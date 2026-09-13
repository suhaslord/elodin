#!/usr/bin/env python3
"""Leakage-resistant held-out DSN_1k evaluation for the frozen TranAD+ dff16 checkpoint.

Design (frozen before any final-test inference):
- preserve the upstream 799/200 official train/test track split;
- fit only the input MinMax scaler on the 799 training tracks;
- split the 200 official test tracks by a deterministic SHA-256 track hash;
- use 100 complete tracks for threshold calibration;
- choose one global anomaly-score threshold from calibration labels only;
- only after freezing that threshold, run inference on the other 100 tracks once.

The metric is pointwise global anomaly F1/precision/recall. It intentionally does
not run the repository's legacy POT/BF test-label threshold search.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path.cwd()
if not (ROOT / "TranADPlus").is_dir():
    raise SystemExit("Run from the parent directory containing TranADPlus/")
sys.path.insert(0, str(ROOT))

from TranADPlus.src import helpers, models, preprocess_data  # noqa: E402


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float


def metrics(labels: np.ndarray, pred: np.ndarray) -> Metrics:
    y = np.asarray(labels, dtype=bool).ravel()
    p = np.asarray(pred, dtype=bool).ravel()
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    tn = int(np.sum(~y & ~p))
    fn = int(np.sum(y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(tp, fp, tn, fn, precision, recall, f1)


def exact_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, Metrics]:
    """Choose the exact best >= threshold on calibration only in O(n log n)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.uint8).ravel()
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    if s.size == 0 or s.shape != y.shape:
        raise ValueError("invalid calibration arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")

    order = np.argsort(-s, kind="mergesort")
    ss = s[order]
    yy = y[order].astype(np.int64)
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    positives = int(yy.sum())

    # Evaluate only after the final member of each equal-score group so that
    # the cumulative prediction exactly represents score >= threshold.
    end = np.r_[ss[1:] != ss[:-1], True]
    idx = np.flatnonzero(end)
    tp_c = tp[idx].astype(np.float64)
    fp_c = fp[idx].astype(np.float64)
    fn_c = positives - tp_c
    precision = np.divide(tp_c, tp_c + fp_c, out=np.zeros_like(tp_c), where=(tp_c + fp_c) != 0)
    recall = np.divide(tp_c, tp_c + fn_c, out=np.zeros_like(tp_c), where=(tp_c + fn_c) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    thresholds = ss[idx]

    # Same deterministic tie-break policy as the existing helper:
    # F1, then recall, then precision, then the higher threshold.
    candidates = np.flatnonzero(f1 == np.max(f1))
    candidates = candidates[recall[candidates] == np.max(recall[candidates])]
    candidates = candidates[precision[candidates] == np.max(precision[candidates])]
    best = int(candidates[np.argmax(thresholds[candidates])])
    threshold = float(thresholds[best])
    return threshold, metrics(y, s >= threshold)


class ArrayWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int, padding: bool, downsample: int | None):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if padding:
            pad_x = np.tile(x[0], (window_size - 1, 1))
            pad_y = np.tile(y[0], (window_size - 1, 1))
            x = np.vstack([pad_x, x])
            y = np.vstack([pad_y, y])
        if downsample is not None:
            x = helpers.downsample(x, int(downsample)).astype(np.float32, copy=False)
            y = helpers.downsample(y, int(downsample)).astype(np.float32, copy=False)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.window_size = int(window_size)
        self.count = int(self.x.shape[0]) - self.window_size + 1
        if self.count <= 0:
            raise ValueError("track shorter than model window")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, i: int):
        return self.x[i : i + self.window_size], self.y[i : i + self.window_size]


def load_original_track(track_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with track_path.open("rb") as f:
        df = pickle.load(f)
    with label_path.open("rb") as f:
        ranges = pickle.load(f)
    # This matches the repository's default DSN preprocessing: byte columns -> 0,
    # NaNs -> 0, float array, then binary anomaly labels across all channels.
    x = preprocess_data.preprocess_numpy(df, byte_mod="constant", nan_value=0.0).astype(np.float32, copy=False)
    y = helpers.convert_anoms_to_tranad(ranges, x.shape).astype(np.float32, copy=False)
    del df
    return x, y


def fit_scaler(train_dir: Path) -> MinMaxScaler:
    scaler = MinMaxScaler()
    paths = sorted(train_dir.glob("*.df.pkl"), key=lambda p: p.stem)
    if len(paths) != 799:
        raise ValueError(f"expected 799 training tracks, got {len(paths)}")
    for i, path in enumerate(paths, 1):
        with path.open("rb") as f:
            df = pickle.load(f)
        x = preprocess_data.preprocess_numpy(df, byte_mod="constant", nan_value=0.0).astype(np.float32, copy=False)
        scaler.partial_fit(x)
        del df, x
        if i % 100 == 0 or i == len(paths):
            print(f"SCALER_PROGRESS {i}/{len(paths)}", flush=True)
        gc.collect()
    return scaler


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint["hp_dict"]
    if hp.get("model_str") != "TranAD" or hp.get("dataset_str") != "DSN_1k":
        raise ValueError(f"unexpected checkpoint metadata: {hp.get('model_str')} / {hp.get('dataset_str')}")
    # In TranAD+ checkpoint naming, dff16 is a feature-count multiplier rather
    # than a literal hidden width: 16 * 129 DSN features = 2064 units.
    parsed_dff = int(hp["features"]) * int(hp["dim_feedforward"])
    model = models.TranAD(
        n_feats=hp["features"],
        dim_feedforward=parsed_dff,
        batch_sz=hp["batch_sz"],
        window_sz=hp["window_sz"],
        num_encoder_layers=hp["num_layers"],
        num_decoder_layers=hp["num_layers"],
    ).to(device).to(torch.float32)
    model.load_state_dict(checkpoint["model_states"])
    model.eval()
    return model, hp


def split_tracks(test_paths: list[Path], seed: str) -> tuple[list[Path], list[Path]]:
    ranked = sorted(
        test_paths,
        key=lambda p: hashlib.sha256(f"{seed}:{p.stem}".encode()).hexdigest(),
    )
    if len(ranked) != 200:
        raise ValueError(f"expected 200 official test tracks, got {len(ranked)}")
    return ranked[:100], ranked[100:]


def infer_tracks(paths: list[Path], base: Path, scaler: MinMaxScaler, model, hp: dict, device: str, label: str):
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    loss_fn = torch.nn.MSELoss(reduction="none")
    for i, track_path in enumerate(paths, 1):
        label_path = base / "Labels" / track_path.name.replace(".df.pkl", ".l.pkl")
        x, y = load_original_track(track_path, label_path)
        x = scaler.transform(x).astype(np.float32, copy=False)
        ds = ArrayWindowDataset(
            x,
            y,
            window_size=int(hp["window_sz"]),
            padding=bool(hp["padding"]),
            downsample=hp.get("downsample"),
        )
        loader = DataLoader(ds, batch_size=int(hp["batch_sz"]), shuffle=False, num_workers=0)
        loss, _, _, _, anoms = helpers.gen_TranAD_predictions(
            model,
            loss_fn,
            loader,
            alpha=0.5,
            beta=0.5,
            device=device,
        )
        all_scores.append(np.mean(loss, axis=1).astype(np.float32, copy=False))
        all_labels.append((np.sum(anoms, axis=1) >= 1).astype(np.uint8, copy=False))
        del x, y, ds, loader, loss, anoms
        gc.collect()
        if i % 10 == 0 or i == len(paths):
            print(f"{label}_INFERENCE_PROGRESS {i}/{len(paths)}", flush=True)
    return np.concatenate(all_scores), np.concatenate(all_labels)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="results/dsn")
    p.add_argument("--split-seed", default="dsn-heldout-v1-20260913")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / "TranADPlus" / "Datasets" / "Original" / "DSN_1k"
    train_dir = original / "Train" / "Tracks"
    test_dir = original / "Test" / "Tracks"

    print("Fitting scaler on official 799-track training split only", flush=True)
    scaler = fit_scaler(train_dir)
    model, hp = load_model(Path(args.checkpoint), "cpu")
    test_paths = sorted(test_dir.glob("*.df.pkl"), key=lambda p: p.stem)
    cal_paths, final_paths = split_tracks(test_paths, args.split_seed)

    manifest = {
        "design": "official 799 training tracks for scaling; official 200 test tracks split at whole-track boundaries before metrics; threshold selected on 100 calibration tracks only; remaining 100 tracks inferred only after threshold freeze",
        "split_seed": args.split_seed,
        "score": "mean per-channel TranAD reconstruction loss, alpha=0.5 beta=0.5",
        "metric": "pointwise global binary anomaly classification; prediction iff score >= frozen threshold",
        "checkpoint": Path(args.checkpoint).name,
        "checkpoint_hp": {k: (v.item() if hasattr(v, "item") else v) for k, v in hp.items() if isinstance(v, (str, int, float, bool, type(None), np.generic))},
        "calibration_tracks": [p.stem.replace(".df", "") for p in cal_paths],
        "final_test_tracks": [p.stem.replace(".df", "") for p in final_paths],
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Calibration happens first. No final-test track is loaded or inferred here.
    cal_scores, cal_labels = infer_tracks(cal_paths, original / "Test", scaler, model, hp, "cpu", "CAL")
    threshold, cal_metrics = exact_f1_threshold(cal_scores, cal_labels)
    np.save(out / "cal_scores.npy", cal_scores)
    np.save(out / "cal_labels.npy", cal_labels)
    frozen = {
        "threshold": threshold,
        "calibration": asdict(cal_metrics),
        "calibration_points": int(cal_scores.size),
        "calibration_anomaly_points": int(cal_labels.sum()),
    }
    (out / "frozen_threshold.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"THRESHOLD_FROZEN {threshold:.17g}", flush=True)
    print("CALIBRATION_METRICS " + json.dumps(asdict(cal_metrics), sort_keys=True), flush=True)

    # Untouched final test: first inference occurs only after THRESHOLD_FROZEN.
    test_scores, test_labels = infer_tracks(final_paths, original / "Test", scaler, model, hp, "cpu", "FINAL_TEST")
    test_metrics = metrics(test_labels, test_scores >= threshold)
    np.save(out / "test_scores.npy", test_scores)
    np.save(out / "test_labels.npy", test_labels)
    result = {
        "status": "PASS_HELD_OUT",
        "threshold_frozen_on": "calibration only",
        "threshold": threshold,
        "calibration": asdict(cal_metrics),
        "final_test": asdict(test_metrics),
        "calibration_points": int(cal_scores.size),
        "final_test_points": int(test_scores.size),
        "calibration_anomaly_points": int(cal_labels.sum()),
        "final_test_anomaly_points": int(test_labels.sum()),
        "calibration_tracks": len(cal_paths),
        "final_test_tracks": len(final_paths),
    }
    (out / "heldout_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DSN_HELDOUT_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
