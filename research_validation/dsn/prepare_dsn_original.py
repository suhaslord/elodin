#!/usr/bin/env python3
"""Prepare the official TranAD+ DSN_1k train/test tracks with minimal disk duplication.

Run from the directory that contains the cloned ``TranADPlus`` package.
The upstream initializer is preserved; its train/test copies are replaced with
hard-links so the 3.7 GB raw dataset does not get duplicated unnecessarily.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "TranADPlus").is_dir():
    raise SystemExit("Run from the parent directory containing TranADPlus/")
sys.path.insert(0, str(ROOT))

from TranADPlus.src import preprocess_data  # noqa: E402


def hardlink_copy(src: str, dst: str, *args, **kwargs) -> str:
    """Drop-in replacement for shutil.copy used by the DSN initializer."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    os.link(src, dst)
    return str(dst_path)


def main() -> None:
    raw = ROOT / "TranADPlus" / "Datasets" / "Raw" / "DSN_1k"
    required = [raw / "mons.pkl", raw / "drs.pkl"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing raw DSN input(s): {missing}")

    # Upstream initialize_dsn_data_split_names calls shutil.copy for each
    # official train/test track. Hard-linking preserves identical bytes while
    # keeping peak disk use manageable on a hosted runner.
    preprocess_data.shutil.copy = hardlink_copy
    preprocess_data.initialize_dataset("DSN_1k")

    base = ROOT / "TranADPlus" / "Datasets" / "Original" / "DSN_1k"
    train = sorted((base / "Train" / "Tracks").glob("*.df.pkl"))
    test = sorted((base / "Test" / "Tracks").glob("*.df.pkl"))
    if len(train) != 799 or len(test) != 200:
        raise SystemExit(f"Unexpected official split: train={len(train)}, test={len(test)}")

    print(f"DSN_PREP_PASS train_tracks={len(train)} test_tracks={len(test)}")


if __name__ == "__main__":
    main()
