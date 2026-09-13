#!/usr/bin/env python3
"""Run the held-out DSN evaluation while honoring the checkpoint's stored architecture."""
import sys
from pathlib import Path

import torch

# Direct script execution puts research_validation/dsn on sys.path, not the repo root.
sys.path.insert(0, str(Path.cwd()))

import research_validation.dsn.run_dsn_heldout as base
from TranADPlus.src import models


# TranAD+'s gen_TranAD_predictions() reads dataloader.dataset.padding. The
# lightweight held-out dataset must expose the same public attribute as the
# upstream TSDataset_tracks implementation.
_original_dataset_init = base.ArrayWindowDataset.__init__


def _dataset_init(self, x, y, window_size, padding, downsample):
    _original_dataset_init(self, x, y, window_size, padding, downsample)
    self.padding = bool(padding)


base.ArrayWindowDataset.__init__ = _dataset_init


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hp = checkpoint["hp_dict"]
    if hp.get("model_str") != "TranAD" or hp.get("dataset_str") != "DSN_1k":
        raise ValueError(
            f"unexpected checkpoint metadata: {hp.get('model_str')} / {hp.get('dataset_str')}"
        )

    # training_loop.py parses dim_feedforward before saving hp_dict, so the
    # checkpoint already contains the literal model width (16 for this run).
    parsed_dff = int(hp["dim_feedforward"])
    model = models.TranAD(
        n_feats=int(hp["features"]),
        dim_feedforward=parsed_dff,
        batch_sz=int(hp["batch_sz"]),
        window_sz=int(hp["window_sz"]),
        num_encoder_layers=int(hp["num_layers"]),
        num_decoder_layers=int(hp["num_layers"]),
    ).to(device).to(torch.float32)
    model.load_state_dict(checkpoint["model_states"])
    model.eval()
    print(f"CHECKPOINT_ARCH features={hp['features']} dim_feedforward={parsed_dff}", flush=True)
    return model, hp


base.load_model = load_model
base.main()
