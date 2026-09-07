"""Shared pytest fixtures for HAT+TGA tests.

Sets up a minimal Hydra config object and resolves data paths so tests
work whether data lives under hat-tga/data/ (per README) or under the
parent Project_Wind/data/ (CI shortcut via HATTGA_DATA_ROOT env var).
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# HATTGA_DATA_ROOT: path to processed data; defaults to hat-tga/data/
_REPO_ROOT = Path(__file__).parent.parent
_DATA_ROOT  = Path(os.environ.get("HATTGA_DATA_ROOT", str(_REPO_ROOT / "data")))

# HATTGA_CKPT_PATH: checkpoint path; auto-discovered if not set
def _default_ckpt() -> str:
    candidates = [
        _REPO_ROOT / "outputs" / "checkpoints" / "hat_tga" / "best.ckpt",
        _REPO_ROOT.parent / "outputs_hat_tga" / "checkpoints" / "hat_tga"
        / "best-epoch=045-rmseval_rmse=0.7378.ckpt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""

CKPT_PATH = os.environ.get("HATTGA_CKPT_PATH", _default_ckpt())


@pytest.fixture(scope="session")
def base_cfg():
    """Minimal Hydra config mirroring conf/training/hat_tga.yaml."""
    cfg = OmegaConf.create({
        "seed": 42,
        "output_dir": str(_REPO_ROOT / "outputs"),
        "model": {
            "name": "hat_tga",
            "generator": {
                "in_channels": 5,
                "dem_enc_in_channels": 5,
                "dem_enc_channels": 64,
                "embed_dim": 180,
                "depths": [6, 6, 6, 6],
                "num_heads": [6, 6, 6, 6],
                "window_size": 8,
                "scale_factor": 4,
            },
            "discriminator": {
                "in_channels": 1,
                "num_features": 64,
            },
            "perceptual": {"depth": 5},
        },
        "dataset": {
            "name": "wind_sr",
            "split": "test",
            "era5_dir":  str(_DATA_ROOT / "processed" / "era5"),
            "cldas_dir": str(_DATA_ROOT / "processed" / "cldas"),
            "dem_path":  str(_DATA_ROOT / "processed" / "dem" / "dem_features_0.0625deg.nc"),
            "station_dir": str(_DATA_ROOT / "raw" / "stations"),
            "train_years": [2017, 2019, 2020, 2021],
            "val_years":   [2022],
            "test_years":  [2023, 2024],
            "lr_mean":  [-0.0362, 0.1926, 77502.28, 282.36, 399.04],
            "lr_std":   [1.2003, 1.3726, 13582.92, 11.235, 502.58],
            "dem_mean": [2380.05, 17.47, 0.002, -0.009, 26.39],
            "dem_std":  [1533.35, 10.14, 0.655, 0.644, 15.78],
            "hr_speed_mean": 1.2314,
            "hr_speed_std":  0.9811,
            "hr_patch_size": 64,
            "lr_patch_size": 16,
            "lat_min": 26.0,
            "lat_max": 34.0,
            "lon_min": 97.0,
            "lon_max": 109.0,
            "use_station_loss": False,
            "augmentation": {"random_flip_h": False},
        },
        "trainer": {
            "max_epochs": 200,
            "lr_g": 1e-4,
            "lr_d": 1e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "gradient_clip_val": 1.0,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "lambda_pixel": 1.0,
            "lambda_perceptual": 0.1,
            "lambda_adv": 0.005,
            "lambda_physical": 0.05,
            "lambda_station": 0.1,
            "lambda_wavelet": 0.2,
            "wavelet_hf_weight": 2.0,
            "elev_weight": 2.0,
            "elev_threshold": 2000.0,
        },
    })
    return cfg
