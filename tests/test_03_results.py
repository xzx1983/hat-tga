"""Test 03 — Results: reproduce manuscript evaluation numbers.

Loads the trained checkpoint, runs inference over the full test set
(2023–2024), and asserts the metrics are within tolerance of the
published values.

Published results (Table 2, npj Climate and Atmospheric Science):
    RMSE  = 0.726 m/s
    MAE   = 0.509 m/s
    Bias  = -0.010 m/s
    r     = 0.615

Exact values from eval_results.json (used as regression targets):
    RMSE  = 0.7256
    MAE   = 0.5090
    Bias  = -0.0096
    r     = 0.6146

Run:
    uv run pytest tests/test_03_results.py -v -s

Requires:
    - Processed data in data/processed/ (or HATTGA_DATA_ROOT)
    - Trained checkpoint (HATTGA_CKPT_PATH or sibling outputs_hat_tga/)
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import CKPT_PATH, _DATA_ROOT

# Expected values from the manuscript; tolerance ±0.002 for all metrics.
EXPECTED = {
    "rmse": 0.7256,
    "mae":  0.5090,
    "bias": -0.0096,
    "r":    0.6146,
}
TOL = {
    "rmse": 0.002,
    "mae":  0.002,
    "bias": 0.002,
    "r":    0.002,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _data_available() -> bool:
    era5_dir  = _DATA_ROOT / "processed" / "era5"
    cldas_dir = _DATA_ROOT / "processed" / "cldas"
    return (
        (era5_dir  / "era5_2023.nc").exists()  and
        (era5_dir  / "era5_2024.nc").exists()  and
        (cldas_dir / "cldas_2023.nc").exists() and
        (cldas_dir / "cldas_2024.nc").exists()
    )


def _ckpt_available() -> bool:
    return bool(CKPT_PATH) and Path(CKPT_PATH).exists()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_model(base_cfg):
    if not _ckpt_available():
        pytest.skip(f"Checkpoint not found at '{CKPT_PATH}'. "
                    "Set HATTGA_CKPT_PATH or place checkpoint at "
                    "outputs/checkpoints/hat_tga/best.ckpt")
    import torch
    # PyTorch ≥2.6 sets weights_only=True by default, which rejects OmegaConf
    # objects stored in Lightning checkpoints.  Patch for the load call only.
    _orig = torch.load
    torch.load = lambda f, map_location=None, **kw: _orig(
        f, map_location=map_location, weights_only=False, **{k: v for k, v in kw.items() if k != "weights_only"}
    )
    try:
        from hat_tga.trainer.tga_trainer import TGATrainer
        model = TGATrainer.load_from_checkpoint(CKPT_PATH, cfg=base_cfg, strict=False)
    finally:
        torch.load = _orig
    model.eval()
    return model


@pytest.fixture(scope="module")
def test_loader(base_cfg):
    if not _data_available():
        pytest.skip(
            "Test data not found. Run `python scripts/setup_test_data.py` "
            "to link data, then retry."
        )
    from hat_tga.data import WindSRDataModule
    dm = WindSRDataModule(base_cfg)
    dm.setup("test")
    return dm.test_dataloader()


# ── Inference helper ──────────────────────────────────────────────────────────

def _run_inference(model, loader, cfg) -> tuple[np.ndarray, np.ndarray]:
    hr_mean = float(cfg.dataset.hr_speed_mean)
    hr_std  = float(cfg.dataset.hr_speed_std)
    device  = next(model.parameters()).device

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            lr  = batch["lr"].to(device)
            dem = batch["dem"].to(device)
            tgt = batch["hr_speed_raw"].to(device)

            pred = model.model(lr, dem).float() * hr_std + hr_mean
            pred = pred.clamp(min=0.0)
            preds.append(pred.cpu().numpy())
            targets.append(tgt.cpu().numpy())

    return np.concatenate(preds), np.concatenate(targets)


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    diff = preds - targets
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    p = preds.flatten()
    t = targets.flatten()
    r = float(np.corrcoef(p, t)[0, 1])
    return {"rmse": rmse, "mae": mae, "bias": bias, "r": r}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_checkpoint_exists():
    """Checkpoint file must exist at the expected path."""
    if not _ckpt_available():
        pytest.skip("Checkpoint not found; set HATTGA_CKPT_PATH")
    assert Path(CKPT_PATH).exists(), f"Checkpoint missing: {CKPT_PATH}"
    size_mb = Path(CKPT_PATH).stat().st_size / 1e6
    print(f"\n  Checkpoint: {CKPT_PATH}  ({size_mb:.1f} MB)")


def test_data_files_exist():
    """Processed test data files must exist."""
    if not _data_available():
        pytest.skip("Test data not found")
    base = _DATA_ROOT / "processed"
    for year in [2023, 2024]:
        assert (base / "era5"  / f"era5_{year}.nc").exists()
        assert (base / "cldas" / f"cldas_{year}.nc").exists()
    assert (base / "dem" / "dem_features_0.0625deg.nc").exists()


def test_dataset_loads(test_loader):
    """Dataset must load and yield correctly shaped batches."""
    batch = next(iter(test_loader))
    assert "lr"           in batch
    assert "dem"          in batch
    assert "hr_speed_raw" in batch

    lr  = batch["lr"]
    dem = batch["dem"]
    tgt = batch["hr_speed_raw"]

    assert lr.shape[1]  == 5,  f"Expected 5 ERA5 channels, got {lr.shape[1]}"
    assert dem.shape[1] == 5,  f"Expected 5 DEM channels, got {dem.shape[1]}"
    assert tgt.shape[1] == 1,  f"Expected 1 target channel, got {tgt.shape[1]}"
    # Spatial scale factor: HR = 4× LR
    assert dem.shape[2] == lr.shape[2] * 4, "DEM/LR spatial mismatch"
    print(f"\n  Batch LR: {tuple(lr.shape)}  DEM: {tuple(dem.shape)}  HR: {tuple(tgt.shape)}")


def test_model_output_shape(loaded_model, test_loader, base_cfg):
    """Model output must have shape (B, 1, H_hr, W_hr)."""
    batch = next(iter(test_loader))
    hr_mean = float(base_cfg.dataset.hr_speed_mean)
    hr_std  = float(base_cfg.dataset.hr_speed_std)
    device  = next(loaded_model.parameters()).device

    with torch.no_grad():
        pred = loaded_model.model(batch["lr"].to(device), batch["dem"].to(device))
    pred_ms = pred.float() * hr_std + hr_mean

    assert pred_ms.shape[1] == 1, "Output must have 1 channel (wind speed)"
    assert pred_ms.shape[2:] == batch["hr_speed_raw"].shape[2:], "Output spatial mismatch"
    assert (pred_ms.clamp(min=0.0) >= 0).all(), "Wind speed must be non-negative"


def test_rmse_matches_manuscript(loaded_model, test_loader, base_cfg):
    """Overall RMSE must match the published value within ±0.002 m/s."""
    device = next(loaded_model.parameters()).device
    loaded_model.to(device)
    preds, targets = _run_inference(loaded_model, test_loader, base_cfg)
    metrics = _compute_metrics(preds, targets)

    print(f"\n  Computed RMSE : {metrics['rmse']:.4f} m/s")
    print(f"  Expected RMSE : {EXPECTED['rmse']:.4f} m/s  (tol ±{TOL['rmse']})")

    assert abs(metrics["rmse"] - EXPECTED["rmse"]) <= TOL["rmse"], (
        f"RMSE {metrics['rmse']:.4f} differs from manuscript {EXPECTED['rmse']:.4f} "
        f"by more than ±{TOL['rmse']}"
    )


def test_mae_matches_manuscript(loaded_model, test_loader, base_cfg):
    """Overall MAE must match the published value within ±0.002 m/s."""
    device = next(loaded_model.parameters()).device
    preds, targets = _run_inference(loaded_model, test_loader, base_cfg)
    metrics = _compute_metrics(preds, targets)

    print(f"\n  Computed MAE : {metrics['mae']:.4f} m/s")
    print(f"  Expected MAE : {EXPECTED['mae']:.4f} m/s  (tol ±{TOL['mae']})")

    assert abs(metrics["mae"] - EXPECTED["mae"]) <= TOL["mae"]


def test_bias_matches_manuscript(loaded_model, test_loader, base_cfg):
    """Overall Bias must match the published value within ±0.002 m/s."""
    device = next(loaded_model.parameters()).device
    preds, targets = _run_inference(loaded_model, test_loader, base_cfg)
    metrics = _compute_metrics(preds, targets)

    print(f"\n  Computed Bias : {metrics['bias']:.4f} m/s")
    print(f"  Expected Bias : {EXPECTED['bias']:.4f} m/s  (tol ±{TOL['bias']})")

    assert abs(metrics["bias"] - EXPECTED["bias"]) <= TOL["bias"]


def test_correlation_matches_manuscript(loaded_model, test_loader, base_cfg):
    """Pearson r must match the published value within ±0.002."""
    device = next(loaded_model.parameters()).device
    preds, targets = _run_inference(loaded_model, test_loader, base_cfg)
    metrics = _compute_metrics(preds, targets)

    print(f"\n  Computed r : {metrics['r']:.4f}")
    print(f"  Expected r : {EXPECTED['r']:.4f}  (tol ±{TOL['r']})")

    assert abs(metrics["r"] - EXPECTED["r"]) <= TOL["r"]


def test_full_metrics_summary(loaded_model, test_loader, base_cfg):
    """Print a full metrics table and save results/test_metrics.json."""
    device = next(loaded_model.parameters()).device
    preds, targets = _run_inference(loaded_model, test_loader, base_cfg)
    metrics = _compute_metrics(preds, targets)

    print("\n" + "=" * 50)
    print(f"  {'Metric':<8}  {'Computed':>10}  {'Expected':>10}  {'Diff':>8}")
    print("-" * 50)
    for k in ("rmse", "mae", "bias", "r"):
        diff = metrics[k] - EXPECTED[k]
        ok   = "✓" if abs(diff) <= TOL[k] else "✗"
        print(f"  {k:<8}  {metrics[k]:>10.4f}  {EXPECTED[k]:>10.4f}  {diff:>+8.4f}  {ok}")
    print("=" * 50)

    # Save for CI record
    out = Path(base_cfg.output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_metrics.json", "w") as f:
        json.dump({"computed": metrics, "expected": EXPECTED}, f, indent=2)
    print(f"  Saved: {out / 'test_metrics.json'}")
