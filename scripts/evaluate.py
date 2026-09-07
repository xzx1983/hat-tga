"""Evaluate a trained HAT+TGA checkpoint on the test set.

Usage:
    uv run python scripts/evaluate.py ckpt_path=<path/to/checkpoint.ckpt>
    uv run python scripts/evaluate.py ckpt_path=outputs/checkpoints/hat_tga/best.ckpt
"""

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

_orig_torch_load = torch.load


def _patched_load(f, map_location=None, **kw):
    kw.pop("weights_only", None)
    return _orig_torch_load(f, map_location=map_location, weights_only=False, **kw)


torch.load = _patched_load

from hat_tga.data import WindSRDataModule  # noqa: E402
from hat_tga.trainer.tga_trainer import TGATrainer  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="training/hat_tga", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.get("seed", 42))

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("Provide ckpt_path=<path/to/checkpoint.ckpt>")

    log.info(f"Loading checkpoint: {ckpt_path}")
    model = TGATrainer.load_from_checkpoint(ckpt_path, cfg=cfg, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    datamodule = WindSRDataModule(cfg)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    log.info(f"Test batches: {len(test_loader)}")

    hr_mean = float(cfg.dataset.hr_speed_mean)
    hr_std  = float(cfg.dataset.hr_speed_std)

    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            lr  = batch["lr"].to(device)
            dem = batch["dem"].to(device)
            tgt = batch["hr_speed_raw"].to(device)  # raw m/s

            pred_norm = model.model(lr, dem)
            pred_ms   = pred_norm.float() * hr_std + hr_mean
            pred_ms   = pred_ms.clamp(min=0.0)

            all_preds.append(pred_ms.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N, 1, H, W)
    targets = np.concatenate(all_targets, axis=0)

    rmse  = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae   = float(np.mean(np.abs(preds - targets)))
    bias  = float(np.mean(preds - targets))

    # Pearson correlation
    p_flat = preds.flatten()
    t_flat = targets.flatten()
    corr   = float(np.corrcoef(p_flat, t_flat)[0, 1])

    results = {
        "rmse": round(rmse, 4),
        "mae":  round(mae,  4),
        "bias": round(bias, 4),
        "r":    round(corr, 4),
        "n_samples": int(preds.shape[0]),
    }

    out_dir  = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results: RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:.4f}  r={corr:.4f}")
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
