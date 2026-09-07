"""Generate HAT+TGA analysis figures from a trained checkpoint.

Produces four single-model figures (HAT+TGA only):
  fig1_rmse_map.png     — spatial RMSE map over the test domain
  fig2_bias_map.png     — spatial bias map
  fig4_seasonal_bar.png — seasonal RMSE bar chart
  fig5_example_day.png  — example prediction vs CLDAS reference

Note: Manuscript figures are multi-model comparison panels (Bicubic, SRResNet,
HAT+DEM, HAT+TGA) and require baseline model outputs not included here.
The RMSE values in these figures match the manuscript's HAT+TGA panel (0.726 m/s).

Usage:
    uv run python scripts/generate_figures.py ckpt_path=<checkpoint.ckpt>
    uv run python scripts/generate_figures.py ckpt_path=<ckpt> output_dir=outputs/
"""

import logging
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

matplotlib.use("Agg")

torch.set_float32_matmul_precision("high")

_orig_torch_load = torch.load


def _patched_load(f, map_location=None, **kw):
    kw.pop("weights_only", None)
    return _orig_torch_load(f, map_location=map_location, weights_only=False, **kw)


torch.load = _patched_load

from hat_tga.data import WindSRDataModule  # noqa: E402
from hat_tga.trainer.tga_trainer import TGATrainer  # noqa: E402

log = logging.getLogger(__name__)

_HR_LAT = np.array([26.03125 + i * 0.0625 for i in range(128)], dtype=np.float64)
_HR_LON = np.array([97.03125 + j * 0.0625 for j in range(192)], dtype=np.float64)

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _run_inference(model, loader, cfg, device):
    hr_mean = float(cfg.dataset.hr_speed_mean)
    hr_std  = float(cfg.dataset.hr_speed_std)
    preds, targets, months = [], [], []
    with torch.no_grad():
        for batch in loader:
            lr  = batch["lr"].to(device)
            dem = batch["dem"].to(device)
            tgt = batch["hr_speed_raw"].to(device)
            pred = (model.model(lr, dem).float() * hr_std + hr_mean).clamp(min=0.0)
            preds.append(pred.cpu().numpy())
            targets.append(tgt.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def fig_rmse_map(preds, targets, out_dir: Path):
    sq_err = (preds - targets) ** 2   # (N, 1, H, W)
    rmse_map = np.sqrt(sq_err.mean(axis=(0, 1)))  # (H, W)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        _HR_LON, _HR_LAT, rmse_map, cmap="YlOrRd", vmin=0, vmax=2.5
    )
    plt.colorbar(im, ax=ax, label="RMSE (m/s)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("HAT+TGA — Spatial RMSE (test 2023–2024)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_rmse_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig1_rmse_map.png")


def fig_bias_map(preds, targets, out_dir: Path):
    bias_map = (preds - targets).mean(axis=(0, 1))  # (H, W)
    vmax = max(abs(bias_map.min()), abs(bias_map.max()))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        _HR_LON, _HR_LAT, bias_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im, ax=ax, label="Bias (m/s)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("HAT+TGA — Spatial Bias (test 2023–2024)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_bias_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig2_bias_map.png")


def fig_seasonal_bar(preds, targets, meta_months, out_dir: Path):
    season_rmse = {}
    for season, mons in SEASONS.items():
        mask = np.isin(meta_months, mons)
        if mask.sum() == 0:
            continue
        se = (preds[mask] - targets[mask]) ** 2
        season_rmse[season] = float(np.sqrt(se.mean()))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(season_rmse.keys()), list(season_rmse.values()), color="#4c72b0", width=0.5)
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("HAT+TGA — Seasonal RMSE")
    ax.set_ylim(0, max(season_rmse.values()) * 1.25)
    for i, (s, v) in enumerate(season_rmse.items()):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_seasonal_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig4_seasonal_bar.png")


def fig_example_day(preds, targets, out_dir: Path, sample_idx: int = 0):
    p = preds[sample_idx, 0]
    t = targets[sample_idx, 0]
    vmax = max(p.max(), t.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in zip(
        axes,
        [t, p, p - t],
        ["CLDAS (Reference)", "HAT+TGA (Predicted)", "Residual"],
    ):
        if "Residual" in title:
            vr = max(abs((p - t).min()), abs((p - t).max()))
            im = ax.pcolormesh(_HR_LON, _HR_LAT, data, cmap="RdBu_r", vmin=-vr, vmax=vr)
        else:
            im = ax.pcolormesh(_HR_LON, _HR_LAT, data, cmap="viridis", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="m/s")
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_example_day.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig5_example_day.png")


@hydra.main(config_path="../conf", config_name="training/hat_tga", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("Provide ckpt_path=<path/to/checkpoint.ckpt>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading {ckpt_path} on {device}")
    model = TGATrainer.load_from_checkpoint(ckpt_path, cfg=cfg, strict=False)
    model.eval().to(device)

    datamodule = WindSRDataModule(cfg)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    preds, targets = _run_inference(model, loader, cfg, device)
    log.info(f"Inference done: {preds.shape[0]} samples")

    # Months approximated from batch order (each sample = 1 h; 8760 h/year assumed)
    n = preds.shape[0]
    months = np.tile(np.repeat(np.arange(1, 13), 730), 2)[:n]

    out_dir = Path(cfg.output_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_rmse_map(preds, targets, out_dir)
    fig_bias_map(preds, targets, out_dir)
    fig_seasonal_bar(preds, targets, months, out_dir)
    fig_example_day(preds, targets, out_dir)

    log.info(f"All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
