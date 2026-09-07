"""Training entry point for HAT+TGA.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py trainer.max_epochs=50
    uv run python scripts/train.py ckpt_path=outputs/checkpoints/hat_tga/last.ckpt
"""

import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")

# Allow loading checkpoints saved without weights_only
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
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    log.info(f"Training {cfg.model.name} | seed={cfg.get('seed', 42)}")

    datamodule = WindSRDataModule(cfg)
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()
    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = TGATrainer(cfg)

    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", cfg.model.name)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:03d}-rmse{val_rmse:.4f}",
            monitor="val_rmse",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val_rmse",
            patience=cfg.trainer.get("early_stop_patience", 20),
            mode="min",
        ),
    ]

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.output_dir, "logs"),
        name=cfg.model.name,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.get("precision", "bf16-mixed"),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=False,
        enable_progress_bar=True,
    )

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        log.info(f"Resuming from checkpoint: {ckpt_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    log.info(f"Training complete. Best model: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
