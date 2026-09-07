"""Trainer modules for HAT+TGA."""

from hat_tga.trainer.base import WindSRLightningModule
from hat_tga.trainer.tga_trainer import TGATrainer

__all__ = ["WindSRLightningModule", "TGATrainer"]
