"""Base PyTorch Lightning trainer for HAT+TGA GAN training."""

import logging

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from hat_tga.model import ModelFactory

logger = logging.getLogger(__name__)


class WindSRLightningModule(pl.LightningModule):
    """Lightning module for GAN-based wind SR.

    Training strategy:
      - Epochs 0 … warmup_epochs-1 : pixel-only (adversarial loss suppressed)
      - Epochs warmup_epochs …     : full GAN training
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        t = cfg.trainer
        self.lr_g           = t.get("lr_g", 1e-4)
        self.lr_d           = t.get("lr_d", 1e-4)
        self.warmup_epochs  = t.get("warmup_epochs", 5)
        self.max_epochs     = t.get("max_epochs", 200)
        self.scheduler_type = t.get("lr_scheduler", "cosine")

        model_cls  = ModelFactory(cfg.model.name)
        self.model = model_cls(cfg)
        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch: dict, batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()

        loss_g, log_g, pred = self.model.generator_step(batch)
        if self.current_epoch < self.warmup_epochs:
            loss_g = loss_g - (self.model.loss_fn.lam_adv * log_g["loss_adv_g"])

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=self.cfg.trainer.get("gradient_clip_val", 1.0))
        opt_g.step()

        if self.current_epoch >= self.warmup_epochs:
            loss_d, log_d = self.model.discriminator_step(batch, pred)
            opt_d.zero_grad()
            self.manual_backward(loss_d)
            self.clip_gradients(opt_d, gradient_clip_val=self.cfg.trainer.get("gradient_clip_val", 1.0))
            opt_d.step()
            self.log_dict({f"train/{k}": v for k, v in log_d.items()}, on_step=True)

        self.log_dict({f"train/{k}": v for k, v in log_g.items()}, on_step=True)

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        lr, dem, target = batch["lr"], batch["dem"], batch["hr_speed"]
        pred = self.model(lr, dem)
        mae  = torch.mean(torch.abs(pred - target))
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        bias = torch.mean(pred - target)
        self.log("val_mae",  mae,  on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
        self.log("val_bias", bias, on_epoch=True)
        return {"val_mae": mae, "val_rmse": rmse}

    def configure_optimizers(self) -> tuple[list, list]:
        opt_g = Adam(self.model.generator.parameters(),      lr=self.lr_g, betas=(0.9, 0.999))
        opt_d = Adam(self.model.discriminator.parameters(), lr=self.lr_d, betas=(0.9, 0.999))
        schedulers = []
        for opt in (opt_g, opt_d):
            if self.scheduler_type == "cosine":
                warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs)
                cosine = CosineAnnealingLR(opt, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-7)
                sched  = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
            else:
                sched = LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=self.max_epochs)
            schedulers.append({"scheduler": sched, "interval": "epoch"})
        return [opt_g, opt_d], schedulers

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.warmup_epochs:
            logger.info(f"Epoch {self.current_epoch}: switching to full GAN training.")
