"""TGATrainer — extends the base trainer with station loss and elevation weighting."""

import logging

import torch
from omegaconf import DictConfig

from hat_tga.trainer.base import WindSRLightningModule

logger = logging.getLogger(__name__)


class TGATrainer(WindSRLightningModule):
    """WindSRLightningModule + station observation loss + elevation-weighted pixel loss.

    The station loss computes L1 between predicted wind speed (m/s, denormalised)
    and daily GSOD station observations at the matching grid cells.

    The elevation mask is derived on-the-fly from the normalised DEM channel 0
    in each batch, so it automatically adapts to patch vs full-domain inputs.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        t = cfg.trainer
        self.lambda_station  = float(t.get("lambda_station", 0.1))
        self.elev_threshold  = float(t.get("elev_threshold", 2000.0))
        self.hr_mean         = float(cfg.dataset.hr_speed_mean)
        self.hr_std          = float(cfg.dataset.hr_speed_std)

        dem_mean0 = float(cfg.dataset.get("dem_mean", [2380.05])[0])
        dem_std0  = float(cfg.dataset.get("dem_std",  [1533.35])[0])
        self._elev_thresh_norm = (self.elev_threshold - dem_mean0) / dem_std0

        elev_weight = float(t.get("elev_weight", 1.0))
        logger.info(
            f"TGATrainer: lambda_station={self.lambda_station}, "
            f"elev_weight={elev_weight} (>{self.elev_threshold:.0f}m)"
        )

    def _make_elev_mask(self, dem: torch.Tensor) -> torch.Tensor:
        elev_norm = dem[:, 0:1, :, :]
        return (elev_norm > self._elev_thresh_norm).to(dtype=dem.dtype)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        opt_g, opt_d = self.optimizers()

        elev_mask = self._make_elev_mask(batch["dem"])
        loss_g, log_g, pred = self.model.generator_step(batch, elev_mask=elev_mask)

        if self.current_epoch < self.warmup_epochs:
            loss_g = loss_g - (self.model.loss_fn.lam_adv * log_g["loss_adv_g"])

        if self.lambda_station > 0:
            mask = batch.get("station_mask")
            vals = batch.get("station_vals")
            if mask is not None:
                mask = mask.to(pred.device)
                vals = vals.to(pred.device)
                n_stn = mask.sum()
                if n_stn > 0:
                    pred_ms = pred.float() * self.hr_std + self.hr_mean
                    L_stn   = (mask * (pred_ms - vals).abs()).sum() / n_stn
                    loss_g  = loss_g + self.lambda_station * L_stn
                    log_g["loss_station"] = L_stn.item()

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
