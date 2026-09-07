"""Loss functions for HAT+TGA wind super-resolution.

Combines:
  - L1 pixel loss with optional elevation weighting  (weight: lambda_pixel)
  - Conv-feature perceptual loss                      (weight: lambda_perceptual)
  - Adversarial GAN loss                              (weight: lambda_adv)
  - Physical gradient loss                            (weight: lambda_physical)
  - Wavelet high-frequency subband L1 loss            (weight: lambda_wavelet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ConvFeatureExtractor(nn.Module):
    """Lightweight multi-scale feature extractor for single-channel maps."""

    def __init__(self, depth: int = 5) -> None:
        super().__init__()
        channels = [1, 32, 64, 128, 128, 128][: depth + 1]
        layers: list = []
        for i in range(depth):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),
            ]
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def haar_dwt2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D Haar discrete wavelet transform."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def pixel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    elev_weight: float = 1.0,
    elev_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if elev_mask is None or elev_weight == 1.0:
        return F.l1_loss(pred, target)
    w = torch.ones_like(target)
    w = w + (elev_weight - 1.0) * elev_mask.to(target.device, dtype=target.dtype)
    return (w * (pred - target).abs()).mean()


def perceptual_loss(
    extractor: ConvFeatureExtractor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(extractor(pred), extractor(target))


def adversarial_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def adversarial_loss_discriminator(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return (loss_real + loss_fake) * 0.5


def physical_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    B = pred.shape[0]
    p = pred.view(B, 1, *pred.shape[-2:])
    t = target.view(B, 1, *target.shape[-2:])
    gx = F.conv2d(p, sobel_x, padding=1)
    gy = F.conv2d(p, sobel_y, padding=1)
    tx = F.conv2d(t, sobel_x, padding=1)
    ty = F.conv2d(t, sobel_y, padding=1)
    return F.l1_loss(gx, tx) + F.l1_loss(gy, ty)


def wavelet_hf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hf_weight: float = 2.0,
) -> torch.Tensor:
    ll_p, lh_p, hl_p, hh_p = haar_dwt2d(pred)
    ll_t, lh_t, hl_t, hh_t = haar_dwt2d(target)
    l_ll = F.l1_loss(ll_p, ll_t)
    l_lh = F.l1_loss(lh_p, lh_t)
    l_hl = F.l1_loss(hl_p, hl_t)
    l_hh = F.l1_loss(hh_p, hh_t)
    return l_ll + hf_weight * (l_lh + l_hl + l_hh)


class WindSRLoss(nn.Module):
    """Combined generator loss for wind SR."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        t = cfg.trainer
        self.lam_pixel      = t.get("lambda_pixel",      1.0)
        self.lam_perceptual = t.get("lambda_perceptual", 0.1)
        self.lam_adv        = t.get("lambda_adv",        0.005)
        self.lam_physical   = t.get("lambda_physical",   0.05)
        self.lam_wavelet    = t.get("lambda_wavelet",    0.0)
        self.elev_weight    = float(t.get("elev_weight", 1.0))
        self.hf_weight      = float(t.get("wavelet_hf_weight", 2.0))
        depth = cfg.model.get("perceptual", {}).get("depth", 5)
        self.extractor = ConvFeatureExtractor(depth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fake_logits: torch.Tensor,
        elev_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        l_pixel = pixel_loss(pred, target, self.elev_weight, elev_mask)
        l_perc  = perceptual_loss(self.extractor, pred, target)
        l_adv   = adversarial_loss_generator(fake_logits)
        l_phys  = physical_gradient_loss(pred, target)

        total = (
            self.lam_pixel      * l_pixel
            + self.lam_perceptual * l_perc
            + self.lam_adv        * l_adv
            + self.lam_physical   * l_phys
        )
        log = {
            "loss_pixel":      l_pixel.item(),
            "loss_perceptual": l_perc.item(),
            "loss_adv_g":      l_adv.item(),
            "loss_physical":   l_phys.item(),
            "loss_g_total":    total.item(),
        }
        if self.lam_wavelet > 0:
            l_wav = wavelet_hf_loss(pred, target, self.hf_weight)
            total = total + self.lam_wavelet * l_wav
            log["loss_wavelet"] = l_wav.item()
        return total, log
