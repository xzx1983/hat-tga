"""PatchGAN discriminator for wind super-resolution."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field).

    Args:
        cfg: Hydra config with model.discriminator sub-tree.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        d    = cfg.model.discriminator
        inch = d.get("in_channels", 1)
        nf   = d.get("num_features", 64)

        def block(in_ch: int, out_ch: int, stride: int = 2,
                  norm: bool = True) -> nn.Sequential:
            layers: list = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(inch,      nf,      norm=False),
            block(nf,        nf * 2),
            block(nf * 2,    nf * 4),
            block(nf * 4,    nf * 8, stride=1),
            nn.Conv2d(nf * 8, 1, 4, 1, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, H, W). Returns: patch_scores (B, 1, H', W')."""
        return self.model(x)
