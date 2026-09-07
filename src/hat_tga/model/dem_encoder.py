"""Lightweight CNN encoder for DEM features."""

import torch
import torch.nn as nn


class DEMEncoder(nn.Module):
    """Lightweight 3-conv CNN that maps DEM features to a dense embedding.

    Args:
        in_channels:  Number of DEM input channels (default 5: elev, slope, sin/cos aspect, TRI).
        out_channels: Output embedding dimension (default 64).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
