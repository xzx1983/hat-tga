"""HAT+TGA — Terrain-Guided Attention for wind speed super-resolution.

Extends the HAT/SwinIR backbone by injecting DEM features directly into every
window attention computation as a learned additive terrain bias.  For each
window, DEM features (64-dim) are projected to per-head terrain scores; the
outer sum of those scores forms a (num_heads, T, T) terrain bias added to the
attention logits before softmax.

Key design choices:
- terrain_proj is zero-initialized: at epoch 0 TGA equals standard
  WindowAttention (no training instability from DEM injection).
- Three DEMEncoder modules produce DEM features at coarse/mid/fine scales,
  all at LR spatial resolution, assigned to RSTB tiers 0/1/2-3 respectively.
- Input stem uses 5 ERA5 channels only (DEM is injected via TGA, not
  concatenated at the input).
- For shifted-window blocks, DEM features are cyclically rolled by the same
  shift_size as tokens so dem windows align with token windows.

Registered as 'hat_tga'.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from hat_tga.model import register_model
from hat_tga.model.dem_encoder import DEMEncoder
from hat_tga.model.discriminator import PatchDiscriminator
from hat_tga.model.losses import WindSRLoss, adversarial_loss_discriminator

logger = logging.getLogger(__name__)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class TerrainGuidedAttention(nn.Module):
    """Window-based multi-head self-attention with additive terrain bias.

    The terrain bias B[h,i,j] = t_h[i] + t_h[j] is an outer-sum of per-token
    per-head terrain scores derived from DEM features via a linear projection.
    terrain_proj is zero-initialized so TGA = standard attention at epoch 0.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        dem_channels: int = 64,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Zero-initialized so TGA = standard attention at epoch 0
        self.terrain_proj = nn.Linear(dem_channels, num_heads, bias=True)
        nn.init.zeros_(self.terrain_proj.weight)
        nn.init.zeros_(self.terrain_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dem_windows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if dem_windows is not None:
            terrain_scores = self.terrain_proj(dem_windows).permute(0, 2, 1)
            terrain_bias = terrain_scores.unsqueeze(-1) + terrain_scores.unsqueeze(-2)
            attn = attn + terrain_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class TGASwinBlock(nn.Module):
    """Swin Transformer block with Terrain-Guided Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dem_channels: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.dem_channels = dem_channels

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TerrainGuidedAttention(
            dim, window_size=window_size, num_heads=num_heads, dem_channels=dem_channels
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def _compute_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(
        self, x: torch.Tensor, H: int, W: int, dem_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        dem_windows_flat: Optional[torch.Tensor] = None
        if dem_feat is not None:
            dem_2d = dem_feat.permute(0, 2, 3, 1)
            if pad_r > 0 or pad_b > 0:
                dem_2d = F.pad(dem_2d, (0, 0, 0, pad_r, 0, pad_b))
            if self.shift_size > 0:
                dem_2d = torch.roll(dem_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            dem_wins = window_partition(dem_2d, self.window_size)
            dem_windows_flat = dem_wins.view(-1, self.window_size ** 2, self.dem_channels)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        mask = self._compute_mask(Hp, Wp, x.device)
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_windows = self.attn(x_windows, mask=mask, dem_windows=dem_windows_flat)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class PatchUnembed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x.transpose(1, 2).view(-1, self.embed_dim, H, W)


class RSTB_TGA(nn.Module):
    """Residual Swin Transformer Block with TGA in every attention layer."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        dem_channels: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TGASwinBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dem_channels=dem_channels,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(dim, dim)
        self.patch_unembed = PatchUnembed(dim)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        dem_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual_2d = self.patch_unembed(x, H, W)
        feat = x
        for blk in self.blocks:
            feat = blk(feat, H, W, dem_feat)
        feat_2d = self.patch_unembed(feat, H, W)
        feat_2d = self.conv(feat_2d) + residual_2d
        feat, H, W = self.patch_embed(feat_2d)
        return feat


class HATGeneratorTGA(nn.Module):
    """HAT-family generator with Terrain-Guided Attention.

    ERA5 LR (5 ch) goes through a standard Swin backbone; DEM is injected
    via per-head attention biases from three DEMEncoders at coarse/mid/fine
    scales.  Output: predicted wind speed (1 ch, 4x HR).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        g = cfg.model.generator
        in_ch      = g.in_channels
        dem_in_ch  = g.get("dem_enc_in_channels", 5)
        dem_enc_ch = g.get("dem_enc_channels", 64)
        embed_dim  = g.embed_dim
        depths     = list(g.depths)
        num_heads  = list(g.num_heads)
        window_size = g.window_size
        scale_factor = g.scale_factor

        assert scale_factor == 4
        assert len(depths) == 4

        self.dem_enc_coarse = DEMEncoder(in_channels=dem_in_ch, out_channels=dem_enc_ch)
        self.dem_enc_mid    = DEMEncoder(in_channels=dem_in_ch, out_channels=dem_enc_ch)
        self.dem_enc_fine   = DEMEncoder(in_channels=dem_in_ch, out_channels=dem_enc_ch)

        self.conv_first = nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1)

        self.patch_embed = PatchEmbed(embed_dim, embed_dim)
        self.rstb_layers = nn.ModuleList([
            RSTB_TGA(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                dem_channels=dem_enc_ch,
            )
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_last = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)

        self._init_weights()
        self._zero_terrain_proj()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _zero_terrain_proj(self) -> None:
        for m in self.modules():
            if isinstance(m, TerrainGuidedAttention):
                nn.init.zeros_(m.terrain_proj.weight)
                nn.init.zeros_(m.terrain_proj.bias)

    def _encode_dem_multiscale(
        self, dem: torch.Tensor, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dem_half = F.interpolate(dem, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
        cond_coarse = self.dem_enc_coarse(dem_half)
        cond_coarse = F.interpolate(cond_coarse, size=(H, W), mode="bilinear", align_corners=False)

        dem_lr = F.interpolate(dem, size=(H, W), mode="bilinear", align_corners=False)
        cond_mid = self.dem_enc_mid(dem_lr)

        cond_fine_hr = self.dem_enc_fine(dem)
        cond_fine = F.interpolate(cond_fine_hr, size=(H, W), mode="bilinear", align_corners=False)

        return cond_coarse, cond_mid, cond_fine

    def forward(self, lr: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        B, C, H, W = lr.shape
        cond_coarse, cond_mid, cond_fine = self._encode_dem_multiscale(dem, H, W)
        dem_per_rstb = [cond_coarse, cond_mid, cond_fine, cond_fine]

        x = self.conv_first(lr)
        shallow = x

        feat, fH, fW = self.patch_embed(x)
        for i, rstb in enumerate(self.rstb_layers):
            feat = rstb(feat, fH, fW, dem_per_rstb[i])
        feat = self.norm(feat)
        deep = self.patch_unembed(feat, fH, fW)
        deep = self.conv_after_body(deep) + shallow

        out = self.upsample(deep)
        return self.conv_last(out)


@register_model("hat_tga")
class HATWithTGA(nn.Module):
    """HAT+TGA GAN wrapper — terrain-conditioned wind speed super-resolution."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator    = HATGeneratorTGA(cfg)
        self.discriminator = PatchDiscriminator(cfg)
        self.loss_fn      = WindSRLoss(cfg)

        n_params = sum(p.numel() for p in self.generator.parameters())
        logger.info(f"HATWithTGA initialised. Generator params: {n_params:,}")

    def forward(self, lr: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        return self.generator(lr, dem)

    def generator_step(
        self,
        batch: dict,
        elev_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        lr, dem, target = batch["lr"], batch["dem"], batch["hr_speed"]
        pred = self.generator(lr, dem)
        with torch.no_grad():
            fake_logits = self.discriminator(pred)
        loss_g, log_d = self.loss_fn(pred, target, fake_logits, elev_mask=elev_mask)
        return loss_g, log_d, pred

    def discriminator_step(
        self, batch: dict, pred: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        target = batch["hr_speed"]
        real_logits = self.discriminator(target)
        fake_logits = self.discriminator(pred.detach())
        loss_d = adversarial_loss_discriminator(real_logits, fake_logits)
        return loss_d, {"loss_d": loss_d.item()}
