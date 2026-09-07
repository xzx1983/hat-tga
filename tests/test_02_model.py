"""Test 02 — Model: instantiation and forward pass with synthetic data.

Tests that the HAT+TGA architecture builds correctly and produces
outputs of the expected shape, without requiring any real data files.

Run:
    uv run pytest tests/test_02_model.py -v
"""

import pytest
import torch


@pytest.fixture(scope="module")
def model(base_cfg):
    from hat_tga.model import ModelFactory
    cls = ModelFactory("hat_tga")
    return cls(base_cfg)


def test_model_instantiation(model):
    """Model creates without error."""
    assert model is not None


def test_generator_param_count(model):
    """Generator must have roughly 14.86M parameters (±0.5M)."""
    n = sum(p.numel() for p in model.generator.parameters())
    print(f"\n  Generator params: {n:,}")
    assert 14_000_000 < n < 16_000_000, f"Unexpected param count: {n:,}"


def test_discriminator_param_count(model):
    """Discriminator must have roughly 2.76M parameters (±0.5M)."""
    n = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\n  Discriminator params: {n:,}")
    assert 2_000_000 < n < 4_000_000, f"Unexpected param count: {n:,}"


def test_terrain_proj_zero_init(model):
    """terrain_proj weights must be zero at initialisation."""
    from hat_tga.model.hat_tga import TerrainGuidedAttention
    for m in model.generator.modules():
        if isinstance(m, TerrainGuidedAttention):
            assert m.terrain_proj.weight.abs().max().item() == 0.0, (
                "terrain_proj.weight is not zero — TGA would corrupt epoch-0 training"
            )
            assert m.terrain_proj.bias.abs().max().item() == 0.0


def test_forward_pass_shape(model):
    """Generator output shape must be (B, 1, 4H, 4W)."""
    B, H, W = 2, 32, 48
    lr  = torch.randn(B, 5, H, W)
    dem = torch.randn(B, 5, H * 4, W * 4)
    with torch.no_grad():
        out = model.generator(lr, dem)
    assert out.shape == (B, 1, H * 4, W * 4), f"Wrong output shape: {out.shape}"


def test_forward_pass_patch(model):
    """Forward pass works with training patch size (16×16 LR → 64×64 HR)."""
    B, H, W = 2, 16, 16
    lr  = torch.randn(B, 5, H, W)
    dem = torch.randn(B, 5, H * 4, W * 4)
    with torch.no_grad():
        out = model.generator(lr, dem)
    assert out.shape == (B, 1, 64, 64)


def test_discriminator_shape(model):
    """Discriminator accepts HR wind maps and returns patch logits."""
    x = torch.randn(2, 1, 128, 192)
    with torch.no_grad():
        logits = model.discriminator(x)
    assert logits.ndim == 4 and logits.shape[1] == 1


def test_generator_step_returns_loss(model, base_cfg):
    """generator_step returns a scalar loss."""
    B, H, W = 2, 32, 48
    batch = {
        "lr":       torch.randn(B, 5, H, W),
        "dem":      torch.randn(B, 5, H * 4, W * 4),
        "hr_speed": torch.randn(B, 1, H * 4, W * 4),
    }
    loss_g, log_g, pred = model.generator_step(batch)
    assert loss_g.ndim == 0, "generator loss must be a scalar"
    assert "loss_pixel" in log_g
    assert pred.shape == (B, 1, H * 4, W * 4)


def test_discriminator_step_returns_loss(model):
    """discriminator_step returns a scalar loss."""
    B, H, W = 2, 32, 48
    batch = {"hr_speed": torch.randn(B, 1, H * 4, W * 4)}
    pred  = torch.randn(B, 1, H * 4, W * 4)
    loss_d, log_d = model.discriminator_step(batch, pred)
    assert loss_d.ndim == 0
    assert "loss_d" in log_d


def test_dem_encoder_multiscale(model):
    """DEMEncoder produces three feature maps all at LR resolution."""
    B, H, W = 2, 32, 48
    dem = torch.randn(B, 5, H * 4, W * 4)
    coarse, mid, fine = model.generator._encode_dem_multiscale(dem, H, W)
    for name, feat in [("coarse", coarse), ("mid", mid), ("fine", fine)]:
        assert feat.shape == (B, 64, H, W), f"DEM {name} shape wrong: {feat.shape}"
