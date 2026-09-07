"""Test 04 — Figure generation: verify all four figure functions run and produce files.

Uses synthetic numpy arrays — no checkpoint or data files are required.
Array shapes match the real inference output: (N, 1, 128, 192).

Run:
    uv run pytest tests/test_04_figures.py -v
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


# ── Load the figures module from its file path ────────────────────────────────

def _load_generate_figures():
    """Load scripts/generate_figures.py as a module without Hydra side-effects."""
    path = Path(__file__).parent.parent / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gf():
    return _load_generate_figures()


# ── Synthetic data ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def arrays():
    """(8, 1, 128, 192) float32 arrays matching real inference output shape."""
    rng     = np.random.default_rng(42)
    preds   = rng.uniform(0.0, 5.0, (8, 1, 128, 192)).astype(np.float32)
    targets = rng.uniform(0.0, 5.0, (8, 1, 128, 192)).astype(np.float32)
    months  = np.array([1, 2, 3, 4, 6, 7, 9, 12])
    return preds, targets, months


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("figures")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fig_rmse_map(gf, arrays, out_dir):
    """fig_rmse_map creates fig1_rmse_map.png."""
    preds, targets, _ = arrays
    gf.fig_rmse_map(preds, targets, out_dir)
    f = out_dir / "fig1_rmse_map.png"
    assert f.exists() and f.stat().st_size > 1_000


def test_fig_bias_map(gf, arrays, out_dir):
    """fig_bias_map creates fig2_bias_map.png."""
    preds, targets, _ = arrays
    gf.fig_bias_map(preds, targets, out_dir)
    f = out_dir / "fig2_bias_map.png"
    assert f.exists() and f.stat().st_size > 1_000


def test_fig_seasonal_bar(gf, arrays, out_dir):
    """fig_seasonal_bar creates fig4_seasonal_bar.png."""
    preds, targets, months = arrays
    gf.fig_seasonal_bar(preds, targets, months, out_dir)
    f = out_dir / "fig4_seasonal_bar.png"
    assert f.exists() and f.stat().st_size > 1_000


def test_fig_example_day(gf, arrays, out_dir):
    """fig_example_day creates fig5_example_day.png."""
    preds, targets, _ = arrays
    gf.fig_example_day(preds, targets, out_dir, sample_idx=0)
    f = out_dir / "fig5_example_day.png"
    assert f.exists() and f.stat().st_size > 1_000


def test_no_fig3(out_dir):
    """fig3_elevation_corr.png is not produced (not in scope)."""
    assert not (out_dir / "fig3_elevation_corr.png").exists()
