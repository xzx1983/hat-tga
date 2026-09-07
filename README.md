# HAT+TGA: Terrain-Guided Attention for Wind Speed Downscaling

Official code for **"Terrain-Guided Attention Improves Deep Learning Wind Speed Downscaling over Complex Terrain"** (*npj Climate and Atmospheric Science*, under review).

HAT+TGA injects DEM-derived terrain features directly into every window-attention layer of a Swin-family backbone via an outer-sum additive bias. This lets the model learn which grid-cell pairs interact strongly given the intervening terrain, without concatenating DEM at the input stem.

**Key results (Sichuan, China, test 2023–2024):**
- Grid-scale RMSE: 0.726 m/s (−38.3% vs bicubic, −0.3% vs HAT+DEM)
- Station-scale RMSE: 0.880 m/s (−23% vs bicubic, −28% vs HAT+DEM)
- Only model improving both grid-scale and station-scale accuracy simultaneously

---

## Requirements

- Python 3.11+
- CUDA GPU (trained on NVIDIA A100 80GB; inference runs on any CUDA GPU)
- [uv](https://docs.astral.sh/uv/) for environment management

## Installation

```bash
git clone https://github.com/xzx1983/hat-tga.git
cd hat-tga
uv sync
```

`uv sync` installs all dependencies including PyTorch with CUDA 12.1 (configured in `pyproject.toml`).
For CPU-only machines, remove the `[[tool.uv.index]]` block from `pyproject.toml` before running `uv sync`.

---

## Data Preparation

HAT+TGA uses three data sources. CLDAS V2.0 must be obtained directly from
the China Meteorological Administration (CMA); ERA5 and SRTM are publicly accessible.

### Directory layout

```
data/
  raw/
    era5/           # ERA5 quarterly .nc files (from CDS API)
    cldas/          # CLDAS V2.0 hourly .nc files (from CMA)
      2017/
      2019/
      2020/
      ...
    dem/            # SRTM 30m tiles
    stations/       # GSOD station CSVs
  processed/        # created by process_data.py
```

### Step 1 — ERA5 (ECMWF Copernicus)

Register at https://cds.climate.copernicus.eu/ and save your API key to `~/.cdsapirc`.

```bash
uv pip install hat-tga[data]
uv run python scripts/prepare_data/download_era5.py
```

Downloads `u10, v10, sp, t2m, blh` at 0.25° hourly for 2017–2024
(~33 MB per quarter, ~68 files total).

### Step 2 — CLDAS V2.0 (China Meteorological Administration)

CLDAS V2.0 is available via the CMA Data Service Centre:
https://data.cma.cn/

Request the hourly 10m wind speed product (`HOR-WIN`) for 2017–2024
(excluding 2018, which is in MICAPS binary format). Place the `.nc` files
under `data/raw/cldas/YYYY/`.

### Step 3 — SRTM 30m DEM

```bash
# Using NASA earthaccess (requires free account at https://urs.earthdata.nasa.gov/)
export EARTHDATA_USERNAME=<your-username>
export EARTHDATA_PASSWORD=<your-password>
uv run python scripts/prepare_data/download_dem.py --method earthaccess
```

### Step 4 — GSOD station observations (optional, needed for station loss)

```bash
uv run python scripts/prepare_data/download_stations.py
```

### Step 5 — Pre-process all data

```bash
uv run python scripts/prepare_data/process_data.py
```

Produces `data/processed/era5/era5_YYYY.nc`, `data/processed/cldas/cldas_YYYY.nc`,
`data/processed/dem/dem_features_0.0625deg.nc`, and `data/processed/stats.json`.

---

## Training

```bash
uv run python scripts/train.py

# Override any config value via Hydra
uv run python scripts/train.py trainer.max_epochs=100 trainer.batch_size=4

# Resume from checkpoint
uv run python scripts/train.py ckpt_path=outputs/checkpoints/hat_tga/last.ckpt
```

Checkpoints are saved to `outputs/checkpoints/hat_tga/`.
TensorBoard logs are saved to `outputs/logs/`.

Training configuration: `conf/training/hat_tga.yaml`

---

## Evaluation

```bash
uv run python scripts/evaluate.py ckpt_path=<path/to/best.ckpt>
```

Outputs `outputs/eval_results.json` with RMSE, MAE, Bias, and Pearson r
on the test set (2023–2024).

---

## Figure Generation

```bash
uv run python scripts/generate_figures.py ckpt_path=<path/to/best.ckpt>
```

Saves HAT+TGA analysis figures to `outputs/figures/`:
- `fig1_rmse_map.png` — spatial RMSE map (HAT+TGA, test 2023–2024)
- `fig2_bias_map.png` — spatial bias map (HAT+TGA)
- `fig4_seasonal_bar.png` — seasonal RMSE bar chart (HAT+TGA)
- `fig5_example_day.png` — example prediction vs CLDAS reference

These are single-model figures for HAT+TGA. The manuscript comparison figures
(multi-panel with Bicubic, SRResNet, HAT+DEM, etc.) require baseline model
outputs not included here. The RMSE values match the manuscript's HAT+TGA
panel (0.726 m/s overall).

---

## Model Architecture

```
ERA5 LR (5 ch, 32×48)
       │
  Conv stem (embed_dim=180)
       │
  ┌────┴────┐
  │ RSTB×4  │   ← each block: 6 × TGASwinBlock (window=8, alternating shift)
  │  + TGA  │   ← DEM bias injected at every attention layer
  └────┬────┘
       │                    SRTM DEM (5 ch, 128×192)
       │                         │
       │              DEMEncoder × 3 (coarse/mid/fine)
       │              → terrain_proj → outer-sum bias B[h,i,j]
       │
  PixelShuffle ×2 ×2  (4× upsampling)
       │
  Wind speed (1 ch, 128×192)
```

**Terrain bias:** For window tokens i,j and head h:
```
B[h,i,j] = terrain_proj(dem_feat)[b,h,i] + terrain_proj(dem_feat)[b,h,j]
```
`terrain_proj` is zero-initialized, so at epoch 0 TGA = standard WindowAttention.

**Parameters:** 14.86M (generator) + 2.76M (discriminator)

---

## Temporal Split

| Split | Years | Samples |
|-------|-------|---------|
| Train | 2017, 2019, 2020, 2021 | ~28,000 h |
| Val   | 2022 | ~8,700 h |
| Test  | 2023–2024 | ~17,500 h |

2018 is excluded: CLDAS 2018 is in MICAPS binary format.

---

## Citation

```bibtex
@article{xu2025hattga,
  title   = {Terrain-Guided Attention for Wind Speed Downscaling Simultaneously
             Improves Grid-Scale and Station-Scale Accuracy over Complex Terrain},
  author  = {Xu, Zi-Xin and Xie, Shengcai and Yang, Ling and Yang, Zhipeng},
  journal = {npj Climate and Atmospheric Science},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT
