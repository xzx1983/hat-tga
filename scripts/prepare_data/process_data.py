"""Pre-process raw data into aligned NetCDF files for training.

Runs three pipelines (all by default, or selectively via flags):
  --dem-only  : SRTM tiles → dem_features_0.0625deg.nc
  --era5-only : ERA5 quarterly .nc → annual era5_YYYY.nc on a 32×48 LR grid
  --cldas     : CLDAS hourly .nc  → annual cldas_YYYY.nc on the 128×192 HR grid
                (CLDAS files must be obtained separately; see README)

Outputs:
  data/processed/dem/dem_features_0.0625deg.nc
  data/processed/era5/era5_YYYY.nc    (T, 5 vars, 32×48)
  data/processed/cldas/cldas_YYYY.nc  (T, 128×192)
  data/processed/stats.json           (normalisation stats from train years)

Usage:
    uv run python scripts/prepare_data/process_data.py          # all
    uv run python scripts/prepare_data/process_data.py --dem-only
    uv run python scripts/prepare_data/process_data.py --era5-only --years 2020 2021
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Canonical grids ────────────────────────────────────────────────────────────
_all_lat = np.arange(0.03125, 65.0, 0.0625)
_all_lon = np.arange(60.03125, 160.0, 0.0625)
HR_LAT = _all_lat[(_all_lat >= 26) & (_all_lat <= 34)]   # 128 pts
HR_LON = _all_lon[(_all_lon >= 97) & (_all_lon <= 109)]  # 192 pts
LR_LAT = np.array([(HR_LAT[i*4] + HR_LAT[i*4+3]) / 2 for i in range(len(HR_LAT) // 4)])
LR_LON = np.array([(HR_LON[j*4] + HR_LON[j*4+3]) / 2 for j in range(len(HR_LON) // 4)])

ERA5_VARS  = ["u10", "v10", "sp", "t2m", "blh"]
TRAIN_YEARS = [2017, 2019, 2020, 2021]

ROOT       = Path(".")
RAW_ERA5   = ROOT / "data/raw/era5"
RAW_CLDAS  = ROOT / "data/raw/cldas"
RAW_DEM    = ROOT / "data/raw/dem"
PROC_ERA5  = ROOT / "data/processed/era5"
PROC_CLDAS = ROOT / "data/processed/cldas"
PROC_DEM   = ROOT / "data/processed/dem"
PROC_ROOT  = ROOT / "data/processed"

for d in (PROC_ERA5, PROC_CLDAS, PROC_DEM):
    d.mkdir(parents=True, exist_ok=True)


# ── DEM processing ─────────────────────────────────────────────────────────────

def process_dem():
    """Merge SRTM tiles and compute terrain features at CLDAS 0.0625° resolution."""
    try:
        import rioxarray as rxr
        from scipy import ndimage
    except ImportError:
        logger.error("rioxarray and scipy are required. Run: uv pip install hat-tga[data]")
        return

    tile_files = sorted(RAW_DEM.rglob("*.tif")) + sorted(RAW_DEM.rglob("*.hgt"))
    if not tile_files:
        logger.error(f"No DEM tiles found in {RAW_DEM}. Run download_dem.py first.")
        return

    logger.info(f"Merging {len(tile_files)} DEM tiles ...")
    from rasterio.merge import merge
    import rasterio

    datasets = [rasterio.open(f) for f in tile_files]
    mosaic, transform = merge(datasets)
    for ds in datasets:
        ds.close()

    elev = mosaic[0].astype(np.float32)

    # Compute slope, aspect, TRI
    dz_dx = ndimage.sobel(elev, axis=1) / (8 * 90.0)
    dz_dy = ndimage.sobel(elev, axis=0) / (8 * 90.0)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.arctan2(-dz_dy, dz_dx)
    sin_aspect = np.sin(aspect)
    cos_aspect = np.cos(aspect)

    # TRI: std of 3×3 window
    from scipy.ndimage import generic_filter
    tri = generic_filter(elev, np.std, size=3)

    # Resample to 0.0625° grid using rioxarray
    logger.info("Resampling to 0.0625° grid ...")

    def to_da(arr, name):
        # Approximate lat/lon from mosaic shape
        h, w = arr.shape
        lat = np.linspace(34.0, 26.0, h)
        lon = np.linspace(97.0, 109.0, w)
        return xr.DataArray(arr, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name=name)

    target = xr.Dataset({"lat": HR_LAT, "lon": HR_LON})
    arrays = {}
    for name, arr in zip(
        ["elevation", "slope", "sin_aspect", "cos_aspect", "tri"],
        [elev, slope, sin_aspect, cos_aspect, tri],
    ):
        da = to_da(arr, name)
        arrays[name] = da.interp(lat=HR_LAT, lon=HR_LON, method="linear")

    ds_out = xr.Dataset(arrays)
    out_path = PROC_DEM / "dem_features_0.0625deg.nc"
    ds_out.to_netcdf(out_path)
    logger.info(f"Saved {out_path}")


# ── ERA5 processing ────────────────────────────────────────────────────────────

def process_era5_year(year: int):
    out_path = PROC_ERA5 / f"era5_{year}.nc"
    if out_path.exists():
        logger.info(f"  era5_{year}.nc exists, skipping.")
        return

    quarterly = sorted(RAW_ERA5.glob(f"era5_{year}_Q*.nc"))
    if not quarterly:
        logger.warning(f"  No quarterly ERA5 files found for {year}. Run download_era5.py first.")
        return

    logger.info(f"  Processing ERA5 {year} ({len(quarterly)} quarters) ...")
    ds = xr.open_mfdataset(quarterly, combine="by_coords")
    ds = ds.interp(latitude=LR_LAT, longitude=LR_LON, method="linear")
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds[[v for v in ERA5_VARS if v in ds]]
    ds.to_netcdf(out_path)
    ds.close()
    logger.info(f"  Saved {out_path}")


# ── CLDAS processing ───────────────────────────────────────────────────────────

_CLDAS_RE = re.compile(r"HOR-WIN-(\d{10})\.nc$")


def _parse_cldas_time(path: Path) -> pd.Timestamp | None:
    m = _CLDAS_RE.search(path.name)
    if not m:
        return None
    s = m.group(1)
    try:
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:00")
    except ValueError:
        return None


def process_cldas_year(year: int):
    out_path = PROC_CLDAS / f"cldas_{year}.nc"
    if out_path.exists():
        logger.info(f"  cldas_{year}.nc exists, skipping.")
        return

    cldas_year_dir = RAW_CLDAS / str(year)
    if not cldas_year_dir.exists():
        logger.warning(f"  CLDAS directory not found: {cldas_year_dir}")
        return

    files = sorted(cldas_year_dir.rglob("*.nc"))
    logger.info(f"  Processing CLDAS {year}: {len(files)} files ...")

    records = []
    for f in files:
        ts = _parse_cldas_time(f)
        if ts is None:
            continue
        try:
            ds = xr.open_dataset(f)
            # Expected variable name variations
            ws_var = next((v for v in ds.data_vars if "WIN" in v.upper() or "WS" in v.upper()), None)
            if ws_var is None:
                ds.close()
                continue
            arr = ds[ws_var].values.astype(np.float32)
            ds.close()
            records.append((ts, arr))
        except Exception as e:
            logger.debug(f"  Skipping {f.name}: {e}")

    if not records:
        logger.warning(f"  No valid CLDAS records for {year}.")
        return

    records.sort(key=lambda x: x[0])
    times = [r[0] for r in records]
    arrays = np.stack([r[1] for r in records], axis=0)

    # Crop to 128×192 HR grid
    target_h, target_w = len(HR_LAT), len(HR_LON)
    if arrays.shape[1:] != (target_h, target_w):
        arrays = arrays[:, :target_h, :target_w]

    ds_out = xr.Dataset(
        {"wind_speed": (["time", "lat", "lon"], arrays)},
        coords={"time": times, "lat": HR_LAT, "lon": HR_LON},
    )
    ds_out.to_netcdf(out_path)
    logger.info(f"  Saved {out_path}")


# ── Normalisation stats ────────────────────────────────────────────────────────

def compute_stats():
    stats_path = PROC_ROOT / "stats.json"
    if stats_path.exists():
        logger.info("stats.json exists, skipping.")
        return

    logger.info("Computing normalisation statistics from train years ...")
    era5_arrays, cldas_arrays = [], []

    for year in TRAIN_YEARS:
        p = PROC_ERA5 / f"era5_{year}.nc"
        if p.exists():
            ds = xr.open_dataset(p)
            era5_arrays.append(
                np.stack([ds[v].values for v in ERA5_VARS if v in ds], axis=1)
            )
            ds.close()
        p = PROC_CLDAS / f"cldas_{year}.nc"
        if p.exists():
            ds = xr.open_dataset(p)
            cldas_arrays.append(ds["wind_speed"].values)
            ds.close()

    if not era5_arrays:
        logger.warning("No ERA5 train data found; skipping stats.")
        return

    era5 = np.concatenate(era5_arrays, axis=0)   # (T, 5, H, W)
    cldas = np.concatenate(cldas_arrays, axis=0) # (T, H, W)

    stats = {
        "lr_mean":       era5.mean(axis=(0, 2, 3)).tolist(),
        "lr_std":        era5.std(axis=(0, 2, 3)).tolist(),
        "hr_speed_mean": float(cldas.mean()),
        "hr_speed_std":  float(cldas.std()),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved {stats_path}")
    logger.info(f"  lr_mean:  {stats['lr_mean']}")
    logger.info(f"  lr_std:   {stats['lr_std']}")
    logger.info(f"  hr_mean:  {stats['hr_speed_mean']:.4f}")
    logger.info(f"  hr_std:   {stats['hr_speed_std']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem-only",  action="store_true")
    parser.add_argument("--era5-only", action="store_true")
    parser.add_argument("--cldas",     action="store_true")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2017, 2025)))
    args = parser.parse_args()

    run_all = not (args.dem_only or args.era5_only or args.cldas)

    if run_all or args.dem_only:
        logger.info("=== DEM ===")
        process_dem()

    if run_all or args.era5_only:
        logger.info("=== ERA5 ===")
        for year in args.years:
            process_era5_year(year)

    if run_all or args.cldas:
        logger.info("=== CLDAS ===")
        for year in args.years:
            process_cldas_year(year)

    if run_all:
        logger.info("=== Stats ===")
        compute_stats()

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
