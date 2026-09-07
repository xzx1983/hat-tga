"""Wind super-resolution dataset.

Loads from annual processed NetCDF files produced by scripts/prepare_data/process_data.py.

Batch dict keys:
    "lr"           (5, H, W)       ERA5 LR, z-score normalised
    "dem"          (5, 4H, 4W)     SRTM DEM, z-score normalised
    "hr_speed"     (1, 4H, 4W)     CLDAS wind speed, normalised  (supervised target)
    "hr_speed_raw" (1, 4H, 4W)     CLDAS wind speed, raw m/s     (metric computation)
    "hr_u"         (1, 4H, 4W)     derived HR u-component, raw m/s
    "hr_v"         (1, 4H, 4W)     derived HR v-component, raw m/s
    "lr_bicubic"   (1, 4H, 4W)     bicubic ERA5 speed upsampled, normalised
    "station_mask" (1, 4H, 4W)     station grid mask (train only)
    "station_vals" (1, 4H, 4W)     station wind speed obs m/s (train only)
    "meta"         dict{year, era5_t, cldas_t}
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from omegaconf import DictConfig
from torch.utils.data import Dataset

from hat_tga.data import register_dataset

logger = logging.getLogger(__name__)

ERA5_VARS = ["u10", "v10", "sp", "t2m", "blh"]
DEM_VARS  = ["elevation", "slope", "sin_aspect", "cos_aspect", "tri"]

_HR_LAT = np.array([26.03125 + i * 0.0625 for i in range(128)], dtype=np.float32)
_HR_LON = np.array([97.03125 + j * 0.0625 for j in range(192)], dtype=np.float32)


@register_dataset("wind_sr")
class WindSRDataset(Dataset):
    """Wind SR dataset backed by annual processed NetCDF files.

    Args:
        cfg: Dataset config (OmegaConf), i.e. cfg.dataset from Hydra.
             See conf/dataset/wind_sr.yaml for all fields.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg   = cfg
        self.split = str(cfg.split)

        _year_map = {
            "train": list(cfg.train_years),
            "val":   list(cfg.val_years),
            "test":  list(cfg.test_years),
        }
        if self.split not in _year_map:
            raise ValueError(f"Unknown split '{self.split}'. Use train/val/test.")
        self.years = _year_map[self.split]

        self.lr_mean       = np.array(cfg.lr_mean,  dtype=np.float32)
        self.lr_std        = np.array(cfg.lr_std,   dtype=np.float32)
        self.dem_mean      = np.array(cfg.dem_mean, dtype=np.float32)
        self.dem_std       = np.array(cfg.dem_std,  dtype=np.float32)
        self.hr_speed_mean = float(cfg.hr_speed_mean)
        self.hr_speed_std  = float(cfg.hr_speed_std)

        self.hr_patch    = int(cfg.hr_patch_size)
        self.lr_patch    = int(cfg.lr_patch_size)
        self.full_domain = (self.split != "train")
        self.aug_flip_h  = (self.split == "train") and bool(
            cfg.augmentation.get("random_flip_h", False)
        )

        era5_dir  = Path(cfg.get("era5_dir",  "data/processed/era5"))
        cldas_dir = Path(cfg.get("cldas_dir", "data/processed/cldas"))
        dem_path  = Path(cfg.get("dem_path",  "data/processed/dem/dem_features_0.0625deg.nc"))

        self._dem = self._load_dem(dem_path)

        self._era5:       dict[int, np.ndarray] = {}
        self._era5_dates: dict[int, np.ndarray] = {}
        self._cldas:      dict[int, np.ndarray] = {}
        self.index:       list[tuple[int, int, int]] = []

        for year in self.years:
            self._load_year(year, era5_dir, cldas_dir)

        self._station_obs:  dict[str, dict[tuple, float]] = {}
        self._use_stn_loss = (
            self.split == "train" and bool(cfg.get("use_station_loss", False))
        )
        if self._use_stn_loss:
            stn_dir    = Path(cfg.get("station_dir", "data/raw/stations"))
            holdout_ids = set(cfg.get("station_holdout_ids", []))
            self._load_station_obs(stn_dir, holdout_ids=holdout_ids)
            logger.info(
                f"  Station loss enabled: {len(self._station_obs)} training days loaded. "
                f"Held out {len(holdout_ids)} station(s)."
            )

        logger.info(
            f"[WindSRDataset:{self.split}] {len(self.index)} samples "
            f"from years {self.years}"
        )

    def _load_dem(self, path: Path) -> np.ndarray:
        ds = xr.open_dataset(path)
        ds_crop = ds.sel(lat=slice(26.0, 34.0), lon=slice(97.0, 109.0))
        hr_lat = np.arange(0.03125, 65.0, 0.0625)
        hr_lon = np.arange(60.03125, 160.0, 0.0625)
        hr_lat = hr_lat[(hr_lat >= 26) & (hr_lat <= 34)]
        hr_lon = hr_lon[(hr_lon >= 97) & (hr_lon <= 109)]
        ds_crop = ds_crop.reindex(lat=hr_lat, lon=hr_lon, method="nearest", tolerance=0.04)
        arrays = [ds_crop[v].values.astype(np.float32) for v in DEM_VARS]
        ds.close()
        dem = np.stack(arrays, axis=0)
        assert dem.shape == (5, 128, 192), f"Unexpected DEM shape: {dem.shape}"
        return dem

    def _load_station_obs(self, stn_dir: Path, holdout_ids: Optional[set] = None) -> None:
        meta_path = stn_dir / "sichuan_stations.csv"
        if not meta_path.exists():
            logger.warning("sichuan_stations.csv not found; station loss disabled.")
            self._use_stn_loss = False
            return

        meta = pd.read_csv(meta_path)
        train_years_set = set(self.years)
        holdout_ids = holdout_ids or set()

        for _, row in meta.iterrows():
            sid = f"{int(row['USAF']):06d}99999"
            if sid in holdout_ids:
                continue
            fpath = stn_dir / f"{sid}.csv"
            if not fpath.exists():
                continue
            i = int(np.argmin(np.abs(_HR_LAT - float(row["LAT"]))))
            j = int(np.argmin(np.abs(_HR_LON - float(row["LON"]))))
            df = pd.read_csv(fpath, parse_dates=["DATE"])
            df = df[df["DATE"].dt.year.isin(train_years_set)]
            for _, srow in df.iterrows():
                speed = float(srow["wdsp_ms"])
                if np.isnan(speed) or speed < 0:
                    continue
                date_str = srow["DATE"].strftime("%Y-%m-%d")
                self._station_obs.setdefault(date_str, {})[(i, j)] = speed

    def _load_year(self, year: int, era5_dir: Path, cldas_dir: Path) -> None:
        era5_path  = era5_dir  / f"era5_{year}.nc"
        cldas_path = cldas_dir / f"cldas_{year}.nc"
        for p in (era5_path, cldas_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Processed file not found: {p}. "
                    "Run scripts/prepare_data/process_data.py first."
                )

        ds_e = xr.open_dataset(era5_path)
        era5_times = pd.DatetimeIndex(ds_e.time.values)
        era5_arr = np.stack(
            [ds_e[v].values.astype(np.float32) for v in ERA5_VARS], axis=1
        )
        ds_e.close()
        self._era5[year]       = era5_arr
        self._era5_dates[year] = np.array(
            [t.strftime("%Y-%m-%d") for t in era5_times], dtype=object
        )

        ds_c = xr.open_dataset(cldas_path)
        cldas_times = pd.DatetimeIndex(ds_c.time.values)
        cldas_arr   = ds_c["wind_speed"].values.astype(np.float32)
        ds_c.close()
        self._cldas[year] = cldas_arr

        common    = era5_times.intersection(cldas_times)
        era5_idx  = era5_times.get_indexer(common)
        cldas_idx = cldas_times.get_indexer(common)

        n_skipped = 0
        for e_i, c_i in zip(era5_idx, cldas_idx):
            if np.isnan(cldas_arr[c_i]).any():
                n_skipped += 1
                continue
            self.index.append((year, int(e_i), int(c_i)))
        if n_skipped:
            logger.warning(f"  [{self.split}] {year}: skipped {n_skipped} timesteps with NaN")
        logger.info(
            f"  [{self.split}] {year}: ERA5={len(era5_times)}h, "
            f"CLDAS={len(cldas_times)}h, common={len(common)}h"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        year, era5_ti, cldas_ti = self.index[idx]
        lr_raw  = self._era5[year][era5_ti].copy()
        ws_raw  = self._cldas[year][cldas_ti].copy()
        dem_raw = self._dem.copy()

        r0, c0 = 0, 0
        if not self.full_domain:
            r0, c0 = self._rand_patch_origin()
            lr_r, lr_c = r0 // 4, c0 // 4
            lr_raw  = lr_raw[:, lr_r:lr_r + self.lr_patch, lr_c:lr_c + self.lr_patch]
            ws_raw  = ws_raw[r0:r0 + self.hr_patch, c0:c0 + self.hr_patch]
            dem_raw = dem_raw[:, r0:r0 + self.hr_patch, c0:c0 + self.hr_patch]

        H_hr, W_hr = ws_raw.shape
        u10_lr = lr_raw[0]
        v10_lr = lr_raw[1]
        ws_era5_lr = np.sqrt(u10_lr**2 + v10_lr**2 + 1e-10)

        ws_lr_t    = torch.from_numpy(ws_era5_lr[None, None])
        lr_bicubic = F.interpolate(
            ws_lr_t, size=(H_hr, W_hr), mode="bicubic", align_corners=False
        ).clamp(min=0.0)[0].numpy()

        u10_hr = F.interpolate(
            torch.from_numpy(u10_lr[None, None]),
            size=(H_hr, W_hr), mode="bilinear", align_corners=False,
        )[0].numpy()
        v10_hr = F.interpolate(
            torch.from_numpy(v10_lr[None, None]),
            size=(H_hr, W_hr), mode="bilinear", align_corners=False,
        )[0].numpy()
        ws_era5_hr = np.sqrt(u10_hr**2 + v10_hr**2 + 1e-10)

        ws_cldas = ws_raw[None]
        hr_u = ws_cldas * (u10_hr / ws_era5_hr)
        hr_v = ws_cldas * (v10_hr / ws_era5_hr)

        lr_norm  = (lr_raw  - self.lr_mean[:, None, None])  / (self.lr_std[:, None, None]  + 1e-8)
        dem_norm = (dem_raw - self.dem_mean[:, None, None]) / (self.dem_std[:, None, None] + 1e-8)
        hr_norm  = (ws_cldas - self.hr_speed_mean) / (self.hr_speed_std + 1e-8)
        bic_norm = (lr_bicubic - self.hr_speed_mean) / (self.hr_speed_std + 1e-8)

        station_mask = np.zeros((1, H_hr, W_hr), dtype=np.float32)
        station_vals = np.zeros((1, H_hr, W_hr), dtype=np.float32)
        if self._use_stn_loss:
            date_str = self._era5_dates[year][era5_ti]
            for (gi, gj), speed in self._station_obs.get(date_str, {}).items():
                pi, pj = gi - r0, gj - c0
                if 0 <= pi < H_hr and 0 <= pj < W_hr:
                    station_mask[0, pi, pj] = 1.0
                    station_vals[0, pi, pj] = speed

        if self.aug_flip_h and np.random.random() > 0.5:
            lr_norm      = lr_norm[:, :, ::-1].copy()
            dem_norm     = dem_norm[:, :, ::-1].copy()
            hr_norm      = hr_norm[:, :, ::-1].copy()
            bic_norm     = bic_norm[:, :, ::-1].copy()
            hr_u         = hr_u[:, :, ::-1].copy()
            hr_v         = hr_v[:, :, ::-1].copy()
            ws_cldas     = ws_cldas[:, :, ::-1].copy()
            station_mask = station_mask[:, :, ::-1].copy()
            station_vals = station_vals[:, :, ::-1].copy()

        return {
            "lr":           torch.from_numpy(lr_norm.astype(np.float32)),
            "dem":          torch.from_numpy(dem_norm.astype(np.float32)),
            "hr_speed":     torch.from_numpy(hr_norm.astype(np.float32)),
            "hr_speed_raw": torch.from_numpy(ws_cldas.astype(np.float32)),
            "hr_u":         torch.from_numpy(hr_u.astype(np.float32)),
            "hr_v":         torch.from_numpy(hr_v.astype(np.float32)),
            "lr_bicubic":   torch.from_numpy(bic_norm.astype(np.float32)),
            "station_mask": torch.from_numpy(station_mask),
            "station_vals": torch.from_numpy(station_vals),
            "meta":         {"year": year, "era5_t": era5_ti, "cldas_t": cldas_ti},
        }

    def _rand_patch_origin(self) -> tuple[int, int]:
        lr_max_r = 32 - self.lr_patch
        lr_max_c = 48 - self.lr_patch
        lr_r = np.random.randint(0, lr_max_r + 1)
        lr_c = np.random.randint(0, lr_max_c + 1)
        return lr_r * 4, lr_c * 4

    def denormalize_hr_speed(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hr_speed_std + self.hr_speed_mean
