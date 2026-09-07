"""Download GSOD (Global Summary of the Day) station observations for Sichuan.

Downloads daily wind speed from NOAA GSOD for 43 stations within the study
domain (26°N–34°N, 97°E–109°E), covering 2017–2024.

Source: NOAA Global Surface Summary of Day
        https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day

Output:
    data/raw/stations/sichuan_stations.csv  — station metadata (USAF, LAT, LON)
    data/raw/stations/<USAF>99999.csv       — daily obs per station (DATE, wdsp_ms)

CLDAS is the primary HR reference. Station data is used only for:
  1. The station-observation loss term during training (L_station).
  2. Independent point-scale validation at the 43 station locations.

Usage:
    uv run python scripts/prepare_data/download_stations.py
    uv run python scripts/prepare_data/download_stations.py --start 2020 --end 2022
"""

import argparse
import io
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

LAT_MIN, LAT_MAX = 26.0, 34.0
LON_MIN, LON_MAX = 97.0, 109.0

OUT_DIR = Path("data/raw/stations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOAA_GSOD_BASE = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"

WDSP_MISSING = 999.9


def knots_to_ms(knots: float) -> float:
    return knots * 0.51444


def fetch_station_year(usaf: str, year: int) -> pd.DataFrame | None:
    sid = f"{usaf}99999"
    url = f"{NOAA_GSOD_BASE}/{year}/{sid}.csv"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            df = pd.read_csv(io.BytesIO(r.read()), low_memory=False)
        if "WDSP" not in df.columns:
            return None
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["wdsp_ms"] = df["WDSP"].replace(WDSP_MISSING, np.nan).apply(
            lambda x: knots_to_ms(float(x)) if not pd.isna(x) else np.nan
        )
        return df[["DATE", "wdsp_ms"]].dropna(subset=["wdsp_ms"])
    except Exception:
        return None


def discover_stations() -> pd.DataFrame:
    """Load station inventory and filter to the study domain."""
    inventory_url = (
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
    )
    logger.info("Downloading station inventory ...")
    meta = pd.read_csv(inventory_url, low_memory=False)
    meta.columns = meta.columns.str.strip()
    meta = meta.rename(columns={"USAF": "USAF", "LAT": "LAT", "LON": "LON"})
    meta = meta.dropna(subset=["LAT", "LON"])
    meta = meta[
        (meta["LAT"] >= LAT_MIN) & (meta["LAT"] <= LAT_MAX) &
        (meta["LON"] >= LON_MIN) & (meta["LON"] <= LON_MAX)
    ]
    meta = meta[meta["END"].astype(str) >= "20170101"]
    logger.info(f"Found {len(meta)} stations in domain")
    return meta[["USAF", "LAT", "LON", "STATION NAME"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end",   type=int, default=2024)
    args = parser.parse_args()

    stations = discover_stations()
    stations.to_csv(OUT_DIR / "sichuan_stations.csv", index=False)
    logger.info(f"Station metadata saved: {len(stations)} stations")

    for _, row in stations.iterrows():
        usaf = f"{int(row['USAF']):06d}"
        sid  = f"{usaf}99999"
        out_path = OUT_DIR / f"{sid}.csv"
        if out_path.exists():
            logger.info(f"  {sid}: already exists, skipping")
            continue

        yearly = []
        for year in range(args.start, args.end + 1):
            df = fetch_station_year(usaf, year)
            if df is not None:
                yearly.append(df)
            time.sleep(0.1)

        if yearly:
            pd.concat(yearly).to_csv(out_path, index=False)
            logger.info(f"  {sid}: {sum(len(d) for d in yearly)} daily records saved")
        else:
            logger.warning(f"  {sid}: no data found")

    logger.info("Station download complete.")


if __name__ == "__main__":
    main()
