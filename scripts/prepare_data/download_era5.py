"""Download ERA5 10m wind + auxiliary variables for the Sichuan study region.

Downloads in quarterly batches to reduce API calls. Requires a CDS API key
(see https://cds.climate.copernicus.eu/ for registration).

Data spec:
  Source    : ERA5 single levels (ECMWF Copernicus)
  Variables : u10, v10, sp, t2m, blh
  Region    : 26°N–34°N, 97°E–109°E
  Period    : 2017–2024  (2018 processed separately — see NOTES)
  Temporal  : hourly
  Output    : data/raw/era5/era5_YYYY_QN.nc

NOTES:
  - 2018 CLDAS data is in MICAPS binary format and not included in training.
    You may still download ERA5 2018 for completeness.
  - Set up your CDS API key at ~/.cdsapirc:
      url: https://cds.climate.copernicus.eu/api/v2
      key: <UID>:<API-KEY>

Usage:
    uv run python scripts/prepare_data/download_era5.py
    uv run python scripts/prepare_data/download_era5.py --start 2020 --end 2021
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

NORTH, SOUTH = 34.0, 26.0
WEST, EAST   = 97.0, 109.0
AREA = [NORTH, WEST, SOUTH, EAST]

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "2m_temperature",
    "boundary_layer_height",
]

QUARTERS = [
    (1, [1, 2, 3]),
    (2, [4, 5, 6]),
    (3, [7, 8, 9]),
    (4, [10, 11, 12]),
]

OUT_DIR = Path("data/raw/era5")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_quarter(client, year: int, qnum: int, months: list) -> None:
    out_path = OUT_DIR / f"era5_{year}_Q{qnum}.nc"
    if out_path.exists():
        logger.info(f"  {out_path.name} already exists, skipping.")
        return

    logger.info(f"  Requesting {year} Q{qnum} ({months}) ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
        },
        str(out_path),
    )
    logger.info(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end",   type=int, default=2024)
    args = parser.parse_args()

    try:
        import cdsapi
    except ImportError:
        logger.error("cdsapi not installed. Run: uv pip install cdsapi  (or add to [data] extras)")
        return

    client = cdsapi.Client()
    for year in range(args.start, args.end + 1):
        logger.info(f"Year {year}")
        for qnum, months in QUARTERS:
            download_quarter(client, year, qnum, months)

    logger.info("ERA5 download complete.")
    logger.info("Next step: run scripts/prepare_data/process_data.py")


if __name__ == "__main__":
    main()
