"""Download SRTM 30m DEM tiles for the Sichuan study region.

Covers 26°N–34°N, 97°E–109°E (64 one-degree tiles).

Two download methods:
  Method A (recommended): NASA earthaccess (requires free Earthdata account)
  Method B (fallback)   : OpenTopography API (requires free API key)

Prerequisites (Method A):
    Register at https://urs.earthdata.nasa.gov/
    uv pip install earthaccess
    Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD as environment variables.

Prerequisites (Method B):
    Register at https://portal.opentopography.org/
    Set OPENTOPO_API_KEY as an environment variable.

Usage:
    uv run python scripts/prepare_data/download_dem.py --method earthaccess
    uv run python scripts/prepare_data/download_dem.py --method opentopo

After download, run scripts/prepare_data/process_data.py --dem-only to
compute slope, aspect, and TRI and save dem_features_0.0625deg.nc.
"""

import argparse
import logging
import os
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

LAT_MIN, LAT_MAX = 26, 34
LON_MIN, LON_MAX = 97, 109

OUT_DIR = Path("data/raw/dem/tiles")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_earthaccess():
    try:
        import earthaccess
    except ImportError:
        logger.error("earthaccess not installed. Run: uv pip install earthaccess")
        return

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name="SRTMGL1",
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    logger.info(f"Found {len(results)} SRTM tiles")
    earthaccess.download(results, str(OUT_DIR))
    logger.info(f"Tiles saved to {OUT_DIR}")


def download_opentopo():
    api_key = os.environ.get("OPENTOPO_API_KEY", "")
    if not api_key:
        logger.error("Set OPENTOPO_API_KEY environment variable.")
        return

    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL1&south={LAT_MIN}&north={LAT_MAX}"
        f"&west={LON_MIN}&east={LON_MAX}&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    out_path = OUT_DIR / "srtm_sichuan.tif"
    logger.info(f"Downloading merged SRTM tile from OpenTopography ...")
    urllib.request.urlretrieve(url, out_path)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["earthaccess", "opentopo"], default="earthaccess")
    args = parser.parse_args()

    if args.method == "earthaccess":
        download_earthaccess()
    else:
        download_opentopo()

    logger.info("DEM download complete.")
    logger.info("Next step: run scripts/prepare_data/process_data.py --dem-only")


if __name__ == "__main__":
    main()
