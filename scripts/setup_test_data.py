"""Set up data symlinks for testing.

Creates symlinks from hat-tga/data/ → the sibling Project_Wind/data/
so tests can run without copying large NetCDF files.

Usage (from hat-tga/ directory):
    uv run python scripts/setup_test_data.py

To use custom data paths instead of symlinking from Project_Wind:
    uv run python scripts/setup_test_data.py --data-root /path/to/your/data

After running this script, the data layout will be:
    data/processed/era5/        → ../data/processed/era5/
    data/processed/cldas/       → ../data/processed/cldas/
    data/processed/dem/         → ../data/processed/dem/
    data/raw/stations/          → ../data/raw/stations/
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Default: look for Project_Wind/data/ one level above hat-tga/
DEFAULT_SRC = REPO_ROOT.parent / "data"

LINKS = [
    # (relative path inside hat-tga/data/, sub-path inside source data root)
    ("processed/era5",    "processed/era5"),
    ("processed/cldas",   "processed/cldas"),
    ("processed/dem",     "processed/dem"),
    ("raw/stations",      "raw/stations"),
]


def make_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            existing = link_path.resolve()
            if existing == target.resolve():
                print(f"  OK (already linked): {link_path} → {target}")
                return
            link_path.unlink()
            print(f"  Updated: {link_path} → {target}")
        else:
            print(f"  SKIP (real directory exists): {link_path}")
            return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target.resolve())
    print(f"  Linked: {link_path} → {target.resolve()}")


def verify_links() -> bool:
    ok = True
    required = [
        REPO_ROOT / "data" / "processed" / "era5"  / "era5_2023.nc",
        REPO_ROOT / "data" / "processed" / "era5"  / "era5_2024.nc",
        REPO_ROOT / "data" / "processed" / "cldas" / "cldas_2023.nc",
        REPO_ROOT / "data" / "processed" / "cldas" / "cldas_2024.nc",
        REPO_ROOT / "data" / "processed" / "dem"   / "dem_features_0.0625deg.nc",
    ]
    print("\nVerifying required test files:")
    for p in required:
        exists = p.exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {p.relative_to(REPO_ROOT)}")
        if not exists:
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_SRC),
        help=f"Source data root directory (default: {DEFAULT_SRC})",
    )
    args = parser.parse_args()

    src = Path(args.data_root)
    if not src.exists():
        print(f"ERROR: Source data root not found: {src}")
        print(
            "Either:\n"
            "  1. Run from inside the Project_Wind repository (parent of hat-tga/)\n"
            "  2. Pass --data-root /path/to/your/processed/data"
        )
        sys.exit(1)

    print(f"Setting up data symlinks from: {src}")
    for link_rel, src_rel in LINKS:
        link_path = REPO_ROOT / "data" / link_rel
        target    = src / src_rel
        if not target.exists():
            print(f"  WARNING: source not found, skipping: {target}")
            continue
        make_symlink(link_path, target)

    all_ok = verify_links()
    if all_ok:
        print("\nAll test data files are accessible. Ready to run:")
        print("  uv run pytest tests/ -v")
    else:
        print("\nSome files are missing. Check the source data root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
