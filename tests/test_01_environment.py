"""Test 01 — Environment: verify all required packages are importable.

Run:
    uv run pytest tests/test_01_environment.py -v
"""

import importlib
import sys


REQUIRED = [
    ("torch",              "2.0"),
    ("pytorch_lightning",  "2.0"),
    ("omegaconf",          "2.3"),
    ("hydra",              None),
    ("numpy",              "1.26"),
    ("pandas",             "2.0"),
    ("xarray",             None),
    ("netCDF4",            None),
    ("matplotlib",         "3.7"),
    ("tqdm",               None),
    ("tensorboard",        None),
]

DATA_EXTRAS = [
    "cdsapi",
    "rasterio",
    "rioxarray",
    "scipy",
]


def _version_ok(pkg_name: str, min_version: str | None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(pkg_name)
    except ImportError as e:
        return False, str(e)
    if min_version is None:
        return True, getattr(mod, "__version__", "unknown")
    installed = getattr(mod, "__version__", "0.0.0").split(".")
    required  = min_version.split(".")
    try:
        ok = tuple(int(x) for x in installed[:2]) >= tuple(int(x) for x in required[:2])
    except ValueError:
        ok = True  # non-numeric version; trust the user
    return ok, getattr(mod, "__version__", "unknown")


def test_python_version():
    assert sys.version_info >= (3, 11), (
        f"Python 3.11+ required, got {sys.version}"
    )


def test_core_packages():
    failures = []
    for pkg, min_ver in REQUIRED:
        ok, ver = _version_ok(pkg, min_ver)
        if ok:
            print(f"  OK  {pkg} {ver}")
        else:
            failures.append(f"{pkg} (need >={min_ver}): {ver}")
    assert not failures, "Missing or outdated packages:\n" + "\n".join(failures)


def test_torch_cuda():
    """CUDA is not required for tests, but log whether it is available."""
    import torch
    print(f"\n  torch version : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")


def test_data_extras_optional():
    """[data] extras are optional — warn if missing, don't fail."""
    missing = []
    for pkg in DATA_EXTRAS:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        import warnings
        warnings.warn(
            f"Optional [data] extras not installed: {missing}. "
            "Run `uv pip install hat-tga[data]` if you need to pre-process raw data.",
            stacklevel=2,
        )


def test_hat_tga_importable():
    """The hat_tga package itself must be importable after `uv sync`."""
    import hat_tga
    assert hasattr(hat_tga, "__version__")

    from hat_tga.model import ModelFactory, MODEL_REGISTRY
    from hat_tga.data import DatasetFactory, DATASET_REGISTRY
    from hat_tga.trainer import TGATrainer

    assert "hat_tga" in MODEL_REGISTRY,  f"hat_tga not registered. Got: {list(MODEL_REGISTRY)}"
    assert "wind_sr" in DATASET_REGISTRY, f"wind_sr not registered. Got: {list(DATASET_REGISTRY)}"
    print(f"\n  Models:   {list(MODEL_REGISTRY.keys())}")
    print(f"  Datasets: {list(DATASET_REGISTRY.keys())}")
