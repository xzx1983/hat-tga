"""Dataset registry for HAT+TGA."""

from typing import Dict, Type

from torch.utils.data import Dataset

DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def DatasetFactory(name: str) -> Type[Dataset]:
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not registered. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


from hat_tga.data import dataset  # noqa: F401, E402
from hat_tga.data.datamodule import WindSRDataModule  # noqa: F401, E402

__all__ = ["DatasetFactory", "register_dataset", "WindSRDataModule", "DATASET_REGISTRY"]
