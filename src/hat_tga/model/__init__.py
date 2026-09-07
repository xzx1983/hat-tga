"""Model registry for HAT+TGA."""

from typing import Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def ModelFactory(name: str) -> Type[nn.Module]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]


# Register all models
from hat_tga.model import hat_tga  # noqa: F401, E402

__all__ = ["ModelFactory", "register_model", "MODEL_REGISTRY"]
