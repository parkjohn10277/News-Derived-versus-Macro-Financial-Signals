"""Model registry -- see README 'Adding a model'."""
from .darnn import build as build_darnn

REGISTRY = {"darnn": build_darnn}


def build_model(name: str, n_features: int, T: int, params: dict):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name}; registered: {list(REGISTRY)}")
    return REGISTRY[name](n_features, T, params)
