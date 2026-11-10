"""Model registry -- see README 'Adding a model'."""
from .bilstm import build as build_bilstm

REGISTRY = {"bilstm": build_bilstm}


def build_model(name: str, n_features: int, T: int, params: dict):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name}; registered: {list(REGISTRY)}")
    return REGISTRY[name](n_features, T, params)
