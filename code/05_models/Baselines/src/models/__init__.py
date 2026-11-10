"""
Model registry for the baselines (same interface as Final_run/models/<tree>/src/models).

    NAME                          str
    GRID                          dict of hyperparameter lists for RandomizedSearchCV
    estimator(seed, pos_weight)   a fresh sklearn-compatible CLASSIFIER
"""
from . import logistic

REGISTRY = {"logistic": logistic}


def get_spec(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name}; registered: {list(REGISTRY)}")
    return REGISTRY[name]
