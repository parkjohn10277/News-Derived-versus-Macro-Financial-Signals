"""
Model registry.  A spec module exposes:

    NAME                          str
    GRID                          dict of hyperparameter lists for RandomizedSearchCV
    estimator(seed, pos_weight)   a fresh sklearn-compatible CLASSIFIER

To add a model, drop a module here and register it below.
"""
from . import lightgbm_model, logistic, random_forest, xgboost_model

REGISTRY = {
    "random_forest": random_forest,
    "xgboost": xgboost_model,
    "lightgbm": lightgbm_model,
    "logistic": logistic,
}


def get_spec(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name}; registered: {list(REGISTRY)}")
    return REGISTRY[name]
