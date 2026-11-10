"""
Random forest CLASSIFIER (direction of foreign net buying).

Grid follows Chun et al. (2024) Table 8: max depth 5-15, max features 10-100.
Their 10-100 is out of 137 predictors (7%-73%), so it is expressed here as a
fraction to scale with each experiment's feature count.  n_estimators and
min_samples_leaf are not listed in Table 8; sensible values are searched.

Class imbalance is handled by class_weight="balanced", the sklearn analogue of
the neural folders' pos_weight = n_neg / n_pos.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

NAME = "random_forest"

GRID = {
    "max_depth": [5, 7, 10, 12, 15],
    "max_features": [0.1, 0.2, 0.3, 0.5, 0.7],
    "n_estimators": [200, 500],
    "min_samples_leaf": [1, 2, 5, 10],
}


def estimator(seed: int, pos_weight: float = 1.0) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
