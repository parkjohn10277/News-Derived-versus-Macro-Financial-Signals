"""
LightGBM CLASSIFIER (direction of foreign net buying).

Grid follows Chun et al. (2024) Table 8: learning rate 0.001-0.1, max depth
10-100, number of estimators 100-1000.  num_leaves is not in Table 8 but
governs LightGBM capacity far more than max_depth, so a small range is searched.

`scale_pos_weight` = n_neg / n_pos on the TRAINING rows.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier

NAME = "lightgbm"

GRID = {
    "learning_rate": [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
    "max_depth": [10, 20, 30, 50, 100],
    "n_estimators": [100, 200, 500, 1000],
    "num_leaves": [15, 31, 63],
}


def estimator(seed: int, pos_weight: float = 1.0) -> LGBMClassifier:
    return LGBMClassifier(objective="binary", scale_pos_weight=float(pos_weight),
                          random_state=seed, n_jobs=1, verbosity=-1)
