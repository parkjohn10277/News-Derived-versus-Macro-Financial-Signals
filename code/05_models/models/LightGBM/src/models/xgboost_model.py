"""
XGBoost CLASSIFIER (direction of foreign net buying).

Grid follows Chun et al. (2024) Table 8: learning rate 0.001-0.1, max depth
5-15.  Table 8 gives no n_estimators for XGB, so it is searched; subsample and
colsample_bytree are added for robustness on a short sample (set both to [1.0]
for strict replication).

`scale_pos_weight` = n_neg / n_pos on the TRAINING rows -- the XGBoost spelling
of the neural folders' pos_weight.
"""
from __future__ import annotations

from xgboost import XGBClassifier

NAME = "xgboost"

GRID = {
    "learning_rate": [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
    "max_depth": [5, 7, 10, 12, 15],
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def estimator(seed: int, pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         scale_pos_weight=float(pos_weight),
                         random_state=seed, n_jobs=1, verbosity=0)
