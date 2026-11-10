"""
L2-regularised logistic regression CLASSIFIER (direction of foreign net buying).

The linear reference for the ten nonlinear models.  It goes through exactly the
same protocol as the tree models (src/protocol.py): same lagged design matrix,
Yeo-Johnson + standardisation and AUX z-clip fit on training rows only, the
penalty C chosen by validation log loss, refit on train+validation, one
prediction for the test block.

Class imbalance: class_weight="balanced", as for the random forest.
The lbfgs solution is deterministic, so every seed gives the same model.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

NAME = "logistic"

# inverse regularisation strength, 1e-4 ... 10 (half-decade steps)
GRID = {"C": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]}


def estimator(seed: int, pos_weight: float = 1.0) -> LogisticRegression:
    return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000,
                              class_weight="balanced", random_state=seed)
