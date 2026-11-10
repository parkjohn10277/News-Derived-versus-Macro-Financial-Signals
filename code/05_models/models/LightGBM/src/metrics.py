"""
Classification metrics for the DIRECTION of foreign net buying.

Target:  1 = net buy (y >= 0),  0 = net sell.

Skill reference
---------------
`Brier_Skill` is the classification analogue of R2_oos: it benchmarks the model
against a forecaster that always predicts the TRAINING base rate.

    Brier_Skill = 1 - Brier(model) / Brier(constant train base rate)

    0.00   no better than always predicting the historical base rate
    > 0    genuine skill
    < 0    worse than the base rate

`LogLoss_Skill` is the same idea under log loss.

Chance references: MCC 0, AUC 0.5, Balanced_Accuracy 0.5.  Raw
Directional_Accuracy is base-rate contaminated, so `base_rate_test` is reported
beside it and should always be quoted with it.
"""
from __future__ import annotations

import numpy as np


def brier_skill_score(y_true, prob, base_rate: float) -> float:
    y_true, prob = np.asarray(y_true, float), np.asarray(prob, float)
    num = np.mean((prob - y_true) ** 2)
    den = np.mean((base_rate - y_true) ** 2)
    return float(1.0 - num / den) if den > 0 else float("nan")


def log_loss_skill_score(y_true, prob, base_rate: float, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, float)
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    b = float(np.clip(base_rate, eps, 1 - eps))
    ll = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    ll_b = -np.mean(y_true * np.log(b) + (1 - y_true) * np.log(1 - b))
    return float(1.0 - ll / ll_b) if ll_b > 0 else float("nan")


def classification_metrics(y_true, prob, base_rate: float, threshold: float = 0.5) -> dict:
    """`prob` = predicted P(net buy);  `base_rate` = share of net-buy days in TRAIN."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
        log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true, int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    two = len(np.unique(y_true)) > 1

    return {
        # skill scores (0 = no better than the train base rate)
        "Brier_Skill": brier_skill_score(y_true, prob, base_rate),
        "LogLoss_Skill": log_loss_skill_score(y_true, prob, base_rate),
        # threshold-free ranking
        "AUC": float(roc_auc_score(y_true, prob)) if two else float("nan"),
        # threshold-dependent
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "F1_Score": float(f1_score(y_true, pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "Directional_Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        # raw probabilistic losses
        "Brier": float(brier_score_loss(y_true, prob)),
        "LogLoss": float(log_loss(y_true, prob, labels=[0, 1])),
        # context
        "base_rate_train": float(base_rate),
        "base_rate_test": float(y_true.mean()),
        "pred_positive_rate": float(pred.mean()),
        "threshold": float(threshold),
    }
