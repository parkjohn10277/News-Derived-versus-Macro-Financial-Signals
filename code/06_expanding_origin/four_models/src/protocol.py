"""
Two-phase protocol for tree DIRECTION CLASSIFIERS -- the exact analogue of the
regression folder, with log loss replacing MSE as the search criterion.

    Phase 1  fit on [start, val_start)  /  score on [val_start, train_end]
             randomised search over the model grid -> best hyperparameters
    Phase 2  refit on the official train [start, train_end] with those
             hyperparameters -> predict P(net buy) on [test_start, end]

Split dates, lag design, AUX clipping and the feature groups are identical to
the regression folder and to the sequence folders, so every variant is
comparable on the same days.

What differs from regression
----------------------------
  target      (y >= 0) as 0/1.  No arcsinh on the target, no StandardScaler, no
              sinh inversion, no prediction clipping -- none of it applies to a
              probability.
  scoring     neg_log_loss (the loss the classifier actually optimises).
  pos_weight  n_neg / n_pos on the TRAINING rows, passed to the estimator as
              class_weight="balanced" (RF) or scale_pos_weight (XGB, LGBM).
  scale_c     still applied to the KRX *input* columns before lagging, because
              the flow's own history is a predictor and is still heavy tailed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, TimeSeriesSplit

from .metrics import classification_metrics
from .tabular import build_design, fit_transforms_clf, scale_constant


def _cv(kind: str, n_train: int, n_val: int, n_splits: int = 5):
    """Validation scheme for the search.  'holdout' reproduces the sequence
    folders' single 2025 validation block; 'tscv' is a rolling alternative."""
    if kind == "holdout":
        fold = np.full(n_train + n_val, -1)
        fold[n_train:] = 0
        return PredefinedSplit(fold)
    if kind == "tscv":
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError(f"unknown cv scheme {kind}")


def run_single_split(df, feature_cols, groups, split, spec, seed: int, *,
                     lags=None, aux_z_clip: float = 3.0, threshold: float = 0.5,
                     cv_scheme: str = "holdout", n_iter: int = 30, n_jobs: int = -1,
                     verbose: bool = False) -> dict:
    """spec is a models/ module exposing estimator(seed, pos_weight) and GRID."""
    X0, y_raw, dates = build_design(df, feature_cols, groups, lags)
    names = list(X0.columns)

    sel_end = int((dates < pd.Timestamp(split["val_start"])).sum())
    off_end = int((dates <= pd.Timestamp(split["train_end"])).sum())
    test_start = int((dates < pd.Timestamp(split["test_start"])).sum())
    if not (0 < sel_end < off_end <= test_start < len(dates)):
        raise ValueError(f"bad split: sel_end={sel_end} off_end={off_end} "
                         f"test_start={test_start} n={len(dates)}")

    # arcsinh scale unit for the KRX input columns, training rows only
    scale_c = scale_constant(y_raw, off_end)
    X, _, _ = build_design(df, feature_cols, groups, lags, scale_c)

    # ---- target: direction
    y_dir = (y_raw >= 0).astype(int)                      # 1 = net buy, 0 = net sell
    n_pos = int(y_dir[:off_end].sum())
    n_neg = int(off_end - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    base_rate = float(y_dir[:off_end].mean())

    # ---- phase 1 : hyperparameter selection on the validation block
    X1, _, clip1 = fit_transforms_clf(X, sel_end, groups, aux_z_clip)
    search = RandomizedSearchCV(
        estimator=spec.estimator(seed, pos_weight), param_distributions=spec.GRID,
        n_iter=n_iter, scoring="neg_log_loss",
        cv=_cv(cv_scheme, sel_end, off_end - sel_end),
        random_state=seed, n_jobs=n_jobs, refit=False, error_score="raise",
    )
    search.fit(X1[:off_end], y_dir[:off_end])
    best_params = dict(search.best_params_)
    best_val = float(-search.best_score_)
    if verbose:
        print(f"    [SEARCH] best val logloss={best_val:.6f}  params={best_params}")

    # ---- phase 2 : refit on the official train, predict the test block
    X2, x_scaler, clip2 = fit_transforms_clf(X, off_end, groups, aux_z_clip)
    model = spec.estimator(seed, pos_weight).set_params(**best_params)
    model.fit(X2[:off_end], y_dir[:off_end])

    test_idx = np.arange(test_start, len(dates))
    prob = model.predict_proba(X2[test_idx])[:, 1]        # P(net buy)
    y_true = y_dir[test_idx]

    met = classification_metrics(y_true, prob, base_rate, threshold)
    met.update({
        "target_scale_c": float(scale_c),
        "pos_weight": float(pos_weight),
        "best_params": best_params, "best_validation_loss": best_val,
        "search_n_iter": int(n_iter), "cv_scheme": cv_scheme,
        "aux_values_clipped_phase1": clip1, "aux_values_clipped_phase2": clip2,
        "n_test": int(len(test_idx)), "n_design_features": len(names),
        "selection_train": f"{dates[0].date()}~{dates[sel_end - 1].date()} ({sel_end} rows)",
        "validation": f"{dates[sel_end].date()}~{dates[off_end - 1].date()} ({off_end - sel_end} rows)",
        "official_train": f"{dates[0].date()}~{dates[off_end - 1].date()} ({off_end} rows)",
        "test": f"{dates[test_idx[0]].date()}~{dates[test_idx[-1]].date()} ({len(test_idx)} rows)",
    })
    preds = pd.DataFrame({
        "date": dates[test_idx],
        "actual_flow_krw": y_raw[test_idx],
        "actual_direction": y_true,
        "prob_net_buy": prob,
        "predicted_direction": (prob >= threshold).astype(int),
    })
    return {"metrics": met, "predictions": preds, "model": model, "feature_names": names,
            "X_scaled": X2, "y_dir": y_dir, "y_raw": y_raw, "dates": dates,
            "test_idx": test_idx, "off_end": off_end, "base_rate": base_rate,
            "x_scaler": x_scaler, "cv_results": pd.DataFrame(search.cv_results_)}
