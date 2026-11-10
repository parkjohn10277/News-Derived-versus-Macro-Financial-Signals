"""
Two-phase protocol -- binary classification variant.

    Phase 1  train [start, val_start)  /  validate [val_start, train_end]
             early stopping on validation BCE -> best_epoch
    Phase 2  refit on [start, train_end] for best_epoch epochs -> predict test

Split dates, feature groups and the AUX z-clip are identical to the regression
folders, so the two variants are directly comparable on the same days.

What differs from regression
----------------------------
  target      (y >= 0) as 0/1, NOT arcsinh-standardised.  No y_scaler, no sinh
              inversion, no target_clip -- none of it applies to a probability.
  pos_weight  n_neg / n_pos computed on the TRAINING rows only.
  weights     optional per-day magnitude weight |arcsinh(y / scale_c)|, so a
              3-trillion-won day counts more than a 1-billion-won day.
  scale_c     still applied to the KRX *input feature* (the flow's own history),
              because that is a predictor and still needs its tail tamed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .metrics import classification_metrics
from .train import TrainCfg, predict_at, train_fixed_epochs, train_fold


def clip_aux(Xs, feature_cols, aux_cols, z):
    idx = [i for i, c in enumerate(feature_cols) if c in aux_cols]
    if not idx or not z or z <= 0:
        return Xs, 0
    Xo = Xs.copy()
    n = int((np.abs(Xo[:, idx]) > z).sum())
    Xo[:, idx] = np.clip(Xo[:, idx], -z, z)
    return Xo, n


def scale_constant(y_raw, n_train: int) -> float:
    """arcsinh scale unit for the KRX input feature: median |flow| over training rows."""
    return float(np.median(np.abs(np.asarray(y_raw, float)[:n_train])))


def transform_features(df, feature_cols, groups, scale_c: float = 1.0):
    """arcsinh(x / scale_c) on the KRX columns; everything else untouched."""
    X = df[feature_cols].to_numpy(np.float64, copy=True)
    krx = [i for i, c in enumerate(feature_cols) if c in groups["KRX"]]
    X[:, krx] = np.arcsinh(X[:, krx] / scale_c)
    return X


def apply_scalers(df, feature_cols, groups, x_scaler, aux_z_clip, scale_c: float = 1.0):
    """Re-create the phase-2 model inputs from a saved scaler (for report scripts)."""
    Xs, _ = clip_aux(x_scaler.transform(transform_features(df, feature_cols, groups, scale_c)),
                     feature_cols, groups["AUX"], aux_z_clip)
    y = (df[groups["KRX"][0]].to_numpy(np.float64) >= 0).astype(np.float32)
    return Xs, y


def run_single_split(df, feature_cols, groups, split, build_model, tcfg: TrainCfg,
                     aux_z_clip: float = 3.0, threshold: float = 0.5,
                     magnitude_weighting: bool = False, device=None, verbose=False) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dates = pd.DatetimeIndex(df.index)
    target = groups["KRX"][0]

    sel_end = int((dates < pd.Timestamp(split["val_start"])).sum())
    off_end = int((dates <= pd.Timestamp(split["train_end"])).sum())
    test_start = int((dates < pd.Timestamp(split["test_start"])).sum())
    if not (tcfg.T < sel_end < off_end <= test_start < len(df)):
        raise ValueError(f"bad split: sel_end={sel_end} off_end={off_end} "
                         f"test_start={test_start} n={len(df)}")

    y_raw = df[target].to_numpy(np.float64)
    y_dir = (y_raw >= 0).astype(np.float32)                 # 1 = net buy, 0 = net sell
    scale_c = scale_constant(y_raw, off_end)

    # per-day weight: how economically large that day was (1.0 everywhere if off)
    if magnitude_weighting:
        w = np.abs(np.arcsinh(y_raw / scale_c)).astype(np.float32)
        w = (w / w[:off_end].mean()).astype(np.float32)      # normalised on train rows
    else:
        w = np.ones(len(y_raw), dtype=np.float32)

    X_t = transform_features(df, feature_cols, groups, scale_c)

    # Continuous flow history for NARX-style decoders (DA-RNN eq. 15).  The
    # TARGET stays the 0/1 label; only the history the model reads is continuous.
    # Standardised on the training rows, exactly as the regression folder did.
    y_as = np.arcsinh(y_raw / scale_c).reshape(-1, 1)
    yh_scaler = StandardScaler().fit(y_as[:off_end])
    y_cont = yh_scaler.transform(y_as).ravel().astype(np.float32)

    # class imbalance correction, training rows only
    n_pos = float(y_dir[:off_end].sum())
    n_neg = float(off_end - n_pos)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else None
    base_rate = float(y_dir[:off_end].mean())

    cfg1 = TrainCfg(**{**tcfg.__dict__, "pos_weight": pos_weight,
                       "magnitude_weighting": magnitude_weighting})

    # ---- phase 1 : epoch selection
    xs1 = StandardScaler().fit(X_t[:sel_end])
    X1, clip1 = clip_aux(xs1.transform(X_t), feature_cols, groups["AUX"], aux_z_clip)
    _, pack = train_fold(X1[:off_end], y_dir[:off_end], w[:off_end],
                         sel_end, off_end - sel_end, build_model, cfg1, device, verbose,
                         y_hist=y_cont[:off_end])
    best_epochs = pack["best_epoch"]

    # ---- phase 2 : refit + test
    xs2 = StandardScaler().fit(X_t[:off_end])
    X2, clip2 = clip_aux(xs2.transform(X_t), feature_cols, groups["AUX"], aux_z_clip)
    model, refit_hist = train_fixed_epochs(X2, y_dir, w, off_end, best_epochs,
                                           build_model, cfg1, device, verbose,
                                           y_hist=y_cont)

    test_pos = np.arange(max(test_start, tcfg.T), len(df))
    prob, extras = predict_at(model, X2, y_dir, w, test_pos, tcfg.T, device, y_hist=y_cont)
    y_true = y_dir[test_pos].astype(int)

    met = classification_metrics(y_true, prob, base_rate, threshold)
    met.update({
        "target_scale_c": float(scale_c),
        "pos_weight": float(pos_weight) if pos_weight else None,
        "magnitude_weighting": bool(magnitude_weighting),
        "best_epoch": int(best_epochs), "best_validation_loss": float(pack["best_val_loss"]),
        "phase1_epochs_run": int(pack["epochs_run"]),
        "aux_values_clipped_phase1": clip1, "aux_values_clipped_phase2": clip2,
        "n_test": int(len(test_pos)),
        "selection_train": f"{dates[0].date()}~{dates[sel_end - 1].date()} ({sel_end} rows)",
        "validation": f"{dates[sel_end].date()}~{dates[off_end - 1].date()} ({off_end - sel_end} rows)",
        "official_train": f"{dates[0].date()}~{dates[off_end - 1].date()} ({off_end} rows)",
        "test": f"{dates[test_pos[0]].date()}~{dates[test_pos[-1]].date()} ({len(test_pos)} rows)",
    })
    preds = pd.DataFrame({
        "date": dates[test_pos],
        "actual_flow_krw": y_raw[test_pos],
        "actual_direction": y_true,
        "prob_net_buy": prob,
        "predicted_direction": (prob >= threshold).astype(int),
    })
    alpha_df = None
    a = extras.get("alpha")
    if a is not None and a.ndim in (2, 3) and a.shape[-1] == len(feature_cols):
        alpha_df = pd.DataFrame(a.mean(axis=1) if a.ndim == 3 else a,
                                index=dates[test_pos], columns=feature_cols)
    return {"metrics": met, "predictions": preds, "extras": extras, "alpha": alpha_df,
            "selection_history": pd.DataFrame(pack["history"]),
            "refit_history": pd.DataFrame(refit_hist),
            "model": model, "x_scaler": xs2, "y_scaler": None}
