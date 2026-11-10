"""
Tabular featurisation for tree models.

The sequence folders (Att_LSTM / LSTM / BiLSTM) feed a (T, n) window to a
recurrent model.  Trees need a flat design matrix, so every driving series is
expanded into a small set of lags and the window collapses to one row per
prediction date:

    to predict  y_{tau+1}   ->   { <col>_lag<k> : col in cols, k in LAGS }
    <col>_lag0 = value on day tau, the most recent observation available

LAGS defaults to [0, 1, 2, 5, 10, 20] -- the same 30-day span the recurrent
models see, summarised at six points.  Chun et al. (2024) use only the current
day (lag 0); the extra lags keep the trees on an equal footing with the
sequence models.  Set `lags: [0]` in config.yaml for strict replication.

Transforms, all fit on training rows only:
    KRX columns   arcsinh            (heavy-tailed signed flows)
    predictors    Yeo-Johnson power transform, standardised   (Chun et al. 2024)
    AUX columns   z-score clip to +-aux_z_clip                (protocol parity)
    target        NONE -- the classifier target is a 0/1 direction label
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

DEFAULT_LAGS = [0, 1, 2, 5, 10, 20]


def scale_constant(y_raw, n_train: int) -> float:
    """arcsinh scale unit: median |flow| over the training rows only.

    arcsinh(x) is linear for |x| << 1 and logarithmic for |x| >> 1.  Feeding raw
    won (~1e11) puts every observation deep in the log regime, so the transformed
    target splits into two clumps near +-27 and its centre maps back to 0 KRW.
    Dividing by this constant first puts the data in arcsinh's transition zone so
    the model's expressible output range matches the target's actual range.
    """
    return float(np.median(np.abs(np.asarray(y_raw, float)[:n_train])))


def build_design(df: pd.DataFrame, feature_cols: list[str], groups: dict,
                 lags: list[int] | None = None,
                 scale_c: float = 1.0) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Flat design matrix X, raw target y (KRW) and the prediction dates.

    Row i of X holds the lagged predictors available for forecasting the target
    on date i; y[i] is that date's realised foreign_net_buy.
    """
    lags = list(lags if lags is not None else DEFAULT_LAGS)
    target = groups["KRX"][0]
    base = df[feature_cols].copy()
    for c in groups["KRX"]:
        if c in base:
            base[c] = np.arcsinh(base[c] / scale_c)       # tame the flow tail before lagging

    parts = {f"{c}_lag{k}": base[c].shift(1 + k) for k in lags for c in feature_cols}
    X = pd.DataFrame(parts, index=df.index)[[f"{c}_lag{k}" for k in lags for c in feature_cols]]
    y = df[target].to_numpy(np.float64)

    keep = X.notna().all(axis=1).to_numpy()
    return X.loc[keep], y[keep], pd.DatetimeIndex(df.index[keep])


def feature_group(name: str, groups: dict) -> str:
    """Map a lagged feature name back to its input group (KRX / LDA / LLM / AUX)."""
    col = name.rsplit("_lag", 1)[0]
    for g in ("KRX", "LDA", "LLM", "AUX"):
        if col in groups[g]:
            return g
    return "OTHER"


def clip_aux(Xs: np.ndarray, feature_names: list[str], groups: dict, z: float) -> tuple[np.ndarray, int]:
    idx = [i for i, f in enumerate(feature_names) if feature_group(f, groups) == "AUX"]
    if not idx or not z or z <= 0:
        return Xs, 0
    out = Xs.copy()
    n = int((np.abs(out[:, idx]) > z).sum())
    out[:, idx] = np.clip(out[:, idx], -z, z)
    return out, n


def fit_transforms(X: pd.DataFrame, y: np.ndarray, n_train: int, groups: dict, aux_z_clip: float,
                   scale_c: float = 1.0):
    """Fit Yeo-Johnson on predictors and arcsinh+StandardScaler on the target,
    using the first `n_train` rows only.  Returns transformed full arrays."""
    names = list(X.columns)
    xt = PowerTransformer(method="yeo-johnson", standardize=True).fit(X.iloc[:n_train])
    Xs, n_clipped = clip_aux(xt.transform(X), names, groups, aux_z_clip)
    y_as = np.arcsinh(y / scale_c).reshape(-1, 1)
    yt = StandardScaler().fit(y_as[:n_train])
    return Xs, yt.transform(y_as).ravel(), xt, yt, n_clipped


def inverse_target(y_scaled, y_scaler, scale_c: float = 1.0) -> np.ndarray:
    """Scaled prediction -> raw KRW."""
    return np.sinh(y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()) * scale_c


def fit_transforms_clf(X: pd.DataFrame, n_train: int, groups: dict, aux_z_clip: float):
    """Classifier variant: predictors only.

    The target is a 0/1 label, so there is nothing to arcsinh or standardise on
    the y side.  Predictors still get Yeo-Johnson + standardisation (Chun et al.
    2024) fit on the first `n_train` rows, then the AUX z-clip.
    """
    names = list(X.columns)
    xt = PowerTransformer(method="yeo-johnson", standardize=True).fit(X.iloc[:n_train])
    Xs, n_clipped = clip_aux(xt.transform(X), names, groups, aux_z_clip)
    return Xs, xt, n_clipped
