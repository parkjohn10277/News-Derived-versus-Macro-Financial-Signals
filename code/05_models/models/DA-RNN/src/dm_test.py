"""Diebold-Mariano test of equal predictive accuracy.

Sebastian & Tantia 2025, sec 3.4.2 / Diebold & Mariano (1995), with the
Harvey-Leybourne-Newbold (1997) small-sample correction.

    H0 : E[ g(e_A) - g(e_B) ] = 0        (the two forecasts are equally accurate)
    g  : loss, "squared" or "abs"
    dm > 0  => model B has the smaller loss (A worse); dm < 0 => A better.

For a 1-step-ahead forecast the HLN correction factor is 1, so this reduces to
the standard DM statistic with a Student-t reference distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMResult:
    dm_stat: float
    p_value: float
    better: str        # "A", "B" or "tie"
    horizon: int
    loss: str
    n: int


def _loss(err: np.ndarray, kind: str) -> np.ndarray:
    if kind == "squared":
        return err ** 2
    if kind == "abs":
        return np.abs(err)
    raise ValueError(kind)


def dm_test(y_true, pred_a, pred_b, horizon: int = 1, loss: str = "squared", alpha: float = 0.05) -> DMResult:
    y_true = np.asarray(y_true, float)
    d = _loss(y_true - np.asarray(pred_a, float), loss) - _loss(y_true - np.asarray(pred_b, float), loss)
    n = d.size
    dbar = d.mean()

    # long-run variance of dbar: Newey-West up to lag h-1
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, horizon):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2.0 * (1.0 - k / horizon) * gk
    var_dbar = lrv / n
    dm = dbar / np.sqrt(var_dbar) if var_dbar > 0 else np.nan

    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm *= hln
    p = 2.0 * stats.t.sf(np.abs(dm), df=n - 1) if np.isfinite(dm) else np.nan

    better = "tie"
    if np.isfinite(p) and p < alpha:
        better = "B" if dm > 0 else "A"
    return DMResult(float(dm), float(p), better, horizon, loss, int(n))
