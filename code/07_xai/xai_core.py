"""
Exact grouped Shapley machinery (model-agnostic).

Players are the information GROUPS (Domestic news, Foreign news, Market,
Macro).  With 4 players there are only 2^4 = 16 coalitions, so every Shapley
value and every pairwise interaction is computed EXACTLY -- no sampling.

A coalition S is an integer bitmask: bit i set <=> player i is present.
For a model f and a day x, the value of S is the interventional expectation

    v_x(S) = E_b [ f( x_S , b_{not S} ) ]        b ~ background (training days)

i.e. the groups outside S are replaced by the values of randomly drawn
training windows.  This is the same value function KernelSHAP and
shap.PermutationExplainer estimate; with 16 coalitions we just enumerate it.

Two kinds of games are built from the same cached v_x(S):

  LOCAL        v_x(S)                          -> phi_g(x): group SHAP value of
                                                  one day's P(net buy)
  PERFORMANCE  v(S) = AUC( y, v_.(S) ) - 0.5   -> contribution of each group to
               (or log-loss reduction)            OUT-OF-SAMPLE predictive power
                                                  (SAGE-style, Covert et al. 2020)

Shapley interaction index (Grabisch & Roubens 1999) for a pair (i, j):

    I_ij = sum_{S in N\\{i,j}} |S|!(n-|S|-2)!/(n-1)! *
           [ v(S+i+j) - v(S+i) - v(S+j) + v(S) ]

Read on the PERFORMANCE game:  I_ij < 0  -> the two groups carry OVERLAPPING
information (together they add less than separately);  I_ij ~ 0 -> they
contribute INDEPENDENTLY;  I_ij > 0 -> they are complementary.

    python xai_core.py        # self-test on games with known answers
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata


# ------------------------------------------------------------------ weights
def _popcount(x: int) -> int:
    return bin(x).count("1")


def shapley_weights(n: int) -> np.ndarray:
    return np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                     for s in range(n)])


def interaction_weights(n: int) -> np.ndarray:
    return np.array([math.factorial(s) * math.factorial(n - s - 2) / math.factorial(n - 1)
                     for s in range(n - 1)])


# ------------------------------------------------------------------ Shapley
def shapley_values(V: np.ndarray, n: int) -> np.ndarray:
    """V[..., S] = value of coalition S.  Returns phi[..., n]."""
    V = np.asarray(V, dtype=float)
    if V.shape[-1] != 1 << n:
        raise ValueError(f"expected last axis 2^{n}={1 << n}, got {V.shape[-1]}")
    w = shapley_weights(n)
    phi = np.zeros(V.shape[:-1] + (n,))
    for i in range(n):
        bit = 1 << i
        for S in range(1 << n):
            if S & bit:
                continue
            phi[..., i] += w[_popcount(S)] * (V[..., S | bit] - V[..., S])
    return phi


def interaction_values(V: np.ndarray, n: int) -> np.ndarray:
    """Shapley interaction index for every pair.  Returns I[..., n, n] (zero diagonal)."""
    V = np.asarray(V, dtype=float)
    I = np.zeros(V.shape[:-1] + (n, n))
    if n < 2:
        return I
    w = interaction_weights(n)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = 1 << i, 1 << j
            acc = np.zeros(V.shape[:-1])
            for S in range(1 << n):
                if S & bi or S & bj:
                    continue
                acc = acc + w[_popcount(S)] * (V[..., S | bi | bj] - V[..., S | bi]
                                               - V[..., S | bj] + V[..., S])
            I[..., i, j] = acc
            I[..., j, i] = acc
    return I


# ------------------------------------------------------------------ performance game
def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (constant scores -> 0.5)."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(s, dtype=float))
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def coalition_performance(V_sub: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    """v(S) for every coalition, on the days in V_sub (rows) with labels y.

    auc      v(S) = AUC(y, p_S) - 0.5          v(empty) = 0 (constant prediction)
    logloss  v(S) = LL(p_empty) - LL(p_S)      loss REDUCTION, v(empty) = 0
    """
    n_coal = V_sub.shape[1]
    if metric == "auc":
        return np.array([fast_auc(y, V_sub[:, S]) - 0.5 for S in range(n_coal)])
    if metric == "logloss":
        base = log_loss(y, V_sub[:, 0])
        return np.array([base - log_loss(y, V_sub[:, S]) for S in range(n_coal)])
    raise ValueError(f"unknown metric {metric}")


def block_bootstrap_indices(n: int, block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap (B x n): contiguous blocks keep the serial
    dependence of daily data, which an iid bootstrap would destroy."""
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[..., None] + np.arange(block)) % n
    return idx.reshape(B, -1)[:, :n]


def performance_decomposition(V_sub: np.ndarray, y: np.ndarray, n: int, metric: str,
                              B: int, block: int, rng: np.random.Generator) -> dict:
    """Exact Shapley + interaction decomposition of out-of-sample performance,
    with block-bootstrap confidence intervals over days."""
    y = np.asarray(y)
    v = coalition_performance(V_sub, y, metric)
    out = {"total": float(v[-1]), "phi": shapley_values(v, n), "I": interaction_values(v, n),
           "n_days": int(len(y)), "n_pos": int(y.sum())}
    tot, phis, Is = [], [], []
    for idx in block_bootstrap_indices(len(y), block, B, rng):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        vb = coalition_performance(V_sub[idx], yb, metric)
        tot.append(vb[-1])
        phis.append(shapley_values(vb, n))
        Is.append(interaction_values(vb, n))
    out["boot_total"] = np.array(tot)
    out["boot_phi"] = np.array(phis).reshape(-1, n)
    out["boot_I"] = np.array(Is).reshape(-1, n, n)
    return out


def ci(a: np.ndarray, axis: int = 0, level: float = 0.95):
    """Percentile interval along `axis`; NaN of the right shape if no replicates."""
    a = np.asarray(a, dtype=float)
    lo = (1 - level) / 2 * 100
    if a.shape[axis] == 0:
        shape = a.shape[:axis] + a.shape[axis + 1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    return np.nanpercentile(a, lo, axis=axis), np.nanpercentile(a, 100 - lo, axis=axis)


# ------------------------------------------------------------------ regimes / lags
def assign_regimes(x: np.ndarray, ref_mask: np.ndarray, method: str = "tercile"):
    """Label each day low / mid / high by the regime variable, thresholds from
    the reference (training) days only."""
    x = np.asarray(x, dtype=float)
    ref = x[np.asarray(ref_mask, bool) & np.isfinite(x)]
    if ref.size == 0:
        raise ValueError("no finite regime-variable values on the reference days")
    if method == "tercile":
        lo, hi = np.percentile(ref, [100 / 3, 200 / 3])
        lab = np.where(x <= lo, "low", np.where(x >= hi, "high", "mid"))
    elif method == "median":
        lo = hi = float(np.median(ref))
        lab = np.where(x <= lo, "low", "high")
    else:
        raise ValueError(f"unknown regime method {method}")
    lab = np.where(np.isfinite(x), lab, "na").astype(object)
    return lab, {"method": method, "low_max": float(lo), "high_min": float(hi),
                 "n_reference_days": int(ref.size)}


def lag_bucket_index(lag_days: np.ndarray, buckets) -> np.ndarray:
    lag_days = np.asarray(lag_days)
    out = np.full(lag_days.shape, -1, dtype=int)
    for b, (_, lo, hi) in enumerate(buckets):
        out[(lag_days >= lo) & (lag_days <= hi)] = b
    if (out < 0).any():
        raise ValueError(f"lags without a bucket: {sorted(set(lag_days[out < 0].tolist()))}")
    return out


# ------------------------------------------------------------------ self-test
def _game(n: int, fn) -> np.ndarray:
    return np.array([fn({i for i in range(n) if S >> i & 1}) for S in range(1 << n)], float)


def selftest() -> None:
    n = 4
    # 1. additive game: phi = weights, no interaction
    a = np.array([0.3, -0.1, 0.5, 0.2])
    v = _game(n, lambda s: sum(a[i] for i in s))
    assert np.allclose(shapley_values(v, n), a)
    assert np.allclose(interaction_values(v, n), 0)
    # 2. redundant pair (0,1 carry the SAME information) + additive player 2
    v = _game(n, lambda s: 1.0 * (0 in s or 1 in s) + 0.4 * (2 in s))
    phi, I = shapley_values(v, n), interaction_values(v, n)
    assert np.allclose(phi, [0.5, 0.5, 0.4, 0.0]), phi
    assert np.isclose(I[0, 1], -1.0) and np.isclose(I[0, 2], 0.0), I
    # 3. complementary pair (useful only together)
    v = _game(n, lambda s: 1.0 * (0 in s and 1 in s))
    phi, I = shapley_values(v, n), interaction_values(v, n)
    assert np.allclose(phi, [0.5, 0.5, 0, 0]) and np.isclose(I[0, 1], 1.0)
    # 4. efficiency on a random game
    rng = np.random.default_rng(0)
    v = rng.normal(size=1 << n)
    assert np.isclose(shapley_values(v, n).sum(), v[-1] - v[0])
    # 5. AUC helper
    assert fast_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert fast_auc(np.array([0, 1, 0, 1]), np.full(4, 0.3)) == 0.5
    # 6. vectorised over days
    Vd = rng.normal(size=(7, 1 << n))
    assert np.allclose(shapley_values(Vd, n).sum(1), Vd[:, -1] - Vd[:, 0])
    print("xai_core self-test passed: exact Shapley values, interactions, efficiency, AUC.")


if __name__ == "__main__":
    selftest()
