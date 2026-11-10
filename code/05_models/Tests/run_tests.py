"""
Statistical tests on the saved out-of-sample predictions of the 10 direction
classifiers (E0-E7).  Nothing is retrained: every test reads predictions.csv.

    python run_tests.py                 # seed 42 (what exists today)
    python run_tests.py --seeds 42 43 44 45 46    # after the multi-seed rerun

Outputs (Desktop/Tests/results/):
  1_within_model_pairs.csv   DeLong + block-bootstrap (AUC) and Diebold-Mariano
                             (Brier, log-loss) for every "add one block" pair
                             of experiments, inside each model
  2_block_wilcoxon.csv       Wilcoxon signed-rank: does adding a block help
                             ACROSS models?
  3_cross_model_E{k}.csv     each model vs the best-AUC model (DeLong, DM)
  4_mcs_E{k}.csv             Hansen-Lunde-Nason model confidence set
  summary.txt                the numbers to quote

Unit of evidence: DeLong / DM / MCS use the 278 test DAYS of one run;
Wilcoxon uses the MODELS (x seeds) as paired observations.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DESKTOP = Path(r".")
OUT = Path(__file__).resolve().parent / "results"

# model -> directory that holds E{k}/seed_{s}/predictions.csv
FINAL = DESKTOP / "Submission_version" / "GDP_corrected" / "models"   # GDP-corrected rerun
MODELS = {
    "LSTM": FINAL / "LSTM/results/lstm",
    "CNN-LSTM": FINAL / "CNN-LSTM/results/cnn_lstm",
    "GRU-LSTM": FINAL / "GRU-LSTM/results/gru_lstm",
    "BiGRU-LSTM": FINAL / "BiGRU-LSTM/results/bigru_lstm",
    "BiLSTM": FINAL / "BiLSTM/results/bilstm",
    "DA-RNN": FINAL / "DA-RNN/results/darnn",
    "iTransformer": FINAL / "iTransformer/results/itransformer",
    "Random Forest": FINAL / "RandomForest/results/random_forest",
    "XGBoost": FINAL / "XGBoost/results/xgboost",
    "LightGBM": FINAL / "LightGBM/results/lightgbm",
}
EXPS = [f"E{k}" for k in range(8)]
FEATURES = {"E0": "KRX", "E1": "KRX+AUX", "E2": "KRX+LDA", "E3": "KRX+LLM",
            "E4": "KRX+LDA+LLM", "E5": "KRX+LDA+AUX", "E6": "KRX+LLM+AUX", "E7": "ALL"}

# (with block, without block): each block is added to the 4 sets that lack it
BLOCK_PAIRS = {
    "AUX": [("E1", "E0"), ("E5", "E2"), ("E6", "E3"), ("E7", "E4")],
    "LLM": [("E3", "E0"), ("E4", "E2"), ("E6", "E1"), ("E7", "E5")],
    "LDA": [("E2", "E0"), ("E4", "E3"), ("E5", "E1"), ("E7", "E6")],
}
EXTRA_PAIRS = [("E6", "E0"), ("E7", "E0"), ("E7", "E1")]

BLOCK = 10          # circular block length (trading days), as in the XAI analysis
B_BOOT = 5000
NW_LAG = 5          # Newey-West bandwidth for DM: floor(4 (n/100)^(2/9)) = 5 at n = 278
MCS_ALPHA = 0.10    # 90% model confidence set
EPS = 1e-12


# ============================================================ loading
def load_predictions(seed: int) -> tuple[np.ndarray, dict]:
    """Returns y (test directions) and P[model][exp] = prob array, all date-aligned."""
    y_ref, dates_ref, P = None, None, {}
    for m, root in MODELS.items():
        P[m] = {}
        for e in EXPS:
            f = root / e / f"seed_{seed}" / "predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
            if dates_ref is None:
                dates_ref, y_ref = d["date"].values, d["actual_direction"].values.astype(int)
            if len(d) != len(dates_ref) or not (d["date"].values == dates_ref).all():
                raise SystemExit(f"{f}: test dates differ from the other runs")
            if not (d["actual_direction"].values == y_ref).all():
                raise SystemExit(f"{f}: target differs from the other runs")
            P[m][e] = np.clip(d["prob_net_buy"].values.astype(float), EPS, 1 - EPS)
    return y_ref, P


# ============================================================ metrics
def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def brier_loss(y, p):
    return (p - y) ** 2


def log_loss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ============================================================ DeLong
def _placements(y, p):
    pos, neg = p[y == 1], p[y == 0]
    cmp = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return cmp.mean(axis=1), cmp.mean(axis=0)          # V10 (per positive), V01 (per negative)


def delong(y, pa, pb):
    """DeLong, DeLong & Clarke-Pearson (1988) test of AUC_a = AUC_b (paired)."""
    v10a, v01a = _placements(y, pa)
    v10b, v01b = _placements(y, pb)
    m, n = len(v10a), len(v01a)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    diff = v10a.mean() - v10b.mean()
    if var <= 0:
        return diff, np.nan, 1.0
    z = diff / np.sqrt(var)
    return diff, z, 2 * stats.norm.sf(abs(z))


# ============================================================ block bootstrap
def block_indices(n, rng, block=BLOCK, B=B_BOOT):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def boot_auc_diff(y, pa, pb, idx):
    """Two-sided circular-block-bootstrap p-value for AUC_a - AUC_b (keeps autocorrelation)."""
    d0 = auc(y, pa) - auc(y, pb)
    ds = np.empty(len(idx))
    for b, ix in enumerate(idx):
        yy = y[ix]
        if yy.min() == yy.max():
            ds[b] = np.nan
            continue
        ds[b] = auc(yy, pa[ix]) - auc(yy, pb[ix])
    ds = ds[~np.isnan(ds)]
    p = np.mean(np.abs(ds - ds.mean()) >= abs(d0))
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5), max(p, 1 / len(ds))


# ============================================================ Diebold-Mariano
def nw_var(d, lag=NW_LAG):
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(la, lb, h=1, lag=NW_LAG):
    """DM test of E[loss_a - loss_b] = 0 with Newey-West variance and the
    Harvey-Leybourne-Newbold small-sample correction (t_{n-1} reference).
    Negative statistic = model a has the LOWER loss (a is better)."""
    d = la - lb
    n = len(d)
    v = nw_var(d, lag)
    if v <= 0:
        return d.mean(), np.nan, 1.0
    dm = d.mean() / np.sqrt(v / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return d.mean(), dm, 2 * stats.t.sf(abs(dm), df=n - 1)


# ============================================================ model confidence set
def mcs(L: np.ndarray, names: list[str], rng, alpha=MCS_ALPHA, B=B_BOOT, block=BLOCK):
    """Hansen, Lunde & Nason (2011) MCS with the T_max statistic and a circular
    block bootstrap.  L is (n days x m models) of losses.  Returns a DataFrame
    with each model's MCS p-value; models with p >= alpha are in the set."""
    n = L.shape[0]
    idx = block_indices(n, rng, block, B)
    alive = list(range(L.shape[1]))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        Lm = L[:, alive]
        di = Lm - Lm.mean(axis=1, keepdims=True)          # loss relative to the set average
        dbar = di.mean(axis=0)
        dboot = np.stack([di[ix].mean(axis=0) for ix in idx])
        se = np.sqrt(((dboot - dbar) ** 2).mean(axis=0))
        t = dbar / se
        tmax = t.max()
        tstar = ((dboot - dbar) / se).max(axis=1)
        p = np.mean(tstar >= tmax)
        running = max(running, p)
        worst = alive[int(np.argmax(t))]
        if p >= alpha:
            for a in alive:
                pvals[a] = running
            break
        pvals[worst] = running
        alive.remove(worst)
    if len(alive) == 1:
        pvals[alive[0]] = 1.0
    out = pd.DataFrame({"model": names, "mean_loss": L.mean(axis=0),
                        "mcs_pvalue": [pvals[i] for i in range(len(names))]})
    out["in_90pct_MCS"] = out["mcs_pvalue"] >= alpha
    return out.sort_values("mean_loss").reset_index(drop=True)


# ============================================================ helpers
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    runs = {s: load_predictions(s) for s in args.seeds}
    y = runs[args.seeds[0]][0]
    idx = block_indices(len(y), rng)
    print(f"test days: {len(y)}   net-buy share: {y.mean():.3f}   seeds: {args.seeds}")

    # ---------------------------------------------------- 1. within-model pairs
    rows = []
    pair_list = [(blk, a, b) for blk, prs in BLOCK_PAIRS.items() for a, b in prs] + \
                [("extra", a, b) for a, b in EXTRA_PAIRS]
    for s, (_, P) in runs.items():
        for m in MODELS:
            for blk, a, b in pair_list:
                if a not in P[m] or b not in P[m]:
                    continue
                pa, pb = P[m][a], P[m][b]
                d_auc, z, p_dl = delong(y, pa, pb)
                _, lo, hi, p_bt = boot_auc_diff(y, pa, pb, idx)
                d_br, dm_br, p_br = diebold_mariano(brier_loss(y, pa), brier_loss(y, pb))
                d_ll, dm_ll, p_ll = diebold_mariano(log_loss(y, pa), log_loss(y, pb))
                rows.append(dict(seed=s, model=m, block=blk, with_=a, without=b,
                                 comparison=f"{a} vs {b}",
                                 auc_with=auc(y, pa), auc_without=auc(y, pb), d_auc=d_auc,
                                 d_auc_ci_lo=lo, d_auc_ci_hi=hi,
                                 delong_z=z, delong_p=p_dl, boot_p=p_bt,
                                 d_brier=d_br, dm_brier=dm_br, dm_brier_p=p_br,
                                 d_logloss=d_ll, dm_logloss=dm_ll, dm_logloss_p=p_ll))
    W = pd.DataFrame(rows).rename(columns={"with_": "with"})
    for col in ["delong_p", "boot_p", "dm_brier_p", "dm_logloss_p"]:
        W[col + "_holm"] = np.nan
        for s in W.seed.unique():                     # Holm over all pairs x models of one seed
            mask = W.seed == s
            W.loc[mask, col + "_holm"] = holm(W.loc[mask, col].values)
    W.to_csv(OUT / "1_within_model_pairs.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------- 2. Wilcoxon across models
    wrows = []
    B = W[W.block != "extra"]
    for blk in BLOCK_PAIRS:
        sub = B[B.block == blk]
        # primary: one number per model = mean effect over its 4 pairs (and seeds)
        per_model = sub.groupby("model")[["d_auc", "d_brier", "d_logloss"]].mean()
        for metric, better in [("d_auc", "greater"), ("d_brier", "less"), ("d_logloss", "less")]:
            v = per_model[metric].values
            stat, p2 = stats.wilcoxon(v, alternative="two-sided", zero_method="wilcox")
            wrows.append(dict(block=blk, unit="model (mean of 4 pairs)", metric=metric,
                              n=len(v), mean_effect=v.mean(), median_effect=np.median(v),
                              n_improves=int((v > 0).sum() if better == "greater" else (v < 0).sum()),
                              wilcoxon_W=stat, p_two_sided=p2))
        # secondary: every (model, pair, seed) -- not independent, shown for reference
        for metric, better in [("d_auc", "greater")]:
            v = sub[metric].values
            stat, p2 = stats.wilcoxon(v, alternative="two-sided", zero_method="wilcox")
            wrows.append(dict(block=blk, unit="all model x pair (non-independent)", metric=metric,
                              n=len(v), mean_effect=v.mean(), median_effect=np.median(v),
                              n_improves=int((v > 0).sum()), wilcoxon_W=stat, p_two_sided=p2))
    for a, b in EXTRA_PAIRS + [("E6", "E1"), ("E7", "E6")]:
        sub = W[(W["with"] == a) & (W.without == b)].groupby("model")["d_auc"].mean()
        stat, p2 = stats.wilcoxon(sub.values)
        wrows.append(dict(block=f"{a} vs {b}", unit="model", metric="d_auc", n=len(sub),
                          mean_effect=sub.mean(), median_effect=sub.median(),
                          n_improves=int((sub > 0).sum()), wilcoxon_W=stat, p_two_sided=p2))
    Wx = pd.DataFrame(wrows)
    Wx.to_csv(OUT / "2_block_wilcoxon.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------- 3 + 4. across models, per experiment
    # across models: average each model's test probability over the seeds (seed ensemble)
    P0 = {m: {e: np.mean([runs[s][1][m][e] for s in args.seeds], axis=0)
              for e in EXPS if all(e in runs[s][1][m] for s in args.seeds)} for m in MODELS}
    cross_all, mcs_all = [], []
    for e in EXPS:
        names = [m for m in MODELS if e in P0[m]]
        aucs = {m: auc(y, P0[m][e]) for m in names}
        best = max(aucs, key=aucs.get)
        rows = []
        for m in names:
            if m == best:
                rows.append(dict(exp=e, model=m, auc=aucs[m], vs=best, d_auc=0.0))
                continue
            d, z, p_dl = delong(y, P0[m][e], P0[best][e])
            _, dm_b, p_b = diebold_mariano(brier_loss(y, P0[m][e]), brier_loss(y, P0[best][e]))
            rows.append(dict(exp=e, model=m, auc=aucs[m], vs=best, d_auc=d,
                             delong_p=p_dl, dm_brier=dm_b, dm_brier_p=p_b))
        C = pd.DataFrame(rows).sort_values("auc", ascending=False)
        C["delong_p_holm"] = np.nan
        mask = C.delong_p.notna()
        C.loc[mask, "delong_p_holm"] = holm(C.loc[mask, "delong_p"].values)
        cross_all.append(C)
        for lname, lf in [("brier", brier_loss), ("logloss", log_loss)]:
            L = np.column_stack([lf(y, P0[m][e]) for m in names])
            M = mcs(L, names, rng)
            M.insert(0, "loss", lname); M.insert(0, "exp", e)
            mcs_all.append(M)
    pd.concat(cross_all).to_csv(OUT / "3_cross_model.csv", index=False, float_format="%.6g")
    pd.concat(mcs_all).to_csv(OUT / "4_mcs.csv", index=False, float_format="%.6g")

    # ---------------------------------------------------- summary
    lines = [f"Seeds: {args.seeds}   test days: {len(y)}   block bootstrap: {BLOCK}-day blocks, B={B_BOOT}",
             "", "=== Wilcoxon: does adding a block help ACROSS the 10 models? (unit = model)"]
    for _, r in Wx[Wx.unit.str.startswith("model")].iterrows():
        lines.append(f"  {r.block:10s} {r.metric:10s} mean {r.mean_effect*100:+6.2f} (x100)  "
                     f"improves {r.n_improves}/{r.n}  p = {r.p_two_sided:.4f} {stars(r.p_two_sided)}")
    lines += ["", "=== Within-model pairs: how many of the individual comparisons are significant (raw p < 0.05 / Holm < 0.05)"]
    for blk in list(BLOCK_PAIRS) + ["extra"]:
        sub = W[W.block == blk]
        lines.append(f"  {blk:6s} n={len(sub):3d}  DeLong {int((sub.delong_p<.05).sum()):2d}/{int((sub.delong_p_holm<.05).sum()):2d}   "
                     f"block-boot {int((sub.boot_p<.05).sum()):2d}/{int((sub.boot_p_holm<.05).sum()):2d}   "
                     f"DM-Brier {int((sub.dm_brier_p<.05).sum()):2d}/{int((sub.dm_brier_p_holm<.05).sum()):2d}")
    lines += ["", "=== E6 vs E1 (does LLM sentiment help in the best feature set?)"]
    for _, r in W[(W["with"] == "E6") & (W.without == "E1")].iterrows():
        lines.append(f"  {r.model:14s} dAUC {r.d_auc:+.4f}  DeLong p={r.delong_p:.3f}  boot p={r.boot_p:.3f}  "
                     f"DM-Brier p={r.dm_brier_p:.3f}")
    for e in ["E6", "E7"]:
        C = [c for c in cross_all if c.exp.iloc[0] == e][0]
        lines += ["", f"=== {e}: each model vs best-AUC model ({C.vs.iloc[0]})"]
        for _, r in C.iterrows():
            extra = "" if r.model == r.vs else f"  DeLong p={r.delong_p:.3f} (Holm {r.delong_p_holm:.3f})  DM-Brier p={r.dm_brier_p:.3f}"
            lines.append(f"  {r.model:14s} AUC {r.auc:.4f}{extra}")
        M = pd.concat(mcs_all)
        for lname in ["brier", "logloss"]:
            mm = M[(M.exp == e) & (M.loss == lname)]
            lines.append(f"  90% MCS ({lname}): {', '.join(mm[mm.in_90pct_MCS].model)}   "
                         f"excluded: {', '.join(mm[~mm.in_90pct_MCS].model) or '-'}")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
