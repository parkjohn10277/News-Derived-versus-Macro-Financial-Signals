"""
Block effects per forecast origin (run after run_expanding.py).  Nothing is retrained.

Each model's probabilities are averaged over the five seeds (seed ensemble), as in the paper.
For every window, model and "add one block" pair (4 pairs per block) it reports the change in
AUC with DeLong, circular-block-bootstrap and Diebold-Mariano (Brier) p-values.

    python analyze_expanding.py

Outputs  analysis/
  auc_by_window.csv        AUC of every window x model x feature set (+ majority, persistence)
  block_pairs.csv          window x model x pair: dAUC, DeLong p, bootstrap p, DM-Brier p
  block_by_window.csv      window x block: mean dAUC over the 4 models x 4 pairs, signs, significance
  summary.txt              the numbers to quote
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_expanding import EXPS, MODELS, RES, SEEDS, WINDOWS, load_input
from stat_tests import BLOCK_PAIRS, EXTRA_PAIRS, auc, block_indices, boot_auc_diff, brier_loss, \
    delong, diebold_mariano

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
NAMES = {"logistic": "Logit", "random_forest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def ensemble(w, m, e):
    ps = []
    for s in SEEDS:
        f = RES / w / m / e / f"seed_{s}" / "predictions.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f, parse_dates=["date"], encoding="utf-8-sig")
        ps.append(d.set_index("date"))
    y = ps[0]["actual_direction"].to_numpy()
    p = np.mean([d["prob_net_buy"].to_numpy() for d in ps], axis=0)
    return ps[0].index, y, p


def naive(w, dates):
    """Majority (training base rate) and persistence (rank by the day-t flow) on the same days."""
    raw = load_input()["foreign_net_buy"]
    prev = raw.shift(1).reindex(dates).to_numpy()
    y = (raw.reindex(dates).to_numpy() >= 0).astype(int)
    return {"persistence": auc(y, prev), "majority": 0.5,
            "base_rate_test": y.mean(), "n_test": len(y)}


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    auc_rows, pair_rows, info = [], [], {}
    for w in WINDOWS:
        preds = {(m, e): ensemble(w, m, e) for m in MODELS for e in EXPS}
        if any(v is None for v in preds.values()):
            print(f"{w}: incomplete, skipped"); continue
        dates, y, _ = preds[(next(iter(MODELS)), "E0")]
        info[w] = naive(w, dates)
        info[w]["test"] = f"{dates[0].date()} .. {dates[-1].date()}"
        idx = block_indices(len(y), rng)
        for m in MODELS:
            for e in EXPS:
                auc_rows.append({"window": w, "model": NAMES[m], "experiment": e,
                                 "AUC": auc(y, preds[(m, e)][2])})
            pairs = [(b, a, c) for b, ps in BLOCK_PAIRS.items() for a, c in ps] + \
                    [("extra", a, c) for a, c in EXTRA_PAIRS]
            for blk, a, c in pairs:
                pa, pc = preds[(m, a)][2], preds[(m, c)][2]
                d, _, p_dl = delong(y, pa, pc)
                _, lo, hi, p_bt = boot_auc_diff(y, pa, pc, idx)
                _, _, p_dm = diebold_mariano(brier_loss(y, pa), brier_loss(y, pc))
                pair_rows.append({"window": w, "model": NAMES[m], "block": blk, "pair": f"{a} vs {c}",
                                  "dAUC": d, "ci_lo": lo, "ci_hi": hi, "p_delong": p_dl,
                                  "p_boot": p_bt, "p_dm_brier": p_dm})
        for k in ("persistence", "majority"):
            auc_rows.append({"window": w, "model": k, "experiment": "-", "AUC": info[w][k]})
        print(f"{w}: done")

    A = pd.DataFrame(auc_rows)
    P = pd.DataFrame(pair_rows)
    A.to_csv(OUT / "auc_by_window.csv", index=False, float_format="%.4f")
    P.to_csv(OUT / "block_pairs.csv", index=False, float_format="%.5f")

    blk = P[P.block != "extra"]
    G = blk.groupby(["window", "block"])
    S = pd.DataFrame({
        "mean_dAUC_x100": G.dAUC.mean() * 100,
        "n_pos": G.dAUC.apply(lambda s: int((s > 0).sum())),
        "n": G.size(),
        "sig_pos": G.apply(lambda g: int(((g.dAUC > 0) & (g.p_delong < .05)).sum())),
        "sig_neg": G.apply(lambda g: int(((g.dAUC < 0) & (g.p_delong < .05)).sum())),
    }).reset_index()
    # pooled over windows: unit = window x model (mean over its 4 pairs), Wilcoxon signed-rank
    unit = blk.groupby(["block", "window", "model"]).dAUC.mean().reset_index()
    pooled = []
    for b, g in unit.groupby("block"):
        pooled.append({"window": "all", "block": b, "mean_dAUC_x100": g.dAUC.mean() * 100,
                       "n_pos": int((g.dAUC > 0).sum()), "n": len(g),
                       "sig_pos": int(S[(S.block == b)].sig_pos.sum()),
                       "sig_neg": int(S[(S.block == b)].sig_neg.sum()),
                       "wilcoxon_p": stats.wilcoxon(g.dAUC).pvalue if len(g) > 5 else np.nan})
    S = pd.concat([S, pd.DataFrame(pooled)], ignore_index=True)
    S.to_csv(OUT / "block_by_window.csv", index=False, float_format="%.4f")

    # ---------------------------------------------------------------- summary
    L = [f"Expanding-origin check. Seed-ensemble probabilities over seeds {SEEDS}. "
         f"Models: {', '.join(NAMES.values())}. DeLong unless stated.\n"]
    L.append("=== Test windows")
    for w, v in info.items():
        L.append(f"  {w}  {v['test']}  n={v['n_test']}  net-buy share {v['base_rate_test']:.3f}  "
                 f"persistence AUC {v['persistence']:.3f}")
    L.append("\n=== AUC by window (seed ensemble)")
    tab = A[A.experiment != "-"].pivot_table(index=["window", "model"], columns="experiment", values="AUC")
    L.append(tab.round(3).to_string())
    L.append("\nMean over the four models:")
    L.append(A[A.experiment != "-"].pivot_table(index="window", columns="experiment", values="AUC")
             .round(3).to_string())
    L.append("\n=== Block effects (16 comparisons per window = 4 models x 4 pairs; dAUC x100)")
    for _, r in S.iterrows():
        extra = f"  Wilcoxon over window x model p={r['wilcoxon_p']:.4f}" if r["window"] == "all" else ""
        L.append(f"  {r['window']:3s} {r['block']:3s}  mean {r['mean_dAUC_x100']:+6.2f}  "
                 f"positive {int(r['n_pos'])}/{int(r['n'])}  significant + {int(r['sig_pos'])}  "
                 f"- {int(r['sig_neg'])}{extra}")
    L.append("\n=== E6 vs E1 (does LLM sentiment help on top of KRX+AUX?)")
    for _, r in P[P.pair == "E6 vs E1"].iterrows():
        L.append(f"  {r['window']} {r['model']:13s} dAUC {r['dAUC'] * 100:+6.2f}  DeLong p={r['p_delong']:.3f}  "
                 f"boot p={r['p_boot']:.3f}  DM-Brier p={r['p_dm_brier']:.3f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
