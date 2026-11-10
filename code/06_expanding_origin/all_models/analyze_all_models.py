"""
Combined expanding-origin analysis: all ten models of the paper (plus the logit benchmark) in all four windows.

    python run_windows.py --collect      # results_nn_all.csv (7 networks, W1-W3)
    python analyze_all_models.py         # analysis/

Sources (read only)
  results_nn_all.csv                       7 neural networks, W1-W3 (this folder)
  ../Expanding_origin/results_all.csv      logit, RF, XGBoost, LightGBM, W1-W4
  ../GDP_corrected/results_all.csv         7 neural networks, W4 (= the paper's split and runs)

Metric and tests follow the main paper (Table tbl:block_auc): for each model the AUC is averaged over the five
seeds; a block's effect is the mean change in AUC over its four matched pairs; the ten model effects are tested
against zero with a two-sided Wilcoxon signed-rank test, per window and pooled over the 4 x 10 window-model effects.
The logistic regression is reported as a benchmark and is not one of the ten models in the tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
TREE_NAMES = {"logistic": "Logit", "random_forest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
NN = ["LSTM", "CNN-LSTM", "GRU-LSTM", "BiGRU-LSTM", "BiLSTM", "DA-RNN", "iTransformer"]
TEN = ["RandomForest", "XGBoost", "LightGBM", "iTransformer"] + [m for m in NN if m != "iTransformer"]
FAMILY = {m: "Machine learning" for m in ["RandomForest", "XGBoost", "LightGBM"]}
FAMILY.update({"iTransformer": "Transformer", **{m: "LSTM-based" for m in NN if m != "iTransformer"}})
PAIRS = {"AUX": [("E1", "E0"), ("E5", "E2"), ("E6", "E3"), ("E7", "E4")],
         "LLM": [("E3", "E0"), ("E4", "E2"), ("E6", "E1"), ("E7", "E5")],
         "LDA": [("E2", "E0"), ("E4", "E3"), ("E5", "E1"), ("E7", "E6")]}
EXPS = [f"E{k}" for k in range(8)]
WINDOWS = {"W1": "2022", "W2": "2023", "W3": "2024", "W4": "2025-26 (paper)"}


def load() -> pd.DataFrame:
    nn = pd.read_csv(HERE / "results_nn_all.csv")[["window", "model", "experiment", "seed", "AUC"]]
    tr = pd.read_csv(HERE.parent / "Expanding_origin" / "results_all.csv")[["window", "model", "experiment", "seed", "AUC"]]
    tr["model"] = tr.model.map(TREE_NAMES)
    w4 = pd.read_csv(HERE.parent / "GDP_corrected" / "results_all.csv")[["model", "experiment", "seed", "AUC"]]
    w4 = w4[w4.model.isin(NN)].assign(window="W4")
    return pd.concat([nn, tr, w4], ignore_index=True)


def wil(x):
    x = np.asarray(x, float)
    return float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    n = d.groupby(["window", "model"]).size().unstack(0)
    complete = (n == 40)                                   # 8 sets x 5 seeds
    A = d.groupby(["window", "model", "experiment"]).AUC.mean().unstack("experiment")[EXPS]
    A.round(6).to_csv(OUT / "auc_by_window.csv")

    rows = []
    for (w, m), r in A.iterrows():
        if not complete.loc[m, w]:
            continue
        for blk, pairs in PAIRS.items():
            rows.append({"window": w, "model": m, "family": FAMILY.get(m, "Benchmark"), "block": blk,
                         "d_auc_x100": 100 * np.mean([r[a] - r[b] for a, b in pairs])})
    E = pd.DataFrame(rows)
    E.round(4).to_csv(OUT / "block_effects_by_model.csv", index=False)

    lines = ["Expanding-origin check with all ten models (seed-mean AUC; block effect = mean over 4 matched pairs, x100).",
             "Wilcoxon signed-rank test across the ten models (per window) and across window x model (pooled).", ""]
    status = complete.reindex(index=TEN + ["Logit"]).replace({True: "ok", False: "MISSING"})
    lines += ["Runs complete (40 per window and model):", status.to_string(), ""]
    S = []
    T = E[E.model.isin(TEN)]
    for blk in PAIRS:
        for w in list(WINDOWS) + ["pooled"]:
            x = T[T.block == blk] if w == "pooled" else T[(T.block == blk) & (T.window == w)]
            if x.empty:
                continue
            S.append({"block": blk, "window": w, "test_period": WINDOWS.get(w, "all four"), "n_models": len(x),
                      "mean_d_auc_x100": x.d_auc_x100.mean(), "n_improved": int((x.d_auc_x100 > 0).sum()),
                      "wilcoxon_p": wil(x.d_auc_x100)})
    S = pd.DataFrame(S)
    S.round(4).to_csv(OUT / "block_by_window.csv", index=False)
    lines += ["=== Block effects, ten models", S.round(4).to_string(index=False), ""]

    F = T.groupby(["block", "family", "window"]).d_auc_x100.mean().unstack("window")
    F["all"] = T.groupby(["block", "family"]).d_auc_x100.mean()
    F.round(2).to_csv(OUT / "block_by_family.csv")
    lines += ["=== Mean effect by model family", F.round(2).to_string(), ""]

    L = E[E.model == "Logit"].pivot(index="block", columns="window", values="d_auc_x100")
    lines += ["=== Logit benchmark (same protocol), block effects", L.round(2).to_string(), ""]
    M = A.reset_index()
    M = M[M.model.isin(TEN)].groupby("window")[EXPS].mean()
    M.loc[:, "E1-E0"] = M.E1 - M.E0
    lines += ["=== Mean AUC of the ten models by window", M.round(3).to_string(), ""]
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
