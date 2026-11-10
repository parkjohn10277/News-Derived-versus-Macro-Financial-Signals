"""
Table: significance of the feature blocks and of the key feature-set comparisons.

Unit of the Wilcoxon test = model (n = 10): each model's effect is its mean over the
5 seeds (and, for a block, over the 4 matched experiment pairs).  Raw p-values.
individual within-model DeLong tests (model x seed x pair) with raw p < 0.05, split by
the sign of the AUC change.

    python make_tbl_block_tests.py   -> tbl_block_tests_auc.tex / tbl_block_tests_dm.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
PAIRS = HERE.parent.parent / "GDP_corrected" / "Tests" / "results" / "1_within_model_pairs.csv"

ROWS = [  # label, block or comparison filter, pairs text
    ("Adding AUX", ("block", "AUX"), "E1--E0, E5--E2, E6--E3, E7--E4"),
    ("Adding LLM", ("block", "LLM"), "E3--E0, E4--E2, E6--E1, E7--E5"),
    ("Adding LDA", ("block", "LDA"), "E2--E0, E4--E3, E5--E1, E7--E6"),
    ("E6 vs E0", ("comparison", "E6 vs E0"), "E6--E0"),
    ("E7 vs E0", ("comparison", "E7 vs E0"), "E7--E0"),
    ("E6 vs E1", ("comparison", "E6 vs E1"), "E6--E1"),
    ("E7 vs E1", ("comparison", "E7 vs E1"), "E7--E1"),
    ("E7 vs E6", ("comparison", "E7 vs E6"), "E7--E6"),
]

d = pd.read_csv(PAIRS)
# the LDA block contains "E7 vs E6" and the LLM block "E6 vs E1"; comparison rows use any block
res = []
for label, (col, key), pairs in ROWS:
    sub = d[d[col] == key].drop_duplicates(["seed", "model", "comparison"])
    per_model = sub.groupby("model")[["d_auc", "d_brier"]].mean()
    r = {"label": label, "pairs": pairs, "n": len(sub)}
    for m, better in [("d_auc", 1), ("d_brier", -1)]:
        x = per_model[m].to_numpy()
        r[f"{m}_mean"] = x.mean() * 100
        r[f"{m}_imp"] = int((better * x > 0).sum())
        r[f"{m}_p"] = wilcoxon(x).pvalue
    sig = sub[sub["delong_p"] < 0.05]
    r["sig_pos"], r["sig_neg"] = int((sig["d_auc"] > 0).sum()), int((sig["d_auc"] < 0).sum())
    res.append(r)
res = pd.DataFrame(res)
for m in ["d_auc", "d_brier"]:
    res[f"{m}_holm"] = multipletests(res[f"{m}_p"], method="holm")[1]


def pfmt(p):
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    t = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return t + (rf"$^{{{stars}}}$" if stars else "")


def sfmt(v):
    return f"{v:+.2f}".replace("-", "$-$")


OUT = {"auc": HERE / "tbl_block_tests_auc.tex", "dm": HERE / "tbl_block_tests_dm.tex"}
SHORT = {"Adding AUX": "AUX", "Adding LLM": "LLM", "Adding LDA": "LDA"}
SPEC = {"auc": ("d_auc", "AUC", "tbl:block_auc", r"change in AUC ($\times 100$)", "improves"),
        "dm": ("d_brier", "DM", "tbl:block_dm", r"change in Brier loss ($\times 100$; negative values are improvements)", "improves")}
NOTE = (r"\par\smallskip\raggedright\footnotesize\textit{Note:} Blocks are tested over the four matched experiment pairs "
        r"(AUX: E1--E0, E5--E2, E6--E3, E7--E4; LLM: E3--E0, E4--E2, E6--E1, E7--E5; LDA: E2--E0, E4--E3, E5--E1, E7--E6). "
        r"Each model's effect is averaged over five seeds (and the pairs); $p$ is from a two-sided Wilcoxon signed-rank "
        r"test across the 10 models. Improved = number of models that improve. "
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")

for key, (col, name, lab, delta_txt, _) in SPEC.items():
    L = [r"\begin{table}[t]",
         rf"\caption{{Across-model tests of the information blocks and feature sets, {name} (test period). "
         rf"$\Delta$ is the mean {delta_txt} across the 10 models.}}",
         rf"\label{{{lab}}}",
         r"\centering\footnotesize",
         r"\setlength{\tabcolsep}{7pt}",
         r"\begin{tabular}{@{} l r c r @{}}",
         r"\toprule",
         r"Comparison & $\Delta$ & Improved & $p$ \\",
         r"\midrule",
         r"\multicolumn{4}{@{}l}{\textit{Information blocks}} \\"]
    for i, r in res.iterrows():
        if i == 3:
            L += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{Feature sets}} \\"]
        L.append(rf"\quad {SHORT.get(r.label, r.label)} & {sfmt(r[col + '_mean'])} & {r[col + '_imp']}/10 & "
                 f"{pfmt(r[col + '_p'])}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", NOTE, r"\end{table}"]
    OUT[key].write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved", OUT[key])
print(res.round(4).to_string())
