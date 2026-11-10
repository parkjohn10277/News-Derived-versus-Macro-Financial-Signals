"""
Table: within-model tests of the three information blocks (part i of the significance analysis).
Per model and block: mean change in AUC and Brier score over the 20 matched comparisons
(4 experiment pairs x 5 seeds) and the Cauchy-combined p-value of the DeLong (AUC) and
Diebold-Mariano (Brier) tests.  Cells with p < 0.05 are set in bold (improvement) or italics
(deterioration); no shading, as required for submission.

    python make_tbl_within_model.py   ->  tbl_within_model.tex
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "GDP_corrected" / "Tests" / "results" / "1_within_model_pairs.csv"
OUT = {"auc": HERE / "tbl_within_model_auc.tex", "dm": HERE / "tbl_within_model_dm.tex"}

BLOCKS = [("AUX", "AUX"), ("LLM", "LLM"), ("LDA", "LDA")]
GROUPS = [("Machine learning", ["Random Forest", "XGBoost", "LightGBM"]),
          ("Transformer", ["iTransformer"]),
          ("LSTM-based", ["DA-RNN", "BiLSTM", "LSTM", "CNN-LSTM", "GRU-LSTM", "BiGRU-LSTM"])]
GOOD, BAD = "bold", "italic"
ALPHA = 0.05


def cauchy(p):
    """Cauchy combination test (Liu and Xie, 2020), equal weights."""
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return 0.5 - np.arctan(np.mean(np.tan((0.5 - p) * np.pi))) / np.pi


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


d = pd.read_csv(SRC)
d = d[d.block.isin([b for b, _ in BLOCKS])]
g = d.groupby(["model", "block"])
R = pd.DataFrame({"dauc": g.d_auc.mean() * 100, "p_auc": g.delong_p.apply(cauchy),
                  "dbr": g.d_brier.mean() * 100, "p_br": g.dm_brier_p.apply(cauchy)})
assert (g.size() == 20).all()


def mark(delta, p, better_if_positive):
    if p >= ALPHA:
        return ""
    return GOOD if (delta > 0) == better_if_positive else BAD


def styled(text, style):
    if style == "bold":
        return r"{\boldmath\bfseries " + text + "}"
    if style == "italic":
        return r"\textit{" + text + "}"
    return text


def num(x, sign=True):
    s = f"{abs(x):.2f}"
    return (r"$-$" if x < 0 else (r"\phantom{$-$}" if sign else "")) + s


CAP = {"auc": (r"Within-model tests of the information blocks, AUC (test period). For each model, $\Delta$ is the mean "
               r"change in AUC ($\times 100$) from adding the block over four matched experiment pairs and five seeds, "
               r"and $p$ is the Cauchy-combined $p$-value of the 20 DeLong tests. Bold and italic entries mark combined "
               r"$p<0.05$ for an improvement and a deterioration, respectively.", "tbl:within_auc"),
       "dm": (r"Within-model tests of the information blocks, DM (test period). For each model, $\Delta$ is the mean "
              r"change in Brier loss ($\times 100$; negative values are improvements) from adding the block over four matched "
              r"experiment pairs and five seeds, and $p$ is the Cauchy-combined $p$-value of the 20 Diebold--Mariano tests. "
              r"Bold and italic entries mark combined $p<0.05$ for an improvement and a deterioration, respectively.", "tbl:within_dm")}
COLS = {"auc": ("dauc", "p_auc", True), "dm": ("dbr", "p_br", False)}

for key, (dcol, pcol, pos_good) in COLS.items():
    cap, lab = CAP[key]
    L = [r"\begin{table}[t]", rf"\caption{{{cap}}}", rf"\label{{{lab}}}",
         r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{} l " + "rr " * len(BLOCKS) + r"@{}}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for _, t in BLOCKS) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BLOCKS))),
         "Model" + r" & $\Delta$ & $p$" * len(BLOCKS) + r" \\", r"\midrule"]
    for gi, (gname, models) in enumerate(GROUPS):
        L.append(rf"\multicolumn{{{1 + 2 * len(BLOCKS)}}}{{@{{}}l}}{{\textit{{{gname}}}}} \\")
        for m in models:
            cells = []
            for b, _ in BLOCKS:
                r = R.loc[(m, b)]
                st = mark(r[dcol], r[pcol], pos_good)
                cells += [styled(num(r[dcol]), st), styled(fmt_p(r[pcol]), st)]
            L.append(rf"\quad {m} & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT[key].write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved", OUT[key])
