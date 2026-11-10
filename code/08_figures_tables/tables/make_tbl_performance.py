"""
Table: test-set performance of all 10 models in every experiment (E0-E7), seed means.
Best value per row in bold, second best in italics.

    python make_tbl_performance.py   ->  tbl_performance.tex
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent.parent / "GDP_corrected" / "results_summary.csv"
OUT = HERE / "tbl_performance.tex"

# ---- edit here --------------------------------------------------------------------
METRICS = [("AUC", "AUC"), ("MCC", "MCC"),
           ("Directional_Accuracy", "DA")]          # add ("Brier_Skill","BSS") etc. if wanted
SHADE = ""                                          # no shading (not allowed in submission)
SHADE2 = ""
GROUP_RGB = {"Machine learning": "42,120,214", "Transformer": "235,104,52", "LSTM-based": "27,175,122"}
MODEL_GROUPS = [
    ("Machine learning", [("RandomForest", "RF"), ("XGBoost", "XGB"), ("LightGBM", "LGBM")]),
    ("Transformer", [("iTransformer", "iTransformer")]),
    ("LSTM-based", [("DA-RNN", "DA-RNN"), ("BiLSTM", "BiLSTM"), ("LSTM", "LSTM"),
                    ("CNN-LSTM", "CNN-LSTM"), ("GRU-LSTM", "GRU-LSTM"),
                    ("BiGRU-LSTM", "BiGRU-LSTM")]),
]
EXPERIMENTS = [("E0", "KRX"), ("E1", "+AUX"), ("E2", "+LDA"), ("E3", "+LLM"),
               ("E4", "+LDA+LLM"), ("E5", "+LDA+AUX"), ("E6", "+LLM+AUX"), ("E7", "All")]
DEC = 3
# -----------------------------------------------------------------------------------

s = pd.read_csv(SUMMARY).set_index(["model", "experiment"])
models = [m for _, g in MODEL_GROUPS for m, _ in g]
n = len(models)


def cell(model, exp, metric):
    """Rank each value within its model column, i.e. across the experiments."""
    col = sorted({round(s.loc[(model, e), f"{metric}_mean"], DEC) for e, _ in EXPERIMENTS},
                 reverse=True)
    v = round(s.loc[(model, exp), f"{metric}_mean"], DEC)
    t = f"{v:.{DEC}f}"
    if v == col[0]:
        return rf"{SHADE}\textbf{{{t}}}"
    return rf"{SHADE2}\textit{{{t}}}" if len(col) > 1 and v == col[1] else t


def row_cells(exp, metric):
    return [cell(m, exp, metric) for m in models]


L = [r"\begin{table*}[t]",
     r"\caption{Out-of-sample classification performance of all models across the feature sets "
     r"(test period, mean over five seeds). For each model and metric, the best feature set is "
     r"in bold and the second best in italics.}",
     r"\label{tbl:performance}",
     r"\centering\scriptsize",
     r"\setlength{\tabcolsep}{3.6pt}",
     r"\renewcommand{\arraystretch}{0.95}",
     r"\begin{tabular}{@{} l l " + "c" * n + r" @{}}",
     r"\toprule"]
head1, rules, col = [], [], 3
for g, ms in MODEL_GROUPS:
    head1.append(rf"\multicolumn{{{len(ms)}}}{{c}}{{{g}}}")
    rules.append(rf"\cmidrule(lr){{{col}-{col + len(ms) - 1}}}")
    col += len(ms)
L.append(" & & " + " & ".join(head1) + r" \\")
L.append(" ".join(rules))
L.append(r"Exp. & Metric & " + " & ".join(lab for _, g in MODEL_GROUPS for _, lab in g) + r" \\")
L.append(r"\midrule")
for i, (exp, feat) in enumerate(EXPERIMENTS):
    for j, (metric, mlab) in enumerate(METRICS):
        first = rf"\textbf{{{exp}}}" if j == 0 else (feat if j == 1 else "")
        L.append(f"{first} & {mlab} & " + " & ".join(row_cells(exp, metric)) + r" \\")
    if i < len(EXPERIMENTS) - 1:
        L.append(r"\midrule")
L += [r"\bottomrule",
      r"\end{tabular}",
      r"\par\smallskip\raggedright\footnotesize\textit{Note:} KRX = foreign net-buying history; "
      r"AUX = 13 auxiliary market and macro variables; LDA = 11 topic intensities; "
      r"LLM = 4 sentiment series. AUC = area under the ROC curve; MCC = Matthews correlation "
      r"coefficient; DA = directional accuracy.",
      r"\end{table*}"]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("saved", OUT)
