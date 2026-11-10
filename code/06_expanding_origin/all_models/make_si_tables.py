"""
LaTeX for the expanding-origin SI tables, from analysis/ (run analyze_all_models.py first).

    python make_si_tables.py      -> analysis/tbl_S14_block_by_window.tex, analysis/tbl_S16_auc_by_window.tex
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE / "analysis"
WIN = {"W1": "2022", "W2": "2023", "W3": "2024", "W4": "2025--26"}
ORDER = ["RandomForest", "XGBoost", "LightGBM", "iTransformer", "LSTM", "CNN-LSTM", "GRU-LSTM",
         "BiGRU-LSTM", "BiLSTM", "DA-RNN", "Logit"]
LABEL = {"RandomForest": "Random Forest", "Logit": "Logistic regression"}
BLOCK = {"AUX": "Macro-financial (AUX)", "LDA": "LDA topics", "LLM": "LLM sentiment"}


def pfmt(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def s14():
    S = pd.read_csv(A / "block_by_window.csv")
    E = pd.read_csv(A / "block_effects_by_model.csv")
    logit = E[E.model == "Logit"].set_index(["block", "window"]).d_auc_x100
    cols = list(WIN) + ["pooled"]
    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.15}",
           r"\caption{Change in test AUC ($\times 100$) from adding each information block, by test year. "
           r"In each window the models are tuned on the validation year, refitted on all earlier data and evaluated once "
           r"on the test year; 2025--26 is the main test period. Each entry is the mean over the ten models of the "
           r"average change across the block's four matched pairs of feature sets, with the five seeds averaged first. "
           r"In parentheses, the number of models that improve. $p$ is from a two-sided Wilcoxon signed-rank test across "
           r"the ten models (40 window--model effects for the pooled column). The logistic regression is shown for "
           r"reference and is not part of the tests. The 2025--26 column equals Table~\ref{tbl:block_auc} of the main text. "
           r"The LDA topics are those of the main analysis, estimated on the full corpus, and the LLM may have seen news "
           r"from 2022--24 in pretraining; both can only favour the text features in the earlier windows.}",
           r"\label{tbl:expanding_origin_blocks}",
           r"\begin{tabular}{@{} l l ccccc @{}}", r"\toprule",
           r" & & " + " & ".join(WIN.values()) + r" & Pooled \\", r"\midrule"]
    for blk in ["AUX", "LDA", "LLM"]:
        x = S[S.block == blk].set_index("window")
        mean = " & ".join(f"${x.loc[w, 'mean_d_auc_x100']:+.2f}$ ({int(x.loc[w, 'n_improved'])}/{int(x.loc[w, 'n_models'])})"
                          for w in cols)
        p = " & ".join(pfmt(x.loc[w, "wilcoxon_p"]) for w in cols)
        lg = " & ".join(f"${logit.loc[(blk, w)]:+.2f}$" for w in WIN) + " & --"
        out += [BLOCK[blk] + r" & Ten models & " + mean + r" \\",
                r" & $p$ & " + p + r" \\",
                r" & Logistic regression & " + lg + r" \\"]
        if blk != "LLM":
            out.append(r"\addlinespace[3pt]")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out).replace("+", "$+$").replace("$$+$", "$+").replace("-", "$-$", 0)


def s16():
    D = pd.read_csv(A / "auc_by_window.csv")
    out = [r"\begin{footnotesize}", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{@{} l l cccccccc @{}}",
           r"\caption{Test AUC by test year, model and feature set (mean over five seeds; the logistic regression is "
           r"deterministic). The 2025--26 rows equal Table~\ref{tbl:performance} of the main text.}",
           r"\label{tbl:expanding_origin_auc}\\", r"\toprule",
           r"Test year & Model & E0 & E1 & E2 & E3 & E4 & E5 & E6 & E7 \\", r"\midrule", r"\endfirsthead",
           r"\toprule", r"Test year & Model & E0 & E1 & E2 & E3 & E4 & E5 & E6 & E7 \\", r"\midrule", r"\endhead",
           r"\bottomrule", r"\endlastfoot"]
    for i, (w, yr) in enumerate(WIN.items()):
        x = D[D.window == w].set_index("model")
        for j, m in enumerate(ORDER):
            vals = " & ".join(f"{x.loc[m, f'E{k}']:.3f}" for k in range(8))
            out.append(f"{yr if j == 0 else ''} & {LABEL.get(m, m)} & {vals} \\\\")
        if i < len(WIN) - 1:
            out.append(r"\midrule")
    out += [r"\end{longtable}", r"\end{footnotesize}"]
    return "\n".join(out)


if __name__ == "__main__":
    t14 = s14()
    (A / "tbl_S14_block_by_window.tex").write_text(t14, encoding="utf-8")
    (A / "tbl_S16_auc_by_window.tex").write_text(s16(), encoding="utf-8")
    print(t14)
