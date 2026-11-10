"""
The two forecasting-performance figures of Section 5, rebuilt from the GDP-corrected run.

    python make_fig_forecasting.py
        -> auc_by_set.pdf / .png              (fig:auc_by_set)    "Test AUC by feature set"
        -> delta_auc.pdf / .png               (fig:delta_auc)     "Change in test AUC from adding each information block"

Inputs (nothing is recomputed):
    ../../GDP_corrected/results_summary.csv                  mean test AUC over five seeds
    ../../GDP_corrected/Tests/results/1_within_model_pairs.csv    DeLong test per model x pair x seed
    ../../GDP_corrected/Tests/results/2_block_wilcoxon.csv        Wilcoxon test across the ten models

Delta-AUC markers: each model's change in AUC averaged over the 4 matched pairs x 5 seeds; filled if the
Cauchy-combined DeLong p-value of those 20 tests is < 0.05 (as in tbl_within_model).  The bottom row is the mean
over the ten models, filled if the across-model Wilcoxon p-value is < 0.05 (as in tbl_block_tests).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
GDP = HERE.parent.parent / "GDP_corrected"
SUMMARY = GDP / "results_summary.csv"
PAIRS = GDP / "Tests" / "results" / "1_within_model_pairs.csv"
WILCOXON = GDP / "Tests" / "results" / "2_block_wilcoxon.csv"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

BLUE, LIGHT, ORANGE, GREEN, RED, GREY = "#2a78d6", "#c5dbf5", "#eb6834", "#1baf7a", "#e34948", "#8c8c8c"
EXPS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
SET_LABEL = {"E0": "KRX", "E1": "+AUX", "E2": "+LDA", "E3": "+LLM", "E4": "+LDA+LLM",
             "E5": "+LDA+AUX", "E6": "+LLM+AUX", "E7": "All"}
WITH_AUX = {"E1", "E5", "E6", "E7"}
FAMILIES = {"Machine learning": (["RandomForest", "XGBoost", "LightGBM"], BLUE),
            "Transformer": (["iTransformer"], ORANGE),
            "LSTM-based": (["DA-RNN", "BiLSTM", "LSTM", "CNN-LSTM", "GRU-LSTM", "BiGRU-LSTM"], GREEN)}
ALPHA = 0.05


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight")
    print("saved", HERE / f"{stem}.pdf")


# ------------------------------------------------------------------ figure 1: AUC by feature set
def fig_auc_by_set():
    s = pd.read_csv(SUMMARY).pivot(index="model", columns="experiment", values="AUC_mean")[EXPS]
    mean = s.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.06})
    y = np.arange(len(EXPS))[::-1]

    a.barh(y, mean.values, height=0.65, color=[BLUE if e in WITH_AUX else LIGHT for e in EXPS])
    for yi, e in zip(y, EXPS):
        a.text(mean[e] - 0.002, yi, f"{mean[e]:.3f}", va="center", ha="right",
               color="white" if e in WITH_AUX else "black")
    a.axvline(mean["E0"], color="black", ls="--", lw=0.9)
    a.set_yticks(y, [f"{e}  {SET_LABEL[e]}" for e in EXPS])
    a.tick_params(axis="y", length=0)
    a.set_xlabel("Test AUC, mean of 10 models")
    a.legend(handles=[Patch(color=BLUE, label="With AUX"), Patch(color=LIGHT, label="Without AUX")],
             loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)

    h = 0.27
    for k, (fam, (models, col)) in enumerate(FAMILIES.items()):
        b.barh(y + (1 - k) * h, s.loc[models].mean().values, height=h, color=col, label=fam)
    b.set_yticks(y, [])
    b.tick_params(axis="y", length=0)
    b.set_xlabel("Test AUC, mean within family")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)

    for ax, tag in ((a, "a"), (b, "b")):
        ax.set_xlim(0.55, 0.725)
        ax.set_xticks([0.55, 0.60, 0.65, 0.70])
        ax.set_ylim(-0.6, len(EXPS) - 0.4)
        ax.text(-0.02 if ax is b else -0.30, 1.03, tag, transform=ax.transAxes, fontweight="bold",
                fontsize=10, va="bottom")
    save(fig, "auc_by_set")
    return mean


# ------------------------------------------------------------------ figure 2: change in AUC by block
def cauchy(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return 0.5 - np.arctan(np.mean(np.tan((0.5 - p) * np.pi))) / np.pi


def fig_delta_auc():
    d = pd.read_csv(PAIRS)
    d = d[d.block.isin(["AUX", "LLM", "LDA"])]
    g = d.groupby(["block", "model"])
    assert (g.size() == 20).all()
    eff = g.d_auc.mean() * 100
    pv = g.delong_p.apply(cauchy)
    w = pd.read_csv(WILCOXON)
    w = w[(w.metric == "d_auc") & w.unit.str.startswith("model")].set_index("block")

    rows = [("Random Forest", 0), ("XGBoost", 1), ("LightGBM", 2), ("iTransformer", 3.5),
            ("DA-RNN", 5), ("BiLSTM", 6), ("LSTM", 7), ("CNN-LSTM", 8), ("GRU-LSTM", 9), ("BiGRU-LSTM", 10)]
    y_mean = 11.8
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True, gridspec_kw={"wspace": 0.04})

    def marker(ax, x, yy, p, kind="o", size=34):
        if p < ALPHA:
            ax.scatter(x, yy, s=size, marker=kind, color=BLUE if x > 0 else RED, zorder=3, linewidths=0)
        else:
            ax.scatter(x, yy, s=size, marker=kind, facecolor="white", edgecolor=GREY, linewidths=1.2, zorder=3)

    for ax, blk in zip(axes, ["AUX", "LLM", "LDA"]):
        ax.axvline(0, color="black", lw=0.9)
        ax.axhline((10 + y_mean) / 2, color="#d0d0d0", lw=0.8)
        for m, yy in rows:
            marker(ax, eff[(blk, m)], yy, pv[(blk, m)])
        marker(ax, w.loc[blk, "mean_effect"] * 100, y_mean, w.loc[blk, "p_two_sided"], kind="D", size=48)
        ax.set_title(f"+{blk}", fontweight="bold")
        ax.set_xlim(-9.5, 9.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([-5, 0, 5], ["−5", "0", "5"])
    axes[0].set_yticks([yy for _, yy in rows] + [y_mean], [m for m, _ in rows] + ["Mean"])
    axes[0].set_ylim(y_mean + 0.8, -0.8)
    axes[0].tick_params(axis="y", length=0)
    axes[1].set_xlabel("Change in AUC (×100)")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="Increase (p < 0.05)"),
               Line2D([], [], marker="o", ls="", color=RED, label="Decrease (p < 0.05)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY,
                      markeredgewidth=1.2, label="Not significant")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    save(fig, "delta_auc")
    return eff.unstack(0), pv.unstack(0), w


if __name__ == "__main__":
    mean = fig_auc_by_set()
    print("mean AUC by set:", mean.round(3).to_dict())
    eff, pv, w = fig_delta_auc()
    print((eff.round(2)).to_string())
    print("Wilcoxon:", w[["mean_effect", "p_two_sided"]].round(4).to_dict())
