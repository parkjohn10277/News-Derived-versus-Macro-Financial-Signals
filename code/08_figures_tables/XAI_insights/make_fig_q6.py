"""
Q6 figure: do the models agree?   (test days, seed-averaged probability per model)

  (a) Spearman correlation of predicted probabilities, models grouped by family
  (b) Single-model AUC against two benchmarks on the same test days: the logistic regression on the same
      feature set (dashed) and the persistence forecast (dotted), both from ../../GDP_corrected/Baselines
   Chronological record: every column is a test day, every row a model.  Top strip: size of the actual flow.

    python make_fig_q6.py --exp E7      ->  E7/q6_agreement_figure.pdf / .png
"""
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

import run_insights as R

C_OK, C_BAD = "#00b88a", "#ff2d6f"
C_BUY, C_SELL = "#2a78d6", "#ffab00"
CMAP = plt.get_cmap("viridis")


def family_strip(ax, colors, w):
    """Thin family-coloured bar just left of the axis, one block per model row."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((-1.6 * w, i - 0.5), w, 1, color=c, lw=0, transform=tr, clip_on=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E7")
    exp = ap.parse_args().exp
    out = R.HERE / exp
    P, flow, y = R.ensemble_predictions(exp)
    keys = list(P.columns)
    labels = [{"RF": "Random Forest"}.get(R.LABEL[k], R.LABEL[k]) for k in keys]
    fam_col = [R.FAM_COLORS[R.FAMILY[k]] for k in keys]
    yv = y.to_numpy()
    # AUC as in the performance table: per seed, then averaged over seeds.
    per_seed = {}
    for sd in R.SEEDS:
        fr = {}
        for k in keys:
            f = R.run_dir(k, exp, sd) / "predictions.csv"
            if f.exists():
                fr[k] = pd.read_csv(f, parse_dates=["date"]).set_index("date")["prob_net_buy"].reindex(P.index)
        if len(fr) == len(keys):
            per_seed[sd] = pd.DataFrame(fr)
    aucs = np.array([np.mean([R.core.fast_auc(yv, d[k].to_numpy()) for d in per_seed.values()]) for k in keys])
    # benchmarks: logit on the same feature set (deterministic, seed 42) and persistence (rank by day-t flow)
    base = R.C.SUBMISSION / "GDP_corrected" / "Baselines" / "results" / "input_gdp_corrected"
    lg = pd.read_csv(base / "logit" / exp / "seed_42" / "predictions.csv", parse_dates=["date"],
                     encoding="utf-8-sig").set_index("date").reindex(P.index)
    assert lg.prob_net_buy.notna().all() and (lg.actual_direction.to_numpy() == yv).all()
    logit = R.core.fast_auc(yv, lg.prob_net_buy.to_numpy())
    ps = pd.read_csv(base / "persistence" / "all" / "seed_0" / "predictions.csv", parse_dates=["date"],
                     encoding="utf-8-sig").set_index("date").reindex(P.index)
    persist = R.core.fast_auc(yv, ps.flow_day_t.to_numpy())
    print("seeds used", list(per_seed), "logit", round(logit, 4), "persistence", round(persist, 4),
          dict(zip(keys, aucs.round(4))))
    correct = ((P >= 0.5).astype(int).values == yv[:, None])
    nc = correct.sum(axis=1)
    order = np.arange(len(nc))  # chronological
    xd = mdates.date2num(P.index.to_pydatetime())
    half = 0.5
    n = len(keys)

    fig = plt.figure(figsize=(7.2, 3.5), layout="constrained")
    ax_a, ax_b = fig.subplots(1, 2, width_ratios=[1.15, 1])

    # (a) correlation heatmap
    M = P.corr(method="spearman").to_numpy().copy()
    M[np.triu_indices(n, 1)] = np.nan
    im = ax_a.imshow(np.ma.masked_invalid(M), cmap=CMAP, vmin=0.7, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            ax_a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                      color="white" if M[i, j] < 0.88 else R.INK)
    ax_a.set_xticks(range(n), labels, rotation=45, ha="right")
    ax_a.set_yticks(range(n), labels)
    family_strip(ax_a, fam_col, 0.012)
    ax_a.tick_params(axis="y", pad=5)
    ax_a.tick_params(length=0)
    ax_a.xaxis.set_minor_locator(plt.NullLocator())
    ax_a.yaxis.set_minor_locator(plt.NullLocator())
    for s in ax_a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.02)
    cb.set_label("Spearman correlation")
    cb.ax.minorticks_off()
    ax_a.text(-0.3, 1.02, "a", transform=ax_a.transAxes, fontsize=8, fontweight="bold")

    # (b) single-model AUC vs the logit and persistence benchmarks
    pos = np.arange(n)[::-1]
    ax_b.barh(pos, aucs, color=fam_col, height=0.65, linewidth=0, zorder=2)
    for p_, a in zip(pos, aucs):
        ax_b.text(a - 0.003, p_, f"{a:.3f}", va="center", ha="right", fontsize=6, color="white")
    ax_b.axvline(logit, color="#7a1fd1", lw=1.2, ls="--", zorder=0)
    ax_b.axvline(persist, color="#6f6f6f", lw=1.2, ls=":", zorder=0)
    ax_b.set_yticks(pos, labels)
    ax_b.tick_params(axis="y", pad=3)
    ax_b.set_xlim(0.5, max(aucs.max(), logit) + 0.045)
    ax_b.set_xlabel("Test AUC")
    ax_b.yaxis.set_minor_locator(plt.NullLocator())
    ax_b.xaxis.set_minor_locator(plt.NullLocator())
    ax_b.legend(handles=[Patch(color=c, label=f) for f, c in R.FAM_COLORS.items()] +
                [plt.Line2D([], [], color="#7a1fd1", ls="--", label=f"Logistic regression ({logit:.3f})"),
                 plt.Line2D([], [], color="#6f6f6f", ls=":", label=f"Persistence ({persist:.3f})")],
                loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2, frameon=False, fontsize=6, handlelength=1.4)
    ax_b.text(-0.25, 1.02, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold")

    R.save(fig, out / "q6_agreement_figure.pdf")
    print(f"all-correct {int((nc == n).sum())}, all-wrong {int((nc == 0).sum())}, days {len(nc)}")


if __name__ == "__main__":
    main()
