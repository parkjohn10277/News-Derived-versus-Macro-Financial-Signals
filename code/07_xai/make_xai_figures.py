"""
XAI figures -- the three panels of the analysis design, submission style
(SciencePlots 'nature', no titles, panel letters only, vector PDF).

    python make_xai_figures.py              # every model that has outputs
    python make_xai_figures.py --exp E7

Reads   outputs/<model>/<exp>/            (written by run_xai.py)
Writes  figures/<exp>/*.pdf  and  figures/<exp>/tables/*.csv  (every plotted number)

Panel 1  Temporal importance of information sources
  xai_1a_temporal_importance.pdf     share of |group SHAP| per year, mean of all models
  xai_1b_temporal_by_family.pdf      same, one heatmap per model family
Panel 2  Feature interactions -> incremental predictive power (out-of-sample)
  xai_2a_interaction_matrix.pdf      diagonal: each group's Shapley share of (AUC - 0.5)
                                     lower triangle: pairwise Shapley interaction
                                     (< 0 overlapping information, ~0 independent, > 0 complementary)
  xai_2b_interaction_network.pdf     the same numbers as a network
Panel 3  Information use by volatility regime (VKOSPI, previous day)
  xai_3a_regime_lag_heatmaps.pdf     groups x days-before-forecast: low | high | high - low
  xai_3b_regime_lag_by_family.pdf    same per model family
  xai_3c_regime_group_share.pdf      share of |group SHAP|, low vs high volatility
  xai_3d_regime_performance.pdf      OOS AUC contribution of each group, low vs high (H2)
Supplementary
  xai_s1_temporal_per_model.pdf      panel 1 for every model
  xai_s2_feature_importance.pdf      top-20 individual features (native SHAP)
  xai_s3_attention_vs_shap.pdf       DA-RNN / iTransformer attention vs SHAP
  tables/xai_qc_table.csv            verification, efficiency and agreement checks per model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors            # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402
import scienceplots                            # noqa: E402,F401
from matplotlib.cm import ScalarMappable       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.lines import Line2D            # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle            # noqa: E402
from matplotlib.ticker import NullLocator      # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xai_config as C                         # noqa: E402

plt.style.use(["science", "nature", "no-latex"])
plt.rcParams.update({"savefig.dpi": 600, "figure.dpi": 150})

INK = "#0b0b0b"
INK_2 = "#3d3c39"
NAN_COLOR = "#eeeeec"
RED = "#e34948"
BLUE = "#2a78d6"
MID = "#f0efec"
NEUTRAL = "#6b6a66"
DIV = LinearSegmentedColormap.from_list("xai_div", [RED, MID, BLUE])
GROUPS = C.GROUPS


def slug(g: str) -> str:
    return C.GROUP_SLUG[g]


# =================================================================== loading
def load_all(exp: str) -> list[dict]:
    recs = []
    for key, spec in C.MODELS.items():
        d = C.OUT_ROOT / key / exp
        if not (d / "meta.json").exists():
            print(f"  [skip] {key:<14} no outputs in {d}")
            continue
        rec = {"key": key, "label": spec["label"], "family": spec["family"],
               "meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}
        if (d / "local_group_shap.csv").exists():
            rec["local"] = pd.read_csv(d / "local_group_shap.csv", parse_dates=["date"], encoding="utf-8-sig")
        if (d / "perf_decomposition.csv").exists():
            rec["perf"] = pd.read_csv(d / "perf_decomposition.csv", encoding="utf-8-sig")
        if (d / "native_shap.npz").exists():
            with np.load(d / "native_shap.npz", allow_pickle=False) as z:
                rec["native"] = {k: z[k] for k in z.files}
        if (d / "attention_by_group.csv").exists():
            rec["att"] = pd.read_csv(d / "attention_by_group.csv", parse_dates=["date"], encoding="utf-8-sig")
        parts = [p for p in ("local", "perf", "native", "att") if p in rec]
        print(f"  [ok]   {key:<14} {', '.join(parts)}")
        recs.append(rec)
    return recs


# =================================================================== drawing helpers
def _tint(color: str, t: float) -> np.ndarray:
    """t = 0 -> near white, t = 1 -> the (slightly darkened) group colour."""
    c = np.array(mcolors.to_rgb(color)) * 0.8
    lo = np.array([0.975, 0.975, 0.97])
    return lo + (c - lo) * float(np.clip(t, 0, 1))


def _heatmap_axes(ax, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0, top=False, right=False)
    for s in ax.spines.values():
        s.set_visible(False)


def _cat_axis(ax, which: str) -> None:
    """No minor ticks on a categorical axis."""
    (ax.yaxis if which == "y" else ax.xaxis).set_minor_locator(NullLocator())


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.01, 1.03, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


def group_heatmap(ax, M, groups, col_labels, vmax, scale=100.0, fmt="{:.0f}",
                  annotate=True, fontsize=5.0) -> None:
    """Rows keep their group hue; intensity = value, one shared scale."""
    G, K = M.shape
    img = np.zeros((G, K, 3))
    for i, g in enumerate(groups):
        for k in range(K):
            v = M[i, k]
            img[i, k] = (mcolors.to_rgb(NAN_COLOR) if not np.isfinite(v)
                         else _tint(C.GROUP_COLORS[g], 0.06 + 0.94 * (v / vmax if vmax > 0 else 0)))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    _heatmap_axes(ax, G, K)
    ax.set_xticks(range(K))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(G))
    ax.set_yticklabels(groups)
    if annotate:
        for i in range(G):
            for k in range(K):
                v = M[i, k]
                if np.isfinite(v):
                    t = v / vmax if vmax > 0 else 0
                    ax.text(k, i, fmt.format(v * scale), ha="center", va="center",
                            fontsize=fontsize, color="white" if t > 0.62 else INK)


def diverging_heatmap(ax, M, groups, col_labels, vlim, scale=100.0, fmt="{:+.1f}",
                      annotate=True, fontsize=5.0):
    cmap = DIV.copy()
    cmap.set_bad(NAN_COLOR)
    im = ax.imshow(np.ma.masked_invalid(M * scale), cmap=cmap, vmin=-vlim * scale,
                   vmax=vlim * scale, aspect="auto", interpolation="nearest")
    _heatmap_axes(ax, *M.shape)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(groups)
    if annotate:
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                v = M[i, k]
                if np.isfinite(v):
                    ax.text(k, i, fmt.format(v * scale), ha="center", va="center", fontsize=fontsize,
                            color="white" if abs(v) > 0.62 * vlim else INK)
    return im


def tint_colorbar(fig, axes, vmax, label, scale=100.0):
    cmap = LinearSegmentedColormap.from_list("xai_tint", [_tint(NEUTRAL, 0.06), _tint(NEUTRAL, 1.0)])
    sm = ScalarMappable(norm=Normalize(0, vmax * scale), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015, aspect=18)
    _style_cb(cb, label)
    return cb


def _style_cb(cb, label):
    cb.set_label(label, fontsize=6)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4, which="major")
    cb.ax.minorticks_off()


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# =================================================================== panel 1: temporal
def _phi_cols(df: pd.DataFrame) -> dict[str, str]:
    return {g: f"phi_{slug(g)}" for g in GROUPS if f"phi_{slug(g)}" in df.columns}


def temporal_tables(local: pd.DataFrame):
    cols = _phi_cols(local)
    a = local[list(cols.values())].abs()
    a.columns = list(cols)
    mag = a.groupby(local["date"].dt.year).mean()
    mag.index.name = "year"
    share = mag.div(mag.sum(axis=1), axis=0)
    return mag.reindex(columns=GROUPS), share.reindex(columns=GROUPS)


def _stack_mean(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean over models of year x group tables."""
    arr = pd.concat(frames, names=["model", "year"])
    out = arr.groupby(level=1).mean()
    out.index.name = "year"
    return out


def _test_start_line(ax, years: list[int], recs: list[dict]) -> None:
    rec = next((r for r in recs if "local" in r), None)
    if rec is None:
        return
    ts = pd.Timestamp(rec["meta"]["loader"]["test_start"])
    if ts.year not in years:
        return
    d = rec["local"]["date"]
    in_year = d[d.dt.year == ts.year]
    frac = float((in_year < ts).mean()) if len(in_year) else 0.0
    ax.axvline(years.index(ts.year) - 0.5 + frac, color=INK, lw=0.7, ls=(0, (2.5, 1.5)))


def fig_temporal(recs, outdir: Path, tables: Path) -> None:
    shares, mags = {}, {}
    for r in recs:
        if "local" in r:
            mags[r["key"]], shares[r["key"]] = temporal_tables(r["local"])
    if not shares:
        print("  [panel 1] no local group SHAP yet -- skipped")
        return
    long = pd.concat({k: v.stack() for k, v in shares.items()}, names=["model", "year", "group"])
    long.rename("share").reset_index().to_csv(tables / "xai_1_temporal_share_by_model.csv", index=False)
    pd.concat({k: v.stack() for k, v in mags.items()}, names=["model", "year", "group"]) \
        .rename("mean_abs_phi").reset_index().to_csv(tables / "xai_1_temporal_magnitude_by_model.csv",
                                                     index=False)

    mean = _stack_mean(shares).reindex(columns=GROUPS)
    mean.to_csv(tables / "xai_1a_temporal_share_mean.csv")
    years = [int(y) for y in mean.index]
    M = mean.T.to_numpy()
    vmax = float(np.nanmax(M))

    fig = plt.figure(figsize=(7.0, 1.55), layout="constrained")
    ax = fig.subplots()
    group_heatmap(ax, M, GROUPS, [str(y) for y in years], vmax)
    _test_start_line(ax, years, recs)
    tint_colorbar(fig, ax, vmax, "Share of |SHAP| (%)")
    save(fig, outdir / "xai_1a_temporal_importance.pdf")

    # ---- by family
    fams = [f for f in C.FAMILIES if any(r["family"] == f and r["key"] in shares for r in recs)]
    fam_mean = {f: _stack_mean({r["key"]: shares[r["key"]] for r in recs
                                if r["family"] == f and r["key"] in shares}).reindex(index=mean.index,
                                                                                       columns=GROUPS)
                for f in fams}
    vmax_f = float(np.nanmax([np.nanmax(m.to_numpy()) for m in fam_mean.values()]))
    fig = plt.figure(figsize=(7.0, 1.15 * len(fams) + 0.35), layout="constrained")
    axes = fig.subplots(len(fams), 1, squeeze=False, sharex=True)[:, 0]
    for ax, f in zip(axes, fams):
        group_heatmap(ax, fam_mean[f].T.to_numpy(), GROUPS, [str(y) for y in years], vmax_f,
                      fontsize=4.6)
        _test_start_line(ax, years, recs)
        ax.set_ylabel(f, fontsize=6.3, labelpad=4)
    tint_colorbar(fig, list(axes), vmax_f, "Share of |SHAP| (%)")
    save(fig, outdir / "xai_1b_temporal_by_family.pdf")
    pd.concat(fam_mean, names=["family"]).to_csv(tables / "xai_1b_temporal_share_by_family.csv")

    # ---- supplementary: every model
    keys = [r["key"] for r in recs if r["key"] in shares]
    labels = {r["key"]: r["label"] for r in recs}
    ncols = 2
    nrows = int(np.ceil(len(keys) / ncols))
    vmax_m = float(np.nanmax([np.nanmax(shares[k].to_numpy()) for k in keys]))
    fig = plt.figure(figsize=(7.0, 1.02 * nrows + 0.3), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(keys):
            ax.axis("off")
            continue
        k = keys[idx]
        sh = shares[k].reindex(index=mean.index)
        group_heatmap(ax, sh.T.to_numpy(), GROUPS, [str(y) for y in years], vmax_m, annotate=False)
        _test_start_line(ax, years, recs)
        ax.set_title(labels[k], fontsize=6.3, pad=2, loc="left")
        ax.tick_params(axis="x", labelsize=4.8, rotation=90)
        ax.tick_params(axis="y", labelsize=5.2)
        if idx % ncols:
            ax.set_yticklabels([])
    tint_colorbar(fig, axes.ravel().tolist(), vmax_m, "Share of |SHAP| (%)")
    save(fig, outdir / "xai_s1_temporal_per_model.pdf")


# =================================================================== panel 2: interactions
def perf_matrices(recs, scope: str = "test", metric: str = "auc"):
    """Per model: G x G matrix -- diagonal = Shapley share of performance,
    off-diagonal = pairwise Shapley interaction; plus the total."""
    out = {}
    for r in recs:
        p = r.get("perf")
        if p is None:
            continue
        q = p[(p["scope"] == scope) & (p["metric"] == metric)]
        if q.empty:
            continue
        M = np.full((len(GROUPS), len(GROUPS)), np.nan)
        total = np.nan
        for row in q.itertuples(index=False):
            if row.kind == "total":
                total = row.value
            elif row.kind == "main" and row.a in GROUPS:
                i = GROUPS.index(row.a)
                M[i, i] = row.value
            elif row.kind == "interaction" and row.a in GROUPS and row.b in GROUPS:
                i, j = GROUPS.index(row.a), GROUPS.index(row.b)
                M[i, j] = M[j, i] = row.value
        out[r["key"]] = (M, total)
    return out


def _mean_se(stack: np.ndarray):
    n = np.sum(np.isfinite(stack), axis=0)
    mean = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.full(mean.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = sd / np.sqrt(n)
    return mean, se, n


def fig_interactions(recs, outdir: Path, tables: Path) -> None:
    mats = perf_matrices(recs, "test", "auc")
    if not mats:
        print("  [panel 2] no performance decomposition yet -- skipped")
        return
    stack = np.stack([m for m, _ in mats.values()])
    totals = np.array([t for _, t in mats.values()])
    mean, se, n = _mean_se(stack)
    npos = np.sum(stack > 0, axis=0)

    rows = []
    for i, a in enumerate(GROUPS):
        for j, b in enumerate(GROUPS):
            if j > i:
                continue
            rows.append({"a": a, "b": b if j != i else "", "kind": "main" if i == j else "interaction",
                         "mean_auc_x100": mean[i, j] * 100, "se_x100": se[i, j] * 100,
                         "n_models": int(n[i, j]), "n_models_positive": int(npos[i, j])})
    pd.DataFrame(rows).to_csv(tables / "xai_2_interaction_mean.csv", index=False)
    per_model = []
    for k, (M, t) in mats.items():
        per_model.append({"model": k, "a": "ALL", "b": "", "value_x100": t * 100})
        for i, a in enumerate(GROUPS):
            for j, b in enumerate(GROUPS):
                if j <= i:
                    per_model.append({"model": k, "a": a, "b": b if j != i else "",
                                      "value_x100": M[i, j] * 100})
    pd.DataFrame(per_model).to_csv(tables / "xai_2_interaction_by_model.csv", index=False)
    print(f"  [panel 2] models: {list(mats)} | mean AUC-0.5 = {np.nanmean(totals) * 100:+.2f} x100")

    def star(i, j):
        return "*" if n[i, j] >= 3 and np.isfinite(se[i, j]) and abs(mean[i, j]) > 1.96 * se[i, j] else ""

    # ---- 2a: matrix
    G = len(GROUPS)
    off = np.where(np.tril(np.ones((G, G), bool), -1), mean, np.nan)
    vlim = float(np.nanmax(np.abs(off))) if np.isfinite(off).any() else 1e-6
    fig = plt.figure(figsize=(3.4, 2.7), layout="constrained")
    ax = fig.subplots()
    cmap = DIV.copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(off * 100), cmap=cmap, vmin=-vlim * 100, vmax=vlim * 100,
                   interpolation="nearest")
    for i, g in enumerate(GROUPS):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor=_tint(C.GROUP_COLORS[g], 0.3),
                               edgecolor="none", zorder=2))
        if np.isfinite(mean[i, i]):
            ax.text(i, i, f"{mean[i, i] * 100:+.2f}{star(i, i)}", ha="center", va="center",
                    fontsize=5.6, fontweight="bold", color=INK, zorder=3)
    for i in range(G):
        for j in range(i):
            v = off[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:+.2f}{star(i, j)}", ha="center", va="center", fontsize=5.4,
                        color="white" if abs(v) > 0.62 * vlim else INK, zorder=3)
    _heatmap_axes(ax, G, G)
    ax.set_xticks(range(G))
    ax.set_xticklabels(GROUPS, rotation=30, ha="right")
    ax.set_yticks(range(G))
    ax.set_yticklabels(GROUPS)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, aspect=16)
    _style_cb(cb, r"Interaction ($\Delta$AUC $\times$100)")
    save(fig, outdir / "xai_2a_interaction_matrix.pdf")

    # ---- 2b: network (node area = group contribution, edge = interaction)
    pos = {"Domestic news": (-1.0, 0.8), "Foreign news": (1.0, 0.8),
           "Market": (-1.0, -0.8), "Macro": (1.0, -0.8)}
    main = {g: mean[i, i] for i, g in enumerate(GROUPS)}
    max_main = max([abs(v) for v in main.values() if np.isfinite(v)] + [1e-9])
    inter = {(GROUPS[i], GROUPS[j]): mean[i, j] for i in range(G) for j in range(i)}
    max_int = max([abs(v) for v in inter.values() if np.isfinite(v)] + [1e-9])
    indep_tol = 0.10 * max_main         # |I| below 10% of the largest main effect = "independent"
    radius = {g: 0.16 + 0.22 * np.sqrt(abs(v) / max_main) if np.isfinite(v) else 0.16
              for g, v in main.items()}

    fig = plt.figure(figsize=(3.3, 2.9), layout="constrained")
    ax = fig.subplots()
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.5, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    diagonal = {frozenset(("Domestic news", "Macro")), frozenset(("Foreign news", "Market"))}
    for (a, b), v in inter.items():
        if not np.isfinite(v) or a not in pos or b not in pos:
            continue
        pa, pb = np.array(pos[a]), np.array(pos[b])
        u = (pb - pa) / np.linalg.norm(pb - pa)
        sa, sb = pa + u * radius[a], pb - u * radius[b]
        if abs(v) < indep_tol:
            color, ls = "#9e9d98", (0, (1.2, 1.2))
        else:
            color, ls = (RED if v < 0 else BLUE), "-"
        ax.plot([sa[0], sb[0]], [sa[1], sb[1]], color=color, lw=0.6 + 2.6 * abs(v) / max_int,
                ls=ls, solid_capstyle="round", zorder=1)
        f = 0.3 if frozenset((a, b)) in diagonal else 0.5
        lx, ly = sa + (sb - sa) * f
        ax.text(lx, ly, f"{v * 100:+.2f}", fontsize=5.0, ha="center", va="center", color=INK_2,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=4)
    for g, (x, y) in pos.items():
        v = main.get(g, np.nan)
        neg = np.isfinite(v) and v < 0
        ax.add_patch(Circle((x, y), radius[g], facecolor=_tint(C.GROUP_COLORS[g], 0.28),
                            edgecolor=RED if neg else C.GROUP_COLORS[g], lw=0.9,
                            ls="--" if neg else "-", zorder=3))
        ax.text(x, y + 0.05, g.replace(" ", "\n"), ha="center", va="center", fontsize=5.8,
                color=INK, zorder=5, linespacing=0.95)
        if np.isfinite(v):
            ax.text(x, y - radius[g] - 0.1, f"{v * 100:+.2f}", ha="center", va="top", fontsize=5.4,
                    fontweight="bold", color=INK, zorder=5)
    handles = [Line2D([], [], color=RED, lw=1.4, label="Overlapping (I < 0)"),
               Line2D([], [], color="#9e9d98", lw=1.0, ls=(0, (1.2, 1.2)), label="Independent (|I| small)"),
               Line2D([], [], color=BLUE, lw=1.4, label="Complementary (I > 0)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=4.9, handlelength=1.8, columnspacing=0.9, frameon=False)
    save(fig, outdir / "xai_2b_interaction_network.pdf")


# =================================================================== panel 3: regimes
def regime_lag_shares(rec):
    nat = rec.get("native")
    if nat is None:
        return None
    A, players = nat["A"], [str(p) for p in nat["players"]]
    reg = nat["regime"].astype(str)
    out = {}
    for r in ("low", "high"):
        m = reg == r
        if m.sum() < 20:
            return None
        M = np.abs(A[m]).mean(axis=0)
        S = M / M.sum()
        full = np.full((len(GROUPS), A.shape[2]), np.nan)
        for i, g in enumerate(players):
            if g in GROUPS:
                full[GROUPS.index(g)] = S[i]
        out[r] = full
    return out


def _regime_triplet(fig, axes, low, high, buckets, vmax, vlim, letters=None, xlabel=True,
                    colorbars=True):
    group_heatmap(axes[0], low, GROUPS, buckets, vmax)
    group_heatmap(axes[1], high, GROUPS, buckets, vmax)
    im = diverging_heatmap(axes[2], high - low, GROUPS, buckets, vlim)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    if xlabel:
        for ax in axes:
            ax.set_xlabel("Days before the forecast day", fontsize=6)
    if letters:
        for ax, l in zip(axes, letters):
            panel_letter(ax, l)
    if colorbars:
        tint_colorbar(fig, list(axes[:2]), vmax, "Share of |SHAP| (%)")
        cb = fig.colorbar(im, ax=axes[2], fraction=0.05, pad=0.03, aspect=16)
        _style_cb(cb, "High $-$ low (pp)")


def fig_regimes(recs, outdir: Path, tables: Path) -> None:
    # ---- 3a/3b: where in the window each group matters, by regime (native SHAP)
    lag = {r["key"]: s for r in recs if (s := regime_lag_shares(r)) is not None}
    if lag:
        buckets = [lab for lab, _, _ in C.LAG_BUCKETS]
        low = np.nanmean(np.stack([v["low"] for v in lag.values()]), axis=0)
        high = np.nanmean(np.stack([v["high"] for v in lag.values()]), axis=0)
        vmax = float(np.nanmax([low, high]))
        vlim = float(np.nanmax(np.abs(high - low))) or 1e-6
        fig = plt.figure(figsize=(7.0, 1.8), layout="constrained")
        axes = fig.subplots(1, 3)
        _regime_triplet(fig, axes, low, high, buckets, vmax, vlim, letters="abc")
        save(fig, outdir / "xai_3a_regime_lag_heatmaps.pdf")
        rows = []
        for k, v in lag.items():
            for r in ("low", "high"):
                for i, g in enumerate(GROUPS):
                    for b, lab in enumerate(buckets):
                        rows.append({"model": k, "regime": r, "group": g, "lag_bucket": lab,
                                     "share": v[r][i, b]})
        pd.DataFrame(rows).to_csv(tables / "xai_3a_regime_lag_share_by_model.csv", index=False)

        fams = [f for f in C.FAMILIES if any(r["family"] == f and r["key"] in lag for r in recs)]
        fam = {}
        for f in fams:
            ks = [r["key"] for r in recs if r["family"] == f and r["key"] in lag]
            fam[f] = (np.nanmean(np.stack([lag[k]["low"] for k in ks]), axis=0),
                      np.nanmean(np.stack([lag[k]["high"] for k in ks]), axis=0))
        vmax_f = float(np.nanmax([np.nanmax(np.stack(v)) for v in fam.values()]))
        vlim_f = float(np.nanmax([np.nanmax(np.abs(h - l)) for l, h in fam.values()])) or 1e-6
        fig = plt.figure(figsize=(7.0, 1.35 * len(fams) + 0.35), layout="constrained")
        axes = fig.subplots(len(fams), 3, squeeze=False)
        for ri, f in enumerate(fams):
            l, h = fam[f]
            _regime_triplet(fig, axes[ri], l, h, buckets, vmax_f, vlim_f,
                            letters="abc" if ri == 0 else None,
                            xlabel=(ri == len(fams) - 1), colorbars=False)
            axes[ri, 0].set_ylabel(f, fontsize=6.3, labelpad=4)
        tint_colorbar(fig, axes[:, :2].ravel().tolist(), vmax_f, "Share of |SHAP| (%)")
        cmap = DIV.copy()
        sm = ScalarMappable(norm=Normalize(-vlim_f * 100, vlim_f * 100), cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[:, 2].tolist(), fraction=0.05, pad=0.03, aspect=18)
        _style_cb(cb, "High $-$ low (pp)")
        save(fig, outdir / "xai_3b_regime_lag_by_family.pdf")
    else:
        print("  [panel 3a/3b] no native SHAP yet -- skipped")

    # ---- 3c: share of |group SHAP| by regime (exact group SHAP, all days)
    share = {}
    for r in recs:
        loc = r.get("local")
        if loc is None:
            continue
        cols = _phi_cols(loc)
        d = {}
        for reg in ("low", "high"):
            a = loc.loc[loc["regime"] == reg, list(cols.values())].abs().mean()
            a.index = list(cols)
            d[reg] = (a / a.sum()).reindex(GROUPS)
        share[r["key"]] = d
    if share:
        low = np.stack([v["low"].to_numpy() for v in share.values()])
        high = np.stack([v["high"].to_numpy() for v in share.values()])
        lm, ls_, _ = _mean_se(low)
        hm, hs, _ = _mean_se(high)
        n_up = np.sum(high > low, axis=0)
        pd.DataFrame({"group": GROUPS, "low_mean": lm, "low_se": ls_, "high_mean": hm, "high_se": hs,
                      "diff_mean": hm - lm, "n_models": len(share), "n_models_high_gt_low": n_up}) \
            .to_csv(tables / "xai_3c_regime_group_share.csv", index=False)
        _regime_bars(GROUPS, lm, ls_, hm, hs, "Share of |SHAP| (%)",
                     outdir / "xai_3c_regime_group_share.pdf", colors=[C.GROUP_COLORS[g] for g in GROUPS])

    # ---- 3d: out-of-sample AUC contribution by regime (performance game, H2)
    lo_m = perf_matrices(recs, "test_low", "auc")
    hi_m = perf_matrices(recs, "test_high", "auc")
    keys = [k for k in lo_m if k in hi_m]
    if keys:
        lo = np.stack([np.append(np.diag(lo_m[k][0]), lo_m[k][1]) for k in keys])
        hi = np.stack([np.append(np.diag(hi_m[k][0]), hi_m[k][1]) for k in keys])
        lm, ls_, _ = _mean_se(lo)
        hm, hs, _ = _mean_se(hi)
        labels = GROUPS + ["Total"]
        pd.DataFrame({"group": labels, "low_mean_auc_x100": lm * 100, "low_se_x100": ls_ * 100,
                      "high_mean_auc_x100": hm * 100, "high_se_x100": hs * 100,
                      "n_models": len(keys)}).to_csv(tables / "xai_3d_regime_performance.csv", index=False)
        _regime_bars(labels, lm, ls_, hm, hs, r"Contribution to AUC ($\times$100)",
                     outdir / "xai_3d_regime_performance.pdf",
                     colors=[C.GROUP_COLORS[g] for g in GROUPS] + [NEUTRAL], zero_line=True)
    else:
        print("  [panel 3d] no regime performance decomposition (too few test days per regime?)")


def _regime_bars(labels, lm, ls_, hm, hs, xlabel, path, colors, zero_line=False) -> None:
    y = np.arange(len(labels))
    h = 0.36
    fig = plt.figure(figsize=(3.3, 0.36 * len(labels) + 0.7), layout="constrained")
    ax = fig.subplots()
    ek = dict(elinewidth=0.5, capsize=1.2, capthick=0.5, ecolor=INK_2)
    ax.barh(y - h / 2, lm * 100, h, color=[_tint(c, 0.35) for c in colors],
            xerr=np.nan_to_num(1.96 * ls_ * 100), error_kw=ek, linewidth=0)
    ax.barh(y + h / 2, hm * 100, h, color=[_tint(c, 1.0) for c in colors],
            xerr=np.nan_to_num(1.96 * hs * 100), error_kw=ek, linewidth=0)
    if zero_line:
        ax.axvline(0, color=INK_2, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    _cat_axis(ax, "y")
    ax.set_xlabel(xlabel, fontsize=6.3)
    ax.tick_params(labelsize=5.8)
    ax.legend(handles=[Patch(facecolor=_tint(NEUTRAL, 0.35), label="Low volatility"),
                       Patch(facecolor=_tint(NEUTRAL, 1.0), label="High volatility")],
              fontsize=5.4, loc="lower right", frameon=False)
    save(fig, path)


# =================================================================== supplementary
def fig_features(recs, outdir: Path, tables: Path, top: int = 20) -> None:
    per = {}
    for r in recs:
        nat = r.get("native")
        if nat is None:
            continue
        imp = np.abs(nat["feat_phi"]).mean(axis=0)
        names = [C.canonical_feature(str(n)) for n in nat["feat_names"]]
        s = pd.Series(imp, index=names).groupby(level=0).sum()
        per[r["key"]] = s / s.sum()
    if not per:
        return
    df = pd.DataFrame(per)
    df.to_csv(tables / "xai_s2_feature_share_by_model.csv")
    mean = df.mean(axis=1, skipna=True).sort_values(ascending=False)
    top_s = mean.head(top)[::-1]
    fig = plt.figure(figsize=(3.3, 0.13 * len(top_s) + 0.6), layout="constrained")
    ax = fig.subplots()
    colors = [C.GROUP_COLORS.get(_safe_group(n), NEUTRAL) for n in top_s.index]
    ax.barh(range(len(top_s)), top_s.to_numpy() * 100, color=colors, height=0.7, linewidth=0)
    ax.set_yticks(range(len(top_s)))
    ax.set_yticklabels(top_s.index, fontsize=5.2)
    _cat_axis(ax, "y")
    ax.set_xlabel("Mean share of |SHAP| (%)", fontsize=6.3)
    ax.tick_params(axis="x", labelsize=5.6)
    ax.legend(handles=[Patch(facecolor=C.GROUP_COLORS[g], label=g) for g in GROUPS],
              fontsize=5.0, loc="lower right", frameon=False)
    save(fig, outdir / "xai_s2_feature_importance.pdf")


def _safe_group(name: str) -> str:
    try:
        return C.group_of(name)
    except KeyError:
        return ""


def fig_attention(recs, outdir: Path, tables: Path) -> None:
    items = [r for r in recs if "att" in r and "local" in r]
    if not items:
        return
    fig = plt.figure(figsize=(3.3 * len(items), 1.7), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(items)))
    rows = []
    y = np.arange(len(GROUPS))
    h = 0.36
    for ax, r, letter in zip(axes, items, "abcdef"):
        att, loc = r["att"], r["local"]
        a = np.array([att[f"att_{slug(g)}"].mean() if f"att_{slug(g)}" in att else np.nan for g in GROUPS])
        u = np.array([att[f"uniform_{slug(g)}"].iloc[0] if f"uniform_{slug(g)}" in att else np.nan
                      for g in GROUPS])
        cols = _phi_cols(loc)
        s = loc[list(cols.values())].abs().mean()
        s.index = list(cols)
        s = (s / s.sum()).reindex(GROUPS).to_numpy()
        colors = [C.GROUP_COLORS[g] for g in GROUPS]
        ax.barh(y - h / 2, a * 100, h, color=[_tint(c, 0.35) for c in colors], linewidth=0)
        ax.barh(y + h / 2, s * 100, h, color=[_tint(c, 1.0) for c in colors], linewidth=0)
        ax.scatter(u * 100, y - h / 2, marker="|", s=40, color=INK, linewidths=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(GROUPS if ax is axes[0] else [])
        ax.invert_yaxis()
        _cat_axis(ax, "y")
        ax.set_xlabel("Share (%)", fontsize=6.3)
        ax.tick_params(labelsize=5.8)
        ax.set_title(r["label"], fontsize=6.3, pad=2, loc="left")
        panel_letter(ax, letter)
        for g, ai, ui, si in zip(GROUPS, a, u, s):
            rows.append({"model": r["key"], "group": g, "attention_share": ai,
                         "uniform_share": ui, "shap_share": si})
    axes[-1].legend(handles=[Patch(facecolor=_tint(NEUTRAL, 0.35), label="Attention"),
                             Patch(facecolor=_tint(NEUTRAL, 1.0), label="SHAP"),
                             Line2D([], [], color=INK, marker="|", ls="none", markersize=6,
                                    label="Uniform")],
                    fontsize=5.2, loc="lower right", frameon=False)
    pd.DataFrame(rows).to_csv(tables / "xai_s3_attention_vs_shap.csv", index=False)
    save(fig, outdir / "xai_s3_attention_vs_shap.pdf")


def qc_table(recs, tables: Path) -> None:
    rows = []
    for r in recs:
        m = r["meta"]
        g = m.get("group", {})
        nat = m.get("native", {})
        agree = nat.get("spearman_abs_exact_vs_native", {}) or {}
        rows.append({
            "model": r["key"], "label": r["label"], "family": r["family"], "n_days": m.get("n_days"),
            "verify_max_abs_diff": m.get("verify", {}).get("max_abs_diff"),
            "test_auc_rebuilt": m.get("verify", {}).get("auc_test_rebuilt"),
            "group_efficiency_max_error": g.get("efficiency_max_error"),
            "native_method": nat.get("method"),
            "native_completeness_corr": nat.get("corr_sum_phi_vs_fx_minus_E"),
            "exact_vs_native_spearman_mean": float(np.mean(list(agree.values()))) if agree else np.nan,
            "attention_entropy_over_max": m.get("attention", {}).get("mean_entropy_over_max"),
            "regime_low_max": m.get("regime", {}).get("low_max"),
            "regime_high_min": m.get("regime", {}).get("high_min"),
            "test_regime_low_max": m.get("regime_test", {}).get("low_max"),
            "test_regime_high_min": m.get("regime_test", {}).get("high_min"),
            "test_days_low": m.get("regime_test", {}).get("counts_test", {}).get("low"),
            "test_days_high": m.get("regime_test", {}).get("counts_test", {}).get("high"),
            "runtime_s": m.get("runtime_s"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(tables / "xai_qc_table.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[["model", "verify_max_abs_diff", "test_auc_rebuilt", "group_efficiency_max_error",
                  "native_completeness_corr", "exact_vs_native_spearman_mean"]].to_string(index=False))


# =================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=C.EXPERIMENT)
    args = ap.parse_args()

    outdir = C.FIG_ROOT / args.exp
    tables = outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    print(f"reading {C.OUT_ROOT} ({args.exp})")
    recs = load_all(args.exp)
    if not recs:
        raise SystemExit("no model has XAI outputs yet -- run run_xai.py first")
    print(f"figures -> {outdir}")
    qc_table(recs, tables)
    fig_temporal(recs, outdir, tables)
    fig_interactions(recs, outdir, tables)
    fig_regimes(recs, outdir, tables)
    fig_features(recs, outdir, tables)
    fig_attention(recs, outdir, tables)


if __name__ == "__main__":
    main()
