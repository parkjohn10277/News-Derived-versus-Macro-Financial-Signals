"""
Variable importance for the tree DIRECTION CLASSIFIERS, following Chun et al. (2024).

Their measure is NOT sklearn's feature_importances_, not permutation importance
and not SHAP.  It is an out-of-sample ablation (their "Analysis of the key
predictors" section, after Gu et al. 2020).  The regression folder measures the
drop in R2_oos; the classifier measures the drop in the skill score that
replaces it:

    I_j = Brier_Skill(full model)  -  Brier_Skill(model with predictor j zeroed)

Zero is the training mean here, because predictors are Yeo-Johnson standardised.
A large drop means the predictor matters.  Appendix Figs. 11-13 normalise the
values to sum to one within each model; `normalized` reproduces that.

This script produces, per experiment, averaged over seeds:

  1. individual ablation      every design feature (<col>_lag<k>)
  2. column ablation          all lags of one input column removed together
  3. GROUP ablation           whole information blocks removed together --
                              KRX / LDA / LLM / AUX, LDA split EN vs KO, and
                              LLM split foreign (GDELT, NYT) vs domestic (NAVER)
                              vs MCST.
  4. marginal association     paper Figs. 4/6/8 -- sweep one predictor across
                              its range with the others held at their median,
                              reading out P(net buy)
  5. SHAP                     secondary interpretability (tree explainer)

    python report_importance.py                  # every experiment with results
    python report_importance.py --exp E7 --seeds 42 --no-shap

Figures + csv tables -> figures/<model>/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run import load_input, prepare                                        # noqa: E402
from src.features import EXPERIMENTS, experiment_features, feature_groups  # noqa: E402
from src.metrics import brier_skill_score                                  # noqa: E402
from src.tabular import build_design, clip_aux, feature_group              # noqa: E402

FOREIGN_NEWS = ["gdelt_sentiment", "nyt_sentiment"]
DOMESTIC_NEWS = ["naver_sentiment"]
MCST_NEWS = ["mcst_sentiment"]

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


# ---------------------------------------------------------------- core ablation
def _skill(model, X, test_idx, y_dir, base_rate) -> float:
    """Out-of-sample Brier Skill vs the training base rate."""
    prob = model.predict_proba(X[test_idx])[:, 1]
    return brier_skill_score(y_dir[test_idx], prob, base_rate)


def ablation(model, X, test_idx, y_dir, base_rate, blocks: dict[str, list[int]]) -> pd.DataFrame:
    """blocks maps a label to the column indices of X zeroed together."""
    base = _skill(model, X, test_idx, y_dir, base_rate)
    rows = []
    for label, cols in blocks.items():
        Xa = X.copy()
        Xa[:, cols] = 0.0                                  # 0 == training mean after standardising
        ab = _skill(model, Xa, test_idx, y_dir, base_rate)
        rows.append({"item": label, "baseline_skill": base, "ablated_skill": ab,
                     "importance": base - ab})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Appendix normalisation: clip negatives at zero, scale to sum one."""
    out = df.copy()
    pos = out["importance"].clip(lower=0)
    total = pos.sum()
    out["normalized"] = pos / total if total > 0 else 0.0
    return out


def marginal_association(model, X, test_idx, names, targets, n_points: int = 41) -> pd.DataFrame:
    """Paper Figs. 4/6/8: sweep one predictor from its 1st to 99th test-set
    percentile (rescaled to [-1, 1]) with every other predictor at its median.
    The read-out is P(net buy) rather than a KRW amount."""
    Xt = X[test_idx]
    med = np.median(Xt, axis=0)
    rows = []
    for f in targets:
        j = names.index(f)
        lo, hi = np.percentile(Xt[:, j], [1, 99])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            continue
        grid = np.linspace(lo, hi, n_points)
        Xg = np.tile(med, (n_points, 1))
        Xg[:, j] = grid
        prob = model.predict_proba(Xg)[:, 1]
        rows += [{"feature": f, "x_scaled": float(g), "x_norm": float(2 * (g - lo) / (hi - lo) - 1),
                  "prob_net_buy": float(p)} for g, p in zip(grid, prob)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-seed run
def load_run(model_dir: Path, exp: str, seed: int, raw, groups, cfg):
    od = model_dir / exp / f"seed_{seed}"
    if not (od / "best_model.joblib").exists():
        return None
    feats = experiment_features(groups, exp)
    df = prepare(raw, groups, feats)
    m = json.loads((od / "metrics.json").read_text(encoding="utf-8"))
    scale_c = float(m.get("target_scale_c", 1.0))
    X, y_raw, dates = build_design(df, feats, groups, cfg["features"]["lags"], scale_c)
    names = list(X.columns)
    with open(od / "x_scaler.pkl", "rb") as f:
        xs = pickle.load(f)
    Xs, _ = clip_aux(xs.transform(X), names, groups, cfg["preprocess"]["aux_z_clip"])
    off_end = int((dates <= pd.Timestamp(cfg["split"]["train_end"])).sum())
    test_start = int((dates < pd.Timestamp(cfg["split"]["test_start"])).sum())
    y_dir = (y_raw >= 0).astype(int)
    return {"model": joblib.load(od / "best_model.joblib"), "X": Xs, "names": names,
            "y_raw": y_raw, "y_dir": y_dir, "test_idx": np.arange(test_start, len(dates)),
            "base_rate": float(y_dir[:off_end].mean()), "scale_c": scale_c,
            "feats": feats, "dates": dates}


def blocks_for(names: list[str], feats: list[str], groups: dict):
    """Design-column indices for the individual / column / group ablations."""
    ind = {f: [i] for i, f in enumerate(names)}
    col = {c: [i for i, f in enumerate(names) if f.rsplit("_lag", 1)[0] == c] for c in feats}
    grp: dict[str, list[int]] = {}
    for g in ("KRX", "LDA", "LLM", "AUX"):
        idx = [i for i, f in enumerate(names) if feature_group(f, groups) == g]
        if idx:
            grp[g] = idx
    for label, cols in (("LDA_EN", [c for c in feats if c.startswith("lda_en")]),
                        ("LDA_KO", [c for c in feats if c.startswith("lda_ko")]),
                        ("NEWS_foreign_EN", [c for c in feats if c in FOREIGN_NEWS]),
                        ("NEWS_domestic_KO", [c for c in feats if c in DOMESTIC_NEWS]),
                        ("NEWS_MCST", [c for c in feats if c in MCST_NEWS])):
        idx = [i for i, f in enumerate(names) if f.rsplit("_lag", 1)[0] in cols]
        if idx:
            grp[label] = idx
    return ind, col, grp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--top", type=int, default=20, help="features shown in the figures")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]["name"]
    model_dir = ROOT / cfg["results"] / model_name
    out = ROOT / "figures" / model_name
    out.mkdir(parents=True, exist_ok=True)

    raw = load_input(cfg)
    groups = feature_groups(raw)
    exps = [e for e in (args.exp or list(EXPERIMENTS)) if (model_dir / e).exists()]
    if not exps:
        print(f"no results under {model_dir}"); return

    group_all, col_all = [], []
    for exp in exps:
        seeds = args.seeds or sorted(int(p.name.split("_")[1]) for p in (model_dir / exp).glob("seed_*"))
        runs = [r for s in seeds if (r := load_run(model_dir, exp, s, raw, groups, cfg)) is not None]
        if not runs:
            continue
        ind_f, col_f, grp_f = [], [], []
        for r in runs:
            ind_b, col_b, grp_b = blocks_for(r["names"], r["feats"], groups)
            kw = dict(test_idx=r["test_idx"], y_dir=r["y_dir"], base_rate=r["base_rate"])
            ind_f.append(ablation(r["model"], r["X"], blocks=ind_b, **kw).set_index("item"))
            col_f.append(ablation(r["model"], r["X"], blocks=col_b, **kw).set_index("item"))
            grp_f.append(ablation(r["model"], r["X"], blocks=grp_b, **kw).set_index("item"))
        ind = normalized(sum(ind_f).div(len(ind_f)).reset_index())
        colr = normalized(sum(col_f).div(len(col_f)).reset_index())
        grp = normalized(sum(grp_f).div(len(grp_f)).reset_index())
        for d in (ind, colr, grp):
            d.insert(0, "experiment", exp)
        ind.to_csv(out / f"ablation_feature_{exp}.csv", index=False)
        colr.to_csv(out / f"ablation_column_{exp}.csv", index=False)
        grp.to_csv(out / f"ablation_group_{exp}.csv", index=False)
        group_all.append(grp); col_all.append(colr)
        print(f"{exp}: seeds {seeds}  baseline Brier_Skill={ind.baseline_skill.iloc[0]:+.4f}  "
              f"top column={colr.item.iloc[0]} ({colr.importance.iloc[0]:+.4f})")

        # ---- per-experiment figures (top columns + groups)
        fig, axes = plt.subplots(1, 2, figsize=(13, 0.30 * min(args.top, len(colr)) + 2.4))
        t = colr.head(args.top).iloc[::-1]
        axes[0].barh(t["item"], t["importance"], color="#2b6cb0")
        axes[0].set_title(f"{exp} - column ablation (top {args.top})", fontsize=9)
        axes[0].set_xlabel("drop in out-of-sample Brier Skill")
        g = grp.iloc[::-1]
        axes[1].barh(g["item"], g["importance"],
                     color=["#d1495b" if "domestic" in i else "#1f5fbf" if "foreign" in i else "#6c8f7d"
                            for i in g["item"]])
        axes[1].set_title(f"{exp} - information-block ablation", fontsize=9)
        axes[1].set_xlabel("drop in out-of-sample Brier Skill")
        fig.suptitle(f"{model_name} classifier | {exp}: {'+'.join(EXPERIMENTS[exp])}  "
                     f"(Chun et al. 2024 ablation, mean of {len(runs)} seeds)", y=1.0)
        fig.tight_layout(); fig.savefig(out / f"ablation_{exp}.png", bbox_inches="tight"); plt.close(fig)

        # ---- marginal association for the strongest columns (paper Figs. 4/6/8)
        r = runs[0]
        top_feats = [f for f in ind.sort_values("importance", ascending=False)["item"].head(8)]
        ma = marginal_association(r["model"], r["X"], r["test_idx"], r["names"], top_feats)
        if not ma.empty:
            ma.insert(0, "experiment", exp)
            ma.to_csv(out / f"marginal_association_{exp}.csv", index=False)
            n = ma.feature.nunique()
            fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(2.6 * ((n + 1) // 2), 5.2), squeeze=False)
            for ax, (f, sub) in zip(axes.ravel(), ma.groupby("feature", sort=False)):
                ax.plot(sub.x_norm, sub.prob_net_buy, color="#2b6cb0")
                ax.set_title(f, fontsize=7); ax.axhline(0.5, color="k", lw=0.5, ls="--")
                ax.set_ylim(0, 1)
                ax.set_xlabel("predictor (scaled to [-1,1])", fontsize=7)
            axes[0, 0].set_ylabel("P(net buy)", fontsize=8)
            fig.suptitle(f"{model_name} classifier | {exp} - marginal association "
                         f"(others at median)", y=1.0)
            fig.tight_layout(); fig.savefig(out / f"marginal_association_{exp}.png", bbox_inches="tight")
            plt.close(fig)

        # ---- SHAP (secondary)
        if not args.no_shap:
            try:
                import shap
                r = runs[0]
                ex = shap.TreeExplainer(r["model"])
                sv = ex.shap_values(r["X"][r["test_idx"]])
                if isinstance(sv, list):                 # RF returns one array per class
                    sv = sv[1]
                if getattr(sv, "ndim", 2) == 3:          # (n, features, classes)
                    sv = sv[:, :, 1]
                sh = pd.DataFrame({"feature": r["names"], "mean_abs_shap": np.abs(sv).mean(0)})
                sh.insert(0, "experiment", exp)
                sh.sort_values("mean_abs_shap", ascending=False).to_csv(
                    out / f"shap_{exp}.csv", index=False)
                fig = plt.figure()
                shap.summary_plot(sv, r["X"][r["test_idx"]], feature_names=r["names"],
                                  plot_type="bar", max_display=args.top, show=False)
                plt.title(f"{model_name} classifier | {exp} - SHAP (seed {seeds[0]})", fontsize=9)
                plt.tight_layout(); plt.savefig(out / f"shap_{exp}.png", bbox_inches="tight"); plt.close(fig)
            except Exception as e:                                    # noqa: BLE001
                print(f"  [warn] SHAP skipped for {exp}: {e}")

    if not group_all:
        return
    groups_df = pd.concat(group_all, ignore_index=True)
    groups_df.to_csv(out / "ablation_group_all.csv", index=False)
    pd.concat(col_all, ignore_index=True).to_csv(out / "ablation_column_all.csv", index=False)

    piv = groups_df.pivot(index="item", columns="experiment", values="importance")
    order = ["KRX", "AUX", "LDA", "LDA_EN", "LDA_KO", "LLM",
             "NEWS_foreign_EN", "NEWS_domestic_KO", "NEWS_MCST"]
    piv = piv.reindex([i for i in order if i in piv.index])
    fig, ax = plt.subplots(figsize=(1.25 * piv.shape[1] + 4, 0.42 * len(piv) + 2.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv)):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7,
                        color="w" if v < np.nanmax(piv.to_numpy()) * 0.6 else "k")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="drop in out-of-sample Brier Skill")
    ax.set_title(f"{model_name} classifier - information-block ablation across experiments\n"
                 f"(Chun et al. 2024 method; higher = removing the block hurts more)")
    fig.tight_layout(); fig.savefig(out / "ablation_group_heatmap.png", bbox_inches="tight"); plt.close(fig)

    print("\nInformation-block ablation (drop in out-of-sample Brier Skill):")
    print(piv.round(4).to_string())
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
