"""
GDP-CORRECTED copy of ../../Baselines/run_baselines.py: same baselines and protocol,
run on ../data/input_gdp_corrected.csv and compared with the rerun models in
../models (via ../Tests/run_tests.py).

Simple baselines for the next-day direction of foreign net buying, under the
same split, inputs and test days as the ten models.

    python run_baselines.py              # run every baseline on both inputs, then compare
    python run_baselines.py --compare    # only rebuild the comparison tables

Baselines
  majority     always the training majority class (net sell); P = training base rate
  persistence  tomorrow's direction = today's direction (sign of the flow on day t);
               AUC uses the day-t flow itself as the score
  logit_E0..E7 L2 logistic regression on each feature set, same protocol as the
               tree models (lags 0,1,2,5,10,20,29; Yeo-Johnson; AUX z-clip 3;
               C chosen on validation log loss; refit on train+validation)
  logit_min    logistic regression on the day-t flow and the day-t SOX return only

Inputs
  input_aligned   data/input_aligned.csv  = the file every Final_run model used
  input_auxfixed  data/input_auxfixed.csv = macro variables shifted by publication lags
                  (only comparable to the models once they are rerun on this file)

Outputs  results/<input>/<baseline>/<E>/seed_<s>/   metrics.json, predictions.csv, ...
         results/<input>/summary.csv                 one row per baseline x experiment
         results/input_aligned/compare_*.csv          tests against the Final_run models
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.features import EXPERIMENTS, TARGET_COL, experiment_features, feature_groups  # noqa: E402
from src.metrics import classification_metrics                                          # noqa: E402
from src.models import get_spec                                                         # noqa: E402
from src.protocol import run_single_split                                               # noqa: E402
from src.tabular import build_design                                                    # noqa: E402

# identical to Final_run/models/RandomForest/config.yaml
START = "2015-02-28"
SPLIT = {"val_start": "2024-03-13", "train_end": "2025-05-07", "test_start": "2025-05-08"}
LAGS = [0, 1, 2, 5, 10, 20, 29]
AUX_Z_CLIP = 3.0
THRESHOLD = 0.5
SEEDS = [42, 43, 44, 45, 46]
INPUTS = {"input_gdp_corrected": ROOT.parent / "data" / "input_gdp_corrected.csv"}
RES = ROOT / "results"

KEEP = ["AUC", "Brier_Skill", "LogLoss_Skill", "MCC", "Balanced_Accuracy",
        "Directional_Accuracy", "F1_Score", "Precision", "Recall", "Brier",
        "base_rate_train", "base_rate_test", "pred_positive_rate", "n_test"]


# ------------------------------------------------------------------ data (as in Final_run run.py)
def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig").set_index("date").sort_index()
    return df[df.index >= pd.Timestamp(START)]


def prepare(df, groups, feature_cols):
    """Missing-value policy of the tree models: LDA/LLM -> 0, AUX -> ffill, drop NaN rows."""
    d = df[[TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]].copy()
    for c in groups["LDA"] + groups["LLM"]:
        if c in d:
            d[c] = d[c].fillna(0.0)
    aux = [c for c in groups["AUX"] if c in d]
    if aux:
        d[aux] = d[aux].ffill()
    return d.dropna()


def save(od: Path, metrics: dict, preds: pd.DataFrame, extra: dict | None = None):
    od.mkdir(parents=True, exist_ok=True)
    (od / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    preds.to_csv(od / "predictions.csv", index=False, encoding="utf-8-sig")
    for name, obj in (extra or {}).items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(od / name, index=False, encoding="utf-8-sig")
        elif name.endswith(".joblib"):
            joblib.dump(obj, od / name)
        else:
            with open(od / name, "wb") as f:
                pickle.dump(obj, f)


# ------------------------------------------------------------------ baselines
def run_logit(df, groups, feats, seed, lags):
    spec = get_spec("logistic")
    d = prepare(df, groups, feats)
    res = run_single_split(d, feats, groups, SPLIT, spec, seed, lags=lags, aux_z_clip=AUX_Z_CLIP,
                           threshold=THRESHOLD, cv_scheme="holdout", n_iter=len(spec.GRID["C"]),
                           n_jobs=1)
    coefs = pd.DataFrame({"design_feature": res["feature_names"],
                          "coef_standardised": res["model"].coef_.ravel()})
    coefs = coefs.reindex(coefs.coef_standardised.abs().sort_values(ascending=False).index)
    extra = {"coefficients.csv": coefs, "cv_results.csv": res["cv_results"],
             "best_model.joblib": res["model"], "x_scaler.pkl": res["x_scaler"]}
    return res["metrics"], res["predictions"], extra


def naive_baselines(df, groups):
    """Majority class and persistence on exactly the test days of the models."""
    d = prepare(df, groups, ["foreign_net_buy"])
    _, y_raw, dates = build_design(d, ["foreign_net_buy"], groups, LAGS)   # same rows as E0
    prev = d["foreign_net_buy"].shift(1).reindex(dates).to_numpy()          # flow on day t
    y = (y_raw >= 0).astype(int)
    train = dates <= pd.Timestamp(SPLIT["train_end"])
    test = dates >= pd.Timestamp(SPLIT["test_start"])
    base = float(y[train].mean())
    out = {}

    # majority: constant P = training base rate (< 0.5, so always "net sell")
    p = np.full(test.sum(), base)
    m = classification_metrics(y[test], p, base, THRESHOLD)
    out["majority"] = (m, pd.DataFrame({"date": dates[test], "actual_flow_krw": y_raw[test],
                                        "actual_direction": y[test], "prob_net_buy": p,
                                        "predicted_direction": (p >= THRESHOLD).astype(int)}))

    # persistence: hard forecast = today's sign; AUC from the day-t flow as a ranking score
    hard = (prev[test] >= 0).astype(float)
    m = classification_metrics(y[test], hard, base, THRESHOLD)
    from sklearn.metrics import roc_auc_score
    m["AUC_hard"] = m["AUC"]
    m["AUC"] = float(roc_auc_score(y[test], prev[test]))
    m["note"] = ("AUC ranks days by the day-t flow; AUC_hard uses the 0/1 forecast. "
                 "Brier/LogLoss use the 0/1 forecast and are not comparable to probabilistic models.")
    out["persistence"] = (m, pd.DataFrame({"date": dates[test], "actual_flow_krw": y_raw[test],
                                           "actual_direction": y[test], "flow_day_t": prev[test],
                                           "prob_net_buy": hard,
                                           "predicted_direction": hard.astype(int)}))
    return out


def run_all(seeds):
    for tag, path in INPUTS.items():
        raw = load_input(path)
        groups = feature_groups(raw)
        print(f"\n######## {tag}: {path.name}  rows={len(raw)}  "
              f"AUX={len(groups['AUX'])} LDA={len(groups['LDA'])} LLM={len(groups['LLM'])}")
        rows = []

        for name, (m, p) in naive_baselines(raw, groups).items():
            save(RES / tag / name / "all" / "seed_0", m, p)
            rows.append({"baseline": name, "experiment": "-", "seed": 0, "n_features": 1,
                         **{k: m.get(k) for k in KEEP}})
            print(f"  {name:12s} AUC={m['AUC']:.4f}  DA={m['Directional_Accuracy']:.4f}  "
                  f"BalAcc={m['Balanced_Accuracy']:.4f}  MCC={m['MCC']:+.3f}")

        jobs = [("logit", e, experiment_features(groups, e), LAGS) for e in EXPERIMENTS]
        jobs.append(("logit_min", "flow+SOX", ["foreign_net_buy", "sox_logret"], [0]))
        for name, exp, feats, lags in jobs:
            for seed in seeds:
                m, p, extra = run_logit(raw, groups, feats, seed, lags)
                m.update({"baseline": name, "experiment": exp, "seed": seed, "lags": lags,
                          "features": feats, "input": path.name})
                save(RES / tag / name / exp / f"seed_{seed}", m, p, extra)
                rows.append({"baseline": name, "experiment": exp, "seed": seed,
                             "n_features": len(feats), "C": m["best_params"]["C"],
                             **{k: m.get(k) for k in KEEP}})
                print(f"  {name:9s} {exp:8s} seed {seed}  C={m['best_params']['C']:<7g} "
                      f"AUC={m['AUC']:.4f}  BrierSkill={m['Brier_Skill']:+.4f}  "
                      f"DA={m['Directional_Accuracy']:.4f}  MCC={m['MCC']:+.3f}")
        S = pd.DataFrame(rows)
        S.to_csv(RES / tag / "summary_all_seeds.csv", index=False, float_format="%.6g")
        g = S.groupby(["baseline", "experiment"], sort=False)
        mean = g[[c for c in KEEP + ["C"] if c in S]].mean()
        mean.insert(0, "n_seeds", g.size())
        mean.insert(1, "auc_sd_over_seeds", g["AUC"].std().fillna(0.0))
        mean.reset_index().to_csv(RES / tag / "summary.csv", index=False, float_format="%.6g")


# ------------------------------------------------------------------ comparison with the models
def compare(seeds):
    sys.path.insert(0, str(ROOT.parent / "Tests"))
    import run_tests as T                                    # DeLong, DM, block bootstrap

    tag = "input_gdp_corrected"
    y, P = None, {}
    runs = {s: T.load_predictions(s) for s in seeds}
    y = runs[seeds[0]][0]
    P = {m: {e: np.mean([runs[s][1][m][e] for s in seeds], axis=0) for e in T.EXPS}
         for m in T.MODELS}                                  # seed ensemble, as in run_tests.py

    def logit_p(exp):
        d = pd.read_csv(RES / tag / "logit" / exp / f"seed_{seeds[0]}" / "predictions.csv",
                        encoding="utf-8-sig", parse_dates=["date"]).sort_values("date")
        if not (d["actual_direction"].values == y).all():
            raise SystemExit("baseline test days differ from the models")
        return np.clip(d["prob_net_buy"].values, T.EPS, 1 - T.EPS)

    L = {e: logit_p(e) for e in T.EXPS}
    rng = np.random.default_rng(2026)
    idx = T.block_indices(len(y), rng)

    # 1. every model vs the logistic regression on the same feature set
    rows = []
    for e in T.EXPS:
        pl = L[e]
        rows.append(dict(exp=e, model="Logistic regression", auc=T.auc(y, pl),
                         brier=T.brier_loss(y, pl).mean()))
        for m in T.MODELS:
            pm = P[m][e]
            d, z, p_dl = T.delong(y, pm, pl)
            d_b, dm_b, p_b = T.diebold_mariano(T.brier_loss(y, pm), T.brier_loss(y, pl))
            rows.append(dict(exp=e, model=m, auc=T.auc(y, pm), brier=T.brier_loss(y, pm).mean(),
                             d_auc_vs_logit=d, delong_z=z, delong_p=p_dl,
                             d_brier_vs_logit=d_b, dm_brier=dm_b, dm_brier_p=p_b))
    C = pd.DataFrame(rows)
    C["delong_p_holm"] = np.nan
    for e in T.EXPS:                                         # Holm over the 10 models of one set
        mk = (C.exp == e) & C.delong_p.notna()
        C.loc[mk, "delong_p_holm"] = T.holm(C.loc[mk, "delong_p"].values)
    C.to_csv(RES / tag / "compare_models_vs_logit.csv", index=False, float_format="%.6g")

    # 2. block effects inside the logistic regression (same 12 matched pairs as the paper)
    rows = []
    for blk, prs in T.BLOCK_PAIRS.items():
        for a, b in prs:
            d, z, p_dl = T.delong(y, L[a], L[b])
            _, lo, hi, p_bt = T.boot_auc_diff(y, L[a], L[b], idx)
            d_b, dm_b, p_b = T.diebold_mariano(T.brier_loss(y, L[a]), T.brier_loss(y, L[b]))
            rows.append(dict(block=blk, comparison=f"{a} vs {b}", auc_with=T.auc(y, L[a]),
                             auc_without=T.auc(y, L[b]), d_auc=d, ci_lo=lo, ci_hi=hi,
                             delong_p=p_dl, boot_p=p_bt, d_brier=d_b, dm_brier=dm_b,
                             dm_brier_p=p_b))
    B = pd.DataFrame(rows)
    B.to_csv(RES / tag / "compare_logit_block_effects.csv", index=False, float_format="%.6g")

    # 3. block effects: logit vs the mean of the ten models
    rows = []
    for blk, prs in T.BLOCK_PAIRS.items():
        for a, b in prs:
            dm = np.mean([T.auc(y, P[m][a]) - T.auc(y, P[m][b]) for m in T.MODELS])
            rows.append(dict(block=blk, comparison=f"{a} vs {b}",
                             d_auc_logit=T.auc(y, L[a]) - T.auc(y, L[b]),
                             d_auc_mean_of_10_models=dm))
    pd.DataFrame(rows).to_csv(RES / tag / "compare_block_effects_logit_vs_models.csv",
                              index=False, float_format="%.6g")

    # text summary
    S = pd.read_csv(RES / tag / "summary.csv")
    lines = [f"Test days {len(y)}, net-buy share {y.mean():.3f}. Models: seed-ensemble "
             f"probabilities over seeds {seeds}. Input: input_gdp_corrected.csv (as in GDP_corrected/models).", ""]
    lines.append("=== Naive baselines")
    for _, r in S[S.baseline.isin(["majority", "persistence"])].iterrows():
        lines.append(f"  {r.baseline:12s} AUC {r.AUC:.4f}  DA {r.Directional_Accuracy:.4f}  "
                     f"BalAcc {r.Balanced_Accuracy:.4f}  MCC {r.MCC:+.3f}")
    r = S[S.baseline == "logit_min"].iloc[0]
    lines.append(f"  logit flow+SOX AUC {r.AUC:.4f}  DA {r.Directional_Accuracy:.4f}  "
                 f"BrierSkill {r.Brier_Skill:+.4f}")
    lines += ["", "=== AUC by feature set: logistic regression vs the ten models (seed ensemble)"]
    piv = C.pivot(index="model", columns="exp", values="auc")
    piv.loc["mean of 10 models"] = piv.drop("Logistic regression").mean()
    lines.append(piv.round(4).to_string())
    lines += ["", "=== Models significantly different from the logit (DeLong raw p<0.05, Holm<0.05)"]
    for e in T.EXPS:
        s = C[(C.exp == e) & C.delong_p.notna()]
        up = s[(s.delong_p < .05) & (s.d_auc_vs_logit > 0)].model.tolist()
        dn = s[(s.delong_p < .05) & (s.d_auc_vs_logit < 0)].model.tolist()
        lines.append(f"  {e}: better {up or '-'}  worse {dn or '-'}  "
                     f"(Holm<0.05: {int((s.delong_p_holm < .05).sum())})")
    lines += ["", "=== Block effects inside the logistic regression (x100 AUC)"]
    for _, r in B.iterrows():
        lines.append(f"  {r.block:4s} {r.comparison:9s} dAUC {r.d_auc*100:+6.2f}  DeLong p={r.delong_p:.3f}  "
                     f"boot p={r.boot_p:.3f}  DM-Brier p={r.dm_brier_p:.3f}")
    txt = "\n".join(lines)
    (RES / tag / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    if not a.compare:
        run_all(a.seeds)
    compare(a.seeds)


if __name__ == "__main__":
    main()
