"""
Run the E0-E7 grid for one tree DIRECTION CLASSIFIER under the shared protocol.

    python run.py                             # model + seeds from config.yaml
    python run.py --exp E0 E7 --seeds 42
    python run.py --n-iter 5                  # smoke test (tiny search budget)

Outputs  results/<model>/E{k}/seed_{s}/
    metrics.json, predictions.csv (with prob_net_buy), features_used.csv,
    design_features.csv, best_model.joblib, x_scaler.pkl, cv_results.csv
and      results/<model>/summary.csv, summary_mean.csv
         (rebuilt from every metrics.json on disk after each run)
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import EXPERIMENTS, TARGET_COL, experiment_features, feature_groups  # noqa: E402
from src.models import get_spec                                                         # noqa: E402
from src.protocol import run_single_split                                               # noqa: E402

SUMMARY_COLS = ["Brier_Skill", "MCC", "AUC", "Balanced_Accuracy", "LogLoss_Skill",
                "F1_Score", "Directional_Accuracy", "Precision", "Recall",
                "Brier", "LogLoss", "base_rate_train", "base_rate_test",
                "pred_positive_rate", "n_design_features", "n_test"]


def load_input(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["input"], parse_dates=["date"]).set_index("date").sort_index()
    df = df[df.index >= pd.Timestamp(cfg["data"]["start"])]
    if cfg["data"].get("end"):
        df = df[df.index <= pd.Timestamp(cfg["data"]["end"])]
    return df


def prepare(df: pd.DataFrame, groups: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Missing-value policy: LDA/LLM -> 0, AUX -> forward-fill, then drop remaining NaN rows."""
    d = df[[TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]].copy()
    for c in groups["LDA"] + groups["LLM"]:
        if c in d:
            d[c] = d[c].fillna(0.0)
    aux = [c for c in groups["AUX"] if c in d]
    if aux:
        d[aux] = d[aux].ffill()
    n0 = len(d)
    d = d.dropna()
    if len(d) != n0:
        print(f"    dropped {n0 - len(d)} rows with missing features")
    return d


def rebuild_summary(model_dir: Path) -> pd.DataFrame:
    rows = []
    for f in model_dir.glob("E*/seed_*/metrics.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"experiment": m["experiment"], "seed": m["seed"],
                     "features": "+".join(EXPERIMENTS[m["experiment"]]),
                     "n_features": m["n_features_used"],
                     # tolerate runs written before a metric was added
                     **{k: m.get(k, float("nan")) for k in SUMMARY_COLS}})
    summ = pd.DataFrame(rows).sort_values(["experiment", "seed"])
    summ.to_csv(model_dir / "summary.csv", index=False)
    mean = summ.groupby(["experiment", "features", "n_features"])[SUMMARY_COLS].mean().reset_index()
    mean.insert(3, "n_seeds", summ.groupby("experiment").size().values)
    mean.to_csv(model_dir / "summary_mean.csv", index=False)
    return mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--model", default=None, help="override config model.name")
    ap.add_argument("--exp", nargs="*", default=None, help="subset of E0..E7")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n-iter", type=int, default=None, help="override search budget")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]["name"]
    spec = get_spec(model_name)
    exps = args.exp or list(EXPERIMENTS)
    seeds = args.seeds or cfg["seeds"]
    n_iter = args.n_iter or cfg["search"]["n_iter"]
    thr = args.threshold if args.threshold is not None else cfg["classification"]["threshold"]
    model_dir = ROOT / cfg["results"] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = load_input(cfg)
    groups = feature_groups(raw)
    lags = cfg["features"]["lags"]
    print(f"model={model_name} (CLASSIFIER)  rows={len(raw)}  "
          f"{raw.index.min().date()} -> {raw.index.max().date()}")
    print(f"groups: KRX={len(groups['KRX'])}  LDA={len(groups['LDA'])}  "
          f"LLM={len(groups['LLM'])}  AUX={len(groups['AUX'])}  | lags={lags}  "
          f"n_iter={n_iter}  threshold={thr}")

    for exp in exps:
        feats = experiment_features(groups, exp)
        df = prepare(raw, groups, feats)
        print(f"\n=== {exp}: {'+'.join(EXPERIMENTS[exp])}  columns={len(feats)}  "
              f"design={len(feats) * len(lags)}  rows={len(df)} ===")
        for seed in seeds:
            res = run_single_split(
                df, feats, groups, cfg["split"], spec, seed, lags=lags,
                aux_z_clip=cfg["preprocess"]["aux_z_clip"], threshold=thr,
                cv_scheme=cfg["search"]["cv"], n_iter=n_iter,
                n_jobs=cfg["search"].get("n_jobs", -1), verbose=args.verbose,
            )
            m = res["metrics"]
            m.update({"model": model_name, "task": "classification", "experiment": exp,
                      "seed": seed, "lags": lags, "prediction_horizon": 1,
                      "n_features_used": len(feats),
                      **{f"{g.lower()}_features": len([c for c in feats if c in groups[g]])
                         for g in ("KRX", "LDA", "LLM", "AUX")}})
            print(f"  seed {seed}: BrierSkill={m['Brier_Skill']:+.4f}  MCC={m['MCC']:+.3f}  "
                  f"AUC={m['AUC']:.3f}  BalAcc={m['Balanced_Accuracy']:.3f}  "
                  f"F1={m['F1_Score']:.3f}")

            od = model_dir / exp / f"seed_{seed}"
            od.mkdir(parents=True, exist_ok=True)
            (od / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
            res["predictions"].to_csv(od / "predictions.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"feature": feats}).to_csv(od / "features_used.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"design_feature": res["feature_names"]}).to_csv(
                od / "design_features.csv", index=False, encoding="utf-8-sig")
            res["cv_results"].to_csv(od / "cv_results.csv", index=False)
            joblib.dump(res["model"], od / "best_model.joblib")
            with open(od / "x_scaler.pkl", "wb") as f:
                pickle.dump(res["x_scaler"], f)

    mean = rebuild_summary(model_dir)
    print(f"\n================  {model_name} (classifier): mean over seeds  ================")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(mean[["experiment", "features", "n_features", "n_seeds",
                    "Brier_Skill", "MCC", "AUC", "Balanced_Accuracy",
                    "F1_Score", "Directional_Accuracy"]].to_string(index=False))
    print("\n(chance references: Brier_Skill 0, MCC 0, AUC 0.5, Balanced_Accuracy 0.5)")
    print(f"results in {model_dir}")


if __name__ == "__main__":
    main()
