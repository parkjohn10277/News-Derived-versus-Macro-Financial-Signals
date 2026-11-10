"""
Run the E0-E7 grid for the direction CLASSIFIER.

    python run.py                          # model + seeds from config.yaml
    python run.py --exp E0 E7 --seeds 42
    python run.py --epochs 2               # smoke test
    python run.py --magnitude-weighting    # weight each day by |arcsinh(flow/c)|

Outputs  results/<model>/E{k}/seed_{s}/
    metrics.json, predictions.csv (with prob_net_buy), features_used.csv,
    best_model.pt, x_scaler.pkl, epoch histories, attention_alpha.csv if the
    model exposes it
and      results/<model>/summary.csv, summary_mean.csv
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import EXPERIMENTS, TARGET_COL, experiment_features, feature_groups  # noqa: E402
from src.models import build_model                                                      # noqa: E402
from src.protocol import run_single_split                                               # noqa: E402
from src.train import TrainCfg                                                          # noqa: E402

SUMMARY_COLS = ["Brier_Skill", "MCC", "AUC", "Balanced_Accuracy", "LogLoss_Skill",
                "F1_Score", "Directional_Accuracy", "Precision", "Recall",
                "Brier", "LogLoss", "base_rate_train", "base_rate_test",
                "pred_positive_rate", "best_epoch", "n_test"]


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
    ap.add_argument("--model", default=None)
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs (smoke test)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--magnitude-weighting", action="store_true",
                    help="weight each training day by |arcsinh(flow / scale_c)|")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]["name"]
    exps = args.exp or list(EXPERIMENTS)
    seeds = args.seeds or cfg["seeds"]
    thr = args.threshold if args.threshold is not None else cfg["classification"]["threshold"]
    magw = args.magnitude_weighting or cfg["classification"]["magnitude_weighting"]
    model_dir = ROOT / cfg["results"] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_input(cfg)
    groups = feature_groups(raw)
    print(f"model={model_name} (CLASSIFIER)  device={device}  rows={len(raw)}  "
          f"{raw.index.min().date()} -> {raw.index.max().date()}")
    print(f"groups: KRX={len(groups['KRX'])}  LDA={len(groups['LDA'])}  "
          f"LLM={len(groups['LLM'])}  AUX={len(groups['AUX'])}  | "
          f"threshold={thr}  magnitude_weighting={magw}")

    tc, mc = cfg["train"], cfg["model"]
    base = TrainCfg(T=mc["T"], batch_size=tc["batch_size"], max_epochs=args.epochs or tc["max_epochs"],
                    lr=tc["lr"], weight_decay=tc["weight_decay"], grad_clip=tc["grad_clip"],
                    early_stopping_patience=tc["early_stopping_patience"], min_delta=tc["min_delta"],
                    loss="bce", optimizer=tc["optimizer"], use_scheduler=tc["use_scheduler"],
                    attn_entropy_lambda=tc.get("attn_entropy_lambda", 0.0))
    factory = lambda n, T: build_model(model_name, n, T, mc.get("params", {}) or {})

    for exp in exps:
        feats = experiment_features(groups, exp)
        df = prepare(raw, groups, feats)
        print(f"\n=== {exp}: {'+'.join(EXPERIMENTS[exp])}  features={len(feats)}  rows={len(df)} ===")
        for seed in seeds:
            tcfg = TrainCfg(**{**base.__dict__, "seed": seed})
            res = run_single_split(df, feats, groups, cfg["split"], factory, tcfg,
                                   aux_z_clip=cfg["preprocess"]["aux_z_clip"], threshold=thr,
                                   magnitude_weighting=magw, device=device, verbose=args.verbose)
            m = res["metrics"]
            m.update({"model": model_name, "task": "classification", "experiment": exp,
                      "seed": seed, "lookback": tcfg.T, "prediction_horizon": 1,
                      "n_features_used": len(feats),
                      **{f"{g.lower()}_features": len([c for c in feats if c in groups[g]])
                         for g in ("KRX", "LDA", "LLM", "AUX")}})
            if res["alpha"] is not None:
                m["attention_top5"] = res["alpha"].mean().sort_values(ascending=False).head(5).round(4).to_dict()
            print(f"  seed {seed}: best_epoch={m['best_epoch']}  BrierSkill={m['Brier_Skill']:+.4f}  "
                  f"MCC={m['MCC']:+.3f}  AUC={m['AUC']:.3f}  BalAcc={m['Balanced_Accuracy']:.3f}  "
                  f"F1={m['F1_Score']:.3f}")

            od = model_dir / exp / f"seed_{seed}"
            od.mkdir(parents=True, exist_ok=True)
            (od / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")
            res["predictions"].to_csv(od / "predictions.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"feature": feats}).to_csv(od / "features_used.csv", index=False,
                                                    encoding="utf-8-sig")
            res["selection_history"].to_csv(od / "epoch_selection_history.csv", index=False)
            res["refit_history"].to_csv(od / "full_train_refit_history.csv", index=False)
            if res["alpha"] is not None:
                res["alpha"].to_csv(od / "attention_alpha.csv")
            for name, arr in res["extras"].items():
                np.save(od / f"extras_{name}.npy", arr)
            torch.save(res["model"].state_dict(), od / "best_model.pt")
            with open(od / "x_scaler.pkl", "wb") as f:
                pickle.dump(res["x_scaler"], f)

    mean = rebuild_summary(model_dir)
    print(f"\n================  {model_name} (classifier): mean over seeds  ================")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(mean[["experiment", "features", "n_features", "n_seeds", "Brier_Skill", "MCC", "AUC",
                    "Balanced_Accuracy", "F1_Score", "Directional_Accuracy", "best_epoch"]].to_string(index=False))
    print(f"\n(chance references: Brier_Skill 0, MCC 0, AUC 0.5, Balanced_Accuracy 0.5)")
    print(f"results in {model_dir}")


if __name__ == "__main__":
    main()
