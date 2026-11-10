"""
GDP-CORRECTED RUN -- a copy of ../Final_run/run_final.py (code unchanged) on the
corrected input data/input_gdp_corrected.csv: macro variables dated by release
(publication lags CPI 5, unemployment 16, semiconductor production 31, GDP 70,
EPU/GPR 7 days).  The file is placed under the names the models read
(models/data/input_aligned.csv and models/<Model>/data/input.csv); main() checks
that every copy matches it before running.  ../Final_run is left untouched.

FINAL RUN -- all 10 direction classifiers x E0-E7 x 5 seeds, one protocol.

    python run_final.py                         # everything (resumable: finished runs are skipped)
    python run_final.py --models LSTM --exps E7 --seeds 42     # a single run
    python run_final.py --collect               # only rebuild the summary tables

Common settings (all models)
  input      models/data/input_aligned.csv  (KRX + 13 AUX + 11 LDA + 4 LLM; E7 = 29 features)
  split      train <= 2024-03-12 | validation 2024-03-13..2025-05-07 | test 2025-05-08..2026-06-26
  lookback   30 trading days (trees: lags t-1 ... t-30), horizon 1
  protocol   phase 1 = epoch / hyper-parameter selection on the validation block,
             phase 2 = refit on train+validation, predict test
  seeds      42 43 44 45 46
  threshold  0.5, AUX z-clip 3
Model-specific (documented): DA-RNN patience 20; iTransformer lr 1e-4;
trees tune hyper-parameters by a 30-draw random search on the validation block.

Results  models/<Model>/results/<name>/E<k>/seed_<s>/{metrics.json, predictions.csv, ...}
Tables   results_all.csv (one row per run), results_summary.csv (mean, sd over seeds)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
M = HERE / "models"

# name -> (kind, working dir, script, results dir name)
MODELS = {
    "LSTM":         ("jun", M / "LSTM", "lstm_classifier.py", "lstm"),
    "CNN-LSTM":     ("jun", M / "CNN-LSTM", "cnn_lstm_classifier.py", "cnn_lstm"),
    "GRU-LSTM":     ("jun", M / "GRU-LSTM", "gru_lstm_classifier.py", "gru_lstm"),
    "BiGRU-LSTM":   ("jun", M / "BiGRU-LSTM", "bigru_lstm_classifier.py", "bigru_lstm"),
    "BiLSTM":       ("template", M / "BiLSTM", "run.py", "bilstm"),
    "DA-RNN":       ("template", M / "DA-RNN", "run.py", "darnn"),
    "iTransformer": ("template", M / "iTransformer", "run.py", "itransformer"),
    "RandomForest": ("template", M / "RandomForest", "run.py", "random_forest"),
    "XGBoost":      ("template", M / "XGBoost", "run.py", "xgboost"),
    "LightGBM":     ("template", M / "LightGBM", "run.py", "lightgbm"),
}
EXPS = [f"E{k}" for k in range(8)]
SEEDS = [42, 43, 44, 45, 46]
LOOKBACK, HORIZON = 30, 1
JUN_ARGS = ["--lookback", str(LOOKBACK), "--horizon", str(HORIZON), "--batch-size", "64",
            "--epochs", "100", "--patience", "10", "--lr", "0.001", "--aux-z-clip", "3.0",
            "--threshold", "0.5"]


def run_dir(model: str, exp: str, seed: int) -> Path:
    kind, wd, _, name = MODELS[model]
    return wd / "results" / name / exp / f"seed_{seed}"


def command(model: str, exp: str, seed: int) -> list[str]:
    kind, wd, script, name = MODELS[model]
    base = [sys.executable, "-W", "ignore", str(wd / script), "--exp", exp]
    if kind == "jun":
        return base + ["--seed", str(seed), *JUN_ARGS, "--outdir", str(wd / "results" / name)]
    return base + ["--seeds", str(seed)]      # lookback / lr / patience come from config.yaml


def collect() -> None:
    rows = []
    for model, (kind, wd, _, name) in MODELS.items():
        for f in (wd / "results" / name).glob("E*/seed_*/metrics.json"):
            m = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"model": model, "experiment": f.parents[1].name,
                         "seed": int(f.parent.name.split("_")[1]),
                         **{k: m.get(k) for k in ["AUC", "MCC", "Brier_Skill", "LogLoss_Skill",
                                                  "Balanced_Accuracy", "Directional_Accuracy",
                                                  "F1_Score", "Precision", "Recall",
                                                  "pred_positive_rate", "best_epoch",
                                                  "best_validation_loss"]},
                         "n_features": m.get("n_features_used")})
    if not rows:
        print("no finished runs yet"); return
    df = pd.DataFrame(rows).sort_values(["model", "experiment", "seed"])
    df.to_csv(HERE / "results_all.csv", index=False, float_format="%.6g")
    metrics = ["AUC", "MCC", "Brier_Skill", "Balanced_Accuracy", "Directional_Accuracy", "F1_Score"]
    g = df.groupby(["model", "experiment"])
    s = g[metrics].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n_seeds", g.size())
    s.reset_index().to_csv(HERE / "results_summary.csv", index=False, float_format="%.4f")
    auc = df.pivot_table(index="model", columns="experiment", values="AUC", aggfunc="mean")
    print(f"\n{len(df)} runs collected.  Mean AUC over seeds:\n{auc.round(3).to_string()}")


def check_inputs() -> None:
    """Every model must read the GDP-corrected input."""
    import hashlib
    md5 = lambda f: hashlib.md5(f.read_bytes()).hexdigest()
    ref = md5(HERE / "data" / "input_gdp_corrected.csv")
    files = [M / "data" / "input_aligned.csv"] + sorted(M.glob("*/data/input.csv"))
    bad = [str(f) for f in files if md5(f) != ref]
    if bad:
        raise SystemExit(f"these inputs are not input_gdp_corrected.csv: {bad}")
    print(f"inputs OK: {len(files)} copies of data/input_gdp_corrected.csv (md5 {ref})")


def main():
    check_inputs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--exps", nargs="+", default=EXPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect(); return

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    (HERE / "logs").mkdir(exist_ok=True)
    todo = [(m, e, s) for s in args.seeds for m in args.models for e in args.exps]
    t0 = time.time()
    for k, (m, e, s) in enumerate(todo, 1):
        if (run_dir(m, e, s) / "metrics.json").exists():
            continue
        t = time.time()
        with open(HERE / "logs" / f"{m}_{e}_s{s}.log", "w", encoding="utf-8") as fh:
            r = subprocess.run(command(m, e, s), stdout=fh, stderr=subprocess.STDOUT,
                               cwd=MODELS[m][1], env=env)
        ok = r.returncode == 0 and (run_dir(m, e, s) / "metrics.json").exists()
        print(f"[{k}/{len(todo)}] {'OK  ' if ok else 'FAIL'} {m:12s} {e} seed={s}  "
              f"{time.time() - t:6.1f}s  total {(time.time() - t0) / 60:6.1f} min", flush=True)
    collect()


if __name__ == "__main__":
    main()
