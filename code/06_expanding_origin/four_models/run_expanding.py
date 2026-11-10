"""
EXPANDING-ORIGIN ROBUSTNESS CHECK -- the paper's protocol re-run with four forecast origins.

Each window refits on all data before its validation year, tunes on that year, refits on
train+validation and scores the following test block once.  Window W4 is the paper's split,
so its results must match ../GDP_corrected exactly (a reproducibility check).

    window  train (phase 2 = train + val)   validation (tuning)       test (scored once)
    W1      2015-03 .. 2020-12              2021-01-01 .. 2021-12-31  2022-01-01 .. 2022-12-31
    W2      2015-03 .. 2021-12              2022-01-01 .. 2022-12-31  2023-01-01 .. 2023-12-31
    W3      2015-03 .. 2022-12              2023-01-01 .. 2023-12-31  2024-01-01 .. 2024-12-31
    W4      2015-03 .. 2024-03-12           2024-03-13 .. 2025-05-07  2025-05-08 .. 2026-06-26  (paper)

Models   logistic regression, random forest, XGBoost, LightGBM  (same code as ../GDP_corrected)
Sets     E0-E7      Seeds 42-46
Input    data/input_gdp_corrected.csv  (release-dated macro, GDP lag 70 days; md5 checked)
Protocol identical to the paper: lags [0,1,2,5,10,20,29]; Yeo-Johnson + standardisation and the
         AUX z-clip (3) fit on training rows only; hyper-parameters by validation log loss
         (trees: 30-draw random search; logit: all 11 values of C); refit on train+validation.

    python run_expanding.py                       # everything (resumable: finished runs are skipped)
    python run_expanding.py --windows W1 --models logistic --exps E0 --seeds 42
    python run_expanding.py --collect             # rebuild results_all.csv / results_summary.csv only

Outputs  results/<window>/<model>/E<k>/seed_<s>/{metrics.json, predictions.csv, cv_results.csv}
         results_all.csv (one row per run), results_summary.csv (mean, sd over seeds)

Caveats (state in the paper): the LDA topic models and the EXAONE sentiment model were fitted on,
or trained with, text that overlaps the earlier test years.  Both can only favour the text features.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import EXPERIMENTS, TARGET_COL, experiment_features, feature_groups  # noqa: E402
from src.models import get_spec                                                         # noqa: E402
from src.protocol import run_single_split                                               # noqa: E402

INPUT = ROOT / "data" / "input_gdp_corrected.csv"
INPUT_MD5 = "52cc0eb4c8e9b10637f45a606a336ad0"
RES = ROOT / "results"
START = "2015-02-28"

WINDOWS = {
    "W1": {"val_start": "2021-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "end": "2022-12-31"},
    "W2": {"val_start": "2022-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "end": "2023-12-31"},
    "W3": {"val_start": "2023-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "end": "2024-12-31"},
    "W4": {"val_start": "2024-03-13", "train_end": "2025-05-07", "test_start": "2025-05-08", "end": None},
}
MODELS = {          # name -> search budget (None = the whole grid)
    "logistic": None,
    "lightgbm": 30,
    "random_forest": 30,
    "xgboost": 30,
}
EXPS = list(EXPERIMENTS)
SEEDS = [42, 43, 44, 45, 46]
LAGS = [0, 1, 2, 5, 10, 20, 29]
AUX_Z_CLIP, THRESHOLD = 3.0, 0.5

KEEP = ["AUC", "Brier_Skill", "LogLoss_Skill", "MCC", "Balanced_Accuracy", "Directional_Accuracy",
        "F1_Score", "Brier", "base_rate_train", "base_rate_test", "pred_positive_rate", "n_test",
        "best_validation_loss"]


def load_input() -> pd.DataFrame:
    md5 = hashlib.md5(INPUT.read_bytes()).hexdigest()
    if md5 != INPUT_MD5:
        raise SystemExit(f"{INPUT.name} has md5 {md5}, expected {INPUT_MD5}")
    df = pd.read_csv(INPUT, parse_dates=["date"], encoding="utf-8-sig").set_index("date").sort_index()
    return df[df.index >= pd.Timestamp(START)]


def prepare(df, groups, feature_cols):
    """Missing-value policy of the paper: LDA/LLM -> 0, AUX -> forward fill, drop NaN rows."""
    d = df[[TARGET_COL] + [c for c in feature_cols if c != TARGET_COL]].copy()
    for c in groups["LDA"] + groups["LLM"]:
        if c in d:
            d[c] = d[c].fillna(0.0)
    aux = [c for c in groups["AUX"] if c in d]
    if aux:
        d[aux] = d[aux].ffill()
    return d.dropna()


def run_dir(w, m, e, s) -> Path:
    return RES / w / m / e / f"seed_{s}"


def run_one(raw, groups, w, model, exp, seed) -> dict:
    win = WINDOWS[w]
    df = raw if win["end"] is None else raw[raw.index <= pd.Timestamp(win["end"])]
    feats = experiment_features(groups, exp)
    d = prepare(df, groups, feats)
    spec = get_spec(model)
    n_iter = MODELS[model] or len(next(iter(spec.GRID.values())))
    res = run_single_split(d, feats, groups, win, spec, seed, lags=LAGS, aux_z_clip=AUX_Z_CLIP,
                           threshold=THRESHOLD, cv_scheme="holdout", n_iter=n_iter,
                           n_jobs=1 if model == "logistic" else -1)
    m = res["metrics"]
    m.update({"window": w, "model": model, "experiment": exp, "seed": seed, "lags": LAGS,
              "n_features_used": len(feats)})
    od = run_dir(w, model, exp, seed)
    od.mkdir(parents=True, exist_ok=True)
    res["predictions"].to_csv(od / "predictions.csv", index=False, encoding="utf-8-sig")
    res["cv_results"].to_csv(od / "cv_results.csv", index=False)
    (od / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    return m


def collect() -> None:
    rows = []
    for f in RES.glob("W*/*/E*/seed_*/metrics.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"window": m["window"], "model": m["model"], "experiment": m["experiment"],
                     "seed": m["seed"], "test": m.get("test"), "validation": m.get("validation"),
                     **{k: m.get(k) for k in KEEP}})
    if not rows:
        print("no finished runs yet"); return
    df = pd.DataFrame(rows).sort_values(["window", "model", "experiment", "seed"])
    df.to_csv(ROOT / "results_all.csv", index=False, float_format="%.6g")
    g = df.groupby(["window", "model", "experiment"])
    s = g[["AUC", "Brier_Skill", "MCC", "Balanced_Accuracy", "Directional_Accuracy"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n_seeds", g.size())
    s.reset_index().to_csv(ROOT / "results_summary.csv", index=False, float_format="%.4f")
    auc = df.pivot_table(index=["window", "model"], columns="experiment", values="AUC", aggfunc="mean")
    print(f"\n{len(df)} runs collected.  Mean AUC over seeds:\n{auc.round(3).to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--exps", nargs="+", default=EXPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect(); return

    raw = load_input()
    groups = feature_groups(raw)
    todo = [(w, m, e, s) for w in args.windows for m in args.models for e in args.exps for s in args.seeds]
    t0 = time.time()
    for k, (w, m, e, s) in enumerate(todo, 1):
        if (run_dir(w, m, e, s) / "metrics.json").exists():
            continue
        t = time.time()
        try:
            met = run_one(raw, groups, w, m, e, s)
            msg = f"OK    AUC={met['AUC']:.3f}  test {met['test']}"
        except Exception as ex:                                    # keep going, report at the end
            msg = f"FAIL  {type(ex).__name__}: {ex}"
        print(f"[{k}/{len(todo)}] {w} {m:13s} {e} seed={s}  {msg}  "
              f"{time.time() - t:6.1f}s  total {(time.time() - t0) / 60:6.1f} min")
    collect()


if __name__ == "__main__":
    main()
