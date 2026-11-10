"""
Run the seven neural networks of the paper on the earlier forecast origins W1-W3 (test years 2022, 2023, 2024).

    python setup_windows.py                 # once: per-window code copies in windows/<W>/models
    python run_windows.py                   # all 7 models x E0-E7 x seeds 42-46 x W1-W3 = 840 runs (resumable)
    python run_windows.py --windows W1 --models LSTM --exps E0 --seeds 42      # a single run
    python run_windows.py --collect         # results_nn_all.csv only

Commands, hyper-parameters and seeds are exactly those of ../GDP_corrected/run_final.py; only the split dates and
the end of the input differ (set by setup_windows.py).  The fast networks run first in every window, DA-RNN
(about 75% of the GPU time) last.

Results  windows/<W>/models/<Model>/results/<name>/E<k>/seed_<s>/{metrics.json, predictions.csv, ...}
Log      run_log.txt (one line per run), logs/<W>_<Model>_<E>_s<seed>.log (full output of each run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "GDP_corrected" / "data" / "input_gdp_corrected.csv"
WINDOWS = {"W1": ("2022-01-01", "2022-12-31"), "W2": ("2023-01-01", "2023-12-31"), "W3": ("2024-01-01", "2024-12-31")}
MODELS = {   # name -> (kind, script, results dir name), as in ../GDP_corrected/run_final.py
    "LSTM":         ("jun", "lstm_classifier.py", "lstm"),
    "CNN-LSTM":     ("jun", "cnn_lstm_classifier.py", "cnn_lstm"),
    "GRU-LSTM":     ("jun", "gru_lstm_classifier.py", "gru_lstm"),
    "BiGRU-LSTM":   ("jun", "bigru_lstm_classifier.py", "bigru_lstm"),
    "BiLSTM":       ("template", "run.py", "bilstm"),
    "iTransformer": ("template", "run.py", "itransformer"),
    "DA-RNN":       ("template", "run.py", "darnn"),
}
EXPS = [f"E{k}" for k in range(8)]
SEEDS = [42, 43, 44, 45, 46]
JUN_ARGS = ["--lookback", "30", "--horizon", "1", "--batch-size", "64", "--epochs", "100", "--patience", "10",
            "--lr", "0.001", "--aux-z-clip", "3.0", "--threshold", "0.5"]


def wd(w, m):
    return HERE / "windows" / w / "models" / m


def run_dir(w, m, e, s):
    return wd(w, m) / "results" / MODELS[m][2] / e / f"seed_{s}"


def command(w, m, e, s):
    kind, script, name = MODELS[m]
    base = [sys.executable, "-W", "ignore", str(wd(w, m) / script), "--exp", e]
    if kind == "jun":
        return base + ["--seed", str(s), *JUN_ARGS, "--outdir", str(wd(w, m) / "results" / name)]
    return base + ["--seeds", str(s)]


def check_inputs():
    """Every window input must be a prefix of input_gdp_corrected.csv ending in the window's test year."""
    ref = REF.read_bytes()
    for w, (t0, t1) in WINDOWS.items():
        files = [HERE / "windows" / w / "models" / "data" / "input_aligned.csv"] + \
                [wd(w, m) / "data" / "input.csv" for m, v in MODELS.items() if v[0] == "template"]
        for f in files:
            b = f.read_bytes()
            if not ref.startswith(b):
                raise SystemExit(f"{f} is not a prefix of input_gdp_corrected.csv; rerun setup_windows.py")
        last = pd.read_csv(files[0], encoding="utf-8-sig", usecols=["date"]).date.iloc[-1]
        if not (t0 <= last <= t1):
            raise SystemExit(f"{w}: input ends {last}, expected in {t0}..{t1}")
    print(f"inputs OK: every window input is a byte prefix of {REF.name} "
          f"(md5 {hashlib.md5(ref).hexdigest()})", flush=True)


def check_test_days(w, m, e, s):
    p = pd.read_csv(run_dir(w, m, e, s) / "predictions.csv", encoding="utf-8-sig", usecols=["date"])
    t0, t1 = WINDOWS[w]
    return p.date.min()[:10] >= t0 and p.date.max()[:10] <= t1


def collect():
    rows = []
    for w in WINDOWS:
        for m, (_, _, name) in MODELS.items():
            for f in (wd(w, m) / "results" / name).glob("E*/seed_*/metrics.json"):
                j = json.loads(f.read_text(encoding="utf-8"))
                rows.append({"window": w, "model": m, "experiment": f.parents[1].name,
                             "seed": int(f.parent.name.split("_")[1]),
                             **{k: j.get(k) for k in ["AUC", "MCC", "Brier_Skill", "Balanced_Accuracy",
                                                      "Directional_Accuracy", "best_epoch", "best_validation_loss"]}})
    if not rows:
        print("no finished runs yet"); return
    df = pd.DataFrame(rows).sort_values(["window", "model", "experiment", "seed"])
    df.to_csv(HERE / "results_nn_all.csv", index=False, float_format="%.6g")
    piv = df.pivot_table(index=["window", "model"], columns="experiment", values="AUC", aggfunc="mean")
    print(f"\n{len(df)} runs collected.  Mean AUC over seeds:\n{piv.round(3).to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=list(WINDOWS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--exps", nargs="+", default=EXPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect(); return
    check_inputs()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    (HERE / "logs").mkdir(exist_ok=True)
    order = [m for m in MODELS if m in a.models]       # DA-RNN last
    todo = [(w, m, e, s) for m in order for w in a.windows for s in a.seeds for e in a.exps]
    t0 = time.time()
    for k, (w, m, e, s) in enumerate(todo, 1):
        if (run_dir(w, m, e, s) / "metrics.json").exists():
            continue
        t = time.time()
        with open(HERE / "logs" / f"{w}_{m}_{e}_s{s}.log", "w", encoding="utf-8") as fh:
            r = subprocess.run(command(w, m, e, s), stdout=fh, stderr=subprocess.STDOUT, cwd=wd(w, m), env=env)
        ok = r.returncode == 0 and (run_dir(w, m, e, s) / "metrics.json").exists() and check_test_days(w, m, e, s)
        msg = f"[{k}/{len(todo)}] {'OK  ' if ok else 'FAIL'} {w} {m:12s} {e} seed={s}  " \
              f"{time.time() - t:6.1f}s  total {(time.time() - t0) / 60:6.1f} min"
        print(msg, flush=True)
        with open(HERE / "run_log.txt", "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    collect()


if __name__ == "__main__":
    main()
