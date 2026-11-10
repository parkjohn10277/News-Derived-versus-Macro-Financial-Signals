"""
Build one self-contained copy of the GDP-corrected model code per earlier forecast origin (W1-W3), so the seven
neural networks can be run on the same windows as ../Expanding_origin (logit and tree models).

    python setup_windows.py

For each window it
  1. copies ../GDP_corrected/models (code only; no results/checkpoints) to windows/<W>/models,
  2. truncates every model input (data/input_aligned.csv and <Model>/data/input.csv) at the end of the test year,
  3. sets the split dates: phase 1 trains on [start, val_start) and selects the epoch on [val_start, train_end];
     phase 2 refits on [start, train_end] and predicts [test_start, end].
Nothing else changes: the input is input_gdp_corrected.csv (release-dated macro inputs, GDP lag 70 days), and the
architectures, lookback 30, seeds, epochs, patience and learning rates are those of the paper's run.
W4 (the paper's split) is not rerun: it is ../GDP_corrected itself.
"""
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "GDP_corrected" / "models"
REF = HERE.parent / "GDP_corrected" / "data" / "input_gdp_corrected.csv"
WINDOWS = {   # identical to ../Expanding_origin/run_expanding.py
    "W1": {"val_start": "2021-01-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "end": "2022-12-31"},
    "W2": {"val_start": "2022-01-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "end": "2023-12-31"},
    "W3": {"val_start": "2023-01-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "end": "2024-12-31"},
}
JUN = {"LSTM": "lstm_classifier.py", "CNN-LSTM": "cnn_lstm_classifier.py", "GRU-LSTM": "gru_lstm_classifier.py",
       "BiGRU-LSTM": "bigru_lstm_classifier.py"}
TEMPLATE = ["BiLSTM", "DA-RNN", "iTransformer"]


def md5(f: Path) -> str:
    return hashlib.md5(f.read_bytes()).hexdigest()


def sub_once(text, pattern, repl, what):
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        raise SystemExit(f"{what}: expected one match for {pattern!r}, found {n}")
    return new


def main():
    ref_md5 = md5(REF)
    for f in [SRC / "data" / "input_aligned.csv"] + [SRC / m / "data" / "input.csv" for m in TEMPLATE]:
        assert md5(f) == ref_md5, f"{f} is not input_gdp_corrected.csv"
    full = pd.read_csv(REF, encoding="utf-8-sig")
    date_col = full.columns[0]
    assert date_col.lower() == "date", date_col

    for w, s in WINDOWS.items():
        dst = HERE / "windows" / w / "models"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        shutil.copytree(SRC / "data", dst / "data")
        for m in list(JUN) + TEMPLATE:
            shutil.copytree(SRC / m, dst / m, ignore=shutil.ignore_patterns("results", "__pycache__"))

        # 2. truncated input: the original text lines up to the end of the test year (byte-identical prefix)
        d = pd.to_datetime(full[date_col])
        cut = full[d <= pd.Timestamp(s["end"])]
        raw = REF.read_bytes().splitlines(keepends=True)          # header + one line per row
        assert len(raw) == len(full) + 1
        body = b"".join(raw[: len(cut) + 1])
        for f in [dst / "data" / "input_aligned.csv"] + [dst / m / "data" / "input.csv" for m in TEMPLATE]:
            f.write_bytes(body)
        assert REF.read_bytes().startswith(body)

        # 3a. hard-coded dates in the four standalone scripts
        for m, script in JUN.items():
            p = dst / m / script
            t = p.read_text(encoding="utf-8")
            t = sub_once(t, r'^OFFICIAL_TRAIN_END = pd\.Timestamp\("[\d-]+"\)',
                         f'OFFICIAL_TRAIN_END = pd.Timestamp("{s["train_end"]}")', p)
            t = sub_once(t, r'^TEST_START = pd\.Timestamp\("[\d-]+"\)', f'TEST_START = pd.Timestamp("{s["test_start"]}")', p)
            t = sub_once(t, r'^VAL_START = pd\.Timestamp\("[\d-]+"\)', f'VAL_START = pd.Timestamp("{s["val_start"]}")', p)
            t = sub_once(t, r'^VAL_END = pd\.Timestamp\("[\d-]+"\)', f'VAL_END = pd.Timestamp("{s["train_end"]}")', p)
            p.write_text(f"# EXPANDING-ORIGIN COPY, window {w}: dates set by ../../../setup_windows.py\n" + t,
                         encoding="utf-8")
        # 3b. config.yaml of the three template models
        for m in TEMPLATE:
            p = dst / m / "config.yaml"
            t = p.read_text(encoding="utf-8")
            for k in ("val_start", "train_end", "test_start"):
                t = sub_once(t, rf"^(\s+{k}:\s+)[\d-]+", rf"\g<1>{s[k]}", f"{p} {k}")
            p.write_text(f"# EXPANDING-ORIGIN COPY, window {w}: split set by ../../../setup_windows.py\n" + t,
                         encoding="utf-8")
        n = lambda a, b: int(((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))).sum())
        print(f"{w}: rows {len(cut)} | phase-1 train <{s['val_start']}: {int((d < pd.Timestamp(s['val_start'])).sum())}"
              f" | validation {n(s['val_start'], s['train_end'])} | test {n(s['test_start'], s['end'])}")


if __name__ == "__main__":
    main()
