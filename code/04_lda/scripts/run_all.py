"""
Run the whole pipeline end to end:  00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06

Each stage caches its output, so in practice you re-run from the earliest stage
whose inputs (config or code) changed. This runner is a convenience for a clean
full build.

    python scripts/run_all.py            # all stages
    python scripts/run_all.py 02 06      # inclusive range of stages
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import importlib
import sys
import time

STAGES = [
    ("00_diagnostic", "00  diagnostics"),
    ("01_filter_and_split", "01  domain filter + language split"),
    ("02_preprocess", "02  tokenisation (ko: kiwipiepy, en: NLTK)"),
    ("03_build_corpus", "03  dictionary + BoW"),
    ("04_train_lda", "04  LDA (K=20 per language)"),
    ("05_visualize", "05  pyLDAvis"),
    ("06_timeseries", "06  TR_{t,k} topic time series"),
]


def main(argv: list[str]) -> None:
    lo, hi = 0, len(STAGES) - 1
    if len(argv) == 2:
        lo = next(i for i, (m, _) in enumerate(STAGES) if m.startswith(argv[0]))
        hi = next(i for i, (m, _) in enumerate(STAGES) if m.startswith(argv[1]))

    for mod_name, label in STAGES[lo:hi + 1]:
        print("\n" + "=" * 70 + f"\n{label}\n" + "=" * 70)
        t0 = time.time()
        importlib.import_module(mod_name).main()
        print(f"[{label}] {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
