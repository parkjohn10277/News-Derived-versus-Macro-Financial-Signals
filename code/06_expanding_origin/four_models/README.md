# Expanding-origin robustness check

This folder reruns the paper's protocol at four forecast origins, so the information-set result can be checked outside
the single 2025–26 test window. `../GDP_corrected` and `../Final_run` are not modified.

| Window | Training (phase 2 = train + validation) | Validation (tuning) | Test (scored once) |
|---|---|---|---|
| W1 | 2015 – 2020 | 2021 | 2022 |
| W2 | 2015 – 2021 | 2022 | 2023 |
| W3 | 2015 – 2022 | 2023 | 2024 |
| W4 | 2015 – 2024-03-12 | 2024-03-13 – 2025-05-07 | 2025-05-08 – 2026-06-26 (the paper's split) |

In each window the models are tuned on the validation year, refit on everything before the test block, and scored
once on the test block. W4 is the paper's split, so its results must reproduce `../GDP_corrected` exactly. The W4
logit E1 gives AUC 0.7118, the same as the baseline.

- **Models:** logistic regression, random forest, XGBoost and LightGBM. The code is the tree `src/` of
  `../GDP_corrected`, plus `src/models/logistic.py`.
- **Feature sets and seeds:** E0–E7, seeds 42–46 (640 runs).
- **Input:** `data/input_gdp_corrected.csv`, md5 `52cc0eb4c8e9b10637f45a606a336ad0`. The macro data are release-dated
  and the GDP lag is 70 days.
- **Protocol:** identical to the paper.
  - Lags 0, 1, 2, 5, 10, 20 and 29.
  - Yeo-Johnson, standardization and the AUX clip are fit on training rows only.
  - Hyperparameters are chosen by validation log loss.
  - The model is refit on train + validation.

## Run

```
cd .\Expanding_origin
python run_expanding.py            # all 640 runs, resumable (log: run_log.txt)
python run_expanding.py --collect  # results_all.csv, results_summary.csv
python analyze_expanding.py        # analysis/
```

Here `python` is `<your conda env>\python.exe`.

## Outputs

- `results/<window>/<model>/E<k>/seed_<s>/`: `metrics.json`, `predictions.csv` and `cv_results.csv`.
- `analysis/auc_by_window.csv`: seed-ensemble AUC per window, model and set, plus the majority and persistence
  baselines.
- `analysis/block_pairs.csv`: for every window, model and "add one block" pair, the dAUC with DeLong, block-bootstrap
  and DM-Brier p-values.
- `analysis/block_by_window.csv`: the AUX, LDA and LLM effects per window, plus a test pooled over windows (Wilcoxon,
  with window × model as the unit).
- `analysis/summary.txt`: the numbers to quote.

## Caveats

The LDA topic models were fitted on the full 2015–2026 corpus, and the EXAONE sentiment model may have seen text from
2022–24 in pretraining. Both can only favour the text features in the earlier windows, which works against the finding
that they add nothing.
