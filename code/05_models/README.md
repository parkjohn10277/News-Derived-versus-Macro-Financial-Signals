# GDP-corrected rerun: 10 classifiers × E0–E7 × 5 seeds, plus baselines and tests

This is a full rerun of `../Final_run` on a corrected input. **`../Final_run` is not modified.**

## Why

The reported results in `../Final_run` read `input_aligned.csv`, which dates the monthly and quarterly macro variables by
the **end of their reference period**, with no publication lag. This rerun uses the release-dated input that the
manuscript describes in Eq. (3). It also corrects the GDP lag. The series is **nominal** GDP, which the Bank of Korea first
publishes with the *preliminary* estimate about 64–70 days after the quarter ends. The 28-day *advance* estimate covers
real GDP only.

| Variable | Lag (days after period end) |
|---|---|
| CPI y/y | 5 |
| Unemployment rate | 16 |
| Semiconductor production y/y | 31 |
| **Nominal GDP y/y** | **70** (was 28 in `../data/input_auxfixed.csv`, 0 in the Final_run input) |
| Global EPU, GPR global, GPR Korea | 7 |

Nothing else changes: the code, split, seeds, hyperparameter searches, daily variables, LDA and LLM columns are all the same.

## Contents

| Path | What |
|---|---|
| `data/build_input_gdp_corrected.py` | Copy of `../data/build_input_auxfixed.py` with the GDP lag set to 70 days |
| `data/input_gdp_corrected.csv` | The corrected input (md5 `52cc0eb4c8e9b10637f45a606a336ad0`) |
| `models/` | Frozen copy of `../Final_run/models` (code only). The corrected file sits under the names the code reads: `models/data/input_aligned.csv` and `models/<Model>/data/input.csv` |
| `run_final.py` | Copy of `../Final_run/run_final.py`. Before running, it checks that every model input matches `input_gdp_corrected.csv` |
| `Tests/run_tests.py` | Copy of `../Tests/run_tests.py` pointed at `GDP_corrected/models` |
| `Baselines/` | Copy of `../Baselines` (majority, persistence, logit E0–E7, logit flow+SOX) on the corrected input |

## Run

```
cd .\GDP_corrected
python run_final.py --models LSTM CNN-LSTM GRU-LSTM BiGRU-LSTM BiLSTM DA-RNN iTransformer   # GPU  (log: run_gpu_log.txt)
python run_final.py --models RandomForest XGBoost LightGBM                                   # CPU  (log: run_cpu_log.txt)
python run_final.py --collect                       # results_all.csv, results_summary.csv
python Tests\run_tests.py                           # Tests/results/
python Baselines\run_baselines.py --compare         # Baselines/results/input_gdp_corrected/
```

Here `python` is `<your conda env>\python.exe`. Runs resume: finished runs are skipped.
