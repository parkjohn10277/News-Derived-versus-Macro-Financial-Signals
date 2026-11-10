# Tree models — same experiment as Att_LSTM / LSTM / BiLSTM

One folder = one model (`RandomForest`, `XGBoost`, `LightGBM`). Data, feature
groups, E0–E7 grid, split dates, target transform, prediction clipping, metric
set and output layout are identical to the sequence folders, so
`results/<model>/summary_mean.csv` is directly comparable across all six models.

Method reference: **Chun, D., Kang, J. & Kim, J. (2024), _Forecasting returns
with machine learning and optimizing global portfolios: evidence from the
Korean and U.S. stock markets_, Financial Innovation 10:124.**

```
PY=C:/Users/User/.conda/envs/duplicates/python.exe
"$PY" run.py                              # E0-E7 x seeds 42-46 from config.yaml
"$PY" run.py --exp E0 E7 --seeds 42       # subset
"$PY" run.py --n-iter 3                   # smoke test
"$PY" report_importance.py                # ablation importance + marginal association + SHAP
```

## What follows the paper, and what does not

| Element | Chun et al. (2024) | Here |
|---|---|---|
| Models | RF, XGBoost, LightGBM (+ linear, MLP) | the three tree models |
| Predictor transform | Yeo–Johnson power transform | same, fit on training rows only |
| Hyperparameters | Table 8 grids, randomised CV | same grids (see `src/models/*.py`) |
| Window | 1250 days = 1000 train + 250 validation, re-estimated yearly | our shared split (below) — kept for cross-model comparability |
| Variable importance | drop in out-of-sample R² when a predictor is set to zero | same (`report_importance.py`) |
| Importance normalisation | appendix: clip negatives, scale to sum 1 | same (`normalized` column) |
| Marginal association | predictors scaled to [−1,1], one varied, others at median | same |
| Target | daily returns (KRW/USD, S&P 500, KOSPI) | foreign net buy, arcsinh + standardised |

Deviations are deliberate and documented in the source: `max_features` is a
fraction (their 10–100 of 137 predictors ≈ 7–73 %); `n_estimators` is searched
for RF and XGB (Table 8 omits it); `num_leaves` is searched for LightGBM;
`subsample` / `colsample_bytree` are added for XGB on a short sample.

## Featurisation

Trees need a flat design matrix. Each input column is expanded into lags
`[0, 1, 2, 5, 10, 20]` — lag 0 is the most recent day, the set spans the same
30 days the recurrent models see:

```
predict y_{t+1}   from   { <col>_lag<k> }        E0 → 6 features, E7 → 210
```

Set `features.lags: [0]` in `config.yaml` for strict Chun et al. replication
(current day only).

## Protocol

* **Phase 1** — fit on 2015-02-28 … 2024-12-31, score on 2025 →
  `RandomizedSearchCV` picks the hyperparameters. This is the exact slot where
  the recurrent folders pick `best_epoch`, on the same block.
* **Phase 2** — refit on the official train 2015 … 2025 with those
  hyperparameters → predict the test block 2026-01-01 … 2026-06-26 (118 days).
* Transforms fit on the training rows of each phase only; AUX z-clip ±3;
  predictions clipped to the official-train target range; seeds 42–46.

## Metrics

Regression `RMSE, MAE, MAPE` · classification (direction = sign)
`Directional_Accuracy, F1_Score, AUC, MCC, R2`, plus `R2_oos_vs_train_mean` and
Diebold–Mariano against the historical mean.

Note: raw-KRW `R2` and `MAPE` sit at the zero-forecast value for every model in
every folder — an artefact of inverting the arcsinh, not a model difference.
Compare on `MCC`, `AUC` and `F1_Score`. For the same reason
`report_importance.py` computes its ablation R² on the **modelled** target
(arcsinh, standardised), not in raw KRW where the measure degenerates to zero.

## Importance outputs (`figures/<model>/`)

| File | Contents |
|---|---|
| `ablation_feature_<E>.csv` | every design feature `<col>_lag<k>` |
| `ablation_column_<E>.csv` | all lags of an input column removed together |
| `ablation_group_<E>.csv` | information blocks: KRX / LDA / LLM / AUX, LDA EN vs KO, news foreign (GDELT+NYT) vs domestic (NAVER) vs MCST |
| `ablation_group_heatmap.png` | blocks × experiments, the H1 table |
| `marginal_association_<E>.*` | paper Figs. 4/6/8 style curves |
| `shap_<E>.*` | secondary SHAP summary (tree explainer) |

The **group ablation is the H1 instrument**: how much out-of-sample R² is lost
when foreign-language news is removed versus when domestic news is removed.

## Layout

```
data/input.csv          final model input (date + 35 columns)
config.yaml             split · seeds · model name · lags · search budget
run.py                  E0-E7 grid runner      -> results/<model>/
report_importance.py    Chun et al. ablation   -> figures/<model>/
src/
  features.py           groups + E0-E7 definitions
  tabular.py            lag design matrix, Yeo-Johnson / arcsinh transforms
  protocol.py           two-phase split, search, refit, metrics
  metrics.py, dm_test.py
  models/__init__.py    registry
  models/random_forest.py, xgboost_model.py, lightgbm_model.py
```

All three model specs ship in every folder; `config.yaml` selects which one
runs. To add a model, drop a spec exposing `NAME`, `GRID` and
`estimator(seed)` into `src/models/` and register it.
