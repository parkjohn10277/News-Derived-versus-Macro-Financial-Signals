# BiLSTM — direction CLASSIFIER

Same data, split, feature groups and AUX clipping as `Desktop/BiLSTM`
(the regression variant), so the two are directly comparable on identical days.
The difference is what the model is asked to do.

```
PY=C:/Users/User/.conda/envs/duplicates/python.exe
"$PY" run.py                          # E0-E7 x seeds 42-46
"$PY" run.py --exp E0 E7 --seeds 42   # subset
"$PY" run.py --epochs 3               # smoke test
"$PY" run.py --magnitude-weighting    # weight each day by |arcsinh(flow / scale_c)|
```

## What changed vs the regression folder

| | regression (`Desktop/BiLSTM`) | classifier (this folder) |
|---|---|---|
| target | `arcsinh(y / scale_c)`, standardised | **`(y >= 0)` as 0/1** |
| loss | `SmoothL1Loss` | **`BCEWithLogitsLoss`** |
| final layer | `Linear(2*hidden, 1)` → a value | `Linear(2*hidden, 1)` → **a logit** (unchanged layer) |
| sigmoid | — | applied **only at predict time**, never in the model |
| class imbalance | — | **`pos_weight = n_neg / n_pos`** from training rows |
| y scaler / `sinh` inversion / `target_clip` | yes | **none** — a probability needs no inversion |
| headline metric | `R2_oos` | **`Brier_Skill`** |
| direction call | `sign(pred_raw)` — an arbitrary threshold | `P(net buy) >= 0.5` — tunable, calibrated |

`scale_c` is still applied to the **KRX input feature**, because the flow's own
history is a predictor and still needs its tail tamed. It just no longer touches
the target.

## Reading the metrics

| metric | chance value | meaning |
|---|---|---|
| **`Brier_Skill`** | **0.00** | fraction of the Brier score removed vs always predicting the training base rate. The classification analogue of `R2_oos`. |
| `LogLoss_Skill` | 0.00 | same idea under log loss |
| **`MCC`** | 0.00 | balanced, symmetric, imbalance-robust |
| **`AUC`** | 0.50 | threshold-free ranking; uses the probability, not the hard call |
| `Balanced_Accuracy` | 0.50 | accuracy corrected for the base rate |
| `Directional_Accuracy` | = `base_rate_test` | **always quote the base rate with it** — it is inflated by imbalance |
| `Precision` / `Recall` / `F1_Score` | — | net-buy class only; asymmetric |
| `Brier` / `LogLoss` | — | raw probabilistic losses, lower is better |

`base_rate_train`, `base_rate_test` and `pred_positive_rate` are in every
`metrics.json` so nothing has to be read without its reference point.

## Magnitude weighting

`--magnitude-weighting` (or `classification.magnitude_weighting: true`) weights
each training day by `|arcsinh(flow / scale_c)|`, normalised on the training
rows. This keeps the loss aligned with the research question while still
recognising that a 3-trillion-won day matters more than a 1-billion-won day —
information plain classification throws away. Off by default.

## Outputs

`results/bilstm/E{k}/seed_{s}/` — `metrics.json`, `predictions.csv` (includes
`prob_net_buy` and `actual_flow_krw`, so the probabilities can be re-thresholded
or backtested later), `features_used.csv`, `best_model.pt`, `x_scaler.pkl`,
epoch histories. There is no `y_scaler.pkl` — the target needs no inversion.

`results/bilstm/summary.csv` and `summary_mean.csv` are rebuilt from every
`metrics.json` on disk after each run.
