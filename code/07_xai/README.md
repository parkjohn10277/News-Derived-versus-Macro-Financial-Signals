# XAI — information contribution analysis (direction classifiers only)

Explains the **saved E7 classifiers** of all 10 models. Nothing is retrained.
Every model is rebuilt from its own checkpoint and must reproduce its saved test
predictions before anything is explained.

| family | models | explainer for the lag-level panel |
|---|---|---|
| Hybrid LSTM | LSTM, CNN-LSTM, GRU-LSTM, BiGRU-LSTM (Desktop/Jun), BiLSTM, DA-RNN | `shap.GradientExplainer` |
| Transformer | iTransformer | `shap.GradientExplainer` (gradient-correct wrapper, see below) |
| Tree ensemble | Random Forest, XGBoost, LightGBM (TreeModels_Classifier) | `shap.TreeExplainer` |

Information groups (rows of every figure; edit in `xai_config.py`):

| group | columns |
|---|---|
| Domestic news | Korean LDA topics (`lda_ko_*`), NAVER sentiment |
| Foreign news | English LDA topics (`lda_en_*`), GDELT, NYT and MCST sentiment |
| Market | foreign net-buy history (KRX), VKOSPI, KRW/USD, S&P 500, global and SOX returns |
| Macro | rate spreads, CPI, GDP, semiconductor production, unemployment, EPU, GPR |

## Run order

Use the project environment: `<your conda env>\python.exe`

```
cd .\XAI
python xai_core.py                  # self-test of the Shapley math (seconds, no models)
python run_xai.py --list            # every model should say "found"
```

Optional quick trial (about a minute), to confirm one model loads end to end:
```
python run_xai.py --model lstm --max-days 200 --steps group
```

Then one model per call (each writes to `outputs/<model>/E7/`):
```
python run_xai.py --model lstm
python run_xai.py --model cnn_lstm
python run_xai.py --model gru_lstm
python run_xai.py --model bigru_lstm
python run_xai.py --model bilstm
python run_xai.py --model darnn
python run_xai.py --model itransformer
python run_xai.py --model random_forest
python run_xai.py --model xgboost
python run_xai.py --model lightgbm
python make_xai_figures.py          # figures/E7/*.pdf + figures/E7/tables/*.csv
```
`make_xai_figures.py` uses whichever models are finished, so you can run it after
any subset. Rerunning a model overwrites its outputs.

Rough runtimes on your GPU: the `group` step takes 1–3 min per model. The `native`
step takes 5–15 min per LSTM-type model (DA-RNN is the slowest) and 10–20 min for
XGBoost (1000 trees). To go faster: `--nsamples 50` for sequence models,
`--tree-bg 50` for trees.

## What the console checks mean

| line | pass condition | if it fails |
|---|---|---|
| `[verify] max \|rebuilt - saved\|` | ≤ 1e-4 and all test days matched | stops: the data or checkpoint changed since training |
| `[group] efficiency check` | ~1e-15 | should never fail (exact math) |
| `[native] completeness corr` | > 0.9 is good | expected gradients is a sampling approximation; raise `--nsamples` |
| `[native] agreement ... Spearman` | positive, ideally > 0.5 | the two methods disagree about that group |
| `[regime] out-of-sample split` | ≥ 30 low and ≥ 30 high test days | the regime performance game is skipped for that regime |

All of these are also collected in `figures/E7/tables/xai_qc_table.csv`.

## The three panels

### 1 · Temporal importance: `xai_1a`, `xai_1b`, `xai_s1`

Each day, P(net buy) is decomposed **exactly** into the four group Shapley
values. Cells show each group's share of total |SHAP| in that year: the
equal-weight mean over models in 1a, per family in 1b, per model in S1.

- The dashed line marks the test start. Left of it the model is explaining days
  it was fitted on (in-sample); right of it is out-of-sample.
- Say this in the caption.

### 2 · Interactions and incremental predictive power: `xai_2a`, `xai_2b`

This is a performance game on the **test days**: v(S) = AUC − 0.5 with only the
groups in S informative.

- **Diagonal / node size:** each group's Shapley contribution to out-of-sample AUC.
  The four sum to the model's AUC − 0.5.
- **Off-diagonal / edges:** the Shapley interaction index.
  - **< 0 (red):** overlapping information. The two groups add less together than
    separately.
  - **≈ 0 (grey dotted):** independent information.
  - **> 0 (blue):** complementary information.
- `*` marks a cross-model mean more than 1.96 SE away from 0.
- Per-model bootstrap CIs are in `outputs/<model>/E7/perf_decomposition.csv`.

This panel answers H1 directly: the news groups' contribution *given* market and
macro information, and whether domestic and foreign news overlap.

### 3 · Volatility regimes: `xai_3a–3d`

Regime is set by VKOSPI on the previous trading day. Low is the bottom third and
high is the top third, with thresholds taken from training days only.

- **3a/3b:** native SHAP by group × days-before-forecast, low | high | high − low.
  Shows *where in the lookback window* each group matters, and how that shifts
  when volatility is high.
- **3c:** share of |group SHAP| in low vs high regimes. This covers all days, with
  ±95% intervals across models.
- **3d:** each group's contribution to **out-of-sample** AUC within each regime.
  This is the direct test of H2 ("news signals matter more in volatile periods").

**Important — the whole test period is a high-volatility episode.** Test-period
VKOSPI has a median of 31.6 (training median 16.3), and its *minimum* (17.9) sits
at the training 67th percentile. Under training thresholds, 275 of the 278 test
days are "high" and none are "low". Panel 3d therefore splits the **test days by
their own terciles**: relatively calm (VKOSPI ≤ ~24.7) vs relatively turbulent
(≥ ~42.0), with 93 days each. This is a descriptive stratification that never
feeds the model. State it in the caption, and note in the paper that all
out-of-sample results come from an unusually volatile year.

### Supplementary

- **`xai_s2`:** top-20 individual features, from native SHAP.
- **`xai_s3`:** native attention vs SHAP for DA-RNN and iTransformer. The attention
  weights are near-uniform (entropy ≈ maximum), which is why the paper uses SHAP
  rather than attention to attribute importance.

## Methods paragraph (draft)

> Feature attributions are computed post hoc on each trained classifier; no model is
> re-estimated. Input variables are partitioned into four information groups
> (domestic news, foreign news, market, macro). Because there are only four groups,
> group Shapley values and pairwise Shapley interaction indices (Grabisch & Roubens,
> 1999) are computed exactly by enumerating all 2⁴ coalitions under an
> interventional value function, with absent groups replaced by randomly drawn
> training windows (50 per day). Local attributions decompose each day's predicted
> probability of net buying. Global attributions decompose out-of-sample
> discrimination, v(S) = AUC(S) − 0.5 (cf. SAGE; Covert et al., 2020), with
> circular block-bootstrap confidence intervals (block length 10 trading days).
> Within-window attributions use SHAP (Lundberg & Lee, 2017): TreeSHAP with an
> interventional background for tree ensembles, and expected gradients
> (GradientExplainer) for recurrent and Transformer models. Volatility regimes are
> defined by terciles of the previous day's VKOSPI, with thresholds taken from the
> training period.

## Implementation notes (why the code looks the way it does)

- **One model per process.** Every template/tree folder has its own `src` package,
  so mixing them in one Python process would import the wrong code.
- **iTransformer** detaches the window mean inside `forward()`. Values are
  unchanged, but gradient attributions would silently be wrong.
  `ITransformerGradCore` recomputes the same forward pass without the detach and is
  checked for numerical identity at load time.
- **cuDNN is disabled** during the gradient step. It refuses LSTM/GRU backward
  passes in eval mode.
- **DA-RNN's decoder input** (the continuous flow history, `y_hist`) belongs to the
  Market group. It is masked together with the KRX column.
- **Jun models** are rebuilt with the exact constructor call read from their own
  script (`final_model = …`), so the architecture always matches the checkpoint.
- **Background data is the one soft assumption** behind every Shapley value here.
  Absent groups are replaced by *independent* draws from training days, which
  ignores correlation between groups. This is the standard interventional-SHAP
  choice, but worth one sentence in the limitations.
