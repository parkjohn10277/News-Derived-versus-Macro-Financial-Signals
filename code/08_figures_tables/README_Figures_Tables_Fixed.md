# Figures and tables rebuilt from the GDP-corrected run

This folder rebuilds every manuscript figure and table that depends on model results or on the macro inputs. The
source is `../GDP_corrected`, the full rerun on the release-dated input with GDP lagged by 70 days. The original
folders `../XAI`, `../tables`, `../ESWA_Park_John/figs` and `../Final_run` are not modified.

## What is rebuilt

| Manuscript item | Script here | Output under `final/` |
|---|---|---|
| Table `tbl:performance` | `tables/make_tbl_performance.py` | `tables/tbl_performance.tex` |
| Tables `tbl:block_auc`, `tbl:block_dm` | `tables/make_tbl_block_tests.py` | `tables/tbl_block_tests_{auc,dm}.tex` |
| Suppl. within-model tables | `tables/make_tbl_within_model.py` | `tables/tbl_within_model_{auc,dm}.tex` |
| Fig. `auc_by_set` | `forecasting/make_fig_forecasting.py` | `Test AUC by feature set.png` (+ pdf) |
| Fig. `delta_auc` | `forecasting/make_fig_forecasting.py` | `Change in test AUC from adding each information block.png` (+ pdf) |
| Fig. `model_agreement` | `XAI/insights/make_fig_q6.py` (panel b now shows the logit and persistence benchmarks, not the ensemble) | `figs/forecasting/model_agreement.pdf` |
| XAI figures, Section 5.3 | `XAI/make_fig_{shap_summary,network,temporal,lags}.py` | `figs/xai/xai_*.pdf` |
| Flow size, global equity, volatility | `XAI/insights/make_fig_q{1,2,4}.py` | `figs/xai/{magnitude,global_equity,volatility}.pdf` |
| AUX input figure (appendix and SI) | `figs/make_fig_aux_inputs.py` | `figs/fig_aux_inputs.pdf` |

The two PNG figures in the main text had no script on disk, so `forecasting/make_fig_forecasting.py` recreates them
with the same layout, colours and statistics:
- the mean AUC over the ten models and the mean within each family;
- for each model, the change in AUC over 4 pairs × 5 seeds, with the Cauchy-combined DeLong p-value;
- for the mean, the Wilcoxon p-value across models.

## Not rebuilt

These items do not depend on the macro inputs or the model results, so they are unchanged:
- the timeline, the overview and the architecture diagrams;
- the LDA and sentiment figures and tables;
- the topic networks;
- the window-grid table, which comes from the lookback selection.

## XAI

`XAI/` is a code copy of `../XAI`. In `xai_config.py`, `MODEL_ROOT` now points to `../GDP_corrected/models`. Every
model is rebuilt from its saved checkpoint and must reproduce its saved test predictions before it is explained.
For LightGBM E7 seed 42, the maximum difference is 8e-17.

```
cd XAI
python run_all.py --exp E7 --models seq     # GPU, about 2.5 h   (log: xai_E7_seq_log.txt)
python run_all.py --exp E7 --models tree    # CPU, about 30 min  (log: xai_E7_tree_log.txt)
cd ..
python build_all.py                         # all figures and tables, then copies them into final/
```

Here `python` is `<your conda env>\python.exe`.
