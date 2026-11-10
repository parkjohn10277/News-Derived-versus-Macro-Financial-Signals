# News-Derived versus Macro-Financial Signals: Explainable Forecasting of Foreign Investor Trading Direction in the Korean Semiconductor Industry

Code, public data and aggregate results for the paper submitted to *Expert Systems with Applications*.

John Park, Maximiliano Islas Solis, Seohyun Lee, Duk-Jo Kong

## Overview

The study forecasts the next-day **net-buy versus net-sell direction of foreign investors** across 150 Korean semiconductor firms. A 2³ factorial design (E0–E7) measures what each information block adds on top of the KRX flow history:

| Set | KRX | LDA | LLM | AUX | Features |
|---|:-:|:-:|:-:|:-:|:-:|
| E0 | ✓ | | | | 1 |
| E1 | ✓ | | | ✓ | 14 |
| E2 | ✓ | ✓ | | | 12 |
| E3 | ✓ | | ✓ | | 5 |
| E4 | ✓ | ✓ | ✓ | | 16 |
| E5 | ✓ | ✓ | | ✓ | 25 |
| E6 | ✓ | | ✓ | ✓ | 18 |
| E7 | ✓ | ✓ | ✓ | ✓ | 29 |

- **KRX:** aggregate foreign net-buy flow.
- **LDA:** daily topic shares from Korean and English news.
- **LLM:** daily sentiment scores from EXAONE 3.5 7.8B for each of the four news sources.
- **AUX:** release-dated macro-financial variables.

There are ten models: LSTM, BiLSTM, CNN-LSTM, GRU-LSTM, BiGRU-LSTM, DA-RNN, iTransformer, Random Forest, XGBoost and LightGBM. Each is run with five seeds (42–46). The models are compared with persistence and logistic-regression baselines, an expanding-origin robustness check (test years 2022–2024) and SHAP-based explanations.

## Repository layout

```
code/
  01_news/               news collection, preprocessing and LLM sentiment scoring (prompt in PROMPT.txt)
  02_krx/                aggregation of the 150 firm-level KRX investor-type files
  03_auxiliary/          daily macro panel and the release-date (publication-lag) correction
  04_lda/                Korean and English LDA pipeline (scripts 00–06)
  05_models/             main experiment: 10 models x E0–E7 x 5 seeds, statistical tests, baselines
  06_expanding_origin/   robustness windows W1–W3 (four_models = logit and trees, all_models = networks and the combined analysis)
  07_xai/                SHAP and attention analysis
  08_figures_tables/     scripts that build the figures and LaTeX tables in the paper
data/
  macro_public/          original public macro-financial series (see data/README.md)
  input_schema.md        column layout of the model input panel (the panel itself is not included)
results/
  main_GDP_corrected/    results_all.csv (per model x set x seed) and results_summary.csv (Table "performance")
  statistical_tests/     DeLong, block bootstrap, Diebold–Mariano, Wilcoxon and MCS tests
  baselines/             majority, persistence and logistic-regression comparisons
  expanding_origin/      results by window and the block effects
  xai/                   aggregate SHAP and attention tables (Section 5.4)
  lda/                   topic coherence and top terms
  tables/                LaTeX table bodies used in the paper
figures/                 figures as they appear in the paper
```

## Data availability

The study combines data from several providers, and not all of it can be redistributed. This repository contains only what can be shared.

| Data | In this repository | How to obtain |
|---|---|---|
| Firm-level investor-type trading records (KRX) | **No**: proprietary | [KRX Data Marketplace](https://data.krx.co.kr), under its terms of use |
| MCST NOW Platform news | **No**: provided for this research only | Ministry of Culture, Sports and Tourism, [now.go.kr](https://www.now.go.kr) |
| NAVER News, GDELT, The New York Times articles | **No**: subject to the providers' terms of service | the providers' public APIs, using `code/01_news/` |
| VKOSPI, S&P 500, PHLX SOX and other global equity indices | **No**: index licences | KRX, S&P Dow Jones Indices, Nasdaq or a market data vendor |
| CPI, GDP, semiconductor production, unemployment (KOSIS) | Yes, `data/macro_public/` | [Statistics Korea (KOSIS)](https://kosis.kr) |
| KRW/USD exchange rate, interest-rate spread (Bank of Korea) | Yes, `data/macro_public/` | [Bank of Korea ECOS](https://ecos.bok.or.kr) |
| Economic Policy Uncertainty (global, Korea) | Yes, `data/macro_public/` | [policyuncertainty.com](https://www.policyuncertainty.com) |
| Geopolitical Risk index | Yes, `data/macro_public/` | [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm) |
| Model metrics, statistical tests, XAI summaries, figures | Yes, `results/` and `figures/` | — |

Daily predictions, model checkpoints and the model input panel are **not** included, because they contain the KRX flow series. Researchers with access to the KRX data can rebuild the panel with `code/02_krx/` → `code/04_lda/` and `code/01_news/sentiment_llm/` → `code/03_auxiliary/`, then follow the column layout in `data/input_schema.md`.

## Reproducing the results

1. **Environment.** Python 3.11. Each model folder has its own `requirements.txt`. The main dependencies are PyTorch, scikit-learn, XGBoost, LightGBM, SHAP, gensim, kiwipiepy and pandas.
2. **Build the input panel.** Save it as `data/input_gdp_corrected.csv` (see `data/input_schema.md`). `code/03_auxiliary/build_input_gdp_corrected.py` applies the publication lags:

   | Variable | Lag (days) |
   |---|---|
   | CPI | 5 |
   | Unemployment | 16 |
   | Semiconductor production | 31 |
   | Nominal GDP | 70 |
   | EPU / GPR | 7 |

3. **Main experiment.** Run `code/05_models/run_final.py`, which covers 10 models × E0–E7 × seeds 42–46. Then run `Tests/run_tests.py` and `Baselines/run_baselines.py`. Further details are in `code/05_models/README.md`.
4. **Robustness.** Follow `code/06_expanding_origin/*/README.md`.
5. **XAI and figures.** Use `code/07_xai/` and `code/08_figures_tables/build_all.py`.

Scripts use paths relative to the repository root. You may need to adjust the path constants at the top of some scripts to match your folder layout.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The files in `data/macro_public/` remain under the terms of their original publishers; please cite those sources.

## Citation

If you use this code, please cite the paper (full reference to be added on publication).
