<div align="center">

# News-Derived versus Macro-Financial Signals

### Explainable Forecasting of Foreign Investor Trading Direction in the Korean Semiconductor Industry

**John Park · Maximiliano Islas Solis · Seohyun Lee · Duk-Jo Kong**

*Submitted to Expert Systems with Applications*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/models-10-success.svg)](#models)
[![Runs](https://img.shields.io/badge/runs-400%20main%20%2B%201%2C480%20robustness-informational.svg)](#robustness)

</div>

<div align="center">

<table>
  <tr>
    <td align="center" valign="middle" width="50%"><img src="assets/photo_john_park.png" width="140" alt="John Park"/></td>
    <td align="center" valign="middle" width="50%"><img src="assets/photo_maximiliano_islas_solis.png" width="140" alt="Maximiliano Islas Solis"/></td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="50%"><h3>John Park</h3><b>박준</b></td>
    <td align="center" valign="middle" width="50%"><h3>Maximiliano Islas Solis</h3><b>MAX</b></td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="50%">M.S. Course · <a href="https://impact.gist.ac.kr/team/students.php">GIST IMPACT Lab</a></td>
    <td align="center" valign="middle" width="50%">M.S. Course · <a href="https://impact.gist.ac.kr/team/students.php">GIST IMPACT Lab</a></td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="50%">Semiconductor &amp; Financial<br/>Data Forecasting<br/>AI Strategy &amp; Data Analytics</td>
    <td align="center" valign="middle" width="50%">Data Analysis &amp;<br/>ML Econometrics</td>
  </tr>
  <tr>
    <td align="center" valign="middle" width="50%"><a href="https://github.com/parkjohn10277"><b>@parkjohn10277</b></a></td>
    <td align="center" valign="middle" width="50%"><a href="https://github.com/MaximilianoIS"><b>@MaximilianoIS</b></a></td>
  </tr>
</table>

</div>

---

This repository holds the code, public data and aggregate results of the paper. The paper forecasts **whether foreign investors are net buyers or net sellers** of Korean semiconductor stocks on the next trading day. It also asks which information drives the forecast: **macro-financial variables**, **LDA news topics**, or **LLM news sentiment**.

<p align="center">
  <img src="figures/overview.png" width="100%" alt="Research framework">
</p>

## Key findings

| | |
|---|---|
| **Macro-financial variables carry the signal** | Adding them raises the mean test AUC from **0.647 to 0.693**, and 9 of 10 models improve (Wilcoxon *p* = 0.004). |
| **News does not add to them** | LDA topics lower the AUC (−1.16 points, 1 of 10 models improve). LLM sentiment has no significant effect (−0.22 points, *p* = 0.49). |
| **Best model** | LightGBM on KRX + macro (E1), **AUC 0.718**. |
| **Simple baselines matter** | Without the macro block, no model beats a persistence forecast. No model beats a logistic regression on the same inputs. |
| **The result holds over time** | In an expanding-origin evaluation (test years 2022, 2023 and 2024), the macro block raises the AUC in every year. |

<p align="center">
  <img src="figures/Test%20AUC%20by%20feature%20set.png" width="85%" alt="Test AUC by feature set">
</p>

## Experimental design

The design is a 2³ factorial over three information blocks, each added to the history of foreign net buying (KRX):

| Set | KRX | LDA | LLM | AUX | Features | Mean AUC (10 models) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| E0 | ✓ | | | | 1 | 0.647 |
| E1 | ✓ | | | ✓ | 14 | **0.693** |
| E2 | ✓ | ✓ | | | 12 | 0.635 |
| E3 | ✓ | | ✓ | | 5 | 0.645 |
| E4 | ✓ | ✓ | ✓ | | 16 | 0.634 |
| E5 | ✓ | ✓ | | ✓ | 25 | 0.684 |
| E6 | ✓ | | ✓ | ✓ | 18 | **0.693** |
| E7 | ✓ | ✓ | ✓ | ✓ | 29 | 0.678 |

- **KRX:** daily foreign net-buy value summed over the semiconductor firms.
- **LDA:** daily topic shares from Korean and English news.
- **LLM:** daily sentiment from EXAONE 3.5 7.8B for four sources: MCST NOW, NAVER News, GDELT and NYT.
- **AUX:** 13 release-dated macro-financial variables, listed below.

### Auxiliary variables

The 13 variables of the AUX block, as listed in the paper. The last column shows whether the original series is in [`data/macro_public/`](data/macro_public).

| No. | Group | Variable | Freq. | In this repo |
|:-:|---|---|:-:|:-:|
| 1 | Global equity markets | S&P 500 return (log) | D | ✗ licence |
| 2 | Global equity markets | Global stock market return (log) | D | ✗ licence |
| 3 | Global equity markets | Philadelphia Semiconductor Index return (log) | D | ✗ licence |
| 4 | Rates & FX | KRW/USD exchange rate | D | ✓ |
| 5 | Rates & FX | 3-year KTB yield minus policy rate | D | ✓ |
| 6 | Macro fundamentals | CPI inflation (y/y) | M | ✓ |
| 7 | Macro fundamentals | Nominal GDP growth (y/y) | Q | ✓ |
| 8 | Macro fundamentals | Unemployment rate | M | ✓ |
| 9 | Macro fundamentals | Semiconductor production growth (y/y) | M | ✓ |
| 10 | Risk & uncertainty | VKOSPI (implied volatility) | D | ✗ licence |
| 11 | Risk & uncertainty | Global economic policy uncertainty | M | ✓ |
| 12 | Risk & uncertainty | Geopolitical risk, global | M | ✓ |
| 13 | Risk & uncertainty | Geopolitical risk, Korea | M | ✓ |

D = daily, M = monthly, Q = quarterly.

### Models

| Family | Models |
|---|---|
| Recurrent | LSTM, BiLSTM, GRU-LSTM, BiGRU-LSTM, CNN-LSTM |
| Attention | DA-RNN, iTransformer |
| Tree ensembles | Random Forest, XGBoost, LightGBM |
| Baselines | majority class, persistence, logistic regression |

Every model is run with five seeds (42–46): 10 models × 8 feature sets × 5 seeds = 400 runs. The runs are compared with DeLong tests, block bootstrap, Diebold–Mariano tests, Wilcoxon tests across models, and the Model Confidence Set.

### Robustness

- **Release-dated macro inputs.** Each variable becomes available only after its publication lag:

  | Variable | Lag (days) |
  |---|---|
  | CPI | 5 |
  | Unemployment | 16 |
  | Semiconductor production | 31 |
  | Nominal GDP | 70 |
  | EPU / GPR | 7 |

- **Expanding-origin evaluation** with test years 2022, 2023 and 2024, for all ten models plus the logit (1,480 additional runs).
- **Explanations.** SHAP and attention analysis by information group, lag and period.

## Repository structure

```
├── code/
│   ├── 01_news/               news collection, preprocessing, LLM sentiment (prompt: PROMPT.txt)
│   ├── 02_krx/                aggregation of the firm-level KRX investor-type records
│   ├── 03_auxiliary/          daily macro panel and the publication-lag correction
│   ├── 04_lda/                Korean and English LDA pipeline (scripts 00–06)
│   ├── 05_models/             main experiment, statistical tests, baselines
│   ├── 06_expanding_origin/   robustness windows W1–W3
│   ├── 07_xai/                SHAP and attention analysis
│   └── 08_figures_tables/     scripts for every figure and table in the paper
├── data/
│   ├── macro_public/          original public macro series (KOSIS, Bank of Korea, EPU, GPR)
│   ├── krx_firm_list.csv      the 150 firms (stock code, Korean and English name; Supplementary Table S1)
│   ├── KRX_DOWNLOAD_GUIDE.md  how to download and rebuild the KRX flow series
│   └── input_schema.md        column layout of the model input panel
├── results/                   all metrics and tests behind the tables (CSV / TXT / LaTeX)
└── figures/                   figures as they appear in the paper
```

## Data availability

Not all of the data can be redistributed. This repository contains only what can be shared, and explains how to obtain the rest.

| Data | Included | How to obtain |
|---|:-:|---|
| Firm-level investor-type trading records (KRX) | ✗ | KRX's terms prohibit redistribution. Download free from the [KRX Data Marketplace](https://data.krx.co.kr) using [`data/KRX_DOWNLOAD_GUIDE.md`](data/KRX_DOWNLOAD_GUIDE.md) and [`data/krx_firm_list.csv`](data/krx_firm_list.csv) |
| MCST NOW Platform news | ✗ | Provided for this research by the Ministry of Culture, Sports and Tourism ([now.go.kr](https://www.now.go.kr)) |
| NAVER News, GDELT, NYT articles | ✗ | The providers' public APIs, using [`code/01_news/`](code/01_news) |
| VKOSPI, S&P 500, PHLX SOX, global indices | ✗ | Index licences. Available from KRX, S&P Dow Jones Indices, Nasdaq or a market-data vendor |
| CPI, GDP, semiconductor production, unemployment | ✓ | [KOSIS](https://kosis.kr) |
| KRW/USD exchange rate, interest-rate spread | ✓ | [Bank of Korea ECOS](https://ecos.bok.or.kr) |
| Economic Policy Uncertainty | ✓ | [policyuncertainty.com](https://www.policyuncertainty.com) |
| Geopolitical Risk index | ✓ | [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm) |
| Metrics, tests, XAI summaries, figures | ✓ | [`results/`](results), [`figures/`](figures) |

Daily predictions, model checkpoints and the model input panel are not included, because they contain the KRX flow series.

## Reproducing the results

1. **Build the KRX target.** Follow [`data/KRX_DOWNLOAD_GUIDE.md`](data/KRX_DOWNLOAD_GUIDE.md), then run `python code/02_krx/combine_krx_150.py`.
2. **Build the text features.** Run the LDA pipeline (`code/04_lda/scripts/run_all.py`) and the LLM sentiment scripts (`code/01_news/sentiment_llm/`).
3. **Build the macro panel and the input file.** Run `code/03_auxiliary/build_daily_panel.py` and then `build_input_gdp_corrected.py`. The result is `data/input_gdp_corrected.csv`; see [`data/input_schema.md`](data/input_schema.md).
4. **Main experiment.** Run `code/05_models/run_final.py`, then `Tests/run_tests.py` and `Baselines/run_baselines.py`.
5. **Robustness, XAI and figures.** Use `code/06_expanding_origin/`, `code/07_xai/` and `code/08_figures_tables/build_all.py`.

Each model folder has its own `requirements.txt`. The main dependencies are PyTorch, scikit-learn, XGBoost, LightGBM, SHAP, gensim, kiwipiepy and pandas. Scripts use paths relative to the repository root; some have path constants at the top that you may need to adjust.

## Citation

If you use this code, please cite the paper (full reference to be added on publication). See also [`CITATION.cff`](CITATION.cff).

```bibtex
@article{park_foreign_flow,
  title   = {News-Derived versus Macro-Financial Signals: Explainable Forecasting of
             Foreign Investor Trading Direction in the Korean Semiconductor Industry},
  author  = {Park, John and Islas Solis, Maximiliano and Lee, Seohyun and Kong, Duk-Jo},
  journal = {Expert Systems with Applications},
  note    = {Under review}
}
```

## Acknowledgements

<p align="center">
  <a href="https://impact.gist.ac.kr/main/main.php"><img src="assets/impact_lab_logo.png" width="180" alt="IMPACT Lab"></a>
</p>

<p align="center">
  <a href="https://impact.gist.ac.kr/team/professor.php"><img src="assets/prof_duk-jo_kong.png" width="150" alt="Prof. Duk-Jo Kong"></a>
</p>

<p align="center">
  <b>Prof. Duk-Jo Kong</b> (공득조)<br>
  Vice Dean, Assistant Professor<br>
  GIST AI Policy &amp; Strategy Graduate School (AIX School)
</p>

This work was carried out at the [**IMPACT Lab**](https://impact.gist.ac.kr/main/main.php) led by Prof. Duk-Jo Kong at the Gwangju Institute of Science and Technology (GIST). We thank the members of the lab for their discussions and support throughout the project.

We are especially grateful to [**Prof. Seohyun Lee**](https://www.kdischool.ac.kr/faculty/view.es?mid=a10401010000&type=1&page=2&emp_no=311000339) of the KDI School of Public Policy and Management for her guidance on the macro-financial side of this work, including economic uncertainty and geopolitical risk.

We also thank the Ministry of Culture, Sports and Tourism of the Republic of Korea for providing the MCST NOW Platform news data.

## Licence

The code is released under the [MIT Licence](LICENSE). The files in `data/macro_public/` remain under the terms of their original publishers.
