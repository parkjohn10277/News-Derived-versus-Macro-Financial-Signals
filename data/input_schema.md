# Model input panel: `input_gdp_corrected.csv`

This file is the input to every model in `code/05_models/`. It has one row per KRX trading day from 2015-02-27 to 2026-06-26 (2,781 rows) and UTF-8 encoding.

**It is not included in this repository**, because `foreign_net_buy` is derived from proprietary KRX data.

| Column | Block | Description |
|---|---|---|
| `date` | — | trading day (YYYY-MM-DD) |
| `foreign_net_buy` | KRX | foreign investors' net-buy value (KRW), summed over the 150 semiconductor firms. Its sign gives the target direction for day t+1 |
| `sp500_logret` | AUX | S&P 500 daily log return |
| `global_logret` | AUX | global equity log return |
| `sox_logret` | AUX | PHLX Semiconductor index log return |
| `krw_usd` | AUX | KRW/USD exchange rate |
| `vkospi` | AUX | VKOSPI volatility index |
| `ktb3y_minus_base` | AUX | 3-year KTB yield minus the BOK base rate |
| `cpi_yoy` | AUX | CPI, year-on-year % (release lag 5 days) |
| `gdp_yoy` | AUX | nominal GDP, year-on-year % (release lag 70 days) |
| `semi_prod_yoy` | AUX | semiconductor production index, year-on-year % (release lag 31 days) |
| `unemp_rate` | AUX | unemployment rate (release lag 16 days) |
| `global_epu` | AUX | Global EPU index (release lag 7 days) |
| `gpr_global`, `gpr_korea` | AUX | Geopolitical Risk indices (release lag 7 days) |
| `lda_en_17`, `lda_en_18`, `lda_en_19` | LDA | daily z-scored topic shares, English LDA |
| `lda_ko_04` … `lda_ko_20` (8 topics: 04, 05, 06, 07, 08, 12, 13, 20) | LDA | daily z-scored topic shares, Korean LDA |
| `mcst_sentiment`, `gdelt_sentiment`, `nyt_sentiment`, `naver_sentiment` | LLM | daily mean EXAONE sentiment score per news source |

Producing scripts:

- **LDA columns:** `code/04_lda/scripts/06_timeseries.py`
- **LLM columns:** `code/01_news/sentiment_llm/`
- **AUX columns:** `code/03_auxiliary/`
- **KRX column:** `code/02_krx/`
