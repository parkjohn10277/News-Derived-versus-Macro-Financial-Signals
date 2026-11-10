# panel_daily_ffill.csv

Daily panel, 2015-02-27 to 2026-06-26, 4138 rows, one row per calendar day.

How each series is put on the daily grid:
- Daily series (markets, FX, rates, VKOSPI): last traded value is carried over weekends and holidays. Log returns are therefore 0 on non-trading days.
- Monthly / quarterly series: each release is carried forward day by day until the next release, and up to one extra period after the final release (1 month / 1 quarter). Nothing is held flat beyond that - when a source series ends, the column becomes empty.

| column | meaning | native freq | source | data from | data to |
|---|---|---|---|---|---|
| `sp500_logret` | S&P 500 index, daily log return | daily | Yahoo Finance ^GSPC | 2015-02-28 | 2026-06-26 |
| `global_logret` | MSCI World index (URTH ETF), daily log return | daily | Yahoo Finance URTH | 2015-02-28 | 2026-06-26 |
| `sox_logret` | Philadelphia Semiconductor Index (SOX), daily log return | daily | NASDAQ/FRED + Yahoo ^SOX | 2015-02-28 | 2026-06-26 |
| `krw_usd` | Won per 1 US dollar, spot rate (level) | daily | ECOS | 2015-02-27 | 2026-06-26 |
| `krw_usd_logret` | Won/dollar daily log return (positive = won weaker) | daily | ECOS | 2015-02-28 | 2026-06-26 |
| `ktb3y_minus_base` | Korea Treasury Bond 3Y yield minus BOK base rate, %p | daily | ECOS / KOFIA | 2015-02-27 | 2026-06-26 |
| `ktb3y_minus_msb91` | KTB 3Y yield minus Monetary Stabilization Bond 91D yield, %p | daily | ECOS / KOFIA | 2015-02-27 | 2026-06-26 |
| `ktb10y_minus_ktb3y` | KTB 10Y yield minus KTB 3Y yield (yield-curve slope), %p | daily | ECOS / KOFIA | 2015-02-27 | 2026-06-26 |
| `vkospi` | VKOSPI - KOSPI 200 implied-volatility index (level), Korea's VIX | daily | KRX | 2015-02-27 | 2026-06-26 |
| `cpi_total` | Korea CPI, all items, whole country, index 2020=100 | monthly | KOSIS | 2015-02-27 | 2026-06-26 |
| `cpi_yoy` | Korea CPI inflation, % change vs 12 months earlier | monthly | KOSIS (derived) | 2015-02-27 | 2026-06-26 |
| `gdp_yoy` | Korea GDP growth, % change vs same quarter last year (nominal, SA) | quarterly | BOK/KOSIS (derived) | 2015-02-27 | 2026-06-26 |
| `gdp_qoq` | Korea GDP growth, % change vs previous quarter (nominal, SA) | quarterly | BOK/KOSIS (derived) | 2015-02-27 | 2026-06-26 |
| `semi_prod_sa` | Korea industrial production index, semiconductors & parts, seasonally adjusted | monthly | KOSIS | 2015-02-27 | 2026-06-26 |
| `semi_prod_yoy` | Semiconductor production index, % change vs 12 months earlier | monthly | KOSIS (derived) | 2015-02-27 | 2026-06-26 |
| `unemp_rate` | Korea unemployment rate, % (total, all ages/education) | monthly | KOSIS | 2015-02-27 | 2026-06-26 |
| `korea_epu` | Korea Economic Policy Uncertainty index, news-based (level) | monthly | Cho & Kim (policyuncertainty.com) | 2015-02-27 | 2025-12-31 |
| `global_epu` | Global Economic Policy Uncertainty index, GDP-weighted (level) | monthly | Davis (policyuncertainty.com) | 2015-02-27 | 2026-06-26 |
| `gpr_global` | Geopolitical Risk index, global (level, 1985-2019 avg = 100) | monthly | Caldara & Iacoviello | 2015-02-27 | 2026-06-26 |
| `gpr_korea` | Geopolitical Risk index, Korea-specific (share of articles, ~0-2 scale) | monthly | Caldara & Iacoviello | 2015-02-27 | 2026-06-26 |

Notes:
- `korea_epu` source stops at Nov 2025, so the column ends 2025-12-31 and is empty after that (not forward-filled).
- GDP is nominal (current prices). 2026 Q1 shows a large jump (+10.5% QoQ) that looks like a KOSIS data break; values are used raw.
- Interest-rate columns are spreads, not yield levels (only spreads are in the source file).
- `vkospi` rises steeply through 2026 (quarterly mean ~20 -> ~68); used raw, not yet cross-checked.
- No ESG variable: no source data available.