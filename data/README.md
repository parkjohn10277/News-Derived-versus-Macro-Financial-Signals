# Public macro-financial data

These are the original files as downloaded from their publishers. They are included for convenience and remain under the publishers' terms of use. If you use them, cite the original sources.

| Folder | Variable(s) in the panel | Source |
|---|---|---|
| `kosis_cpi/` | `cpi_yoy` | Statistics Korea (KOSIS), Consumer Price Index (2020 = 100) |
| `kosis_gdp/` | `gdp_yoy` | KOSIS, GDP and GNI by economic activity, seasonally adjusted, current prices, quarterly |
| `kosis_semiconductor_production/` | `semi_prod_yoy` | KOSIS, Mining and Manufacturing Production Index (table DT_1F02011) |
| `kosis_unemployment/` | `unemp_rate` | KOSIS, unemployment rate |
| `bok_exchange_rate/` | `krw_usd` | Bank of Korea ECOS, KRW/USD exchange rate |
| `bok_interest_rate_spread/` | `ktb3y_minus_base` | Bank of Korea ECOS, 3-year KTB yield minus the base rate |
| `epu_global/` | `global_epu` | Baker, Bloom and Davis, Global Economic Policy Uncertainty index (via FRED/ALFRED, series GEPUCURRENT) |
| `epu_korea/` | — (reference) | Baker, Bloom and Davis, Korea Economic Policy Uncertainty index |
| `gpr/` | `gpr_global`, `gpr_korea` | Caldara and Iacoviello, Geopolitical Risk index |

## Not included

These series also enter the panel but cannot be redistributed:

- `vkospi`: Korea Exchange.
- `sp500_logret`: S&P Dow Jones Indices.
- `sox_logret`: PHLX Semiconductor index, Nasdaq.
- `global_logret`: MSCI World ETF (URTH), Nikkei 225 and TAIEX.

They are available from the index providers or a market-data vendor. `code/03_auxiliary/getSP500.py` shows how the S&P 500 series was retrieved.

## Build steps

1. `code/03_auxiliary/build_daily_panel.py` converts the monthly and quarterly series to daily frequency by forward fill.
2. `code/03_auxiliary/build_input_gdp_corrected.py` then shifts each series by its publication lag (see the main README).
