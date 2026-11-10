"""
Build a daily-frequency panel of macro / market auxiliary variables for the
Korean semiconductor industry.

Sources live in ./1_Original/. Output goes to
./2_DataAugmented_DailyFrequency/panel_daily_ffill.csv

Method: forward fill, nothing else.
* Native daily series (equities, FX, VKOSPI, rate spreads) are put on a full
  calendar-day grid and forward-filled over weekends / holidays.
* Monthly / quarterly series are stamped at their PERIOD END and forward-filled
  to daily: each release is carried until the next release, plus at most one
  extra period after the final release. A source series that ends is NOT held
  flat - the column just goes NaN (e.g. Korea EPU after Nov 2025).
* Transforms:
    equities, FX      -> daily log returns
    GDP               -> YoY % and QoQ % growth
    CPI               -> level + YoY % inflation
    semi production   -> level + YoY %
    everything else   -> level as-is
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "1_Original"
OUT = ROOT / "2_DataAugmented_DailyFrequency"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_START = "2015-02-27"   # study window start (VKOSPI coverage)
PANEL_END = "2026-06-26"     # study window end ("" = use latest available)

# Final columns to keep, in order.
KEEP = [
    # --- markets / daily ---
    "sp500_logret",          # S&P 500 daily log return
    "global_logret",         # MSCI World daily log return (global stock market)
    "sox_logret",            # PHLX/SOX semiconductor index log return
    "krw_usd",               # KRW per USD (level = the exchange-rate variable)
    "krw_usd_logret",        # KRW/USD daily log return
    "ktb3y_minus_base",      # rate spreads (proxy for KTB yields)
    "ktb3y_minus_msb91",
    "ktb10y_minus_ktb3y",
    "vkospi",                # VKOSPI level
    # --- macro (forward-filled to daily) ---
    "cpi_total",             # CPI level
    "cpi_yoy",               # CPI YoY inflation
    "gdp_yoy",               # total Korea GDP YoY growth
    "gdp_qoq",               # total Korea GDP QoQ growth
    "semi_prod_sa",          # semiconductor industrial production index (SA)
    "semi_prod_yoy",         # semiconductor production YoY
    "unemp_rate",            # unemployment rate
    "korea_epu",             # Korea EPU
    "global_epu",            # Global EPU
    "gpr_global",            # Geopolitical Risk (global)
    "gpr_korea",             # Geopolitical Risk (Korea)
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def month_end_index(start_ym: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_ym, periods=n, freq="ME")


def quarter_end_from_label(label: str) -> pd.Timestamp:
    # "2015.1/4" -> 2015Q1 -> 2015-03-31
    y, q = label.split(".")
    q = int(q.split("/")[0])
    return pd.Timestamp(year=int(y), month=q * 3, day=1) + pd.offsets.MonthEnd(0)


def ffill_to_daily(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Stamp a period-END-indexed series onto the daily grid and forward-fill.

    Each release is carried forward day by day until the next release, and for
    at most ONE more period after the final release (1 month for monthly data,
    1 quarter for quarterly). Nothing is filled beyond that - if the source
    series stops, the panel column goes NaN, it is not held flat.
    """
    s = s.dropna().sort_index()
    if s.empty:
        return pd.Series(index=grid, dtype="float64")
    step_days = np.median(np.diff(s.index.values).astype("timedelta64[D]").astype(int))
    horizon = pd.Timedelta(days=31 if step_days < 45 else 92)
    out = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    out[out.index > s.index.max() + horizon] = np.nan
    return out


def yoy(s: pd.Series, periods: int) -> pd.Series:
    return 100.0 * (s / s.shift(periods) - 1.0)


def pct(s: pd.Series, periods: int = 1) -> pd.Series:
    return 100.0 * (s / s.shift(periods) - 1.0)


# ----------------------------------------------------------------------------
# Loaders -> dict of {col_name: Series indexed by period END}
# ----------------------------------------------------------------------------

def load_cpi() -> dict:
    f = next((SRC / "CPI").glob("*.csv"))
    df = pd.read_csv(f, header=0)
    row = df[df.iloc[:, 0].astype(str).str.strip() == "whole country"]
    vals = pd.to_numeric(row.iloc[0, 1:], errors="coerce").to_numpy()
    idx = month_end_index("1965-01-31", len(vals))
    lvl = pd.Series(vals, index=idx, name="cpi_total")
    return {"cpi_total": lvl, "cpi_yoy": yoy(lvl, 12).rename("cpi_yoy")}


def load_gdp() -> dict:
    f = next((SRC / "GDP_GNI ").glob("*.csv"))
    df = pd.read_csv(f)
    qcols = list(df.columns[1:])
    qidx = pd.DatetimeIndex([quarter_end_from_label(c) for c in qcols])
    r = df[df.iloc[:, 0] == "Gross domestic product at market prices(GDP)"]
    vals = pd.to_numeric(r.iloc[0, 1:], errors="coerce").to_numpy(dtype="float64")
    s = pd.Series(vals, index=qidx, name="gdp")
    return {
        "gdp_yoy": yoy(s, 4).rename("gdp_yoy"),
        "gdp_qoq": pct(s, 1).rename("gdp_qoq"),
    }


def load_production_semi() -> dict:
    f = next((SRC / "Production_Growth_Index").glob("*.csv"))
    df = pd.read_csv(f, encoding="cp949")
    m = df[(df["기타분류별"] == "반도체 및 부품")
           & (df["항목"] == "산업생산지수(계절조정)")]
    vals = pd.to_numeric(m.iloc[0, 3:], errors="coerce").to_numpy()
    idx = month_end_index("2005-01-31", len(vals))
    lvl = pd.Series(vals, index=idx, name="semi_prod_sa")
    return {
        "semi_prod_sa": lvl,
        "semi_prod_yoy": yoy(lvl, 12).rename("semi_prod_yoy"),
    }


def load_unemployment() -> dict:
    f = next((SRC / "Unemployment_Rate").glob("*.xlsx"))
    df = pd.read_excel(f, sheet_name="데이터", header=0)
    r = df[(df.iloc[:, 0].astype(str).str.strip() == "계")
           & (df.iloc[:, 1].astype(str).str.strip() == "계")]
    vals = pd.to_numeric(r.iloc[0, 2:], errors="coerce").to_numpy()
    idx = month_end_index("2015-01-31", len(vals))
    return {"unemp_rate": pd.Series(vals, index=idx, name="unemp_rate")}


def load_korea_epu() -> dict:
    f = next((SRC / "Korean_EPU").glob("*.xlsx"))
    df = pd.read_excel(f, sheet_name=0)
    df = df.iloc[:, :2].dropna()
    df.columns = ["date", "v"]
    s = pd.Series(pd.to_numeric(df["v"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0),
                  name="korea_epu")
    return {"korea_epu": s}


def load_global_epu() -> dict:
    f = next((SRC / "Global_EPU").glob("*.csv"))
    df = pd.read_csv(f)
    val_col = [c for c in df.columns if c != "observation_date"][-1]  # latest vintage
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["observation_date"]) + pd.offsets.MonthEnd(0),
                  name="global_epu")
    return {"global_epu": s}


def load_gpr() -> dict:
    f = next((SRC / "Geopolitical_Risk_Index").glob("*.xls"))
    df = pd.read_excel(f, sheet_name="Sheet1")
    df = df[pd.to_datetime(df["month"], errors="coerce").notna()].copy()
    idx = pd.to_datetime(df["month"]) + pd.offsets.MonthEnd(0)
    out = {}
    for key, col in [("gpr_global", "GPR"), ("gpr_korea", "GPRC_KOR")]:
        out[key] = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(),
                             index=idx, name=key)
    return out


# --- native daily -----------------------------------------------------------

def load_sp500() -> pd.DataFrame:
    f = next((SRC / "SP500_Returns").glob("*.csv"))
    df = pd.read_csv(f)
    s = pd.Series(pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["Date"]))
    return s.to_frame("sp500_close")


def load_global_returns() -> pd.DataFrame:
    d = SRC / "Global_Returns"
    out = {}

    sox = pd.read_csv(d / "NASDAQSOX.csv")
    out["sox_close"] = pd.Series(
        pd.to_numeric(sox["NASDAQSOX"], errors="coerce").to_numpy(),
        index=pd.to_datetime(sox["observation_date"]),
    )

    gf = pd.read_csv(d / "global_semi_factors.csv")
    gf.index = pd.to_datetime(gf["Date"])
    ren = {"URTH": "world_close", "^N225": "nikkei_close",
           "^SOX": "sox_yf_close", "^TWII": "taiex_close"}
    for src, dst in ren.items():
        if src in gf.columns:
            out[dst] = pd.to_numeric(gf[src], errors="coerce")

    df = pd.DataFrame(out).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_fx() -> pd.DataFrame:
    f = next((SRC / "Exchange_Rate").glob("*.xlsx"))
    raw = pd.read_excel(f, sheet_name=0, header=None)
    dt = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    m = dt.notna()
    df = pd.DataFrame({
        "krw_usd": pd.to_numeric(raw.iloc[:, 1][m], errors="coerce").to_numpy(),
        "krw_neer": pd.to_numeric(raw.iloc[:, 2][m], errors="coerce").to_numpy(),
    }, index=dt[m])
    return df.sort_index()


def load_interest_spreads() -> pd.DataFrame:
    f = next((SRC / "Interest_Rates").glob("*.xlsx"))
    raw = pd.read_excel(f, sheet_name=0, header=None)
    dt = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    m = dt.notna()
    df = pd.DataFrame({
        "ktb3y_minus_base": pd.to_numeric(raw.iloc[:, 1][m], errors="coerce").to_numpy(),
        "ktb3y_minus_msb91": pd.to_numeric(raw.iloc[:, 2][m], errors="coerce").to_numpy(),
        "ktb10y_minus_ktb3y": pd.to_numeric(raw.iloc[:, 3][m], errors="coerce").to_numpy(),
    }, index=dt[m])
    return df.sort_index()


def load_vkospi() -> pd.DataFrame:
    frames = []
    for fp in sorted(glob.glob(str(SRC / "VKOSPI" / "*.csv"))):
        d = pd.read_csv(fp, encoding="cp949")
        frames.append(pd.DataFrame({
            "vkospi": pd.to_numeric(d["종가"], errors="coerce").to_numpy(),
        }, index=pd.to_datetime(d["일자"], format="%Y/%m/%d")))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def main() -> None:
    sp = load_sp500()
    gr = load_global_returns()
    fx = load_fx()
    ir = load_interest_spreads()
    vk = load_vkospi()

    dmin = pd.Timestamp(PANEL_START)
    dmax = max(sp.index.max(), gr.index.max(), fx.index.max(),
               ir.index.max(), vk.index.max())
    if PANEL_END:
        dmax = min(dmax, pd.Timestamp(PANEL_END))
    grid = pd.date_range(start=dmin, end=dmax, freq="D")

    daily = pd.DataFrame(index=grid)
    daily.index.name = "date"

    # native daily levels -> reindex onto grid + ffill over non-trading days
    for src in (sp, gr, fx, ir, vk):
        for c in src.columns:
            daily[c] = src[c].reindex(src.index.union(grid)).ffill().reindex(grid)

    # log returns for equities + FX (on the ffilled daily grid)
    daily["global_close"] = daily.get("world_close")  # MSCI World = "global"
    for c in ["sp500_close", "sox_close", "global_close", "krw_usd"]:
        if c in daily:
            daily[f"{c.replace('_close', '')}_logret"] = np.log(daily[c]).diff()

    # monthly / quarterly series -> ffill to daily
    periodic: dict[str, pd.Series] = {}
    for loader in (load_cpi, load_gdp, load_production_semi, load_unemployment,
                   load_korea_epu, load_global_epu, load_gpr):
        periodic.update(loader())

    for name, s in periodic.items():
        daily[name] = ffill_to_daily(s, grid).to_numpy()

    missing = [c for c in KEEP if c not in daily.columns]
    if missing:
        raise RuntimeError(f"expected columns not produced: {missing}")

    out = daily[KEEP]
    path = OUT / "panel_daily_ffill.csv"
    out.to_csv(path)
    print(f"wrote {path}  shape={out.shape}  "
          f"range={out.index.min().date()}..{out.index.max().date()}")
    print(out.notna().sum().to_string())

    _write_dictionary(out)


# column -> (plain description, native frequency, source)
VAR_INFO = {
    "sp500_logret":       ("S&P 500 index, daily log return", "daily", "Yahoo Finance ^GSPC"),
    "global_logret":      ("MSCI World index (URTH ETF), daily log return", "daily", "Yahoo Finance URTH"),
    "sox_logret":         ("Philadelphia Semiconductor Index (SOX), daily log return", "daily", "NASDAQ/FRED + Yahoo ^SOX"),
    "krw_usd":            ("Won per 1 US dollar, spot rate (level)", "daily", "ECOS"),
    "krw_usd_logret":     ("Won/dollar daily log return (positive = won weaker)", "daily", "ECOS"),
    "ktb3y_minus_base":   ("Korea Treasury Bond 3Y yield minus BOK base rate, %p", "daily", "ECOS / KOFIA"),
    "ktb3y_minus_msb91":  ("KTB 3Y yield minus Monetary Stabilization Bond 91D yield, %p", "daily", "ECOS / KOFIA"),
    "ktb10y_minus_ktb3y": ("KTB 10Y yield minus KTB 3Y yield (yield-curve slope), %p", "daily", "ECOS / KOFIA"),
    "vkospi":             ("VKOSPI - KOSPI 200 implied-volatility index (level), Korea's VIX", "daily", "KRX"),
    "cpi_total":          ("Korea CPI, all items, whole country, index 2020=100", "monthly", "KOSIS"),
    "cpi_yoy":            ("Korea CPI inflation, % change vs 12 months earlier", "monthly", "KOSIS (derived)"),
    "gdp_yoy":            ("Korea GDP growth, % change vs same quarter last year (nominal, SA)", "quarterly", "BOK/KOSIS (derived)"),
    "gdp_qoq":            ("Korea GDP growth, % change vs previous quarter (nominal, SA)", "quarterly", "BOK/KOSIS (derived)"),
    "semi_prod_sa":       ("Korea industrial production index, semiconductors & parts, seasonally adjusted", "monthly", "KOSIS"),
    "semi_prod_yoy":      ("Semiconductor production index, % change vs 12 months earlier", "monthly", "KOSIS (derived)"),
    "unemp_rate":         ("Korea unemployment rate, % (total, all ages/education)", "monthly", "KOSIS"),
    "korea_epu":          ("Korea Economic Policy Uncertainty index, news-based (level)", "monthly", "Cho & Kim (policyuncertainty.com)"),
    "global_epu":         ("Global Economic Policy Uncertainty index, GDP-weighted (level)", "monthly", "Davis (policyuncertainty.com)"),
    "gpr_global":         ("Geopolitical Risk index, global (level, 1985-2019 avg = 100)", "monthly", "Caldara & Iacoviello"),
    "gpr_korea":          ("Geopolitical Risk index, Korea-specific (share of articles, ~0-2 scale)", "monthly", "Caldara & Iacoviello"),
}


def _write_dictionary(out: pd.DataFrame) -> None:
    lines = [
        "# panel_daily_ffill.csv",
        "",
        f"Daily panel, {out.index.min().date()} to {out.index.max().date()}, "
        f"{len(out)} rows, one row per calendar day.",
        "",
        "How each series is put on the daily grid:",
        "- Daily series (markets, FX, rates, VKOSPI): last traded value is carried "
        "over weekends and holidays. Log returns are therefore 0 on non-trading days.",
        "- Monthly / quarterly series: each release is carried forward day by day "
        "until the next release, and up to one extra period after the final "
        "release (1 month / 1 quarter). Nothing is held flat beyond that - when a "
        "source series ends, the column becomes empty.",
        "",
        "| column | meaning | native freq | source | data from | data to |",
        "|---|---|---|---|---|---|",
    ]
    epu_end = "?"
    for c in out.columns:
        desc, freq, source = VAR_INFO[c]
        nn = out[c].dropna()
        lo = nn.index.min().date() if len(nn) else "-"
        hi = nn.index.max().date() if len(nn) else "-"
        if c == "korea_epu" and len(nn):
            epu_end = hi
        lines.append(f"| `{c}` | {desc} | {freq} | {source} | {lo} | {hi} |")
    lines += [
        "",
        "Notes:",
        f"- `korea_epu` source stops at Nov 2025, so the column ends {epu_end} "
        "and is empty after that (not forward-filled).",
        "- GDP is nominal (current prices). 2026 Q1 shows a large jump "
        "(+10.5% QoQ) that looks like a KOSIS data break; values are used raw.",
        "- Interest-rate columns are spreads, not yield levels (only spreads are in the source file).",
        "- `vkospi` rises steeply through 2026 (quarterly mean ~20 -> ~68); used raw, not yet cross-checked.",
        "- No ESG variable: no source data available.",
    ]
    (OUT / "panel_columns.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'panel_columns.md'}")


if __name__ == "__main__":
    main()
