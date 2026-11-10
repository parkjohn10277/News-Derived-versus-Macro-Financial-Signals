"""
Build input_gdp_corrected.csv from ../../data/input_aligned.csv (kept unchanged).
Identical to ../../data/build_input_auxfixed.py except the GDP lag: 28 -> 70 days.

Fixes to the monthly / quarterly auxiliary variables only:

1. Publication lag.  The original panel (Desktop/3_Auxiliary_Variables/
   build_daily_panel.py) stamps each value at its reference-PERIOD END and
   forward-fills, so e.g. June semiconductor production is visible from 30 June
   although KOSIS publishes it at the end of July.  Here each value is stamped at
   period end + a conservative publication lag, THEN forward-filled (same method).

GDP is NOMINAL (current prices).  The +10.5 % q/q jump in 2026 Q1 is kept: it is
driven almost entirely by value added in "Computer, electronic and optical products"
(+85 % q/q, 84 % of the GDP increase) -- the semiconductor price boom, not a data break.
Only the publication timing changes; no value is altered or dropped.

Daily market variables (S&P 500, global, SOX, KRW/USD, VKOSPI, rate spread),
the KRX target, LDA topics and sentiment are copied unchanged.

    python build_input_auxfixed.py  ->  input_auxfixed.csv, input_auxfixed_changes.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC_INPUT = HERE.parent.parent / "data" / "input_aligned.csv"   # the original panel, unchanged
AUXDIR = Path(r".\3_Auxiliary_Variables")
sys.path.insert(0, str(AUXDIR))
import build_daily_panel as bdp  # noqa: E402  (reuses the original loaders unchanged)

# days after the reference-period end at which the value is first known (conservative)
PUB_LAG_DAYS = {
    "cpi_yoy": 5,          # Statistics Korea CPI: first days of the following month
    "unemp_rate": 16,      # Economically Active Population Survey: ~mid following month
    "semi_prod_yoy": 31,   # Industrial Activity Trends: ~end of following month
    "gdp_yoy": 70,         # NOMINAL GDP is first released with the BOK preliminary estimate, ~64-70 days after quarter end (the 28-day advance estimate covers real GDP only)
    "global_epu": 7,       # policyuncertainty.com: first week of following month
    "gpr_global": 7,       # Caldara-Iacoviello GPR: first week of following month
    "gpr_korea": 7,
}


def load_gdp() -> dict:
    """Same as bdp.load_gdp; the original globs 'GDP_GNI ' (trailing space, Mac folder name)."""
    f = next((bdp.SRC / "GDP_GNI").glob("*.csv"))
    df = pd.read_csv(f)
    qidx = pd.DatetimeIndex([bdp.quarter_end_from_label(c) for c in df.columns[1:]])
    r = df[df.iloc[:, 0] == "Gross domestic product at market prices(GDP)"]
    s = pd.Series(pd.to_numeric(r.iloc[0, 1:], errors="coerce").to_numpy(dtype="float64"), index=qidx)
    return {"gdp_yoy": bdp.yoy(s, 4).rename("gdp_yoy")}


def periodic_series() -> dict[str, pd.Series]:
    s = {}
    for loader in (bdp.load_cpi, load_gdp, bdp.load_production_semi, bdp.load_unemployment,
                   bdp.load_global_epu, bdp.load_gpr):
        s.update(loader())
    return {k: s[k] for k in PUB_LAG_DAYS}


def ffill_daily(s: pd.Series, grid: pd.DatetimeIndex, max_gap_days: int) -> pd.Series:
    """Forward-fill a release-dated series to the daily grid (same rule as the original:
    no holding beyond one period after the last release)."""
    s = s.dropna().sort_index()
    out = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    out[out.index > s.index.max() + pd.Timedelta(days=max_gap_days)] = np.nan
    return out


def main():
    inp = pd.read_csv(SRC_INPUT, encoding="utf-8-sig", parse_dates=["date"])
    grid = pd.date_range(inp["date"].min() - pd.Timedelta(days=400), inp["date"].max(), freq="D")

    new = inp.copy()
    log = []
    for col, s in periodic_series().items():
        released = s.copy()
        released.index = released.index + pd.Timedelta(days=PUB_LAG_DAYS[col])
        # same rule as the original panel: carry at most one period past the last release
        gap = 92 if col == "gdp_yoy" else 31
        daily = ffill_daily(released, grid, gap)
        vals = daily.reindex(pd.DatetimeIndex(inp["date"])).to_numpy()
        changed = ~np.isclose(vals, inp[col].to_numpy(), equal_nan=True)
        new[col] = vals
        log.append(f"{col:15s} lag +{PUB_LAG_DAYS[col]:2d} d   days changed {changed.sum():5d}/{len(inp)}   "
                   f"missing {int(np.isnan(vals).sum())}")

    assert new[list(PUB_LAG_DAYS)].notna().all().all(), "a fixed column has missing values"
    new.to_csv(HERE / "input_gdp_corrected.csv", index=False, encoding="utf-8-sig")
    txt = "input_gdp_corrected.csv = input_aligned.csv with publication-lagged macro variables\n" \
          "(values unchanged, only the date each becomes available).  All other columns identical.\n\n" \
          + "\n".join(log)
    (HERE / "input_gdp_corrected_changes.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
