"""
Figure: the target (foreign net buying) and the 13 auxiliary inputs, one panel each,
ordered and coloured by the groups of the variable table (tbl:aux); panel numbers
match the table's "No." column.  Daily values as used by the models (monthly /
quarterly series forward-filled); validation and test blocks shaded.

    python make_fig_aux_inputs.py   ->  fig_aux_inputs.pdf / .png
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from figstyle import apply_style

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent.parent / "GDP_corrected" / "data" / "input_gdp_corrected.csv"   # release-dated, GDP lag 70 days
VAL_START, TEST_START = "2024-03-13", "2025-05-08"

C_TARGET = "#3d3d3a"
GROUPS = {  # group -> colour (validated categorical slots 1-4)
    "Global equity markets": "#2a78d6",
    "Rates & FX": "#eb6834",
    "Macroeconomic fundamentals": "#1baf7a",
    "Risk & uncertainty": "#eda100",
}
PANELS = [  # column, title, group, scale   (order and numbers = tbl:aux)
    ("foreign_net_buy", "Target: foreign net buying (trillion KRW)", None, 1e-12),
    ("sp500_logret", "(1) S&P 500 return (log)", "Global equity markets", 1),
    ("global_logret", "(2) Global stock market return (log)", "Global equity markets", 1),
    ("sox_logret", "(3) Philadelphia Semiconductor Index return (log)", "Global equity markets", 1),
    ("krw_usd", "(4) KRW/USD exchange rate", "Rates & FX", 1),
    ("ktb3y_minus_base", "(5) 3-year KTB yield minus policy rate (%p)", "Rates & FX", 1),
    ("cpi_yoy", "(6) CPI inflation, y/y (%)", "Macroeconomic fundamentals", 1),
    ("gdp_yoy", "(7) Nominal GDP growth, y/y (%)", "Macroeconomic fundamentals", 1),
    ("unemp_rate", "(8) Unemployment rate (%)", "Macroeconomic fundamentals", 1),
    ("semi_prod_yoy", "(9) Semiconductor production growth, y/y (%)", "Macroeconomic fundamentals", 1),
    ("vkospi", "(10) VKOSPI (implied volatility)", "Risk & uncertainty", 1),
    ("global_epu", "(11) Global economic policy uncertainty", "Risk & uncertainty", 1),
    ("gpr_global", "(12) Geopolitical risk, global", "Risk & uncertainty", 1),
    ("gpr_korea", "(13) Geopolitical risk, Korea", "Risk & uncertainty", 1),
]
THIN = {"sp500_logret", "global_logret", "sox_logret", "foreign_net_buy"}   # daily returns / flows

apply_style()
d = pd.read_csv(INPUT, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")

fig, axes = plt.subplots(7, 2, figsize=(7.2, 8.9), sharex=True)
for ax, (col, title, group, scale) in zip(axes.flat, PANELS):
    ax.axvspan(pd.Timestamp(VAL_START), pd.Timestamp(TEST_START), color="0.93", lw=0, zorder=0)
    ax.axvspan(pd.Timestamp(TEST_START), d.index.max(), color="0.84", lw=0, zorder=0)
    c = GROUPS[group] if group else C_TARGET
    ax.plot(d.index, d[col] * scale, color=c, lw=0.35 if col in THIN else 0.9)
    if col in THIN:
        ax.axhline(0, color="0.5", lw=0.4, zorder=1)
    ax.set_title(title, loc="left", pad=2)
for ax in axes[-1]:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(d.index.min(), d.index.max())

handles = [Patch(color=C_TARGET, label="Target")] + [Patch(color=c, label=g) for g, c in GROUPS.items()]
handles += [Patch(color="0.93", label="Validation"), Patch(color="0.84", label="Test")]
fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
           handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(h_pad=0.45, w_pad=0.8, rect=(0, 0, 1, 0.975))
fig.savefig(HERE / "fig_aux_inputs.pdf")
fig.savefig(HERE / "fig_aux_inputs.png", dpi=200)
print("saved", HERE / "fig_aux_inputs.pdf")
