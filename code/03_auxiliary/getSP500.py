import yfinance as yf
import pandas as pd

# Define your basket of indices and proxies
tickers = [
    "^TWII",  # TAIEX (Taiwan)
    "^N225",  # Nikkei 225 (Japan)
    "^SOX",   # PHLX Semiconductor Index
    "URTH"    # MSCI World ETF Proxy
]

# Download data from 2014 to present
df = yf.download(tickers, start="2014-01-01", interval="1d")

# Extract just the 'Adj Close' prices to account for dividends/splits
adj_close = df['Close']

# Drop rows where all markets were closed (e.g., global holidays)
adj_close = adj_close.dropna(how='all')

# Save the combined dataset to CSV
adj_close.to_csv("global_semi_factors.csv")

print(adj_close.head())