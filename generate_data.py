"""
generate_data.py — Synthetic OHLC stock data for the Stock Trend Analyzer.

Creates one year of daily trading data (business days only) for a fictional
company "NovaTech Ltd" (ticker: NOVA), priced in INR.

How the data is made (all synthetic — no real market data, no external APIs):
  * Geometric Brownian motion: next_price = price * exp((mu - sigma^2/2)*dt
    + sigma * sqrt(dt) * z), with z drawn from a fixed-seed random generator.
  * Two volatility regimes (calm vs choppy) plus one mid-year market shock
    (earnings miss) and one rally, so the series has realistic-looking
    trends, pullbacks, and a stress event for RSI/MA signals to react to.
  * Open/High/Low are derived from Close with small intraday noise;
    Volume follows a log-normal distribution with spikes on big move days.

Running this script creates data/nova_ohlc.csv. The CSV is NOT pushed to
GitHub (README explains this); regenerate it locally with this script.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_DAYS = 260                      # ~1 year of trading days (business days only)
START_PRICE = 1250.0              # INR, plausible mid-cap tech stock price
OUTPUT = Path(__file__).parent / "data" / "nova_ohlc.csv"

rng = np.random.default_rng(SEED)

# Business days ending yesterday (a Friday/Saturday buffer keeps it realistic)
dates = pd.bdate_range(end="2026-10-09", periods=N_DAYS, freq="C")
dates.name = "Date"

closes = np.empty(N_DAYS)
closes[0] = START_PRICE

for i in range(1, N_DAYS):
    # --- regime: calm (sigma=1.2%), choppy (sigma=2.6%), or event-driven ---
    day = i
    if 60 <= day < 75:
        # Market shock: 3 weeks of negative drift + high volatility
        mu, sigma = -0.012, 0.035
    elif 130 <= day < 155:
        # Earnings rally: positive drift, moderate volatility
        mu, sigma = 0.008, 0.020
    elif (i // 40) % 2 == 0:
        mu, sigma = 0.0009, 0.012     # calm uptrend regime
    else:
        mu, sigma = -0.0002, 0.026    # choppy regime

    z = rng.standard_normal()
    closes[i] = closes[i - 1] * np.exp((mu - sigma ** 2 / 2) + sigma * z)

# Intraday structure around the close
intraday_noise = rng.normal(0, 0.004, N_DAYS)
opens = closes * (1 + intraday_noise)
daily_range = np.abs(rng.normal(0.012, 0.005, N_DAYS)) * closes
highs = np.maximum(opens, closes) + daily_range * rng.uniform(0.3, 1.0, N_DAYS)
lows = np.minimum(opens, closes) - daily_range * rng.uniform(0.3, 1.0, N_DAYS)

# Volume: log-normal base, 2-4x spikes on big |return| days
returns_abs = np.abs(np.diff(closes, prepend=closes[0]) / closes)
volume = rng.lognormal(mean=np.log(850_000), sigma=0.45, size=N_DAYS)
volume = volume * (1 + 2.2 * np.clip(returns_abs / 0.05, 0, 1))
volume = volume.astype(int)

df = pd.DataFrame(
    {
        "Date": dates,
        "Open": opens.round(2),
        "High": highs.round(2),
        "Low": lows.round(2),
        "Close": closes.round(2),
        "Volume": volume,
    }
)
df = df.set_index("Date")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT)
print(f"Wrote {len(df)} trading days to {OUTPUT}")
print(df.head(3).to_string())
print("...")
print(df.tail(3).to_string())
