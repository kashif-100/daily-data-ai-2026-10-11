# Day 4 — Stock Trend Analyzer 📈

Technical analysis of a synthetic stock (**NovaTech Ltd**, ticker `NOVA`, INR)
covering one year of trading days (2025-10-13 → 2026-10-09).

**Key findings**
- NOVA returned **−10.2%** over the year (₹1,250.00 → ₹1,122.41) with 1.85% average
  daily volatility — a down year with plenty of chop.
- The SMA-20/SMA-50 crossover system fired **1 golden cross** and **2 death crosses**;
  RSI spent 24 days overbought (>70) and 16 days oversold (<30).
- A simple trend-following strategy (hold when SMA-20 > SMA-50, else cash) returned
  **−14.2%** vs **−10.2%** for buy-and-hold — it *trailed* by 4 points, a reminder
  that lagging indicators charge a "cost of being late."
- A LogisticRegression next-day direction classifier scored **53.5%** accuracy vs a
  **62.8%** majority-class baseline — honestly *worse* than the naive baseline.
  That negative result is the most instructive part of the project (see interview notes).

## What's inside

| File | What it does |
|---|---|
| `generate_data.py` | Creates `data/nova_ohlc.csv` — 260 synthetic trading days via geometric Brownian motion with calm/choppy volatility regimes plus a scripted market shock and rally (fixed seed 42) |
| `analysis.py` | Computes SMA-20/50, EMA-12/26, MACD, RSI-14, Bollinger Bands; detects crossover & RSI signals; backtests the trend strategy; trains the direction classifier; saves 4 charts + `findings.json` |
| `ai_insights.py` | Reads `findings.json` and auto-generates `AI_INSIGHTS.md` — a plain-English "market commentary" (trend, momentum, signals, backtest verdict, honest ML assessment, risks) |
| `INTERVIEW_NOTES.md` | Interview Q&A prep grounded in this project's real numbers |
| `findings.json` | All computed metrics in one machine-readable file |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/nova_ohlc.csv (260 days, seed 42)
python analysis.py        # indicators, signals, backtest, ML model, charts
python ai_insights.py     # writes AI_INSIGHTS.md
```

> The raw CSV (`data/nova_ohlc.csv`) is generated locally and not committed —
> run `generate_data.py` to reproduce it exactly (seeded, so it's identical every time).

## Charts

- `charts/price_trend.png` — Close price with SMA-20/50, Bollinger Bands, golden/death crosses
- `charts/rsi_macd.png` — RSI-14 with overbought/oversold zones + MACD panel
- `charts/strategy_vs_hold.png` — Crossover strategy vs buy-and-hold cumulative growth
- `charts/returns_distribution.png` — Distribution of daily returns

## Tech stack

pandas · numpy · matplotlib · seaborn · scikit-learn (LogisticRegression)

## Disclaimer

All data is synthetic — no real market data, no real company. Nothing here is investment advice.
