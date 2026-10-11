"""
analysis.py — Technical analysis of NovaTech (NOVA) OHLC data.

What it does:
  1. Loads data/nova_ohlc.csv (run generate_data.py first).
  2. Computes indicators: SMA-20/50, EMA-12/26, MACD, RSI-14, Bollinger Bands.
  3. Detects signals: golden/death crosses, RSI overbought/oversold, MACD flips.
  4. Backtests a simple rule-based strategy vs buy-and-hold.
  5. Trains a LogisticRegression next-day direction classifier on lagged
     technical features (the "AI" component, alongside ai_insights.py).
  6. Saves charts to charts/ and key numbers to findings.json.

Run:  python analysis.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")

HERE = Path(__file__).parent
DATA = HERE / "data" / "nova_ohlc.csv"
CHARTS = HERE / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- indicators
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing via exponential means)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["RSI"] = rsi(df["Close"])
df["BB_mid"] = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["BB_upper"] = df["BB_mid"] + 2 * bb_std
df["BB_lower"] = df["BB_mid"] - 2 * bb_std
df["Return"] = df["Close"].pct_change()

# ------------------------------------------------------------------- signals
golden = (df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1))
death = (df["SMA_20"] < df["SMA_50"]) & (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1))
macd_bull = (df["MACD"] > df["MACD_signal"]) & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
macd_bear = (df["MACD"] < df["MACD_signal"]) & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
overbought = df["RSI"] > 70
oversold = df["RSI"] < 30

n_golden = int(golden.sum())
n_death = int(death.sum())
ob_days = int(overbought.sum())
os_days = int(oversold.sum())

# ------------------------------------------------------------- backtest
# Simple rule: hold when SMA_20 > SMA_50 (bullish trend), else cash.
df["Signal"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
df["Strategy_ret"] = df["Signal"].shift(1) * df["Return"]
strat_cum = (1 + df["Strategy_ret"].fillna(0)).prod() - 1
bh_cum = (1 + df["Return"].fillna(0)).prod() - 1
# Annualized Sharpe (risk-free 0), 252 trading days/year
excess = df["Strategy_ret"].dropna()
sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
max_dd = ((1 + df["Strategy_ret"].fillna(0)).cumprod() /
          (1 + df["Strategy_ret"].fillna(0)).cumprod().cummax() - 1).min()

# ------------------------------------------------- ML direction classifier
features = pd.DataFrame(index=df.index)
features["rsi"] = df["RSI"]
features["macd_hist"] = df["MACD"] - df["MACD_signal"]
features["sma_spread"] = (df["SMA_20"] - df["SMA_50"]) / df["Close"]
features["bb_pos"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
features["ret_1"] = df["Return"]
features["ret_5"] = df["Close"].pct_change(5)
features["vol_change"] = df["Volume"].pct_change()
features["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

ml = features.dropna()
X_train, X_test, y_train, y_test = train_test_split(
    ml.drop(columns="target"), ml["target"], test_size=0.2,
    shuffle=False,  # chronological split — no look-ahead bias
)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = float(accuracy_score(y_test, pred))
# Majority-class baseline for honest comparison
baseline = float(y_test.mean()) if y_test.mean() > 0.5 else float(1 - y_test.mean())
feat_imp = dict(sorted(
    zip(X_train.columns, np.abs(clf.coef_[0]).round(4)),
    key=lambda kv: kv[1], reverse=True))

# ------------------------------------------------------------------- charts
last = df.index[-1]
first = df.index[0]

# Chart 1: Price, MAs, Bollinger Bands
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close", color="#1f77b4", linewidth=1.2)
ax.plot(df.index, df["SMA_20"], label="SMA-20", color="#ff7f0e", linewidth=1)
ax.plot(df.index, df["SMA_50"], label="SMA-50", color="#2ca02c", linewidth=1)
ax.fill_between(df.index, df["BB_upper"], df["BB_lower"], color="gray", alpha=0.15,
                label="Bollinger Bands (20, 2σ)")
ax.scatter(df.index[golden], df["Close"][golden], marker="^", s=70,
           color="green", label=f"Golden cross ({n_golden})", zorder=5)
ax.scatter(df.index[death], df["Close"][death], marker="v", s=70,
           color="red", label=f"Death cross ({n_death})", zorder=5)
ax.set_title("NOVA — Price with Moving Averages, Bollinger Bands & Crossovers")
ax.set_ylabel("Price (INR)")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(CHARTS / "price_trend.png", dpi=120)
plt.close(fig)

# Chart 2: RSI + MACD panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(df.index, df["RSI"], color="purple", linewidth=1.2)
ax1.axhline(70, color="red", linestyle="--", linewidth=1)
ax1.axhline(30, color="green", linestyle="--", linewidth=1)
ax1.fill_between(df.index, 70, 100, where=overbought, color="red", alpha=0.2)
ax1.fill_between(df.index, 0, 30, where=oversold, color="green", alpha=0.2)
ax1.set_ylabel("RSI-14")
ax1.set_title(f"NOVA — RSI-14 (overbought: {ob_days}d, oversold: {os_days}d)")
ax2.plot(df.index, df["MACD"], label="MACD", linewidth=1.2)
ax2.plot(df.index, df["MACD_signal"], label="Signal", linewidth=1)
ax2.bar(df.index, df["MACD"] - df["MACD_signal"], color="gray", alpha=0.4, width=2,
        label="Histogram")
ax2.set_ylabel("MACD")
ax2.set_title("NOVA — MACD (12, 26, 9)")
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(CHARTS / "rsi_macd.png", dpi=120)
plt.close(fig)

# Chart 3: Strategy vs buy-and-hold
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, (1 + df["Strategy_ret"].fillna(0)).cumprod(),
        label=f"SMA crossover strategy ({strat_cum:+.1%})", linewidth=1.4)
ax.plot(df.index, (1 + df["Return"].fillna(0)).cumprod(),
        label=f"Buy & hold ({bh_cum:+.1%})", linewidth=1.2, linestyle="--")
ax.set_title("NOVA — Strategy vs Buy-and-Hold (cumulative growth of ₹1)")
ax.set_ylabel("Portfolio value (×)")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "strategy_vs_hold.png", dpi=120)
plt.close(fig)

# Chart 4: Returns distribution
fig, ax = plt.subplots(figsize=(10, 5))
rets = df["Return"].dropna() * 100
ax.hist(rets, bins=40, color="#1f77b4", alpha=0.8, edgecolor="white")
ax.axvline(rets.mean(), color="red", linestyle="--",
           label=f"Mean {rets.mean():.3f}% | Vol (σ) {rets.std():.2f}%")
ax.set_title("NOVA — Distribution of daily returns")
ax.set_xlabel("Daily return (%)")
ax.set_ylabel("Days")
ax.legend()
fig.tight_layout()
fig.savefig(CHARTS / "returns_distribution.png", dpi=120)
plt.close(fig)

# --------------------------------------------------------------- findings
findings = {
    "ticker": "NOVA",
    "company": "NovaTech Ltd",
    "currency": "INR",
    "period": {"start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"),
               "trading_days": len(df)},
    "price": {
        "first_close": round(float(df["Close"].iloc[0]), 2),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "period_return_pct": round(float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
        "max_close": round(float(df["Close"].max()), 2),
        "min_close": round(float(df["Close"].min()), 2),
        "avg_daily_volume": int(df["Volume"].mean()),
        "daily_volatility_pct": round(float(rets.std()), 2),
    },
    "signals": {
        "golden_crosses": n_golden,
        "death_crosses": n_death,
        "golden_cross_dates": [d.strftime("%Y-%m-%d") for d in df.index[golden]],
        "death_cross_dates": [d.strftime("%Y-%m-%d") for d in df.index[death]],
        "rsi_overbought_days": ob_days,
        "rsi_oversold_days": os_days,
        "rsi_now": round(float(df["RSI"].iloc[-1]), 1),
        "macd_now": round(float(df["MACD"].iloc[-1]), 2),
        "macd_signal_now": round(float(df["MACD_signal"].iloc[-1]), 2),
        "trend_now": "bullish" if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] else "bearish",
        "price_vs_bb": "above upper band" if df["Close"].iloc[-1] > df["BB_upper"].iloc[-1]
        else ("below lower band" if df["Close"].iloc[-1] < df["BB_lower"].iloc[-1] else "inside bands"),
    },
    "backtest": {
        "strategy": "hold when SMA_20 > SMA_50, else cash",
        "strategy_return_pct": round(strat_cum * 100, 2),
        "buy_hold_return_pct": round(bh_cum * 100, 2),
        "outperformance_pct": round((strat_cum - bh_cum) * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
    },
    "ml_direction_model": {
        "model": "LogisticRegression on 7 lagged technical features",
        "test_accuracy": round(acc, 4),
        "baseline_majority_class": round(baseline, 4),
        "beat_baseline": acc > baseline,
        "test_size": len(y_test),
        "feature_importance_abs_coef": feat_imp,
        "note": "Chronological train/test split (no shuffling) avoids look-ahead bias.",
    },
    "seed": 42,  # must match SEED in generate_data.py
    "data_note": "Synthetic OHLC generated by generate_data.py (seed 42); no real market data.",
}

(HERE / "findings.json").write_text(json.dumps(findings, indent=2))
print(json.dumps(findings, indent=2))
print("\nCharts saved to", CHARTS)
