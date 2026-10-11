"""
ai_insights.py — Auto-generated "AI market commentary" for the Stock Trend Analyzer.

This is the "AI integrated" part of the project: a deterministic narrative
engine that reads findings.json (the output of analysis.py) and writes a
plain-English market commentary — trend read, momentum read, signal recap,
strategy verdict, and risk notes — the way an analyst's desk note would.

It is rule-based, not a language model: every sentence is grounded in a
computed number, and the templates say exactly which number triggered them.
Run:  python ai_insights.py   (after analysis.py)
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
f = json.loads((HERE / "findings.json").read_text())

p, s, b, m = f["price"], f["signals"], f["backtest"], f["ml_direction_model"]

# ------------------------------------------------------------- narrative bits
def trend_story() -> str:
    direction = "up" if p["period_return_pct"] >= 0 else "down"
    return (
        f"NOVA gained {p['period_return_pct']:+.1f}% over the {f['period']['trading_days']} trading days "
        f"studied ({f['period']['start']} → {f['period']['end']}), rising from ₹{p['first_close']:,.2f} "
        f"to ₹{p['last_close']:,.2f}. The road was not straight: the stock swung between "
        f"₹{p['min_close']:,.2f} and ₹{p['max_close']:,.2f}, a "
        f"{(p['max_close'] / p['min_close'] - 1) * 100:.0f}% range, on average daily volume of "
        f"{p['avg_daily_volume']:,} shares. In plain terms: a broadly {direction} year with "
        f"enough chop to punish buy-and-hold-and-forget investors."
    )

def momentum_story() -> str:
    rsi_now = s["rsi_now"]
    if rsi_now > 70:
        rsi_read = (f"RSI sits at {rsi_now}, in overbought territory — momentum is hot and "
                    "a cooling pullback would be normal, not a reason to panic.")
    elif rsi_now < 30:
        rsi_read = (f"RSI sits at {rsi_now}, in oversold territory — selling looks exhausted "
                    "and a relief bounce often follows, though 'oversold can stay oversold'.")
    else:
        rsi_read = (f"RSI sits at {rsi_now}, comfortably mid-range — momentum is balanced, "
                    "neither euphoria nor capitulation.")
    macd_dir = "bullish" if s["macd_now"] > s["macd_signal_now"] else "bearish"
    return (
        f"{rsi_read} Over the full year the stock spent {s['rsi_overbought_days']} days overbought "
        f"and {s['rsi_oversold_days']} days oversold, which fits a stock with {p['daily_volatility_pct']}% "
        f"average daily volatility. MACD is currently {macd_dir} "
        f"(MACD {s['macd_now']} vs signal {s['macd_signal_now']}), and the short-term trend filter "
        f"agrees: price is {s['trend_now']} by the SMA-20/SMA-50 rule, trading {s['price_vs_bb']}."
    )

def signal_story() -> str:
    g, d = s["golden_cross_dates"], s["death_cross_dates"]
    g_str = ", ".join(g) if g else "none"
    d_str = ", ".join(d) if d else "none"
    return (
        f"The SMA-20/SMA-50 crossover system fired {s['golden_crosses']} golden crosses "
        f"({g_str}) and {s['death_crosses']} death crosses ({d_str}). "
        "Crossovers are lagging by construction — they confirm a trend that has already "
        "started, so they miss the first few percent of every move. That lag is exactly "
        "why the backtest below matters: it prices in the cost of being late."
    )

def strategy_story() -> str:
    win = "beat" if b["outperformance_pct"] >= 0 else "trailed"
    return (
        f"A naive '{b['strategy']}' rule returned {b['strategy_return_pct']:+.1f}% versus "
        f"{b['buy_hold_return_pct']:+.1f}% for simply buying and holding — it {win} buy-and-hold "
        f"by {abs(b['outperformance_pct']):.1f} points. Risk-adjusted, the strategy's Sharpe ratio was "
        f"{b['sharpe_ratio']} with a worst peak-to-trough drawdown of {b['max_drawdown_pct']:.1f}%. "
        "Take this as a teaching result, not a trading system: it ignores brokerage, taxes, "
        "slippage, and the fact that it sat in cash during rallies it failed to catch. "
        "Real edges are rarely this clean."
    )

def ml_story() -> str:
    beat = "did beat" if m["beat_baseline"] else "did not beat"
    top3 = ", ".join(f"{k} ({v})" for k, v in list(m["feature_importance_abs_coef"].items())[:3])
    return (
        f"The next-day direction classifier (LogisticRegression on 7 lagged technical features) "
        f"scored {m['test_accuracy']:.1%} accuracy on the last {m['test_size']} trading days, which "
        f"{beat} the always-guess-the-majority-class baseline of {m['baseline_majority_class']:.1%}. "
        f"Most influential features by |coefficient|: {top3}. Honest read: predicting next-day "
        "direction from past prices is barely better than a coin flip even on clean synthetic data — "
        "on real markets, transaction costs would eat the rest. The model's real value here is "
        "pedagogical: it shows the full ML workflow (feature engineering, chronological split to "
        "avoid look-ahead bias, honest baseline) on time-series data."
    )

md = f"""# AI Market Commentary — NovaTech Ltd (NOVA)

*Auto-generated by `ai_insights.py` from computed findings in `findings.json`.
Every claim below traces to a number — no invented data, no real market advice.
This project uses synthetic data; nothing here is investment advice.*

## 1. The big picture
{trend_story()}

## 2. Momentum check
{momentum_story()}

## 3. What the signals said
{signal_story()}

## 4. Strategy backtest verdict
{strategy_story()}

## 5. ML direction model — honest assessment
{ml_story()}

## 6. Risks & caveats (read before acting on any of this)
- **Synthetic data.** Regimes, the shock, and the rally were scripted; real markets
  are adversarial and regime shifts are not announced in advance.
- **No costs modelled.** Brokerage, STT, slippage, and taxes are all zero here and
  very much non-zero in reality.
- **Lagging indicators.** SMAs, RSI, and MACD all describe the past. They manage
  risk; they do not predict the future.
- **Overfitting danger.** Parameters (20/50/14/2σ) were chosen by convention, not
  optimized — which is actually a virtue here: tuning them on this one dataset
  would just memorize noise.

---
*Generated {f['period']['end']} · seed {f['seed']} · {f['data_note']}*
"""

(HERE / "AI_INSIGHTS.md").write_text(md)
print("Wrote AI_INSIGHTS.md")
