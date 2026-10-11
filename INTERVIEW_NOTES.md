# INTERVIEW_NOTES.md — Day 4: Stock Trend Analyzer

Interview Q&A prep for **this** project only. Every answer is grounded in what the
project actually does and the numbers it actually produced. No bluffing.

## 30-second pitch

"I built a technical-analysis toolkit on one year of synthetic OHLC data for a
fictional stock, NOVA. It computes the classic indicator set — SMA-20/50, MACD,
RSI-14, Bollinger Bands — detects golden and death crosses, backtests a
trend-following strategy against buy-and-hold, and trains a logistic-regression
model to predict next-day direction. The interesting part: the strategy trailed
buy-and-hold by 4 points and the ML model scored 53.5%, below the 62.8%
naive baseline — so the project's real lesson is an honest one about how hard
prediction is. An `ai_insights.py` script auto-generates a plain-English market
commentary from the computed findings."

## Project walkthrough (end-to-end)

**Q: Walk me through the project.**
"A: First `generate_data.py` synthesizes 260 trading days of OHLC data for a
fictional company using geometric Brownian motion with a fixed seed — calm and
choppy volatility regimes, plus a scripted mid-year market shock and an earnings
rally, so indicators have realistic structure to react to. Then `analysis.py`
computes the indicator set, detects signals like golden/death crosses and
RSI overbought/oversold days, backtests a 'hold when SMA-20 > SMA-50, else cash'
strategy, and trains a logistic-regression next-day direction classifier on seven
lagged technical features with a chronological train/test split. Finally
`ai_insights.py` reads `findings.json` and writes a narrative market commentary.
The script saves four charts: price with moving averages and Bollinger Bands,
an RSI+MACD panel, strategy-vs-buy-and-hold, and the returns distribution."

**Q: Why synthetic data instead of real market data?**
"A: Three reasons: reproducibility — seed 42 means anyone gets the identical
dataset; legality and simplicity — no API keys, no rate limits, no licensing;
and pedagogy — I could script known regimes like a shock and a rally, so the
indicators have ground truth to respond to. The honest limitation is that
synthetic data is well-behaved by construction, so real markets would break
these models harder. That's also why the README carries a not-investment-advice
disclaimer."

## Technical deep-dives

**Q: Explain the RSI calculation.**
"A: RSI-14 measures momentum on a 0–100 scale. You take daily price changes,
separate gains and losses, smooth each with an exponential moving average using
Wilder's smoothing factor alpha = 1/14, compute RS = average gain / average
loss, then RSI = 100 − 100/(1+RS). Readings above 70 are 'overbought',
below 30 'oversold'. NOVA spent 24 days overbought and 16 oversold. The honest
limitation: in a strong trend, RSI can stay overbought for weeks — it's a
mean-reversion tool being applied to trending data, so it whipsaws."

**Q: What is MACD and what does it tell you here?**
"A: MACD is the EMA-12 minus the EMA-26 — a momentum oscillator built from two
trend-following averages, with a 9-day EMA of the MACD line as the 'signal'
line. Crosses of MACD above/below its signal line flag momentum shifts, and the
histogram shows their strength. On NOVA's last day, MACD was −11.44 versus a
signal of −20.04, i.e. MACD above its signal line — short-term bullish momentum
inside an overall bearish SMA trend. That kind of mixed reading is normal and
why analysts never trade on one indicator."

**Q: How do golden and death crosses work?**
"A: A golden cross fires when the fast average (SMA-20) crosses above the slow
one (SMA-50); a death cross is the reverse. They're pure trend-confirmation
signals — NOVA produced 1 golden cross (2026-04-27) and 2 death crosses
(2026-01-20, 2026-08-14). The limitation is structural: moving averages lag by
construction, so a crossover confirms a trend that already started and misses
the first few percent of every move. The backtest prices in exactly that cost."

**Q: Explain the backtest and its result.**
"A: The rule is dead simple: hold the stock when SMA-20 is above SMA-50,
otherwise sit in cash. Over the year it returned −14.2% versus −10.2% for
buy-and-hold, with a Sharpe of −0.67 and a max drawdown of −22.9%. It lost
because the lagging signals kept it invested through drawdowns and in cash
during rebounds. I present this as a teaching result, not a trading system —
it ignores brokerage, STT, taxes, and slippage, which would make it worse."

**Q: Walk through the ML model.**
"A: It's a LogisticRegression predicting whether tomorrow's close is higher
than today's, using seven lagged features: RSI, MACD histogram, SMA-20/50
spread, Bollinger Band position, 1-day and 5-day returns, and volume change.
Crucially the split is chronological — first 80% train, last 20% test, no
shuffling — because shuffling time series leaks the future into training
(look-ahead bias). Top features by |coefficient| were Bollinger position, then
volume change, then 1-day return."

**Q: The model scored 53.5% — worse than the 62.8% baseline. Isn't that a failure?**
"A: It's the most valuable result in the project. The baseline is 'always guess
the majority class', and a fancy model losing to it is the honest, normal
outcome of predicting next-day direction from past prices — markets are close
to a random walk at that horizon. I'd rather present a negative result with a
correct evaluation setup than an inflated 90% from a shuffled split. It also
demonstrates the full ML workflow: feature engineering, leakage-free splits,
and benchmarking against a naive baseline before claiming anything."

**Q: What is look-ahead bias and how did you avoid it?**
"A: Look-ahead bias is training on information that wouldn't have been available
at prediction time — for example shuffling daily rows so the model trains on
future days and 'predicts' past ones. I avoided it two ways: the train/test
split is chronological, and every feature is computed only from data up to day
t to predict day t+1's direction."

## Data-analyst thinking

**Q: What were the KPIs and the key insight?**
"A: KPIs: period return (−10.2%), daily volatility (1.85%), signal counts
(1 golden / 2 death crosses, 24 overbought / 16 oversold days), strategy
outperformance (−4.0 pp vs buy-and-hold), Sharpe (−0.67), max drawdown (−22.9%),
and classifier accuracy vs baseline (53.5% vs 62.8%). The key insight is a
two-part story: the indicators correctly described a choppy down year, but
neither the rule-based strategy nor the ML model could turn description into
profitable prediction — which is precisely the gap a good analyst must
communicate to stakeholders."

**Q: How did the analysis become recommendations?**
"A: The `ai_insights.py` commentary translates each finding into plain English:
the trend read ('a broadly down year with enough chop to punish
buy-and-hold-and-forget'), the momentum read (RSI 49, balanced; mixed MACD),
and the strategy verdict ('a teaching result, not a trading system'). The
recommendation embedded in it is meta: describe risk honestly, model costs
before trusting a backtest, and never present a lagging indicator as a forecast."

## The "AI integrated" part

**Q: What does "AI integrated" mean in this project?**
"A: Two things. First, the machine-learning model — logistic regression for
next-day direction on engineered technical features. Second, `ai_insights.py`,
a narrative engine that turns the computed findings into a plain-English market
commentary: trend, momentum, signals, backtest verdict, honest ML assessment,
and risk caveats. It's rule-based rather than an LLM call, which is deliberate:
every sentence is traceable to a number in `findings.json`, so nothing is
hallucinated — the trade-off is less fluent prose in exchange for verifiability."

**Q: Why not just call an LLM API for the commentary?**
"A: An LLM would write smoother prose, but it can also invent numbers and it
needs an API key, costs money per call, and isn't reproducible offline. The
template engine is deterministic, free, runs anywhere, and — most importantly
for an analyst deliverable — every claim cites the metric that produced it.
For a real product I'd consider an LLM with the findings injected as context
and strict instructions to only use supplied numbers."

## With more time I would…

- "Model transaction costs, STT, and slippage in the backtest so the strategy comparison is realistic."
- "Add walk-forward validation for the ML model instead of a single chronological split."
- "Test on multiple synthetic regimes and, eventually, real data to check robustness."
- "Add position sizing and stop-losses to the strategy — raw returns without risk management are incomplete."
- "Compare more models (random forest, gradient boosting) with proper time-series cross-validation."
