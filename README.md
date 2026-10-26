# Local Crypto Day-Trader Lab

A local-only cryptocurrency analysis, SMC pattern recognition, machine-learning
win-probability evaluation, and paper trading dashboard. It scans a configurable
watchlist, displays Binance candles, detects geometric formations and Smart
Money Concepts (liquidity sweeps / stop hunts), predicts setup outcomes via
a local `RandomForestClassifier`, and manages paper trades with automatic
stop-loss and target execution. It does not connect to a broker or place live orders.

The default watchlist is BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, and
GRASSUSDT. Symbols, API settings, risk limits, refresh intervals, RSS feeds,
and local storage paths are defined in `config.json`.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The Streamlit server is configured to bind to `127.0.0.1` on port `8501`. Keep the process,
computer, and network connection running for continuous updates. If the
computer sleeps or the network is unavailable, polling pauses; the dashboard
shows data errors and the last candle timestamp rather than inventing a quote.

## Market data and signals

The app uses public Binance REST endpoints and does not need an API key.
It requests Spot klines first and falls back to Binance USD-M Futures only
when the requested symbol is not listed on Spot. The configured GRASSUSDT pair
is therefore expected to use Futures. Each result displays its source market.
The client supports 1-minute and 5-minute candles, pages historical requests
within Binance's kline limit, and excludes incomplete candles from signals.

## Smart Trade Advisor (SMC & Geometric Patterns)

`technical_analysis.py` (pure Python, look-ahead free) and `analyze_setup()` in
`crypto_engine.py` analyse the closed candles of every coin at any time of day
and use the live price for entries:

- **Indicators:** EMA 20/50/200, daily (UTC) and rolling-24h VWAP, RSI(14),
  MACD(12,26,9), ATR(14), Bollinger Bands(20, 2.0) with squeeze detection, and
  relative volume (RVOL, last closed candle vs. the previous 20).
- **Candlestick patterns:** Bullish/Bearish Engulfing, Hammer / Pin Bar,
  Shooting Star, Morning Star, Doji (last two closed candles).
- **Structure and chart patterns:** swing-based support/resistance zones,
  Higher-Highs/Higher-Lows vs. Lower-Highs/Lower-Lows, Double Bottom (W),
  Ascending / Descending / Symmetrical Triangles, Bull Flag (EMA20 retest),
  and Rolling 1h/4h High Breakout.
- **Smart Money Concepts (SMC):** Liquidity Sweep / Stop Hunt detection where
  price wicks below swing-low support (hunting sell-stops) and closes back above,
  with invalidation anchoring the stop directly beneath the sweep wick.
- **Setup Score (0-100):** trend 30, momentum 25, volume 10, structure 20,
  patterns and SMC 15, minus penalties for fresh bearish reversal candles or
  bearish sweeps. Ratings: 75-100 STRONG BUY, 60-74 BUY / WATCH, 40-59 NEUTRAL / WAIT,
  0-39 AVOID.
- **Levels:** the entry is the live price (or an EMA20 pullback for BUY /
  WATCH when price is extended). The stop sits below the nearest swing-low
  support or sweep wick, or at entry - 1.5 ATR. T1 sits before the first
  swing-high resistance, never below 1.5R; T2 is entry + 3 ATR or 2.5R.

## Machine Learning Win-Probability Classifier

`ml_model.py` provides a local, lightweight machine learning model powered by
`scikit-learn` (`RandomForestClassifier`):

- **Features (24):** distances to EMAs/VWAP in ATR units, ATR%, volatility regime,
  RVOL, RSI, MACD histogram, Bollinger position, distance to confirmed swing
  support/resistance, market structure, geometric patterns (Double Bottom, Triangles),
  SMC liquidity sweeps, candlestick reversals, and time-of-day cyclics.
- **Label:** 1 if price reaches the 1.5×ATR target before the 1.5×ATR stop within
  48 bars, else 0.
- **Validation:** Chronological purged holdout test reporting ROC AUC, Brier
  score, calibration curves, and feature importance with directional local effects.

## Dashboard

A dark, TradingView-style terminal. A top ticker shows the 24-hour change,
Setup Score, advisor rating, recognised pattern, safe T1 return, and source market;
four tabs hold the rest:

- **Élő Chart & Scanner**: interactive Plotly candlesticks with volume,
  Entry, Stop-Loss, and Target levels, switchable EMA 20/50/200, VWAP, Bollinger Bands,
  support/resistance zones, SMC & Alakzatok overlays, the Okos Tanácsadó panel,
  and the live scanner table.
- **🤖 ML Valószínűség**: local RandomForest win probability prediction for the
  active setup, top 3 decision factors (Feature Importance) with local impact,
  out-of-sample calibration curve, and feature table.
- **Paper Trading & Kockázat**: fractional quantity calculator based on USD
  risk, Binance step/tick sizes, fees, and slippage. Automatically closes open
  positions on live updates when Stop or Target is touched, factoring in fees
  and slippage.
- **Hírek, Hangulat & Riasztások**: crypto RSS headlines, Alternative.me Fear & Greed
  Index, and persistent price alerts.

## Command-line helpers

The existing module paths remain available for small standalone tasks:

```powershell
python trader-scanner\scan.py
python trader-entry\entry.py BTCUSDT
python trader-timemachine\timemachine.py BTCUSDT --interval 5m
python trader-budget\size.py BTCUSDT --entry 60000 --stop 58000 --risk-usd 5
python trader-news\news.py
python trader-alerts\alerts.py
```

## Tests

The tests use synthetic candles and mocked HTTP responses; no live Binance,
RSS, or sentiment API calls are required:

```powershell
python -m unittest discover -s tests -v
```

## Risk notice

This software is for education and simulation, not financial advice. Crypto
markets are volatile and may trade differently across Spot and Futures.
Historical simulation and ML probability estimates are not predictions or
guarantees. Stop prices do not guarantee a fill or a maximum loss. Verify all
data independently and never risk money you cannot afford to lose.
