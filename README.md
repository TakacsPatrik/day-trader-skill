# Local Crypto Day-Trader Lab

A local-only cryptocurrency analysis and paper trading dashboard. It scans a
configurable watchlist, evaluates UTC daily opening-range breakouts, displays
Binance candles, simulates 10,000 historical price paths, and records paper
trades on your computer. It does not connect to a broker or place live orders.

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

The Streamlit server is configured to bind to `127.0.0.1`. Keep the process,
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

The default Opening Range starts at 00:00 UTC every day and uses the first
15 minutes. A breakout requires a later, completed candle to close above the
range high. This is a technical signal, not an instruction to trade.

## Smart Trade Advisor

`technical_analysis.py` (pure Python, no TA-Lib/numpy) and `analyze_setup()` in
`crypto_engine.py` analyse the closed candles of every coin at any time of day
and use the live price for entries:

- **Indicators:** EMA 20/50/200, daily (UTC) and rolling-24h VWAP, RSI(14),
  MACD(12,26,9), ATR(14), Bollinger Bands(20, 2.0) with squeeze detection, and
  relative volume (RVOL, last closed candle vs. the previous 20).
- **Candlestick patterns:** Bullish/Bearish Engulfing, Hammer / Pin Bar,
  Shooting Star, Morning Star, Doji (last two closed candles).
- **Structure and chart patterns:** swing-based support/resistance zones,
  Higher-Highs/Higher-Lows vs. Lower-Highs/Lower-Lows, Double Bottom,
  Ascending Triangle, Bull Flag (EMA20 retest), and Rolling 1h/4h High
  Breakout.
- **Setup Score (0-100):** trend 30, momentum 25, volume 10, structure 20,
  patterns and Bollinger 15, minus a penalty for a fresh bearish candle.
  Ratings: 75-100 STRONG BUY, 60-74 BUY / WATCH, 40-59 NEUTRAL / WAIT, 0-39
  AVOID. STRONG BUY additionally requires price above EMA50 and VWAP, a
  positive MACD histogram, RSI 40-68, RVOL >= 1 and a bullish pattern;
  otherwise the score is capped at 74. A downtrend below the EMA50, RSI above
  75, or a bearish reversal candle at resistance forces AVOID (max 39), and a
  T1 smaller than the round-trip fees and slippage caps the score at 59.
- **Levels:** the entry is the live price (or an EMA20 pullback for BUY /
  WATCH when price is extended). The stop sits below the nearest swing-low
  support (0.75-2.5 ATR away) or at entry - 1.5 ATR. T1 ("safe target") sits
  just before the first swing-high resistance or at entry + 1.5 ATR, never
  below 1.5R and capped at 3 ATR; T2 is entry + 3 ATR or 2.5R. The expected
  return percentages are conservative estimates, not guarantees.

On quiet markets the 1.5R target can be smaller than fees; the dashboard shows
the net-of-fees return and downgrades such setups.

## Time Machine

The simulator chains contiguous blocks of historical candles (default one hour,
12h horizon) and rescales each block by current ATR% / ATR% at the block's
start. The historical drift is removed, bars that touch both stop and target
count as a stop, and runs are seeded. Next to it, the **historical hit rate**
replays real paths that started in a comparable volatility regime (ATR within
1.5x of the current one) and reports how often the target was reached before
the stop. These windows overlap, so the independent sample is smaller than the
printed count. Horizon, block length, and the 10,000 runs are configurable.

## Dashboard

A dark, TradingView-style terminal. A top ticker shows the 24-hour change,
Setup Score, advisor rating, recognised pattern, safe T1 return, and breakout
state of every configured coin; four tabs hold the rest:

- **Élő Chart & Scanner**: interactive Plotly candlesticks with volume, UTC
  Opening Range, Entry, Stop-Loss, and Target levels, switchable EMA 20/50/200,
  VWAP, Bollinger Bands, and support/resistance zones, the "Okos Tanácsadó &
  Mintázatok" panel (score, T1/T2 returns, stop distance, ✅/⚠️/❌ reasoning,
  one-click plan loading), and the scanner table.
- **Time Machine**: 10,000-run volatility-adjusted block-bootstrap simulation
  of the advisor's (or your own) Entry/Stop/Target with target, stop, and
  neither probabilities, the historical hit rate, a return histogram, and
  sample paths.
- **Paper Trading & Kockázat**: fractional quantity calculator based on USD
  risk, Binance step/tick sizes, minimum notional, fees, and slippage; open
  positions with live unrealized P&L; closed-trade history.
- **Hírek, Hangulat & Riasztások**: crypto RSS headlines, the Alternative.me
  Fear & Greed Index, and persistent price alerts.

Market data is cached for at most 5 seconds. The sidebar can switch automatic
refreshing on or off, choose 5 / 10 / 30 seconds, and clear the cache with
"Frissítés most". The status bar shows the last refresh to the second in UTC and
local time.

Paper trades are stored in `data/paper_trades.json`. Alert rules are stored in
`alerts.json` (loaded on startup, so they survive restarts; see
`trader-alerts/alerts.example.json` for the format), alert trigger state in
`data/alert_state.json`. The command-line `trader-alerts\alerts.py` reads the
same `alerts.json`. These are local files and are not uploaded by this
application.

## Command-line helpers

The existing module paths remain available for small standalone tasks:

```powershell
python trader-scanner\scan.py
python trader-entry\entry.py BTCUSDT
python trader-budget\size.py BTCUSDT --entry 60000 --stop 58000 --risk-usd 5
python trader-timemachine\timemachine.py BTCUSDT --entry 60000 --stop 58000 --target 64000
python trader-news\news.py
python trader-alerts\alerts.py
```

`trader-exit\exit.py` creates a tick-aligned long-side paper plan, and
`trader-risk\check.py` checks a plan's fractional quantity against the USD
risk rules. These helpers also read the central `config.json`.

## Tests

The tests use synthetic candles and mocked HTTP responses; no live Binance,
RSS, or sentiment API calls are required:

```powershell
python -m unittest discover -s tests -v
```

## Risk notice

This software is for education and simulation, not financial advice. Crypto
markets are volatile and may trade differently across Spot and Futures.
Historical simulation results are not predictions and do not include every
future market condition. Stop prices do not guarantee a fill or a maximum
loss. Verify all data independently and never risk money you cannot afford
to lose.
