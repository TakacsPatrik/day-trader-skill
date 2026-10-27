import datetime
import math
import os
import random
import sys
import tempfile
import unittest
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bot
import ml_model
import mtf
import technical_analysis as ta
from binance_client import load_config
from crypto_engine import TradingError, analyze_setup, build_plan, rating_for
from paper_trading import list_trades

STEP = 300_000
START = int(datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
FILTERS = {"tick_size": "0.01", "step_size": "0.0001", "min_qty": "0.0001", "min_notional": "5"}


def make_bars(closes, spread=0.2, volumes=None, start=START, step=STEP):
    bars, previous = [], closes[0]
    for i, close in enumerate(closes):
        open_ = previous
        high = max(open_, close) + spread
        low = min(open_, close) - spread
        bars.append({"open_time": start + i * step, "close_time": start + (i + 1) * step - 1,
                     "open": Decimal(repr(round(open_, 6))), "high": Decimal(repr(round(high, 6))),
                     "low": Decimal(repr(round(low, 6))), "close": Decimal(repr(round(close, 6))),
                     "volume": Decimal(repr(volumes[i] if volumes else 10.0)), "market": "spot"})
        previous = close
    return bars


def after(bars):
    return datetime.datetime.fromtimestamp((bars[-1]["close_time"] + 1) / 1000, datetime.timezone.utc)


def random_walk(n, sigma, drift=0.0, seed=3, start_price=100.0):
    rng = random.Random(seed)
    price, closes = start_price, []
    for _ in range(n):
        price *= 1 + drift + rng.gauss(0, sigma)
        closes.append(price)
    return closes


def fbar(o, h, low, c, index=0):
    return {"t": START + index * STEP, "ct": START + (index + 1) * STEP - 1,
            "o": o, "h": h, "l": low, "c": c, "v": 10.0}


def with_context(last_bars, trend):
    """Ten bars drifting in ``trend`` direction followed by the pattern bars."""
    first = last_bars[0]["o"]
    lead = [fbar(first - trend * (10 - i), first - trend * (10 - i) + 0.5,
                 first - trend * (10 - i) - 0.5, first - trend * (10 - i - 1), i) for i in range(10)]
    pattern = [dict(b, t=START + (10 + k) * STEP, ct=START + (11 + k) * STEP - 1)
               for k, b in enumerate(last_bars)]
    return lead + pattern


class IndicatorTests(unittest.TestCase):
    def test_ema_matches_hand_calculation(self):
        out = ta.ema([float(i) for i in range(1, 11)], 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[-1], 9.0)
        self.assertEqual(ta.ema([5.0] * 30, 20)[-1], 5.0)

    def test_rsi_extremes_and_flat(self):
        self.assertEqual(ta.rsi([float(i) for i in range(40)])[-1], 100.0)
        self.assertEqual(ta.rsi([float(40 - i) for i in range(40)])[-1], 0.0)
        self.assertEqual(ta.rsi([7.0] * 40)[-1], 50.0)
        zigzag = [100 + (i % 2) for i in range(80)]
        self.assertAlmostEqual(ta.rsi(zigzag)[-1], 50.0, delta=3.0)

    def test_macd_is_zero_for_constant_series_and_warmup_is_correct(self):
        line, signal, hist = ta.macd([50.0] * 80)
        self.assertAlmostEqual(line[-1], 0.0)
        self.assertAlmostEqual(hist[-1], 0.0)
        self.assertEqual(next(i for i, v in enumerate(hist) if v is not None), 33)
        rising = ta.macd([float(i) for i in range(120)])
        self.assertGreater(rising[0][-1], 0)

    def test_atr_of_constant_range(self):
        closes = [100.0] * 40
        value = ta.atr([101.0] * 40, [99.0] * 40, closes)
        self.assertAlmostEqual(value[-1], 2.0)
        self.assertIsNone(value[0])

    def test_bollinger_bands_and_squeeze(self):
        mid, up, low, width = ta.bollinger([9.0, 11.0] * 20)
        self.assertAlmostEqual(mid[-1], 10.0)
        self.assertAlmostEqual(up[-1], 12.0)
        self.assertAlmostEqual(low[-1], 8.0)
        self.assertAlmostEqual(ta.bollinger([5.0] * 30)[3][-1], 0.0)
        shrinking = [1.0 - i * 0.004 for i in range(100)]
        self.assertTrue(ta.bollinger_squeeze(shrinking)["active"])
        self.assertFalse(ta.bollinger_squeeze([0.1 + i * 0.004 for i in range(100)])["active"])

    def test_vwap_resets_every_utc_day(self):
        day = ta.DAY_MS
        bars = [{"t": 0, "h": 11, "l": 9, "c": 10, "v": 1}, {"t": STEP, "h": 21, "l": 19, "c": 20, "v": 3},
                {"t": day, "h": 31, "l": 29, "c": 30, "v": 5}]
        out = ta.vwap_daily(bars)
        self.assertAlmostEqual(out[1], (10 * 1 + 20 * 3) / 4)
        self.assertAlmostEqual(out[2], 30.0)
        self.assertAlmostEqual(ta.vwap_rolling(bars, 2)[2], (20 * 3 + 30 * 5) / 8)

    def test_relative_volume(self):
        out = ta.rvol([1.0] * 20 + [3.0])
        self.assertAlmostEqual(out[-1], 3.0)
        self.assertIsNone(out[0])


class PatternTests(unittest.TestCase):
    def detect(self, last_bars, trend):
        f = with_context(last_bars, trend)
        found = ta.candle_patterns(f, 2.0, [None] * len(f), ages=(0,))
        return {p["key"]: p for p in found}

    def test_bullish_and_bearish_engulfing(self):
        bull = self.detect([fbar(101, 101.2, 98.8, 99), fbar(98.5, 102.3, 98.3, 102)], 1.0)
        self.assertIn("bullish_engulfing", bull)
        bear = self.detect([fbar(99, 101.2, 98.8, 101), fbar(101.5, 101.7, 97.7, 98)], -1.0)
        self.assertIn("bearish_engulfing", bear)
        self.assertNotIn("bullish_engulfing", bear)

    def test_hammer_and_shooting_star_need_context(self):
        self.assertIn("hammer", self.detect([fbar(100, 100.5, 97, 100.4)], -1.0))
        self.assertIn("shooting_star", self.detect([fbar(100.4, 103.9, 99.9, 100)], 1.0))
        self.assertNotIn("hammer", self.detect([fbar(100, 100.5, 97, 100.4)], 1.0))
        self.assertNotIn("shooting_star", self.detect([fbar(100.4, 103.9, 99.9, 100)], -1.0))

    def test_doji_and_morning_star(self):
        self.assertIn("doji", self.detect([fbar(100, 101.5, 98.5, 100.05)], 1.0))
        star = self.detect([fbar(105, 105.3, 101.8, 102), fbar(101.8, 102.0, 101.4, 101.6),
                            fbar(102, 104.3, 101.9, 104)], 1.0)
        self.assertIn("morning_star", star)
        self.assertEqual(star["morning_star"]["type"], "bullish")

    def test_swing_points_and_structure(self):
        highs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 4, 12, 4, 3, 2, 1, 2, 3, 4, 5, 6.0]
        lows = [h - 0.5 for h in highs]
        swing_highs, swing_lows = ta.swing_points(highs, lows, 3)
        self.assertEqual([i for i, _ in swing_highs], [3, 10])
        self.assertEqual(ta.market_structure([(1, 10), (5, 12)], [(2, 5), (6, 7)], 1.0)["trend"], "uptrend")
        self.assertEqual(ta.market_structure([(1, 12), (5, 10)], [(2, 7), (6, 5)], 1.0)["trend"], "downtrend")
        self.assertEqual(ta.market_structure([(1, 12), (5, 10)], [(2, 5), (6, 7)], 1.0)["trend"], "range")

    def test_support_resistance_zones_surround_price(self):
        closes = [100 + 3 * math.sin(i / 5.0) for i in range(300)]
        f = ta.to_floats(make_bars(closes))
        sh, sl = ta.swing_points([b["h"] for b in f], [b["l"] for b in f], 5)
        supports, resistances = ta.support_resistance(f, sh, sl, 100.0, 0.5, 300)
        self.assertTrue(all(z["level"] < 100 for z in supports))
        self.assertTrue(all(z["level"] > 100 for z in resistances))
        self.assertTrue(resistances and supports)
        self.assertGreater(resistances[0]["touches"], 1)

    def test_double_bottom_and_rolling_breakout(self):
        points = [(0, 110), (25, 100), (40, 106), (55, 100.2), (70, 104.5)]
        closes = []
        for (i0, p0), (i1, p1) in zip(points, points[1:]):
            closes += [p0 + (p1 - p0) * k / (i1 - i0) for k in range(i1 - i0)]
        closes.append(points[-1][1])
        f = ta.to_floats(make_bars(closes, spread=0.1))
        atr_value = ta.last(ta.atr([b["h"] for b in f], [b["l"] for b in f], [b["c"] for b in f]))
        sh, sl = ta.swing_points([b["h"] for b in f], [b["l"] for b in f], 3)
        patterns = ta.chart_patterns(f, atr_value, 104.0, 103.0, sh, sl, STEP)
        double = next((p for p in patterns if p["key"] == "double_bottom"), None)
        self.assertIsNotNone(double)
        self.assertEqual(double["status"], "forming")
        flat = [100.0] * 80 + [100.8]
        g = ta.to_floats(make_bars(flat, spread=0.05))
        found = ta.chart_patterns(g, 0.3, 100.0, 99.0, [], [], STEP)
        self.assertTrue(any(p["key"].startswith("rolling_") for p in found))

    def test_liquidity_sweep_detection(self):
        closes = [100.0] * 40
        f = ta.to_floats(make_bars(closes, spread=0.1))
        # Create a clear swing low at index 15
        f[15]["l"] = 95.0
        f[15]["c"] = 96.0
        # And a bullish liquidity sweep on the last bar (wick below 95, close above 95)
        f[-1]["l"] = 94.2
        f[-1]["c"] = 96.5
        sh, sl = ta.swing_points([b["h"] for b in f], [b["l"] for b in f], 5)
        sweeps = ta.liquidity_sweeps(f, len(f) - 1, sh, sl, 1.0, 5)
        bull = next((s for s in sweeps if s["type"] == "bullish"), None)
        self.assertIsNotNone(bull)
        self.assertEqual(bull["level"], 95.0)
        self.assertAlmostEqual(bull["sweep_price"], 94.2)


class AdvisorTests(unittest.TestCase):
    def test_rating_thresholds(self):
        self.assertEqual([rating_for(s) for s in (100, 75, 74, 60, 59, 40, 39, 0)],
                         ["STRONG_BUY", "STRONG_BUY", "BUY_WATCH", "BUY_WATCH", "NEUTRAL",
                          "NEUTRAL", "AVOID", "AVOID"])

    def test_insufficient_data(self):
        bars = make_bars(random_walk(30, 0.001))
        result = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=after(bars))
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["score"])

    def test_levels_follow_the_rules_for_an_uptrend(self):
        closes = [100 + 0.04 * i + 3 * math.sin(i / 7.0) for i in range(420)]
        volumes = [10 + 8 * (i % 5 == 0) for i in range(420)]
        bars = make_bars(closes, volumes=volumes)
        result = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=after(bars), cost_rate=0.0005)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(0 <= result["score"] <= 100)
        self.assertEqual(result["rating"], rating_for(result["score"]))
        if result["entry"] is not None:
            entry, stop, t1, t2 = (result[k] for k in ("entry", "stop", "t1", "t2"))
            self.assertTrue(stop < entry < t1 < t2)
            risk = entry - stop
            self.assertGreaterEqual((t1 - entry) / risk, Decimal("1.45"))
            self.assertLess(result["stop_pct"], 0)
            self.assertAlmostEqual(result["t1_pct"], float((t1 / entry - 1) * 100), places=9)
            for value in (entry, stop, t1, t2):
                self.assertEqual(value % Decimal("0.01"), 0)
        if result["rating"] == "STRONG_BUY":
            self.assertTrue(all(result["conditions"].values()))

    def test_overbought_rsi_is_avoided(self):
        closes = [100 * (1.002 ** i) for i in range(300)]
        bars = make_bars(closes, spread=0.05)
        result = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=after(bars))
        self.assertGreater(result["indicators"]["rsi"], 75)
        self.assertEqual(result["rating"], "AVOID")
        self.assertLessEqual(result["score"], 39)
        self.assertIn("RSI", " ".join(result["avoid"]))

    def test_downtrend_is_avoided(self):
        closes = [150 - 0.06 * i + 3 * math.sin(i / 6.0) for i in range(400)]
        bars = make_bars(closes)
        result = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=after(bars))
        self.assertEqual(result["structure"]["trend"], "downtrend")
        self.assertEqual(result["rating"], "AVOID")

    def test_costs_larger_than_target_cap_the_rating(self):
        closes = [100 + 0.04 * i + 3 * math.sin(i / 7.0) for i in range(420)]
        bars = make_bars(closes)
        result = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=after(bars), cost_rate=0.05)
        self.assertLessEqual(result["score"], 59)
        self.assertFalse(result["actionable"])

    def test_forming_candle_sets_price_but_not_indicators(self):
        closes = [100 + 3 * math.sin(i / 9.0) for i in range(300)]
        bars = make_bars(closes)
        now = after(bars) - datetime.timedelta(seconds=30)
        with_live = analyze_setup(bars, "BTCUSDT", "5m", FILTERS, now=now)
        without = analyze_setup(bars[:-1], "BTCUSDT", "5m", FILTERS, now=now)
        self.assertTrue(with_live["forming"])
        self.assertEqual(with_live["price"], bars[-1]["close"])
        self.assertEqual(with_live["indicators"]["rsi"], without["indicators"]["rsi"])

    def test_build_plan_accepts_explicit_target(self):
        plan = build_plan("100.004", "99", 2, FILTERS, target="101.505")
        self.assertEqual(plan["target"], Decimal("101.50"))
        self.assertAlmostEqual(float(plan["reward_risk"]), 1.5, places=2)
        with self.assertRaises(TradingError):
            build_plan("100", "99", 2, FILTERS, target="99.5")


class MLModelTests(unittest.TestCase):
    def test_train_and_predict_win_probability_and_top_factors(self):
        closes = [100 + 0.02 * i + 2.5 * math.sin(i / 8.0) + 1.2 * math.cos(i / 3.0)
                  for i in range(1100)]
        volumes = [10.0 + (i % 7) * 2.0 for i in range(1100)]
        bars = make_bars(closes, spread=0.25, volumes=volumes)
        now = after(bars)
        cfg = {"ml": {"min_train_samples": 300, "n_estimators": 50, "horizon_bars": 24}}
        bundle = ml_model.train_model(bars, "5m", now=now, config=cfg)
        self.assertGreaterEqual(bundle["samples"], 300)
        self.assertEqual(len(bundle["importance"]), len(ml_model.FEATURE_KEYS))
        pred = ml_model.predict_setup(bundle, bars, now=now)
        self.assertTrue(0.0 <= pred["probability"] <= 1.0)
        self.assertAlmostEqual(pred["breakeven"], 1.8 / (1.8 + 4.5), places=6)
        self.assertAlmostEqual(pred["required_threshold"], 1.25 * (1.8 / (1.8 + 4.5)), places=6)
        self.assertEqual(len(pred["top_factors"]), 3)
        self.assertTrue(pred["stop"] < pred["entry"] < pred["target"])

    def test_insufficient_samples_raises_model_error(self):
        bars = make_bars(random_walk(120, 0.002))
        with self.assertRaises(ml_model.ModelError):
            ml_model.train_model(bars, "5m", now=after(bars))

    def test_mtf_filter_caps_advisor_and_ml_probability_when_misaligned(self):
        closes = [100 + 0.03 * i + 2.5 * math.sin(i / 8.0) + 1.1 * math.cos(i / 3.0)
                  for i in range(1100)]
        bars = make_bars(closes, spread=0.25)
        now = after(bars)
        end_ms = int(now.timestamp() * 1000)
        # Build rising 1h/4h bars and falling 1h/4h bars
        h1_step, h4_step = 3_600_000, 14_400_000
        h1_start = START - 70 * h1_step
        h4_start = START - 70 * h4_step
        bull_htf = {
            "1h": make_bars([100 + 0.5 * i for i in range(200)], start=h1_start, step=h1_step),
            "4h": make_bars([100 + 1.5 * i for i in range(120)], start=h4_start, step=h4_step),
        }
        mixed_htf = {
            "1h": make_bars([100 + 0.5 * i for i in range(200)], start=h1_start, step=h1_step),
            "4h": make_bars([300 - 1.5 * i for i in range(120)], start=h4_start, step=h4_step),
        }
        bear_htf = {
            "1h": make_bars([300 - 0.5 * i for i in range(200)], start=h1_start, step=h1_step),
            "4h": make_bars([300 - 1.5 * i for i in range(120)], start=h4_start, step=h4_step),
        }
        bull_state = mtf.evaluate(mtf.build_series(bull_htf), end_ms)
        mixed_state = mtf.evaluate(mtf.build_series(mixed_htf), end_ms)
        bear_state = mtf.evaluate(mtf.build_series(bear_htf), end_ms)
        self.assertTrue(bull_state["aligned"])
        self.assertEqual(bull_state["size_multiplier"], 1.0)
        self.assertFalse(mixed_state["aligned"])
        self.assertTrue(mixed_state["allow_long"])
        self.assertEqual(mixed_state["size_multiplier"], 0.5)
        self.assertFalse(bear_state["aligned"])
        self.assertFalse(bear_state["allow_long"])
        self.assertEqual(bear_state["size_multiplier"], 0.0)

        # Advisor caps score at <= 59 when 1h is bearish, but allows half-size when 1h is bullish & 4h bearish
        adv_bear = analyze_setup(bars[-450:], "BTCUSDT", "5m", FILTERS, now=now, htf=bear_state)
        self.assertLessEqual(adv_bear["score"], 59)
        self.assertFalse(adv_bear["actionable"])
        self.assertEqual(adv_bear["position_scale"], 0.0)

        adv_mixed = analyze_setup(bars[-450:], "BTCUSDT", "5m", FILTERS, now=now, htf=mixed_state)
        self.assertEqual(adv_mixed["position_scale"], 0.5)

        # ML model caps probability at base_rate when 1h is bearish, but allows 0.5x when 1h is bullish
        cfg = {"ml": {"min_train_samples": 250, "n_estimators": 40, "horizon_bars": 24, "auto_threshold": 0.55}}
        bundle = ml_model.train_model(bars, "5m", now=now, config=cfg, htf_bars=bull_htf)
        pred_bear = ml_model.predict_setup(bundle, bars, now=now, htf_bars=bear_htf)
        self.assertFalse(pred_bear["mtf_aligned"])
        self.assertEqual(pred_bear["size_multiplier"], 0.0)
        self.assertLessEqual(pred_bear["probability"], pred_bear["base_rate"] + 1e-9)
        self.assertFalse(pred_bear["signal"])

        pred_mixed = ml_model.predict_setup(bundle, bars, now=now, htf_bars=mixed_htf)
        self.assertEqual(pred_mixed["size_multiplier"], 0.5)
        self.assertTrue(pred_mixed["mtf_allow_long"])

        # On 1h trading interval, bearish 4h blocks trades completely (size_multiplier == 0.0)
        mixed_state_1h = mtf.evaluate(mtf.build_series(mixed_htf), end_ms, trading_interval="1h")
        self.assertFalse(mixed_state_1h["allow_long"])
        self.assertEqual(mixed_state_1h["size_multiplier"], 0.0)
        adv_mixed_1h = analyze_setup(bars[-450:], "BTCUSDT", "1h", FILTERS, now=now, htf=mixed_state_1h)
        self.assertEqual(adv_mixed_1h["position_scale"], 0.0)
        self.assertFalse(adv_mixed_1h["actionable"])


class BotAndBacktestTests(unittest.TestCase):
    def test_walk_forward_backtest_runs_and_produces_report(self):
        # 2400 x 15m candles = 25 days; train on first 8 days, walk-forward test on rest
        closes = [100 + 0.015 * i + 2.8 * math.sin(i / 7.0) + 1.3 * math.cos(i / 3.0)
                  for i in range(2400)]
        bars = make_bars(closes, spread=0.35, step=900_000)
        now = after(bars)
        h1_step, h4_step = 3_600_000, 14_400_000
        htf_bars = {
            "1h": make_bars([100 + 0.4 * i for i in range(720)], start=START - 80 * h1_step, step=h1_step),
            "4h": make_bars([100 + 1.2 * i for i in range(240)], start=START - 80 * h4_step, step=h4_step),
        }
        config = load_config()
        config = dict(config, ml=dict(config["ml"], min_train_samples=250, n_estimators=30, horizon_bars=24))
        report = bot.run_backtest(
            "BTCUSDT", interval="15m", train_days=8.0, retrain_days=5.0,
            threshold=0.15, require_mtf=True, config=config, now=now,
            bars=bars, htf_bars=htf_bars, filters=FILTERS)
        self.assertGreater(report["strategy"]["models_trained"], 0)
        self.assertGreater(report["trades"], 0)
        self.assertIn("total_profit_usd", report)
        self.assertIn("win_rate", report)
        self.assertIn("breakeven_win_rate", report)
        self.assertAlmostEqual(report["breakeven_win_rate"], 1.8 / (1.8 + 4.5), places=6)
        self.assertAlmostEqual(
            report["strategy"]["required_ev_threshold"], 1.25 * (1.8 / (1.8 + 4.5)), places=6)
        self.assertIn("max_drawdown_usd", report)
        formatted = bot.format_report(report)
        self.assertIn("Total Profit", formatted)
        self.assertIn("Win Rate", formatted)
        self.assertIn("Break-even Win Rate", formatted)
        self.assertIn("Dinamikus Küszöb (Required +EV Threshold)", formatted)
        self.assertIn("Max Drawdown", formatted)

    def test_paper_bot_opens_trade_when_probability_and_mtf_pass(self):
        closes = [100 + 0.02 * i + 2.6 * math.sin(i / 7.0) + 1.2 * math.cos(i / 3.0)
                  for i in range(1100)]
        bars = make_bars(closes, spread=0.3)
        now = after(bars)
        h1_step, h4_step = 3_600_000, 14_400_000
        bull_htf = {
            "1h": make_bars([100 + 0.5 * i for i in range(240)], start=START - 80 * h1_step, step=h1_step),
            "4h": make_bars([100 + 1.5 * i for i in range(120)], start=START - 80 * h4_step, step=h4_step),
        }

        class FakeClient:
            def get_klines(self, symbol, interval, limit=500, start_time=None, end_time=None, include_open=False):
                if interval in ("1h", "4h"):
                    return {"symbol": symbol, "interval": interval, "market": "spot",
                            "bars": bull_htf[interval], "filters": FILTERS}
                return {"symbol": symbol, "interval": interval, "market": "spot",
                        "bars": bars, "filters": FILTERS}

        with tempfile.TemporaryDirectory() as tmp:
            paper_path = os.path.join(tmp, "paper_trades.json")
            config = load_config()
            config = dict(
                config,
                symbols=[{"symbol": "BTCUSDT", "base": "BTC"}],
                ml=dict(config["ml"], min_train_samples=250, n_estimators=30, horizon_bars=18, auto_threshold=0.05),
                storage=dict(config["storage"], paper_trades=paper_path),
            )
            pbot = bot.PaperBot(config=config, client=FakeClient(), interval="5m",
                                threshold=0.05, symbols=["BTCUSDT"])
            opened = pbot.cycle(now=now)
            self.assertEqual(len(opened), 1)
            self.assertEqual(opened[0]["symbol"], "BTCUSDT")
            self.assertEqual(opened[0]["status"], "open")
            # Second cycle on the same closed candle does not open a duplicate position
            self.assertEqual(pbot.cycle(now=now), [])
            saved = list_trades(config=config, path=paper_path)
            self.assertEqual(len(saved), 1)

            # Verify half-size position when 1h is bullish and 4h is bearish
            mixed_htf = {
                "1h": bull_htf["1h"],
                "4h": make_bars([300 - 1.5 * i for i in range(120)], start=START - 80 * h4_step, step=h4_step),
            }

            class MixedClient:
                def get_klines(self, symbol, interval, limit=500, start_time=None, end_time=None, include_open=False):
                    if interval in ("1h", "4h"):
                        return {"symbol": symbol, "interval": interval, "market": "spot",
                                "bars": mixed_htf[interval], "filters": FILTERS}
                    return {"symbol": symbol, "interval": interval, "market": "spot",
                            "bars": bars, "filters": FILTERS}

            paper_path_half = os.path.join(tmp, "paper_half.json")
            cfg_half = dict(config, storage=dict(config["storage"], paper_trades=paper_path_half))
            pbot_half = bot.PaperBot(config=cfg_half, client=MixedClient(), interval="5m",
                                     threshold=0.05, symbols=["BTCUSDT"])
            opened_half = pbot_half.cycle(now=now)
            self.assertEqual(len(opened_half), 1)
            self.assertAlmostEqual(
                float(opened_half[0]["quantity"]), float(opened[0]["quantity"]) * 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
