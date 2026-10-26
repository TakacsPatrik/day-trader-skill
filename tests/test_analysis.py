import datetime
import math
import os
import random
import sys
import unittest
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ml_model
import technical_analysis as ta
from crypto_engine import TradingError, analyze_setup, build_plan, rating_for

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
        self.assertEqual(len(pred["top_factors"]), 3)
        self.assertTrue(pred["stop"] < pred["entry"] < pred["target"])

    def test_insufficient_samples_raises_model_error(self):
        bars = make_bars(random_walk(120, 0.002))
        with self.assertRaises(ml_model.ModelError):
            ml_model.train_model(bars, "5m", now=after(bars))


if __name__ == "__main__":
    unittest.main()
