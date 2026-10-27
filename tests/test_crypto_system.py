import datetime
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, BinanceError, load_config
from crypto_engine import (TradingError, analyze_setup,
                           size_position, validate_risk)
from paper_trading import auto_close_trades, close_trade, list_trades, record_entry, unrealized_pnl


def module_from_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


news = module_from_file("crypto_news_test", os.path.join("trader-news", "news.py"))
alerts = module_from_file("crypto_alerts_test", os.path.join("trader-alerts", "alerts.py"))


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = {}
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()

    def json(self):
        if isinstance(self.body, bytes):
            return json.loads(self.body.decode())
        return self.body

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("HTTP %s" % self.status_code)


class FakeBinanceSession:
    def __init__(self, rows=None, absent_spot=False):
        self.rows = rows or []
        self.absent_spot = absent_spot
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/api/v3/klines") and self.absent_spot:
            return FakeResponse({"code": -1121, "msg": "Invalid symbol."}, 400)
        if url.endswith("/klines"):
            start = int((params or {}).get("startTime", 0))
            limit = int((params or {}).get("limit", 500))
            rows = [row for row in self.rows if row[0] >= start]
            return FakeResponse(rows[:limit])
        if url.endswith("/exchangeInfo"):
            return FakeResponse({"symbols": [{
                "symbol": (params or {}).get("symbol"),
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"}
                ]}]})
        raise AssertionError("Unexpected URL " + url)


class BinanceClientTests(unittest.TestCase):
    def test_default_config_is_exact_watchlist(self):
        config = load_config()
        self.assertEqual([item["symbol"] for item in config["symbols"]],
                         ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "GRASSUSDT"])
        self.assertEqual(config["intervals"], ["1m", "5m", "15m", "1h"])
        self.assertEqual(config["default_interval"], "1h")

    def test_spot_unknown_symbol_falls_back_to_futures(self):
        now = 1_000_000
        row = [900_000, "10", "11", "9", "10.5", "4", 959_999]
        session = FakeBinanceSession([row], absent_spot=True)
        config = load_config()
        config["symbols"] = [item for item in config["symbols"] if item["symbol"] == "GRASSUSDT"]
        client = BinanceClient(config, session, clock=lambda: now, sleeper=lambda _: None)
        result = client.get_klines("GRASSUSDT", "1m", limit=1)
        self.assertEqual(result["market"], "futures")
        self.assertEqual(result["bars"][0]["close"], Decimal("10.5"))
        self.assertTrue(any(url.endswith("/api/v3/klines") for url, _ in session.calls))
        self.assertTrue(any(url.endswith("/fapi/v1/klines") for url, _ in session.calls))

    def test_pagination_and_forming_candle_filter(self):
        rows = []
        for index in range(1001):
            open_ms = 1_000_000 + index * 60_000
            rows.append([open_ms, "1", "2", "0.5", "1.5", "10", open_ms + 59_999])
        session = FakeBinanceSession(rows)
        config = load_config()
        config["symbols"] = [item for item in config["symbols"] if item["symbol"] == "BTCUSDT"]
        config["binance"]["page_limit"] = 1000
        client = BinanceClient(config, session, clock=lambda: 1_000_000 + 1001 * 60_000,
                               sleeper=lambda _: None)
        result = client.get_klines("BTCUSDT", "1m", limit=1001, start_time=1_000_000)
        self.assertEqual(len(result["bars"]), 1001)
        self.assertEqual(sum(url.endswith("/api/v3/klines") for url, _ in session.calls), 2)
        with self.assertRaises(BinanceError):
            client.get_klines("NOPEUSDT", "1m", limit=1)


def candle(open_ms, interval_ms, o, h, low, close, market="spot"):
    return {"open_time": open_ms, "close_time": open_ms + interval_ms - 1,
            "open": Decimal(str(o)), "high": Decimal(str(h)),
            "low": Decimal(str(low)), "close": Decimal(str(close)),
            "volume": Decimal("1"), "market": market}


class CryptoEngineTests(unittest.TestCase):
    def test_fractional_sizing_uses_decimal_step_and_usd_risk(self):
        filters = {"step_size": "0.0001", "min_qty": "0.0001",
                   "tick_size": "0.01", "min_notional": "5"}
        result = size_position("60000", "58000", "5", filters)
        self.assertEqual(result["quantity"], Decimal("0.0025"))
        self.assertEqual(result["estimated_loss_usd"], Decimal("5.0000"))
        sol = size_position("150", "149", "15.4",
                            {"step_size": "0.1", "min_qty": "0.1",
                             "tick_size": "0.01", "min_notional": "5"})
        self.assertEqual(sol["quantity"], Decimal("15.4"))

    def test_risk_check_blocks_excess_loss(self):
        result = validate_risk({"entry": "100", "stop": "98"}, "6",
                               {"risk_per_trade_usd": 10, "daily_loss_limit_usd": 20})
        self.assertFalse(result["passed"])
        self.assertIn("per-trade", result["problems"][0])

class NewsAndAlertTests(unittest.TestCase):
    def test_rss_only_matches_configured_coin_aliases(self):
        xml = b"""<rss><channel>
          <item><title>Bitcoin rises as market turns</title><guid>a</guid>
          <link>https://example.test/a</link><pubDate>Wed, 01 Jan 2025 00:00:00 +0000</pubDate></item>
          <item><title>Company reports earnings</title><guid>b</guid></item>
        </channel></rss>"""
        coins = load_config()["symbols"]
        items = news.parse_feed(xml, "Test", coins)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["coins"], ["BTCUSDT"])

    def test_fear_greed_parser_and_bad_response(self):
        class Session:
            def __init__(self, body):
                self.body = body

            def get(self, *args, **kwargs):
                return FakeResponse(self.body)

        record = {"data": [{"value": "67", "value_classification": "Greed",
                            "timestamp": "1760000000"}]}
        self.assertEqual(news.fetch_fear_greed("https://test", Session(record))["value"], 67)
        with self.assertRaises(news.NewsError):
            news.fetch_fear_greed("https://test", Session({"data": []}))

    def test_alerts_are_24_7_and_ring_once(self):
        now = datetime.datetime(2026, 1, 1, 2, 0, tzinfo=datetime.timezone.utc)
        rules = [{"id": "target", "symbol": "BTCUSDT", "type": "target", "price": "100"}]
        quote = {"BTCUSDT": {"price": Decimal("101"), "timestamp": now.isoformat()}}
        rings, state = alerts.evaluate(rules, quote, {}, now)
        self.assertEqual(len(rings), 1)
        self.assertTrue(alerts.market_status(now)["open"])
        self.assertEqual(alerts.evaluate(rules, quote, state, now)[0], [])

    def test_percent_and_daily_utc_alerts(self):
        now = datetime.datetime(2026, 1, 1, 2, 0, tzinfo=datetime.timezone.utc)
        quote = {"BTCUSDT": {"price": Decimal("105"), "reference_price": Decimal("100")}}
        rules = [{"id": "move", "symbol": "BTCUSDT", "type": "pct_move",
                  "pct": "4", "direction": "up"},
                 {"id": "review", "type": "time", "at": "01:00"}]
        rings, state = alerts.evaluate(rules, quote, {}, now)
        self.assertEqual({ring["id"].split(":")[0] for ring in rings}, {"move", "review"})
        self.assertEqual(alerts.evaluate(rules, quote, state, now)[0], [])


    def test_alert_rules_persist_to_json_file(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "alerts.json")
            self.assertEqual(alerts.load_rules(config, path), [])
            rule = alerts.add_rule({"symbol": "btcusdt", "type": "target", "price": "70000.50"},
                                   config=config, path=path)
            self.assertEqual(rule["symbol"], "BTCUSDT")
            reloaded = alerts.load_rules(config, path)
            self.assertEqual([item["id"] for item in reloaded], [rule["id"]])
            self.assertEqual(reloaded[0]["price"], "70000.50")
            with self.assertRaises(alerts.AlertError):
                alerts.add_rule({"symbol": "DOGEUSDT", "type": "above", "price": "1"},
                                config=config, path=path)
            with self.assertRaises(alerts.AlertError):
                alerts.add_rule({"symbol": "BTCUSDT", "type": "above", "price": "-1"},
                                config=config, path=path)
            self.assertEqual(alerts.delete_rule(rule["id"], config=config, path=path), [])
            self.assertEqual(alerts.load_rules(config, path), [])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{broken")
            with self.assertRaises(alerts.AlertError):
                alerts.load_rules(config, path)


class PaperTradingTests(unittest.TestCase):
    def test_paper_entry_exit_and_pnl(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paper.json")
            trade = record_entry("BTCUSDT", "long", "0.0025", "60000",
                                 "58000", "64000", config=config, path=path,
                                 trade_id="paper-1")
            self.assertEqual(trade["quantity"], "0.0025")
            self.assertEqual(unrealized_pnl(trade, "61000"), Decimal("2.5000"))
            self.assertEqual(unrealized_pnl(trade, "61000", fee_rate="0.001", slippage_rate="0.0005"),
                             Decimal("2.046250000"))
            closed = close_trade("paper-1", "61000", fee_rate="0.001", slippage_rate="0.0005",
                                 config=config, path=path)
            self.assertEqual(closed["status"], "closed")
            self.assertEqual(Decimal(closed["realized_pnl_usd"]), Decimal("2.046250000"))
            self.assertEqual(list_trades(config=config, path=path)[0]["id"], "paper-1")

    def test_auto_close_trades_hits_stop_and_target_with_slippage(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paper.json")
            record_entry("BTCUSDT", "long", "1", "100", "95", "110",
                         config=config, path=path, trade_id="t-stop")
            record_entry("ETHUSDT", "long", "2", "200", "190", "220",
                         config=config, path=path, trade_id="t-target")
            closed = auto_close_trades(
                {"BTCUSDT": {"price": "96", "high": "99", "low": "94.5"},
                 "ETHUSDT": {"price": "221", "high": "222", "low": "205"}},
                fee_rate="0.001", slippage_rate="0.0005", config=config, path=path)
            self.assertEqual(len(closed), 2)
            by_id = {t["id"]: t for t in list_trades(config=config, path=path)}
            self.assertEqual(by_id["t-stop"]["status"], "closed")
            self.assertEqual(by_id["t-stop"]["exit_reason"], "stop")
            self.assertEqual(Decimal(by_id["t-stop"]["exit_price"]), Decimal("95"))
            # 1 * (95 - 100) - 1 * (100 + 95) * 0.0015 = -5 - 0.2925 = -5.2925
            self.assertEqual(Decimal(by_id["t-stop"]["realized_pnl_usd"]), Decimal("-5.2925"))
            self.assertEqual(by_id["t-target"]["status"], "closed")
            self.assertEqual(by_id["t-target"]["exit_reason"], "target")
            self.assertEqual(Decimal(by_id["t-target"]["exit_price"]), Decimal("220"))
            # 2 * (220 - 200) - 2 * (200 + 220) * 0.0015 = 40 - 1.26 = 38.74
            self.assertEqual(Decimal(by_id["t-target"]["realized_pnl_usd"]), Decimal("38.74"))

    def test_trailing_stop_moves_to_breakeven_and_trails_profit(self):
        from paper_trading import compute_trailing_stop
        config = load_config()
        # Verify fee-buffered breakeven floor directly when stop_atr == trigger_atr (2.0)
        be_stop, be_moved = compute_trailing_stop(
            "100", "94", "106", atr_value="3", stop_atr="2.0",
            trigger_atr="2.0", breakeven_buffer="0.004")
        self.assertTrue(be_moved)
        self.assertEqual(be_stop, Decimal("100.400"))

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paper.json")
            # Entry 100, initial stop 91 (1.8 * ATR where ATR = 5), target 122.5 (4.5 * ATR), partial 112.5 (2.5 * ATR)
            record_entry("BTCUSDT", "long", "1", "100", "91", "122.5",
                         config=config, path=path, trade_id="t-trail", atr="5",
                         partial_target="112.5", partial_ratio="0.5")
            # Step 1: Price reaches +1.0 * ATR (105) -> does NOT trigger yet (needs +2.5 * ATR = 112.5)
            closed = auto_close_trades(
                {"BTCUSDT": {"price": "104.5", "high": "105", "low": "99"}},
                config=config, path=path)
            self.assertEqual(closed, [])
            trade = list_trades(config=config, path=path)[0]
            self.assertEqual(Decimal(trade["stop"]), Decimal("91"))
            self.assertFalse(trade["trailing_active"])
            self.assertFalse(trade["partial_closed"])

            # Step 2: Price climbs to 113 (+2.6 * ATR) -> Partial TP (50% @ 112.5) + Stop trails at max(100.4, 113 - 1.8*5) = 104
            closed = auto_close_trades(
                {"BTCUSDT": {"price": "112", "high": "113", "low": "105"}},
                config=config, path=path)
            self.assertEqual(closed, [])
            trade = list_trades(config=config, path=path)[0]
            self.assertEqual(Decimal(trade["stop"]), Decimal("104"))
            self.assertTrue(trade["trailing_active"])
            self.assertTrue(trade["partial_closed"])
            self.assertEqual(Decimal(trade["quantity"]), Decimal("0.5"))
            self.assertEqual(Decimal(trade["partial_pnl_usd"]), Decimal("6.25"))

            # Step 3: Candle wicks down to 103.5 -> closes remaining 0.5 qty at trailed stop 104 (+2.0) + partial (+6.25) = +8.25
            closed = auto_close_trades(
                {"BTCUSDT": {"price": "105", "high": "106", "low": "103.5"}},
                fee_rate="0", slippage_rate="0", config=config, path=path)
            self.assertEqual(len(closed), 1)
            self.assertEqual(Decimal(closed[0]["exit_price"]), Decimal("104"))
            self.assertEqual(Decimal(closed[0]["realized_pnl_usd"]), Decimal("8.25"))


if __name__ == "__main__":
    unittest.main()
