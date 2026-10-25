"""UTC-midnight opening-range breakout evaluation on Binance candles."""

import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import evaluate_breakout


def _day_start_ms(now):
    return int(datetime.datetime.combine(now.date(), datetime.time(),
                                         datetime.timezone.utc).timestamp() * 1000)


def _day_limit(interval):
    return 1441 if interval == "1m" else 289


def evaluate(chart, now_utc=None, or_minutes=15):
    return evaluate_breakout(chart["bars"], chart["symbol"], chart["interval"],
                             now_utc, or_minutes)


def watch(symbol, interval=None, every=None, client=None, config=None, until=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    interval = interval or config["default_interval"]
    every = max(5, int(every or config.get("refresh_seconds", 30)))
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        result = client.get_klines(symbol, interval, limit=_day_limit(interval),
                                   start_time=_day_start_ms(now),
                                   end_time=int(now.timestamp() * 1000), include_open=True)
        signal = evaluate_breakout(result["bars"], symbol, interval, now,
                                   config["opening_range_minutes"])
        if signal["status"] == "triggered":
            return signal
        if until and now >= until:
            return signal
        time.sleep(every)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check a crypto breakout")
    parser.add_argument("symbol")
    parser.add_argument("--interval", choices=["1m", "5m"], default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.symbol.upper() not in {coin["symbol"] for coin in cfg["symbols"]}:
        parser.error("symbol must be configured in config.json")
    client = BinanceClient(cfg)
    now = datetime.datetime.now(datetime.timezone.utc)
    interval = args.interval or cfg["default_interval"]
    result = client.get_klines(args.symbol.upper(), interval, limit=_day_limit(interval),
                               start_time=_day_start_ms(now),
                               end_time=int(now.timestamp() * 1000), include_open=True)
    signal = evaluate_breakout(result["bars"], args.symbol.upper(),
                               interval, now,
                               cfg["opening_range_minutes"])
    print(signal)
    raise SystemExit(0 if signal["status"] == "triggered" else 1)
