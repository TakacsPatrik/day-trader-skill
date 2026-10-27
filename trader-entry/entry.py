"""SMC & setup-advisor evaluation on Binance candles."""

import datetime
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import analyze_setup
import mtf


def evaluate(chart, now_utc=None, filters=None, cost_rate=0, htf=None):
    return analyze_setup(chart["bars"], chart["symbol"], chart["interval"],
                         filters=filters or chart.get("filters"), now=now_utc,
                         cost_rate=cost_rate, htf=htf)


def watch(symbol, interval=None, every=None, client=None, config=None, until=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    interval = interval or config["default_interval"]
    every = max(5, int(every or config.get("refresh_seconds", 30)))
    risk = config.get("risk", {})
    cost_rate = float(risk.get("fee_rate", 0)) + float(risk.get("slippage_rate", 0))
    htf_cache = mtf.HTFCache(client, ttl=120)
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        result = client.get_klines(symbol, interval, limit=500, include_open=True)
        htf_state = htf_cache.state(symbol, int(now.timestamp() * 1000))
        advice = analyze_setup(result["bars"], symbol, interval, result["filters"],
                               now=now, cost_rate=cost_rate, htf=htf_state)
        if advice.get("actionable"):
            return advice
        if until and now >= until:
            return advice
        time.sleep(every)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a crypto setup with SMC, MTF & pattern analysis")
    parser.add_argument("symbol")
    parser.add_argument("--interval", choices=["1m", "5m", "15m", "1h"], default=None)
    args = parser.parse_args()
    cfg = load_config()
    if args.symbol.upper() not in {coin["symbol"] for coin in cfg["symbols"]}:
        parser.error("symbol must be configured in config.json")
    client = BinanceClient(cfg)
    now = datetime.datetime.now(datetime.timezone.utc)
    interval = args.interval or cfg["default_interval"]
    sym = args.symbol.upper()
    result = client.get_klines(sym, interval, limit=500, include_open=True)
    htf_state = mtf.HTFCache(client).state(sym, int(now.timestamp() * 1000))
    risk = cfg.get("risk", {})
    cost_rate = float(risk.get("fee_rate", 0)) + float(risk.get("slippage_rate", 0))
    advice = analyze_setup(result["bars"], sym, interval,
                           result["filters"], now=now, cost_rate=cost_rate, htf=htf_state)
    print(advice)
    raise SystemExit(0 if advice.get("actionable") else 1)
