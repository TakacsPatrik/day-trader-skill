"""Scan the configured crypto watchlist with Binance public candle data and SMC setup advisor."""

import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import analyze_setup
import mtf


def scan(client=None, config=None, now=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    interval = config["default_interval"]
    risk = config.get("risk", {})
    cost_rate = float(risk.get("fee_rate", 0)) + float(risk.get("slippage_rate", 0))
    htf_cache = mtf.HTFCache(client, ttl=120)
    now_ms = int(now.timestamp() * 1000)
    results = []
    for coin in config["symbols"]:
        symbol = coin["symbol"]
        try:
            data = client.get_klines(symbol, interval, limit=500, include_open=True)
            try:
                htf_state = htf_cache.state(symbol, now_ms)
            except Exception:
                htf_state = None
            advice = analyze_setup(data["bars"], symbol, data["interval"], data["filters"],
                                   now=now, cost_rate=cost_rate, htf=htf_state)
            price = data["bars"][-1]["close"] if data["bars"] else None
            results.append({"symbol": symbol, "price": price, "market": data["market"],
                            "advice": advice, "error": None})
        except Exception as exc:
            results.append({"symbol": symbol, "price": None, "market": None,
                            "advice": None, "error": str(exc)})
    return results


def main():
    data = scan()
    print(json.dumps([{"symbol": row["symbol"],
                       "price": format(row["price"], "f") if row["price"] is not None else None,
                       "market": row["market"],
                       "rating": row["advice"]["rating"] if row["advice"] else "error",
                       "score": row["advice"]["score"] if row["advice"] else None,
                       "mtf": row["advice"]["mtf"]["label"] if row["advice"] and row["advice"].get("mtf") else None,
                       "error": row["error"]} for row in data], indent=2))
    return 0 if any(row["advice"] for row in data) else 1


if __name__ == "__main__":
    raise SystemExit(main())
