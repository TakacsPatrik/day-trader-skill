"""Scan the configured crypto watchlist with Binance public candle data."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import evaluate_breakout


def scan(client=None, config=None, now=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    day_start = int(datetime.datetime.combine(now.date(), datetime.time(),
                                              datetime.timezone.utc).timestamp() * 1000)
    interval = config["default_interval"]
    limit = 1441 if interval == "1m" else 289
    results = []
    for coin in config["symbols"]:
        symbol = coin["symbol"]
        try:
            data = client.get_klines(symbol, interval, limit=limit, start_time=day_start,
                                     end_time=int(now.timestamp() * 1000), include_open=True)
            signal = evaluate_breakout(data["bars"], symbol, data["interval"], now,
                                       config["opening_range_minutes"])
            price = data["bars"][-1]["close"] if data["bars"] else None
            results.append({"symbol": symbol, "price": price, "market": data["market"],
                            "signal": signal, "error": None})
        except Exception as exc:
            results.append({"symbol": symbol, "price": None, "market": None,
                            "signal": None, "error": str(exc)})
    return results


def main():
    data = scan()
    print(json.dumps([{"symbol": row["symbol"],
                       "price": format(row["price"], "f") if row["price"] is not None else None,
                       "market": row["market"],
                       "status": row["signal"]["status"] if row["signal"] else "error",
                       "error": row["error"]} for row in data], indent=2))
    return 0 if any(row["signal"] for row in data) else 1


if __name__ == "__main__":
    raise SystemExit(main())
