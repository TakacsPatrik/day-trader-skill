"""Historical Binance-candle Monte Carlo simulator for crypto paper trades."""

import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import simulate_trade


def run(symbol, entry, stop, target, interval="5m", runs=10_000, seed=19,
        config=None, client=None, now=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    start = int((now - datetime.timedelta(days=int(config.get("history_days", 60)))).timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    history = client.get_klines(symbol, interval, limit=20_000,
                                start_time=start, end_time=end)
    result = simulate_trade(history["bars"], entry, stop, target, interval, runs, seed, now)
    result["symbol"] = symbol
    result["market"] = history["market"]
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 10,000 simulated crypto price paths")
    parser.add_argument("symbol")
    parser.add_argument("--entry", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--interval", choices=["1m", "5m"], default="5m")
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    result = run(args.symbol.upper(), args.entry, args.stop, args.target,
                 args.interval, args.runs, args.seed)
    result.pop("terminal_returns_pct", None)
    result.pop("sample_paths", None)
    print(json.dumps(result, indent=2))
