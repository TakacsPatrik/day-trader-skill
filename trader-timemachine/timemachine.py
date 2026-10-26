"""Local RandomForest win-probability evaluator for crypto setups."""

import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
import ml_model


def run(symbol, interval="5m", config=None, client=None, now=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    days = int(config.get("history_days", 30))
    start = int((now - datetime.timedelta(days=days)).timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    history = client.get_klines(symbol, interval, limit=20_000,
                                start_time=start, end_time=end)
    bundle = ml_model.train_model(history["bars"], interval, now=now, config=config)
    prediction = ml_model.predict_setup(bundle, history["bars"], now=now)
    return {
        "symbol": symbol,
        "market": history["market"],
        "interval": interval,
        "win_probability": prediction["probability"],
        "base_rate": prediction["base_rate"],
        "edge_pp": prediction["edge_pp"],
        "top_factors": prediction["top_factors"],
        "metrics": bundle["metrics"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate setup win probability with RandomForestClassifier")
    parser.add_argument("symbol")
    parser.add_argument("--interval", choices=["1m", "5m"], default="5m")
    args = parser.parse_args()
    result = run(args.symbol.upper(), args.interval)
    print(json.dumps(result, indent=2))
