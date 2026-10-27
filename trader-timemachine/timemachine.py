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
import mtf


def run(symbol, interval=None, config=None, client=None, now=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    interval = interval or config.get("default_interval", "15m")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    days = int(config.get("history_days", 30))
    start = int((now - datetime.timedelta(days=days)).timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    history = client.get_klines(symbol, interval, limit=20_000,
                                start_time=start, end_time=end)
    try:
        htf_bars = mtf.fetch_htf(client, symbol, start, end)
    except Exception:
        htf_bars = None
    bundle = ml_model.train_model(history["bars"], interval, now=now, config=config, htf_bars=htf_bars)
    prediction = ml_model.predict_setup(bundle, history["bars"], now=now, htf_bars=htf_bars)
    return {
        "symbol": symbol,
        "market": history["market"],
        "interval": interval,
        "win_probability": prediction["probability"],
        "raw_probability": prediction["raw_probability"],
        "mtf_aligned": prediction["mtf_aligned"],
        "mtf": prediction["mtf"],
        "signal": prediction["signal"],
        "base_rate": prediction["base_rate"],
        "edge_pp": prediction["edge_pp"],
        "top_factors": prediction["top_factors"],
        "metrics": bundle["metrics"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate setup win probability with RandomForestClassifier")
    parser.add_argument("symbol")
    parser.add_argument("--interval", choices=["1m", "5m", "15m", "1h"], default="1h")
    args = parser.parse_args()
    result = run(args.symbol.upper(), args.interval)
    print(json.dumps(result, indent=2))
