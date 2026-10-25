"""Build tick-aligned long-side crypto paper-trade plans."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import build_plan


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Create a crypto paper-trade plan")
    parser.add_argument("symbol")
    parser.add_argument("--entry", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--target-r", default="2")
    args = parser.parse_args(argv)
    config = load_config()
    client = BinanceClient(config)
    info = client.get_klines(args.symbol.upper(), config["default_interval"], limit=1)
    plan = build_plan(args.entry, args.stop, args.target_r, info["filters"])
    print(json.dumps({key: format(value, "f") if hasattr(value, "as_tuple") else value
                      for key, value in plan.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
