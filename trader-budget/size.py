"""Risk-based fractional crypto position sizing."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import BinanceClient, load_config
from crypto_engine import size_position


def size_trade(symbol, entry, stop, risk_usd=None, config=None, client=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    risk = config.get("risk", {})
    market_data = client.get_klines(symbol, config["default_interval"], limit=1)
    return size_position(entry, stop, risk_usd or risk.get("risk_per_trade_usd", 50),
                         market_data["filters"], risk.get("max_position_usd"),
                         risk.get("fee_rate", 0), risk.get("slippage_rate", 0))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Calculate a crypto quantity from USD risk")
    parser.add_argument("symbol")
    parser.add_argument("--entry", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--risk-usd", type=float)
    args = parser.parse_args(argv)
    result = size_trade(args.symbol.upper(), args.entry, args.stop, args.risk_usd)
    print(json.dumps({key: format(value, "f") if hasattr(value, "as_tuple") else value
                      for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
