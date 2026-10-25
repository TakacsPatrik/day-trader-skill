"""Crypto paper-trade risk checks in USD, with fractional quantities."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from binance_client import load_config
from crypto_engine import validate_risk


def check(plan, quantity, config=None, current_daily_loss=0, trades_today=0):
    config = config or load_config()
    return validate_risk(plan, quantity, config.get("risk", {}),
                         current_daily_loss=current_daily_loss, trades_today=trades_today)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Check a crypto paper trade against USD risk limits")
    parser.add_argument("plan")
    parser.add_argument("--quantity", required=True)
    args = parser.parse_args(argv)
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)
    result = check(plan, args.quantity)
    print(json.dumps({**result, "estimated_loss_usd": format(result["estimated_loss_usd"], "f")},
                     indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
