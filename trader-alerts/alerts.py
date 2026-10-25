"""24/7 crypto price alerts backed by the public Binance API."""

import datetime
import json
import os
import tempfile
import threading
import uuid

from binance_client import BinanceClient, BinanceError, ROOT, load_config
from crypto_engine import TradingError, dec


class AlertError(Exception):
    pass


def market_status(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    return {"open": True, "market": "crypto", "now_utc": now.astimezone(
        datetime.timezone.utc).isoformat(), "reason": "Crypto markets trade 24/7"}


def latest_prices(client=None, config=None):
    config = config or load_config()
    client = client or BinanceClient(config)
    quotes, errors = {}, []
    for item in config["symbols"]:
        symbol = item["symbol"]
        try:
            interval = config["default_interval"]
            limit = 1441 if interval == "1m" else 289
            now = datetime.datetime.now(datetime.timezone.utc)
            start_ms = int(now.timestamp() * 1000) - 86_400_000
            result = client.get_klines(symbol, interval, limit=limit,
                                       start_time=start_ms, end_time=int(now.timestamp() * 1000),
                                       include_open=True)
            if result["bars"]:
                bar = result["bars"][-1]
                reference = result["bars"][0]["close"]
                quotes[symbol] = {"price": bar["close"], "reference_price": reference,
                                  "timestamp": bar["open_time_utc"],
                                  "market": result["market"]}
            else:
                errors.append("%s: no candle data" % symbol)
        except BinanceError as exc:
            errors.append("%s: %s" % (symbol, exc))
    return quotes, errors


def evaluate(rules, quotes, state=None, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    state = dict(state or {})
    seen = set(state.get("triggered", []))
    rings = []
    for rule in rules:
        symbol = str(rule.get("symbol", "")).upper()
        kind = rule.get("type")
        ident = str(rule.get("id") or "%s:%s:%s" %
                     (symbol, kind, rule.get("price", rule.get("at"))))
        if kind == "time":
            try:
                hour, minute = [int(part) for part in str(rule["at"]).split(":", 1)]
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError()
            except (KeyError, ValueError):
                raise AlertError("Time alerts need an HH:MM UTC value")
            stamp = now.astimezone(datetime.timezone.utc)
            day_ident = "%s:%s" % (ident, stamp.date().isoformat())
            if (stamp.hour, stamp.minute) >= (hour, minute) and day_ident not in seen:
                rings.append({"id": day_ident, "symbol": symbol, "type": kind,
                              "price": None, "level": rule["at"],
                              "timestamp": stamp.isoformat()})
                seen.add(day_ident)
            continue
        quote = quotes.get(symbol)
        if not quote:
            continue
        price = dec(quote["price"])
        if kind == "pct_move":
            reference = rule.get("reference_price", quote.get("reference_price"))
            if not reference or dec(reference) <= 0:
                continue
            change = (price / dec(reference) - 1) * 100
            threshold = dec(rule.get("pct", 0))
            direction = rule.get("direction", "any")
            triggered = threshold > 0 and ((direction == "up" and change >= threshold) or
                                           (direction == "down" and change <= -threshold) or
                                           (direction == "any" and abs(change) >= threshold))
            level = threshold
        else:
            level = dec(rule.get("price", 0))
            triggered = (kind in ("above", "target") and price >= level) or (
                kind in ("below", "stop") and price <= level)
        if triggered and ident not in seen:
            rings.append({"id": ident, "symbol": symbol, "type": kind,
                          "price": price, "level": level, "timestamp": quote.get("timestamp")})
            seen.add(ident)
        if rule.get("rearm") and ((kind in ("above", "target") and price < level) or
                                  (kind in ("below", "stop") and price > level)):
            seen.discard(ident)
    state["triggered"] = sorted(seen)
    return rings, state


def rules_from_plan(plan):
    symbol = str(plan.get("symbol", "")).upper()
    if not symbol or plan.get("stop") is None or plan.get("target") is None:
        raise AlertError("A plan needs symbol, stop, and target levels")
    return [{"id": "%s-plan-stop" % symbol, "symbol": symbol, "type": "stop",
             "price": str(plan["stop"])},
            {"id": "%s-plan-target" % symbol, "symbol": symbol, "type": "target",
             "price": str(plan["target"])}]


def _read_state(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise AlertError("Cannot read alert state %s: %s" % (path, exc))


PRICE_KINDS = ("above", "below", "stop", "target")
_RULES_LOCK = threading.RLock()


def rules_path(config=None, path=None):
    config = config or load_config()
    path = path or config.get("storage", {}).get("alert_rules", "alerts.json")
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def validate_rule(rule, symbols=None):
    """Return a normalized copy of a rule or raise AlertError."""
    if not isinstance(rule, dict):
        raise AlertError("Alert rule must be an object")
    kind = rule.get("type")
    clean = {"id": str(rule.get("id") or uuid.uuid4().hex[:10]), "type": kind}
    if kind == "time":
        try:
            hour, minute = [int(part) for part in str(rule["at"]).split(":", 1)]
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except (KeyError, ValueError):
            raise AlertError("Time alerts need an HH:MM UTC value")
        clean["at"] = "%02d:%02d" % (hour, minute)
        return clean
    symbol = str(rule.get("symbol", "")).upper()
    if symbols is not None and symbol not in symbols:
        raise AlertError("Alert symbol %r is not in config.json" % symbol)
    if not symbol:
        raise AlertError("Alert rule needs a symbol")
    clean["symbol"] = symbol
    try:
        if kind in PRICE_KINDS:
            price = dec(rule.get("price"), "alert price")
            if price <= 0:
                raise AlertError("Alert price must be above zero")
            clean["price"] = format(price, "f")
        elif kind == "pct_move":
            pct = dec(rule.get("pct"), "alert percentage")
            if pct <= 0:
                raise AlertError("Alert percentage must be above zero")
            clean["pct"] = format(pct, "f")
            clean["direction"] = rule.get("direction", "any")
            if clean["direction"] not in ("up", "down", "any"):
                raise AlertError("Alert direction must be up, down, or any")
        else:
            raise AlertError("Unsupported alert type %r" % kind)
    except TradingError as exc:
        raise AlertError(str(exc))
    clean["rearm"] = bool(rule.get("rearm", False))
    return clean


def load_rules(config=None, path=None):
    """Read persisted rules; a missing file means no rules, a corrupt file is an error."""
    config = config or load_config()
    target = rules_path(config, path)
    symbols = {row["symbol"] for row in config["symbols"]}
    with _RULES_LOCK:
        if not os.path.exists(target):
            return []
        document = _read_state(target)
    raw = document.get("rules") if isinstance(document, dict) else document
    if not isinstance(raw, list):
        raise AlertError("%s must contain a rules list" % target)
    return [validate_rule(rule, symbols) for rule in raw]


def save_rules(rules, config=None, path=None):
    config = config or load_config()
    target = rules_path(config, path)
    symbols = {row["symbol"] for row in config["symbols"]}
    clean = [validate_rule(rule, symbols) for rule in rules]
    with _RULES_LOCK:
        _write_state(target, {"rules": clean})
    return clean


def add_rule(rule, config=None, path=None):
    config = config or load_config()
    with _RULES_LOCK:
        rules = load_rules(config, path)
        symbols = {row["symbol"] for row in config["symbols"]}
        new_rule = validate_rule(rule, symbols)
        if any(item["id"] == new_rule["id"] for item in rules):
            raise AlertError("Alert id already exists: %s" % new_rule["id"])
        rules.append(new_rule)
        save_rules(rules, config, path)
    return new_rule


def delete_rule(rule_id, config=None, path=None):
    config = config or load_config()
    with _RULES_LOCK:
        rules = load_rules(config, path)
        remaining = [rule for rule in rules if rule["id"] != rule_id]
        if len(remaining) == len(rules):
            raise AlertError("No alert with id %s" % rule_id)
        save_rules(remaining, config, path)
    return remaining


def _write_state(path, state):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".alerts-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise AlertError("Cannot save alert state %s: %s" % (path, exc))


def check_alerts(rules, quotes=None, client=None, config=None, state_path=None):
    config = config or load_config()
    if quotes is None:
        quotes, errors = latest_prices(client, config)
    else:
        errors = []
    path = state_path or config.get("storage", {}).get("alert_state", "data/alert_state.json")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    state = _read_state(path)
    rings, state = evaluate(rules, quotes, state)
    _write_state(path, state)
    return {"rings": rings, "quotes": quotes, "errors": errors, "market": market_status()}


def main():
    config = load_config()
    try:
        rules = load_rules(config)
    except AlertError as exc:
        print("Alert rules error: %s" % exc)
        return 1
    result = check_alerts(rules, config=config)
    print("CRYPTO MARKET OPEN 24/7 (UTC %s)." % result["market"]["now_utc"])
    for ring in result["rings"]:
        print("ALERT %s %s at %s (level %s)." %
              (ring["symbol"], ring["type"], ring["price"], ring["level"]))
    for error in result["errors"]:
        print("Data warning: " + error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
