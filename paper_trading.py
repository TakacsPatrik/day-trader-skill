"""Local-only paper trading journal. It never connects to a broker."""

import datetime
import json
import os
import tempfile
import threading
import uuid
from decimal import Decimal

from binance_client import ROOT, load_config
from crypto_engine import TradingError, dec

_LOCK = threading.RLock()


def _journal_path(config=None, path=None):
    config = config or load_config()
    path = path or config.get("storage", {}).get("paper_trades", "data/paper_trades.json")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    return path


def _load(path):
    if not os.path.exists(path):
        return {"trades": []}
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError) as exc:
        raise TradingError("Cannot read paper trading journal %s: %s" % (path, exc))
    if not isinstance(doc, dict) or not isinstance(doc.get("trades"), list):
        raise TradingError("Paper trading journal must contain a trades list")
    return doc


def _save(path, document):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".paper-trades-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise TradingError("Cannot save paper trading journal %s: %s" % (path, exc))


def _stamp(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    return now.astimezone(datetime.timezone.utc).isoformat()


def list_trades(config=None, path=None):
    target = _journal_path(config, path)
    with _LOCK:
        return list(_load(target)["trades"])


def daily_risk(trades, now=None, fee_rate=Decimal("0"), slippage_rate=Decimal("0")):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    today = now.astimezone(datetime.timezone.utc).date().isoformat()
    fees, slippage = dec(fee_rate), dec(slippage_rate)
    loss = Decimal(0)
    count = 0
    for trade in trades:
        if str(trade.get("entry_at", "")).startswith(today):
            count += 1
        if trade.get("status") == "closed" and str(trade.get("exit_at", "")).startswith(today):
            realized = dec(trade.get("realized_pnl_usd", 0))
            if realized < 0:
                loss -= realized
        elif trade.get("status") == "open" and str(trade.get("entry_at", "")).startswith(today):
            qty, entry = dec(trade["quantity"]), dec(trade["entry_price"])
            stop = dec(trade["stop"]) if trade.get("stop") else entry
            loss += qty * (max(Decimal(0), entry - stop) +
                           (entry + stop) * (fees + slippage))
    return {"trades_today": count, "loss_usd": loss}


def record_entry(symbol, side, quantity, entry_price, stop=None, target=None,
                 note="", config=None, path=None, trade_id=None, now=None):
    config = config or load_config()
    symbol = str(symbol).upper()
    allowed = {row["symbol"] for row in config["symbols"]}
    if symbol not in allowed:
        raise TradingError("Paper trade symbol %s is not in config.json" % symbol)
    if str(side).lower() != "long":
        raise TradingError("Paper trading currently supports long entries only")
    qty, entry = dec(quantity, "quantity"), dec(entry_price, "entry price")
    if qty <= 0 or entry <= 0:
        raise TradingError("Quantity and entry price must be above zero")
    stop_value = dec(stop, "stop") if stop not in (None, "") else None
    target_value = dec(target, "target") if target not in (None, "") else None
    if stop_value is None:
        raise TradingError("A paper entry requires a stop price")
    if stop_value is not None and stop_value >= entry:
        raise TradingError("Stop must be below entry for a long paper trade")
    if target_value is not None and target_value <= entry:
        raise TradingError("Target must be above entry for a long paper trade")
    target_path = _journal_path(config, path)
    trade = {"id": trade_id or uuid.uuid4().hex, "symbol": symbol, "side": "long",
             "quantity": format(qty, "f"), "entry_price": format(entry, "f"),
             "stop": format(stop_value, "f") if stop_value is not None else None,
             "target": format(target_value, "f") if target_value is not None else None,
             "entry_at": _stamp(now), "exit_at": None, "exit_price": None,
             "status": "open", "realized_pnl_usd": None, "note": str(note)}
    with _LOCK:
        document = _load(target_path)
        if any(item.get("id") == trade["id"] for item in document["trades"]):
            raise TradingError("Paper trade id already exists: %s" % trade["id"])
        document["trades"].append(trade)
        _save(target_path, document)
    return trade


def close_trade(trade_id, exit_price, fee_rate=Decimal("0"), slippage_rate=Decimal("0"),
                reason="manual", config=None, path=None, now=None):
    config = config or load_config()
    exit_value = dec(exit_price, "exit price")
    if exit_value <= 0:
        raise TradingError("Exit price must be above zero")
    fees = dec(fee_rate, "fee rate")
    slippage = dec(slippage_rate, "slippage rate")
    target_path = _journal_path(config, path)
    with _LOCK:
        document = _load(target_path)
        trade = next((item for item in document["trades"]
                      if item.get("id") == trade_id), None)
        if trade is None:
            raise TradingError("No paper trade with id %s" % trade_id)
        if trade.get("status") != "open":
            raise TradingError("Paper trade %s is already closed" % trade_id)
        qty = dec(trade["quantity"])
        entry = dec(trade["entry_price"])
        notional = qty * (entry + exit_value)
        pnl = qty * (exit_value - entry) - notional * (fees + slippage)
        trade.update(status="closed", exit_price=format(exit_value, "f"),
                     exit_at=_stamp(now), exit_reason=str(reason),
                     realized_pnl_usd=format(pnl, "f"))
        _save(target_path, document)
    return trade


def auto_close_trades(quotes, fee_rate=Decimal("0"), slippage_rate=Decimal("0"),
                      config=None, path=None, now=None):
    """Automatically close open paper trades when price hits Stop-Loss or Target.

    ``quotes`` maps symbol -> current price (or dict with ``price``, ``high``, ``low``).
    If both stop and target are touched in the same update, stop wins (conservative).
    """
    config = config or load_config()
    fees = dec(fee_rate, "fee rate")
    slippage = dec(slippage_rate, "slippage rate")
    target_path = _journal_path(config, path)
    closed_now = []
    with _LOCK:
        document = _load(target_path)
        changed = False
        for trade in document["trades"]:
            if trade.get("status") != "open":
                continue
            symbol = trade.get("symbol")
            quote = (quotes or {}).get(symbol)
            if quote is None:
                continue
            if isinstance(quote, dict):
                price = dec(quote["price"], "price")
                high = dec(quote.get("high", price), "high")
                low = dec(quote.get("low", price), "low")
            else:
                price = high = low = dec(quote, "price")
            stop = dec(trade["stop"], "stop") if trade.get("stop") not in (None, "") else None
            target = dec(trade["target"], "target") if trade.get("target") not in (None, "") else None
            hit_reason = None
            exit_value = None
            if stop is not None and (price <= stop or low <= stop):
                hit_reason = "stop"
                exit_value = min(price, stop)
            elif target is not None and (price >= target or high >= target):
                hit_reason = "target"
                exit_value = target
            if hit_reason is not None and exit_value is not None and exit_value > 0:
                qty = dec(trade["quantity"])
                entry = dec(trade["entry_price"])
                notional = qty * (entry + exit_value)
                pnl = qty * (exit_value - entry) - notional * (fees + slippage)
                trade.update(status="closed", exit_price=format(exit_value, "f"),
                             exit_at=_stamp(now), exit_reason=hit_reason,
                             realized_pnl_usd=format(pnl, "f"))
                closed_now.append(dict(trade))
                changed = True
        if changed:
            _save(target_path, document)
    return closed_now


def unrealized_pnl(trade, current_price, fee_rate=Decimal("0"), slippage_rate=Decimal("0")):
    if trade.get("status") != "open":
        return dec(trade.get("realized_pnl_usd", "0"))
    qty = dec(trade["quantity"])
    entry = dec(trade["entry_price"])
    current = dec(current_price)
    fees = dec(fee_rate)
    slippage = dec(slippage_rate)
    return qty * (current - entry) - qty * (entry + current) * (fees + slippage)
