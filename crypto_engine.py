"""Shared crypto signal, risk, sizing, setup-advisor, and Monte Carlo calculations."""

import datetime
import random
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

import technical_analysis as ta
from technical_analysis import (atr, bollinger, ema, macd, rsi, rvol,  # noqa: F401  (public API)
                                vwap_daily, vwap_rolling)

class TradingError(Exception):
    pass


def dec(value, name="value"):
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise TradingError("%s must be a valid number" % name)
    if not result.is_finite():
        raise TradingError("%s must be finite" % name)
    return result


def floor_step(value, step):
    value, step = dec(value), dec(step)
    if step <= 0:
        raise TradingError("step size must be above zero")
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def floor_tick(value, tick):
    return floor_step(value, tick)


def _interval_ms(interval):
    if interval == "1m":
        return 60_000
    if interval == "5m":
        return 300_000
    raise TradingError("Interval must be 1m or 5m")


def _utc_day(ms):
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).date()


def evaluate_breakout(bars, symbol, interval="5m", now=None, or_minutes=15):
    """Build one UTC-day opening range and find its first confirmed breakout."""
    step_ms = _interval_ms(interval)
    if or_minutes <= 0 or (or_minutes * 60_000) % step_ms:
        raise TradingError("Opening range must be a positive whole number of %s bars" % interval)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    else:
        now = now.astimezone(datetime.timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    day_start = int(datetime.datetime.combine(now.date(), datetime.time(),
                                               datetime.timezone.utc).timestamp() * 1000)
    range_ms = or_minutes * 60_000
    range_end = day_start + range_ms
    closed = [bar for bar in bars if bar.get("close_time", 0) < now_ms
              and day_start <= bar.get("open_time", -1) < day_start + 86_400_000]
    closed.sort(key=lambda bar: bar["open_time"])
    opening = [bar for bar in closed if bar["open_time"] < range_end]
    base = {"symbol": symbol, "interval": interval, "date_utc": now.date().isoformat(),
            "status": "forming", "opening_range": None, "signal": None,
            "last_close": closed[-1]["close"] if closed else None}
    if now_ms < range_end:
        base["reason"] = "UTC daily opening range is still forming"
        return base
    expected = range_ms // step_ms
    expected_starts = [day_start + offset * step_ms for offset in range(expected)]
    opening_by_time = {bar["open_time"]: bar for bar in opening}
    if any(timestamp not in opening_by_time for timestamp in expected_starts):
        base.update(status="no_data", reason="Opening range has %d of %d closed bars" %
                    (len(opening_by_time), expected))
        return base
    opening = [opening_by_time[timestamp] for timestamp in expected_starts]
    high = max(bar["high"] for bar in opening)
    low = min(bar["low"] for bar in opening)
    base["opening_range"] = {"high": high, "low": low, "minutes": or_minutes,
                             "bars": expected, "start_utc_ms": day_start,
                             "end_utc_ms": range_end}
    for bar in closed:
        if bar["open_time"] < range_end:
            continue
        if bar["close"] > high:
            base.update(status="triggered", reason="First closed candle above the UTC opening-range high",
                        signal={"entry": bar["close"], "bar_open_time": bar["open_time"],
                                "bar_close_time": bar["close_time"], "bar_high": bar["high"],
                                "bar_low": bar["low"], "market": bar.get("market", "unknown")})
            return base
    base.update(status="waiting", reason="No closed candle has broken the opening-range high")
    return base


def build_plan(entry, stop, target_r, filters, side="long", target=None):
    if side != "long":
        raise TradingError("Only long-side paper simulations are supported")
    entry, stop = dec(entry, "entry"), dec(stop, "stop")
    target_r = dec(target_r, "target R") if target is None else None
    tick = dec(filters.get("tick_size", "0.00000001"), "tick size")
    entry, stop = floor_tick(entry, tick), floor_tick(stop, tick)
    if entry <= 0 or stop <= 0 or stop >= entry:
        raise TradingError("Entry must be positive and stop must be below entry")
    if target is not None:
        target = floor_tick(dec(target, "target"), tick)
        target_r = (target - entry) / (entry - stop)
    elif target_r <= 0:
        raise TradingError("Target R must be above zero")
    else:
        target = floor_tick(entry + (entry - stop) * target_r, tick)
    if target <= entry:
        raise TradingError("Target must be above the entry price")
    return {"entry": entry, "stop": stop, "target": target, "target_r": target_r,
            "risk_per_unit": entry - stop, "reward_risk": (target - entry) / (entry - stop)}


def size_position(entry, stop, risk_usd, filters, max_position_usd=None,
                  fee_rate=Decimal("0"), slippage_rate=Decimal("0")):
    entry, stop = dec(entry, "entry"), dec(stop, "stop")
    risk_usd = dec(risk_usd, "USD risk")
    step = dec(filters.get("step_size", "0.00000001"), "step size")
    min_qty = dec(filters.get("min_qty", "0"), "minimum quantity")
    tick = dec(filters.get("tick_size", "0.00000001"), "tick size")
    min_notional = dec(filters.get("min_notional", "0"), "minimum notional")
    if entry <= 0 or stop <= 0 or stop >= entry:
        raise TradingError("Entry must be positive and stop must be below entry")
    if risk_usd <= 0:
        raise TradingError("USD risk must be above zero")
    unit_loss = entry - stop + (entry + stop) * (dec(fee_rate) + dec(slippage_rate))
    qty = floor_step(risk_usd / unit_loss, step)
    if max_position_usd is not None:
        cap_qty = floor_step(dec(max_position_usd) / entry, step)
        qty = min(qty, cap_qty)
    notional = qty * entry
    estimated_loss = qty * unit_loss
    if qty < min_qty:
        raise TradingError("Calculated quantity is below Binance minimum quantity")
    if notional < min_notional:
        raise TradingError("Calculated notional is below Binance minimum notional")
    return {"quantity": qty, "notional_usd": notional, "estimated_loss_usd": estimated_loss,
            "risk_per_unit_usd": unit_loss, "step_size": step, "tick_size": tick}


def validate_risk(plan, quantity, settings, current_daily_loss=0, trades_today=0):
    qty = dec(quantity, "quantity")
    entry, stop = dec(plan["entry"]), dec(plan["stop"])
    risk = dec(settings.get("risk_per_trade_usd", 0), "risk per trade")
    daily_cap = dec(settings.get("daily_loss_limit_usd", 0), "daily loss limit")
    daily_loss = dec(current_daily_loss, "current daily loss")
    max_trades = int(settings.get("max_trades_per_day", 0) or 0)
    fee_rate = dec(settings.get("fee_rate", 0))
    slippage = dec(settings.get("slippage_rate", 0))
    loss = qty * (entry - stop + (entry + stop) * (fee_rate + slippage))
    problems = []
    if qty <= 0:
        problems.append("Quantity must be above zero")
    if loss > risk:
        problems.append("Stop loss including fees/slippage exceeds the per-trade USD risk")
    if daily_loss + loss > daily_cap:
        problems.append("Projected daily loss exceeds the configured USD limit")
    if max_trades and int(trades_today) >= max_trades:
        problems.append("Configured maximum trades per UTC day has been reached")
    return {"passed": not problems, "estimated_loss_usd": loss, "problems": problems}


RATINGS = {
    "STRONG_BUY": ("🟢 ERŐS VÉTEL (STRONG BUY)", "🟢 STRONG BUY"),
    "BUY_WATCH": ("🔵 VÉTEL / FIGYELÉS (BUY / WATCH)", "🔵 BUY / WATCH"),
    "NEUTRAL": ("🟡 VÁRAKOZÁS (NEUTRAL / WAIT)", "🟡 NEUTRAL / WAIT"),
    "AVOID": ("🔴 KERÜLENDŐ / VESZÉLYES (AVOID)", "🔴 AVOID"),
}
MIN_ANALYSIS_BARS = 60


def rating_for(score):
    if score >= 75:
        return "STRONG_BUY"
    if score >= 60:
        return "BUY_WATCH"
    if score >= 40:
        return "NEUTRAL"
    return "AVOID"


def _trade_levels(entry, atr_value, supports, resistances):
    """Smart stop and safe/extended targets from swing zones and ATR."""
    stop, stop_basis = entry - 1.5 * atr_value, "Entry − 1.5×ATR"
    support = next((z for z in supports if z["level"] < entry), None)
    if support:
        candidate = support["low"] - 0.2 * atr_value
        distance = entry - candidate
        if 0.75 * atr_value <= distance <= 2.5 * atr_value:
            stop, stop_basis = candidate, "a legközelebbi Swing Low támasz alatt"
    if stop <= 0:
        stop, stop_basis = entry * 0.98, "Entry − 2%"
    risk = entry - stop
    minimum_t1 = entry + 1.5 * risk
    ceiling = max(entry + 3.0 * atr_value, minimum_t1)
    resistance = next((z for z in resistances if z["level"] > entry), None)
    resistance_target = resistance["low"] - 0.1 * atr_value if resistance else None
    blocked = False
    if resistance_target is not None and resistance_target >= minimum_t1:
        t1 = min(resistance_target, ceiling)
        t1_basis = "az első Swing High ellenállás előtt" if t1 == resistance_target \
            else "Entry + 3×ATR (az ellenállás távolabb van)"
    else:
        t1 = max(entry + 1.5 * atr_value, minimum_t1)
        t1_basis = "Entry + 1.5R (min. 1.5R)"
        blocked = resistance is not None and resistance["low"] < t1
    t2 = max(entry + 3.0 * atr_value, entry + 2.5 * risk)
    t2 = max(t2, t1 + atr_value)
    return {"entry": entry, "stop": stop, "t1": t1, "t2": t2, "risk": risk,
            "stop_basis": stop_basis, "t1_basis": t1_basis, "resistance": resistance,
            "resistance_blocks_t1": blocked,
            "room_r": (resistance["low"] - entry) / risk if resistance else None}


def _near_zone(value, zones, tolerance):
    return next((z for z in zones if z["low"] - tolerance <= value <= z["high"] + tolerance), None)


def analyze_setup(bars, symbol="", interval="5m", filters=None, now=None, cost_rate=0):
    """Score a long setup (0-100) and derive entry, stop, and targets.

    Indicators and patterns use closed candles only (stable, no flicker);
    the live price of the forming candle is used for entry and distances.
    """
    step_ms = _interval_ms(interval)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    now_ms = int(now.astimezone(datetime.timezone.utc).timestamp() * 1000)
    ordered = sorted(bars, key=lambda b: b["open_time"])
    closed = [b for b in ordered if b.get("close_time", 0) < now_ms]
    forming = next((b for b in reversed(ordered) if b.get("close_time", 0) >= now_ms), None)
    result = {"symbol": symbol, "interval": interval, "status": "insufficient_data",
              "score": None, "rating": None, "label": "⚪ NINCS ELÉG ADAT",
              "short_label": "⚪ N/A", "actionable": False, "patterns": [],
              "pattern_summary": "—", "reasons": [], "breakdown": [], "supports": [],
              "resistances": [], "entry": None, "stop": None, "t1": None, "t2": None,
              "stop_pct": None, "t1_pct": None, "t2_pct": None, "advice": ""}
    if len(closed) < MIN_ANALYSIS_BARS:
        result["advice"] = "Legalább %d lezárt gyertya kell az elemzéshez (most: %d)." % (
            MIN_ANALYSIS_BARS, len(closed))
        result["reasons"] = [{"status": "warn", "text": result["advice"]}]
        return result

    f = ta.to_floats(closed)
    series = ta.indicator_series(f, step_ms)
    price = float((forming or closed[-1])["close"])
    last = ta.last
    ema20, ema50, ema200 = last(series["ema20"]), last(series["ema50"]), last(series["ema200"])
    today = now_ms // ta.DAY_MS
    daily_bars = sum(1 for b in f if b["t"] // ta.DAY_MS == today)
    vwap_mode = "daily" if daily_bars >= 12 else "rolling24h"
    vwap = last(series["vwap_daily" if vwap_mode == "daily" else "vwap_rolling"])
    rsi_value = last(series["rsi"])
    macd_line, macd_signal, hist = last(series["macd"]), last(series["macd_signal"]), last(series["macd_hist"])
    hist_values = [v for v in series["macd_hist"] if v is not None]
    hist_prev = hist_values[-2] if len(hist_values) >= 2 else None
    atr_value = max(last(series["atr"]) or 0.0, price * 0.0005)
    rvol_value = last(series["rvol"])
    bb_mid, bb_up, bb_low = last(series["bb_mid"]), last(series["bb_up"]), last(series["bb_low"])
    squeeze = ta.bollinger_squeeze(series["bb_width"])

    window = f[-400:]
    swing_size = 5 if interval == "5m" else 8
    swing_highs, swing_lows = ta.swing_points([b["h"] for b in window], [b["l"] for b in window],
                                              swing_size)
    supports, resistances = ta.support_resistance(window, swing_highs, swing_lows, price,
                                                  atr_value, len(window))
    structure = ta.market_structure(swing_highs, swing_lows, atr_value)
    candles = ta.candle_patterns(f, atr_value, series["ema20"])
    charts = ta.chart_patterns(window, atr_value, ema20, ema50, swing_highs, swing_lows, step_ms)
    bullish_candles = [p for p in candles if p["type"] == "bullish"]
    bearish_candles = [p for p in candles if p["type"] == "bearish"]
    bullish_charts = [p for p in charts if p["type"] == "bullish"]

    parts = []

    def part(group, name, earned, maximum, status, text):
        parts.append({"group": group, "name": name, "earned": earned, "max": maximum,
                      "status": status, "text": text})

    def above(label, level, points, group="Trend"):
        if level is None:
            return
        article = "az" if label.startswith("EMA") else "a"
        if price > level:
            part(group, label, points, points, "ok", "Ár %s %s felett (%.6g)" % (article, label, level))
        else:
            part(group, label, 0, points, "bad", "Ár %s %s alatt (%.6g)" % (article, label, level))

    above("EMA20", ema20, 5)
    above("EMA50", ema50, 8)
    above("VWAP", vwap, 7)
    if ema20 is not None and ema50 is not None:
        if ema20 > ema50:
            part("Trend", "EMA20>EMA50", 5, 5, "ok", "EMA20 az EMA50 felett – bikás sorrend")
        else:
            part("Trend", "EMA20>EMA50", 0, 5, "bad", "EMA20 az EMA50 alatt – gyenge sorrend")
    above("EMA200", ema200, 5)

    if hist is not None:
        if hist > 0 and (hist_prev is None or hist >= hist_prev):
            part("Momentum", "MACD", 8, 8, "ok", "MACD hisztogram pozitív és erősödik")
        elif hist > 0:
            part("Momentum", "MACD", 5, 8, "warn", "MACD pozitív, de a hisztogram gyengül")
        else:
            part("Momentum", "MACD", 0, 8, "bad", "MACD hisztogram negatív")
        if macd_line is not None and macd_line > 0:
            part("Momentum", "MACD vonal", 4, 4, "ok", "MACD vonal a nulla felett")
        else:
            part("Momentum", "MACD vonal", 0, 4, "bad", "MACD vonal a nulla alatt")
    if rsi_value is not None:
        if 45 <= rsi_value <= 62:
            part("Momentum", "RSI", 13, 13, "ok", "RSI %.0f – ideális, nem túlvásárolt" % rsi_value)
        elif 40 <= rsi_value <= 68:
            part("Momentum", "RSI", 9, 13, "ok", "RSI %.0f – elfogadható, van tér felfelé" % rsi_value)
        elif 68 < rsi_value <= 75:
            part("Momentum", "RSI", 4, 13, "warn", "RSI %.0f – melegedő, korlátozott tér" % rsi_value)
        elif rsi_value > 75:
            part("Momentum", "RSI", 0, 13, "bad", "RSI %.0f – extrém túlvásárolt" % rsi_value)
        elif rsi_value >= 30:
            part("Momentum", "RSI", 4, 13, "warn", "RSI %.0f – gyenge momentum" % rsi_value)
        else:
            part("Momentum", "RSI", 2, 13, "warn", "RSI %.0f – túladott, a visszapattanás kockázatos" % rsi_value)

    if rvol_value is not None:
        if rvol_value >= 1.5:
            part("Volume", "RVOL", 10, 10, "ok", "RVOL %.2f – erős, átlag feletti forgalom" % rvol_value)
        elif rvol_value >= 1.0:
            part("Volume", "RVOL", 7, 10, "ok", "RVOL %.2f – átlag feletti forgalom" % rvol_value)
        elif rvol_value >= 0.7:
            part("Volume", "RVOL", 3, 10, "warn", "RVOL %.2f – a volume még átlagos" % rvol_value)
        else:
            part("Volume", "RVOL", 0, 10, "bad", "RVOL %.2f – gyenge forgalom" % rvol_value)

    trend = structure["trend"]
    if trend == "uptrend":
        part("Struktúra", "Trendstruktúra", 10, 10, "ok", "Higher Highs + Higher Lows (emelkedő trend)")
    elif trend == "downtrend":
        part("Struktúra", "Trendstruktúra", 0, 10, "bad", "Lower Highs + Lower Lows (csökkenő trend)")
    else:
        part("Struktúra", "Trendstruktúra", 4, 10, "warn", "Oldalazó / vegyes piaci struktúra")

    levels = _trade_levels(price, atr_value, supports, resistances)
    room = levels["room_r"]
    if room is None:
        part("Struktúra", "Tér felfelé", 8, 10, "ok", "Nincs közeli ellenállás az ár felett")
    elif room >= 2.0:
        part("Struktúra", "Tér felfelé", 10, 10, "ok", "Szabad tér az ellenállásig: %.1fR" % room)
    elif room >= 1.5:
        part("Struktúra", "Tér felfelé", 7, 10, "ok", "Elegendő tér az ellenállásig: %.1fR" % room)
    elif room >= 1.0:
        part("Struktúra", "Tér felfelé", 3, 10, "warn", "Az ellenállás közel van (%.1fR) – gyenge R:R" % room)
    else:
        part("Struktúra", "Tér felfelé", 0, 10, "bad", "Az ellenállás túl közel van (%.1fR)" % room)

    def at_support(index):
        bar = f[index]
        return _near_zone(bar["l"], supports, 0.75 * atr_value) is not None

    recent_bullish = max(bullish_candles, key=lambda p: -p["age"], default=None)
    if recent_bullish:
        place = " a támaszon" if at_support(recent_bullish["index"]) else ""
        part("Mintázat", "Gyertyaminta", 5, 5, "ok", "%s gyertya%s" % (recent_bullish["name"], place))
    else:
        part("Mintázat", "Gyertyaminta", 0, 5, "warn", "Nincs bikás gyertyaminta az utolsó gyertyákon")
    if bullish_charts:
        confirmed = any(p["status"] == "confirmed" for p in bullish_charts)
        names = ", ".join(p["name"] for p in bullish_charts)
        part("Mintázat", "Chart alakzat", 7 if confirmed else 5, 7, "ok",
             "%s (%s)" % (names, "megerősítve" if confirmed else "kialakulóban"))
    else:
        part("Mintázat", "Chart alakzat", 0, 7, "warn", "Nincs bikás chart alakzat")
    if bb_up is not None and bb_mid is not None:
        if price > bb_up:
            part("Mintázat", "Bollinger", 0, 3, "warn", "Ár a felső Bollinger-sáv felett – kinyúlt")
        elif price > bb_mid:
            part("Mintázat", "Bollinger", 3, 3, "ok", "Ár a Bollinger középvonal felett")
        else:
            part("Mintázat", "Bollinger", 0, 3, "bad", "Ár a Bollinger középvonal alatt")

    extras = []
    if squeeze["active"]:
        extras.append({"status": "warn", "text": "Bollinger Squeeze: szűkülés, nagy kitörés közeleg "
                       "(az irány még nem dőlt el)"})
    for pattern in candles:
        if pattern["type"] == "bearish" and pattern["age"] <= 1:
            extras.append({"status": "bad", "text": "%s – %s" % (pattern["name"], pattern["detail"])})
        elif pattern["type"] == "neutral" and pattern["age"] == 0:
            extras.append({"status": "warn", "text": "Doji – bizonytalanság, várj megerősítésre"})
    if levels["resistance_blocks_t1"]:
        extras.append({"status": "warn", "text": "Ellenállás van a T1 előtt (%.6g): a cél nehezebben "
                       "teljesül" % levels["resistance"]["low"]})

    earned, maximum = sum(p["earned"] for p in parts), sum(p["max"] for p in parts)
    penalty = 8 if any(p["age"] <= 1 for p in bearish_candles) else 0
    score = max(0, int(round(100.0 * earned / maximum)) - penalty) if maximum else 0

    near_resistance_bearish = any(
        _near_zone(f[p["index"]]["h"], resistances, 0.75 * atr_value) is not None
        for p in bearish_candles if p["age"] <= 1)
    avoid = []
    if trend == "downtrend" and ema50 is not None and price < ema50:
        avoid.append("Csökkenő trend (LH + LL) az EMA50 alatt")
    if rsi_value is not None and rsi_value > 75:
        avoid.append("Extrém túlvásárolt RSI (%.0f)" % rsi_value)
    if near_resistance_bearish:
        avoid.append("Medvés fordulógyertya az ellenállásnál")

    conditions = {
        "Ár > EMA50": ema50 is not None and price > ema50,
        "Ár > VWAP": vwap is not None and price > vwap,
        "MACD pozitív": hist is not None and hist > 0,
        "RSI 40–68": rsi_value is not None and 40 <= rsi_value <= 68,
        "Átlag feletti volume": rvol_value is not None and rvol_value >= 1.0,
        "Bikás minta": bool(recent_bullish or bullish_charts),
    }
    gate_notes = []
    cost_pct_f = float(cost_rate) * 200 if cost_rate else 0.0
    net_t1_f = (levels["t1"] / price - 1) * 100 - cost_pct_f
    if avoid:
        score = min(score, 39)
    elif cost_pct_f and net_t1_f <= 0:
        score = min(score, 59)
        gate_notes.append("A T1 hozam (%+.2f%%) nem fedezi a díjakat és csúszást (−%.2f%%): "
                          "ilyen kis céllal nem éri meg belépni" %
                          ((levels["t1"] / price - 1) * 100, cost_pct_f))
    elif score >= 75 and not all(conditions.values()):
        score = 74
        gate_notes.append("Az ERŐS VÉTEL feltételei nem teljesülnek: %s" %
                          ", ".join(k for k, v in conditions.items() if not v))
    rating = rating_for(score)

    entry_mode = "market"
    if rating == "BUY_WATCH" and ema20 and price - ema20 > 0.5 * atr_value:
        levels = _trade_levels(ema20, atr_value, supports, resistances)
        entry_mode = "pullback"
    entry = levels["entry"]

    tick = dec(filters.get("tick_size", "0.00000001")) if filters else None

    def to_decimal(value):
        number = dec(format(value, ".12g"))
        return floor_tick(number, tick) if tick else number

    entry_d, stop_d = to_decimal(entry), to_decimal(levels["stop"])
    t1_d, t2_d = to_decimal(levels["t1"]), to_decimal(levels["t2"])
    valid = 0 < stop_d < entry_d < t1_d < t2_d
    cost_pct = cost_pct_f
    if valid:
        stop_pct = float((stop_d / entry_d - 1) * 100)
        t1_pct = float((t1_d / entry_d - 1) * 100)
        t2_pct = float((t2_d / entry_d - 1) * 100)
        risk_d = entry_d - stop_d
        rr1, rr2 = float((t1_d - entry_d) / risk_d), float((t2_d - entry_d) / risk_d)
    else:
        stop_pct = t1_pct = t2_pct = rr1 = rr2 = None
        entry_d = stop_d = t1_d = t2_d = None

    reasons = [{"status": p["status"], "text": p["text"]} for p in parts] + extras
    reasons += [{"status": "bad", "text": text} for text in avoid]
    reasons += [{"status": "warn", "text": text} for text in gate_notes]
    if entry_mode == "pullback":
        reasons.append({"status": "warn", "text": "Az ár %.1f ATR-rel az EMA20 felett jár – érdemes megvárni a "
                        "visszatesztet az EMA20-ra (%.6g)" % ((price - ema20) / atr_value, ema20)})
    order = {"ok": 0, "warn": 1, "bad": 2}
    reasons.sort(key=lambda r: order[r["status"]])

    pattern_list = [dict(p, group="candle") for p in candles] + [dict(p, group="chart") for p in charts]
    named = [p["name"] for p in pattern_list if p["type"] == "bullish" and p.get("age", 0) <= 1]
    named += ["⚠️ " + p["name"] for p in pattern_list if p["type"] == "bearish" and p.get("age", 0) <= 1]
    named += [p["name"] for p in pattern_list if p["type"] == "neutral" and p.get("age", 0) == 0]

    label, short_label = RATINGS[rating]
    if not valid:
        advice = "A szintek nem számíthatók a coin tick méretével; nincs javasolt terv."
    elif rating == "STRONG_BUY":
        advice = ("Az összes kulcsfeltétel teljesül: belépés most piaci áron (%s) lehetséges, a stop %s "
                  "(%+.2f%%), a biztonságos cél T1 %+.2f%%." %
                  (entry_d, stop_d, stop_pct, t1_pct))
    elif rating == "BUY_WATCH":
        advice = ("Ígéretes struktúra, de érdemes megvárni egy visszatesztet: javasolt limit belépő %s "
                  "(stop %+.2f%%, T1 %+.2f%%)." % (entry_d, stop_pct, t1_pct)) if entry_mode == "pullback" \
            else ("Ígéretes struktúra, de még nem tökéletes: figyeld a megerősítést (belépő %s, "
                  "stop %+.2f%%, T1 %+.2f%%)." % (entry_d, stop_pct, t1_pct))
    elif rating == "NEUTRAL":
        advice = "Várakozás: oldalazó piac vagy ellentmondó indikátorok. Nincs elég előny a belépéshez."
    else:
        advice = "Kerüld a long belépést: " + ("; ".join(avoid) if avoid else "gyenge technikai kép") + "."

    result.update(
        status="ok", score=score, rating=rating, label=label, short_label=short_label,
        actionable=rating in ("STRONG_BUY", "BUY_WATCH") and valid, entry_mode=entry_mode,
        price=dec(format(price, ".12g")), entry=entry_d, stop=stop_d, t1=t1_d, t2=t2_d,
        stop_pct=stop_pct, t1_pct=t1_pct, t2_pct=t2_pct, t1_net_pct=(t1_pct - cost_pct) if valid else None,
        rr1=rr1, rr2=rr2, stop_basis=levels["stop_basis"], t1_basis=levels["t1_basis"],
        patterns=pattern_list, pattern_summary=", ".join(named[:3]) or "—",
        reasons=reasons, breakdown=parts, conditions=conditions, avoid=avoid, structure=structure,
        supports=supports, resistances=resistances, advice=advice, vwap_mode=vwap_mode,
        squeeze=squeeze, forming=forming is not None,
        indicators={"ema20": ema20, "ema50": ema50, "ema200": ema200, "vwap": vwap,
                    "rsi": rsi_value, "macd": macd_line, "macd_signal": macd_signal,
                    "macd_hist": hist, "atr": atr_value, "atr_pct": atr_value / price * 100,
                    "bb_up": bb_up, "bb_mid": bb_mid, "bb_low": bb_low, "rvol": rvol_value})
    return result


def _first_hit(o, h, low, stop, target):
    """Which barrier a bar touches first; both in one bar counts as the stop."""
    if o <= stop:
        return "stop", o
    if o >= target:
        return "target", o
    hit_stop, hit_target = low <= stop, h >= target
    if hit_stop:
        return "stop", stop
    if hit_target:
        return "target", target
    return None, None


def simulate_trade(bars, entry, stop, target, interval="5m", runs=10_000, seed=19,
                   now=None, max_steps=None, current_atr=None, block_size=None,
                   demean=True, similar_vol_ratio=1.5):
    """Volatility-adjusted block-bootstrap simulation of a long trade.

    Contiguous blocks of historical candles are chained until the horizon is
    reached. Each block's returns are rescaled by (current ATR% / ATR% at the
    block start), so quiet history is not replayed at today's volatility and
    vice versa. ``historical`` reports the deterministic hit rate of real
    paths that started in a comparable volatility regime.
    """
    if not 1 <= int(runs) <= 100_000:
        raise TradingError("Simulation runs must be between 1 and 100000")
    step_ms = _interval_ms(interval)
    entry, stop, target = dec(entry, "entry"), dec(stop, "stop"), dec(target, "target")
    if not (0 < stop < entry < target):
        raise TradingError("Long simulation requires stop < entry < target")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    now_ms = int(now.astimezone(datetime.timezone.utc).timestamp() * 1000)
    block = int(block_size or (3_600_000 // step_ms))
    horizon = int(max_steps or (12 * 3_600_000 // step_ms))
    if block < 1 or horizon < 1:
        raise TradingError("Block size and horizon must be positive")

    closed = sorted((b for b in bars if b.get("close_time", 0) < now_ms),
                    key=lambda b: b["open_time"])
    f = ta.to_floats(closed)
    n = len(f)
    atr_series = ta.atr([b["h"] for b in f], [b["l"] for b in f], [b["c"] for b in f])
    # atr_pct[i] is the volatility known *before* bar i starts.
    atr_pct = [None] + [(atr_series[i - 1] / f[i - 1]["c"]) if atr_series[i - 1] and f[i - 1]["c"] > 0
                        else None for i in range(1, n)]
    entry_f, stop_f, target_f = float(entry), float(stop), float(target)
    if current_atr is not None and float(current_atr) > 0:
        current_pct = float(current_atr) / entry_f
    elif n and atr_series[-1]:
        current_pct = atr_series[-1] / f[-1]["c"]
    else:
        raise TradingError("Not enough historical candles to measure volatility")

    breaks = [1] * n
    ro, rh, rl, rc = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(1, n):
        previous = f[i - 1]["c"]
        if f[i]["t"] - f[i - 1]["t"] == step_ms and previous > 0:
            breaks[i] = 0
            ro[i], rh[i] = f[i]["o"] / previous - 1, f[i]["h"] / previous - 1
            rl[i], rc[i] = f[i]["l"] / previous - 1, f[i]["c"] / previous - 1
    if demean:
        usable = [i for i in range(1, n) if not breaks[i]]
        drift = sum(rc[i] for i in usable) / len(usable) if usable else 0.0
        for i in usable:
            ro[i] -= drift
            rh[i] -= drift
            rl[i] -= drift
            rc[i] -= drift
    prefix = [0]
    for flag in breaks:
        prefix.append(prefix[-1] + flag)
    starts = [i for i in range(1, n - block + 1)
              if atr_pct[i] and prefix[i + block] - prefix[i] == 0]
    if not starts:
        raise TradingError("Not enough contiguous historical candles for the simulation")
    scales = {i: min(4.0, max(0.25, current_pct / atr_pct[i])) for i in starts}

    rng = random.Random(int(seed))
    outcomes = {"target": 0, "stop": 0, "neither": 0}
    terminal_returns, sample_paths = [], []
    sample_limit = min(40, int(runs))
    for _ in range(int(runs)):
        sampling = len(sample_paths) < sample_limit
        current, outcome, done = entry_f, "neither", 0
        path = [entry_f] if sampling else None
        while done < horizon and outcome == "neither":
            begin = starts[rng.randrange(len(starts))]
            scale = scales[begin]
            for j in range(begin, begin + block):
                if done >= horizon:
                    break
                o = current * (1 + scale * ro[j])
                h = max(current * (1 + scale * rh[j]), o)
                low = min(current * (1 + scale * rl[j]), o)
                close = current * (1 + scale * rc[j])
                h, low = max(h, close), min(low, close)
                hit, price = _first_hit(o, h, low, stop_f, target_f)
                done += 1
                if hit:
                    outcome, current = hit, price
                    if sampling:
                        path.append(price)
                    break
                current = close
                if sampling:
                    path.append(current)
        outcomes[outcome] += 1
        terminal_returns.append((current / entry_f - 1) * 100)
        if sampling:
            sample_paths.append(path)

    historical = _historical_hit_rate(f, breaks, prefix, atr_pct, entry_f, stop_f, target_f,
                                      current_pct, horizon, block, similar_vol_ratio)
    total = float(runs)
    warnings = []
    span_days = n * step_ms / 86_400_000
    if span_days < 14:
        warnings.append("Csak %.1f napnyi történelmi adat áll rendelkezésre; az eredmény bizonytalan." %
                        span_days)
    if len(starts) < 100:
        warnings.append("Kevés (%d) használható gyertyablokk; az eredmény bizonytalan." % len(starts))
    if historical["similar"] < 20:
        warnings.append("Csak %d hasonló volatilitású történelmi időszak van; a találati arány "
                        "statisztikailag gyenge." % historical["similar"])
    scale_values = sorted(scales.values())
    return {"runs": int(runs), "seed": int(seed), "interval": interval,
            "method": "block_bootstrap_vol_adjusted", "demeaned": bool(demean),
            "probabilities": {name: count / total for name, count in outcomes.items()},
            "counts": outcomes, "terminal_returns_pct": terminal_returns,
            "sample_paths": sample_paths, "historical_days": int(span_days),
            "horizon_bars": horizon, "block_size": block, "blocks_available": len(starts),
            "current_atr_pct": current_pct * 100,
            "vol_scale": {"median": scale_values[len(scale_values) // 2],
                          "min": scale_values[0], "max": scale_values[-1]},
            "historical": historical, "warnings": warnings}


def _historical_hit_rate(f, breaks, prefix, atr_pct, entry, stop, target, current_pct,
                         horizon, block, ratio):
    """Replay real paths that began in a similar-volatility regime."""
    n = len(f)
    stop_ratio, target_ratio = stop / entry, target / entry
    low_bound, high_bound = current_pct / ratio, current_pct * ratio
    counts = {"target": 0, "stop": 0, "neither": 0}
    windows = similar = 0
    for i in range(1, n - horizon + 1, block):
        if prefix[i + horizon] - prefix[i] != 0:
            continue
        windows += 1
        if not (atr_pct[i] and low_bound <= atr_pct[i] <= high_bound):
            continue
        similar += 1
        base = f[i - 1]["c"]
        stop_level, target_level = base * stop_ratio, base * target_ratio
        outcome = "neither"
        for j in range(i, i + horizon):
            hit, _ = _first_hit(f[j]["o"], f[j]["h"], f[j]["l"], stop_level, target_level)
            if hit:
                outcome = hit
                break
        counts[outcome] += 1
    return {"windows": windows, "similar": similar, "target_first": counts["target"],
            "stop_first": counts["stop"], "neither": counts["neither"],
            "target_rate": counts["target"] / similar if similar else None,
            "stop_rate": counts["stop"] / similar if similar else None,
            "vol_range_pct": [low_bound * 100, high_bound * 100], "step_bars": block}
