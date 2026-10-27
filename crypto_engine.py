"""Shared crypto setup advisor (indicators + geometry + SMC), risk, and sizing calculations.

The ML win-probability model lives in ``ml_model.py``.
"""

import datetime
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
    if interval == "15m":
        return 900_000
    if interval == "1h":
        return 3_600_000
    raise TradingError("Interval must be 1m, 5m, 15m or 1h")


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
                  fee_rate=Decimal("0"), slippage_rate=Decimal("0"),
                  size_multiplier=Decimal("1")):
    entry, stop = dec(entry, "entry"), dec(stop, "stop")
    mult = dec(size_multiplier, "size multiplier")
    if mult <= 0:
        raise TradingError("Size multiplier must be above zero")
    risk_usd = dec(risk_usd, "USD risk") * mult
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
        cap_qty = floor_step(dec(max_position_usd) * mult / entry, step)
        qty = min(qty, cap_qty)
    notional = qty * entry
    estimated_loss = qty * unit_loss
    if qty < min_qty:
        raise TradingError("Calculated quantity is below Binance minimum quantity")
    if notional < min_notional:
        raise TradingError("Calculated notional is below Binance minimum notional")
    return {"quantity": qty, "notional_usd": notional, "estimated_loss_usd": estimated_loss,
            "risk_per_unit_usd": unit_loss, "step_size": step, "tick_size": tick,
            "size_multiplier": mult}


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


def breakeven_win_rate(stop_atr=1.8, target_atr=4.5):
    """Breakeven win rate from Risk / (Risk + Reward)."""
    s, t = float(stop_atr), float(target_atr)
    return s / (s + t) if (s + t) > 0 else 0.5


def required_ev_threshold(stop_atr=1.8, target_atr=4.5, multiplier=1.25):
    """Dynamic +EV win-probability threshold (default: 1.25x breakeven win rate)."""
    return min(0.95, breakeven_win_rate(stop_atr, target_atr) * float(multiplier))


def _trade_levels(entry, atr_value, supports, resistances, sweep_low=None):
    """Smart stop and safe/extended targets from swing zones, sweep wicks and ATR."""
    stop, stop_basis = entry - 1.5 * atr_value, "Entry − 1.5×ATR"
    support = next((z for z in supports if z["level"] < entry), None)
    if support:
        candidate = support["low"] - 0.2 * atr_value
        distance = entry - candidate
        if 0.75 * atr_value <= distance <= 2.5 * atr_value:
            stop, stop_basis = candidate, "a legközelebbi Swing Low támasz alatt"
    if sweep_low is not None and sweep_low < entry:
        # after a stop hunt the invalidation point is the sweep wick itself
        candidate = sweep_low - 0.2 * atr_value
        if 0.5 * atr_value <= entry - candidate <= 2.5 * atr_value:
            stop, stop_basis = candidate, "a likviditás-lesöprés kanóca alatt"
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


def analyze_setup(bars, symbol="", interval="5m", filters=None, now=None, cost_rate=0, htf=None):
    """Score a long setup (0-100) and derive entry, stop, and targets.

    ``htf`` is the optional ``mtf.evaluate()`` macro trend state (1h/4h). When
    given, a long can only rate BUY/WATCH or better if every higher timeframe
    closes above its EMA50 (trend alignment); otherwise the score is capped at 59.

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
    swing_size = ta.swing_size(step_ms)
    swing_highs, swing_lows = ta.swing_points([b["h"] for b in window], [b["l"] for b in window],
                                              swing_size)
    supports, resistances = ta.support_resistance(window, swing_highs, swing_lows, price,
                                                  atr_value, len(window))
    structure = ta.market_structure(swing_highs, swing_lows, atr_value)
    candles = ta.candle_patterns(f, atr_value, series["ema20"])
    charts = ta.chart_patterns(window, atr_value, ema20, ema50, swing_highs, swing_lows, step_ms,
                               swing_size=swing_size)
    sweeps = [p for p in charts if p["key"].endswith("_sweep")]
    charts_geo = [p for p in charts if not p["key"].endswith("_sweep")]
    bull_sweep = next((p for p in sweeps if p["type"] == "bullish"), None)
    bear_sweep = next((p for p in sweeps if p["type"] == "bearish" and p["age"] <= 1), None)
    sweep_low = bull_sweep["sweep_price"] if bull_sweep else None
    bullish_candles = [p for p in candles if p["type"] == "bullish"]
    bearish_candles = [p for p in candles if p["type"] == "bearish"]
    bullish_charts = [p for p in charts_geo if p["type"] == "bullish"]
    bearish_charts = [p for p in charts_geo if p["type"] == "bearish"]

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

    levels = _trade_levels(price, atr_value, supports, resistances, sweep_low)
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

    # Smart Money Concepts: an untouched swing low swept and reclaimed is the
    # classic institutional "stop hunt" long entry.
    if bull_sweep:
        part("SMC", "Likviditás-lesöprés", 8, 8, "ok",
             "Bikás liquidity sweep %s: stop hunt a %.6g swing low alatt, visszazárás fölé" %
             ("az utolsó gyertyán" if bull_sweep["age"] == 0 else "%d gyertyája" % bull_sweep["age"],
              bull_sweep["level"]))
    else:
        part("SMC", "Likviditás-lesöprés", 0, 8, "warn", "Nincs friss bikás likviditás-lesöprés")

    if htf is not None:
        bulls, total = htf.get("bull_count", 0), htf.get("total", 0)
        mtf_points = int(round(10.0 * bulls / total)) if total else 0
        mtf_status = "ok" if htf.get("aligned") else ("warn" if bulls else "bad")
        part("MTF", "Makro trend (1h/4h)", mtf_points, 10, mtf_status,
             "Makro trend %s: %s" % (htf.get("label", ""), htf.get("detail", "")))

    extras = []
    if squeeze["active"]:
        extras.append({"status": "warn", "text": "Bollinger Squeeze: szűkülés, nagy kitörés közeleg "
                       "(az irány még nem dőlt el)"})
    for pattern in candles:
        if pattern["type"] == "bearish" and pattern["age"] <= 1:
            extras.append({"status": "bad", "text": "%s – %s" % (pattern["name"], pattern["detail"])})
        elif pattern["type"] == "neutral" and pattern["age"] == 0:
            extras.append({"status": "warn", "text": "Doji – bizonytalanság, várj megerősítésre"})
    if bear_sweep:
        extras.append({"status": "bad", "text": "%s – %s" % (bear_sweep["name"], bear_sweep["detail"])})
    for pattern in bearish_charts:
        extras.append({"status": "warn", "text": "%s (%s) – %s" % (
            pattern["name"], "megerősítve" if pattern["status"] == "confirmed" else "kialakulóban",
            pattern["detail"])})
    if levels["resistance_blocks_t1"]:
        extras.append({"status": "warn", "text": "Ellenállás van a T1 előtt (%.6g): a cél nehezebben "
                       "teljesül" % levels["resistance"]["low"]})

    earned, maximum = sum(p["earned"] for p in parts), sum(p["max"] for p in parts)
    penalty = 8 if any(p["age"] <= 1 for p in bearish_candles) else 0
    penalty += 8 if bear_sweep else 0
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
        "Bikás minta": bool(recent_bullish or bullish_charts or bull_sweep),
    }
    position_scale = 1.0
    if htf is not None:
        position_scale = float(htf.get("size_multiplier", 1.0 if htf.get("aligned") else 0.0))
        if interval == "1h" and not htf.get("aligned"):
            position_scale = 0.0
        conditions["Makro trend 1h/4h ↑"] = bool(htf.get("aligned"))
    gate_notes = []
    cost_pct_f = float(cost_rate) * 200 if cost_rate else 0.0
    t1_gross_pct_f = (levels["t1"] / price - 1) * 100
    net_t1_f = t1_gross_pct_f - cost_pct_f
    if avoid:
        score = min(score, 39)
    else:
        if htf is not None and position_scale <= 0.0:
            score = min(score, 59)
            gate_notes.append("A makro trend (4h/1h EMA50) medvés – long belépés tiltva: %s"
                              % htf.get("detail", ""))
        elif htf is not None and position_scale < 1.0:
            score = min(score, 74)
            gate_notes.append("Vegyes makro trend (1h bikás, 4h medvés): a belépés engedélyezett, "
                              "de csökkentett (50%%-os) kockázattal és pozíciómérettel (%s)"
                              % htf.get("detail", ""))
        if cost_pct_f and (net_t1_f <= 0 or t1_gross_pct_f < 2.5 * cost_pct_f):
            score = min(score, 59)
            gate_notes.append("A T1 hozam (%+.2f%%) túl kicsi a díjakhoz és csúszáshoz (−%.2f%%) képest: "
                              "alacsony volatilitás miatt a költség-kapu blokkolja a belépést" %
                              (t1_gross_pct_f, cost_pct_f))
        elif score >= 75 and not all(conditions.values()):
            score = 74
            gate_notes.append("Az ERŐS VÉTEL feltételei nem teljesülnek: %s" %
                              ", ".join(k for k, v in conditions.items() if not v))
    rating = rating_for(score)

    entry_mode = "market"
    if rating == "BUY_WATCH" and ema20 and price - ema20 > 0.5 * atr_value:
        levels = _trade_levels(ema20, atr_value, supports, resistances, sweep_low)
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
        be_wr = 1.0 / (1.0 + rr1)
        req_ev = min(0.95, 1.25 * be_wr)
    else:
        stop_pct = t1_pct = t2_pct = rr1 = rr2 = be_wr = req_ev = None
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
        rr1=rr1, rr2=rr2, breakeven_win_rate=be_wr, required_threshold=req_ev,
        stop_basis=levels["stop_basis"], t1_basis=levels["t1_basis"],
        patterns=pattern_list, pattern_summary=", ".join(named[:3]) or "—",
        reasons=reasons, breakdown=parts, conditions=conditions, avoid=avoid, structure=structure,
        supports=supports, resistances=resistances, advice=advice, vwap_mode=vwap_mode,
        squeeze=squeeze, forming=forming is not None,
        smc={"bullish_sweep": bull_sweep, "bearish_sweep": bear_sweep,
             "sweeps": sweeps, "geometry": charts_geo}, mtf=htf,
        position_scale=position_scale,
        indicators={"ema20": ema20, "ema50": ema50, "ema200": ema200, "vwap": vwap,
                    "rsi": rsi_value, "macd": macd_line, "macd_signal": macd_signal,
                    "macd_hist": hist, "atr": atr_value, "atr_pct": atr_value / price * 100,
                    "bb_up": bb_up, "bb_mid": bb_mid, "bb_low": bb_low, "rvol": rvol_value})
    return result
