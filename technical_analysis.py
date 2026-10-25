"""Pure-Python technical indicators, candlestick patterns, and market structure.

Everything works on plain floats and lists (no TA-Lib, numpy, or pandas).
Series are aligned with the input bars; warm-up positions are ``None``.
"""

import math

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def to_floats(bars):
    """Convert normalized Decimal bars into compact float dicts."""
    return [{"t": int(b["open_time"]), "ct": int(b.get("close_time", b["open_time"])),
             "o": float(b["open"]), "h": float(b["high"]), "l": float(b["low"]),
             "c": float(b["close"]), "v": float(b.get("volume", 0))} for b in bars]


def _mean(values):
    return sum(values) / len(values)


def ema(values, period):
    out = [None] * len(values)
    if period < 1 or len(values) < period:
        return out
    prev = _mean(values[:period])
    out[period - 1] = prev
    k = 2.0 / (period + 1)
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def _ema_valid(values, period):
    """EMA over a series that starts with None warm-up entries."""
    first = next((i for i, v in enumerate(values) if v is not None), None)
    out = [None] * len(values)
    if first is None:
        return out
    tail = ema(values[first:], period)
    out[first:] = tail
    return out


def rsi(closes, period=14):
    n = len(closes)
    out = [None] * n
    if n <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        delta = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain, avg_loss):
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def macd(closes, fast=12, slow=26, signal=9):
    fast_ema, slow_ema = ema(closes, fast), ema(closes, slow)
    line = [a - b if a is not None and b is not None else None
            for a, b in zip(fast_ema, slow_ema)]
    sig = _ema_valid(line, signal)
    hist = [l - s if l is not None and s is not None else None for l, s in zip(line, sig)]
    return line, sig, hist


def atr(highs, lows, closes, period=14):
    """Wilder's Average True Range."""
    n = len(closes)
    out = [None] * n
    if n < period:
        return out
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                      abs(lows[i] - closes[i - 1])))
    prev = _mean(tr[:period])
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def bollinger(closes, period=20, mult=2.0):
    n = len(closes)
    mid, upper, lower, width = [None] * n, [None] * n, [None] * n, [None] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        mean = _mean(window)
        deviation = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        mid[i], upper[i], lower[i] = mean, mean + mult * deviation, mean - mult * deviation
        width[i] = (upper[i] - lower[i]) / mean if mean else None
    return mid, upper, lower, width


def bollinger_squeeze(width, lookback=120, percentile=20.0, min_history=40):
    """Squeeze = bandwidth in the lowest percentile of its recent history."""
    history = [w for w in width if w is not None][-lookback:]
    result = {"active": False, "bandwidth": history[-1] if history else None,
              "percentile_rank": None, "expanding": False}
    if len(history) < min_history:
        return result
    current = history[-1]
    rank = 100.0 * sum(1 for w in history if w <= current) / len(history)
    result["percentile_rank"] = rank
    result["active"] = rank <= percentile
    result["expanding"] = len(history) > 3 and current > history[-4] * 1.05
    return result


def vwap_daily(f):
    """VWAP anchored at 00:00 UTC; resets every UTC day."""
    out, day, cum_pv, cum_v = [], None, 0.0, 0.0
    for bar in f:
        bar_day = bar["t"] // DAY_MS
        if bar_day != day:
            day, cum_pv, cum_v = bar_day, 0.0, 0.0
        typical = (bar["h"] + bar["l"] + bar["c"]) / 3.0
        cum_pv += typical * bar["v"]
        cum_v += bar["v"]
        out.append(cum_pv / cum_v if cum_v > 0 else typical)
    return out


def vwap_rolling(f, window):
    out, pv, vol = [], [0.0], [0.0]
    for bar in f:
        typical = (bar["h"] + bar["l"] + bar["c"]) / 3.0
        pv.append(pv[-1] + typical * bar["v"])
        vol.append(vol[-1] + bar["v"])
        start = max(0, len(pv) - 1 - window)
        total = vol[-1] - vol[start]
        out.append((pv[-1] - pv[start]) / total if total > 0 else typical)
    return out


def rvol(volumes, period=20):
    """Volume divided by the average of the previous ``period`` bars."""
    out = [None] * len(volumes)
    for i in range(period, len(volumes)):
        average = _mean(volumes[i - period:i])
        out[i] = volumes[i] / average if average > 0 else None
    return out


def last(series):
    return next((v for v in reversed(series) if v is not None), None)


def indicator_series(f, interval_ms):
    closes = [b["c"] for b in f]
    highs, lows = [b["h"] for b in f], [b["l"] for b in f]
    macd_line, macd_signal, macd_hist = macd(closes)
    bb_mid, bb_up, bb_low, bb_width = bollinger(closes)
    return {"ema20": ema(closes, 20), "ema50": ema(closes, 50), "ema200": ema(closes, 200),
            "vwap_daily": vwap_daily(f), "vwap_rolling": vwap_rolling(f, DAY_MS // interval_ms),
            "rsi": rsi(closes), "macd": macd_line, "macd_signal": macd_signal,
            "macd_hist": macd_hist, "atr": atr(highs, lows, closes),
            "bb_mid": bb_mid, "bb_up": bb_up, "bb_low": bb_low, "bb_width": bb_width,
            "rvol": rvol([b["v"] for b in f])}


def chart_series(bars, interval_ms):
    """Overlay series for plotting, aligned with every given bar."""
    f = to_floats(bars)
    full = indicator_series(f, interval_ms)
    return {key: full[key] for key in ("ema20", "ema50", "ema200", "vwap_daily",
                                        "bb_mid", "bb_up", "bb_low")}


# ---------------------------------------------------------------- structure

def swing_points(highs, lows, size=3):
    """Fractal pivots confirmed by ``size`` bars on both sides."""
    swing_highs, swing_lows = [], []
    for i in range(size, len(highs) - size):
        if highs[i] > max(highs[i - size:i]) and highs[i] >= max(highs[i + 1:i + size + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] < min(lows[i - size:i]) and lows[i] <= min(lows[i + 1:i + size + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def cluster_levels(points, tolerance, pad):
    """Merge nearby pivots into horizontal zones."""
    zones = []
    for index, price in sorted(points, key=lambda p: p[1]):
        if zones and price - zones[-1]["_max"] <= tolerance:
            zone = zones[-1]
            zone["_prices"].append(price)
            zone["_max"] = price
            zone["last_index"] = max(zone["last_index"], index)
        else:
            zones.append({"_prices": [price], "_max": price, "last_index": index})
    result = []
    for zone in zones:
        prices = zone.pop("_prices")
        zone.pop("_max")
        zone.update(level=_mean(prices), low=min(prices) - pad, high=max(prices) + pad,
                    touches=len(prices))
        result.append(zone)
    return result


def support_resistance(f, swing_highs, swing_lows, price, atr_value, window_bars):
    """Nearest support zones below and resistance zones above ``price``."""
    first = max(0, len(f) - window_bars)
    tolerance = max(0.5 * atr_value, price * 0.0005)
    pad = 0.15 * atr_value
    highs = [p for p in swing_highs if p[0] >= first]
    lows = [p for p in swing_lows if p[0] >= first]
    recent = f[-window_bars // 4:] if window_bars >= 4 else f
    offset = len(f) - len(recent)
    if recent:
        top = max(range(len(recent)), key=lambda i: recent[i]["h"])
        bottom = min(range(len(recent)), key=lambda i: recent[i]["l"])
        highs.append((offset + top, recent[top]["h"]))
        lows.append((offset + bottom, recent[bottom]["l"]))
    resistances = [z for z in cluster_levels(highs, tolerance, pad) if z["level"] > price]
    supports = [z for z in cluster_levels(lows, tolerance, pad) if z["level"] < price]
    resistances.sort(key=lambda z: z["level"] - price)
    supports.sort(key=lambda z: price - z["level"])
    return supports[:3], resistances[:3]


def market_structure(swing_highs, swing_lows, atr_value):
    tol = 0.1 * atr_value
    highs, lows = swing_highs[-3:], swing_lows[-3:]
    higher_high = len(highs) >= 2 and highs[-1][1] > highs[-2][1] + tol
    lower_high = len(highs) >= 2 and highs[-1][1] < highs[-2][1] - tol
    higher_low = len(lows) >= 2 and lows[-1][1] > lows[-2][1] + tol
    lower_low = len(lows) >= 2 and lows[-1][1] < lows[-2][1] - tol
    if higher_high and higher_low:
        trend = "uptrend"
    elif lower_high and lower_low:
        trend = "downtrend"
    else:
        trend = "range"
    return {"trend": trend, "higher_highs": higher_high, "higher_lows": higher_low,
            "lower_highs": lower_high, "lower_lows": lower_low}


# ------------------------------------------------------------ candlesticks

def _parts(bar):
    body = abs(bar["c"] - bar["o"])
    return {"body": body, "range": bar["h"] - bar["l"],
            "upper": bar["h"] - max(bar["o"], bar["c"]),
            "lower": min(bar["o"], bar["c"]) - bar["l"]}


def candle_patterns(f, atr_value, ema20_series, ages=(0, 1)):
    """Detect classic patterns ending at each of the last ``ages`` closed bars."""
    found = []
    n = len(f)
    for age in ages:
        i = n - 1 - age
        if i < 5 or atr_value <= 0:
            continue
        bar, prev = f[i], f[i - 1]
        p, pp = _parts(bar), _parts(prev)
        if p["range"] <= 0:
            continue
        ema_prev = ema20_series[i - 1]
        down_context = (ema_prev is not None and prev["c"] < ema_prev) or prev["c"] < f[i - 5]["c"]
        up_context = (ema_prev is not None and prev["c"] > ema_prev) or prev["c"] > f[i - 5]["c"]

        def add(key, name, kind, detail):
            found.append({"key": key, "name": name, "type": kind, "age": age,
                          "index": i, "detail": detail})

        bullish = bar["c"] > bar["o"]
        bearish = bar["c"] < bar["o"]
        if bullish and prev["c"] < prev["o"] and bar["o"] <= prev["c"] and bar["c"] >= prev["o"] \
                and p["body"] > pp["body"] and p["body"] >= 0.4 * atr_value:
            add("bullish_engulfing", "Bullish Engulfing", "bullish",
                "A bikás gyertya teljesen bekebelezi az előző medvés gyertyát.")
        if bearish and prev["c"] > prev["o"] and bar["o"] >= prev["c"] and bar["c"] <= prev["o"] \
                and p["body"] > pp["body"] and p["body"] >= 0.4 * atr_value:
            add("bearish_engulfing", "Bearish Engulfing", "bearish",
                "A medvés gyertya teljesen bekebelezi az előző bikás gyertyát.")
        if p["range"] >= 0.5 * atr_value:
            body = max(p["body"], 0.05 * p["range"])
            if down_context and p["lower"] >= 2 * body and p["lower"] >= 0.55 * p["range"] \
                    and p["upper"] <= 0.2 * p["range"]:
                add("hammer", "Hammer / Pin Bar", "bullish",
                    "Hosszú alsó kanóc eső mozgás után: az eladók elfogytak.")
            if up_context and p["upper"] >= 2 * body and p["upper"] >= 0.55 * p["range"] \
                    and p["lower"] <= 0.2 * p["range"]:
                add("shooting_star", "Shooting Star", "bearish",
                    "Hosszú felső kanóc emelkedés után: az eladók visszaverték az árat.")
        if p["range"] >= 0.3 * atr_value and p["body"] <= 0.1 * p["range"]:
            add("doji", "Doji", "neutral", "Bizonytalanság: a vevők és eladók egyensúlyban.")
        if i >= 2:
            first, middle = f[i - 2], f[i - 1]
            first_parts, middle_parts = _parts(first), _parts(middle)
            if first["c"] < first["o"] and first_parts["body"] >= 0.6 * atr_value \
                    and middle_parts["body"] <= 0.35 * first_parts["body"] \
                    and max(middle["o"], middle["c"]) <= first["c"] + 0.15 * atr_value \
                    and bullish and bar["c"] > (first["o"] + first["c"]) / 2.0:
                add("morning_star", "Morning Star", "bullish",
                    "Háromgyertyás fordulómintázat: eső, tétovázó, majd erős bikás gyertya.")
    return found


# ---------------------------------------------------------- chart patterns

def chart_patterns(f, atr_value, ema20_value, ema50_value, swing_highs, swing_lows, interval_ms):
    found = []
    n = len(f)
    if n < 30 or atr_value <= 0:
        return found
    close = f[-1]["c"]
    highs, lows = [b["h"] for b in f], [b["l"] for b in f]

    recent_lows = [p for p in swing_lows if n - 1 - p[0] <= 200][-4:]
    best = None
    min_height = max(2.5 * atr_value, close * 0.0025)
    for a in range(len(recent_lows)):
        for b in range(a + 1, len(recent_lows)):
            (i1, p1), (i2, p2) = recent_lows[a], recent_lows[b]
            tol = max(0.7 * atr_value, close * 0.0015)
            if not 12 <= i2 - i1 <= 200 or abs(p1 - p2) > tol or n - 1 - i2 > 80:
                continue
            neckline = max(highs[i1:i2 + 1])
            if neckline - max(p1, p2) < min_height or close < min(p1, p2) - 0.5 * atr_value:
                continue
            if min(lows[i1:i2 + 1]) < min(p1, p2) - 0.2 * atr_value:
                continue
            if max(highs[max(0, i1 - 20):i1 + 1]) - p1 < 2 * atr_value:
                continue
            if best is None or i2 > best[1]:
                best = (i1, i2, p1, p2, neckline)
    if best:
        _, _, p1, p2, neckline = best
        confirmed = close > neckline
        found.append({"key": "double_bottom", "name": "Double Bottom (W)", "type": "bullish",
                      "status": "confirmed" if confirmed else "forming", "level": neckline,
                      "detail": "Két közel azonos alj (%.6g / %.6g), nyakvonal: %.6g%s" %
                                (p1, p2, neckline, " – kitörés megtörtént." if confirmed else ".")})

    window = [p for p in swing_highs if n - 1 - p[0] <= 150][-3:]
    if len(window) >= 2:
        tops = [p[1] for p in window]
        resistance = max(tops)
        if resistance - min(tops) <= max(0.5 * atr_value, close * 0.002):
            rising = [p for p in swing_lows if p[0] > window[0][0]][-3:]
            if len(rising) >= 2 and all(rising[k + 1][1] > rising[k][1] + 0.15 * atr_value
                                        for k in range(len(rising) - 1)) \
                    and n - 1 - window[0][0] >= 15 and close < resistance + 0.2 * atr_value \
                    and close > rising[-1][1]:
                broke = close > resistance + 0.1 * atr_value
                found.append({"key": "ascending_triangle", "name": "Ascending Triangle",
                              "type": "bullish", "status": "confirmed" if broke else "forming",
                              "level": resistance,
                              "detail": "Fix ellenállás %.6g alatt egyre magasabb alj-pontok." % resistance})

    lookback = f[-25:]
    if len(lookback) >= 10 and ema20_value is not None and ema50_value is not None:
        pole_top = max(range(len(lookback)), key=lambda i: lookback[i]["h"])
        top_index = n - len(lookback) + pole_top
        if n - 1 - top_index >= 3:
            start = max(0, top_index - 20)
            base = min(range(start, top_index + 1), key=lambda i: f[i]["l"])
            height = f[top_index]["h"] - f[base]["l"]
            if base < top_index and height >= max(4 * atr_value, 0.006 * close):
                pullback = min(lows[top_index:])
                retrace = (f[top_index]["h"] - pullback) / height
                retest = min(lows[-8:]) <= ema20_value + 0.5 * atr_value
                holding = close > ema20_value - 0.3 * atr_value and close > ema50_value
                if retrace <= 0.5 and retest and holding:
                    broke = close > f[top_index]["h"]
                    found.append({"key": "bull_flag", "name": "Bull Flag", "type": "bullish",
                                  "status": "confirmed" if broke else "forming",
                                  "level": f[top_index]["h"],
                                  "detail": "Erős emelkedés (zászlórúd), majd visszateszt az EMA20-ra "
                                            "(%.0f%%-os korrekció)." % (retrace * 100)})

    for hours in (4, 1):
        bars_needed = max(3, hours * HOUR_MS // interval_ms)
        if n > bars_needed + 1:
            prior_high = max(highs[-1 - bars_needed:-1])
            if close > prior_high:
                found.append({"key": "rolling_%dh_breakout" % hours,
                              "name": "Rolling %dh High Breakout" % hours, "type": "bullish",
                              "status": "confirmed", "level": prior_high,
                              "detail": "A lezárt gyertya az elmúlt %d óra csúcsa (%.6g) fölött zárt." %
                                        (hours, prior_high)})
                break
    return found
