"""Pure-Python technical indicators, candlestick patterns, and market structure.

Everything works on plain floats and lists (no TA-Lib, numpy, or pandas).
Series are aligned with the input bars; warm-up positions are ``None``.
"""

import bisect
import math

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


def swing_size(interval_ms):
    """Fractal half-width used for swing pivots: 5 bars on 5m, 8 bars on 1m."""
    return 5 if interval_ms >= 300_000 else 8


def known_pivots(pivots, pivot_index, end, size, keep=12):
    """The last ``keep`` pivots already confirmed at bar ``end`` (no look-ahead).

    A fractal pivot at index ``j`` needs ``size`` later bars, so it is only
    known once bar ``j + size`` has closed. ``pivot_index`` is the sorted list
    of pivot indices, used for a fast bisect.
    """
    cut = bisect.bisect_right(pivot_index, end - size)
    return pivots[max(0, cut - keep):cut]


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


def candle_patterns(f, atr_value, ema20_series, ages=(0, 1), end=None):
    """Detect classic patterns ending at each of the last ``ages`` closed bars.

    ``end`` evaluates the series as if bar ``end`` were the latest one (used to
    build historical ML features without slicing or look-ahead).
    """
    found = []
    n = len(f) if end is None else end + 1
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


# ------------------------------------------- geometric patterns and SMC
#
# Every detector below evaluates the market "as of" bar ``end`` and only uses
# bars up to ``end`` plus swing pivots that were already confirmed by then.
# The same functions therefore drive the live advisor (``end`` = last closed
# bar) and the historical ML feature builder (``end`` = every past bar).

def _max_high(f, start, stop):
    return max(f[k]["h"] for k in range(start, stop))


def _min_low(f, start, stop):
    return min(f[k]["l"] for k in range(start, stop))


def double_bottom(f, end, swing_lows, atr_value):
    """W pattern: two similar confirmed lows with a meaningful neckline between."""
    close = f[end]["c"]
    recent = [p for p in swing_lows if end - p[0] <= 200][-4:]
    min_height = max(2.5 * atr_value, close * 0.0025)
    tol = max(0.7 * atr_value, close * 0.0015)
    best = None
    for a in range(len(recent)):
        for b in range(a + 1, len(recent)):
            (i1, p1), (i2, p2) = recent[a], recent[b]
            # cheap rejections first; the slice scans below are rare
            if not 12 <= i2 - i1 <= 200 or abs(p1 - p2) > tol or end - i2 > 80:
                continue
            if close < min(p1, p2) - 0.5 * atr_value:
                continue
            neckline = _max_high(f, i1, i2 + 1)
            if neckline - max(p1, p2) < min_height:
                continue
            if _min_low(f, i1, i2 + 1) < min(p1, p2) - 0.2 * atr_value:
                continue
            if _max_high(f, max(0, i1 - 20), i1 + 1) - p1 < 2 * atr_value:
                continue
            if best is None or i2 > best[1]:
                best = (i1, i2, p1, p2, neckline)
    if not best:
        return None
    _, i2, p1, p2, neckline = best
    confirmed = close > neckline
    return {"key": "double_bottom", "name": "Double Bottom (W)", "type": "bullish",
            "status": "confirmed" if confirmed else "forming", "level": neckline,
            "t": f[i2]["t"],
            "detail": "Két közel azonos alj (%.6g / %.6g), nyakvonal: %.6g%s" %
                      (p1, p2, neckline, " – kitörés megtörtént." if confirmed else ".")}


def ascending_triangle(f, end, swing_highs, swing_lows, atr_value):
    """Flat resistance with a series of higher lows (bullish continuation)."""
    close = f[end]["c"]
    window = [p for p in swing_highs if end - p[0] <= 150][-3:]
    if len(window) < 2:
        return None
    tops = [p[1] for p in window]
    resistance = max(tops)
    if resistance - min(tops) > max(0.5 * atr_value, close * 0.002):
        return None
    rising = [p for p in swing_lows if p[0] > window[0][0]][-3:]
    if len(rising) >= 2 and all(rising[k + 1][1] > rising[k][1] + 0.15 * atr_value
                                for k in range(len(rising) - 1)) \
            and end - window[0][0] >= 15 and close < resistance + 0.2 * atr_value \
            and close > rising[-1][1]:
        broke = close > resistance + 0.1 * atr_value
        return {"key": "ascending_triangle", "name": "Ascending Triangle", "type": "bullish",
                "status": "confirmed" if broke else "forming", "level": resistance,
                "t": f[window[-1][0]]["t"],
                "detail": "Fix ellenállás %.6g alatt egyre magasabb alj-pontok." % resistance}
    return None


def descending_triangle(f, end, swing_highs, swing_lows, atr_value):
    """Flat support with a series of lower highs (bearish continuation)."""
    close = f[end]["c"]
    window = [p for p in swing_lows if end - p[0] <= 150][-3:]
    if len(window) < 2:
        return None
    bottoms = [p[1] for p in window]
    support = min(bottoms)
    if max(bottoms) - support > max(0.5 * atr_value, close * 0.002):
        return None
    falling = [p for p in swing_highs if p[0] > window[0][0]][-3:]
    if len(falling) >= 2 and all(falling[k + 1][1] < falling[k][1] - 0.15 * atr_value
                                 for k in range(len(falling) - 1)) \
            and end - window[0][0] >= 15 and close > support - 0.2 * atr_value \
            and close < falling[-1][1]:
        broke = close < support - 0.1 * atr_value
        return {"key": "descending_triangle", "name": "Descending Triangle", "type": "bearish",
                "status": "confirmed" if broke else "forming", "level": support,
                "t": f[window[-1][0]]["t"],
                "detail": "Fix támasz %.6g felett egyre alacsonyabb csúcsok – eladói nyomás." % support}
    return None


def symmetrical_triangle(f, end, swing_highs, swing_lows, atr_value):
    """Lower highs and higher lows converging; direction decided by the break."""
    close = f[end]["c"]
    highs = [p for p in swing_highs if end - p[0] <= 150][-2:]
    lows = [p for p in swing_lows if end - p[0] <= 150][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    (_, h1), (_, h2) = highs
    (_, l1), (_, l2) = lows
    if not (h2 < h1 - 0.15 * atr_value and l2 > l1 + 0.15 * atr_value):
        return None
    if h2 - l2 > 0.8 * (h1 - l1) or h2 <= l2:
        return None
    if close > h2 + 0.1 * atr_value:
        kind, status, text = "bullish", "confirmed", "felfelé tört ki"
    elif close < l2 - 0.1 * atr_value:
        kind, status, text = "bearish", "confirmed", "lefelé tört ki"
    elif l2 <= close <= h2:
        kind, status, text = "neutral", "forming", "az irány még nem dőlt el"
    else:
        return None
    return {"key": "symmetrical_triangle", "name": "Symmetrical Triangle", "type": kind,
            "status": status, "level": h2 if kind != "bearish" else l2,
            "t": f[max(highs[-1][0], lows[-1][0])]["t"],
            "detail": "Szűkülő ársáv (csúcsok %.6g→%.6g, aljak %.6g→%.6g): %s." %
                      (h1, h2, l1, l2, text)}


def liquidity_sweeps(f, end, swing_highs, swing_lows, atr_value, size, lookback=3, max_age=150):
    """Liquidity sweep / stop hunt on the last ``lookback`` bars up to ``end``.

    Bullish sweep: a candle wicks at least 0.05 ATR below an untouched,
    previously confirmed swing low (where sell stops rest) and closes back
    above it, and no later candle has closed below that level since.
    Bearish sweep is the mirror image above a swing high (buy-stop raid).
    """
    found = []
    pen = 0.05 * atr_value
    for kind, pivots in (("bullish", swing_lows), ("bearish", swing_highs)):
        for k in range(end, max(end - lookback, 0), -1):
            bar = f[k]
            best = None
            for j, level in reversed(pivots):
                if j + size >= k:          # pivot not yet confirmed when bar k traded
                    continue
                if k - j > max_age:
                    break
                if kind == "bullish":
                    if not (bar["l"] < level - pen and bar["c"] > level):
                        continue
                    if j + 1 < k and _min_low(f, j + 1, k) < level:
                        continue           # pool already taken earlier
                    if any(f[m]["c"] < level for m in range(k + 1, end + 1)):
                        continue           # sweep failed: price accepted below
                    if best is None or level < best[1]:
                        best = (j, level)
                else:
                    if not (bar["h"] > level + pen and bar["c"] < level):
                        continue
                    if j + 1 < k and _max_high(f, j + 1, k) > level:
                        continue
                    if any(f[m]["c"] > level for m in range(k + 1, end + 1)):
                        continue
                    if best is None or level > best[1]:
                        best = (j, level)
            if best:
                j, level = best
                depth = (level - bar["l"]) if kind == "bullish" else (bar["h"] - level)
                if kind == "bullish":
                    name = "Liquidity Sweep (bikás stop hunt)"
                    detail = ("A gyertya %.2f ATR-rel leszúrt a %.6g swing low alá (eladói stopok "
                              "lesöprése), majd visszazárt fölé." % (depth / atr_value, level))
                else:
                    name = "Liquidity Sweep (medvés bull trap)"
                    detail = ("A gyertya %.2f ATR-rel átszúrt a %.6g swing high fölé (vevői stopok "
                              "lesöprése), majd visszazárt alá." % (depth / atr_value, level))
                found.append({"key": "%s_sweep" % kind, "name": name, "type": kind,
                              "status": "confirmed", "age": end - k, "index": k,
                              "t": bar["t"], "level": level,
                              "sweep_price": bar["l"] if kind == "bullish" else bar["h"],
                              "depth_atr": depth / atr_value, "detail": detail})
                break                      # most recent sweep of this kind only
    return found


def structure_patterns(f, end, swing_highs, swing_lows, atr_value, size):
    """All geometric + SMC patterns as of bar ``end`` (pivots must be known by then)."""
    if end < 30 or atr_value <= 0:
        return []
    found = []
    for detector in (lambda: double_bottom(f, end, swing_lows, atr_value),
                     lambda: ascending_triangle(f, end, swing_highs, swing_lows, atr_value),
                     lambda: descending_triangle(f, end, swing_highs, swing_lows, atr_value),
                     lambda: symmetrical_triangle(f, end, swing_highs, swing_lows, atr_value)):
        pattern = detector()
        if pattern:
            found.append(pattern)
    found += liquidity_sweeps(f, end, swing_highs, swing_lows, atr_value, size)
    return found


def chart_patterns(f, atr_value, ema20_value, ema50_value, swing_highs, swing_lows, interval_ms,
                   swing_size=5):
    found = []
    n = len(f)
    if n < 30 or atr_value <= 0:
        return found
    close = f[-1]["c"]
    lows = [b["l"] for b in f]
    highs = [b["h"] for b in f]
    found += structure_patterns(f, n - 1, swing_highs, swing_lows, atr_value, swing_size)

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
                                  "level": f[top_index]["h"], "t": f[top_index]["t"],
                                  "detail": "Erős emelkedés (zászlórúd), majd visszateszt az EMA20-ra "
                                            "(%.0f%%-os korrekció)." % (retrace * 100)})

    for hours in (4, 1):
        bars_needed = max(3, hours * HOUR_MS // interval_ms)
        if n > bars_needed + 1:
            prior_high = max(highs[-1 - bars_needed:-1])
            if close > prior_high:
                found.append({"key": "rolling_%dh_breakout" % hours,
                              "name": "Rolling %dh High Breakout" % hours, "type": "bullish",
                              "status": "confirmed", "level": prior_high, "t": f[-1]["t"],
                              "detail": "A lezárt gyertya az elmúlt %d óra csúcsa (%.6g) fölött zárt." %
                                        (hours, prior_high)})
                break
    return found
