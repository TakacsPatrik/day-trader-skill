"""Multi-timeframe (1h / 4h) macro trend filter.

A long setup on 1m/5m is only "trend aligned" when, on every higher timeframe,
the last *fully closed* candle closed above its EMA50. ``HTFSeries.state_at``
answers that question for any past moment with a bisect, so the same code
serves the live dashboard, the ML feature builder and the backtester without
look-ahead.
"""

import bisect
import threading
import time

import technical_analysis as ta

HTF_INTERVALS = ("1h", "4h")
HTF_MS = {"1h": 3_600_000, "4h": 14_400_000}
EMA_PERIOD = 50
LIVE_LIMIT = 300


class HTFSeries:
    def __init__(self, interval, bars, period=EMA_PERIOD):
        ordered = sorted(bars, key=lambda b: b["open_time"])
        self.interval = interval
        self.period = period
        self.close_times = [int(b["close_time"]) for b in ordered]
        self.closes = [float(b["close"]) for b in ordered]
        self.ema = ta.ema(self.closes, period)

    def state_at(self, at_ms):
        """Trend of the last candle whose close_time is before ``at_ms``."""
        k = bisect.bisect_left(self.close_times, int(at_ms)) - 1
        base = {"interval": self.interval, "status": "unknown", "above": None,
                "close": None, "ema": None, "dist_pct": None, "slope_pct": 0.0,
                "rising": True, "close_time": None}
        if k < 0 or self.ema[k] is None:
            return base
        close, ema = self.closes[k], self.ema[k]
        above = close > ema
        prev_ema = self.ema[k - 1] if (k > 0 and self.ema[k - 1] is not None) else None
        slope = ((ema - prev_ema) / prev_ema * 100.0) if (prev_ema and prev_ema > 0) else 0.0
        rising = slope >= -0.03
        base.update(status="bull" if above else "bear", above=above, close=close, ema=ema,
                    dist_pct=(close / ema - 1) * 100 if ema else 0.0,
                    slope_pct=slope, rising=rising,
                    close_time=self.close_times[k])
        return base


def build_series(htf_bars, period=EMA_PERIOD):
    """``{"1h": bars, "4h": bars}`` -> ``{"1h": HTFSeries, ...}`` (None if empty)."""
    if not htf_bars:
        return None
    series = {iv: HTFSeries(iv, bars, period) for iv, bars in htf_bars.items() if bars}
    return series or None


def evaluate(series, at_ms, trading_interval=None):
    """Combined macro trend state at ``at_ms`` (milliseconds, UTC).

    Tiered position sizing rules:
    - 1h AND 4h above EMA50 -> ``size_multiplier = 1.0`` (full position size)
    - On sub-hourly charts (1m/5m/15m): 1h above EMA50, 4h below -> ``size_multiplier = 0.5``
    - On 1h chart: 4h is the primary macro timeframe, so 4h below EMA50 -> ``size_multiplier = 0.0``
    - 1h below EMA50 -> ``size_multiplier = 0.0`` (trade blocked)
    """
    frames = {iv: s.state_at(at_ms) for iv, s in (series or {}).items()}
    total = len(frames)
    bulls = sum(1 for f in frames.values() if f["above"] is True)
    unknown = sum(1 for f in frames.values() if f["status"] == "unknown")
    h4_frame = frames.get("4h", {})
    h4_rising = h4_frame.get("rising", True) if "4h" in frames else True
    aligned = total > 0 and bulls == total and h4_rising
    h1_above = frames.get("1h", {}).get("above")
    h4_above = frames.get("4h", {}).get("above")
    strict_4h = trading_interval == "1h" and "4h" in frames
    h4_ok = (h4_above is True) and (h4_rising or h4_frame.get("dist_pct", 0.0) >= 0.35)
    if not total or unknown == total:
        status, label = "unknown", "⚪ Ismeretlen"
        size_multiplier = 0.0
    elif aligned:
        status, label = "bull", "🟢 Bikás (%s · 100%% méret)" % "+".join(frames)
        size_multiplier = 1.0
    elif strict_4h and not h4_ok:
        status = "bear" if bulls == 0 else "mixed"
        label = "🔴 4h Medvés / Csökkenő EMA (1h charton tiltva)"
        size_multiplier = 0.0
    elif h1_above is True or ("1h" not in frames and bulls > 0):
        status, label = "mixed", "🟡 Vegyes (1h Bikás · 50% méret)"
        size_multiplier = 0.5
    elif bulls == 0 and not unknown:
        status, label = "bear", "🔴 Medvés (%s · tiltva)" % "+".join(frames)
        size_multiplier = 0.0
    else:
        status, label = "mixed", "🟡 Vegyes (1h Medvés · tiltva)"
        size_multiplier = 0.0
    parts = []
    for iv, f in frames.items():
        if f["status"] == "unknown":
            parts.append("%s: nincs elég adat" % iv)
        else:
            parts.append("%s: ár az EMA%d %s (%+.2f%%)" % (
                iv, EMA_PERIOD, "felett" if f["above"] else "alatt", f["dist_pct"]))
    return {"frames": frames, "aligned": aligned, "h1_bull": h1_above is True,
            "h4_bull": h4_above is True, "allow_long": size_multiplier > 0.0,
            "size_multiplier": size_multiplier, "bull_count": bulls, "total": total,
            "status": status, "label": label, "detail": " · ".join(parts) or "nincs adat"}


def fetch_htf(client, symbol, start_ms=None, end_ms=None, limit=LIVE_LIMIT):
    """Closed 1h/4h candles. With ``start_ms`` an EMA warm-up is prepended."""
    out = {}
    for iv in HTF_INTERVALS:
        if start_ms is None:
            result = client.get_klines(symbol, iv, limit=limit)
        else:
            end = int(end_ms if end_ms is not None else time.time() * 1000)
            warm = int(start_ms) - (EMA_PERIOD + 10) * HTF_MS[iv]
            count = (end - warm) // HTF_MS[iv] + 1
            result = client.get_klines(symbol, iv, limit=count, start_time=warm, end_time=end)
        out[iv] = result["bars"]
    return out


class HTFCache:
    """Thread-safe TTL cache of live higher-timeframe candles per symbol."""

    def __init__(self, client, ttl=120):
        self.client = client
        self.ttl = ttl
        self._lock = threading.Lock()
        self._data = {}

    def get(self, symbol):
        with self._lock:
            hit = self._data.get(symbol)
            if hit and time.time() - hit[0] < self.ttl:
                return hit[1]
        bars = fetch_htf(self.client, symbol)
        with self._lock:
            self._data[symbol] = (time.time(), bars)
        return bars

    def state(self, symbol, at_ms=None, trading_interval=None):
        series = build_series(self.get(symbol))
        return evaluate(series, at_ms if at_ms is not None else int(time.time() * 1000),
                        trading_interval=trading_interval)

