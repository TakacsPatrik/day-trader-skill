"""Local, lightweight ML win-probability model for long setups.

A ``RandomForestClassifier`` is trained per symbol and interval on recent
Binance history. Every closed candle is treated as a hypothetical long entry
at its close:

* **Features** (known at that candle's close, no look-ahead): distances from
  the EMAs/VWAP in ATR units, ATR% and volatility regime, relative volume,
  momentum, distance to confirmed swing support/resistance, market structure,
  geometric patterns (double bottom, triangles), SMC liquidity sweeps,
  candlestick reversals, and the hour of day.
* **Label**: 1 if price reached ``entry + target_atr x ATR`` before
  ``entry - stop_atr x ATR`` within ``horizon_bars`` candles, else 0 (stop
  first, both in the same candle, or timeout).

Validation is chronological: the newest ``test_fraction`` of samples is held
out, and training samples whose label window overlaps the test period are
purged. Reported metrics (AUC, Brier score, calibration, permutation
importance) are out-of-sample; the live model is then refit on all labels.
"""

import datetime
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

import technical_analysis as ta

INTERVAL_MS = {"1m": 60_000, "5m": 300_000}

FEATURES = [
    ("dist_ema20", "Távolság az EMA20-tól (ATR)"),
    ("dist_ema50", "Távolság az EMA50-től (ATR)"),
    ("dist_ema200", "Távolság az EMA200-tól (ATR)"),
    ("dist_vwap", "Távolság a 24h VWAP-tól (ATR)"),
    ("atr_pct", "ATR (% az árhoz)"),
    ("atr_regime", "Volatilitás-rezsim (ATR / átlag ATR)"),
    ("rvol", "Relatív forgalom (RVOL)"),
    ("rsi", "RSI(14)"),
    ("macd_hist", "MACD hisztogram (ATR)"),
    ("bb_pos", "Pozíció a Bollinger-sávban (0–1)"),
    ("mom_1h", "1 órás momentum (ATR)"),
    ("dist_support", "Távolság a swing támasztól (ATR)"),
    ("dist_resistance", "Távolság a swing ellenállástól (ATR)"),
    ("structure", "Piaci struktúra (−1 LH/LL … +1 HH/HL)"),
    ("sweep_bull", "Bikás likviditás-lesöprés (stop hunt)"),
    ("sweep_bear", "Medvés likviditás-lesöprés (bull trap)"),
    ("double_bottom", "Dupla alj / W (1 kialakulóban, 2 megerősítve)"),
    ("asc_triangle", "Emelkedő háromszög (1/2)"),
    ("desc_triangle", "Ereszkedő háromszög (1/2)"),
    ("sym_triangle", "Szimmetrikus háromszög (−1 le, 0 forming, +1 fel)"),
    ("bull_candle", "Bikás fordulógyertya"),
    ("bear_candle", "Medvés fordulógyertya"),
    ("hour_sin", "Napszak (sin, UTC)"),
    ("hour_cos", "Napszak (cos, UTC)"),
]
FEATURE_KEYS = [key for key, _ in FEATURES]
FEATURE_LABELS = dict(FEATURES)
# Missing support/resistance (no pivot on that side) is encoded as "far away".
FAR_ATR = 10.0

DEFAULTS = {
    "target_atr": 1.5,
    "stop_atr": 1.5,
    "horizon_bars": 48,
    "test_fraction": 0.25,
    "min_train_samples": 600,
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 40,
    "seed": 19,
}


class ModelError(Exception):
    """The model cannot be trained or applied safely."""


def settings(config=None):
    merged = dict(DEFAULTS)
    merged.update((config or {}).get("ml", {}) or {})
    for key in ("target_atr", "stop_atr", "test_fraction"):
        merged[key] = float(merged[key])
    for key in ("horizon_bars", "min_train_samples", "n_estimators", "max_depth",
                "min_samples_leaf", "seed"):
        merged[key] = int(merged[key])
    if merged["target_atr"] <= 0 or merged["stop_atr"] <= 0 or merged["horizon_bars"] < 1:
        raise ModelError("ml.target_atr, ml.stop_atr and ml.horizon_bars must be positive")
    if not 0.05 <= merged["test_fraction"] <= 0.5:
        raise ModelError("ml.test_fraction must be between 0.05 and 0.5")
    return merged


def _interval_ms(interval):
    if interval not in INTERVAL_MS:
        raise ModelError("Interval must be 1m or 5m")
    return INTERVAL_MS[interval]


def _closed_floats(bars, now):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    closed = sorted((b for b in bars if b.get("close_time", 0) < now_ms),
                    key=lambda b: b["open_time"])
    return ta.to_floats(closed)


def _level(pattern):
    if not pattern:
        return 0.0
    return 2.0 if pattern["status"] == "confirmed" else 1.0


# ------------------------------------------------------------------ features

def build_features(f, interval_ms):
    """One feature row per closed bar (``None`` during indicator warm-up).

    Row ``i`` only uses bars ``0..i`` and swing pivots confirmed by bar ``i``.
    """
    n = len(f)
    rows = [None] * n
    if n == 0:
        return rows
    series = ta.indicator_series(f, interval_ms)
    closes = [b["c"] for b in f]
    atr_series = series["atr"]
    atr_avg = ta._ema_valid(atr_series, 100)
    size = ta.swing_size(interval_ms)
    swing_highs, swing_lows = ta.swing_points([b["h"] for b in f], [b["l"] for b in f], size)
    high_index = [p[0] for p in swing_highs]
    low_index = [p[0] for p in swing_lows]
    bars_hour = max(1, ta.HOUR_MS // interval_ms)
    ema20, ema50, ema200 = series["ema20"], series["ema50"], series["ema200"]
    vwap, rsi, hist = series["vwap_rolling"], series["rsi"], series["macd_hist"]
    bb_up, bb_low, rvol = series["bb_up"], series["bb_low"], series["rvol"]

    for i in range(n):
        values = (ema20[i], ema50[i], ema200[i], atr_series[i], atr_avg[i], rsi[i], hist[i],
                  bb_up[i], bb_low[i], rvol[i])
        if any(v is None for v in values) or i < bars_hour:
            continue
        c = closes[i]
        a = max(atr_series[i], c * 0.0005)
        highs_known = ta.known_pivots(swing_highs, high_index, i, size, keep=30)
        lows_known = ta.known_pivots(swing_lows, low_index, i, size, keep=30)
        support = max((p for j, p in lows_known if p < c and i - j <= 400), default=None)
        resistance = min((p for j, p in highs_known if p > c and i - j <= 400), default=None)
        recent_highs, recent_lows = highs_known[-12:], lows_known[-12:]
        trend = ta.market_structure(recent_highs, recent_lows, a)["trend"]
        patterns = {p["key"]: p for p in
                    ta.structure_patterns(f, i, recent_highs, recent_lows, a, size)}
        candles = ta.candle_patterns(f, a, ema20, ages=(0,), end=i)
        sym = patterns.get("symmetrical_triangle")
        band = bb_up[i] - bb_low[i]
        hour = (f[i]["t"] // ta.HOUR_MS) % 24
        rows[i] = [
            (c - ema20[i]) / a,
            (c - ema50[i]) / a,
            (c - ema200[i]) / a,
            (c - vwap[i]) / a,
            a / c * 100.0,
            atr_series[i] / atr_avg[i] if atr_avg[i] else 1.0,
            min(rvol[i], 10.0),
            rsi[i],
            hist[i] / a,
            (c - bb_low[i]) / band if band > 0 else 0.5,
            (c - closes[i - bars_hour]) / a,
            min((c - support) / a, FAR_ATR) if support is not None else FAR_ATR,
            min((resistance - c) / a, FAR_ATR) if resistance is not None else FAR_ATR,
            {"uptrend": 1.0, "downtrend": -1.0}.get(trend, 0.0),
            1.0 if "bullish_sweep" in patterns else 0.0,
            1.0 if "bearish_sweep" in patterns else 0.0,
            _level(patterns.get("double_bottom")),
            _level(patterns.get("ascending_triangle")),
            _level(patterns.get("descending_triangle")),
            ({"bullish": 1.0, "bearish": -1.0}.get(sym["type"], 0.0) if sym else 0.0),
            1.0 if any(p["type"] == "bullish" for p in candles) else 0.0,
            1.0 if any(p["type"] == "bearish" for p in candles) else 0.0,
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
        ]
    return rows


def label_outcomes(f, interval_ms, target_atr=1.5, stop_atr=1.5, horizon=48):
    """1 = target before stop within ``horizon`` bars, 0 = not, NaN = unknown yet.

    Entry is the close of bar ``i``; barriers are checked from bar ``i + 1``.
    A candle touching both barriers counts as a stop (conservative). Windows
    that cross a data gap and the last ``horizon`` bars stay NaN.
    """
    n = len(f)
    labels = np.full(n, np.nan)
    if n <= horizon:
        return labels
    high = np.array([b["h"] for b in f])
    low = np.array([b["l"] for b in f])
    close = np.array([b["c"] for b in f])
    times = np.array([b["t"] for b in f], dtype=np.int64)
    raw_atr = ta.atr(list(high), list(low), list(close))
    atr = np.array([v if v is not None else np.nan for v in raw_atr])
    atr = np.maximum(atr, close * 0.0005)
    target = close + target_atr * atr
    stop = close - stop_atr * atr
    never = horizon + 1
    first_target = np.full(n, never)
    first_stop = np.full(n, never)
    for k in range(1, horizon + 1):
        span = n - k
        hit_t = (high[k:] >= target[:span]) & (first_target[:span] == never)
        hit_s = (low[k:] <= stop[:span]) & (first_stop[:span] == never)
        first_target[:span][hit_t] = k
        first_stop[:span][hit_s] = k
    valid = np.zeros(n, dtype=bool)
    valid[:n - horizon] = (times[horizon:] - times[:n - horizon]) == horizon * interval_ms
    valid &= ~np.isnan(atr)
    win = (first_target < first_stop) & (first_target <= horizon)
    labels[valid] = win[valid].astype(float)
    return labels


# ------------------------------------------------------------------ training

def _forest(cfg):
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], max_features="sqrt",
        n_jobs=-1, random_state=cfg["seed"])


def _positive_proba(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def _calibration(probabilities, outcomes, bins=10):
    table = []
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        mask = (probabilities >= low) & ((probabilities < high) if b < bins - 1 else (probabilities <= high))
        count = int(mask.sum())
        if count:
            table.append({"low": low, "high": high, "count": count,
                          "mean_pred": float(probabilities[mask].mean()),
                          "actual": float(outcomes[mask].mean())})
    return table


def train_model(bars, interval, now=None, config=None):
    """Train, validate out-of-sample, and refit the win-probability model."""
    cfg = settings(config)
    step = _interval_ms(interval)
    f = _closed_floats(bars, now)
    rows = build_features(f, step)
    labels = label_outcomes(f, step, cfg["target_atr"], cfg["stop_atr"], cfg["horizon_bars"])
    usable = [i for i in range(len(f)) if rows[i] is not None and not np.isnan(labels[i])]
    if len(usable) < cfg["min_train_samples"]:
        raise ModelError("Túl kevés címkézett minta a tanításhoz (%d, minimum %d). Tölts le több "
                         "történelmi adatot." % (len(usable), cfg["min_train_samples"]))
    X = np.array([rows[i] for i in usable], dtype=float)
    y = labels[usable].astype(int)

    n_test = max(1, int(len(usable) * cfg["test_fraction"]))
    test_start_bar = usable[len(usable) - n_test]
    # purge: a training label must be fully resolved before the test period starts
    train_mask = np.array([i + cfg["horizon_bars"] < test_start_bar for i in usable])
    test_mask = np.zeros(len(usable), dtype=bool)
    test_mask[len(usable) - n_test:] = True
    X_train, y_train, X_test, y_test = X[train_mask], y[train_mask], X[test_mask], y[test_mask]
    if len(set(y_train.tolist())) < 2:
        raise ModelError("A tanító időszakban csak egyféle kimenetel fordult elő; a modell nem "
                         "tanítható (próbálj más cél/stop ATR-szorzót).")

    validation = _forest(cfg).fit(X_train, y_train)
    p_test = _positive_proba(validation, X_test)
    base_train = float(y_train.mean())
    both_classes = len(set(y_test.tolist())) == 2
    metrics = {
        "base_rate_train": base_train,
        "base_rate_test": float(y_test.mean()),
        "auc": float(roc_auc_score(y_test, p_test)) if both_classes else None,
        "brier": float(brier_score_loss(y_test, p_test)),
        "brier_baseline": float(brier_score_loss(y_test, np.full(len(y_test), base_train))),
        "mean_pred_test": float(p_test.mean()),
        "calibration": _calibration(p_test, y_test),
    }
    try:
        perm = permutation_importance(validation, X_test, y_test,
                                      scoring="roc_auc" if both_classes else "neg_brier_score",
                                      n_repeats=5, random_state=cfg["seed"], n_jobs=1)
        raw = np.clip(perm.importances_mean, 0, None)
        method = "permutation (holdout)"
    except ValueError:
        raw = np.zeros(len(FEATURE_KEYS))
    if raw.sum() <= 0:
        raw, method = validation.feature_importances_, "impurity (Gini)"
    shares = raw / raw.sum() if raw.sum() > 0 else raw
    importance = sorted(({"key": k, "label": FEATURE_LABELS[k], "importance": float(v)}
                         for k, v in zip(FEATURE_KEYS, shares)),
                        key=lambda item: -item["importance"])

    final = _forest(cfg).fit(X, y)
    span_days = (f[-1]["t"] - f[0]["t"]) / ta.DAY_MS if f else 0
    warnings = []
    if metrics["auc"] is not None and metrics["auc"] < 0.55:
        warnings.append("A holdout AUC %.2f: a modell alig jobb a véletlennél ezen a coinon/idősíkon. "
                        "A valószínűséget kezeld óvatosan." % metrics["auc"])
    if metrics["brier"] >= metrics["brier_baseline"]:
        warnings.append("A holdout Brier-pontszám nem jobb, mint a puszta alapráta: a modell "
                        "nem ad hozzá előrejelző információt.")
    if n_test < 300:
        warnings.append("Csak %d holdout minta: a validációs metrikák bizonytalanok." % n_test)
    return {
        "model": final, "settings": cfg, "interval": interval,
        "trained_at": datetime.datetime.now(datetime.timezone.utc),
        "samples": len(usable), "n_train": int(train_mask.sum()), "n_test": n_test,
        "span_days": span_days, "metrics": metrics, "importance": importance,
        "importance_method": method, "warnings": warnings,
        "medians": np.median(X, axis=0),
    }


# ---------------------------------------------------------------- prediction

def predict_setup(bundle, bars, now=None):
    """Win probability for a long entered at the latest closed candle."""
    cfg = bundle["settings"]
    step = _interval_ms(bundle["interval"])
    f = _closed_floats(bars, now)
    rows = build_features(f, step)
    if not rows or rows[-1] is None:
        raise ModelError("Nincs elég lezárt gyertya az aktuális jellemzők kiszámításához "
                         "(legalább ~250 kell az EMA200 miatt).")
    x = np.array([rows[-1]], dtype=float)
    model = bundle["model"]
    probability = float(_positive_proba(model, x)[0])

    # local sensitivity: how the probability moves if one feature were "typical"
    medians = bundle["medians"]
    probes = np.repeat(x, len(FEATURE_KEYS), axis=0)
    for j in range(len(FEATURE_KEYS)):
        probes[j, j] = medians[j]
    effects = probability - _positive_proba(model, probes)
    values = dict(zip(FEATURE_KEYS, x[0].tolist()))
    factors = []
    for item in bundle["importance"]:
        j = FEATURE_KEYS.index(item["key"])
        factors.append(dict(item, value=values[item["key"]], typical=float(medians[j]),
                            effect=float(effects[j])))

    close = f[-1]["c"]
    atr_value = max(ta.last(ta.atr([b["h"] for b in f], [b["l"] for b in f],
                                   [b["c"] for b in f])) or 0.0, close * 0.0005)
    bucket = next((b for b in bundle["metrics"]["calibration"]
                   if b["low"] <= probability < b["high"] or (b["high"] == 1.0 and probability == 1.0)),
                  None)
    return {
        "probability": probability,
        "base_rate": bundle["metrics"]["base_rate_train"],
        "edge_pp": (probability - bundle["metrics"]["base_rate_train"]) * 100,
        "bar_time": f[-1]["t"], "entry": close,
        "stop": close - cfg["stop_atr"] * atr_value,
        "target": close + cfg["target_atr"] * atr_value,
        "atr": atr_value, "target_pct": cfg["target_atr"] * atr_value / close * 100,
        "stop_pct": -cfg["stop_atr"] * atr_value / close * 100,
        "factors": factors, "top_factors": factors[:3], "values": values,
        "calibration_bucket": bucket,
    }

