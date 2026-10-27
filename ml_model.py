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
import mtf

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}

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
    ("htf1_dist", "1h trend: távolság az EMA50-től (%)"),
    ("htf4_dist", "4h trend: távolság az EMA50-től (%)"),
    ("mtf_aligned", "Makro trend egyezés (1h+4h EMA50 felett)"),
]
FEATURE_KEYS = [key for key, _ in FEATURES]
FEATURE_LABELS = dict(FEATURES)
# Missing support/resistance (no pivot on that side) is encoded as "far away".
FAR_ATR = 10.0

DEFAULTS = {
    "target_atr": 4.5,
    "stop_atr": 1.8,
    "trail_trigger_atr": 2.5,
    "trail_distance_atr": 1.0,
    "partial_tp_atr": 2.5,
    "partial_tp_ratio": 0.5,
    "breakeven_buffer_pct": 0.004,
    "min_cost_multiple": 4.0,
    "strict_4h_on_1h": True,
    "entry_gate": True,
    "cooldown_bars_after_stop": 6,
    "cooldown_bars_after_win": 2,
    "max_train_days": 90,
    "horizon_bars": 48,
    "test_fraction": 0.25,
    "min_train_samples": 250,
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "seed": 19,
    "ev_multiplier": 1.25,
    "auto_threshold": None,
}

_IDX = {k: i for i, k in enumerate(FEATURE_KEYS)}


class ModelError(Exception):
    """The model cannot be trained or applied safely."""


def breakeven_rate(stop_atr, target_atr):
    """Breakeven Win Rate = Risk / (Risk + Reward) = stop_atr / (stop_atr + target_atr)."""
    s, t = float(stop_atr), float(target_atr)
    if s <= 0 or t <= 0:
        raise ModelError("stop_atr and target_atr must be positive")
    return s / (s + t)


def dynamic_threshold(stop_atr, target_atr, multiplier=1.25):
    """Required +EV win probability threshold: multiplier (1.25x) * breakeven win rate."""
    return min(0.95, breakeven_rate(stop_atr, target_atr) * float(multiplier))


def evaluate_entry_gate(bar, feat_row, prev_feat_row=None):
    """Price-action & technical confirmation gate for a long entry candle.

    Requires a bullish (green) candle with price above EMA20/EMA50, positive
    MACD momentum, non-bearish market structure, and no bearish liquidity sweep.
    Returns ``(passed, reasons)``.
    """
    if bar is None or feat_row is None:
        return False, ["hiányzó gyertya vagy jellemző"]
    reasons = []
    o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
    rng = h - l
    if c < o:
        reasons.append("piros (medvés) belépési gyertya")
    if rng > 0 and (c - l) / rng < 0.35:
        reasons.append("erős felső kanóc / elutasítás a gyertya tetején")

    dist20 = float(feat_row[_IDX["dist_ema20"]])
    dist50 = float(feat_row[_IDX["dist_ema50"]])
    rsi = float(feat_row[_IDX["rsi"]])
    macd_h = float(feat_row[_IDX["macd_hist"]])
    struct = float(feat_row[_IDX["structure"]])
    bear_c = float(feat_row[_IDX["bear_candle"]])
    sweep_b = float(feat_row[_IDX["sweep_bear"]])

    if dist20 < -0.25:
        reasons.append("ár az EMA20 alatt (%.2f ATR)" % dist20)
    if dist50 < 0.0:
        reasons.append("ár az EMA50 alatt (%.2f ATR)" % dist50)
    if dist50 < dist20 - 0.20:
        reasons.append("EMA20 az EMA50 alatt")
    if rsi < 45.0:
        reasons.append("gyenge RSI momentum (%.1f < 45)" % rsi)
    if macd_h < 0.0:
        reasons.append("negatív MACD hisztogram (%.2f ATR)" % macd_h)
    if struct < 0.0:
        reasons.append("lokális csökkenő piaci struktúra (LH/LL)")
    if sweep_b > 0.0 or (bear_c > 0.0 and c <= o):
        reasons.append("medvés fordulógyertya vagy bull trap likviditás-lesöprés")
    return len(reasons) == 0, reasons


def settings(config=None):
    merged = dict(DEFAULTS)
    merged.update((config or {}).get("ml", {}) or {})
    risk = (config or {}).get("risk", {}) or {}
    for key in ("target_atr", "stop_atr", "trail_trigger_atr", "trail_distance_atr",
                "partial_tp_atr", "partial_tp_ratio", "breakeven_buffer_pct",
                "min_cost_multiple", "test_fraction", "ev_multiplier"):
        merged[key] = float(merged[key])
    for key in ("horizon_bars", "min_train_samples", "n_estimators", "max_depth",
                "min_samples_leaf", "seed", "cooldown_bars_after_stop",
                "cooldown_bars_after_win", "max_train_days"):
        merged[key] = int(merged[key])
    merged["strict_4h_on_1h"] = bool(merged.get("strict_4h_on_1h", True))
    merged["entry_gate"] = bool(merged.get("entry_gate", True))
    if "cost_rate" not in merged:
        merged["cost_rate"] = float(risk.get("fee_rate", 0.0)) + float(risk.get("slippage_rate", 0.0))
    else:
        merged["cost_rate"] = float(merged["cost_rate"])
    if merged["target_atr"] <= 0 or merged["stop_atr"] <= 0 or merged["horizon_bars"] < 1:
        raise ModelError("ml.target_atr, ml.stop_atr and ml.horizon_bars must be positive")
    if not 0.05 <= merged["test_fraction"] <= 0.5:
        raise ModelError("ml.test_fraction must be between 0.05 and 0.5")
    merged["breakeven"] = breakeven_rate(merged["stop_atr"], merged["target_atr"])
    merged["dynamic_threshold"] = dynamic_threshold(
        merged["stop_atr"], merged["target_atr"], merged["ev_multiplier"])
    if merged.get("auto_threshold") is None:
        merged["auto_threshold"] = merged["dynamic_threshold"]
    else:
        merged["auto_threshold"] = float(merged["auto_threshold"])
    return merged


def _interval_ms(interval):
    if interval not in INTERVAL_MS:
        raise ModelError("Interval must be 1m, 5m, 15m or 1h")
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

def build_features(f, interval_ms, htf=None):
    """One feature row per closed bar (``None`` during indicator warm-up).

    Row ``i`` only uses bars ``0..i``, swing pivots confirmed by bar ``i`` and
    (if ``htf`` series are given) higher-timeframe candles closed by then.
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
        ] + _htf_features(htf, f[i]["ct"] + 1)
    return rows


def _htf_features(htf, at_ms):
    if not htf:
        return [0.0, 0.0, 0.0]
    state = mtf.evaluate(htf, at_ms)
    dist = {iv: frame["dist_pct"] for iv, frame in state["frames"].items()}
    clip = lambda v: max(-20.0, min(20.0, v)) if v is not None else 0.0  # noqa: E731
    return [clip(dist.get("1h")), clip(dist.get("4h")), float(state["size_multiplier"])]


def label_outcomes(f, interval_ms, target_atr=3.6, stop_atr=1.5, horizon=72, cost_rate=0.0,
                   partial_tp_atr=0.0, partial_tp_ratio=0.0):
    """1 = fee-adjusted target reached before stop within ``horizon`` bars, 0 = not, NaN = unknown.

    Entry is the close of bar ``i``; barriers are checked from bar ``i + 1``.
    When ``partial_tp_ratio > 0`` and ``partial_tp_atr > 0``, the effective target
    barrier uses the blended two-stage target ``ratio * partial_tp_atr + (1 - ratio) * target_atr``.
    When ``cost_rate > 0``, the round-trip transaction cost (``2 * cost_rate * close``)
    is added to the target barrier and low-volatility setups whose target move is
    smaller than ``2 * round_trip_cost`` are labeled 0 so the model never learns
    fee-losing trades as positive examples.
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
    round_trip = 2.0 * max(0.0, float(cost_rate)) * close
    ratio = max(0.0, min(1.0, float(partial_tp_ratio)))
    eff_target_atr = (ratio * float(partial_tp_atr) + (1.0 - ratio) * float(target_atr)) \
        if (ratio > 0.0 and float(partial_tp_atr) > 0.0) else float(target_atr)
    target = close + eff_target_atr * atr + round_trip
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
    if cost_rate > 0:
        win &= (target_atr * atr) >= (2.0 * round_trip)
    labels[valid] = win[valid].astype(float)
    return labels


# ------------------------------------------------------------------ training

def _forest(cfg):
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], max_features="sqrt",
        n_jobs=-1, random_state=cfg["seed"])


def _sample_weights(X):
    """Upweight training samples from bullish macro & local trend regimes (2.5x)."""
    if len(X) == 0:
        return None
    weights = np.ones(len(X), dtype=float)
    mtf_col = X[:, _IDX["mtf_aligned"]]
    ema50_col = X[:, _IDX["dist_ema50"]]
    struct_col = X[:, _IDX["structure"]]
    bull_regime = (mtf_col > 0.0) & (ema50_col >= -0.2) & (struct_col >= 0.0)
    bear_regime = (ema50_col < -0.5) | (struct_col < 0.0)
    weights[bull_regime] = 2.5
    weights[bear_regime] = 0.5
    return weights


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


def atr_floor(f):
    """ATR series floored at 0.05% of price (the same ATR used by the labels)."""
    raw = ta.atr([b["h"] for b in f], [b["l"] for b in f], [b["c"] for b in f])
    return [max(v, b["c"] * 0.0005) if v is not None else None for v, b in zip(raw, f)]


def dataset(f, interval_ms, cfg, htf=None):
    """Feature matrix (NaN rows during warm-up), labels, and the valid-row mask."""
    rows = build_features(f, interval_ms, htf)
    labels = label_outcomes(
        f, interval_ms, cfg["target_atr"], cfg["stop_atr"], cfg["horizon_bars"],
        cost_rate=cfg.get("cost_rate", 0.0))
    X = np.full((len(f), len(FEATURE_KEYS)), np.nan)
    valid = np.zeros(len(f), dtype=bool)
    for i, row in enumerate(rows):
        if row is not None:
            X[i] = row
            valid[i] = True
    return X, labels, valid


def fit_forest(X, y, cfg):
    return _forest(cfg).fit(X, y, sample_weight=_sample_weights(X))


def positive_proba(model, X):
    return _positive_proba(model, X)


def train_model(bars, interval, now=None, config=None, htf_bars=None):
    """Train, validate out-of-sample, and refit the win-probability model.

    ``htf_bars`` (``{"1h": bars, "4h": bars}``) adds the macro trend features.
    """
    cfg = settings(config)
    step = _interval_ms(interval)
    f = _closed_floats(bars, now)
    htf = mtf.build_series(htf_bars)
    rows = build_features(f, step, htf)
    labels = label_outcomes(
        f, step, cfg["target_atr"], cfg["stop_atr"], cfg["horizon_bars"],
        cost_rate=cfg.get("cost_rate", 0.0))
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

    validation = _forest(cfg).fit(X_train, y_train, sample_weight=_sample_weights(X_train))
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

    final = _forest(cfg).fit(X, y, sample_weight=_sample_weights(X))
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
        "importance_method": method, "warnings": warnings, "uses_mtf": htf is not None,
        "medians": np.median(X, axis=0),
    }


# ---------------------------------------------------------------- prediction

def predict_setup(bundle, bars, now=None, htf_bars=None):
    """Win probability for a long entered at the latest closed candle.

    With ``htf_bars`` the result is trend-gated: when the 1h/4h trend is not
    aligned (price below EMA50), ``probability`` is capped at the historical
    base rate (no edge claimed) and ``signal`` is False. ``raw_probability``
    always holds the unfiltered model output.
    """
    cfg = bundle["settings"]
    step = _interval_ms(bundle["interval"])
    f = _closed_floats(bars, now)
    htf = mtf.build_series(htf_bars)
    rows = build_features(f, step, htf)
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
    raw_probability = probability
    mtf_state = mtf.evaluate(
        htf, f[-1]["ct"] + 1, trading_interval=bundle["interval"]) if htf else None
    mtf_aligned = mtf_state["aligned"] if mtf_state else None
    size_mult = float(mtf_state["size_multiplier"]) if mtf_state else 1.0
    if bundle["interval"] == "1h" and cfg.get("strict_4h_on_1h", True) and mtf_state is not None:
        if not mtf_aligned:
            size_mult = 0.0
    mtf_allow = size_mult > 0.0
    target_pct = cfg["target_atr"] * atr_value / close * 100
    stop_pct = -cfg["stop_atr"] * atr_value / close * 100
    round_trip_pct = 2.0 * cfg.get("cost_rate", 0.0) * 100.0
    min_target_pct = cfg.get("min_cost_multiple", 4.0) * round_trip_pct
    cost_ok = (target_pct >= min_target_pct) if round_trip_pct > 0 else True
    prev_row = rows[-2] if len(rows) >= 2 else None
    gate_ok, gate_reasons = evaluate_entry_gate(f[-1], rows[-1], prev_row)
    if (mtf_state is not None and not mtf_allow) or not cost_ok:
        probability = min(raw_probability, bundle["metrics"]["base_rate_train"])
    breakeven = breakeven_rate(cfg["stop_atr"], cfg["target_atr"])
    req_threshold = cfg["auto_threshold"]
    signal = probability >= req_threshold and mtf_allow and cost_ok and (
        gate_ok if cfg.get("entry_gate", False) else True)
    return {
        "probability": probability,
        "raw_probability": raw_probability,
        "mtf": mtf_state, "mtf_aligned": mtf_aligned,
        "mtf_allow_long": mtf_allow, "size_multiplier": size_mult,
        "cost_gate_passed": cost_ok, "min_target_pct": min_target_pct,
        "entry_gate_passed": gate_ok, "entry_gate_reasons": gate_reasons,
        "signal": signal,
        "breakeven": breakeven,
        "required_threshold": req_threshold,
        "ev_multiplier": cfg.get("ev_multiplier", 1.25),
        "threshold": req_threshold,
        "base_rate": bundle["metrics"]["base_rate_train"],
        "edge_pp": (probability - bundle["metrics"]["base_rate_train"]) * 100,
        "bar_time": f[-1]["t"], "entry": close,
        "stop": close - cfg["stop_atr"] * atr_value,
        "partial_target": close + cfg.get("partial_tp_atr", 1.8) * atr_value,
        "target": close + cfg["target_atr"] * atr_value,
        "atr": atr_value, "target_pct": target_pct,
        "stop_pct": stop_pct,
        "factors": factors, "top_factors": factors[:3], "values": values,
        "calibration_bucket": bucket,
    }

