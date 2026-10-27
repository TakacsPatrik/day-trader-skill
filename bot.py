"""Autonomous strategy robot: walk-forward backtester and auto-paper trader.

Usage::

    python bot.py backtest BTCUSDT --days 90            # replay ~3 months, print report
    python bot.py backtest ETHUSDT --threshold 0.6 --csv  # lower bar, export trades
    python bot.py paper                                  # daemon: auto-open paper trades
    python bot.py paper --once                           # a single scan cycle

Strategy (identical in both modes): at the close of a 1m/5m candle the
RandomForest win probability must exceed ``ml.auto_threshold`` (default 75%)
AND the 1h + 4h closes must sit above their EMA50 (MTF trend alignment).
The trade goes long at that close with stop = entry - stop_atr x ATR and
target = entry + target_atr x ATR, sized by the USD risk per trade from
config.json (Binance tick/step filters, fees and slippage included).

Backtest honesty rules: the model is trained walk-forward (it only ever sees
labels that were fully resolved before the bar it trades), higher timeframes
only expose candles closed by the decision time, a candle touching both stop
and target counts as a stop, and positions time out after ``horizon_bars``.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from decimal import Decimal

import numpy as np

import ml_model
import mtf
import technical_analysis as ta
from binance_client import INTERVAL_MS, ROOT, BinanceClient, load_config
from crypto_engine import TradingError, build_plan, dec, size_position, validate_risk
from paper_trading import auto_close_trades, daily_risk, list_trades, record_entry

DAY_MS = 86_400_000
UTC = datetime.timezone.utc
LOG = logging.getLogger("bot")
MTF_INDEX = ml_model.FEATURE_KEYS.index("mtf_aligned")


def _now(now=None):
    now = now or datetime.datetime.now(UTC)
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def _risk(config):
    risk = config.get("risk", {})
    return {"fee": float(risk.get("fee_rate", 0)), "slippage": float(risk.get("slippage_rate", 0)),
            "risk_usd": risk.get("risk_per_trade_usd", 50), "max_position": risk.get("max_position_usd"),
            "account": float(risk.get("account_usd", 5000)),
            "max_trades": int(risk.get("max_trades_per_day", 0) or 0),
            "daily_limit": float(risk.get("daily_loss_limit_usd", 0) or 0), "raw": risk}


def _plan_and_size(entry, stop, target, filters, risk, size_mult=1.0):
    """Tick-aligned plan + step-aligned quantity (Decimal, Binance filters)."""
    plan = build_plan(format(entry, ".12g"), format(stop, ".12g"), 1, filters or {},
                      target=format(target, ".12g"))
    size = size_position(plan["entry"], plan["stop"], risk["risk_usd"], filters or {},
                         risk["max_position"], risk["fee"], risk["slippage"],
                         size_multiplier=format(size_mult, ".4g"))
    return plan, size


# ================================================================== backtest

def fetch_history(client, symbol, interval, days, now):
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - int(days * DAY_MS)
    step = INTERVAL_MS[interval]
    result = client.get_klines(symbol, interval, limit=(end_ms - start_ms) // step + 1,
                               start_time=start_ms, end_time=end_ms)
    htf_bars = mtf.fetch_htf(client, symbol, start_ms, end_ms)
    return result["bars"], htf_bars, result["filters"], result["market"]


def run_backtest(symbol, interval=None, days=120, train_days=60, retrain_days=2.0, threshold=None,
                 require_mtf=True, config=None, client=None, now=None, bars=None, htf_bars=None,
                 filters=None, market="spot", progress=None):
    """Walk-forward replay of the ML + MTF + Trailing Stop strategy. Returns a report dict."""
    config = config or load_config()
    interval = interval or config.get("default_interval", "1h")
    cfg = ml_model.settings(config)
    threshold = cfg["auto_threshold"] if threshold is None else float(threshold)
    risk = _risk(config)
    now = _now(now)
    if interval not in ("1m", "5m", "15m", "1h"):
        raise TradingError("Backtest interval must be 1m, 5m, 15m or 1h")
    step = INTERVAL_MS[interval]
    if bars is None:
        client = client or BinanceClient(config)
        bars, htf_bars, filters, market = fetch_history(client, symbol, interval, days, now)
    say = progress or (lambda message: None)

    f = ml_model._closed_floats(bars, now)
    n = len(f)
    if n < 300:
        raise TradingError("Too little history for a backtest (%d candles)" % n)
    series = mtf.build_series(htf_bars)
    use_mtf = require_mtf and series is not None
    say("Jellemzők számítása %d gyertyán..." % n)
    X, labels, valid = ml_model.dataset(f, step, cfg, series)
    atr = ml_model.atr_floor(f)
    horizon = cfg["horizon_bars"]
    labeled = valid & ~np.isnan(labels)
    index = np.arange(n)

    test_start = next((i for i in range(n) if f[i]["t"] >= f[0]["t"] + train_days * DAY_MS), n)
    if test_start >= n - 1:
        raise TradingError("The training window covers the whole history; lower --train-days")
    retrain = max(1, int(retrain_days * DAY_MS // step))
    max_train_days = max(float(train_days), float(cfg.get("max_train_days", 90) or 90))
    max_train_bars = int(max_train_days * DAY_MS // step)
    probs = np.full(n, np.nan)
    models = 0
    for seg in range(test_start, n, retrain):
        seg_end = min(n, seg + retrain)
        win_start = max(0, seg - max_train_bars)
        train_idx = index[labeled & (index >= win_start) & (index < seg - horizon)]
        if len(train_idx) < cfg["min_train_samples"]:
            train_idx = index[labeled & (index < seg - horizon)]
        if len(train_idx) < cfg["min_train_samples"] or len(set(labels[train_idx].tolist())) < 2:
            continue
        model = ml_model.fit_forest(X[train_idx], labels[train_idx].astype(int), cfg)
        models += 1
        live = index[seg:seg_end][valid[seg:seg_end]]
        if len(live):
            probs[live] = ml_model.positive_proba(model, X[live])
        say("Walk-forward modell #%d (%d minta) · %s" % (
            models, len(train_idx), datetime.datetime.fromtimestamp(f[seg]["t"] / 1000, UTC).date()))

    equity = peak = risk["account"]
    max_dd = max_dd_pct = 0.0
    trades, curve = [], [(f[test_start]["t"], equity)]
    stats = {"signals": 0, "blocked_mtf": 0, "half_size_mtf": 0,
             "blocked_gate": 0, "blocked_cooldown": 0,
             "blocked_cost": 0, "blocked_risk": 0, "blocked_size": 0}
    daily = {}
    position = None
    cooldown_until = -1
    cost = risk["fee"] + risk["slippage"]
    round_trip_cost = 2.0 * cost
    min_target_move = cfg.get("min_cost_multiple", 4.0) * round_trip_cost
    trail_trigger = cfg.get("trail_trigger_atr", 2.5)
    trail_dist_atr = cfg.get("trail_distance_atr", 1.0)
    partial_tp_atr = cfg.get("partial_tp_atr", 2.5)
    partial_tp_ratio = max(0.0, min(0.9, float(cfg.get("partial_tp_ratio", 0.5))))
    be_buffer = cfg.get("breakeven_buffer_pct", 0.004)
    strict_4h = bool(cfg.get("strict_4h_on_1h", True)) and interval == "1h"
    use_entry_gate = bool(cfg.get("entry_gate", True)) and require_mtf
    cooldown_stop = int(cfg.get("cooldown_bars_after_stop", 3))
    cooldown_win = int(cfg.get("cooldown_bars_after_win", 1))

    def close(i, price, reason):
        nonlocal equity, peak, max_dd, max_dd_pct, position, cooldown_until
        p = position
        rem_pnl = p["qty"] * (price - p["entry"]) - p["qty"] * (p["entry"] + price) * cost
        pnl = p["realized_pnl"] + rem_pnl
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100 if peak else 0.0)
        day = daily.setdefault(f[i]["t"] // DAY_MS, {"count": 0, "loss": 0.0})
        if pnl < 0:
            day["loss"] -= pnl
            cooldown_until = max(cooldown_until, i + cooldown_stop)
        else:
            cooldown_until = max(cooldown_until, i + cooldown_win)
        trades.append({"entry_time": datetime.datetime.fromtimestamp(p["t"] / 1000, UTC).isoformat(),
                       "exit_time": datetime.datetime.fromtimestamp(f[i]["ct"] / 1000, UTC).isoformat(),
                       "entry": p["entry"], "initial_stop": p["initial_stop"],
                       "stop": p["stop"], "target": p["target"], "exit": price,
                       "qty": p["initial_qty"], "size_mult": p["size_mult"],
                       "partial_closed": p["partial_closed"],
                       "trailed": p["trailed"], "probability": p["prob"],
                       "reason": reason, "pnl_usd": pnl,
                       "r_multiple": pnl / p["risk_usd"] if p["risk_usd"] else 0.0,
                       "bars_held": i - p["i"], "equity": equity})
        curve.append((f[i]["ct"], equity))
        position = None

    for i in range(test_start, n):
        bar = f[i]
        if position and i > position["i"]:
            p = position
            stop_reason = "trailing_stop" if p["trailed"] else "stop"
            if bar["o"] <= p["stop"]:
                close(i, bar["o"], stop_reason)
            elif bar["l"] <= p["stop"]:
                close(i, p["stop"], stop_reason)
            else:
                if (partial_tp_ratio > 0.0 and not p["partial_closed"]
                        and bar["h"] >= p["partial_target"]):
                    part_exit = max(bar["o"], p["partial_target"])
                    part_qty = p["qty"] * partial_tp_ratio
                    part_pnl = (part_qty * (part_exit - p["entry"])
                                - part_qty * (p["entry"] + part_exit) * cost)
                    p["realized_pnl"] += part_pnl
                    p["qty"] -= part_qty
                    p["partial_closed"] = True
                if bar["o"] >= p["target"]:
                    close(i, bar["o"], "target")
                elif bar["h"] >= p["target"]:
                    close(i, p["target"], "target")
                elif i - p["i"] >= horizon:
                    close(i, bar["c"], "timeout")
                else:
                    p["peak"] = max(p["peak"], bar["h"])
                    if p["peak"] - p["entry"] >= trail_trigger * p["atr"]:
                        be_floor = p["entry"] * (1.0 + max(be_buffer, 2.2 * cost))
                        candidate = max(be_floor, p["peak"] - trail_dist_atr * p["atr"])
                        if candidate > p["stop"]:
                            p["stop"] = candidate
                            p["trailed"] = True
                    if p["trailed"] and bar["c"] <= p["stop"]:
                        close(i, p["stop"], "trailing_stop")
        if position or np.isnan(probs[i]) or probs[i] < threshold:
            continue
        stats["signals"] += 1
        if i <= cooldown_until:
            stats["blocked_cooldown"] += 1
            continue
        mtf_mult = float(X[i, MTF_INDEX]) if use_mtf else 1.0
        if use_mtf and (mtf_mult <= 0.0 or (strict_4h and mtf_mult < 1.0)):
            stats["blocked_mtf"] += 1
            continue
        if use_entry_gate:
            prev_row = X[i - 1] if (i > 0 and valid[i - 1]) else None
            gate_ok, _ = ml_model.evaluate_entry_gate(bar, X[i], prev_row)
            if not gate_ok:
                stats["blocked_gate"] += 1
                continue
        entry = bar["c"]
        if round_trip_cost > 0 and (cfg["target_atr"] * atr[i] / entry) < min_target_move:
            stats["blocked_cost"] += 1
            continue
        if use_mtf and mtf_mult < 1.0:
            stats["half_size_mtf"] += 1
        day = daily.setdefault(bar["t"] // DAY_MS, {"count": 0, "loss": 0.0})
        eff_risk = float(risk["risk_usd"]) * mtf_mult
        if (risk["max_trades"] and day["count"] >= risk["max_trades"]) or \
                (risk["daily_limit"] and day["loss"] + eff_risk > risk["daily_limit"]):
            stats["blocked_risk"] += 1
            continue
        try:
            plan, size = _plan_and_size(entry, entry - cfg["stop_atr"] * atr[i],
                                        entry + cfg["target_atr"] * atr[i], filters, risk,
                                        size_mult=mtf_mult)
        except TradingError:
            stats["blocked_size"] += 1
            continue
        day["count"] += 1
        position = {"i": i, "t": bar["ct"], "entry": float(plan["entry"]),
                    "initial_stop": float(plan["stop"]), "stop": float(plan["stop"]),
                    "partial_target": float(plan["entry"]) + partial_tp_atr * float(atr[i]),
                    "target": float(plan["target"]),
                    "initial_qty": float(size["quantity"]), "qty": float(size["quantity"]),
                    "realized_pnl": 0.0, "partial_closed": False,
                    "risk_usd": float(size["estimated_loss_usd"]), "prob": float(probs[i]),
                    "atr": float(atr[i]), "peak": float(plan["entry"]),
                    "trailed": False, "size_mult": mtf_mult}
    if position:
        close(n - 1, f[-1]["c"], "end")

    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    test_labels = labels[test_start:][~np.isnan(labels[test_start:])]
    scored = probs[~np.isnan(probs)]
    breakeven_wr = ml_model.breakeven_rate(cfg["stop_atr"], cfg["target_atr"])
    dyn_threshold = ml_model.dynamic_threshold(cfg["stop_atr"], cfg["target_atr"], cfg["ev_multiplier"])
    rr_ratio = cfg["target_atr"] / cfg["stop_atr"]
    report = {
        "symbol": symbol, "interval": interval, "market": market,
        "period": {"history_start": datetime.datetime.fromtimestamp(f[0]["t"] / 1000, UTC).isoformat(),
                   "test_start": datetime.datetime.fromtimestamp(f[test_start]["t"] / 1000, UTC).isoformat(),
                   "end": datetime.datetime.fromtimestamp(f[-1]["ct"] / 1000, UTC).isoformat(),
                   "test_days": (f[-1]["t"] - f[test_start]["t"]) / DAY_MS, "candles": n},
        "strategy": {"threshold": threshold, "required_ev_threshold": dyn_threshold,
                     "ev_multiplier": cfg["ev_multiplier"], "mtf_filter": use_mtf,
                     "target_atr": cfg["target_atr"], "stop_atr": cfg["stop_atr"],
                     "reward_risk_ratio": rr_ratio, "breakeven_win_rate": breakeven_wr,
                     "horizon_bars": horizon, "risk_per_trade_usd": float(risk["risk_usd"]),
                     "fee_rate": risk["fee"], "slippage_rate": risk["slippage"],
                     "models_trained": models, "retrain_bars": retrain},
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else None,
        "breakeven_win_rate": breakeven_wr,
        "required_ev_threshold": threshold,
        "reward_risk_ratio": rr_ratio,
        "total_profit_usd": sum(pnls), "return_pct": sum(pnls) / risk["account"] * 100,
        "profit_factor": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "avg_r": sum(t["r_multiple"] for t in trades) / len(trades) if trades else None,
        "expectancy_usd": sum(pnls) / len(trades) if trades else None,
        "max_drawdown_usd": max_dd, "max_drawdown_pct": max_dd_pct,
        "start_equity": risk["account"], "end_equity": equity,
        "exits": {reason: sum(1 for t in trades if t["reason"] == reason)
                  for reason in ("target", "trailing_stop", "stop", "timeout", "end")},
        "filters": stats,
        "base_rate_test": float(test_labels.mean()) if len(test_labels) else None,
        "mean_probability": float(scored.mean()) if len(scored) else None,
        "max_probability": float(scored.max()) if len(scored) else None,
        "buy_and_hold_pct": (f[-1]["c"] / f[test_start]["c"] - 1) * 100,
        "trade_list": trades, "equity_curve": curve,
    }
    return report


def format_report(r):
    def pct(v):
        return "—" if v is None else "%.1f%%" % (v * 100)

    def num(v, fmt="%.2f"):
        return "—" if v is None else fmt % v

    s, p = r["strategy"], r["period"]
    lines = [
        "=" * 64,
        " BACKTEST · %s · %s · %s" % (r["symbol"], r["interval"], r["market"].upper()),
        "=" * 64,
        " Teszt időszak : %s → %s (%.1f nap, walk-forward)" % (p["test_start"][:16], p["end"][:16], p["test_days"]),
        " Stratégia     : ML p >= %.1f%%%s · cél +%.1f×ATR · stop −%.1f×ATR · max %d gyertya" % (
            s["threshold"] * 100, " + MTF (1h/4h EMA50)" if s["mtf_filter"] else " (MTF szűrő NÉLKÜL)",
            s["target_atr"], s["stop_atr"], s["horizon_bars"]),
        " Modellek      : %d újratanítás (%d gyertyánként)" % (s["models_trained"], s["retrain_bars"]),
        "-" * 64,
        " Kötések       : %d  (nyerő %d / vesztes %d)" % (r["trades"], r["wins"], r["losses"]),
        " Win Rate      : %s   (alapráta a tesztben: %s)" % (pct(r["win_rate"]), pct(r["base_rate_test"])),
        " Break-even Win Rate : %s   (Kockázat / (Kockázat + Hozam) = %.1f / (%.1f + %.1f), R:R = 1:%.2f)" % (
            pct(r["breakeven_win_rate"]), s["stop_atr"], s["stop_atr"], s["target_atr"], r["reward_risk_ratio"]),
        " Dinamikus Küszöb (Required +EV Threshold) : %s   (%.2f × Break-even WR)" % (
            pct(r.get("required_ev_threshold", s["threshold"])), s.get("ev_multiplier", 1.25)),
        " Total Profit  : %+.2f USD  (%+.2f%% a %.0f USD számlán)" % (
            r["total_profit_usd"], r["return_pct"], r["start_equity"]),
        " Max Drawdown  : %.2f USD  (%.2f%%)" % (r["max_drawdown_usd"], r["max_drawdown_pct"]),
        " Profit Factor : %s · Átlag R: %s · Várható érték/kötés: %s USD" % (
            num(r["profit_factor"]), num(r["avg_r"], "%+.2f"), num(r["expectancy_usd"], "%+.2f")),
        " Kilépések     : target %(target)d · trailing stop %(trailing_stop)d · stop %(stop)d · timeout %(timeout)d · vége %(end)d" % r["exits"],
        " Jelzések      : %d küszöb feletti · MTF 0.5x %d · MTF tiltva %d · PA-kapu %d · cooldown %d · költség %d · limit %d" % (
            r["filters"]["signals"], r["filters"].get("half_size_mtf", 0),
            r["filters"]["blocked_mtf"], r["filters"].get("blocked_gate", 0),
            r["filters"].get("blocked_cooldown", 0),
            r["filters"].get("blocked_cost", 0), r["filters"]["blocked_risk"]),
        " Valószínűség  : átlag %s · maximum %s" % (pct(r["mean_probability"]), pct(r["max_probability"])),
        " Buy & Hold    : %+.2f%% ugyanebben az időszakban" % r["buy_and_hold_pct"],
        "=" * 64,
    ]
    if not r["trades"]:
        lines.append(" Nem volt belépő: a modell egyszer sem adott %.0f%% feletti esélyt MTF-egyezéssel."
                     % (s["threshold"] * 100))
        lines.append(" Próbáld alacsonyabb küszöbbel, pl.: --threshold 0.6")
    lines.append(" Figyelem: múltbeli eredmény, nem jövőbeli garancia.")
    return "\n".join(lines)


def save_report(report, csv_too=False):
    folder = os.path.join(ROOT, "data", "backtests")
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = os.path.join(folder, "%s_%s_%s" % (report["symbol"], report["interval"], stamp))
    with open(base + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    if csv_too and report["trade_list"]:
        with open(base + ".csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["trade_list"][0]))
            writer.writeheader()
            writer.writerows(report["trade_list"])
    return base


# =============================================================== auto-paper

class PaperBot:
    """Background auto-paper trader using the same ML + MTF rules as the backtest."""

    def __init__(self, config=None, client=None, interval=None, threshold=None,
                 retrain_minutes=60, symbols=None):
        self.config = config or load_config()
        self.client = client or BinanceClient(self.config)
        self.interval = interval or self.config["default_interval"]
        self.cfg = ml_model.settings(self.config)
        self.threshold = self.cfg["auto_threshold"] if threshold is None else float(threshold)
        self.cfg["auto_threshold"] = self.threshold
        self.retrain_seconds = retrain_minutes * 60
        self.symbols = symbols or [c["symbol"] for c in self.config["symbols"]]
        self.risk = _risk(self.config)
        self.htf_cache = mtf.HTFCache(self.client, ttl=300)
        self.models = {}
        self.last_bar = {}

    def _model(self, symbol, now):
        cached = self.models.get(symbol)
        if cached and time.time() - cached[0] < self.retrain_seconds:
            return cached[1]
        days = int(self.config.get("history_days", 30))
        step = INTERVAL_MS[self.interval]
        days = min(days, max(3, 20_000 // (DAY_MS // step)))
        bars, htf_bars, _, _ = fetch_history(self.client, symbol, self.interval, days, now)
        config = dict(self.config, ml=dict(self.config.get("ml", {}), auto_threshold=self.threshold))
        bundle = ml_model.train_model(bars, self.interval, now=now, config=config, htf_bars=htf_bars)
        self.models[symbol] = (time.time(), bundle)
        LOG.info("%s: modell tanítva (%d minta, holdout AUC %s)", symbol, bundle["samples"],
                 "%.3f" % bundle["metrics"]["auc"] if bundle["metrics"]["auc"] is not None else "—")
        return bundle

    def cycle(self, now=None):
        """One scan: close hits, then evaluate every symbol. Returns opened trades."""
        now = _now(now)
        live = {}
        for symbol in self.symbols:
            try:
                live[symbol] = self.client.get_klines(symbol, self.interval, limit=1000, include_open=True)
            except Exception as exc:  # network / Binance errors must not kill the daemon
                LOG.warning("%s: adatlekérés sikertelen: %s", symbol, exc)
        quotes = {s: {"price": r["bars"][-1]["close"], "high": r["bars"][-1]["high"],
                      "low": r["bars"][-1]["low"]} for s, r in live.items() if r["bars"]}
        for trade in auto_close_trades(quotes, self.risk["fee"], self.risk["slippage"], config=self.config):
            LOG.info("%s: pozíció automatikusan zárva (%s) @ %s, P&L %s USD", trade["symbol"],
                     trade["exit_reason"], trade["exit_price"], trade["realized_pnl_usd"])
        opened = []
        for symbol, result in live.items():
            try:
                trade = self._evaluate(symbol, result, now)
                if trade:
                    opened.append(trade)
            except Exception as exc:
                LOG.warning("%s: kiértékelés sikertelen: %s", symbol, exc)
        return opened

    def _evaluate(self, symbol, result, now):
        bars = result["bars"]
        if not bars:
            return None
        bundle = self._model(symbol, now)
        prediction = ml_model.predict_setup(bundle, bars, now=now, htf_bars=self.htf_cache.get(symbol))
        if self.last_bar.get(symbol) == prediction["bar_time"]:
            return None                       # this closed candle was already judged
        self.last_bar[symbol] = prediction["bar_time"]
        size_mult = float(prediction.get("size_multiplier",
                                         1.0 if prediction.get("mtf_aligned") else 0.0))
        LOG.info("%s: ML p=%.1f%% (szűrt %.1f%%, +EV küszöb %.1f%%) · makro %s (méret %.0f%%)",
                 symbol, prediction["raw_probability"] * 100,
                 prediction["probability"] * 100, self.threshold * 100,
                 prediction["mtf"]["label"] if prediction["mtf"] else "—",
                 size_mult * 100)
        if not (prediction["probability"] >= self.threshold and size_mult > 0.0):
            return None
        if not prediction.get("cost_gate_passed", True):
            LOG.info("%s: jelzés blokkolva a költség-kapu által (cél %.2f%% < min %.2f%%)",
                     symbol, prediction.get("target_pct", 0.0), prediction.get("min_target_pct", 0.0))
            return None
        if (self.cfg.get("entry_gate", True)
                and self.threshold >= self.cfg["dynamic_threshold"] * 0.5
                and not prediction.get("entry_gate_passed", True)):
            LOG.info("%s: jelzés blokkolva a Price-Action kapu által (%s)",
                     symbol, ", ".join(prediction.get("entry_gate_reasons", [])))
            return None
        trades = list_trades(config=self.config)
        if any(t["status"] == "open" and t["symbol"] == symbol for t in trades):
            LOG.info("%s: jelzés, de már van nyitott pozíció – kihagyva", symbol)
            return None
        entry = float(bars[-1]["close"])      # live price of the forming candle
        atr_value = prediction["atr"]
        plan, size = _plan_and_size(entry, entry - self.cfg["stop_atr"] * atr_value,
                                    entry + self.cfg["target_atr"] * atr_value, result["filters"],
                                    self.risk, size_mult=size_mult)
        daily = daily_risk(trades, fee_rate=self.risk["fee"], slippage_rate=self.risk["slippage"])
        check = validate_risk(plan, size["quantity"], self.risk["raw"],
                              current_daily_loss=daily["loss_usd"], trades_today=daily["trades_today"])
        if not check["passed"]:
            LOG.info("%s: jelzés blokkolva: %s", symbol, "; ".join(check["problems"]))
            return None
        p_ratio = float(self.cfg.get("partial_tp_ratio", 0.5))
        p_target = (float(plan["entry"]) + float(self.cfg.get("partial_tp_atr", 1.8)) * atr_value) \
            if p_ratio > 0 else None
        trade = record_entry(symbol, "long", size["quantity"], plan["entry"], plan["stop"], plan["target"],
                             note="auto-bot · ML %.1f%% · MTF %s (%.0f%% méret)" % (
                                 prediction["raw_probability"] * 100,
                                 prediction["mtf"]["label"] if prediction["mtf"] else "—",
                                 size_mult * 100),
                             config=self.config, atr=format(atr_value, ".12g"),
                             partial_target=format(p_target, ".12g") if p_target else None,
                             partial_ratio=format(p_ratio, ".4g") if p_target else None)
        LOG.info("%s: 🤖 AUTO PAPER LONG %s @ %s · stop %s · target %s", symbol, trade["quantity"],
                 trade["entry_price"], trade["stop"], trade["target"])
        return trade

    def run_forever(self, every=60):
        LOG.info("Auto-paper robot indul: %s · %s · küszöb %.0f%% · %d mp-enként",
                 ", ".join(self.symbols), self.interval, self.threshold * 100, every)
        while True:
            started = time.time()
            try:
                self.cycle()
            except Exception:
                LOG.exception("Váratlan hiba a ciklusban")
            time.sleep(max(1.0, every - (time.time() - started)))


# ===================================================================== CLI

def _setup_logging():
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(ROOT, "data", "bot.log"), encoding="utf-8")])


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Crypto ML + MTF robot (backtest / auto-paper)")
    sub = parser.add_subparsers(dest="mode", required=True)
    bt = sub.add_parser("backtest", help="walk-forward historical replay")
    bt.add_argument("symbol")
    bt.add_argument("--interval", choices=["1m", "5m", "15m", "1h"], default=None,
                    help="trading timeframe (default: config.json default_interval = 1h)")
    bt.add_argument("--days", type=float, default=120, help="total history to download (default 120)")
    bt.add_argument("--train-days", type=float, default=60, help="initial training window (default 60)")
    bt.add_argument("--retrain-days", type=float, default=2, help="walk-forward retrain step (default 2)")
    bt.add_argument("--threshold", type=float, default=None,
                    help="ML probability threshold (default: dynamic 1.25x breakeven)")
    bt.add_argument("--no-mtf", action="store_true", help="disable the 1h/4h trend filter")
    bt.add_argument("--csv", action="store_true", help="also export the trade list as CSV")
    pp = sub.add_parser("paper", help="auto-paper daemon")
    pp.add_argument("--interval", choices=["1m", "5m", "15m", "1h"], default=None)
    pp.add_argument("--every", type=int, default=60, help="seconds between scans (default 60)")
    pp.add_argument("--threshold", type=float, default=None)
    pp.add_argument("--retrain-minutes", type=int, default=60)
    pp.add_argument("--symbols", nargs="*", default=None)
    pp.add_argument("--once", action="store_true", help="run a single scan and exit")
    args = parser.parse_args(argv)
    _setup_logging()
    config = load_config()
    if args.mode == "backtest":
        symbol = args.symbol.upper()
        if symbol not in {c["symbol"] for c in config["symbols"]}:
            parser.error("symbol must be configured in config.json")
        report = run_backtest(symbol, args.interval, args.days, args.train_days, args.retrain_days,
                              args.threshold, not args.no_mtf, config=config,
                              progress=lambda m: LOG.info(m))
        print(format_report(report))
        print(" Riport mentve: %s.json" % save_report(report, args.csv))
        return 0
    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    bot = PaperBot(config, interval=args.interval, threshold=args.threshold,
                   retrain_minutes=args.retrain_minutes, symbols=symbols)
    if args.once:
        bot.cycle()
        return 0
    try:
        bot.run_forever(args.every)
    except KeyboardInterrupt:
        LOG.info("Robot leállítva.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

