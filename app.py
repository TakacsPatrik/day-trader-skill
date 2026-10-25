"""Crypto terminal: live scanner, chart, Time Machine, paper journal, alerts."""

import concurrent.futures
import datetime
import importlib.util
import os
from decimal import Decimal

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from binance_client import INTERVAL_MS, BinanceClient, ROOT, load_config
import technical_analysis as ta
from crypto_engine import (TradingError, analyze_setup, build_plan, dec, evaluate_breakout,
                           simulate_trade, size_position, validate_risk)
from paper_trading import (close_trade, daily_risk, list_trades, record_entry,
                           unrealized_pnl)

CACHE_TTL_SECONDS = 5
REFRESH_CHOICES = (5, 10, 30)
DEFAULT_TARGET_R = Decimal("2")
UP, DOWN, ACCENT = "#26a69a", "#ef5350", "#2962ff"
BG, GRID, MUTED = "#131722", "#2a2e39", "#787b86"
UTC = datetime.timezone.utc

SIGNAL_LABELS = {"triggered": ("🟢", "BREAKOUT"), "waiting": ("👀", "WATCHING"),
                 "forming": ("⏳", "FORMING"), "no_data": ("⚠️", "NO DATA"),
                 "error": ("⚠️", "ERROR")}
ALERT_LABELS = {"above": "Ár fölé emelkedik", "below": "Ár alá esik",
                "target": "Target elérése", "stop": "Stop elérése"}
LAYER_CHOICES = ["EMA 20", "EMA 50", "EMA 200", "VWAP", "Bollinger", "Támasz / Ellenállás"]
DEFAULT_LAYERS = ["EMA 20", "EMA 50", "Bollinger", "Támasz / Ellenállás"]
STATUS_ICONS = {"ok": "✅", "warn": "⚠️", "bad": "❌"}

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1650px;}
header[data-testid="stHeader"] {background: transparent;}
.terminal-title {font-size: 1.5rem; font-weight: 700; letter-spacing: .08em; margin: 0 0 .6rem 0;}
.terminal-title span {color: #2962ff;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 10px;}
[data-testid="stMetric"] {background: #131722; border-radius: 8px; padding: .35rem .6rem;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-weight: 600; font-size: 1.45rem;}
[data-testid="stMetricValue"] * {overflow: visible !important; text-overflow: clip !important;}
.sig {display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: .72rem;
      font-weight: 700; letter-spacing: .05em; background: #2a2e39; color: #d1d4dc;}
.sig-triggered, .sig-strong_buy {background: rgba(38,166,154,.22); color: #26a69a;}
.sig-waiting, .sig-buy_watch {background: rgba(41,98,255,.20); color: #5c8dff;}
.sig-forming, .sig-no_data, .sig-error, .sig-neutral {background: rgba(255,179,0,.18); color: #ffb300;}
.sig-avoid {background: rgba(239,83,80,.20); color: #ef5350;}
.src {color: #787b86; font-size: .7rem; margin-left: 6px;}
.mini {color: #9598a1; font-size: .72rem; white-space: nowrap; overflow: hidden;
       text-overflow: ellipsis; margin-top: 3px;}
.mini b {color: #d1d4dc;}
.hero {background: #131722; border: 1px solid #2a2e39; border-radius: 10px; padding: .8rem 1rem;
       margin: .4rem 0 .8rem 0;}
.hero .lbl {color: #9598a1; font-size: .85rem;}
.hero .val {font-size: 2.1rem; font-weight: 700; color: #26a69a; font-variant-numeric: tabular-nums;}
.hero .val.neg {color: #ef5350;}
.hero .sub {color: #9598a1; font-size: .8rem;}
.live-bar {background: #131722; border: 1px solid #2a2e39; border-radius: 8px;
           padding: .45rem .8rem; margin: .5rem 0 1rem 0; font-size: .85rem;}
.live-bar b {font-variant-numeric: tabular-nums;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
"""


def _load_module(name, *parts):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *parts))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_news = _load_module("crypto_news_module", "trader-news", "news.py")
_alerts = _load_module("crypto_alerts_module", "trader-alerts", "alerts.py")


@st.cache_resource
def _client():
    return BinanceClient(load_config())


def _fetch_symbol(client, symbol, interval, kwargs, cost_rate, now):
    try:
        result = client.get_klines(symbol, interval, include_open=True, **kwargs)
    except Exception as exc:
        return None, None, str(exc)
    try:
        advice = analyze_setup(result["bars"], symbol, interval, result["filters"],
                               now=now, cost_rate=cost_rate)
    except Exception as exc:
        advice = {"status": "error", "reasons": [{"status": "bad", "text": "Elemzési hiba: %s" % exc}]}
    return result, advice, None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_all(interval):
    """Recent candles (>= 24h, including the forming one) plus the setup analysis per coin."""
    config = load_config()
    client = _client()
    now = datetime.datetime.now(UTC)
    end_ms = int(now.timestamp() * 1000)
    bars_per_day = 86_400_000 // INTERVAL_MS[interval]
    if bars_per_day + 1 <= 1000:
        kwargs = {"limit": 1000}
    else:
        kwargs = {"limit": bars_per_day + 1, "start_time": end_ms - 86_400_000, "end_time": end_ms}
    risk = config.get("risk", {})
    cost_rate = float(risk.get("fee_rate", 0)) + float(risk.get("slippage_rate", 0))
    symbols = [row["symbol"] for row in config["symbols"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(symbols))) as pool:
        outcomes = list(pool.map(
            lambda symbol: _fetch_symbol(client, symbol, interval, kwargs, cost_rate, now), symbols))
    return {"interval": interval, "fetched_at": now,
            "results": {s: r for s, (r, _, _) in zip(symbols, outcomes) if r is not None},
            "advice": {s: a for s, (_, a, _) in zip(symbols, outcomes) if a is not None},
            "errors": {s: e for s, (_, _, e) in zip(symbols, outcomes) if e}}


def _last_24h(bars):
    if not bars:
        return bars
    cutoff = bars[-1]["open_time"] - 86_400_000
    return [bar for bar in bars if bar["open_time"] >= cutoff]


@st.cache_data(ttl=900, show_spinner=False)
def _load_news():
    return _news.fetch_crypto_news(load_config())


def _plain(value):
    return format(dec(value).normalize(), "f")


def _fmt_price(value):
    number = dec(value)
    places = 2 if abs(number) >= 100 else 4 if abs(number) >= 1 else 5
    return "{:,.{}f}".format(number, places)


def _usd(value, signed=False):
    number = dec(value)
    return "%s$%s" % ("-" if number < 0 else "+" if signed and number > 0 else "",
                      "{:,.2f}".format(abs(number)))


def _clock(moment):
    local = moment.astimezone()
    offset = local.strftime("%z")
    zone = "UTC%s:%s" % (offset[:3], offset[3:]) if offset else "helyi"
    return (moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "%s (%s)" % (local.strftime("%H:%M:%S"), zone))


def _snapshot(config, interval):
    data = _load_all(interval)
    now = datetime.datetime.now(UTC)
    now_ms = int(now.timestamp() * 1000)
    rows = []
    for coin in config["symbols"]:
        symbol = coin["symbol"]
        result = data["results"].get(symbol)
        row = {"symbol": symbol, "base": coin.get("base", symbol[:-4]), "result": result,
               "error": data["errors"].get(symbol), "price": None, "change": None,
               "signal": {"status": "no_data", "reason": data["errors"].get(symbol, "No data")},
               "stale": False}
        if result and result["bars"]:
            bars = result["bars"]
            try:
                row["signal"] = evaluate_breakout(bars, symbol, interval, now,
                                                  config["opening_range_minutes"])
            except Exception as exc:
                row["signal"] = {"status": "error", "reason": str(exc)}
            row["price"] = bars[-1]["close"]
            day_bars = _last_24h(bars)
            if day_bars[0]["open"]:
                row["change"] = (bars[-1]["close"] / day_bars[0]["open"] - 1) * 100
            row["stale"] = now_ms - bars[-1]["open_time"] > 3 * INTERVAL_MS[interval]
        advice = data["advice"].get(symbol)
        row["advice"] = advice if advice and advice.get("status") == "ok" else None
        rows.append(row)
    return rows, data, now


def _status_bar(data, now, refresh, auto):
    utc_text, local_text = _clock(now)
    fetched = data["fetched_at"].astimezone(UTC).strftime("%H:%M:%S UTC")
    if auto:
        mode = "🟢 <b>ÉLŐ</b> · automatikus frissítés %d másodpercenként" % refresh
    else:
        mode = "⏸ <b>PAUZA</b> · automatikus frissítés kikapcsolva"
    st.markdown(
        '<div class="live-bar">%s · Utolsó frissítés: <b>%s</b> / <b>%s</b> · '
        'Binance adat lekérve: <b>%s</b></div>' % (mode, utc_text, local_text, fetched),
        unsafe_allow_html=True)


def _render_ticker(config, interval, refresh, auto):
    @st.fragment(run_every=refresh if auto else None)
    def ticker():
        rows, data, now = _snapshot(config, interval)
        per_row = 6
        for offset in range(0, len(rows), per_row):
            columns = st.columns(per_row)
            for column, row in zip(columns, rows[offset:offset + per_row]):
                status = row["signal"]["status"]
                icon, label = SIGNAL_LABELS.get(status, ("⚠️", status.upper()))
                with column.container(border=True):
                    if row["price"] is None:
                        st.metric(row["base"], "—")
                    else:
                        st.metric(row["base"], _fmt_price(row["price"]),
                                  "%+.2f%%" % float(row["change"]) if row["change"] is not None else None)
                    market = row["result"]["market"].upper() if row["result"] else "—"
                    stale = " · ELAVULT" if row["stale"] else ""
                    advice = row["advice"]
                    if advice:
                        st.markdown(
                            '<span class="sig sig-%s">%s · %d</span>'
                            '<div class="mini" title="%s">🕯 %s</div>'
                            '<div class="mini">🎯 T1 <b>%+.2f%%</b> · SL <b>%+.2f%%</b></div>' %
                            (advice["rating"].lower(), advice["short_label"], advice["score"],
                             advice["pattern_summary"], advice["pattern_summary"],
                             advice["t1_pct"] or 0, advice["stop_pct"] or 0),
                            unsafe_allow_html=True)
                    st.markdown('<span class="sig sig-%s">%s %s%s</span><span class="src">%s</span>' %
                                (status, icon, label, stale, market), unsafe_allow_html=True)
        _status_bar(data, now, refresh, auto)
        for symbol, message in data["errors"].items():
            st.warning("%s: %s" % (symbol, message))

    ticker()


def _active_plan(selected, signal, advice=None, actionable_only=False):
    """Plan levels in priority order: loaded/calculator plan, advisor, UTC breakout."""
    saved = st.session_state.get("last_plan")
    if saved and saved.get("symbol") == selected:
        return {"entry": dec(saved["entry"]), "stop": dec(saved["stop"]), "target": dec(saved["target"]),
                "t2": dec(saved["t2"]) if saved.get("t2") else None,
                "source": saved.get("source", "kalkulátor terv")}
    if advice and advice.get("entry") is not None and (advice["actionable"] or not actionable_only):
        return {"entry": advice["entry"], "stop": advice["stop"], "target": advice["t1"],
                "t2": advice["t2"], "source": "Okos Tanácsadó (T1 / T2)"}
    sig, opening = signal.get("signal"), signal.get("opening_range")
    if sig and opening and opening["low"] < sig["entry"]:
        entry, stop = sig["entry"], opening["low"]
        return {"entry": entry, "stop": stop, "target": entry + (entry - stop) * DEFAULT_TARGET_R,
                "t2": None, "source": "UTC breakout (OR low stop, %sR)" % DEFAULT_TARGET_R}
    return None


def _candlestick(symbol, result, signal, plan, candles, layers=(), advice=None):
    bars = result["bars"][-candles:]
    step = INTERVAL_MS[result["interval"]]
    xs = [datetime.datetime.fromtimestamp(bar["open_time"] / 1000, UTC) for bar in bars]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(
        x=xs, open=[float(b["open"]) for b in bars], high=[float(b["high"]) for b in bars],
        low=[float(b["low"]) for b in bars], close=[float(b["close"]) for b in bars],
        increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
        name=symbol, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(
        x=xs, y=[float(b["volume"]) for b in bars], name="Volume", showlegend=False,
        marker_color=[UP if b["close"] >= b["open"] else DOWN for b in bars], opacity=0.55),
        row=2, col=1)

    overlays = ta.chart_series(result["bars"], step)

    def line(key, name, color, dash=None, width=1.3):
        fig.add_trace(go.Scatter(x=xs, y=overlays[key][-candles:], mode="lines", name=name,
                                 line=dict(color=color, width=width, dash=dash), connectgaps=False),
                      row=1, col=1)

    if "EMA 20" in layers:
        line("ema20", "EMA 20", "#ffb300")
    if "EMA 50" in layers:
        line("ema50", "EMA 50", "#ab47bc")
    if "EMA 200" in layers:
        line("ema200", "EMA 200", "#e0e0e0", width=1.6)
    if "VWAP" in layers:
        line("vwap_daily", "VWAP (napi)", "#26c6da", "dashdot")
    if "Bollinger" in layers:
        line("bb_up", "BB felső", "rgba(92,141,255,0.8)")
        fig.add_trace(go.Scatter(x=xs, y=overlays["bb_low"][-candles:], mode="lines", name="BB alsó",
                                 line=dict(color="rgba(92,141,255,0.8)", width=1.3),
                                 fill="tonexty", fillcolor="rgba(92,141,255,0.07)"), row=1, col=1)
        line("bb_mid", "BB közép", "rgba(92,141,255,0.5)", "dot", 1)

    prices = [float(b["high"]) for b in bars] + [float(b["low"]) for b in bars]
    opening = signal.get("opening_range")
    if opening:
        fig.add_shape(type="rect", row=1, col=1, layer="below", line_width=0,
                      x0=datetime.datetime.fromtimestamp(opening["start_utc_ms"] / 1000, UTC),
                      x1=xs[-1] + datetime.timedelta(milliseconds=step),
                      y0=float(opening["low"]), y1=float(opening["high"]),
                      fillcolor="rgba(41,98,255,0.10)")
        for label, price in (("OR High", opening["high"]), ("OR Low", opening["low"])):
            fig.add_hline(y=float(price), row=1, col=1, line_dash="dot", line_color="#5c8dff",
                          annotation_text="%s %s" % (label, _fmt_price(price)),
                          annotation_position="right", annotation_font_color="#5c8dff")
            prices.append(float(price))
    if plan:
        levels = [("Entry", plan["entry"], "#ffb300"), ("Stop-Loss", plan["stop"], DOWN),
                  ("Target", plan["target"], "#00e676")]
        if plan.get("t2"):
            levels.append(("Target 2", plan["t2"], "#69f0ae"))
        for label, price, color in levels:
            fig.add_hline(y=float(price), row=1, col=1, line_dash="dash", line_width=1.6,
                          line_color=color, annotation_text="%s %s" % (label, _fmt_price(price)),
                          annotation_position="right", annotation_font_color=color)
            prices.append(float(price))
    last_close = float(bars[-1]["close"])
    fig.add_hline(y=last_close, row=1, col=1, line_dash="dot", line_color=MUTED, line_width=1)

    pad = (max(prices) - min(prices)) * 0.04 or last_close * 0.01
    low_view, high_view = min(prices) - pad, max(prices) + pad
    if advice and "Támasz / Ellenállás" in layers:
        for zones, label, color in ((advice["supports"], "Támasz", "rgba(38,166,154,0.18)"),
                                    (advice["resistances"], "Ellenállás", "rgba(239,83,80,0.18)")):
            for zone in zones:
                if zone["high"] < low_view or zone["low"] > high_view:
                    continue
                fig.add_hrect(y0=zone["low"], y1=zone["high"], row=1, col=1, fillcolor=color,
                              line_width=0, layer="below",
                              annotation_text="%s %s (%dx)" % (label[0], _fmt_price(zone["level"]),
                                                              zone["touches"]),
                              annotation_position="top left", annotation_font_size=10,
                              annotation_font_color="#9598a1")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=640,
        margin=dict(l=10, r=130, t=45, b=10), hovermode="x unified", dragmode="pan",
        title="%s · %s · %s" % (symbol, result["market"].upper(), result["interval"]),
        xaxis_rangeslider_visible=False, uirevision="%s-%s" % (symbol, result["interval"]),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)))
    fig.update_xaxes(gridcolor=GRID, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, side="right")
    fig.update_yaxes(range=[low_view, high_view], row=1, col=1)
    return fig


def _scanner_table(rows):
    table = []
    for row in rows:
        icon, label = SIGNAL_LABELS.get(row["signal"]["status"], ("⚠️", "?"))
        opening = row["signal"].get("opening_range")
        change = row["change"]
        advice = row["advice"]
        table.append({
            "Coin": row["symbol"],
            "Ár (USDT)": _fmt_price(row["price"]) if row["price"] is not None else "—",
            "24h": ("%s %+.2f%%" % ("▲" if change >= 0 else "▼", float(change))
                    if change is not None else "—"),
            "Setup Score": advice["score"] if advice else None,
            "Tanács": advice["short_label"] if advice else "—",
            "Mintázat": advice["pattern_summary"] if advice else "—",
            "T1 hozam": "%+.2f%%" % advice["t1_pct"] if advice and advice["t1_pct"] is not None else "—",
            "Stop": "%+.2f%%" % advice["stop_pct"] if advice and advice["stop_pct"] is not None else "—",
            "UTC breakout": "%s %s%s" % (icon, label, " (elavult)" if row["stale"] else ""),
            "OR High": _fmt_price(opening["high"]) if opening else "—",
            "OR Low": _fmt_price(opening["low"]) if opening else "—",
            "Utolsó gyertya (UTC)": (row["result"]["bars"][-1]["open_time_utc"][11:19]
                                     if row["result"] and row["result"]["bars"] else "—"),
            "Forrás": row["result"]["market"].upper() if row["result"] else "—"})
    return table


def _load_advisor_plan(config, selected, advice, filters):
    """Copy the advisor's levels into the calculator and the plan used by Time Machine."""
    cfg_risk = config.get("risk", {})
    plan = build_plan(advice["entry"], advice["stop"], DEFAULT_TARGET_R, filters, target=advice["t1"])
    size = size_position(plan["entry"], plan["stop"], cfg_risk.get("risk_per_trade_usd", 50), filters,
                         cfg_risk.get("max_position_usd"), cfg_risk.get("fee_rate", 0),
                         cfg_risk.get("slippage_rate", 0))
    st.session_state["last_plan"] = {
        "symbol": selected, "entry": str(plan["entry"]), "stop": str(plan["stop"]),
        "target": str(plan["target"]), "t2": str(advice["t2"]), "quantity": str(size["quantity"]),
        "notional": str(size["notional_usd"]), "loss": str(size["estimated_loss_usd"]),
        "step": str(size["step_size"]), "source": "Okos Tanácsadó (betöltve)"}
    for name, value in (("entry", plan["entry"]), ("stop", plan["stop"]), ("target", plan["target"])):
        st.session_state["calc_%s_%s" % (name, selected)] = _plain(value)


def _render_advisor(config, selected, advice, filters):
    st.subheader("🧠 Okos Tanácsadó & Mintázatok")
    if not advice or advice.get("status") != "ok":
        text = advice["advice"] if advice and advice.get("advice") else "Az elemzés nem elérhető."
        st.info(text)
        return
    notice = st.session_state.pop("plan_notice", None)
    if notice:
        st.success(notice)
    rating = advice["rating"]
    box = {"STRONG_BUY": st.success, "BUY_WATCH": st.info, "NEUTRAL": st.warning, "AVOID": st.error}[rating]
    box("**%s** · Setup Score: **%d / 100**\n\n%s" % (advice["label"], advice["score"], advice["advice"]))
    st.progress(advice["score"] / 100.0)
    if advice["t1_pct"] is not None:
        net = advice["t1_net_pct"]
        st.markdown(
            '<div class="hero"><div class="lbl">Biztonsággal várható hozam (T1)</div>'
            '<div class="val%s">%+.2f%%</div><div class="sub">díjak és csúszás után: %+.2f%% · '
            'R:R %.1f · cél: %s</div></div>' %
            (" neg" if net is not None and net <= 0 else "", advice["t1_pct"], net, advice["rr1"],
             advice["t1_basis"]), unsafe_allow_html=True)
        cols = st.columns(5)
        cols[0].metric("Setup Score", "%d / 100" % advice["score"])
        cols[1].metric("T1 hozam (biztonságos)", "%+.2f%%" % advice["t1_pct"], _fmt_price(advice["t1"]),
                       delta_color="off")
        cols[2].metric("T2 hozam (kiterjesztett)", "%+.2f%%" % advice["t2_pct"], _fmt_price(advice["t2"]),
                       delta_color="off")
        cols[3].metric("Stop-Loss távolság", "%+.2f%%" % advice["stop_pct"], _fmt_price(advice["stop"]),
                       delta_color="off")
        mode = "limit (visszateszt)" if advice["entry_mode"] == "pullback" else "piaci ár"
        cols[4].metric("Optimum Entry", _fmt_price(advice["entry"]), mode, delta_color="off")
        st.caption("Stop: %s. A „biztonságos” hozam konzervatív becslés (ATR + swing szintek), nem garancia; "
                   "a Time Machine fülön ellenőrizheted a történelmi találati arányát." % advice["stop_basis"])
    else:
        st.warning("A szintek nem számíthatók a coin tick méretével.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Indoklás**")
        st.markdown("\n".join("- %s %s" % (STATUS_ICONS[r["status"]], r["text"]) for r in advice["reasons"]))
    with right:
        st.markdown("**Felismert mintázatok**")
        patterns = advice["patterns"]
        if not patterns:
            st.caption("Nincs felismert gyertya- vagy chart-mintázat.")
        for pattern in patterns:
            icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[pattern["type"]]
            age = "" if pattern["group"] == "chart" else (" (utolsó gyertya)" if pattern["age"] == 0
                                                          else " (az előző gyertya)")
            status = {"confirmed": " – megerősítve", "forming": " – kialakulóban"}.get(pattern.get("status"), "")
            st.markdown("- %s **%s**%s%s  \n<small>%s</small>" %
                        (icon, pattern["name"], age, status, pattern["detail"]), unsafe_allow_html=True)
        trend = {"uptrend": "📈 Emelkedő trend (HH + HL)", "downtrend": "📉 Csökkenő trend (LH + LL)",
                 "range": "↔️ Oldalazó / vegyes struktúra"}[advice["structure"]["trend"]]
        st.markdown("**Piaci struktúra:** %s" % trend)
        ind = advice["indicators"]
        st.caption("RSI %.0f · MACD hiszt. %+.4g · ATR %.3f%% · RVOL %s · VWAP: %s" %
                   (ind["rsi"] or 0, ind["macd_hist"] or 0, ind["atr_pct"],
                    "%.2f" % ind["rvol"] if ind["rvol"] is not None else "—",
                    "napi" if advice["vwap_mode"] == "daily" else "gördülő 24h"))
    with st.expander("Pontszám részletei"):
        st.dataframe([{"Csoport": p["group"], "Szempont": p["name"],
                       "Pont": "%d / %d" % (p["earned"], p["max"]), "Megjegyzés": p["text"]}
                      for p in advice["breakdown"]], hide_index=True)
    if advice["entry"] is not None and st.button("⚡ Javasolt terv betöltése a Kalkulátorba és a Time Machine-be",
                                                 type="primary", key="load-advisor-plan"):
        try:
            _load_advisor_plan(config, selected, advice, filters)
            st.session_state["plan_notice"] = (
                "⚡ A Tanácsadó terve betöltve (Entry %s, Stop %s, T1 %s, T2 %s). "
                "Nézd meg a Time Machine és a Paper Trading fület." %
                (_fmt_price(advice["entry"]), _fmt_price(advice["stop"]),
                 _fmt_price(advice["t1"]), _fmt_price(advice["t2"])))
            st.rerun()
        except (TradingError, ValueError) as exc:
            st.error("A terv nem tölthető be: %s" % exc)


def _render_live_tab(config, interval, selected, candles, layers, refresh, auto):
    @st.fragment(run_every=refresh if auto else None)
    def live():
        rows, data, now = _snapshot(config, interval)
        row = next(item for item in rows if item["symbol"] == selected)
        result, signal, advice = row["result"], row["signal"], row["advice"]
        if not result or not result["bars"]:
            st.error("Nincs elérhető gyertyaadat a(z) %s párhoz: %s" %
                     (selected, row["error"] or "üres válasz"))
        else:
            plan = _active_plan(selected, signal, advice, actionable_only=True)
            columns = st.columns(4)
            columns[0].metric("Ár (USDT)", _fmt_price(row["price"]),
                              "%+.2f%% (24h)" % float(row["change"]) if row["change"] is not None else None)
            columns[1].metric("Tanács", advice["short_label"] if advice else "—",
                              "Score %d" % advice["score"] if advice else None, delta_color="off")
            opening = signal.get("opening_range")
            columns[2].metric("UTC nyitó sáv",
                              "%s – %s" % (_fmt_price(opening["low"]), _fmt_price(opening["high"]))
                              if opening else "—")
            columns[3].metric("Adatforrás", result["market"].upper())
            st.plotly_chart(_candlestick(selected, result, signal, plan, candles, layers, advice),
                            key="live-chart", config={"displaylogo": False, "scrollZoom": True})
            st.caption("Entry/Stop/Target: %s. Az élő (képződő) gyertya nem erősít meg kitörést. %s" %
                       (plan["source"] if plan else "nincs javasolt belépő – lásd a Tanácsadót",
                        signal.get("reason", "")))
            st.divider()
            _render_advisor(config, selected, advice, result["filters"])
        st.divider()
        st.subheader("Scanner")
        st.dataframe(_scanner_table(rows), hide_index=True, column_config={
            "Setup Score": st.column_config.ProgressColumn("Setup Score", min_value=0, max_value=100,
                                                            format="%d")})

    live()


def _histogram(output, plan):
    returns = output["terminal_returns_pct"]
    low, high = min(returns), max(returns)
    size = (high - low) / 60 or 1.0
    bins = dict(start=low, end=high + size, size=size)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[v for v in returns if v < 0], xbins=bins, name="Veszteség",
                               marker_color=DOWN))
    fig.add_trace(go.Histogram(x=[v for v in returns if v >= 0], xbins=bins, name="Nyereség",
                               marker_color=UP))
    entry, stop, target = plan
    for label, price, color in (("Stop", stop, DOWN), ("Target", target, "#00e676")):
        fig.add_vline(x=float((price / entry - 1) * 100), line_dash="dash", line_color=color,
                      annotation_text="%s %+.2f%%" % (label, float((price / entry - 1) * 100)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, barmode="overlay",
                      title="Szimulált záró hozam eloszlása (%)", xaxis_title="Hozam (%)",
                      yaxis_title="Futások", height=420, margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def _path_fan(output, plan):
    fig = go.Figure()
    for path in output["sample_paths"]:
        fig.add_trace(go.Scatter(y=path, mode="lines", line=dict(width=1, color="rgba(92,141,255,0.35)"),
                                 showlegend=False, hoverinfo="skip"))
    for label, price, color in (("Entry", plan[0], "#ffb300"), ("Stop", plan[1], DOWN),
                                ("Target", plan[2], "#00e676")):
        fig.add_hline(y=float(price), line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      title="Minta árfolyamutak (%d)" % len(output["sample_paths"]),
                      xaxis_title="Gyertya", yaxis_title="Ár", height=420, margin=dict(t=50, b=10))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def _odds_bar(probs):
    fig = go.Figure()
    for key, label, color in (("target", "Target előbb", "#00e676"), ("stop", "Stop előbb", DOWN),
                              ("neither", "Egyik sem", MUTED)):
        fig.add_trace(go.Bar(y=[""], x=[probs[key] * 100], orientation="h", name=label,
                             marker_color=color, text="%.1f%%" % (probs[key] * 100),
                             textposition="inside"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, barmode="stack",
                      height=130, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 100]),
                      legend=dict(orientation="h", y=-0.4))
    return fig


def _render_time_machine(config, interval, selected, signal, advice):
    st.caption("Volatilitás-igazított block bootstrap: történelmi gyertyablokkok, az aktuális ATR-hez "
               "skálázva. A valószínűségek szimulációs kimenetek, nem előrejelzések.")
    plan = _active_plan(selected, signal, advice)
    levels = None
    if plan:
        targets = {"T1 (biztonságos cél)": plan["target"]}
        if plan.get("t2"):
            targets["T2 (kiterjesztett cél)"] = plan["t2"]
        choice = st.radio("Szimulált célár", list(targets), horizontal=True,
                          key="sim_target_choice") if len(targets) > 1 else next(iter(targets))
        levels = (plan["entry"], plan["stop"], targets[choice])
    source = plan["source"] if plan else None
    with st.expander("Saját szintek megadása (opcionális, felülírja az aktív tervet)"):
        cols = st.columns(3)
        custom = [cols[0].text_input("Entry", key="sim_entry_" + selected),
                  cols[1].text_input("Stop", key="sim_stop_" + selected),
                  cols[2].text_input("Target", key="sim_target_" + selected)]
    with st.expander("Haladó beállítások"):
        cols = st.columns(2)
        horizon_hours = cols[0].number_input(
            "Tartási idő (óra)", 1, 72, int(config.get("simulation_horizon_hours", 12)))
        block_minutes = cols[1].number_input(
            "Blokkhossz (perc)", 5, 360, int(config.get("simulation_block_minutes", 60)), step=5)
    if any(item.strip() for item in custom):
        try:
            levels = tuple(dec(item, name) for item, name in zip(custom, ("entry", "stop", "target")))
            if not levels[1] < levels[0] < levels[2]:
                raise TradingError("A szintekre Stop < Entry < Target kell teljesüljön")
            source = "saját szintek"
        except TradingError as exc:
            st.error(str(exc))
            levels = None
    if levels is None:
        st.info("Nincs aktív terv. Töltsd be a Tanácsadó javaslatát az 1. fülön, számolj tervet a 3. fülön, "
                "vagy adj meg saját szinteket.")
    else:
        cols = st.columns(5)
        cols[0].metric("Entry", _fmt_price(levels[0]))
        cols[1].metric("Stop-Loss", "%s (%+.2f%%)" % (_fmt_price(levels[1]), float((levels[1] / levels[0] - 1) * 100)))
        cols[2].metric("Target", "%s (%+.2f%%)" % (_fmt_price(levels[2]), float((levels[2] / levels[0] - 1) * 100)))
        cols[3].metric("Forrás", source.split(" (")[0])
        cols[4].metric("Aktuális ATR", "%.3f%%" % advice["indicators"]["atr_pct"] if advice else "auto")
    runs = int(config.get("default_replays", 10_000))
    if st.button("▶ %s szimuláció futtatása" % "{:,}".format(runs).replace(",", " "),
                 type="primary", disabled=levels is None):
        try:
            now = datetime.datetime.now(UTC)
            step = INTERVAL_MS[interval]
            bars_per_day = 86_400_000 // step
            days = min(int(config.get("history_days", 60)), max(3, 20_000 // bars_per_day))
            start = int((now - datetime.timedelta(days=days)).timestamp() * 1000)
            with st.spinner("Történelmi adatok letöltése és szimuláció..."):
                history = _client().get_klines(selected, interval, limit=days * bars_per_day,
                                               start_time=start, end_time=int(now.timestamp() * 1000))
                output = simulate_trade(
                    history["bars"], levels[0], levels[1], levels[2], interval, runs,
                    int(config.get("simulation_seed", 19)), now,
                    max_steps=int(horizon_hours * 3_600_000 // step),
                    current_atr=advice["indicators"]["atr"] if advice else None,
                    block_size=max(1, int(block_minutes * 60_000 // step)))
            st.session_state["simulation"] = {"output": output, "plan": levels, "symbol": selected,
                                              "interval": interval, "market": history["market"],
                                              "days": days}
        except Exception as exc:
            st.error("A szimuláció nem sikerült: %s" % exc)
    saved = st.session_state.get("simulation")
    if not saved or saved["symbol"] != selected:
        return
    output, sim_plan = saved["output"], saved["plan"]
    for warning in output.get("warnings", []):
        st.warning(warning)
    hist = output["historical"]
    if hist["similar"]:
        st.markdown(
            '<div class="hero"><div class="lbl">Történelmi találati arány (hasonló volatilitású időszakok)</div>'
            '<div class="val">%.1f%%</div><div class="sub">Az elmúlt %d nap %d hasonló volatilitású '
            '(ATR %.2f%%–%.2f%%) időszakából <b>%d alkalommal érte el a célárat a stop előtt</b>, '
            '%d alkalommal a stop jött előbb, %d alkalommal egyik sem.<br>'
            'A %d gyertyás lépésközű ablakok átfedik egymást, ezért a tényleges független minta kisebb.</div></div>' %
            (hist["target_rate"] * 100, saved["days"], hist["similar"], hist["vol_range_pct"][0],
             hist["vol_range_pct"][1], hist["target_first"], hist["stop_first"], hist["neither"],
             hist["step_bars"]), unsafe_allow_html=True)
    else:
        st.warning("Nem volt hasonló volatilitású történelmi időszak a teljes mintában (%d ablak)." %
                   hist["windows"])
    probs = output["probabilities"]
    cols = st.columns(4)
    cols[0].metric("🎯 Target előbb", "%.1f%%" % (probs["target"] * 100))
    cols[1].metric("🛑 Stop előbb", "%.1f%%" % (probs["stop"] * 100))
    cols[2].metric("⏱ Egyik sem", "%.1f%%" % (probs["neither"] * 100))
    cols[3].metric("Volatilitás-skála", "×%.2f" % output["vol_scale"]["median"],
                   "medián, ATR %.3f%%" % output["current_atr_pct"], delta_color="off")
    st.plotly_chart(_odds_bar(probs), key="odds-bar", config={"displayModeBar": False})
    left, right = st.columns(2)
    left.plotly_chart(_histogram(output, sim_plan), key="sim-hist")
    right.plotly_chart(_path_fan(output, sim_plan), key="sim-paths")
    st.caption("%d futás (Monte Carlo) · seed %d · %d gyertyás blokkok · tartási idő %d gyertya · "
               "%d történelmi nap · %s · %s gyertyák · a múltbeli trend kiszűrve (drift-mentes)" %
               (output["runs"], output["seed"], output["block_size"], output["horizon_bars"],
                output["historical_days"], saved["market"].upper(), saved["interval"]))


def _last_prices(interval):
    data = _load_all(interval)
    return {symbol: result["bars"][-1]["close"]
            for symbol, result in data["results"].items() if result["bars"]}


def _render_open_positions(config, interval, refresh, auto):
    fee = config.get("risk", {}).get("fee_rate", 0)

    @st.fragment(run_every=refresh if auto else None)
    def positions():
        open_trades = [t for t in list_trades(config=config) if t["status"] == "open"]
        if not open_trades:
            st.info("Nincs nyitott paper pozíció.")
            return
        prices = _last_prices(interval)
        table, total = [], Decimal(0)
        for trade in open_trades:
            price = prices.get(trade["symbol"])
            pnl = unrealized_pnl(trade, price, fee) if price is not None else None
            cost = dec(trade["quantity"]) * dec(trade["entry_price"])
            total += pnl or 0
            table.append({"ID": trade["id"][:8], "Coin": trade["symbol"], "Mennyiség": trade["quantity"],
                          "Entry": trade["entry_price"], "Stop": trade["stop"], "Target": trade["target"],
                          "Aktuális ár": _fmt_price(price) if price is not None else "—",
                          "Nem realizált P&L ($)": "%+.2f" % pnl if pnl is not None else "—",
                          "P&L (%)": "%+.2f%%" % float(pnl / cost * 100) if pnl is not None and cost else "—"})
        st.metric("Nyitott pozíciók nem realizált P&L", _usd(total, signed=True))
        st.dataframe(table, hide_index=True)

    positions()


def _render_paper_tab(config, interval, selected, result, refresh, auto):
    cfg_risk = config.get("risk", {})
    last_price = _plain(result["bars"][-1]["close"]) if result and result["bars"] else ""
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Tört pozícióméret kalkulátor")
        keys = {name: "calc_%s_%s" % (name, selected) for name in ("entry", "stop", "risk", "r", "target")}
        st.session_state.setdefault(keys["entry"], last_price)
        st.session_state.setdefault(keys["risk"], str(cfg_risk.get("risk_per_trade_usd", 50)))
        st.session_state.setdefault(keys["r"], str(DEFAULT_TARGET_R))
        st.button("Aktuális ár betöltése", disabled=not last_price,
                  on_click=lambda: st.session_state.update({keys["entry"]: last_price}))
        with st.form("position-calculator"):
            entry_text = st.text_input("Entry ár (USDT)", key=keys["entry"])
            stop_text = st.text_input("Stop ár (USDT)", key=keys["stop"])
            risk_text = st.text_input("Kockáztatott összeg (USD)", key=keys["risk"])
            r_text = st.text_input("Target (R-többszörös)", key=keys["r"])
            target_text = st.text_input("Vagy konkrét Target ár (USDT, felülírja az R-t)",
                                        key=keys["target"])
            calculate = st.form_submit_button("Számolás", type="primary")
        if calculate:
            try:
                filters = result["filters"] if result else _client().get_symbol_filters(selected)
                entry, stop = dec(entry_text, "entry"), dec(stop_text, "stop")
                if not entry > stop > 0:
                    raise TradingError("Az Entry legyen nagyobb a pozitív Stop árnál")
                if target_text.strip():
                    plan = build_plan(entry, stop, DEFAULT_TARGET_R, filters, target=dec(target_text, "target"))
                else:
                    plan = build_plan(entry, stop, dec(r_text, "target R"), filters)
                size = size_position(plan["entry"], plan["stop"], dec(risk_text, "risk"), filters,
                                     cfg_risk.get("max_position_usd"), cfg_risk.get("fee_rate", 0),
                                     cfg_risk.get("slippage_rate", 0))
                st.session_state["last_plan"] = {
                    "symbol": selected, "entry": str(plan["entry"]), "stop": str(plan["stop"]),
                    "target": str(plan["target"]), "quantity": str(size["quantity"]),
                    "notional": str(size["notional_usd"]), "loss": str(size["estimated_loss_usd"]),
                    "step": str(size["step_size"]), "source": "kalkulátor terv"}
            except (TradingError, ValueError) as exc:
                st.session_state.pop("last_plan", None)
                st.error(str(exc))
        plan = st.session_state.get("last_plan")
        if plan and plan["symbol"] == selected:
            st.caption("Aktív terv forrása: %s" % plan.get("source", "kalkulátor terv"))
            cols = st.columns(2)
            cols[0].metric("Mennyiség", "%s %s" % (_plain(plan["quantity"]), selected[:-4]))
            cols[1].metric("Pozícióérték", _usd(plan["notional"]))
            cols = st.columns(2)
            cols[0].metric("Becsült max. veszteség", _usd(plan["loss"]))
            cols[1].metric("Target", _fmt_price(plan["target"]))
            st.caption("Binance lépésköz (%s) szerint lefelé kerekítve; díj és csúszás becslés." % plan["step"])
            if st.button("📝 Paper belépő naplózása"):
                try:
                    trades = list_trades(config=config)
                    daily = daily_risk(trades, fee_rate=cfg_risk.get("fee_rate", 0),
                                       slippage_rate=cfg_risk.get("slippage_rate", 0))
                    check = validate_risk({"entry": plan["entry"], "stop": plan["stop"]}, plan["quantity"],
                                          cfg_risk, current_daily_loss=daily["loss_usd"],
                                          trades_today=daily["trades_today"])
                    if not check["passed"]:
                        st.error("Blokkolva: " + "; ".join(check["problems"]))
                    else:
                        record_entry(selected, "long", plan["quantity"], plan["entry"], plan["stop"],
                                     plan["target"], config=config)
                        st.success("Paper belépő elmentve.")
                except TradingError as exc:
                    st.error(str(exc))
    with right:
        st.subheader("Nyitott pozíciók")
        _render_open_positions(config, interval, refresh, auto)
        _render_closing_and_history(config, interval)


def _render_closing_and_history(config, interval):
    fee = config.get("risk", {}).get("fee_rate", 0)
    trades = list_trades(config=config)
    open_trades = [t for t in trades if t["status"] == "open"]
    closed = [t for t in trades if t["status"] == "closed"]
    if open_trades:
        with st.form("close-paper-trade"):
            trade_id = st.selectbox(
                "Zárandó pozíció", [t["id"] for t in open_trades],
                format_func=lambda tid: next("%s · %s · %s" % (t["symbol"], t["quantity"], t["id"][:8])
                                             for t in open_trades if t["id"] == tid))
            use_market = st.checkbox("Zárás az aktuális piaci áron", value=True)
            manual = st.text_input("Vagy kézi kilépő ár (USDT)")
            if st.form_submit_button("Pozíció lezárása"):
                try:
                    symbol = next(t["symbol"] for t in open_trades if t["id"] == trade_id)
                    price = _last_prices(interval).get(symbol) if use_market else manual
                    if price in (None, ""):
                        raise TradingError("Nincs elérhető kilépő ár")
                    done = close_trade(trade_id, price, fee, config=config)
                    st.success("Lezárva. Realizált P&L: %s" % _usd(done["realized_pnl_usd"], signed=True))
                except TradingError as exc:
                    st.error(str(exc))
        closed = [t for t in list_trades(config=config) if t["status"] == "closed"]
    st.subheader("Lezárt tesztkötések")
    if not closed:
        st.info("Még nincs lezárt paper kötés.")
        return
    pnls = [dec(t["realized_pnl_usd"]) for t in closed]
    cols = st.columns(3)
    cols[0].metric("Realizált P&L", _usd(sum(pnls), signed=True))
    cols[1].metric("Lezárt kötések", len(closed))
    cols[2].metric("Nyerési arány", "%.0f%%" % (100 * sum(1 for p in pnls if p > 0) / len(pnls)))
    st.dataframe([{"ID": t["id"][:8], "Coin": t["symbol"], "Mennyiség": t["quantity"],
                   "Entry": t["entry_price"], "Kilépő": t["exit_price"],
                   "P&L ($)": "%+.2f" % float(dec(t["realized_pnl_usd"])), "Zárás (UTC)": t["exit_at"][:19]}
                  for t in reversed(closed)], hide_index=True)


def _describe_rule(rule):
    if rule["type"] == "time":
        return "Napi UTC emlékeztető %s" % rule["at"]
    if rule["type"] == "pct_move":
        return "%s mozgás ≥ %s%%" % (rule.get("direction", "any"), rule["pct"])
    return "%s: %s" % (ALERT_LABELS.get(rule["type"], rule["type"]), _fmt_price(rule["price"]))


def _delete_alert(config, rule_id):
    try:
        _alerts.delete_rule(rule_id, config=config)
    except _alerts.AlertError as exc:
        st.session_state["alert_error"] = str(exc)


def _render_alerts(config, interval, selected, refresh, auto):
    st.subheader("🔔 Riasztások")
    with st.form("new-price-alert", clear_on_submit=True):
        symbol = st.selectbox("Coin", [c["symbol"] for c in config["symbols"]],
                              index=[c["symbol"] for c in config["symbols"]].index(selected))
        kind = st.selectbox("Típus", list(ALERT_LABELS), format_func=ALERT_LABELS.get)
        level = st.text_input("Ár (USDT)")
        rearm = st.checkbox("Újrafegyverzés, ha az ár visszalép a szint alá/fölé")
        if st.form_submit_button("Riasztás mentése", type="primary"):
            try:
                _alerts.add_rule({"symbol": symbol, "type": kind, "price": level, "rearm": rearm},
                                 config=config)
                st.success("Riasztás elmentve az alerts.json fájlba.")
            except _alerts.AlertError as exc:
                st.error(str(exc))
    if st.session_state.get("alert_error"):
        st.error(st.session_state.pop("alert_error"))
    try:
        rules = _alerts.load_rules(config)
    except _alerts.AlertError as exc:
        st.error("Az alerts.json nem olvasható, javítsd vagy töröld: %s" % exc)
        return
    if not rules:
        st.info("Nincs mentett riasztás.")
    for rule in rules:
        text, button = st.columns([5, 1])
        text.write("**%s** · %s" % (rule.get("symbol", "UTC"), _describe_rule(rule)))
        button.button("🗑", key="del-" + rule["id"], on_click=_delete_alert, args=(config, rule["id"]),
                      help="Riasztás törlése")

    @st.fragment(run_every=refresh if auto else None)
    def monitor():
        try:
            current = _alerts.load_rules(config)
            data = _load_all(interval)
            quotes = {symbol: {"price": res["bars"][-1]["close"],
                               "timestamp": res["bars"][-1]["open_time_utc"],
                               "reference_price": _last_24h(res["bars"])[0]["open"]}
                      for symbol, res in data["results"].items() if res["bars"]}
            outcome = _alerts.check_alerts(current, quotes=quotes, config=config)
            history = st.session_state.setdefault("alert_history", [])
            for ring in outcome["rings"]:
                history.insert(0, {"Idő (UTC)": datetime.datetime.now(UTC).strftime("%H:%M:%S"),
                                   "Coin": ring["symbol"] or "—", "Típus": ring["type"],
                                   "Ár": _fmt_price(ring["price"]) if ring["price"] is not None else "—",
                                   "Szint": str(ring["level"])})
                st.toast("🔔 %s %s elérve: %s" % (ring["symbol"], ring["type"], ring["level"]))
            st.caption("%d aktív szabály · utoljára ellenőrizve: %s" %
                       (len(current), _clock(datetime.datetime.now(UTC))[0]))
            if history:
                st.dataframe(history[:20], hide_index=True)
        except Exception as exc:
            st.error("A riasztás-ellenőrzés nem sikerült: %s" % exc)

    monitor()


def _render_news():
    with st.spinner("RSS hírek és Fear & Greed betöltése..."):
        data = _load_news()
    sentiment = data.get("fear_greed")
    st.subheader("😨 Fear & Greed index")
    if sentiment:
        value = int(sentiment["value"])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=value, title={"text": sentiment["classification"]},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT},
                   "steps": [{"range": [0, 25], "color": "#7f2a2a"}, {"range": [25, 45], "color": "#a35a2a"},
                             {"range": [45, 55], "color": "#555a64"}, {"range": [55, 75], "color": "#2a6e68"},
                             {"range": [75, 100], "color": "#1b7a4e"}]}))
        gauge.update_layout(template="plotly_dark", paper_bgcolor=BG, height=260,
                            margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, key="fng-gauge", config={"displayModeBar": False})
        st.caption("Alternative.me · %s" % sentiment["timestamp"])
        if sentiment.get("stale"):
            st.warning("A Fear & Greed adat két napnál régebbi.")
    else:
        st.warning("A Fear & Greed adat nem elérhető.")
    st.subheader("📰 Kripto hírek")
    for error in data.get("errors", []):
        st.warning(error)
    if not data.get("items"):
        st.info("Nincs a figyelt coinokra illő friss hír.")
    for item in data.get("items", [])[:20]:
        coins = ", ".join(item.get("coins", []))
        title = (item.get("title") or "Cím nélkül").replace("[", "\\[").replace("]", "\\]")
        line = "[%s](%s)" % (title, item["link"]) if item.get("link") else title
        st.markdown("%s  \n<small>%s · %s · %s</small>" %
                    (line, coins, item.get("source", "RSS"), item.get("published_at", "")),
                    unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Crypto Terminal", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="terminal-title"><span>◢</span> CRYPTO TERMINAL '
                '<small style="color:#787b86;font-weight:400;font-size:.8rem">'
                'helyi elemzés és paper trading · nincs valódi megbízás</small></div>',
                unsafe_allow_html=True)
    try:
        config = load_config()
    except Exception as exc:
        st.error("Konfigurációs hiba: %s" % exc)
        st.stop()

    symbols = [coin["symbol"] for coin in config["symbols"]]
    with st.sidebar:
        st.header("⚙️ Vezérlők")
        selected = st.selectbox("Coin", symbols, key="selected_symbol")
        interval = st.radio("Gyertya idősík", config["intervals"], horizontal=True, key="interval",
                            index=config["intervals"].index(config["default_interval"]))
        candles = st.select_slider("Megjelenített gyertyák", options=[60, 120, 240, 480], value=240)
        st.divider()
        auto = st.toggle("Automatikus frissítés", value=True, key="auto_refresh")
        default_refresh = int(config.get("refresh_seconds", 10))
        refresh = st.segmented_control(
            "Frissítési sűrűség", REFRESH_CHOICES, format_func=lambda v: "%d mp" % v,
            default=default_refresh if default_refresh in REFRESH_CHOICES else 10,
            disabled=not auto, key="refresh_seconds") or 10
        if st.button("🔄 Frissítés most", width="stretch"):
            _load_all.clear()
        st.caption("Adatforrás: Binance publikus API (Spot, GRASS: USD-M Futures). "
                   "A piaci adat gyorsítótára legfeljebb %d mp." % CACHE_TTL_SECONDS)

    _render_ticker(config, interval, refresh, auto)
    rows, _, _ = _snapshot(config, interval)
    current = next(row for row in rows if row["symbol"] == selected)
    tabs = st.tabs(["📊 Élő Chart & Scanner", "⏳ Time Machine",
                    "💼 Paper Trading & Kockázat", "📰 Hírek, Hangulat & Riasztások"])
    with tabs[0]:
        layers = st.multiselect("Chart rétegek", LAYER_CHOICES, default=DEFAULT_LAYERS, key="chart_layers")
        _render_live_tab(config, interval, selected, candles, layers, refresh, auto)
    with tabs[1]:
        _render_time_machine(config, interval, selected, current["signal"], current["advice"])
    with tabs[2]:
        _render_paper_tab(config, interval, selected, current["result"], refresh, auto)
    with tabs[3]:
        news_col, alert_col = st.columns([3, 2])
        with news_col:
            _render_news()
        with alert_col:
            _render_alerts(config, interval, selected, refresh, auto)


if __name__ == "__main__":
    main()
