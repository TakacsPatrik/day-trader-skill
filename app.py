"""Crypto terminal: live SMC + MTF scanner, chart, ML setup classifier, backtester, paper journal, alerts."""

import concurrent.futures
import datetime
import importlib.util
import os
from decimal import Decimal

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from binance_client import INTERVAL_MS, BinanceClient, ROOT, load_config
import bot
import ml_model
import mtf
import technical_analysis as ta
from crypto_engine import (TradingError, analyze_setup, build_plan, dec,
                           size_position, validate_risk)
from paper_trading import (auto_close_trades, close_trade, daily_risk, list_trades,
                           record_entry, unrealized_pnl)

CACHE_TTL_SECONDS = 5
REFRESH_CHOICES = (5, 10, 30)
DEFAULT_TARGET_R = Decimal("2.5")
UP, DOWN, ACCENT = "#26a69a", "#ef5350", "#2962ff"
BG, GRID, MUTED = "#131722", "#2a2e39", "#787b86"
UTC = datetime.timezone.utc

ALERT_LABELS = {"above": "Ár fölé emelkedik", "below": "Ár alá esik",
                "target": "Target elérése", "stop": "Stop elérése"}
LAYER_CHOICES = ["EMA 20", "EMA 50", "EMA 200", "VWAP", "Bollinger", "Támasz / Ellenállás", "SMC & Alakzatok"]
DEFAULT_LAYERS = ["EMA 20", "EMA 50", "Bollinger", "Támasz / Ellenállás", "SMC & Alakzatok"]
STATUS_ICONS = {"ok": "✅", "warn": "⚠️", "bad": "❌"}
TREND_LABELS = {"uptrend": "📈 Emelkedő (HH+HL)", "downtrend": "📉 Csökkenő (LH+LL)",
                "range": "↔️ Oldalazó"}

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
.sig-strong_buy {background: rgba(38,166,154,.22); color: #26a69a;}
.sig-buy_watch {background: rgba(41,98,255,.20); color: #5c8dff;}
.sig-no_data, .sig-error, .sig-neutral {background: rgba(255,179,0,.18); color: #ffb300;}
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
def _client(version=3):
    return BinanceClient(load_config())


@st.cache_resource
def _htf_cache(version=3):
    return mtf.HTFCache(_client(), ttl=120)


def _fetch_symbol(client, symbol, interval, kwargs, cost_rate, now):
    try:
        result = client.get_klines(symbol, interval, include_open=True, **kwargs)
    except Exception as exc:
        return None, None, str(exc)
    try:
        htf_state = _htf_cache().state(symbol, int(now.timestamp() * 1000))
    except Exception:
        htf_state = None
    try:
        advice = analyze_setup(result["bars"], symbol, interval, result["filters"],
                               now=now, cost_rate=cost_rate, htf=htf_state)
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
    results = {s: r for s, (r, _, _) in zip(symbols, outcomes) if r is not None}
    quotes = {s: {"price": r["bars"][-1]["close"],
                  "high": r["bars"][-1]["high"],
                  "low": r["bars"][-1]["low"]}
              for s, r in results.items() if r.get("bars")}
    try:
        auto_closed = auto_close_trades(quotes, fee_rate=risk.get("fee_rate", 0),
                                        slippage_rate=risk.get("slippage_rate", 0),
                                        config=config, now=now)
    except Exception:
        auto_closed = []
    return {"interval": interval, "fetched_at": now,
            "results": results,
            "advice": {s: a for s, (_, a, _) in zip(symbols, outcomes) if a is not None},
            "errors": {s: e for s, (_, _, e) in zip(symbols, outcomes) if e},
            "auto_closed": auto_closed}


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
    for closed_trade in data.get("auto_closed", []):
        seen = st.session_state.setdefault("notified_auto_closed", set())
        if closed_trade["id"] not in seen:
            seen.add(closed_trade["id"])
            reason_lbl = "🎯 TARGET" if closed_trade.get("exit_reason") == "target" else "🛑 STOP-LOSS"
            st.toast("%s elérve (%s): %s lezárva @ %s (P&L: %s)" %
                     (reason_lbl, closed_trade["symbol"], closed_trade["id"][:8],
                      _fmt_price(closed_trade["exit_price"]),
                      _usd(closed_trade["realized_pnl_usd"], signed=True)))
    now = datetime.datetime.now(UTC)
    now_ms = int(now.timestamp() * 1000)
    rows = []
    for coin in config["symbols"]:
        symbol = coin["symbol"]
        result = data["results"].get(symbol)
        row = {"symbol": symbol, "base": coin.get("base", symbol[:-4]), "result": result,
               "error": data["errors"].get(symbol), "price": None, "change": None,
               "stale": False}
        if result and result["bars"]:
            bars = result["bars"]
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
                        trend_txt = TREND_LABELS.get(advice["structure"]["trend"], "—")
                        mtf_lbl = advice["mtf"]["label"] if advice.get("mtf") else "—"
                        mtf_det = advice["mtf"]["detail"] if advice.get("mtf") else ""
                        st.markdown(
                            '<span class="sig sig-%s">%s · %d</span><span class="src">%s%s</span>'
                            '<div class="mini" title="%s">🧭 Makro (1h/4h): <b>%s</b></div>'
                            '<div class="mini" title="%s">🕯 %s</div>'
                            '<div class="mini">%s · 🎯 T1 <b>%+.2f%%</b> · SL <b>%+.2f%%</b></div>' %
                            (advice["rating"].lower(), advice["short_label"], advice["score"],
                             market, stale, mtf_det, mtf_lbl,
                             advice["pattern_summary"], advice["pattern_summary"],
                             trend_txt, advice["t1_pct"] or 0, advice["stop_pct"] or 0),
                            unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="sig sig-no_data">⚪ N/A%s</span><span class="src">%s</span>' %
                                    (stale, market), unsafe_allow_html=True)
        _status_bar(data, now, refresh, auto)
        for symbol, message in data["errors"].items():
            st.warning("%s: %s" % (symbol, message))

    ticker()


def _active_plan(selected, advice=None, actionable_only=False):
    """Plan levels in priority order: loaded/calculator plan, advisor (SMC + swing levels)."""
    saved = st.session_state.get("last_plan")
    if saved and saved.get("symbol") == selected:
        return {"entry": dec(saved["entry"]), "stop": dec(saved["stop"]), "target": dec(saved["target"]),
                "t2": dec(saved["t2"]) if saved.get("t2") else None,
                "source": saved.get("source", "kalkulátor terv")}
    if advice and advice.get("entry") is not None and (advice["actionable"] or not actionable_only):
        return {"entry": advice["entry"], "stop": advice["stop"], "target": advice["t1"],
                "t2": advice["t2"], "source": "Okos Tanácsadó (T1 / T2)"}
    return None


def _candlestick(symbol, result, plan, candles, layers=(), advice=None):
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
    if advice and "SMC & Alakzatok" in layers and advice.get("smc"):
        for sweep in advice["smc"].get("sweeps", []):
            is_bull = sweep["type"] == "bullish"
            color = "#00e676" if is_bull else "#ff5252"
            fig.add_hline(y=float(sweep["level"]), row=1, col=1, line_dash="dot", line_width=1.2,
                          line_color=color,
                          annotation_text="SMC Sweep %s" % _fmt_price(sweep["level"]),
                          annotation_position="bottom left" if is_bull else "top left",
                          annotation_font_color=color, annotation_font_size=10)
        for geo in advice["smc"].get("geometry", []):
            if geo.get("level") is not None and low_view <= float(geo["level"]) <= high_view:
                fig.add_hline(y=float(geo["level"]), row=1, col=1, line_dash="dashdot", line_width=1.1,
                              line_color="#5c8dff",
                              annotation_text="%s (%s)" % (geo["name"], _fmt_price(geo["level"])),
                              annotation_position="top left", annotation_font_color="#5c8dff",
                              annotation_font_size=10)
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
        change = row["change"]
        advice = row["advice"]
        smc_summary = "—"
        if advice and advice.get("smc"):
            sweeps = advice["smc"].get("sweeps", [])
            if sweeps:
                s = sweeps[0]
                smc_summary = ("🟢 Bikás Sweep" if s["type"] == "bullish" else "🔴 Medvés Sweep")
        mtf_label = advice["mtf"]["label"] if advice and advice.get("mtf") else "—"
        table.append({
            "Coin": row["symbol"],
            "Ár (USDT)": _fmt_price(row["price"]) if row["price"] is not None else "—",
            "24h": ("%s %+.2f%%" % ("▲" if change >= 0 else "▼", float(change))
                    if change is not None else "—"),
            "Setup Score": advice["score"] if advice else None,
            "Tanács": advice["short_label"] if advice else "—",
            "Makro Trend (1h/4h)": mtf_label,
            "Piaci struktúra": TREND_LABELS.get(advice["structure"]["trend"], "—") if advice else "—",
            "SMC / Likviditás": smc_summary,
            "Mintázat": advice["pattern_summary"] if advice else "—",
            "T1 hozam": "%+.2f%%" % advice["t1_pct"] if advice and advice["t1_pct"] is not None else "—",
            "Stop": "%+.2f%%" % advice["stop_pct"] if advice and advice["stop_pct"] is not None else "—",
            "Utolsó gyertya (UTC)": (row["result"]["bars"][-1]["open_time_utc"][11:19]
                                     if row["result"] and row["result"]["bars"] else "—"),
            "Forrás": row["result"]["market"].upper() if row["result"] else "—"})
    return table


def _load_advisor_plan(config, selected, advice, filters):
    """Copy the advisor's levels into the calculator and active plan."""
    cfg_risk = config.get("risk", {})
    scale = float(advice.get("position_scale", 1.0)) if advice.get("position_scale") else 1.0
    plan = build_plan(advice["entry"], advice["stop"], DEFAULT_TARGET_R, filters, target=advice["t1"])
    size = size_position(plan["entry"], plan["stop"], cfg_risk.get("risk_per_trade_usd", 50), filters,
                         cfg_risk.get("max_position_usd"), cfg_risk.get("fee_rate", 0),
                         cfg_risk.get("slippage_rate", 0),
                         size_multiplier=format(scale, ".4g"))
    st.session_state["last_plan"] = {
        "symbol": selected, "entry": str(plan["entry"]), "stop": str(plan["stop"]),
        "target": str(plan["target"]), "t2": str(advice["t2"]), "quantity": str(size["quantity"]),
        "notional": str(size["notional_usd"]), "loss": str(size["estimated_loss_usd"]),
        "step": str(size["step_size"]),
        "source": "Okos Tanácsadó (betöltve · %.0f%% méret)" % (scale * 100)}
    for name, value in (("entry", plan["entry"]), ("stop", plan["stop"]), ("target", plan["target"])):
        st.session_state["calc_%s_%s" % (name, selected)] = _plain(value)


def _render_advisor(config, selected, advice, filters):
    st.subheader("🧠 Okos Tanácsadó, Makro Trend (1h/4h) & SMC")
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
        st.caption("Stop: %s. A „biztonságos” hozam konzervatív becslés (ATR + swing/SMC szintek), nem garancia; "
                   "az ML Valószínűség fülön ellenőrizheted a setup gépi tanulási nyerési esélyét." %
                   advice["stop_basis"])
    else:
        st.warning("A szintek nem számíthatók a coin tick méretével.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Indoklás**")
        st.markdown("\n".join("- %s %s" % (STATUS_ICONS[r["status"]], r["text"]) for r in advice["reasons"]))
    with right:
        st.markdown("**Felismert geometriai, SMC és gyertyamintázatok**")
        patterns = advice["patterns"]
        if not patterns:
            st.caption("Nincs felismert gyertya-, chart- vagy likviditás-mintázat.")
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
        if advice.get("mtf"):
            st.markdown("**Makro Trend (1h/4h):** %s — <small>%s</small>" %
                        (advice["mtf"]["label"], advice["mtf"]["detail"]), unsafe_allow_html=True)
        ind = advice["indicators"]
        st.caption("RSI %.0f · MACD hiszt. %+.4g · ATR %.3f%% · RVOL %s · VWAP: %s" %
                   (ind["rsi"] or 0, ind["macd_hist"] or 0, ind["atr_pct"],
                    "%.2f" % ind["rvol"] if ind["rvol"] is not None else "—",
                    "napi" if advice["vwap_mode"] == "daily" else "gördülő 24h"))
    with st.expander("Pontszám részletei"):
        st.dataframe([{"Csoport": p["group"], "Szempont": p["name"],
                       "Pont": "%d / %d" % (p["earned"], p["max"]), "Megjegyzés": p["text"]}
                      for p in advice["breakdown"]], hide_index=True)
    if advice["entry"] is not None and st.button("⚡ Javasolt terv betöltése a Kalkulátorba",
                                                 type="primary", key="load-advisor-plan"):
        try:
            _load_advisor_plan(config, selected, advice, filters)
            st.session_state["plan_notice"] = (
                "⚡ A Tanácsadó terve betöltve (Entry %s, Stop %s, T1 %s, T2 %s). "
                "Nézd meg az ML Valószínűség és a Paper Trading fület." %
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
        result, advice = row["result"], row["advice"]
        if not result or not result["bars"]:
            st.error("Nincs elérhető gyertyaadat a(z) %s párhoz: %s" %
                     (selected, row["error"] or "üres válasz"))
        else:
            plan = _active_plan(selected, advice, actionable_only=True)
            columns = st.columns(5)
            columns[0].metric("Ár (USDT)", _fmt_price(row["price"]),
                              "%+.2f%% (24h)" % float(row["change"]) if row["change"] is not None else None)
            columns[1].metric("Tanács", advice["short_label"] if advice else "—",
                              "Score %d" % advice["score"] if advice else None, delta_color="off")
            mtf_info = advice.get("mtf") if advice else None
            columns[2].metric("Makro Trend (1h/4h)",
                              mtf_info["label"] if mtf_info else "—",
                              mtf_info["detail"] if mtf_info else None, delta_color="off")
            columns[3].metric("Piaci struktúra (SMC)",
                              TREND_LABELS.get(advice["structure"]["trend"], "—") if advice else "—")
            columns[4].metric("Adatforrás", result["market"].upper())
            st.plotly_chart(_candlestick(selected, result, plan, candles, layers, advice),
                            key="live-chart", config={"displaylogo": False, "scrollZoom": True})
            st.caption("Entry/Stop/Target: %s. Az indikátorok és alakzatok lezárt gyertyákból számolódnak." %
                       (plan["source"] if plan else "nincs javasolt belépő – lásd a Tanácsadót"))
            st.divider()
            _render_advisor(config, selected, advice, result["filters"])
        st.divider()
        st.subheader("Scanner")
        st.dataframe(_scanner_table(rows), hide_index=True, column_config={
            "Setup Score": st.column_config.ProgressColumn("Setup Score", min_value=0, max_value=100,
                                                            format="%d")})

    live()


@st.cache_data(ttl=600, show_spinner=False)
def _load_trained_ml(symbol, interval, days):
    config = load_config()
    client = _client()
    now = datetime.datetime.now(UTC)
    step = INTERVAL_MS[interval]
    bars_per_day = 86_400_000 // step
    days = min(int(days), max(3, 20_000 // bars_per_day))
    end_ms = int(now.timestamp() * 1000)
    start = int((now - datetime.timedelta(days=days)).timestamp() * 1000)
    history = client.get_klines(symbol, interval, limit=days * bars_per_day,
                                start_time=start, end_time=end_ms)
    try:
        htf_bars = mtf.fetch_htf(client, symbol, start, end_ms)
    except Exception:
        htf_bars = None
    bundle = ml_model.train_model(history["bars"], interval, now=now, config=config, htf_bars=htf_bars)
    bundle["symbol"] = symbol
    bundle["market"] = history["market"]
    bundle["days"] = days
    return bundle


def _importance_chart(factors):
    top = list(reversed(factors[:10]))
    fig = go.Figure(go.Bar(
        x=[f["importance"] * 100 for f in top],
        y=[f["label"] for f in top],
        orientation="h",
        marker_color=[UP if f["effect"] >= 0 else DOWN for f in top],
        text=["%.1f%% (%+.1f pp)" % (f["importance"] * 100, f["effect"] * 100) for f in top],
        textposition="auto"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Legfontosabb döntési faktorok (Feature Importance & lokális hatás)",
        xaxis_title="Fontosság (%)", height=380, margin=dict(l=10, r=20, t=45, b=10))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def _calibration_chart(calibration, base_rate):
    fig = go.Figure()
    if calibration:
        fig.add_trace(go.Scatter(
            x=[b["mean_pred"] * 100 for b in calibration],
            y=[b["actual"] * 100 for b in calibration],
            mode="lines+markers+text",
            text=["n=%d" % b["count"] for b in calibration],
            textposition="top center",
            marker=dict(size=9, color=ACCENT),
            line=dict(color=ACCENT, width=2),
            name="Valós találati arány"))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        line=dict(color=MUTED, dash="dash"), name="Tökéletes kalibráció"))
    fig.add_hline(y=base_rate * 100, line_dash="dot", line_color="#ffb300",
                  annotation_text="Alapráta %.1f%%" % (base_rate * 100))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
        title="Out-of-Sample Kalibráció (Holdout teszt)",
        xaxis_title="Modell által becsült valószínűség (%)",
        yaxis_title="Tényleges győzelmi arány (%)",
        height=380, margin=dict(l=10, r=20, t=45, b=10),
        legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(gridcolor=GRID, range=[0, 100])
    fig.update_yaxes(gridcolor=GRID, range=[0, 100])
    return fig


def _render_backtest_section(config, interval, selected):
    st.divider()
    st.subheader("⏪ Walk-Forward Backtester & Auto-Paper Robot (`bot.py`)")
    st.caption(
        "Teszteld a stratégiát múltbeli gyertyákon (jövőbelátás nélkül, walk-forward újratanítással és "
        "1h/4h EMA50 makro szűrővel), vagy futtasd önálló háttérfolyamatként: "
        "`python bot.py backtest %s --days 90` · `python bot.py paper`" % selected)
    cfg_ml = ml_model.settings(config)
    dyn_pct = cfg_ml["auto_threshold"] * 100.0
    be_pct = cfg_ml["breakeven"] * 100.0
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    bt_days = c1.number_input("Történelem (nap)", min_value=15, max_value=180, value=90, step=15,
                              key="bt_days_" + selected)
    bt_train = c2.number_input("Kezdeti tanítás (nap)", min_value=7, max_value=90, value=30, step=5,
                               key="bt_train_" + selected)
    bt_thresh = c3.slider(
        "Dinamikus +EV Küszöb (%)", min_value=20, max_value=85,
        value=int(round(dyn_pct)), step=1,
        help="Alapértelmezett: 1.25 × Break-even Win Rate (%.1f%% × 1.25 = %.1f%%)" % (be_pct, dyn_pct),
        key="bt_thresh_" + selected)
    bt_mtf = c4.checkbox("1h/4h EMA50 szűrő kötelező", value=True, key="bt_mtf_" + selected)

    b_col1, b_col2 = st.columns(2)
    if b_col1.button("▶️ Walk-Forward Backtest indítása (%s · %s)" % (selected, interval), type="primary"):
        try:
            status = st.empty()
            with st.spinner("Múltbeli adatok letöltése és gyertyáról gyertyára történő szimuláció..."):
                rep = bot.run_backtest(
                    selected, interval=interval, days=float(bt_days), train_days=float(bt_train),
                    retrain_days=2.0, threshold=bt_thresh / 100.0, require_mtf=bt_mtf,
                    config=config, client=_client(), progress=lambda msg: status.caption(msg))
                saved_path = bot.save_report(rep, csv_too=True)
            status.empty()
            st.session_state["last_backtest_" + selected] = (rep, saved_path)
        except Exception as exc:
            st.error("Backtest hiba: %s" % exc)

    if b_col2.button("🤖 Auto-Paper Robot: 1 ciklus futtatása most (összes coin)", type="secondary"):
        try:
            with st.spinner("Auto-Paper ciklus futtatása (ML > %.0f%% + 1h/4h Makro Trend)..." % bt_thresh):
                pbot = bot.PaperBot(config=config, client=_client(), interval=interval,
                                    threshold=bt_thresh / 100.0)
                opened = pbot.cycle()
            if opened:
                st.success("🤖 Auto-Paper nyitott %d új pozíciót: %s" %
                           (len(opened), ", ".join("%s @ %s" % (t["symbol"], t["entry_price"]) for t in opened)))
            else:
                st.info("A ciklus lefutott: jelenleg egyik coin sem haladta meg az ML > %.0f%% + 1h/4h Bikás "
                        "feltételt (vagy már van nyitott pozíció)." % bt_thresh)
        except Exception as exc:
            st.error("Auto-Paper hiba: %s" % exc)

    saved_bt = st.session_state.get("last_backtest_" + selected)
    if saved_bt:
        rep, saved_path = saved_bt
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Profit", "%+.2f USD" % rep["total_profit_usd"],
                  "%+.2f%%" % rep["return_pct"])
        m2.metric("Win Rate", "%.1f%%" % (rep["win_rate"] * 100) if rep["win_rate"] is not None else "—",
                  "Break-even: %.1f%% · +EV küszöb: %.1f%%" % (
                      rep.get("breakeven_win_rate", 0) * 100,
                      rep.get("required_ev_threshold", rep["strategy"]["threshold"]) * 100), delta_color="off")
        m3.metric("Max Drawdown", "%.2f USD" % rep["max_drawdown_usd"],
                  "%.2f%%" % rep["max_drawdown_pct"], delta_color="inverse")
        m4.metric("Profit Factor", "%.2f" % rep["profit_factor"] if rep["profit_factor"] is not None else "—",
                  "Átlag R: %+.2f" % rep["avg_r"] if rep["avg_r"] is not None else None, delta_color="off")
        m5.metric("Buy & Hold (ugyanekkor)", "%+.2f%%" % rep["buy_and_hold_pct"])
        st.caption("Mentve: `%s.json`" % saved_path)
        with st.expander("Részletes Backtest riport és kötéslista", expanded=False):
            st.code(bot.format_report(rep), language="text")
            if rep["trade_list"]:
                st.dataframe(rep["trade_list"], hide_index=True)


def _render_ml_tab(config, interval, selected, result):
    st.subheader("🤖 ML Valószínűségi Osztályozó (RandomForestClassifier + 1h/4h MTF)")
    cfg_ml = ml_model.settings(config)
    st.caption(
        "Helyben tanuló gépi tanulási modell (`scikit-learn RandomForestClassifier`), amely a múltbeli "
        "gyertyák geometriai alakzataiból, SMC likviditás-lesöpréseiből, relatív forgalmából (RVOL), "
        "ATR volatilitásából, a mozgóátlagoktól való távolságból és a **magasabb idősíkú (1h/4h EMA50) "
        "makro trendből** becsüli meg annak valószínűségét, hogy az aktuális setup eléri az "
        "**%.1f×ATR célárat** az **%.1f×ATR stop** előtt (%d gyertyán belül)." %
        (cfg_ml["target_atr"], cfg_ml["stop_atr"], cfg_ml["horizon_bars"]))

    if not result or not result.get("bars"):
        st.warning("Nincs elérhető piaci adat a(z) %s párhoz." % selected)
        return

    default_days = int(config.get("history_days", 60))
    col_btn, col_days = st.columns([3, 1])
    days = col_days.number_input("Tanító időszak (nap)", min_value=5, max_value=180, value=default_days, step=5,
                                 key="ml_days_" + selected)
    retrain = col_btn.button("🔄 Modell újratanítása a legfrissebb történelmi adatokon", type="secondary")
    if retrain:
        _load_trained_ml.clear()

    try:
        with st.spinner("Helyi RandomForest modell tanítása és kiértékelése (%s · %s)..." % (selected, interval)):
            bundle = _load_trained_ml(selected, interval, days)
            try:
                live_htf = _htf_cache().get(selected)
            except Exception:
                live_htf = None
            prediction = ml_model.predict_setup(bundle, result["bars"], htf_bars=live_htf)
    except Exception as exc:
        st.error("Az ML modell nem futtatható: %s" % exc)
        return

    for warning in bundle.get("warnings", []):
        st.warning(warning)

    win_prob = prediction["probability"] * 100.0
    raw_prob = prediction["raw_probability"] * 100.0
    base_prob = prediction["base_rate"] * 100.0
    be_prob = prediction["breakeven"] * 100.0
    req_prob = prediction["required_threshold"] * 100.0
    edge = prediction["edge_pp"]
    is_neg = win_prob < req_prob
    mtf_state = prediction.get("mtf")
    mtf_lbl = mtf_state["label"] if mtf_state else "⚪ Ismeretlen"
    mtf_det = mtf_state["detail"] if mtf_state else ""
    size_mult = prediction.get("size_multiplier", 1.0)
    if prediction.get("mtf_allow_long") is False:
        st.warning(
            "🧭 **Makro Trend szűrő aktív (%s):** %s. Mivel az 1h idősíkon az árfolyam "
            "nincs az EMA50 felett, a modell nyers becslése (**%.1f%%**) az alaprátában (**%.1f%%**) "
            "maximalizálva lett, és vételi jelzés nem adható." %
            (mtf_lbl, mtf_det, raw_prob, base_prob))
    elif size_mult == 0.5:
        st.info(
            "🧭 **Rugalmas MTF szűrő (%s):** %s. Az 1h trend bikás, de a 4h még nem, ezért a belépés "
            "**50%%-os (felezett) kockázattal és pozíciómérettel** engedélyezett." %
            (mtf_lbl, mtf_det))
    st.markdown(
        '<div class="hero"><div class="lbl">🤖 ML Győzelmi Valószínűség az aktuális setupra (Win Probability): '
        '<b>%.1f%%</b> · Makro Trend (1h/4h): <b>%s</b></div>'
        '<div class="val%s">%.1f%%</div>'
        '<div class="sub">Dinamikus Küszöb (Required +EV Threshold): <b>%.1f%%</b> '
        '(%.2f× Break-even WR: <b>%.1f%%</b>) · Történelmi alapráta: %.1f%% · Edge: <b>%+.1f pp</b> · '
        'Auto-Paper jelzés (≥%.1f%% + MTF): <b>%s</b><br>'
        'Cél: <b>%s</b> (%+.2f%%, +%.1f×ATR) · Stop: <b>%s</b> (%+.2f%%, −%.1f×ATR)</div></div>' %
        (win_prob, mtf_lbl, " neg" if is_neg else "", win_prob, req_prob,
         prediction.get("ev_multiplier", 1.25), be_prob, base_prob, edge, req_prob,
         "🟢 AKTÍV +EV VÉTEL" if prediction.get("signal") else "⏸ VÁRAKOZÁS",
         _fmt_price(prediction["target"]), prediction["target_pct"], cfg_ml["target_atr"],
         _fmt_price(prediction["stop"]), prediction["stop_pct"], cfg_ml["stop_atr"]),
        unsafe_allow_html=True)

    st.markdown("#### 🏆 A 3 legfontosabb döntési faktor (Feature Importance)")
    top3_cols = st.columns(3)
    for idx, factor in enumerate(prediction["top_factors"]):
        direction = "⬆️ növeli" if factor["effect"] >= 0 else "⬇️ csökkenti"
        top3_cols[idx].metric(
            "#%d · %s" % (idx + 1, factor["label"]),
            "%.1f%% súly" % (factor["importance"] * 100),
            "%s az esélyt (%+.1f pp) · érték: %.2f" % (direction, factor["effect"] * 100, factor["value"]),
            delta_color="normal" if factor["effect"] >= 0 else "inverse")

    metrics = bundle["metrics"]
    m_cols = st.columns(4)
    m_cols[0].metric("Tanító / Teszt minták", "%d / %d" % (bundle["n_train"], bundle["n_test"]))
    m_cols[1].metric("Holdout ROC AUC", "%.3f" % metrics["auc"] if metrics["auc"] is not None else "N/A")
    m_cols[2].metric("Holdout Brier Score", "%.4f" % metrics["brier"],
                     "alap: %.4f" % metrics["brier_baseline"], delta_color="off")
    m_cols[3].metric("Történelmi időtáv", "%.1f nap (%s)" % (bundle["span_days"], bundle["market"].upper()))

    left, right = st.columns(2)
    left.plotly_chart(_importance_chart(prediction["factors"]), key="ml-imp-chart")
    right.plotly_chart(_calibration_chart(metrics["calibration"], prediction["base_rate"]), key="ml-cal-chart")

    with st.expander("Összes bemeneti változó (Features) aktuális értéke és hatása"):
        st.dataframe([
            {"Faktor": f["label"],
             "Fontosság (%)": "%.2f%%" % (f["importance"] * 100),
             "Aktuális érték": "%.4g" % f["value"],
             "Tipikus (medián)": "%.4g" % f["typical"],
             "Lokális hatás (pp)": "%+.2f pp" % (f["effect"] * 100)}
            for f in prediction["factors"]
        ], hide_index=True)

    _render_backtest_section(config, interval, selected)


def _last_prices(interval):
    data = _load_all(interval)
    return {symbol: result["bars"][-1]["close"]
            for symbol, result in data["results"].items() if result["bars"]}


def _render_open_positions(config, interval, refresh, auto):
    cfg_risk = config.get("risk", {})
    fee = cfg_risk.get("fee_rate", 0)
    slippage = cfg_risk.get("slippage_rate", 0)

    @st.fragment(run_every=refresh if auto else None)
    def positions():
        data = _load_all(interval)
        open_trades = [t for t in list_trades(config=config) if t["status"] == "open"]
        if not open_trades:
            st.info("Nincs nyitott paper pozíció. (A Stop-Loss vagy Target elérésekor a rendszer "
                    "minden élő frissítésnél automatikusan lezárja a pozíciókat.)")
            return
        prices = {symbol: res["bars"][-1]["close"]
                  for symbol, res in data["results"].items() if res["bars"]}
        table, total = [], Decimal(0)
        for trade in open_trades:
            price = prices.get(trade["symbol"])
            pnl = unrealized_pnl(trade, price, fee, slippage) if price is not None else None
            cost = dec(trade["quantity"]) * dec(trade["entry_price"])
            total += pnl or 0
            table.append({"ID": trade["id"][:8], "Coin": trade["symbol"], "Mennyiség": trade["quantity"],
                          "Entry": trade["entry_price"], "Stop": trade["stop"], "Target": trade["target"],
                          "Aktuális ár": _fmt_price(price) if price is not None else "—",
                          "Nem realizált P&L ($)": "%+.2f" % pnl if pnl is not None else "—",
                          "P&L (%)": "%+.2f%%" % float(pnl / cost * 100) if pnl is not None and cost else "—",
                          "Megjegyzés": trade.get("note", "")})
        st.metric("Nyitott pozíciók nem realizált P&L (díj + slippage után)", _usd(total, signed=True))
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
    cfg_risk = config.get("risk", {})
    fee = cfg_risk.get("fee_rate", 0)
    slippage = cfg_risk.get("slippage_rate", 0)
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
                    done = close_trade(trade_id, price, fee, slippage, reason="manual", config=config)
                    st.success("Lezárva. Realizált P&L (díj + slippage után): %s" %
                               _usd(done["realized_pnl_usd"], signed=True))
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
    reason_map = {"target": "🎯 Target (auto)", "stop": "🛑 Stop (auto)", "manual": "✋ Kézi"}
    st.dataframe([{"ID": t["id"][:8], "Coin": t["symbol"], "Mennyiség": t["quantity"],
                   "Entry": t["entry_price"], "Kilépő": t["exit_price"],
                   "Mód": reason_map.get(t.get("exit_reason", "manual"), t.get("exit_reason", "—")),
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
                'SMC struktúra, 1h/4h Makro Trend, ML valószínűség, Backtester és Auto-Paper Trading</small></div>',
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
            _htf_cache.clear()
        st.caption("Adatforrás: Binance publikus API (Spot, GRASS: USD-M Futures). "
                   "A piaci adat gyorsítótára legfeljebb %d mp." % CACHE_TTL_SECONDS)

    _render_ticker(config, interval, refresh, auto)
    rows, _, _ = _snapshot(config, interval)
    current = next(row for row in rows if row["symbol"] == selected)
    tabs = st.tabs(["📊 Élő Chart & Scanner", "🤖 ML Valószínűség & Backtester",
                    "💼 Paper Trading & Kockázat", "📰 Hírek, Hangulat & Riasztások"])
    with tabs[0]:
        layers = st.multiselect("Chart rétegek", LAYER_CHOICES, default=DEFAULT_LAYERS, key="chart_layers")
        _render_live_tab(config, interval, selected, candles, layers, refresh, auto)
    with tabs[1]:
        _render_ml_tab(config, interval, selected, current["result"])
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
