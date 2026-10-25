"""Unauthenticated Binance Spot/Futures market data client."""

import datetime
import json
import os
import time
from decimal import Decimal, InvalidOperation

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
INTERVAL_MS = {"1m": 60_000, "5m": 300_000}
UNKNOWN_SYMBOL_CODES = {-1121}
FUTURES_MEMO_MS = 3_600_000


class BinanceError(Exception):
    """A Binance request or response could not be used safely."""


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.json")
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise BinanceError("Cannot read configuration %s: %s" % (path, exc))
    symbols = config.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise BinanceError("config.json needs a non-empty symbols list")
    normalized = []
    seen = set()
    for item in symbols:
        symbol = (item.get("symbol") if isinstance(item, dict) else item)
        symbol = str(symbol or "").upper().strip()
        if not symbol.endswith("USDT") or not symbol[:-4].isalnum():
            raise BinanceError("Invalid USDT symbol in config.json: %r" % symbol)
        if symbol in seen:
            raise BinanceError("Duplicate symbol in config.json: %s" % symbol)
        seen.add(symbol)
        normalized.append(dict(item, symbol=symbol) if isinstance(item, dict) else
                          {"symbol": symbol, "base": symbol[:-4], "name": symbol[:-4],
                           "aliases": [symbol[:-4]]})
    intervals = config.get("intervals")
    if not isinstance(intervals, list) or not intervals or any(i not in INTERVAL_MS for i in intervals):
        raise BinanceError("config.json intervals may contain only 1m and 5m")
    if config.get("default_interval") not in intervals:
        raise BinanceError("default_interval must be one of the configured intervals")
    config["symbols"] = normalized
    return config


def _decimal(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise BinanceError("Invalid %s from Binance: %r" % (label, value))
    if not number.is_finite():
        raise BinanceError("Invalid %s from Binance: %r" % (label, value))
    return number


def _error_code(response):
    try:
        return int(response.json().get("code"))
    except (AttributeError, TypeError, ValueError, requests.RequestException):
        return None


def _unknown_symbol(response):
    return response.status_code == 404 or _error_code(response) in UNKNOWN_SYMBOL_CODES


class BinanceClient:
    def __init__(self, config=None, session=None, clock=None, sleeper=None):
        self.config = config or load_config()
        self.session = session or requests.Session()
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.sleeper = sleeper or time.sleep
        self._filters = {}
        self._futures_only = {}

    def _request_json(self, url, params=None):
        settings = self.config.get("binance", {})
        retries = max(0, int(settings.get("retries", 2)))
        timeout = float(settings.get("timeout_seconds", 10))
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                if response.status_code in (429, 418) or response.status_code >= 500:
                    if attempt < retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = min(10.0, float(retry_after)) if retry_after else 0.5 * (2 ** attempt)
                        except ValueError:
                            delay = 0.5 * (2 ** attempt)
                        self.sleeper(delay)
                        continue
                if not response.ok:
                    code = _error_code(response)
                    message = "HTTP %s" % response.status_code
                    try:
                        body = response.json()
                        message = "%s (%s)" % (body.get("msg", message), code) if code is not None else body.get("msg", message)
                    except (ValueError, AttributeError, requests.RequestException):
                        pass
                    raise BinanceError("%s: %s" % (url, message))
                try:
                    return response.json()
                except ValueError as exc:
                    raise BinanceError("Binance returned invalid JSON from %s: %s" % (url, exc))
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries:
                    self.sleeper(0.5 * (2 ** attempt))
        if last_error:
            raise BinanceError("Binance network request failed for %s: %s" % (url, last_error))
        raise BinanceError("Binance request failed for %s" % url)

    def _base_url(self, market):
        settings = self.config.get("binance", {})
        key = "futures_base_url" if market == "futures" else "spot_base_url"
        return str(settings.get(key, "https://fapi.binance.com" if market == "futures"
                                else "https://api.binance.com")).rstrip("/")

    def _raw_klines(self, symbol, interval, market, params):
        path = "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"
        url = self._base_url(market) + path
        try:
            return self._request_json(url, dict(params, symbol=symbol, interval=interval))
        except BinanceError as exc:
            # The fallback is reserved for a symbol that is genuinely absent;
            # network/rate-limit/server errors should not silently switch markets.
            text = str(exc)
            if "-1121" in text or "HTTP 404" in text:
                raise _UnknownSymbol(text)
            raise

    @staticmethod
    def _normalize(symbol, interval, market, row):
        try:
            return {
                "symbol": symbol,
                "interval": interval,
                "market": market,
                "open_time": int(row[0]),
                "open_time_utc": datetime.datetime.fromtimestamp(
                    int(row[0]) / 1000, datetime.timezone.utc).isoformat(),
                "open": _decimal(row[1], "open"),
                "high": _decimal(row[2], "high"),
                "low": _decimal(row[3], "low"),
                "close": _decimal(row[4], "close"),
                "volume": _decimal(row[5], "volume"),
                "close_time": int(row[6]),
            }
        except (IndexError, TypeError, ValueError) as exc:
            raise BinanceError("Malformed kline for %s: %s" % (symbol, exc))

    def get_klines(self, symbol, interval="5m", limit=500, start_time=None,
                   end_time=None, include_open=False, market="spot"):
        symbol = str(symbol).upper()
        if interval not in INTERVAL_MS:
            raise BinanceError("Unsupported interval %r; use 1m or 5m" % interval)
        configured = {row["symbol"] for row in self.config["symbols"]}
        if symbol not in configured:
            raise BinanceError("Symbol %s is not in config.json" % symbol)
        if market not in ("spot", "futures"):
            raise BinanceError("Market must be spot or futures")
        page_limit = min(1000, max(1, int(self.config.get("binance", {}).get("page_limit", 1000))))
        limit = max(1, int(limit))
        if start_time is None:
            limit = min(limit, page_limit)
        selected_market = market
        # Skip the doomed Spot request for symbols recently found on Futures only.
        if (market == "spot" and self._futures_only.get(symbol, 0) > int(self.clock())
                and self.config.get("binance", {}).get("fallback_to_futures", True)):
            selected_market = "futures"
        params = {"limit": min(limit, page_limit)}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        try:
            first = self._raw_klines(symbol, interval, selected_market, params)
        except _UnknownSymbol as exc:
            if selected_market != "spot" or not self.config.get("binance", {}).get("fallback_to_futures", True):
                raise BinanceError(str(exc))
            selected_market = "futures"
            try:
                first = self._raw_klines(symbol, interval, selected_market, params)
            except _UnknownSymbol as futures_exc:
                raise BinanceError("Symbol %s is unavailable on Spot and USD-M Futures: %s" %
                                   (symbol, futures_exc))
            self._futures_only[symbol] = int(self.clock()) + FUTURES_MEMO_MS

        raw_rows = list(first or [])
        target_count = min(limit, page_limit) if start_time is None else limit
        cursor = int(raw_rows[-1][0]) + INTERVAL_MS[interval] if raw_rows else None
        while start_time is not None and len(raw_rows) < target_count and cursor is not None:
            if end_time is not None and cursor > int(end_time):
                break
            page_params = {"limit": min(page_limit, target_count - len(raw_rows)), "startTime": cursor}
            if end_time is not None:
                page_params["endTime"] = int(end_time)
            page = self._raw_klines(symbol, interval, selected_market, page_params)
            if not page:
                break
            raw_rows.extend(page)
            next_cursor = int(page[-1][0]) + INTERVAL_MS[interval]
            if next_cursor <= cursor:
                raise BinanceError("Binance pagination did not advance for %s" % symbol)
            cursor = next_cursor

        latest_open = {}
        now = int(self.clock())
        for row in raw_rows:
            bar = self._normalize(symbol, interval, selected_market, row)
            if include_open or bar["close_time"] < now:
                latest_open[bar["open_time"]] = bar
        bars = [latest_open[key] for key in sorted(latest_open)]
        if start_time is not None:
            bars = bars[:target_count]
        return {"symbol": symbol, "interval": interval, "market": selected_market,
                "bars": bars, "filters": self.get_symbol_filters(symbol, selected_market)}

    def get_symbol_filters(self, symbol, market="spot"):
        symbol = str(symbol).upper()
        key = (market, symbol)
        if key in self._filters:
            return self._filters[key]
        endpoint = "/fapi/v1/exchangeInfo" if market == "futures" else "/api/v3/exchangeInfo"
        doc = self._request_json(self._base_url(market) + endpoint, {"symbol": symbol})
        records = doc.get("symbols", []) if isinstance(doc, dict) else []
        record = next((item for item in records if item.get("symbol") == symbol), None)
        if record is None:
            raise BinanceError("No %s exchange filters for %s" % (market, symbol))
        filters = {item.get("filterType"): item for item in record.get("filters", [])}
        lot = filters.get("LOT_SIZE") or filters.get("MARKET_LOT_SIZE") or {}
        price = filters.get("PRICE_FILTER") or {}
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        result = {
            "market": market,
            "step_size": _decimal(lot.get("stepSize", "0.00000001"), "stepSize"),
            "min_qty": _decimal(lot.get("minQty", "0"), "minQty"),
            "tick_size": _decimal(price.get("tickSize", "0.00000001"), "tickSize"),
            "min_notional": _decimal(notional.get("minNotional", "0"), "minNotional"),
        }
        self._filters[key] = result
        return result


class _UnknownSymbol(BinanceError):
    pass
