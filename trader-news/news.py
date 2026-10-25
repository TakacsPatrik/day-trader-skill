"""Crypto RSS headlines and the global Alternative.me Fear & Greed Index."""

import datetime
import email.utils
import re
import xml.etree.ElementTree as ET

import requests

from binance_client import load_config


class NewsError(Exception):
    pass


def _text(element, names):
    for name in names:
        value = element.findtext(name)
        if value:
            return value.strip()
        # RSS/Atom namespaces vary by publisher.
        for child in element.iter():
            if child.tag.rsplit("}", 1)[-1] == name and child.text:
                return child.text.strip()
    return ""


def _published(value):
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed.astimezone(datetime.timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return value


def _matches(text, item):
    aliases = [item.get("base", ""), item.get("name", "")]
    aliases.extend(item.get("aliases") or [])
    for alias in filter(None, aliases):
        words = str(alias).strip()
        if not words:
            continue
        pattern = r"(?<![A-Za-z0-9])" + re.escape(words) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            # Very short ticker aliases are matched case-sensitively to avoid
            # turning ordinary English words into crypto headlines.
            if len(words) <= 3 and words.casefold() not in ("btc", "eth", "bnb", "xrp", "sol"):
                if re.search(r"(?<![A-Za-z0-9])" + re.escape(words) + r"(?![A-Za-z0-9])", text):
                    return True
                continue
            return True
    return False


def parse_feed(xml_data, feed_name, symbols):
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        raise NewsError("%s returned invalid RSS/XML: %s" % (feed_name, exc))
    items = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        title = _text(element, ("title",))
        description = _text(element, ("description", "summary", "content"))
        link = _text(element, ("link",))
        if not link:
            for child in element:
                if child.tag.rsplit("}", 1)[-1] == "link":
                    link = child.attrib.get("href", "")
                    break
        guid = _text(element, ("guid", "id")) or link or title
        matched = [coin for coin in symbols
                   if _matches("%s %s" % (title, description), coin)]
        if not matched:
            continue
        items.append({"id": "%s:%s" % (feed_name.casefold(), guid),
                      "title": title, "description": description, "link": link,
                      "source": feed_name,
                      "published_at": _published(_text(element, ("pubDate", "published", "updated"))),
                      "coins": [coin["symbol"] for coin in matched]})
    return items


def fetch_fear_greed(url=None, session=None, timeout=10):
    if not url:
        config = load_config()
        url = config.get("news", {}).get("fear_greed_url",
                                        "https://api.alternative.me/fng/?limit=1")
    session = session or requests
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        doc = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise NewsError("Fear & Greed API request failed: %s" % exc)
    try:
        record = doc["data"][0]
        value = int(record["value"])
        if not 0 <= value <= 100:
            raise ValueError("value outside 0..100")
        parsed_timestamp = datetime.datetime.fromtimestamp(int(record["timestamp"]),
                                                           datetime.timezone.utc)
        timestamp = parsed_timestamp.isoformat()
        stale = datetime.datetime.now(datetime.timezone.utc) - parsed_timestamp > datetime.timedelta(days=2)
        return {"value": value, "classification": str(record["value_classification"]),
                "timestamp": timestamp, "source": "Alternative.me", "stale": stale}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise NewsError("Fear & Greed API returned invalid data: %s" % exc)


def fetch_crypto_news(config=None, session=None, timeout=10):
    config = config or load_config()
    session = session or requests
    coins = config["symbols"]
    items, errors = [], []
    for feed in config.get("news", {}).get("feeds", []):
        name, url = feed.get("name", "RSS"), feed.get("url")
        if not url:
            errors.append("%s: no URL configured" % name)
            continue
        try:
            response = session.get(url, timeout=timeout,
                                   headers={"User-Agent": "crypto-day-trader/1.0"})
            response.raise_for_status()
            items.extend(parse_feed(response.content, name, coins))
        except (requests.RequestException, NewsError) as exc:
            errors.append("%s: %s" % (name, exc))
    deduped = {}
    for item in items:
        if item["id"] in deduped:
            existing = deduped[item["id"]]
            existing["coins"] = sorted(set(existing["coins"]) | set(item["coins"]))
        else:
            deduped[item["id"]] = item
    items = sorted(deduped.values(),
                   key=lambda item: item.get("published_at") or "", reverse=True)
    try:
        sentiment = fetch_fear_greed(
            config.get("news", {}).get("fear_greed_url"), session=session, timeout=timeout)
    except NewsError as exc:
        sentiment = None
        errors.append(str(exc))
    return {"items": items, "fear_greed": sentiment, "errors": errors,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    result = fetch_crypto_news()
    for item in result["items"]:
        print("%s | %s | %s | %s" % (item.get("published_at") or "time unknown",
                                     ",".join(item["coins"]), item["source"], item["title"]))
        if item.get("link"):
            print("  " + item["link"])
    if result["fear_greed"]:
        fng = result["fear_greed"]
        print("Fear & Greed: %d (%s) at %s" %
              (fng["value"], fng["classification"], fng["timestamp"]))
    for error in result["errors"]:
        print("Data warning: " + error)
    return 1 if result["errors"] and not result["items"] and not result["fear_greed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
