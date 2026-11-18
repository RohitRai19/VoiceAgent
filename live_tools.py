"""Live data tools the chat model can call: weather (Open-Meteo), share prices (Yahoo Finance via
yfinance) and web search (Tavily). Each returns JSON text, or text starting with "Error:" so the
model can tell the user what went wrong."""
from __future__ import annotations

import json
import logging
import re

import requests
from langchain_core.tools import tool

import config

log = logging.getLogger("live_tools")
HTTP_TIMEOUT = 15

# WMO weather codes used by Open-Meteo.
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _err(message: str) -> str:
    return f"Error: {message}"


@tool
def get_weather(location: str, days: int = 1) -> str:
    """Live weather for a place: current temperature, feels-like, humidity, wind, rain and conditions,
    plus a daily forecast (days = 1 to 3). Give the place name in English, e.g. "Delhi", "Mumbai",
    "Gurugram", "London". Use this for any question about weather, temperature, rain or forecast."""
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": location, "count": 1, "language": "en", "format": "json"},
                           timeout=HTTP_TIMEOUT).json()
        if not geo.get("results"):
            return _err(f"could not find a place called '{location}'.")
        place = geo["results"][0]
        days = max(1, min(int(days or 1), 3))
        fc = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "forecast_days": days,
        }, timeout=HTTP_TIMEOUT).json()
        cur = fc.get("current", {})
        daily = fc.get("daily", {})
        return json.dumps({
            "place": ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p),
            "local_time": cur.get("time"),
            "current": {
                "temperature_c": cur.get("temperature_2m"), "feels_like_c": cur.get("apparent_temperature"),
                "humidity_percent": cur.get("relative_humidity_2m"), "wind_kmh": cur.get("wind_speed_10m"),
                "precipitation_mm": cur.get("precipitation"),
                "conditions": _WEATHER_CODES.get(cur.get("weather_code"), "unknown"),
            },
            "forecast": [
                {"date": d, "min_c": lo, "max_c": hi, "rain_chance_percent": p,
                 "conditions": _WEATHER_CODES.get(code, "unknown")}
                for d, lo, hi, p, code in zip(daily.get("time", []), daily.get("temperature_2m_min", []),
                                              daily.get("temperature_2m_max", []),
                                              daily.get("precipitation_probability_max", []),
                                              daily.get("weather_code", []))
            ],
            "source": "Open-Meteo",
        }, ensure_ascii=False)
    except Exception as exc:
        log.warning("weather failed: %s", exc)
        return _err(f"weather service unavailable ({type(exc).__name__}).")


_TICKER_RE = re.compile(r"^\^?[A-Z0-9.\-=&]{1,15}$")


def _find_symbol(query: str) -> str | None:
    """Ticker for a company name, preferring Indian listings (NSE, then BSE) when there are several."""
    import yfinance as yf

    q = query.strip()
    if _TICKER_RE.match(q) and (any(ch in q for ch in ".^=") or q.isupper()):
        return q
    quotes = [x for x in (yf.Search(q, max_results=8).quotes or []) if x.get("symbol")]
    if not quotes:
        return None
    for suffix in (".NS", ".BO"):
        for x in quotes:
            if x["symbol"].endswith(suffix) and x.get("quoteType") in ("EQUITY", "ETF", "INDEX", None):
                return x["symbol"]
    return quotes[0]["symbol"]


@tool
def get_stock_price(company_or_symbol: str) -> str:
    """Latest share price from Yahoo Finance: price, change since the previous close, day range,
    52-week range, market cap and currency. Accepts a company name ("Reliance", "Tata Motors", "Apple")
    or a ticker ("RELIANCE.NS", "AAPL", "^NSEI" for Nifty 50, "^BSESN" for Sensex). Prices may be delayed."""
    try:
        import yfinance as yf

        symbol = _find_symbol(company_or_symbol)
        if not symbol:
            return _err(f"could not find a listed company or ticker for '{company_or_symbol}'.")
        t = yf.Ticker(symbol)
        fi = t.fast_info
        price = fi.get("lastPrice")
        prev = fi.get("previousClose")
        if price is None:
            return _err(f"no price data for {symbol}.")
        change = (price - prev) if prev else None
        name = None
        try:
            name = (yf.Search(symbol, max_results=1).quotes or [{}])[0].get("longname")
        except Exception:
            pass
        r2 = lambda v: round(float(v), 2) if v is not None else None  # noqa: E731
        return json.dumps({
            "symbol": symbol, "name": name, "currency": fi.get("currency"), "exchange": fi.get("exchange"),
            "price": r2(price), "previous_close": r2(prev), "change": r2(change),
            "change_percent": r2(change / prev * 100) if change is not None and prev else None,
            "day_low": r2(fi.get("dayLow")), "day_high": r2(fi.get("dayHigh")),
            "year_low": r2(fi.get("yearLow")), "year_high": r2(fi.get("yearHigh")),
            "market_cap": int(fi["marketCap"]) if fi.get("marketCap") else None,
            "note": "Yahoo Finance data; may be delayed by about 15 minutes.",
        })
    except Exception as exc:
        log.warning("stock price failed: %s", exc)
        return _err(f"share price service unavailable ({type(exc).__name__}: {str(exc)[:120]}).")


@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the web for current or recent information: news, events, recent releases, facts that may
    have changed, or anything not in the files, Salesforce, weather or share-price tools. Returns a short
    answer plus the top sources (title, URL, snippet)."""
    if not config.TAVILY_API_KEY:
        return _err("web search is not configured (TAVILY_API_KEY is missing).")
    try:
        r = requests.post("https://api.tavily.com/search",
                          headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"},
                          json={"query": query, "max_results": max(1, min(int(max_results or 4), 6)),
                                "search_depth": "basic", "include_answer": True},
                          timeout=HTTP_TIMEOUT + 10)
        if r.status_code != 200:
            return _err(f"web search failed (HTTP {r.status_code}).")
        data = r.json()
        return json.dumps({
            "answer": data.get("answer"),
            "results": [{"title": x.get("title"), "url": x.get("url"), "snippet": (x.get("content") or "")[:350],
                         "published": x.get("published_date")} for x in data.get("results", [])],
            "source": "Tavily web search",
        }, ensure_ascii=False)
    except Exception as exc:
        log.warning("web search failed: %s", exc)
        return _err(f"web search unavailable ({type(exc).__name__}).")


def load_live_tools() -> list:
    tools = []
    if config.WEATHER_TOOL_ENABLED:
        tools.append(get_weather)
    if config.STOCK_TOOL_ENABLED:
        tools.append(get_stock_price)
    if config.WEB_SEARCH_ENABLED and config.TAVILY_API_KEY:
        tools.append(web_search)
    return tools
