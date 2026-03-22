from __future__ import annotations

import contextlib
import importlib
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests
import typer
from rich.console import Console

app = typer.Typer(help="Stock market analysis & alerts")
console = Console()

QUOTE_LOOKBACK_DAYS = 450
RSI_WINDOW = 14
ATR_WINDOW = 14
MA_SHORT = 20
MA_LONG = 50
VOLUME_WINDOW = 20
HIGH_52W_WINDOW = 252
DRAWDOWN_WINDOW = 20
REQUEST_TIMEOUT = 20
DEFAULT_ALERT_HOURS = 72
DEFAULT_ALERT_DISPLAY_LIMIT = 5
DEFAULT_ALERT_RAW_LIMIT = 75
DEFAULT_QUOTE_NEWS_HOURS = 72
DEFAULT_QUOTE_NEWS_LIMIT = 3
POLYGON_BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
NEW_YORK_TZ = ZoneInfo("America/New_York")
MARKET_NEWS_KEYWORDS = [
    "s&p 500",
    "nasdaq",
    "dow",
    "stocks",
    "wall street",
    "market",
    "fed",
    "federal reserve",
    "interest rate",
    "inflation",
    "cpi",
    "pce",
    "jobs",
    "earnings",
    "guidance",
    "treasury",
    "treasuries",
    "yield",
    "ai",
    "artificial intelligence",
    "semiconductor",
    "chip",
    "tech",
    "megacap",
    "magnificent seven",
    "risk appetite",
    "risk-off",
    "recession fears",
    "tariff",
]
POSITIVE_SENTIMENT_TERMS = [
    "rally",
    "surge",
    "gain",
    "beat",
    "bullish",
    "easing",
    "cooling inflation",
    "optimism",
    "rebound",
    "upgrade",
]
NEGATIVE_SENTIMENT_TERMS = [
    "selloff",
    "risk-off",
    "decline",
    "drop",
    "hot inflation",
    "sticky inflation",
    "reaccelerating inflation",
    "hawkish",
    "tariff",
    "warning",
    "downgrade",
    "recession",
]
STRICT_TICKER_ALIASES = {
    "AAPL": ["aapl", "apple"],
    "CRCL": ["crcl", "circle", "circle internet group"],
    "GOOG": ["goog", "google", "alphabet"],
    "GOOGL": ["googl", "google", "alphabet"],
    "IBKR": ["ibkr", "interactive brokers", "interactive brokers group"],
    "NVDA": ["nvda", "nvidia"],
    "AMD": ["amd", "advanced micro devices"],
    "MU": ["mu", "micron", "micron technology"],
    "MSTR": ["mstr", "microstrategy"],
    "AMZN": ["amzn", "amazon"],
    "PLTR": ["pltr", "palantir", "palantir technologies"],
    "NBIS": ["nbis", "nebius", "nebius group"],
    "AAOI": ["aaoi", "applied optoelectronics", "applied optoelectronics inc"],
    "LITE": ["lite", "lumentum", "lumentum holdings"],
    "SAND": ["sand", "sandstorm gold", "sandstorm gold royalties"],
    "SNDK": ["sndk", "sandisk"],
    "ASTS": ["asts", "ast spacemobile", "ast space mobile"],
    "SWMR": ["swmr", "swell medical"],
    "RKLB": ["rklb", "rocket lab", "rocket lab usa"],
    "ONDS": ["onds", "ondas", "ondas holdings"],
    "AEHR": ["aehr", "aehr test systems"],
    "TSEM": ["tsem", "tower semiconductor", "tower semi"],
    "POET": ["poet", "poet technologies"],
    "COHR": ["cohr", "coherent"],
    "AXTI": ["axti", "axt"],
    "IREN": ["iren", "iris energy"],
    "ANET": ["anet", "arista", "arista networks"],
    "FN": ["fn", "fabrinet"],
    "GLW": ["glw", "corning"],
    "MSFT": ["msft", "microsoft"],
    "META": ["meta", "meta platforms", "facebook"],
    "TSLA": ["tsla", "tesla"],
    "NFLX": ["nflx", "netflix"],
}
ETF_TICKERS = {"QQQ", "VOO", "SMH", "XOP", "USO"}
ETF_ALIASES = {
    "QQQ": [
        "qqq",
        "invesco qqq",
        "nasdaq 100",
        "nasdaq-100",
        "ndx",
        "big tech",
        "mega-cap tech",
    ],
    "VOO": [
        "voo",
        "vanguard s&p 500",
        "s&p 500",
        "sp500",
        "large cap us stocks",
    ],
    "SMH": [
        "smh",
        "van eck semiconductor",
        "vaneck semiconductor",
        "semiconductor etf",
        "semiconductor stocks",
        "chip stocks",
        "semis",
    ],
    "XOP": [
        "xop",
        "s&p oil & gas exploration & production",
        "oil and gas exploration and production",
        "exploration and production stocks",
        "e&p stocks",
        "oil producers",
    ],
    "USO": [
        "uso",
        "united states oil fund",
        "oil etf",
        "wti crude",
        "crude oil prices",
        "oil prices",
    ],
}


def _load_analysis_dependencies() -> tuple[Any, Any]:
    """Load pandas and vectorbt lazily for CLI usage."""
    try:
        pd = importlib.import_module("pandas")
        vbt = importlib.import_module("vectorbt")
    except ImportError as exc:
        raise RuntimeError(
            "Missing stock dependencies. Install pandas and vectorbt."
        ) from exc
    return pd, vbt


def _pct(value: float) -> float:
    """Convert a decimal ratio to percentage points."""
    return round(value * 100, 2)


def _quote_start_from_lookback(lookback_days: int = QUOTE_LOOKBACK_DAYS) -> str:
    """Return a start date derived from a fixed lookback window."""
    return (datetime.now(UTC).date() - timedelta(days=lookback_days)).isoformat()


def _require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _rolling_max_drawdown(close: Any, window: int) -> Any:
    """Compute rolling max drawdown over a fixed window."""

    def calc(values: Any) -> float:
        drawdown = values / values.cummax() - 1
        return float(drawdown.min())

    return close.rolling(window).apply(calc, raw=False)


def _format_news_timestamp(value: str) -> str | None:
    """Format an API timestamp into a compact New York market time label."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    with contextlib.suppress(ValueError):
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        ny_time = parsed.astimezone(NEW_YORK_TZ)
        return ny_time.strftime("%Y-%m-%d %H:%M ET")
    return value


def _format_metric(value: object, *, suffix: str = "", digits: int = 2) -> str:
    """Format numeric metrics while tolerating missing values."""
    if value is None:
        return "暂无"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _contains_keyword(text: str, keyword: str) -> bool:
    """Match keywords with simple word boundaries to reduce substring false positives."""
    escaped = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _contains_alias(text: str, alias: str, *, strict: bool = False) -> bool:
    """Match ticker symbols or company aliases with configurable strictness."""
    if strict or " " not in alias:
        return _contains_keyword(text, alias)
    return alias.lower() in text.lower()


def _is_etf_ticker(ticker: str) -> bool:
    """Return whether the ticker should use ETF-style thematic news matching."""
    return ticker.upper() in ETF_TICKERS


def _get_news_aliases(ticker: str) -> list[str]:
    """Return the configured aliases for a ticker."""
    normalized = ticker.upper()
    if _is_etf_ticker(normalized):
        return ETF_ALIASES.get(normalized, [normalized.lower()])
    return STRICT_TICKER_ALIASES.get(normalized, [normalized.lower()])


def _matches_strict_alias(text: str, alias: str) -> bool:
    """Use boundary-aware matching for stock tickers and aliases."""
    return _contains_keyword(text, alias)


def _matches_etf_alias(text: str, alias: str) -> bool:
    """Allow broader thematic matching for ETF aliases while keeping token aliases strict."""
    return _contains_alias(text, alias, strict=False)


def _matches_ticker_news(ticker: str, text: str) -> bool:
    """Apply stock-vs-ETF-specific local relevance matching."""
    aliases = _get_news_aliases(ticker)
    matcher = _matches_etf_alias if _is_etf_ticker(ticker) else _matches_strict_alias
    return any(matcher(text, alias) for alias in aliases)


def _fetch_polygon_daily_bars(ticker: str, start: str) -> Any:
    """Fetch daily OHLCV bars from Polygon and normalize them to a DataFrame."""
    pd, _ = _load_analysis_dependencies()
    api_key = _require_env("POLYGON_API_KEY")
    end = datetime.now(UTC).date().isoformat()
    response = requests.get(
        POLYGON_BASE_URL.format(ticker=ticker.upper(), start=start, end=end),
        params={
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
            "apiKey": api_key,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        detail = ""
        with contextlib.suppress(ValueError):
            payload = response.json()
            detail = payload.get("error") or payload.get("message") or ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Polygon request failed with status {response.status_code}{suffix}")

    payload = response.json()
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Polygon returned no daily bars for {ticker.upper()}")

    frame = pd.DataFrame(results)
    required = {"o", "h", "l", "c", "v", "t"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Polygon daily bars for {ticker.upper()} are missing expected fields")

    frame = frame.rename(
        columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        }
    )
    frame.index = (
        pd.to_datetime(frame["t"], unit="ms", utc=True)
        .dt.tz_convert(NEW_YORK_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _classify_trend(latest: Any, prev: Any, pd: Any) -> str | None:
    """Classify the latest technical trend state from available indicators."""
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    close = latest.get("close")
    rsi = latest.get("rsi14")
    prev_ma20 = prev.get("ma20")
    prev_ma50 = prev.get("ma50")

    if pd.isna(close) or pd.isna(rsi):
        return None

    has_ma20 = not pd.isna(ma20) and not pd.isna(prev_ma20)
    has_ma50 = not pd.isna(ma50) and not pd.isna(prev_ma50)

    if has_ma20 and has_ma50:
        ma20_rising = ma20 >= prev_ma20
        ma50_rising = ma50 >= prev_ma50
        above_ma20 = close >= ma20
        above_ma50 = close >= ma50
        if above_ma20 and above_ma50 and ma20_rising and ma50_rising and rsi >= 50:
            return "强"
        if (not above_ma20) and (not above_ma50) and (not ma20_rising) and rsi < 45:
            return "弱"
        return "中性"

    if has_ma20:
        ma20_rising = ma20 >= prev_ma20
        above_ma20 = close >= ma20
        if above_ma20 and ma20_rising and rsi >= 50:
            return "偏强"
        if (not above_ma20) and (not ma20_rising) and rsi < 45:
            return "偏弱"
        return "中性"

    if rsi >= 60:
        return "偏强"
    if rsi <= 40:
        return "偏弱"
    return "中性"


def _classify_risk(latest: Any, trend: str | None, pd: Any) -> str | None:
    """Assign a coarse risk label from the latest indicator snapshot."""
    score = 0
    checks = 0

    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    close = latest.get("close")
    rsi14 = latest.get("rsi14")
    atr_pct = latest.get("atr_pct")
    rvol = latest.get("rvol")
    drawdown_52w = latest.get("drawdown_from_52w_high")
    max_drawdown_20d = latest.get("max_drawdown_20d")

    if not pd.isna(close) and not pd.isna(ma20):
        checks += 1
        if close < ma20:
            score += 1
    if not pd.isna(close) and not pd.isna(ma50):
        checks += 1
        if close < ma50:
            score += 2
    if not pd.isna(rsi14):
        checks += 1
        if rsi14 > 75 or rsi14 < 25:
            score += 1
    if not pd.isna(atr_pct):
        checks += 1
        if atr_pct >= 0.06:
            score += 1
    if not pd.isna(rvol):
        checks += 1
        if rvol >= 2.0:
            score += 1
    if not pd.isna(drawdown_52w):
        checks += 1
        if drawdown_52w <= -0.2:
            score += 1
    if not pd.isna(max_drawdown_20d):
        checks += 1
        if max_drawdown_20d <= -0.1:
            score += 1
    if trend in {"弱", "偏弱"}:
        checks += 1
        score += 1

    if checks == 0:
        return None
    if score >= 5:
        return "警戒"
    if score >= 2:
        return "注意"
    return "正常"


def _analyze_price_frame(ticker: str, price_df: Any) -> dict[str, object]:
    """Compute the quote snapshot from a normalized OHLCV price DataFrame."""
    pd, vbt = _load_analysis_dependencies()
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(price_df.columns):
        missing = ", ".join(sorted(required_columns - set(price_df.columns)))
        raise ValueError(f"Price data is missing required columns: {missing}")

    if price_df.empty:
        raise ValueError(f"No price data available for {ticker}")

    close = price_df["Close"].astype(float)
    high = price_df["High"].astype(float)
    low = price_df["Low"].astype(float)
    volume = price_df["Volume"].astype(float)
    history_days = len(price_df)

    ma20 = vbt.MA.run(close, MA_SHORT).ma
    ma50 = vbt.MA.run(close, MA_LONG).ma
    rsi14 = vbt.RSI.run(close, window=RSI_WINDOW).rsi
    atr14 = vbt.ATR.run(high, low, close, window=ATR_WINDOW).atr
    day_change_pct = close.pct_change()
    high_52w = high.rolling(HIGH_52W_WINDOW, min_periods=1).max()
    drawdown_from_52w_high = close / high_52w - 1
    max_drawdown_20d = _rolling_max_drawdown(close, DRAWDOWN_WINDOW)
    volume_available = volume.fillna(0).gt(0).any()

    if volume_available:
        avg_volume_20d = volume.rolling(VOLUME_WINDOW, min_periods=1).mean()
        rvol = volume / avg_volume_20d
    else:
        avg_volume_20d = pd.Series(index=close.index, dtype=float)
        rvol = pd.Series(index=close.index, dtype=float)

    metrics = pd.DataFrame(
        {
            "close": close,
            "day_change_pct": day_change_pct,
            "ma20": ma20,
            "ma50": ma50,
            "dev_ma20": close / ma20 - 1,
            "dev_ma50": close / ma50 - 1,
            "rsi14": rsi14,
            "atr_pct": atr14 / close,
            "avg_volume_20d": avg_volume_20d,
            "rvol": rvol,
            "drawdown_from_52w_high": drawdown_from_52w_high,
            "max_drawdown_20d": max_drawdown_20d,
        }
    )

    latest = metrics.iloc[-1]
    prev = metrics.iloc[-2] if len(metrics) >= 2 else metrics.iloc[-1]
    latest_date = price_df.index[-1]
    history_warning: str | None = None

    if history_days < VOLUME_WINDOW:
        history_warning = "历史短于 20 个交易日，部分技术指标暂不可用。"
    elif history_days < MA_LONG:
        history_warning = "历史短于 50 个交易日，MA50 与部分趋势/风险判断暂不可用。"
    elif history_days < HIGH_52W_WINDOW:
        history_warning = "历史短于 252 个交易日，52 周最高价回撤改用样本期最高价计算。"

    ma20_value = None if history_days < MA_SHORT or pd.isna(latest["ma20"]) else round(float(latest["ma20"]), 2)
    ma50_value = None if history_days < MA_LONG or pd.isna(latest["ma50"]) else round(float(latest["ma50"]), 2)
    dev_ma20_value = None if ma20_value is None or pd.isna(latest["dev_ma20"]) else _pct(float(latest["dev_ma20"]))
    dev_ma50_value = None if ma50_value is None or pd.isna(latest["dev_ma50"]) else _pct(float(latest["dev_ma50"]))
    rsi14_value = None if history_days < RSI_WINDOW or pd.isna(latest["rsi14"]) else round(float(latest["rsi14"]), 2)
    atr_pct_value = None if history_days < ATR_WINDOW or pd.isna(latest["atr_pct"]) else _pct(float(latest["atr_pct"]))
    avg_volume_20d_value = None
    rvol_value = None
    if history_days >= VOLUME_WINDOW and pd.notna(latest["avg_volume_20d"]):
        avg_volume_20d_value = round(float(latest["avg_volume_20d"]))
    if history_days >= VOLUME_WINDOW and pd.notna(latest["rvol"]):
        rvol_value = round(float(latest["rvol"]), 2)
    drawdown_52w_value = None if pd.isna(latest["drawdown_from_52w_high"]) else _pct(float(latest["drawdown_from_52w_high"]))
    max_drawdown_20d_value = None if history_days < DRAWDOWN_WINDOW or pd.isna(latest["max_drawdown_20d"]) else _pct(float(latest["max_drawdown_20d"]))

    latest_for_classification = {
        "close": latest["close"],
        "ma20": latest["ma20"] if ma20_value is not None else pd.NA,
        "ma50": latest["ma50"] if ma50_value is not None else pd.NA,
        "rsi14": latest["rsi14"] if rsi14_value is not None else pd.NA,
        "atr_pct": latest["atr_pct"] if atr_pct_value is not None else pd.NA,
        "rvol": latest["rvol"] if rvol_value is not None else pd.NA,
        "drawdown_from_52w_high": latest["drawdown_from_52w_high"] if drawdown_52w_value is not None else pd.NA,
        "max_drawdown_20d": latest["max_drawdown_20d"] if max_drawdown_20d_value is not None else pd.NA,
    }
    prev_for_classification = {
        "ma20": prev["ma20"] if len(metrics) >= 2 and history_days >= MA_SHORT else pd.NA,
        "ma50": prev["ma50"] if len(metrics) >= 2 and history_days >= MA_LONG else pd.NA,
    }
    trend = _classify_trend(latest_for_classification, prev_for_classification, pd)
    risk = _classify_risk(latest_for_classification, trend, pd)

    return {
        "ticker": ticker,
        "date": latest_date.strftime("%Y-%m-%d"),
        "history_days": history_days,
        "history_warning": history_warning,
        "close": round(float(latest["close"]), 2),
        "day_change_pct": None if pd.isna(latest["day_change_pct"]) else _pct(float(latest["day_change_pct"])),
        "ma20": ma20_value,
        "ma50": ma50_value,
        "dev_ma20": dev_ma20_value,
        "dev_ma50": dev_ma50_value,
        "rsi14": rsi14_value,
        "atr_pct": atr_pct_value,
        "avg_volume_20d": avg_volume_20d_value,
        "rvol": rvol_value,
        "drawdown_from_52w_high": drawdown_52w_value,
        "drawdown_label": "距 52 周最高价回撤" if history_days >= HIGH_52W_WINDOW else "距样本期最高价回撤",
        "max_drawdown_20d": max_drawdown_20d_value,
        "trend": trend,
        "risk": risk,
    }


def _marketaux_request(params: dict[str, object]) -> list[dict[str, Any]]:
    """Execute a Marketaux news request and normalize the response."""
    token = _require_env("MARKETAUX_API_TOKEN")
    response = requests.get(
        MARKETAUX_NEWS_URL,
        params={**params, "api_token": token},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        detail = ""
        with contextlib.suppress(ValueError):
            payload = response.json()
            detail = payload.get("error") or payload.get("message") or ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"Marketaux request failed with status {response.status_code}{suffix}"
        )

    payload = response.json()
    raw_items = payload.get("data") or []
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        normalized.append(
            {
                "title": (item.get("title") or "").strip(),
                "summary": (item.get("description") or item.get("snippet") or "").strip(),
                "source": (item.get("source") or item.get("domain") or "未知来源").strip(),
                "published_at": item.get("published_at") or "",
                "url": item.get("url") or "",
                "entities": item.get("entities") or [],
            }
        )
    return normalized


def _fetch_marketaux_news(raw_limit: int = DEFAULT_ALERT_RAW_LIMIT, hours: int = DEFAULT_ALERT_HOURS) -> list[dict[str, Any]]:
    """Fetch recent broad finance news from Marketaux."""
    published_after = (datetime.now(UTC) - timedelta(hours=hours)).replace(microsecond=0)
    return _marketaux_request(
        {
            "language": "en",
            "limit": raw_limit,
            "published_after": published_after.strftime("%Y-%m-%dT%H:%M:%S"),
            "group_similar": "true",
        }
    )


def _fetch_ticker_news(
    ticker: str,
    limit: int = DEFAULT_QUOTE_NEWS_LIMIT,
    hours: int = DEFAULT_QUOTE_NEWS_HOURS,
) -> list[dict[str, Any]]:
    """Fetch recent ticker-specific news from Marketaux."""
    published_after = (datetime.now(UTC) - timedelta(hours=hours)).replace(microsecond=0)
    news_items = _marketaux_request(
        {
            "language": "en",
            "limit": max(limit * 3, 10),
            "published_after": published_after.strftime("%Y-%m-%dT%H:%M:%S"),
            "group_similar": "true",
            "symbols": ticker.upper(),
            "filter_entities": "true",
        }
    )
    matched: list[dict[str, Any]] = []
    for item in news_items:
        haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        if _matches_ticker_news(ticker, haystack):
            matched.append(item)
    return matched[:limit]


def _split_market_news(news_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep only high-relevance market news items."""
    matched: list[dict[str, Any]] = []

    for item in news_items:
        haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        if any(_contains_keyword(haystack, keyword) for keyword in MARKET_NEWS_KEYWORDS):
            matched.append(item)

    return {"matched": matched}


def _summarize_market_news(news_items: list[dict[str, Any]]) -> str | None:
    """Generate a simple rule-based market tone summary from matched market news only."""
    if not news_items:
        return None

    positive_score = 0
    negative_score = 0
    for item in news_items:
        haystack = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        positive_score += sum(_contains_keyword(haystack, term) for term in POSITIVE_SENTIMENT_TERMS)
        negative_score += sum(_contains_keyword(haystack, term) for term in NEGATIVE_SENTIMENT_TERMS)

    if negative_score >= positive_score + 2:
        return "新闻流偏谨慎，风险偏好承压，短线更适合控制仓位并关注波动。"
    if positive_score >= negative_score + 2:
        return "新闻流偏积极，风险偏好改善，科技与成长方向更容易获得资金关注。"
    return "新闻流多空交织，市场更像事件驱动震荡，交易上宜保持选择性。"


def _build_market_news_overview(
    display_limit: int = DEFAULT_ALERT_DISPLAY_LIMIT,
    hours: int = DEFAULT_ALERT_HOURS,
    raw_limit: int = DEFAULT_ALERT_RAW_LIMIT,
) -> dict[str, object]:
    """Build the alert payload from Marketaux news only."""
    split = _split_market_news(_fetch_marketaux_news(raw_limit=raw_limit, hours=hours))
    matched = split["matched"][:display_limit]
    return {
        "matched_news_items": matched,
        "conclusion": _summarize_market_news(matched),
        "no_match_message": (
            f"最近 {hours} 小时未找到高相关度的市场主线新闻，可尝试扩大时间窗口或放宽相关性条件。"
            if not matched
            else None
        ),
    }


def _print_news_section(title: str, news_items: list[dict[str, Any]], empty_message: str) -> None:
    """Render a list of news items with compact metadata."""
    if title:
        console.print(title)
    if not news_items:
        console.print(empty_message)
        return
    for item in news_items:
        source = item.get("source") or "未知来源"
        title_text = item.get("title") or "无标题"
        summary = item.get("summary") or "暂无摘要。"
        published_at = _format_news_timestamp(str(item.get("published_at") or ""))
        url = item.get("url") or ""
        console.print(f"- [{source}] {title_text}")
        console.print(f"  {summary}")
        if published_at:
            console.print(f"  发布时间: {published_at}")
        if url:
            console.print(f"  链接: {url}")


def _print_market_news_overview(overview: dict[str, object], hours: int = DEFAULT_ALERT_HOURS) -> None:
    """Render the market news alert output."""
    matched_news_items = overview["matched_news_items"]
    console.print("=== 市场新闻摘要 ===")
    _print_news_section(
        "",
        matched_news_items,
        f"最近 {hours} 小时暂无高相关度市场主题新闻",
    )
    if overview.get("no_match_message"):
        console.print(str(overview["no_match_message"]))
    conclusion = overview.get("conclusion")
    if conclusion:
        console.print(f"市场风向结论: {conclusion}")


def _print_quote_news(ticker: str, news_items: list[dict[str, Any]], hours: int) -> None:
    """Render the ticker-specific recent news section."""
    console.print(f"=== {ticker} 相关新闻 ===")
    if not news_items:
        console.print(f"最近 {hours} 小时暂无高相关度的 {ticker} 相关新闻")
        return
    for item in news_items:
        source = item.get("source") or "未知来源"
        title = item.get("title") or "无标题"
        summary = item.get("summary") or "暂无摘要。"
        published_at = _format_news_timestamp(str(item.get("published_at") or ""))
        url = item.get("url") or ""
        console.print(f"- [{source}] {title}")
        console.print(f"  {summary}")
        if published_at:
            console.print(f"  发布时间: {published_at}")
        if url:
            console.print(f"  链接: {url}")


def _print_quote(item: dict[str, object]) -> None:
    """Render the quote snapshot."""
    console.print(f"[{item['ticker']}] {item['date']}")
    console.print(f"收盘价: {_format_metric(item['close'])}")
    console.print(f"当日涨跌幅: {_format_metric(item['day_change_pct'], suffix='%')}")
    console.print(f"MA20: {_format_metric(item.get('ma20'))}")
    console.print(f"MA50: {_format_metric(item.get('ma50'))}")
    console.print(f"距 MA20 偏离率: {_format_metric(item.get('dev_ma20'), suffix='%')}")
    console.print(f"距 MA50 偏离率: {_format_metric(item.get('dev_ma50'), suffix='%')}")
    console.print(f"RSI14: {_format_metric(item.get('rsi14'))}")
    console.print(f"ATR%: {_format_metric(item.get('atr_pct'), suffix='%')}")
    avg_volume = (
        f"{int(item['avg_volume_20d']):,}" if item.get("avg_volume_20d") is not None else "暂无"
    )
    console.print(f"20 日平均成交量: {avg_volume}")
    console.print(f"相对成交量 RVOL: {_format_metric(item.get('rvol'))}")
    console.print(
        f"{item.get('drawdown_label', '距 52 周最高价回撤')}: "
        f"{_format_metric(item.get('drawdown_from_52w_high'), suffix='%')}"
    )
    console.print(f"20 日最大回撤: {_format_metric(item.get('max_drawdown_20d'), suffix='%')}")
    console.print(f"趋势状态: {item.get('trend') or '暂无'}")
    console.print(f"风险标签: {item.get('risk') or '暂无'}")
    if item.get("history_warning"):
        console.print(f"提示: {item['history_warning']}")


@app.command("alert")
def stock_alert(
    limit: int = typer.Option(DEFAULT_ALERT_DISPLAY_LIMIT, min=1, help="展示新闻条数"),
    hours: int = typer.Option(DEFAULT_ALERT_HOURS, min=1, help="回看新闻小时数"),
) -> None:
    """显示近期市场新闻摘要。"""
    try:
        with console.status("[bold green]正在获取市场新闻..."):
            overview = _build_market_news_overview(
                display_limit=limit,
                hours=hours,
            )
    except Exception as exc:
        console.print(f"[red]stock alert 执行失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_market_news_overview(overview, hours=hours)


@app.command("quote")
def stock_quote(
    ticker: str = typer.Argument(..., help="股票代码"),
    news_limit: int = typer.Option(DEFAULT_QUOTE_NEWS_LIMIT, min=1, help="ticker 新闻展示条数"),
    news_hours: int = typer.Option(DEFAULT_QUOTE_NEWS_HOURS, min=1, help="ticker 新闻回看小时数"),
    lookback_days: int = typer.Option(QUOTE_LOOKBACK_DAYS, min=30, help="技术分析价格回看天数"),
) -> None:
    """查询单只股票的近期新闻和技术状态。"""
    normalized_ticker = ticker.upper()
    news_items: list[dict[str, Any]] = []
    news_warning: str | None = None

    try:
        news_items = _fetch_ticker_news(
            normalized_ticker,
            limit=news_limit,
            hours=news_hours,
        )
    except Exception as exc:
        news_warning = f"相关新闻获取失败：{exc}"

    try:
        with console.status(f"[bold green]正在获取 {ticker.upper()} 行情并计算技术指标..."):
            resolved_start = _quote_start_from_lookback(lookback_days)
            price_df = _fetch_polygon_daily_bars(normalized_ticker, resolved_start)
            item = _analyze_price_frame(normalized_ticker, price_df)
    except Exception as exc:
        console.print(f"[red]stock quote 执行失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_quote_news(item["ticker"], news_items, news_hours)
    if news_warning:
        console.print(f"提示: {news_warning}")
    console.print()
    _print_quote(item)
