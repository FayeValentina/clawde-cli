from __future__ import annotations

import contextlib
import importlib
import io
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console

app = typer.Typer(help="Stock market analysis & alerts")
console = Console()

MARKET_TICKERS = ["SPY", "QQQ"]
VIX_TICKER = "^VIX"
DEFAULT_START = "2024-01-01"
ATR_WINDOW = 14
MA_SHORT = 20
MA_LONG = 50
VOLUME_WINDOW = 20
HIGH_52W_WINDOW = 252
DRAWDOWN_WINDOW = 20
REQUEST_TIMEOUT = 20
FED_CALENDAR_URL = "https://www.federalreserve.gov/newsevents/{year}-{month}.htm"
BLS_SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/home.htm"
BEA_SCHEDULE_URL = "https://www.bea.gov/news/schedule"
USER_AGENT = "Mozilla/5.0 (compatible; clawde-stock/1.0)"
WATCHLIST_PATH = Path.home() / ".openclaw" / "stock" / "watchlist.txt"
MACRO_EVENT_MAP = {
    "Consumer Price Index": "CPI",
    "Employment Situation": "非农",
    "Personal Income and Outlays": "PCE",
    "GDP": "GDP",
}


def _load_analysis_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        pd = importlib.import_module("pandas")
        vbt = importlib.import_module("vectorbt")
        yf = importlib.import_module("yfinance")
        bs4 = importlib.import_module("bs4")
    except ImportError as exc:
        raise RuntimeError(
            "Missing stock dependencies. Install pandas, vectorbt, yfinance, beautifulsoup4."
        ) from exc
    return pd, vbt, yf, bs4.BeautifulSoup


def _pct(value: float) -> float:
    return round(value * 100, 2)


def _fetch_html(url: str, *, force_curl: bool = False) -> str:
    if not force_curl:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            pass

    result = subprocess.run(
        [
            "curl",
            "-sL",
            "--compressed",
            "-A",
            USER_AGENT,
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _load_watchlist_tickers(path: Path = WATCHLIST_PATH) -> list[str]:
    if not path.exists():
        return []

    tickers: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            ticker = token.upper()
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def _rolling_max_drawdown(close: Any, window: int) -> Any:
    def calc(values: Any) -> float:
        drawdown = values / values.cummax() - 1
        return float(drawdown.min())

    return close.rolling(window).apply(calc, raw=False)


def _week_bounds(reference_day: date) -> tuple[date, date]:
    week_start = reference_day - timedelta(days=reference_day.weekday())
    return week_start, week_start + timedelta(days=6)


def _month_anchors_for_week(reference_day: date) -> list[date]:
    week_start, week_end = _week_bounds(reference_day)
    anchors = [date(week_start.year, week_start.month, 1)]
    if (week_end.year, week_end.month) != (week_start.year, week_start.month):
        anchors.append(date(week_end.year, week_end.month, 1))
    return anchors


def _classify_trend(latest: Any, prev: Any) -> str:
    ma20_rising = latest["ma20"] >= prev["ma20"]
    ma50_rising = latest["ma50"] >= prev["ma50"]
    above_ma20 = latest["close"] >= latest["ma20"]
    above_ma50 = latest["close"] >= latest["ma50"]
    rsi = latest["rsi14"]

    if above_ma20 and above_ma50 and ma20_rising and ma50_rising and rsi >= 50:
        return "强"
    if (not above_ma20) and (not above_ma50) and (not ma20_rising) and rsi < 45:
        return "弱"
    return "中性"


def _classify_risk(latest: Any, trend: str, pd: Any) -> str:
    score = 0

    if latest["close"] < latest["ma20"]:
        score += 1
    if latest["close"] < latest["ma50"]:
        score += 2
    if latest["rsi14"] > 75 or latest["rsi14"] < 25:
        score += 1
    if latest["atr_pct"] >= 0.06:
        score += 1
    if pd.notna(latest["rvol"]) and latest["rvol"] >= 2.0:
        score += 1
    if latest["drawdown_from_52w_high"] <= -0.2:
        score += 1
    if latest["max_drawdown_20d"] <= -0.1:
        score += 1
    if trend == "弱":
        score += 1

    if score >= 5:
        return "警戒"
    if score >= 2:
        return "注意"
    return "正常"


def _analyze_ticker(ticker: str, start: str) -> dict[str, object]:
    pd, vbt, _, _ = _load_analysis_dependencies()

    data = vbt.YFData.download(ticker, start=start)
    close = data.get("Close")
    high = data.get("High")
    low = data.get("Low")
    volume = data.get("Volume")

    ma20 = vbt.MA.run(close, MA_SHORT).ma
    ma50 = vbt.MA.run(close, MA_LONG).ma
    rsi14 = vbt.RSI.run(close, window=14).rsi
    atr14 = vbt.ATR.run(high, low, close, window=ATR_WINDOW).atr
    day_change_pct = close.pct_change()
    high_52w = close.rolling(HIGH_52W_WINDOW).max()
    drawdown_from_52w_high = close / high_52w - 1
    max_drawdown_20d = _rolling_max_drawdown(close, DRAWDOWN_WINDOW)
    volume_available = volume is not None and volume.fillna(0).gt(0).any()

    if volume_available:
        avg_volume_20d = volume.rolling(VOLUME_WINDOW).mean()
        rvol = volume / avg_volume_20d
    else:
        avg_volume_20d = pd.Series(index=close.index, dtype=float)
        rvol = pd.Series(index=close.index, dtype=float)

    df = pd.DataFrame(
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
            "volume": volume,
        }
    )

    required_cols = [
        "close",
        "day_change_pct",
        "ma20",
        "ma50",
        "dev_ma20",
        "dev_ma50",
        "rsi14",
        "atr_pct",
        "drawdown_from_52w_high",
        "max_drawdown_20d",
    ]
    df = df.dropna(subset=required_cols)

    if len(df) < 2:
        raise ValueError("有效行情不足，至少需要 252 个交易日数据")

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    trend = _classify_trend(latest, prev)
    risk = _classify_risk(latest, trend, pd)

    return {
        "ticker": ticker,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(float(latest["close"]), 2),
        "day_change_pct": _pct(float(latest["day_change_pct"])),
        "ma20": round(float(latest["ma20"]), 2),
        "ma50": round(float(latest["ma50"]), 2),
        "dev_ma20": _pct(float(latest["dev_ma20"])),
        "dev_ma50": _pct(float(latest["dev_ma50"])),
        "rsi14": round(float(latest["rsi14"]), 2),
        "atr_pct": _pct(float(latest["atr_pct"])),
        "avg_volume_20d": round(float(latest["avg_volume_20d"]))
        if pd.notna(latest["avg_volume_20d"])
        else None,
        "rvol": round(float(latest["rvol"]), 2) if pd.notna(latest["rvol"]) else None,
        "drawdown_from_52w_high": _pct(float(latest["drawdown_from_52w_high"])),
        "max_drawdown_20d": _pct(float(latest["max_drawdown_20d"])),
        "trend": trend,
        "risk": risk,
    }


def _fetch_fed_events_for_week(reference_day: date) -> list[str]:
    _, _, _, BeautifulSoup = _load_analysis_dependencies()
    week_start, week_end = _week_bounds(reference_day)
    events: list[str] = []

    for month_anchor in _month_anchors_for_week(reference_day):
        url = FED_CALENDAR_URL.format(
            year=month_anchor.year, month=month_anchor.strftime("%B").lower()
        )
        soup = BeautifulSoup(_fetch_html(url), "html.parser")

        for section in soup.select(".cal-nojs__rowTitle"):
            category = section.get_text(" ", strip=True)
            node = section.find_next_sibling()
            while node and "cal-nojs__rowTitle" not in (node.get("class") or []):
                classes = node.get("class") or []
                if "row" in classes:
                    panel = node.select_one(".panel-body .row")
                    if panel:
                        cols = panel.find_all("div", recursive=False)
                        if len(cols) >= 3:
                            time_text = cols[0].get_text(" ", strip=True)
                            headline = cols[1].find("p")
                            title = cols[1].find("p", class_="calendar__title")
                            day_text = cols[2].get_text(" ", strip=True)
                            if headline and day_text.isdigit():
                                event_date = date(
                                    month_anchor.year, month_anchor.month, int(day_text)
                                )
                                if week_start <= event_date <= week_end:
                                    line = (
                                        f"{event_date:%m-%d} {category}: "
                                        f"{headline.get_text(' ', strip=True)}"
                                    )
                                    if title:
                                        detail = title.get_text(" ", strip=True).strip()
                                        if detail:
                                            line = f"{line} | {detail}"
                                    if time_text:
                                        line = f"{line} | {time_text}"
                                    events.append(line)
                node = node.find_next_sibling()

    deduped: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event not in seen:
            seen.add(event)
            deduped.append(event)
    return deduped[:8]


def _parse_bls_date(value: str) -> date:
    return datetime.strptime(value, "%A, %B %d, %Y").date()


def _parse_bea_date(value: str, year: int) -> date:
    return datetime.strptime(f"{value} {year}", "%B %d %Y").date()


def _fetch_bls_events_for_week(reference_day: date) -> list[dict[str, str]]:
    _, _, _, BeautifulSoup = _load_analysis_dependencies()
    week_start, week_end = _week_bounds(reference_day)
    html = _fetch_html(BLS_SCHEDULE_URL.format(year=reference_day.year), force_curl=True)
    soup = BeautifulSoup(html, "html.parser")

    events: list[dict[str, str]] = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) != 3 or cells[0] == "Date":
            continue

        date_text, time_text, release_text = cells
        event_name = None
        for key, alias in MACRO_EVENT_MAP.items():
            if key in release_text and alias in {"CPI", "非农"}:
                event_name = alias
                break
        if event_name is None:
            continue

        event_date = _parse_bls_date(date_text)
        if not (week_start <= event_date <= week_end):
            continue

        events.append(
            {
                "date": event_date.strftime("%m-%d"),
                "time": time_text or "时间未提供",
                "event": event_name,
                "source": "BLS",
                "detail": release_text,
            }
        )

    return events


def _fetch_bea_events_for_week(reference_day: date) -> list[dict[str, str]]:
    _, _, _, BeautifulSoup = _load_analysis_dependencies()
    week_start, week_end = _week_bounds(reference_day)
    soup = BeautifulSoup(_fetch_html(BEA_SCHEDULE_URL), "html.parser")

    events: list[dict[str, str]] = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < 3 or cells[0].startswith("Year "):
            continue

        release_meta, _, release_text = cells[:3]
        matched = None
        for key, alias in MACRO_EVENT_MAP.items():
            if key in release_text and alias in {"PCE", "GDP"}:
                matched = alias
                break
        if matched is None:
            continue

        parts = release_meta.split()
        if len(parts) < 4:
            continue
        month_day = f"{parts[0]} {parts[1]}"
        time_text = " ".join(parts[2:])
        event_date = _parse_bea_date(month_day, reference_day.year)
        if not (week_start <= event_date <= week_end):
            continue

        events.append(
            {
                "date": event_date.strftime("%m-%d"),
                "time": time_text,
                "event": matched,
                "source": "BEA",
                "detail": release_text,
            }
        )

    return events


def _fetch_macro_events_for_week(reference_day: date) -> list[dict[str, str]]:
    events = _fetch_bls_events_for_week(reference_day) + _fetch_bea_events_for_week(
        reference_day
    )
    events.sort(key=lambda item: (item["date"], item["time"], item["event"]))
    return events


def _normalize_earnings_dates(raw_dates: object) -> list[date]:
    pd, _, _, _ = _load_analysis_dependencies()
    if raw_dates is None:
        return []
    if isinstance(raw_dates, (list, tuple)):
        items = raw_dates
    elif isinstance(raw_dates, pd.Series):
        items = raw_dates.dropna().tolist()
    elif isinstance(raw_dates, pd.Index):
        items = raw_dates.dropna().tolist()
    else:
        items = [raw_dates]

    dates: list[date] = []
    for item in items:
        if isinstance(item, pd.Timestamp):
            dates.append(item.date())
        elif isinstance(item, datetime):
            dates.append(item.date())
        elif isinstance(item, date):
            dates.append(item)
    return dates


def _fetch_earnings_events_for_week(
    reference_day: date, tickers: list[str]
) -> list[dict[str, str]]:
    _, _, yf, _ = _load_analysis_dependencies()
    week_start, week_end = _week_bounds(reference_day)
    events: list[dict[str, str]] = []

    for ticker in tickers:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                calendar = yf.Ticker(ticker).calendar
        except Exception:
            continue

        earnings_dates = _normalize_earnings_dates(calendar.get("Earnings Date"))
        for earnings_date in earnings_dates:
            if not (week_start <= earnings_date <= week_end):
                continue

            events.append(
                {
                    "date": earnings_date.strftime("%m-%d"),
                    "ticker": ticker,
                    "time": "时间未提供",
                    "source": "Yahoo Finance",
                    "detail": "Earnings Date",
                }
            )

    events.sort(key=lambda item: (item["date"], item["ticker"]))
    return events


def _build_market_conclusion(
    spy: dict[str, object], qqq: dict[str, object], vix: dict[str, object]
) -> str:
    vix_value = float(vix["close"])
    strong_count = sum(item["trend"] == "强" for item in (spy, qqq))
    weak_count = sum(item["trend"] == "弱" for item in (spy, qqq))
    risk_count = sum(item["risk"] == "警戒" for item in (spy, qqq))

    if weak_count == 2 or risk_count >= 1 or vix_value >= 25:
        return "风险偏好走弱，优先控制仓位和波动敞口。"
    if strong_count == 2 and vix_value < 20:
        return "风险偏好较强，趋势资产仍占优，但不宜追高。"
    return "市场处于震荡观察区，适合精选个股并跟踪事件驱动。"


def _analyze_market_overview(start: str) -> dict[str, object]:
    spy_state = _analyze_ticker("SPY", start)
    qqq_state = _analyze_ticker("QQQ", start)
    vix_state = _analyze_ticker(VIX_TICKER, start)

    today = date.today()

    try:
        fed_events = _fetch_fed_events_for_week(today)
    except Exception as exc:
        fed_events = [f"事件抓取失败：{exc}"]

    try:
        macro_events = _fetch_macro_events_for_week(today)
    except Exception as exc:
        macro_events = [
            {
                "date": "--",
                "time": "--",
                "event": "抓取失败",
                "source": "系统",
                "detail": str(exc),
            }
        ]

    try:
        earnings_events = _fetch_earnings_events_for_week(today, _load_watchlist_tickers())
    except Exception as exc:
        earnings_events = [
            {
                "date": "--",
                "ticker": "--",
                "time": "--",
                "source": "系统",
                "detail": str(exc),
            }
        ]

    return {
        "spy": spy_state,
        "qqq": qqq_state,
        "vix": vix_state,
        "events": fed_events if fed_events else ["本周暂无从 Fed 日历识别出的重点事件"],
        "macro_events": macro_events,
        "earnings_events": earnings_events,
        "conclusion": _build_market_conclusion(spy_state, qqq_state, vix_state),
    }


def _print_market_overview(overview: dict[str, object]) -> None:
    spy = overview["spy"]
    qqq = overview["qqq"]
    vix = overview["vix"]

    console.print("=== 市场总览 ===")
    console.print(f"SPY 状态: {spy['trend']} / {spy['risk']} / {spy['day_change_pct']}%")
    console.print(f"QQQ 状态: {qqq['trend']} / {qqq['risk']} / {qqq['day_change_pct']}%")
    console.print(f"VIX: {vix['close']} ({vix['day_change_pct']}%)")
    console.print("本周宏观事件:")
    if overview["macro_events"]:
        for event in overview["macro_events"]:
            console.print(
                f"- {event['date']} {event['time']} ET | "
                f"{event['event']} | {event['source']} | {event['detail']}"
            )
    else:
        console.print("- 本周暂无识别到 CPI / PCE / 非农 / GDP")
    console.print("本周重点财报:")
    if overview["earnings_events"]:
        for event in overview["earnings_events"]:
            console.print(
                f"- {event['date']} | {event['ticker']} | {event['time']} | {event['source']}"
            )
    else:
        console.print("- 当前股票池本周暂无财报事件")
    console.print("本周重大事件:")
    for event in overview["events"]:
        console.print(f"- {event}")
    console.print(f"市场风向结论: {overview['conclusion']}")


def _print_quote(item: dict[str, object]) -> None:
    console.print(f"[{item['ticker']}] {item['date']}")
    console.print(f"收盘价: {item['close']}")
    console.print(f"当日涨跌幅: {item['day_change_pct']}%")
    console.print(f"MA20: {item['ma20']}")
    console.print(f"MA50: {item['ma50']}")
    console.print(f"距 MA20 偏离率: {item['dev_ma20']}%")
    console.print(f"距 MA50 偏离率: {item['dev_ma50']}%")
    console.print(f"RSI14: {item['rsi14']}")
    console.print(f"ATR%: {item['atr_pct']}%")
    avg_volume = (
        f"{int(item['avg_volume_20d']):,}" if item["avg_volume_20d"] is not None else "N/A"
    )
    rvol = item["rvol"] if item["rvol"] is not None else "N/A"
    console.print(f"20 日平均成交量: {avg_volume}")
    console.print(f"相对成交量 RVOL: {rvol}")
    console.print(f"距 52 周高点回撤: {item['drawdown_from_52w_high']}%")
    console.print(f"20 日最大回撤: {item['max_drawdown_20d']}%")
    console.print(f"趋势状态: {item['trend']}")
    console.print(f"风险标签: {item['risk']}")


@app.command("alert")
def stock_alert(
    start: str = typer.Option(DEFAULT_START, help="指标计算起始日期"),
) -> None:
    """显示市场总览。"""
    try:
        with console.status("[bold green]正在分析市场总览..."):
            overview = _analyze_market_overview(start)
    except Exception as exc:
        console.print(f"[red]stock alert failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_market_overview(overview)


@app.command("quote")
def stock_quote(
    ticker: str = typer.Argument(..., help="股票代码"),
    start: str = typer.Option(DEFAULT_START, help="指标计算起始日期"),
) -> None:
    """查询单只股票的技术状态。"""
    try:
        with console.status(f"[bold green]正在分析 {ticker.upper()}..."):
            item = _analyze_ticker(ticker.upper(), start)
    except Exception as exc:
        console.print(f"[red]stock quote failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_quote(item)
