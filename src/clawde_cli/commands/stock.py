from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import requests
import typer
from rich.console import Console

app = typer.Typer(help="Stock quote and technical snapshot")
console = Console()

# ---------- Constants ----------
QUOTE_LOOKBACK_DAYS = 260
REQUEST_TIMEOUT = 20

FMP_HISTORICAL_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

MA_SHORT = 20
MA_LONG = 50
RSI_WINDOW = 14
ATR_WINDOW = 14
VOLUME_WINDOW = 20
HIGH_52W_WINDOW = 252
DRAWDOWN_WINDOW = 20

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_WINDOW = 20
BB_STD = 2

STOCH_RSI_WINDOW = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3

ADX_WINDOW = 14


# ---------- Helpers ----------
def _load_analysis_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("缺少 pandas，请先安装：pip install pandas") from exc

    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError("缺少 vectorbt，请先安装：pip install vectorbt") from exc

    return pd, vbt


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


def _pct(value: float) -> float:
    return value * 100.0


def _quote_start_from_lookback(lookback_days: int) -> str:
    extra_buffer = 120
    start_date = datetime.now(UTC).date() - timedelta(days=lookback_days + extra_buffer)
    return start_date.isoformat()


def _format_metric(value: object, *, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "暂无"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _resolve_metric(
    value: object,
    *,
    pd: Any,
    history_days: int,
    min_history: int = 0,
    transform: Any | None = None,
) -> object:
    if history_days < min_history or pd.isna(value):
        return None
    resolved = float(value)
    return transform(resolved) if transform is not None else resolved


def _rolling_max_drawdown(close: Any, window: int) -> Any:
    def calc(values: Any) -> float:
        drawdown = values / values.cummax() - 1
        return float(drawdown.min())

    return close.rolling(window).apply(calc, raw=False)


# ---------- Data Fetch ----------
def _fetch_fmp_daily_bars(ticker: str, start: str) -> Any:
    pd, _ = _load_analysis_dependencies()
    api_key = _require_env("FMP_API_KEY")

    response = requests.get(
        FMP_HISTORICAL_URL,
        params={
            "symbol": ticker.upper(),
            "from": start,
            "to": datetime.now(UTC).date().isoformat(),
            "apikey": api_key,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("error") or payload.get("message") or ""
        except ValueError:
            pass
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"FMP request failed with status {response.status_code}{suffix}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"FMP returned no daily bars for {ticker.upper()}")

    frame = pd.DataFrame(payload)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"FMP daily bars for {ticker.upper()} are missing expected fields: {missing}")

    frame = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    frame.index = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    if frame.empty:
        raise ValueError(f"FMP returned an empty normalized frame for {ticker.upper()}")

    return frame


# ---------- Indicator Calculations ----------
def _ema(series: Any, span: int) -> Any:
    return series.ewm(span=span, adjust=False).mean()


def _compute_macd(close: Any) -> tuple[Any, Any, Any]:
    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    macd = ema_fast - ema_slow
    signal = _ema(macd, MACD_SIGNAL)
    hist = macd - signal
    return macd, signal, hist


def _compute_bollinger(close: Any, pd: Any) -> tuple[Any, Any, Any, Any, Any]:
    middle = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std(ddof=0)
    upper = middle + BB_STD * std
    lower = middle - BB_STD * std
    width = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    percent_b = percent_b.where((upper - lower) != 0, pd.NA)
    return upper, middle, lower, width, percent_b


def _compute_obv(close: Any, volume: Any, pd: Any) -> Any:
    direction = pd.Series(0.0, index=close.index)
    direction = direction.mask(close > close.shift(1), 1.0)
    direction = direction.mask(close < close.shift(1), -1.0)
    return (direction * volume.fillna(0)).cumsum()


def _compute_adx(high: Any, low: Any, close: Any, window: int, pd: Any) -> Any:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm = plus_dm.mask((up_move > down_move) & (up_move > 0), up_move)
    minus_dm = minus_dm.mask((down_move > up_move) & (down_move > 0), down_move)

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx


def _compute_stoch_rsi(close: Any, pd: Any) -> tuple[Any, Any]:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_WINDOW, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_min = rsi.rolling(STOCH_RSI_WINDOW, min_periods=STOCH_RSI_WINDOW).min()
    rsi_max = rsi.rolling(STOCH_RSI_WINDOW, min_periods=STOCH_RSI_WINDOW).max()
    raw = (rsi - rsi_min) / (rsi_max - rsi_min)
    raw = raw.where((rsi_max - rsi_min) != 0, pd.NA)

    k = raw.rolling(STOCH_RSI_SMOOTH_K, min_periods=STOCH_RSI_SMOOTH_K).mean() * 100
    d = k.rolling(STOCH_RSI_SMOOTH_D, min_periods=STOCH_RSI_SMOOTH_D).mean()
    return k, d


def _classify_trend(latest: dict[str, Any]) -> str:
    close = latest.get("close")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    rsi14 = latest.get("rsi14")
    macd = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    adx14 = latest.get("adx14")

    if None in (close, ma20, ma50, rsi14):
        return "样本不足"

    bullish = close > ma20 and close > ma50 and rsi14 >= 50
    bearish = close < ma20 and close < ma50 and rsi14 < 50

    macd_bull = macd is not None and macd_signal is not None and macd > macd_signal
    macd_bear = macd is not None and macd_signal is not None and macd < macd_signal
    strong_trend = adx14 is not None and adx14 >= 25

    if bullish and macd_bull and strong_trend:
        return "强势上升"
    if bullish:
        return "偏强"
    if bearish and macd_bear and strong_trend:
        return "强势下降"
    if bearish:
        return "偏弱"
    return "震荡"


def _classify_risk(latest: dict[str, Any]) -> str:
    risk_score = 0

    atr_pct = latest.get("atr_pct")
    rvol = latest.get("rvol")
    drawdown = latest.get("drawdown_from_52w_high")
    max_dd_20 = latest.get("max_drawdown_20d")
    bb_width = latest.get("bb_width")
    stochrsi_k = latest.get("stochrsi_k")

    if atr_pct is not None and atr_pct >= 6:
        risk_score += 2
    elif atr_pct is not None and atr_pct >= 4:
        risk_score += 1

    if rvol is not None and rvol >= 2.0:
        risk_score += 1

    if drawdown is not None and drawdown <= -30:
        risk_score += 2
    elif drawdown is not None and drawdown <= -15:
        risk_score += 1

    if max_dd_20 is not None and max_dd_20 <= -12:
        risk_score += 2
    elif max_dd_20 is not None and max_dd_20 <= -7:
        risk_score += 1

    if bb_width is not None and bb_width >= 12:
        risk_score += 1

    if stochrsi_k is not None and (stochrsi_k >= 90 or stochrsi_k <= 10):
        risk_score += 1

    if risk_score >= 5:
        return "高波动"
    if risk_score >= 3:
        return "中等偏高"
    return "正常"


def _analyze_price_frame(ticker: str, frame: Any) -> dict[str, Any]:
    pd, vbt = _load_analysis_dependencies()

    if len(frame) < 2:
        raise ValueError(f"{ticker} 历史数据不足，至少需要 2 根日线")

    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    volume = frame["Volume"].astype(float)

    ma20 = vbt.MA.run(close, MA_SHORT).ma
    ma50 = vbt.MA.run(close, MA_LONG).ma
    rsi14 = vbt.RSI.run(close, window=RSI_WINDOW).rsi
    atr14 = vbt.ATR.run(high, low, close, window=ATR_WINDOW).atr

    macd, macd_signal, macd_hist = _compute_macd(close)
    bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b = _compute_bollinger(close, pd)
    adx14 = _compute_adx(high, low, close, ADX_WINDOW, pd)
    obv = _compute_obv(close, volume, pd)
    stochrsi_k, stochrsi_d = _compute_stoch_rsi(close, pd)

    day_change_pct = close.pct_change()
    avg_volume_20d = volume.rolling(VOLUME_WINDOW, min_periods=1).mean()
    rvol = volume / avg_volume_20d

    high_52w = high.rolling(HIGH_52W_WINDOW, min_periods=1).max()
    drawdown_from_52w_high = close / high_52w - 1
    max_drawdown_20d = _rolling_max_drawdown(close, DRAWDOWN_WINDOW)

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
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "bb_percent_b": bb_percent_b,
            "adx14": adx14,
            "obv": obv,
            "stochrsi_k": stochrsi_k,
            "stochrsi_d": stochrsi_d,
        }
    )

    latest = metrics.iloc[-1]
    history_days = len(frame)

    history_warning: str | None = None
    if history_days < VOLUME_WINDOW:
        history_warning = f"历史数据仅 {history_days} 根，部分技术指标暂不可用"
    elif history_days < MA_LONG:
        history_warning = f"历史数据仅 {history_days} 根，MA{MA_LONG} 及部分趋势/风险判断参考性有限"
    elif history_days < HIGH_52W_WINDOW:
        history_warning = f"历史数据仅 {history_days} 根，52 周高点回撤将基于当前样本期计算"

    ma20_value = _resolve_metric(latest["ma20"], pd=pd, history_days=history_days, min_history=MA_SHORT, transform=lambda v: round(v, 2))
    ma50_value = _resolve_metric(latest["ma50"], pd=pd, history_days=history_days, min_history=MA_LONG, transform=lambda v: round(v, 2))
    dev_ma20_value = _resolve_metric(latest["dev_ma20"], pd=pd, history_days=history_days, min_history=MA_SHORT, transform=_pct)
    dev_ma50_value = _resolve_metric(latest["dev_ma50"], pd=pd, history_days=history_days, min_history=MA_LONG, transform=_pct)
    rsi14_value = _resolve_metric(latest["rsi14"], pd=pd, history_days=history_days, min_history=RSI_WINDOW, transform=lambda v: round(v, 2))
    atr_pct_value = _resolve_metric(latest["atr_pct"], pd=pd, history_days=history_days, min_history=ATR_WINDOW, transform=_pct)
    avg_volume_20d_value = _resolve_metric(
        latest["avg_volume_20d"],
        pd=pd,
        history_days=history_days,
        min_history=VOLUME_WINDOW,
        transform=lambda v: round(v),
    )
    rvol_value = _resolve_metric(latest["rvol"], pd=pd, history_days=history_days, min_history=VOLUME_WINDOW, transform=lambda v: round(v, 2))
    drawdown_52w_value = _resolve_metric(latest["drawdown_from_52w_high"], pd=pd, history_days=history_days, transform=_pct)
    max_drawdown_20d_value = _resolve_metric(
        latest["max_drawdown_20d"],
        pd=pd,
        history_days=history_days,
        min_history=DRAWDOWN_WINDOW,
        transform=_pct,
    )
    macd_value = _resolve_metric(latest["macd"], pd=pd, history_days=history_days, transform=lambda v: round(v, 4))
    macd_signal_value = _resolve_metric(latest["macd_signal"], pd=pd, history_days=history_days, transform=lambda v: round(v, 4))
    macd_hist_value = _resolve_metric(latest["macd_hist"], pd=pd, history_days=history_days, transform=lambda v: round(v, 4))
    bb_upper_value = _resolve_metric(latest["bb_upper"], pd=pd, history_days=history_days, min_history=BB_WINDOW, transform=lambda v: round(v, 2))
    bb_middle_value = _resolve_metric(latest["bb_middle"], pd=pd, history_days=history_days, min_history=BB_WINDOW, transform=lambda v: round(v, 2))
    bb_lower_value = _resolve_metric(latest["bb_lower"], pd=pd, history_days=history_days, min_history=BB_WINDOW, transform=lambda v: round(v, 2))
    bb_width_value = _resolve_metric(latest["bb_width"], pd=pd, history_days=history_days, min_history=BB_WINDOW, transform=_pct)
    bb_percent_b_value = _resolve_metric(
        latest["bb_percent_b"],
        pd=pd,
        history_days=history_days,
        min_history=BB_WINDOW,
        transform=lambda v: round(v, 4),
    )
    adx14_value = _resolve_metric(latest["adx14"], pd=pd, history_days=history_days, min_history=ADX_WINDOW, transform=lambda v: round(v, 2))
    obv_value = _resolve_metric(latest["obv"], pd=pd, history_days=history_days, transform=lambda v: round(v))
    stochrsi_k_value = _resolve_metric(
        latest["stochrsi_k"],
        pd=pd,
        history_days=history_days,
        min_history=RSI_WINDOW + STOCH_RSI_WINDOW + STOCH_RSI_SMOOTH_K - 1,
        transform=lambda v: round(v, 2),
    )
    stochrsi_d_value = _resolve_metric(
        latest["stochrsi_d"],
        pd=pd,
        history_days=history_days,
        min_history=RSI_WINDOW + STOCH_RSI_WINDOW + STOCH_RSI_SMOOTH_K + STOCH_RSI_SMOOTH_D - 2,
        transform=lambda v: round(v, 2),
    )

    result = {
        "ticker": ticker.upper(),
        "date": frame.index[-1].date().isoformat(),
        "history_days": history_days,
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
        "drawdown_label": "距 52 周高点回撤" if history_days >= HIGH_52W_WINDOW else "距样本期高点回撤",
        "max_drawdown_20d": max_drawdown_20d_value,
        "macd": macd_value,
        "macd_signal": macd_signal_value,
        "macd_hist": macd_hist_value,
        "bb_upper": bb_upper_value,
        "bb_middle": bb_middle_value,
        "bb_lower": bb_lower_value,
        "bb_width": bb_width_value,
        "bb_percent_b": bb_percent_b_value,
        "adx14": adx14_value,
        "obv": obv_value,
        "stochrsi_k": stochrsi_k_value,
        "stochrsi_d": stochrsi_d_value,
        "history_warning": history_warning,
    }

    result["trend"] = _classify_trend(result)
    result["risk"] = _classify_risk(result)
    return result


# ---------- Output ----------
def _print_quote(item: dict[str, Any]) -> None:
    console.print(f"\n[bold cyan]{item['ticker']}[/bold cyan]  ({item['date']})")
    console.print(f"收盘价: {_format_metric(item.get('close'))}")
    console.print(f"当日涨跌幅: {_format_metric(item.get('day_change_pct'), suffix='%')}")
    console.print(f"MA20: {_format_metric(item.get('ma20'))}")
    console.print(f"MA50: {_format_metric(item.get('ma50'))}")
    console.print(f"距 MA20 偏离: {_format_metric(item.get('dev_ma20'), suffix='%')}")
    console.print(f"距 MA50 偏离: {_format_metric(item.get('dev_ma50'), suffix='%')}")
    console.print(f"RSI14: {_format_metric(item.get('rsi14'))}")
    console.print(f"ATR%: {_format_metric(item.get('atr_pct'), suffix='%')}")
    console.print(f"20 日平均成交量: {_format_metric(item.get('avg_volume_20d'))}")
    console.print(f"RVOL: {_format_metric(item.get('rvol'))}")
    console.print(f"{item.get('drawdown_label', '距 52 周高点回撤')}: {_format_metric(item.get('drawdown_from_52w_high'), suffix='%')}")
    console.print(f"20 日最大回撤: {_format_metric(item.get('max_drawdown_20d'), suffix='%')}")

    console.print(f"MACD: {_format_metric(item.get('macd'), digits=4)}")
    console.print(f"MACD Signal: {_format_metric(item.get('macd_signal'), digits=4)}")
    console.print(f"MACD Hist: {_format_metric(item.get('macd_hist'), digits=4)}")

    console.print(f"Boll 上轨: {_format_metric(item.get('bb_upper'))}")
    console.print(f"Boll 中轨: {_format_metric(item.get('bb_middle'))}")
    console.print(f"Boll 下轨: {_format_metric(item.get('bb_lower'))}")
    console.print(f"Boll 带宽: {_format_metric(item.get('bb_width'), suffix='%')}")
    console.print(f"%B: {_format_metric(item.get('bb_percent_b'), digits=4)}")

    console.print(f"ADX14: {_format_metric(item.get('adx14'))}")

    obv_value = f"{int(item['obv']):,}" if item.get("obv") is not None else "暂无"
    console.print(f"OBV: {obv_value}")

    console.print(f"Stoch RSI %K: {_format_metric(item.get('stochrsi_k'))}")
    console.print(f"Stoch RSI %D: {_format_metric(item.get('stochrsi_d'))}")

    console.print(f"趋势状态: {item.get('trend', '暂无')}")
    console.print(f"风险标签: {item.get('risk', '暂无')}")

    if item.get("history_warning"):
        console.print(f"[yellow]提示: {item['history_warning']}[/yellow]")


# ---------- CLI ----------
@app.command("quote")
def stock_quote(
    ticker: str = typer.Argument(..., help="股票代码"),
    lookback_days: int = typer.Option(QUOTE_LOOKBACK_DAYS, min=30, help="技术分析价格回看天数"),
) -> None:
    """查询单只股票的技术状态。"""
    normalized_ticker = ticker.upper()

    try:
        with console.status(f"[bold green]正在从 FMP 获取 {normalized_ticker} 行情并计算技术指标..."):
            resolved_start = _quote_start_from_lookback(lookback_days)
            price_df = _fetch_fmp_daily_bars(normalized_ticker, resolved_start)
            price_df = price_df.tail(lookback_days)
            item = _analyze_price_frame(normalized_ticker, price_df)
    except Exception as exc:
        console.print(f"[red]stock quote 执行失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_quote(item)


if __name__ == "__main__":
    app()
