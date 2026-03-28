from __future__ import annotations

import contextlib
import importlib
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import requests
import typer
from rich.console import Console

app = typer.Typer(help="Stock market quote analysis")
console = Console()

QUOTE_LOOKBACK_DAYS = 450
RSI_WINDOW = 14
ATR_WINDOW = 14
MA_SHORT = 20
MA_LONG = 50
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9
VOLUME_WINDOW = 20
HIGH_52W_WINDOW = 252
DRAWDOWN_WINDOW = 20
ADX_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2
STOCH_RSI_WINDOW = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_SMOOTH_D = 3
CCI_WINDOW = 20
CMF_WINDOW = 20
MFI_WINDOW = 14
REQUEST_TIMEOUT = 20
POLYGON_BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
NEW_YORK_TZ = ZoneInfo("America/New_York")


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


def _format_metric(value: object, *, suffix: str = "", digits: int = 2) -> str:
    """Format numeric metrics while tolerating missing values."""
    if value is None:
        return "暂无"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _ema(series: Any, span: int) -> Any:
    """Compute exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _compute_macd(close: Any) -> tuple[Any, Any, Any]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)
    macd = ema_fast - ema_slow
    signal = _ema(macd, MACD_SIGNAL)
    hist = macd - signal
    return macd, signal, hist


def _compute_bollinger(close: Any, pd: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Compute Bollinger Bands and derived measures."""
    middle = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std(ddof=0)
    upper = middle + BB_STD * std
    lower = middle - BB_STD * std
    width = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    percent_b = percent_b.where((upper - lower) != 0, pd.NA)
    return upper, middle, lower, width, percent_b


def _compute_obv(close: Any, volume: Any, pd: Any) -> Any:
    """Compute On-Balance Volume."""
    direction = pd.Series(0.0, index=close.index)
    direction = direction.mask(close > close.shift(1), 1.0)
    direction = direction.mask(close < close.shift(1), -1.0)
    return (direction * volume.fillna(0)).cumsum()


def _compute_adx(high: Any, low: Any, close: Any, window: int, pd: Any) -> Any:
    """Compute Average Directional Index."""
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
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def _compute_stoch_rsi(close: Any, pd: Any) -> tuple[Any, Any]:
    """Compute Stochastic RSI %K and %D."""
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


def _compute_williams_r(high: Any, low: Any, close: Any, window: int, pd: Any) -> Any:
    """Compute Williams %R."""
    highest_high = high.rolling(window, min_periods=window).max()
    lowest_low = low.rolling(window, min_periods=window).min()
    denominator = highest_high - lowest_low
    values = -100 * (highest_high - close) / denominator
    return values.where(denominator != 0, pd.NA)


def _compute_cci(high: Any, low: Any, close: Any, window: int, pd: Any) -> Any:
    """Compute Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window, min_periods=window).mean()
    mean_dev = typical_price.rolling(window, min_periods=window).apply(
        lambda values: float((abs(values - values.mean())).mean()),
        raw=False,
    )
    denominator = 0.015 * mean_dev
    cci = (typical_price - sma_tp) / denominator
    return cci.where(denominator != 0, pd.NA)


def _compute_cmf(high: Any, low: Any, close: Any, volume: Any, window: int, pd: Any) -> Any:
    """Compute Chaikin Money Flow."""
    denominator = high - low
    multiplier = ((close - low) - (high - close)) / denominator
    multiplier = multiplier.where(denominator != 0, 0.0)
    money_flow_volume = multiplier * volume.fillna(0)
    volume_sum = volume.rolling(window, min_periods=window).sum()
    cmf = money_flow_volume.rolling(window, min_periods=window).sum() / volume_sum
    return cmf.where(volume_sum != 0, pd.NA)


def _compute_mfi(high: Any, low: Any, close: Any, volume: Any, window: int, pd: Any) -> Any:
    """Compute Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume.fillna(0)
    direction = typical_price.diff()
    positive_flow = raw_money_flow.where(direction > 0, 0.0)
    negative_flow = raw_money_flow.where(direction < 0, 0.0)
    positive_sum = positive_flow.rolling(window, min_periods=window).sum()
    negative_sum = negative_flow.rolling(window, min_periods=window).sum().abs()
    money_ratio = positive_sum / negative_sum
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.where(negative_sum != 0, pd.NA)


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
            "vw": "VWAP",
            "n": "Transactions",
        }
    )
    frame.index = (
        pd.to_datetime(frame["t"], unit="ms", utc=True)
        .dt.tz_convert(NEW_YORK_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if "VWAP" in frame.columns:
        columns.append("VWAP")
    if "Transactions" in frame.columns:
        columns.append("Transactions")
    frame = frame[columns].sort_index()
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
    vwap = (
        price_df["VWAP"].astype(float)
        if "VWAP" in price_df.columns
        else pd.Series(index=price_df.index, dtype=float)
    )
    transactions = (
        price_df["Transactions"].astype(float)
        if "Transactions" in price_df.columns
        else pd.Series(index=price_df.index, dtype=float)
    )
    history_days = len(price_df)

    ma20 = vbt.MA.run(close, MA_SHORT).ma
    ma50 = vbt.MA.run(close, MA_LONG).ma
    rsi14 = vbt.RSI.run(close, window=RSI_WINDOW).rsi
    atr14 = vbt.ATR.run(high, low, close, window=ATR_WINDOW).atr
    ema12 = _ema(close, EMA_FAST)
    ema26 = _ema(close, EMA_SLOW)
    macd, macd_signal, macd_hist = _compute_macd(close)
    bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b = _compute_bollinger(close, pd)
    adx14 = _compute_adx(high, low, close, ADX_WINDOW, pd)
    obv = _compute_obv(close, volume, pd)
    stochrsi_k, stochrsi_d = _compute_stoch_rsi(close, pd)
    williams_r14 = _compute_williams_r(high, low, close, RSI_WINDOW, pd)
    cci20 = _compute_cci(high, low, close, CCI_WINDOW, pd)
    cmf20 = _compute_cmf(high, low, close, volume, CMF_WINDOW, pd)
    mfi14 = _compute_mfi(high, low, close, volume, MFI_WINDOW, pd)
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
            "ema12": ema12,
            "ema26": ema26,
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
            "williams_r14": williams_r14,
            "cci20": cci20,
            "cmf20": cmf20,
            "mfi14": mfi14,
            "vwap": vwap,
            "transactions": transactions,
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
    ema12_value = None if history_days < EMA_FAST or pd.isna(latest["ema12"]) else round(float(latest["ema12"]), 2)
    ema26_value = None if history_days < EMA_SLOW or pd.isna(latest["ema26"]) else round(float(latest["ema26"]), 2)
    dev_ma20_value = None if ma20_value is None or pd.isna(latest["dev_ma20"]) else _pct(float(latest["dev_ma20"]))
    dev_ma50_value = None if ma50_value is None or pd.isna(latest["dev_ma50"]) else _pct(float(latest["dev_ma50"]))
    rsi14_value = None if history_days < RSI_WINDOW or pd.isna(latest["rsi14"]) else round(float(latest["rsi14"]), 2)
    atr_pct_value = None if history_days < ATR_WINDOW or pd.isna(latest["atr_pct"]) else _pct(float(latest["atr_pct"]))
    macd_value = None if history_days < EMA_SLOW or pd.isna(latest["macd"]) else round(float(latest["macd"]), 4)
    macd_signal_value = None if history_days < EMA_SLOW or pd.isna(latest["macd_signal"]) else round(float(latest["macd_signal"]), 4)
    macd_hist_value = None if history_days < EMA_SLOW or pd.isna(latest["macd_hist"]) else round(float(latest["macd_hist"]), 4)
    avg_volume_20d_value = None
    rvol_value = None
    if history_days >= VOLUME_WINDOW and pd.notna(latest["avg_volume_20d"]):
        avg_volume_20d_value = round(float(latest["avg_volume_20d"]))
    if history_days >= VOLUME_WINDOW and pd.notna(latest["rvol"]):
        rvol_value = round(float(latest["rvol"]), 2)
    bb_upper_value = None if history_days < BB_WINDOW or pd.isna(latest["bb_upper"]) else round(float(latest["bb_upper"]), 2)
    bb_middle_value = None if history_days < BB_WINDOW or pd.isna(latest["bb_middle"]) else round(float(latest["bb_middle"]), 2)
    bb_lower_value = None if history_days < BB_WINDOW or pd.isna(latest["bb_lower"]) else round(float(latest["bb_lower"]), 2)
    bb_width_value = None if history_days < BB_WINDOW or pd.isna(latest["bb_width"]) else _pct(float(latest["bb_width"]))
    bb_percent_b_value = None if history_days < BB_WINDOW or pd.isna(latest["bb_percent_b"]) else round(float(latest["bb_percent_b"]), 4)
    drawdown_52w_value = None if pd.isna(latest["drawdown_from_52w_high"]) else _pct(float(latest["drawdown_from_52w_high"]))
    max_drawdown_20d_value = None if history_days < DRAWDOWN_WINDOW or pd.isna(latest["max_drawdown_20d"]) else _pct(float(latest["max_drawdown_20d"]))
    adx14_value = None if history_days < ADX_WINDOW or pd.isna(latest["adx14"]) else round(float(latest["adx14"]), 2)
    obv_value = None if pd.isna(latest["obv"]) else round(float(latest["obv"]))
    stochrsi_k_value = None if history_days < (RSI_WINDOW + STOCH_RSI_WINDOW + STOCH_RSI_SMOOTH_K - 1) or pd.isna(latest["stochrsi_k"]) else round(float(latest["stochrsi_k"]), 2)
    stochrsi_d_value = None if history_days < (RSI_WINDOW + STOCH_RSI_WINDOW + STOCH_RSI_SMOOTH_K + STOCH_RSI_SMOOTH_D - 2) or pd.isna(latest["stochrsi_d"]) else round(float(latest["stochrsi_d"]), 2)
    williams_r14_value = None if history_days < RSI_WINDOW or pd.isna(latest["williams_r14"]) else round(float(latest["williams_r14"]), 2)
    cci20_value = None if history_days < CCI_WINDOW or pd.isna(latest["cci20"]) else round(float(latest["cci20"]), 2)
    cmf20_value = None if history_days < CMF_WINDOW or pd.isna(latest["cmf20"]) else round(float(latest["cmf20"]), 4)
    mfi14_value = None if history_days < MFI_WINDOW or pd.isna(latest["mfi14"]) else round(float(latest["mfi14"]), 2)
    vwap_value = None if pd.isna(latest["vwap"]) else round(float(latest["vwap"]), 2)
    transactions_value = None if pd.isna(latest["transactions"]) else round(float(latest["transactions"]))

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
        "ema12": ema12_value,
        "ema26": ema26_value,
        "dev_ma20": dev_ma20_value,
        "dev_ma50": dev_ma50_value,
        "rsi14": rsi14_value,
        "atr_pct": atr_pct_value,
        "avg_volume_20d": avg_volume_20d_value,
        "rvol": rvol_value,
        "drawdown_from_52w_high": drawdown_52w_value,
        "drawdown_label": "距 52 周最高价回撤" if history_days >= HIGH_52W_WINDOW else "距样本期最高价回撤",
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
        "williams_r14": williams_r14_value,
        "cci20": cci20_value,
        "cmf20": cmf20_value,
        "mfi14": mfi14_value,
        "vwap": vwap_value,
        "transactions": transactions_value,
        "trend": trend,
        "risk": risk,
    }


def _print_quote(item: dict[str, object]) -> None:
    """Render the quote snapshot."""
    console.print(f"[{item['ticker']}] {item['date']}")
    console.print(f"收盘价: {_format_metric(item['close'])}")
    console.print(f"当日涨跌幅: {_format_metric(item['day_change_pct'], suffix='%')}")
    console.print(f"MA20: {_format_metric(item.get('ma20'))}")
    console.print(f"MA50: {_format_metric(item.get('ma50'))}")
    console.print(f"EMA12: {_format_metric(item.get('ema12'))}")
    console.print(f"EMA26: {_format_metric(item.get('ema26'))}")
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
    console.print(f"Williams %R: {_format_metric(item.get('williams_r14'))}")
    console.print(f"CCI20: {_format_metric(item.get('cci20'))}")
    console.print(f"CMF20: {_format_metric(item.get('cmf20'), digits=4)}")
    console.print(f"MFI14: {_format_metric(item.get('mfi14'))}")
    console.print(f"VWAP: {_format_metric(item.get('vwap'))}")
    transactions_value = (
        f"{int(item['transactions']):,}" if item.get("transactions") is not None else "暂无"
    )
    console.print(f"成交笔数: {transactions_value}")
    console.print(f"趋势状态: {item.get('trend') or '暂无'}")
    console.print(f"风险标签: {item.get('risk') or '暂无'}")
    if item.get("history_warning"):
        console.print(f"提示: {item['history_warning']}")


@app.command("quote")
def stock_quote(
    ticker: str = typer.Argument(..., help="股票代码"),
    lookback_days: int = typer.Option(QUOTE_LOOKBACK_DAYS, min=30, help="技术分析价格回看天数"),
) -> None:
    """查询单只股票的技术状态。"""
    normalized_ticker = ticker.upper()

    try:
        with console.status(f"[bold green]正在获取 {normalized_ticker} 行情并计算技术指标..."):
            resolved_start = _quote_start_from_lookback(lookback_days)
            price_df = _fetch_polygon_daily_bars(normalized_ticker, resolved_start)
            item = _analyze_price_frame(normalized_ticker, price_df)
    except Exception as exc:
        console.print(f"[red]stock quote 执行失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    _print_quote(item)
