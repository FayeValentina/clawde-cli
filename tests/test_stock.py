from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from clawde_cli.commands import stock as stock_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_stock_quote_prints_ticker_snapshot(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stock_cmd,
        "_fetch_fmp_daily_bars",
        lambda ticker, start: captured.update({"ticker": ticker, "start": start}) or pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.0],
                "Low": [1.0],
                "Close": [1.0],
                "Volume": [1.0],
            },
            index=pd.date_range("2026-03-20", periods=1),
        ),
    )
    monkeypatch.setattr(
        stock_cmd,
        "_analyze_price_frame",
        lambda ticker, price_df: {
            "ticker": ticker,
            "date": "2026-03-20",
            "close": 172.7,
            "day_change_pct": -3.28,
            "ma20": 183.12,
            "ma50": 184.59,
            "dev_ma20": -5.69,
            "dev_ma50": -6.44,
            "rsi14": 37.39,
            "atr_pct": 3.2,
            "avg_volume_20d": 198907674,
            "rvol": 1.05,
            "drawdown_from_52w_high": -16.58,
            "drawdown_label": "距 52 周高点回撤",
            "max_drawdown_20d": -11.68,
            "macd": -1.2345,
            "macd_signal": -0.9876,
            "macd_hist": -0.2469,
            "bb_upper": 190.12,
            "bb_middle": 182.45,
            "bb_lower": 174.78,
            "bb_width": 8.41,
            "bb_percent_b": 0.2213,
            "adx14": 27.45,
            "obv": 123456789,
            "stochrsi_k": 12.34,
            "stochrsi_d": 23.45,
            "trend": "偏弱",
            "risk": "中等偏高",
            "history_warning": None,
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda"])

    assert result.exit_code == 0
    assert captured["ticker"] == "NVDA"
    assert captured["start"] == stock_cmd._quote_start_from_lookback(stock_cmd.QUOTE_LOOKBACK_DAYS)
    assert "NVDA" in result.stdout
    assert "收盘价: 172.70" in result.stdout
    assert "MACD: -1.2345" in result.stdout
    assert "OBV: 123,456,789" in result.stdout
    assert "风险标签: 中等偏高" in result.stdout


def test_stock_quote_cli_uses_lookback_days(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stock_cmd,
        "_fetch_fmp_daily_bars",
        lambda ticker, start: captured.update({"ticker": ticker, "start": start}) or pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.0, 2.0],
                "Low": [1.0, 2.0],
                "Close": [1.0, 2.0],
                "Volume": [1.0, 2.0],
            },
            index=pd.date_range("2026-03-19", periods=2),
        ),
    )
    monkeypatch.setattr(
        stock_cmd,
        "_analyze_price_frame",
        lambda ticker, price_df: {
            "ticker": ticker,
            "date": "2026-03-20",
            "close": 2.0,
            "day_change_pct": 100.0,
            "ma20": None,
            "ma50": None,
            "dev_ma20": None,
            "dev_ma50": None,
            "rsi14": None,
            "atr_pct": None,
            "avg_volume_20d": None,
            "rvol": None,
            "drawdown_from_52w_high": 0.0,
            "drawdown_label": "距样本期高点回撤",
            "max_drawdown_20d": None,
            "macd": 0.1,
            "macd_signal": 0.05,
            "macd_hist": 0.05,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_width": None,
            "bb_percent_b": None,
            "adx14": None,
            "obv": 3,
            "stochrsi_k": None,
            "stochrsi_d": None,
            "trend": "样本不足",
            "risk": "正常",
            "history_warning": "历史数据仅 2 根，部分技术指标暂不可用",
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda", "--lookback-days", "300"])

    assert result.exit_code == 0
    assert captured["ticker"] == "NVDA"
    assert captured["start"] == stock_cmd._quote_start_from_lookback(300)


def test_analyze_price_frame_handles_missing_volume(monkeypatch):
    index = pd.date_range("2025-01-01", periods=260, freq="D")
    close = pd.Series(range(100, 360), index=index, dtype=float)
    price_df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 0.0,
        },
        index=index,
    )

    class FakeIndicator:
        def __init__(self, values):
            self.ma = values
            self.rsi = values
            self.atr = values

    class FakeMA:
        @staticmethod
        def run(series, window):
            return FakeIndicator(series.rolling(window).mean())

    class FakeRSI:
        @staticmethod
        def run(series, window):
            return FakeIndicator(pd.Series(55.0, index=series.index))

    class FakeATR:
        @staticmethod
        def run(high, low, close, window):
            return FakeIndicator(pd.Series(2.0, index=close.index))

    monkeypatch.setattr(
        stock_cmd,
        "_load_analysis_dependencies",
        lambda: (
            pd,
            SimpleNamespace(MA=FakeMA, RSI=FakeRSI, ATR=FakeATR),
        ),
    )

    item = stock_cmd._analyze_price_frame("^VIX", price_df)

    assert item["ticker"] == "^VIX"
    assert item["avg_volume_20d"] == 0
    assert item["rvol"] is None
    assert item["drawdown_label"] == "距 52 周高点回撤"


def test_analyze_price_frame_degrades_for_short_history(monkeypatch):
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    close = pd.Series(range(100, 130), index=index, dtype=float)
    price_df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.5,
            "Low": close - 1,
            "Close": close,
            "Volume": pd.Series(1000.0, index=index),
        },
        index=index,
    )

    class FakeIndicator:
        def __init__(self, values):
            self.ma = values
            self.rsi = values
            self.atr = values

    class FakeMA:
        @staticmethod
        def run(series, window):
            return FakeIndicator(series.rolling(window).mean())

    class FakeRSI:
        @staticmethod
        def run(series, window):
            return FakeIndicator(pd.Series(58.0, index=series.index))

    class FakeATR:
        @staticmethod
        def run(high, low, close, window):
            return FakeIndicator(pd.Series(2.5, index=close.index))

    monkeypatch.setattr(
        stock_cmd,
        "_load_analysis_dependencies",
        lambda: (
            pd,
            SimpleNamespace(MA=FakeMA, RSI=FakeRSI, ATR=FakeATR),
        ),
    )

    item = stock_cmd._analyze_price_frame("IPO", price_df)

    assert item["ma20"] is not None
    assert item["ma50"] is None
    assert item["trend"] == "样本不足"
    assert item["history_warning"] is not None
    assert item["drawdown_label"] == "距样本期高点回撤"


def test_fetch_fmp_daily_bars_normalizes_response(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return [
                {"date": "2024-03-19", "open": 100.0, "high": 110.0, "low": 99.0, "close": 105.0, "volume": 1000},
                {"date": "2024-03-20", "open": 106.0, "high": 111.0, "low": 104.0, "close": 109.0, "volume": 1200},
            ]

    monkeypatch.setattr(stock_cmd, "requests", SimpleNamespace(get=lambda *args, **kwargs: FakeResponse()))
    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (pd, None))

    frame = stock_cmd._fetch_fmp_daily_bars("NVDA", "2024-01-01")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.iloc[0]["Close"] == 105.0
    assert frame.index.is_monotonic_increasing


def test_stock_cli_rejects_invalid_numeric_options():
    quote_result = runner.invoke(app, ["stock", "quote", "nvda", "--lookback-days", "0"])

    assert quote_result.exit_code != 0


def test_classify_trend_returns_strong_uptrend():
    latest = {
        "close": 110.0,
        "ma20": 100.0,
        "ma50": 95.0,
        "rsi14": 58.0,
        "macd": 1.2,
        "macd_signal": 0.9,
        "adx14": 27.0,
    }

    assert stock_cmd._classify_trend(latest) == "强势上升"


def test_classify_risk_prefers_high_volatility_label():
    latest = {
        "atr_pct": 6.5,
        "rvol": 2.3,
        "drawdown_from_52w_high": -31.0,
        "max_drawdown_20d": -13.0,
        "bb_width": 13.0,
        "stochrsi_k": 95.0,
    }

    assert stock_cmd._classify_risk(latest) == "高波动"
