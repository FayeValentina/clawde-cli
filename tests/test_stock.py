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
        "_fetch_polygon_daily_bars",
        lambda ticker, start: captured.update({"ticker": ticker, "start": start}) or pd.DataFrame(
            {
                "Open": [170.0, 171.0],
                "High": [173.0, 174.0],
                "Low": [169.0, 170.5],
                "Close": [172.0, 172.7],
                "Volume": [1000.0, 1200.0],
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
            "close": 172.7,
            "day_change_pct": -3.28,
            "ma20": 183.12,
            "ma50": 184.59,
            "ema12": 179.42,
            "ema26": 181.77,
            "dev_ma20": -5.69,
            "dev_ma50": -6.44,
            "rsi14": 37.39,
            "atr_pct": 3.2,
            "avg_volume_20d": 198907674,
            "rvol": 1.05,
            "drawdown_from_52w_high": -16.58,
            "drawdown_label": "距 52 周最高价回撤",
            "max_drawdown_20d": -11.68,
            "macd": -1.2456,
            "macd_signal": -0.8821,
            "macd_hist": -0.3635,
            "bb_upper": 188.22,
            "bb_middle": 183.12,
            "bb_lower": 178.02,
            "bb_width": 5.57,
            "bb_percent_b": 0.2734,
            "adx14": 24.18,
            "obv": 1234567890,
            "stochrsi_k": 12.34,
            "stochrsi_d": 18.76,
            "williams_r14": -82.45,
            "cci20": -109.32,
            "cmf20": -0.1234,
            "mfi14": 31.28,
            "vwap": 173.21,
            "transactions": 456789,
            "trend": "弱",
            "risk": "警戒",
            "history_warning": None,
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda"])

    assert result.exit_code == 0
    assert captured["ticker"] == "NVDA"
    assert captured["start"] == stock_cmd._quote_start_from_lookback(stock_cmd.QUOTE_LOOKBACK_DAYS)
    assert "[NVDA] 2026-03-20" in result.stdout
    assert "收盘价: 172.70" in result.stdout
    assert "ATR%: 3.20%" in result.stdout
    assert "MACD: -1.2456" in result.stdout
    assert "ADX14: 24.18" in result.stdout
    assert "OBV: 1,234,567,890" in result.stdout
    assert "成交笔数: 456,789" in result.stdout
    assert "20 日平均成交量: 198,907,674" in result.stdout
    assert "风险标签: 警戒" in result.stdout


def test_stock_quote_cli_uses_lookback_days(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stock_cmd,
        "_fetch_polygon_daily_bars",
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
            "ema12": None,
            "ema26": None,
            "dev_ma20": None,
            "dev_ma50": None,
            "rsi14": None,
            "atr_pct": None,
            "avg_volume_20d": None,
            "rvol": None,
            "drawdown_from_52w_high": 0.0,
            "drawdown_label": "距样本期最高价回撤",
            "max_drawdown_20d": None,
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_width": None,
            "bb_percent_b": None,
            "adx14": None,
            "obv": None,
            "stochrsi_k": None,
            "stochrsi_d": None,
            "williams_r14": None,
            "cci20": None,
            "cmf20": None,
            "mfi14": None,
            "vwap": None,
            "transactions": None,
            "trend": "暂无",
            "risk": "暂无",
            "history_warning": "历史短于 20 个交易日，部分技术指标暂不可用。",
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
    assert item["avg_volume_20d"] is None
    assert item["rvol"] is None
    assert item["drawdown_label"] == "距 52 周最高价回撤"


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
    assert item["trend"] in {"偏强", "中性", "偏弱"}
    assert item["history_warning"] is not None
    assert item["drawdown_label"] == "距样本期最高价回撤"


def test_fetch_polygon_daily_bars_normalizes_response(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "results": [
                    {"t": 1710806400000, "o": 100.0, "h": 110.0, "l": 99.0, "c": 105.0, "v": 1000},
                    {"t": 1710892800000, "o": 106.0, "h": 111.0, "l": 104.0, "c": 109.0, "v": 1200, "vw": 107.5, "n": 321},
                ]
            }

    monkeypatch.setattr(stock_cmd, "requests", SimpleNamespace(get=lambda *args, **kwargs: FakeResponse()))
    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (pd, None))

    frame = stock_cmd._fetch_polygon_daily_bars("NVDA", "2024-01-01")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume", "VWAP", "Transactions"]
    assert frame.iloc[0]["Close"] == 105.0
    assert frame.index.is_monotonic_increasing


def test_fetch_polygon_daily_bars_preserves_optional_polygon_fields(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "results": [
                    {"t": 1710892800000, "o": 106.0, "h": 111.0, "l": 104.0, "c": 109.0, "v": 1200, "vw": 107.5, "n": 321},
                ]
            }

    monkeypatch.setattr(stock_cmd, "requests", SimpleNamespace(get=lambda *args, **kwargs: FakeResponse()))
    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (pd, None))

    frame = stock_cmd._fetch_polygon_daily_bars("NVDA", "2024-01-01")

    assert "VWAP" in frame.columns
    assert "Transactions" in frame.columns
    assert frame.iloc[0]["VWAP"] == 107.5
    assert frame.iloc[0]["Transactions"] == 321


def test_stock_cli_rejects_invalid_numeric_options():
    quote_result = runner.invoke(app, ["stock", "quote", "nvda", "--lookback-days", "0"])

    assert quote_result.exit_code != 0


def test_classify_trend_returns_strong_state():
    latest = {
        "close": 110.0,
        "ma20": 100.0,
        "ma50": 95.0,
        "rsi14": 58.0,
    }
    prev = {
        "ma20": 99.0,
        "ma50": 94.0,
    }

    assert stock_cmd._classify_trend(latest, prev, pd) == "强"


def test_classify_risk_prefers_alert_label():
    latest = {
        "close": 80.0,
        "ma20": 100.0,
        "ma50": 105.0,
        "rsi14": 20.0,
        "atr_pct": 0.07,
        "rvol": 2.3,
        "drawdown_from_52w_high": -0.31,
        "max_drawdown_20d": -0.13,
    }

    assert stock_cmd._classify_risk(latest, "弱", pd) == "警戒"
