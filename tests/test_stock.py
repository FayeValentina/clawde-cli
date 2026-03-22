from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from clawde_cli.commands import stock as stock_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_stock_alert_prints_market_news(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_build_market_news_overview",
        lambda display_limit, hours, raw_limit=stock_cmd.DEFAULT_ALERT_RAW_LIMIT: {
            "matched_news_items": [
                {
                    "source": "Reuters",
                    "title": "Wall Street slips as yields rise",
                    "summary": "Treasury yields climbed while investors reassessed rate-cut timing.",
                    "published_at": "2026-03-23T13:00:00Z",
                    "url": "https://example.com/reuters-story",
                },
                {
                    "source": "Bloomberg",
                    "title": "AI trade lifts chip stocks",
                    "summary": "Semiconductor shares outperformed on renewed AI spending optimism.",
                    "published_at": "2026-03-23T14:30:00Z",
                    "url": "https://example.com/bloomberg-story",
                },
            ],
            "conclusion": "新闻流多空交织，市场更像事件驱动震荡，交易上宜保持选择性。",
            "no_match_message": None,
        },
    )

    result = runner.invoke(app, ["stock", "alert"])

    assert result.exit_code == 0
    assert "=== 市场新闻摘要 ===" in result.stdout
    assert "- [Reuters] Wall Street slips as yields rise" in result.stdout
    assert "AI trade lifts chip stocks" in result.stdout
    assert "发布时间:" in result.stdout
    assert "链接: https://example.com/reuters-story" in result.stdout
    assert "市场风向结论: 新闻流多空交织，市场更像事件驱动震荡，交易上宜保持选择性。" in result.stdout


def test_stock_alert_handles_no_high_relevance_matches(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_build_market_news_overview",
        lambda display_limit, hours, raw_limit=stock_cmd.DEFAULT_ALERT_RAW_LIMIT: {
            "matched_news_items": [],
            "conclusion": None,
            "no_match_message": "最近 72 小时未找到高相关度的市场主线新闻，可尝试扩大时间窗口或放宽相关性条件。",
        },
    )

    result = runner.invoke(app, ["stock", "alert"])

    assert result.exit_code == 0
    assert "最近 72 小时暂无高相关度市场主题新闻" in result.stdout
    assert "最近 72 小时未找到高相关度的市场主线新闻" in result.stdout
    assert "=== 其他参考新闻 ===" not in result.stdout
    assert "市场风向结论:" not in result.stdout


def test_stock_quote_prints_news_and_ticker_snapshot(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stock_cmd,
        "_fetch_ticker_news",
        lambda ticker, limit, hours: [
            {
                "source": "Reuters",
                "title": "Nvidia gains on AI demand",
                "summary": "Investors focused on data-center demand.",
                "published_at": "2026-03-23T13:00:00Z",
                "url": "https://example.com/nvda-story",
            }
        ],
    )
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_polygon_daily_bars",
        lambda ticker, start: captured.setdefault("start", start) or object(),
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
            "avg_volume_20d": None,
            "rvol": 1.05,
            "drawdown_from_52w_high": -16.58,
            "drawdown_label": "距 52 周最高价回撤",
            "max_drawdown_20d": -11.68,
            "trend": "弱",
            "risk": "警戒",
            "history_warning": None,
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda"])

    assert result.exit_code == 0
    assert captured["start"] == stock_cmd._quote_start_from_lookback()
    assert "=== NVDA 相关新闻 ===" in result.stdout
    assert "Nvidia gains on AI demand" in result.stdout
    assert "发布时间:" in result.stdout
    assert "[NVDA] 2026-03-20" in result.stdout
    assert "收盘价: 172.70" in result.stdout
    assert "20 日平均成交量: 暂无" in result.stdout
    assert "风险标签: 警戒" in result.stdout


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
                    {"t": 1710892800000, "o": 106.0, "h": 111.0, "l": 104.0, "c": 109.0, "v": 1200},
                ]
            }

    monkeypatch.setattr(stock_cmd, "requests", SimpleNamespace(get=lambda *args, **kwargs: FakeResponse()))
    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (pd, None))

    frame = stock_cmd._fetch_polygon_daily_bars("NVDA", "2024-01-01")

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.iloc[0]["Close"] == 105.0
    assert frame.index.is_monotonic_increasing


def test_build_market_news_overview_uses_only_matched_items_for_conclusion(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_marketaux_news",
        lambda raw_limit, hours: [
            {"title": "Fed turns hawkish", "summary": "Markets brace for tighter policy."},
            {"title": "Company opens new store", "summary": "Retail expansion continues."},
        ],
    )
    monkeypatch.setattr(
        stock_cmd,
        "_summarize_market_news",
        lambda items: f"used:{len(items)}:{items[0]['title']}" if items else None,
    )

    overview = stock_cmd._build_market_news_overview(display_limit=5, hours=72, raw_limit=75)

    assert overview["conclusion"] == "used:1:Fed turns hawkish"
    assert len(overview["matched_news_items"]) == 1


def test_stock_alert_cli_uses_stable_surface(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stock_cmd,
        "_build_market_news_overview",
        lambda display_limit, hours, raw_limit=stock_cmd.DEFAULT_ALERT_RAW_LIMIT: captured.update(
            {"display_limit": display_limit, "hours": hours, "raw_limit": raw_limit}
        ) or {
            "matched_news_items": [],
            "conclusion": None,
            "no_match_message": None,
        },
    )

    result = runner.invoke(app, ["stock", "alert", "--hours", "96", "--limit", "4"])

    assert result.exit_code == 0
    assert captured == {
        "display_limit": 4,
        "hours": 96,
        "raw_limit": stock_cmd.DEFAULT_ALERT_RAW_LIMIT,
    }


def test_stock_quote_continues_when_news_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_ticker_news",
        lambda ticker, limit, hours: (_ for _ in ()).throw(RuntimeError("news unavailable")),
    )
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_polygon_daily_bars",
        lambda ticker, start: object(),
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
            "drawdown_label": "距 52 周最高价回撤",
            "max_drawdown_20d": -11.68,
            "trend": "弱",
            "risk": "警戒",
            "history_warning": None,
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda"])

    assert result.exit_code == 0
    assert "提示: 相关新闻获取失败：" in result.stdout
    assert "[NVDA] 2026-03-20" in result.stdout


def test_stock_quote_cli_uses_lookback_days(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(stock_cmd, "_fetch_ticker_news", lambda ticker, limit, hours: [])
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_polygon_daily_bars",
        lambda ticker, start: captured.update({"ticker": ticker, "start": start}) or object(),
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
            "drawdown_label": "距 52 周最高价回撤",
            "max_drawdown_20d": -11.68,
            "trend": "弱",
            "risk": "警戒",
            "history_warning": None,
        },
    )

    result = runner.invoke(
        app,
        ["stock", "quote", "nvda", "--lookback-days", "300", "--news-hours", "48", "--news-limit", "2"],
    )

    assert result.exit_code == 0
    assert captured["ticker"] == "NVDA"
    assert captured["start"] == stock_cmd._quote_start_from_lookback(300)


def test_matches_ticker_news_uses_strict_matching_for_stocks():
    assert stock_cmd._matches_ticker_news("NVDA", "nvidia guides higher on ai demand")
    assert not stock_cmd._matches_ticker_news("AMD", "broad market demand improves")


def test_matches_ticker_news_allows_thematic_etf_aliases():
    assert stock_cmd._matches_ticker_news("QQQ", "big tech leads the rally after rate relief")
    assert stock_cmd._matches_ticker_news("USO", "crude oil prices climb after supply disruption")


def test_stock_cli_rejects_invalid_numeric_options():
    alert_result = runner.invoke(app, ["stock", "alert", "--limit", "0"])
    quote_result = runner.invoke(app, ["stock", "quote", "nvda", "--lookback-days", "0"])

    assert alert_result.exit_code != 0
    assert quote_result.exit_code != 0


def test_summarize_market_news_prefers_cautious_bias():
    news_items = [
        {"title": "Stocks fall in selloff", "summary": "Risk-off mood returns as sticky inflation stays elevated."},
        {"title": "Fed seen as hawkish", "summary": "Markets price slower rate cuts and broader decline."},
    ]

    summary = stock_cmd._summarize_market_news(news_items)

    assert "偏谨慎" in summary or "承压" in summary


def test_summarize_market_news_does_not_penalize_cooling_inflation():
    news_items = [
        {"title": "Cooling inflation helps stocks rally", "summary": "Investors turn bullish as price pressures ease."},
    ]

    summary = stock_cmd._summarize_market_news(news_items)

    assert summary is not None
    assert "偏积极" in summary or "改善" in summary
