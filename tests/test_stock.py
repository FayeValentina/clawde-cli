from datetime import date
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from clawde_cli.commands import stock as stock_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_stock_alert_prints_market_overview(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_analyze_market_overview",
        lambda start: {
            "spy": {"trend": "弱", "risk": "注意", "day_change_pct": -1.43},
            "qqq": {"trend": "弱", "risk": "注意", "day_change_pct": -1.85},
            "vix": {"close": 26.78, "day_change_pct": 11.31},
            "macro_events": [
                {
                    "date": "03-24",
                    "time": "08:30",
                    "event": "CPI",
                    "source": "BLS",
                    "detail": "Consumer Price Index",
                }
            ],
            "earnings_events": [
                {
                    "date": "03-25",
                    "ticker": "NVDA",
                    "time": "时间未提供",
                    "source": "Yahoo Finance",
                }
            ],
            "events": ["03-26 Speeches: Chair remarks | 1:30 p.m."],
            "conclusion": "风险偏好走弱，优先控制仓位和波动敞口。",
        },
    )

    result = runner.invoke(app, ["stock", "alert"])

    assert result.exit_code == 0
    assert "=== 市场总览 ===" in result.stdout
    assert "SPY 状态: 弱 / 注意 / -1.43%" in result.stdout
    assert "03-24 08:30 ET | CPI | BLS | Consumer Price Index" in result.stdout
    assert "03-25 | NVDA | 时间未提供 | Yahoo Finance" in result.stdout
    assert "市场风向结论: 风险偏好走弱，优先控制仓位和波动敞口。" in result.stdout


def test_stock_quote_prints_ticker_snapshot(monkeypatch):
    monkeypatch.setattr(
        stock_cmd,
        "_analyze_ticker",
        lambda ticker, start: {
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
            "max_drawdown_20d": -11.68,
            "trend": "弱",
            "risk": "警戒",
        },
    )

    result = runner.invoke(app, ["stock", "quote", "nvda"])

    assert result.exit_code == 0
    assert "[NVDA] 2026-03-20" in result.stdout
    assert "收盘价: 172.7" in result.stdout
    assert "当日涨跌幅: -3.28%" in result.stdout
    assert "MA20: 183.12" in result.stdout
    assert "20 日平均成交量: 198,907,674" in result.stdout
    assert "风险标签: 警戒" in result.stdout


def test_analyze_ticker_handles_missing_volume(monkeypatch):
    index = pd.date_range("2025-01-01", periods=260, freq="D")
    close = pd.Series(range(100, 360), index=index, dtype=float)
    high = close + 1
    low = close - 1

    class FakeYFData:
        @staticmethod
        def download(ticker, start):
            assert ticker == "^VIX"
            return SimpleNamespace(
                get=lambda field: {
                    "Close": close,
                    "High": high,
                    "Low": low,
                    "Volume": None,
                }[field]
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
            SimpleNamespace(YFData=FakeYFData, MA=FakeMA, RSI=FakeRSI, ATR=FakeATR),
            None,
            None,
        ),
    )

    item = stock_cmd._analyze_ticker("^VIX", "2024-01-01")

    assert item["ticker"] == "^VIX"
    assert item["avg_volume_20d"] is None
    assert item["rvol"] is None


def test_fetch_fed_events_for_cross_month_week(monkeypatch):
    april_html = """
    <div class="cal-nojs__rowTitle">Speeches</div>
    <div class="row">
      <div class="panel-body">
        <div class="row">
          <div>1:30 p.m.</div>
          <div>
            <p>Chair remarks</p>
            <p class="calendar__title">At conference</p>
          </div>
          <div>1</div>
        </div>
      </div>
    </div>
    """
    march_html = """
    <div class="cal-nojs__rowTitle">Testimony</div>
    <div class="row">
      <div class="panel-body">
        <div class="row">
          <div>9:00 a.m.</div>
          <div>
            <p>Vice Chair testimony</p>
          </div>
          <div>31</div>
        </div>
      </div>
    </div>
    """

    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (None, None, None, __import__("bs4").BeautifulSoup))
    monkeypatch.setattr(
        stock_cmd,
        "_fetch_html",
        lambda url, force_curl=False: april_html if "2025-april" in url else march_html,
    )

    events = stock_cmd._fetch_fed_events_for_week(date(2025, 4, 1))

    assert "03-31 Testimony: Vice Chair testimony | 9:00 a.m." in events
    assert "04-01 Speeches: Chair remarks | At conference | 1:30 p.m." in events


def test_normalize_earnings_dates_accepts_pandas_containers(monkeypatch):
    monkeypatch.setattr(stock_cmd, "_load_analysis_dependencies", lambda: (pd, None, None, None))

    series_dates = pd.Series([pd.Timestamp("2025-03-25"), pd.NaT, pd.Timestamp("2025-03-26")])
    index_dates = pd.DatetimeIndex(["2025-03-27", "2025-03-28"])

    assert stock_cmd._normalize_earnings_dates(series_dates) == [
        date(2025, 3, 25),
        date(2025, 3, 26),
    ]
    assert stock_cmd._normalize_earnings_dates(index_dates) == [
        date(2025, 3, 27),
        date(2025, 3, 28),
    ]


def test_load_watchlist_tickers_from_file(tmp_path):
    watchlist = tmp_path / "watchlist.txt"
    watchlist.write_text(
        """
        # my list
        nvda, aapl tsla

        qqq
        nvda
        """,
        encoding="utf-8",
    )

    assert stock_cmd._load_watchlist_tickers(watchlist) == ["NVDA", "AAPL", "TSLA", "QQQ"]
