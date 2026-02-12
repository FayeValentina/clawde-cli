"""Weather query commands."""

from typing import Any

import requests
import typer
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from urllib3.util.retry import Retry

app = typer.Typer(help="Weather queries via Open-Meteo")
console = Console()

# Open-Meteo 的地理编码与天气预报 API
GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
# 单次请求超时时间（秒）
HTTP_TIMEOUT_SECONDS = 20

# Open-Meteo weather_code 到人类可读文本的映射
WEATHER_CODE_MAP = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm + slight hail",
    99: "Thunderstorm + heavy hail",
}


def _build_session() -> requests.Session:
    # 针对临时网络波动配置重试，减少偶发失败
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _build_session()


def _weather_label(code: Any) -> str:
    # 将天气编码转换为文本；未知编码保留原值便于排查
    if isinstance(code, int):
        return WEATHER_CODE_MAP.get(code, f"Unknown ({code})")
    return "Unknown"


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    # 统一 GET + JSON 解析，异常由调用方处理
    response = SESSION.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _resolve_location(location: str) -> dict[str, Any]:
    # 通过地理编码接口把城市名解析为经纬度（只取最匹配的一条）
    data = _get_json(
        GEOCODE_API,
        {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Location not found: {location}")
    return results[0]


def _get_weather_data(latitude: float, longitude: float, days: int) -> dict[str, Any]:
    # 一次请求同时拿 current + daily，减少网络往返
    return _get_json(
        FORECAST_API,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "forecast_days": days,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                ]
            ),
        },
    )


@app.command()
def now(
    location: str = typer.Argument("Tokyo", help="Location to check"),
):
    """Get current weather."""
    try:
        # 先解析地点，再按经纬度取天气
        loc = _resolve_location(location)
        data = _get_weather_data(loc["latitude"], loc["longitude"], days=1)
        current = data.get("current", {})
        city = loc.get("name", location)
        country = loc.get("country", "")
        place = f"{city}, {country}".strip(", ")
        weather_text = _weather_label(current.get("weather_code"))
        temp = current.get("temperature_2m", "?")
        feels = current.get("apparent_temperature", "?")
        humidity = current.get("relative_humidity_2m", "?")
        wind = current.get("wind_speed_10m", "?")

        console.print(
            Panel(
                (
                    f"[bold]{place}[/bold]\n"
                    f"{weather_text}, {temp}°C (feels {feels}°C)\n"
                    f"Humidity: {humidity}% | Wind: {wind} km/h"
                ),
                title="🌤️  Current Weather",
                border_style="cyan",
            )
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except requests.RequestException as e:
        console.print(f"[red]Error fetching weather: {e}[/red]")
        raise typer.Exit(code=1) from e
    except OSError as e:
        console.print(f"[red]Output error: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def full(
    location: str = typer.Argument("Tokyo", help="Location to check"),
):
    """Get full weather report."""
    try:
        # full 模式固定拉取 3 天数据：当前概览 + 3 日表格
        loc = _resolve_location(location)
        data = _get_weather_data(loc["latitude"], loc["longitude"], days=3)
        current = data.get("current", {})
        daily = data.get("daily", {})

        city = loc.get("name", location)
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")
        place = f"{city}, {admin1}, {country}".strip(", ")

        summary = (
            f"[bold]{place}[/bold]\n"
            f"Lat/Lon: {loc.get('latitude')}, {loc.get('longitude')}\n"
            f"Now: {_weather_label(current.get('weather_code'))}, "
            f"{current.get('temperature_2m', '?')}°C "
            f"(feels {current.get('apparent_temperature', '?')}°C), "
            f"Humidity {current.get('relative_humidity_2m', '?')}%, "
            f"Wind {current.get('wind_speed_10m', '?')} km/h"
        )
        console.print(
            Panel(summary, title="🌤️  Full Weather Report", border_style="cyan")
        )

        dates = daily.get("time", [])
        weather_codes = daily.get("weather_code", [])
        t_min = daily.get("temperature_2m_min", [])
        t_max = daily.get("temperature_2m_max", [])
        rain_prob = daily.get("precipitation_probability_max", [])

        table = Table(
            title="📅 3-Day Forecast", show_header=True, header_style="bold cyan"
        )
        table.add_column("Date", style="cyan")
        table.add_column("Condition", style="green")
        table.add_column("Temp Range", style="yellow")
        table.add_column("Rain Chance", style="blue")

        for i, date in enumerate(dates):
            condition = (
                _weather_label(weather_codes[i]) if i < len(weather_codes) else "?"
            )
            low = t_min[i] if i < len(t_min) else "?"
            high = t_max[i] if i < len(t_max) else "?"
            rain = rain_prob[i] if i < len(rain_prob) else "?"
            table.add_row(date, condition, f"{low}°C - {high}°C", f"{rain}%")

        console.print(table)
    except requests.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except OSError as e:
        console.print(f"[red]Output error: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def forecast(
    location: str = typer.Argument("Tokyo", help="Location to check"),
    days: int = typer.Option(3, help="Number of days"),
):
    """Get weather forecast."""
    # Open-Meteo daily 预报最大支持 16 天
    if days <= 0 or days > 16:
        console.print("[red]`days` must be between 1 and 16.[/red]")
        raise typer.Exit(code=1)

    try:
        loc = _resolve_location(location)
        data = _get_weather_data(loc["latitude"], loc["longitude"], days=days)

        table = Table(
            title=f"📅 {loc.get('name', location)} Forecast",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Date", style="cyan")
        table.add_column("Condition", style="green")
        table.add_column("Temp Range", style="yellow")
        table.add_column("Chance of Rain", style="blue")

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        weather_codes = daily.get("weather_code", [])
        t_min = daily.get("temperature_2m_min", [])
        t_max = daily.get("temperature_2m_max", [])
        rain_prob = daily.get("precipitation_probability_max", [])

        for i, date in enumerate(dates):
            condition = (
                _weather_label(weather_codes[i]) if i < len(weather_codes) else "?"
            )
            low = t_min[i] if i < len(t_min) else "?"
            high = t_max[i] if i < len(t_max) else "?"
            rain = rain_prob[i] if i < len(rain_prob) else "?"
            table.add_row(date, condition, f"{low}°C - {high}°C", f"{rain}%")

        if not dates:
            console.print("[yellow]No forecast data returned.[/yellow]")
            raise typer.Exit(code=1)
        console.print(table)
    except requests.RequestException as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except OSError as e:
        console.print(f"[red]Output error: {e}[/red]")
        raise typer.Exit(code=1) from e
