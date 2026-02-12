"""Event log commands."""

from __future__ import annotations

import json
import shlex
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Event log operations")
console = Console()

WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace"
LOG_DIR = WORKSPACE_DIR / "logs"
LOG_FILE = LOG_DIR / "events.jsonl"


def _ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_since(since: str | None) -> datetime | None:
    if since is None:
        return None
    text = since.strip().lower()
    if not text:
        return None

    unit = text[-1]
    if unit not in {"m", "h", "d"}:
        raise ValueError("Invalid --since format. Use examples like 30m, 12h, 2d.")

    try:
        value = int(text[:-1])
    except ValueError as exc:
        raise ValueError("Invalid --since format. Use examples like 30m, 12h, 2d.") from exc

    if value < 0:
        raise ValueError("--since value must be non-negative.")

    delta: timedelta
    if unit == "m":
        delta = timedelta(minutes=value)
    elif unit == "h":
        delta = timedelta(hours=value)
    else:
        delta = timedelta(days=value)
    return datetime.now(timezone.utc) - delta


def _parse_event_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    value = raw
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_events(limit: int | None = None) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    if limit is None:
        return list(_iter_events())

    window: deque[dict[str, Any]] = deque(maxlen=limit)
    for event in _iter_events():
        window.append(event)
    return list(window)


def _iter_events() -> Iterator[dict[str, Any]]:
    if not LOG_FILE.exists():
        return
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def _read_events_from_offset(offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not LOG_FILE.exists():
        return 0, []

    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        content = f.read()
        new_offset = f.tell()

    if not content:
        return new_offset, []

    events: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return new_offset, events


def _append_event(event: dict[str, Any]) -> None:
    _ensure_log_dir()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _print_events_table(events: list[dict[str, Any]], title: str) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Time", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Level", style="yellow")
    table.add_column("Message", style="white")

    for event in events:
        ts = str(event.get("ts", "?"))
        evt_type = str(event.get("type", "?"))
        level = str(event.get("level", "info"))
        message = str(event.get("message", ""))
        table.add_row(ts, evt_type, level, message[:120])

    console.print(table)


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("add")
def add_event(
    event_type: str = typer.Option(..., "--type", help="Event type, e.g. TOOL_CALL"),
    data: str = typer.Option("{}", "--data", help="JSON object payload"),
    level: str = typer.Option("info", "--level", help="Event level, e.g. info/warn/error"),
    message: str = typer.Option("", "--message", help="Human-readable summary"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Append one JSONL event."""
    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid --data JSON: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    event = {
        "ts": _utc_now_iso(),
        "type": event_type,
        "level": level,
        "message": message,
        "data": parsed_data,
    }
    try:
        _append_event(event)
    except OSError as exc:
        console.print(f"[red]Failed to write log event: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if json_output:
        _emit_json(event)
        return
    console.print("[green]Log event added.[/green]")


@app.command("tail")
def tail_events(
    lines: int = typer.Option(20, "--lines", min=1, help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", help="Keep watching for new events"),
    interval: float = typer.Option(1.0, "--interval", min=0.1, help="Follow poll interval"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Show the latest events."""
    if not LOG_FILE.exists():
        if json_output:
            _emit_json([])
            return
        console.print("[dim]No log file found.[/dim]")
        return

    last_offset = 0
    try:
        events = _read_events(limit=lines)
        if json_output:
            _emit_json(events)
            return
        if events:
            _print_events_table(events, title=f"🧾 Last {min(lines, len(events))} Events")
        else:
            console.print("[dim]Log file is empty.[/dim]")

        if not follow:
            return

        last_offset = LOG_FILE.stat().st_size
        console.print("[dim]Following log... press Ctrl+C to stop.[/dim]")
        while True:
            time.sleep(interval)
            current_size = LOG_FILE.stat().st_size
            if current_size < last_offset:
                last_offset = 0
            if current_size <= last_offset:
                continue
            last_offset, events = _read_events_from_offset(last_offset)
            if events:
                _print_events_table(events, title=f"🧾 New Events ({len(events)})")
    except KeyboardInterrupt:
        console.print("[dim]Stopped following log.[/dim]")
    except OSError as exc:
        console.print(f"[red]Failed to read log file: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("grep")
def grep_events(
    keyword: str = typer.Argument(..., help="Keyword to search"),
    since: str | None = typer.Option(None, "--since", help="Time window, e.g. 2d/12h/30m"),
    limit: int = typer.Option(20, "--limit", min=1, help="Max matched events to show"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Search events by keyword and time window."""
    try:
        cutoff = _parse_since(since)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    matches: list[dict[str, Any]] = []
    needle = keyword.lower()
    window: deque[dict[str, Any]] = deque(maxlen=limit)
    for event in _iter_events():
        ts = _parse_event_time(event.get("ts"))
        if cutoff is not None and ts is not None and ts < cutoff:
            continue
        blob = json.dumps(event, ensure_ascii=False).lower()
        if needle in blob:
            window.append(event)
    matches = list(window)

    if not matches:
        if json_output:
            _emit_json([])
            return
        console.print(f"[dim]No events matched '{keyword}'.[/dim]")
        return

    if json_output:
        _emit_json(matches)
        return
    _print_events_table(matches, title=f"🔎 Matched Events ({len(matches)})")


@app.command("stats")
def stats_events(
    since: str | None = typer.Option("7d", "--since", help="Time window, e.g. 2d/12h/30m"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Summarize event log stats."""
    try:
        cutoff = _parse_since(since)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    total = 0
    failures = 0
    duration_values: list[float] = []
    failed_commands: dict[str, int] = {}

    for event in _iter_events():
        ts = _parse_event_time(event.get("ts"))
        if cutoff is not None and ts is not None and ts < cutoff:
            continue

        total += 1
        level = str(event.get("level", "info")).lower()
        data = event.get("data", {})

        duration_ms: float | None = None
        if isinstance(data, dict):
            raw_duration = data.get("duration_ms")
            if isinstance(raw_duration, (int, float)):
                duration_ms = float(raw_duration)
            exit_code = data.get("exit_code")
            cmd = data.get("cmd")
            failed = bool(level in {"error", "fatal"})
            if isinstance(exit_code, int) and exit_code != 0:
                failed = True
                if isinstance(cmd, list):
                    cmd = shlex.join(str(part) for part in cmd)
                if isinstance(cmd, str) and cmd:
                    failed_commands[cmd] = failed_commands.get(cmd, 0) + 1
            if failed:
                failures += 1
        else:
            if level in {"error", "fatal"}:
                failures += 1

        if duration_ms is not None:
            duration_values.append(duration_ms)

    avg_duration = (sum(duration_values) / len(duration_values)) if duration_values else 0.0
    failure_rate = (failures / total * 100) if total else 0.0
    top_failed = sorted(failed_commands.items(), key=lambda item: item[1], reverse=True)[:3]
    stats_payload = {
        "window": since or "all",
        "total_events": total,
        "failures": failures,
        "failure_rate": round(failure_rate, 2),
        "avg_duration_ms": round(avg_duration, 2),
        "top_failed_commands": [
            {"cmd": cmd, "count": count} for cmd, count in top_failed
        ],
    }
    if json_output:
        _emit_json(stats_payload)
        return

    summary = Table(title="📊 Log Stats", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Window", since or "all")
    summary.add_row("Total events", str(total))
    summary.add_row("Failures", str(failures))
    summary.add_row("Failure rate", f"{failure_rate:.1f}%")
    summary.add_row("Avg duration", f"{avg_duration:.1f} ms")
    console.print(summary)

    if top_failed:
        table = Table(title="Top Failed Commands", show_header=True, header_style="bold red")
        table.add_column("Command", style="yellow")
        table.add_column("Count", style="red", justify="right")
        for cmd, count in top_failed:
            table.add_row(cmd[:100], str(count))
        console.print(table)
