"""Reliable external command runner."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    help="Reliable command execution",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()

TIMEOUT_EXIT_CODE = 124
NOT_FOUND_EXIT_CODE = 127
DEFAULT_TAIL_BYTES = 4000


def _tail_text(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""

    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text

    tail_raw = raw[-max_bytes:]
    trimmed = tail_raw.decode("utf-8", errors="replace")
    dropped = len(raw) - len(tail_raw)
    return f"[truncated {dropped} bytes]\\n{trimmed}"


def _parse_env(env_items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in env_items or []:
        if "=" not in item:
            raise ValueError(f"Invalid env pair: {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid env pair: {item!r}. Empty KEY is not allowed.")
        parsed[key] = value
    return parsed


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            proc.kill()
            return

        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()


def _run_once(
    command: list[str],
    timeout: float | None,
    cwd: Path | None,
    env_overrides: dict[str, str],
    stdin_data: str | None,
    max_bytes: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cmd_display = shlex.join(command)

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **env_overrides},
            stdin=subprocess.PIPE if stdin_data is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )
    except FileNotFoundError:
        duration_ms = int((time.perf_counter() - started) * 1000)
        message = f"Command not found: {command[0]}"
        return {
            "cmd": cmd_display,
            "cwd": str(cwd.resolve()) if cwd else str(Path.cwd()),
            "exit_code": NOT_FOUND_EXIT_CODE,
            "timed_out": False,
            "duration_ms": duration_ms,
            "stdout_tail": "",
            "stderr_tail": message,
            "stdout_bytes": 0,
            "stderr_bytes": len(message.encode("utf-8", errors="replace")),
        }
    except OSError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        message = f"Failed to start process: {exc}"
        return {
            "cmd": cmd_display,
            "cwd": str(cwd.resolve()) if cwd else str(Path.cwd()),
            "exit_code": 1,
            "timed_out": False,
            "duration_ms": duration_ms,
            "stdout_tail": "",
            "stderr_tail": message,
            "stdout_bytes": 0,
            "stderr_bytes": len(message.encode("utf-8", errors="replace")),
        }

    timed_out = False
    stdout = ""
    stderr = ""

    try:
        stdout, stderr = proc.communicate(
            input=stdin_data,
            timeout=timeout,
        )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except (subprocess.TimeoutExpired, ValueError):
            stdout = stdout or ""
            stderr = stderr or ""
        exit_code = TIMEOUT_EXIT_CODE

    duration_ms = int((time.perf_counter() - started) * 1000)
    stdout_raw = stdout.encode("utf-8", errors="replace")
    stderr_raw = stderr.encode("utf-8", errors="replace")
    return {
        "cmd": cmd_display,
        "cwd": str(cwd.resolve()) if cwd else str(Path.cwd()),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout_tail": _tail_text(stdout, max_bytes=max_bytes),
        "stderr_tail": _tail_text(stderr, max_bytes=max_bytes),
        "stdout_bytes": len(stdout_raw),
        "stderr_bytes": len(stderr_raw),
    }


def _print_human(result: dict[str, Any]) -> None:
    title_style = "green" if result["exit_code"] == 0 else "red"
    status_line = (
        f"exit={result['exit_code']} | duration={result['duration_ms']}ms"
        f" | timed_out={result['timed_out']}"
    )
    content = [f"[bold]{result['cmd']}[/bold]", status_line]
    if result["stdout_tail"]:
        content.append("\n[cyan]stdout tail:[/cyan]\n" + result["stdout_tail"])
    if result["stderr_tail"]:
        content.append("\n[yellow]stderr tail:[/yellow]\n" + result["stderr_tail"])

    console.print(
        Panel(
            "\n".join(content),
            title="Run Result",
            border_style=title_style,
        )
    )


@app.callback()
def run(
    ctx: typer.Context,
    command: list[str] = typer.Argument(
        ...,
        help="External command to execute",
    ),
    timeout: float = typer.Option(
        30.0,
        min=0.1,
        help="Timeout in seconds for each attempt",
    ),
    retry: int = typer.Option(
        0,
        min=0,
        help="Retry count on failure",
    ),
    retry_delay: float = typer.Option(
        0.5,
        min=0.0,
        help="Delay in seconds between retries",
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        help="Working directory",
    ),
    env: list[str] | None = typer.Option(
        None,
        "--env",
        help="Extra env vars (KEY=VALUE), repeatable",
    ),
    stdin_text: str | None = typer.Option(
        None,
        "--stdin",
        help="Raw stdin text",
    ),
    stdin_file: Path | None = typer.Option(
        None,
        "--stdin-file",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Read stdin from file",
    ),
    max_bytes: int = typer.Option(
        DEFAULT_TAIL_BYTES,
        "--max-bytes",
        min=1,
        help="Max bytes kept for stdout/stderr tail",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON",
    ),
):
    """Run a command with timeout/retry/log-friendly output."""
    if ctx.invoked_subcommand is not None:
        return

    if stdin_text is not None and stdin_file is not None:
        console.print("[red]Use either --stdin or --stdin-file, not both.[/red]")
        raise typer.Exit(code=2)

    try:
        env_overrides = _parse_env(env)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    stdin_data: str | None = stdin_text
    if stdin_file is not None:
        try:
            stdin_data = stdin_file.read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Failed to read stdin file: {exc}[/red]")
            raise typer.Exit(code=2) from exc

    attempts = retry + 1
    result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        result = _run_once(
            command=command,
            timeout=timeout,
            cwd=cwd,
            env_overrides=env_overrides,
            stdin_data=stdin_data,
            max_bytes=max_bytes,
        )
        result["attempt"] = attempt
        result["max_attempts"] = attempts

        if result["exit_code"] == 0:
            break
        if attempt < attempts:
            time.sleep(retry_delay)

    assert result is not None
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)

    raise typer.Exit(code=int(result["exit_code"]))
