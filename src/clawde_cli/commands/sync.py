"""Gemini auth sync commands."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="sync",
    help="Refresh Gemini OAuth authentication",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


def _find_openclaw_command() -> str | None:
    """Return the openclaw executable path if it is present in PATH."""
    return shutil.which("openclaw")


def _probe_openclaw() -> tuple[bool, str | None]:
    """Check whether openclaw exists and responds sanely."""
    executable = _find_openclaw_command()
    if executable is None:
        return False, "OpenClaw command is not installed or not in PATH."

    try:
        subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True, None
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        return False, f"OpenClaw command is present but failed to run{suffix}"
    except FileNotFoundError:
        return False, "OpenClaw command is not installed or not in PATH."


def _build_tty_command(base_command: list[str]) -> list[str]:
    """Wrap a command in a PTY-backed launcher when available."""
    script_cmd = shutil.which("script")
    if script_cmd is None:
        return base_command

    quoted = shlex.join(base_command)

    if os.uname().sysname == "Darwin":
        return [script_cmd, "-q", "/dev/null", "sh", "-lc", quoted]
    return [script_cmd, "-qec", quoted, "/dev/null"]


@app.callback()
def sync(ctx: typer.Context):
    """Refresh Gemini OAuth authentication via OpenClaw."""
    if ctx.invoked_subcommand is not None:
        return

    openclaw_ready, openclaw_error = _probe_openclaw()
    if not openclaw_ready:
        console.print(
            Panel(
                f"[bold red]OpenClaw is unavailable[/bold red]\n{openclaw_error}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    executable = _find_openclaw_command() or "openclaw"
    command = [
        executable,
        "models",
        "auth",
        "setup-token",
        "--provider",
        "google-gemini-cli",
        "--yes",
    ]
    wrapped_command = _build_tty_command(command)

    console.print(
        Panel(
            "[bold cyan]Refreshing Gemini OAuth authentication[/bold cyan]\n"
            "This runs:\n"
            "openclaw models auth setup-token --provider google-gemini-cli --yes",
            title="Clawde Sync",
            border_style="cyan",
        )
    )

    try:
        result = subprocess.run(wrapped_command, text=True)
    except FileNotFoundError as exc:
        missing_command = exc.filename or wrapped_command[0]
        console.print(
            Panel(
                "[bold red]Error: failed to launch sync command[/bold red]\n"
                f"Missing executable: {missing_command}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Sync cancelled by user.[/yellow]")
        raise typer.Exit(code=130)

    if result.returncode != 0:
        console.print(
            Panel(
                "[bold red]✗ Gemini auth refresh failed[/bold red]\n"
                "Command exited with a non-zero status.\n"
                "You can retry manually with:\n"
                "  openclaw models auth setup-token --provider google-gemini-cli --yes",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=result.returncode)

    console.print(
        Panel(
            "[bold green]✓ Gemini auth refresh completed![/bold green]\n"
            "Provider: google-gemini-cli",
            title="Sync Complete",
            border_style="green",
        )
    )
