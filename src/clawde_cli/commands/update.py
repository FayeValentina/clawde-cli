"""OpenClaw update commands."""

from __future__ import annotations

import re
import shutil
import subprocess
import time

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="update",
    help="Update OpenClaw to the latest version",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()


def _find_openclaw_command() -> str | None:
    """Return the openclaw executable path if it is present in PATH."""
    return shutil.which("openclaw")


def _find_npm_command() -> str | None:
    """Return the npm executable path if it is present in PATH."""
    return shutil.which("npm")


def _probe_openclaw() -> tuple[bool, str | None]:
    """Check whether openclaw exists and responds sanely.

    Returns:
        Tuple of (is_usable, error_message)
    """
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


def _probe_npm() -> tuple[bool, str | None]:
    """Check whether npm exists and responds sanely."""
    executable = _find_npm_command()
    if executable is None:
        return False, "npm is not installed or not in PATH."

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
        return False, f"npm is present but failed to run{suffix}"
    except FileNotFoundError:
        return False, "npm is not installed or not in PATH."


def _get_current_version() -> str:
    """Get current OpenClaw version."""
    executable = _find_openclaw_command() or "openclaw"
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gateway_status_output(timeout: int = 5) -> str:
    """Return raw gateway status output."""
    executable = _find_openclaw_command() or "openclaw"
    result = subprocess.run(
        [executable, "gateway", "status"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (result.stdout or "") + (result.stderr or "")


def _service_is_loaded(status_text: str) -> bool:
    """Check whether LaunchAgent is actually loaded."""
    return "Service: LaunchAgent (loaded)" in status_text


def _rpc_probe_is_ok(status_text: str) -> bool:
    """Check whether gateway probe succeeds."""
    return "RPC probe: ok" in status_text


def _runtime_is_running(status_text: str) -> bool:
    """Check whether the gateway runtime reports as running."""
    return "Runtime: running" in status_text


def _gateway_is_listening(status_text: str) -> bool:
    """Check whether gateway reports one or more listening addresses."""
    return bool(re.search(r"Listening:\s+\S+", status_text))


def _gateway_ready_via_service(status_text: str) -> bool:
    """Treat the gateway as healthy using OpenClaw's documented baseline."""
    return _runtime_is_running(status_text) and _rpc_probe_is_ok(status_text)


def _install_latest_openclaw() -> tuple[bool, str]:
    """Install the latest OpenClaw via npm."""
    npm = _find_npm_command()
    if npm is None:
        return False, "npm is not installed or not in PATH."

    console.print("[yellow]→ Installing latest OpenClaw via npm...[/yellow]")
    result = subprocess.run(
        [npm, "install", "-g", "openclaw@latest"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        return False, "npm install -g openclaw@latest failed"
    return True, "OpenClaw CLI updated via npm"


def _install_gateway_service() -> tuple[bool, str]:
    """Install the gateway LaunchAgent explicitly after updating the CLI."""
    executable = _find_openclaw_command() or "openclaw"
    console.print("[yellow]→ Reinstalling gateway LaunchAgent...[/yellow]")
    result = subprocess.run(
        [executable, "gateway", "install"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown error"
        return False, f"openclaw gateway install failed: {detail}"
    return True, "Gateway LaunchAgent installed"


def _wait_for_gateway(timeout: int = 30) -> bool:
    """Wait for gateway service to become ready.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if gateway service is ready, False otherwise
    """
    console.print(f"[yellow]→ Waiting for gateway service to start (max {timeout}s)...[/yellow]")
    for _ in range(timeout):
        try:
            status_text = _gateway_status_output(timeout=5)
            if _gateway_ready_via_service(status_text):
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)
    return False


def _build_gateway_status_summary() -> tuple[str, str]:
    """Return a compact status label and supporting details for the gateway."""
    try:
        status_text = _gateway_status_output(timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return "unknown", f"Unable to query gateway status: {exc}"

    if _gateway_ready_via_service(status_text):
        return "ready", "Runtime running and RPC probe ok."

    details = []
    if _runtime_is_running(status_text):
        details.append("runtime running")
    else:
        details.append("runtime not running")
    if _service_is_loaded(status_text):
        details.append("LaunchAgent loaded")
    else:
        details.append("LaunchAgent not loaded")
    if _rpc_probe_is_ok(status_text):
        details.append("RPC probe ok")
    else:
        details.append("RPC probe failed")
    if _gateway_is_listening(status_text):
        details.append("gateway listening")
    else:
        details.append("gateway not listening")
    return "degraded", ", ".join(details)


@app.callback()
def update(
    ctx: typer.Context,
    status_only: bool = typer.Option(
        False,
        "--status",
        "-s",
        help="Show current version and gateway status without installing",
    ),
    skip_launchagent: bool = typer.Option(
        False,
        "--skip-launchagent",
        help="Skip gateway LaunchAgent install and readiness verification",
    ),
):
    """Update OpenClaw to the latest version.

    This is the ONLY authorized way to update OpenClaw.
    DO NOT run 'openclaw update' directly or use other package managers.

    This command will:
    1. Update OpenClaw to the latest version
    2. Reinstall LaunchAgent to prevent "Gateway service not loaded" errors
    3. Wait for gateway service to become ready
    """
    if ctx.invoked_subcommand is not None:
        return

    openclaw_ready, openclaw_error = _probe_openclaw()
    if status_only and not openclaw_ready:
        console.print(
            Panel(
                f"[bold red]OpenClaw is unavailable[/bold red]\n{openclaw_error}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    current_version = _get_current_version()
    if openclaw_ready:
        gateway_state, gateway_details = _build_gateway_status_summary()
    else:
        gateway_state = "unavailable"
        gateway_details = openclaw_error or "OpenClaw command is unavailable."

    if status_only:
        console.print(
            Panel(
                f"[bold cyan]Current OpenClaw version:[/bold cyan] {current_version}\n"
                f"[bold cyan]Gateway status:[/bold cyan] {gateway_state}\n"
                f"[dim]{gateway_details}[/dim]\n"
                "Run without --status to update OpenClaw and repair the gateway service.",
                title="Update Status",
                border_style="cyan",
            )
        )
        return

    npm_ready, npm_error = _probe_npm()
    if not npm_ready:
        console.print(
            Panel(
                f"[bold red]npm is unavailable[/bold red]\n{npm_error}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Current version:[/bold cyan] {current_version}\n"
            "This will update OpenClaw to the latest version.\n"
            "The gateway LaunchAgent will be explicitly reinstalled and verified.",
            title="OpenClaw Update",
            border_style="cyan",
            )
        )

    if not openclaw_ready:
        console.print(
            Panel(
                "[yellow]OpenClaw is currently unavailable, but the updater will try to repair it.[/yellow]\n"
                f"{openclaw_error}",
                title="Recovery Mode",
                border_style="yellow",
            )
        )

    if skip_launchagent:
        console.print(
            "[yellow]→ LaunchAgent repair and gateway readiness checks are being skipped by request.[/yellow]"
        )

    console.print("[yellow]→ Updating OpenClaw via npm...[/yellow]")

    try:
        success, message = _install_latest_openclaw()
        if not success:
            console.print(
                Panel(
                    "[bold red]✗ Update failed[/bold red]\n"
                    f"{message}\n"
                    "If the issue persists, try:\n"
                    "  1. Verify npm: npm --version\n"
                    "  2. Install manually: npm install -g openclaw@latest\n"
                    "  3. Reinstall service: openclaw gateway install",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        new_version = _get_current_version()
        console.print(
            Panel(
                f"[bold green]✓ Update completed![/bold green]\n"
                f"[dim]Previous:[/dim] {current_version}\n"
                f"[bold cyan]Current:[/bold cyan] {new_version}",
                title="Version Updated",
                border_style="green",
            )
        )

        if not skip_launchagent:
            success, message = _install_gateway_service()
            if success:
                console.print(f"\n[green]✓ {message}[/green]")
            else:
                console.print(
                    Panel(
                        f"[yellow]⚠ Gateway install issue:[/yellow] {message}\n"
                        "The update succeeded, but the service was not verified as healthy.\n"
                        "Try: openclaw gateway install",
                        title="Warning",
                        border_style="yellow",
                    )
                )

        if skip_launchagent:
            console.print(
                Panel(
                    "[bold green]✓ OpenClaw update completed![/bold green]\n"
                    f"Version: {new_version}\n"
                    "LaunchAgent repair and gateway verification were skipped.",
                    title="Update Complete",
                    border_style="green",
                )
            )
        elif success:
            if _wait_for_gateway(timeout=30):
                console.print(
                    Panel(
                        "[bold green]✓ OpenClaw is ready![/bold green]\n"
                        f"Version: {new_version}\n"
                        "LaunchAgent is loaded and gateway is accepting connections.",
                        title="Update Complete",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        "[yellow]⚠ Gateway service verification did not complete in time[/yellow]\n"
                        "Please check with:\n"
                        "  openclaw gateway status",
                        title="Starting",
                        border_style="yellow",
                    )
                )
        else:
            console.print(
                Panel(
                    "[yellow]⚠ Skipping readiness wait because gateway reinstall did not succeed[/yellow]\n"
                    "Please fix the service first:\n"
                    "  openclaw gateway install\n"
                    "Then verify with:\n"
                    "  openclaw gateway status",
                    title="Verification Skipped",
                    border_style="yellow",
                )
            )

    except FileNotFoundError:
        console.print(
            Panel(
                "[bold red]Error: 'openclaw' command not found[/bold red]\n"
                "OpenClaw may not be installed or not in PATH.",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Update cancelled by user.[/yellow]")
        raise typer.Exit(code=130)


@app.command(name="status")
def update_status():
    """Show current version and gateway status without installing updates."""
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

    current_version = _get_current_version()
    gateway_state, gateway_details = _build_gateway_status_summary()

    console.print(
        Panel(
            f"[bold cyan]Current OpenClaw version:[/bold cyan] {current_version}\n"
            f"[bold cyan]Gateway status:[/bold cyan] {gateway_state}\n"
            f"[dim]{gateway_details}[/dim]\n"
            "[dim]To run the safe wrapper update flow:[/dim] clawde update",
            title="Status",
            border_style="cyan",
        )
    )
