"""OpenClaw update commands."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

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
LAUNCHAGENT_LABEL = "ai.openclaw.gateway"


def _check_openclaw_installed() -> bool:
    """Check if openclaw command is available."""
    try:
        subprocess.run(
            ["openclaw", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_current_version() -> str:
    """Get current OpenClaw version."""
    try:
        result = subprocess.run(
            ["openclaw", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _get_uid() -> str:
    """Get current user id for launchctl gui domain."""
    result = subprocess.run(
        ["id", "-u"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _gateway_status_output(timeout: int = 5) -> str:
    """Return raw gateway status output."""
    result = subprocess.run(
        ["openclaw", "gateway", "status"],
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


def _gateway_is_listening(status_text: str) -> bool:
    """Check whether gateway is listening on loopback."""
    return bool(re.search(r"Listening:\s+127\.0\.0\.1:\d+", status_text))


def _gateway_ready_via_service(status_text: str) -> bool:
    """Require both a loaded LaunchAgent and a healthy gateway probe."""
    return (
        _service_is_loaded(status_text)
        and _rpc_probe_is_ok(status_text)
        and _gateway_is_listening(status_text)
    )


def _launchctl_print_loaded(uid: str) -> bool:
    """Check whether launchd knows about the service label."""
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{LAUNCHAGENT_LABEL}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _reinstall_launchagent() -> tuple[bool, str]:
    """Reinstall LaunchAgent to fix 'Gateway service not loaded' issue.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        plist_path = Path.home() / "Library/LaunchAgents/ai.openclaw.gateway.plist"
        if not plist_path.exists():
            return False, f"LaunchAgent plist not found at {plist_path}"

        uid = _get_uid()
        gui_domain = f"gui/{uid}"
        service_target = f"{gui_domain}/{LAUNCHAGENT_LABEL}"

        console.print("[yellow]→ Booting out existing LaunchAgent (ignore errors if absent)...[/yellow]")
        subprocess.run(
            ["launchctl", "bootout", gui_domain, str(plist_path)],
            capture_output=True,
            text=True,
        )
        time.sleep(1)

        console.print("[yellow]→ Bootstrapping LaunchAgent...[/yellow]")
        bootstrap = subprocess.run(
            ["launchctl", "bootstrap", gui_domain, str(plist_path)],
            capture_output=True,
            text=True,
        )
        if bootstrap.returncode != 0 and "already bootstrapped" not in (bootstrap.stderr or "").lower():
            return False, f"Failed to bootstrap LaunchAgent: {(bootstrap.stderr or bootstrap.stdout).strip()}"

        console.print("[yellow]→ Kickstarting LaunchAgent...[/yellow]")
        kickstart = subprocess.run(
            ["launchctl", "kickstart", "-k", service_target],
            capture_output=True,
            text=True,
        )
        if kickstart.returncode != 0:
            return False, f"Failed to kickstart LaunchAgent: {(kickstart.stderr or kickstart.stdout).strip()}"

        time.sleep(2)

        if not _launchctl_print_loaded(uid):
            return False, "LaunchAgent did not appear in launchctl after bootstrap/kickstart"

        status_text = _gateway_status_output(timeout=10)
        if _gateway_ready_via_service(status_text):
            return True, "LaunchAgent reinstalled and gateway service is healthy"

        details = []
        if not _service_is_loaded(status_text):
            details.append("service not loaded")
        if not _rpc_probe_is_ok(status_text):
            details.append("RPC probe not ok")
        if not _gateway_is_listening(status_text):
            details.append("gateway not listening")
        suffix = ", ".join(details) if details else "post-install verification failed"
        return False, f"LaunchAgent installed but verification failed: {suffix}"

    except Exception as e:
        return False, f"Error reinstalling LaunchAgent: {e}"


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


@app.callback()
def update(
    ctx: typer.Context,
    status_only: bool = typer.Option(
        False,
        "--status",
        "-s",
        help="Check update status without installing",
    ),
    skip_launchagent: bool = typer.Option(
        False,
        "--skip-launchagent",
        help="Skip LaunchAgent reinstallation (not recommended)",
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

    if not _check_openclaw_installed():
        console.print(
            Panel(
                "[bold red]OpenClaw is not installed[/bold red]\n"
                "Please install OpenClaw first before using this command.",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    current_version = _get_current_version()

    if status_only:
        console.print(
            Panel(
                f"[bold cyan]Current OpenClaw version:[/bold cyan] {current_version}\n"
                "Run without --status to check for updates and install.",
                title="Update Status",
                border_style="cyan",
            )
        )
        return

    console.print(
        Panel(
            f"[bold cyan]Current version:[/bold cyan] {current_version}\n"
            "This will update OpenClaw to the latest version.\n"
            "The gateway service will be automatically restarted and verified.",
            title="OpenClaw Update",
            border_style="cyan",
        )
    )

    console.print("[yellow]→ Running 'openclaw update'...[/yellow]")

    try:
        result = subprocess.run(
            ["openclaw", "update"],
            capture_output=False,
            text=True,
        )

        if result.returncode != 0:
            console.print(
                Panel(
                    "[bold red]✗ Update failed[/bold red]\n"
                    "Please check the error message above.\n"
                    "If the issue persists, try:\n"
                    "  1. Check status: openclaw gateway status\n"
                    "  2. Reinstall service: openclaw gateway install\n"
                    "  3. Verify service: openclaw gateway status",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=result.returncode)

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
            console.print("\n[yellow]→ Reinstalling LaunchAgent to prevent service errors...[/yellow]")
            success, message = _reinstall_launchagent()

            if success:
                console.print(f"[green]✓ {message}[/green]")
            else:
                console.print(
                    Panel(
                        f"[yellow]⚠ LaunchAgent reinstall issue:[/yellow] {message}\n"
                        "The update succeeded, but the service was not verified as healthy.\n"
                        "Try: openclaw gateway install",
                        title="Warning",
                        border_style="yellow",
                    )
                )

        console.print("\n[yellow]→ Waiting for gateway service to become ready...[/yellow]")
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
    """Check if an update is available without installing."""
    if not _check_openclaw_installed():
        console.print(
            Panel(
                "[bold red]OpenClaw is not installed[/bold red]",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    current_version = _get_current_version()

    console.print(
        Panel(
            f"[bold cyan]Current OpenClaw version:[/bold cyan] {current_version}\n"
            "[dim]To check for available updates, run:[/dim] clawde update --status",
            title="Version Info",
            border_style="cyan",
        )
    )
