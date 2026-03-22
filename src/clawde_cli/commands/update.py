"""OpenClaw update commands."""

from __future__ import annotations

import subprocess
import sys
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


def _reinstall_launchagent() -> tuple[bool, str]:
    """Reinstall LaunchAgent to fix 'Gateway service not loaded' issue.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Step 1: Check if plist exists
        plist_path = Path.home() / "Library/LaunchAgents/ai.openclaw.gateway.plist"
        if not plist_path.exists():
            return False, f"LaunchAgent plist not found at {plist_path}"
        
        # Step 2: Unload if loaded (ignore errors)
        console.print("[yellow]→ Unloading existing LaunchAgent...[/yellow]")
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
        )
        time.sleep(1)
        
        # Step 3: Reload LaunchAgent
        console.print("[yellow]→ Reloading LaunchAgent...[/yellow]")
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Try bootstrap as fallback
            result = subprocess.run(
                ["launchctl", "bootstrap", f"gui/{subprocess.run(['id', '-u'], capture_output=True, text=True).stdout.strip()}", str(plist_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return False, f"Failed to load LaunchAgent: {result.stderr}"
        
        time.sleep(2)
        
        # Step 4: Verify gateway is running
        result = subprocess.run(
            ["openclaw", "gateway", "status"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "running" in result.stdout.lower():
            return True, "LaunchAgent reinstalled and gateway is running"
        
        return True, "LaunchAgent reinstalled (gateway starting)"
        
    except Exception as e:
        return False, f"Error reinstalling LaunchAgent: {e}"


def _wait_for_gateway(timeout: int = 30) -> bool:
    """Wait for gateway to become ready.
    
    Args:
        timeout: Maximum seconds to wait
        
    Returns:
        True if gateway is ready, False otherwise
    """
    console.print(f"[yellow]→ Waiting for gateway to start (max {timeout}s)...[/yellow]")
    for i in range(timeout):
        try:
            result = subprocess.run(
                ["openclaw", "gateway", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "running" in result.stdout.lower():
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
    3. Wait for gateway to become ready
    """
    if ctx.invoked_subcommand is not None:
        return

    # Check if openclaw is installed
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

    # Show current version and confirm
    console.print(
        Panel(
            f"[bold cyan]Current version:[/bold cyan] {current_version}\n"
            "This will update OpenClaw to the latest version.\n"
            "The gateway service will be automatically restarted.",
            title="OpenClaw Update",
            border_style="cyan",
        )
    )

    # Run openclaw update
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
                    "  1. Stop gateway: openclaw gateway stop\n"
                    "  2. Manual update: pnpm update -g openclaw\n"
                    "  3. Start gateway: openclaw gateway start",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=result.returncode)
        
        # Update succeeded
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
        
        # Step 2: Reinstall LaunchAgent (unless skipped)
        if not skip_launchagent:
            console.print("\n[yellow]→ Reinstalling LaunchAgent to prevent service errors...[/yellow]")
            success, message = _reinstall_launchagent()
            
            if success:
                console.print(f"[green]✓ {message}[/green]")
            else:
                console.print(
                    Panel(
                        f"[yellow]⚠ LaunchAgent reinstall issue:[/yellow] {message}\n"
                        "Gateway may still work. If you see 'Gateway service not loaded' errors,\n"
                        "try running: openclaw gateway install",
                        title="Warning",
                        border_style="yellow",
                    )
                )
        
        # Step 3: Wait for gateway to be ready
        console.print("\n[yellow]→ Waiting for gateway to become ready...[/yellow]")
        if _wait_for_gateway(timeout=30):
            console.print(
                Panel(
                    "[bold green]✓ OpenClaw is ready![/bold green]\n"
                    f"Version: {new_version}\n"
                    "Gateway is running and accepting connections.",
                    title="Update Complete",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]⚠ Gateway may still be starting[/yellow]\n"
                    "Please wait a moment and check with:\n"
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
