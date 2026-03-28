"""Self-upgrade command for the globally installed Clawde CLI."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="upgrade",
    help="Upgrade the global Clawde CLI from the original local workspace",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()
PROJECT_NAME = "clawde-cli"
PROJECT_SENTINEL = 'name = "clawde-cli"'


def _find_uv_command() -> str | None:
    """Return the uv executable path if it is present in PATH."""
    return shutil.which("uv")


def _probe_uv() -> tuple[bool, str | None]:
    """Check whether uv exists and responds sanely."""
    executable = _find_uv_command()
    if executable is None:
        return False, "uv is not installed or not in PATH."

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
        return False, f"uv is present but failed to run{suffix}"
    except FileNotFoundError:
        return False, "uv is not installed or not in PATH."


def _looks_like_project_root(path: Path) -> bool:
    """Return whether path appears to be the local clawde-cli project root."""
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False

    try:
        content = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False

    return PROJECT_SENTINEL in content


def _find_project_root(start: Path | None = None) -> Path | None:
    """Locate the local clawde-cli workspace by walking upward from start."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if _looks_like_project_root(candidate):
            return candidate
    return None


def _get_installed_source_root() -> Path | None:
    """Return the original local source path used to install this tool, if any."""
    try:
        dist = importlib.metadata.distribution(PROJECT_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None

    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return None

    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    url = payload.get("url")
    if not isinstance(url, str):
        return None

    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None

    candidate = Path(unquote(parsed.path)).resolve()
    if _looks_like_project_root(candidate):
        return candidate
    return None


def _resolve_upgrade_source() -> Path | None:
    """Resolve the clawde-cli workspace to use for self-upgrade."""
    installed_source = _get_installed_source_root()
    if installed_source is not None:
        return installed_source
    return _find_project_root()


def _install_workspace_editable(project_root: Path) -> tuple[bool, str]:
    """Install the current workspace as the global clawde tool in editable mode."""
    uv = _find_uv_command()
    if uv is None:
        return False, "uv is not installed or not in PATH."

    result = subprocess.run(
        [
            uv,
            "tool",
            "install",
            "--force",
            "--editable",
            "--reinstall",
            "--refresh",
            str(project_root),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown error"
        return False, f"uv tool install failed: {detail}"
    return True, f"Global {PROJECT_NAME} upgraded from {project_root}"


@app.callback()
def upgrade(
    ctx: typer.Context,
):
    """Upgrade the global Clawde CLI from the original local workspace."""
    if ctx.invoked_subcommand is not None:
        return

    uv_ready, uv_error = _probe_uv()
    if not uv_ready:
        console.print(
            Panel(
                f"[bold red]uv is unavailable[/bold red]\n{uv_error}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    project_root = _resolve_upgrade_source()
    if project_root is None:
        console.print(
            Panel(
                "[bold red]clawde-cli workspace not found[/bold red]\n"
                "Unable to resolve the original repository path from the installed tool metadata.\n"
                "Run this once manually:\n"
                "  uv tool install --force --editable --reinstall --refresh /path/to/clawde-cli",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Workspace:[/bold cyan] {project_root}\n"
            "This will reinstall the global `clawde` command from the original local workspace.\n"
            "Editable mode is used so local source changes are immediately reflected.",
            title="Clawde Upgrade",
            border_style="cyan",
        )
    )

    try:
        success, message = _install_workspace_editable(project_root)
        if not success:
            console.print(
                Panel(
                    "[bold red]✗ Upgrade failed[/bold red]\n"
                    f"{message}\n"
                    "If the issue persists, try:\n"
                    f"  1. Verify uv: {_find_uv_command() or 'uv'} --version\n"
                    f"  2. Retry manually: uv tool install --force --editable --reinstall --refresh {project_root}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        console.print(
            Panel(
                "[bold green]✓ Clawde upgrade completed![/bold green]\n"
                f"{message}",
                title="Upgrade Complete",
                border_style="green",
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Upgrade cancelled by user.[/yellow]")
        raise typer.Exit(code=130)
