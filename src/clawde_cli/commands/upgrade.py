"""Self-upgrade command for the globally installed Clawde CLI."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
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
SOURCE_ROOT_ENV = "CLAWDE_SOURCE_ROOT"
SOURCE_ROOT_FILE = Path.home() / ".openclaw" / "workspace" / "clawde" / "source_root.txt"


def _find_uv_command() -> str | None:
    """Return the uv executable path if it is present in PATH."""
    return shutil.which("uv")


def _find_git_command() -> str | None:
    """Return the git executable path if it is present in PATH."""
    return shutil.which("git")


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


def _get_env_source_root() -> Path | None:
    """Return an explicit source root from the environment, if valid."""
    raw_value = os.environ.get(SOURCE_ROOT_ENV)
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser().resolve()
    if _looks_like_project_root(candidate):
        return candidate
    return None


def _get_persisted_source_root() -> Path | None:
    """Return the persisted source root from disk, if valid."""
    try:
        raw_value = SOURCE_ROOT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser().resolve()
    if _looks_like_project_root(candidate):
        return candidate
    return None


def _persist_source_root(project_root: Path) -> tuple[bool, str | None]:
    """Persist the source root for future self-upgrades."""
    try:
        SOURCE_ROOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_ROOT_FILE.write_text(f"{project_root}\n", encoding="utf-8")
        return True, None
    except OSError as exc:
        return False, str(exc)


def _resolve_upgrade_source() -> Path | None:
    """Resolve the clawde-cli workspace to use for self-upgrade."""
    env_source = _get_env_source_root()
    if env_source is not None:
        return env_source

    persisted_source = _get_persisted_source_root()
    if persisted_source is not None:
        return persisted_source

    cwd_source = _find_project_root()
    if cwd_source is not None:
        return cwd_source

    installed_source = _get_installed_source_root()
    if installed_source is not None:
        return installed_source
    return None


def _tracked_workspace_files(project_root: Path) -> tuple[bool, list[Path] | str]:
    """Return tracked files from the workspace, using working tree contents."""
    git = _find_git_command()
    if git is None:
        return False, "git is not installed or not in PATH."

    result = subprocess.run(
        [git, "ls-files", "-z"],
        capture_output=True,
        cwd=project_root,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip() or "unknown error"
        return False, f"git ls-files failed: {detail}"

    paths = [
        Path(item.decode("utf-8"))
        for item in result.stdout.split(b"\x00")
        if item
    ]
    return True, paths


def _required_build_files(project_root: Path) -> list[Path]:
    """Return extra local files required by build metadata."""
    pyproject = project_root / "pyproject.toml"
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    project = payload.get("project")
    if not isinstance(project, dict):
        return []

    readme = project.get("readme")
    if isinstance(readme, str):
        return [Path(readme)]
    if isinstance(readme, dict):
        file_value = readme.get("file")
        if isinstance(file_value, str):
            return [Path(file_value)]
    return []


def _materialize_tracked_snapshot(project_root: Path, snapshot_root: Path) -> tuple[bool, str]:
    """Copy tracked working tree files into a clean snapshot directory."""
    tracked_ok, tracked_result = _tracked_workspace_files(project_root)
    if not tracked_ok:
        return False, str(tracked_result)

    tracked_files = tracked_result
    assert isinstance(tracked_files, list)
    required_files = _required_build_files(project_root)
    files_to_copy: list[Path] = []
    seen: set[Path] = set()
    for relative_path in [*tracked_files, *required_files]:
        if relative_path in seen:
            continue
        seen.add(relative_path)
        files_to_copy.append(relative_path)

    if not files_to_copy:
        return False, "No tracked files found in the clawde-cli workspace."

    for relative_path in files_to_copy:
        source = project_root / relative_path
        destination = snapshot_root / relative_path
        if not source.is_file():
            return False, f"Required snapshot file is missing from working tree: {relative_path}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return True, f"Prepared tracked snapshot with {len(files_to_copy)} files"


def _install_workspace_snapshot(project_root: Path, snapshot_root: Path) -> tuple[bool, str]:
    """Install a tracked workspace snapshot as the global clawde tool."""
    uv = _find_uv_command()
    if uv is None:
        return False, "uv is not installed or not in PATH."

    result = subprocess.run(
        [
            uv,
            "tool",
            "install",
            "--force",
            "--reinstall",
            "--refresh",
            str(snapshot_root),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown error"
        return False, f"uv tool install failed: {detail}"
    return True, f"Global {PROJECT_NAME} upgraded from tracked workspace snapshot at {project_root}"


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
                "  CLAWDE_SOURCE_ROOT=/path/to/clawde-cli clawde upgrade\n"
                "or:\n"
                "  uv tool install --force --reinstall --refresh /path/to/clawde-cli",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Workspace:[/bold cyan] {project_root}\n"
            "This will reinstall the global `clawde` command from a clean snapshot of tracked workspace files.\n"
            "Untracked files are excluded from the installed tool.",
            title="Clawde Upgrade",
            border_style="cyan",
        )
    )

    try:
        with tempfile.TemporaryDirectory(prefix="clawde-upgrade-") as tmpdir:
            snapshot_root = Path(tmpdir) / PROJECT_NAME
            snapshot_root.mkdir(parents=True, exist_ok=True)

            success, message = _materialize_tracked_snapshot(project_root, snapshot_root)
            if not success:
                console.print(
                    Panel(
                        "[bold red]✗ Upgrade failed[/bold red]\n"
                        f"{message}",
                        title="Error",
                        border_style="red",
                    )
                )
                raise typer.Exit(code=1)

            success, message = _install_workspace_snapshot(project_root, snapshot_root)
        if not success:
            console.print(
                Panel(
                    "[bold red]✗ Upgrade failed[/bold red]\n"
                    f"{message}\n"
                    "If the issue persists, try:\n"
                    f"  1. Verify uv: {_find_uv_command() or 'uv'} --version\n"
                    f"  2. Verify tracked files: {(_find_git_command() or 'git')} -C {project_root} ls-files\n"
                    f"  3. Retry manually: uv tool install --force --reinstall --refresh {project_root}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

        persisted_ok, persisted_error = _persist_source_root(project_root)
        if not persisted_ok:
            console.print(
                Panel(
                    "[yellow]⚠ Upgrade completed, but the source root could not be persisted[/yellow]\n"
                    f"{persisted_error}\n"
                    f"You can still override it later with {SOURCE_ROOT_ENV}.",
                    title="Warning",
                    border_style="yellow",
                )
            )

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
