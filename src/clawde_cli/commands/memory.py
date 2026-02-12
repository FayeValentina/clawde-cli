"""Memory log commands."""

import typer
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="Memory log operations")
console = Console()

# memory 日志默认目录（按日期命名的 Markdown 文件）
MEMORY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
# 为避免终端渲染过大内容，单次最多渲染 200KB
MAX_RENDER_BYTES = 200_000


def get_today_memory_file() -> Path:
    """Get today's memory file path."""
    # 约定文件名格式：YYYY-MM-DD.md
    return MEMORY_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.md"


def _safe_memory_file(date: str) -> Path:
    """Build and validate memory log path to prevent path traversal."""
    # 先解析根目录和候选路径，再校验候选路径是否仍在根目录下
    memory_root = MEMORY_DIR.resolve()
    candidate = (MEMORY_DIR / f"{date}.md").resolve()
    try:
        candidate.relative_to(memory_root)
    except ValueError as exc:
        raise ValueError(f"Invalid date path: {date}") from exc
    return candidate


def _read_markdown_limited(path: Path) -> str:
    """Read markdown content with a size cap to avoid rendering overflow."""
    size = path.stat().st_size
    if size <= MAX_RENDER_BYTES:
        return path.read_text(encoding="utf-8")

    # 超过阈值时截断展示，并在结尾提示文件过大
    content = path.read_text(encoding="utf-8", errors="replace")
    trimmed = content[:MAX_RENDER_BYTES]
    return (
        f"{trimmed}\n\n"
        f"... [truncated, file size {size} bytes exceeds limit {MAX_RENDER_BYTES} bytes]"
    )


@app.command()
def today():
    """Show today's memory log."""
    # 仅查看当天日志，不自动创建新文件
    memory_file = get_today_memory_file()

    try:
        if memory_file.exists():
            content = _read_markdown_limited(memory_file)
            console.print(
                Panel(
                    Markdown(content),
                    title=f"🧠 Today's Memory ({memory_file.name})",
                    border_style="cyan",
                )
            )
        else:
            console.print("[dim]No memory log for today yet.[/dim]")
    except OSError as e:
        console.print(f"[red]Failed to read today's memory log: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def list():
    """List all memory log files."""
    try:
        if not MEMORY_DIR.exists():
            console.print("[dim]No memory directory found.[/dim]")
            return

        # 逆序显示，最近日期在前
        files = sorted(MEMORY_DIR.glob("*.md"), reverse=True)

        if not files:
            console.print("[dim]No memory logs found.[/dim]")
            return

        console.print("[bold cyan]🧠 Memory Logs:[/bold cyan]")
        for f in files[:10]:  # 仅展示最近 10 条，避免输出过长
            size = f.stat().st_size
            console.print(f"  • [green]{f.name}[/green] ({size} bytes)")

        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more")
    except OSError as e:
        console.print(f"[red]Failed to list memory logs: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def view(
    date: str = typer.Argument(
        datetime.now().strftime("%Y-%m-%d"), help="Date to view (YYYY-MM-DD)"
    ),
):
    """View memory log for a specific date."""
    try:
        # 对 date 做路径安全校验，防止传入路径穿越
        memory_file = _safe_memory_file(date)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    try:
        if memory_file.exists():
            content = _read_markdown_limited(memory_file)
            console.print(
                Panel(
                    Markdown(content), title=f"🧠 Memory: {date}", border_style="cyan"
                )
            )
        else:
            console.print(f"[red]No memory log found for {date}[/red]")
    except OSError as e:
        console.print(f"[red]Failed to read memory log for {date}: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def grep(
    keyword: str = typer.Argument(..., help="Keyword to search"),
):
    """Search memory logs for a keyword."""
    if not MEMORY_DIR.exists():
        console.print("[dim]No memory directory found.[/dim]")
        return

    # 收集命中结果：(文件名, 关键字附近上下文)
    matches = []
    try:
        for f in MEMORY_DIR.glob("*.md"):
            try:
                content = _read_markdown_limited(f)
            except OSError:
                continue
            if keyword.lower() in content.lower():
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        # 取命中行前后各两行，便于快速定位语境
                        context = "\n".join(lines[max(0, i - 2) : i + 3])
                        matches.append((f.name, context))
                        break  # 每个文件只展示首个命中，避免噪声过多
    except OSError as e:
        console.print(f"[red]Failed while searching memory logs: {e}[/red]")
        raise typer.Exit(code=1) from e

    if matches:
        console.print(
            f"[bold cyan]🧠 Found '{keyword}' in {len(matches)} files:[/bold cyan]\n"
        )
        # 最多展示 5 个文件，控制输出长度
        for filename, context in matches[:5]:
            console.print(
                Panel(
                    Markdown(context),
                    title=f"[green]{filename}[/green]",
                    border_style="green",
                )
            )
    else:
        console.print(f"[dim]No matches found for '{keyword}'[/dim]")
