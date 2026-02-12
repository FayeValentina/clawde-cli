"""Notes management commands."""

import typer
import webbrowser
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Quick notes operations")
console = Console()

# 默认笔记目录，按日期落盘为 Markdown 文件
NOTES_DIR = Path.home() / ".openclaw" / "workspace" / "notes"


def ensure_notes_dir():
    """Ensure notes directory exists."""
    # 首次使用时自动创建目录
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return NOTES_DIR


@app.command()
def add(
    content: str = typer.Argument(..., help="Note content to add"),
):
    """Add a quick note."""
    try:
        notes_dir = ensure_notes_dir()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 当天所有记录汇总到同一个 YYYY-MM-DD.md 文件
        today_file = notes_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        with today_file.open("a", encoding="utf-8") as f:
            f.write(f"\n## {timestamp}\n\n{content}\n\n")

        console.print(
            Panel(
                f"[green]✓ Note added to {today_file.name}[/green]\n\n{content}",
                title="📝 Quick Note",
                border_style="green",
            )
        )
    except OSError as e:
        console.print(f"[red]Failed to add note: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def today():
    """Show today's notes."""
    try:
        notes_dir = ensure_notes_dir()
        today_file = notes_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

        if today_file.exists():
            content = today_file.read_text(encoding="utf-8")
            console.print(
                Panel(
                    content or "No notes for today yet.",
                    title=f"📝 Today's Notes ({today_file.name})",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                "[dim]No notes for today yet. Use 'clawde notes add <content>' to add one.[/dim]"
            )
    except OSError as e:
        console.print(f"[red]Failed to read today's notes: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command("list")
def list_notes():
    """List all note files."""
    try:
        notes_dir = ensure_notes_dir()
        # 逆序展示，最近日期排在前面
        files = sorted(notes_dir.glob("*.md"), reverse=True)

        if not files:
            console.print("[dim]No notes found.[/dim]")
            return

        console.print("[bold cyan]📝 Available Notes:[/bold cyan]")
        for f in files:
            size = f.stat().st_size
            console.print(f"  • [green]{f.name}[/green] ({size} bytes)")
    except OSError as e:
        console.print(f"[red]Failed to list notes: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command("open")
def open_today():
    """Open today's notes in default editor."""
    try:
        notes_dir = ensure_notes_dir()
        today_file = notes_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        # 不存在时先创建空文件，确保可被系统默认应用打开
        today_file.touch(exist_ok=True)
    except OSError as e:
        console.print(f"[red]Failed to prepare today's note file: {e}[/red]")
        raise typer.Exit(code=1) from e

    opened = webbrowser.open(today_file.resolve().as_uri())
    if not opened:
        console.print("[red]Failed to open note in default application.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Opened {today_file.name} in default editor[/green]")
