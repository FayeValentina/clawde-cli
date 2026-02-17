"""Session utterance extraction commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="Extract session utterances",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()

DEFAULT_SRC = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
DEFAULT_DST = Path.home() / ".openclaw" / "workspace" / "summaries" / "sessions_text"
MAX_CONTEXT_CHARS = 12000


def _truncate(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _extract_context_from_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            continue

        if item_type == "toolCall":
            tool_name = item.get("name") or "unknown_tool"
            args = item.get("arguments")
            args_str = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
            parts.append(f"[toolCall] {tool_name} args={args_str}")
            continue

        parts.append(f"[{item_type}] {json.dumps(item, ensure_ascii=False)}")

    return _truncate("\n".join(parts).strip())


def _extract_utterance(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "message":
        return None

    msg = event.get("message")
    if not isinstance(msg, dict):
        return None

    role = msg.get("role")
    if not isinstance(role, str) or not role:
        return None

    timestamp = event.get("timestamp") or msg.get("timestamp")
    if timestamp is not None and not isinstance(timestamp, str):
        timestamp = str(timestamp)

    context = _extract_context_from_content(msg.get("content"))
    if not context:
        return None

    speaker = role
    if role == "toolResult":
        speaker = f"tool:{msg.get('toolName') or 'unknown_tool'}"

    return {
        "user": speaker,
        "timestamp": timestamp,
        "context": context,
        "role": role,
        "message_id": event.get("id"),
        "parent_id": event.get("parentId"),
    }


@app.callback()
def extract(
    ctx: typer.Context,
    src: Path = typer.Option(
        DEFAULT_SRC,
        "--src",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Session JSONL directory",
    ),
    dst: Path = typer.Option(
        DEFAULT_DST,
        "--dst",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Output directory for extracted utterances",
    ),
):
    """Extract per-utterance records from OpenClaw session JSONL files."""
    if ctx.invoked_subcommand is not None:
        return

    if not src.exists() or not src.is_dir():
        console.print(f"[red]Source directory not found: {src}[/red]")
        raise typer.Exit(code=1)

    files = sorted(src.glob("*.jsonl"))
    if not files:
        console.print(f"[yellow]No session files found in {src}[/yellow]")
        return

    try:
        dst.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        console.print(f"[red]Failed to create output directory: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[cyan]Found {len(files)} session files[/cyan]")
    for session_file in files:
        records: list[dict[str, Any]] = []
        try:
            with session_file.open("r", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    record = _extract_utterance(event)
                    if record:
                        records.append(record)
        except OSError as exc:
            console.print(f"[red]Failed to read {session_file.name}: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        out_jsonl = dst / f"{session_file.stem}.jsonl"
        out_txt = dst / f"{session_file.stem}.txt"
        try:
            with out_jsonl.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            txt_lines = []
            for record in records:
                one_line_context = record["context"].replace("\n", "\\n")
                txt_lines.append(
                    f"user:{record['user']},timestamp:{record.get('timestamp')},context:{one_line_context}"
                )
            out_txt.write_text(
                "\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8"
            )
        except OSError as exc:
            console.print(f"[red]Failed to write outputs for {session_file.name}: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        console.print(
            f"[green]wrote:[/green] {out_jsonl} + {out_txt} [dim]({len(records)} utterances)[/dim]"
        )
