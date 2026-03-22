"""Search utterances from extracted session records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_SRC = Path.home() / ".openclaw" / "workspace" / "summaries" / "sessions_text"
DEFAULT_QMD_COMMAND = Path.home() / ".local" / "bin" / "qmd"


def _load_records(jsonl_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with jsonl_file.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
    except OSError:
        return []
    return records


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value


def _contains_query(text: str, query: str, ignore_case: bool) -> bool:
    if ignore_case:
        return query.lower() in text.lower()
    return query in text


def _parse_timestamp(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _line_preview(text: str, max_chars: int = 220) -> str:
    one = text.replace("\n", " ").strip()
    if len(one) <= max_chars:
        return one
    return one[:max_chars] + "..."


def _strip_leading_thinking(context: str) -> str:
    text = context.strip()
    if not text.startswith("[thinking] "):
        return text
    parts = text.split("\n", 1)
    if len(parts) == 1:
        return ""
    return parts[1].strip()


def _search_grep(
    query: str, src: Path, ignore_case: bool, limit: int
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for jsonl_file in sorted(src.glob("*.jsonl")):
        records = _load_records(jsonl_file)
        for rec in records:
            raw_context = _normalize_text(rec.get("context"))
            context = _strip_leading_thinking(raw_context)
            if not context:
                continue
            if not _contains_query(context, query, ignore_case):
                continue
            matches.append(
                {
                    "source": "grep",
                    "session": jsonl_file.stem,
                    "user": _normalize_text(rec.get("user")) or "unknown",
                    "timestamp": _parse_timestamp(rec.get("timestamp")),
                    "context": context,
                    "role": _normalize_text(rec.get("role")),
                    "message_id": _normalize_text(rec.get("message_id")),
                    "parent_id": _normalize_text(rec.get("parent_id")),
                    "qmd_score": None,
                }
            )

    matches.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return matches[:limit]


def _extract_session_stem_from_qmd_item(item: dict[str, Any]) -> str | None:
    title = item.get("title")
    if isinstance(title, str) and title:
        return title
    file_value = item.get("file")
    if not isinstance(file_value, str):
        return None
    # Example: qmd://sessions-text/<session-id>.txt
    tail = file_value.rsplit("/", 1)[-1]
    if not tail:
        return None
    if tail.endswith(".txt"):
        return tail[:-4]
    return tail


def _search_qmd_candidates(
    query: str,
    src: Path,
    qmd_command: Path,
    qmd_collection: str,
    qmd_limit: int,
    ignore_case: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    cmd = [
        str(qmd_command),
        "search",
        query,
        "--json",
        "-n",
        str(qmd_limit),
        "-c",
        qmd_collection,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return [], f"Failed to run qmd: {exc}"

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}"
        return [], f"qmd search failed: {detail}"

    try:
        payload = json.loads(proc.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        return [], f"qmd output is not valid JSON: {exc}"

    if not isinstance(payload, list):
        return [], "qmd output JSON is not a list"

    matches: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        stem = _extract_session_stem_from_qmd_item(item)
        if not stem:
            continue
        session_jsonl = src / f"{stem}.jsonl"
        if not session_jsonl.exists():
            continue

        records = _load_records(session_jsonl)
        for rec in records:
            raw_context = _normalize_text(rec.get("context"))
            context = _strip_leading_thinking(raw_context)
            if not context:
                continue
            if not _contains_query(context, query, ignore_case):
                continue
            matches.append(
                {
                    "source": "qmd+grep",
                    "session": stem,
                    "user": _normalize_text(rec.get("user")) or "unknown",
                    "timestamp": _parse_timestamp(rec.get("timestamp")),
                    "context": context,
                    "role": _normalize_text(rec.get("role")),
                    "message_id": _normalize_text(rec.get("message_id")),
                    "parent_id": _normalize_text(rec.get("parent_id")),
                    "qmd_score": item.get("score"),
                }
            )

    matches.sort(
        key=lambda x: (
            float(x.get("qmd_score")) if isinstance(x.get("qmd_score"), (int, float)) else 0.0,
            x.get("timestamp") or "",
        ),
        reverse=True,
    )
    return matches, None


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for rec in results:
        key = (
            rec.get("session") or "",
            rec.get("message_id") or "",
            rec.get("timestamp") or "",
            rec.get("context") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped


def _print_human(results: list[dict[str, Any]], query: str) -> None:
    if not results:
        console.print(f"[yellow]No matches found for '{query}'.[/yellow]")
        return

    table = Table(title=f"🔎 Search Results ({len(results)})", show_header=True, header_style="bold cyan")
    table.add_column("Source", style="cyan")
    table.add_column("User", style="green")
    table.add_column("Timestamp", style="yellow")
    table.add_column("Session", style="blue")
    table.add_column("Context", style="white")

    for rec in results:
        table.add_row(
            str(rec.get("source", "")),
            str(rec.get("user", "")),
            str(rec.get("timestamp", "")),
            str(rec.get("session", "")),
            _line_preview(str(rec.get("context", ""))),
        )
    console.print(table)


def search_command(
    query: str = typer.Argument(..., help="Search query text"),
    src: Path = typer.Option(
        DEFAULT_SRC,
        "--src",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory containing extracted session *.jsonl files",
    ),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum merged results"),
    grep_limit: int = typer.Option(60, "--grep-limit", min=1, help="Maximum grep matches before merge"),
    qmd_limit: int = typer.Option(20, "--qmd-limit", min=1, help="Number of qmd BM25 hits"),
    use_qmd: bool = typer.Option(True, "--qmd/--no-qmd", help="Enable qmd BM25 candidate search"),
    qmd_command: Path = typer.Option(
        DEFAULT_QMD_COMMAND,
        "--qmd-command",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to qmd executable",
    ),
    qmd_collection: str = typer.Option(
        "sessions-text", "--qmd-collection", help="qmd collection name for session text"
    ),
    ignore_case: bool = typer.Option(
        True, "--ignore-case/--case-sensitive", help="Case-insensitive matching for grep refinement"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
):
    """Search session utterances using grep precision + qmd BM25 recall."""
    if not src.exists() or not src.is_dir():
        console.print(f"[red]Source directory not found: {src}[/red]")
        raise typer.Exit(code=1)

    grep_results = _search_grep(query=query, src=src, ignore_case=ignore_case, limit=grep_limit)

    qmd_results: list[dict[str, Any]] = []
    qmd_error: str | None = None
    if use_qmd:
        qmd_results, qmd_error = _search_qmd_candidates(
            query=query,
            src=src,
            qmd_command=qmd_command,
            qmd_collection=qmd_collection,
            qmd_limit=qmd_limit,
            ignore_case=ignore_case,
        )

    merged = _dedupe_results(grep_results + qmd_results)
    merged = merged[:limit]

    payload = {
        "query": query,
        "source_dir": str(src),
        "counts": {
            "grep": len(grep_results),
            "qmd_refined": len(qmd_results),
            "merged": len(merged),
        },
        "qmd": {
            "enabled": use_qmd,
            "command": str(qmd_command),
            "collection": qmd_collection,
            "error": qmd_error,
        },
        "results": merged,
    }

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if qmd_error:
        console.print(f"[yellow]{qmd_error}[/yellow]")
    _print_human(merged, query=query)
