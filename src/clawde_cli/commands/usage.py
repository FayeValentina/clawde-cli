"""Usage reporting commands for provider quotas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="usage",
    help="Show provider quota usage",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()
HERMES_AUTH_PATH = Path.home() / ".hermes" / "auth.json"
WHAM_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
REQUEST_TIMEOUT = 20


@dataclass(frozen=True)
class CodexAuthEntry:
    """One Codex account/token source from Hermes auth storage."""

    label: str
    access_token: str
    account_id: str | None = None


def _dedupe_auth_entries(entries: list[CodexAuthEntry]) -> list[CodexAuthEntry]:
    """Deduplicate entries by access token while preserving order.

    Prefer the first-seen label, but upgrade it with a non-empty account_id if a
    later duplicate has one.
    """
    deduped: list[CodexAuthEntry] = []
    index_by_token: dict[str, int] = {}
    for entry in entries:
        existing_idx = index_by_token.get(entry.access_token)
        if existing_idx is None:
            index_by_token[entry.access_token] = len(deduped)
            deduped.append(entry)
            continue
        existing = deduped[existing_idx]
        if existing.account_id is None and entry.account_id is not None:
            deduped[existing_idx] = CodexAuthEntry(
                label=existing.label,
                access_token=existing.access_token,
                account_id=entry.account_id,
            )
    return deduped


def _load_codex_auth_entries(auth_path: Path = HERMES_AUTH_PATH) -> list[CodexAuthEntry]:
    """Load all available OpenAI Codex auth entries from Hermes storage."""
    try:
        payload = json.loads(auth_path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"Hermes auth file not found: {auth_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Hermes auth file is not valid JSON: {auth_path}") from exc

    entries: list[CodexAuthEntry] = []

    pool_entries = payload.get("credential_pool", {}).get("openai-codex", [])
    if isinstance(pool_entries, list):
        for idx, entry in enumerate(pool_entries, start=1):
            if not isinstance(entry, dict):
                continue
            access_token = str(entry.get("access_token") or "").strip()
            if not access_token:
                continue
            label = str(entry.get("label") or "").strip() or f"account-{idx}"
            account_id = str(entry.get("account_id") or "").strip() or None
            entries.append(
                CodexAuthEntry(
                    label=label,
                    access_token=access_token,
                    account_id=account_id,
                )
            )

    provider = payload.get("providers", {}).get("openai-codex", {})
    tokens = provider.get("tokens", {})
    access_token = str(tokens.get("access_token") or "").strip()
    account_id = str(tokens.get("account_id") or "").strip() or None
    if access_token:
        entries.append(
            CodexAuthEntry(
                label="current",
                access_token=access_token,
                account_id=account_id,
            )
        )

    entries = _dedupe_auth_entries(entries)
    if not entries:
        raise RuntimeError("OpenAI Codex is not logged in under ~/.hermes/auth.json")
    return entries


def _load_codex_auth(auth_path: Path = HERMES_AUTH_PATH) -> tuple[str, str | None]:
    """Backward-compatible single-entry loader for existing callers/tests."""
    first = _load_codex_auth_entries(auth_path)[0]
    return first.access_token, first.account_id


def _fetch_codex_usage(access_token: str, account_id: str | None) -> dict[str, Any]:
    """Fetch Codex quota usage from ChatGPT's WHAM endpoint."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "CodexBar",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    try:
        response = requests.get(WHAM_USAGE_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch Codex usage: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip()
        if len(detail) > 240:
            detail = detail[:240] + "..."
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Codex usage request failed with HTTP {response.status_code}{suffix}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Codex usage endpoint returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Codex usage endpoint returned an unexpected response shape")
    return payload


def _window_label(limit_window_seconds: object, *, secondary: bool = False) -> str:
    """Convert a window size into a human-readable label."""
    if not isinstance(limit_window_seconds, (int, float)) or limit_window_seconds <= 0:
        return "Window"
    hours = int(limit_window_seconds // 3600)
    if secondary and hours >= 168:
        return "Week"
    if hours >= 24:
        days = hours // 24
        return f"{days}d"
    return f"{hours}h"


def _format_reset(seconds: object) -> str:
    """Format reset countdown from seconds into a compact string."""
    if not isinstance(seconds, (int, float)):
        return "unknown"
    total = max(int(seconds), 0)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return "<1m"


def _format_rate_limit_section(title: str, payload: dict[str, Any]) -> list[str]:
    """Format one rate-limit section into display lines."""
    lines = [f"[bold]{title}[/bold]"]
    allowed = payload.get("allowed")
    limit_reached = payload.get("limit_reached")
    if isinstance(allowed, bool):
        status = "allowed" if allowed else "blocked"
        if isinstance(limit_reached, bool) and limit_reached:
            status += " · limit reached"
        lines.append(f"状态: {status}")

    primary = payload.get("primary_window")
    secondary = payload.get("secondary_window")
    if isinstance(primary, dict):
        lines.append(_format_window_line(primary, secondary=False))
    if isinstance(secondary, dict):
        lines.append(_format_window_line(secondary, secondary=True))
    if len(lines) == 1:
        lines.append("暂无窗口信息")
    return lines


def _format_window_line(window: dict[str, Any], *, secondary: bool) -> str:
    """Format one usage window line."""
    used_percent = window.get("used_percent")
    if isinstance(used_percent, (int, float)):
        used = max(min(int(round(used_percent)), 100), 0)
        left = 100 - used
        percent_text = f"{left}% left"
    else:
        percent_text = "usage unknown"

    label = _window_label(window.get("limit_window_seconds"), secondary=secondary)
    reset_text = _format_reset(window.get("reset_after_seconds"))
    return f"- {label}: {percent_text} · resets in {reset_text}"


def _build_usage_panel(payload: dict[str, Any], *, label: str | None = None) -> Panel:
    """Build a rich panel for Codex usage output."""
    plan_type = str(payload.get("plan_type") or "unknown").strip() or "unknown"
    body: list[str] = []
    if label:
        body.append(f"Account: {label}")
    body.append(f"Plan: {plan_type}")

    rate_limit = payload.get("rate_limit")
    if isinstance(rate_limit, dict):
        body.extend(["", *_format_rate_limit_section("Codex", rate_limit)])

    code_review_limit = payload.get("code_review_rate_limit")
    if isinstance(code_review_limit, dict):
        body.extend(["", *_format_rate_limit_section("Code Review", code_review_limit)])

    credits = payload.get("credits")
    if isinstance(credits, dict):
        balance = credits.get("balance")
        unlimited = credits.get("unlimited")
        if unlimited is True:
            body.extend(["", "Credits: unlimited"])
        elif balance is not None:
            body.extend(["", f"Credits balance: {balance}"])

    return Panel(
        "\n".join(body),
        title="Codex Usage" if not label else f"Codex Usage · {label}",
        border_style="cyan",
    )


@app.callback()
def usage(ctx: typer.Context):
    """Show Codex usage based on Hermes OAuth credentials."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        auth_entries = _load_codex_auth_entries()
    except RuntimeError as exc:
        console.print(
            Panel(
                f"[bold red]Unable to load Codex usage[/bold red]\n{exc}",
                title="Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    successes = 0
    errors: list[tuple[str, str]] = []

    for entry in auth_entries:
        try:
            payload = _fetch_codex_usage(entry.access_token, entry.account_id)
        except RuntimeError as exc:
            errors.append((entry.label, str(exc)))
            continue
        console.print(_build_usage_panel(payload, label=entry.label if len(auth_entries) > 1 else None))
        successes += 1

    for label, message in errors:
        console.print(
            Panel(
                f"[bold yellow]Unable to load Codex usage for {label}[/bold yellow]\n{message}",
                title="Warning" if successes else "Error",
                border_style="yellow" if successes else "red",
            )
        )

    if successes == 0:
        raise typer.Exit(code=1)
