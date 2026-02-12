"""Environment diagnostic commands."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Environment diagnostics",
    invoke_without_command=True,
    no_args_is_help=False,
)
console = Console()

OPENCLAW_ROOT = Path.home() / ".openclaw"
WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace"
CHECK_DIRS = [
    WORKSPACE_DIR,
    WORKSPACE_DIR / "notes",
    WORKSPACE_DIR / "memory",
    WORKSPACE_DIR / "logs",
]
REQUIRED_PACKAGES = ["typer", "rich", "requests", "psutil"]
DNS_HOSTS = ["geocoding-api.open-meteo.com", "api.open-meteo.com"]


@dataclass
class CheckResult:
    id: str
    ok: bool
    severity: str
    message: str
    details: dict[str, Any]
    duration_ms: int


def _result(
    check_id: str,
    severity: str,
    message: str,
    details: dict[str, Any],
    started: float,
) -> CheckResult:
    duration_ms = int((time.perf_counter() - started) * 1000)
    return CheckResult(
        id=check_id,
        ok=severity != "error",
        severity=severity,
        message=message,
        details=details,
        duration_ms=duration_ms,
    )


def _skipped_result(check_id: str, message: str, details: dict[str, Any]) -> CheckResult:
    return CheckResult(
        id=check_id,
        ok=True,
        severity="info",
        message=message,
        details={"skipped": True, **details},
        duration_ms=0,
    )


def _check_runtime() -> CheckResult:
    started = time.perf_counter()
    pyver = platform.python_version()
    details = {
        "python_version": pyver,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    return _result("runtime", "info", f"Python {pyver}", details, started)


def _check_dependencies() -> CheckResult:
    started = time.perf_counter()
    versions: dict[str, str] = {}
    missing: list[str] = []

    for pkg in REQUIRED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            missing.append(pkg)

    details = {"versions": versions, "missing": missing}
    if missing:
        return _result(
            "dependencies",
            "error",
            f"Missing dependencies: {', '.join(missing)}",
            details,
            started,
        )

    return _result("dependencies", "info", "Dependency versions OK", details, started)


def _check_openclaw_root() -> CheckResult:
    started = time.perf_counter()
    path = OPENCLAW_ROOT
    details = {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}

    if not path.exists():
        return _result(
            "openclaw_root",
            "warn",
            "OpenClaw root not found (~/.openclaw)",
            details,
            started,
        )
    if not path.is_dir():
        return _result(
            "openclaw_root",
            "error",
            "OpenClaw root exists but is not a directory",
            details,
            started,
        )
    return _result("openclaw_root", "info", "OpenClaw root directory found", details, started)


def _check_workspace() -> CheckResult:
    started = time.perf_counter()
    details: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for path in CHECK_DIRS:
        state: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "writable": False,
        }

        if not path.exists():
            warnings.append(f"missing: {path}")
            details[str(path)] = state
            continue

        if not path.is_dir():
            errors.append(f"not a directory: {path}")
            details[str(path)] = state
            continue

        probe = path / f".doctor-write-probe-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            state["writable"] = True
        except OSError as exc:
            errors.append(f"not writable: {path}")
            state["write_error"] = str(exc)
        finally:
            try:
                if probe.exists():
                    probe.unlink()
            except OSError:
                pass

        details[str(path)] = state

    if errors:
        return _result("workspace", "error", "; ".join(errors), details, started)
    if warnings:
        return _result(
            "workspace",
            "warn",
            "Workspace directories missing (commands may auto-create some paths)",
            {"issues": warnings, **details},
            started,
        )

    return _result("workspace", "info", "Workspace directories are ready", details, started)


def _check_dns() -> CheckResult:
    started = time.perf_counter()
    resolved: dict[str, str] = {}
    failed: dict[str, str] = {}

    for host in DNS_HOSTS:
        try:
            resolved[host] = socket.gethostbyname(host)
        except OSError as exc:
            failed[host] = str(exc)

    details = {"resolved": resolved, "failed": failed}
    if failed:
        return _result(
            "dns",
            "error",
            f"DNS lookup failed for {len(failed)} host(s)",
            details,
            started,
        )

    return _result("dns", "info", "DNS resolution OK", details, started)


def _check_http(timeout: float) -> CheckResult:
    started = time.perf_counter()
    url = "https://geocoding-api.open-meteo.com/v1/search"

    try:
        response = requests.get(
            url,
            params={"name": "Tokyo", "count": 1, "format": "json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        count = len(payload.get("results", []) or [])
        return _result(
            "http_open_meteo",
            "info",
            "Open-Meteo HTTP probe OK",
            {
                "url": response.url,
                "status_code": response.status_code,
                "results_count": count,
            },
            started,
        )
    except requests.RequestException as exc:
        return _result(
            "http_open_meteo",
            "error",
            "Open-Meteo HTTP probe failed",
            {
                "url": url,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            started,
        )


def _check_env() -> CheckResult:
    started = time.perf_counter()
    proxy_vars = {
        key: os.getenv(key)
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ]
        if os.getenv(key)
    }

    ssl_vars = {
        key: os.getenv(key)
        for key in ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
        if os.getenv(key)
    }

    details = {
        "home": str(Path.home()),
        "proxy_env": proxy_vars,
        "ssl_env": ssl_vars,
    }
    return _result("environment", "info", "Environment variables captured", details, started)


def _run_all_checks(timeout: float, no_network: bool) -> list[CheckResult]:
    checks = [
        _check_runtime(),
        _check_dependencies(),
        _check_openclaw_root(),
        _check_workspace(),
        _check_env(),
    ]
    if no_network:
        checks.append(
            _skipped_result(
                "dns",
                "DNS check skipped by --no-network",
                {"reason": "no_network"},
            )
        )
        checks.append(
            _skipped_result(
                "http_open_meteo",
                "HTTP probe skipped by --no-network",
                {"reason": "no_network"},
            )
        )
        return checks

    checks.append(_check_dns())
    checks.append(_check_http(timeout=timeout))
    return checks


def _print_human(results: list[CheckResult]) -> None:
    table = Table(title="🩺 Clawde Doctor", show_header=True, header_style="bold cyan")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Message", style="white")
    table.add_column("Duration", justify="right", style="dim")

    for item in results:
        if item.severity == "info":
            status = "[green]OK[/green]"
        elif item.severity == "warn":
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[red]FAIL[/red]"
        table.add_row(item.id, status, item.message, f"{item.duration_ms}ms")

    console.print(table)


def _print_json(results: list[CheckResult]) -> None:
    payload = [asdict(item) for item in results]
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _has_errors(results: list[CheckResult]) -> bool:
    return any(item.severity == "error" for item in results)


def _should_fail(results: list[CheckResult], strict: bool) -> bool:
    if _has_errors(results):
        return True
    if strict and any(item.severity == "warn" for item in results):
        return True
    return False


@app.callback()
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    timeout: float = typer.Option(5.0, min=0.1, help="HTTP probe timeout in seconds"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat warnings as failures (non-zero exit)",
    ),
    no_network: bool = typer.Option(
        False,
        "--no-network",
        help="Skip DNS/HTTP checks",
    ),
):
    """Run environment diagnostics."""
    if ctx.invoked_subcommand is not None:
        return

    results = _run_all_checks(timeout=timeout, no_network=no_network)
    if json_output:
        _print_json(results)
    else:
        _print_human(results)

    if _should_fail(results, strict=strict):
        raise typer.Exit(code=1)
