"""Deterministic image generation pipeline for OpenClaw agents."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
import typer

DEFAULT_SCRIPT_CANDIDATES = [
    Path.home() / ".npm-global/lib/node_modules/openclaw/skills/nano-banana-pro/scripts/generate_image.py",
    Path.home() / ".openclaw/skills/nano-banana-pro/scripts/generate_image.py",
]
DEFAULT_OUTPUT_DIR = Path.home() / ".openclaw/workspace"
DEFAULT_PUBLISH_DIR = Path("/tmp/openclaw-line-media")
DEFAULT_RESOLUTION = "1K"
RESOLUTION_CHOICES = {"1K", "2K", "4K"}


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if normalized:
        return normalized[:48]
    return "image"


def _default_filename(prompt: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{_slugify(prompt)}-{stamp}.png"


def _detect_script(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in DEFAULT_SCRIPT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Nano Banana script not found. Pass --script or install openclaw nano-banana-pro skill."
    )


def _parse_generator_output(stdout: str) -> tuple[str | None, str | None]:
    image_saved: str | None = None
    media_token: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("image saved:"):
            image_saved = line.split(":", 1)[1].strip() or image_saved
            continue
        if line.startswith("MEDIA:"):
            media_token = line.split(":", 1)[1].strip() or media_token
    return image_saved, media_token


def _build_public_url(base_url: str, filename: str, cache_bust: bool = True) -> str:
    cleaned = base_url.rstrip("/")
    url = f"{cleaned}/{quote(filename)}"
    if cache_bust:
        url = f"{url}?v={int(time.time())}"
    return url


def _wait_for_url(url: str, timeout_sec: float, interval_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=min(3.0, max(timeout_sec, 0.2)))
            if 200 <= resp.status_code < 300:
                return True
        except requests.RequestException:
            pass
        time.sleep(max(0.05, interval_sec))
    return False


def _resolve_local_path(
    image_saved: str | None,
    media_token: str | None,
    output_dir: Path,
    filename: str,
) -> Path | None:
    for candidate in (media_token, image_saved):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = output_dir / path
        if path.exists():
            return path.resolve()
    fallback = (output_dir / filename).resolve()
    if fallback.exists():
        return fallback
    return None


def generate_command(
    prompt: str = typer.Argument(..., help="Image prompt text"),
    filename: str | None = typer.Option(None, "--filename", help="Output filename (.png preferred)"),
    resolution: str = typer.Option(DEFAULT_RESOLUTION, "--resolution", help="1K, 2K, or 4K"),
    script: Path | None = typer.Option(
        None,
        "--script",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
        help="Path to nano-banana generate_image.py",
    ),
    uv_bin: str = typer.Option("uv", "--uv-bin", help="uv executable path"),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where generated files are written",
    ),
    publish_dir: Path = typer.Option(
        DEFAULT_PUBLISH_DIR,
        "--publish-dir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Static directory to publish image for web access",
    ),
    base_url: str = typer.Option(
        "",
        "--base-url",
        help="Public base URL for published files, e.g. https://host/openclaw-media",
    ),
    check_url: bool = typer.Option(True, "--check-url/--no-check-url", help="Verify public URL is reachable"),
    check_timeout: float = typer.Option(8.0, "--check-timeout", min=0.2, help="URL check timeout (seconds)"),
    check_interval: float = typer.Option(0.4, "--check-interval", min=0.05, help="URL check interval (seconds)"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON payload"),
    emit_media_line: bool = typer.Option(True, "--emit-media-line/--no-emit-media-line", help="Print 'MEDIA: <url>'"),
) -> None:
    """Generate image, publish to static dir, and emit a LINE-compatible media URL."""
    if resolution.upper() not in RESOLUTION_CHOICES:
        raise typer.BadParameter(f"Unsupported resolution: {resolution}. Use one of {sorted(RESOLUTION_CHOICES)}")

    selected_script = _detect_script(script)
    if not selected_script.exists():
        raise typer.BadParameter(f"Script not found: {selected_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    publish_dir.mkdir(parents=True, exist_ok=True)

    final_filename = filename or _default_filename(prompt)
    command = [
        uv_bin,
        "run",
        str(selected_script),
        "--prompt",
        prompt,
        "--filename",
        final_filename,
        "--resolution",
        resolution.upper(),
    ]

    proc = subprocess.run(
        command,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or "generator failed with unknown error"
        typer.echo(f"generate failed: {detail}", err=True)
        raise typer.Exit(code=1)

    image_saved, media_token = _parse_generator_output(proc.stdout)
    public_url: str | None = None
    local_path: Path | None = None
    published_path: Path | None = None

    if media_token and re.match(r"^https?://", media_token, flags=re.IGNORECASE):
        public_url = media_token
    else:
        local_path = _resolve_local_path(image_saved=image_saved, media_token=media_token, output_dir=output_dir, filename=final_filename)
        if local_path is None:
            typer.echo("generate failed: generator succeeded but image file was not found", err=True)
            raise typer.Exit(code=1)

        dest = (publish_dir / local_path.name).resolve()
        if local_path != dest:
            shutil.copy2(local_path, dest)
        published_path = dest

        effective_base_url = base_url.strip()
        if not effective_base_url:
            raise typer.BadParameter("Missing --base-url (required when generator does not return an HTTP MEDIA URL)")
        public_url = _build_public_url(effective_base_url, dest.name, cache_bust=True)

    if check_url and public_url and re.match(r"^https?://", public_url, flags=re.IGNORECASE):
        if not _wait_for_url(public_url, timeout_sec=check_timeout, interval_sec=check_interval):
            typer.echo(f"generate failed: public URL is not reachable yet: {public_url}", err=True)
            raise typer.Exit(code=1)

    payload = {
        "ok": True,
        "prompt": prompt,
        "filename": final_filename,
        "resolution": resolution.upper(),
        "local_path": str(local_path) if local_path else (image_saved or ""),
        "published_path": str(published_path) if published_path else "",
        "public_url": public_url or "",
    }

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False))
    else:
        typer.echo(f"Image generated: {payload['filename']}")
        if payload["local_path"]:
            typer.echo(f"Local: {payload['local_path']}")
        if payload["published_path"]:
            typer.echo(f"Published: {payload['published_path']}")
        if payload["public_url"]:
            typer.echo(f"URL: {payload['public_url']}")

    if emit_media_line and payload["public_url"]:
        typer.echo(f"MEDIA: {payload['public_url']}")
