"""Nano Banana image generation/editing commands."""

from __future__ import annotations

import base64
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import typer
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from urllib3.util.retry import Retry

console = Console()

MODEL_NAME = "gemini-3.1-flash-image-preview"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
HTTP_TIMEOUT_SECONDS = 180
DEFAULT_FILENAME_PREFIX = "clawde-generate"


def _build_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _build_session()


def _resolve_api_key() -> str:
    api_key = os.getenv("NANO_BANANA_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "NANO_BANANA_KEY is not set. Add it to your shell environment and reload ~/.zshrc."
        )
    return api_key


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    suffix = path.suffix.lower()
    if suffix == ".jpg":
        return "image/jpeg"
    if suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    raise ValueError(f"Unsupported image type for input file: {path}")


def _build_contents(prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for path in image_paths:
        raw = path.read_bytes()
        parts.append(
            {
                "inlineData": {
                    "mimeType": _guess_mime_type(path),
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            }
        )
    return [{"parts": parts}]


def _build_payload(prompt: str, image_paths: list[Path]) -> dict[str, Any]:
    return {
        "contents": _build_contents(prompt, image_paths),
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }


def _request_image_generation(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE}/{MODEL_NAME}:generateContent"
    response = SESSION.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=HTTP_TIMEOUT_SECONDS,
    )

    try:
        data = response.json()
    except ValueError as exc:
        response.raise_for_status()
        raise RuntimeError("Gemini returned a non-JSON response.") from exc

    if response.status_code >= 400:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = error.get("message") or response.text or "unknown error"
        raise RuntimeError(f"Gemini request failed: {message}")

    return data


def _extract_text_parts(response_data: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for candidate in response_data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return texts


def _extract_image_parts(response_data: dict[str, Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for candidate in response_data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue
            data = inline_data.get("data")
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type")
            if data and mime_type:
                images.append({"data": data, "mime_type": mime_type})
    return images


def _extension_for_mime_type(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".bin"


def _default_output_dir() -> Path:
    return Path.cwd()


def _resolve_output_paths(output: Path | None, images: list[dict[str, str]]) -> list[Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    total = len(images)

    if output is None:
        directory = _default_output_dir()
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, image in enumerate(images, start=1):
            suffix = _extension_for_mime_type(image["mime_type"])
            serial = f"-{index}" if total > 1 else ""
            paths.append(directory / f"{DEFAULT_FILENAME_PREFIX}-{timestamp}{serial}{suffix}")
        return paths

    if output.suffix and total == 1:
        output.parent.mkdir(parents=True, exist_ok=True)
        return [output]

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(images, start=1):
        suffix = _extension_for_mime_type(image["mime_type"])
        paths.append(output / f"{DEFAULT_FILENAME_PREFIX}-{timestamp}-{index}{suffix}")
    return paths


def _save_images(images: list[dict[str, str]], output: Path | None) -> list[Path]:
    paths = _resolve_output_paths(output, images)
    for image, path in zip(images, paths, strict=True):
        raw = base64.b64decode(image["data"])
        path.write_bytes(raw)
    return paths


def _render_summary(prompt: str, image_paths: list[Path], saved_paths: list[Path], texts: list[str]) -> None:
    mode = "Image Edit" if image_paths else "Image Generation"
    lines = [
        f"[bold]Model:[/bold] {MODEL_NAME}",
        f"[bold]Mode:[/bold] {mode}",
        f"[bold]Prompt:[/bold] {prompt}",
    ]
    if image_paths:
        lines.append("[bold]Inputs:[/bold] " + ", ".join(str(path) for path in image_paths))
    lines.append("[bold]Saved:[/bold] " + ", ".join(str(path) for path in saved_paths))
    if texts:
        lines.append("\n[bold]Model notes:[/bold]\n" + "\n".join(texts))
    console.print(
        Panel(
            "\n".join(lines),
            title="Nano Banana",
            border_style="green",
        )
    )


def _render_files_table(saved_paths: list[Path]) -> None:
    table = Table(title="Generated Files", header_style="bold cyan")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Path", style="green")
    table.add_column("Bytes", style="yellow", justify="right")
    for index, path in enumerate(saved_paths, start=1):
        table.add_row(str(index), str(path), str(path.stat().st_size))
    console.print(table)


def generate_command(
    prompt: str = typer.Argument(..., help="Prompt describing the image to generate or edit"),
    input_images: list[Path] = typer.Option(
        None,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Input image(s) for editing. Repeatable.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        resolve_path=True,
        help="Output file (single image) or output directory",
    ),
):
    """Generate or edit an image with Nano Banana."""
    try:
        api_key = _resolve_api_key()
        image_paths = input_images or []
        payload = _build_payload(prompt, image_paths)
        response_data = _request_image_generation(api_key, payload)
        images = _extract_image_parts(response_data)
        texts = _extract_text_parts(response_data)
        if not images:
            details = "\n".join(texts).strip()
            if details:
                raise RuntimeError(f"Gemini returned no image data. Response text:\n{details}")
            raise RuntimeError("Gemini returned no image data.")
        saved_paths = _save_images(images, output)
        _render_summary(prompt, image_paths, saved_paths, texts)
        _render_files_table(saved_paths)
    except ValueError as exc:
        console.print(Panel(str(exc), title="Error", border_style="red"))
        raise typer.Exit(code=1) from exc
    except requests.RequestException as exc:
        console.print(Panel(f"Network error: {exc}", title="Error", border_style="red"))
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(Panel(str(exc), title="Error", border_style="red"))
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(Panel(f"File error: {exc}", title="Error", border_style="red"))
        raise typer.Exit(code=1) from exc
