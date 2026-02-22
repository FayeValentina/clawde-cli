import json
from pathlib import Path

from typer.testing import CliRunner

from clawde_cli.main import app

runner = CliRunner()


def test_generate_uses_http_media_directly(monkeypatch, tmp_path):
    script = tmp_path / "generate_image.py"
    script.write_text("# stub", encoding="utf-8")

    class _Proc:
        returncode = 0
        stdout = (
            "Generating image with resolution 1K...\n"
            "Image saved: /tmp/ignored.png\n"
            "MEDIA: https://example.com/media/demo.png?v=1\n"
        )
        stderr = ""

    def _fake_run(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr("clawde_cli.commands.generate.subprocess.run", _fake_run)

    result = runner.invoke(
        app,
        [
            "generate",
            "test prompt",
            "--script",
            str(script),
            "--no-check-url",
        ],
    )

    assert result.exit_code == 0
    assert "MEDIA: https://example.com/media/demo.png?v=1" in result.stdout


def test_generate_publishes_local_file_to_public_url(monkeypatch, tmp_path):
    script = tmp_path / "generate_image.py"
    script.write_text("# stub", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    publish_dir = tmp_path / "public"
    publish_dir.mkdir(parents=True, exist_ok=True)

    local_image = output_dir / "demo.png"
    local_image.write_bytes(b"png-bytes")

    class _Proc:
        returncode = 0
        stdout = f"Image saved: {local_image}\nMEDIA: {local_image}\n"
        stderr = ""

    def _fake_run(*args, **kwargs):
        return _Proc()

    monkeypatch.setattr("clawde_cli.commands.generate.subprocess.run", _fake_run)

    result = runner.invoke(
        app,
        [
            "generate",
            "another prompt",
            "--script",
            str(script),
            "--output-dir",
            str(output_dir),
            "--publish-dir",
            str(publish_dir),
            "--base-url",
            "https://example.com/openclaw-media",
            "--no-check-url",
            "--json",
            "--no-emit-media-line",
            "--filename",
            "demo.png",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["published_path"] == str((publish_dir / "demo.png").resolve())
    assert payload["public_url"].startswith("https://example.com/openclaw-media/demo.png")
    assert (publish_dir / "demo.png").exists()
