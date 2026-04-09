from pathlib import Path

from typer.testing import CliRunner

from clawde_cli.commands import generate as generate_cmd
from clawde_cli.main import app

runner = CliRunner()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_build_payload_with_input_images(tmp_path):
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"png-bytes")

    payload = generate_cmd._build_payload("make it dreamy", [image_path])

    assert payload["contents"][0]["parts"][0]["text"] == "make it dreamy"
    inline_data = payload["contents"][0]["parts"][1]["inlineData"]
    assert inline_data["mimeType"] == "image/png"
    assert inline_data["data"]
    assert payload["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]


def test_generate_saves_returned_image(monkeypatch, tmp_path):
    monkeypatch.setenv("NANO_BANANA_KEY", "test-key")

    response_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "done"},
                        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
                    ]
                }
            }
        ]
    }

    def fake_request(api_key: str, payload: dict):
        assert api_key == "test-key"
        assert payload["contents"][0]["parts"][0]["text"] == "draw a cat"
        return response_payload

    monkeypatch.setattr(generate_cmd, "_request_image_generation", fake_request)

    output_file = tmp_path / "cat.png"
    result = runner.invoke(app, ["generate", "draw a cat", "--output", str(output_file)])

    assert result.exit_code == 0
    assert output_file.read_bytes() == b"hello"
    assert "Nano Banana" in result.stdout
    assert output_file.name in result.stdout


def test_generate_supports_editing_with_input_image(monkeypatch, tmp_path):
    monkeypatch.setenv("NANO_BANANA_KEY", "test-key")

    input_image = tmp_path / "base.jpg"
    input_image.write_bytes(b"jpg-bytes")

    observed_payload = {}

    def fake_request(api_key: str, payload: dict):
        observed_payload.update(payload)
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"mimeType": "image/jpeg", "data": "/9g="}},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(generate_cmd, "_request_image_generation", fake_request)

    output_file = tmp_path / "edited.jpg"
    result = runner.invoke(
        app,
        ["generate", "turn this into watercolor", "-i", str(input_image), "-o", str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    inline_data = observed_payload["contents"][0]["parts"][1]["inlineData"]
    assert inline_data["mimeType"] == "image/jpeg"


def test_generate_requires_key(monkeypatch):
    monkeypatch.delenv("NANO_BANANA_KEY", raising=False)

    result = runner.invoke(app, ["generate", "draw something"])

    assert result.exit_code == 1
    assert "NANO_BANANA_KEY is not set" in result.stdout


def test_request_image_generation_surfaces_api_error(monkeypatch):
    def fake_post(url, params, json, timeout):
        return _FakeResponse(400, {"error": {"message": "bad request"}})

    monkeypatch.setattr(generate_cmd.SESSION, "post", fake_post)

    try:
        generate_cmd._request_image_generation("k", {"contents": []})
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "bad request" in str(exc)


def test_resolve_output_directory_for_multiple_images(tmp_path):
    images = [
        {"mime_type": "image/png", "data": "aGVsbG8="},
        {"mime_type": "image/jpeg", "data": "aGVsbG8="},
    ]

    paths = generate_cmd._resolve_output_paths(tmp_path, images)

    assert len(paths) == 2
    assert all(path.parent == tmp_path for path in paths)
    assert paths[0].suffix == ".png"
    assert paths[1].suffix == ".jpg"
