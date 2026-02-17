import json

from typer.testing import CliRunner

from clawde_cli.main import app

runner = CliRunner()


def test_extract_outputs_structured_records(tmp_path):
    src = tmp_path / "sessions"
    dst = tmp_path / "out"
    src.mkdir(parents=True)

    session = src / "sample.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "m1",
                        "parentId": None,
                        "timestamp": "2026-02-17T01:00:00.000Z",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "id": "m2",
                        "parentId": "m1",
                        "timestamp": "2026-02-17T01:00:01.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "name": "browser",
                                    "arguments": {"action": "open"},
                                },
                                {"type": "text", "text": "done"},
                            ],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["extract", "--src", str(src), "--dst", str(dst)])
    assert result.exit_code == 0

    out_jsonl = dst / "sample.jsonl"
    out_txt = dst / "sample.txt"
    assert out_jsonl.exists()
    assert out_txt.exists()

    records = [
        json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(records) == 2
    assert records[0]["user"] == "user"
    assert records[0]["timestamp"] == "2026-02-17T01:00:00.000Z"
    assert records[0]["context"] == "hello"
    assert records[1]["user"] == "assistant"
    assert "[toolCall] browser args=" in records[1]["context"]
    assert "done" in records[1]["context"]

    text_lines = [line for line in out_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert text_lines[0].startswith("user:user,timestamp:2026-02-17T01:00:00.000Z,context:hello")
