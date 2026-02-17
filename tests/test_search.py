import json

from typer.testing import CliRunner

from clawde_cli.main import app

runner = CliRunner()


def test_search_no_qmd_returns_exact_matches(tmp_path):
    src = tmp_path / "sessions_text"
    src.mkdir(parents=True)

    records = [
        {
            "user": "user",
            "timestamp": "2026-02-17T01:00:00.000Z",
            "context": "hello clawde world",
            "role": "user",
            "message_id": "m1",
            "parent_id": "",
        },
        {
            "user": "assistant",
            "timestamp": "2026-02-17T01:00:01.000Z",
            "context": "something else",
            "role": "assistant",
            "message_id": "m2",
            "parent_id": "m1",
        },
        {
            "user": "assistant",
            "timestamp": "2026-02-17T01:00:02.000Z",
            "context": "HELLO from assistant",
            "role": "assistant",
            "message_id": "m3",
            "parent_id": "m2",
        },
    ]
    sample = src / "sample.jsonl"
    sample.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["search", "hello", "--src", str(src), "--no-qmd", "--json"],
    )
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["counts"]["grep"] == 2
    assert payload["counts"]["qmd_refined"] == 0
    assert payload["counts"]["merged"] == 2
    assert payload["results"][0]["timestamp"] == "2026-02-17T01:00:02.000Z"
    assert payload["results"][1]["timestamp"] == "2026-02-17T01:00:00.000Z"
    assert payload["results"][0]["source"] == "grep"
