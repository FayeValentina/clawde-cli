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


def test_search_vector_fallback_triggers_when_results_insufficient(tmp_path):
    src = tmp_path / "sessions_text"
    src.mkdir(parents=True)

    records = [
        {
            "user": "user",
            "timestamp": "2026-02-17T01:00:00.000Z",
            "context": "system breakage in staging",
            "role": "user",
            "message_id": "m1",
            "parent_id": "",
        },
        {
            "user": "assistant",
            "timestamp": "2026-02-17T01:00:01.000Z",
            "context": "critical system outage happened",
            "role": "assistant",
            "message_id": "m2",
            "parent_id": "m1",
        },
    ]
    sample = src / "sample.jsonl"
    sample.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "search",
            "outtage",
            "--src",
            str(src),
            "--no-qmd",
            "--vector",
            "--vector-trigger-below",
            "5",
            "--vector-limit",
            "5",
            "--vector-min-score",
            "0.2",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["counts"]["grep"] == 0
    assert payload["counts"]["vector_fallback"] >= 1
    assert payload["results"][0]["source"] == "vector"


def test_search_vector_fallback_skips_when_results_sufficient(tmp_path):
    src = tmp_path / "sessions_text"
    src.mkdir(parents=True)

    records = [
        {
            "user": "user",
            "timestamp": "2026-02-17T01:00:00.000Z",
            "context": "hello world one",
            "role": "user",
            "message_id": "m1",
            "parent_id": "",
        },
        {
            "user": "assistant",
            "timestamp": "2026-02-17T01:00:01.000Z",
            "context": "hello world two",
            "role": "assistant",
            "message_id": "m2",
            "parent_id": "m1",
        },
    ]
    sample = src / "sample.jsonl"
    sample.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "search",
            "hello",
            "--src",
            str(src),
            "--no-qmd",
            "--vector",
            "--vector-trigger-below",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["counts"]["grep"] == 2
    assert payload["counts"]["vector_fallback"] == 0


def test_search_vector_auto_reduce_scan_limit(tmp_path):
    src = tmp_path / "sessions_text"
    src.mkdir(parents=True)

    for i in range(3):
        record = {
            "user": "user",
            "timestamp": f"2026-02-17T01:00:0{i}.000Z",
            "context": "outage alert in cluster",
            "role": "user",
            "message_id": f"m{i}",
            "parent_id": "",
        }
        (src / f"s{i}.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "search",
            "outtage",
            "--src",
            str(src),
            "--no-qmd",
            "--vector",
            "--vector-auto-reduce-files",
            "1",
            "--vector-scan-limit",
            "5000",
            "--vector-auto-reduced-scan-limit",
            "700",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["vector"]["auto_reduced"] is True
    assert payload["vector"]["effective_scan_limit"] == 700
    assert payload["vector"]["session_file_count"] == 3
