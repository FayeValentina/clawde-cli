import json
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from clawde_cli.commands import log as log_cmd
from clawde_cli.main import app

runner = CliRunner()


def _configure_log_paths(monkeypatch, tmp_path):
    workspace = tmp_path / ".openclaw" / "workspace"
    log_dir = workspace / "logs"
    log_file = log_dir / "events.jsonl"
    monkeypatch.setattr(log_cmd, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(log_cmd, "LOG_DIR", log_dir)
    monkeypatch.setattr(log_cmd, "LOG_FILE", log_file)
    return log_file


def test_log_add_and_grep(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "log",
            "add",
            "--type",
            "TOOL_CALL",
            "--level",
            "info",
            "--message",
            "hello",
            "--data",
            '{"cmd":"echo hi","exit_code":0}',
        ],
    )
    assert result.exit_code == 0
    assert log_file.exists()

    grep = runner.invoke(app, ["log", "grep", "TOOL_CALL"])
    assert grep.exit_code == 0
    assert "TOOL_CALL" in grep.stdout


def test_log_add_json_output(monkeypatch, tmp_path):
    _configure_log_paths(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "log",
            "add",
            "--type",
            "TOOL_CALL",
            "--message",
            "json add",
            "--data",
            '{"cmd":"echo hi","exit_code":0}',
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["type"] == "TOOL_CALL"
    assert payload["message"] == "json add"
    assert payload["data"]["cmd"] == "echo hi"


def test_log_grep_since_filters_old_events(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    new_ts = datetime.now(timezone.utc).isoformat()
    events = [
        {"ts": old_ts, "type": "OLD", "level": "info", "message": "past", "data": {}},
        {"ts": new_ts, "type": "NEW", "level": "info", "message": "recent", "data": {}},
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for item in events:
            f.write(json.dumps(item) + "\n")

    grep = runner.invoke(app, ["log", "grep", "NEW", "--since", "1d"])
    assert grep.exit_code == 0
    assert "NEW" in grep.stdout
    assert "OLD" not in grep.stdout


def test_log_stats_counts_failures(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    events = [
        {
            "ts": now,
            "type": "TOOL_CALL",
            "level": "info",
            "message": "ok",
            "data": {"cmd": "echo ok", "exit_code": 0, "duration_ms": 10},
        },
        {
            "ts": now,
            "type": "TOOL_CALL",
            "level": "error",
            "message": "fail",
            "data": {"cmd": "bad cmd", "exit_code": 2, "duration_ms": 30},
        },
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for item in events:
            f.write(json.dumps(item) + "\n")

    stats = runner.invoke(app, ["log", "stats", "--since", "7d"])
    assert stats.exit_code == 0
    assert "Total events" in stats.stdout
    assert "Failures" in stats.stdout
    assert "Top Failed Commands" in stats.stdout
    assert "bad cmd" in stats.stdout


def test_log_tail_json_empty_when_missing_file(monkeypatch, tmp_path):
    _configure_log_paths(monkeypatch, tmp_path)

    result = runner.invoke(app, ["log", "tail", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == []


def test_log_grep_json_output(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": now,
                    "type": "TOOL_CALL",
                    "level": "info",
                    "message": "recent",
                    "data": {"cmd": "echo hi"},
                }
            )
            + "\n"
        )

    result = runner.invoke(app, ["log", "grep", "TOOL_CALL", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["type"] == "TOOL_CALL"


def test_log_stats_json_output(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": now,
                    "type": "TOOL_CALL",
                    "level": "error",
                    "message": "fail",
                    "data": {"cmd": "bad cmd", "exit_code": 2, "duration_ms": 30},
                }
            )
            + "\n"
        )

    result = runner.invoke(app, ["log", "stats", "--since", "7d", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total_events"] == 1
    assert payload["failures"] == 1
    assert payload["top_failed_commands"][0]["cmd"] == "bad cmd"


def test_log_stats_supports_list_command(monkeypatch, tmp_path):
    log_file = _configure_log_paths(monkeypatch, tmp_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with log_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": now,
                    "type": "TOOL_CALL",
                    "level": "error",
                    "message": "fail",
                    "data": {"cmd": ["python", "-c", "print(1)"], "exit_code": 2, "duration_ms": 30},
                }
            )
            + "\n"
        )

    result = runner.invoke(app, ["log", "stats", "--since", "7d", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["top_failed_commands"][0]["cmd"] == "python -c 'print(1)'"
