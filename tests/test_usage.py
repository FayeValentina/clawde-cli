from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from clawde_cli.commands import usage as usage_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_usage_prints_codex_windows(monkeypatch):
    monkeypatch.setattr(
        usage_cmd,
        "_load_codex_auth_entries",
        lambda: [usage_cmd.CodexAuthEntry(label="current", access_token="token", account_id="acct")],
    )
    monkeypatch.setattr(
        usage_cmd,
        "_fetch_codex_usage",
        lambda access_token, account_id: {
            "plan_type": "plus",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
                "primary_window": {
                    "used_percent": 33,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 7200,
                },
                "secondary_window": {
                    "used_percent": 15,
                    "limit_window_seconds": 604800,
                    "reset_after_seconds": 345600,
                },
            },
            "code_review_rate_limit": {
                "allowed": True,
                "limit_reached": False,
                "primary_window": {
                    "used_percent": 0,
                    "limit_window_seconds": 604800,
                    "reset_after_seconds": 604800,
                },
                "secondary_window": None,
            },
        },
    )

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0
    assert "Codex Usage" in result.stdout
    assert "Plan: plus" in result.stdout
    assert "Codex" in result.stdout
    assert "5h: 67% left" in result.stdout
    assert "Week: 85% left" in result.stdout
    assert "Code Review" in result.stdout


def test_usage_fails_when_auth_is_missing(monkeypatch):
    monkeypatch.setattr(
        usage_cmd,
        "_load_codex_auth_entries",
        lambda: (_ for _ in ()).throw(RuntimeError("OpenAI Codex is not logged in")),
    )

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 1
    assert "Unable to load Codex usage" in result.stdout
    assert "OpenAI Codex is not logged in" in result.stdout


def test_load_codex_auth_reads_hermes_auth_file(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        '{"providers":{"openai-codex":{"tokens":{"access_token":"abc","account_id":"acct-1"}}}}'
    )

    access_token, account_id = usage_cmd._load_codex_auth(auth_path)

    assert access_token == "abc"
    assert account_id == "acct-1"


def test_load_codex_auth_errors_without_access_token(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"providers":{"openai-codex":{"tokens":{}}}}')

    try:
        usage_cmd._load_codex_auth(auth_path)
    except RuntimeError as exc:
        assert "OpenAI Codex is not logged in" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when access token is missing")


def test_load_codex_auth_entries_reads_pool_and_dedupes_current(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        '{'
        '"providers":{"openai-codex":{"tokens":{"access_token":"token-a","account_id":"acct-1"}}},'
        '"credential_pool":{"openai-codex":['
        '{"label":"primary","access_token":"token-a"},'
        '{"label":"backup","access_token":"token-b","account_id":"acct-2"}'
        ']}'
        '}'
    )

    entries = usage_cmd._load_codex_auth_entries(auth_path)

    assert entries == [
        usage_cmd.CodexAuthEntry(label="primary", access_token="token-a", account_id="acct-1"),
        usage_cmd.CodexAuthEntry(label="backup", access_token="token-b", account_id="acct-2"),
    ]


def test_usage_prints_multiple_accounts(monkeypatch):
    monkeypatch.setattr(
        usage_cmd,
        "_load_codex_auth_entries",
        lambda: [
            usage_cmd.CodexAuthEntry(label="primary", access_token="token-a", account_id="acct-1"),
            usage_cmd.CodexAuthEntry(label="backup", access_token="token-b", account_id="acct-2"),
        ],
    )
    monkeypatch.setattr(
        usage_cmd,
        "_fetch_codex_usage",
        lambda access_token, account_id: {
            "plan_type": "plus",
            "rate_limit": {
                "allowed": True,
                "limit_reached": False,
                "primary_window": {
                    "used_percent": 25 if access_token == "token-a" else 60,
                    "limit_window_seconds": 18000,
                    "reset_after_seconds": 7200,
                },
            },
        },
    )

    result = runner.invoke(app, ["usage"])

    assert result.exit_code == 0
    assert "Codex Usage · primary" in result.stdout
    assert "Codex Usage · backup" in result.stdout
    assert "Account: primary" in result.stdout
    assert "Account: backup" in result.stdout
    assert "5h: 75% left" in result.stdout
    assert "5h: 40% left" in result.stdout


def test_format_reset_outputs_compact_windows():
    assert usage_cmd._format_reset(59) == "<1m"
    assert usage_cmd._format_reset(65) == "1m"
    assert usage_cmd._format_reset(7260) == "2h 1m"
    assert usage_cmd._format_reset(176400) == "2d 1h"
