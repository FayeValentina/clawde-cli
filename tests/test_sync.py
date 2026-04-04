from importlib import import_module
from types import SimpleNamespace

from typer.testing import CliRunner

sync_cmd = import_module("clawde_cli.commands.sync")
from clawde_cli.main import app

runner = CliRunner()


def test_sync_runs_gemini_auth_refresh_command(monkeypatch):
    calls = []

    monkeypatch.setattr(sync_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(sync_cmd, "_find_openclaw_command", lambda: "/usr/local/bin/openclaw")
    monkeypatch.setattr(
        sync_cmd,
        "_build_tty_command",
        lambda command: ["script", "-q", "/dev/null", "sh", "-lc", "wrapped"],
    )

    def fake_run(args, text=True):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sync_cmd.subprocess, "run", fake_run)

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert calls == [["script", "-q", "/dev/null", "sh", "-lc", "wrapped"]]
    assert "Refreshing Gemini OAuth authentication" in result.stdout
    assert "Gemini auth refresh completed" in result.stdout


def test_sync_fails_when_openclaw_is_unavailable(monkeypatch):
    monkeypatch.setattr(sync_cmd, "_probe_openclaw", lambda: (False, "OpenClaw command is not installed or not in PATH."))

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "OpenClaw is unavailable" in result.stdout


def test_sync_surfaces_nonzero_exit_code(monkeypatch):
    monkeypatch.setattr(sync_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(sync_cmd, "_find_openclaw_command", lambda: "/usr/local/bin/openclaw")
    monkeypatch.setattr(sync_cmd, "_build_tty_command", lambda command: ["wrapped"])
    monkeypatch.setattr(
        sync_cmd.subprocess,
        "run",
        lambda args, text=True: SimpleNamespace(returncode=7),
    )

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 7
    assert "Gemini auth refresh failed" in result.stdout


def test_sync_module_is_importable_directly():
    sync_module = import_module("clawde_cli.commands.sync")

    assert hasattr(sync_module, "app")


def test_build_tty_command_returns_base_command_without_script(monkeypatch):
    command = ["openclaw", "models", "auth"]
    monkeypatch.setattr(sync_cmd.shutil, "which", lambda name: None)

    result = sync_cmd._build_tty_command(command)

    assert result == command
