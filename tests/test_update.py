import subprocess
from types import SimpleNamespace

from typer.testing import CliRunner

from clawde_cli.commands import update as update_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_update_status_flag_prints_version_and_gateway_status(monkeypatch):
    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_get_current_version", lambda: "openclaw 1.2.3")
    monkeypatch.setattr(
        update_cmd,
        "_build_gateway_status_summary",
        lambda: ("ready", "LaunchAgent loaded, RPC probe ok, and gateway is listening."),
    )

    result = runner.invoke(app, ["update", "--status"])

    assert result.exit_code == 0
    assert "Current OpenClaw version:" in result.stdout
    assert "openclaw 1.2.3" in result.stdout
    assert "Gateway status:" in result.stdout
    assert "ready" in result.stdout
    assert "repair the gateway service" in result.stdout


def test_update_status_subcommand_prints_version_and_gateway_status(monkeypatch):
    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_get_current_version", lambda: "openclaw 2.0.0")
    monkeypatch.setattr(
        update_cmd,
        "_build_gateway_status_summary",
        lambda: ("degraded", "LaunchAgent not loaded, RPC probe failed, gateway not listening"),
    )

    result = runner.invoke(app, ["update", "status"])

    assert result.exit_code == 0
    assert "Current OpenClaw version:" in result.stdout
    assert "openclaw 2.0.0" in result.stdout
    assert "Gateway status:" in result.stdout
    assert "degraded" in result.stdout
    assert "clawde update" in result.stdout


def test_update_skip_launchagent_skips_repair_and_wait(monkeypatch):
    version_calls = iter(["openclaw 1.0.0", "openclaw 1.1.0"])
    repair_called = False
    wait_called = False

    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_get_current_version", lambda: next(version_calls))
    monkeypatch.setattr(
        update_cmd,
        "_build_gateway_status_summary",
        lambda: ("degraded", "LaunchAgent not loaded, RPC probe failed, gateway not listening"),
    )

    def fake_reinstall():
        nonlocal repair_called
        repair_called = True
        return True, "unexpected"

    def fake_wait(timeout: int = 30) -> bool:
        nonlocal wait_called
        wait_called = True
        return True

    monkeypatch.setattr(update_cmd, "_reinstall_launchagent", fake_reinstall)
    monkeypatch.setattr(update_cmd, "_wait_for_gateway", fake_wait)
    monkeypatch.setattr(update_cmd, "_find_openclaw_command", lambda: "/usr/local/bin/openclaw")
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = runner.invoke(app, ["update", "--skip-launchagent"])

    assert result.exit_code == 0
    assert repair_called is False
    assert wait_called is False
    assert "skipped by request" in result.stdout
    assert "gateway verification were skipped" in result.stdout


def test_probe_openclaw_reports_broken_command(monkeypatch):
    monkeypatch.setattr(update_cmd.shutil, "which", lambda name: "/usr/local/bin/openclaw")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr="broken install",
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    ok, message = update_cmd._probe_openclaw()

    assert ok is False
    assert message is not None
    assert "present but failed to run" in message
    assert "broken install" in message
