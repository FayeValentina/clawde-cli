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
        lambda: ("ready", "Runtime running and RPC probe ok."),
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
        lambda: ("degraded", "runtime not running, LaunchAgent not loaded, RPC probe failed"),
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
    install_called = False
    wait_called = False

    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_probe_npm", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_get_current_version", lambda: next(version_calls))
    monkeypatch.setattr(
        update_cmd,
        "_build_gateway_status_summary",
        lambda: ("degraded", "runtime not running, LaunchAgent not loaded, RPC probe failed"),
    )

    def fake_install_latest():
        return True, "updated"

    def fake_install_gateway():
        nonlocal install_called
        install_called = True
        return True, "unexpected"

    def fake_wait(timeout: int = 30) -> bool:
        nonlocal wait_called
        wait_called = True
        return True

    monkeypatch.setattr(update_cmd, "_install_latest_openclaw", fake_install_latest)
    monkeypatch.setattr(update_cmd, "_install_gateway_service", fake_install_gateway)
    monkeypatch.setattr(update_cmd, "_wait_for_gateway", fake_wait)

    result = runner.invoke(app, ["update", "--skip-launchagent"])

    assert result.exit_code == 0
    assert install_called is False
    assert wait_called is False
    assert "skipped by request" in result.stdout
    assert "gateway verification were skipped" in result.stdout


def test_gateway_ready_matches_openclaw_documented_baseline():
    status = "Runtime: running\nRPC probe: ok\nService: LaunchAgent (not loaded)\n"

    assert update_cmd._gateway_ready_via_service(status) is True


def test_install_latest_openclaw_uses_npm(monkeypatch):
    calls = []

    monkeypatch.setattr(update_cmd, "_find_npm_command", lambda: "/opt/homebrew/bin/npm")

    def fake_run(args, capture_output=False, text=True):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    success, message = update_cmd._install_latest_openclaw()

    assert success is True
    assert "updated via npm" in message
    assert calls == [["/opt/homebrew/bin/npm", "install", "-g", "openclaw@latest"]]


def test_update_skips_repair_when_gateway_is_healthy_after_update(monkeypatch):
    version_calls = iter(["openclaw 1.0.0", "openclaw 1.1.0"])
    install_called = False
    wait_called = False

    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_probe_npm", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_get_current_version", lambda: next(version_calls))
    monkeypatch.setattr(
        update_cmd,
        "_build_gateway_status_summary",
        lambda: ("ready", "Runtime running and RPC probe ok."),
    )

    def fake_install_latest():
        return True, "updated"

    def fake_install_gateway():
        nonlocal install_called
        install_called = True
        return True, "Gateway LaunchAgent installed"

    def fake_wait(timeout: int = 30) -> bool:
        nonlocal wait_called
        wait_called = True
        return True

    monkeypatch.setattr(update_cmd, "_install_latest_openclaw", fake_install_latest)
    monkeypatch.setattr(update_cmd, "_install_gateway_service", fake_install_gateway)
    monkeypatch.setattr(update_cmd, "_wait_for_gateway", fake_wait)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert install_called is True
    assert wait_called is True
    assert "Gateway LaunchAgent installed" in result.stdout


def test_update_fails_when_npm_is_unavailable(monkeypatch):
    monkeypatch.setattr(update_cmd, "_probe_openclaw", lambda: (True, None))
    monkeypatch.setattr(update_cmd, "_probe_npm", lambda: (False, "npm is not installed or not in PATH."))

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "npm is unavailable" in result.stdout


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
