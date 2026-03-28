from types import SimpleNamespace

from typer.testing import CliRunner

from clawde_cli.commands import upgrade as upgrade_cmd
from clawde_cli.main import app

runner = CliRunner()


def test_upgrade_uses_installed_source_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "clawde-cli"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "clawde-cli"\n',
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(upgrade_cmd, "_probe_uv", lambda: (True, None))
    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: repo_root)
    monkeypatch.setattr(upgrade_cmd, "_find_uv_command", lambda: "/opt/homebrew/bin/uv")

    def fake_run(args, capture_output=True, text=True):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(upgrade_cmd.subprocess, "run", fake_run)

    result = runner.invoke(app, ["upgrade"])

    assert result.exit_code == 0
    assert calls == [[
        "/opt/homebrew/bin/uv",
        "tool",
        "install",
        "--force",
        "--editable",
        "--reinstall",
        "--refresh",
        str(repo_root),
    ]]
    assert "Clawde upgrade completed" in result.stdout


def test_upgrade_falls_back_to_cwd_project_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "clawde-cli"
    nested = repo_root / "src"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "clawde-cli"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: None)

    result = upgrade_cmd._find_project_root(nested)

    assert result == repo_root


def test_upgrade_fails_when_workspace_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr(upgrade_cmd, "_probe_uv", lambda: (True, None))
    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_find_project_root", lambda start=None: None)

    result = runner.invoke(app, ["upgrade"])

    assert result.exit_code == 1
    assert "workspace not found" in result.stdout
