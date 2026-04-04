from pathlib import Path

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
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    monkeypatch.setattr(upgrade_cmd, "_probe_uv", lambda: (True, None))
    monkeypatch.setattr(upgrade_cmd, "_get_env_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_get_persisted_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_find_project_root", lambda start=None: None)
    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: repo_root)

    class FakeTemporaryDirectory:
        def __init__(self, path: Path):
            self.path = path

        def __enter__(self):
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    materialize_calls = []
    install_calls = []

    monkeypatch.setattr(
        upgrade_cmd.tempfile,
        "TemporaryDirectory",
        lambda prefix="": FakeTemporaryDirectory(snapshot_root),
    )

    def fake_materialize(project_root_arg, snapshot_root_arg):
        materialize_calls.append((project_root_arg, snapshot_root_arg))
        return True, "prepared"

    def fake_install(project_root_arg, snapshot_root_arg):
        install_calls.append((project_root_arg, snapshot_root_arg))
        return True, "installed"

    monkeypatch.setattr(upgrade_cmd, "_materialize_tracked_snapshot", fake_materialize)
    monkeypatch.setattr(upgrade_cmd, "_install_workspace_snapshot", fake_install)
    monkeypatch.setattr(upgrade_cmd, "_persist_source_root", lambda project_root_arg: (True, None))

    result = runner.invoke(app, ["upgrade"])

    assert result.exit_code == 0
    assert materialize_calls == [(repo_root, snapshot_root / "clawde-cli")]
    assert install_calls == [(repo_root, snapshot_root / "clawde-cli")]
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


def test_resolve_upgrade_source_prefers_persisted_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "clawde-cli"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "clawde-cli"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(upgrade_cmd, "_get_env_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_get_persisted_source_root", lambda: repo_root)
    monkeypatch.setattr(upgrade_cmd, "_find_project_root", lambda start=None: None)
    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: None)

    result = upgrade_cmd._resolve_upgrade_source()

    assert result == repo_root


def test_materialize_tracked_snapshot_copies_tracked_files(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "clawde-cli"\n',
        encoding="utf-8",
    )
    source_file = repo_root / "src" / "clawde_cli" / "main.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("print('ok')\n", encoding="utf-8")
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    monkeypatch.setattr(
        upgrade_cmd,
        "_tracked_workspace_files",
        lambda project_root: (True, [Path("src/clawde_cli/main.py")]),
    )

    success, message = upgrade_cmd._materialize_tracked_snapshot(repo_root, snapshot_root)

    assert success is True
    assert "Prepared tracked snapshot" in message
    assert (snapshot_root / "src" / "clawde_cli" / "main.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_materialize_tracked_snapshot_includes_project_readme(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "clawde-cli"\nreadme = "README.md"\n',
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    monkeypatch.setattr(upgrade_cmd, "_tracked_workspace_files", lambda project_root: (True, []))

    success, message = upgrade_cmd._materialize_tracked_snapshot(repo_root, snapshot_root)

    assert success is True
    assert "Prepared tracked snapshot" in message
    assert (snapshot_root / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_upgrade_fails_when_workspace_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr(upgrade_cmd, "_probe_uv", lambda: (True, None))
    monkeypatch.setattr(upgrade_cmd, "_get_env_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_get_persisted_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_get_installed_source_root", lambda: None)
    monkeypatch.setattr(upgrade_cmd, "_find_project_root", lambda start=None: None)

    result = runner.invoke(app, ["upgrade"])

    assert result.exit_code == 1
    assert "workspace not found" in result.stdout
