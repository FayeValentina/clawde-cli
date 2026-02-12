import json

from typer.testing import CliRunner

from clawde_cli.commands import doctor as doctor_cmd
from clawde_cli.main import app

runner = CliRunner()


def _configure_doctor_paths(monkeypatch, tmp_path, create_children: bool) -> None:
    openclaw_root = tmp_path / ".openclaw"
    workspace = openclaw_root / "workspace"
    workspace.mkdir(parents=True)

    if create_children:
        for name in ("notes", "memory", "logs"):
            (workspace / name).mkdir()

    monkeypatch.setattr(doctor_cmd, "OPENCLAW_ROOT", openclaw_root)
    monkeypatch.setattr(doctor_cmd, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(
        doctor_cmd,
        "CHECK_DIRS",
        [workspace, workspace / "notes", workspace / "memory", workspace / "logs"],
    )


def test_doctor_json_no_network_success(monkeypatch, tmp_path):
    _configure_doctor_paths(monkeypatch, tmp_path, create_children=True)

    result = runner.invoke(app, ["doctor", "--json", "--no-network"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    checks = {item["id"]: item for item in payload}
    assert checks["dns"]["details"]["skipped"] is True
    assert checks["http_open_meteo"]["details"]["skipped"] is True


def test_doctor_strict_fails_on_warning(monkeypatch, tmp_path):
    _configure_doctor_paths(monkeypatch, tmp_path, create_children=False)

    result = runner.invoke(app, ["doctor", "--json", "--no-network", "--strict"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    checks = {item["id"]: item for item in payload}
    assert checks["workspace"]["severity"] == "warn"


def test_doctor_strict_passes_when_no_warnings(monkeypatch, tmp_path):
    _configure_doctor_paths(monkeypatch, tmp_path, create_children=True)

    result = runner.invoke(app, ["doctor", "--json", "--no-network", "--strict"])

    assert result.exit_code == 0
