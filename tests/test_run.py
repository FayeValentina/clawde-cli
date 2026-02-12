import json
import sys

from typer.testing import CliRunner

from clawde_cli.main import app

runner = CliRunner()


def _json_from_output(output: str) -> dict:
    return json.loads(output)


def test_run_json_success():
    result = runner.invoke(
        app,
        ["run", "--json", sys.executable, "-c", "print('hello')"],
    )

    assert result.exit_code == 0
    payload = _json_from_output(result.stdout)
    assert payload["exit_code"] == 0
    assert "hello" in payload["stdout_tail"]


def test_run_nonzero_exit_code_propagates():
    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            sys.executable,
            "-c",
            "import sys; print('err', file=sys.stderr); sys.exit(3)",
        ],
    )

    assert result.exit_code == 3
    payload = _json_from_output(result.stdout)
    assert payload["exit_code"] == 3
    assert "err" in payload["stderr_tail"]


def test_run_timeout_returns_124():
    result = runner.invoke(
        app,
        ["run", "--json", "--timeout", "0.2", sys.executable, "-c", "import time; time.sleep(1.0)"],
    )

    assert result.exit_code == 124
    payload = _json_from_output(result.stdout)
    assert payload["timed_out"] is True
    assert payload["exit_code"] == 124


def test_run_retry_eventually_succeeds(tmp_path):
    marker = tmp_path / "marker.txt"
    script = (
        "from pathlib import Path\n"
        f"marker = Path(r'{marker}')\n"
        "if marker.exists():\n"
        "    print('ok')\n"
        "else:\n"
        "    marker.write_text('1', encoding='utf-8')\n"
        "    raise SystemExit(2)\n"
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--retry",
            "1",
            "--retry-delay",
            "0",
            sys.executable,
            "-c",
            script,
        ],
    )

    assert result.exit_code == 0
    payload = _json_from_output(result.stdout)
    assert payload["exit_code"] == 0
    assert payload["attempt"] == 2
    assert payload["max_attempts"] == 2
