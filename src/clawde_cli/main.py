"""Clawde CLI - Personal toolkit for automation."""

import typer
from rich.console import Console
from rich.panel import Panel

from clawde_cli.commands import search
from clawde_cli.commands.doctor import app as doctor_app
from clawde_cli.commands.extract import app as extract_app
from clawde_cli.commands.log import app as log_app
from clawde_cli.commands.memory import app as memory_app
from clawde_cli.commands.run import app as run_app
from clawde_cli.commands.stock import app as stock_app
from clawde_cli.commands.sync import app as sync_app
from clawde_cli.commands.system import app as system_app
from clawde_cli.commands.update import app as update_app
from clawde_cli.commands.upgrade import app as upgrade_app
from clawde_cli.commands.usage import app as usage_app
from clawde_cli.commands.weather import app as weather_app

app = typer.Typer(
    name="clawde",
    help="🐾 Clawde's personal CLI toolkit",
    no_args_is_help=True,
)
console = Console()

# 注册子命令组，统一挂载到 `clawde` 根命令下
app.add_typer(system_app, name="system", help="System status checks")
app.add_typer(weather_app, name="weather", help="Weather queries")
app.add_typer(memory_app, name="memory", help="Memory log operations")
app.add_typer(stock_app, name="stock", help="Stock quote and technical snapshot")
app.add_typer(extract_app, name="extract", help="Extract session utterances")
app.add_typer(doctor_app, name="doctor", help="Environment diagnostics")
app.add_typer(run_app, name="run", help="Reliable command execution")
app.add_typer(log_app, name="log", help="Event log operations")
app.add_typer(update_app, name="update", help="Safe OpenClaw update wrapper")
app.add_typer(upgrade_app, name="upgrade", help="Upgrade the global Clawde CLI from the original local workspace")
app.add_typer(sync_app, name="sync", help="Refresh Gemini OAuth authentication")
app.add_typer(usage_app, name="usage", help="Show Codex quota usage")
app.command(name="search", help="Search session utterances")(search.search_command)


@app.callback()
def callback():
    """🐾 Clawde's personal CLI toolkit."""
    pass


@app.command()
def hello():
    """Say hello from Clawde."""
    console.print(
        Panel(
            "[bold cyan]Hello! I'm Clawde 🐾[/bold cyan]\n"
            "Your digital assistant CLI toolkit is ready!",
            title="Welcome",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
