"""Clawde CLI - Personal toolkit for automation."""

import typer
from rich.console import Console
from rich.panel import Panel

from clawde_cli.commands import doctor, extract, log, memory, run, search, stock, system, update, upgrade, weather

app = typer.Typer(
    name="clawde",
    help="🐾 Clawde's personal CLI toolkit",
    no_args_is_help=True,
)
console = Console()

# 注册子命令组，统一挂载到 `clawde` 根命令下
app.add_typer(system.app, name="system", help="System status checks")
app.add_typer(weather.app, name="weather", help="Weather queries")
app.add_typer(memory.app, name="memory", help="Memory log operations")
app.add_typer(stock.app, name="stock", help="Stock quote and technical snapshot")
app.add_typer(extract.app, name="extract", help="Extract session utterances")
app.add_typer(doctor.app, name="doctor", help="Environment diagnostics")
app.add_typer(run.app, name="run", help="Reliable command execution")
app.add_typer(log.app, name="log", help="Event log operations")
app.add_typer(update.app, name="update", help="Safe OpenClaw update wrapper")
app.add_typer(upgrade.app, name="upgrade", help="Upgrade the global Clawde CLI from the original local workspace")
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
