"""Clawde CLI - Personal toolkit for automation."""

import typer

from clawde_cli.commands.generate import generate_command
from clawde_cli.commands.stock import stock_quote
from clawde_cli.commands.system import app as system_app

app = typer.Typer(
    name="clawde",
    help="🐾 Clawde's personal CLI toolkit",
    no_args_is_help=True,
)

app.add_typer(system_app, name="system", help="System status checks")
app.command(name="generate", help="Generate or edit images with Nano Banana")(generate_command)
app.command(name="stock", help="Stock quote and technical snapshot")(stock_quote)


@app.callback()
def callback():
    """🐾 Clawde's personal CLI toolkit."""
    pass


if __name__ == "__main__":
    app()
