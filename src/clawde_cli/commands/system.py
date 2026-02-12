"""System status commands."""

import typer
import psutil
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="System status checks")
console = Console()


@app.command()
def status():
    """Show system status (CPU, Memory, Disk)."""
    try:
        # interval=1 会阻塞 1 秒进行采样，得到更稳定的 CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # 读取内存与根分区磁盘使用情况
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        table = Table(
            title="🖥️  System Status", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="dim")

        table.add_row(
            "CPU Usage",
            f"{cpu_percent}%",
            f"{cpu_count} cores",
        )

        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        table.add_row(
            "Memory",
            f"{mem.percent}%",
            f"{mem_used_gb:.1f} / {mem_total_gb:.1f} GB",
        )

        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        table.add_row(
            "Disk",
            f"{disk_percent:.1f}%",
            f"{disk_used_gb:.1f} / {disk_total_gb:.1f} GB",
        )

        boot_time = psutil.boot_time()
        from datetime import datetime

        # 将系统启动时间转换为可读字符串
        boot_str = datetime.fromtimestamp(boot_time).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            "Boot Time",
            boot_str,
            "Last system start",
        )
        console.print(table)
    except psutil.Error as exc:
        console.print(f"[red]Failed to fetch system status: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]System status error: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def processes(top: int = typer.Option(10, help="Number of processes to show")):
    """Show top processes by CPU usage."""
    if top <= 0:
        console.print("[red]`top` must be greater than 0.[/red]")
        raise typer.Exit(code=1)

    try:
        procs = []

        # 先预热 CPU 采样；首次读取通常是 0.0，不具参考意义
        warmup_procs = list(psutil.process_iter(["pid", "name"]))
        for proc in warmup_procs:
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # 第二轮读取时再采集可用指标
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                procs.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # 按 CPU 占用降序排序，取前 N 条
        procs.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)

        table = Table(
            title=f"🔥 Top {top} Processes by CPU",
            show_header=True,
            header_style="bold red",
        )
        table.add_column("PID", style="cyan", justify="right")
        table.add_column("Name", style="green")
        table.add_column("CPU %", style="yellow", justify="right")
        table.add_column("Memory %", style="blue", justify="right")

        for proc in procs[:top]:
            name = proc.get("name") or "<unknown>"
            table.add_row(
                str(proc.get("pid", "?")),
                name[:30],
                f"{proc.get('cpu_percent') or 0:.1f}",
                f"{proc.get('memory_percent') or 0:.1f}",
            )

        console.print(table)
    except psutil.Error as exc:
        console.print(f"[red]Failed to fetch process list: {exc}[/red]")
        raise typer.Exit(code=1) from exc
