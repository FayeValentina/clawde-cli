# clawde-cli

Clawde's personal CLI toolkit for Linux.

This build exposes three commands:

- `generate`: generate or edit images with Nano Banana.
- `stock`: fetch a stock quote and technical snapshot.
- `system`: inspect local CPU, memory, disk, and processes.

## Installation

Install from the local workspace with `uv`:

```bash
uv tool install --force --reinstall --refresh .
```

Verify the installed command:

```bash
clawde --help
```

## Environment Variables

`generate` requires:

```bash
export NANO_BANANA_KEY="your-api-key"
```

`stock` requires:

```bash
export POLYGON_API_KEY="your-api-key"
export FINNHUB_API_KEY="your-api-key"
```

For persistent Linux shell configuration, add the exports to `~/.bashrc`,
`~/.profile`, or your shell's startup file, then reload the shell:

```bash
source ~/.bashrc
```

## Commands

### Generate Images

Generate a new image:

```bash
clawde generate "a clean product photo of a white ceramic mug" --output mug.png
```

Edit an existing image:

```bash
clawde generate "make the background transparent" --input input.png --output output.png
```

Use multiple input images by repeating `--input`:

```bash
clawde generate "combine these references into one image" -i ref-1.png -i ref-2.png -o ./outputs
```

### Stock Snapshot

Query a ticker directly:

```bash
clawde stock NVDA
```

Override the technical-analysis lookback window:

```bash
clawde stock NVDA --lookback-days 300
```

`--lookback-days` must be at least `30`; the default is `450`.

### System Status

Show CPU, memory, disk, and boot time:

```bash
clawde system status
```

Show top processes by CPU usage:

```bash
clawde system processes
```

Limit the process list:

```bash
clawde system processes --top 5
```

## Help

```bash
clawde --help
clawde generate --help
clawde stock --help
clawde system --help
```
