---
title: Every other tool
nav_order: 4
description: One command sets up AttogradDB memory for Codex, Gemini CLI, OpenCode, and any MCP host
---

# Every other tool

Cursor and Claude Code have a [plugin](agents.md). For everything else — Codex, Gemini CLI,
OpenCode, and any other MCP host — one command does the whole setup for you.

## Two steps

```bash
pip install attogradDB
attograddb setup
```

`attograddb setup` detects the AI coding tools installed on your machine, asks which ones to
configure and whether to set them up for **this project** or **globally**, then writes each
tool's MCP server config and an instructions file so the agent actually calls the memory
tools. No manual editing.

```text
$ attograddb setup
Detected these tools:
  1. Codex
  2. Gemini CLI
  3. OpenCode
Configure which? [numbers, comma-separated, or 'all']: all

Install scope:
  1. This project (writes into the current directory)
  2. Global (writes into your home config for every project)
Scope? [1/2, default 1]: 1

Will configure Codex, Gemini CLI, OpenCode at project scope.
Proceed? [y/N]: y
```

Restart the tool (or start a new session) afterwards. Then ask it to call `recall_decisions`
before it plans — or rely on the instructions file the setup wrote.

### Non-interactive

Everything can be passed as flags, for scripts or dotfiles:

```bash
attograddb setup --tools codex,gemini,opencode --scope project --yes
attograddb setup --tools cursor --scope global --yes
```

`--tools` accepts any of `codex`, `gemini`, `opencode`, `cursor`, `claude`. `--project-dir`
overrides the target project (default: current directory).

## Requirements

- Python 3.11+.
- Optional: [uv](https://docs.astral.sh/uv/getting-started/installation/). If `uvx` is on
  your PATH, the generated config launches the server with it (self-contained and
  version-pinned). Otherwise it uses the Python you installed AttogradDB into.

## What it writes

Project scope writes into the current directory; global scope writes into your home config.

- **Codex** — `mcp_servers.attograd-memory` in `.codex/config.toml` (or `~/.codex/config.toml`),
  plus an instructions block in `AGENTS.md`.
- **Gemini CLI** — `mcpServers.attograd-memory` in `.gemini/settings.json` (or
  `~/.gemini/settings.json`), plus `GEMINI.md`.
- **OpenCode** — `mcp.attograd-memory` (`type: "local"`) in `opencode.json` (or
  `~/.config/opencode/opencode.json`), plus `AGENTS.md`.
- **Cursor** — `mcpServers.attograd-memory` in `.cursor/mcp.json` (or `~/.cursor/mcp.json`),
  plus a `.cursor/rules/attograd-memory.mdc` rule. Use this only if the
  [plugin](agents.md) is unavailable.
- **Claude Code** — `mcpServers.attograd-memory` in `.mcp.json` (or `~/.claude.json`), plus a
  `CLAUDE.md` block. Use this only if the [plugin](agents.md) is unavailable.

Re-running `attograddb setup` is safe: it replaces its own entry and leaves the rest of each
file untouched.

## Manual MCP (Hermes, Pi, anything else)

Any host that launches a local stdio MCP server can use AttogradDB directly. The server takes
an optional `--project-root`; omit it and it uses `$CLAUDE_PROJECT_DIR` or the working
directory.

```bash
attograddb-mcp --project-root /path/to/project
```

The equivalent host config, using `uvx` so nothing needs to be pre-installed:

```json
{
  "mcpServers": {
    "attograd-memory": {
      "command": "uvx",
      "args": [
        "--from",
        "attogradDB[mcp]==1.1.0",
        "attograddb-mcp",
        "--project-root",
        "/path/to/project"
      ]
    }
  }
}
```

- **Hermes** — `mcp_servers:` map in `~/.hermes/config.yaml` (Linux/macOS/WSL2). Reload MCP
  or start a new session after editing.
- **Pi** — with the `pi-mcp-adapter` extension installed, the standard `mcpServers` map in
  `~/.pi/agent/mcp.json` (global) or `.pi/mcp.json` (project).

Pass the real project directory, not a shared global folder, so each workspace gets its own
`.attograd-memory.db`.
