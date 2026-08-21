---
title: Other harnesses
nav_order: 4
description: Use AttogradDB from OpenCode, Hermes, Pi, or any stdio MCP client
---

# Other harnesses

Any host that can launch a local stdio MCP server can use AttogradDB. There is no
Hermes, OpenCode, or Pi plugin listing. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then point the host at `uvx`. The server still takes `--project-root` so each
workspace gets its own `.attograd-memory.db`.

The tools are the same everywhere: `recall_decisions` and `remember_decision`.
Call recall before planning or editing. See [Agent plugins](agents.md) for the
tool contract.

Generic command:

```bash
uvx --from "attogradDB[mcp]==1.0.1" attograddb-mcp --project-root /path/to/project
```

If the host starts the server with the project as its working directory, `.` is a
valid `--project-root`.

## OpenCode

Project file `opencode.jsonc`, or global `~/.config/opencode/opencode.jsonc`.
OpenCode v2 nests servers under `mcp.servers`. `command` is one array. Default
Code Mode wraps MCP tools; set `codemode` to `false` so recall and remember stay
on the model's tool list.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "attograd-memory": {
        "type": "local",
        "command": [
          "uvx",
          "--from",
          "attogradDB[mcp]==1.0.1",
          "attograddb-mcp",
          "--project-root",
          "."
        ],
        "cwd": ".",
        "codemode": false
      }
    }
  }
}
```

## Hermes

`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  attograd-memory:
    command: uvx
    args:
      - --from
      - attogradDB[mcp]==1.0.1
      - attograddb-mcp
      - --project-root
      - .
```

Prefer editing the YAML. `hermes mcp add attograd-memory --command uvx` only sets
the executable; add the `args` list in `config.yaml` if the CLI does not take them.
Reload MCP or start a new session after editing.

## Pi

Put the standard MCP map in `~/.pi/agent/mcp.json` (user-global) or `.mcp.json`
(project). Some Pi builds also read `.pi/mcp.json`.

```json
{
  "mcpServers": {
    "attograd-memory": {
      "command": "uvx",
      "args": [
        "--from",
        "attogradDB[mcp]==1.0.1",
        "attograddb-mcp",
        "--project-root",
        "."
      ]
    }
  }
}
```

If a Pi extension requires `"transport": "stdio"`, add that field. The command
and args stay the same.

## Any other stdio MCP client

Shape the host expects is usually one of:

```json
{
  "command": "uvx",
  "args": [
    "--from",
    "attogradDB[mcp]==1.0.1",
    "attograddb-mcp",
    "--project-root",
    "/path/to/project"
  ]
}
```

or a single argv array:

```json
{
  "command": [
    "uvx",
    "--from",
    "attogradDB[mcp]==1.0.1",
    "attograddb-mcp",
    "--project-root",
    "/path/to/project"
  ]
}
```

Pass the real project directory, not a shared global folder. After connect, ask
the agent to call `recall_decisions` before it plans.
