---
title: Agent plugins
nav_order: 3
description: Install AttogradDB in Claude Code, Cursor, or Codex — plugin or manual MCP
---

# Agent plugins

Important project decisions are not a similarity-search problem. A later task can depend on
a prior constraint without using similar words. The MCP server keeps a small structured set
of *active* decisions and returns all of them on recall. Vector search remains for documents
and traces; see [Python usage](usage.md).

Install the plugin **once** at the host's global/user scope. The host still launches one
workspace-aware server for each project. Each project keeps its own `.attograd-memory.db`.
The model does not supply a project or session name.

The plugin uses `uvx` to install `attogradDB[mcp]==1.0.1` on first launch. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first. You do not need a
separate `pip install` when using the plugin or the `uvx` manual commands below.

```bash
uv --version
```

## Until the public marketplace listing

A public listing in Claude Code, Cursor, or Codex is not required. Users get the same tools
and the same database layout from any of:

1. **This GitHub repository as a plugin source** — the manifests in the repo.
2. **Manual MCP configuration** — host-specific files, or the generic stdio command.
3. **A later marketplace install** — same server, same tools, once the listing is approved.

Prefer the plugin when the host can load it. Use the manual MCP config when you want an
explicit command line or the marketplace UI is not available yet. OpenCode, Hermes, Pi, and
other stdio clients are documented in [Other harnesses](other-harnesses.md). How listings
get submitted is in [Marketplace listings](marketplace.md).

## Claude Code

From a Claude Code session, add this repository as a marketplace and install globally:

```text
/plugin marketplace add gouthamk16/AttogradDB
/plugin install attograd-memory@attograd-plugins
```

Choose the global/user installation scope. The plugin passes Claude Code's active project
directory to the server. Run `/mcp` to confirm the connection, then start a new session.

That repository marketplace is the supported path until Claude Code lists the plugin
publicly. After a listing exists, install from the Claude Code plugin marketplace instead.

### Manual MCP (same tools)

```bash
claude mcp add --scope user attograd-memory -- uvx --from "attogradDB[mcp]==1.0.1" attograddb-mcp --project-root /absolute/path/to/project
```

Replace the project path. Manual setup does not automatically change the project path as you
move between repositories; the plugin does.

## Cursor

Until Cursor Marketplace lists AttogradDB, load the plugin from this repository
(`.cursor-plugin/plugin.json`) or add a direct MCP config.

Open **Customize**, add the repository as a plugin source if you are testing locally, and
choose a global/user installation scope. Confirm `attograd-memory` is enabled in MCP
settings, then start a new Agent task. Cursor passes `${workspaceFolder}` to the server.

### Manual MCP (same tools)

Create `.cursor/mcp.json` in a project, or put the same entry in `~/.cursor/mcp.json` for a
global install that still opens the active workspace's database:

```json
{
  "mcpServers": {
    "attograd-memory": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "attogradDB[mcp]==1.0.1",
        "attograddb-mcp",
        "--project-root",
        "${workspaceFolder}"
      ]
    }
  }
}
```

If `uvx` is unavailable and AttogradDB is already installed in Python:

```json
{
  "mcpServers": {
    "attograd-memory": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "attogradDB.mcp_server", "--project-root", "${workspaceFolder}"]
    }
  }
}
```

## Codex

Until Codex lists the plugin publicly, add this repository's marketplace
(`.agents/plugins/marketplace.json`) and install **AttogradDB Project Memory** at
global/user scope. The bundled server uses the active working directory as `--project-root`.

### Manual MCP (same tools)

`~/.codex/config.toml` (global; still scoped to the current working directory):

```toml
[mcp_servers.attograd-memory]
command = "uvx"
args = ["--from", "attogradDB[mcp]==1.0.1", "attograddb-mcp", "--project-root", "."]
```

For a trusted project-only configuration, put the same entry in `.codex/config.toml`.

## What gets installed

```text
host plugin or MCP config
        → uvx
        → attograddb-mcp --project-root active-workspace
        → active-workspace/.attograd-memory.db
```

Generic MCP host config is in [Other harnesses](other-harnesses.md). You can also run the
server yourself:

```bash
attograddb-mcp --project-root /path/to/project
```

The default database is `/path/to/project/.attograd-memory.db`. Pass `--db` to put it
elsewhere. The project name is the directory name of `--project-root`.

## Tools

- `recall_decisions()` — every active decision for this project, in insertion order.
- `remember_decision(claim, rationale, scope=None, evidence=None, supersedes=None)` —
  write durable project truth. Pass the prior decision's id as `supersedes` when replacing
  it. Missing, cross-project, or already-superseded targets are rejected without a partial
  write.

Each decision has `claim`, `rationale`, optional `scope` and `evidence`, `status`
(`active` or `superseded`), and `superseded_by`.

Call `recall_decisions` before planning or editing. Treat returned decisions as constraints.
Call `remember_decision` for durable choices. Pass `supersedes` when replacing an active
decision. Do not store secrets or transient task notes.

The bundled skill and server instructions tell the model to do this. Generic MCP cannot
force a host to inject that context. For critical workflows, keep the same requirement in
your team or project instructions.

A library-level counterpart (no MCP host required) is `examples/mcp_memory.py`.
