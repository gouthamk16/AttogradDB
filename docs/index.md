---
title: Home
nav_order: 1
description: AttogradDB documentation — install, Python API, and agent plugins
---

# AttogradDB

A lightweight, local-first vector store. One SQLite file, no server, no index to maintain.
Search one project or one session without the rest bleeding in. Optional MCP tools remember
and recall durable project decisions.

Version 1.0.0

[Python usage](usage.md){: .btn .btn-primary .mr-2 }
[Agent plugins](agents.md){: .btn .btn-outline }
[GitHub](https://github.com/gouthamk16/AttogradDB){: .btn .btn-outline }

## Install the library

```bash
pip install attogradDB
```

Decision memory over MCP:

```bash
pip install "attogradDB[mcp]"
```

Or let a host plugin bootstrap that extra with `uvx`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

## Choose a path

| Goal | Path |
| --- | --- |
| Ingest documents and search them from Python | [Python usage](usage.md) |
| Let Claude Code, Cursor, or Codex remember project decisions | [Agent plugins](agents.md) |
| Both | Point the Python store and the MCP server at the same SQLite file |

## Plugins and manual install

You can use AttogradDB with Claude Code, Cursor, or Codex **without waiting for a public marketplace listing**. The plugin in this repository and a direct MCP config expose the same two tools:

- `recall_decisions`
- `remember_decision`

Until those marketplaces list AttogradDB, install from the GitHub repo or add the MCP server by hand. The experience is the same: a global install, a workspace-aware server, and a per-project `.attograd-memory.db`. See [Agent plugins](agents.md).
