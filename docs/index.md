---
title: Home
nav_order: 1
description: Light speed memory for your agents — a local-first vector store and MCP decision memory
---

<div class="attograd-hero" markdown="0">
  <p class="attograd-eyebrow">AttogradDB · v1.1.0</p>
  <h1 class="attograd-tagline">Light speed memory for your agents</h1>
  <p class="attograd-sub">A local-first vector store in one SQLite file. No server, no index to tune. Search one project or one session without the rest bleeding in — and give Claude Code, Cursor, or Codex durable project memory over MCP.</p>
</div>

[Python usage](usage.md){: .btn .btn-primary .mr-2 }
[Claude Code & Cursor](agents.md){: .btn .mr-2 }
[Every other tool](other-harnesses.md){: .btn .mr-2 }
[GitHub](https://github.com/gouthamk16/AttogradDB){: .btn }

## Install

```bash
pip install attogradDB
```

Add the MCP decision-memory server:

```bash
pip install "attogradDB[mcp]"
```

Host plugins bootstrap that extra for you with `uvx` — install [uv](https://docs.astral.sh/uv/getting-started/installation/) and skip the manual `pip`.

## Where to go next

| You want to… | Go to |
| --- | --- |
| Give Claude Code or Cursor project memory (plugin) | [Claude Code and Cursor](agents.md) |
| Set up Codex, Gemini CLI, OpenCode, or any MCP host | [Every other tool](other-harnesses.md) |
| Ingest documents and search them from Python | [Python usage](usage.md) |
| Get AttogradDB listed in a plugin marketplace | [Marketplace listings](marketplace.md) |

## Two tools, one database

The agent surface is deliberately small. Both tools operate on a per-project `.attograd-memory.db`, and the model never supplies a project or session name — the host passes the active workspace.

- **`recall_decisions`** — every active decision for this project, in order. Call it before planning or editing.
- **`remember_decision`** — record a durable choice, with an explicit `supersedes` when it replaces an older one.

Install the [plugin](agents.md) in Cursor or Claude Code, or run [`attograddb setup`](other-harnesses.md) for every other tool. Either way you get the same tools and the same database layout.
