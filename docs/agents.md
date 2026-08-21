---
title: Claude Code and Cursor
nav_order: 3
description: Install AttogradDB project memory in Cursor and Claude Code from the plugin marketplace
---

# Claude Code and Cursor

Cursor and Claude Code load AttogradDB as a **plugin** — one click or one command, no
config files to edit. The plugin runs a per-project decision-memory server and exposes two
tools:

- `recall_decisions` — every active decision for this project, in order.
- `remember_decision` — record a durable choice, superseding an older one when it replaces it.

Each project keeps its own `.attograd-memory.db`. The plugin passes the active workspace to
the server, so you never type a project path.

Using another tool (Codex, Gemini CLI, OpenCode, or anything else)? See
[Every other tool](other-harnesses.md).

## Cursor

Install from the Cursor plugin directory:
[cursor.directory/plugins/attograd-memory](https://cursor.directory/plugins/attograd-memory).

1. Open the listing and click **Install**, or in Cursor open **Settings → Plugins**, search
   for `attograd-memory`, and install it.
2. Choose the global (user) scope so it applies to every project.
3. Confirm `attograd-memory` is enabled under **Settings → MCP**, then start a new Agent chat.

Cursor passes `${workspaceFolder}` to the server, so memory is scoped to whichever project
you have open.

## Claude Code

The plugin installs from a marketplace inside a Claude Code session:

```text
/plugin marketplace add gouthamk16/AttogradDB
/plugin install attograd-memory@attograd-plugins
```

Pick the global/user scope when prompted. Run `/mcp` to confirm `attograd-memory` connected,
then start a new session. Claude Code passes the active project directory to the server.

Once the plugin is approved in Anthropic's community directory it also installs as
`attograd-memory@claude-community` (`/plugin marketplace add anthropics/claude-plugins-community`).
Both give you the same tools and the same per-project database.

## Using it

The bundled skill tells the agent to call `recall_decisions` before it plans and to record
durable choices with `remember_decision`. You can also ask directly:

- "Recall the project decisions before you start."
- "Remember that we use SQLite, not Redis, for durability — supersede the old Redis decision."

`remember_decision` takes `claim`, `rationale`, and optional `scope`, `evidence`, and
`supersedes` (the id of the decision it replaces). A missing, cross-project, or
already-superseded `supersedes` target is rejected without a partial write.

Prefer the Python API instead of an agent? See [Python usage](usage.md). How listings get
submitted is in [Marketplace listings](marketplace.md).
