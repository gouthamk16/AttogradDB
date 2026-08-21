---
title: Marketplace listings
nav_order: 5
description: How AttogradDB gets into Claude, Cursor, and Codex plugin marketplaces
---

# Marketplace listings

A marketplace listing is discovery, not a different product. Until a listing is
approved, install from this GitHub repository or with a direct MCP config. The
tools and the per-project database are the same. See [Agent plugins](agents.md)
and [Other harnesses](other-harnesses.md).

The manifests are already on a public `main`, so every submission below can be
made now. Each host's "until approval" fallback keeps working in the meantime.

## Claude Code / Claude Cowork

Third-party plugins go to the **community** marketplace
(`anthropics/claude-plugins-community`), not Anthropic's official catalog.

1. Validate locally: `claude plugin validate .`
2. Submit the public GitHub URL (`https://github.com/gouthamk16/AttogradDB`) at
   [claude.ai plugin submissions](https://claude.ai/admin-settings/directory/submissions/plugins/new)
   or the [console form](https://platform.claude.com/plugins/submit).
   Shortcut: [clau.de/plugin-directory-submission](https://clau.de/plugin-directory-submission).
3. Anthropic runs automated review. Approved plugins are pinned by commit SHA and
   synced into the community catalog (often overnight).

Until that lands, users install from this repo:

```text
/plugin marketplace add gouthamk16/AttogradDB
/plugin install attograd-memory@attograd-plugins
```

After approval:

```text
/plugin marketplace add anthropics/claude-plugins-community
/plugin install attograd-memory@claude-community
```

There is no application for `claude-plugins-official`. Anthropic curates that
list separately.

## Cursor

1. Submit the public repo at [cursor.directory/plugins/new](https://cursor.directory/plugins/new)
   (community directory; this is the current listing path).
2. Optionally also use [cursor.com/marketplace/publish](https://cursor.com/marketplace/publish)
   while signed into Cursor. First-party Marketplace review is slower and often
   reserved for company integrations.

Until a listing appears, load `.cursor-plugin/plugin.json` from this repository
or add the MCP server in `.cursor/mcp.json` / `~/.cursor/mcp.json`.

## Codex / ChatGPT

Codex CLI, desktop, and the ChatGPT desktop app can load this repo's plugin via
`.codex-plugin/plugin.json` and `.agents/plugins/marketplace.json`. That is the
supported path today.

The **public** ChatGPT/Codex Plugins Directory is a different pipeline: the
[OpenAI plugin submission portal](https://developers.openai.com/plugins/deploy/submission)
reviews an MCP **server URL**. AttogradDB is a local stdio server, not a hosted
HTTP MCP. Do not submit it there unless a hosted endpoint exists. Use the repo
marketplace or `~/.codex/config.toml` instead.

## After submit

| Host | Form | What users do until approval |
| --- | --- | --- |
| Claude Code | [Plugin directory submission](https://clau.de/plugin-directory-submission) | `/plugin marketplace add gouthamk16/AttogradDB` |
| Cursor | [cursor.directory/plugins/new](https://cursor.directory/plugins/new) | Repo plugin or `.cursor/mcp.json` |
| Codex | Repo marketplace only (no hosted MCP URL) | `.agents/plugins/marketplace.json` or `config.toml` |
| OpenCode, Hermes, Pi | None | Direct MCP config in [Other harnesses](other-harnesses.md) |
