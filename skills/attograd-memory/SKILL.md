---
name: attograd-memory
description: Recall and record durable project decisions with AttogradDB. Use when starting work in a repository, planning or editing code, or deciding whether a project constraint should persist.
---

# AttogradDB project memory

Use the `attograd-memory` MCP tools for decisions that should survive the current task.

## Start of a task

Before planning or editing:

1. Call `recall_decisions`.
2. Treat every returned active decision as a project constraint.
3. Do not contradict an active decision without an explicit replacement.

## Record durable decisions

Call `remember_decision` when the user or team makes a durable choice about architecture,
dependencies, APIs, workflow, security, or scope. Include:

- `claim`: the decision in one clear sentence.
- `rationale`: why it was chosen.
- `scope`: the affected area when useful.
- `evidence`: a file, benchmark, issue, or other supporting reference when available.

When a new decision replaces an active one, pass the old decision's id as `supersedes`.
Do not create a contradictory active decision.

Do not store transient task details, guesses, or secrets.
