"""`attograddb setup`: wire AttogradDB decision memory into AI coding tools.

Detects installed hosts (Codex, Gemini CLI, OpenCode, Cursor, Claude Code), then
writes each one's MCP server config plus an auto-loaded instructions block, at
project or global scope. No manual file editing required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import tomlkit

NAME = "attograd-memory"

START = "<!-- attograd-memory:start -->"
END = "<!-- attograd-memory:end -->"

INSTRUCTION_BODY = """## Project memory (AttogradDB)

This project uses AttogradDB decision memory through the `attograd-memory` MCP server.

- Before planning or editing, call `recall_decisions` and treat each active decision as a
  project constraint. Do not contradict one without an explicit replacement.
- When a durable choice is made (architecture, dependencies, APIs, workflow, security, or
  scope), call `remember_decision` with a one-sentence `claim` and a `rationale`. Pass the
  prior decision's id as `supersedes` when it replaces an active one.
- Do not store secrets or transient task notes.
"""

TOOL_ORDER = ["codex", "gemini", "opencode", "cursor", "claude"]
LABELS = {
    "codex": "Codex",
    "gemini": "Gemini CLI",
    "opencode": "OpenCode",
    "cursor": "Cursor",
    "claude": "Claude Code",
}


def _pin() -> str:
    try:
        from importlib.metadata import version

        return f"attogradDB[mcp]=={version('attogradDB')}"
    except Exception:
        return "attogradDB[mcp]"


def _launcher() -> Callable[[str | None], list[str]]:
    """Return a function mapping a project-root token to the server launch argv.

    Prefers `uvx` (self-contained, version-pinned) and falls back to the current
    interpreter so setup works even without uv installed.
    """
    uvx = shutil.which("uvx")
    pin = _pin()

    def build(project_root: str | None) -> list[str]:
        tail = [] if project_root is None else ["--project-root", project_root]
        if uvx:
            return [uvx, "--from", pin, "attograddb-mcp", *tail]
        return [sys.executable, "-m", "attogradDB.mcp_server", *tail]

    return build


# --- file helpers -----------------------------------------------------------


def _read_json_lenient(path: Path) -> dict:
    """Load JSON, tolerating // and /* */ comments (opencode.jsonc)."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return json.loads(_strip_json_comments(text))


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _upsert_json(path: Path, top_key: str, entry: dict) -> None:
    data = _read_json_lenient(path)
    servers = data.get(top_key)
    if not isinstance(servers, dict):
        servers = {}
        data[top_key] = servers
    servers[NAME] = entry
    _write_json(path, data)


def _upsert_toml(path: Path, command: str, args: list[str]) -> None:
    doc = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    servers = doc.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = tomlkit.table()
        doc["mcp_servers"] = servers
    entry = tomlkit.table()
    entry["command"] = command
    entry["args"] = args
    servers[NAME] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _upsert_block(path: Path, block: str) -> None:
    """Insert or replace the instructions block between markers, keeping the rest."""
    body = f"{START}\n{block.strip()}\n{END}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if START in text and END in text:
            pre = text[: text.index(START)]
            post = text[text.index(END) + len(END) :]
            new = f"{pre}{body}{post}"
        else:
            joiner = "" if text == "" else ("\n" if text.endswith("\n") else "\n\n")
            new = f"{text}{joiner}{body}\n"
    else:
        new = f"{body}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")


def _write_cursor_rule(path: Path) -> None:
    frontmatter = "---\ndescription: AttogradDB project memory\nalwaysApply: true\n---\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter + INSTRUCTION_BODY.strip() + "\n", encoding="utf-8")


# --- per-tool configurers ---------------------------------------------------

Launcher = Callable[[str | None], list[str]]


def _configure_cursor(scope: str, project: Path, home: Path, launch: Launcher) -> list[Path]:
    if scope == "project":
        cfg = project / ".cursor" / "mcp.json"
        argv = launch(str(project))
        rule = project / ".cursor" / "rules" / "attograd-memory.mdc"
    else:
        cfg = home / ".cursor" / "mcp.json"
        argv = launch("${workspaceFolder}")
        rule = None
    _upsert_json(cfg, "mcpServers", {"type": "stdio", "command": argv[0], "args": argv[1:]})
    written = [cfg]
    if rule is not None:
        _write_cursor_rule(rule)
        written.append(rule)
    return written


def _configure_claude(scope: str, project: Path, home: Path, launch: Launcher) -> list[Path]:
    if scope == "project":
        cfg = project / ".mcp.json"
        argv = launch(str(project))
        instr = project / "CLAUDE.md"
    else:
        cfg = home / ".claude.json"
        argv = launch(None)  # Claude injects CLAUDE_PROJECT_DIR into the server env
        instr = home / ".claude" / "CLAUDE.md"
    _upsert_json(cfg, "mcpServers", {"command": argv[0], "args": argv[1:]})
    _upsert_block(instr, INSTRUCTION_BODY)
    return [cfg, instr]


def _configure_codex(scope: str, project: Path, home: Path, launch: Launcher) -> list[Path]:
    if scope == "project":
        cfg = project / ".codex" / "config.toml"
        argv = launch(str(project))
        instr = project / "AGENTS.md"
    else:
        cfg = home / ".codex" / "config.toml"
        argv = launch(None)  # server falls back to the working directory
        instr = home / ".codex" / "AGENTS.md"
    _upsert_toml(cfg, argv[0], argv[1:])
    _upsert_block(instr, INSTRUCTION_BODY)
    return [cfg, instr]


def _configure_gemini(scope: str, project: Path, home: Path, launch: Launcher) -> list[Path]:
    if scope == "project":
        cfg = project / ".gemini" / "settings.json"
        argv = launch(str(project))
        instr = project / "GEMINI.md"
    else:
        cfg = home / ".gemini" / "settings.json"
        argv = launch(None)
        instr = home / ".gemini" / "GEMINI.md"
    _upsert_json(cfg, "mcpServers", {"command": argv[0], "args": argv[1:]})
    _upsert_block(instr, INSTRUCTION_BODY)
    return [cfg, instr]


def _configure_opencode(scope: str, project: Path, home: Path, launch: Launcher) -> list[Path]:
    if scope == "project":
        existing = project / "opencode.jsonc"
        cfg = existing if existing.exists() else project / "opencode.json"
        argv = launch(str(project))
        instr = project / "AGENTS.md"
    else:
        cfg = home / ".config" / "opencode" / "opencode.json"
        argv = launch(None)
        instr = home / ".config" / "opencode" / "AGENTS.md"
    _upsert_json(cfg, "mcp", {"type": "local", "command": argv, "enabled": True})
    _upsert_block(instr, INSTRUCTION_BODY)
    return [cfg, instr]


CONFIGURERS: dict[str, Callable[[str, Path, Path, Launcher], list[Path]]] = {
    "codex": _configure_codex,
    "gemini": _configure_gemini,
    "opencode": _configure_opencode,
    "cursor": _configure_cursor,
    "claude": _configure_claude,
}


def configure(key: str, scope: str, project: Path, home: Path) -> list[Path]:
    """Write config + instructions for one tool. Returns the paths written."""
    return CONFIGURERS[key](scope, project, home, _launcher())


# --- detection --------------------------------------------------------------


def _detectors(home: Path) -> dict[str, Callable[[], bool]]:
    def either(cmd: str, *paths: Path) -> Callable[[], bool]:
        return lambda: shutil.which(cmd) is not None or any(p.exists() for p in paths)

    return {
        "codex": either("codex", home / ".codex"),
        "gemini": either("gemini", home / ".gemini"),
        "opencode": either("opencode", home / ".config" / "opencode"),
        "cursor": either("cursor", home / ".cursor"),
        "claude": either("claude", home / ".claude", home / ".claude.json"),
    }


def detect(home: Path | None = None) -> list[str]:
    home = home or Path.home()
    checks = _detectors(home)
    return [key for key in TOOL_ORDER if checks[key]()]


# --- interactive flow -------------------------------------------------------


def _prompt_tools(detected: list[str]) -> list[str]:
    candidates = detected or TOOL_ORDER
    if detected:
        print("Detected these tools:")
    else:
        print("No tools auto-detected. Choose from all supported hosts:")
    for i, key in enumerate(candidates, 1):
        print(f"  {i}. {LABELS[key]}")
    raw = input("Configure which? [numbers, comma-separated, or 'all']: ").strip()
    if raw.lower() in ("", "all"):
        return candidates
    chosen = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(candidates):
            chosen.append(candidates[int(part) - 1])
    return chosen


def _prompt_scope() -> str:
    print("\nInstall scope:")
    print("  1. This project (writes into the current directory)")
    print("  2. Global (writes into your home config for every project)")
    raw = input("Scope? [1/2, default 1]: ").strip()
    return "global" if raw == "2" else "project"


def run_setup(args: argparse.Namespace) -> int:
    project = Path(args.project_dir or os.getcwd()).resolve()
    home = Path.home()

    if args.tools:
        keys = [t.strip() for t in args.tools.split(",") if t.strip()]
        unknown = [k for k in keys if k not in CONFIGURERS]
        if unknown:
            print(f"error: unknown tool(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"choose from: {', '.join(TOOL_ORDER)}", file=sys.stderr)
            return 2
    else:
        keys = _prompt_tools(detect(home))

    if not keys:
        print("Nothing selected. Exiting.")
        return 0

    scope = args.scope or (_prompt_scope() if not args.yes else "project")

    print(f"\nWill configure {', '.join(LABELS[k] for k in keys)} at {scope} scope.")
    if scope == "project":
        print(f"Project: {project}")
    if not args.yes:
        if input("Proceed? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Aborted. Nothing written.")
            return 0

    for key in keys:
        written = configure(key, scope, project, home)
        print(f"\n{LABELS[key]}:")
        for path in written:
            print(f"  wrote {path}")

    print("\nDone. Restart the tool (or start a new session) to load the memory server.")
    print("Then ask it to call recall_decisions before it plans.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="attograddb", description="AttogradDB command-line tools"
    )
    sub = parser.add_subparsers(dest="command")
    setup_parser = sub.add_parser("setup", help="Configure AttogradDB memory for AI coding tools")
    setup_parser.add_argument(
        "--tools", help="Comma-separated: codex, gemini, opencode, cursor, claude"
    )
    setup_parser.add_argument("--scope", choices=["project", "global"])
    setup_parser.add_argument(
        "--project-dir", default=None, help="Project directory (default: current directory)"
    )
    setup_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip prompts (non-interactive)"
    )
    args = parser.parse_args(argv)

    if args.command != "setup":
        parser.print_help()
        return 0
    return run_setup(args)


if __name__ == "__main__":
    raise SystemExit(main())
