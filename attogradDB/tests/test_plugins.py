import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
PIN = "attogradDB[mcp]==1.0.1"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _expected_args(project_root: str) -> list[str]:
    return ["--from", PIN, "attograddb-mcp", "--project-root", project_root]


def test_claude_plugin_declares_inline_mcp_server() -> None:
    """Claude Code loads inline mcpServers from plugin.json with its own project var."""
    manifest = _load(".claude-plugin/plugin.json")
    server = manifest["mcpServers"]["attograd-memory"]

    assert manifest["name"] == "attograd-memory"
    assert manifest["version"] == "1.0.1"
    assert manifest["skills"] == "./skills/"
    assert server["command"] == "uvx"
    assert server["args"] == _expected_args("${CLAUDE_PROJECT_DIR}")


def test_cursor_plugin_discovers_mcp_json() -> None:
    """Cursor's marketplace scanner finds MCP servers in mcp.json, not inline config."""
    manifest = _load(".cursor-plugin/plugin.json")
    server = _load("mcp.json")["mcpServers"]["attograd-memory"]

    assert manifest["name"] == "attograd-memory"
    assert manifest["version"] == "1.0.1"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "mcp.json"
    assert server["command"] == "uvx"
    assert server["args"] == _expected_args("${workspaceFolder}")


def test_codex_plugin_points_at_dot_mcp_json() -> None:
    """Codex reads mcpServers as a path to .mcp.json, which uses the mcp_servers key."""
    manifest = _load(".codex-plugin/plugin.json")
    server = _load(".mcp.json")["mcp_servers"]["attograd-memory"]

    assert manifest["name"] == "attograd-memory"
    assert manifest["version"] == "1.0.1"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert manifest["interface"]["displayName"]
    assert server["command"] == "uvx"
    assert server["args"] == _expected_args(".")


def test_codex_marketplace_entry_has_required_policy() -> None:
    """Codex requires policy.installation, policy.authentication, and category per entry."""
    entry = _load(".agents/plugins/marketplace.json")["plugins"][0]

    assert entry["name"] == "attograd-memory"
    assert entry["source"]["path"] == "./"
    assert entry["policy"]["installation"] == "AVAILABLE"
    assert entry["policy"]["authentication"] == "ON_INSTALL"
    assert entry["category"] == "Productivity"


def test_plugin_guidance_is_shared() -> None:
    content = (ROOT / "skills" / "attograd-memory" / "SKILL.md").read_text()

    assert "recall_decisions" in content
    assert "remember_decision" in content
    assert "supersedes" in content
