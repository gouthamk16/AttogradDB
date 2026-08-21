import json
import sys
import tomllib

import pytest

from attogradDB import setup


@pytest.fixture(autouse=True)
def force_interpreter_launch(monkeypatch):
    """Pin the launcher to the interpreter path so generated argv is deterministic."""
    monkeypatch.setattr(setup.shutil, "which", lambda _cmd: None)


def _expected_argv(project=None):
    tail = [] if project is None else ["--project-root", str(project)]
    return [sys.executable, "-m", "attogradDB.mcp_server", *tail]


def test_detect_reports_tools_by_dir_or_binary(tmp_path, monkeypatch):
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    monkeypatch.setattr(setup.shutil, "which", lambda _cmd: None)

    assert setup.detect(tmp_path) == ["codex", "claude"]


def test_cursor_project_writes_stdio_config_and_rule(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    written = setup.configure("cursor", "project", project, tmp_path / "home")

    cfg = json.loads((project / ".cursor" / "mcp.json").read_text())
    entry = cfg["mcpServers"]["attograd-memory"]
    assert entry["type"] == "stdio"
    assert [entry["command"], *entry["args"]] == _expected_argv(project)

    rule = project / ".cursor" / "rules" / "attograd-memory.mdc"
    assert "alwaysApply: true" in rule.read_text()
    assert set(written) == {project / ".cursor" / "mcp.json", rule}


def test_cursor_global_uses_workspace_variable_and_no_rule(tmp_path):
    home = tmp_path / "home"
    written = setup.configure("cursor", "global", tmp_path / "proj", home)

    cfg = json.loads((home / ".cursor" / "mcp.json").read_text())
    assert "${workspaceFolder}" in cfg["mcpServers"]["attograd-memory"]["args"]
    assert written == [home / ".cursor" / "mcp.json"]


def test_claude_project_config_has_no_type_and_writes_claude_md(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    setup.configure("claude", "project", project, tmp_path / "home")

    entry = json.loads((project / ".mcp.json").read_text())["mcpServers"]["attograd-memory"]
    assert "type" not in entry
    assert [entry["command"], *entry["args"]] == _expected_argv(project)
    assert "recall_decisions" in (project / "CLAUDE.md").read_text()


def test_claude_global_omits_project_root(tmp_path):
    home = tmp_path / "home"
    setup.configure("claude", "global", tmp_path / "proj", home)

    entry = json.loads((home / ".claude.json").read_text())["mcpServers"]["attograd-memory"]
    assert "--project-root" not in entry["args"]
    assert (home / ".claude" / "CLAUDE.md").exists()


def test_codex_project_writes_toml_table_and_agents_md(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    setup.configure("codex", "project", project, tmp_path / "home")

    doc = tomllib.loads((project / ".codex" / "config.toml").read_text())
    entry = doc["mcp_servers"]["attograd-memory"]
    assert [entry["command"], *entry["args"]] == _expected_argv(project)
    assert setup.START in (project / "AGENTS.md").read_text()


def test_gemini_project_writes_settings_and_gemini_md(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    setup.configure("gemini", "project", project, tmp_path / "home")

    entry = json.loads(
        (project / ".gemini" / "settings.json").read_text()
    )["mcpServers"]["attograd-memory"]
    assert [entry["command"], *entry["args"]] == _expected_argv(project)
    assert "recall_decisions" in (project / "GEMINI.md").read_text()


def test_opencode_project_uses_local_type_and_command_array(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    setup.configure("opencode", "project", project, tmp_path / "home")

    entry = json.loads((project / "opencode.json").read_text())["mcp"]["attograd-memory"]
    assert entry["type"] == "local"
    assert entry["enabled"] is True
    assert entry["command"] == _expected_argv(project)
    assert setup.START in (project / "AGENTS.md").read_text()


def test_json_config_preserves_other_keys_and_is_idempotent(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    settings = project / ".gemini" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}))

    setup.configure("gemini", "project", project, tmp_path / "home")
    setup.configure("gemini", "project", project, tmp_path / "home")

    data = json.loads(settings.read_text())
    assert data["theme"] == "dark"
    assert set(data["mcpServers"]) == {"other", "attograd-memory"}


def test_instruction_block_is_upserted_once(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    agents = project / "AGENTS.md"
    agents.write_text("# House rules\n\nBe careful.\n")

    setup.configure("codex", "project", project, tmp_path / "home")
    setup.configure("codex", "project", project, tmp_path / "home")

    text = agents.read_text()
    assert text.count(setup.START) == 1
    assert text.count(setup.END) == 1
    assert "Be careful." in text


def test_opencode_reads_existing_jsonc_with_comments(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    jsonc = project / "opencode.jsonc"
    jsonc.write_text('{\n  // a comment\n  "$schema": "https://opencode.ai/config.json"\n}\n')

    setup.configure("opencode", "project", project, tmp_path / "home")

    data = json.loads(jsonc.read_text())
    assert data["$schema"] == "https://opencode.ai/config.json"
    assert data["mcp"]["attograd-memory"]["type"] == "local"


def test_run_setup_rejects_unknown_tool(tmp_path, capsys):
    args = setup.argparse.Namespace(
        tools="codex,notatool", scope="project", project_dir=str(tmp_path), yes=True
    )
    assert setup.run_setup(args) == 2
    assert "unknown tool" in capsys.readouterr().err
