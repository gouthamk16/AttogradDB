import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("path", "project_root"),
    [
        (".claude-plugin/plugin.json", "${CLAUDE_PROJECT_DIR}"),
        (".cursor-plugin/plugin.json", "${workspaceFolder}"),
        (".codex-plugin/plugin.json", "."),
    ],
)
def test_plugin_manifests_bind_the_active_project(path: str, project_root: str) -> None:
    manifest = json.loads((ROOT / path).read_text())
    server = manifest["mcpServers"]["attograd-memory"]

    assert manifest["name"] == "attograd-memory"
    assert manifest["version"] == "1.0.0"
    assert manifest["skills"] == "./skills/"
    assert server["command"] == "uvx"
    assert server["args"] == [
        "--from",
        "attogradDB[mcp]==1.0.0",
        "attograddb-mcp",
        "--project-root",
        project_root,
    ]


def test_plugin_guidance_is_shared() -> None:
    skill = ROOT / "skills" / "attograd-memory" / "SKILL.md"
    content = skill.read_text()

    assert "recall_decisions" in content
    assert "remember_decision" in content
    assert "supersedes" in content
