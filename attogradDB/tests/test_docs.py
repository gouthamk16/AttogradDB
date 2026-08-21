from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_docs_site_pages_exist() -> None:
    docs = ROOT / "docs"
    for name in ("_config.yml", "index.md", "usage.md", "agents.md"):
        text = (docs / name).read_text(encoding="utf-8")
        assert text.strip(), name


def test_agent_docs_cover_manual_and_plugin_install() -> None:
    text = (ROOT / "docs" / "agents.md").read_text(encoding="utf-8")
    assert "claude mcp add" in text
    assert ".cursor/mcp.json" in text
    assert "mcp_servers.attograd-memory" in text
    assert "/plugin install attograd-memory@attograd-plugins" in text
    assert "until" in text.lower()
