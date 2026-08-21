from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"


def test_docs_site_pages_exist() -> None:
    for name in (
        "_config.yml",
        "index.md",
        "usage.md",
        "agents.md",
        "other-harnesses.md",
        "marketplace.md",
    ):
        text = (DOCS / name).read_text(encoding="utf-8")
        assert text.strip(), name


def test_agent_docs_cover_manual_and_plugin_install() -> None:
    text = (DOCS / "agents.md").read_text(encoding="utf-8")
    assert "claude mcp add" in text
    assert ".cursor/mcp.json" in text
    assert "mcp_servers.attograd-memory" in text
    assert "/plugin install attograd-memory@attograd-plugins" in text
    assert "until" in text.lower()


def test_other_harnesses_cover_opencode_hermes_pi() -> None:
    text = (DOCS / "other-harnesses.md").read_text(encoding="utf-8")
    assert "mcp.servers" in text or '"servers"' in text
    assert "mcp_servers:" in text
    assert "~/.pi/agent/mcp.json" in text
    assert "attograddb-mcp" in text


def test_marketplace_docs_include_submission_urls() -> None:
    text = (DOCS / "marketplace.md").read_text(encoding="utf-8")
    assert "clau.de/plugin-directory-submission" in text
    assert "cursor.directory/plugins/new" in text
    assert "developers.openai.com/plugins/deploy/submission" in text
