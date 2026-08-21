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


def test_agents_page_is_marketplace_only() -> None:
    text = (DOCS / "agents.md").read_text(encoding="utf-8")
    assert "cursor.directory/plugins/attograd-memory" in text
    assert "/plugin install attograd-memory@attograd-plugins" in text
    assert "recall_decisions" in text
    # No manual MCP config on the marketplace page; that lives in other-harnesses.md.
    assert ".cursor/mcp.json" not in text
    assert "claude mcp add" not in text


def test_other_harnesses_lead_with_setup_cli() -> None:
    text = (DOCS / "other-harnesses.md").read_text(encoding="utf-8")
    assert "attograddb setup" in text
    for tool in ("Codex", "Gemini", "OpenCode"):
        assert tool in text
    assert "attograddb-mcp" in text
    # Hermes and Pi remain documented as manual stdio hosts.
    assert "~/.pi/agent/mcp.json" in text


def test_marketplace_docs_include_submission_urls() -> None:
    text = (DOCS / "marketplace.md").read_text(encoding="utf-8")
    assert "clau.de/plugin-directory-submission" in text
    assert "cursor.directory/plugins/new" in text
    assert "developers.openai.com/plugins/deploy/submission" in text
