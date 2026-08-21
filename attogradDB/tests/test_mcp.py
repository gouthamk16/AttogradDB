import pytest
from mcp import Client

from attogradDB.mcp_server import create_server, resolve_project_root


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(tmp_path):
    async with Client(create_server(tmp_path), raise_exceptions=True) as connected:
        yield connected


def test_project_root_prefers_explicit_then_env_then_cwd(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    env = tmp_path / "from-env"
    cwd = tmp_path / "cwd"
    for path in (explicit, env, cwd):
        path.mkdir()
    monkeypatch.chdir(cwd)

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(env))
    assert resolve_project_root(str(explicit)) == str(explicit)
    assert resolve_project_root(None) == str(env)

    monkeypatch.delenv("CLAUDE_PROJECT_DIR")
    assert resolve_project_root(None) == str(cwd)


@pytest.mark.anyio
async def test_server_exposes_only_decision_tools(client):
    tools = await client.list_tools()
    assert [tool.name for tool in tools.tools] == ["remember_decision", "recall_decisions"]


@pytest.mark.anyio
async def test_server_instructions_require_startup_recall(client):
    assert "before planning or editing" in client.instructions
    assert "supersedes" in client.instructions


@pytest.mark.anyio
async def test_remember_and_recall_return_structured_decisions(client):
    remembered = await client.call_tool(
        "remember_decision",
        {
            "claim": "Use SQLite.",
            "rationale": "Memory is local-first.",
            "scope": "storage",
            "evidence": "Architecture decision",
        },
    )
    recalled = await client.call_tool("recall_decisions", {})

    assert remembered.structured_content["claim"] == "Use SQLite."
    assert remembered.structured_content["project"]
    assert recalled.structured_content["result"] == [remembered.structured_content]


@pytest.mark.anyio
async def test_tool_can_supersede_an_active_decision(client):
    old = await client.call_tool(
        "remember_decision",
        {"claim": "Use Redis.", "rationale": "Initial choice."},
    )
    new = await client.call_tool(
        "remember_decision",
        {
            "claim": "Use SQLite.",
            "rationale": "Durability is required.",
            "supersedes": old.structured_content["id"],
        },
    )
    recalled = await client.call_tool("recall_decisions", {})

    assert recalled.structured_content["result"] == [new.structured_content]


@pytest.mark.anyio
async def test_decisions_persist_across_server_lifecycles(tmp_path):
    async with Client(create_server(tmp_path), raise_exceptions=True) as first:
        await first.call_tool(
            "remember_decision",
            {"claim": "Persist this.", "rationale": "Sessions end."},
        )

    async with Client(create_server(tmp_path), raise_exceptions=True) as reopened:
        recalled = await reopened.call_tool("recall_decisions", {})

    assert [item["claim"] for item in recalled.structured_content["result"]] == ["Persist this."]


@pytest.mark.anyio
async def test_recall_does_not_load_the_embedding_model(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("decision recall must not download the embedding model")

    monkeypatch.setattr("attogradDB.embedding.snapshot_download", fail_download)
    async with Client(create_server(tmp_path), raise_exceptions=True) as connected:
        recalled = await connected.call_tool("recall_decisions", {})

    assert recalled.structured_content["result"] == []
