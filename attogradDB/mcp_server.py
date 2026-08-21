import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
except ImportError as exc:
    raise ImportError(
        'attograddb-mcp requires the optional MCP extra. '
        'Install it with: pip install "attogradDB[mcp]"'
    ) from exc

from attogradDB.attodb import VectorStore
from attogradDB.memory import Decision, init_memory
from attogradDB.memory import recall_decisions as read_decisions
from attogradDB.memory import remember_decision as write_decision

INSTRUCTIONS = """
Call recall_decisions before planning or editing so active project decisions are in context.
Write durable decisions explicitly with remember_decision. When a new decision replaces an
existing one, pass its id as supersedes; never leave contradictory decisions active.
""".strip()


@dataclass
class AppContext:
    store: VectorStore
    project: str


def _lifespan(root: Path, path: Path):
    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
        store = VectorStore(path=str(path))
        init_memory(store.db)
        try:
            yield AppContext(store=store, project=root.name)
        finally:
            store.close()

    return lifespan


# Async handlers stay on the event-loop thread; sync MCP handlers run in a worker thread,
# which cannot use the SQLite connection created by the server lifespan.
async def remember_decision(
    ctx: Context[AppContext],
    claim: str,
    rationale: str,
    scope: str | None = None,
    evidence: str | None = None,
    supersedes: int | None = None,
) -> Decision:
    """Persist a project decision, optionally superseding an active decision by id."""
    app = ctx.request_context.lifespan_context
    return write_decision(
        app.store.db,
        app.project,
        claim,
        rationale,
        scope=scope,
        evidence=evidence,
        supersedes=supersedes,
    )


async def recall_decisions(ctx: Context[AppContext]) -> list[Decision]:
    """Return every active decision for this project in insertion order."""
    app = ctx.request_context.lifespan_context
    return read_decisions(app.store.db, app.project)


def create_server(
    project_root: str | Path,
    db_path: str | Path | None = None,
) -> MCPServer[AppContext]:
    """Create a project-scoped decision-memory server."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError(f"project root does not exist or is not a directory: {root}")
    path = Path(db_path).resolve() if db_path else root / ".attograd-memory.db"

    server = MCPServer(
        "AttogradDB Memory",
        instructions=INSTRUCTIONS,
        lifespan=_lifespan(root, path),
    )
    server.add_tool(remember_decision, structured_output=True)
    server.add_tool(recall_decisions, structured_output=True)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AttogradDB decision memory over MCP stdio")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--db", help="Override the default <project-root>/.attograd-memory.db")
    args = parser.parse_args()
    create_server(args.project_root, args.db).run()


if __name__ == "__main__":
    main()
