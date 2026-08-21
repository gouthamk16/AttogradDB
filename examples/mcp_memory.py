"""Record and recall a project decision without an MCP host.

The same SQLite file is what `attograddb-mcp --project-root` would open.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from attogradDB.attodb import VectorStore
from attogradDB.memory import init_memory, recall_decisions, remember_decision


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "payments"
        root.mkdir()
        db = root / ".attograd-memory.db"

        store = VectorStore(path=str(db))
        init_memory(store.db)
        written = remember_decision(
            store.db,
            project=root.name,
            claim="Use SQLite, not Redis, for the queue.",
            rationale="We need durability the current Redis setup does not give us.",
            scope="storage",
            evidence="examples/quickstart.py",
        )
        active = recall_decisions(store.db, root.name)
        store.close()

        print(f"remembered #{written.id}: {written.claim}")
        print(f"active decisions: {len(active)}")
        for decision in active:
            print(f"  {decision.id}  {decision.claim}")


if __name__ == "__main__":
    main()
