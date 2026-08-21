# AttogradDB

**Light speed memory for your agents.** A local-first vector store in one SQLite file — no server, no index to tune — plus project decision memory over MCP for Claude Code, Cursor, and Codex.

[![PyPI](https://img.shields.io/pypi/v/attogradDB)](https://pypi.org/project/attogradDB/)
[![Downloads](https://static.pepy.tech/badge/attograddb)](https://pepy.tech/projects/attograddb)

Docs: **[gouthamk16.github.io/AttogradDB](https://gouthamk16.github.io/AttogradDB/)**

## Install

```bash
pip install attogradDB            # library
pip install "attogradDB[mcp]"     # + MCP decision-memory server
```

## Usage

```python
from attogradDB import VectorStore

store = VectorStore(path="memory.db")   # omit path for in-memory
store.add(["retry logic lives in client.py"], project="payments")

for doc_id, score, text in store.search("where is retry handled", project="payments"):
    print(score, text)

store.close()
```

`add`, `search`, and `delete` scope by `project` / `session` / `kind` (a value or a list). Decision memory lives in the same file and is exposed over MCP as `remember_decision` and `recall_decisions` — see the [docs](https://gouthamk16.github.io/AttogradDB/).

## License

MIT — see [`LICENSE`](LICENSE).
