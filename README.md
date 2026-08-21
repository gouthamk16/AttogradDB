# AttogradDB

A lightweight, local-first vector store for semantic retrieval. One SQLite file, no server, no index to maintain. Built for scoped retrieval -- search one project or one session without the rest bleeding in.

Version 1.0.0

[![PyPI Downloads](https://static.pepy.tech/badge/attograddb)](https://pepy.tech/projects/attograddb)

Docs: [https://gouthamk16.github.io/AttogradDB/](https://gouthamk16.github.io/AttogradDB/)
Python API, Claude Code / Cursor / Codex plugins, other MCP harnesses, and marketplace
status. Runnable copies: `examples/quickstart.py`, `examples/mcp_memory.py`.

## Features

- Single-file SQLite storage, no server and no separate index
- Scoped search by project, session, or kind — one value or several at once
- Deletion as a first-class operation -- forgetting matters as much as remembering
- Plaintext, PDF and JSON ingestion with overlap-aware chunking
- Exhaustive exact search: no approximate-recall tradeoff
- Matryoshka dimensions: search at 256-d, keep the full 1024-d on disk
- Project-scoped decision memory over MCP with explicit supersession

## Installation

### Method 1: Install the PyPI package

```bash
pip install attogradDB
```

Install the optional MCP server with:

```bash
pip install "attogradDB[mcp]"
```

### Method 2: Clone and build from source

```bash
git clone https://github.com/gouthamk16/AttogradDB.git
```
Setup and activate python virtual environment

```bash
cd AttogradDB
python3 -m venv .venv
source .venv/bin/activate
```
Install in editable mode with the dev extras and run the tests
```bash
pip install -e ".[dev]"
python -m pytest attogradDB/tests
```

## Usage

```python
from attogradDB import VectorStore

store = VectorStore(path="memory.db")          # omit path for an in-memory store

ids = store.add(
    ["retry logic lives in client.py", "we ruled out Redis: no durability"],
    project="payments",
    session="2026-08-18",
)

for doc_id, score, text in store.search("why not redis", top_n=3):
    print(score, text)

store.search("redis", project="payments")      # scoped
store.delete(session="2026-08-18")             # forget
store.close()
```

`add`, `search`, and `delete` take `project` / `session` / `kind` as a value or a list. Decision memory is a separate table in the same file, exposed over MCP (`attograddb-mcp --project-root ...`) as `remember_decision` and `recall_decisions`. Full docs: [https://gouthamk16.github.io/AttogradDB/](https://gouthamk16.github.io/AttogradDB/).

## Design notes

Search scans every candidate vector rather than using an ANN index. Measured at 100k chunks
of 768 dimensions: 8 ms unfiltered, 3 ms scoped to 2% of the store. An HNSW index is faster
unfiltered but roughly 275x slower once a filter is applied, because filtering disconnects its
graph while it only shortens an exhaustive scan. Since almost every query here is scoped, and
8 ms is invisible next to an LLM call, the index is not worth its cost. Past roughly a million
vectors a scan reaches ~230 ms and that trade changes; `CLAUDE.md` holds the full numbers.

## Roadmap

- Add a method for performance logging.
- LLM based chunking.
- Adding support for more embedding models and indexing methods.
- Adding support for more document types (currently we have pdf, json and txt. Need to add support for docx and images).
- Extract decision candidates from session traces.
- Retrieve long-tail facts using the agent's working context rather than only the user query.
- Verify stored evidence against repository state and compact oversized active decision sets.

## Contributing

Contributions are welcome! If you encounter bugs or have feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
