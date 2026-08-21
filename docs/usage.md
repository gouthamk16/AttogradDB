# AttogradDB usage guide

AttogradDB is a local-first vector store. One SQLite file holds the documents; a normalised copy of the vectors lives in memory for search. There is no server to run and no approximate index to tune.

Use this guide for the full API. The README is the short entry point; runnable copies of the examples below live in `examples/`.

## Install

From PyPI:

```bash
pip install attogradDB
```

Decision memory over MCP is optional:

```bash
pip install "attogradDB[mcp]"
```

From source:

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python -m pytest attogradDB/tests
```

Python 3.11 or newer. The first embedding call downloads Qwen3-Embedding-0.6B (~600MB, int8) and caches it via `huggingface_hub`. Opening a store does not trigger that download.

## Open a store

```python
from attogradDB import VectorStore

store = VectorStore()                      # in-memory, discarded on close
store = VectorStore(path="memory.db")      # durable SQLite file
```

`path=None` (the default) is an in-memory store. A file path creates or reopens a SQLite database.

Optional constructor arguments:

- `embedding_model="qwen3"` — the only model shipped today. Reopening a store under a different name raises, because vectors from different models are not comparable.
- `dim=256` — search dimension. Full 1024-d vectors always go to disk; the in-memory index is truncated to `dim` (Matryoshka) and re-normalised. You can reopen the same file at a different `dim` without re-embedding.

Call `store.close()` when you are done. `len(store)` is the number of stored chunks.

## Add, search, scope, forget

Ids are generated. You never invent them.

```python
from attogradDB import VectorStore

store = VectorStore(path="memory.db")

ids = store.add(
    [
        "The retry logic lives in client.py and backs off exponentially.",
        "We ruled out Redis for the queue: no durability guarantee we could rely on.",
    ],
    project="payments",
    session="2026-08-18",
)

for doc_id, score, text in store.search("why not redis", top_n=3):
    print(score, text)

store.search("redis", project="payments")                 # one project
store.search("redis", project=["payments", "api"])        # several
store.search("what broke", session=["mon", "tue"], kind="note")

removed = store.delete(session="2026-08-18")
store.close()
```

A runnable version is `examples/quickstart.py`.

### `add`

`add(texts, project=None, session=None, kind=None) -> list[int]`

Embeds a string or a list of strings and stores them. An empty list is a no-op. Scope fields are optional labels used later by `search` and `delete`.

### `search`

`search(query, top_n=5, project=None, session=None, kind=None) -> list[tuple[int, float, str]]`

Returns `(id, score, text)`, best first. Scores are cosine similarities against the truncated, normalised index. Each scope takes a single value or a list. Omit a scope to search across all of it. An empty list matches nothing, not everything.

An empty store, or a scope that matches nothing, returns `[]`. `top_n` larger than the store returns the whole store.

### `delete`

`delete(ids=None, project=None, session=None, kind=None) -> int`

Removes matching chunks and returns how many went. Requires at least one filter, so an empty call cannot wipe the store.

## Ingest PDFs and long text

```python
from attogradDB import VectorStore, read_pdf
from attogradDB.io import TextSplitter

store = VectorStore(path="memory.db")
splitter = TextSplitter(chunk_size=300, chunk_overlap=20)
splitter.split_text(read_pdf("paper.pdf"))
store.add(splitter.get_docs(), project="papers", kind="pdf")
```

`TextSplitter` is fixed-size character chunking. `chunk_size` must be positive; `chunk_overlap` must be in `[0, chunk_size)` or the splitter would not advance.

`read_pdf(path)` concatenates text from every page via pypdf. Pages without a text layer contribute nothing rather than aborting the read.

## Embedding

Documents and queries are embedded differently on purpose. Qwen3 is trained asymmetrically: `embed_query()` prepends an instruction, `embed_document()` does not. The store handles this; collapsing them into one call costs retrieval quality.

Native output is 1024-d. Search uses the truncated `dim` (256 by default). Embedding is one text at a time: batching measured flat on CPU, and padding changes the vector.

The default path is CPU-only int8. A GPU path was measured and removed: ingest was about 2× faster, search about 7× slower and far less predictable, because DirectML recompiles per input shape. See **Measured decisions** in `CLAUDE.md` before adding one back.

## Decision memory over MCP

Important project decisions are not a similarity-search problem. A later task can depend on a prior constraint without using similar words. The MCP server therefore keeps a small structured set of *active* decisions and returns all of them on recall. Vector search remains for the larger body of documents and traces.

### Run the server

```bash
attograddb-mcp --project-root /path/to/project
```

The default database is `/path/to/project/.attograd-memory.db`. Pass `--db` to put it somewhere else. The project name is the directory name of `--project-root`.

Generic MCP host config:

```json
{
  "command": "attograddb-mcp",
  "args": ["--project-root", "/path/to/project"]
}
```

### Tools

- `recall_decisions()` — every active decision for this project, in insertion order.
- `remember_decision(claim, rationale, scope=None, evidence=None, supersedes=None)` — write durable project truth. Pass the prior decision's id as `supersedes` when replacing it. Missing, cross-project, or already-superseded targets are rejected without a partial write.

Each decision has `claim`, `rationale`, optional `scope` and `evidence`, `status` (`active` or `superseded`), and `superseded_by`.

The server instructs the model to call `recall_decisions` before planning or editing. Generic MCP cannot force a host to inject that context. The host or agent must follow the instruction. Automatic transcript extraction and Cursor- or Claude-specific hooks are not part of this release.

A library-level counterpart (no MCP host required) is `examples/mcp_memory.py`.

## What this store does not do

Search is an exhaustive scan of the in-memory index, not an ANN graph. At the scale this is built for — hundreds of thousands of scoped chunks — that is a few milliseconds. An HNSW index wins unfiltered and loses badly once a filter is applied; almost every query here is scoped. Revisit past roughly a million vectors.

There is one store. Decision records live in the same SQLite file as chunks, in a `decisions` table, not a second engine.
