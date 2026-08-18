# AttogradDB

A lightweight, local-first vector store for semantic retrieval. One SQLite file, no server, no index to maintain. Built for scoped retrieval -- search one project or one session without the rest bleeding in.

Version 0.9.0

[![PyPI Downloads](https://static.pepy.tech/badge/attograddb)](https://pepy.tech/projects/attograddb)

## Features

- Single-file SQLite storage, no server and no separate index
- Scoped search by project, session, or kind — one value or several at once
- Deletion as a first-class operation -- forgetting matters as much as remembering
- Plaintext, PDF and JSON ingestion with overlap-aware chunking
- Exhaustive exact search: no approximate-recall tradeoff
- Matryoshka dimensions: search at 256-d, keep the full 1024-d on disk


## Installation

### Method 1: Install the PyPI package

```bash
pip install attogradDB
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

Examples can be found at `AttogradDB/examples`

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

A runnable version is in `examples/quickstart.py`.

## Documentation

### VectorStore

-   `__init__(path=None, embedding_model="qwen3", dim=256, profile="int8-cpu")` Open a store. `path` is a SQLite file; `None` gives an in-memory store discarded on close. `dim` is the search dimension: full 1024-d vectors are always stored, and the in-memory index is truncated to `dim` (Matryoshka), so you can reopen at a different `dim` without re-embedding. Reopening with a different `embedding_model` is refused, since vectors from different models are not comparable.

-   `add(texts, project=None, session=None, kind=None) -> list[int]` Embed and store a string or list of strings. Ids are generated and returned.

-   `search(query, top_n=5, project=None, session=None, kind=None) -> list[tuple[int, float, str]]` Return the closest chunks as `(id, score, text)`, best first. Each scope takes a single value or a list of them, so one query can span several projects or sessions. Omitting a scope searches across all of it.

    ```python
    store.search("why not redis")                              # everything
    store.search("why not redis", project="payments")          # one project
    store.search("why not redis", project=["payments", "api"]) # several
    store.search("what broke", session=["mon", "tue"], kind="note")
    ```

-   `delete(ids=None, project=None, session=None, kind=None) -> int` Delete matching chunks and return how many were removed. Scopes accept a list, matching `search()`. Requires at least one filter, so an empty call cannot wipe the store.

-   `similarity(vector_a, vector_b) -> float` Cosine similarity between two vectors.

-   `len(store)` Number of stored chunks. `store.close()` closes the database.

### Ingestion

-   `TextSplitter(chunk_size=200, chunk_overlap=20)` from `attogradDB.io` — fixed-size character chunking. `split_text(text)` then `get_docs()`.

-   `read_pdf(path)` from `attogradDB.utils` — extract text from every page via pypdf.

### Embedding

#### `QwenEmbedding`

-   Qwen3-Embedding-0.6B (Apache-2.0) via ONNX Runtime, CPU by default. The model downloads on first embed call (~600MB, int8) and is cached by `huggingface_hub`; opening a store does not trigger it.

-   Native output is 1024-d. Queries carry an instruction prefix and documents do not — the model is trained asymmetrically, and the store handles this for you.

-   Embeds one text at a time. Batching gives no CPU throughput gain and padding perturbs the result, so it is deliberately not offered.

-   Two profiles, pairing a model build with an execution provider. The default `int8-cpu` runs identically everywhere including macOS. `fp16-gpu` requires `pip install onnxruntime-directml` or `onnxruntime-gpu`, and falls back to CPU if the accelerator cannot start.

-   **GPU is opt-in and only worth it for bulk ingestion.** Measured on an RTX 4060 over 96 PDF chunks:

    | profile | ingest | query (median) |
    |---|---|---|
    | `int8-cpu` | 9.3 s | **34 ms** |
    | `fp16-gpu` | **4.6 s** | 238 ms |

    Ingest is 2x faster, but search is ~7x slower: DirectML recompiles per input shape and real queries vary in length. If you search more than you ingest — which agent memory does — stay on the default.

-   The two builds produce different vectors (0.913 cosine), so the profile is recorded in the store and reopening under a different one raises rather than silently degrading results.

-   CoreML on macOS is not used automatically, since it can be slower than CPU on a model it has to partition heavily. Opt in explicitly if you measure a win:

    ```python
    from attogradDB.embedding import QwenEmbedding
    QwenEmbedding(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    ```

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
- Tests for lading and saving indexes locally.
- Adding support for more embedding models and indexing methods.
- Adding support for more document types (currently we have pdf, json and txt. Need to add support for docx and images).

## Contributing

Contributions are welcome! If you encounter bugs or have feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.