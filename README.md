# AttogradDB

A lightweight document based vector store for fast and efficient semantic retrieval. Lightning fast vector-based search for NoSQL and plaintext documents, embedded using BERT. 

Version 0.5.0

[![PyPI Downloads](https://static.pepy.tech/badge/attograddb)](https://pepy.tech/projects/attograddb)

## Features

- NoSQL Key Value Store
- Plaintext document processing
- Document Embedding
- Customizable Vector Store
- HNSW Indexing
- Semantic search for NoSQL documents


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

## Documentation

### VectorStore

-   `__init__(indexing="hnsw", embedding_model="bert", save_path=None, load_path=None)` Initialize a vector store. `indexing` is `"hnsw"` or `"brute-force"`; an unknown value raises `ValueError`. Setting `save_path` writes the index after every `add_text`; setting `load_path` restores one at construction.

-   `add_text(vector_id, input_data)` Add a single text document to the vector store after embedding.

-   `add_documents(docs)` Bulk-add a list of JSON documents to the vector store after converting to text and embedding.

-   `get_similar(query_text, top_n=5, decode_results=True)` Find top N semantically similar documents for a given query text. Returns list of tuples containing (vector_id, similarity_score, document_text) if decode_results=True, otherwise returns (vector_id, similarity_score).

-   `similarity(vector_a, vector_b, method="cosine")` Calculate cosine similarity between two vectors.

-   `save_index(path=None)` Write the vectors and their source text to a JSON file. Falls back to the constructor's `save_path`, then to `stored_indices.json`.

-   `load_index(path=None)` Restore an index written by `save_index`, rebuilding the HNSW graph. Falls back to the same defaults.

### keyValueStore

-   `create_master_collection(name)` Create a new master collection to group related collections.

-   `create_collection(name, master_collection="default")` Create a new collection within a master collection.

-   `use_collection(collection, master_collection="default")` Switch to a specific collection. Raises `FileNotFoundError` if it does not exist.

-   `add(data, doc_id=None)` Add document(s) to current collection with optional custom IDs.

-   `add_json(json_file)` Add documents from a JSON file to current collection.

-   `search(key, value)` Search documents by key-value pair in current collection.

-   `to_vector(indexing="brute-force", embedding_model="bert", collection=None, master_collection=None)` Convert collection documents to a vector store. The internal `_id` is excluded from the embedded text. (`toVector()` still works but is deprecated.)

### Embedding

#### `BertEmbedding`

-   Generates BERT-based embeddings for input text (768-d, mean-pooled last hidden state).

### Indexing

#### `HNSW`

-   Implements Hierarchical Navigable Small World indexing over `hnswlib`.

-   Provides efficient approximate nearest-neighbor search for large data. Capacity doubles automatically as the store grows.

#### `Clustered Brute-Force`

-   Implements brute-force search of clustered documents.

-   Lightspeed search for small to medium sized text and NoSQL documents.

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