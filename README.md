# AttogradDB

A lightweight document based vector store for fast and efficient semantic retrieval. Lightning fast vector-based search for NoSQL and plaintext documents, embedded using BERT. 

Version 0.4.2 (pip package for 0.4.2 not available yet)

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
pip3 install attogradDB==0.3.1
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
Build setup dependencies
```bash
pip3 install -e .
```

## Usage

Examples can be found at `AttogradDB/examples`

## Documentation

### VectorStore

-   `__init__(indexing="hnsw", embedding_model="bert", save_index=False, save_path=None)` Initialize vector store with specified indexing and embedding model.

-   `add_text(vector_id, input_data)` Add a single text document to the vector store after embedding.

-   `add_documents(docs)` Bulk-add a list of JSON documents to the vector store after converting to text and embedding.

-   `get_similar(query_text, top_n=5, decode_results=True)` Find top N semantically similar documents for a given query text. Returns list of tuples containing (vector_id, similarity_score, document_text) if decode_results=True, otherwise returns (vector_id, similarity_score).

-   `similarity(vector_a, vector_b, method="cosine")` Calculate cosine similarity between two vectors.

-   `save_path()` Save the vector index to a json file. `None` by default.

-   `load_path()` Loads a json file containing indexes into the VectorDB. `None` by default.

### keyValueStore

-   `create_master_collection(name)` Create a new master collection to group related collections.

-   `create_collection(name, master_collection="default")` Create a new collection within a master collection.

-   `use_collection(collection, master_collection="default")` Switch to a specific collection for operations.

-   `add(data, doc_id=None)` Add document(s) to current collection with optional custom IDs.

-   `add_json(json_file)` Add documents from a JSON file to current collection.

-   `search(key, value)` Search documents by key-value pair in current collection.

-   `toVector(indexing="brute-force", embedding_model="bert", collection=None, master_collection=None)` Convert collection documents to vector store with specified indexing and embedding model.

### Embedding

#### `BertEmbedding`

-   Generates BERT-based embeddings for input text.

-   Supports reverse mapping from embeddings back to text.

### Indexing

#### `HNSW`

-   Implements Hierarchical Navigable Small Words indexing.

-   Provides efficient approximate nearest-neighbor search for large data.

#### `Clustered Brute-Force`

-   Implements brute-force search of clustered documents.

-   Lightspeed search for small to medium sized text and NoSQL documents.

## Roadmap

- Add a method for performance logging.
- LLM based chunking.
- Tests for lading and saving indexes locally.
- Add support for GPU-accelerated embedding generation and vector search using cuda.
- C/Rust backend for similarity search and indexing.
- Adding support for more embedding models and indexing methods.
- Adding support for more document types (currently we have pdf, json and txt. Need to add support for docx and images).

## Contributing

Contributions are welcome! If you encounter bugs or have feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.