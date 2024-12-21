# AttogradDB

A lightweight document based vector store for fast and efficient semantic retrieval. Lightning fast vector-based search for NoSQL and plaintext documents, embedded using BERT. 

Version 0.3.1

## Features

- NoSQL Key Value Store
- Plaintext document processing
- Document Embedding
- Customizable Vector Store
- HNSW Indexing
- Semantic search for NoSQL documents


## Installation

pip package will be available in the upcoming release.

Clone source

```bash
git clone https://github.com/gouthamk16/AttogradDB.git
```
Setup and activate python virtual environment

```bash
cd AttogradDB
python -m venv .venv
source .venv/bin/activate
```
Build setup dependencies
```bash
pip install -e .
```

## Usage

Examples can be found at `AttogradDB/examples`

## Documentation

### VectorStore

-   `add_text(vector_id, input_data)` Add a single vectorized text to the database.

-   `add_documents(docs)` Bulk-add a list of documents.

-   `get_vector(vector_id, decode_results=False)` Retrieve a stored vector by its ID.

-   `get_similar(query_text, top_n=5, decode_results=True)` Find top N similar vectors for a given query.

### keyValueStore

-   `add(data)` Add a new dictionary to the JSON file.

-   `search(key, value)` Search for entries by a specific key-value pair.

-   `toVector(indexing, embedding_model)` Convert the key-value store data into a document stored in the vector database.

### Embedding

#### `BertEmbedding`

-   Generates BERT-based embeddings for input text.

-   Supports reverse mapping from embeddings back to text.

### Indexing

#### `HNSW`

-   Implements Hierarchical Navigable Small World indexing.

-   Provides efficient approximate nearest-neighbor search for large data.

#### `Clustered Brute-Force`

-   Implements brute-force search of clustered documents.

-   Lightspeed search for small to medium sized documents and NoSQL databases.

## Roadmap

- Add support for GPU-accelerated embedding generation and vector search using cuda.
- C/Rust backend for similarity search and indexing.
- Performance logging for HNSW indexing. 
- Publishing the library on PyPI.
- Adding support for more embedding models and indexing methods.

## Contributing

Contributions are welcome! If you encounter bugs or have feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.