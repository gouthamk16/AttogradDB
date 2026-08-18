import json
from pathlib import Path

import numpy as np

from attogradDB.embedding import BertEmbedding
from attogradDB.indexing import HNSW

DEFAULT_INDEX_PATH = "stored_indices.json"
INDEXING_METHODS = ("hnsw", "brute-force")
EMBEDDING_MODELS = ("bert",)


class VectorStore:
    def __init__(
        self,
        indexing: str = "hnsw",
        embedding_model: str = "bert",
        save_path: str | None = None,
        load_path: str | None = None,
    ):
        if indexing not in INDEXING_METHODS:
            raise ValueError(f"Unknown indexing {indexing!r}, expected one of {INDEXING_METHODS}")
        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding_model {embedding_model!r}, expected one of {EMBEDDING_MODELS}"
            )

        self.indexing = indexing
        self.save_path = save_path
        self.embedding_model = BertEmbedding()
        self.vector = {}
        self.text = {}
        self.idx = 0
        self.index = HNSW() if indexing == "hnsw" else None
        self._label_ids = {}

        if load_path:
            self.load_index(load_path)

    @staticmethod
    def similarity(vector_a, vector_b, method: str = "cosine") -> float:
        """Cosine similarity. A vector with no direction scores 0 rather than nan."""
        if method != "cosine":
            raise ValueError(f"Unknown similarity method: {method!r}")

        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        if norms == 0:
            return 0.0
        return float(np.dot(a, b) / norms)

    def add_text(self, vector_id: str, input_data: str) -> None:
        """Embed input_data and store it under vector_id."""
        embedding = np.asarray(self.embedding_model.embed(input_data), dtype=np.float32)
        self.vector[vector_id] = embedding
        self.text[vector_id] = input_data

        if self.index is not None:
            self._label_ids[self.index.add_node(embedding)] = vector_id

        if self.save_path:
            self.save_index()

    def add_documents(self, docs: list[str]) -> None:
        """Add documents under generated doc_N ids."""
        for doc in docs:
            self.add_text(f"doc_{self.idx}", doc)
            self.idx += 1

    def get_vector(self, vector_id: str, decode_results: bool = False):
        """Return the stored vector, or its source text when decode_results is set."""
        if decode_results:
            return self.text.get(vector_id)
        return self.vector.get(vector_id)

    def get_similar(self, query_text: str, top_n: int = 5, decode_results: bool = True) -> list:
        """Return the top_n most similar documents as (id, score) or (id, score, text)."""
        query_vector = np.asarray(self.embedding_model.embed(query_text), dtype=np.float32)

        if self.index is not None:
            results = [
                (self._label_ids[label], score)
                for label, score in self.index.search(query_vector, top_n)
            ]
        else:
            results = [
                (vector_id, self.similarity(query_vector, vector))
                for vector_id, vector in self.vector.items()
            ]
            results.sort(key=lambda pair: pair[1], reverse=True)
            results = results[:top_n]

        if decode_results:
            return [(vector_id, score, self.text.get(vector_id)) for vector_id, score in results]
        return results

    def save_index(self, path: str | None = None) -> None:
        """Write vectors and their source text to disk as JSON."""
        path = path or self.save_path or DEFAULT_INDEX_PATH
        payload = {
            "indexing": self.indexing,
            "vectors": {vid: vec.tolist() for vid, vec in self.vector.items()},
            "texts": self.text,
        }
        Path(path).write_text(json.dumps(payload))

    def load_index(self, path: str | None = None) -> None:
        """Restore vectors and text from disk, rebuilding the ANN graph if one is in use."""
        path = path or self.save_path or DEFAULT_INDEX_PATH
        try:
            payload = json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"No saved index at {path}") from None
        except json.JSONDecodeError as err:
            raise ValueError(f"Saved index at {path} is not valid JSON: {err}") from None

        self.vector = {
            vid: np.asarray(vec, dtype=np.float32) for vid, vec in payload["vectors"].items()
        }
        self.text = payload.get("texts", {})
        self.idx = len(self.vector)

        if self.index is not None:
            self.index = HNSW()
            self._label_ids = {
                self.index.add_node(vec): vid for vid, vec in self.vector.items()
            }


# Re-exported for backwards compatibility. Imported at the bottom because
# kvstore imports VectorStore from this module.
from attogradDB.kvstore import keyValueStore  # noqa: E402,F401
