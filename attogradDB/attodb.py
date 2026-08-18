import json
import os
import uuid
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


class keyValueStore:
    def __init__(self, json_path="data.json"):
        '''
        Initialize key-value store with master and default collections.
        Creates directory structure if it doesn't exist.
        '''
        self.base_path = json_path.rsplit('.', 1)[0]
        os.makedirs(self.base_path, exist_ok=True)
        
        # Initialize master collection structure
        self.master_collections = {}
        self.current_master = "default"
        self.current_collection = "default"
        
        # Create default master collection and collection
        master_path = os.path.join(self.base_path, "default")
        os.makedirs(master_path, exist_ok=True)
        collection_path = os.path.join(master_path, "default.json")
        
        if not os.path.exists(collection_path):
            with open(collection_path, "w") as f:
                json.dump({"documents": []}, f)

    def create_master_collection(self, name):
        '''Create a new master collection'''
        path = os.path.join(self.base_path, name)
        os.makedirs(path, exist_ok=True)
        self.master_collections[name] = {}
        
    def create_collection(self, name, master_collection="default"):
        '''Create a new collection within a master collection'''
        master_path = os.path.join(self.base_path, master_collection)
        collection_path = os.path.join(master_path, f"{name}.json")
        
        if not os.path.exists(collection_path):
            with open(collection_path, "w") as f:
                json.dump({"documents": []}, f)

    def use_collection(self, collection, master_collection="default"):
        '''Switch to a specific collection'''
        self.current_master = master_collection
        self.current_collection = collection

    def add(self, data, doc_id=None):
        '''
        Add document(s) to current collection
        '''
        collection_path = os.path.join(self.base_path, self.current_master, 
                                     f"{self.current_collection}.json")
        
        with open(collection_path, "r") as f:
            try:
                collection_data = json.load(f)
            except json.JSONDecodeError:
                collection_data = {"documents": []}

        if isinstance(data, list):
            for idx, doc in enumerate(data):
                doc_with_id = doc.copy()
                if doc_id is None:
                    doc_with_id["_id"] = str(uuid.uuid4())
                else:
                    doc_with_id["_id"] = f"{doc_id}_{idx}"
                collection_data["documents"].append(doc_with_id)
        else:
            doc_with_id = data.copy()
            doc_with_id["_id"] = doc_id or str(uuid.uuid4())
            collection_data["documents"].append(doc_with_id)

        with open(collection_path, "w") as f:
            json.dump(collection_data, f, indent=4)

    def add_json(self, json_file):
        '''
        Add documents from a JSON file to current collection
        
        Args:
            json_file (str): Path to JSON file containing documents
        '''
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Handle both single document and list of documents
            if isinstance(data, dict):
                self.add(data)
            elif isinstance(data, list):
                self.add(data)
            else:
                raise ValueError("JSON file must contain either a single document or list of documents")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_file}")

    def __getitem__(self, key):
        '''Retrieve document by index'''
        collection_path = os.path.join(self.base_path, self.current_master,
                                     f"{self.current_collection}.json")
        with open(collection_path, "r") as f:
            collection_data = json.load(f)
        return collection_data["documents"][key]

    def search(self, key, value):
        '''Search documents by key-value pair'''
        collection_path = os.path.join(self.base_path, self.current_master,
                                     f"{self.current_collection}.json")
        with open(collection_path, "r") as f:
            collection_data = json.load(f)
        
        return [doc for doc in collection_data["documents"] if doc.get(key) == value]

    def toVector(self, indexing="brute-force", embedding_model="bert", collection=None, master_collection=None):
        '''
        Convert collection documents to vector store
        '''
        if collection:
            self.use_collection(collection, master_collection or self.current_master)
            
        collection_path = os.path.join(self.base_path, self.current_master,
                                     f"{self.current_collection}.json")
        
        with open(collection_path, "r") as f:
            collection_data = json.load(f)

        docs = [json.dumps(doc, separators=(',', ':')) for doc in collection_data["documents"]]
        
        vectorStore = VectorStore(indexing=indexing, embedding_model=embedding_model)
        vectorStore.add_documents(docs)
        
        return vectorStore