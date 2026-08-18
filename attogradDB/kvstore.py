import json
import uuid
import warnings
from pathlib import Path

from attogradDB.attodb import VectorStore


class keyValueStore:
    """JSON-file-backed document store: master collection -> collection -> documents.

    All filesystem access funnels through _path/_read/_write so the backend can be
    replaced without touching the public methods.
    """

    def __init__(self, json_path: str = "data.json"):
        self.base_path = Path(json_path).with_suffix("")
        self.current_master = "default"
        self.current_collection = "default"
        self.create_collection("default")

    def _path(self, collection: str | None = None, master: str | None = None) -> Path:
        collection = collection or self.current_collection
        master = master or self.current_master
        return self.base_path / master / f"{collection}.json"

    def _read(self, collection: str | None = None, master: str | None = None) -> list[dict]:
        path = self._path(collection, master)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"Collection not found: {path}") from None
        except json.JSONDecodeError as err:
            raise ValueError(f"Collection file {path} is not valid JSON: {err}") from None

        if "documents" not in payload:
            raise ValueError(f"Collection file {path} has no 'documents' key")
        return payload["documents"]

    def _write(
        self, documents: list[dict], collection: str | None = None, master: str | None = None
    ) -> None:
        path = self._path(collection, master)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"documents": documents}, indent=4))

    def create_master_collection(self, name: str) -> None:
        (self.base_path / name).mkdir(parents=True, exist_ok=True)

    def create_collection(self, name: str, master_collection: str = "default") -> None:
        if not self._path(name, master_collection).exists():
            self._write([], name, master_collection)

    def use_collection(self, collection: str, master_collection: str = "default") -> None:
        path = self._path(collection, master_collection)
        if not path.exists():
            raise FileNotFoundError(
                f"Collection not found: {path}. Create it with create_collection() first."
            )
        self.current_master = master_collection
        self.current_collection = collection

    def add(self, data: dict | list[dict], doc_id: str | None = None) -> None:
        """Add a document, or a list of documents, to the current collection."""
        documents = self._read()
        batch = data if isinstance(data, list) else [data]

        for idx, doc in enumerate(batch):
            if not isinstance(doc, dict):
                raise TypeError(f"Documents must be dicts, got {type(doc).__name__}")
            stored = dict(doc)
            if doc_id is None:
                stored["_id"] = str(uuid.uuid4())
            elif isinstance(data, list):
                stored["_id"] = f"{doc_id}_{idx}"
            else:
                stored["_id"] = doc_id
            documents.append(stored)

        self._write(documents)

    def add_json(self, json_file: str) -> None:
        """Add documents from a JSON file to the current collection."""
        # Scoped tightly to the input file so a failure writing the collection is not
        # reported as the caller's file being missing.
        try:
            data = json.loads(Path(json_file).read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file}") from None
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON in {json_file}: {err}") from None

        if not isinstance(data, (dict, list)):
            raise ValueError("JSON file must contain a document or a list of documents")
        self.add(data)

    def __getitem__(self, key):
        return self._read()[key]

    def search(self, key: str, value) -> list[dict]:
        return [doc for doc in self._read() if doc.get(key) == value]

    def to_vector(
        self,
        indexing: str = "brute-force",
        embedding_model: str = "bert",
        collection: str | None = None,
        master_collection: str | None = None,
    ) -> VectorStore:
        """Promote this collection's documents into a VectorStore."""
        if collection:
            self.use_collection(collection, master_collection or self.current_master)

        # _id is a storage detail; embedding it feeds 36 random characters to the model.
        docs = [
            json.dumps({k: v for k, v in doc.items() if k != "_id"}, separators=(",", ":"))
            for doc in self._read()
        ]

        store = VectorStore(indexing=indexing, embedding_model=embedding_model)
        store.add_documents(docs)
        return store

    def toVector(self, *args, **kwargs) -> VectorStore:
        warnings.warn(
            "toVector() is deprecated, use to_vector()", DeprecationWarning, stacklevel=2
        )
        return self.to_vector(*args, **kwargs)
