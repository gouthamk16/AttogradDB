import sqlite3
import time

import numpy as np

from attogradDB.embedding import NATIVE_DIM, QwenEmbedding

EMBEDDING_MODELS = ("qwen3",)
SCOPE_FIELDS = ("project", "session", "kind")

# A scope is one value, several values, or unset (meaning "don't filter on this").
Scope = str | list[str] | None

# Vectors are stored at NATIVE_DIM and truncated to this for searching. Qwen3 is trained
# with Matryoshka representation learning, so early dimensions carry the most signal;
# top-1 hits survived truncation to 64 in local testing, and 256 leaves headroom.
DEFAULT_DIM = 256

# Below this share of the store, copying the matching rows beats scanning everything;
# above it the copy dominates. Measured at 100k x 768d; see "Measured decisions"
# in CLAUDE.md before changing this.
SUBSET_SCAN_THRESHOLD = 0.10

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    vector     BLOB NOT NULL,
    project    TEXT,
    session    TEXT,
    kind       TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_scope ON chunks(project, session, kind);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class VectorStore:
    """Vectors live in SQLite; a normalised copy is held in memory for scanning.

    Search is an exhaustive dot product. At the scale this is built for -- hundreds of
    thousands of chunks -- that costs single-digit milliseconds unfiltered and far less
    once scoped, which is cheaper than maintaining an ANN index that degrades under
    exactly the filtered queries this store exists to serve.
    """

    def __init__(
        self,
        path: str | None = None,
        embedding_model: str = "qwen3",
        dim: int = DEFAULT_DIM,
    ):
        if embedding_model not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding_model {embedding_model!r}, expected one of {EMBEDDING_MODELS}"
            )
        if not 0 < dim <= NATIVE_DIM:
            raise ValueError(f"dim must be in (0, {NATIVE_DIM}], got {dim}")

        self.dim = dim
        self.embedding_model = QwenEmbedding()
        self.db = sqlite3.connect(path or ":memory:")
        self.db.executescript(SCHEMA)
        self._check_model(embedding_model)
        self._load()

    def _check_model(self, embedding_model: str) -> None:
        """Refuse to mix vectors from different models in one store.

        Embedding models turn over faster than stores do, and a swapped model does not
        error -- it quietly returns worse results. This makes that case loud.
        """
        stored = dict(self.db.execute("SELECT key, value FROM meta"))
        if not stored:
            with self.db:
                self.db.execute("INSERT INTO meta VALUES ('model', ?)", (embedding_model,))
        elif stored.get("model") != embedding_model:
            raise ValueError(
                f"store was built with model={stored.get('model')!r}, cannot reopen as "
                f"{embedding_model!r} -- vectors are not comparable"
            )

    def _load(self) -> None:
        rows = self.db.execute(
            f"SELECT id, vector, {', '.join(SCOPE_FIELDS)} FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            self._ids = np.empty(0, dtype=np.int64)
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._scopes = {field: np.empty(0, dtype="U1") for field in SCOPE_FIELDS}
            return
        self._ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
        stored = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        self._vectors = self._truncate(stored)
        self._scopes = {
            field: np.array([row[2 + i] or "" for row in rows], dtype="U")
            for i, field in enumerate(SCOPE_FIELDS)
        }

    def _truncate(self, vectors: np.ndarray) -> np.ndarray:
        """Cut to self.dim and renormalise -- slicing a unit vector does not leave one."""
        cut = vectors[..., : self.dim]
        norms = np.linalg.norm(cut, axis=-1, keepdims=True)
        return np.divide(cut, norms, out=np.zeros_like(cut), where=norms != 0)

    def _embed_query(self, text: str) -> np.ndarray:
        """Truncated, normalised query vector, ready to dot against the index."""
        raw = np.asarray(self.embedding_model.embed_query(text), dtype=np.float32)
        return self._truncate(raw)

    @staticmethod
    def _scope_sql(project, session, kind) -> tuple[list[str], list]:
        """SQL clauses for the given scope. Used by delete(); search() matches in memory."""
        clauses, params = [], []
        for field, value in zip(SCOPE_FIELDS, (project, session, kind)):
            if value is None:
                continue
            if isinstance(value, str):
                clauses.append(f"{field} = ?")
                params.append(value)
            else:
                values = list(value)
                if not values:
                    clauses.append("0")  # an empty set matches nothing
                    continue
                clauses.append(f"{field} IN ({','.join('?' * len(values))})")
                params.extend(values)
        return clauses, params

    def _scope_rows(self, project, session, kind) -> np.ndarray | None:
        """Row indices matching the scope, or None when nothing was scoped.

        Matched in memory rather than via SQL: pulling 90k ids back out of SQLite and
        intersecting them costs ~20ms, against ~0.2ms for a numpy comparison.
        """
        active = [
            (field, value)
            for field, value in zip(SCOPE_FIELDS, (project, session, kind))
            if value is not None
        ]
        if not active:
            return None
        mask = np.ones(len(self._ids), dtype=bool)
        for field, value in active:
            column = self._scopes[field]
            if isinstance(value, str):
                mask &= column == value
            else:
                mask &= np.isin(column, list(value))
        return np.flatnonzero(mask)

    def _texts(self, ids) -> dict[int, str]:
        placeholders = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", [int(i) for i in ids]
        )
        return dict(rows)

    def add(
        self,
        texts: str | list[str],
        project: str | None = None,
        session: str | None = None,
        kind: str | None = None,
    ) -> list[int]:
        """Embed and store text. Returns the generated ids."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        vectors = self.embedding_model.embed_documents(texts)
        created_at = time.time()
        ids = []
        with self.db:
            for text, vector in zip(texts, vectors):
                cursor = self.db.execute(
                    "INSERT INTO chunks (text, vector, project, session, kind, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (text, vector.tobytes(), project, session, kind, created_at),
                )
                ids.append(cursor.lastrowid)

        stacked = self._truncate(np.stack(vectors))
        self._ids = np.concatenate([self._ids, np.array(ids, dtype=np.int64)])
        self._vectors = stacked if self._vectors.size == 0 else np.vstack([self._vectors, stacked])
        for field, value in zip(SCOPE_FIELDS, (project, session, kind)):
            added = np.array([value or ""] * len(texts), dtype="U")
            self._scopes[field] = np.concatenate([self._scopes[field], added])
        return ids

    def search(
        self,
        query: str,
        top_n: int = 5,
        project: Scope = None,
        session: Scope = None,
        kind: Scope = None,
    ) -> list[tuple[int, float, str]]:
        """Return the top_n closest chunks as (id, score, text), best first.

        Each scope takes a single value or a list of them, so one query can span
        several projects or sessions. Omit a scope to search across all of it.
        """
        if len(self._ids) == 0:
            return []

        if top_n <= 0:
            return []

        rows = self._scope_rows(project, session, kind)
        if rows is not None and len(rows) == 0:
            return []

        query_vector = self._embed_query(query)

        if rows is None:
            scores = self._vectors @ query_vector
            ids = self._ids
        elif len(rows) <= SUBSET_SCAN_THRESHOLD * len(self._ids):
            scores = self._vectors[rows] @ query_vector
            ids = self._ids[rows]
        else:
            # Past the threshold, materialising self._vectors[rows] costs more than
            # scanning everything -- at 40% selectivity the copy alone is 4x the scan.
            scores = self._vectors @ query_vector
            keep = np.zeros(len(scores), dtype=bool)
            keep[rows] = True
            scores = np.where(keep, scores, -np.inf)
            ids = self._ids

        k = min(top_n, len(scores) if rows is None else len(rows))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        chosen = ids[top]
        texts = self._texts(chosen)
        return [(int(i), float(s), texts[int(i)]) for i, s in zip(chosen, scores[top])]

    def delete(
        self,
        ids: list[int] | None = None,
        project: Scope = None,
        session: Scope = None,
        kind: Scope = None,
    ) -> int:
        """Delete chunks matching the given filters. Returns how many were removed.

        Scopes accept a single value or a list, matching search().
        """
        if ids is not None and not ids:
            return 0

        clauses, params = self._scope_sql(project, session, kind)
        if ids is not None:
            clauses.append(f"id IN ({','.join('?' * len(ids))})")
            params.extend(int(i) for i in ids)
        if not clauses:
            raise ValueError("delete() needs at least one of: ids, project, session, kind")

        with self.db:
            cursor = self.db.execute("DELETE FROM chunks WHERE " + " AND ".join(clauses), params)
        if cursor.rowcount:
            self._load()
        return cursor.rowcount

    @staticmethod
    def similarity(vector_a, vector_b) -> float:
        """Cosine similarity between two vectors. Zero when either has no direction."""
        a = np.asarray(vector_a, dtype=np.float64)
        b = np.asarray(vector_b, dtype=np.float64)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norms) if norms else 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def close(self) -> None:
        self.db.close()
