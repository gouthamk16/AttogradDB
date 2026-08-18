import hashlib

import numpy as np
import pytest

from attogradDB import attodb
from attogradDB.embedding import NATIVE_DIM


class StubEmbedding:
    """Deterministic stand-in for QwenEmbedding: same text always maps to the same vector.

    Keeps the store suite fast and reproducible -- the real model is a 600MB download
    and ~50ms per call, neither of which this logic needs.
    """

    def _vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).random(NATIVE_DIM).astype(np.float32)

    def embed_document(self, text):
        return self._vector(text)

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def stub_embedding(monkeypatch):
    monkeypatch.setattr(attodb, "QwenEmbedding", StubEmbedding)


@pytest.fixture
def spy_embedding(monkeypatch):
    """Patches the embedder and returns a log of (kind, text) calls."""
    calls = []

    class Spy(StubEmbedding):
        def embed_document(self, text):
            calls.append(("document", text))
            return super().embed_document(text)

        def embed_query(self, text):
            calls.append(("query", text))
            return super().embed_query(text)

    monkeypatch.setattr(attodb, "QwenEmbedding", Spy)
    return calls
