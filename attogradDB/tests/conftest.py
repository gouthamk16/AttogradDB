import hashlib

import numpy as np
import pytest

from attogradDB import attodb


class StubEmbedding:
    """Deterministic stand-in for BertEmbedding: same text always maps to the same vector.

    Keeps the persistence and retrieval suites fast and reproducible -- loading real
    BERT costs ~30s per test and its output is not needed to exercise this logic.
    """

    def embed(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        return np.random.default_rng(seed).random(768).astype(np.float32)


@pytest.fixture
def stub_embedding(monkeypatch):
    monkeypatch.setattr(attodb, "BertEmbedding", StubEmbedding)
