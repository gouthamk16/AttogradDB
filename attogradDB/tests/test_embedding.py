"""Checks against the real embedding model. Few, because each one loads it."""

import numpy as np

from attogradDB.embedding import NATIVE_DIM, QwenEmbedding


def test_query_and_document_differ():
    """The model is trained asymmetrically; the prefix must change the vector."""
    embedder = QwenEmbedding()
    assert not np.allclose(embedder.embed_query("redis"), embedder.embed_document("redis"))


def test_embed_documents_matches_one_at_a_time():
    embedder = QwenEmbedding()
    texts = ["first document", "a rather longer second document"]
    batched = embedder.embed_documents(texts)

    assert batched.shape == (2, NATIVE_DIM)
    for vector, text in zip(batched, texts):
        assert np.allclose(vector, embedder.embed_document(text))


def test_embed_documents_on_empty_input():
    assert QwenEmbedding().embed_documents([]).shape == (0, NATIVE_DIM)


def test_embeddings_are_native_dimension():
    assert QwenEmbedding().embed_document("hello").shape == (NATIVE_DIM,)
