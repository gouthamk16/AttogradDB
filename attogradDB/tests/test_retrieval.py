import numpy as np
import pytest

from attogradDB.attodb import VectorStore

DOCS = {
    "vec1": "The quick brown fox",
    "vec2": "The cat sat on the mat",
    "vec3": "The rabbit hole is very deep",
}


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_top_n_larger_than_store(stub_embedding, indexing):
    """get_similar defaults to top_n=5; a 3-document store must not raise."""
    store = VectorStore(indexing=indexing)
    for vector_id, text in DOCS.items():
        store.add_text(vector_id, text)

    assert len(store.get_similar("The quick brown fox")) == 3


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_query_on_empty_store(stub_embedding, indexing):
    assert VectorStore(indexing=indexing).get_similar("anything") == []


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_duplicate_texts_keep_distinct_ids(stub_embedding, indexing):
    """Identical text yields identical embeddings; both ids must stay reachable."""
    store = VectorStore(indexing=indexing)
    store.add_text("vec1", "same text")
    store.add_text("vec2", "same text")

    returned = {vector_id for vector_id, *_ in store.get_similar("same text", top_n=2)}
    assert returned == {"vec1", "vec2"}


def test_hnsw_grows_past_initial_capacity(stub_embedding):
    """The index is created with max_elements=1000; inserting more must resize."""
    store = VectorStore(indexing="hnsw")
    for i in range(1100):
        store.add_text(f"vec{i}", f"document number {i}")

    assert store.get_similar("document number 1050", top_n=1)[0][0] == "vec1050"


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_results_are_ordered_by_descending_similarity(stub_embedding, indexing):
    store = VectorStore(indexing=indexing)
    for vector_id, text in DOCS.items():
        store.add_text(vector_id, text)

    scores = [score for _, score, *_ in store.get_similar("The quick brown fox")]
    assert scores == sorted(scores, reverse=True)


def test_similarity_of_identical_vectors_is_one():
    assert VectorStore.similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_similarity_with_zero_vector_does_not_produce_nan():
    """A zero vector has no direction; returning nan poisons every downstream sort."""
    assert not np.isnan(VectorStore.similarity([0.0, 0.0], [1.0, 2.0]))


def test_similarity_rejects_unknown_method():
    with pytest.raises(ValueError):
        VectorStore.similarity([1.0], [1.0], method="euclidean")
