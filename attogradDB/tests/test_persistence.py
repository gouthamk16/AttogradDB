import json

import pytest

from attogradDB.attodb import VectorStore

DOCS = {
    "vec1": "The quick brown fox",
    "vec2": "The cat sat on the mat",
    "vec3": "The rabbit hole is very deep",
}


def _populate(store):
    for vector_id, text in DOCS.items():
        store.add_text(vector_id, text)
    return store


def test_constructs_with_no_arguments(stub_embedding):
    assert VectorStore().indexing == "hnsw"


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_add_text_without_save_path(stub_embedding, indexing):
    """add_text must not depend on an attribute only set when save_path is given."""
    store = _populate(VectorStore(indexing=indexing))
    assert store.get_vector("vec1", decode_results=True) == DOCS["vec1"]


@pytest.mark.parametrize("indexing", ["hnsw", "brute-force"])
def test_save_load_round_trip(stub_embedding, tmp_path, indexing):
    path = tmp_path / "index.json"
    _populate(VectorStore(indexing=indexing, save_path=str(path)))

    assert path.exists(), "save_path was set but no index file was written"

    reloaded = VectorStore(indexing=indexing, load_path=str(path))
    results = reloaded.get_similar(DOCS["vec2"], top_n=1)

    assert results[0][0] == "vec2"
    assert results[0][2] == DOCS["vec2"], "decoded text did not survive the reload"


def test_saved_index_is_valid_json(stub_embedding, tmp_path):
    path = tmp_path / "index.json"
    _populate(VectorStore(indexing="brute-force", save_path=str(path)))

    payload = json.loads(path.read_text())
    assert set(payload["vectors"]) == set(DOCS)
    assert len(payload["vectors"]["vec1"]) == 768


def test_explicit_save_and_load_paths(stub_embedding, tmp_path):
    path = tmp_path / "explicit.json"
    store = _populate(VectorStore(indexing="brute-force"))
    store.save_index(str(path))

    reloaded = VectorStore(indexing="brute-force")
    reloaded.load_index(str(path))
    assert reloaded.get_vector("vec3", decode_results=True) == DOCS["vec3"]


def test_add_documents_after_reload_does_not_collide(stub_embedding, tmp_path):
    path = tmp_path / "index.json"
    store = VectorStore(indexing="brute-force", save_path=str(path))
    store.add_documents(["alpha", "beta"])

    reloaded = VectorStore(indexing="brute-force", load_path=str(path))
    reloaded.add_documents(["gamma"])

    assert reloaded.get_vector("doc_0", decode_results=True) == "alpha"
    assert reloaded.get_vector("doc_2", decode_results=True) == "gamma"


def test_get_vector_missing_id_returns_none(stub_embedding):
    store = _populate(VectorStore(indexing="brute-force"))
    assert store.get_vector("nope") is None
    assert store.get_vector("nope", decode_results=True) is None


@pytest.mark.parametrize(
    "kwargs", [{"indexing": "bruteforce"}, {"embedding_model": "gpt"}]
)
def test_invalid_configuration_fails_at_construction(stub_embedding, kwargs):
    """A typo must raise where it is made, not as an AttributeError deep in add_text."""
    with pytest.raises(ValueError):
        VectorStore(**kwargs)
