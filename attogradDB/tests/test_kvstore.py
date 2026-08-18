import json

import pytest

from attogradDB.kvstore import keyValueStore

DOCS = [
    {"id": 101, "name": "Alice", "contact": "alice@example.com"},
    {"id": 102, "name": "Bob", "contact": None},
]


@pytest.fixture
def store(tmp_path):
    return keyValueStore(json_path=str(tmp_path / "data.json"))


def test_add_and_search(store):
    store.add(DOCS)
    assert [doc["id"] for doc in store.search("name", "Alice")] == [101]


def test_add_assigns_unique_ids(store):
    store.add(DOCS)
    found = store.search("name", "Alice") + store.search("name", "Bob")
    assert len({doc["_id"] for doc in found}) == 2


def test_add_list_with_explicit_doc_id(store):
    store.add(DOCS, doc_id="emp")
    assert {store[0]["_id"], store[1]["_id"]} == {"emp_0", "emp_1"}


def test_add_rejects_non_dict_documents(store):
    with pytest.raises(TypeError):
        store.add(["just a string"])


def test_create_collection_creates_missing_master(store):
    """create_collection must not assume the master directory already exists."""
    store.create_collection("employees", master_collection="users")
    store.use_collection("employees", master_collection="users")
    store.add(DOCS[0])
    assert store.search("name", "Alice")


def test_use_unknown_collection_raises(store):
    with pytest.raises(FileNotFoundError):
        store.use_collection("does-not-exist")


def test_add_json_reports_the_input_file(store, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.json"):
        store.add_json(str(tmp_path / "missing.json"))


def test_add_json_does_not_blame_input_for_collection_errors(store, tmp_path):
    """A readable input file must never be reported as missing."""
    source = tmp_path / "good.json"
    source.write_text(json.dumps(DOCS))
    (tmp_path / "data" / "default" / "default.json").unlink()

    with pytest.raises(FileNotFoundError) as err:
        store.add_json(str(source))
    assert "good.json" not in str(err.value)


def test_add_json_rejects_malformed_json(store, tmp_path):
    source = tmp_path / "bad.json"
    source.write_text("{not json")
    with pytest.raises(ValueError):
        store.add_json(str(source))


def test_to_vector_excludes_the_internal_id(store, stub_embedding):
    """_id is a storage detail; embedding it feeds random uuid characters to the model."""
    store.add(DOCS)
    vector_store = store.to_vector(indexing="brute-force")

    embedded = list(vector_store.text.values())
    assert len(embedded) == 2
    assert all("_id" not in text for text in embedded)
    assert any("Alice" in text for text in embedded)


def test_toVector_alias_still_works_but_warns(store, stub_embedding):
    store.add(DOCS)
    with pytest.warns(DeprecationWarning):
        assert len(store.toVector(indexing="brute-force").text) == 2
