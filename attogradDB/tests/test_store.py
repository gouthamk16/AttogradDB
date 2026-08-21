import sqlite3

import numpy as np
import pytest

from attogradDB.attodb import VectorStore
from attogradDB.embedding import NATIVE_DIM

DOCS = ["the quick brown fox", "the cat sat on the mat", "the rabbit hole is deep"]


@pytest.fixture
def store(stub_embedding):
    return VectorStore()


def test_add_returns_generated_ids(store):
    ids = store.add(DOCS)
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_add_accepts_a_bare_string(store):
    assert len(store.add("just one")) == 1


def test_add_empty_list_is_a_noop(store):
    assert store.add([]) == []
    assert len(store) == 0


def test_search_returns_the_closest_document(store):
    store.add(DOCS)
    top = store.search("the cat sat on the mat", top_n=1)
    assert top[0][2] == "the cat sat on the mat"


def test_search_on_empty_store(store):
    assert store.search("anything") == []


def test_search_top_n_larger_than_store(store):
    store.add(DOCS)
    assert len(store.search("fox", top_n=99)) == 3


def test_search_top_n_zero_returns_empty(store):
    store.add(DOCS)
    assert store.search("fox", top_n=0) == []


def test_results_are_ordered_by_descending_score(store):
    store.add(DOCS)
    scores = [score for _, score, _ in store.search("fox")]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_texts_get_distinct_ids(store):
    ids = store.add(["same", "same"])
    assert ids[0] != ids[1]
    assert len(store.search("same", top_n=2)) == 2


# --- scoping: the thing the old store could not do at all ---


def test_search_is_scoped_by_project(store):
    store.add(["alpha document"], project="one")
    store.add(["alpha document"], project="two")

    results = store.search("alpha document", project="one")
    assert len(results) == 1


def test_search_is_scoped_by_session(store):
    store.add(DOCS, session="a")
    store.add(["unrelated"], session="b")

    returned = {text for _, _, text in store.search("the", top_n=99, session="a")}
    assert returned == set(DOCS)


def test_scopes_combine(store):
    store.add(["hit"], project="p", session="s")
    store.add(["miss"], project="p", session="other")

    results = store.search("hit", project="p", session="s")
    assert [text for _, _, text in results] == ["hit"]


def test_unknown_scope_returns_nothing(store):
    store.add(DOCS, project="real")
    assert store.search("fox", project="nope") == []


# --- deletion: forgetting is a first-class operation for agent memory ---


def test_delete_by_id(store):
    ids = store.add(DOCS)
    assert store.delete(ids=[ids[0]]) == 1
    assert len(store) == 2
    assert all(doc_id != ids[0] for doc_id, _, _ in store.search("fox", top_n=99))


def test_delete_by_session(store):
    store.add(DOCS, session="old")
    store.add(["keep me"], session="new")

    assert store.delete(session="old") == 3
    assert [text for _, _, text in store.search("keep me")] == ["keep me"]


def test_delete_non_matching_filter_removes_nothing(store):
    store.add(DOCS)
    assert store.delete(session="never-existed") == 0
    assert len(store) == 3


def test_delete_requires_a_filter(store):
    """Refuse to wipe the whole store through an empty filter."""
    store.add(DOCS)
    with pytest.raises(ValueError):
        store.delete()


def test_search_after_delete_does_not_return_stale_vectors(store):
    ids = store.add(DOCS)
    store.delete(ids=ids[:2])
    results = store.search("the", top_n=99)
    assert len(results) == 1
    assert results[0][2] == DOCS[2]


# --- persistence ---


def test_data_survives_reopen(stub_embedding, tmp_path):
    path = str(tmp_path / "memory.db")
    original = VectorStore(path=path)
    original.add(DOCS, project="p")
    original.close()

    reopened = VectorStore(path=path)
    assert len(reopened) == 3
    assert reopened.search("the cat sat on the mat", top_n=1, project="p")[0][2] == DOCS[1]


def test_deletes_survive_reopen(stub_embedding, tmp_path):
    path = str(tmp_path / "memory.db")
    original = VectorStore(path=path)
    ids = original.add(DOCS)
    original.delete(ids=[ids[0]])
    original.close()

    assert len(VectorStore(path=path)) == 2


def test_broad_filter_uses_the_mask_path_and_stays_correct(store):
    """Filters above SUBSET_SCAN_THRESHOLD take a different branch in search()."""
    wanted = [f"wanted document {i}" for i in range(18)]
    store.add(wanted, project="big")
    store.add(["excluded one", "excluded two"], project="small")

    results = store.search("wanted document 7", top_n=99, project="big")
    assert len(results) == len(wanted)
    assert all(text.startswith("wanted") for _, _, text in results)
    assert results[0][2] == "wanted document 7"


def test_broad_and_narrow_filters_agree(store):
    store.add([f"doc {i}" for i in range(30)], project="wide")
    store.add(["needle in here"], project="narrow")

    assert store.search("needle in here", top_n=1, project="narrow")[0][2] == "needle in here"
    assert len(store.search("doc 3", top_n=99, project="wide")) == 30


# --- dimensions and model identity ---


@pytest.mark.parametrize("dim", [0, -1, 99999])
def test_invalid_dim_rejected(stub_embedding, dim):
    with pytest.raises(ValueError):
        VectorStore(dim=dim)


def test_index_is_truncated_but_storage_is_not(stub_embedding, tmp_path):
    """Full vectors go to disk so dim can change later without re-embedding."""
    path = str(tmp_path / "m.db")
    store = VectorStore(path=path, dim=128)
    store.add(DOCS)

    assert store._vectors.shape[1] == 128
    stored = store.db.execute("SELECT vector FROM chunks LIMIT 1").fetchone()[0]
    assert len(stored) // 4 == NATIVE_DIM
    store.close()

    # Same data, different search dimension, no re-embedding needed.
    wider = VectorStore(path=path, dim=512)
    assert wider._vectors.shape[1] == 512
    assert wider.search(DOCS[1], top_n=1)[0][2] == DOCS[1]


def test_truncated_index_vectors_stay_normalised(stub_embedding):
    store = VectorStore(dim=64)
    store.add(DOCS)
    assert np.allclose(np.linalg.norm(store._vectors, axis=1), 1.0, atol=1e-6)


def test_reopening_with_a_different_model_is_refused(stub_embedding, tmp_path):
    path = str(tmp_path / "m.db")
    VectorStore(path=path).close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE meta SET value = 'some-other-model' WHERE key = 'model'")
    with pytest.raises(ValueError, match="not comparable"):
        VectorStore(path=path)


def test_queries_and_documents_take_different_paths(spy_embedding):
    """The model is trained asymmetrically; the store must not embed both the same way."""
    calls = spy_embedding
    store = VectorStore()
    store.add(["a document"])
    store.search("a question")

    assert [kind for kind, _ in calls] == ["document", "query"]


# --- searching across several projects or sessions at once ---


def test_search_across_several_projects(store):
    store.add(["alpha note"], project="a")
    store.add(["beta note"], project="b")
    store.add(["gamma note"], project="c")

    found = {text for _, _, text in store.search("note", top_n=99, project=["a", "b"])}
    assert found == {"alpha note", "beta note"}


def test_search_across_several_sessions(store):
    for name in ("mon", "tue", "wed"):
        store.add([f"{name} work"], session=name)

    found = {text for _, _, text in store.search("work", top_n=99, session=["mon", "wed"])}
    assert found == {"mon work", "wed work"}


def test_list_scope_combines_with_single_scope(store):
    store.add(["hit one"], project="a", kind="note")
    store.add(["hit two"], project="b", kind="note")
    store.add(["miss"], project="a", kind="code")

    found = {text for _, _, text in store.search("hit", top_n=99, project=["a", "b"], kind="note")}
    assert found == {"hit one", "hit two"}


def test_empty_list_scope_matches_nothing(store):
    store.add(DOCS, project="a")
    assert store.search("fox", project=[]) == []


def test_single_element_list_matches_the_plain_string(store):
    store.add(["alpha note"], project="a")
    store.add(["beta note"], project="b")
    assert store.search("note", top_n=99, project=["a"]) == store.search(
        "note", top_n=99, project="a"
    )


def test_delete_across_several_sessions(store):
    for name in ("mon", "tue", "wed"):
        store.add([f"{name} work"], session=name)

    assert store.delete(session=["mon", "tue"]) == 2
    assert [text for _, _, text in store.search("work", top_n=99)] == ["wed work"]
