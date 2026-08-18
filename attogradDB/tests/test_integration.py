"""End-to-end checks against the real embedding model.

Deliberately few: the first run downloads ~600MB and each embed call costs ~50ms.
Behaviour that does not depend on real embeddings belongs in test_store.py.
"""

import pytest

from attogradDB.attodb import VectorStore

DOCS = [
    "The cat sat on the mat",
    "Postgres is a relational database",
    "A kitten rested on the rug",
]


@pytest.fixture(scope="module")
def store():
    store = VectorStore()
    store.add(DOCS, project="demo")
    yield store
    store.close()


def test_semantically_similar_text_outranks_unrelated(store):
    top = store.search("a small cat lying down", top_n=2)
    returned = {text for _, _, text in top}

    assert "Postgres is a relational database" not in returned
    assert returned <= {DOCS[0], DOCS[2]}


def test_scores_are_valid_cosine_similarities(store):
    for _, score, _ in store.search("database", top_n=3):
        assert -1.0 <= score <= 1.0


def test_exact_match_ranks_first(store):
    doc_id, score, text = store.search(DOCS[1], top_n=1)[0]
    assert text == DOCS[1]
    # Not ~1.0: the query carries an instruction prefix the document does not.
    assert score > 0.5


def test_irrelevant_document_is_clearly_separated(store):
    """The failure this model was adopted to fix: BERT put an unrelated note within
    0.075 of the correct answer for a query like this."""
    ranked = store.search("a small cat lying down", top_n=3)
    by_text = {text: score for _, score, text in ranked}
    assert by_text[DOCS[0]] - by_text["Postgres is a relational database"] > 0.15


def test_scoping_still_applies_with_real_embeddings(store):
    assert store.search("cat", project="nonexistent") == []
    assert len(store.search("cat", project="demo", top_n=99)) == 3
