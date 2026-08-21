import sqlite3

import pytest

from attogradDB.memory import init_memory, recall_decisions, remember_decision


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:")
    init_memory(connection)
    yield connection
    connection.close()


def test_remember_and_recall_active_decisions(db):
    created = remember_decision(
        db,
        project="attograd",
        claim="Search stays exhaustive.",
        rationale="Filtered HNSW was slower.",
        scope="retrieval",
        evidence="CLAUDE.md measured decisions",
    )

    assert recall_decisions(db, "attograd") == [created]
    assert created.status == "active"
    assert created.superseded_by is None


def test_recall_is_project_scoped(db):
    remember_decision(db, "one", "Use SQLite.", "Local durability.")
    remember_decision(db, "two", "Use Postgres.", "Shared service.")

    assert [decision.claim for decision in recall_decisions(db, "one")] == ["Use SQLite."]


def test_recall_order_is_deterministic(db):
    first = remember_decision(db, "p", "First.", "Reason one.")
    second = remember_decision(db, "p", "Second.", "Reason two.")

    assert [decision.id for decision in recall_decisions(db, "p")] == [first.id, second.id]


def test_superseding_replaces_active_truth_and_keeps_history(db):
    old = remember_decision(db, "p", "Use Redis.", "Initial choice.")
    new = remember_decision(
        db,
        "p",
        "Use SQLite.",
        "Durability is required.",
        supersedes=old.id,
    )

    assert recall_decisions(db, "p") == [new]
    status = db.execute(
        "SELECT status, superseded_by FROM decisions WHERE id = ?", (old.id,)
    ).fetchone()
    assert status == ("superseded", new.id)


def test_missing_supersession_target_does_not_write(db):
    with pytest.raises(ValueError):
        remember_decision(db, "p", "Replacement.", "New reason.", supersedes=999)

    assert recall_decisions(db, "p") == []


def test_cross_project_supersession_does_not_write(db):
    other = remember_decision(db, "other", "Other decision.", "Other reason.")

    with pytest.raises(ValueError, match="another project"):
        remember_decision(db, "p", "Replacement.", "New reason.", supersedes=other.id)

    assert recall_decisions(db, "p") == []


def test_already_superseded_decision_cannot_be_replaced_again(db):
    old = remember_decision(db, "p", "Old.", "Old reason.")
    remember_decision(db, "p", "Current.", "Current reason.", supersedes=old.id)

    with pytest.raises(ValueError, match="active"):
        remember_decision(db, "p", "Another.", "Another reason.", supersedes=old.id)

    assert [decision.claim for decision in recall_decisions(db, "p")] == ["Current."]


@pytest.mark.parametrize("field", ["project", "claim", "rationale"])
def test_required_text_fields_cannot_be_empty(db, field):
    values = {"project": "p", "claim": "Decision.", "rationale": "Reason."}
    values[field] = " "

    with pytest.raises(ValueError, match=field):
        remember_decision(db, **values)


def test_decisions_survive_reopen(tmp_path):
    path = tmp_path / "memory.db"
    first = sqlite3.connect(path)
    init_memory(first)
    remember_decision(first, "p", "Persist this.", "Sessions end.")
    first.close()

    reopened = sqlite3.connect(path)
    init_memory(reopened)
    assert [decision.claim for decision in recall_decisions(reopened, "p")] == ["Persist this."]
    reopened.close()
