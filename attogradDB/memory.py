import sqlite3
import time
from dataclasses import dataclass

DECISION_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY,
    project       TEXT NOT NULL,
    scope         TEXT,
    claim         TEXT NOT NULL,
    rationale     TEXT NOT NULL,
    evidence      TEXT,
    status        TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    superseded_by INTEGER REFERENCES decisions(id),
    created_at    REAL NOT NULL,
    CHECK (
        (status = 'active' AND superseded_by IS NULL)
        OR (status = 'superseded' AND superseded_by IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_decisions_project_status
    ON decisions(project, status, id);
"""

DECISION_COLUMNS = (
    "id, project, scope, claim, rationale, evidence, status, superseded_by, created_at"
)


@dataclass(frozen=True)
class Decision:
    id: int
    project: str
    scope: str | None
    claim: str
    rationale: str
    evidence: str | None
    status: str
    superseded_by: int | None
    created_at: float


def init_memory(db: sqlite3.Connection) -> None:
    """Create the structured decision-memory schema on an AttogradDB connection."""
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(DECISION_SCHEMA)


def _require_text(name: str, value: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _get_decision(db: sqlite3.Connection, decision_id: int) -> Decision:
    row = db.execute(
        f"SELECT {DECISION_COLUMNS} FROM decisions WHERE id = ?", (decision_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"decision {decision_id} does not exist")
    return Decision(*row)


def _active_prior(
    db: sqlite3.Connection,
    project: str,
    supersedes: int | None,
) -> Decision | None:
    if supersedes is None:
        return None
    prior = _get_decision(db, supersedes)
    if prior.project != project:
        raise ValueError(f"decision {supersedes} belongs to another project")
    if prior.status != "active":
        raise ValueError(f"decision {supersedes} is not active")
    return prior


def remember_decision(
    db: sqlite3.Connection,
    project: str,
    claim: str,
    rationale: str,
    scope: str | None = None,
    evidence: str | None = None,
    supersedes: int | None = None,
) -> Decision:
    """Write a decision, optionally replacing one active decision atomically."""
    project = _require_text("project", project)
    claim = _require_text("claim", claim)
    rationale = _require_text("rationale", rationale)

    with db:
        prior = _active_prior(db, project, supersedes)
        cursor = db.execute(
            "INSERT INTO decisions "
            "(project, scope, claim, rationale, evidence, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?)",
            (project, scope, claim, rationale, evidence, time.time()),
        )
        decision_id = cursor.lastrowid
        if prior is not None:
            db.execute(
                "UPDATE decisions SET status = 'superseded', superseded_by = ? WHERE id = ?",
                (decision_id, prior.id),
            )
    return _get_decision(db, decision_id)


def recall_decisions(db: sqlite3.Connection, project: str) -> list[Decision]:
    """Return the project's active decisions in insertion order."""
    project = _require_text("project", project)
    rows = db.execute(
        f"SELECT {DECISION_COLUMNS} FROM decisions "
        "WHERE project = ? AND status = 'active' ORDER BY id",
        (project,),
    )
    return [Decision(*row) for row in rows]
