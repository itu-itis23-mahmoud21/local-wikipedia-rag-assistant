"""SQLite metadata layer for the Local Wikipedia RAG Assistant.

This module stores non-vector metadata for configured entities, ingested
documents, generated chunks, and ingestion run status. Vector data will live in
Chroma in a later step.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from src import config
from src.entities import ENTITY_TYPES


class MetadataDB:
    """Small SQLite wrapper for local RAG metadata."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Create a metadata database wrapper.

        Args:
            db_path: Optional SQLite file path. Defaults to
                ``config.SQLITE_DB_PATH``.
        """

        self.db_path = Path(db_path) if db_path is not None else config.SQLITE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries and foreign keys."""

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def init_schema(self) -> None:
        """Create all required metadata tables if they do not already exist."""

        with closing(self.connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE,
                    entity_type TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'wikipedia',
                    source_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(name, entity_type)
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    source_url TEXT,
                    raw_path TEXT,
                    processed_path TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(entity_id) REFERENCES entities(id)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    entity_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_estimate INTEGER,
                    char_count INTEGER NOT NULL,
                    vector_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id),
                    FOREIGN KEY(entity_id) REFERENCES entities(id),
                    UNIQUE(document_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    total_entities INTEGER NOT NULL DEFAULT 0,
                    successful_entities INTEGER NOT NULL DEFAULT 0,
                    failed_entities INTEGER NOT NULL DEFAULT 0,
                    notes TEXT
                );
                """
            )
            connection.commit()

    def reset(self) -> None:
        """Drop known metadata tables and recreate the schema."""

        with closing(self.connect()) as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS entities;
                DROP TABLE IF EXISTS ingestion_runs;
                """
            )
            connection.commit()

        self.init_schema()

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        source_url: str | None = None,
    ) -> int:
        """Insert or update an entity and return its row id."""

        _validate_name(name)
        _validate_entity_type(entity_type)
        now = _utc_now()

        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO entities (
                    name,
                    entity_type,
                    source_url,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name, entity_type) DO UPDATE SET
                    source_url = excluded.source_url,
                    updated_at = excluded.updated_at;
                """,
                (name.strip(), entity_type, source_url, now, now),
            )
            row = connection.execute(
                """
                SELECT id
                FROM entities
                WHERE name = ? COLLATE NOCASE
                  AND entity_type = ?;
                """,
                (name.strip(), entity_type),
            ).fetchone()
            connection.commit()

        if row is None:
            raise RuntimeError("Failed to upsert entity.")
        return int(row["id"])

    def get_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> dict | None:
        """Find an entity by case-insensitive name and optional type."""

        _validate_name(name)
        with closing(self.connect()) as connection:
            if entity_type is None:
                row = connection.execute(
                    """
                    SELECT *
                    FROM entities
                    WHERE name = ? COLLATE NOCASE
                    ORDER BY id
                    LIMIT 1;
                    """,
                    (name.strip(),),
                ).fetchone()
            else:
                _validate_entity_type(entity_type)
                row = connection.execute(
                    """
                    SELECT *
                    FROM entities
                    WHERE name = ? COLLATE NOCASE
                      AND entity_type = ?
                    LIMIT 1;
                    """,
                    (name.strip(), entity_type),
                ).fetchone()

        return _row_to_dict(row)

    def get_entity_by_id(self, entity_id: int) -> dict | None:
        """Return an entity by id."""

        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM entities
                WHERE id = ?;
                """,
                (entity_id,),
            ).fetchone()

        return _row_to_dict(row)

    def list_entities(self, entity_type: str | None = None) -> list[dict]:
        """List entities, optionally filtered by entity type."""

        with closing(self.connect()) as connection:
            if entity_type is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM entities
                    ORDER BY id;
                    """
                ).fetchall()
            else:
                _validate_entity_type(entity_type)
                rows = connection.execute(
                    """
                    SELECT *
                    FROM entities
                    WHERE entity_type = ?
                    ORDER BY id;
                    """,
                    (entity_type,),
                ).fetchall()

        return _rows_to_dicts(rows)

    def create_document(
        self,
        entity_id: int,
        title: str,
        source_url: str | None,
        raw_path: str | None,
        processed_path: str | None,
        status: str,
        error_message: str | None = None,
    ) -> int:
        """Create a document metadata record and return its row id."""

        _validate_name(title, field_name="title")
        _validate_status(status)
        now = _utc_now()

        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents (
                    entity_id,
                    title,
                    source_url,
                    raw_path,
                    processed_path,
                    status,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    entity_id,
                    title.strip(),
                    source_url,
                    raw_path,
                    processed_path,
                    status.strip(),
                    error_message,
                    now,
                    now,
                ),
            )
            document_id = _lastrowid_to_int(
                cursor.lastrowid,
                "creating document",
            )
            connection.commit()

        return document_id

    def update_document_status(
        self,
        document_id: int,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update a document status and optional error message."""

        _validate_status(status)
        now = _utc_now()

        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?,
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?;
                """,
                (status.strip(), error_message, now, document_id),
            )
            connection.commit()

    def get_document(self, document_id: int) -> dict | None:
        """Return a document by id."""

        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?;
                """,
                (document_id,),
            ).fetchone()

        return _row_to_dict(row)

    def list_documents(self, status: str | None = None) -> list[dict]:
        """List documents, optionally filtered by status."""

        with closing(self.connect()) as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM documents
                    ORDER BY id;
                    """
                ).fetchall()
            else:
                _validate_status(status)
                rows = connection.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE status = ?
                    ORDER BY id;
                    """,
                    (status.strip(),),
                ).fetchall()

        return _rows_to_dicts(rows)

    def add_chunk(
        self,
        document_id: int,
        entity_id: int,
        chunk_index: int,
        text: str,
        vector_id: str | None = None,
    ) -> int:
        """Create a chunk metadata record and return its row id."""

        char_count = len(text)
        token_estimate = max(1, char_count // 4)
        now = _utc_now()

        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    entity_id,
                    chunk_index,
                    text,
                    token_estimate,
                    char_count,
                    vector_id,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    document_id,
                    entity_id,
                    chunk_index,
                    text,
                    token_estimate,
                    char_count,
                    vector_id,
                    now,
                ),
            )
            chunk_id = _lastrowid_to_int(cursor.lastrowid, "adding chunk")
            connection.commit()

        return chunk_id

    def list_chunks(
        self,
        document_id: int | None = None,
        entity_id: int | None = None,
    ) -> list[dict]:
        """List chunks, optionally filtered by document or entity."""

        clauses: list[str] = []
        parameters: list[int] = []

        if document_id is not None:
            clauses.append("document_id = ?")
            parameters.append(document_id)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            parameters.append(entity_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT *
            FROM chunks
            {where_sql}
            ORDER BY document_id, chunk_index;
        """

        with closing(self.connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()

        return _rows_to_dicts(rows)

    def delete_chunks(self, document_id: int | None = None) -> int:
        """Delete chunks globally or for one document and return rows deleted."""

        with closing(self.connect()) as connection:
            if document_id is None:
                cursor = connection.execute("DELETE FROM chunks;")
            else:
                cursor = connection.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = ?;
                    """,
                    (document_id,),
                )
            deleted_count = int(cursor.rowcount)
            connection.commit()

        return deleted_count

    def update_chunk_vector_id(self, chunk_id: int, vector_id: str | None) -> None:
        """Update the stored vector id for one chunk."""

        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE chunks
                SET vector_id = ?
                WHERE id = ?;
                """,
                (vector_id, chunk_id),
            )
            connection.commit()

    def clear_chunk_vector_ids(self, document_id: int | None = None) -> int:
        """Clear stored vector ids globally or for one document."""

        with closing(self.connect()) as connection:
            if document_id is None:
                cursor = connection.execute(
                    """
                    UPDATE chunks
                    SET vector_id = NULL
                    WHERE vector_id IS NOT NULL;
                    """
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE chunks
                    SET vector_id = NULL
                    WHERE document_id = ?
                      AND vector_id IS NOT NULL;
                    """,
                    (document_id,),
                )
            cleared_count = int(cursor.rowcount)
            connection.commit()

        return cleared_count

    def count_chunk_vector_ids(self) -> int:
        """Return the number of chunks that currently have a stored vector id."""

        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM chunks
                WHERE vector_id IS NOT NULL;
                """
            ).fetchone()

        return int(row["count"])

    def start_ingestion_run(
        self,
        total_entities: int,
        notes: str | None = None,
    ) -> int:
        """Create an ingestion run record in running status."""

        if total_entities < 0:
            raise ValueError("total_entities must be non-negative.")

        now = _utc_now()
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO ingestion_runs (
                    started_at,
                    status,
                    total_entities,
                    notes
                )
                VALUES (?, ?, ?, ?);
                """,
                (now, "running", total_entities, notes),
            )
            run_id = _lastrowid_to_int(
                cursor.lastrowid,
                "starting ingestion run",
            )
            connection.commit()

        return run_id

    def finish_ingestion_run(
        self,
        run_id: int,
        status: str,
        successful_entities: int,
        failed_entities: int,
        notes: str | None = None,
    ) -> None:
        """Mark an ingestion run as finished."""

        _validate_status(status)
        if successful_entities < 0:
            raise ValueError("successful_entities must be non-negative.")
        if failed_entities < 0:
            raise ValueError("failed_entities must be non-negative.")

        now = _utc_now()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = ?,
                    status = ?,
                    successful_entities = ?,
                    failed_entities = ?,
                    notes = COALESCE(?, notes)
                WHERE id = ?;
                """,
                (
                    now,
                    status.strip(),
                    successful_entities,
                    failed_entities,
                    notes,
                    run_id,
                ),
            )
            connection.commit()

    def get_ingestion_run(self, run_id: int) -> dict | None:
        """Return an ingestion run by id."""

        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM ingestion_runs
                WHERE id = ?;
                """,
                (run_id,),
            ).fetchone()

        return _row_to_dict(row)

    def get_summary_counts(self) -> dict:
        """Return high-level counts for local metadata tables."""

        with closing(self.connect()) as connection:
            return {
                "entities": _count_rows(connection, "entities"),
                "documents": _count_rows(connection, "documents"),
                "chunks": _count_rows(connection, "chunks"),
                "people": _count_rows(
                    connection,
                    "entities",
                    "entity_type = ?",
                    ("person",),
                ),
                "places": _count_rows(
                    connection,
                    "entities",
                    "entity_type = ?",
                    ("place",),
                ),
                "successful_documents": _count_rows(
                    connection,
                    "documents",
                    "LOWER(status) = ?",
                    ("success",),
                ),
                "failed_documents": _count_rows(
                    connection,
                    "documents",
                    "LOWER(status) = ?",
                    ("failed",),
                ),
            }


def initialize_database(db_path: str | Path | None = None) -> None:
    """Initialize the SQLite schema using the default or provided path."""

    MetadataDB(db_path).init_schema()


def _utc_now() -> str:
    """Return the current UTC time in ISO format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Convert a SQLite row to a plain dictionary."""

    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert non-null SQLite rows to plain dictionaries."""

    return [dict(row) for row in rows]


def _lastrowid_to_int(lastrowid: int | None, operation: str) -> int:
    """Return a SQLite inserted row id or raise a clear runtime error."""

    if lastrowid is None:
        raise RuntimeError(f"Failed to get inserted row id after {operation}.")
    return int(lastrowid)


def _count_rows(
    connection: sqlite3.Connection,
    table_name: str,
    where_clause: str | None = None,
    parameters: tuple = (),
) -> int:
    """Count rows from a known table with an optional WHERE clause."""

    query = f"SELECT COUNT(*) AS count FROM {table_name}"
    if where_clause is not None:
        query = f"{query} WHERE {where_clause}"
    row = connection.execute(query, parameters).fetchone()
    return int(row["count"])


def _validate_entity_type(entity_type: str) -> None:
    """Validate a supported entity type."""

    if entity_type not in ENTITY_TYPES:
        allowed = ", ".join(ENTITY_TYPES)
        raise ValueError(f"entity_type must be one of: {allowed}.")


def _validate_status(status: str) -> None:
    """Validate a non-blank status string."""

    if not status.strip():
        raise ValueError("status must not be blank.")


def _validate_name(name: str, field_name: str = "name") -> None:
    """Validate a non-blank text field."""

    if not name.strip():
        raise ValueError(f"{field_name} must not be blank.")
