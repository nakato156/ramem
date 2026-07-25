from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from ramem.conversation.models import (
    ConversationMessage,
    ConversationSession,
    MemoryChunk,
    SessionStatus,
    StorageStats,
)

SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class ConversationStore:
    """Canonical, transactional store for conversations and index work."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )"""
            )
            current_row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
            current_version = int(current_row[0])
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current_version} is newer than RAMEM {SCHEMA_VERSION}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'archived'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS memory_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    message_ids_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('pending', 'running', 'completed', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(operation, entity_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_sequence
                    ON messages(session_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_chunks_session_created
                    ON memory_chunks(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON index_jobs(status, created_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS messages_are_immutable
                BEFORE UPDATE ON messages
                BEGIN
                    SELECT RAISE(ABORT, 'messages are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS memory_chunks_are_immutable
                BEFORE UPDATE ON memory_chunks
                BEGIN
                    SELECT RAISE(ABORT, 'memory chunks are immutable');
                END;
                """
            )
            now = _iso(_utc_now())
            connection.executemany(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                ((version, now) for version in range(1, SCHEMA_VERSION + 1)),
            )
            connection.execute(
                """INSERT INTO memory_chunks_fts(chunk_id, text)
                SELECT c.chunk_id, c.text FROM memory_chunks AS c
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_chunks_fts AS f WHERE f.chunk_id = c.chunk_id
                )"""
            )
            connection.execute(
                """UPDATE index_jobs SET status = 'pending', error = ?, updated_at = ?
                WHERE status = 'running'""",
                ("recovered after interrupted indexing", now),
            )
            connection.commit()

    def create_session(self, title: str | None = None) -> ConversationSession:
        now = _utc_now()
        session = ConversationSession(title=(title or f"Conversación {now:%Y-%m-%d %H:%M}").strip())
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO sessions(session_id, title, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    str(session.session_id),
                    session.title,
                    _iso(session.created_at),
                    _iso(session.updated_at),
                    session.status.value,
                ),
            )
        return session

    def get_session(self, session_id: UUID | str) -> ConversationSession | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
        return self._session_from_row(row) if row else None

    def latest_session(self) -> ConversationSession | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, created_at DESC LIMIT 1"
            ).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, limit: int = 100) -> tuple[ConversationSession, ...]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(self._session_from_row(row) for row in rows)

    def rename_session(self, session_id: UUID | str, title: str) -> ConversationSession:
        normalized = title.strip()
        if not normalized:
            raise ValueError("session title cannot be empty")
        now = _utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (normalized, _iso(now), str(session_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown session {session_id}")
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError("renamed session disappeared")
        return session

    def delete_session(self, session_id: UUID | str) -> tuple[str, ...]:
        with self.transaction() as connection:
            chunk_rows = connection.execute(
                "SELECT chunk_id FROM memory_chunks WHERE session_id = ?", (str(session_id),)
            ).fetchall()
            chunk_ids = tuple(str(row["chunk_id"]) for row in chunk_rows)
            for chunk_id in chunk_ids:
                connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id = ?", (chunk_id,))
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_id = ?", (str(session_id),)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown session {session_id}")
            for chunk_id in chunk_ids:
                self._upsert_job(connection, "delete", chunk_id)
        return chunk_ids

    def delete_all(self) -> tuple[str, ...]:
        with self.transaction() as connection:
            rows = connection.execute("SELECT chunk_id FROM memory_chunks").fetchall()
            chunk_ids = tuple(str(row["chunk_id"]) for row in rows)
            connection.execute("DELETE FROM memory_chunks_fts")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM index_jobs")
            for chunk_id in chunk_ids:
                self._upsert_job(connection, "delete", chunk_id)
        return chunk_ids

    def append_message(
        self,
        session_id: UUID | str,
        role: str,
        content: str,
    ) -> ConversationMessage:
        normalized = content.strip()
        if not normalized:
            raise ValueError("message content cannot be empty")
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role {role!r}")
        now = _utc_now()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if not exists:
                raise KeyError(f"unknown session {session_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM messages WHERE session_id = ?",
                (str(session_id),),
            ).fetchone()
            sequence = int(row["next"])
            message = ConversationMessage(
                session_id=UUID(str(session_id)),
                sequence=sequence,
                role=cast(Literal["user", "assistant"], role),
                content=normalized,
                created_at=now,
                content_hash=digest,
            )
            connection.execute(
                """INSERT INTO messages
                (message_id, session_id, sequence, role, content, created_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(message.message_id),
                    str(message.session_id),
                    message.sequence,
                    message.role,
                    message.content,
                    _iso(message.created_at),
                    message.content_hash,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_iso(now), str(session_id)),
            )
        return message

    def messages(
        self,
        session_id: UUID | str,
        *,
        limit: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence"
        params: tuple[object, ...] = (str(session_id),)
        if limit is not None:
            sql = """SELECT * FROM (
                SELECT * FROM messages WHERE session_id = ? ORDER BY sequence DESC LIMIT ?
            ) ORDER BY sequence"""
            params = (str(session_id), limit)
        with closing(self.connect()) as connection:
            rows = connection.execute(sql, params).fetchall()
        return tuple(self._message_from_row(row) for row in rows)

    def message_count(self, session_id: UUID | str) -> int:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?", (str(session_id),)
            ).fetchone()
        return int(row["count"])

    def add_chunk(self, chunk: MemoryChunk) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_chunks
                (chunk_id, session_id, message_ids_json, text, start_offset, end_offset,
                 token_count, created_at, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(chunk.chunk_id),
                    str(chunk.session_id),
                    json.dumps([str(item) for item in chunk.message_ids]),
                    chunk.text,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.token_count,
                    _iso(chunk.created_at),
                    chunk.content_hash,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                connection.execute(
                    "INSERT INTO memory_chunks_fts(chunk_id, text) VALUES (?, ?)",
                    (str(chunk.chunk_id), chunk.text),
                )
                self._upsert_job(connection, "upsert", str(chunk.chunk_id))
        return inserted

    def get_chunk(self, chunk_id: UUID | str) -> MemoryChunk | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM memory_chunks WHERE chunk_id = ?", (str(chunk_id),)
            ).fetchone()
        return self._chunk_from_row(row) if row else None

    def all_chunks(self) -> tuple[MemoryChunk, ...]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM memory_chunks ORDER BY created_at, chunk_id"
            ).fetchall()
        return tuple(self._chunk_from_row(row) for row in rows)

    def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_chunk_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        """Search canonical conversational text with SQLite FTS5."""
        terms = re.findall(r"[\wáéíóúüñ]+", query.casefold(), flags=re.UNICODE)
        if not terms:
            return []
        quoted_terms = ['"' + term.replace('"', '""') + '"' for term in terms]
        expression = " OR ".join(quoted_terms)
        excluded = excluded_chunk_ids or set()
        fetch_limit = max(limit * 4, limit)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT c.*, -bm25(memory_chunks_fts) AS _score
                FROM memory_chunks_fts
                JOIN memory_chunks AS c USING(chunk_id)
                WHERE memory_chunks_fts MATCH ?
                ORDER BY bm25(memory_chunks_fts), c.created_at DESC
                LIMIT ?""",
                (expression, fetch_limit),
            ).fetchall()
        return [dict(row) for row in rows if str(row["chunk_id"]) not in excluded][:limit]

    def pending_jobs(self, limit: int = 100) -> tuple[sqlite3.Row, ...]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM index_jobs
                WHERE status = 'pending' OR (status = 'failed' AND attempts < 5)
                ORDER BY created_at LIMIT ?""",
                (limit,),
            ).fetchall()
        return tuple(rows)

    def mark_job_running(self, job_id: str) -> None:
        self._set_job(job_id, "running", increment_attempt=True)

    def mark_job_completed(self, job_id: str) -> None:
        self._set_job(job_id, "completed", error=None)

    def mark_job_failed(self, job_id: str, error: str) -> None:
        self._set_job(job_id, "failed", error=error[:1000])

    def reset_index_jobs(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM index_jobs")
            for row in connection.execute("SELECT chunk_id FROM memory_chunks"):
                self._upsert_job(connection, "upsert", str(row["chunk_id"]))

    def stats(self) -> StorageStats:
        with closing(self.connect()) as connection:
            values = {
                "sessions": connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
                "messages": connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "chunks": connection.execute("SELECT COUNT(*) FROM memory_chunks").fetchone()[0],
                "pending_jobs": connection.execute(
                    "SELECT COUNT(*) FROM index_jobs WHERE status IN ('pending', 'running')"
                ).fetchone()[0],
                "failed_jobs": connection.execute(
                    "SELECT COUNT(*) FROM index_jobs WHERE status = 'failed'"
                ).fetchone()[0],
            }
        return StorageStats(
            **values,
            database_bytes=sum(
                path.stat().st_size
                for path in (
                    self.path,
                    Path(str(self.path) + "-wal"),
                    Path(str(self.path) + "-shm"),
                )
                if path.exists()
            ),
        )

    def _set_job(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        now = _iso(_utc_now())
        attempts = ", attempts = attempts + 1" if increment_attempt else ""
        with self.transaction() as connection:
            cursor = connection.execute(
                f"UPDATE index_jobs SET status = ?, error = ?, updated_at = ?{attempts} "
                "WHERE job_id = ?",
                (status, error, now, job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown index job {job_id}")

    @staticmethod
    def _upsert_job(connection: sqlite3.Connection, operation: str, entity_id: str) -> None:
        now = _iso(_utc_now())
        connection.execute(
            """INSERT INTO index_jobs
            (job_id, operation, entity_id, status, attempts, error, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?)
            ON CONFLICT(operation, entity_id) DO UPDATE SET
                status = 'pending', error = NULL, updated_at = excluded.updated_at""",
            (str(uuid4()), operation, entity_id, now, now),
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> ConversationSession:
        return ConversationSession(
            session_id=UUID(str(row["session_id"])),
            title=str(row["title"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            status=SessionStatus(str(row["status"])),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            message_id=UUID(str(row["message_id"])),
            session_id=UUID(str(row["session_id"])),
            sequence=int(row["sequence"]),
            role=cast(Literal["user", "assistant"], str(row["role"])),
            content=str(row["content"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            content_hash=str(row["content_hash"]),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> MemoryChunk:
        return MemoryChunk(
            chunk_id=UUID(str(row["chunk_id"])),
            session_id=UUID(str(row["session_id"])),
            message_ids=tuple(UUID(item) for item in json.loads(row["message_ids_json"])),
            text=str(row["text"]),
            start_offset=int(row["start_offset"]),
            end_offset=int(row["end_offset"]),
            token_count=int(row["token_count"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            content_hash=str(row["content_hash"]),
        )
