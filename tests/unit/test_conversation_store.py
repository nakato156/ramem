import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from ramem.conversation.store import ConversationStore
from ramem.memory.chunking import ConversationChunker


def test_session_messages_chunks_and_delete_are_durable(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "ramem.sqlite3")
    session = store.create_session("Proyecto memoria")
    user = store.append_message(session.session_id, "user", "Mi color favorito es el verde.")
    assistant = store.append_message(
        session.session_id,
        "assistant",
        "Recordaré que tu color favorito es el verde.",
    )
    chunks = ConversationChunker(max_tokens=64).build_exchange_chunks((user, assistant))

    assert len(chunks) == 1
    assert store.add_chunk(chunks[0])
    assert not store.add_chunk(chunks[0])
    assert store.message_count(session.session_id) == 2
    assert store.stats().pending_jobs == 1

    reopened = ConversationStore(tmp_path / "ramem.sqlite3")
    assert reopened.get_session(session.session_id) == store.get_session(session.session_id)
    assert reopened.messages(session.session_id) == (user, assistant)
    assert reopened.get_chunk(chunks[0].chunk_id) == chunks[0]

    chunk_ids = reopened.delete_session(session.session_id)
    assert chunk_ids == (str(chunks[0].chunk_id),)
    assert reopened.get_session(session.session_id) is None
    assert reopened.messages(session.session_id) == ()
    assert reopened.get_chunk(chunks[0].chunk_id) is None


def test_messages_are_append_only_and_sequence_is_stable(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "ramem.sqlite3")
    session = store.create_session()
    first = store.append_message(session.session_id, "user", "Uno")
    second = store.append_message(session.session_id, "assistant", "Dos")

    assert first.sequence == 1
    assert second.sequence == 2
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as connection:
        connection.execute(
            """UPDATE messages SET sequence = 1 WHERE message_id = ?""",
            (str(second.message_id),),
        )


def test_message_content_is_immutable_and_fts_is_canonical(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "ramem.sqlite3")
    session = store.create_session()
    user = store.append_message(session.session_id, "user", "Mi destino es Cusco")
    assistant = store.append_message(session.session_id, "assistant", "Tu destino es Cusco")
    chunk = ConversationChunker(max_tokens=64).build_exchange_chunks((user, assistant))[0]
    store.add_chunk(chunk)

    assert store.lexical_search("destino Cusco", limit=10)[0]["chunk_id"] == str(chunk.chunk_id)
    with (
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
        store.transaction() as connection,
    ):
        connection.execute(
            "UPDATE messages SET content = 'alterado' WHERE message_id = ?",
            (str(user.message_id),),
        )


def test_running_index_job_is_recovered_when_store_reopens(tmp_path: Path) -> None:
    path = tmp_path / "ramem.sqlite3"
    store = ConversationStore(path)
    session = store.create_session()
    user = store.append_message(session.session_id, "user", "Uno")
    assistant = store.append_message(session.session_id, "assistant", "Dos")
    chunk = ConversationChunker(max_tokens=64).build_exchange_chunks((user, assistant))[0]
    store.add_chunk(chunk)
    job = store.pending_jobs()[0]
    store.mark_job_running(str(job["job_id"]))

    recovered = ConversationStore(path)

    assert recovered.pending_jobs()[0]["status"] == "pending"


def test_oversized_message_is_split_but_normal_messages_keep_boundaries(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "ramem.sqlite3")
    session = store.create_session()
    user = store.append_message(session.session_id, "user", "A" * 300)
    assistant = store.append_message(session.session_id, "assistant", "Respuesta corta")

    chunks = ConversationChunker(max_tokens=32, chars_per_token=2).build_exchange_chunks(
        (user, assistant)
    )

    user_chunks = [chunk for chunk in chunks if chunk.message_ids == (user.message_id,)]
    assert len(user_chunks) > 1
    assert user_chunks[0].start_offset == 0
    assert user_chunks[-1].end_offset == len(user.content)
    assert any(chunk.message_ids == (assistant.message_id,) for chunk in chunks)


def test_schema_migration_records_all_versions_and_enables_fts(tmp_path: Path) -> None:
    path = tmp_path / "ramem.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            ("2025-01-01T00:00:00+00:00",),
        )

    store = ConversationStore(path)

    with closing(store.connect()) as connection:
        versions = tuple(
            row[0]
            for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        )
        fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_chunks_fts'"
        ).fetchone()
    assert versions == (1, 2)
    assert fts is not None


def test_newer_database_schema_is_rejected_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "ramem.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (999, ?)",
            ("2025-01-01T00:00:00+00:00",),
        )

    with pytest.raises(RuntimeError, match="newer than RAMEM"):
        ConversationStore(path)
