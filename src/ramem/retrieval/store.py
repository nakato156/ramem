from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ramem.domain.models import Document
from ramem.retrieval.vectors import hashing_vector


class SQLiteDocumentStore:
    def __init__(self, path: Path, embedding_dimension: int = 256) -> None:
        self.path = path
        self.embedding_dimension = embedding_dimension
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source_uri TEXT,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    document_id UNINDEXED,
                    title,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )

    def ingest(self, documents: list[Document]) -> int:
        with self.connect() as connection:
            for document in documents:
                vector_json = json.dumps(
                    hashing_vector(document.text, self.embedding_dimension), separators=(",", ":")
                )
                connection.execute(
                    "DELETE FROM documents_fts WHERE document_id = ?", (document.document_id,)
                )
                connection.execute(
                    """INSERT OR REPLACE INTO documents
                    (document_id, title, text, source_uri, content_hash, metadata_json, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document.document_id,
                        document.title,
                        document.text,
                        document.source_uri,
                        document.content_hash,
                        json.dumps(document.metadata, ensure_ascii=False),
                        vector_json,
                    ),
                )
                connection.execute(
                    "INSERT INTO documents_fts(document_id, title, text) VALUES (?, ?, ?)",
                    (document.document_id, document.title, document.text),
                )
        return len(documents)

    def count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"]) if row else 0
