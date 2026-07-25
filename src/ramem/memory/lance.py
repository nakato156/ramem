from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.rerankers import RRFReranker

from ramem.conversation.models import MemoryChunk
from ramem.memory.embeddings import TextEmbedder

TABLE_NAME = "conversation_memory"


class LanceMemoryIndex:
    """Rebuildable LanceDB retrieval index; SQLite remains canonical."""

    def __init__(self, path: Path, embedder: TextEmbedder) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.db = lancedb.connect(path)
        self.recreated = False
        self._ensure_table()

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("message_ids_json", pa.string()),
                pa.field("text", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("token_count", pa.int32()),
                pa.field("start_offset", pa.int32()),
                pa.field("end_offset", pa.int32()),
                pa.field("content_hash", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedder.dimension)),
            ]
        )

    def _ensure_table(self) -> None:
        if TABLE_NAME in self.db.list_tables().tables:
            field = self.db.open_table(TABLE_NAME).schema.field("vector")
            list_size = getattr(field.type, "list_size", None)
            if list_size != self.embedder.dimension:
                self.db.drop_table(TABLE_NAME)
                self.recreated = True
        if TABLE_NAME not in self.db.list_tables().tables:
            self.db.create_table(TABLE_NAME, schema=self._schema())

    @property
    def table(self):  # type: ignore[no-untyped-def]
        return self.db.open_table(TABLE_NAME)

    def count(self) -> int:
        return int(self.table.count_rows())

    def contains_chunks(self, chunk_ids: list[str] | tuple[str, ...]) -> bool:
        """Return whether any requested canonical chunk is still in the rebuildable index."""
        if not chunk_ids or self.count() == 0:
            return False
        quoted = ", ".join(f"'{self._escape(item)}'" for item in chunk_ids)
        return bool(self.table.count_rows(f"chunk_id IN ({quoted})"))

    def upsert(self, chunks: list[MemoryChunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        ids = [str(chunk.chunk_id) for chunk in chunks]
        self.delete_chunks(ids)
        rows = [
            {
                "chunk_id": str(chunk.chunk_id),
                "session_id": str(chunk.session_id),
                "message_ids_json": json.dumps([str(item) for item in chunk.message_ids]),
                "text": chunk.text,
                "created_at": chunk.created_at.isoformat(),
                "token_count": chunk.token_count,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "content_hash": chunk.content_hash,
                "vector": vector.tolist(),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.table.add(rows)
        self._refresh_fts()

    def delete_chunks(self, chunk_ids: list[str] | tuple[str, ...]) -> None:
        if not chunk_ids or self.count() == 0:
            return
        quoted = ", ".join(f"'{self._escape(item)}'" for item in chunk_ids)
        self.table.delete(f"chunk_id IN ({quoted})")

    def delete_session(self, session_id: str) -> None:
        if self.count() > 0:
            self.table.delete(f"session_id = '{self._escape(session_id)}'")

    def clear(self) -> None:
        if TABLE_NAME in self.db.list_tables().tables:
            self.db.drop_table(TABLE_NAME)
        self._ensure_table()

    def dense_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_chunk_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        if self.count() == 0:
            return []
        vector = self.embedder.embed_query(query)
        results = self.table.search(vector).limit(limit * 2).to_list()
        return self._filter_results(results, limit, excluded_chunk_ids)

    def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_chunk_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        if self.count() == 0 or not query.strip():
            return []
        try:
            results = self.table.search(query, query_type="fts").limit(limit * 2).to_list()
        except Exception:
            self._refresh_fts()
            results = self.table.search(query, query_type="fts").limit(limit * 2).to_list()
        return self._filter_results(results, limit, excluded_chunk_ids)

    def hybrid_search(
        self,
        query: str,
        vector: list[float],
        *,
        limit: int,
        excluded_chunk_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        """Use LanceDB's native FTS/vector retrieval and tested RRF implementation."""
        if self.count() == 0 or not query.strip():
            return []
        try:
            results = (
                self.table.search(query_type="hybrid")
                .vector(vector)
                .text(query)
                .rerank(RRFReranker())
                .limit(limit * 2)
                .to_list()
            )
        except Exception:
            self._refresh_fts()
            results = (
                self.table.search(query_type="hybrid")
                .vector(vector)
                .text(query)
                .rerank(RRFReranker())
                .limit(limit * 2)
                .to_list()
            )
        return self._filter_results(results, limit, excluded_chunk_ids)

    def rows(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], self.table.to_arrow().to_pylist())

    def _refresh_fts(self) -> None:
        if self.count() == 0:
            return
        self.table.create_index(
            "text",
            config=FTS(language="Spanish"),
            replace=True,
        )

    @staticmethod
    def _filter_results(
        results: list[dict[str, object]],
        limit: int,
        excluded_chunk_ids: set[str] | None,
    ) -> list[dict[str, object]]:
        excluded = excluded_chunk_ids or set()
        return [row for row in results if str(row["chunk_id"]) not in excluded][:limit]

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "''")
