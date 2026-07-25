from __future__ import annotations

import json
import re
from datetime import datetime
from uuid import UUID

import numpy as np

from ramem.config import RetrievalConfig
from ramem.conversation.models import MemoryChunk, RetrievedMemory
from ramem.conversation.store import ConversationStore
from ramem.memory.embeddings import TextEmbedder
from ramem.memory.lance import LanceMemoryIndex


class ConversationalRetriever:
    def __init__(
        self,
        index: LanceMemoryIndex,
        embedder: TextEmbedder,
        config: RetrievalConfig,
        store: ConversationStore,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.config = config
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        excluded_chunk_ids: set[str] | None = None,
    ) -> tuple[RetrievedMemory, ...]:
        if self.config.mode == "hybrid":
            return self._retrieve_native_hybrid(query, excluded_chunk_ids)
        lexical: list[dict[str, object]] = []
        dense: list[dict[str, object]] = []
        if self.config.mode in {"hybrid", "lexical"}:
            lexical = self.store.lexical_search(
                query,
                limit=self.config.lexical_k,
                excluded_chunk_ids=excluded_chunk_ids,
            )
        if self.config.mode in {"hybrid", "dense"}:
            dense = self.index.dense_search(
                query,
                limit=self.config.dense_k,
                excluded_chunk_ids=excluded_chunk_ids,
            )
        ranked: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        rows: dict[str, dict[str, object]] = {}
        dense_scores: dict[str, float] = {}
        lexical_scores: dict[str, float] = {}
        for source, values in (("dense", dense), ("lexical", lexical)):
            for rank, row in enumerate(values, start=1):
                chunk_id = str(row["chunk_id"])
                rows[chunk_id] = {**rows.get(chunk_id, {}), **row}
                ranked[chunk_id] = ranked.get(chunk_id, 0.0) + 1.0 / (self.config.rrf_k + rank)
                sources.setdefault(chunk_id, []).append(source)
                if source == "dense":
                    dense_scores[chunk_id] = 1.0 - self._float(
                        row.get("_distance"),
                        default=1.0,
                    )
                else:
                    lexical_scores[chunk_id] = self._float(
                        row.get("_score"),
                        default=0.0,
                    )
        ordered = sorted(ranked, key=lambda item: (-ranked[item], item))
        ordered = self._remove_near_duplicates(ordered, rows)
        diversified = self._mmr(query, ordered, rows)
        results: list[RetrievedMemory] = []
        for chunk_id in diversified[: self.config.final_k]:
            row = rows[chunk_id]
            results.append(
                RetrievedMemory(
                    chunk=self._chunk(row),
                    dense_score=dense_scores.get(chunk_id),
                    lexical_score=lexical_scores.get(chunk_id),
                    fused_score=ranked[chunk_id],
                    rank_sources=tuple(sources[chunk_id]),
                )
            )
        return tuple(results)

    def _retrieve_native_hybrid(
        self,
        query: str,
        excluded_chunk_ids: set[str] | None,
    ) -> tuple[RetrievedMemory, ...]:
        query_vector = self.embedder.embed_query(query)
        candidate_limit = max(
            self.config.lexical_k,
            self.config.dense_k,
            self.config.final_k * 2,
        )
        native = self.index.hybrid_search(
            query,
            query_vector.tolist(),
            limit=candidate_limit,
            excluded_chunk_ids=excluded_chunk_ids,
        )
        rows = {str(row["chunk_id"]): row for row in native}
        ordered = self._remove_near_duplicates(list(rows), rows)
        diversified = self._mmr(query, ordered, rows, query_vector=query_vector)
        return tuple(
            RetrievedMemory(
                chunk=self._chunk(rows[chunk_id]),
                fused_score=self._float(
                    rows[chunk_id].get("_relevance_score"),
                    default=0.0,
                ),
                rank_sources=("dense", "lexical"),
            )
            for chunk_id in diversified[: self.config.final_k]
        )

    def _mmr(
        self,
        query: str,
        ordered: list[str],
        rows: dict[str, dict[str, object]],
        *,
        query_vector: np.ndarray | None = None,
    ) -> list[str]:
        if len(ordered) < 2:
            return ordered
        query_vector = (
            query_vector if query_vector is not None else self.embedder.embed_query(query)
        )
        missing = [chunk_id for chunk_id in ordered if "vector" not in rows[chunk_id]]
        if missing:
            embedded = self.embedder.embed_documents([str(rows[item]["text"]) for item in missing])
            for chunk_id, vector in zip(missing, embedded, strict=True):
                rows[chunk_id]["vector"] = vector
        vectors = {
            chunk_id: np.asarray(rows[chunk_id]["vector"], dtype=np.float32) for chunk_id in ordered
        }
        selected: list[str] = []
        remaining = set(ordered)
        while remaining:
            best_id = max(
                remaining,
                key=lambda chunk_id: (
                    self.config.mmr_lambda * float(vectors[chunk_id] @ query_vector)
                    - (1.0 - self.config.mmr_lambda)
                    * max(
                        (float(vectors[chunk_id] @ vectors[item]) for item in selected),
                        default=0.0,
                    ),
                    rows[chunk_id]["created_at"],
                    chunk_id,
                ),
            )
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    def _remove_near_duplicates(
        self,
        ordered: list[str],
        rows: dict[str, dict[str, object]],
    ) -> list[str]:
        selected: list[str] = []
        hashes: set[str] = set()
        signatures: list[set[str]] = []
        for chunk_id in ordered:
            content_hash = str(rows[chunk_id].get("content_hash", ""))
            if content_hash and content_hash in hashes:
                continue
            signature = set(re.findall(r"\w+", str(rows[chunk_id]["text"]).casefold()))
            duplicate = any(
                self._jaccard(signature, previous) >= self.config.near_duplicate_threshold
                for previous in signatures
            )
            if duplicate:
                continue
            selected.append(chunk_id)
            hashes.add(content_hash)
            signatures.append(signature)
        return selected

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    @staticmethod
    def _chunk(row: dict[str, object]) -> MemoryChunk:
        return MemoryChunk(
            chunk_id=UUID(str(row["chunk_id"])),
            session_id=UUID(str(row["session_id"])),
            message_ids=tuple(UUID(item) for item in json.loads(str(row["message_ids_json"]))),
            text=str(row["text"]),
            start_offset=ConversationalRetriever._int(row["start_offset"]),
            end_offset=ConversationalRetriever._int(row["end_offset"]),
            token_count=ConversationalRetriever._int(row["token_count"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            content_hash=str(row["content_hash"]),
        )

    @staticmethod
    def _float(value: object, *, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float, str)):
            return float(value)
        raise TypeError(f"expected numeric value, got {type(value).__name__}")

    @staticmethod
    def _int(value: object) -> int:
        if isinstance(value, (int, float, str)):
            return int(value)
        raise TypeError(f"expected integer value, got {type(value).__name__}")
