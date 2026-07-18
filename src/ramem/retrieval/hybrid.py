from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ramem.config import RetrievalConfig
from ramem.domain.models import Candidate, RetrievalFilters
from ramem.retrieval.store import SQLiteDocumentStore
from ramem.retrieval.vectors import cosine_similarity, hashing_vector, tokenize


@dataclass(frozen=True)
class _Ranked:
    document_id: str
    score: float


def _fts_query(query: str) -> str:
    tokens = [token.replace('"', '""') for token in tokenize(query)]
    return " OR ".join(f'"{token}"' for token in tokens)


class HybridRetriever:
    def __init__(self, store: SQLiteDocumentStore, config: RetrievalConfig) -> None:
        self.store = store
        self.config = config

    def _lexical(self, query: str) -> list[_Ranked]:
        expression = _fts_query(query)
        if not expression:
            return []
        with self.store.connect() as connection:
            rows = connection.execute(
                """SELECT document_id, -bm25(documents_fts) AS score
                FROM documents_fts WHERE documents_fts MATCH ?
                ORDER BY bm25(documents_fts) LIMIT ?""",
                (expression, self.config.lexical_k),
            ).fetchall()
        return [_Ranked(str(row["document_id"]), float(row["score"])) for row in rows]

    def _dense(self, query: str) -> list[_Ranked]:
        query_vector = hashing_vector(query, self.config.embedding_dimension)
        query_tokens = set(tokenize(query))
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT document_id, text, vector_json FROM documents"
            ).fetchall()
        ranked = [
            _Ranked(
                str(row["document_id"]),
                cosine_similarity(query_vector, json.loads(row["vector_json"])),
            )
            for row in rows
            if query_tokens.intersection(tokenize(str(row["text"])))
        ]
        meaningful = [item for item in ranked if item.score > 0.0]
        return sorted(meaningful, key=lambda item: (-item.score, item.document_id))[
            : self.config.dense_k
        ]

    def retrieve(self, query: str, filters: RetrievalFilters) -> tuple[Candidate, ...]:
        del filters  # Semantic document rows do not carry user/time fields yet.
        lexical = self._lexical(query)
        dense = self._dense(query)
        if self.config.mode == "lexical":
            dense = []
        elif self.config.mode == "dense":
            lexical = []

        fused: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        lexical_scores = {item.document_id: item.score for item in lexical}
        dense_scores = {item.document_id: item.score for item in dense}
        for source_name, ranking in (("lexical", lexical), ("dense", dense)):
            for rank, item in enumerate(ranking, start=1):
                fused[item.document_id] = fused.get(item.document_id, 0.0) + 1.0 / (
                    self.config.rrf_k + rank
                )
                sources.setdefault(item.document_id, []).append(source_name)

        selected_ids = sorted(fused, key=lambda key: (-fused[key], key))[: self.config.final_k]
        if not selected_ids:
            return ()
        placeholders = ",".join("?" for _ in selected_ids)
        with self.store.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents WHERE document_id IN ({placeholders})",
                selected_ids,
            ).fetchall()
        by_id: dict[str, sqlite3.Row] = {str(row["document_id"]): row for row in rows}
        candidates: list[Candidate] = []
        for document_id in selected_ids:
            row = by_id[document_id]
            text = str(row["text"])
            candidates.append(
                Candidate(
                    document_id=document_id,
                    title=str(row["title"]),
                    text=text,
                    source_uri=row["source_uri"],
                    start_offset=0,
                    end_offset=len(text),
                    lexical_score=lexical_scores.get(document_id),
                    dense_score=dense_scores.get(document_id),
                    fused_score=fused[document_id],
                    rank_sources=tuple(sources[document_id]),
                )
            )
        return tuple(candidates)
