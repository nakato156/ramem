from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ramem.conversation.service import ConversationService


class BenchmarkTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str
    assistant: str


class BenchmarkSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    source_id: str | None = None
    turns: tuple[BenchmarkTurn, ...]


class MemoryBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    corpus_id: str | None = None
    category: Literal["extraction", "cross_session", "update", "temporal", "abstention"]
    sessions: tuple[BenchmarkSession, ...]
    query: str
    relevant_phrases: tuple[str, ...] = ()
    relevant_session_ids: tuple[str, ...] = ()
    split: Literal["development", "holdout"] = "development"


class CategoryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cases: int
    recall_at_k: float = Field(ge=0.0, le=1.0)


class MemoryBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    cases: int
    k: int
    recall_at_k: float = Field(ge=0.0, le=1.0)
    categories: dict[str, CategoryMetrics]
    mean_latency_ms: float
    p95_latency_ms: float
    indexed_messages: int
    indexed_chunks: int


def load_memory_benchmark(path: Path) -> tuple[MemoryBenchmarkCase, ...]:
    cases: list[MemoryBenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(MemoryBenchmarkCase.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"invalid benchmark row {line_number} in {path}") from error
    if not cases:
        raise ValueError(f"memory benchmark is empty: {path}")
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("memory benchmark case_id values must be unique")
    return tuple(cases)


def evaluate_memory_retrieval(
    service: ConversationService,
    cases: tuple[MemoryBenchmarkCase, ...],
    *,
    k: int = 10,
) -> MemoryBenchmarkResult:
    if k <= 0:
        raise ValueError("k must be positive")
    source_sessions: dict[tuple[str, str], str] = {}
    for case in cases:
        corpus_id = case.corpus_id or case.case_id
        for source_session in case.sessions:
            if source_session.source_id is not None:
                source_key = (corpus_id, source_session.source_id)
                if source_key in source_sessions:
                    continue
            session = service.open_session(new=True, title=f"{corpus_id}:{source_session.title}")
            if source_session.source_id is not None:
                source_sessions[source_key] = str(session.session_id)
            for turn in source_session.turns:
                service.store.append_message(session.session_id, "user", turn.user)
                service.store.append_message(session.session_id, "assistant", turn.assistant)
                messages = service.store.messages(session.session_id, limit=3)
                for chunk in service.chunker.build_exchange_chunks(messages):
                    service.store.add_chunk(chunk)
    service.worker.process(limit=max(100, service.store.stats().chunks + 1))

    latencies: list[float] = []
    recalls: list[float] = []
    by_category: dict[str, list[float]] = defaultdict(list)
    original_config = service.retriever.config
    service.retriever.config = original_config.model_copy(update={"final_k": max(k, 1)})
    try:
        for case in cases:
            started = time.perf_counter()
            memories = service.search(case.query)[:k]
            latencies.append((time.perf_counter() - started) * 1000)
            if case.relevant_session_ids:
                corpus_id = case.corpus_id or case.case_id
                expected = {
                    source_sessions[(corpus_id, source_id)]
                    for source_id in case.relevant_session_ids
                }
                retrieved_sessions = {str(memory.chunk.session_id) for memory in memories}
                recall = len(expected & retrieved_sessions) / len(expected)
            elif not case.relevant_phrases:
                recall = 1.0
            else:
                retrieved = "\n".join(memory.chunk.text.casefold() for memory in memories)
                hits = sum(phrase.casefold() in retrieved for phrase in case.relevant_phrases)
                recall = hits / len(case.relevant_phrases)
            recalls.append(recall)
            by_category[case.category].append(recall)
    finally:
        service.retriever.config = original_config

    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    stats = service.store.stats()
    return MemoryBenchmarkResult(
        dataset="RaMem-Memory-ES",
        cases=len(cases),
        k=k,
        recall_at_k=statistics.fmean(recalls),
        categories={
            category: CategoryMetrics(cases=len(values), recall_at_k=statistics.fmean(values))
            for category, values in sorted(by_category.items())
        },
        mean_latency_ms=statistics.fmean(latencies),
        p95_latency_ms=ordered[p95_index],
        indexed_messages=stats.messages,
        indexed_chunks=stats.chunks,
    )


def write_benchmark_result(result: MemoryBenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
