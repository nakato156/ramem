from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ramem.domain.models import (
    Candidate,
    Document,
    ExecutionState,
    GeneratedAnswer,
    MemoryWriteCandidate,
    PackedContext,
    QueryRequest,
    RetrievalFilters,
    RouteResult,
    VerificationResult,
)


class Router(Protocol):
    def route(self, request: QueryRequest) -> RouteResult: ...


class QueryRewriter(Protocol):
    def rewrite(self, request: QueryRequest, route: RouteResult) -> tuple[str, ...]: ...


class DocumentStore(Protocol):
    def ingest(self, documents: list[Document]) -> int: ...

    def count(self) -> int: ...


class Retriever(Protocol):
    def retrieve(self, query: str, filters: RetrievalFilters) -> tuple[Candidate, ...]: ...


class ContextPacker(Protocol):
    def pack(self, query: str, candidates: tuple[Candidate, ...]) -> PackedContext: ...


class Generator(Protocol):
    def generate(self, query: str, context: PackedContext) -> GeneratedAnswer: ...


class Verifier(Protocol):
    def verify(self, answer: GeneratedAnswer, context: PackedContext) -> VerificationResult: ...


class MemoryWriter(Protocol):
    def propose(
        self, request: QueryRequest, answer: GeneratedAnswer
    ) -> tuple[MemoryWriteCandidate, ...]: ...


class Evaluator(Protocol):
    def evaluate(self, suite_path: Path) -> dict[str, float | int | str]: ...


class TraceSink(Protocol):
    def write(self, state: ExecutionState) -> Path: ...
