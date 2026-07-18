from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RouteLabel(StrEnum):
    DIRECT = "direct"
    EPISODIC = "episodic"
    KNOWLEDGE = "knowledge"
    WEB = "web"
    HYBRID = "hybrid"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WEB = "web"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class WriteDecision(StrEnum):
    WRITE = "write"
    UPDATE = "update"
    IGNORE = "ignore"
    ASK_CONSENT = "ask_consent"


class QueryRequest(FrozenModel):
    query: str = Field(min_length=1)
    session_id: str = "local"
    user_id_hash: str = "anonymous"
    conversation_window: tuple[str, ...] = ()


class RouteResult(FrozenModel):
    labels: tuple[RouteLabel, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class RetrievalFilters(FrozenModel):
    memory_types: tuple[MemoryType, ...] = (MemoryType.SEMANTIC,)
    user_id_hash: str | None = None
    valid_at: datetime | None = None


class Document(FrozenModel):
    document_id: str
    title: str
    text: str = Field(min_length=1)
    source_uri: str | None = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(FrozenModel):
    evidence_id: str = ""
    document_id: str
    title: str
    text: str
    source_uri: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    lexical_score: float | None = None
    dense_score: float | None = None
    fused_score: float = 0.0
    rank_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> Candidate:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be >= start_offset")
        return self


class PackedContext(FrozenModel):
    text: str
    evidence: tuple[Candidate, ...]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)


class Citation(FrozenModel):
    evidence_id: str
    document_id: str
    source_uri: str | None = None
    start_offset: int
    end_offset: int


class GeneratedAnswer(FrozenModel):
    text: str
    citations: tuple[Citation, ...] = ()
    abstained: bool = False


class VerificationResult(FrozenModel):
    status: SupportStatus
    valid_citations: bool
    citation_precision: float = Field(ge=0.0, le=1.0)
    reason: str


class MemoryWriteCandidate(FrozenModel):
    decision: WriteDecision
    memory_type: MemoryType = MemoryType.EPISODIC
    fact: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    sensitivity: str = "personal"
    valid_from: datetime | None = None
    supersedes_id: UUID | None = None
    reason_code: str


class StageTiming(FrozenModel):
    stage: str
    duration_ms: float = Field(ge=0.0)


class ExecutionState(FrozenModel):
    request_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str
    user_id_hash: str
    query_original: str
    conversation_window: tuple[str, ...] = ()
    route_labels: tuple[RouteLabel, ...] = ()
    route_confidence: float = 0.0
    rewritten_queries: tuple[str, ...] = ()
    retrieval_filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    candidates: tuple[Candidate, ...] = ()
    selected_evidence: tuple[Candidate, ...] = ()
    context_token_budget: int = 1024
    generation_config: dict[str, Any] = Field(default_factory=dict)
    draft_answer: str | None = None
    citations: tuple[Citation, ...] = ()
    verification_result: VerificationResult | None = None
    final_answer: str | None = None
    memory_write_candidates: tuple[MemoryWriteCandidate, ...] = ()
    trace_timings: tuple[StageTiming, ...] = ()
    resource_metrics: dict[str, float] = Field(default_factory=dict)
    abstained: bool = False
