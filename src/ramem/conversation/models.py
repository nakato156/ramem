from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationSession(FrozenModel):
    session_id: UUID = Field(default_factory=uuid4)
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: SessionStatus = SessionStatus.ACTIVE


class ConversationMessage(FrozenModel):
    message_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str


class MemoryChunk(FrozenModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    message_ids: tuple[UUID, ...]
    text: str = Field(min_length=1)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(ge=0)
    token_count: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str


class RetrievedMemory(FrozenModel):
    chunk: MemoryChunk
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rank_sources: tuple[str, ...] = ()
    evidence_id: str = ""


class ContextBundle(FrozenModel):
    query: str
    recent_messages: tuple[ConversationMessage, ...]
    memories: tuple[RetrievedMemory, ...]
    rendered_context: str
    memory_tokens: int = Field(ge=0)
    recent_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)


class ChatTurnResult(FrozenModel):
    session: ConversationSession
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    context: ContextBundle
    answer: str
    cited_evidence_ids: tuple[str, ...] = ()
    invalid_citation_ids: tuple[str, ...] = ()


class StorageStats(FrozenModel):
    sessions: int
    messages: int
    chunks: int
    pending_jobs: int
    failed_jobs: int
    database_bytes: int
