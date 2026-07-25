from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ramem.conversation.models import ContextBundle


class PrivacyTraceSink:
    """Persist operational traces while excluding conversation content."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(
        self,
        *,
        session_id: UUID | str,
        query: str,
        context: ContextBundle | None,
        backend: str | None,
        duration_ms: float,
        outcome: str,
        error: BaseException | None = None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "trace_id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            "session_id": str(session_id),
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_chars": len(query),
            "backend": backend,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "memory_count": len(context.memories) if context else 0,
            "memory_tokens": context.memory_tokens if context else 0,
            "recent_tokens": context.recent_tokens if context else 0,
            "evidence_ids": [item.evidence_id for item in context.memories] if context else [],
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
        path = self.directory / f"{payload['trace_id']}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
