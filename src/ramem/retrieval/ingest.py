from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ramem.domain.models import Document


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    text: str = Field(min_length=1)
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_jsonl_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = SourceDocument.model_validate_json(line)
            if source.id in seen:
                raise ValueError(f"duplicate document id {source.id!r} at line {line_number}")
            seen.add(source.id)
            normalized = " ".join(source.text.split())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            documents.append(
                Document(
                    document_id=source.id,
                    title=source.title,
                    text=normalized,
                    source_uri=source.source_uri,
                    content_hash=digest,
                    metadata=source.metadata,
                )
            )
    return documents
