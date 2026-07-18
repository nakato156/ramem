from __future__ import annotations

import json
from pathlib import Path

from ramem.domain.models import ExecutionState


class JsonTraceSink:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, state: ExecutionState) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{state.request_id}.json"
        path.write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
