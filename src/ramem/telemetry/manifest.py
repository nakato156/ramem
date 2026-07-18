from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ramem.config import AppConfig


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_manifest(
    directory: Path,
    config: AppConfig,
    *,
    run_type: str,
    seed: int = 42,
    dataset_hashes: dict[str, str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_type": run_type,
        "seed": seed,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config.model_dump(mode="json"),
        "dataset_hashes": dataset_hashes or {},
    }
    path = directory / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
