from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalConfig(StrictConfig):
    mode: str = "hybrid"
    lexical_k: int = Field(default=20, gt=0)
    dense_k: int = Field(default=20, gt=0)
    final_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    embedding_dimension: int = Field(default=256, gt=0)


class ContextConfig(StrictConfig):
    token_budget: int = Field(default=1024, gt=0)
    chars_per_token: int = Field(default=4, gt=0)


class GenerationConfig(StrictConfig):
    provider: str = "transformers"
    model_id: str = "google/gemma-3-1b-it"
    adapter_path: Path | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = Field(default=256, gt=0)


class StorageConfig(StrictConfig):
    index_path: Path = Path("artifacts/ramem.sqlite3")


class TelemetryConfig(StrictConfig):
    traces_dir: Path = Path("artifacts/traces")


class AppConfig(StrictConfig):
    version: int = 1
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    generation: GenerationConfig = GenerationConfig()
    storage: StorageConfig = StorageConfig()
    telemetry: TelemetryConfig = TelemetryConfig()


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path or os.environ.get("RAMEM_CONFIG", "configs/default.yaml"))
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = AppConfig.model_validate(raw)
    artifacts = os.environ.get("RAMEM_ARTIFACTS_DIR")
    if artifacts:
        root = Path(artifacts)
        config = config.model_copy(
            update={
                "storage": config.storage.model_copy(update={"index_path": root / "ramem.sqlite3"}),
                "telemetry": config.telemetry.model_copy(update={"traces_dir": root / "traces"}),
            }
        )
    return config
