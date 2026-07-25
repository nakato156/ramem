from __future__ import annotations

import os
from pathlib import Path

import yaml
from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalConfig(StrictConfig):
    mode: str = "hybrid"
    lexical_k: int = Field(default=20, gt=0)
    dense_k: int = Field(default=20, gt=0)
    final_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    embedding_dimension: int = Field(default=256, gt=0)
    embedding_model_id: str = "google/embeddinggemma-300m"
    embedding_provider: str = "embeddinggemma"
    chunk_tokens: int = Field(default=512, gt=0)
    memory_token_budget: int = Field(default=2048, gt=0)
    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    near_duplicate_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_embedding_dimension(self) -> RetrievalConfig:
        if self.embedding_provider == "embeddinggemma" and self.embedding_dimension not in {
            128,
            256,
            768,
        }:
            raise ValueError("EmbeddingGemma dimension must be one of 128, 256, or 768")
        return self


class ContextConfig(StrictConfig):
    token_budget: int = Field(default=2048, gt=0)
    chars_per_token: int = Field(default=4, gt=0)
    recent_token_budget: int = Field(default=768, gt=0)
    instruction_token_budget: int = Field(default=256, gt=0)
    total_token_budget: int = Field(default=4096, gt=0)


class GenerationConfig(StrictConfig):
    provider: str = "auto"
    model_id: str = "nakato156/ramem-gemma-1b"
    model_revision: str = "main"
    gguf_filename: str = "ramem-gemma-1b-q4_k_m.gguf"
    gguf_path: Path | None = None
    adapter_path: Path | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = Field(default=1024, gt=0)
    context_length: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    gpu_layers: int = 0
    seed: int = 42


class StorageConfig(StrictConfig):
    index_path: Path | None = None
    lance_path: Path | None = None


class TelemetryConfig(StrictConfig):
    traces_dir: Path | None = None


class AppConfig(StrictConfig):
    version: int = 1
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    generation: GenerationConfig = GenerationConfig()
    storage: StorageConfig = StorageConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

    @model_validator(mode="after")
    def validate_context_budget(self) -> AppConfig:
        allocated = (
            self.context.instruction_token_budget
            + self.context.recent_token_budget
            + self.context.token_budget
            + self.generation.max_new_tokens
        )
        if allocated > self.context.total_token_budget:
            raise ValueError(
                "instruction, recent, memory, and response budgets exceed total_token_budget"
            )
        return self


def load_config(path: Path | str | None = None) -> AppConfig:
    configured_path = path or os.environ.get("RAMEM_CONFIG")
    if configured_path is None:
        config = AppConfig()
    else:
        with Path(configured_path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = AppConfig.model_validate(raw)
    artifacts = os.environ.get("RAMEM_ARTIFACTS_DIR")
    if artifacts:
        root = Path(artifacts)
        config = config.model_copy(
            update={
                "telemetry": config.telemetry.model_copy(update={"traces_dir": root / "traces"}),
            }
        )
    return config


def resolve_data_paths(config: AppConfig) -> tuple[Path, Path, Path]:
    """Return stable per-user paths without depending on the working directory."""
    root = Path(os.environ.get("RAMEM_DATA_DIR", user_data_path("ramem", "ramem")))
    sqlite_path = config.storage.index_path or root / "ramem.sqlite3"
    lance_path = config.storage.lance_path or root / "memory.lance"
    traces_path = config.telemetry.traces_dir or root / "traces"
    return sqlite_path.expanduser(), lance_path.expanduser(), traces_path.expanduser()
