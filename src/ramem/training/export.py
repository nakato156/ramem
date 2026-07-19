from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model_id: str
    adapter_path: Path
    output_dir: Path
    evaluation_summary: Path | None = None


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_merged(config: ExportConfig) -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before export") from error
    if config.output_dir.exists():
        raise FileExistsError(f"Export is immutable and already exists: {config.output_dir}")
    if not config.adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter not found: {config.adapter_path}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to load the gated Gemma base model")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        token=token,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, config.adapter_path)
    merged = model.merge_and_unload(safe_merge=True)
    tokenizer = AutoTokenizer.from_pretrained(config.adapter_path)
    config.output_dir.mkdir(parents=True)
    merged.save_pretrained(config.output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(config.output_dir)
    model_files = sorted(config.output_dir.glob("*.safetensors"))
    if not model_files:
        raise RuntimeError("Merged export did not produce safetensors weights")
    manifest: dict[str, Any] = {
        "format": "merged_transformers_safetensors",
        "base_model_id": config.base_model_id,
        "adapter_path": str(config.adapter_path),
        "adapter_sha256": _sha256(config.adapter_path / "adapter_model.safetensors"),
        "git_commit": _git_commit(),
        "dtype": str(dtype),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in model_files
        },
    }
    if config.evaluation_summary and config.evaluation_summary.is_file():
        manifest["evaluation"] = json.loads(config.evaluation_summary.read_text(encoding="utf-8"))
    (config.output_dir / "ramem_export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a RaMem LoRA adapter for inference")
    parser.add_argument("--config", type=Path, default=Path("configs/export/gemma_1b_ramem.yaml"))
    args = parser.parse_args()
    config = ExportConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    export_merged(config)


if __name__ == "__main__":
    main()
