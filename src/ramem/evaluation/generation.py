from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import string
import subprocess
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    adapter_path: Path
    dataset_path: Path
    output_dir: Path
    seed: int = 42
    max_samples: int = Field(default=256, gt=0)
    max_input_length: int = Field(default=1024, gt=0)
    max_new_tokens: int = Field(default=64, gt=0)


@dataclass(frozen=True)
class AnswerMetrics:
    exact_match: float
    token_f1: float
    cites_d1: float
    valid_citations: float


_CITATION = re.compile(r"\s*\[D1\]\s*", re.IGNORECASE)
_SPANISH_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}


def _answer_text(text: str) -> str:
    return _CITATION.sub(" ", text).strip()


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", _answer_text(text)).casefold()
    text = "".join(" " if char in string.punctuation else char for char in text)
    return " ".join(token for token in text.split() if token not in _SPANISH_ARTICLES)


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: str) -> AnswerMetrics:
    citation_ids = [
        citation.upper() for citation in re.findall(r"\[(D\d+)\]", prediction, re.IGNORECASE)
    ]
    return AnswerMetrics(
        exact_match=float(normalize_answer(prediction) == normalize_answer(reference)),
        token_f1=token_f1(prediction, reference),
        cites_d1=float(bool(re.search(r"\[D1\]", prediction, re.IGNORECASE))),
        valid_citations=float(bool(citation_ids) and set(citation_ids) == {"D1"}),
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_id: str, adapter_path: Path | None, token: str) -> tuple[Any, Any, Any]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map={"": 0},
        dtype=dtype,
        quantization_config=quantization,
    )
    if adapter_path is not None:
        from peft import PeftModel  # type: ignore[import-not-found]

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer, torch


def _evaluate_variant(
    *,
    name: str,
    rows: Any,
    config: EvaluationConfig,
    token: str,
    adapter_path: Path | None,
) -> dict[str, Any]:
    model, tokenizer, torch = _load_model(config.model_id, adapter_path, token)
    output_path = config.output_dir / f"predictions-{name}.jsonl"
    totals = Counter(
        {
            "exact_match": 0.0,
            "token_f1": 0.0,
            "cites_d1": 0.0,
            "valid_citations": 0.0,
        }
    )
    latencies: list[float] = []
    started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows):
            inputs = tokenizer(
                row["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=config.max_input_length,
            ).to(model.device)
            sample_started = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            latency = time.perf_counter() - sample_started
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            reference = str(row["completion"])
            metrics = score_answer(prediction, reference)
            for key in totals:
                totals[key] += getattr(metrics, key)
            latencies.append(latency)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "id": str(row["id"]),
                        "prompt": str(row["prompt"]),
                        "prediction": prediction,
                        "reference": reference,
                        "input_tokens": int(inputs["input_ids"].shape[-1]),
                        "input_at_max_length": bool(
                            inputs["input_ids"].shape[-1] == config.max_input_length
                        ),
                        "generated_tokens": int(generated.shape[-1]),
                        "latency_seconds": latency,
                        **metrics.__dict__,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            if (index + 1) % 25 == 0:
                print(f"{name}: {index + 1}/{len(rows)}")
    count = len(rows)
    result = {
        "samples": count,
        **{key: value / count for key, value in totals.items()},
        "mean_latency_seconds": sum(latencies) / count,
        "total_runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_from_disk  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before evaluation") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Generation evaluation requires a CUDA GPU")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not config.adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter not found: {config.adapter_path}")
    rows = load_from_disk(str(config.dataset_path))["validation"]
    rows = rows.shuffle(seed=config.seed).select(range(min(config.max_samples, len(rows))))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.model_dump(mode="json") | {
        "resolved_samples": len(rows),
        "git_commit": _git_commit(),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "peft", "datasets", "bitsandbytes")
        },
        "preparation_manifest_sha256": _sha256(config.dataset_path / "preparation_manifest.json"),
        "adapter_sha256": _sha256(config.adapter_path / "adapter_model.safetensors"),
    }
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    results = {
        "base": _evaluate_variant(
            name="base", rows=rows, config=config, token=token, adapter_path=None
        ),
        "adapter": _evaluate_variant(
            name="adapter",
            rows=rows,
            config=config,
            token=token,
            adapter_path=config.adapter_path,
        ),
    }
    results["delta_adapter_minus_base"] = {
        key: results["adapter"][key] - results["base"][key]
        for key in (
            "exact_match",
            "token_f1",
            "cites_d1",
            "valid_citations",
            "mean_latency_seconds",
        )
    }
    (config.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base Gemma against a RaMem adapter")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/gemma_1b_t4.yaml"))
    args = parser.parse_args()
    config = EvaluationConfig.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    evaluate(config)


if __name__ == "__main__":
    main()
