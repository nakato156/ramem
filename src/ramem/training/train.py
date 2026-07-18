from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    dataset_path: Path
    output_dir: Path
    seed: int = 42
    max_sequence_length: int = Field(gt=0)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    epochs: float = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    logging_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    eval_steps: int = Field(gt=0)
    save_total_limit: Literal[2] = 2
    resume_from_checkpoint: bool = True
    max_train_samples: int | None = Field(default=None, gt=0)
    max_eval_samples: int | None = Field(default=None, gt=0)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.exists():
        return None
    for candidate in output_dir.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            step = int(candidate.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, candidate))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def _target_modules(model: Any) -> list[str]:
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
    targets = sorted(wanted.intersection(found))
    if not targets:
        raise RuntimeError("No supported LoRA target modules found in the actual checkpoint")
    return targets


def train(config: TrainConfig, max_samples: int | None = None) -> None:
    try:
        import torch  # type: ignore[import-not-found]
        from datasets import load_from_disk  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before training") from error

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; switch the Lightning Studio to T4 or L4")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; add it as a Lightning Studio secret")
    set_seed(config.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        token=token,
        device_map={"": 0},
        dtype=dtype,
        quantization_config=quantization,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = _target_modules(model)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(config.dataset_path))
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    sample_limit = max_samples or config.max_train_samples
    if sample_limit:
        train_data = train_data.shuffle(seed=config.seed).select(
            range(min(sample_limit, len(train_data)))
        )
    eval_limit = config.max_eval_samples
    if max_samples and eval_limit is None:
        eval_limit = max(16, max_samples // 10)
    if eval_limit:
        validation_data = validation_data.shuffle(seed=config.seed).select(
            range(min(eval_limit, len(validation_data)))
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(config.output_dir) if config.resume_from_checkpoint else None
    resolved = config.model_dump(mode="json") | {
        "resolved_target_modules": targets,
        "gpu": torch.cuda.get_device_name(0),
        "compute_dtype": str(dtype),
        "resolved_train_samples": len(train_data),
        "resolved_eval_samples": len(validation_data),
        "resolved_resume_checkpoint": str(checkpoint) if checkpoint else None,
    }
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    args = SFTConfig(
        output_dir=str(config.output_dir),
        max_length=config.max_sequence_length,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        report_to="none",
        seed=config.seed,
        dataset_num_proc=max(1, (os.cpu_count() or 2) // 2),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if checkpoint:
        print(f"Resuming training from {checkpoint}")
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(config.output_dir / "adapter-final"))
    tokenizer.save_pretrained(config.output_dir / "adapter-final")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a real Gemma QLoRA adapter")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("RAMEM_TRAIN_CONFIG", "configs/training/gemma_1b_t4_qlora.yaml")
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Real-data smoke-test limit")
    args = parser.parse_args()
    config = TrainConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    artifacts_root = Path(os.environ.get("RAMEM_ARTIFACTS_DIR", "artifacts"))
    if not config.output_dir.is_absolute() and config.output_dir.parts[0] == "artifacts":
        config = config.model_copy(
            update={"output_dir": artifacts_root.joinpath(*config.output_dir.parts[1:])}
        )
    train(config, args.max_samples)


if __name__ == "__main__":
    main()
