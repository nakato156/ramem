from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SYSTEM_POLICY = (
    "Responde en español únicamente con la evidencia. Cita la evidencia como [D1]. "
    "No agregues hechos que no aparezcan en el fragmento."
)


def _is_validation(example_id: str, percent: int) -> bool:
    bucket = int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) % 100
    return bucket < percent


def prepare_squad(source: Path, output: Path, validation_percent: int) -> dict[str, int]:
    try:
        from datasets import Dataset, DatasetDict, load_from_disk  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before preparing data") from error

    raw = load_from_disk(str(source))["train"]
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    for row in raw:
        answers = row["answers"]["text"]
        if not answers:
            continue
        prompt = (
            f"{SYSTEM_POLICY}\n\nEVIDENCIA:\n[D1] {row['context']}\n\n"
            f"PREGUNTA:\n{row['question']}\n\nRESPUESTA:"
        )
        prepared = {"id": str(row["id"]), "prompt": prompt, "completion": f"{answers[0]} [D1]"}
        target = (
            validation_rows if _is_validation(str(row["id"]), validation_percent) else train_rows
        )
        target.append(prepared)
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; processed datasets are versioned artifacts"
        )
    dataset = DatasetDict(
        {"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(validation_rows)}
    )
    dataset.save_to_disk(output)
    stats = {"train": len(train_rows), "validation": len(validation_rows)}
    (output / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "validation_percent": validation_percent,
                "stats": stats,
                "transformation": (
                    "SQuAD-es train -> grounded prompt/completion with source citation"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare real training data without synthetic rows"
    )
    raw_root = Path(os.environ.get("RAMEM_RAW_DATA_DIR", "data/raw"))
    processed_root = Path(os.environ.get("RAMEM_PROCESSED_DATA_DIR", "data/processed"))
    parser.add_argument("--source", type=Path, default=raw_root / "squad-es")
    parser.add_argument("--output", type=Path, default=processed_root / "grounded-qa-es-v1")
    parser.add_argument("--validation-percent", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 20:
        parser.error("--validation-percent must be between 1 and 20")
    print(json.dumps(prepare_squad(args.source, args.output, args.validation_percent), indent=2))


if __name__ == "__main__":
    main()
