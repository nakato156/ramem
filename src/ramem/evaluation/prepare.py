from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ramem.training.prepare import SYSTEM_POLICY


def _normalized_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _training_hashes(training_source: Path) -> tuple[set[str], set[str]]:
    from datasets import load_from_disk  # type: ignore[import-not-found]

    if not training_source.exists():
        return set(), set()
    rows = load_from_disk(str(training_source))["train"]
    contexts = {_normalized_hash(str(row["context"])) for row in rows}
    questions = {_normalized_hash(str(row["question"])) for row in rows}
    return contexts, questions


def _download_record(source: Path) -> dict[str, Any] | None:
    manifest_path = source.parent / "download_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return next(
        (record for record in manifest.get("datasets", []) if record.get("name") == source.name),
        None,
    )


def prepare_external_qa(
    source: Path,
    output: Path,
    *,
    source_split: str = "validation",
    training_source: Path | None = None,
    purpose: str = "external_development_evaluation",
) -> dict[str, Any]:
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before preparing data") from error
    if output.exists():
        raise FileExistsError(f"{output} already exists; evaluation datasets are immutable")
    raw = load_from_disk(str(source))
    if source_split not in raw:
        raise KeyError(f"split {source_split!r} is absent from {source}")
    training_contexts: set[str] = set()
    training_questions: set[str] = set()
    if training_source is not None:
        training_contexts, training_questions = _training_hashes(training_source)
    rows: list[dict[str, str]] = []
    context_overlap = 0
    question_overlap = 0
    for row in raw[source_split]:
        answers = row["answers"]["text"]
        if not answers:
            continue
        context = str(row["context"])
        question = str(row["question"])
        context_overlap += int(_normalized_hash(context) in training_contexts)
        question_overlap += int(_normalized_hash(question) in training_questions)
        prompt = (
            f"{SYSTEM_POLICY}\n\nEVIDENCIA:\n[D1] {context}\n\n"
            f"PREGUNTA:\n{question}\n\nRESPUESTA:\n"
        )
        rows.append(
            {
                "id": str(row["id"]),
                "prompt": prompt,
                "completion": f"{answers[0]} [D1]",
            }
        )
    if context_overlap or question_overlap:
        raise RuntimeError(
            "External evaluation overlaps training data exactly: "
            f"contexts={context_overlap}, questions={question_overlap}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"validation": Dataset.from_list(rows)}).save_to_disk(output)
    manifest: dict[str, Any] = {
        "source": str(source),
        "source_split": source_split,
        "output": str(output),
        "rows": len(rows),
        "purpose": purpose,
        "training_source": str(training_source) if training_source else None,
        "exact_context_overlap": context_overlap,
        "exact_question_overlap": question_overlap,
        "download_record": _download_record(source),
        "transformation": "extractive QA -> grounded prompt/completion with [D1] citation",
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an external grounded-QA evaluation set")
    raw_root = Path(os.environ.get("RAMEM_RAW_DATA_DIR", "data/raw"))
    processed_root = Path(os.environ.get("RAMEM_PROCESSED_DATA_DIR", "data/processed"))
    parser.add_argument("--source", type=Path, default=raw_root / "mlqa-es-dev")
    parser.add_argument("--output", type=Path, default=processed_root / "mlqa-es-grounded-dev-v1")
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--training-source", type=Path, default=raw_root / "squad-es")
    parser.add_argument(
        "--purpose",
        choices=("external_development_evaluation", "external_final_evaluation"),
        default="external_development_evaluation",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_external_qa(
                args.source,
                args.output,
                source_split=args.source_split,
                training_source=args.training_source,
                purpose=args.purpose,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
