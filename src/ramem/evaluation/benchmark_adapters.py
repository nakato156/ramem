from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from ramem.evaluation.memory import (
    BenchmarkSession,
    BenchmarkTurn,
    MemoryBenchmarkCase,
)

Category = Literal["extraction", "cross_session", "update", "temporal", "abstention"]

LONGMEM_CATEGORY: dict[str, Category] = {
    "single-session-user": "extraction",
    "single-session-assistant": "extraction",
    "single-session-preference": "extraction",
    "multi-session": "cross_session",
    "knowledge-update": "update",
    "temporal-reasoning": "temporal",
}


def _paired_turns(messages: list[dict[str, Any]]) -> tuple[BenchmarkTurn, ...]:
    turns: list[BenchmarkTurn] = []
    pending_user: str | None = None
    for message in messages:
        role = str(message.get("role", "")).casefold()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            turns.append(BenchmarkTurn(user=pending_user, assistant=content))
            pending_user = None
    return tuple(turns)


def convert_longmemeval(path: Path) -> tuple[MemoryBenchmarkCase, ...]:
    """Convert the official dataset without reimplementing its answer evaluator."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LongMemEval input must be a JSON list")
    cases: list[MemoryBenchmarkCase] = []
    for row in payload:
        question_id = str(row["question_id"])
        abstention = question_id.endswith("_abs")
        session_ids = [str(value) for value in row["haystack_session_ids"]]
        dates = [str(value) for value in row["haystack_dates"]]
        histories = row["haystack_sessions"]
        if not (len(session_ids) == len(dates) == len(histories)):
            raise ValueError(f"unaligned LongMemEval sessions in {question_id}")
        sessions_list: list[BenchmarkSession] = []
        for source_id, date, messages in zip(
            session_ids,
            dates,
            histories,
            strict=True,
        ):
            turns = _paired_turns(messages)
            if turns:
                sessions_list.append(
                    BenchmarkSession(
                        title=f"{source_id} · {date}",
                        source_id=source_id,
                        turns=turns,
                    )
                )
        category: Category = (
            "abstention"
            if abstention
            else LONGMEM_CATEGORY.get(str(row["question_type"]), "cross_session")
        )
        cases.append(
            MemoryBenchmarkCase(
                case_id=question_id,
                category=category,
                sessions=tuple(sessions_list),
                query=str(row["question"]),
                relevant_session_ids=()
                if abstention
                else tuple(str(value) for value in row["answer_session_ids"]),
            )
        )
    return tuple(cases)


def convert_locomo(path: Path) -> tuple[MemoryBenchmarkCase, ...]:
    """Convert LoCoMo QA evidence IDs to session-level retrieval cases."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo input must be a JSON list")
    cases: list[MemoryBenchmarkCase] = []
    for sample in payload:
        sample_id = str(sample["sample_id"])
        conversation = sample["conversation"]
        speaker_a = str(conversation["speaker_a"])
        session_keys = sorted(
            (
                key
                for key, value in conversation.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and isinstance(value, list)
            ),
            key=lambda value: int(value.removeprefix("session_")),
        )
        dialog_to_session: dict[str, str] = {}
        sessions: list[BenchmarkSession] = []
        for key in session_keys:
            messages: list[dict[str, Any]] = []
            for turn in conversation[key]:
                dialog_to_session[str(turn["dia_id"])] = key
                messages.append(
                    {
                        "role": "user" if str(turn["speaker"]) == speaker_a else "assistant",
                        "content": str(turn["text"]),
                    }
                )
            paired = _paired_turns(messages)
            if paired:
                date = str(conversation.get(f"{key}_date_time", ""))
                sessions.append(
                    BenchmarkSession(
                        title=f"{key} · {date}",
                        source_id=key,
                        turns=paired,
                    )
                )
        for index, qa in enumerate(sample["qa"]):
            evidence = tuple(
                dict.fromkeys(
                    dialog_to_session[str(dialog_id)]
                    for dialog_id in qa.get("evidence", [])
                    if str(dialog_id) in dialog_to_session
                )
            )
            raw_category = int(qa.get("category", 1))
            category_by_number: dict[int, Category] = {
                1: "extraction",
                2: "cross_session",
                3: "temporal",
                4: "abstention",
            }
            category = category_by_number.get(raw_category, "cross_session")
            cases.append(
                MemoryBenchmarkCase(
                    case_id=f"locomo-{sample_id}-{index:04d}",
                    corpus_id=f"locomo-{sample_id}",
                    category=category,
                    sessions=tuple(sessions) if index == 0 else (),
                    query=str(qa["question"]),
                    relevant_session_ids=evidence,
                )
            )
    return tuple(cases)


def write_cases(cases: tuple[MemoryBenchmarkCase, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt official memory datasets to RAMEM retrieval cases"
    )
    parser.add_argument("format", choices=("longmemeval", "locomo"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    converter = convert_longmemeval if args.format == "longmemeval" else convert_locomo
    cases = converter(args.input)
    write_cases(cases, args.output)
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
