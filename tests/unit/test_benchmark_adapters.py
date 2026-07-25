import json
from pathlib import Path

from ramem.evaluation.benchmark_adapters import convert_locomo, convert_longmemeval


def test_longmemeval_adapter_preserves_official_evidence_sessions(tmp_path: Path) -> None:
    source = tmp_path / "longmem.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "knowledge-update",
                    "question": "¿Cuál es el dato vigente?",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "Ahora es verde"},
                            {"role": "assistant", "content": "Entendido"},
                        ]
                    ],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )

    case = convert_longmemeval(source)[0]

    assert case.category == "update"
    assert case.sessions[0].source_id == "s1"
    assert case.relevant_session_ids == ("s1",)


def test_locomo_adapter_maps_dialog_evidence_to_session(tmp_path: Path) -> None:
    source = tmp_path / "locomo.json"
    source.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample",
                    "conversation": {
                        "speaker_a": "Ana",
                        "speaker_b": "Luis",
                        "session_1_date_time": "1 Jan 2026",
                        "session_1": [
                            {"speaker": "Ana", "dia_id": "d1", "text": "Viajo a Lima"},
                            {"speaker": "Luis", "dia_id": "d2", "text": "Buen viaje"},
                        ],
                    },
                    "qa": [
                        {
                            "question": "¿Adónde viaja Ana?",
                            "answer": "Lima",
                            "category": 1,
                            "evidence": ["d1"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    case = convert_locomo(source)[0]

    assert case.category == "extraction"
    assert case.relevant_session_ids == ("session_1",)
