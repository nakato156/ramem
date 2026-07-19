import json
from pathlib import Path

import yaml

from ramem.evaluation.analysis import AnalysisConfig, analyze, paired_bootstrap_interval


def test_paired_bootstrap_interval_is_deterministic_and_positive() -> None:
    lower, upper = paired_bootstrap_interval([0.1, 0.2, 0.3], samples=1000, seed=42)

    assert 0.09 <= lower <= upper <= 0.31


def test_external_analysis_config_is_valid() -> None:
    config = AnalysisConfig.model_validate(
        yaml.safe_load(Path("configs/evaluation/mlqa_es_analysis.yaml").read_text())
    )

    assert config.acceptance.min_valid_citation_rate == 0.98


def test_analysis_writes_decision_and_worst_cases(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    summary_path = tmp_path / "summary.json"
    base_path.write_text(json.dumps({"id": "1", "token_f1": 0.0, "exact_match": 0.0}) + "\n")
    adapter_path.write_text(
        json.dumps(
            {
                "id": "1",
                "prediction": "Lima [D1]",
                "reference": "Lima [D1]",
                "token_f1": 1.0,
                "exact_match": 1.0,
            }
        )
        + "\n"
    )
    summary_path.write_text(
        json.dumps(
            {
                "base": {
                    "exact_match": 0.0,
                    "token_f1": 0.0,
                    "cites_d1": 0.0,
                    "valid_citations": 0.0,
                    "mean_latency_seconds": 2.0,
                },
                "adapter": {
                    "exact_match": 1.0,
                    "token_f1": 1.0,
                    "cites_d1": 1.0,
                    "valid_citations": 1.0,
                    "mean_latency_seconds": 1.0,
                },
            }
        )
    )
    raw = yaml.safe_load(Path("configs/evaluation/mlqa_es_analysis.yaml").read_text())
    raw.update(
        {
            "predictions_base": base_path,
            "predictions_adapter": adapter_path,
            "summary": summary_path,
            "output_dir": tmp_path / "output",
            "bootstrap_samples": 100,
        }
    )

    result = analyze(AnalysisConfig.model_validate(raw))

    assert result["decision"]["status"] == "accepted_external_dev"
    assert (tmp_path / "output" / "external-dev-report.md").is_file()
