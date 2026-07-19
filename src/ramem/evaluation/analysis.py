from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AcceptanceCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_exact_match: float = Field(ge=0.0, le=1.0)
    min_token_f1: float = Field(ge=0.0, le=1.0)
    min_f1_delta: float = Field(ge=-1.0, le=1.0)
    min_citation_rate: float = Field(ge=0.0, le=1.0)
    min_valid_citation_rate: float = Field(ge=0.0, le=1.0)
    max_mean_latency_seconds: float = Field(gt=0.0)


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictions_base: Path
    predictions_adapter: Path
    summary: Path
    output_dir: Path
    bootstrap_samples: int = Field(default=10_000, gt=99)
    seed: int = 42
    worst_cases: int = Field(default=25, gt=0)
    acceptance: AcceptanceCriteria


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paired_bootstrap_interval(
    differences: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not differences:
        raise ValueError("at least one paired difference is required")
    rng = random.Random(seed)
    count = len(differences)
    means = sorted(
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def _category(row: dict[str, Any]) -> str:
    if row["exact_match"] == 1.0:
        return "exact"
    if row["token_f1"] >= 0.5:
        return "partial_high"
    if row["token_f1"] > 0.0:
        return "partial_low"
    return "wrong_no_overlap"


def _report_markdown(analysis: dict[str, Any], summary: dict[str, Any]) -> str:
    base = summary["base"]
    adapter = summary["adapter"]
    decision = analysis["decision"]
    rows = []
    for label, key, digits in (
        ("Exact match", "exact_match", 4),
        ("Token F1", "token_f1", 4),
        ("Citation `[D1]`", "cites_d1", 4),
        ("Valid citations only", "valid_citations", 4),
        ("Mean latency (s)", "mean_latency_seconds", 3),
    ):
        rows.append(
            f"| {label} | {base[key]:.{digits}f} | {adapter[key]:.{digits}f} | "
            f"{adapter[key] - base[key]:+.{digits}f} |"
        )
    metric_rows = "\n".join(rows)
    f1_interval = analysis["confidence_intervals"]["token_f1_delta_95"]
    checks = json.dumps(decision["checks"], sort_keys=True)
    return f"""# RaMem external development evaluation

Status: **{decision["status"]}**

This report is a development evaluation, not a final test result. The external test split remains
reserved and must not be used for prompt, threshold, or training decisions.

| Metric | Base | Adapter | Delta |
|---|---:|---:|---:|
{metric_rows}

Paired-bootstrap 95% confidence interval for adapter-minus-base token F1:
`[{f1_interval[0]:.4f}, {f1_interval[1]:.4f}]`.

Acceptance checks: `{checks}`.
"""


def analyze(config: AnalysisConfig) -> dict[str, Any]:
    base_rows = _load_jsonl(config.predictions_base)
    adapter_rows = _load_jsonl(config.predictions_adapter)
    if len(base_rows) != len(adapter_rows):
        raise ValueError("base and adapter prediction counts differ")
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for base, adapter in zip(base_rows, adapter_rows, strict=True):
        if base["id"] != adapter["id"]:
            raise ValueError("base and adapter prediction ordering differs")
        paired.append((base, adapter))
    summary = json.loads(config.summary.read_text(encoding="utf-8"))
    f1_differences = [
        float(adapter["token_f1"]) - float(base["token_f1"]) for base, adapter in paired
    ]
    em_differences = [
        float(adapter["exact_match"]) - float(base["exact_match"]) for base, adapter in paired
    ]
    checks = {
        "exact_match": summary["adapter"]["exact_match"] >= config.acceptance.min_exact_match,
        "token_f1": summary["adapter"]["token_f1"] >= config.acceptance.min_token_f1,
        "f1_delta": (
            summary["adapter"]["token_f1"] - summary["base"]["token_f1"]
            >= config.acceptance.min_f1_delta
        ),
        "citation_rate": (summary["adapter"]["cites_d1"] >= config.acceptance.min_citation_rate),
        "valid_citation_rate": (
            summary["adapter"]["valid_citations"] >= config.acceptance.min_valid_citation_rate
        ),
        "latency": (
            summary["adapter"]["mean_latency_seconds"] <= config.acceptance.max_mean_latency_seconds
        ),
    }
    categorized = Counter(_category(row) for row in adapter_rows)
    analysis: dict[str, Any] = {
        "samples": len(paired),
        "decision": {
            "status": "accepted_external_dev" if all(checks.values()) else "needs_review",
            "checks": checks,
            "criteria": config.acceptance.model_dump(),
        },
        "adapter_error_categories": dict(categorized),
        "confidence_intervals": {
            "token_f1_delta_95": paired_bootstrap_interval(
                f1_differences, samples=config.bootstrap_samples, seed=config.seed
            ),
            "exact_match_delta_95": paired_bootstrap_interval(
                em_differences, samples=config.bootstrap_samples, seed=config.seed + 1
            ),
        },
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(
        adapter_rows,
        key=lambda row: (float(row["token_f1"]), float(row["exact_match"])),
    )[: config.worst_cases]
    with (config.output_dir / "worst-cases-adapter.jsonl").open("w", encoding="utf-8") as handle:
        for row in worst:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (config.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    (config.output_dir / "external-dev-report.md").write_text(
        _report_markdown(analysis, summary), encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired RaMem generation predictions")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/mlqa_es_analysis.yaml")
    )
    args = parser.parse_args()
    config = AnalysisConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    analyze(config)


if __name__ == "__main__":
    main()
