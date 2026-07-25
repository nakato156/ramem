from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ReleaseMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_10: float = Field(ge=0.0, le=1.0)
    category_recall_at_10: dict[str, float]
    recent_window_accuracy: float = Field(ge=0.0, le=1.0)
    retrieved_accuracy: float = Field(ge=0.0, le=1.0)
    oracle_accuracy: float = Field(gt=0.0, le=1.0)
    retrieved_to_oracle_token_ratio: float = Field(ge=0.0)
    valid_citation_rate: float = Field(ge=0.0, le=1.0)
    transformers_bf16_token_f1: float = Field(ge=0.0, le=1.0)
    gguf_q4_token_f1: float = Field(ge=0.0, le=1.0)
    retrieval_packaging_p95_seconds: float = Field(ge=0.0)
    gguf_cpu_tokens_per_second: float = Field(ge=0.0)
    gguf_cpu_first_token_seconds: float = Field(ge=0.0)
    durability_passed: bool
    deletion_passed: bool
    clean_install_passed: bool
    ci_passed: bool


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    observed: float | bool
    threshold: str


class ReleaseGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    gates: tuple[GateResult, ...]


def evaluate_release_gates(metrics: ReleaseMetrics) -> ReleaseGateReport:
    minimum_category = min(metrics.category_recall_at_10.values(), default=0.0)
    gates = (
        GateResult(
            name="recall_at_10_global",
            passed=metrics.recall_at_10 >= 0.90,
            observed=metrics.recall_at_10,
            threshold=">= 0.90",
        ),
        GateResult(
            name="recall_at_10_categories",
            passed=minimum_category >= 0.80,
            observed=minimum_category,
            threshold=">= 0.80",
        ),
        GateResult(
            name="improvement_over_recent_window",
            passed=metrics.retrieved_accuracy - metrics.recent_window_accuracy >= 0.10,
            observed=metrics.retrieved_accuracy - metrics.recent_window_accuracy,
            threshold=">= 0.10",
        ),
        GateResult(
            name="oracle_accuracy_retention",
            passed=metrics.retrieved_accuracy / metrics.oracle_accuracy >= 0.80,
            observed=metrics.retrieved_accuracy / metrics.oracle_accuracy,
            threshold=">= 0.80",
        ),
        GateResult(
            name="oracle_token_efficiency",
            passed=metrics.retrieved_to_oracle_token_ratio <= 0.10,
            observed=metrics.retrieved_to_oracle_token_ratio,
            threshold="<= 0.10",
        ),
        GateResult(
            name="valid_citations",
            passed=metrics.valid_citation_rate >= 0.98,
            observed=metrics.valid_citation_rate,
            threshold=">= 0.98",
        ),
        GateResult(
            name="gguf_q4_f1_loss",
            passed=metrics.transformers_bf16_token_f1 > 0
            and metrics.gguf_q4_token_f1 > 0
            and metrics.transformers_bf16_token_f1 - metrics.gguf_q4_token_f1 <= 0.02,
            observed=metrics.transformers_bf16_token_f1 - metrics.gguf_q4_token_f1,
            threshold="<= 0.02",
        ),
        GateResult(
            name="retrieval_packaging_p95",
            passed=metrics.retrieval_packaging_p95_seconds <= 1.5,
            observed=metrics.retrieval_packaging_p95_seconds,
            threshold="<= 1.5 s",
        ),
        GateResult(
            name="gguf_cpu_throughput",
            passed=metrics.gguf_cpu_tokens_per_second >= 5.0,
            observed=metrics.gguf_cpu_tokens_per_second,
            threshold=">= 5 tokens/s",
        ),
        GateResult(
            name="gguf_cpu_first_token",
            passed=metrics.gguf_cpu_first_token_seconds < 5.0,
            observed=metrics.gguf_cpu_first_token_seconds,
            threshold="< 5 s",
        ),
        *(
            GateResult(name=name, passed=value, observed=value, threshold="true")
            for name, value in (
                ("durability", metrics.durability_passed),
                ("complete_deletion", metrics.deletion_passed),
                ("clean_install", metrics.clean_install_passed),
                ("ci", metrics.ci_passed),
            )
        ),
    )
    return ReleaseGateReport(passed=all(gate.passed for gate in gates), gates=gates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all RAMEM V1 release gates")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = ReleaseMetrics.model_validate_json(args.metrics.read_text(encoding="utf-8"))
    report = evaluate_release_gates(metrics)
    rendered = report.model_dump_json(indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
