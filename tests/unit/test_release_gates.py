from ramem.evaluation.release_gates import ReleaseMetrics, evaluate_release_gates


def test_release_gate_report_only_passes_when_every_threshold_passes() -> None:
    metrics = ReleaseMetrics(
        recall_at_10=0.91,
        category_recall_at_10={"extraction": 0.90, "temporal": 0.82},
        recent_window_accuracy=0.60,
        retrieved_accuracy=0.75,
        oracle_accuracy=0.90,
        retrieved_to_oracle_token_ratio=0.10,
        valid_citation_rate=0.99,
        transformers_bf16_token_f1=0.80,
        gguf_q4_token_f1=0.79,
        retrieval_packaging_p95_seconds=1.4,
        gguf_cpu_tokens_per_second=5.1,
        gguf_cpu_first_token_seconds=4.9,
        durability_passed=True,
        deletion_passed=True,
        clean_install_passed=True,
        ci_passed=True,
    )

    assert evaluate_release_gates(metrics).passed


def test_release_gate_report_exposes_failure() -> None:
    metrics = ReleaseMetrics(
        recall_at_10=0.89,
        category_recall_at_10={"extraction": 0.90},
        recent_window_accuracy=0.60,
        retrieved_accuracy=0.75,
        oracle_accuracy=0.90,
        retrieved_to_oracle_token_ratio=0.10,
        valid_citation_rate=0.99,
        transformers_bf16_token_f1=0.80,
        gguf_q4_token_f1=0.79,
        retrieval_packaging_p95_seconds=1.4,
        gguf_cpu_tokens_per_second=5.1,
        gguf_cpu_first_token_seconds=4.9,
        durability_passed=True,
        deletion_passed=True,
        clean_install_passed=True,
        ci_passed=True,
    )

    report = evaluate_release_gates(metrics)

    assert not report.passed
    assert report.gates[0].name == "recall_at_10_global"


def test_missing_model_scores_do_not_accidentally_pass_quantization_gate() -> None:
    metrics = ReleaseMetrics(
        recall_at_10=0.91,
        category_recall_at_10={"extraction": 0.90},
        recent_window_accuracy=0.60,
        retrieved_accuracy=0.75,
        oracle_accuracy=0.90,
        retrieved_to_oracle_token_ratio=0.10,
        valid_citation_rate=0.99,
        transformers_bf16_token_f1=0.0,
        gguf_q4_token_f1=0.0,
        retrieval_packaging_p95_seconds=1.4,
        gguf_cpu_tokens_per_second=5.1,
        gguf_cpu_first_token_seconds=4.9,
        durability_passed=True,
        deletion_passed=True,
        clean_install_passed=True,
        ci_passed=True,
    )

    report = evaluate_release_gates(metrics)

    quantization = next(gate for gate in report.gates if gate.name == "gguf_q4_f1_loss")
    assert not quantization.passed
