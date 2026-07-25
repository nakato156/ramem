from pathlib import Path

from ramem.config import AppConfig, RetrievalConfig, StorageConfig, TelemetryConfig
from ramem.conversation.service import ConversationService
from ramem.evaluation.memory import evaluate_memory_retrieval, load_memory_benchmark
from ramem.memory.embeddings import HashingEmbedder


def test_memory_benchmark_is_reproducible(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        '{"case_id":"one","category":"extraction","sessions":'
        '[{"title":"p","turns":[{"user":"Mi clave es orquídea azul",'
        '"assistant":"La clave es orquídea azul"}]}],"query":"¿Cuál es la clave?",'
        '"relevant_phrases":["orquídea azul"],"split":"development"}\n',
        encoding="utf-8",
    )
    config = AppConfig(
        retrieval=RetrievalConfig(
            embedding_provider="hashing",
            embedding_dimension=64,
            final_k=10,
        ),
        storage=StorageConfig(
            index_path=tmp_path / "ramem.sqlite3",
            lance_path=tmp_path / "memory.lance",
        ),
        telemetry=TelemetryConfig(traces_dir=tmp_path / "traces"),
    )
    service = ConversationService(config, embedder=HashingEmbedder(64))

    result = evaluate_memory_retrieval(service, load_memory_benchmark(dataset), k=10)

    assert result.recall_at_k == 1.0
    assert result.indexed_messages == 2
