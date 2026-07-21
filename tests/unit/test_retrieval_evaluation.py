import gzip
import json
from pathlib import Path

from ramem.evaluation.retrieval import (
    RetrievalEvalConfig,
    iter_corpus,
    load_qrels,
    load_topics,
    reciprocal_rank_fusion,
    retrieval_metrics,
    run,
    select_pilot_corpus,
)


def test_miracl_readers_and_pilot_selection(tmp_path: Path) -> None:
    topics = tmp_path / "topics.tsv"
    qrels = tmp_path / "qrels.tsv"
    corpus = tmp_path / "docs.jsonl.gz"
    topics.write_text("q1\tcapital del Perú\n", encoding="utf-8")
    qrels.write_text("q1\tQ0\td2\t1\n", encoding="utf-8")
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        for row in (
            {"docid": "d1", "title": "Chile", "text": "Santiago"},
            {"docid": "d2", "title": "Perú", "text": "Lima"},
            {"docid": "d3", "title": "Bolivia", "text": "Sucre"},
        ):
            handle.write(json.dumps(row) + "\n")

    selected, stats = select_pilot_corpus(
        iter_corpus([corpus]), relevant_docids={"d2"}, max_background_documents=1
    )

    assert load_topics(topics) == {"q1": "capital del Perú"}
    assert load_qrels(qrels) == {"q1": {"d2"}}
    assert [document.docid for document in selected] == ["d1", "d2"]
    assert stats["scanned_documents"] == 3


def test_retrieval_metrics_and_rrf() -> None:
    metrics = retrieval_metrics({"q1": ["d2", "d3"]}, {"q1": {"d2"}})
    fused = reciprocal_rank_fusion(["d1", "d2"], ["d2", "d3"], rrf_k=60, limit=3)

    assert metrics["recall_at_1"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert fused[0] == "d2"


def test_cpu_evaluation_writes_reproducible_artifacts(tmp_path: Path) -> None:
    topics = tmp_path / "topics.tsv"
    qrels = tmp_path / "qrels.tsv"
    corpus = tmp_path / "docs.jsonl.gz"
    output = tmp_path / "output"
    topics.write_text("q1\tcapital peruana\n", encoding="utf-8")
    qrels.write_text("q1\tQ0\td2\t1\n", encoding="utf-8")
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"docid": "d1", "title": "Chile", "text": "Santiago"}) + "\n")
        handle.write(
            json.dumps({"docid": "d2", "title": "Perú", "text": "capital peruana Lima"}) + "\n"
        )

    summary = run(
        RetrievalEvalConfig(
            topics_path=topics,
            qrels_path=qrels,
            corpus_glob=corpus.as_posix(),
            output_dir=output,
            run_dense=False,
        )
    )

    assert summary["queries"] == 1
    assert summary["modes"]["lexical"]["metrics"]["recall_at_1"] == 1.0
    assert summary["provenance"]["input_sha256"][str(corpus)]
    assert (output / "resolved_config.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "rankings.jsonl").is_file()
