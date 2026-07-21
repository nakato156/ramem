from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import platform
import sqlite3
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ramem.retrieval.vectors import tokenize


class RetrievalEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topics_path: Path
    qrels_path: Path
    corpus_glob: str
    output_dir: Path
    max_background_documents: int | None = Field(default=None, gt=0)
    max_queries: int | None = Field(default=None, gt=0)
    lexical_k: int = Field(default=100, gt=0)
    dense_k: int = Field(default=100, gt=0)
    final_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    embedding_model_id: str = "google/embeddinggemma-300m"
    embedding_dimension: int = Field(default=768, gt=0, le=768)
    embedding_batch_size: int = Field(default=8, gt=0)
    embedding_device: str = "cuda"
    run_dense: bool = True
    seed: int = 42


@dataclass(frozen=True)
class CorpusDocument:
    docid: str
    title: str
    text: str


def load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            query_id, query = line.rstrip("\n").split("\t", maxsplit=1)
            if query_id in topics:
                raise ValueError(f"duplicate query {query_id!r} at line {line_number}")
            topics[query_id] = query
    return topics


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"invalid qrel at line {line_number}")
            query_id, _iteration, docid, relevance = fields
            if int(relevance) > 0:
                qrels.setdefault(query_id, set()).add(docid)
    return qrels


def iter_corpus(paths: Iterable[Path]) -> Iterable[CorpusDocument]:
    for path in sorted(paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    yield CorpusDocument(
                        docid=str(row["docid"]),
                        title=str(row.get("title", "")),
                        text=str(row["text"]),
                    )
                except KeyError as error:
                    raise ValueError(f"missing field in {path}:{line_number}") from error


def select_pilot_corpus(
    documents: Iterable[CorpusDocument],
    *,
    relevant_docids: set[str],
    max_background_documents: int | None,
) -> tuple[list[CorpusDocument], dict[str, int]]:
    selected: dict[str, CorpusDocument] = {}
    scanned = 0
    background = 0
    relevant = 0
    for document in documents:
        scanned += 1
        is_relevant = document.docid in relevant_docids
        include_background = (
            max_background_documents is None or background < max_background_documents
        )
        if not is_relevant and not include_background:
            continue
        if document.docid in selected:
            raise ValueError(f"duplicate corpus document {document.docid!r}")
        selected[document.docid] = document
        if is_relevant:
            relevant += 1
        else:
            background += 1
    return list(selected.values()), {
        "scanned_documents": scanned,
        "selected_documents": len(selected),
        "selected_background_documents": background,
        "selected_relevant_documents": relevant,
    }


def _fts_query(query: str) -> str:
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokenize(query))


class LexicalIndex:
    def __init__(self, path: Path) -> None:
        self.path = path

    def build(self, documents: list[CorpusDocument]) -> None:
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite retrieval index: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE VIRTUAL TABLE documents USING fts5("
                "docid UNINDEXED, title, text, tokenize='unicode61 remove_diacritics 2')"
            )
            connection.executemany(
                "INSERT INTO documents(docid, title, text) VALUES (?, ?, ?)",
                ((document.docid, document.title, document.text) for document in documents),
            )

    def search(self, query: str, limit: int) -> list[str]:
        expression = _fts_query(query)
        if not expression:
            return []
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT docid FROM documents WHERE documents MATCH ? "
                "ORDER BY bm25(documents) LIMIT ?",
                (expression, limit),
            ).fetchall()
        return [str(row[0]) for row in rows]


def reciprocal_rank_fusion(
    lexical: list[str], dense: list[str], *, rrf_k: int, limit: int
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (lexical, dense):
        for rank, docid in enumerate(ranking, start=1):
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda docid: (-scores[docid], docid))[:limit]


def retrieval_metrics(
    rankings: dict[str, list[str]], qrels: dict[str, set[str]]
) -> dict[str, float]:
    if not rankings:
        raise ValueError("no rankings to evaluate")
    recalls: dict[int, list[float]] = {cutoff: [] for cutoff in (1, 5, 10, 20)}
    ndcg_10: list[float] = []
    reciprocal_ranks: list[float] = []
    for query_id, ranking in rankings.items():
        relevant = qrels[query_id]
        for cutoff, values in recalls.items():
            values.append(len(set(ranking[:cutoff]).intersection(relevant)) / len(relevant))
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, docid in enumerate(ranking[:10], start=1)
            if docid in relevant
        )
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
        ndcg_10.append(dcg / ideal if ideal else 0.0)
        first = next(
            (rank for rank, docid in enumerate(ranking[:10], start=1) if docid in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first if first else 0.0)
    return {
        **{f"recall_at_{cutoff}": statistics.fmean(values) for cutoff, values in recalls.items()},
        "ndcg_at_10": statistics.fmean(ndcg_10),
        "mrr_at_10": statistics.fmean(reciprocal_ranks),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered) * 1000,
        "p50_ms": ordered[len(ordered) // 2] * 1000,
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "sentence-transformers", "torch", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    peak_rss: int | None = None
    try:
        psutil = import_module("psutil")
        memory = psutil.Process().memory_info()
        peak_rss = int(getattr(memory, "peak_wset", memory.rss))
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "peak_rss_bytes": peak_rss,
    }


def _dense_rankings(
    documents: list[CorpusDocument],
    topics: dict[str, str],
    config: RetrievalEvalConfig,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    try:
        np = import_module("numpy")
        sentence_transformers = import_module("sentence_transformers")
    except ImportError as error:
        raise RuntimeError(
            "Install retrieval dependencies with `uv sync --extra retrieval`"
        ) from error
    model = sentence_transformers.SentenceTransformer(
        config.embedding_model_id, device=config.embedding_device
    )
    document_texts = [
        f"title: {document.title or 'none'} | text: {document.text}" for document in documents
    ]
    started = time.perf_counter()
    document_vectors = model.encode_document(
        document_texts,
        batch_size=config.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )[:, : config.embedding_dimension]
    document_vectors /= np.linalg.norm(document_vectors, axis=1, keepdims=True).clip(min=1e-12)
    build_seconds = time.perf_counter() - started
    query_vectors = model.encode_query(
        list(topics.values()),
        batch_size=config.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[:, : config.embedding_dimension]
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True).clip(min=1e-12)
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    docids = [document.docid for document in documents]
    for query_id, vector in zip(topics, query_vectors, strict=True):
        query_started = time.perf_counter()
        scores = document_vectors @ vector
        count = min(config.dense_k, len(scores))
        candidate_indices = (
            np.arange(len(scores))
            if count == len(scores)
            else np.argpartition(scores, -count)[-count:]
        )
        ordered = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        rankings[query_id] = [docids[int(index)] for index in ordered]
        latencies.append(time.perf_counter() - query_started)
    return rankings, {
        "model_id": config.embedding_model_id,
        "dimension": config.embedding_dimension,
        "embedding_build_seconds": build_seconds,
        "documents_per_second": len(documents) / build_seconds,
        "embedding_bytes": int(document_vectors.nbytes),
        "latency": _latency_summary(latencies),
    }


def run(config: RetrievalEvalConfig) -> dict[str, Any]:
    corpus_paths = sorted(Path(path) for path in glob.glob(config.corpus_glob))
    if not corpus_paths:
        raise FileNotFoundError(f"corpus glob matched no files: {config.corpus_glob}")
    if config.output_dir.exists():
        raise FileExistsError(f"evaluation output is immutable: {config.output_dir}")
    topics = load_topics(config.topics_path)
    qrels = load_qrels(config.qrels_path)
    topics = {query_id: query for query_id, query in topics.items() if query_id in qrels}
    if config.max_queries:
        topics = dict(list(topics.items())[: config.max_queries])
    if not topics:
        raise ValueError("no topics with positive qrels are available")
    relevant_docids = set().union(*(qrels[query_id] for query_id in topics))
    documents, corpus_stats = select_pilot_corpus(
        iter_corpus(corpus_paths),
        relevant_docids=relevant_docids,
        max_background_documents=config.max_background_documents,
    )
    available_docids = {document.docid for document in documents}
    available_qrels = {
        query_id: qrels[query_id].intersection(available_docids)
        for query_id in topics
        if qrels[query_id].intersection(available_docids)
    }
    topics = {query_id: topics[query_id] for query_id in available_qrels}
    if not topics:
        raise ValueError("the selected corpus contains no relevant documents")
    config.output_dir.mkdir(parents=True)
    lexical = LexicalIndex(config.output_dir / "lexical.sqlite3")
    lexical_started = time.perf_counter()
    lexical.build(documents)
    lexical_build_seconds = time.perf_counter() - lexical_started
    lexical_rankings: dict[str, list[str]] = {}
    lexical_latencies: list[float] = []
    for query_id, query in topics.items():
        started = time.perf_counter()
        lexical_rankings[query_id] = lexical.search(query, config.lexical_k)
        lexical_latencies.append(time.perf_counter() - started)
    mode_results: dict[str, Any] = {
        "lexical": {
            "metrics": retrieval_metrics(lexical_rankings, available_qrels),
            "latency": _latency_summary(lexical_latencies),
            "build_seconds": lexical_build_seconds,
            "index_bytes": (config.output_dir / "lexical.sqlite3").stat().st_size,
        }
    }
    all_rankings: dict[str, dict[str, list[str]]] = {"lexical": lexical_rankings}
    if config.run_dense:
        dense_rankings, dense_resources = _dense_rankings(documents, topics, config)
        hybrid_rankings = {
            query_id: reciprocal_rank_fusion(
                lexical_rankings[query_id],
                dense_rankings[query_id],
                rrf_k=config.rrf_k,
                limit=config.final_k,
            )
            for query_id in topics
        }
        mode_results["dense"] = {
            "metrics": retrieval_metrics(dense_rankings, available_qrels),
            **dense_resources,
        }
        mode_results["hybrid"] = {"metrics": retrieval_metrics(hybrid_rankings, available_qrels)}
        all_rankings |= {"dense": dense_rankings, "hybrid": hybrid_rankings}
    summary = {
        "scope": (
            "conditional_pilot" if config.max_background_documents is not None else "full_corpus"
        ),
        "queries": len(topics),
        "corpus": corpus_stats,
        "provenance": {
            "git_commit": _git_commit(),
            "seed": config.seed,
            "input_sha256": {
                str(path): _sha256(path)
                for path in [config.topics_path, config.qrels_path, *corpus_paths]
            },
            "environment": _environment_metadata(),
        },
        "modes": mode_results,
    }
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (config.output_dir / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for mode, rankings in all_rankings.items():
            for query_id, ranking in rankings.items():
                handle.write(
                    json.dumps(
                        {"mode": mode, "query_id": query_id, "ranking": ranking},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical/dense/hybrid MIRACL retrieval")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/retrieval/e02_miracl_pilot.yaml")
    )
    args = parser.parse_args()
    run(RetrievalEvalConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
