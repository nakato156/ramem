from __future__ import annotations

from typing import Protocol, cast

import numpy as np

from ramem.retrieval.vectors import hashing_vector


class TextEmbedder(Protocol):
    dimension: int
    model_id: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _truncate_and_normalize(vectors: np.ndarray, dimension: int) -> np.ndarray:
    truncated = np.asarray(vectors, dtype=np.float32)[:, :dimension]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / np.clip(norms, 1e-12, None)


class EmbeddingGemmaEmbedder:
    def __init__(
        self,
        model_id: str,
        dimension: int,
        *,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError("EmbeddingGemma requiere: uv sync --extra retrieval") from error
        self.model_id = model_id
        self.dimension = dimension
        self._model = SentenceTransformer(model_id, device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _truncate_and_normalize(np.asarray(vectors), self.dimension)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self._model.encode_query(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = _truncate_and_normalize(np.asarray(vectors), self.dimension)[0]
        return cast(np.ndarray, result)


class HashingEmbedder:
    """Deterministic test/development embedder; never the release default."""

    model_id = "ramem/hashing-test-only"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = [hashing_vector(text, self.dimension) for text in texts]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(hashing_vector(text, self.dimension), dtype=np.float32)


def build_embedder(
    provider: str,
    model_id: str,
    dimension: int,
    *,
    device: str | None = None,
) -> TextEmbedder:
    if provider == "hashing":
        return HashingEmbedder(dimension)
    if provider != "embeddinggemma":
        raise ValueError(f"unknown embedding provider {provider!r}")
    return EmbeddingGemmaEmbedder(model_id, dimension, device=device)
