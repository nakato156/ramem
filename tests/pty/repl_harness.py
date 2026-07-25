from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import ramem.cli.app as cli
from ramem.config import load_config
from ramem.conversation.service import ConversationService
from ramem.generation.backends import BackendInfo, GenerationCancelled
from ramem.memory.embeddings import HashingEmbedder


class SlowBackend:
    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="pty-test",
            model="deterministic",
            quantization="none",
            device="cpu",
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        del messages
        parts: list[str] = []
        for _ in range(200):
            if cancelled is not None and cancelled():
                raise GenerationCancelled("cancelled")
            token = "respuesta "
            parts.append(token)
            if on_token is not None:
                on_token(token)
            time.sleep(0.01)
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def truncate_tokens(self, text: str, max_tokens: int, keep_end: bool) -> str:
        characters = max_tokens * 4
        return text[-characters:] if keep_end else text[:characters]


config = load_config(Path(os.environ["RAMEM_PTY_CONFIG"]))
service = ConversationService(
    config,
    backend=SlowBackend(),
    embedder=HashingEmbedder(config.retrieval.embedding_dimension),
)
cli._service = lambda loaded: service
cli.run_repl(config, new=False)
