from __future__ import annotations

import queue
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ramem.config import GenerationConfig
from ramem.generation import backends
from ramem.generation.backends import (
    GenerationCancelled,
    LlamaCppBackend,
    TransformersBackend,
)


class _FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def create_chat_completion(self, **kwargs: Any) -> Iterator[dict[str, object]]:
        del kwargs
        yield {"choices": [{"delta": {"content": "hola "}}]}
        yield {"choices": [{"delta": {"content": "mundo"}}]}

    def tokenize(self, value: bytes, **kwargs: Any) -> list[int]:
        del kwargs
        return list(value)

    def detokenize(self, values: list[int]) -> bytes:
        return bytes(values)


def test_llama_backend_streams_and_uses_deterministic_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=_FakeLlama))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    received: list[str] = []

    backend = LlamaCppBackend(model, GenerationConfig(seed=73))
    answer = backend.generate(
        [{"role": "user", "content": "Hola"}],
        on_token=received.append,
    )

    assert answer == "hola mundo"
    assert received == ["hola ", "mundo"]
    assert backend._llm.kwargs["seed"] == 73
    assert backend.count_tokens("abc") == 3
    assert backend.truncate_tokens("abcdef", 3, False) == "abc"


def test_llama_backend_cancels_only_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=_FakeLlama))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    backend = LlamaCppBackend(model, GenerationConfig())

    with pytest.raises(GenerationCancelled):
        backend.generate(
            [{"role": "user", "content": "Hola"}],
            cancelled=lambda: True,
        )


class _Inputs(dict[str, object]):
    def to(self, device: object) -> _Inputs:
        del device
        return self


class _Tokenizer:
    def apply_chat_template(self, *args: Any, **kwargs: Any) -> _Inputs:
        del args, kwargs
        return _Inputs(input_ids=[1, 2])

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        del kwargs
        return list(text.encode())

    def decode(self, values: list[int], **kwargs: Any) -> str:
        del kwargs
        return bytes(values).decode()


class _Streamer:
    _end = object()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.values: queue.Queue[object] = queue.Queue()

    def emit(self, value: str) -> None:
        self.values.put(value)

    def finish(self) -> None:
        self.values.put(self._end)

    def __iter__(self) -> Iterator[str]:
        while (value := self.values.get(timeout=5)) is not self._end:
            yield str(value)


class _Model:
    device = "cpu"

    def generate(self, **kwargs: Any) -> None:
        streamer = kwargs["streamer"]
        streamer.emit("respuesta ")
        streamer.emit("transformers")
        streamer.finish()


class _Torch:
    seeds: ClassVar[list[int]] = []

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        cls.seeds.append(seed)


def test_transformers_backend_streaming_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = types.SimpleNamespace(
        StoppingCriteria=object,
        StoppingCriteriaList=list,
        TextIteratorStreamer=_Streamer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    backend = TransformersBackend.__new__(TransformersBackend)
    backend.torch = _Torch
    backend.config = GenerationConfig(provider="transformers", load_in_4bit=False, seed=91)
    backend.model_path = "test"
    backend.tokenizer = _Tokenizer()
    backend.model = _Model()
    received: list[str] = []

    answer = backend.generate(
        [{"role": "user", "content": "Hola"}],
        on_token=received.append,
    )

    assert answer == "respuesta transformers"
    assert received == ["respuesta ", "transformers"]
    assert _Torch.seeds[-1] == 91
    assert backend.count_tokens("abc") == 3
    assert backend.truncate_tokens("abcdef", 3, True) == "def"


def test_auto_backend_is_always_gguf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class _Backend:
        def __init__(self, path: Path, config: GenerationConfig) -> None:
            observed["path"] = path
            observed["provider"] = config.provider

        @property
        def info(self) -> object:
            return object()

        def generate(
            self,
            messages: Sequence[dict[str, str]],
            *,
            on_token: Callable[[str], None] | None = None,
            cancelled: Callable[[], bool] | None = None,
        ) -> str:
            del messages, on_token, cancelled
            return ""

        def count_tokens(self, text: str) -> int:
            return len(text)

        def truncate_tokens(self, text: str, max_tokens: int, keep_end: bool) -> str:
            return text[-max_tokens:] if keep_end else text[:max_tokens]

    monkeypatch.setattr(backends, "LlamaCppBackend", _Backend)
    backend = backends.build_generator_backend(
        GenerationConfig(provider="auto"), model_root=tmp_path
    )

    assert isinstance(backend, _Backend)
    assert observed["path"] == tmp_path / "ramem-gemma-1b-q4_k_m.gguf"
