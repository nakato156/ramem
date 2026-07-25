from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from ramem.config import GenerationConfig

TokenCallback = Callable[[str], None]


class GenerationCancelled(RuntimeError):
    """Raised when the caller cancels only the active generation."""


class BackendInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    model: str
    quantization: str
    device: str


class GeneratorBackend(Protocol):
    @property
    def info(self) -> BackendInfo: ...

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: TokenCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def truncate_tokens(self, text: str, max_tokens: int, keep_end: bool) -> str: ...


class LlamaCppBackend:
    def __init__(self, model_path: Path, config: GenerationConfig) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Instala el backend GGUF con: uv sync --extra llama") from error
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el GGUF en {model_path}. Ejecuta `ramem model pull`."
            )
        self.model_path = model_path
        self.config = config
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=config.context_length,
            n_gpu_layers=config.gpu_layers,
            seed=config.seed,
            verbose=False,
        )

    @property
    def info(self) -> BackendInfo:
        device = "GPU" if self.config.gpu_layers != 0 else "CPU"
        return BackendInfo(
            backend="llama_cpp",
            model=str(self.model_path),
            quantization="GGUF Q4_K_M",
            device=device,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: TokenCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        stream = self._llm.create_chat_completion(
            messages=list(messages),
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        )
        parts: list[str] = []
        for event in stream:
            if cancelled and cancelled():
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
                raise GenerationCancelled("generation cancelled")
            choices = event.get("choices", [])
            token = ""
            if choices:
                token = str(choices[0].get("delta", {}).get("content") or "")
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
        return "".join(parts).strip()

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode(), add_bos=False, special=True))

    def truncate_tokens(self, text: str, max_tokens: int, keep_end: bool) -> str:
        tokens = self._llm.tokenize(text.encode(), add_bos=False, special=True)
        selected = tokens[-max_tokens:] if keep_end else tokens[:max_tokens]
        return self._llm.detokenize(selected).decode(errors="replace")


class TransformersBackend:
    def __init__(self, model_path: str | Path, config: GenerationConfig) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Instala el backend Transformers con: uv sync --extra transformers"
            ) from error
        self.torch = torch
        self.config = config
        self.model_path = str(model_path)
        quantization: Any = None
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("Transformers 4-bit requiere CUDA; usa GGUF para CPU")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        remote = not Path(self.model_path).exists()
        load_options = {"revision": config.model_revision} if remote else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, **load_options)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization,
            dtype="auto",
            **load_options,
        )

    @property
    def info(self) -> BackendInfo:
        device = str(self.model.device)
        return BackendInfo(
            backend="transformers",
            model=self.model_path,
            quantization="NF4" if self.config.load_in_4bit else "native",
            device=device,
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: TokenCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        from transformers import (  # type: ignore[import-not-found]
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        inputs = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors: list[BaseException] = []
        was_cancelled = False

        class CancelWhenRequested(StoppingCriteria):
            def __call__(self, *args: Any, **kwargs: Any) -> bool:
                del args, kwargs
                return bool(cancelled and cancelled())

        def run() -> None:
            try:
                self.torch.manual_seed(self.config.seed)
                self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.temperature > 0,
                    temperature=max(self.config.temperature, 1e-6),
                    top_p=self.config.top_p,
                    stopping_criteria=StoppingCriteriaList([CancelWhenRequested()]),
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        parts: list[str] = []
        for token in streamer:
            if cancelled and cancelled():
                was_cancelled = True
                break
            parts.append(token)
            if on_token:
                on_token(token)
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("Transformers did not stop within 10 seconds after cancellation")
        if errors:
            raise RuntimeError("Transformers generation failed") from errors[0]
        if was_cancelled or (cancelled and cancelled()):
            raise GenerationCancelled("generation cancelled")
        return "".join(parts).strip()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, max_tokens: int, keep_end: bool) -> str:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        selected = tokens[-max_tokens:] if keep_end else tokens[:max_tokens]
        return str(self.tokenizer.decode(selected, skip_special_tokens=True))


def build_generator_backend(
    config: GenerationConfig,
    *,
    model_root: Path,
) -> GeneratorBackend:
    gguf_path = config.gguf_path or model_root / config.gguf_filename
    provider = config.provider
    if provider == "auto":
        provider = "llama_cpp"
    if provider == "llama_cpp":
        return LlamaCppBackend(gguf_path, config)
    if provider == "transformers":
        transformers_path = (
            model_root if (model_root / "config.json").is_file() else config.model_id
        )
        return TransformersBackend(transformers_path, config)
    raise ValueError(f"unknown generation provider {provider!r}")
