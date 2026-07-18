from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ramem.config import GenerationConfig
from ramem.domain.models import Citation, GeneratedAnswer, PackedContext

CITATION_PATTERN = re.compile(r"\[(D\d+)\]")


class TransformersGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install the real model runtime with: uv sync --extra training"
            ) from error

        token = __import__("os").environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required to download the gated Gemma checkpoint")
        quantization: Any = None
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit Gemma inference requires a CUDA GPU")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            token=token,
            device_map="auto",
            quantization_config=quantization,
            dtype="auto",
        )
        if config.adapter_path:
            from peft import PeftModel  # type: ignore[import-not-found]

            self.model = PeftModel.from_pretrained(self.model, Path(config.adapter_path))
        self.max_new_tokens = config.max_new_tokens

    def generate(self, query: str, context: PackedContext) -> GeneratedAnswer:
        if not context.evidence:
            return GeneratedAnswer(
                text="No hay evidencia suficiente para responder.", abstained=True
            )
        system = (
            "Responde únicamente con la evidencia proporcionada. Cita cada afirmación verificable "
            "con identificadores como [D1]. Si la evidencia no basta, abstente explícitamente."
        )
        user = f"EVIDENCIA:\n{context.text}\n\nPREGUNTA:\n{query}"
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        by_id = {candidate.evidence_id: candidate for candidate in context.evidence}
        citations: list[Citation] = []
        for evidence_id in dict.fromkeys(CITATION_PATTERN.findall(text)):
            if evidence_id not in by_id:
                continue
            candidate = by_id[evidence_id]
            citations.append(
                Citation(
                    evidence_id=evidence_id,
                    document_id=candidate.document_id,
                    source_uri=candidate.source_uri,
                    start_offset=candidate.start_offset,
                    end_offset=candidate.end_offset,
                )
            )
        abstained = "no hay evidencia suficiente" in text.casefold()
        return GeneratedAnswer(text=text, citations=tuple(citations), abstained=abstained)
