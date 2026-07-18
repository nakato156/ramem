from __future__ import annotations

from ramem.config import ContextConfig
from ramem.domain.models import Candidate, PackedContext


class BoundedContextPacker:
    def __init__(self, config: ContextConfig) -> None:
        self.config = config

    def pack(self, query: str, candidates: tuple[Candidate, ...]) -> PackedContext:
        del query
        remaining_chars = self.config.token_budget * self.config.chars_per_token
        selected: list[Candidate] = []
        blocks: list[str] = []
        for number, candidate in enumerate(candidates, start=1):
            evidence_id = f"D{number}"
            prefix = f"[{evidence_id}] {candidate.title}\n"
            allowance = remaining_chars - len(prefix)
            if allowance <= 0:
                break
            excerpt = candidate.text[:allowance]
            if not excerpt:
                break
            selected_candidate = candidate.model_copy(
                update={"evidence_id": evidence_id, "text": excerpt, "end_offset": len(excerpt)}
            )
            block = prefix + excerpt
            selected.append(selected_candidate)
            blocks.append(block)
            remaining_chars -= len(block)
        packed = "\n\n".join(blocks)
        estimated = (len(packed) + self.config.chars_per_token - 1) // self.config.chars_per_token
        return PackedContext(
            text=packed,
            evidence=tuple(selected),
            estimated_tokens=estimated,
            token_budget=self.config.token_budget,
        )
