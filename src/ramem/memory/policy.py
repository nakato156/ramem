from __future__ import annotations

from ramem.domain.models import (
    GeneratedAnswer,
    MemoryWriteCandidate,
    QueryRequest,
    WriteDecision,
)


class NoBlindWritePolicy:
    def propose(
        self, request: QueryRequest, answer: GeneratedAnswer
    ) -> tuple[MemoryWriteCandidate, ...]:
        del request, answer
        return (
            MemoryWriteCandidate(
                decision=WriteDecision.IGNORE,
                confidence=1.0,
                reason_code="v0_no_blind_writes",
            ),
        )
