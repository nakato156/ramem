from __future__ import annotations

from ramem.domain.models import GeneratedAnswer, PackedContext, SupportStatus, VerificationResult


class CitationVerifier:
    def verify(self, answer: GeneratedAnswer, context: PackedContext) -> VerificationResult:
        if answer.abstained:
            return VerificationResult(
                status=SupportStatus.SUPPORTED,
                valid_citations=True,
                citation_precision=1.0,
                reason="explicit_abstention",
            )
        allowed = {item.evidence_id for item in context.evidence}
        cited = {citation.evidence_id for citation in answer.citations}
        valid_count = len(cited.intersection(allowed))
        precision = valid_count / len(cited) if cited else 0.0
        valid = bool(cited) and cited.issubset(allowed)
        return VerificationResult(
            status=SupportStatus.SUPPORTED if valid else SupportStatus.UNSUPPORTED,
            valid_citations=valid,
            citation_precision=precision,
            reason="citations_present_in_context" if valid else "missing_or_invalid_citation",
        )
