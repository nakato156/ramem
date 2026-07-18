from __future__ import annotations

import re

from ramem.domain.models import QueryRequest, RouteLabel, RouteResult

WEB_TERMS = re.compile(r"\b(hoy|actual|actualmente|últim[oa]s?|precio|clima|noticias)\b", re.I)
MEMORY_TERMS = re.compile(r"\b(recuerdas?|dije|preferencia|mi decisión|antes)\b", re.I)
KNOWLEDGE_TERMS = re.compile(
    r"\b(documento|archivo|manual|fuente|ramem|dónde|qué|cómo|cuándo)\b", re.I
)


class RuleRouter:
    def route(self, request: QueryRequest) -> RouteResult:
        labels: list[RouteLabel] = []
        query = request.query.strip()
        if len(query.split()) < 2:
            return RouteResult(
                labels=(RouteLabel.CLARIFY,), confidence=0.85, reason="query_too_short"
            )
        if WEB_TERMS.search(query):
            labels.append(RouteLabel.WEB)
        if MEMORY_TERMS.search(query):
            labels.append(RouteLabel.EPISODIC)
        if KNOWLEDGE_TERMS.search(query) or not labels:
            labels.append(RouteLabel.KNOWLEDGE)
        if len(labels) > 1:
            labels.append(RouteLabel.HYBRID)
        return RouteResult(labels=tuple(labels), confidence=0.72, reason="deterministic_rules")


class IdentityRewriter:
    def rewrite(self, request: QueryRequest, route: RouteResult) -> tuple[str, ...]:
        del route
        return (" ".join(request.query.split()),)
