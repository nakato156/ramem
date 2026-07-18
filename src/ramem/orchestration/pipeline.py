from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from ramem.config import AppConfig
from ramem.domain.contracts import (
    ContextPacker,
    Generator,
    MemoryWriter,
    QueryRewriter,
    Retriever,
    Router,
    TraceSink,
    Verifier,
)
from ramem.domain.models import ExecutionState, QueryRequest, StageTiming, SupportStatus
from ramem.generation.context import BoundedContextPacker
from ramem.generation.transformers import TransformersGenerator
from ramem.memory.policy import NoBlindWritePolicy
from ramem.retrieval.hybrid import HybridRetriever
from ramem.retrieval.store import SQLiteDocumentStore
from ramem.routing.rules import IdentityRewriter, RuleRouter
from ramem.telemetry.trace import JsonTraceSink
from ramem.verification.rules import CitationVerifier


class RaMemPipeline:
    def __init__(
        self,
        *,
        config: AppConfig,
        router: Router,
        rewriter: QueryRewriter,
        retriever: Retriever,
        packer: ContextPacker,
        generator: Generator,
        verifier: Verifier,
        memory_writer: MemoryWriter,
        trace_sink: TraceSink,
    ) -> None:
        self.config = config
        self.router = router
        self.rewriter = rewriter
        self.retriever = retriever
        self.packer = packer
        self.generator = generator
        self.verifier = verifier
        self.memory_writer = memory_writer
        self.trace_sink = trace_sink

    def run(self, request: QueryRequest) -> ExecutionState:
        timings: list[StageTiming] = []

        @contextmanager
        def timed(stage: str) -> Iterator[None]:
            started = perf_counter()
            try:
                yield
            finally:
                timings.append(
                    StageTiming(stage=stage, duration_ms=(perf_counter() - started) * 1000)
                )

        state = ExecutionState(
            session_id=request.session_id,
            user_id_hash=request.user_id_hash,
            query_original=request.query,
            conversation_window=request.conversation_window,
            context_token_budget=self.config.context.token_budget,
            generation_config=self.config.generation.model_dump(mode="json"),
        )
        with timed("route"):
            route = self.router.route(request)
        state = state.model_copy(
            update={"route_labels": route.labels, "route_confidence": route.confidence}
        )
        with timed("rewrite"):
            rewritten = self.rewriter.rewrite(request, route)
        state = state.model_copy(update={"rewritten_queries": rewritten})
        with timed("retrieve"):
            candidates = self.retriever.retrieve(rewritten[0], state.retrieval_filters)
        state = state.model_copy(update={"candidates": candidates})
        with timed("pack"):
            context = self.packer.pack(request.query, candidates)
        state = state.model_copy(update={"selected_evidence": context.evidence})
        with timed("generate"):
            answer = self.generator.generate(request.query, context)
        state = state.model_copy(
            update={"draft_answer": answer.text, "citations": answer.citations}
        )
        with timed("verify"):
            verification = self.verifier.verify(answer, context)
        final_answer = answer.text
        abstained = answer.abstained
        if verification.status is not SupportStatus.SUPPORTED:
            final_answer = "No hay evidencia suficiente para responder de forma sustentada."
            abstained = True
        with timed("memory_write"):
            writes = self.memory_writer.propose(request, answer)
        state = state.model_copy(
            update={
                "verification_result": verification,
                "final_answer": final_answer,
                "memory_write_candidates": writes,
                "trace_timings": tuple(timings),
                "abstained": abstained,
            }
        )
        self.trace_sink.write(state)
        return state


def build_pipeline(config: AppConfig, *, root: Path = Path(".")) -> RaMemPipeline:
    index_path = config.storage.index_path
    traces_dir = config.telemetry.traces_dir
    if not index_path.is_absolute():
        index_path = root / index_path
    if not traces_dir.is_absolute():
        traces_dir = root / traces_dir
    store = SQLiteDocumentStore(index_path, config.retrieval.embedding_dimension)
    return RaMemPipeline(
        config=config,
        router=RuleRouter(),
        rewriter=IdentityRewriter(),
        retriever=HybridRetriever(store, config.retrieval),
        packer=BoundedContextPacker(config.context),
        generator=TransformersGenerator(config.generation),
        verifier=CitationVerifier(),
        memory_writer=NoBlindWritePolicy(),
        trace_sink=JsonTraceSink(traces_dir),
    )
