from collections.abc import Callable, Sequence
from pathlib import Path

from ramem.config import (
    AppConfig,
    ContextConfig,
    GenerationConfig,
    RetrievalConfig,
    StorageConfig,
    TelemetryConfig,
)
from ramem.conversation.service import ConversationService
from ramem.generation.backends import BackendInfo, GenerationCancelled
from ramem.memory.embeddings import HashingEmbedder


class FakeBackend:
    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            backend="fake",
            model="tests",
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
        del cancelled
        answer = "Respuesta de prueba [D1]" if "[D1]" in messages[-1]["content"] else "Guardado."
        if on_token:
            for token in answer.split(" "):
                on_token(token + " ")
        return answer


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        retrieval=RetrievalConfig(
            mode="hybrid",
            lexical_k=10,
            dense_k=10,
            final_k=5,
            embedding_provider="hashing",
            embedding_dimension=64,
        ),
        context=ContextConfig(),
        generation=GenerationConfig(),
        storage=StorageConfig(
            index_path=tmp_path / "ramem.sqlite3",
            lance_path=tmp_path / "memory.lance",
        ),
        telemetry=TelemetryConfig(traces_dir=tmp_path / "traces"),
    )


def test_chat_indexes_one_session_and_recalls_it_from_another(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = ConversationService(
        config,
        backend=FakeBackend(),
        embedder=HashingEmbedder(64),
    )
    first = service.open_session(new=True, title="Preferencias")
    service.ask(first.session_id, "Mi color favorito es verde")

    assert service.store.message_count(first.session_id) == 2
    assert service.index.count() == 1

    second = service.open_session(new=True, title="Consulta")
    result = service.ask(second.session_id, "¿Cuál es mi color favorito?")

    assert result.context.memories
    assert result.context.memories[0].chunk.session_id == first.session_id
    assert "[D1]" in result.answer
    assert result.cited_evidence_ids == ("D1",)
    assert service.store.message_count(second.session_id) == 2


def test_lancedb_can_be_rebuilt_from_sqlite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = ConversationService(
        config,
        backend=FakeBackend(),
        embedder=HashingEmbedder(64),
    )
    session = service.open_session(new=True)
    service.ask(session.session_id, "Trabajo en un proyecto llamado RAMEM")
    assert service.index.count() == 1

    service.index.clear()
    assert service.index.count() == 0
    assert service.rebuild_index() == 1
    assert service.index.count() == 1
    assert service.search("proyecto RAMEM")


def test_recent_chunk_is_excluded_from_same_turn_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = ConversationService(
        config,
        backend=FakeBackend(),
        embedder=HashingEmbedder(64),
    )
    session = service.open_session(new=True)
    service.ask(session.session_id, "Esta conversación acaba de comenzar")
    result = service.ask(session.session_id, "Continúa")

    assert result.context.memories == ()


class InvalidCitationBackend(FakeBackend):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        del messages, on_token, cancelled
        return "Respuesta [D999]"


class CancelledBackend(FakeBackend):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        on_token: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        del messages, on_token, cancelled
        raise GenerationCancelled("cancelled")


def test_invalid_citations_are_removed_and_trace_contains_no_content(tmp_path: Path) -> None:
    service = ConversationService(
        _config(tmp_path),
        backend=InvalidCitationBackend(),
        embedder=HashingEmbedder(64),
    )
    session = service.open_session(new=True)

    result = service.ask(session.session_id, "Texto ultrasecreto")

    assert result.answer == "Respuesta"
    assert result.invalid_citation_ids == ("D999",)
    trace = next((tmp_path / "traces").glob("*.json")).read_text(encoding="utf-8")
    assert "Texto ultrasecreto" not in trace


def test_cancelled_generation_keeps_only_confirmed_user_message(tmp_path: Path) -> None:
    service = ConversationService(
        _config(tmp_path),
        backend=CancelledBackend(),
        embedder=HashingEmbedder(64),
    )
    session = service.open_session(new=True)

    try:
        service.ask(session.session_id, "Mensaje confirmado")
    except GenerationCancelled:
        pass
    else:
        raise AssertionError("expected cancellation")

    messages = service.store.messages(session.session_id)
    assert len(messages) == 1
    assert messages[0].role == "user"


def test_session_and_global_deletion_remove_canonical_and_rebuildable_data(
    tmp_path: Path,
) -> None:
    service = ConversationService(
        _config(tmp_path),
        backend=FakeBackend(),
        embedder=HashingEmbedder(64),
    )
    first = service.open_session(new=True, title="Primera")
    second = service.open_session(new=True, title="Segunda")
    service.ask(first.session_id, "Mi ciudad preferida es Arequipa")
    service.ask(second.session_id, "Mi bebida preferida es café")

    first_chunk_ids = tuple(
        str(chunk.chunk_id)
        for chunk in service.store.all_chunks()
        if chunk.session_id == first.session_id
    )
    service.delete_session(first.session_id)

    assert service.store.get_session(first.session_id) is None
    assert not service.index.contains_chunks(first_chunk_ids)
    assert all(memory.chunk.session_id != first.session_id for memory in service.search("Arequipa"))
    assert service.store.get_session(second.session_id) is not None

    service.delete_all()

    assert service.store.stats().sessions == 0
    assert service.store.stats().messages == 0
    assert service.store.stats().chunks == 0
    assert service.store.pending_jobs() == ()
    assert service.index.count() == 0
