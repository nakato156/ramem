from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from ramem.config import ContextConfig
from ramem.conversation.models import (
    ContextBundle,
    ConversationMessage,
    RetrievedMemory,
)
from ramem.conversation.store import ConversationStore
from ramem.memory.retriever import ConversationalRetriever


class ConversationalContextBuilder:
    def __init__(
        self,
        store: ConversationStore,
        retriever: ConversationalRetriever,
        config: ContextConfig,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.config = config
        self._token_counter: Callable[[str], int] | None = None
        self._token_truncator: Callable[[str, int, bool], str] | None = None

    def use_tokenizer(
        self,
        counter: Callable[[str], int],
        truncator: Callable[[str, int, bool], str],
    ) -> None:
        self._token_counter = counter
        self._token_truncator = truncator

    def build(self, session_id: UUID | str, query: str) -> ContextBundle:
        messages = self.store.messages(session_id)
        recent = self._recent_window(messages)
        previous_user = [item.content for item in messages[:-1] if item.role == "user"][-2:]
        retrieval_query = "\n".join([*previous_user, query])
        recent_ids = {item.message_id for item in recent}
        excluded = {
            str(chunk.chunk_id)
            for chunk in self.store.all_chunks()
            if any(message_id in recent_ids for message_id in chunk.message_ids)
        }
        memories = self.retriever.retrieve(retrieval_query, excluded_chunk_ids=excluded)
        selected, rendered, tokens = self._pack_memories(memories)
        return ContextBundle(
            query=query,
            recent_messages=recent,
            memories=selected,
            rendered_context=rendered,
            memory_tokens=tokens,
            recent_tokens=sum(self._tokens(item.content) for item in recent),
            token_budget=self.config.token_budget,
        )

    def _recent_window(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> tuple[ConversationMessage, ...]:
        selected: list[ConversationMessage] = []
        used = 0
        for message in reversed(messages):
            tokens = self._tokens(message.content)
            remaining = self.config.recent_token_budget - used
            if remaining <= 0:
                break
            if tokens > remaining:
                excerpt = self._truncate(message.content, remaining, keep_end=True)
                selected.append(message.model_copy(update={"content": excerpt}))
                used += self._tokens(excerpt)
                break
            selected.append(message)
            used += tokens
        return tuple(reversed(selected))

    def _pack_memories(
        self,
        memories: tuple[RetrievedMemory, ...],
    ) -> tuple[tuple[RetrievedMemory, ...], str, int]:
        selected: list[RetrievedMemory] = []
        blocks: list[str] = []
        used = 0
        for number, memory in enumerate(memories, start=1):
            session = self.store.get_session(memory.chunk.session_id)
            title = session.title if session else str(memory.chunk.session_id)
            prefix = f"[D{number}] Sesión: {title}; fecha: {memory.chunk.created_at.isoformat()}\n"
            remaining = self.config.token_budget - used - self._tokens(prefix)
            if remaining <= 0:
                break
            excerpt = self._truncate(memory.chunk.text, remaining, keep_end=False)
            if not excerpt:
                continue
            evidence = memory.model_copy(
                update={
                    "evidence_id": f"D{number}",
                    "chunk": memory.chunk.model_copy(
                        update={
                            "text": excerpt,
                            "end_offset": memory.chunk.start_offset + len(excerpt),
                        }
                    ),
                }
            )
            block = prefix + excerpt
            selected.append(evidence)
            blocks.append(block)
            used += self._tokens(block)
        return tuple(selected), "\n\n".join(blocks), used

    def _tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return max(1, self._token_counter(text))
        return max(1, (len(text) + self.config.chars_per_token - 1) // self.config.chars_per_token)

    def _truncate(self, text: str, max_tokens: int, *, keep_end: bool) -> str:
        if self._token_truncator is not None:
            return self._token_truncator(text, max_tokens, keep_end)
        characters = max_tokens * self.config.chars_per_token
        return text[-characters:] if keep_end else text[:characters]
