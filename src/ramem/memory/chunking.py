from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC

from ramem.conversation.models import ConversationMessage, MemoryChunk


class ConversationChunker:
    """Build immutable memory units without cutting ordinary message boundaries."""

    def __init__(self, max_tokens: int = 512, chars_per_token: int = 4) -> None:
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self._token_counter: Callable[[str], int] | None = None

    def use_tokenizer(self, counter: Callable[[str], int]) -> None:
        self._token_counter = counter

    def build_exchange_chunks(
        self,
        messages: Sequence[ConversationMessage],
    ) -> tuple[MemoryChunk, ...]:
        if len(messages) < 2:
            return ()
        user = messages[-2]
        assistant = messages[-1]
        if user.role != "user" or assistant.role != "assistant":
            raise ValueError("an exchange must end with one user and one assistant message")

        overlap = messages[-3] if len(messages) >= 3 else None
        exchange = [user, assistant]
        selected = ([overlap] if overlap is not None else []) + exchange
        rendered = self._render(selected)
        if self._tokens(rendered) <= self.max_tokens:
            return (self._chunk(selected, rendered, start=0, end=len(rendered)),)

        chunks: list[MemoryChunk] = []
        adjacent = overlap
        for message in exchange:
            current = self._render([message])
            with_overlap = self._render([adjacent, message]) if adjacent is not None else current
            if adjacent is not None and self._tokens(with_overlap) <= self.max_tokens:
                chunks.append(
                    self._chunk(
                        [adjacent, message],
                        with_overlap,
                        start=0,
                        end=len(with_overlap),
                    )
                )
            elif self._tokens(current) <= self.max_tokens:
                chunks.append(self._chunk([message], current, start=0, end=len(current)))
            else:
                chunks.extend(self._split_oversized_message(message))
            adjacent = message
        return tuple(chunks)

    def _split_oversized_message(
        self,
        message: ConversationMessage,
    ) -> list[MemoryChunk]:
        label = "Usuario" if message.role == "user" else "Asistente"
        header = f"Fecha: {message.created_at.astimezone(UTC).isoformat()}\n{label}: "
        chunks: list[MemoryChunk] = []
        start = 0
        while start < len(message.content):
            end = self._largest_fitting_end(message.content, start, header)
            excerpt = message.content[start:end]
            text = header + excerpt
            chunks.append(
                self._chunk(
                    [message],
                    text,
                    start=start,
                    end=end,
                )
            )
            start = end
        return chunks

    def _largest_fitting_end(self, content: str, start: int, header: str) -> int:
        low = start + 1
        high = len(content)
        best = low
        while low <= high:
            middle = (low + high) // 2
            if self._tokens(header + content[start:middle]) <= self.max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _chunk(
        self,
        messages: Sequence[ConversationMessage],
        text: str,
        *,
        start: int,
        end: int,
    ) -> MemoryChunk:
        last = messages[-1]
        identity = ",".join(str(item.message_id) for item in messages)
        digest = hashlib.sha256(
            f"{last.session_id}:{identity}:{start}:{end}:{text}".encode()
        ).hexdigest()
        return MemoryChunk(
            session_id=last.session_id,
            message_ids=tuple(item.message_id for item in messages),
            text=text,
            start_offset=start,
            end_offset=end,
            token_count=self._tokens(text),
            created_at=last.created_at.astimezone(UTC),
            content_hash=digest,
        )

    def _tokens(self, text: str) -> int:
        if self._token_counter is not None:
            return max(1, self._token_counter(text))
        return max(1, (len(text) + self.chars_per_token - 1) // self.chars_per_token)

    @staticmethod
    def _render(messages: Sequence[ConversationMessage | None]) -> str:
        labels = {"user": "Usuario", "assistant": "Asistente"}
        present = [message for message in messages if message is not None]
        if not present:
            return ""
        timestamp = present[-1].created_at.astimezone(UTC)
        body = "\n".join(f"{labels[item.role]}: {item.content}" for item in present)
        return f"Fecha: {timestamp.isoformat()}\n{body}"
