from __future__ import annotations

from ramem.conversation.models import MemoryChunk
from ramem.conversation.store import ConversationStore
from ramem.memory.lance import LanceMemoryIndex


class IndexWorker:
    def __init__(self, store: ConversationStore, index: LanceMemoryIndex) -> None:
        self.store = store
        self.index = index

    def process(self, limit: int = 100) -> tuple[int, int]:
        completed = 0
        failed = 0
        upserts: list[tuple[str, MemoryChunk]] = []
        deletes: list[tuple[str, str]] = []
        for job in self.store.pending_jobs(limit):
            job_id = str(job["job_id"])
            self.store.mark_job_running(job_id)
            operation = str(job["operation"])
            entity_id = str(job["entity_id"])
            if operation == "upsert":
                chunk = self.store.get_chunk(entity_id)
                if chunk is None:
                    self.store.mark_job_completed(job_id)
                    completed += 1
                else:
                    upserts.append((job_id, chunk))
            elif operation == "delete":
                deletes.append((job_id, entity_id))
            else:
                self.store.mark_job_failed(job_id, f"unknown index operation {operation!r}")
                failed += 1

        if upserts:
            try:
                self.index.upsert([chunk for _, chunk in upserts])
            except Exception as error:
                for job_id, _ in upserts:
                    self.store.mark_job_failed(job_id, f"{type(error).__name__}: {error}")
                    failed += 1
            else:
                for job_id, _ in upserts:
                    self.store.mark_job_completed(job_id)
                    completed += 1

        if deletes:
            try:
                self.index.delete_chunks([entity_id for _, entity_id in deletes])
            except Exception as error:
                for job_id, _ in deletes:
                    self.store.mark_job_failed(job_id, f"{type(error).__name__}: {error}")
                    failed += 1
            else:
                for job_id, _ in deletes:
                    self.store.mark_job_completed(job_id)
                    completed += 1
        return completed, failed

    def drain(self, *, batch_size: int = 100) -> tuple[int, int]:
        """Finish all recoverable jobs, including bounded retries after a crash."""
        completed = 0
        while self.store.pending_jobs(limit=1):
            batch_completed, batch_failed = self.process(limit=batch_size)
            completed += batch_completed
            if batch_completed == 0 and batch_failed == 0:
                raise RuntimeError("index worker made no progress")
        return completed, self.store.stats().failed_jobs

    def rebuild(self) -> int:
        self.index.clear()
        chunks = list(self.store.all_chunks())
        self.store.reset_index_jobs()
        _, failed = self.drain(batch_size=max(100, len(chunks) + 1))
        if failed:
            raise RuntimeError(f"failed to rebuild {failed} memory index jobs")
        indexed = self.index.count()
        if indexed != len(chunks):
            raise RuntimeError(
                f"index rebuild count mismatch: SQLite={len(chunks)}, index={indexed}"
            )
        return indexed
