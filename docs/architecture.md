# Architecture

RaMem uses an explicit, synchronous state machine. Every stage receives and returns Pydantic
models; component interfaces are Protocols so deterministic baselines can later be replaced by
Gemma-backed adapters without changing the domain layer.

The current flow is: request → route → rewrite → retrieve → RRF → context pack → generate →
verify → memory-write proposal → trace. A failed verification may abstain, but V0 never loops more
than once. Retrieved documents are untrusted data and never become system instructions.

SQLite stores document metadata, FTS5 text, and compact JSON vectors. The dense implementation is
a deterministic hashing-vector baseline for tests and orchestration validation, not the final
EmbeddingGemma implementation.

## Component boundaries

`domain/contracts.py` defines router, rewriter, document-store, retriever, packer, generator,
verifier, memory-writer, evaluator and trace-sink protocols. `ExecutionState` is frozen; the
orchestrator advances it with validated copies rather than mutating shared dictionaries. This makes
each stage replaceable and every intermediate artifact serializable.

The current index stores one real ingested document per retrieval unit. Parent/child chunks, time filters,
episodic rows and ANN indexing are deliberately deferred, but the typed interfaces already carry
offsets, memory filters, source URIs and evidence IDs needed by those adapters.
