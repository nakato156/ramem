# ADR 0002: Offline retrieval baseline

Status: accepted

Use SQLite FTS5 plus deterministic hashing vectors and RRF for the first executable baseline. It
runs without model downloads and validates orchestration. EmbeddingGemma and an ARM-suitable ANN
index remain replaceable adapters to be selected through E02-E05.
