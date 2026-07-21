# MIRACL retrieval evaluation

E02 compares SQLite FTS5 BM25, EmbeddingGemma dense retrieval and reciprocal-rank fusion on a
fixed MIRACL Spanish development slice. Every completed run stores its resolved configuration,
input hashes, Git commit, seed, package versions, memory, timings, rankings and summary.

## CPU baseline

The lexical baseline does not require a GPU or the retrieval dependency group:

```powershell
uv run --with "psutil>=6,<8" ramem-evaluate-retrieval `
  --config configs/retrieval/e02_miracl_pilot_cpu.yaml
```

## Dense and hybrid pilot

The retrieval environment uses Torch 2.6 with CUDA 12.4 because EmbeddingGemma's bidirectional
attention requires Torch 2.6 or newer:

```powershell
uv sync --extra retrieval
uv run --extra retrieval ramem-evaluate-retrieval `
  --config configs/retrieval/e02_miracl_pilot.yaml
```

Do not combine `--extra training` and `--extra retrieval`. Training remains pinned to Torch 2.5.1
with CUDA 12.1 so its proven environment stays reproducible; `uv` declares the extras mutually
exclusive and will reject an accidental combination. The local GTX 1650 configuration uses batch
size 1 to stay within 4 GB of VRAM.

The pilot scans one 500,000-document shard, retains 10,000 deterministic background documents and
all relevant documents found in that shard, and evaluates only queries whose relevant documents are
available. Its metrics validate implementation and relative behavior; they are not the final E02
scientific result. The final comparison must use every pinned MIRACL Spanish corpus shard, preferably
on the Lightning T4, before E03 selects 768, 256 or 128 dimensions.
