# RaMem V0

RaMem is a local-first, typed memory and retrieval system for small language models. The repository
contains real Hugging Face download, preparation, QLoRA training and Transformers inference paths.
No synthetic dataset, fake generator or model mock is part of the executable workflow.

Data and checkpoints are not committed. A Lightning AI Studio downloads them directly from their
official sources, resolves immutable repository revisions, validates the declared license and writes
local SHA-256 manifests.

## Local quality gate

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run ramem doctor
```

## Real cloud workflow

Follow [LIGHTNING_RUNBOOK.md](docs/LIGHTNING_RUNBOOK.md) exactly. The short path is:

```bash
uv sync --extra dev --extra training
uv run ramem-download --dataset squad-es
uv run ramem-prepare
# Switch the Lightning Studio from CPU to a T4 before this command:
uv run ramem-train --config configs/training/gemma_1b_smoke_qlora.yaml --max-samples 64
```

Gemma is gated on Hugging Face. Accept Google's usage terms for both model repositories and add
`HF_TOKEN` as a Lightning secret before starting.

## Boundaries

- `data/raw`, `data/processed`, Hugging Face caches and training artifacts are ignored by Git.
- SQuAD-es `train` is the only source used by the first grounded-QA preparation job.
- Its validation subset is selected deterministically from that training split; public benchmark
  validation/test sets are not used for selection.
- MIRACL Spanish annotations and corpus have separate opt-in downloads for E02–E05.
- The router training set does not yet exist. It will not be fabricated implicitly; controller
  training remains blocked until RaMem-Route-ES has reviewed, traceable examples.
