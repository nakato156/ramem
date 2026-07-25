.PHONY: install install-llama install-transformers install-training lint typecheck test check build benchmark download prepare smoke train

install:
	uv sync --extra dev

install-llama:
	uv sync --extra dev --extra llama

install-transformers:
	uv sync --extra dev --extra transformers

install-training:
	uv sync --extra dev --extra training

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

build: check
	uv build

benchmark:
	uv run ramem benchmark --dataset data/benchmarks/ramem-memory-es-dev.jsonl

download:
	uv run ramem-download --dataset squad-es

prepare:
	uv run ramem-prepare

smoke:
	uv run ramem-train --config configs/training/gemma_1b_smoke_qlora.yaml --max-samples 64

train:
	uv run ramem-train --config configs/training/gemma_1b_t4_qlora.yaml
