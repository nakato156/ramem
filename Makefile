.PHONY: install install-training lint typecheck test check download prepare smoke train

install:
	uv sync --extra dev

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

download:
	uv run ramem-download --dataset squad-es

prepare:
	uv run ramem-prepare

smoke:
	uv run ramem-train --config configs/training/gemma_1b_smoke_qlora.yaml --max-samples 64

train:
	uv run ramem-train --config configs/training/gemma_1b_t4_qlora.yaml
