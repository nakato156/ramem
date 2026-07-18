#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

LOG_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/training"
LOG_FILE="$LOG_DIR/t4-smoke.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] FAILED with exit code $exit_code"
  exit "$exit_code"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting Lightning T4 smoke workflow"
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

uv sync --extra dev --extra training --locked

RAW_ROOT="${RAMEM_RAW_DATA_DIR:-data/raw}"
PROCESSED_ROOT="${RAMEM_PROCESSED_DATA_DIR:-data/processed}"

if [[ ! -d "$RAW_ROOT/squad-es" ]]; then
  uv run ramem-download --dataset squad-es
else
  echo "Using existing immutable dataset at $RAW_ROOT/squad-es"
fi

if [[ ! -d "$PROCESSED_ROOT/grounded-qa-es-v1" ]]; then
  uv run ramem-prepare
else
  echo "Using existing prepared dataset at $PROCESSED_ROOT/grounded-qa-es-v1"
fi

uv run ramem-train \
  --config configs/training/gemma_1b_smoke_qlora.yaml \
  --max-samples 64

echo "[$(date --iso-8601=seconds)] COMPLETED Lightning T4 smoke workflow"
