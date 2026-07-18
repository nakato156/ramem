#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

LOG_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/training"
LOG_FILE="$LOG_DIR/t4-train.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] FAILED with exit code $exit_code"
  exit "$exit_code"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting budgeted Lightning T4 training"
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

uv sync --extra dev --extra training --locked
uv run ramem-train --config configs/training/gemma_1b_t4_qlora.yaml

echo "[$(date --iso-8601=seconds)] COMPLETED budgeted Lightning T4 training"
