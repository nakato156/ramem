#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

LOG_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/export"
LOG_FILE="$LOG_DIR/merge-adapter.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] FAILED with exit code $exit_code"
  exit "$exit_code"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting immutable merged export"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
uv sync --extra dev --extra training --locked
uv run ramem-export --config configs/export/gemma_1b_ramem.yaml
echo "[$(date --iso-8601=seconds)] COMPLETED immutable merged export"
