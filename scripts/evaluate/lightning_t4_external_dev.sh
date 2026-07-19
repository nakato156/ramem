#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

LOG_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/evaluation"
LOG_FILE="$LOG_DIR/t4-external-dev.log"
RAW_DIR="${RAMEM_RAW_DATA_DIR:-data/raw}/mlqa-es-dev"
PROCESSED_DIR="${RAMEM_PROCESSED_DATA_DIR:-data/processed}/mlqa-es-grounded-dev-v1"
OUTPUT_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/evaluation/mlqa-es-external-dev-seed42"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[$(date --iso-8601=seconds)] FAILED with exit code $exit_code"
  exit "$exit_code"
}
trap on_error ERR

echo "[$(date --iso-8601=seconds)] Starting MLQA-es external development evaluation"
git status --short --branch
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
uv sync --extra dev --extra training --locked

if [[ ! -d "$RAW_DIR" ]]; then
  uv run ramem-download --dataset mlqa-es-dev
fi
if [[ ! -d "$PROCESSED_DIR" ]]; then
  uv run ramem-prepare-external
fi
if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite completed external evaluation: $OUTPUT_DIR" >&2
  exit 2
fi

uv run ramem-evaluate --config configs/evaluation/mlqa_es_external_dev.yaml
uv run ramem-analyze-evaluation --config configs/evaluation/mlqa_es_analysis.yaml
echo "[$(date --iso-8601=seconds)] COMPLETED MLQA-es external development evaluation"
