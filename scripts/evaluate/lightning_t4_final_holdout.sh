#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if [[ "${RAMEM_RELEASE_CANDIDATE_FROZEN:-}" != "yes" ]]; then
  echo "Final holdout is locked. Set RAMEM_RELEASE_CANDIDATE_FROZEN=yes only after freezing." >&2
  exit 2
fi

LOG_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/evaluation"
LOG_FILE="$LOG_DIR/t4-final-holdout.log"
RAW_DIR="${RAMEM_RAW_DATA_DIR:-data/raw}/xquad-es-final"
PROCESSED_DIR="${RAMEM_PROCESSED_DATA_DIR:-data/processed}/xquad-es-grounded-final-v1"
OUTPUT_DIR="${RAMEM_ARTIFACTS_DIR:-artifacts}/evaluation/xquad-es-final-seed42"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date --iso-8601=seconds)] Starting ONE-TIME XQuAD-es final holdout"
uv sync --extra dev --extra training --locked
if [[ ! -d "$RAW_DIR" ]]; then
  uv run ramem-download --dataset xquad-es-final --release-test
fi
if [[ ! -d "$PROCESSED_DIR" ]]; then
  uv run ramem-prepare-external \
    --source "$RAW_DIR" \
    --output "$PROCESSED_DIR" \
    --purpose external_final_evaluation
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite or repeat final holdout: $OUTPUT_DIR" >&2
  exit 2
fi
uv run ramem-evaluate --config configs/evaluation/xquad_es_final.yaml
echo "[$(date --iso-8601=seconds)] COMPLETED ONE-TIME XQuAD-es final holdout"
