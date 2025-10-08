#!/usr/bin/env bash
#SBATCH --job-name=kd_build_shards
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh   # creates/activates env & installs reqs

# Inputs (override with --export)
HF_DATASETS="${HF_DATASETS:-teknium/OpenHermes-2.5}"
SPLIT="${SPLIT:-train}"
WEIGHTS="${WEIGHTS:-}"               # e.g. "0.8,0.2" if multiple datasets
MAX_SAMPLES="${MAX_SAMPLES:-0}"      # 0 = no cap
OUT="${OUT:-data/shards.jsonl}"
STREAMING="${STREAMING:-1}"          # 1 = streaming, 0 = non-streaming
DATA_DIR="${DATA_DIR:-}"             # for offline local cache: set to your path

# Log
echo "[INFO] Datasets: $HF_DATASETS"
echo "[INFO] Split:    $SPLIT"
echo "[INFO] Weights:  ${WEIGHTS:-<uniform>}"
echo "[INFO] Max:      ${MAX_SAMPLES}"
echo "[INFO] Out:      $OUT"
echo "[INFO] Streaming:${STREAMING}"
echo "[INFO] Data dir: ${DATA_DIR:-<none>}"

# Build args
DATASET_ARGS=()
IFS=',' read -ra DS_ARR <<< "$HF_DATASETS"
for d in "${DS_ARR[@]}"; do
  DATASET_ARGS+=(--dataset "$d")
done

[[ "$STREAMING" == "1" ]] && STREAM_FLAG="--streaming" || STREAM_FLAG="--no-streaming"
[[ "$MAX_SAMPLES" == "0" ]] && MAX_ARG=() || MAX_ARG=(--max_samples "$MAX_SAMPLES")
[[ -n "$WEIGHTS" ]] && W_ARG=(--weights "$WEIGHTS") || W_ARG=()
[[ -n "$DATA_DIR" ]] && DD_ARG=(--data_dir "$DATA_DIR") || DD_ARG=()

mkdir -p data
python data/build_shards_from_hf.py \
  "${DATASET_ARGS[@]}" \
  --split "$SPLIT" \
  "${W_ARG[@]}" \
  "${MAX_ARG[@]}" \
  --out "$OUT" \
  $STREAM_FLAG \
  "${DD_ARG[@]}"

echo "[INFO] Wrote $(wc -l < "$OUT") lines to $OUT"
