#!/usr/bin/env bash
#SBATCH --job-name=kd_build_caches
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Input:   $IN"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

# Fail fast if input is missing/empty
if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty." >&2
  exit 2
fi

# Make sure output dirs exist
mkdir -p data/topk_k16 data/fb_hints_L22 data/relb_embeds

# ---- RB top-k caches
python teacher_farm/make_topk_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/topk_k16/ \
  --k 16 \
  --dtype float16

# ---- FB hidden-state caches (e.g., teacher layer 22)
python teacher_farm/make_hidden_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/fb_hints_L22/ \
  --layers 22 \
  --batch_size 1 \
  --max_length 2048 \
  --dtype bfloat16 \
  --flush_every 256


# ---- RelB pooled embedding caches
python teacher_farm/make_embed_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/relb_embeds/

echo "[INFO] Cache build complete"
