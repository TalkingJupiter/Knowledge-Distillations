#!/usr/bin/env bash
#SBATCH --job-name=relb_build_caches
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/cache/relb/%x_%j.out
#SBATCH --error=logs/cache/relb/%x_%j.err

set -euo pipefail
# set -x   # <- uncomment for very chatty bash debug

# ---------- Environment ----------
source scripts/_env_single_node.sh

# Hugging Face caches on fast storage
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/models}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# ONLINE vs OFFLINE toggle (keep =1 once the model is cached)
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # faster downloads when OFFLINE=0
export PYTHONUNBUFFERED=1                                             # streaming logs from Python

# NCCL/torch sane defaults for H100
export NCCL_P2P_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

mkdir -p logs "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ---------- Inputs (override via --export=ALL,VAR=...) ----------
IN=${IN:-data/shards_research_10M.jsonl}   # .jsonl or .jsonl.gz
TEACHER=${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}
DEVICE_MAP=${DEVICE_MAP:-auto}                # pass to all stages


# ---------- RelB pooled embedding caches ----------
RELB_DIR=${RELB_DIR:-data/relb_embeds}
RELB_DTYPE=${RELB_DTYPE:-float16}
RELB_BATCH=${RELB_BATCH:-4}
RELB_MAXLEN=${RELB_MAXLEN:-8192}
RELB_POOLING=${RELB_POOLING:-mean}   # mean | cls_last | last_token
RELB_NORMALIZE=${RELB_NORMALIZE:-1}  # 1 => --normalize
RELB_SHARD_SIZE=${RELB_SHARD_SIZE:-4096}
RELB_STEM=${RELB_STEM:-relb_embeds}
RELB_RESUME=${RELB_RESUME:-1}

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Input:   $IN"
echo "[INFO] GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] OFFLINE: ${HF_DATASETS_OFFLINE}"

# ---------- Preemption trap (SIGUSR1) ----------
cleanup_and_requeue() {
  echo "[WARN] SIGUSR1 received; graceful stop & requeue…"
  pkill -SIGTERM -P $$ python || true
  sleep 10
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap cleanup_and_requeue SIGUSR1

# ---------- Input checks ----------
if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input '$IN' not found or empty." >&2
  exit 2
fi

# ---------- Prepare output dirs ----------
mkdir -p "$RELB_DIR"

# ---------- Diagnostics ----------
echo "[INFO] nvidia-smi:"
nvidia-smi || true


# ---------- RelB pooled embedding caches ----------
python teacher_farm/make_embed_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir "$RELB_DIR" \
  --batch_size "$RELB_BATCH" \
  --max_length "$RELB_MAXLEN" \
  --dtype "$RELB_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --pooling "$RELB_POOLING" \
  $( [[ "$RELB_NORMALIZE" == "1" ]] && echo --normalize ) \
  --shard_size "$RELB_SHARD_SIZE" \
  --stem "$RELB_STEM" \
  $( [[ "$RELB_RESUME" == "1" ]] && echo --resume )

echo "[INFO] RB Cache build complete"
