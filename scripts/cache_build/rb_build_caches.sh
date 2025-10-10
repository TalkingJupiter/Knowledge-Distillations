#!/usr/bin/env bash
#SBATCH --job-name=rb_build_caches
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/cache/rb/%x_%j.out
#SBATCH --error=logs/cache/rb/%x_%j.err

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

# ---------- RB top-k caches ----------
TOPK_DIR=${TOPK_DIR:-data/topk_k16}
TOPK_K=${TOPK_K:-16}
TOPK_DTYPE=${TOPK_DTYPE:-float16}
TOPK_BATCH=${TOPK_BATCH:-1}
TOPK_MAXLEN=${TOPK_MAXLEN:-8192}
TOPK_SHARD_SIZE=${TOPK_SHARD_SIZE:-128}
TOPK_STEM=${TOPK_STEM:-rb_topk}
TOPK_RESUME=${TOPK_RESUME:-1}


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
mkdir -p "$TOPK_DIR"

# ---------- Diagnostics ----------
echo "[INFO] nvidia-smi:"
nvidia-smi || true

# ---------- RB top-k caches ----------
python teacher_farm/make_topk_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir "$TOPK_DIR" \
  --k "$TOPK_K" \
  --batch_size "$TOPK_BATCH" \
  --max_length "$TOPK_MAXLEN" \
  --dtype "$TOPK_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --shard_size "$TOPK_SHARD_SIZE" \
  --stem "$TOPK_STEM" \
  $( [[ "$TOPK_RESUME" == "1" ]] && echo --resume )


echo "[INFO] RB Cache build complete"
