#!/usr/bin/env bash
#SBATCH --job-name=fb_build_caches
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/cache/fb/%x_%j.out
#SBATCH --error=logs/cache/fb/%x_%j.err

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


# ---------- FB hidden-state caches ----------
FB_DIR=${FB_DIR:-data/fb_hints_L22}
FB_LAYERS=${FB_LAYERS:-22}           # "22" or "12 22" supported by the script
FB_BATCH_SIZE=${FB_BATCH_SIZE:-1}
FB_MAXLEN=${FB_MAXLEN:-2048}
FB_DTYPE=${FB_DTYPE:-bfloat16}
FB_FLUSH_EVERY=${FB_FLUSH_EVERY:-256}
FB_STEM=${FB_STEM:-fb_hints}
FB_RESUME=${FB_RESUME:-1}


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
mkdir -p "$FB_DIR"

# ---------- Diagnostics ----------
echo "[INFO] nvidia-smi:"
nvidia-smi || true


# ---------- FB hidden-state caches ----------
python teacher_farm/make_hidden_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir "$FB_DIR" \
  --layers $FB_LAYERS \
  --batch_size "$FB_BATCH_SIZE" \
  --max_length "$FB_MAXLEN" \
  --dtype "$FB_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --flush_every "$FB_FLUSH_EVERY" \
  --stem "$FB_STEM" \
  $( [[ "$FB_RESUME" == "1" ]] && echo --resume )


echo "[INFO] RB Cache build complete"
