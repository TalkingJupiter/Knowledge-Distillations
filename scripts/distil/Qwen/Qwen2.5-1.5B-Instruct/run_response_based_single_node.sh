#!/usr/bin/env bash
#SBATCH --job-name=kd_response_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/Qwen/2.5-1.5B-Instruct/response/%x_%j.out
#SBATCH --error=logs/Qwen/2.5-1.5B-Instruct/response/%x_%j.err


set -euo pipefail
source scripts/_env_single_node.sh

# ---- Shared knobs ----
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
RB_TOPK=${RB_TOPK:-16}
RB_TEMP=${RB_TEMP:-2.0}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-8}      # per-rank batch size
MAX_STEPS=${MAX_STEPS:-0}
SAVE_EVERY=${SAVE_EVERY:-1000}
SQ_LEN=${SEQ_LEN:-1024}  # 0 means no truncation
# DS_CFG=${DS_CFG:-configs/ds_zero3.json}   # only if you really enable DS in Python

# ---- Preemption trap ----
cleanup_and_requeue() {
  echo "[WARN] SIGUSR1; graceful stop & requeue…"
  pkill -SIGTERM -P $$ python || true
  sleep 12
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap cleanup_and_requeue SIGUSR1

# ---- Per-task hygiene ----
SEED=$(( 777 + ${SLURM_ARRAY_TASK_ID:-0} ))
MASTER_PORT=$(( 29700 + (${SLURM_JOB_ID} % 1000) + ${SLURM_ARRAY_TASK_ID:-0} ))

echo "[INFO] RB KD | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES | seed=$SEED | port=$MASTER_PORT | student=$STUDENT"

# ---- Telemetry ----
mkdir -p "logs/telemetry/Qwen/2.5-1.5B-Instruct/response/$SLURM_JOB_ID"
python monitor.py --output "logs/telemetry/Qwen/2.5-1.5B-Instruct/response/$SLURM_JOB_ID/${HOSTNAME}.jsonl" --interval 1 &
MON_PID=$!

# ---- Run dir (stable across requeues) ----
RUN_DIR="serialization_dir/Qwen/2.5-1.5B-Instruct/response/$SLURM_JOB_ID/RB_TOPK${RB_TOPK}_TEMP${RB_TEMP}"  # fixed for resuming from specific run
mkdir -p "$RUN_DIR"

# ---- Data guard ----
shopt -s nullglob
RB_FILES=(data/topk_k16/*.parquet)
[[ ${#RB_FILES[@]} -gt 0 ]] || { echo "[ERROR] no topk_k16 parquet files at data/topk_k16/*.parquet"; exit 2; }

# ---- Launch ----
accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  --module kd.train \
    --kd.mode rb \
    --student "${STUDENT}" \
    --data "data/topk_k16/*.parquet" \
    --rb.topk "${RB_TOPK}" \
    --rb.temperature "${RB_TEMP}" \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SQ_LEN}" \
    --max_steps "${MAX_STEPS}" \
    --save-dir "$RUN_DIR" \
    --save_every "${SAVE_EVERY}" \
    --resume auto


kill "$MON_PID" || true
echo "[INFO] RB KD complete"
