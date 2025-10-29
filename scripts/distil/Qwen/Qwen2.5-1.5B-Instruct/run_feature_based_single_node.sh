#!/usr/bin/env bash
#SBATCH --job-name=kd_feature_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/Qwen/2.5-1.5B-Instruct/feature/%x_%j.out
#SBATCH --error=logs/Qwen/2.5-1.5B-Instruct/feature/%x_%j.err


set -euo pipefail
source scripts/_env_single_node.sh

# ---- Shared knobs ----
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
FB_TEACHER_LAYER=${FB_TEACHER_LAYER:-22}
FB_STUDENT_LAYER=${FB_STUDENT_LAYER:-12}
FB_TOKEN_SUBSET=${FB_TOKEN_SUBSET:-0.25}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-2}      # per-rank
MAX_STEPS=${MAX_STEPS:-0}
SAVE_EVERY=${SAVE_EVERY:-1000}
SQ_LEN=${SEQ_LEN:-1024}  # 0 means no truncation
# DS_CFG=${DS_CFG:-configs/ds_zero3.json}  # only if you truly use DS in Python

# ---- Preemption trap ----
cleanup_and_requeue() {
  echo "[WARN] SIGUSR1; graceful stop & requeue…"
  pkill -SIGTERM -P $$ python || true
  sleep 12
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap cleanup_and_requeue SIGUSR1

# ---- Per-task hygiene ----
SEED=$(( 1337 + ${SLURM_ARRAY_TASK_ID:-0} ))
MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID} % 1000) + ${SLURM_ARRAY_TASK_ID:-0} ))

echo "[INFO] Feature KD | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES | seed=$SEED | port=$MASTER_PORT | student=$STUDENT"

# ---- Telemetry ----
mkdir -p "logs/telemetry/Qwen/2.5-1.5B-Instruct/feature/$SLURM_JOB_ID"
python monitor.py --output "logs/telemetry/Qwen/2.5-1.5B-Instruct/feature/$SLURM_JOB_ID/${HOSTNAME}.jsonl" --interval 1 &
MON_PID=$!

# ---- Data dir (match teacher layer) ----
FB_DATA_DIR="data/fb_hints_L${FB_TEACHER_LAYER}"

# ---- Run dir (stable across requeues) ----
RUN_DIR="serialization_dir/Qwen/2.5-1.5B-Instruct/feature/$SLURM_JOB_ID/FB_TLayer${FB_TEACHER_LAYER}_SLayer${FB_STUDENT_LAYER}_TSR${FB_TOKEN_SUBSET}"
mkdir -p "$RUN_DIR"

# ---- Data guard ----
shopt -s nullglob
FB_FILES=(${FB_DATA_DIR}/*.parquet)
[[ ${#FB_FILES[@]} -gt 0 ]] || { echo "[ERROR] no fb_hints parquet files at ${FB_DATA_DIR}/*.parquet"; exit 2; }

# ---- Launch ----
accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  --module kd.train \
    --kd.mode fb \
    --student "${STUDENT}" \
    --data "${FB_DATA_DIR}/*.parquet" \
    --fb.teacher_layer "${FB_TEACHER_LAYER}" \
    --fb.student_layer "${FB_STUDENT_LAYER}" \
    --fb.token_subset_ratio "${FB_TOKEN_SUBSET}" \
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
echo "[INFO] FB KD complete"
