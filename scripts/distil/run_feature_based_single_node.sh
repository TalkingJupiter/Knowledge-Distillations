#!/usr/bin/env bash
#SBATCH --job-name=kd_feature_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/feature/%x_%j.out
#SBATCH --error=logs/feature/%x_%j.err
#SBATCH --array=0-4

set -euo pipefail
source scripts/_env_single_node.sh

# ---- Shared knobs ----
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
FB_TEACHER_LAYER=${FB_TEACHER_LAYER:-22}
FB_STUDENT_LAYER=${FB_STUDENT_LAYER:-12}
FB_TOKEN_SUBSET=${FB_TOKEN_SUBSET:-0.25}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_STEPS=${MAX_STEPS:-2000}
SAVE_EVERY=${SAVE_EVERY:-200}
DS_CFG=${DS_CFG:-configs/ds_zero3.json}

# ---- Preemption trap ----
cleanup_and_requeue() {
  echo "[WARN] SIGUSR1; graceful stop & requeue…"
  pkill -SIGTERM -P $$ python || true
  sleep 8
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap cleanup_and_requeue SIGUSR1

# ---- Per-task hygiene ----
SEED=$(( 1337 + ${SLURM_ARRAY_TASK_ID:-0} ))
MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID} % 1000) + ${SLURM_ARRAY_TASK_ID:-0} ))

echo "[INFO] Feature KD | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES | seed=$SEED | port=$MASTER_PORT | student=$STUDENT"

# ---- Telemetry ----
mkdir -p "logs/telemetry/2nd/feature/$SLURM_JOB_ID"
python monitor.py --output "logs/telemetry/2nd/feature/$SLURM_JOB_ID/${HOSTNAME}.jsonl" --interval 1 &
MON_PID=$!

# ---- Run dir ----
RUN_DIR="serialization_dir/feature/$(date +%Y%m%d_%H%M)_FB_1n_task${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$RUN_DIR"

# ---- Data guard ----
shopt -s nullglob
FB_FILES=(data/fb_hints_L22/*.parquet)
[[ ${#FB_FILES[@]} -gt 0 ]] || { echo "[ERROR] no fb_hints parquet files at data/fb_hints_L22/*.parquet"; exit 2; }

# ---- Launch ----
accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --deepspeed_config_file "${DS_CFG}" \
  --main_process_port "${MASTER_PORT}" \
  --mixed_precision bf16 \
  --module kd.train \
    --kd.mode fb \
    --student "${STUDENT}" \
    --data "data/fb_hints_L22/*.parquet" \
    --fb.teacher_layer "${FB_TEACHER_LAYER}" \
    --fb.student_layer "${FB_STUDENT_LAYER}" \
    --fb.token_subset_ratio "${FB_TOKEN_SUBSET}" \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --save-dir "$RUN_DIR" \
    --save_every "${SAVE_EVERY}" \
    --seed "${SEED}" \
    --resume auto

kill "$MON_PID" || true
echo "[INFO] FB KD complete"
