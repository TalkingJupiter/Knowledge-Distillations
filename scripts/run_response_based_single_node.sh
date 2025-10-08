#!/usr/bin/env bash
#SBATCH --job-name=kd_response_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/2nd/response/%x_%j.out
#SBATCH --error=logs/2nd/response/%x_%j.err
#SBATCH --nodelist=rpg-93-6
#SBATCH --array=0-4

set -euo pipefail
source scripts/_env_single_node.sh

echo "[INFO] Response-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

# Node-local telemetry
mkdir -p logs/telemetry/$SLURM_JOB_ID
python monitor.py --output logs/telemetry/2nd/response/$SLURM_JOB_ID/${HOSTNAME}.jsonl --interval 1 &
MON_PID=$!

RUN_DIR="serialization_dir/response/$(date +%Y%m%d_%H%M)_RB_1n"
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  --module kd.train \
    --kd.mode rb \
    --student meta-llama/Llama-3.1-8B \
    --data "data/topk_k16/*.parquet" \
    --rb.topk 16 \
    --rb.temperature 2.0 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --bash_size 2 \
    --max_steps 2000 \
    --save-dir "$RUN_DIR" \
    --save_every 200 \
    --resume auto 
    

kill $MON_PID || true
echo "[INFO] RB KD complete"
