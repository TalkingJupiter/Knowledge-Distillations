#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_lighteval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=h100
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

MODEL_DIR=${MODEL_DIR:?Set with --export=MODEL_DIR=/path/to/run}
source scripts/_env_single_node.sh

echo "[INFO] LightEval on ${MODEL_DIR}"
sbatch eval/lighteval_runner.sh "${MODEL_DIR}"

python eval/parse_results.py
echo "[INFO] Wrote results/eval_summary.csv"
