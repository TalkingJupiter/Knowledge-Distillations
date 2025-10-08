#!/usr/bin/env bash
#SBATCH --job-name=kd_merge_eval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lighteval-logs/%x_%j.out
#SBATCH --error=logs/lighteval-logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Args: base model + adapter dir ---
BASE=${1:?Usage: sbatch kd_merge_and_eval.slurm <base_model_id> <adapter_dir>}
ADAPTER=${2:?Usage: sbatch kd_merge_and_eval.slurm <base_model_id> <adapter_dir>}

SAFE_NAME="$(basename "$ADAPTER")"
MERGED="merged/${SAFE_NAME}_full"

# --- Env setup ---
source ~/.bashrc || true
conda activate kd || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

mkdir -p logs results merged

echo "====================================================="
echo "[INFO] Merge + LightEval"
echo " Base model   : ${BASE}"
echo " Adapter dir  : ${ADAPTER}"
echo " Merged out   : ${MERGED}"
echo "====================================================="

# --- Merge adapter into full model ---
python <<PYCODE
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

base = "${BASE}"
adapter = "${ADAPTER}"
merged = "${MERGED}"

print(f"[MERGE] Base={base}")
print(f"[MERGE] Adapter={adapter}")
print(f"[MERGE] Saving merged model -> {merged}")

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
tok.save_pretrained(merged)

model = AutoPeftModelForCausalLM.from_pretrained(
    adapter,
    device_map=None,       # force CPU
    torch_dtype="auto"     # don’t require CUDA ops
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged)

print("[MERGE] Done.")
PYCODE


# --- Run LightEval ---
lighteval accelerate \
  "model_name=${MERGED},batch_size=4" \
  "leaderboard|mmlu|0,leaderboard|gsm8k|0,leaderboard|arc_challenge|0,leaderboard|truthfulqa:mc|0" \
  --output_dir "results/lighteval_${SAFE_NAME}"

echo "[INFO] All done -> results/lighteval_${SAFE_NAME}"
