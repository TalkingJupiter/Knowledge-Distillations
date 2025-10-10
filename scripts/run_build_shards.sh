#!/usr/bin/env bash
#SBATCH --job-name=kd_build_shards_exact
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=zen4
#SBATCH --time=24:00:00
#SBATCH --output=logs/shards/%x_%j.out
#SBATCH --error=logs/shards/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

# Match your interactive environment
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore:.*pynvml package is deprecated.*:FutureWarning"


# (Optional) cache dirs (same as your command)
# export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/hf/datasets}"
# export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/models}"



mkdir -p logs

echo "[INFO] Starting shard build at $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Cache dir: $SCRATCH/hf/datasets"

python data/build_shards_from_hf.py \
  --dataset allenai/c4:en \
  --split train \
  --out data/shards_research_10M.jsonl \
  --cache_dir "$SCRATCH/hf/datasets" \
  --with-meta \
  --shuffle_streaming --buffer_size 200000

rc=$?
echo "[INFO] Finished Python with rc=$rc at $(date)"
exit $rc


##################################################
#
#           DATASETS & OTHER Settings
#
##################################################
  # --dataset oscar-corpus/OSCAR-2301:en \
  # --dataset allenai/c4:en \
  # --dataset wikipedia:20220301.en \
  # --dataset OpenAssistant/oasst2 \
  # --dataset databricks/databricks-dolly-15k \
  # --dataset daily_dialog \
  # --dataset bavard/personachat_truecased \
  # --dataset hotpotqa/hotpot_qa:distractor@train \
  # --dataset BeIR/nq:corpus@test
  #--weights "0.32,0.22,0.08,0.06,0.08,0.03,0.07,0.05,0.04,0.03,0.02" \
  #--max_samples 10000000 \