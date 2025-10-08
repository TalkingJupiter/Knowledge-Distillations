#!/usr/bin/env bash
#SBATCH --job-name=kd_build_shards_exact
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=zen4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

# Match your interactive environment
# export HF_DATASETS_OFFLINE=0
# export PYTHONUNBUFFERED=1
# export CUDA_VISIBLE_DEVICES=""

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
  --dataset oscar-corpus/OSCAR-2301:en \
  --dataset wikipedia:20240520.en \
  --dataset allenai/arxiv-papers \
  --dataset OpenAssistant/oasst2 \
  --dataset databricks/databricks-dolly-15k \
  --dataset li2017dailydialog \
  --dataset bavard/personachat_truecased \
  --dataset hotpotqa/hotpot_qa \
  --dataset BeIR/nq@test \
  --dataset pfb30/multi_woz_v22 \
  --split train \
  --weights "0.32,0.22,0.08,0.06,0.08,0.03,0.07,0.05,0.04,0.03,0.02" \
  --max_samples 10000000 \
  --out data/shards_research_10M.jsonl.gz \
  --cache_dir "$SCRATCH/hf/datasets" \
  --with-meta \
  --gzip \
  --shuffle_streaming --buffer_size 200000

rc=$?
echo "[INFO] Finished Python with rc=$rc at $(date)"
exit $rc
