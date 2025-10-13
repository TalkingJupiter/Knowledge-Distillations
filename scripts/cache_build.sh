#!/usr/bin/env bash
#SBATCH --job-name=cache_build_submit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=zen4
#SBATCH --time=00:15:00
#SBATCH --output=logs/submitter/%x_%j.out
#SBATCH --error=logs/submitter/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "[INFO] Launching KD Cache Build at $(date)"
echo "[INFO] Partition: ${SLURM_JOB_PARTITION:-zen4}"

mkdir -p logs/submitter/

for f in scripts/cache_build/rb_build_caches.sh \
         scripts/cache_build/fb_build_caches.sh \
         scripts/cache_build/relb_build_caches.sh \
         data/shards_research_10M.jsonl
do
  [[ -e "$f" ]] || { echo "[ERROR] Missing file: $f" >&2; exit 1; }
done


# ---- Submit each KD mode ----
RB=$(sbatch --parsable scripts/cache_build/rb_build_caches.sh)
FB=$(sbatch --parsable scripts/cache_build/fb_build_caches.sh)
RELB=$(sbatch --parsable scripts/cache_build/relb_build_caches.sh)

echo "[SUBMITTED]"
echo "  Response-Based KD  → JobID $RB"
echo "  Feature-Based  KD  → JobID $FB"
echo "  Relation-Based KD  → JobID $RELB"

# optional dependency example:
# FB=$(sbatch --dependency=afterok:$RB --parsable scripts/kd_feature_based_single_node.sh)

echo "[INFO] All Cache Build jobs queued successfully."