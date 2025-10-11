#!/usr/bin/env bash
#SBATCH --job-name=kd_submitter_single_node
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:15:00
#SBATCH --output=logs/submitter/%x_%j.out
#SBATCH --error=logs/submitter/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "[INFO] Launching KD single-node suite at $(date)"
echo "[INFO] Partition: ${SLURM_JOB_PARTITION:-h100}"

# ---- Configurable shared student (applied to all modes) ----
STUDENT=${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}
export STUDENT

# ---- Submit each KD mode ----
RB=$(sbatch --parsable scripts/distil/kd_response_based_single_node.sh)
FB=$(sbatch --parsable scripts/distil/kd_feature_based_single_node.sh)
RELB=$(sbatch --parsable scripts/distil/kd_relation_based_single_node.sh)

echo "[SUBMITTED]"
echo "  Response-Based KD  → JobID $RB"
echo "  Feature-Based  KD  → JobID $FB"
echo "  Relation-Based KD  → JobID $RELB"

# optional dependency example:
# FB=$(sbatch --dependency=afterok:$RB --parsable scripts/kd_feature_based_single_node.sh)

echo "[INFO] All KD jobs queued successfully."
