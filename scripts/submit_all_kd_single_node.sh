#!/usr/bin/env bash
#SBATCH --job-name=kd_submitter_single_node
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --time=00:15:00
#SBATCH --output=logs/2nd/%x_%j.out
#SBATCH --error=logs/2nd/%x_%j.err
set -euo pipefail

RB=$(sbatch --parsable scripts/run_response_based_single_node.sh)
FB=$(sbatch --parsable scripts/run_feature_based_single_node.sh)
RELB=$(sbatch --parsable scripts/run_relation_based_single_node.sh)

echo "[SUBMITTED] Response-Based=$RB  Feature-Based=$FB  Relation-Based=$RELB"