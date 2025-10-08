#!/bin/bash
#SBATCH --job-name=Relation_Plotter
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=zen4        # change to your CPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

# -------------------------
# Environment setup
# -------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate your plotting environment (adjust to your setup)
source ~/.bashrc
conda activate kd   # or whichever env has numpy/pandas/matplotlib

# -------------------------
# Input / output paths
# -------------------------
TELEMETRY_FILES="logs/telemetry/14620/*.jsonl"   # change to your file(s), space-separated if multiple
OUTDIR="plots/RB"

mkdir -p logs "$OUTDIR"

# -------------------------
# Run plotting
# -------------------------
echo "[INFO] Running telemetry plotting..."
python plotter.py \
  --telemetry $TELEMETRY_FILES \
  --out "$OUTDIR" \
  --resample-seconds 2 \
  --x elapsed \
  --format png pdf \
  --dpi 180

echo "[INFO] Done. Figures -> $OUTDIR/figures, Tables -> $OUTDIR/tables"
