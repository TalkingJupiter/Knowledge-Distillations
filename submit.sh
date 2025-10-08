#!/usr/bin/env bash
#SBATCH --job-name=pipeline_launcher
#SBATCH --partition=zen4              # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This job only SUBMITS other jobs using sbatch (very light). Those jobs do the real work.

#!/usr/bin/env bash
set -Eeuo pipefail

# =======================
# Config
# =======================
LOGDIR="logs"

# Stage scripts
ENV_JOB="scripts/_env_single_node.sh"
SHARDS_JOB="scripts/run_build_shards.sh"
CACHES_JOB="scripts/build_caches.sh"
KD_JOB="scripts/submit_all_kd_single_node.sh"

# Partition for the tiny cleanup/disarm jobs (CPU ok)
PARTITION_CPU="${PARTITION_CPU:-zen4}"

mkdir -p "$LOGDIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# =======================
# Helpers
# =======================
die() { echo "[FATAL] $*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "[ERROR] Missing required file: $1"
}

# submit: sbatch <opts...> <script> ; echo JobID
submit() {
  local script="$1"; shift || true
  local out jid
  out=$(sbatch "$@" "$script" 2>&1) || die "sbatch failed: $script $* :: $out"
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"$out")
  [[ -n "${jid:-}" ]] || die "Could not parse JobID from: $out"
  echo "$jid"
}

# submit a cleanup job that cancels downstream targets if <dep_expr> fails
submit_cleanup() {
  local dep="$1"; shift
  [[ $# -gt 0 ]] || { echo ""; return 0; }
  submit --job-name="pipeline_cleanup" \
         --partition="${PARTITION_CPU}" \
         --time=00:05:00 \
         --output="${LOGDIR}/%x_%j.out" \
         --error="${LOGDIR}/%x_%j.err" \
         --dependency="${dep}" \
         --wrap "$(printf 'scancel %s || true\n' "$@")"
}

# submit a disarm job that cancels the given cleanup job if upstream succeeds
submit_disarm() {
  local dep_ok="$1"   # e.g., afterok:<upstream_jid>
  local cleanup_jid="$2"
  [[ -n "${cleanup_jid:-}" ]] || return 0
  sbatch --job-name="cleanup_disarm" \
         --partition="${PARTITION_CPU}" \
         --time=00:02:00 \
         --output="${LOGDIR}/%x_%j.out" \
         --error="${LOGDIR}/%x_%j.err" \
         --dependency="${dep_ok}" \
         --wrap "scancel ${cleanup_jid} || true" >/dev/null
}

# =======================
# Presence checks
# =======================
need_file "$ENV_JOB"
need_file "$SHARDS_JOB"
need_file "$CACHES_JOB"
need_file "$KD_JOB"

# =======================
# Submit chain
# =======================
echo "[INFO] Submitting pipeline…"

jid_env=$(submit "$ENV_JOB")
echo "[SUBMIT] env             -> $jid_env"

# Shards with fixed --export values (your request)
jid_shards=$(submit "$SHARDS_JOB" \
  --dependency="afterok:${jid_env}" \
  --export=HF_DATASETS="tatsu-lab/alpaca,cerebras/SlimPajama-627B",WEIGHTS="0.05,0.95",SPLIT=train,STREAMING=1,OUT=data/shards.jsonl)
echo "[SUBMIT] build_shards    -> $jid_shards (afterok:$jid_env)"

jid_caches=$(submit "$CACHES_JOB" --dependency="afterok:${jid_shards}")
echo "[SUBMIT] build_caches    -> $jid_caches (afterok:$jid_shards)"

jid_kd=$(submit "$KD_JOB" --dependency="afterok:${jid_caches}")
echo "[SUBMIT] kd_pipeline     -> $jid_kd (afterok:$jid_caches)"

# =======================
# Cleanup on failure + Disarm on success
# =======================
# If env fails -> cancel shards, caches, kd
jid_clean_env=$(submit_cleanup "afternotok:${jid_env}"   "$jid_shards" "$jid_caches" "$jid_kd")
submit_disarm "afterok:${jid_env}"    "$jid_clean_env"

# If shards fail -> cancel caches, kd
jid_clean_shr=$(submit_cleanup "afternotok:${jid_shards}"              "$jid_caches" "$jid_kd")
submit_disarm "afterok:${jid_shards}" "$jid_clean_shr"

# If caches fail -> cancel kd
jid_clean_cch=$(submit_cleanup "afternotok:${jid_caches}"                           "$jid_kd")
submit_disarm "afterok:${jid_caches}" "$jid_clean_cch"

echo "[INFO] All jobs submitted:"
printf "  env:    %s\n  shards: %s\n  caches: %s\n  kd:     %s\n" "$jid_env" "$jid_shards" "$jid_caches" "$jid_kd"
