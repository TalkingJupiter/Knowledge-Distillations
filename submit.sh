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

set -Eeuo pipefail

# =======================
# Config (override via --export=ALL,VAR=val)
# =======================
LOGDIR="${LOGDIR:-logs}"

# Stage scripts
ENV_JOB="${ENV_JOB:-scripts/_env_single_node.sh}"
SHARDS_JOB="${SHARDS_JOB:-scripts/build_shards_10M.sbatch}"          # shards wrapper
CACHES_JOB="${CACHES_JOB:-scripts/kd_build_caches.sbatch}"           # GPU caches builder
KD_JOB="${KD_JOB:-scripts/kd_submitter_single_node.sbatch}"          # submits FB/RelB/RB

# Partition for tiny cleanup/disarm jobs (CPU ok)
PARTITION_CPU="${PARTITION_CPU:-zen4}"

# Dataset build defaults (human-heavy research mix)
HF_DATASETS="${HF_DATASETS:-c4,oscar-corpus/OSCAR-2301,wikipedia,allenai/arxiv-papers,OpenAssistant/oasst2,databricks/databricks-dolly-15k,li2017dailydialog,bavard/personachat_truecased,hotpotqa/hotpot_qa,BeIR/nq,pfb30/multi_woz_v22}"
WEIGHTS="${WEIGHTS:-0.32,0.22,0.08,0.06,0.08,0.03,0.07,0.05,0.04,0.03,0.02}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-10000000}"  # 10M
OUT="${OUT:-data/shards_research_10M.jsonl.gz}"
STREAMING="${STREAMING:-1}"
DATA_DIR="${DATA_DIR:-}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/hf/datasets}"
WITH_META="${WITH_META:-1}"
GZIP_OUT="${GZIP_OUT:-1}"
SHUFFLE_STREAMING="${SHUFFLE_STREAMING:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-200000}"

# Shared models (passed to downstream stages)
STUDENT="${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER="${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}"

mkdir -p "$LOGDIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# =======================
# Helpers
# =======================
die() { echo "[FATAL] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "[ERROR] Missing required file: $1"; }

submit() {
  local script="$1"; shift || true
  local out jid
  out=$(sbatch "$@" "$script" 2>&1) || die "sbatch failed: $script $* :: $out"
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"$out")
  [[ -n "${jid:-}" ]] || die "Could not parse JobID from: $out"
  echo "$jid"
}

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

submit_disarm() {
  local dep_ok="$1" cleanup_jid="$2"
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

# 1) Env/bootstrap (CPU)
jid_env=$(submit "$ENV_JOB" \
  --job-name=env_bootstrap \
  --partition="${PARTITION_CPU}" \
  --time=00:15:00 \
  --output="${LOGDIR}/%x_%j.out" \
  --error="${LOGDIR}/%x_%j.err")
echo "[SUBMIT] env             -> $jid_env"

# 2) Build shards (let the SBATCH header inside decide its partition)
jid_shards=$(submit "$SHARDS_JOB" \
  --dependency="afterok:${jid_env}" 
echo "[SUBMIT] build_shards    -> $jid_shards (afterok:$jid_env)"

# 3) Build caches (GPU job) — no partition override here
jid_caches=$(submit "$CACHES_JOB" \
  --dependency="afterok:${jid_shards}" \
  --export=ALL,IN="${OUT}",TEACHER="${TEACHER}")
echo "[SUBMIT] build_caches    -> $jid_caches (afterok:$jid_shards)"

# 4) KD submitter (GPU jobs) — pass shared STUDENT
jid_kd=$(submit "$KD_JOB" \
  --dependency="afterok:${jid_caches}" \
  --export=ALL,STUDENT="${STUDENT}")
echo "[SUBMIT] kd_pipeline     -> $jid_kd (afterok:$jid_caches)"

# =======================
# Cleanup on failure + Disarm on success
# =======================
jid_clean_env=$(submit_cleanup "afternotok:${jid_env}"   "$jid_shards" "$jid_caches" "$jid_kd")
submit_disarm "afterok:${jid_env}"    "$jid_clean_env"

jid_clean_shr=$(submit_cleanup "afternotok:${jid_shards}"              "$jid_caches" "$jid_kd")
submit_disarm "afterok:${jid_shards}" "$jid_clean_shr"

jid_clean_cch=$(submit_cleanup "afternotok:${jid_caches}"                           "$jid_kd")
submit_disarm "afterok:${jid_caches}" "$jid_clean_cch"

echo "[INFO] All jobs submitted:"
printf "  env:    %s\n  shards: %s\n  caches: %s\n  kd:     %s\n" "$jid_env" "$jid_shards" "$jid_caches" "$jid_kd"
