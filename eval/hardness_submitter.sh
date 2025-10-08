#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_submitter-feature
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=eval/logs/feature/%x_%j.out
#SBATCH --error=eval/logs/feature/%x_%j.err

set -euo pipefail


# Helper to get absolute paths (portable even if 'readlink -f' is missing)
abspath() { python - "$1" <<'PY'
import os,sys; print(os.path.abspath(sys.argv[1]))
PY
}

# === Bases (Instruct variants; will use chat template flags) ===
BASE_FEATURE="Qwen/Qwen2.5-1.5B-Instruct"
BASE_RESPONSE="meta-llama/Llama-3.1-8B-Instruct"
BASE_RELATION="Qwen/Qwen2.5-1.5B-Instruct"

CHAT_FLAGS=( --apply_chat_template --fewshot_as_multiturn )

# === Your adapter directories (as given) ===

ADAPTERS_FEATURE=(
  "$(abspath serialization_dir/feature/20250923_1156_FB_1n/)"
  "$(abspath serialization_dir/feature/20250923_1229_FB_1n/)"
  "$(abspath serialization_dir/feature/20250923_1305_FB_1n/)"
  "$(abspath serialization_dir/feature/20250923_1340_FB_1n/)"
  "$(abspath serialization_dir/feature/20250923_1413_FB_1n/)"
)

ADAPTERS_RESPONSE=(
  "$(abspath serialization_dir/response/20250923_1156_RB_1n/)"
  "$(abspath serialization_dir/response/20250923_1213_RB_1n/)"
  "$(abspath serialization_dir/response/20250923_1232_RB_1n/)"
  "$(abspath serialization_dir/response/20250923_1253_RB_1n/)"
  "$(abspath serialization_dir/response/20250923_1317_RB_1n/)"
)

ADAPTERS_RELATION=(
  "$(abspath serialization_dir/relation/20250923_1155_RelB_1n/)"
  "$(abspath serialization_dir/relation/20250923_1209_RelB_1n/)"
  "$(abspath serialization_dir/relation/20250923_1218_RelB_1n/)"
  "$(abspath serialization_dir/relation/20250923_1230_RelB_1n/)"
  "$(abspath serialization_dir/relation/20250923_1244_RelB_1n/)"
)

submit_group () {
  local base="$1"; shift
  local -a adapters=( "$@" )

  for ad in "${adapters[@]}"; do
    # validate adapter dir contains adapter_config.json
    if [[ ! -f "$ad/adapter_config.json" ]]; then
      echo "[WARN] Skipping '$ad' (missing adapter_config.json)"
      continue
    fi
    echo "[INFO] Submitting: BASE=$base  ADAPTER=$ad"
    sbatch eval/harness_runner.sh "$base" "$ad" "${CHAT_FLAGS[@]}"
  done
}

# Submit all groups
submit_group "$BASE_FEATURE"  "${ADAPTERS_FEATURE[@]}"
submit_group "$BASE_RESPONSE" "${ADAPTERS_RESPONSE[@]}"
submit_group "$BASE_RELATION" "${ADAPTERS_RELATION[@]}"
