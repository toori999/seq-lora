#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

TASK="${TASK:-obqa}"
EVAL_TASKS="${EVAL_TASKS:-iid,arc-e,arc-c,mmlu_science_high,mmlu_science_college}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_obqa_lora_map_leftpad/obqa_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/obqa_transfer/ensemble}"
EVAL_BSZ="${EVAL_BSZ:-48}"
ENSEMBLE_GROUPS=(
  "seeds_1_3_7:1 3 7"
  "seeds_3_7_11:3 7 11"
  "seeds_7_11_13:7 11 13"
)

for group in "${ENSEMBLE_GROUPS[@]}"; do
  group_id="${group%%:*}"
  seed_list="${group#*:}"
  args=()
  for seed in ${seed_list}; do
    args+=(--map_adapter_dir "${MAP_ROOT}/seed_${seed}/map_step_2000")
  done
  run_and_log "${LOG_ROOT}/${group_id}.log" \
    "${PY}" -m baselines.ensemble_eval \
      --task "${TASK}" \
      "${args[@]}" \
      --eval_tasks "${EVAL_TASKS}" \
      --eval_bsz "${EVAL_BSZ}" \
      --seed 0
done
