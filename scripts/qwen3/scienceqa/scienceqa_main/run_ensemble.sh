#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_main/ensemble}"
EVAL_BSZ="${EVAL_BSZ:-48}"
ENSEMBLE_GROUPS=(
  "0:0 1 2 3 4"
  "1:5 6 7 8 9"
  "2:10 11 12 13 14"
  "3:15 16 17 18 19"
  "4:20 21 22 23 24"
)

for group in "${ENSEMBLE_GROUPS[@]}"; do
  group_id="${group%%:*}"
  seed_list="${group#*:}"
  args=()
  for seed in ${seed_list}; do
    args+=(--map_adapter_dir "${MAP_ROOT}/seed_${seed}/map_step_2000")
  done
  run_and_log "${LOG_ROOT}/group${group_id}.log" \
    "${PY}" -m baselines.ensemble_eval \
      --task "${TASK}" \
      "${args[@]}" \
      --eval_tasks "${EVAL_TASKS}" \
      --eval_bsz "${EVAL_BSZ}" \
      --seed "${group_id}"
done
