#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/map_training_order}"
SEEDS=(${SEEDS:-1 3 7 11 13})
VARIANTS=(${VARIANTS:-reverse random})

adapter_root_for() {
  case "$1" in
    reverse)
      echo "iid_qwen35_8b_scienceqa_lora_map_leftpad_reversegrade/scienceqa_text_closedchoice_grade2_11_reversegrade_qv_lmhead_leftpad"
      ;;
    random)
      echo "iid_qwen35_8b_scienceqa_lora_map_leftpad_random/scienceqa_text_closedchoice_grade2_11_random_qv_lmhead_leftpad"
      ;;
    *)
      echo "[ERROR] unknown variant: $1" >&2
      return 1
      ;;
  esac
}

for variant in "${VARIANTS[@]}"; do
  root="$(adapter_root_for "${variant}")"
  for seed in "${SEEDS[@]}"; do
    run_and_log "${LOG_ROOT}/${variant}/seed_${seed}.log" \
      "${PY}" -m baselines.map_eval \
        --task "${TASK}" \
        --map_adapter_dir "${root}/seed_${seed}/map_step_2000" \
        --eval_tasks "${EVAL_TASKS}" \
        --eval_bsz 48 \
        --seed "${seed}"
  done
done
