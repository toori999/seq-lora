#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_main/mcdrop}"
SEEDS=(${SEEDS:-1 3 7 11 13})
EVAL_BSZ="${EVAL_BSZ:-48}"
MC_SAMPLES="${MC_SAMPLES:-32}"

for seed in "${SEEDS[@]}"; do
  run_and_log "${LOG_ROOT}/seed${seed}.log" \
    "${PY}" -m baselines.mcdrop_eval \
      --task "${TASK}" \
      --map_adapter_dir "${MAP_ROOT}/seed_${seed}/map_step_2000" \
      --eval_tasks "${EVAL_TASKS}" \
      --eval_bsz "${EVAL_BSZ}" \
      --mc_samples "${MC_SAMPLES}" \
      --temp 1.0 \
      --seed "${seed}"
done
