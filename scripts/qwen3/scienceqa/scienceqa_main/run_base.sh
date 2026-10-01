#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_main/base}"
EVAL_BSZ="${EVAL_BSZ:-48}"

run_and_log "${LOG_ROOT}/seed0.log" \
  "${PY}" -m baselines.base_eval \
    --task "${TASK}" \
    --base_model Qwen/Qwen3-8B-Base \
    --eval_tasks "${EVAL_TASKS}" \
    --eval_bsz "${EVAL_BSZ}" \
    --seed 0
