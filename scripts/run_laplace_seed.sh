#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-0}"
EVAL_TASKS="${2:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu,gpqa_main}"

MAP_DIR="${ROOT_DIR}/iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/seed_${SEED}/map_step_2000"
OUT_DIR="${ROOT_DIR}/outputs_laplace_official_source_qv_lmhead_suite/seed_${SEED}"
LOG_PATH="${ROOT_DIR}/laplace_seed${SEED}.log"

mkdir -p "${OUT_DIR}"

cd "${ROOT_DIR}"
python laplace_lora_official_source_eval.py \
  --task_name scienceqa_closedchoice_grade2_11 \
  --map_adapter_dir "${MAP_DIR}" \
  --output_dir "${OUT_DIR}" \
  --eval_tasks "${EVAL_TASKS}" \
  --laplace_sub all \
  --testing_set val \
  --seed 0 \
  --fit_bsz 2 \
  --laplace_bsz 1 \
  --prior_optim_step 100 \
  --laplace_mc_samples 48 \
  --laplace_mc_chunk 8 \
  --max_length 300 | tee "${LOG_PATH}"
