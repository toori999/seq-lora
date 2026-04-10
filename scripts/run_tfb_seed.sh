#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:-0}"
EVAL_TASKS="${2:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu,gpqa_main}"

MAP_DIR="${ROOT_DIR}/iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/seed_${SEED}/map_step_2000"
OUT_DIR="${ROOT_DIR}/outputs_tfb_qv_lmhead_suite/seed_${SEED}"
LOG_PATH="${ROOT_DIR}/tfb_seed${SEED}.log"

mkdir -p "${OUT_DIR}"

cd "${ROOT_DIR}"
python tfb_eval.py \
  --task scienceqa_closedchoice_grade2_11 \
  --map_adapter_dir "${MAP_DIR}" \
  --output_dir "${OUT_DIR}" \
  --eval_tasks "${EVAL_TASKS}" \
  --anchor_split val \
  --anchor_bsz 32 \
  --eval_bsz 256 \
  --anchor_n_samples 10 \
  --eval_n_samples 10 \
  --bayes_beta_max 0.015 \
  --threshold 0.003 \
  --search_iters 5 \
  --bayes_eps 0.05 \
  --max_seq_len 300 \
  --seed 0 | tee "${LOG_PATH}"
