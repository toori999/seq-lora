#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

DATASET="${DATASET:-scienceqa_closedchoice_grade2_11}"
MODEL="${MODEL:-Qwen/Qwen3-8B-Base}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_main/tfb}"
SEEDS=(${SEEDS:-1 3 7 11 13})
EVAL_DATASET="${EVAL_DATASET:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"

for seed in "${SEEDS[@]}"; do
  run_and_log "${LOG_ROOT}/seed${seed}.log" \
    "${PY}" baselines/bayesian-peft/qwen3/run/main.py \
      --nowand \
      --dataset "${DATASET}" \
      --dataset-type benchmark_mcdataset \
      --modelwrapper tfblora \
      --model "${MODEL}" \
      --model-type causallm \
      --max-seq-len 300 \
      --max-train-steps 0 \
      --eval-batch-size 48 \
      --seed "${seed}" \
      --load-lora-path "${MAP_ROOT}/seed_${seed}/map_step_2000" \
      --testing-set train_train_test \
      --eval-dataset "${EVAL_DATASET}" \
      --anchor-size 500 \
      --lora-r 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --apply-classhead-lora \
      --bayes-train-n-samples 10 \
      --bayes-eval-n-samples-final 32 \
      --bayes-beta 0.015 \
      --bayes-final-beta 0.18 \
      --th 0.003 \
      --iter 5
done
