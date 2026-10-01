#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

DATASET="${DATASET:-obqa}"
MODEL="${MODEL:-Qwen/Qwen3-8B-Base}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_obqa_lora_map_leftpad/obqa_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/obqa_transfer/clora}"
SEEDS=(${SEEDS:-1 3 7})
EVAL_DATASET="${EVAL_DATASET:-iid,arc-e,arc-c,mmlu-h,mmlu-c}"

for seed in "${SEEDS[@]}"; do
  run_and_log "${LOG_ROOT}/seed_${seed}.log" \
    "${PY}" baselines/c_lora/qwen3/run/main.py \
      --nowand \
      --dataset "${DATASET}" \
      --dataset-type benchmark_mcdataset \
      --modelwrapper c_lora \
      --model "${MODEL}" \
      --model-type causallm \
      --max-seq-len 300 \
      --max-train-steps 2000 \
      --eval-per-steps 100 \
      --batch-size 8 \
      --eval-batch-size 48 \
      --lr 5e-5 \
      --opt adamw \
      --opt-wd 0.01 \
      --warmup-ratio 0.06 \
      --adam-epsilon 1e-8 \
      --seed "${seed}" \
      --shared-init-lora-path "${MAP_ROOT}/seed_${seed}/init_lora.pt" \
      --save-clora-dir "clora_qwen3_8b_obqa/seed_${seed}/clora" \
      --testing-set train_train_test \
      --eval-dataset "${EVAL_DATASET}" \
      --lora-r 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --apply-classhead-lora \
      --bayes-eval-n-samples-final 32 \
      --bayes-kllr 0.01 \
      --bayes-kllr-std 0.02 \
      --bayes-beta 0.2 \
      --bayes-kl-reweighting 1 \
      --bayes-klreweighting \
      --bayes-datasetrescaling \
      --bayes-opt2-wd 0.0005
done
