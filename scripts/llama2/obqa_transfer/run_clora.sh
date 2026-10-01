#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

SRC_DIR="${SRC_DIR:-${SEQ_LORA_ROOT}/baselines/c_lora}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
LOG_ROOT="${LOG_ROOT:-${SEQ_LORA_ROOT}/logs/llama2/obqa_transfer/clora}"
SEEDS_SPEC="${SEEDS:-1 3 7}"
SEEDS=(${SEEDS_SPEC//,/ })
EVAL_BSZ="${EVAL_BSZ:-48}"
TRAIN_BSZ="${TRAIN_BSZ:-4}"
LR="${LR:-5e-5}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-300}"
MC_SAMPLES="${MC_SAMPLES:-10}"
SUBSET_SIZE="${SUBSET_SIZE:-0.8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
TRAIN_EVAL_PER_STEPS="${TRAIN_EVAL_PER_STEPS:-1000000000}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
OOD_EVAL_DATASETS="${OOD_EVAL_DATASETS:-arc-e,arc-c,mmlu-chem,mmlu-phy}"

mkdir -p "${LOG_ROOT}"
{
  echo "[INFO] log_root=${LOG_ROOT}"
  echo "[INFO] source=${SRC_DIR}"
  echo "[INFO] seeds=${SEEDS[*]}"
  echo "[INFO] ood_eval_datasets=${OOD_EVAL_DATASETS}"
} | tee -a "${LOG_ROOT}/run_all.log"

run_clora() {
  local tag="$1"
  local seed="$2"
  shift 2
  (
    cd "${SRC_DIR}"
    run_and_log "${LOG_ROOT}/${tag}.log" \
      "${PY}" run/main.py \
      --dataset-type mcdataset \
      --dataset obqa \
      --model-type causallm \
      --model "${MODEL}" \
      --modelwrapper c_lora \
      --load-in-8bit "${LOAD_IN_8BIT}" \
      --lr "${LR}" \
      --batch-size "${TRAIN_BSZ}" \
      --eval-batch-size "${EVAL_BSZ}" \
      --opt adamw \
      --warmup-ratio 0.06 \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --seed "${seed}" \
      --testing-set auto \
      --evaluate \
      --nowand \
      --apply-classhead-lora \
      --lora-r 8 \
      --lora-alpha 16 \
      --lora-dropout "${LORA_DROPOUT}" \
      --log-path "${tag}_lr${LR}_drop${LORA_DROPOUT}_step${MAX_TRAIN_STEPS}" \
      --max-train-steps "${MAX_TRAIN_STEPS}" \
      --eval-per-steps "${TRAIN_EVAL_PER_STEPS}" \
      --bayes-kl-reweighting 1 \
      --subset-size "${SUBSET_SIZE}" \
      --bayes-eps 0.05 \
      --bayes-beta 0.2 \
      --bayes-gamma 8 \
      --bayes-kllr 0.01 \
      --bayes-kllr-std 0.01 \
      --bayes-momentum 0 \
      --bayes-opt2-wd 0 \
      --bayes-train-n-samples 1 \
      --bayes-eval-n-samples 1 \
      --bayes-eval-n-samples-final "${MC_SAMPLES}" \
      "$@"
  )
}

for seed in "${SEEDS[@]}"; do
  # Train once on OBQA and evaluate on the OOD targets; the runner saves the adapter
  # under last_models/, which the OBQA evaluation below reloads without retraining.
  run_clora "source_obqa__eval_arc-e_arc-c_mmlu-chem_mmlu-phy__seed${seed}" "${seed}" \
    --eval-dataset "${OOD_EVAL_DATASETS}"
  run_clora "source_obqa__eval_obqa__seed${seed}" "${seed}" \
    --load-checkpoint \
    --load-model-path "${SRC_DIR}/last_models/c_lora/${MODEL}/obqa/default/${seed}" \
    --max-train-steps 0
done
