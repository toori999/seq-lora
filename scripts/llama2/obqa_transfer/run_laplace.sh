#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

TASK="${TASK:-obqa}"
MAP_ROOT="${MAP_ROOT:-baselines/bayesian-peft/checkpoints/mle/meta-llama/Llama-2-7b-hf/obqa}"
LOG_ROOT="${LOG_ROOT:-logs/llama2/obqa_transfer/laplace}"
SEEDS=(${SEEDS:-1 3 7})
OOD_TASKS="${OOD_TASKS:-arc-e,arc-c,mmlu-chem,mmlu-phy}"

for seed in "${SEEDS[@]}"; do
  map_dir="${MAP_ROOT}/lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed${seed}"
  run_and_log "${LOG_ROOT}/source_obqa__eval_obqa__seed${seed}.log" \
    "${PY}" -m baselines.laplace.eval \
      --task_name "${TASK}" \
      --map_adapter_dir "${map_dir}" \
      --eval_tasks iid \
      --seed "${seed}" \
      --max_length 300 \
      --per_device_eval_batch_size 16 \
      --fit_bsz 32 \
      --laplace_bsz 32 \
      --laplace_sub all \
      --laplace_hessian kron \
      --laplace_mc_samples 1000 \
      --testing_set val

  run_and_log "${LOG_ROOT}/source_obqa__eval_arc-e_arc-c_mmlu-chem_mmlu-phy__seed${seed}.log" \
    "${PY}" -m baselines.laplace.eval \
      --task_name "${TASK}" \
      --map_adapter_dir "${map_dir}" \
      --eval_tasks "${OOD_TASKS}" \
      --seed "${seed}" \
      --max_length 300 \
      --per_device_eval_batch_size 16 \
      --fit_bsz 32 \
      --laplace_bsz 32 \
      --laplace_sub all \
      --laplace_hessian kron \
      --laplace_mc_samples 1000 \
      --testing_set val
done
