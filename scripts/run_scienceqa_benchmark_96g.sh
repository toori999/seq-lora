#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${1:-${ROOT_DIR}/benchmark_suite_scienceqa_96g}"
LOG_PATH="${2:-${ROOT_DIR}/benchmark_suite_scienceqa_96g.log}"

cd "${ROOT_DIR}"
python run_scienceqa_benchmark_suite.py \
  --seeds 0,1,2,3,4 \
  --result_root "${RESULT_ROOT}" \
  --eval_tasks "iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu,gpqa_main" \
  --map_micro_bsz 32 \
  --map_grad_accum 1 \
  --map_eval_bsz 128 \
  --constant_q_var 1.0 \
  --blob_eval_n 10 \
  --mcdrop_mc_samples 32 \
  --mcdrop_temp 1.0 \
  --laplace_fit_bsz 32 \
  --laplace_bsz 32 \
  --laplace_prior_optim_step 1000 \
  --laplace_mc_samples 10000 \
  --laplace_mc_chunk 512 \
  --ensemble_total_seeds 25 \
  --ensemble_num_groups 5 \
  --resume \
  2>&1 | tee "${LOG_PATH}"
