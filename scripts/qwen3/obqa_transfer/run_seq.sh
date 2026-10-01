#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

TASK="${TASK:-obqa}"
EVAL_TASKS="${EVAL_TASKS:-iid,arc-e,arc-c,mmlu_science_high,mmlu_science_college}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_obqa_lora_map_leftpad/obqa_qv_lmhead_leftpad}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/obqa_transfer/seq}"
CACHE_ROOT="${CACHE_ROOT:-caches/qwen3/obqa_transfer/seq_random_T10}"
SEEDS=(${SEEDS:-1 3 7})

for seed in "${SEEDS[@]}"; do
  run_and_log "${LOG_ROOT}/seed_${seed}.log" \
    "${PY}" -m seq_lora.eval \
      --task "${TASK}" \
      --eval_protocol default \
      --seed "${seed}" \
      --map_dir "${MAP_ROOT}/seed_${seed}/map_step_2000" \
      --eval_tasks "${EVAL_TASKS}" \
      --random_num_slices 10 \
      --trust_remote_code true \
      --tokenizer_padding_side left \
      --kfac_backend asdl \
      --kfac_bsz 4 \
      --eval_bsz 64 \
      --q_mode module_constant \
      --posterior_eval_mode lgssm_final \
      --mc_eval_samples 32 \
      --mc_eval_chunk 8 \
      --tau_mode auto \
      --tau_search_max 1 \
      --tau_anchor_size 500 \
      --tau_anchor_bsz 64 \
      --tau_anchor_n_samples 10 \
      --posterior_stats_dtype float32 \
      --posterior_stats_cache_path "${CACHE_ROOT}/seed_${seed}_stats.pt"
done
