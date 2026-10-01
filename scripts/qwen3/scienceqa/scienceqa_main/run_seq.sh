#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
SLICE_DIR="${SLICE_DIR:-slice_data/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_ablation/T10/kfac_balanced}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_main/seq}"
CACHE_ROOT="${CACHE_ROOT:-caches/qwen3/scienceqa/scienceqa_main/seq_T10}"
SEEDS=(${SEEDS:-1 3 7 11 13})

for seed in "${SEEDS[@]}"; do
  run_and_log "${LOG_ROOT}/seed${seed}.log" \
    "${PY}" -m seq_lora.eval \
      --task "${TASK}" \
      --eval_protocol default \
      --seed "${seed}" \
      --map_dir "${MAP_ROOT}/seed_${seed}/map_step_2000" \
      --slices_dir "${SLICE_DIR}" \
      --eval_tasks "${EVAL_TASKS}" \
      --trust_remote_code true \
      --tokenizer_padding_side left \
      --kfac_backend asdl \
      --kfac_bsz 4 \
      --max_kfac_samples_per_slice -1 \
      --eval_bsz 48 \
      --q_mode module_constant \
      --posterior_eval_mode lgssm_final \
      --mc_eval_samples 32 \
      --mc_eval_chunk 8 \
      --tau_mode auto \
      --tau_search_max 1 \
      --tau_anchor_size 500 \
      --tau_anchor_bsz 64 \
      --tau_anchor_n_samples 32 \
      --posterior_stats_dtype float32 \
      --posterior_stats_cache_path "${CACHE_ROOT}/seed${seed}_stats.pt"
done
