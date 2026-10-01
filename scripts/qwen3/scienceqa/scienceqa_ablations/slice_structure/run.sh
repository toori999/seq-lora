#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
GRADE_SLICE="${GRADE_SLICE:-slice_data/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_ablation/T10/kfac_balanced}"
RAND_SLICE_ROOT="${RAND_SLICE_ROOT:-slice_data/scienceqa_slice_ablation_controls/full_shuffle}"
REV_SLICE_ROOT="${REV_SLICE_ROOT:-slice_data/scienceqa_slice_ablation_controls/reverse_grade}"
SHUF_SLICE_ROOT="${SHUF_SLICE_ROOT:-slice_data/scienceqa_slice_ablation_controls/shuffled_grade}"
BASE_NLL_SLICE="${BASE_NLL_SLICE:-slice_data/scienceqa_slice_ablation_controls/base_nll_easyhard/kfac_balanced}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_ablations/slice_structure}"
CACHE_ROOT="${CACHE_ROOT:-caches/qwen3/scienceqa/slice_structure}"
SEEDS=(${SEEDS:-1 3 7 11 13})
VARIANTS=(${VARIANTS:-map q0_pooled lgssm_grade lgssm_random lgssm_reverse lgssm_shuffled lgssm_base_nll ind_grade ind_random})

run_seq_variant() {
  local variant="$1"
  local seed="$2"
  local slice_dir="${GRADE_SLICE}"
  local cache_path="${CACHE_ROOT}/${variant}/seed${seed}_stats.pt"
  local mode_args=(--posterior_eval_mode lgssm_final --mc_eval_samples 32 --mc_eval_chunk 8 --tau_mode auto --tau_anchor_n_samples 32)

  case "${variant}" in
    map)
      mode_args=(--posterior_eval_mode lgssm_final --mc_eval_samples 1 --mc_eval_chunk 1 --tau_mode fixed --posterior_tau 0)
      ;;
    q0_pooled)
      mode_args+=(--s_q 0)
      ;;
    lgssm_random)
      slice_dir="${RAND_SLICE_ROOT}/seed_${seed}/kfac_balanced"
      ;;
    lgssm_reverse)
      slice_dir="${REV_SLICE_ROOT}/seed_${seed}/kfac_balanced"
      ;;
    lgssm_shuffled)
      slice_dir="${SHUF_SLICE_ROOT}/seed_${seed}/kfac_balanced"
      ;;
    lgssm_base_nll)
      slice_dir="${BASE_NLL_SLICE}"
      ;;
    ind_grade)
      mode_args=(--posterior_eval_mode independent_slice_ensemble --independent_slice_mc_samples_per_slice 32 --mc_eval_samples 30 --mc_eval_chunk 8 --tau_mode auto --tau_anchor_n_samples 30)
      ;;
    ind_random)
      slice_dir="${RAND_SLICE_ROOT}/seed_${seed}/kfac_balanced"
      mode_args=(--posterior_eval_mode independent_slice_ensemble --independent_slice_mc_samples_per_slice 32 --mc_eval_samples 30 --mc_eval_chunk 8 --tau_mode auto --tau_anchor_n_samples 30)
      ;;
  esac

  run_and_log "${LOG_ROOT}/${variant}/seed${seed}.log" \
    "${PY}" -m seq_lora.eval \
      --task "${TASK}" \
      --eval_protocol default \
      --seed "${seed}" \
      --slices_dir "${slice_dir}" \
      --map_dir "${MAP_ROOT}/seed_${seed}/map_step_2000" \
      --eval_tasks "${EVAL_TASKS}" \
      --trust_remote_code true \
      --tokenizer_padding_side left \
      --kfac_backend asdl \
      --kfac_bsz 4 \
      --max_kfac_samples_per_slice -1 \
      --eval_bsz 64 \
      --q_mode module_constant \
      --tau_search_max 1 \
      --tau_anchor_size 500 \
      --tau_anchor_bsz 64 \
      --posterior_stats_dtype float32 \
      --posterior_stats_cache_path "${cache_path}" \
      "${mode_args[@]}"
}

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_seq_variant "${variant}" "${seed}"
  done
done
