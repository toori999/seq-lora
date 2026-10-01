#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

TASK="${TASK:-obqa}"
EVAL_TASKS="${EVAL_TASKS:-iid,arc-e,arc-c,mmlu-chem,mmlu-phy}"
MAP_ROOT="${MAP_ROOT:-baselines/bayesian-peft/checkpoints/mle/meta-llama/Llama-2-7b-hf/obqa}"
LOG_ROOT="${LOG_ROOT:-logs/llama2/obqa_transfer/seq}"
CACHE_ROOT="${CACHE_ROOT:-caches/seq_llama2_obqa_ood_aligned_random_T10}"
SEEDS=(${SEEDS:-1 3 7})

# The paper's in-distribution (OBQA) and OOD rows come from two separate posterior builds:
# OBQA uses every K-FAC sample with 32 tau-anchor MC samples; the OOD targets use the
# runner's 256-samples-per-slice K-FAC cap with 30 tau-anchor MC samples.
for seed in "${SEEDS[@]}"; do
  map_dir="${MAP_ROOT}/lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed${seed}"
  run_and_log "${LOG_ROOT}/source_obqa__eval_obqa__seed${seed}.log" \
    "${PY}" -m seq_lora.eval \
      --task "${TASK}" \
      --seed "${seed}" \
      --map_dir "${map_dir}" \
      --eval_tasks iid \
      --eval_protocol bayesian_peft \
      --random_num_slices 10 \
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
      --posterior_stats_cache_path "${CACHE_ROOT}/obqa_seed${seed}_iid_stats.pt"

  run_and_log "${LOG_ROOT}/source_obqa__eval_arc-e_arc-c_mmlu-chem_mmlu-phy__seed${seed}.log" \
    "${PY}" -m seq_lora.eval \
      --task "${TASK}" \
      --seed "${seed}" \
      --map_dir "${map_dir}" \
      --eval_tasks "${EVAL_TASKS#iid,}" \
      --eval_protocol bayesian_peft \
      --random_num_slices 10 \
      --tokenizer_padding_side left \
      --kfac_backend asdl \
      --kfac_bsz 4 \
      --max_kfac_samples_per_slice 256 \
      --eval_bsz 48 \
      --q_mode module_constant \
      --posterior_eval_mode lgssm_final \
      --mc_eval_samples 32 \
      --mc_eval_chunk 8 \
      --tau_mode auto \
      --tau_search_max 1 \
      --tau_anchor_size 500 \
      --tau_anchor_bsz 64 \
      --tau_anchor_n_samples 30 \
      --posterior_stats_dtype float32 \
      --posterior_stats_cache_path "${CACHE_ROOT}/obqa_seed${seed}_ood_stats.pt"
done
