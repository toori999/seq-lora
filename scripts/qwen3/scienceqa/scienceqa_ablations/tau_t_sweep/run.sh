#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../../_common.sh"

TASK="${TASK:-scienceqa_closedchoice_grade2_11}"
EVAL_TASKS="${EVAL_TASKS:-iid,mmlu_science_high,mmlu_science_college}"
MAP_ROOT="${MAP_ROOT:-iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad}"
# Dense T grid built by: python -m seq_lora.data.make_scienceqa_slices --preset dense_tau
SLICE_ROOT="${SLICE_ROOT:-slice_data/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_dense_tau_seed1}"
LOG_ROOT="${LOG_ROOT:-logs/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/logs}"
CACHE_ROOT="${CACHE_ROOT:-caches/qwen3/scienceqa/tau_t_sweep}"
SEED="${SEED:-1}"
TS=(${TS:-2 3 4 5 6 7 8 9 10 12 16 20 24 30 36 40})
# MODES entries: auto (anchor-KL tau), tau05 / tau1 (fixed 0.5 / 1.0), or tau0pN (fixed 0.N).
# Fixed-tau sensitivity at T=10 (plotted by plot_t10_tau_grid.py):
#   TS=10 MODES="tau0p1 tau0p2 tau0p3 tau0p4 tau05 tau0p6 tau0p7 tau0p8 tau0p9 tau1" bash <this script>
MODES=(${MODES:-auto tau05 tau1})

for T in "${TS[@]}"; do
  for mode in "${MODES[@]}"; do
    case "${mode}" in
      auto) mode_args=(--tau_mode auto --tau_anchor_n_samples 30) ;;
      tau05) mode_args=(--tau_mode fixed --posterior_tau 0.5) ;;
      tau1) mode_args=(--tau_mode fixed --posterior_tau 1.0) ;;
      tau0p[0-9]*) mode_args=(--tau_mode fixed --posterior_tau "0.${mode#tau0p}") ;;
      *) echo "[ERROR] unknown MODES entry: ${mode}" >&2; exit 1 ;;
    esac

    run_and_log "${LOG_ROOT}/T${T}/${mode}.log" \
      "${PY}" -m seq_lora.eval \
        --task "${TASK}" \
        --eval_protocol default \
        --seed "${SEED}" \
        --map_dir "${MAP_ROOT}/seed_${SEED}/map_step_2000" \
        --slices_dir "${SLICE_ROOT}/T${T}/kfac_balanced" \
        --eval_tasks "${EVAL_TASKS}" \
        --trust_remote_code true \
        --tokenizer_padding_side left \
        --kfac_backend asdl \
        --kfac_bsz 4 \
        --max_kfac_samples_per_slice -1 \
        --eval_bsz 64 \
        --q_mode module_constant \
        --posterior_eval_mode lgssm_final \
        --mc_eval_samples 32 \
        --mc_eval_chunk 8 \
        --tau_search_max 1 \
        --tau_anchor_size 500 \
        --tau_anchor_bsz 64 \
        --posterior_stats_dtype float32 \
        --posterior_stats_cache_path "${CACHE_ROOT}/T${T}/seed${SEED}_stats.pt" \
        "${mode_args[@]}"
  done
done
