#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
CKPT_ROOT="${CKPT_ROOT:-${SEQ_LORA_ROOT}/baselines/bayesian-peft/checkpoints/mle/${MODEL}}"
LOG_ROOT="${LOG_ROOT:-${SEQ_LORA_ROOT}/logs/llama2/obqa_transfer/mcdrop}"
SEEDS_SPEC="${SEEDS:-1 3 7}"
SEEDS=(${SEEDS_SPEC//,/ })
DATASETS=(${DATASETS:-obqa ARC-Challenge ARC-Easy MMLU-chem MMLU-phy})
MC_SAMPLES="${MC_SAMPLES:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-300}"
EVAL_BSZ="${EVAL_BSZ:-48}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"

adapter_dir_for() {
  local seed="$1"
  printf '%s/obqa/lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed%s' "${CKPT_ROOT}" "${seed}"
}

log_task_name() {
  case "$1" in
    obqa) echo "obqa" ;;
    ARC-Easy|arc-e) echo "arc-e" ;;
    ARC-Challenge|arc-c) echo "arc-c" ;;
    MMLU-chem|mmlu-chem) echo "mmlu-chem" ;;
    MMLU-phy|mmlu-phy) echo "mmlu-phy" ;;
    *) echo "$1" | tr '/[:upper:]' '_[:lower:]' ;;
  esac
}

mkdir -p "${LOG_ROOT}"
{
  echo "[INFO] log_root=${LOG_ROOT}"
  echo "[INFO] ckpt_root=${CKPT_ROOT}"
  echo "[INFO] seeds=${SEEDS[*]}"
  echo "[INFO] datasets=${DATASETS[*]}"
} | tee -a "${LOG_ROOT}/run_all.log"

for dataset in "${DATASETS[@]}"; do
  task_name="$(log_task_name "${dataset}")"
  for seed in "${SEEDS[@]}"; do
    adapter_dir="$(adapter_dir_for "${seed}")"
    if [[ ! -d "${adapter_dir}" ]]; then
      echo "[ERROR] missing adapter: ${adapter_dir}" | tee -a "${LOG_ROOT}/run_all.log"
      exit 1
    fi
    cmd=(
      "${PY}" -m baselines.bayesian_peft_checkpoint_eval
      --method mcdrop
      --dataset "${dataset}"
      --model "${MODEL}"
      --eval_split auto
      --eval_bsz "${EVAL_BSZ}"
      --max_seq_len "${MAX_SEQ_LEN}"
      --seed 0
      --mc_samples "${MC_SAMPLES}"
      --adapter_dir "${adapter_dir}"
    )
    if [[ "${LOAD_IN_8BIT}" == "1" ]]; then
      cmd+=(--load_in_8bit)
    fi
    run_and_log "${LOG_ROOT}/source_obqa__eval_${task_name}__seed${seed}.log" \
      "${cmd[@]}"
  done
done
