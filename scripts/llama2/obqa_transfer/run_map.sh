#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
CKPT_ROOT="${CKPT_ROOT:-${SEQ_LORA_ROOT}/baselines/bayesian-peft/checkpoints/mle/${MODEL}/obqa}"
LOG_ROOT="${LOG_ROOT:-${SEQ_LORA_ROOT}/logs/llama2/obqa_transfer/map}"
SEEDS_SPEC="${SEEDS:-1 3 7}"
SEEDS=(${SEEDS_SPEC//,/ })
DATASETS=(${DATASETS:-obqa ARC-Easy ARC-Challenge MMLU-chem MMLU-phy})
EVAL_BSZ="${EVAL_BSZ:-48}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-300}"

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

for seed in "${SEEDS[@]}"; do
  adapter_dir="${CKPT_ROOT}/lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed${seed}"
  if [[ ! -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    echo "[ERROR] missing adapter: ${adapter_dir}" | tee -a "${LOG_ROOT}/run_all.log"
    exit 1
  fi

  for dataset in "${DATASETS[@]}"; do
    task_name="$(log_task_name "${dataset}")"
    run_and_log "${LOG_ROOT}/source_obqa__eval_${task_name}__seed${seed}.log" \
      "${PY}" -m baselines.bayesian_peft_checkpoint_eval \
        --method map \
        --dataset "${dataset}" \
        --eval_split auto \
        --eval_bsz "${EVAL_BSZ}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --model "${MODEL}" \
        --adapter_dir "${adapter_dir}" \
        --seed "${seed}"
  done
done
