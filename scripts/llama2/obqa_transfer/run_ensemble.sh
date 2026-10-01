#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
CKPT_ROOT="${CKPT_ROOT:-${SEQ_LORA_ROOT}/baselines/bayesian-peft/checkpoints/mle/${MODEL}}"
LOG_ROOT="${LOG_ROOT:-${SEQ_LORA_ROOT}/logs/llama2/obqa_transfer/ensemble}"
DATASETS=(${DATASETS:-obqa ARC-Challenge ARC-Easy MMLU-chem MMLU-phy})
ENSEMBLE_SEED_SETS="${ENSEMBLE_SEED_SETS:-1,3,7;1,3;3,7}"
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

collect_adapter_args() {
  local seed_set="$1"
  local -n out_ref="$2"
  out_ref=()
  local seed adapter_dir
  local seed_arr=()
  IFS=',' read -r -a seed_arr <<< "${seed_set}"
  for seed in "${seed_arr[@]}"; do
    adapter_dir="$(adapter_dir_for "${seed}")"
    if [[ ! -d "${adapter_dir}" ]]; then
      echo "[ERROR] missing adapter: ${adapter_dir}" | tee -a "${LOG_ROOT}/run_all.log"
      exit 1
    fi
    out_ref+=(--adapter_dir "${adapter_dir}")
  done
}

mkdir -p "${LOG_ROOT}"
{
  echo "[INFO] log_root=${LOG_ROOT}"
  echo "[INFO] ckpt_root=${CKPT_ROOT}"
  echo "[INFO] datasets=${DATASETS[*]}"
  echo "[INFO] ensemble_seed_sets=${ENSEMBLE_SEED_SETS}"
} | tee -a "${LOG_ROOT}/run_all.log"

IFS=';' read -r -a SEED_SET_ARR <<< "${ENSEMBLE_SEED_SETS}"
for dataset in "${DATASETS[@]}"; do
  task_name="$(log_task_name "${dataset}")"
  ens_idx=1
  for seed_set in "${SEED_SET_ARR[@]}"; do
    adapter_args=()
    collect_adapter_args "${seed_set}" adapter_args
    seed_tag="${seed_set//,/-}"
    cmd=(
      "${PY}" -m baselines.bayesian_peft_checkpoint_eval
      --method ens
      --dataset "${dataset}"
      --model "${MODEL}"
      --eval_split auto
      --eval_bsz "${EVAL_BSZ}"
      --max_seq_len "${MAX_SEQ_LEN}"
      --seed 0
      "${adapter_args[@]}"
    )
    if [[ "${LOAD_IN_8BIT}" == "1" ]]; then
      cmd+=(--load_in_8bit)
    fi
    run_and_log "${LOG_ROOT}/source_obqa__eval_${task_name}__ens${ens_idx}_seeds${seed_tag}.log" \
      "${cmd[@]}"
    ens_idx=$((ens_idx + 1))
  done
done

"${PY}" - "${LOG_ROOT}" <<'PY' | tee -a "${LOG_ROOT}/summary_mean_sd.log"
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
name_re = re.compile(
    r"^source_obqa__eval_(?P<eval_dataset>obqa|arc-e|arc-c|mmlu-chem|mmlu-phy)"
    r"__ens(?P<ens_idx>[0-9]+)_seeds(?P<seed_set>[0-9-]+)\.log$"
)
metric_re = re.compile(
    r"^\[(?P<eval_dataset>.+?)\((?P<split>.+?)\)\]\[ENS\]\s+"
    r"NLL=(?P<nll>[0-9.]+)\s+ACC=(?P<acc>[0-9.]+)%\s+"
    r"ECE=(?P<ece>[0-9.]+)%\s+Brier=(?P<brier>[0-9.]+)"
)


def mean_sd(values):
    return (float(values[0]), 0.0) if len(values) == 1 else (statistics.mean(values), statistics.stdev(values))


rows = []
for path in sorted(run_dir.glob("*.log")):
    name_match = name_re.match(path.name)
    if name_match is None:
        continue
    metric_match = None
    for line in path.read_text(errors="replace").splitlines():
        match = metric_re.match(line.strip())
        if match is not None:
            metric_match = match
    if metric_match is None:
        continue
    rows.append(
        {
            "eval_dataset": metric_match.group("eval_dataset"),
            "split": metric_match.group("split"),
            "ens_idx": int(name_match.group("ens_idx")),
            "seed_set": name_match.group("seed_set").replace("-", ","),
            "nll": float(metric_match.group("nll")),
            "acc_pct": float(metric_match.group("acc")),
            "ece_pct": float(metric_match.group("ece")),
            "brier": float(metric_match.group("brier")),
            "log": path.name,
        }
    )

groups = {}
for row in rows:
    groups.setdefault((row["eval_dataset"], row["split"]), []).append(row)

summary = []
for (eval_dataset, split), group in sorted(groups.items()):
    group = sorted(group, key=lambda row: row["ens_idx"])
    nll_mean, nll_sd = mean_sd([row["nll"] for row in group])
    acc_mean, acc_sd = mean_sd([row["acc_pct"] for row in group])
    ece_mean, ece_sd = mean_sd([row["ece_pct"] for row in group])
    brier_mean, brier_sd = mean_sd([row["brier"] for row in group])
    summary.append(
        {
            "eval_dataset": eval_dataset,
            "split": split,
            "n_ens": len(group),
            "seed_sets": ";".join(row["seed_set"] for row in group),
            "nll_mean": nll_mean,
            "nll_sd": nll_sd,
            "acc_pct_mean": acc_mean,
            "acc_pct_sd": acc_sd,
            "ece_pct_mean": ece_mean,
            "ece_pct_sd": ece_sd,
            "brier_mean": brier_mean,
            "brier_sd": brier_sd,
        }
    )

if rows:
    with (run_dir / "ensemble_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
if summary:
    with (run_dir / "summary_mean_sd.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

print(f"[ENS_SUMMARY] parsed_runs={len(rows)} groups={len(summary)}")
print(f"[ENS_SUMMARY] csv={run_dir / 'summary_mean_sd.csv'}")
for row in summary:
    print(
        f"[ENS_MEAN_SD][obqa_transfer][obqa->{row['eval_dataset']}({row['split']})] "
        f"n={row['n_ens']} seeds={row['seed_sets']} "
        f"NLL={row['nll_mean']:.4f}+/-{row['nll_sd']:.4f} "
        f"ACC={row['acc_pct_mean']:.2f}+/-{row['acc_pct_sd']:.2f}% "
        f"ECE={row['ece_pct_mean']:.2f}+/-{row['ece_pct_sd']:.2f}% "
        f"Brier={row['brier_mean']:.4f}+/-{row['brier_sd']:.4f}"
    )
PY
