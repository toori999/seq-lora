#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ_LORA_ROOT="${SEQ_LORA_ROOT:-$(cd "${SCRIPT_ROOT}/.." && pwd)}"
# Python interpreter used by every launcher; override with PY=/path/to/python.
PY="${PY:-${PYTHON:-python}}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-seq-lora}}"
mkdir -p "${MPLCONFIGDIR}"
cd "${SEQ_LORA_ROOT}"

run_and_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  echo "[RUN] $(date -Is)" | tee "${log_file}"
  printf '[CMD]' | tee -a "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
  local rc=${PIPESTATUS[0]}
  echo "[DONE] rc=${rc} $(date -Is)" | tee -a "${log_file}"
  return "${rc}"
}

contains_word() {
  local needle="$1"
  shift
  local word
  for word in "$@"; do
    [[ "${word}" == "${needle}" ]] && return 0
  done
  return 1
}

