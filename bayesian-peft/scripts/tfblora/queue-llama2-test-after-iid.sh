#!/usr/bin/env bash
set -euo pipefail

base_dir="checkpoints/tfblora/meta-llama/Llama-2-7b-hf"
expected_runs=18
poll_seconds="${POLL_SECONDS:-60}"

count_completed() {
  python - <<'PY'
import os
base = "checkpoints/tfblora/meta-llama/Llama-2-7b-hf"
count = 0
if os.path.isdir(base):
    for root, _, files in os.walk(base):
        if "log.txt" not in files:
            continue
        path = os.path.join(root, "log.txt")
        try:
            text = open(path).read()
        except OSError:
            continue
        if "val_acc:" in text and "-iid-" in root:
            count += 1
print(count)
PY
}

echo "Waiting for ${expected_runs} completed IID runs under ${base_dir}"
while true; do
  completed="$(count_completed)"
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${timestamp}] completed_iid_runs=${completed}/${expected_runs}"
  if [ "${completed}" -ge "${expected_runs}" ]; then
    break
  fi
  sleep "${poll_seconds}"
done

echo "IID runs complete. Starting test reruns for ARC-Challenge / ARC-Easy / obqa."
bash scripts/tfblora/mle-llama2-test-single-gpu.sh
