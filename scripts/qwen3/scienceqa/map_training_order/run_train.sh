#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../_common.sh"

SEEDS="${SEEDS:-0,1,2,3,4}"
TRAIN_ORDERS="${TRAIN_ORDERS:-order reverse random}"
MICRO_BSZ="${MICRO_BSZ:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_BSZ="${EVAL_BSZ:-32}"

for train_order in ${TRAIN_ORDERS}; do
  "${PY}" -m training.scienceqa_map \
    --train_order "${train_order}" \
    --seeds "${SEEDS}" \
    --micro_bsz "${MICRO_BSZ}" \
    --grad_accum "${GRAD_ACCUM}" \
    --eval_bsz "${EVAL_BSZ}"
done
