#!/usr/bin/env bash

# Example MAP training launches for the bayesian-peft-style refactor layout.

python -m refactor.bayesian_peft_layout.run.train_map \
  --variant order \
  --seed 0 \
  --micro-bsz 4 \
  --grad-accum 2 \
  --eval-bsz 32

python -m refactor.bayesian_peft_layout.run.train_map \
  --variant reverse \
  --seed 1 \
  --micro-bsz 4 \
  --grad-accum 2 \
  --eval-bsz 32

python -m refactor.bayesian_peft_layout.run.train_map \
  --backend refactor \
  --variant order \
  --seed 0 \
  --inspect-only \
  --preview-dataset
