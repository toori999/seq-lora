#!/usr/bin/env bash

# Example invocations for the bayesian-peft-style refactor layout.

python -m refactor.bayesian_peft_layout.run.main map \
  --task obqa \
  --adapter-dir /path/to/adapter

python -m refactor.bayesian_peft_layout.run.main mcdrop \
  --task obqa \
  --adapter-dir /path/to/adapter \
  --mc-samples 32

python -m refactor.bayesian_peft_layout.run.main deep-ensemble \
  --task obqa \
  --adapter-dirs /path/to/a,/path/to/b,/path/to/c

python -m refactor.bayesian_peft_layout.run.main seq-constantq \
  --task scienceqa_curriculum_grade2_11 \
  --map-dir /path/to/map_adapter \
  --slice-dir /path/to/train_slices \
  --eval-tasks iid,mmlu

python -m refactor.bayesian_peft_layout.run.main laplace \
  --task scienceqa_curriculum_grade2_11 \
  --adapter-dir /path/to/map_adapter \
  --output-dir /tmp/laplace_eval
