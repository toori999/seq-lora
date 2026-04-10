#!/usr/bin/env bash

python -m refactor.bayesian_peft_layout.run.benchmark_suite \
  --seeds 0,1,2,3,4 \
  --result-root ./benchmark_suite_scienceqa_refactor \
  --resume
