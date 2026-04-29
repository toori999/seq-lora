# Seq-LoRA

Bayesian Seq-LoRA is a framework for uncertainty-aware, parameter-efficient
adaptation of large language models. It builds a post-hoc Bayesian posterior
over LoRA adapter coordinates, then evaluates calibrated Bayesian model
averages for multiple-choice QA without retraining the full base model.


## Overview

This repository contains the training, posterior construction, and evaluation
code for a ScienceQA-centered benchmark of Bayesian and uncertainty-aware LoRA
methods. The main experiments use `Qwen/Qwen3-8B-Base` and cast multiple-choice
prediction as last-token classification over answer choices `A/B/C/D`.

The benchmark compares:

- Frozen base model.
- Deterministic MAP LoRA.
- MC-Dropout over the MAP adapter.
- Deep ensembles of MAP adapters.
- Official-source Laplace over LoRA parameters.
- BLoB, TFB, and C-LoRA baselines.
- Bayesian Seq-LoRA.

Seq-LoRA constructs slice-aware posterior dynamics from LoRA curvature. It
extracts Kronecker-factored curvature per training slice, projects curvature and
gradient observations into low-dimensional adapter subspaces, runs LGSSM/Kalman
filtering over slice order, and samples terminal posterior LoRA coordinates for
Bayesian model averaging.

## Key Results

Macro-average over 7 IID/OOD evaluation tasks from the latest author-side
benchmark aggregation:

| Method | Macro ACC | Macro NLL | Macro ECE | Macro Brier |
| --- | ---: | ---: | ---: | ---: |
| base | 70.96 | 0.7197 | 9.18 | 0.3865 |
| MAP LoRA | 79.81 | 0.7785 | 12.66 | 0.3113 |
| MC-Dropout | 79.76 | 0.7767 | 12.72 | 0.3108 |
| Ensemble | 79.89 | 0.7329 | 11.84 | 0.3010 |
| Laplace | 79.72 | 0.5958 | 7.11 | 0.2784 |
| BLoB | 79.34 | 0.5683 | 6.82 | 0.2762 |
| TFB | 79.62 | 0.5739 | 6.85 | 0.2789 |
| C-LoRA | 78.36 | 0.5841 | 6.79 | 0.2883 |
| Seq-LoRA | 79.63 | **0.5224** | **5.33** | **0.2702** |

Compared with deterministic MAP LoRA, the latest Seq-LoRA table reduces:

- ECE from 12.66% to 5.33%: 57.9% relative reduction.
- NLL from 0.7785 to 0.5224: 32.9% relative reduction.

Seq-LoRA keeps macro accuracy close to the strongest accuracy baselines while
improving the probability-quality metrics most relevant to uncertainty
estimation.

## Benchmark Tasks

Source task:

- `scienceqa_closedchoice_grade2_11`

Evaluation tasks:

- `iid`: ScienceQA grades 2-11 test split.
- `scienceqa_closedchoice_grade12`
- `obqa`
- `arc-c`
- `mmlu_science_high`
- `mmlu_science_college`
- `gpqa_main`

Reported metrics:

- Accuracy.
- Negative log-likelihood.
- Expected calibration error.
- Brier score.
- Runtime and peak GPU memory where available.

The latest author-side table uses seeds `1, 3, 7, 11, 13` for most stochastic methods.
The ensemble baseline uses 5 independent ensemble groups, each containing 5 MAP
members from seeds `0-24`. This means the current reporting protocol is not a
perfectly matched seed protocol across all methods; when writing a paper table,
state the seed protocol explicitly.

## Repository Layout

Core Seq-LoRA implementation:

- `seq_eval_iid_constantq.py`: main Seq-LoRA posterior construction and
  Bayesian evaluation script.
- `seq_lora_subspace_obs.py`: subspace construction, curvature projection, and
  LGSSM observation helpers.
- `lssm_ffbs_obs.py`: Kalman filtering / state-space posterior utilities.
- `kfac.py`: curvature-related utilities.
- `mcqa_slices.py`: slicing utilities for MCQA experiments.

Training and deterministic evaluation:

- `train_scienceqa_qwen35_9b_lora_map_leftpad.py`: ScienceQA MAP LoRA training
  for Qwen3-8B.
- `train_closedchoice_qwen35_8b_lora_map_leftpad.py`: closed-choice Qwen LoRA
  training path.
- `train_closedchoice_llama2_7b_lora_map_leftpad.py`: closed-choice Llama-2
  LoRA training path.
- `map_eval.py`: MAP adapter evaluation.
- `mcdrop_eval.py`: MC-Dropout evaluation.
- `map_ensemble_eval.py`: MAP ensemble evaluation.
- `laplace_lora_official_source_eval.py`: official-source Laplace evaluation
  on LoRA adapters.

Benchmark orchestration and artifacts:

- `run_scienceqa_benchmark_suite.py`: end-to-end benchmark launcher and result
  parser.
- Generated benchmark outputs are written locally under
  `benchmark_suite_scienceqa/` by default.

Third-party and baseline code:

- `third_party/`
- `laplace/`

## Environment

Use Python 3.12 and install a PyTorch build that matches the target GPU/CUDA
runtime. The development environment used CUDA-enabled PyTorch and bf16/fp16
mixed precision.

For the benchmark stack:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-benchmark.txt
```

For Laplace experiments:

```bash
python -m pip install -r requirements-laplace.txt
```

`requirements.txt` records the full local working environment used during the
latest experiments. It is intentionally more specific than
`requirements-benchmark.txt`.

The first run may download:

- `Qwen/Qwen3-8B-Base`
- ScienceQA
- OpenBookQA
- ARC-Challenge
- MMLU Science subsets
- GPQA-main

If running on a machine with pre-downloaded assets, point Hugging Face cache
variables at the local cache:

```bash
export HF_HOME=/path/to/hf_home
export HF_DATASETS_CACHE=/path/to/hf_datasets_cache
```

## Quickstart

Train the ScienceQA MAP LoRA adapter:

```bash
python train_scienceqa_qwen35_9b_lora_map_leftpad.py
```

Evaluate a MAP adapter:

```bash
python map_eval.py \
  --map_adapter_dir iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/seed_1/map_step_2000 \
  --eval_tasks iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main \
  --max_seq_len 300 \
  --eval_bsz 32
```

Run Seq-LoRA on a MAP adapter:

```bash
python seq_eval_iid_constantq.py \
  --task scienceqa_closedchoice_grade2_11 \
  --map_dir iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/seed_1/map_step_2000 \
  --eval_tasks iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main \
  --q_mode module_constant \
  --s_q 1.0 \
  --forecast_horizon 0 \
  --kfac_bsz 8 \
  --eval_bsz 48 \
  --n_kfac 16 \
  --subspace_dim_per_module 64 \
  --mc_eval_samples 32 \
  --max_seq_len 300
```

Reuse expensive posterior statistics during tau/temperature/MC sweeps:

```bash
python seq_eval_iid_constantq.py \
  --task scienceqa_closedchoice_grade2_11 \
  --map_dir iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/seed_1/map_step_2000 \
  --eval_tasks iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu_science_high,mmlu_science_college,gpqa_main \
  --posterior_stats_cache_path caches/seq_lora_seed1_stats.pt \
  --q_mode module_constant \
  --s_q 1.0 \
  --mc_eval_samples 32
```

Run the ScienceQA benchmark suite:

```bash
python run_scienceqa_benchmark_suite.py
```

For slice-order ablations, use `--slice_order sorted`, `--slice_order reverse`,
or `--slice_order shuffle` in `seq_eval_iid_constantq.py`, optionally with
`--random_num_slices` and `--slice_order_seed`.

## Implementation Notes

Seq-LoRA includes several compute and memory optimizations:

- LoRA-only posterior construction over trainable adapter modules.
- K-FAC extraction per training slice with randomized low-rank PSD compression
  for large Kronecker blocks.
- Per-slice K-FAC sample caps and configurable subspace dimension per module.
- Posterior-stat cache for K-FAC/subspace/gradient-observation stages.
- bf16/fp16 autocast, SDPA attention, and TF32 matmul where available.
- Dynamic padding, sequence-length sorting, and padding to multiples of 8.
- Per-batch trimming of left-padded inputs to the active maximum sequence
  length during Seq-LoRA MC evaluation.
- In-place LoRA perturbation updates with `torch._foreach_add_` and
  `torch._foreach_sub_`, avoiding model copies across MC samples.
- MC sample chunking to control peak GPU memory.
- CPU offload of slice curvature statistics between posterior-build stages.

## Reproducing Tables

The benchmark runner parses per-method logs into CSV/JSON summaries. By default,
it writes outputs to `benchmark_suite_scienceqa/`.

Typical generated outputs include:

- `all_metrics.csv` and `all_metrics.json`
- `summary_mean_sd.csv`
- `command_times.csv`
- `training_resource_summary.csv`
- per-method log files under `logs/`

## Notes

- Large checkpoints, local Hugging Face caches, and benchmark artifacts can be
  substantial. Keep them out of commits unless intentionally publishing model
  artifacts.
- Some author-side experiment summaries used for the manuscript are local
  generated artifacts rather than required source files.
- This repository assumes local access to the required base model and datasets,
  or internet access to download them through Hugging Face.
