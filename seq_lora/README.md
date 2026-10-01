# Seq-LoRA Python Package

This package contains the method implementation used by the paper-facing
experiment launchers in `scripts/`.

## Reader Guide

`seq_lora/` contains only the Seq-LoRA method and its data/CLI helpers. It does
not train the MAP anchor adapter; use `training/scienceqa_map.py` for that. It
also does not implement competing baselines; use `baselines/` for Base, MAP,
MC-Dropout, ensemble, Laplace, BLoB, TFB, and C-LoRA.

## Layout

- `seq_lora.algorithm`: Seq-LoRA method code: K-FAC extraction, pseudo-
  observations, subspace construction, process-noise construction, posterior
  sampling, and LGSSM/Kalman inference.
- `seq_lora.data`: ScienceQA slice construction and slice-loading utilities.
- `seq_lora.eval`: Seq-LoRA command-line runner. Shared benchmark, prompt,
  metric, and protocol utilities live in top-level `evaluation/`.

## File-Level Map

| File | Purpose |
| --- | --- |
| `algorithm/kfac.py` | Computes per-slice Kronecker factors for LoRA modules. |
| `algorithm/asdl_backend.py` | Bridges the model modules to the ASDL curvature backend. |
| `algorithm/subspace.py` | Builds and materializes low-dimensional curvature eigenspaces. |
| `algorithm/pseudo_observations.py` | Converts slice gradients into pseudo-observation means. |
| `algorithm/process_noise.py` | Builds the LGSSM process-noise covariance `Q`. |
| `algorithm/lgssm.py` | Implements Kalman filtering/smoothing for the linear Gaussian state-space model. |
| `algorithm/posterior.py` | Samples final Seq-LoRA posterior coordinates or independent-slice ensemble samples. |
| `algorithm/lora_state.py` | Caches LoRA factors and applies/removes posterior perturbations during MC evaluation. |
| `algorithm/tau.py` | Calibrates posterior scale by matching anchor-set KL targets. |
| `data/make_scienceqa_slices.py` | Builds the paper's ScienceQA slice datasets and control slices. |
| `data/paper_slice_orders.json.gz` | Exact row order of the paper's slice datasets (indices into the ScienceQA train split). |
| `data/mcqa_slices.py` | Builds loss/easy-hard based MCQA slices. |
| `data/slices.py` | Provides random slice assignment utilities. |
| `eval/runner.py` | Main `python -m seq_lora.eval` implementation. |
| `eval/cache.py` | Saves and validates posterior-statistics caches. |
| `eval/__main__.py` | Thin CLI entrypoint. |

## CLI Entry Points

Use module execution from the repository root:

```bash
python -m seq_lora.data.make_scienceqa_slices --preset main --overwrite true
python -m seq_lora.eval --task scienceqa_closedchoice_grade2_11 --eval_protocol default ...
python -m seq_lora.eval --task obqa --eval_protocol bayesian_peft ...
```

The full paper command templates live under `scripts/qwen3/` and
`scripts/llama2/`.

MAP anchor training is intentionally kept outside this package:

```bash
python -m training.scienceqa_map --train_order order --seeds 0
```

## Main Seq-LoRA CLI Options

| Option | Meaning |
| --- | --- |
| `--task` | Source task, currently `scienceqa_closedchoice_grade2_11` or `obqa`. |
| `--eval_protocol` | `default` for Qwen3 experiments; `bayesian_peft` for Llama2 transfer compatibility. |
| `--map_dir` | MAP LoRA adapter directory used as the posterior mean/anchor. |
| `--slices_dir` | Explicit slice dataset directory, used by ScienceQA experiments. |
| `--random_num_slices` | Randomly split the source training set when no explicit slice directory is supplied. |
| `--slice_order` | `sorted`, `reverse`, or `shuffle` for slice-order controls. |
| `--kfac_backend` | Curvature backend; paper scripts use `asdl`. |
| `--kfac_bsz` | Batch size for slice curvature extraction. |
| `--q_mode` | Process-noise mode, usually `module_constant`. |
| `--posterior_eval_mode` | `lgssm_final` for Seq-LoRA, `independent_slice_ensemble` for a control. |
| `--mc_eval_samples` | Number of posterior samples used for Bayesian averaging. |
| `--mc_eval_chunk` | Chunk size for applying samples to the model. |
| `--tau_mode` | `auto` for anchor-KL calibration, or `fixed` with `--posterior_tau`. |
| `--posterior_stats_cache_path` | Cache path for expensive K-FAC/subspace/pseudo-observation statistics. |

## Slice Presets

| Preset | Builds | Used by |
| --- | --- | --- |
| `main` | `T=1,5,10,20` plus structure controls | Main table and standard ablations. |
| `t_ablation` | Only `T=1,5,10,20` | T-granularity ablation. |
| `structure_controls` | reverse/shuffled/full-shuffle/base-NLL controls | Slice-structure ablation. |
| `dense_tau` | Dense T grid | Tau/T diagnostic figures. |
| `all` | Every dataset above | Full paper reproduction. |

By default the slices are rebuilt with the exact row order used for the paper
(`data/paper_slice_orders.json.gz`, checked against a checksum of the ScienceQA
training split). The order within a slice decides which examples form its
pseudo-observation batches. Pass `--use_paper_orders false` to regenerate every
partition from its rule instead.
