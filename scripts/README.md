# Experiment Command Entrypoints

Each shell script writes its logs to the matching `logs/...` directory and
exposes the usual knobs through environment variables such as `SEEDS`,
`EVAL_TASKS`, `EVAL_BSZ`, and `PY`. Benchmark methods are split into one method
per entrypoint.

## How To Read This Directory

Use `scripts/` as the source of truth for paper reproduction commands. The
Python modules implement methods; the shell scripts bind those methods to the
seeds, settings, checkpoint paths, evaluation tasks, cache paths, and log paths
of the reported runs.

All scripts source `scripts/_common.sh`, which:

- changes into the repository root (located relative to `scripts/`);
- sets `PY` to `python` unless `PY` (or `PYTHON`) is already set;
- sets `MPLCONFIGDIR` for headless plotting;
- provides `run_and_log`, which records `[RUN]`, `[CMD]`, command output, and
  `[DONE]` into the target log file.

Main entrypoints:

- `qwen3/scienceqa/scienceqa_main/`: Qwen3 ScienceQA main benchmark.
- `qwen3/scienceqa/scienceqa_ablations/`: slice, T, and tau ablations.
- `qwen3/scienceqa/map_training_order/`: deterministic MAP order ablation.
- `qwen3/obqa_transfer/`: Qwen3 OBQA-to-ARC/MMLU transfer benchmark.
- `llama2/obqa_transfer/`: Llama-2 OBQA-to-ARC/MMLU transfer benchmark.

Examples:

```bash
bash scripts/qwen3/scienceqa/scienceqa_main/run_seq.sh
bash scripts/qwen3/scienceqa/scienceqa_main/run_map.sh
bash scripts/qwen3/scienceqa/scienceqa_main/run_mcdrop.sh
VARIANTS="lgssm_grade q0_pooled ind_grade" bash scripts/qwen3/scienceqa/scienceqa_ablations/slice_structure/run.sh
```

## Reproducing Paper Experiments

The launchers can be started from any directory (they change into the
repository root); run the `python -m` commands below from the repository root.
Optionally pick the interpreter first:

```bash
export PY=/path/to/env/bin/python
```

If MAP checkpoints and slice datasets already exist, skip the preparation step.

### 1. Preparation

Train the Qwen3 ScienceQA MAP adapters used by the main table and ablations:

```bash
SEEDS=1,3,7,11,13 TRAIN_ORDERS=order \
  bash scripts/qwen3/scienceqa/map_training_order/run_train.sh

SEEDS=1,3,7,11,13 TRAIN_ORDERS="reverse random" \
  bash scripts/qwen3/scienceqa/map_training_order/run_train.sh
```

The ScienceQA ensemble row uses 25 MAP adapters (seeds 0-24) grouped into five
ensembles; train the seeds not covered above:

```bash
SEEDS=0,2,4,5,6,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24 TRAIN_ORDERS=order \
  bash scripts/qwen3/scienceqa/map_training_order/run_train.sh
```

Build the ScienceQA slice datasets (the exact slices used for the paper; add
`--preset all` for the dense `T` grid of the appendix sweeps):

```bash
"${PY:-python}" -m seq_lora.data.make_scienceqa_slices --preset main
"${PY:-python}" -m seq_lora.data.make_scienceqa_slices --preset all
```

### 2. Qwen3 ScienceQA Main Benchmark

```bash
for method in base map mcdrop ensemble laplace tfb blob clora seq; do
  bash "scripts/qwen3/scienceqa/scienceqa_main/run_${method}.sh"
done
```

Logs are written to `logs/qwen3/scienceqa/scienceqa_main/<method>/`.

### 3. Qwen3 ScienceQA Ablations

```bash
bash scripts/qwen3/scienceqa/map_training_order/run_eval.sh
bash scripts/qwen3/scienceqa/scienceqa_ablations/slice_structure/run.sh
bash scripts/qwen3/scienceqa/scienceqa_ablations/t_granularity/run.sh
bash scripts/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/run.sh

# Fixed-tau sensitivity at T = 10 and its figure (also reads the tau = 0 row
# written by slice_structure/run.sh).
TS=10 MODES="tau0p1 tau0p2 tau0p3 tau0p4 tau05 tau0p6 tau0p7 tau0p8 tau0p9 tau1" \
  bash scripts/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/run.sh
"${PY:-python}" scripts/qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/plot_t10_tau_grid.py
```

Logs are written to `logs/qwen3/scienceqa/map_training_order/` and
`logs/qwen3/scienceqa/scienceqa_ablations/`.

### 4. Qwen3 OBQA-to-ARC/MMLU Transfer

These scripts assume the Qwen3 OBQA MAP checkpoints (`init_lora.pt` and
`map_step_2000/` per seed) already exist under
`iid_qwen35_8b_obqa_lora_map_leftpad/obqa_qv_lmhead_leftpad/`.

```bash
for method in map mcdrop ensemble laplace tfb blob clora seq; do
  bash "scripts/qwen3/obqa_transfer/run_${method}.sh"
done
```

Logs are written to `logs/qwen3/obqa_transfer/<method>/`.

### 5. Llama-2 OBQA-to-ARC/MMLU Transfer

These scripts assume the Llama-2 OBQA MAP/MLE checkpoints already exist under
`baselines/bayesian-peft/checkpoints/mle/meta-llama/Llama-2-7b-hf/obqa/`.

```bash
for method in map mcdrop ensemble laplace clora seq; do
  bash "scripts/llama2/obqa_transfer/run_${method}.sh"
done
```

Logs are written to `logs/llama2/obqa_transfer/<method>/`.

## Entrypoint Reference

### Qwen3 ScienceQA main benchmark

| Script | Python entrypoint | Purpose |
| --- | --- | --- |
| `qwen3/scienceqa/scienceqa_main/run_base.sh` | `python -m baselines.base_eval` | Frozen Qwen3 base model over all ScienceQA IID/OOD tasks. |
| `qwen3/scienceqa/scienceqa_main/run_map.sh` | `python -m baselines.map_eval` | Deterministic MAP LoRA evaluation. |
| `qwen3/scienceqa/scienceqa_main/run_mcdrop.sh` | `python -m baselines.mcdrop_eval` | Dropout MC samples from the MAP adapter. |
| `qwen3/scienceqa/scienceqa_main/run_ensemble.sh` | `python -m baselines.ensemble_eval` | Five MAP ensemble groups, each with five members. |
| `qwen3/scienceqa/scienceqa_main/run_laplace.sh` | `python -m baselines.laplace.eval` | Laplace approximation over LoRA parameters. |
| `qwen3/scienceqa/scienceqa_main/run_blob.sh` | `baselines/bayesian-peft/qwen3/run/main.py --modelwrapper blob` | Qwen3 BLoB baseline. |
| `qwen3/scienceqa/scienceqa_main/run_tfb.sh` | `baselines/bayesian-peft/qwen3/run/main.py --modelwrapper tfblora` | Qwen3 TFB baseline from MAP adapters. |
| `qwen3/scienceqa/scienceqa_main/run_clora.sh` | `baselines/c_lora/qwen3/run/main.py --modelwrapper c_lora` | Qwen3 C-LoRA baseline. |
| `qwen3/scienceqa/scienceqa_main/run_seq.sh` | `python -m seq_lora.eval` | Seq-LoRA default `T=10` result. |

### Qwen3 ScienceQA ablations

| Script | What it changes |
| --- | --- |
| `qwen3/scienceqa/map_training_order/run_train.sh` | Trains MAP adapters under `order`, `reverse`, and `random` example orders. |
| `qwen3/scienceqa/map_training_order/run_eval.sh` | Evaluates reverse/random MAP adapters. |
| `qwen3/scienceqa/scienceqa_ablations/slice_structure/run.sh` | Compares grade slices, random slices, reverse slices, shuffled grade slices, base-NLL easy-hard slices, `Q=0`, and independent-slice posterior ensembles; the `map` variant is the same pipeline at `tau = 0`. |
| `qwen3/scienceqa/scienceqa_ablations/t_granularity/run.sh` | Runs Seq-LoRA for `T=1,5,10,20`. |
| `qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/run.sh` | Runs the dense `T` grid under `auto`, fixed `tau=0.5`, and fixed `tau=1.0` (`MODES` also accepts `tau0pN` for fixed `tau = 0.N`). |
| `qwen3/scienceqa/scienceqa_ablations/tau_t_sweep/plot_t10_tau_grid.py` | Plots the `T=10` fixed-`tau` sensitivity grid from the sweep logs. |

### Transfer benchmarks

| Directory | Purpose |
| --- | --- |
| `qwen3/obqa_transfer/run_*.sh` | Qwen3 source OBQA to OBQA IID, ARC-Easy, ARC-Challenge, MMLU high/college science. |
| `llama2/obqa_transfer/run_*.sh` | Llama-2 source OBQA to OBQA IID, ARC-Easy, ARC-Challenge, MMLU-chem, MMLU-phy. |

## Common Overrides

| Variable | Applies to | Meaning |
| --- | --- | --- |
| `PY` | all scripts | Python interpreter. |
| `SEEDS` | most scripts | Seed list. `run_train.sh` expects comma-separated seeds; most eval scripts accept space-separated seeds. |
| `TRAIN_ORDERS` | `map_training_order/run_train.sh` | MAP training order variants. |
| `EVAL_TASKS` | base/MAP/MC-Dropout/ensemble/Laplace/Seq-LoRA | Evaluation task list. |
| `EVAL_DATASET` | BLoB/TFB/C-LoRA | Upstream-runner evaluation dataset aliases. |
| `MAP_ROOT` | adapter-based methods | Root containing `seed_<seed>/map_step_2000`. |
| `LOG_ROOT` | all scripts | Output log directory; set it to keep reruns separate. |
| `CACHE_ROOT` | Seq-LoRA scripts | Posterior-statistics cache directory. |
| `EVAL_BSZ` | most scripts | Evaluation batch size. |
| `MC_SAMPLES` | MC methods | Number of MC samples. |
| `LOAD_IN_8BIT` | Llama2 scripts | Load base model/checkpoint in 8-bit when supported. |

For a smoke test, use a single seed and a new log root:

```bash
SEEDS="1" LOG_ROOT=logs/smoke/qwen3_seq \
  bash scripts/qwen3/scienceqa/scienceqa_main/run_seq.sh
```
