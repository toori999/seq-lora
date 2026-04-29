# Benchmark Notes For Paper Writing

## 1. What This Benchmark Evaluates

- Source task: `scienceqa_closedchoice_grade2_11`
- Source-task IID evaluation split: `test`
- OOD targets:
  - `scienceqa_closedchoice_grade12`
  - `obqa`
  - `arc-c`
  - `mmlu_science_high`
  - `mmlu_science_college`
  - `gpqa_main`
- Core benchmark script: [run_scienceqa_benchmark_suite.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/run_scienceqa_benchmark_suite.py)

## 2. Backbone And Prediction Format

- Backbone model: `Qwen/Qwen3-8B-Base`
- Maximum input length: `300`
- Prediction format: text-only multiple choice, left-padded, last-token classification over answer tokens `A/B/C/D`
- The LM head is trimmed to 4 choice logits and invalid options are masked
- Main MAP LoRA target modules:
  - all `q_proj`
  - all `v_proj`
  - `lm_head`
- Main MAP LoRA hyperparameters:
  - rank `r = 8`
  - `alpha = 16`
  - dropout `0.05`

Relevant code:

- [train_scienceqa_qwen35_9b_lora_map_leftpad.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/train_scienceqa_qwen35_9b_lora_map_leftpad.py)
- [common_eval_utils.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/common_eval_utils.py)

## 3. Task List And Sample Counts

The task sizes visible in the benchmark logs are:

| Report name | Internal task name | # examples |
| --- | --- | ---: |
| `iid` | `scienceqa_closedchoice_grade2_11(test)` | `2063` |
| `grade12` | `scienceqa_closedchoice_grade12(ood)` | `310` |
| `obqa` | `obqa(ood)` | `500` |
| `arc-c` | `arc-c(ood)` | `1172` |
| `mmlu_science_high` | `mmlu_science_high(ood)` | `664` |
| `mmlu_science_college` | `mmlu_science_college(ood)` | `346` |
| `gpqa_main` | `gpqa_main(ood)` | `448` |

Source:

- [laplace_seed1.log](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/laplace/laplace_seed1.log:95)

## 4. Methods Included In The Main Comparison

- `base`: frozen pretrained backbone, no task adaptation
- `map`: deterministic order-MAP LoRA checkpoint
- `mcdrop`: MC-Dropout evaluation on the MAP adapter
- `ens`: ensemble of MAP adapters
- `laplace`: official-source Laplace applied to the MAP adapter
- `blob sample`: BLoB stochastic prediction (`samp`, not `mean`)
- `tfb sample`: TFB stochastic prediction
- `clora sample`: C-LoRA stochastic prediction (`samp`, not `mean`)
- `seq`: Bayesian Seq-LoRA

## 5. Exact Training / Inference Settings

### 5.1 Order-MAP

- Checkpoint used for evaluation: `map_step_2000`
- Training batch config:
  - `micro_bsz = 4`
  - `grad_accum = 2`
  - effective batch size `8`
  - eval batch size `32`
- `NUM_WORKERS = 0`
- `USE_GRADIENT_CHECKPOINTING = False`
- `FAST_BUT_NONDETERMINISTIC = True`
- Curriculum order:
  - `order`: grade 2 -> grade 11
  - each grade block is shuffled with `seed + grade_num`

Relevant code:

- [train_scienceqa_qwen35_9b_lora_map_leftpad.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/train_scienceqa_qwen35_9b_lora_map_leftpad.py:60)
- [train_scienceqa_qwen35_9b_lora_map_leftpad.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/train_scienceqa_qwen35_9b_lora_map_leftpad.py:405)

### 5.2 MC-Dropout

- Evaluation script: [mcdrop_eval.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/mcdrop_eval.py)
- MC samples: `32`
- temperature: `1.0`
- In the suite command builder, the internal post-hoc eval seed is fixed to `0`; the variation comes from the underlying MAP adapters

### 5.3 Ensemble

- Ensemble script: [map_ensemble_eval.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/map_ensemble_eval.py)
- Each ensemble contains `5` MAP members
- Paper-clean ensemble protocol in the benchmark suite:
  - total MAP seeds: `25`
  - number of ensemble groups: `5`
  - groups are consecutive:
    - group 0: seeds `0-4`
    - group 1: seeds `5-9`
    - group 2: seeds `10-14`
    - group 3: seeds `15-19`
    - group 4: seeds `20-24`

Relevant code:

- [run_scienceqa_benchmark_suite.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/run_scienceqa_benchmark_suite.py:1356)

### 5.4 Laplace

- Script: [laplace_lora_official_source_eval.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/laplace_lora_official_source_eval.py)
- Uses the official-source Laplace path on top of MAP
- `laplace_sub = all`
- fitting split setting: `testing_set = val`
- fit batch size: `2`
- Laplace eval batch size: `4`
- prior optimization steps: `100`
- MC samples: `32`
- MC chunk size: `8`

### 5.5 BLoB

- Uses `shared_init_lora_path` from the corresponding MAP seed
- Training config from current logs:
  - `batch_size = 8`
  - `eval_batch_size = 48`
  - `max_train_steps = 2000`
  - `lr = 5e-5`
  - `weight_decay = 0.01`
  - `warmup_ratio = 0.06`
  - `anchor_size = 500`
  - `bayes_train_n_samples = 1`
  - `bayes_eval_n_samples_final = 32`
- In the comparison table, use `BLoB samp`, not `BLoB mean`

Source:

- [seed_1.log](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/thirdparty_blob_train_once_mc32/seed_1.log:38)

### 5.6 TFB

- Loads the MAP LoRA checkpoint directly
- Current log config:
  - `eval_batch_size = 48`
  - `anchor_size = 500`
  - `bayes_train_n_samples = 10`
  - `bayes_eval_n_samples_final = 32`
  - initial `bayes_beta = 0.015`
  - final fitted beta in seed-1 log: `0.003625`
  - `bayes_final_beta = 0.18`
  - `bayes_flipout = True`
- In the comparison table, use the stochastic TFB prediction

Source:

- [official_tfblora_bench_lora_seed1.log](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/tfb/logs/official_tfblora_bench_lora_seed1.log:46)

### 5.7 C-LoRA

- Uses `shared_init_lora_path` from the corresponding MAP seed
- Current log config:
  - `batch_size = 8`
  - `gradient_accumulation_steps = 1`
  - `eval_batch_size = 48`
  - `max_train_steps = 2000`
  - `lr = 5e-5`
  - `weight_decay = 0.01`
  - `warmup_ratio = 0.06`
  - `bayes_train_n_samples = 1`
  - `bayes_eval_n_samples_final = 32`
  - `bayes_beta = 0.2`
- In the comparison table, use `C-LoRA samp`, not `C-LoRA mean`

Source:

- [seed_1.log](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/thirdparty_clora_train_once_mc32_eval100/seed_1.log:20)

### 5.8 Seq-LoRA

- Script: [seq_eval_iid_constantq.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/seq_eval_iid_constantq.py)
- Benchmark configuration:
  - `q_mode = module_constant`
  - `s_q = 1.0`
  - `forecast_horizon = 0`
  - `KFAC_BSZ = 8`
  - `N_KFAC = 16`
  - `LR_THRESHOLD = 256`
  - `MAX_KFAC_SAMPLES_PER_SLICE = 2048`
  - `MU_OBS_SCALE = 2`
  - `MU_OBS_BATCHES = 32`
  - `P1_VAR = 1.0`
  - `SUBSPACE_DIM_PER_MODULE = 64`
  - `MC_EVAL_SAMPLES = 32`
  - `TEMP_BAYES = 1.0`
- The comparison table uses the Bayesian-only output block

Relevant code:

- [seq_eval_iid_constantq.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/seq_eval_iid_constantq.py:46)
- [seq_eval_iid_constantq.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/seq_eval_iid_constantq.py:890)

## 6. Metrics And Reporting

- Main metrics:
  - Accuracy (`ACC`, in %)
  - Negative log-likelihood (`NLL`)
  - Expected calibration error (`ECE`, in %)
  - Brier score (`Brier`)
- Mean and standard deviation are computed with sample standard deviation (`statistics.stdev`) when `n > 1`
- Current comparison artifacts:
  - [all_methods_mean_sd_table.md](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/all_methods_mean_sd_table.md)
  - [all_methods_mean_sd_stats.csv](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/all_methods_mean_sd_stats.csv)

## 7. Current Seed Protocols In The Latest Table

This is the most important caveat to state clearly before using the current table in a paper.

- `base`: single run from `seed 0`
- `map`, `mcdrop`, `laplace`, `blob sample`, `tfb sample`, `clora sample`, `seq`:
  - current latest table uses seeds `1, 3, 7, 11, 13`
- `ens`:
  - current latest table now uses `5` independent ensemble groups
  - each group contains `5` MAP members
  - the group members come from seeds `0-24`

This means the latest summary table is not a perfectly matched seed protocol across all methods.

If the paper requires a fully aligned reporting protocol, the safest options are:

1. Re-run every stochastic method on the same seed set.
2. Or report the original benchmark-suite numbers from [summary_mean_sd.csv](/home/tori/projects/Seq-LoRA/Seq_LoRA/benchmark_suite_scienceqa/summary_mean_sd.csv), which use the suite's native seed bookkeeping.

## 8. Current Result Snapshot (Latest `logs/` Table)

Macro-average over the 7 benchmark tasks from [all_methods_mean_sd_stats.csv](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/all_methods_mean_sd_stats.csv):

| method | macro ACC | macro NLL | macro ECE | macro Brier |
| --- | ---: | ---: | ---: | ---: |
| `base` | `70.9643` | `0.7197` | `9.1843` | `0.3865` |
| `map` | `79.8051` | `0.7785` | `12.6571` | `0.3113` |
| `mcdrop` | `79.7600` | `0.7767` | `12.7197` | `0.3108` |
| `ens` | `79.8940` | `0.7329` | `11.8420` | `0.3010` |
| `laplace` | `79.7200` | `0.5958` | `7.1071` | `0.2784` |
| `blob sample` | `79.3403` | `0.5683` | `6.8154` | `0.2762` |
| `tfb sample` | `79.6203` | `0.5739` | `6.8460` | `0.2789` |
| `clora sample` | `78.3560` | `0.5841` | `6.7860` | `0.2883` |
| `seq` | `79.6274` | `0.5224` | `5.3309` | `0.2702` |

Interpretation of the current latest table:

- Best macro `ACC`: `ens`
- Best macro `NLL`: `seq`
- Best macro `ECE`: `seq`
- Best macro `Brier`: `seq`

## 9. Training Resource Numbers You May Want In The Paper

Seed-1 training-stage numbers currently visible in the latest logs:

| method | train stage | time | peak alloc | peak reserve |
| --- | --- | ---: | ---: | ---: |
| `map` | MAP train | `1077.99s` | `N/A` | `N/A` |
| `laplace` | Laplace fit | `701.13s` | `60.06 GB` | `94.28 GB` |
| `blob` | BLoB fit | `1881.89s` | `28.87 GB` | `44.97 GB` |
| `tfb` | TFB fit | `135.44s` | `14.63 GB` | `16.03 GB` |
| `clora` | C-LoRA fit | `3603.29s` | `24.01 GB` | `30.56 GB` |
| `seq` | posterior build | `673.13s` | `23.56 GB` | `24.20 GB` |

Note:

- The current visible seed-1 MAP training artifact has time but not logged peak memory.

## 10. A Paper-Ready Methods Paragraph You Can Reuse

Suggested concise wording:

> We evaluate uncertainty-aware LoRA methods on a ScienceQA-centered OOD benchmark. The source task is text-only ScienceQA closed-choice classification over grades 2-11, and IID performance is measured on the source test split. We then evaluate transfer and calibration on six held-out targets: ScienceQA grade-12, OpenBookQA, ARC-Challenge, MMLU High-School Science, MMLU College Science, and GPQA-main. All methods use the same backbone, Qwen/Qwen3-8B-Base, and we cast prediction as left-padded last-token classification over the answer tokens A-D with a 4-way trimmed LM head and invalid-choice masking. We compare the frozen base model, deterministic MAP LoRA, MC-Dropout, deep ensembles, Laplace, BLoB, TFB, C-LoRA, and Seq-LoRA. We report accuracy, negative log-likelihood, expected calibration error, and Brier score, using mean ± standard deviation across random seeds or ensemble groups as applicable.

## 11. Files To Cite In Your Own Notes

- Benchmark runner: [run_scienceqa_benchmark_suite.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/run_scienceqa_benchmark_suite.py)
- Latest comparison table: [all_methods_mean_sd_table.md](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/all_methods_mean_sd_table.md)
- Raw stats table: [all_methods_mean_sd_stats.csv](/home/tori/projects/Seq-LoRA/Seq_LoRA/logs/all_methods_mean_sd_stats.csv)
- Original suite export: [summary_mean_sd.csv](/home/tori/projects/Seq-LoRA/Seq_LoRA/benchmark_suite_scienceqa/summary_mean_sd.csv)
