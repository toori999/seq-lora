# Seq-LoRA

This repository contains the ScienceQA MAP, Seq-LoRA, BLoB, MC-Dropout, and Laplace evaluation code used in the current experiments.

The repository already includes the order-MAP checkpoints for:

- `seed_0`
- `seed_1`
- `seed_2`
- `seed_3`
- `seed_4`

under [iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad](/home/tori/projects/Seq-LoRA/Seq_LoRA/iid_qwen35_8b_scienceqa_lora_map_leftpad/scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad).

## Paperspace Quickstart

These steps are intended for a fresh Paperspace / DigitalOcean GPU machine.

1. Clone the repo:

```bash
git clone git@github.com:toooooori/Seq-LoRA.git
cd Seq-LoRA
```

2. Create the Python environment and install the tested Laplace dependencies:

```bash
bash scripts/setup_paperspace_laplace.sh
source .venv-laplace/bin/activate
```

3. Run Laplace on a specific seed:

```bash
bash scripts/run_laplace_seed.sh 0
```

By default this uses:

- `--eval_tasks "iid,scienceqa_closedchoice_grade12,obqa,arc-c,mmlu,gpqa_main"`
- `--laplace_sub all`
- `--testing_set val`
- `--seed 0`
- `--fit_bsz 2`
- `--laplace_bsz 1`
- `--prior_optim_step 100`
- `--laplace_mc_samples 48`
- `--laplace_mc_chunk 8`
- `--max_length 300`

Logs are written to `laplace_seed<seed>.log`.

## Notes

- First run on a fresh machine will still download the base model `Qwen/Qwen3-8B-Base` and evaluation datasets from Hugging Face.
- The local cache fallbacks in [common_eval_utils.py](/home/tori/projects/Seq-LoRA/Seq_LoRA/common_eval_utils.py) only help if those datasets have already been downloaded once on that machine.
- The benchmark outputs, Laplace caches, and large local experiment artifacts are intentionally excluded from Git.
