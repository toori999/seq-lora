# Llama-2 OBQA Transfer Commands

These scripts reproduce the Llama-2 OBQA-source transfer experiments: the
source task is OBQA and the targets are OBQA (ID), ARC-Easy, ARC-Challenge,
MMLU-chem, and MMLU-phy, with seeds 1, 3, 7. Logs are written to
`logs/llama2/obqa_transfer/<method>/`.

Default checkpoint root for MAP-like methods:

```text
baselines/bayesian-peft/checkpoints/mle/meta-llama/Llama-2-7b-hf/obqa/
```

Expected adapter directory pattern:

```text
lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed<seed>
```

These MAP (MLE) adapters were trained with the Bayesian-PEFT `mle` wrapper using
the settings of `baselines/bayesian-peft/scripts/map/map-llama-all-2000-local.sh`
(lr 5e-5, batch size 4, LoRA rank 8 / alpha 16 / dropout 0.1 on `q_proj`,
`v_proj`, and `lm_head`, 2000 steps) with `--dataset obqa`, seeds 1, 3, 7, and
`--log-path lora-obqa-lr5e-5-bs4-drop0.1-step2000-seed<seed>`; run it from
`baselines/bayesian-peft/`.

Entry points:

- `run_seq.sh`: Seq-LoRA random-slice OBQA transfer.
- `run_laplace.sh`: official-source Laplace-LoRA evaluation.
- `run_map.sh`: deterministic MAP evaluation from saved OBQA LoRA checkpoints.
- `run_mcdrop.sh`: MC-Dropout evaluation from saved OBQA LoRA checkpoints.
- `run_ensemble.sh`: ensemble evaluation from saved OBQA LoRA checkpoints.
- `run_clora.sh`: C-LoRA official-source OBQA transfer evaluation.

The BLoB and TFB rows come from the Bayesian-PEFT runner itself
(`baselines/bayesian-peft/run/main.py`). TFB was applied to the MLE adapters
above with `baselines/bayesian-peft/scripts/tfblora/mle-llama2-iid-single-gpu.sh`;
BLoB was trained with `--modelwrapper blob` and the settings encoded in its
checkpoint name, `blob-obqa-lr5e-5-bs4-drop0.1-step2000-sample10-eps0.05-kllr0.01-beta0.2-gamma8-seed<seed>`.
Both write their checkpoints and evaluation logs under
`baselines/bayesian-peft/checkpoints/{blob,tfblora}/meta-llama/Llama-2-7b-hf/obqa/`.

## Script Details

| Script | Python entrypoint | Output |
| --- | --- | --- |
| `run_map.sh` | `python -m baselines.bayesian_peft_checkpoint_eval --method map` | `logs/llama2/obqa_transfer/map/source_obqa__eval_<target>__seed<seed>.log` |
| `run_mcdrop.sh` | `python -m baselines.bayesian_peft_checkpoint_eval --method mcdrop` | `logs/llama2/obqa_transfer/mcdrop/source_obqa__eval_<target>__seed<seed>.log` |
| `run_ensemble.sh` | `python -m baselines.bayesian_peft_checkpoint_eval --method ens` | Ensembles over seed sets `1,3,7`, `1,3`, `3,7`; logs plus `ensemble_runs.csv` and `summary_mean_sd.csv`. |
| `run_laplace.sh` | `python -m baselines.laplace.eval` | separate IID and OOD logs under `laplace/`. |
| `run_clora.sh` | `cd baselines/c_lora && python run/main.py --modelwrapper c_lora` | Trains C-LoRA once per seed (OOD logs), then reloads it for the OBQA evaluation; logs under `clora/`. |
| `run_seq.sh` | `python -m seq_lora.eval --eval_protocol bayesian_peft` | Seq-LoRA OBQA and OOD logs under `seq/` (two posterior builds per seed, as in the paper). |

## Reproduce All Llama-2 Transfer Runs

```bash
for method in map mcdrop ensemble laplace clora seq; do
  bash "scripts/llama2/obqa_transfer/run_${method}.sh"
done
```

## Common Overrides

| Variable | Meaning |
| --- | --- |
| `PY` | Python interpreter. |
| `MODEL` | Base model name, default `meta-llama/Llama-2-7b-hf`. |
| `CKPT_ROOT` | Checkpoint root for MAP/MC-Dropout/ensemble scripts. |
| `MAP_ROOT` | Checkpoint root for Laplace/Seq-LoRA scripts. |
| `LOG_ROOT` | Log output directory. Set a new value for reruns. |
| `SEEDS` | Seed list, default `1 3 7`. |
| `ENSEMBLE_SEED_SETS` | Ensemble seed sets, default `1,3,7;1,3;3,7`. |
| `DATASETS` | MAP/MC-Dropout/ensemble target list. |
| `OOD_TASKS` | Laplace OOD target list. |
| `EVAL_TASKS` | Seq-LoRA target list. |
| `MC_SAMPLES` | MC sample count for MC-Dropout/C-LoRA. |
| `LOAD_IN_8BIT` | Enable 8-bit loading where supported. |
| `CACHE_ROOT` | Seq-LoRA posterior statistics cache path. |

Example rerun into a separate log directory:

```bash
LOG_ROOT=logs/llama2/obqa_transfer/seq_rerun \
  bash scripts/llama2/obqa_transfer/run_seq.sh
```
