# Refactor Sandbox

This directory contains a non-invasive refactor of the current utility surface.
The existing top-level scripts and modules are left untouched on purpose.

## Scope

The first pass focuses on splitting the old monolithic `common_eval_utils.py`
into smaller modules under `refactor/seq_lora/`:

- `constants.py`: task and prompt constants
- `prompts.py`: choice labels, answer mapping, prompt builders
- `datasets.py`: dataset loading and normalization
- `preprocessing.py`: task-specific preprocessing
- `collators.py`: dynamic collators
- `metrics.py`: metric builders and choice token ids
- `peft_utils.py`: LoRA / adapter helpers
- `choice_head.py`: choice-head cache and restricted-logit paths
- `eval/`: shared evaluation framework extracted from MAP / MCDrop / ensemble scripts

## Migration Idea

Code that currently imports from `common_eval_utils` can be migrated gradually:

```python
from refactor.seq_lora import load_eval_dataset, preprocess_task, ChoiceHeadCache
```

The original module remains the source of truth for now; this folder is a
staging area for cleaner structure and future migration.

## New Unified Eval CLI

You can now exercise the new evaluation refactor without touching the legacy
scripts:

```bash
python -m refactor.seq_lora.cli.evaluate map --task obqa --adapter-dir /path/to/adapter
python -m refactor.seq_lora.cli.evaluate mcdrop --task obqa --adapter-dir /path/to/adapter --mc-samples 32
python -m refactor.seq_lora.cli.evaluate deep-ensemble --task obqa --adapter-dirs /a,/b,/c
python -m refactor.seq_lora.cli.evaluate prob-ensemble --task obqa --adapter-dirs /a,/b,/c
```

## Bayesian-PEFT-Style Layout

There is now a second additive layout under `refactor/bayesian_peft_layout/`
that mirrors the folder split you asked for:

- `datasets/`
- `model/`
- `modelwrappers/`
- `run/`
- `scripts/`
- `utils/`
