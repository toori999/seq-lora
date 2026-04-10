# Evaluation Refactor

This package is a staging area for consolidating the repeated logic currently
spread across:

- `map_eval.py`
- `mcdrop_eval.py`
- `ens_eval.py`
- `map_ensemble_eval.py`

## What Lives Here

- `common.py`: device helpers, timers, eval task parsing, loader preparation
- `adapter_loading.py`: base model + LoRA adapter loading and lm-head trimming
- `methods.py`: method-specific evaluation routines for MAP, MC-Dropout, and ensembles

## Intent

The old scripts remain untouched. New or migrated evaluation CLIs can import
these helpers and stay much thinner than the original single-file scripts.

## Current Entry Point

The new refactor CLI lives at:

```bash
python -m refactor.seq_lora.cli.evaluate --help
```
