This directory vendors patched snapshots of external method repositories used for benchmark integration.

- `bayesian-peft/`: snapshot based on `Wang-ML-Lab/bayesian-peft`, adapted for ScienceQA benchmark train/test/OOD evaluation, trimmed `lm_head`, and benchmark-style timing/resource logging.
- `c_lora/`: snapshot based on `ahra99/c_lora`, adapted for the same benchmark dataset interface and logging conventions.

These copies are included here so the exact benchmark-facing source used in experiments can be versioned inside this repository.
