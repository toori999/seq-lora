# Bayesian-PEFT-Style Layout

This directory mirrors the folder split requested for the refactor:

- `datasets/`
- `model/`
- `modelwrappers/`
- `run/`
- `scripts/`
- `utils/`

It is intentionally additive. Existing project files and the earlier
`refactor/seq_lora/` package remain untouched.

## Main Entry Point

```bash
python -m refactor.bayesian_peft_layout.run.main --help
python -m refactor.bayesian_peft_layout.run.train_map --help
python -m refactor.bayesian_peft_layout.run.benchmark_suite --help
```

## What Is Covered

The refactor entrypoint now centralizes these methods behind `run/main.py`:

- `map`
- `mcdrop`
- `deep-ensemble`
- `prob-ensemble`
- `seq-constantq`
- `laplace`
- `blob`

`run/benchmark_suite.py` uses the same unified CLI, so the orchestration no
longer calls the legacy evaluation scripts directly.

MAP training variants are launched through `run/train_map.py`, which wraps the
existing ScienceQA training scripts behind a stable refactor-facing CLI. It now
also exposes an experimental `--backend refactor` path that runs a training loop
implemented inside this refactor layout. The benchmark suite still pins training
to the legacy backend until that path is fully validated.

## Dataset And Model Helpers

The folder split is now carrying more training-side logic too:

- `datasets/scienceqa_curriculum.py` handles ScienceQA curriculum split loading,
  grade-ordering, and slice export helpers.
- `datasets/tokenization.py` carries training-side ScienceQA prompt/tokenization
  preprocessing.
- `model/lora_training.py` centralizes LoRA target-module discovery, shared init
  state handling, and adapter state-dict helpers.
- `model/training_eval.py` collects training-time choice-logit and validation
  scoring helpers.
- `model/train_runtime.py` and `model/map_training.py` now hold the training-side
  runtime setup and the refactored ScienceQA MAP training loop.
