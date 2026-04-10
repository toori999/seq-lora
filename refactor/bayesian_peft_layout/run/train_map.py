from __future__ import annotations

import argparse

from refactor.bayesian_peft_layout.datasets import (
    load_scienceqa_train_eval_split,
    print_scienceqa_split_summary,
)
from refactor.bayesian_peft_layout.model import run_refactor_scienceqa_map_training
from refactor.bayesian_peft_layout.utils import (
    MAP_VARIANT_MODULES,
    apply_train_config_to_runner,
    build_scienceqa_map_train_config,
    load_map_variant_runner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a ScienceQA MAP training variant from the refactor layout."
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=sorted(MAP_VARIANT_MODULES),
        help="Training order variant to launch.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--micro-bsz", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-bsz", type=int, default=32)
    parser.add_argument(
        "--backend",
        type=str,
        default="legacy",
        choices=["legacy", "refactor"],
        help="Execution backend. Keep legacy for benchmark stability; use refactor to exercise the new training path.",
    )
    parser.add_argument(
        "--preview-dataset",
        action="store_true",
        help="Try loading the ScienceQA raw train/eval split summary before launching training.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the resolved training config and exit without starting training.",
    )
    return parser


def _maybe_preview_dataset(source_eval_split: str) -> None:
    try:
        train_raw, eval_raw = load_scienceqa_train_eval_split(source_eval_split)
    except Exception as exc:
        print(f"[Train Preview] dataset preview unavailable: {exc}")
        return
    print_scienceqa_split_summary("Train Raw", train_raw)
    print_scienceqa_split_summary("Eval Raw", eval_raw)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = build_scienceqa_map_train_config(
        variant=args.variant,
        seed=int(args.seed),
        micro_bsz=int(args.micro_bsz),
        grad_accum=int(args.grad_accum),
        eval_bsz=int(args.eval_bsz),
    )
    runner = apply_train_config_to_runner(
        load_map_variant_runner(args.variant),
        config,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[Train Launcher] "
        f"backend={args.backend} variant={config.variant} seed={config.seed} task={config.task} "
        f"micro_bsz={config.micro_bsz} grad_accum={config.grad_accum} "
        f"effective_train_bsz={config.effective_train_bsz} eval_bsz={config.eval_bsz}"
    )
    print(
        "[Train Launcher] "
        f"run_tag={config.run_tag} output_dir={config.output_dir} "
        f"slice_dir={config.slice_dir} source_eval_split={config.source_eval_split}"
    )
    print(
        "[Train Launcher] "
        f"run_dir={config.run_dir} map_dir={config.map_dir} "
        f"map_step={config.map_step_for_table} max_steps={config.max_steps}"
    )

    if args.preview_dataset:
        _maybe_preview_dataset(config.source_eval_split)
    if args.inspect_only:
        return

    if args.backend == "legacy":
        runner.main()
        return

    run_refactor_scienceqa_map_training(config)


if __name__ == "__main__":
    main()
