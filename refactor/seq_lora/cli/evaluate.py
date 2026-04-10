from __future__ import annotations

import argparse

from ..constants import SCIENCEQA_CURRIC_TASK_NAME
from ..eval import (
    run_deep_ensemble_evaluation,
    run_map_evaluation,
    run_mc_dropout_evaluation,
    run_probability_ensemble_evaluation,
)

TASK_CHOICES = [
    "wgs",
    "wgm",
    "arc-c",
    "arc-e",
    "obqa",
    "boolq",
    "sciq",
    SCIENCEQA_CURRIC_TASK_NAME,
]


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=str, required=True, choices=TASK_CHOICES)
    parser.add_argument("--eval-tasks", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=300)
    parser.add_argument("--eval-bsz", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI for the refactor package."
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    map_parser = subparsers.add_parser("map", help="Run deterministic MAP evaluation.")
    _add_shared_args(map_parser)
    map_parser.add_argument("--adapter-dir", type=str, required=True)

    mcdrop_parser = subparsers.add_parser(
        "mcdrop", help="Run MC-Dropout evaluation."
    )
    _add_shared_args(mcdrop_parser)
    mcdrop_parser.add_argument("--adapter-dir", type=str, required=True)
    mcdrop_parser.add_argument("--mc-samples", type=int, default=32)
    mcdrop_parser.add_argument("--temp", type=float, default=1.0)

    deep_ens_parser = subparsers.add_parser(
        "deep-ensemble",
        help="Run multi-adapter deep ensemble evaluation.",
    )
    _add_shared_args(deep_ens_parser)
    deep_ens_parser.add_argument(
        "--adapter-dirs",
        type=str,
        required=True,
        help="Comma-separated adapter directories.",
    )
    deep_ens_parser.add_argument("--temp", type=float, default=1.0)

    prob_ens_parser = subparsers.add_parser(
        "prob-ensemble",
        help="Average probabilities from multiple MAP adapters.",
    )
    _add_shared_args(prob_ens_parser)
    prob_ens_parser.add_argument(
        "--adapter-dirs",
        type=str,
        required=True,
        help="Comma-separated adapter directories.",
    )

    return parser


def _split_adapter_dirs(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.method == "map":
        run_map_evaluation(
            task=args.task,
            adapter_dir=args.adapter_dir,
            eval_task_spec=args.eval_tasks,
            max_seq_len=args.max_seq_len,
            eval_bsz=args.eval_bsz,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
        )
        return

    if args.method == "mcdrop":
        run_mc_dropout_evaluation(
            task=args.task,
            adapter_dir=args.adapter_dir,
            eval_task_spec=args.eval_tasks,
            max_seq_len=args.max_seq_len,
            eval_bsz=args.eval_bsz,
            seed=args.seed,
            mc_samples=args.mc_samples,
            temp=args.temp,
            trust_remote_code=args.trust_remote_code,
        )
        return

    if args.method == "deep-ensemble":
        run_deep_ensemble_evaluation(
            task=args.task,
            adapter_dirs=_split_adapter_dirs(args.adapter_dirs),
            eval_task_spec=args.eval_tasks,
            max_seq_len=args.max_seq_len,
            eval_bsz=args.eval_bsz,
            seed=args.seed,
            temp=args.temp,
            trust_remote_code=args.trust_remote_code,
        )
        return

    if args.method == "prob-ensemble":
        run_probability_ensemble_evaluation(
            task=args.task,
            adapter_dirs=_split_adapter_dirs(args.adapter_dirs),
            eval_task_spec=args.eval_tasks,
            max_seq_len=args.max_seq_len,
            eval_bsz=args.eval_bsz,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
        )
        return

    raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
