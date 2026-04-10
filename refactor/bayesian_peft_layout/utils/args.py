from __future__ import annotations

from argparse import ArgumentParser

from refactor.seq_lora.constants import SCIENCEQA_CURRIC_TASK_NAME

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


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")


def add_evaluation_args(parser: ArgumentParser) -> None:
    parser.add_argument("--task", type=str, required=True, choices=TASK_CHOICES)
    parser.add_argument("--eval-tasks", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=300)
    parser.add_argument("--eval-bsz", type=int, default=None)


def add_method_subparsers(subparsers) -> None:
    map_parser = subparsers.add_parser("map", help="Run deterministic MAP evaluation.")
    add_management_args(map_parser)
    add_evaluation_args(map_parser)
    map_parser.add_argument("--adapter-dir", type=str, required=True)

    mcdrop_parser = subparsers.add_parser("mcdrop", help="Run MC-Dropout evaluation.")
    add_management_args(mcdrop_parser)
    add_evaluation_args(mcdrop_parser)
    mcdrop_parser.add_argument("--adapter-dir", type=str, required=True)
    mcdrop_parser.add_argument("--mc-samples", type=int, default=32)
    mcdrop_parser.add_argument("--temp", type=float, default=1.0)

    deep_ens_parser = subparsers.add_parser(
        "deep-ensemble",
        help="Run multi-adapter deep ensemble evaluation.",
    )
    add_management_args(deep_ens_parser)
    add_evaluation_args(deep_ens_parser)
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
    add_management_args(prob_ens_parser)
    add_evaluation_args(prob_ens_parser)
    prob_ens_parser.add_argument(
        "--adapter-dirs",
        type=str,
        required=True,
        help="Comma-separated adapter directories.",
    )

    seq_parser = subparsers.add_parser(
        "seq-constantq",
        help="Run Seq-LoRA evaluation with constant process noise.",
    )
    add_management_args(seq_parser)
    add_evaluation_args(seq_parser)
    seq_parser.add_argument("--map-dir", type=str, required=True)
    seq_parser.add_argument("--slice-dir", type=str, default="")
    seq_parser.add_argument("--constant-q-var", type=float, default=1.0)
    seq_parser.add_argument("--forecast-horizon", type=int, default=0)

    laplace_parser = subparsers.add_parser(
        "laplace",
        help="Run Laplace-LoRA evaluation through the refactor entrypoint.",
    )
    add_management_args(laplace_parser)
    add_evaluation_args(laplace_parser)
    laplace_parser.add_argument("--adapter-dir", type=str, required=True)
    laplace_parser.add_argument("--output-dir", type=str, required=True)
    laplace_parser.add_argument("--fit-bsz", type=int, default=2)
    laplace_parser.add_argument("--laplace-bsz", type=int, default=32)
    laplace_parser.add_argument("--prior-optim-step", type=int, default=100)
    laplace_parser.add_argument("--laplace-mc-samples", type=int, default=100000)
    laplace_parser.add_argument("--laplace-mc-chunk", type=int, default=512)
    laplace_parser.add_argument("--testing-set", type=str, default="val")
    laplace_parser.add_argument("--model-name-or-path", type=str, default="")
    laplace_parser.add_argument("--force-refit", action="store_true")
    laplace_parser.add_argument("--force-reprior", action="store_true")

    blob_parser = subparsers.add_parser(
        "blob",
        help="Run BLoB train/eval through the refactor entrypoint.",
    )
    add_management_args(blob_parser)
    add_evaluation_args(blob_parser)
    blob_parser.add_argument("--base-model", type=str, required=True)
    blob_parser.add_argument("--map-dir", type=str, default="")
    blob_parser.add_argument("--shared-init-lora-path", type=str, default="")
    blob_parser.add_argument("--save-blob-dir", type=str, default="")
    blob_parser.add_argument("--load-blob-dir", type=str, default="")
    blob_parser.add_argument("--blob-eval-n", type=int, default=10)
    blob_parser.add_argument("--do-train", action="store_true")
    blob_parser.add_argument("--do-eval", action="store_true")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Bayesian-PEFT-style evaluation entrypoint for the Seq-LoRA refactor."
    )
    subparsers = parser.add_subparsers(dest="method", required=True)
    add_method_subparsers(subparsers)
    return parser


__all__ = [
    "TASK_CHOICES",
    "add_evaluation_args",
    "add_management_args",
    "add_method_subparsers",
    "build_parser",
]
