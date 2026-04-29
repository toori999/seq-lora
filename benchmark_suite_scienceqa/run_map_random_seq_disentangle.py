from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_eval_utils import (  # noqa: E402
    SCIENCEQA_CURRIC_TASK_NAME,
    SCIENCEQA_DATASET_NAME,
    SCIENCEQA_GRADE_MAX,
    SCIENCEQA_GRADE_MIN,
)


DEFAULT_EVAL_TASKS = [
    "iid",
    "scienceqa_closedchoice_grade12",
    "obqa",
    "arc-c",
    "mmlu_science_high",
    "mmlu_science_college",
    "gpqa_main",
]
DEFAULT_VARIANTS = ["flow", "semantic", "semantic_random"]
SUPPORTED_VARIANTS = ["flow", "semantic", "semantic_reverse", "semantic_random"]
SCIENCEQA_TASK_FILTER = "closed choice"
HF_CACHE_DIR = ROOT / ".hf_datasets_cache"


def parse_seed_list(text: str) -> List[int]:
    seeds = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_variant_list(text: str) -> List[str]:
    variants = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not variants:
        raise ValueError("At least one variant is required.")
    unknown = [name for name in variants if name not in SUPPORTED_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}. Choices: {SUPPORTED_VARIANTS}")
    return variants


def _remove_then_add_column(ds: Dataset, name: str, values: Sequence[int]) -> Dataset:
    if name in ds.column_names:
        ds = ds.remove_columns([name])
    return ds.add_column(name, list(values))


def _with_orig_idx(train_ds: Dataset) -> Dataset:
    if "orig_idx" in train_ds.column_names:
        return train_ds
    return train_ds.add_column("orig_idx", list(range(len(train_ds))))


def load_source_train() -> Dataset:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(SCIENCEQA_DATASET_NAME, cache_dir=str(HF_CACHE_DIR))
    train_ds = ds["train"]

    def _keep(ex: Dict) -> bool:
        try:
            grade_num = int(str(ex["grade"]).strip().lower().replace("grade", ""))
        except Exception:
            return False
        return (
            str(ex.get("task", "")).strip().lower() == SCIENCEQA_TASK_FILTER
            and SCIENCEQA_GRADE_MIN <= grade_num <= SCIENCEQA_GRADE_MAX
        )

    def _add_meta(ex: Dict) -> Dict:
        grade_num = int(str(ex["grade"]).strip().lower().replace("grade", ""))
        choices = ex.get("choices", [])
        if isinstance(choices, dict):
            num_choices = len(choices.get("text", []))
        else:
            num_choices = len(choices)
        return {
            "grade_num": grade_num,
            "slice_id": grade_num - SCIENCEQA_GRADE_MIN,
            "num_choices": num_choices,
        }

    train_ds = train_ds.filter(_keep).map(_add_meta)
    return _with_orig_idx(train_ds)


def _grade_parts(train_ds: Dataset, seed: int, ascending: bool) -> List[Dataset]:
    grade_order = range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1)
    if not ascending:
        grade_order = reversed(list(grade_order))

    parts: List[Dataset] = []
    for grade_num in grade_order:
        idxs = [i for i, g in enumerate(train_ds["grade_num"]) if int(g) == int(grade_num)]
        if not idxs:
            continue
        ds_g = train_ds.select(idxs).shuffle(seed=seed + int(grade_num))
        parts.append(ds_g)
    if not parts:
        raise RuntimeError("No ScienceQA grade slices were constructed.")
    return parts


def _concat_with_slice_ids(parts: Sequence[Dataset]) -> Dataset:
    seq_parts: List[Dataset] = []
    for sid, ds_part in enumerate(parts):
        seq_parts.append(_remove_then_add_column(ds_part, "slice_id", [sid] * len(ds_part)))
    return concatenate_datasets(seq_parts)


def build_semantic_slices(train_ds: Dataset, seed: int) -> tuple[Dataset, Dict]:
    parts = _grade_parts(train_ds, seed=seed, ascending=True)
    meta = {
        "variant": "semantic",
        "slice_grade_order": list(range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1)),
        "notes": "Grade-ordered slices (easy-to-hard / low-to-high grade).",
    }
    return _concat_with_slice_ids(parts), meta


def build_semantic_reverse_slices(train_ds: Dataset, seed: int) -> tuple[Dataset, Dict]:
    parts = _grade_parts(train_ds, seed=seed, ascending=False)
    meta = {
        "variant": "semantic_reverse",
        "slice_grade_order": list(range(SCIENCEQA_GRADE_MAX, SCIENCEQA_GRADE_MIN - 1, -1)),
        "notes": "Grade-ordered slices in reverse (hard-to-easy / high-to-low grade).",
    }
    return _concat_with_slice_ids(parts), meta


def build_semantic_random_slices(train_ds: Dataset, seed: int) -> tuple[Dataset, Dict]:
    grade_parts = _grade_parts(train_ds, seed=seed, ascending=True)
    grades = list(range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1))
    perm = np.random.default_rng(seed).permutation(len(grade_parts)).tolist()
    parts = [grade_parts[idx] for idx in perm]
    meta = {
        "variant": "semantic_random",
        "slice_grade_order": [grades[idx] for idx in perm],
        "slice_permutation": perm,
        "notes": "Grade bucket composition preserved, but slice order randomized.",
    }
    return _concat_with_slice_ids(parts), meta


def build_flow_slices(
    train_ds: Dataset,
    seed: int,
    num_slices: int,
    micro_bsz: int,
    grad_accum: int,
) -> tuple[Dataset, Dict]:
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if micro_bsz <= 0 or grad_accum <= 0:
        raise ValueError("micro_bsz and grad_accum must be positive.")

    effective_bsz = int(micro_bsz) * int(grad_accum)
    train_flow = train_ds.shuffle(seed=seed)
    n_updates = len(train_flow) // effective_bsz
    if n_updates < num_slices:
        raise ValueError(
            f"num_slices={num_slices} exceeds available optimizer-step blocks={n_updates} "
            f"for len(train)={len(train_flow)} and effective_bsz={effective_bsz}."
        )

    usable = train_flow.select(range(n_updates * effective_bsz))
    block_ids = np.array_split(np.arange(n_updates), num_slices)

    parts: List[Dataset] = []
    block_ranges: List[List[int]] = []
    for sid, block_group in enumerate(block_ids):
        if len(block_group) == 0:
            continue
        start = int(block_group[0]) * effective_bsz
        stop = (int(block_group[-1]) + 1) * effective_bsz
        block_ranges.append([int(block_group[0]), int(block_group[-1])])
        ds_part = usable.select(range(start, stop))
        ds_part = _remove_then_add_column(ds_part, "slice_id", [sid] * len(ds_part))
        ds_part = _remove_then_add_column(ds_part, "flow_pos", list(range(start, stop)))
        parts.append(ds_part)

    meta = {
        "variant": "flow",
        "notes": (
            "Slices follow the actual MAP-random sample stream, grouped into contiguous "
            "optimizer-step windows."
        ),
        "micro_bsz": int(micro_bsz),
        "grad_accum": int(grad_accum),
        "effective_bsz": effective_bsz,
        "num_optimizer_step_blocks": int(n_updates),
        "num_slices": int(num_slices),
        "slice_block_ranges": block_ranges,
    }
    return concatenate_datasets(parts), meta


def save_slice_dataset(
    train_ds: Dataset,
    out_dir: Path,
    meta: Dict,
    overwrite: bool,
) -> None:
    if out_dir.exists():
        if not overwrite:
            return
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": train_ds}).save_to_disk(str(out_dir))
    meta_path = out_dir.parent / f"{out_dir.name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def build_variant_slices(
    variant: str,
    train_ds: Dataset,
    seed: int,
    num_slices: int,
    micro_bsz: int,
    grad_accum: int,
) -> tuple[Dataset, Dict]:
    if variant == "flow":
        return build_flow_slices(train_ds, seed, num_slices, micro_bsz, grad_accum)
    if variant == "semantic":
        return build_semantic_slices(train_ds, seed)
    if variant == "semantic_reverse":
        return build_semantic_reverse_slices(train_ds, seed)
    if variant == "semantic_random":
        return build_semantic_random_slices(train_ds, seed)
    raise ValueError(f"Unhandled variant: {variant}")


def build_seq_command(
    python_bin: str,
    slice_dir: Path,
    map_dir: Path,
    eval_tasks: Sequence[str],
    q_mode: str,
    s_q: float,
    forecast_horizon: int,
) -> List[str]:
    cmd = [
        python_bin,
        str(ROOT / "seq_eval_iid_constantq.py"),
        "--task",
        SCIENCEQA_CURRIC_TASK_NAME,
        "--slices_dir",
        str(slice_dir),
        "--map_dir",
        str(map_dir),
        "--eval_tasks",
        ",".join(eval_tasks),
        "--q_mode",
        str(q_mode),
        "--s_q",
        str(s_q),
        "--forecast_horizon",
        str(forecast_horizon),
    ]
    return cmd


def run_with_log(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("[CMD] " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, check=False)
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build and optionally run the MAP-random / Seq-LoRA disentanglement experiment. "
            "For each MAP-random seed, this script can construct: "
            "(1) flow-derived slices, "
            "(2) semantic grade-ordered slices, and "
            "(3) randomized semantic-slice controls."
        )
    )
    ap.add_argument("--seeds", type=str, default="1,3,7,11,13")
    ap.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--eval_tasks", type=str, default=",".join(DEFAULT_EVAL_TASKS))
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument(
        "--map_random_base_dir",
        type=str,
        default=str(
            ROOT
            / "iid_qwen35_8b_scienceqa_lora_map_leftpad_random"
            / "scienceqa_text_closedchoice_grade2_11_random_qv_lmhead_leftpad"
        ),
    )
    ap.add_argument(
        "--slice_root",
        type=str,
        default=str(ROOT / "slice_data" / "map_random_seq_disentangle"),
    )
    ap.add_argument(
        "--log_root",
        type=str,
        default=str(ROOT / "logs" / "map_random_seq_disentangle"),
    )
    ap.add_argument("--num_slices", type=int, default=10)
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--q_mode", type=str, default="module_constant", choices=["module_constant", "constant"])
    ap.add_argument("--s_q", type=float, default=1.0)
    ap.add_argument("--forecast_horizon", type=int, default=0)
    ap.add_argument("--build_only", action="store_true", help="Build slices only; do not launch Seq-LoRA runs.")
    ap.add_argument("--overwrite_slices", action="store_true", help="Overwrite existing slice directories.")
    args = ap.parse_args()

    seeds = parse_seed_list(args.seeds)
    variants = parse_variant_list(args.variants)
    eval_tasks = [tok.strip() for tok in args.eval_tasks.split(",") if tok.strip()]

    map_random_base_dir = Path(args.map_random_base_dir).resolve()
    slice_root = Path(args.slice_root).resolve()
    log_root = Path(args.log_root).resolve()
    train_ds = load_source_train()

    manifest: Dict[str, Dict] = {
        "seeds": seeds,
        "variants": variants,
        "eval_tasks": eval_tasks,
        "map_random_base_dir": str(map_random_base_dir),
        "slice_root": str(slice_root),
        "log_root": str(log_root),
        "runs": {},
    }

    for seed in seeds:
        seed_key = f"seed_{seed}"
        manifest["runs"][seed_key] = {}
        map_dir = map_random_base_dir / f"seed_{seed}" / "map_step_2000"
        if not map_dir.is_dir():
            raise FileNotFoundError(f"MAP-random adapter not found: {map_dir}")

        for variant in variants:
            variant_dir = slice_root / seed_key / variant
            variant_log = log_root / seed_key / f"{variant}.log"
            slice_ds, meta = build_variant_slices(
                variant=variant,
                train_ds=train_ds,
                seed=seed,
                num_slices=int(args.num_slices),
                micro_bsz=int(args.micro_bsz),
                grad_accum=int(args.grad_accum),
            )
            meta.update(
                {
                    "seed": int(seed),
                    "map_dir": str(map_dir),
                    "slice_dir": str(variant_dir),
                    "eval_tasks": eval_tasks,
                }
            )
            save_slice_dataset(slice_ds, variant_dir, meta, overwrite=bool(args.overwrite_slices))

            record = {
                "map_dir": str(map_dir),
                "slice_dir": str(variant_dir),
                "meta": meta,
            }

            if not args.build_only:
                cmd = build_seq_command(
                    python_bin=str(args.python_bin),
                    slice_dir=variant_dir,
                    map_dir=map_dir,
                    eval_tasks=eval_tasks,
                    q_mode=str(args.q_mode),
                    s_q=float(args.s_q),
                    forecast_horizon=int(args.forecast_horizon),
                )
                rc = run_with_log(cmd, variant_log)
                record["command"] = cmd
                record["log_path"] = str(variant_log)
                record["returncode"] = int(rc)
                print(f"[Run] seed={seed} variant={variant} rc={rc} log={variant_log}")
                if rc != 0:
                    raise RuntimeError(f"Seq-LoRA run failed for seed={seed} variant={variant}. See {variant_log}")
            else:
                print(f"[Build] seed={seed} variant={variant} slices={variant_dir}")

            manifest["runs"][seed_key][variant] = record

    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = log_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
