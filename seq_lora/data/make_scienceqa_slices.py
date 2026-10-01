#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.common import (  # noqa: E402
    SCIENCEQA_GRADE_MAX,
    SCIENCEQA_GRADE_MIN,
    load_scienceqa_closedchoice_grade2_11,
)


T_ABLATION_ROOT = (
    "slice_data/"
    "scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_ablation"
)
DENSE_TAU_ROOT = (
    "slice_data/"
    "scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_dense_tau_seed1"
)
CONTROL_ROOT = "slice_data/scienceqa_slice_ablation_controls"
MAIN_T_VALUES = [1, 5, 10, 20]
DENSE_T_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 30, 36, 40]
# Exact row order of the slice datasets used for the paper. Within-slice order decides which
# examples form the pseudo-observation batches, so partitions rebuilt from the rules below
# reproduce the reported numbers only approximately.
PAPER_SLICE_ORDERS = Path(__file__).with_name("paper_slice_orders.json.gz")


def _parse_int_list(spec: str, default: Sequence[int]) -> List[int]:
    text = str(spec or "").strip()
    if not text:
        return [int(x) for x in default]
    values: List[int] = []
    for raw in text.replace(",", " ").split():
        value = int(raw)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"Values must be positive, got {value}.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return values


def _parse_str_list(spec: str, default: Sequence[str]) -> List[str]:
    text = str(spec or "").strip()
    if not text:
        return [str(x) for x in default]
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _load_scienceqa_train() -> Dataset:
    train, _, _ = load_scienceqa_closedchoice_grade2_11()
    if "orig_idx" not in train.column_names:
        train = train.add_column("orig_idx", list(range(len(train))))
    return train


def _grade_values(ds: Dataset) -> List[int]:
    grades = sorted({int(g) for g in ds["grade_num"]})
    expected = list(range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1))
    if grades != expected:
        raise ValueError(f"Expected ScienceQA grades {expected}, got {grades}.")
    return grades


def _replace_column(ds: Dataset, name: str, values: Iterable[int]) -> Dataset:
    if name in ds.column_names:
        ds = ds.remove_columns([name])
    return ds.add_column(name, [int(x) for x in values])


def _stable_sort_by_slice(ds: Dataset) -> Dataset:
    order = sorted(range(len(ds)), key=lambda idx: (int(ds[idx]["slice_id"]), idx))
    return ds.select(order)


def _train_sha256(ds: Dataset, columns: Sequence[str]) -> str:
    rows = [
        json.dumps([ex[c] for c in columns], ensure_ascii=False, sort_keys=True, default=str)
        for ex in ds
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _load_paper_orders(train: Dataset) -> Optional[Dict[str, Dict[str, List[int]]]]:
    if not PAPER_SLICE_ORDERS.exists():
        print(f"[Paper orders] {PAPER_SLICE_ORDERS} not found; using rule-based slices.")
        return None
    with gzip.open(PAPER_SLICE_ORDERS, "rt", encoding="utf-8") as f:
        spec = json.load(f)
    if int(spec["num_train_rows"]) != len(train) or _train_sha256(train, spec["digest_columns"]) != spec["train_sha256"]:
        print("[Paper orders] ScienceQA train split differs from the paper version; using rule-based slices.")
        return None
    print(f"[Paper orders] using exact paper slice orders from {PAPER_SLICE_ORDERS}")
    return spec["sets"]


def _from_paper_order(train: Dataset, entry: Dict[str, List[int]]) -> Dataset:
    slice_ids = [sid for sid, size in enumerate(entry["slice_sizes"]) for _ in range(int(size))]
    return _replace_column(train.select([int(i) for i in entry["index"]]), "slice_id", slice_ids)


def _save_train(ds: Dataset, out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            print(f"[Skip] {out_dir}")
            return
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(str(out_dir))
    print(f"[Save] {out_dir}")


def _slice_ids_for_t(ds: Dataset, t_value: int, grades: Sequence[int]) -> List[int]:
    grade_min = int(min(grades))
    grade_pos = [int(g) - grade_min for g in ds["grade_num"]]

    if t_value == 1:
        return [0 for _ in grade_pos]
    if t_value == 5:
        return [int(pos // 2) for pos in grade_pos]
    if t_value == 10:
        return [int(pos) for pos in grade_pos]
    if t_value == 20:
        by_grade: Dict[int, List[int]] = defaultdict(list)
        for idx, grade in enumerate(ds["grade_num"]):
            by_grade[int(grade)].append(idx)

        out = [0 for _ in range(len(ds))]
        for grade in grades:
            idxs = by_grade[int(grade)]
            base = (int(grade) - grade_min) * 2
            for rank, idx in enumerate(idxs):
                out[idx] = base + min(int(rank * 2 // max(len(idxs), 1)), 1)
        return out

    order = sorted(range(len(ds)), key=lambda idx: (int(ds[idx]["grade_num"]), idx))
    out = [0 for _ in range(len(ds))]
    for rank, idx in enumerate(order):
        out[idx] = min(int(rank * int(t_value) // max(len(order), 1)), int(t_value) - 1)
    return out


def _summary(ds: Dataset, t_value: int) -> Dict[str, object]:
    counts = Counter(int(x) for x in ds["slice_id"])
    grades_by_slice: Dict[int, List[int]] = defaultdict(list)
    for slice_id, grade in zip(ds["slice_id"], ds["grade_num"]):
        sid = int(slice_id)
        grade_i = int(grade)
        if grade_i not in grades_by_slice[sid]:
            grades_by_slice[sid].append(grade_i)
    return {
        "T": int(t_value),
        "num_rows": int(len(ds)),
        "slice_counts": {str(k): int(counts[k]) for k in sorted(counts)},
        "grades_by_slice": {str(k): sorted(v) for k, v in sorted(grades_by_slice.items())},
    }


def build_t_slices(
    train: Dataset,
    *,
    output_root: Path,
    t_values: Sequence[int],
    overwrite: bool,
    paper_orders: Optional[Dict[str, Dict[str, List[int]]]] = None,
    order_prefix: str = "t_ablation",
) -> None:
    grades = _grade_values(train)
    summaries = []
    for t_value in t_values:
        entry = (paper_orders or {}).get(f"{order_prefix}/T{int(t_value)}")
        if entry is not None:
            out_ds = _from_paper_order(train, entry)
        else:
            slice_ids = _slice_ids_for_t(train, int(t_value), grades)
            out_ds = _stable_sort_by_slice(_replace_column(train, "slice_id", slice_ids))
        out_dir = output_root / f"T{int(t_value)}" / "kfac_balanced"
        _save_train(out_ds, out_dir, overwrite=overwrite)
        summary = _summary(out_ds, int(t_value))
        summaries.append(summary)
        print(f"  T={int(t_value)} slice_counts={summary['slice_counts']}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "source": "ScienceQA closed-choice grades 2--11 train split",
                "output_root": str(output_root),
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _grade_parts(train: Dataset, seed: int, grade_order: Sequence[int]) -> List[Dataset]:
    parts: List[Dataset] = []
    for grade in grade_order:
        idxs = [idx for idx, g in enumerate(train["grade_num"]) if int(g) == int(grade)]
        if not idxs:
            raise RuntimeError(f"No rows for grade {grade}")
        parts.append(train.select(idxs).shuffle(seed=int(seed) + int(grade)))
    return parts


def _concat_with_slice_ids(parts: Sequence[Dataset]) -> Dataset:
    out = []
    for sid, part in enumerate(parts):
        out.append(_replace_column(part, "slice_id", [sid] * len(part)))
    return concatenate_datasets(out)


def _build_reverse_grade(train: Dataset, seed: int, num_slices: int) -> tuple[Dataset, Dict[str, object]]:
    grades = list(range(SCIENCEQA_GRADE_MAX, SCIENCEQA_GRADE_MIN - 1, -1))
    return _concat_with_slice_ids(_grade_parts(train, seed, grades)), {
        "variant": "reverse_grade",
        "num_slices": int(num_slices),
        "seed": int(seed),
        "slice_grade_order": grades,
        "notes": "Grade buckets kept intact, ordered high-to-low grade.",
    }


def _build_shuffled_grade(train: Dataset, seed: int, num_slices: int) -> tuple[Dataset, Dict[str, object]]:
    grades = list(range(SCIENCEQA_GRADE_MIN, SCIENCEQA_GRADE_MAX + 1))
    perm = np.random.default_rng(int(seed)).permutation(len(grades)).tolist()
    shuffled_grades = [grades[i] for i in perm]
    return _concat_with_slice_ids(_grade_parts(train, seed, shuffled_grades)), {
        "variant": "shuffled_grade",
        "num_slices": int(num_slices),
        "seed": int(seed),
        "slice_grade_order": shuffled_grades,
        "slice_permutation": perm,
        "notes": "Grade buckets kept intact, but grade order is randomized per seed.",
    }


def _build_full_shuffle(train: Dataset, seed: int, num_slices: int) -> tuple[Dataset, Dict[str, object]]:
    shuffled = train.shuffle(seed=int(seed))
    parts = []
    sizes = []
    for sid, idxs in enumerate(np.array_split(np.arange(len(shuffled)), int(num_slices))):
        part = shuffled.select(idxs.astype(int).tolist())
        part = _replace_column(part, "slice_id", [sid] * len(part))
        parts.append(part)
        sizes.append(len(part))
    return concatenate_datasets(parts), {
        "variant": "full_shuffle",
        "num_slices": int(num_slices),
        "seed": int(seed),
        "slice_sizes": sizes,
        "notes": "Whole training set shuffled, then evenly split into slices.",
    }


def build_structure_control_slices(
    train: Dataset,
    *,
    output_root: Path,
    variants: Sequence[str],
    seeds: Sequence[int],
    num_slices: int,
    overwrite: bool,
    args: argparse.Namespace,
    paper_orders: Optional[Dict[str, Dict[str, List[int]]]] = None,
) -> None:
    builders = {
        "reverse_grade": _build_reverse_grade,
        "shuffled_grade": _build_shuffled_grade,
        "full_shuffle": _build_full_shuffle,
    }
    for variant in variants:
        if variant in builders:
            for seed in seeds:
                ds_out, meta = builders[variant](train, int(seed), int(num_slices))
                out_dir = output_root / variant / f"seed_{int(seed)}" / "kfac_balanced"
                _save_train(ds_out, out_dir, overwrite=overwrite)
                (out_dir.parent / "meta.json").write_text(
                    json.dumps(meta, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        elif variant == "base_nll_easyhard":
            entry = (paper_orders or {}).get("structure_controls/base_nll_easyhard")
            if entry is not None:
                # Paper order of the base-model loss ranking; no model scoring needed.
                out_dir = output_root / "base_nll_easyhard" / "kfac_balanced"
                _save_train(_from_paper_order(train, entry), out_dir, overwrite=overwrite)
            else:
                build_base_nll_slices(output_root=output_root, overwrite=overwrite, args=args)
        else:
            raise ValueError(
                f"Unknown variant={variant!r}. Expected one of "
                "reverse_grade, shuffled_grade, full_shuffle, base_nll_easyhard."
            )


def build_base_nll_slices(
    *,
    output_root: Path,
    overwrite: bool,
    args: argparse.Namespace,
) -> None:
    out_dir = output_root / "base_nll_easyhard"
    slice_dir = out_dir / "kfac_balanced"
    if slice_dir.exists() and not overwrite:
        print(f"[Skip] {slice_dir}")
        return
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)

    cmd = [
        str(args.python_bin),
        str(ROOT / "seq_lora" / "data" / "mcqa_slices.py"),
        "--dataset_name",
        "scienceqa_closedchoice_grade2_11",
        "--split",
        "train",
        "--out_dir",
        str(out_dir),
        "--seed",
        "0",
        "--num_slices",
        str(args.num_slices),
        "--kfac_per_slice",
        "0",
        "--slice_strategy",
        "quantile",
        "--model_family",
        "custom",
        "--score_with",
        "base",
        "--base_model",
        str(args.base_model),
        "--tokenizer_padding_side",
        str(args.tokenizer_padding_side),
        "--trust_remote_code",
        str(args.trust_remote_code).lower(),
        "--local_files_only",
        str(args.local_files_only).lower(),
        "--batch_size",
        str(args.base_loss_batch_size),
        "--max_seq_len",
        str(args.max_seq_len),
        "--save_full_train",
        "false",
        "--save_kfac_balanced",
        "true",
    ]
    print("[Run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build all ScienceQA slice datasets used by Seq-LoRA experiments."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="main",
        choices=["main", "t_ablation", "dense_tau", "structure_controls", "all"],
        help=(
            "main builds T=1/5/10/20 and structure controls; all also builds "
            "the dense T grid used for tau/T diagnostics."
        ),
    )
    parser.add_argument("--t_output_root", type=Path, default=Path(T_ABLATION_ROOT))
    parser.add_argument("--dense_output_root", type=Path, default=Path(DENSE_TAU_ROOT))
    parser.add_argument("--control_output_root", type=Path, default=Path(CONTROL_ROOT))
    parser.add_argument("--t_values", type=str, default="")
    parser.add_argument("--dense_t_values", type=str, default="")
    parser.add_argument(
        "--variants",
        type=str,
        default="reverse_grade,shuffled_grade,full_shuffle,base_nll_easyhard",
    )
    parser.add_argument("--seeds", type=str, default="1,3,7,11,13")
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--overwrite", type=_parse_bool, default=False)
    parser.add_argument(
        "--use_paper_orders",
        type=_parse_bool,
        default=True,
        help=(
            "Rebuild the exact slice orders used for the paper (paper_slice_orders.json.gz) "
            "when available; false regenerates every partition from its rule."
        ),
    )

    parser.add_argument("--python_bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--tokenizer_padding_side", type=str, default="left")
    parser.add_argument("--trust_remote_code", type=_parse_bool, default=True)
    parser.add_argument("--local_files_only", type=_parse_bool, default=True)
    parser.add_argument("--base_loss_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=300)
    args = parser.parse_args()

    train = _load_scienceqa_train()
    print(f"[Load] ScienceQA closed-choice G2--11 train rows={len(train)}")
    paper_orders = _load_paper_orders(train) if bool(args.use_paper_orders) else None

    preset = str(args.preset)
    if preset in {"main", "t_ablation", "all"}:
        print(f"[Build] T ablation -> {args.t_output_root}")
        build_t_slices(
            train,
            output_root=args.t_output_root,
            t_values=_parse_int_list(args.t_values, MAIN_T_VALUES),
            overwrite=bool(args.overwrite),
            paper_orders=paper_orders,
            order_prefix="t_ablation",
        )

    if preset in {"dense_tau", "all"}:
        print(f"[Build] dense tau/T slices -> {args.dense_output_root}")
        build_t_slices(
            train,
            output_root=args.dense_output_root,
            t_values=_parse_int_list(args.dense_t_values, DENSE_T_VALUES),
            overwrite=bool(args.overwrite),
            paper_orders=paper_orders,
            order_prefix="dense_tau",
        )

    if preset in {"main", "structure_controls", "all"}:
        print(f"[Build] structure controls -> {args.control_output_root}")
        build_structure_control_slices(
            train,
            output_root=args.control_output_root,
            variants=_parse_str_list(
                args.variants,
                ["reverse_grade", "shuffled_grade", "full_shuffle", "base_nll_easyhard"],
            ),
            seeds=_parse_int_list(args.seeds, [1, 3, 7, 11, 13]),
            num_slices=int(args.num_slices),
            overwrite=bool(args.overwrite),
            args=args,
            paper_orders=paper_orders,
        )

    print("[Done]")


if __name__ == "__main__":
    main()
