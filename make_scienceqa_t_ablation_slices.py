from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_INPUT_DIR = (
    "slice_data/"
    "scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad/"
    "kfac_balanced"
)
DEFAULT_OUTPUT_ROOT = (
    "slice_data/"
    "scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad_T_ablation"
)
SUPPORTED_T = (1, 5, 10, 20)


def _parse_t_values(spec: str) -> List[int]:
    values: List[int] = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        t_value = int(raw)
        if t_value not in SUPPORTED_T:
            raise argparse.ArgumentTypeError(
                f"Unsupported T={t_value}. Expected one of {SUPPORTED_T}."
            )
        values.append(t_value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one T value.")
    return values


def _load_train(input_dir: str) -> Dataset:
    ds_obj = load_from_disk(input_dir)
    if isinstance(ds_obj, DatasetDict):
        if "train" not in ds_obj:
            raise ValueError(f"{input_dir} is a DatasetDict without a train split.")
        ds = ds_obj["train"]
    else:
        ds = ds_obj
    if "grade_num" not in ds.column_names:
        raise ValueError("Input dataset must contain grade_num.")
    return ds


def _grade_values(ds: Dataset) -> List[int]:
    grades = sorted({int(g) for g in ds["grade_num"]})
    if len(grades) != 10:
        raise ValueError(f"Expected 10 ScienceQA grades, got {grades}.")
    if grades != list(range(min(grades), max(grades) + 1)):
        raise ValueError(f"Expected contiguous grade values, got {grades}.")
    return grades


def _slice_ids_for_t(ds: Dataset, t_value: int, grades: List[int]) -> List[int]:
    grade_min = int(min(grades))
    grade_pos = [int(g) - grade_min for g in ds["grade_num"]]

    if t_value == 1:
        return [0 for _ in grade_pos]
    if t_value == 5:
        return [int(pos // 2) for pos in grade_pos]
    if t_value == 10:
        return [int(pos) for pos in grade_pos]
    if t_value != 20:
        raise ValueError(f"Unsupported T={t_value}.")

    by_grade: Dict[int, List[int]] = defaultdict(list)
    for idx, grade in enumerate(ds["grade_num"]):
        by_grade[int(grade)].append(idx)

    out = [0 for _ in range(len(ds))]
    for grade in grades:
        idxs = by_grade[int(grade)]
        n_grade = len(idxs)
        base = (int(grade) - grade_min) * 2
        for rank, idx in enumerate(idxs):
            half = min(int(rank * 2 // max(n_grade, 1)), 1)
            out[idx] = base + half
    return out


def _replace_slice_id(ds: Dataset, slice_ids: Iterable[int]) -> Dataset:
    if "slice_id" in ds.column_names:
        ds = ds.remove_columns(["slice_id"])
    return ds.add_column("slice_id", [int(x) for x in slice_ids])


def _stable_sort_by_slice(ds: Dataset) -> Dataset:
    order = sorted(range(len(ds)), key=lambda idx: (int(ds[idx]["slice_id"]), idx))
    return ds.select(order)


def _summary(ds: Dataset, t_value: int) -> Dict[str, object]:
    counts = Counter(int(x) for x in ds["slice_id"])
    grade_by_slice: Dict[int, List[int]] = defaultdict(list)
    for slice_id, grade in zip(ds["slice_id"], ds["grade_num"]):
        grade_i = int(grade)
        slice_i = int(slice_id)
        if grade_i not in grade_by_slice[slice_i]:
            grade_by_slice[slice_i].append(grade_i)
    return {
        "T": int(t_value),
        "num_rows": int(len(ds)),
        "slice_counts": {str(k): int(counts[k]) for k in sorted(counts)},
        "grades_by_slice": {
            str(k): sorted(v) for k, v in sorted(grade_by_slice.items())
        },
    }


def _save_dataset(ds: Dataset, out_dir: str, *, overwrite: bool) -> None:
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists. Pass --overwrite true to replace it."
            )
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    DatasetDict({"train": ds}).save_to_disk(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ScienceQA grade-based slice-count ablations for Seq-LoRA."
    )
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--t_values", type=_parse_t_values, default=list(SUPPORTED_T))
    parser.add_argument(
        "--overwrite",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"},
        default=False,
    )
    args = parser.parse_args()

    ds = _load_train(args.input_dir)
    grades = _grade_values(ds)
    print(f"[Load] input={args.input_dir} rows={len(ds)} grades={grades}")

    summaries = []
    for t_value in args.t_values:
        slice_ids = _slice_ids_for_t(ds, int(t_value), grades)
        out_ds = _stable_sort_by_slice(_replace_slice_id(ds, slice_ids))
        summary = _summary(out_ds, int(t_value))
        out_dir = os.path.join(args.output_root, f"T{int(t_value)}", "kfac_balanced")
        _save_dataset(out_ds, out_dir, overwrite=bool(args.overwrite))
        summaries.append(summary)

        print(f"\n[Save] T={int(t_value)} -> {out_dir}")
        print(f"  slice_counts={summary['slice_counts']}")
        print(f"  grades_by_slice={summary['grades_by_slice']}")

    summary_path = os.path.join(args.output_root, "summary.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": args.input_dir,
                "output_root": args.output_root,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"\n[Summary] {summary_path}")


if __name__ == "__main__":
    main()
