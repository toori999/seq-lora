from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets

from refactor.seq_lora.datasets import load_scienceqa_closedchoice_grade2_11

ORDER_KEYS = ("order", "reverse", "random")


@dataclass(frozen=True)
class ScienceQACurriculumSplits:
    train: Dataset
    validation: Dataset
    test: Dataset

    def get_eval_split(self, split_name: str) -> Dataset:
        split_name = str(split_name).strip().lower()
        if split_name in {"val", "validation"}:
            return self.validation
        if split_name == "test":
            return self.test
        raise ValueError(f"Unsupported ScienceQA eval split: {split_name!r}")


def load_scienceqa_curriculum_splits() -> ScienceQACurriculumSplits:
    train, validation, test = load_scienceqa_closedchoice_grade2_11()
    return ScienceQACurriculumSplits(
        train=train,
        validation=validation,
        test=test,
    )


def load_scienceqa_train_eval_split(
    source_eval_split: str = "test",
) -> Tuple[Dataset, Dataset]:
    splits = load_scienceqa_curriculum_splits()
    return splits.train, splits.get_eval_split(source_eval_split)


def summarize_grade_counts(ds: Dataset) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if "grade_num" not in ds.column_names:
        return counts
    for grade_num in ds["grade_num"]:
        grade_num = int(grade_num)
        counts[grade_num] = counts.get(grade_num, 0) + 1
    return counts


def summarize_choice_counts(ds: Dataset) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if "num_choices" not in ds.column_names:
        return counts
    for num_choices in ds["num_choices"]:
        num_choices = int(num_choices)
        counts[num_choices] = counts.get(num_choices, 0) + 1
    return counts


def describe_scienceqa_split(ds: Dataset) -> Dict[str, object]:
    return {
        "num_rows": len(ds),
        "grade_counts": summarize_grade_counts(ds),
        "choice_counts": summarize_choice_counts(ds),
    }


def print_scienceqa_split_summary(prefix: str, ds: Dataset) -> None:
    summary = describe_scienceqa_split(ds)
    print(f"[{prefix}] total={summary['num_rows']}")
    for grade_num, count in sorted(summary["grade_counts"].items()):
        print(f"  grade{grade_num}: {count}")
    choice_counts = summary["choice_counts"]
    if choice_counts:
        print(
            f"[{prefix}] choice-counts="
            + ", ".join(
                f"{num_choices}-choice={choice_counts[num_choices]}"
                for num_choices in sorted(choice_counts)
            )
        )


def _grade_order(train_raw: Dataset, descending: bool) -> Iterable[int]:
    grade_values = sorted({int(value) for value in train_raw["grade_num"]})
    if descending:
        grade_values = list(reversed(grade_values))
    return grade_values


def order_scienceqa_train(train_raw: Dataset, order_key: str, seed: int) -> Dataset:
    order_key = str(order_key).strip().lower()
    if order_key == "random":
        return train_raw.shuffle(seed=seed)
    if order_key not in {"order", "reverse"}:
        expected = ", ".join(ORDER_KEYS)
        raise ValueError(f"Unknown order_key={order_key!r}; expected one of: {expected}")

    descending = order_key == "reverse"
    parts: List[Dataset] = []
    for grade_num in _grade_order(train_raw, descending=descending):
        idxs = [
            idx for idx, value in enumerate(train_raw["grade_num"]) if int(value) == grade_num
        ]
        if not idxs:
            continue
        parts.append(train_raw.select(idxs).shuffle(seed=seed + grade_num))
    if not parts:
        raise RuntimeError("No ScienceQA training examples left after grade ordering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def save_kfac_balanced_dataset(train_ds: Dataset, slice_out_dir: str | Path) -> Path:
    slice_out_dir = Path(slice_out_dir)
    slice_out_dir.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(
        range(len(train_ds)),
        key=lambda idx: int(train_ds[idx]["slice_id"]),
    )
    ds_dict = DatasetDict({"train": train_ds.select(order)})
    ds_dict.save_to_disk(str(slice_out_dir))
    print(f"[Save] kfac_balanced slices -> {slice_out_dir}")
    return slice_out_dir


__all__ = [
    "ORDER_KEYS",
    "ScienceQACurriculumSplits",
    "describe_scienceqa_split",
    "load_scienceqa_curriculum_splits",
    "load_scienceqa_train_eval_split",
    "order_scienceqa_train",
    "print_scienceqa_split_summary",
    "save_kfac_balanced_dataset",
    "summarize_choice_counts",
    "summarize_grade_counts",
]
