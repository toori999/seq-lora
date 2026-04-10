from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .constants import (
    AGIEVAL_ENGLISH_CONFIGS,
    MMLU_EVAL_TASK_PREFIX,
    MMLU_GROUPS,
    SCIENCEQA_CURRIC_TASK_NAME,
    SCIENCEQA_DATASET_NAME,
    SCIENCEQA_GRADE12_TASK_NAME,
    SCIENCEQA_GRADE_MAX,
    SCIENCEQA_GRADE_MIN,
    SCIENCEQA_TASK_FILTER,
)
from .prompts import answer_index_to_key, answer_key_to_index, get_choice_labels


def _parse_scienceqa_grade_num(grade_value) -> int:
    text = str(grade_value).strip().lower()
    if text.startswith("grade"):
        return int(text.replace("grade", ""))
    raise ValueError(f"Unexpected ScienceQA grade format: {grade_value}")


def _scienceqa_choice_texts(choices_obj) -> List[str]:
    if hasattr(choices_obj, "tolist"):
        values = choices_obj.tolist()
    elif isinstance(choices_obj, (list, tuple)):
        values = list(choices_obj)
    else:
        values = []
    return [str(value).strip() for value in values if str(value).strip()]


def _concat_available_splits(parts: List[Dataset]) -> Dataset:
    if not parts:
        raise ValueError("No dataset splits available to concatenate.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_openbookqa_from_local_cache() -> Optional[DatasetDict]:
    cache_root = (
        Path.home() / ".cache" / "huggingface" / "datasets" / "openbookqa" / "main" / "0.0.0"
    )
    if not cache_root.exists():
        return None
    revisions = sorted(path for path in cache_root.iterdir() if path.is_dir())
    for rev_dir in reversed(revisions):
        train_path = rev_dir / "openbookqa-train.arrow"
        val_path = rev_dir / "openbookqa-validation.arrow"
        test_path = rev_dir / "openbookqa-test.arrow"
        if train_path.exists() and val_path.exists() and test_path.exists():
            return DatasetDict(
                {
                    "train": Dataset.from_file(str(train_path)),
                    "validation": Dataset.from_file(str(val_path)),
                    "test": Dataset.from_file(str(test_path)),
                }
            )
    return None


def _load_openbookqa_dataset() -> DatasetDict:
    try:
        return load_dataset("openbookqa", "main")
    except Exception as exc:
        cached = _load_openbookqa_from_local_cache()
        if cached is not None:
            print(
                "[Dataset] Falling back to local cached OpenBookQA after "
                f"load_dataset failure: {exc}"
            )
            return cached
        raise


def _load_mmlu_subject_from_local_cache(subject: str) -> Optional[Dataset]:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "cais___mmlu"
        / subject
        / "0.0.0"
    )
    if not cache_root.exists():
        return None
    revisions = sorted(path for path in cache_root.iterdir() if path.is_dir())
    for rev_dir in reversed(revisions):
        for split_name in ["mmlu-test.arrow", "mmlu-validation.arrow", "mmlu-dev.arrow"]:
            split_path = rev_dir / split_name
            if split_path.exists():
                return Dataset.from_file(str(split_path))
    return None


def load_scienceqa_closedchoice_grade2_11() -> Tuple[Dataset, Dataset, Dataset]:
    ds = load_dataset(SCIENCEQA_DATASET_NAME)

    def _keep(ex: Dict) -> bool:
        try:
            grade_num = _parse_scienceqa_grade_num(ex["grade"])
        except Exception:
            return False
        return (
            str(ex.get("task", "")).strip().lower() == SCIENCEQA_TASK_FILTER
            and SCIENCEQA_GRADE_MIN <= grade_num <= SCIENCEQA_GRADE_MAX
        )

    def _add_meta(ex: Dict) -> Dict:
        grade_num = _parse_scienceqa_grade_num(ex["grade"])
        return {
            "grade_num": grade_num,
            "slice_id": grade_num - SCIENCEQA_GRADE_MIN,
            "num_choices": len(_scienceqa_choice_texts(ex["choices"])),
        }

    train_ds = ds["train"].filter(_keep).map(_add_meta)
    val_ds = ds["validation"].filter(_keep).map(_add_meta)
    test_ds = ds["test"].filter(_keep).map(_add_meta)
    return train_ds, val_ds, test_ds


def load_scienceqa_closedchoice_grade12_all() -> Dataset:
    ds = load_dataset(SCIENCEQA_DATASET_NAME)

    def _keep(ex: Dict) -> bool:
        try:
            grade_num = _parse_scienceqa_grade_num(ex["grade"])
        except Exception:
            return False
        return (
            str(ex.get("task", "")).strip().lower() == SCIENCEQA_TASK_FILTER
            and grade_num == 12
        )

    def _add_meta(ex: Dict) -> Dict:
        return {
            "grade_num": 12,
            "num_choices": len(_scienceqa_choice_texts(ex["choices"])),
        }

    parts: List[Dataset] = []
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            parts.append(ds[split_name].filter(_keep).map(_add_meta))
    return _concat_available_splits(parts)


def load_task_dataset(task: str) -> Tuple[Dataset, Dataset, Dataset]:
    print(f"\n=== Loading Dataset for Task: {task} ===")
    if task == "wgs":
        ds = load_dataset("winogrande", "winogrande_s")
        return ds["train"], ds["validation"], ds["validation"]
    if task == "wgm":
        ds = load_dataset("winogrande", "winogrande_m")
        return ds["train"], ds["validation"], ds["validation"]
    if task == "arc-c":
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        return ds["train"], ds["validation"], ds["test"]
    if task == "arc-e":
        ds = load_dataset("ai2_arc", "ARC-Easy")
        return ds["train"], ds["validation"], ds["test"]
    if task == "obqa":
        ds = _load_openbookqa_dataset()
        return ds["train"], ds["validation"], ds["test"]
    if task == "boolq":
        ds = load_dataset("super_glue", "boolq")
        return ds["train"], ds["validation"], ds["validation"]
    if task == "sciq":
        ds = load_dataset("sciq")
        return ds["train"], ds["validation"], ds["test"]
    if task == SCIENCEQA_CURRIC_TASK_NAME:
        return load_scienceqa_closedchoice_grade2_11()
    raise ValueError(f"Unknown task: {task}")


def load_iid_test_set(task: str) -> Dataset:
    _, _, test = load_task_dataset(task)
    return test


def _pick_mmlu_split(ds_dict) -> Dataset:
    for split in ["test", "validation", "dev"]:
        if split in ds_dict:
            return ds_dict[split]
    return ds_dict[list(ds_dict.keys())[0]]


def _load_mmlu_subject(subject: str) -> Dataset:
    try:
        return _pick_mmlu_split(load_dataset("cais/mmlu", subject))
    except Exception as exc:
        cached = _load_mmlu_subject_from_local_cache(subject)
        if cached is not None:
            print(
                "[Dataset] Falling back to local cached MMLU subject "
                f"'{subject}' after load_dataset failure: {exc}"
            )
            return cached
        raise


def _load_mmlu_group(group: str) -> Dataset:
    group = group.lower().strip()
    if group not in MMLU_GROUPS:
        raise ValueError(f"Unknown MMLU group: {group}")

    parts = []
    for subject in MMLU_GROUPS[group]:
        print(f"[MMLU] loading subject: {subject}")
        parts.append(_load_mmlu_subject(subject))
    return concatenate_datasets(parts)


def _is_mmlu_eval_task(task: str) -> bool:
    return task.lower().strip().startswith(MMLU_EVAL_TASK_PREFIX)


def _mmlu_group_from_task(task: str) -> str:
    task = task.lower().strip()
    if not _is_mmlu_eval_task(task):
        raise ValueError(f"Unknown MMLU eval task: {task}")
    return task[len(MMLU_EVAL_TASK_PREFIX) :]


def _make_choice_rows(options: List[str]) -> List[Dict[str, str]]:
    labels = get_choice_labels(len(options))
    return [{"label": labels[i], "text": str(options[i])} for i in range(len(options))]


def _normalize_gpqa_subset(ds: Dataset) -> Dataset:
    def _fmt(ex: Dict) -> Dict:
        question = str(ex.get("Question", ex.get("question", "")))
        correct = str(ex.get("Correct Answer", ex.get("correct_answer", "")))
        distractors = [
            str(ex.get("Incorrect Answer 1", ex.get("incorrect_answer_1", ""))),
            str(ex.get("Incorrect Answer 2", ex.get("incorrect_answer_2", ""))),
            str(ex.get("Incorrect Answer 3", ex.get("incorrect_answer_3", ""))),
        ]
        options = distractors + [correct]
        random.shuffle(options)
        labels = get_choice_labels(len(options))
        answer_key = answer_index_to_key(options.index(correct), labels)
        return {
            "question": question,
            "choices": _make_choice_rows(options),
            "answerKey": answer_key,
        }

    return ds.map(_fmt, remove_columns=ds.column_names)


def _extract_agieval_options(ex: Dict):
    for key in ["options", "choices"]:
        if key in ex and isinstance(ex[key], (list, tuple)):
            return list(ex[key])
    return None


def _extract_agieval_answer(ex: Dict):
    for key in ["label", "answer", "answerKey", "target"]:
        if key in ex:
            return ex[key]
    return None


def _normalize_agieval_subset(ds: Dataset, subset_name: str) -> Dataset:
    def _fmt(ex: Dict) -> Dict:
        question = str(ex.get("question", ex.get("query", ex.get("problem", ""))))
        options = _extract_agieval_options(ex) or []
        answer = _extract_agieval_answer(ex)
        if len(options) != 4:
            return {
                "question": "",
                "choices": [],
                "answerKey": "Z",
                "source_subset": subset_name,
            }

        label_order = get_choice_labels(len(options))
        answer_key = None
        if isinstance(answer, int):
            if 0 <= int(answer) < len(label_order):
                answer_key = answer_index_to_key(int(answer), label_order)
        else:
            try:
                answer_key = answer_index_to_key(
                    answer_key_to_index(answer, label_order), label_order
                )
            except Exception:
                answer_key = None

        if answer_key is None:
            return {
                "question": "",
                "choices": [],
                "answerKey": "Z",
                "source_subset": subset_name,
            }

        return {
            "question": question,
            "choices": _make_choice_rows([str(opt) for opt in options]),
            "answerKey": answer_key,
            "source_subset": subset_name,
        }

    ds2 = ds.map(_fmt, remove_columns=ds.column_names)
    return ds2.filter(
        lambda ex: ex["answerKey"] in get_choice_labels(4) and len(ex["choices"]) == 4
    )


def _load_gpqa_subset(config_name: str) -> Dataset:
    ds = load_dataset("Idavidrein/gpqa", config_name)
    return ds["train"] if "train" in ds else ds[list(ds.keys())[0]]


def _load_agieval_english_mcqa() -> Dataset:
    parts = []
    for subset in AGIEVAL_ENGLISH_CONFIGS:
        print(f"[AGIEval] loading subset: {subset}")
        ds = load_dataset("lighteval/agi_eval_en", subset)
        split = ds["test"] if "test" in ds else ds[list(ds.keys())[0]]
        parts.append(_normalize_agieval_subset(split, subset))
    return concatenate_datasets(parts)


def load_eval_dataset(task: str) -> Dataset:
    task = task.lower().strip()
    if task in ["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq"]:
        return load_iid_test_set(task)
    if task == SCIENCEQA_CURRIC_TASK_NAME:
        _, _, test_ds = load_scienceqa_closedchoice_grade2_11()
        return test_ds
    if task == SCIENCEQA_GRADE12_TASK_NAME:
        return load_scienceqa_closedchoice_grade12_all()
    if task == "hellaswag":
        ds = load_dataset("rowan/hellaswag")["validation"]
        ds = ds.shuffle(seed=42)
        return ds.select(range(min(1000, len(ds))))
    if task == "gpqa":
        return _normalize_gpqa_subset(_load_gpqa_subset("gpqa_diamond"))
    if task == "gpqa_main":
        return _normalize_gpqa_subset(_load_gpqa_subset("gpqa_main"))
    if task == "agieval":
        return _load_agieval_english_mcqa()
    if _is_mmlu_eval_task(task):
        return _load_mmlu_group(_mmlu_group_from_task(task))
    raise ValueError(f"Unknown eval task: {task}")


def get_task_num_classes(task: str) -> int:
    task = task.lower().strip()
    task_num_classes = {
        "wgs": 2,
        "wgm": 2,
        "boolq": 2,
        "arc-c": 4,
        "arc-e": 4,
        "obqa": 4,
        "sciq": 4,
        SCIENCEQA_CURRIC_TASK_NAME: 4,
        SCIENCEQA_GRADE12_TASK_NAME: 4,
        "hellaswag": 4,
        "gpqa": 4,
        "gpqa_main": 4,
        "agieval": 4,
    }
    if task in task_num_classes:
        return task_num_classes[task]
    if _is_mmlu_eval_task(task):
        return 4
    raise ValueError(f"Unknown task: {task}")


__all__ = [
    "_is_mmlu_eval_task",
    "_parse_scienceqa_grade_num",
    "_scienceqa_choice_texts",
    "get_task_num_classes",
    "load_eval_dataset",
    "load_iid_test_set",
    "load_scienceqa_closedchoice_grade2_11",
    "load_scienceqa_closedchoice_grade12_all",
    "load_task_dataset",
]
