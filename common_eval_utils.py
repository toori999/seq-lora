from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Sequence
import hashlib
import math
import random
import string

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

_MulticlassAccuracy = None
_MulticlassCalibrationError = None

try:
    from torchmetrics import Accuracy, CalibrationError
except Exception:
    Accuracy = None
    CalibrationError = None

try:
    from torchmetrics.classification import MulticlassAccuracy as _MulticlassAccuracy
    from torchmetrics.classification import MulticlassCalibrationError as _MulticlassCalibrationError
except Exception:
    _MulticlassAccuracy = None
    _MulticlassCalibrationError = None

Tensor = torch.Tensor

# =========================================================================
# Prompts
# =========================================================================

PROMPT_WG = (
    "Select one of the choices that answer the following question: {question}\n"
    "Choices: A. {option1}. B. {option2}. Answer:"
)

PROMPT_ARC = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

PROMPT_OBQA = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

PROMPT_BOOLQ = (
    "Select one of the choices that answer the following question:\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Choices: A. False. B. True. Answer:"
)

PROMPT_SCIQ = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

PROMPT_4CHOICE = PROMPT_ARC
DEFAULT_CHOICE_LABELS = list(string.ascii_uppercase)

MMLU_GROUPS = {
    "science_high": [
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
    ],
    "science_college": [
        "college_physics",
        "college_chemistry",
        "college_biology",
    ],
}

MMLU_EVAL_TASK_PREFIX = "mmlu_"

AGIEVAL_ENGLISH_CONFIGS = [
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
]

SCIENCEQA_CURRIC_TASK_NAME = "scienceqa_closedchoice_grade2_11"
SCIENCEQA_GRADE12_TASK_NAME = "scienceqa_closedchoice_grade12"
SCIENCEQA_DATASET_NAME = "tcallens/scienceqa-text-only"
SCIENCEQA_GRADE_MIN = 2
SCIENCEQA_GRADE_MAX = 11
SCIENCEQA_TASK_FILTER = "closed choice"


# =========================================================================
# Tokenization helpers
# =========================================================================


def _tokenize_prompts(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dict:
    if pad_to_max_length:
        return tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
    return tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_len,
    )


def _finalize_preprocessed_dataset(
    ds: Dataset,
    keep_cols: Iterable[str] = ("input_ids", "attention_mask", "labels", "slice_id"),
) -> Dataset:
    keep = set(keep_cols)
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def _choices_obj_to_mapping(choices_obj) -> Dict[str, str]:
    if isinstance(choices_obj, dict):
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        return {str(lab): str(txt) for lab, txt in zip(labels, texts)}
    if isinstance(choices_obj, list):
        out = {}
        for c in choices_obj:
            if isinstance(c, dict) and "label" in c and "text" in c:
                out[str(c["label"])] = str(c["text"])
        return out
    return {}


def get_choice_labels(num_choices: int) -> List[str]:
    if num_choices < 2:
        raise ValueError(f"num_choices must be >= 2, got {num_choices}")
    if num_choices > len(DEFAULT_CHOICE_LABELS):
        raise ValueError(
            f"num_choices={num_choices} exceeds supported label budget "
            f"({len(DEFAULT_CHOICE_LABELS)} uppercase letters)"
        )
    return DEFAULT_CHOICE_LABELS[:num_choices]


def _ordered_choice_labels(mapping: Dict[str, str]) -> List[str]:
    labels = [str(label).strip() for label in mapping.keys() if str(label).strip()]
    alpha_rank = {label: idx for idx, label in enumerate(DEFAULT_CHOICE_LABELS)}
    if labels and all(label.upper() in alpha_rank for label in labels):
        return sorted(labels, key=lambda label: alpha_rank[label.upper()])
    return labels


def get_single_token_id(tokenizer: AutoTokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids2 = tokenizer.encode(" " + s, add_special_tokens=False)
    if len(ids2) == 1:
        return int(ids2[0])
    raise ValueError(f'"{s}" is not a single token: ids={ids}, ids_with_space={ids2}')


def answer_key_to_index(answer, label_order: Sequence[str]) -> int:
    labels = [str(label).strip() for label in label_order]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    if isinstance(answer, int):
        idx = int(answer)
        if 0 <= idx < len(labels):
            return idx
        raise ValueError(f"answer int not in 0..{len(labels) - 1}")

    ans = str(answer).strip()
    ans_upper = ans.upper()
    if ans in label_to_idx:
        return label_to_idx[ans]
    if ans_upper in label_to_idx:
        return label_to_idx[ans_upper]
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(labels):
            return idx - 1
        if 0 <= idx < len(labels):
            return idx

    raise ValueError(f"answer={answer} not compatible with labels={labels}")


def answer_index_to_key(answer_idx: int, label_order: Sequence[str]) -> str:
    idx = int(answer_idx)
    labels = [str(label).strip() for label in label_order]
    if not (0 <= idx < len(labels)):
        raise ValueError(f"answer_idx={idx} not in 0..{len(labels) - 1}")
    return labels[idx]


def make_prompt_from_choices(
    question: str,
    mapping: Dict[str, str],
    label_order: Optional[Sequence[str]] = None,
) -> str:
    ordered_labels = (
        [str(label).strip() for label in label_order]
        if label_order is not None
        else _ordered_choice_labels(mapping)
    )
    if len(ordered_labels) < 2:
        raise ValueError("Need at least 2 choices to build a prompt")
    if not all(label in mapping for label in ordered_labels):
        missing = [label for label in ordered_labels if label not in mapping]
        raise ValueError(f"Missing labels in mapping: {missing}")

    choices_text = " ".join(f"{label}. {mapping[label]}" for label in ordered_labels)
    return (
        "Select one of the choices that answers the following question:\n"
        f"{question} Choices: {choices_text} Answer:"
    )


def make_prompt_from_4choices(question: str, mapping: Dict[str, str]) -> str:
    return make_prompt_from_choices(question, mapping, label_order=get_choice_labels(4))


def preprocess_multiple_choice_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    question_field: str,
    choices_field: str,
    answer_field: str,
    pad_to_max_length: bool = True,
    question_is_nested: bool = False,
    choices_is_nested_under_question: bool = False,
    keep_extra_fields: Optional[List[str]] = None,
    expected_num_choices: Optional[int] = None,
    choice_labels: Optional[Sequence[str]] = None,
) -> Dataset:
    keep_extra_fields = keep_extra_fields or []
    static_choice_labels = [str(label).strip() for label in choice_labels] if choice_labels is not None else None

    def _fn(batch: Dict) -> Dict:
        if question_is_nested:
            questions = []
            for q in batch[question_field]:
                if isinstance(q, dict):
                    questions.append(str(q.get("stem", q.get("text", ""))))
                else:
                    questions.append(str(q))
        else:
            questions = [str(x) for x in batch[question_field]]

        if choices_is_nested_under_question:
            choices_list = []
            for q in batch[question_field]:
                if isinstance(q, dict):
                    choices_list.append(q.get(choices_field, {}))
                else:
                    choices_list.append({})
        else:
            choices_list = batch[choices_field]

        answers = batch[answer_field]
        prompts: List[str] = []
        labels: List[int] = []
        for i in range(len(questions)):
            try:
                mapping = _choices_obj_to_mapping(choices_list[i])
                label_order = list(static_choice_labels) if static_choice_labels is not None else _ordered_choice_labels(mapping)
                if expected_num_choices is not None and len(label_order) != int(expected_num_choices):
                    raise ValueError(
                        f"expected {int(expected_num_choices)} choices, got {len(label_order)}"
                    )
                if len(label_order) < 2:
                    raise ValueError("need at least 2 valid choices")
                if not all(label in mapping for label in label_order):
                    raise ValueError("missing expected choice labels")
                y = answer_key_to_index(answers[i], label_order)
                prompts.append(make_prompt_from_choices(questions[i], mapping, label_order=label_order))
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        for k in keep_extra_fields:
            enc[k] = batch[k]
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1)
    return _finalize_preprocessed_dataset(
        ds2,
        keep_cols=("input_ids", "attention_mask", "labels", *keep_extra_fields),
    )


def preprocess_4choice_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    question_field: str,
    choices_field: str,
    answer_field: str,
    pad_to_max_length: bool = True,
    question_is_nested: bool = False,
    choices_is_nested_under_question: bool = False,
    keep_extra_fields: Optional[List[str]] = None,
) -> Dataset:
    return preprocess_multiple_choice_dataset(
        ds=ds,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field=question_field,
        choices_field=choices_field,
        answer_field=answer_field,
        pad_to_max_length=pad_to_max_length,
        question_is_nested=question_is_nested,
        choices_is_nested_under_question=choices_is_nested_under_question,
        keep_extra_fields=keep_extra_fields,
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )


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
    return [str(v).strip() for v in values if str(v).strip()]


def preprocess_scienceqa_curriculum(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    keep_extra = [c for c in ["slice_id", "grade_num", "num_choices"] if c in ds.column_names]

    def _fn(batch: Dict) -> Dict:
        prompts: List[str] = []
        labels: List[int] = []
        num_choices_list: List[int] = []

        for i in range(len(batch["question"])):
            try:
                choices = _scienceqa_choice_texts(batch["choices"][i])
                k = len(choices)
                if k < 2 or k > 4:
                    raise ValueError(f"unsupported num_choices={k}")
                label_order = get_choice_labels(k)
                mapping = {label_order[j]: choices[j] for j in range(k)}
                answer = answer_key_to_index(batch["answer"][i], label_order)
                prompt = make_prompt_from_choices(str(batch["question"][i]), mapping, label_order=label_order)
                prompts.append(prompt)
                labels.append(answer)
                num_choices_list.append(k)
            except Exception:
                prompts.append("")
                labels.append(-1)
                num_choices_list.append(-1)

        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        enc["num_choices"] = num_choices_list
        for k in keep_extra:
            if k != "num_choices":
                enc[k] = batch[k]
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= 4)
    return _finalize_preprocessed_dataset(
        ds2,
        keep_cols=("input_ids", "attention_mask", "labels", "num_choices", *keep_extra),
    )


# =========================================================================
# Task specific preprocessing
# =========================================================================


def preprocess_wg(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        sents, o1, o2, ans = batch["sentence"], batch["option1"], batch["option2"], batch["answer"]
        prompts, labels = [], []
        for i in range(len(sents)):
            prompts.append(PROMPT_WG.format(question=sents[i], option1=o1[i], option2=o2[i]))
            labels.append(0 if str(ans[i]) == "1" else 1)
        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    return _finalize_preprocessed_dataset(ds2)



def preprocess_arc(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    if "question" in ds.column_names and len(ds) > 0 and isinstance(ds[0]["question"], dict):
        return preprocess_multiple_choice_dataset(
            ds=ds,
            tokenizer=tokenizer,
            max_len=max_len,
            question_field="question",
            choices_field="choices",
            answer_field="answerKey",
            pad_to_max_length=pad_to_max_length,
            question_is_nested=True,
            choices_is_nested_under_question=True,
            expected_num_choices=4,
            choice_labels=get_choice_labels(4),
        )

    def _get_q_and_choices(ex_question, ex_choices):
        if isinstance(ex_question, dict):
            return ex_question.get("stem", ""), ex_question.get("choices", ex_choices)
        return str(ex_question), ex_choices

    def _fn(batch: Dict) -> Dict:
        questions, choices_col, answer_keys = batch["question"], batch.get("choices", None), batch["answerKey"]
        prompts, labels = [], []
        for i in range(len(answer_keys)):
            try:
                ex_choices = choices_col[i] if choices_col is not None else None
                qtext, ch = _get_q_and_choices(questions[i], ex_choices)
                if ch is None:
                    raise ValueError("choices is None")
                labs, txts = ch["label"], ch["text"]
                if len(labs) < 2 or len(txts) < 2:
                    raise ValueError("choices < 2")
                if len(labs) != len(txts):
                    raise ValueError("choice labels/text length mismatch")
                label_order = [str(x) for x in labs]
                mapping = {str(lab): str(txt) for lab, txt in zip(labs, txts)}
                y = answer_key_to_index(answer_keys[i], label_order)
                prompts.append(make_prompt_from_choices(qtext, mapping, label_order=label_order))
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True).filter(lambda ex: ex["labels"] != -1)
    if len(ds2) == 0:
        raise RuntimeError("ARC preprocess produced 0 examples.")
    return _finalize_preprocessed_dataset(ds2)



def preprocess_obqa(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    keep = [c for c in ds.column_names if c in ["slice_id"]]
    return preprocess_multiple_choice_dataset(
        ds=ds,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field="question_stem",
        choices_field="choices",
        answer_field="answerKey",
        pad_to_max_length=pad_to_max_length,
        keep_extra_fields=keep,
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )



def preprocess_boolq(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        qs = batch["question"]
        ps = batch["passage"]
        ans = batch["label"]

        prompts = []
        labels = []
        for i in range(len(qs)):
            prompt = PROMPT_BOOLQ.format(question=qs[i], passage=ps[i])
            y = int(ans[i])
            prompts.append(prompt)
            labels.append(y)

        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    return _finalize_preprocessed_dataset(ds2, keep_cols=("input_ids", "attention_mask", "labels"))



def preprocess_sciq(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _format_sciq(ex):
        opts = [ex["distractor1"], ex["distractor2"], ex["distractor3"], ex["correct_answer"]]
        random.shuffle(opts)
        labels = get_choice_labels(len(opts))
        return {
            "question": ex["question"],
            "choices": [{"label": labels[i], "text": str(text)} for i, text in enumerate(opts)],
            "answerKey": answer_index_to_key(opts.index(ex["correct_answer"]), labels),
        }

    remove_cols = [c for c in ["distractor1", "distractor2", "distractor3", "correct_answer", "support"] if c in ds.column_names]
    ds_fmt = ds.map(_format_sciq, remove_columns=remove_cols)
    return preprocess_multiple_choice_dataset(
        ds=ds_fmt,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
        pad_to_max_length=pad_to_max_length,
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )


def preprocess_hellaswag(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _format_hellaswag(ex):
        labels = get_choice_labels(len(ex["endings"]))
        return {
            "question": ex["ctx"],
            "choices": [{"label": labels[i], "text": str(text)} for i, text in enumerate(ex["endings"])],
            "answerKey": answer_index_to_key(int(ex["label"]), labels),
        }

    remove_cols = [c for c in ["activity_label", "ctx_a", "ctx_b", "split", "split_type", "source_id", "endings", "label"] if c in ds.column_names]
    ds_fmt = ds.map(_format_hellaswag, remove_columns=remove_cols)
    return preprocess_multiple_choice_dataset(
        ds=ds_fmt,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
        pad_to_max_length=pad_to_max_length,
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )


def preprocess_mmlu_subset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        questions = [str(x) for x in batch["question"]]
        choices = batch["choices"]
        answers = batch["answer"]
        prompts: List[str] = []
        labels: List[int] = []
        for i in range(len(questions)):
            try:
                ch = choices[i]
                if not isinstance(ch, (list, tuple)) or len(ch) < 2:
                    raise ValueError("choices must have at least 2 entries")
                label_order = get_choice_labels(len(ch))
                mapping = {label_order[j]: str(ch[j]) for j in range(len(ch))}
                y = answer_key_to_index(answers[i], label_order)
                prompts.append(make_prompt_from_choices(questions[i], mapping, label_order=label_order))
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1)
    return _finalize_preprocessed_dataset(ds2, keep_cols=("input_ids", "attention_mask", "labels"))


def preprocess_gpqa(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    return preprocess_multiple_choice_dataset(
        ds=ds,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
        pad_to_max_length=pad_to_max_length,
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )


def preprocess_agieval(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    return preprocess_multiple_choice_dataset(
        ds=ds,
        tokenizer=tokenizer,
        max_len=max_len,
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
        pad_to_max_length=pad_to_max_length,
        keep_extra_fields=["source_subset"],
        expected_num_choices=4,
        choice_labels=get_choice_labels(4),
    )


 # =========================================================================
 # Dataset loaders
 # =========================================================================


def _concat_available_splits(parts: List[Dataset]) -> Dataset:
    if not parts:
        raise ValueError("No dataset splits available to concatenate.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_openbookqa_from_local_cache() -> Optional[DatasetDict]:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "openbookqa" / "main" / "0.0.0"
    if not cache_root.exists():
        return None
    revisions = sorted([p for p in cache_root.iterdir() if p.is_dir()])
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
            print(f"[Dataset] Falling back to local cached OpenBookQA after load_dataset failure: {exc}")
            return cached
        raise


def _load_mmlu_subject_from_local_cache(subject: str) -> Optional[Dataset]:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "cais___mmlu" / subject / "0.0.0"
    if not cache_root.exists():
        return None
    revisions = sorted([p for p in cache_root.iterdir() if p.is_dir()])
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
    elif task == "wgm":
        ds = load_dataset("winogrande", "winogrande_m")
        return ds["train"], ds["validation"], ds["validation"]
    elif task == "arc-c":
        ds = load_dataset("ai2_arc", "ARC-Challenge")
        return ds["train"], ds["validation"], ds["test"]
    elif task == "arc-e":
        ds = load_dataset("ai2_arc", "ARC-Easy")
        return ds["train"], ds["validation"], ds["test"]
    elif task == "obqa":
        ds = _load_openbookqa_dataset()
        return ds["train"], ds["validation"], ds["test"]
    elif task == "boolq":
        ds = load_dataset("super_glue", "boolq")
        return ds["train"], ds["validation"], ds["validation"]
    elif task == "sciq":
        ds = load_dataset("sciq")
        return ds["train"], ds["validation"], ds["test"]
    elif task == SCIENCEQA_CURRIC_TASK_NAME:
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


def _concat_all_splits(ds_dict) -> Dataset:
    preferred = ["auxiliary_train", "dev", "validation", "test"]
    split_names = [name for name in preferred if name in ds_dict]
    split_names.extend([name for name in ds_dict.keys() if name not in split_names])
    if not split_names:
        raise ValueError("No available splits found.")
    parts = [ds_dict[name] for name in split_names]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_mmlu_subject(subject: str) -> Dataset:
    try:
        return _pick_mmlu_split(load_dataset("cais/mmlu", subject))
    except Exception as exc:
        cached = _load_mmlu_subject_from_local_cache(subject)
        if cached is not None:
            print(f"[Dataset] Falling back to local cached MMLU subject '{subject}' after load_dataset failure: {exc}")
            return cached
        raise


def _load_mmlu_group(group: str) -> Dataset:
    group = group.lower().strip()
    if group not in MMLU_GROUPS:
        raise ValueError(f"Unknown MMLU group: {group}")
    subjects = MMLU_GROUPS[group]

    parts = []
    for sub in subjects:
        print(f"[MMLU] loading subject: {sub}")
        parts.append(_load_mmlu_subject(sub))
    return concatenate_datasets(parts)


def _is_mmlu_eval_task(task: str) -> bool:
    task = task.lower().strip()
    return task.startswith(MMLU_EVAL_TASK_PREFIX)


def _mmlu_group_from_task(task: str) -> str:
    task = task.lower().strip()
    if not _is_mmlu_eval_task(task):
        raise ValueError(f"Unknown MMLU eval task: {task}")
    return task[len(MMLU_EVAL_TASK_PREFIX):]


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
        opts = distractors + [correct]
        random.shuffle(opts)
        labels = get_choice_labels(len(opts))
        answer_key = answer_index_to_key(opts.index(correct), labels)
        return {
            "question": question,
            "choices": _make_choice_rows(opts),
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
            return {"question": "", "choices": [], "answerKey": "Z", "source_subset": subset_name}

        label_order = get_choice_labels(len(options))
        answer_key = None
        if isinstance(answer, int):
            if 0 <= int(answer) < len(label_order):
                answer_key = answer_index_to_key(int(answer), label_order)
        else:
            try:
                answer_key = answer_index_to_key(answer_key_to_index(answer, label_order), label_order)
            except Exception:
                answer_key = None

        if answer_key is None:
            return {"question": "", "choices": [], "answerKey": "Z", "source_subset": subset_name}

        return {
            "question": question,
            "choices": _make_choice_rows([str(x) for x in options]),
            "answerKey": answer_key,
            "source_subset": subset_name,
        }

    ds2 = ds.map(_fmt, remove_columns=ds.column_names)
    return ds2.filter(lambda ex: ex["answerKey"] in get_choice_labels(4) and len(ex["choices"]) == 4)


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



def preprocess_task(
    task: str,
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    if task in ["wgs", "wgm"]:
        return preprocess_wg(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task in ["arc-c", "arc-e"]:
        return preprocess_arc(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == "obqa":
        return preprocess_obqa(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == "boolq":
        return preprocess_boolq(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == "sciq":
        return preprocess_sciq(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == "hellaswag":
        return preprocess_hellaswag(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task in ["gpqa", "gpqa_main"]:
        return preprocess_gpqa(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == "agieval":
        return preprocess_agieval(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == SCIENCEQA_CURRIC_TASK_NAME:
        return preprocess_scienceqa_curriculum(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif task == SCIENCEQA_GRADE12_TASK_NAME:
        return preprocess_scienceqa_curriculum(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    elif _is_mmlu_eval_task(task):
        return preprocess_mmlu_subset(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    raise ValueError(f"Unknown task: {task}")


# =========================
# Dynamic eval collator
# =========================


@dataclass
class DynamicEvalCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([int(f["labels"]) for f in features], dtype=torch.long)
        for key in features[0].keys():
            if key not in {"input_ids", "attention_mask", "labels"}:
                batch[key] = [f[key] for f in features]
        return batch


# =========================
# Choice token ids & metrics
# =========================


def get_choice_token_ids(tokenizer: AutoTokenizer, device: torch.device, num_classes: int) -> torch.Tensor:
    choices = get_choice_labels(num_classes)
    ids = [get_single_token_id(tokenizer, c) for c in choices]
    print(f"[Choice token ids] classes={num_classes}, ids={dict(zip(choices, ids))}")
    return torch.tensor(ids, device=device, dtype=torch.long)



def make_accuracy(device: torch.device, num_classes: int):
    if Accuracy is not None:
        try:
            return Accuracy(task="multiclass", num_classes=num_classes).to(device)
        except Exception:
            pass
    if _MulticlassAccuracy is None:
        raise RuntimeError("No usable torchmetrics Accuracy implementation found.")
    return _MulticlassAccuracy(num_classes=num_classes).to(device)



def make_ece(device: torch.device, num_classes: int, n_bins: int = 15):
    if CalibrationError is not None:
        try:
            return CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins, norm="l1").to(device)
        except Exception:
            pass
    if _MulticlassCalibrationError is None:
        raise RuntimeError("No usable torchmetrics CalibrationError implementation found.")
    return _MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm="l1").to(device)


# =========================
# PEFT / lm_head helpers
# =========================


def get_active_adapter_name(model: nn.Module) -> str:
    if hasattr(model, "active_adapter"):
        a = model.active_adapter
        if isinstance(a, str):
            return a
        if isinstance(a, (list, tuple)) and len(a) > 0:
            return str(a[0])
    return "default"



def pick_adapter_module(maybe_mod, adapter_name: str):
    if isinstance(maybe_mod, (nn.ModuleDict, dict)):
        if adapter_name in maybe_mod:
            return maybe_mod[adapter_name]
        try:
            return next(iter(maybe_mod.values()))
        except StopIteration:
            return None
    return maybe_mod



def pick_scaling(maybe_scaling, adapter_name: str):
    if isinstance(maybe_scaling, dict):
        if adapter_name in maybe_scaling:
            return maybe_scaling[adapter_name]
        try:
            return next(iter(maybe_scaling.values()))
        except StopIteration:
            return 1.0
    if isinstance(maybe_scaling, (list, tuple)):
        return maybe_scaling[0] if len(maybe_scaling) > 0 else 1.0
    return maybe_scaling


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


def init_blob_rho_(rho: torch.Tensor, eps: float) -> torch.Tensor:
    if eps < 0:
        nn.init.uniform_(rho, eps - 1.0, eps)
    else:
        nn.init.uniform_(rho, eps / math.sqrt(2.0), eps)
    return rho


def blob_sigma_from_rho(rho: torch.Tensor) -> torch.Tensor:
    return rho.square()


def blob_kl_div_stable(
    mu_q: torch.Tensor,
    rho_q: torch.Tensor,
    mu_p: float = 0.0,
    sigma_p: float = 0.2,
) -> torch.Tensor:
    eps = 1e-6
    sigma_q = blob_sigma_from_rho(rho_q)
    kl = (
        math.log(float(sigma_p) + eps)
        - torch.log(sigma_q.to(torch.float64) + eps)
        + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - float(mu_p)) ** 2)
        / (2 * (float(sigma_p) ** 2) + eps)
        - 0.5
    )
    return kl.sum()


def blob_sample_lora_noise(
    x: torch.Tensor,
    lora_a_weight: torch.Tensor,
    lora_b_weight: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    sigma_a = blob_sigma_from_rho(rho).to(dtype=lora_a_weight.dtype)
    if x.dim() == 2:
        r_a = torch.empty((x.size(0), lora_a_weight.size(1)), device=x.device, dtype=x.dtype).uniform_(-1, 1).sign()
        s_a = torch.empty((x.size(0), lora_a_weight.size(0)), device=x.device, dtype=x.dtype).uniform_(-1, 1).sign()
    elif x.dim() == 3:
        r_a = torch.empty((x.size(0), x.size(1), lora_a_weight.size(1)), device=x.device, dtype=x.dtype).uniform_(-1, 1).sign()
        s_a = torch.empty((x.size(0), x.size(1), lora_a_weight.size(0)), device=x.device, dtype=x.dtype).uniform_(-1, 1).sign()
    else:
        raise ValueError(f"Unsupported BLoB input rank {x.dim()}, expected 2 or 3.")

    lora_noise_a = sigma_a * torch.randn_like(lora_a_weight)
    return (((x * r_a) @ lora_noise_a.transpose(0, 1)) * s_a) @ lora_b_weight.transpose(0, 1)


def get_transformer_and_lm_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "model"):
        transformer = base.model
    elif hasattr(base, "transformer"):
        transformer = base.transformer
    else:
        raise RuntimeError("Cannot locate transformer body.")

    if hasattr(base, "lm_head"):
        lm_head = base.lm_head
    else:
        lm_head = base.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Cannot locate lm_head.")
    return transformer, lm_head



def lm_head_has_lora(lm_head: nn.Module) -> bool:
    return hasattr(lm_head, "lora_A") and hasattr(lm_head, "lora_B")



def get_lm_head_lora_scaling(lm_head: nn.Module, adapter: str) -> float:
    if hasattr(lm_head, "scaling"):
        sc = pick_scaling(getattr(lm_head, "scaling"), adapter)
        if isinstance(sc, (float, int)):
            return float(sc)

    r = getattr(lm_head, "r", None)
    if isinstance(r, dict):
        r = r.get(adapter, None)
    alpha = getattr(lm_head, "lora_alpha", None)
    if isinstance(alpha, dict):
        alpha = alpha.get(adapter, None)
    if r and alpha and float(r) != 0:
        return float(alpha) / float(r)
    return 1.0



def get_lm_head_dropout(lm_head: nn.Module, adapter: str) -> Optional[nn.Module]:
    d = getattr(lm_head, "lora_dropout", None)
    if isinstance(d, (nn.ModuleDict, dict)) and adapter in d:
        return d[adapter]
    if isinstance(d, nn.Module) and not isinstance(d, nn.ModuleDict):
        return d
    return None



def get_lm_head_lora_A_weight(lm_head: nn.Module, adapter: str) -> Optional[torch.Tensor]:
    if not lm_head_has_lora(lm_head):
        return None
    A_mod = pick_adapter_module(getattr(lm_head, "lora_A", None), adapter)
    return A_mod.weight if hasattr(A_mod, "weight") else None



def get_lm_head_lora_B_choice_fp32(
    lm_head: nn.Module,
    adapter: str,
    choice_token_ids: torch.Tensor,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    if not lm_head_has_lora(lm_head):
        return None
    if adapter in cache:
        return cache[adapter]
    B_mod = pick_adapter_module(getattr(lm_head, "lora_B", None), adapter)
    if B_mod is None or not hasattr(B_mod, "weight"):
        return None
    B_choice_fp32 = B_mod.weight.index_select(0, choice_token_ids).detach().to(device=device, dtype=torch.float32).contiguous()
    cache[adapter] = B_choice_fp32
    return B_choice_fp32


@dataclass
class ChoiceHeadCache:
    W_choice_fp32: torch.Tensor
    b_choice_fp32: Optional[torch.Tensor]
    transformer: nn.Module
    lm_head: nn.Module
    choice_token_ids: torch.Tensor
    num_classes: int
    B_choice_fp32_by_adapter: Dict[str, torch.Tensor] = field(default_factory=dict)
    bayes_eps: float = 0.0



def build_choice_head_cache(
    model: nn.Module,
    choice_token_ids: torch.Tensor,
    device: torch.device,
    bayes_eps: float = 0.0,
) -> ChoiceHeadCache:
    transformer, lm_head = get_transformer_and_lm_head(model)
    W_choice = lm_head.weight.index_select(0, choice_token_ids).detach()
    b_choice = (
        lm_head.bias.index_select(0, choice_token_ids).detach()
        if getattr(lm_head, "bias", None) is not None
        else None
    )
    return ChoiceHeadCache(
        W_choice_fp32=W_choice.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        b_choice_fp32=(
            None
            if b_choice is None
            else b_choice.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        ),
        transformer=transformer,
        lm_head=lm_head,
        choice_token_ids=choice_token_ids,
        num_classes=len(choice_token_ids),
        bayes_eps=float(bayes_eps),
    )



def restricted_choice_logits_last_token(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_cache: ChoiceHeadCache,
    amp_dtype: torch.dtype,
    last_idx: Optional[torch.Tensor] = None,
    batch_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = input_ids.device
    bsz = input_ids.size(0)

    if last_idx is None:
        last_token_is_valid = attention_mask[:, -1].to(dtype=torch.bool)
        last_idx = torch.empty((bsz,), device=device, dtype=torch.long)
        last_idx[last_token_is_valid] = attention_mask.size(1) - 1
        if (~last_token_is_valid).any():
            last_idx[~last_token_is_valid] = attention_mask[~last_token_is_valid].sum(dim=1) - 1
    if batch_idx is None:
        batch_idx = torch.arange(bsz, device=device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = choice_cache.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        h_last = out.last_hidden_state[batch_idx, last_idx, :]

    h_last_fp32 = h_last.float()
    logits = h_last_fp32 @ choice_cache.W_choice_fp32.t()
    if choice_cache.b_choice_fp32 is not None:
        logits += choice_cache.b_choice_fp32.view(1, choice_cache.num_classes)

    lm_head = choice_cache.lm_head
    if lm_head_has_lora(lm_head):
        adapter = get_active_adapter_name(model)
        A_w = get_lm_head_lora_A_weight(lm_head, adapter)
        B_choice_fp32 = get_lm_head_lora_B_choice_fp32(
            lm_head=lm_head,
            adapter=adapter,
            choice_token_ids=choice_cache.choice_token_ids,
            device=device,
            cache=choice_cache.B_choice_fp32_by_adapter,
        )

        if A_w is not None and B_choice_fp32 is not None:
            # Laplace Jacobian code paths call this function under autograd.
            # Cached / inference-mode tensors from eval fast paths must be cloned
            # before they participate in ops whose intermediates are saved for backward.
            if torch.is_grad_enabled():
                h_last = h_last.clone()
                A_w = A_w.clone()
                B_choice_fp32 = B_choice_fp32.clone()

            drop = get_lm_head_dropout(lm_head, adapter)
            h_for_lora = h_last if drop is None else drop(h_last)
            h_for_lora = h_for_lora.to(dtype=A_w.dtype)
            z_td = h_for_lora @ A_w.to(dtype=A_w.dtype).t()
            lora_logits = (z_td.float() @ B_choice_fp32.t()) * float(get_lm_head_lora_scaling(lm_head, adapter))

            rho = getattr(lm_head, f"blob_rho_{adapter}", None)
            if isinstance(rho, nn.Parameter) and bool(getattr(lm_head, f"blob_sample_{adapter}", True)):
                lora_logits += blob_sample_lora_noise(
                    x=h_for_lora,
                    lora_a_weight=A_w.to(dtype=A_w.dtype),
                    lora_b_weight=B_choice_fp32.to(dtype=A_w.dtype),
                    rho=rho.to(dtype=A_w.dtype),
                ).float() * float(get_lm_head_lora_scaling(lm_head, adapter))

            logits += lora_logits

    return logits



def logits_via_lm_head_last_token_for_kfac(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_cache: ChoiceHeadCache,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_ids.device
    bsz = input_ids.size(0)
    last_token_is_valid = attention_mask[:, -1].to(dtype=torch.bool)
    last_idx = torch.empty((bsz,), device=device, dtype=torch.long)
    last_idx[last_token_is_valid] = attention_mask.size(1) - 1
    if (~last_token_is_valid).any():
        last_idx[~last_token_is_valid] = attention_mask[~last_token_is_valid].sum(dim=1) - 1
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = choice_cache.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits_v = choice_cache.lm_head(out.last_hidden_state[torch.arange(bsz, device=device), last_idx, :])
    return logits_v.index_select(-1, choice_cache.choice_token_ids)



def set_inference_fast(model: nn.Module):
    if hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "base_model") and hasattr(model.base_model, "config") and hasattr(model.base_model.config, "use_cache"):
        model.base_model.config.use_cache = False


__all__ = [
    "PROMPT_WG",
    "PROMPT_ARC",
    "PROMPT_OBQA",
    "PROMPT_BOOLQ",
    "PROMPT_SCIQ",
    "DEFAULT_CHOICE_LABELS",
    "_tokenize_prompts",
    "get_choice_labels",
    "get_single_token_id",
    "answer_key_to_index",
    "answer_index_to_key",
    "make_prompt_from_choices",
    "preprocess_wg",
    "preprocess_arc",
    "preprocess_obqa",
    "preprocess_boolq",
    "preprocess_sciq",
    "preprocess_hellaswag",
    "preprocess_mmlu_subset",
    "preprocess_gpqa",
    "preprocess_agieval",
    "make_prompt_from_4choices",
    "preprocess_multiple_choice_dataset",
    "preprocess_4choice_dataset",
    "load_task_dataset",
    "load_eval_dataset",
    "load_iid_test_set",
    "get_task_num_classes",
    "preprocess_task",
    "DynamicEvalCollator",
    "get_choice_token_ids",
    "make_accuracy",
    "make_ece",
    "ChoiceHeadCache",
    "get_active_adapter_name",
    "pick_adapter_module",
    "pick_scaling",
    "softplus",
    "init_blob_rho_",
    "blob_sigma_from_rho",
    "blob_kl_div_stable",
    "blob_sample_lora_noise",
    "get_transformer_and_lm_head",
    "lm_head_has_lora",
    "get_lm_head_lora_scaling",
    "get_lm_head_dropout",
    "get_lm_head_lora_A_weight",
    "get_lm_head_lora_B_choice_fp32",
    "build_choice_head_cache",
    "restricted_choice_logits_last_token",
    "logits_via_lm_head_last_token_for_kfac",
    "set_inference_fast",
]
