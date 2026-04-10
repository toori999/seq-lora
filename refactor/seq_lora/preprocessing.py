from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import random

from datasets import Dataset
from transformers import AutoTokenizer

from .constants import (
    PROMPT_BOOLQ,
    PROMPT_WG,
    SCIENCEQA_CURRIC_TASK_NAME,
    SCIENCEQA_GRADE12_TASK_NAME,
)
from .datasets import _is_mmlu_eval_task, _scienceqa_choice_texts
from .prompts import (
    _choices_obj_to_mapping,
    _ordered_choice_labels,
    answer_index_to_key,
    answer_key_to_index,
    get_choice_labels,
    make_prompt_from_choices,
)


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
    return ds.remove_columns([column for column in ds.column_names if column not in keep])


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
    choice_labels: Optional[List[str]] = None,
) -> Dataset:
    keep_extra_fields = keep_extra_fields or []
    static_choice_labels = (
        [str(label).strip() for label in choice_labels]
        if choice_labels is not None
        else None
    )

    def _fn(batch: Dict) -> Dict:
        if question_is_nested:
            questions = []
            for question in batch[question_field]:
                if isinstance(question, dict):
                    questions.append(str(question.get("stem", question.get("text", ""))))
                else:
                    questions.append(str(question))
        else:
            questions = [str(value) for value in batch[question_field]]

        if choices_is_nested_under_question:
            choices_list = []
            for question in batch[question_field]:
                if isinstance(question, dict):
                    choices_list.append(question.get(choices_field, {}))
                else:
                    choices_list.append({})
        else:
            choices_list = batch[choices_field]

        answers = batch[answer_field]
        prompts: List[str] = []
        labels: List[int] = []
        for idx in range(len(questions)):
            try:
                mapping = _choices_obj_to_mapping(choices_list[idx])
                label_order = (
                    list(static_choice_labels)
                    if static_choice_labels is not None
                    else _ordered_choice_labels(mapping)
                )
                if expected_num_choices is not None and len(label_order) != int(expected_num_choices):
                    raise ValueError(
                        f"expected {int(expected_num_choices)} choices, got {len(label_order)}"
                    )
                if len(label_order) < 2:
                    raise ValueError("need at least 2 valid choices")
                if not all(label in mapping for label in label_order):
                    raise ValueError("missing expected choice labels")
                labels.append(answer_key_to_index(answers[idx], label_order))
                prompts.append(
                    make_prompt_from_choices(
                        questions[idx], mapping, label_order=label_order
                    )
                )
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
        enc["labels"] = labels
        for key in keep_extra_fields:
            enc[key] = batch[key]
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


def preprocess_scienceqa_curriculum(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    keep_extra = [
        column
        for column in ["slice_id", "grade_num", "num_choices"]
        if column in ds.column_names
    ]

    def _fn(batch: Dict) -> Dict:
        prompts: List[str] = []
        labels: List[int] = []
        num_choices_list: List[int] = []

        for idx in range(len(batch["question"])):
            try:
                choices = _scienceqa_choice_texts(batch["choices"][idx])
                num_choices = len(choices)
                if num_choices < 2 or num_choices > 4:
                    raise ValueError(f"unsupported num_choices={num_choices}")
                label_order = get_choice_labels(num_choices)
                mapping = {
                    label_order[choice_idx]: choices[choice_idx]
                    for choice_idx in range(num_choices)
                }
                answer = answer_key_to_index(batch["answer"][idx], label_order)
                prompts.append(
                    make_prompt_from_choices(
                        str(batch["question"][idx]), mapping, label_order=label_order
                    )
                )
                labels.append(answer)
                num_choices_list.append(num_choices)
            except Exception:
                prompts.append("")
                labels.append(-1)
                num_choices_list.append(-1)

        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
        enc["labels"] = labels
        enc["num_choices"] = num_choices_list
        for key in keep_extra:
            if key != "num_choices":
                enc[key] = batch[key]
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(
        lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= 4
    )
    return _finalize_preprocessed_dataset(
        ds2,
        keep_cols=("input_ids", "attention_mask", "labels", "num_choices", *keep_extra),
    )


def preprocess_wg(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        sentences = batch["sentence"]
        option1 = batch["option1"]
        option2 = batch["option2"]
        answers = batch["answer"]
        prompts = []
        labels = []
        for idx in range(len(sentences)):
            prompts.append(
                PROMPT_WG.format(
                    question=sentences[idx],
                    option1=option1[idx],
                    option2=option2[idx],
                )
            )
            labels.append(0 if str(answers[idx]) == "1" else 1)
        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
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
        questions = batch["question"]
        choices_col = batch.get("choices", None)
        answer_keys = batch["answerKey"]
        prompts = []
        labels = []
        for idx in range(len(answer_keys)):
            try:
                ex_choices = choices_col[idx] if choices_col is not None else None
                question_text, choices = _get_q_and_choices(questions[idx], ex_choices)
                if choices is None:
                    raise ValueError("choices is None")
                labels_raw, texts_raw = choices["label"], choices["text"]
                if len(labels_raw) < 2 or len(texts_raw) < 2:
                    raise ValueError("choices < 2")
                if len(labels_raw) != len(texts_raw):
                    raise ValueError("choice labels/text length mismatch")
                label_order = [str(label) for label in labels_raw]
                mapping = {
                    str(label): str(text)
                    for label, text in zip(labels_raw, texts_raw)
                }
                labels.append(answer_key_to_index(answer_keys[idx], label_order))
                prompts.append(
                    make_prompt_from_choices(
                        question_text, mapping, label_order=label_order
                    )
                )
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
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
    keep = [column for column in ds.column_names if column in ["slice_id"]]
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
        questions = batch["question"]
        passages = batch["passage"]
        answers = batch["label"]
        prompts = []
        labels = []
        for idx in range(len(questions)):
            prompts.append(
                PROMPT_BOOLQ.format(
                    question=questions[idx],
                    passage=passages[idx],
                )
            )
            labels.append(int(answers[idx]))

        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    return _finalize_preprocessed_dataset(
        ds2, keep_cols=("input_ids", "attention_mask", "labels")
    )


def preprocess_sciq(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    def _format_sciq(ex):
        options = [
            ex["distractor1"],
            ex["distractor2"],
            ex["distractor3"],
            ex["correct_answer"],
        ]
        random.shuffle(options)
        labels = get_choice_labels(len(options))
        return {
            "question": ex["question"],
            "choices": [
                {"label": labels[i], "text": str(text)}
                for i, text in enumerate(options)
            ],
            "answerKey": answer_index_to_key(
                options.index(ex["correct_answer"]), labels
            ),
        }

    remove_cols = [
        column
        for column in [
            "distractor1",
            "distractor2",
            "distractor3",
            "correct_answer",
            "support",
        ]
        if column in ds.column_names
    ]
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
            "choices": [
                {"label": labels[i], "text": str(text)}
                for i, text in enumerate(ex["endings"])
            ],
            "answerKey": answer_index_to_key(int(ex["label"]), labels),
        }

    remove_cols = [
        column
        for column in [
            "activity_label",
            "ctx_a",
            "ctx_b",
            "split",
            "split_type",
            "source_id",
            "endings",
            "label",
        ]
        if column in ds.column_names
    ]
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
        questions = [str(question) for question in batch["question"]]
        choices = batch["choices"]
        answers = batch["answer"]
        prompts: List[str] = []
        labels: List[int] = []
        for idx in range(len(questions)):
            try:
                options = choices[idx]
                if not isinstance(options, (list, tuple)) or len(options) < 2:
                    raise ValueError("choices must have at least 2 entries")
                label_order = get_choice_labels(len(options))
                mapping = {
                    label_order[choice_idx]: str(options[choice_idx])
                    for choice_idx in range(len(options))
                }
                labels.append(answer_key_to_index(answers[idx], label_order))
                prompts.append(
                    make_prompt_from_choices(
                        questions[idx], mapping, label_order=label_order
                    )
                )
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(
            tokenizer, prompts, max_len, pad_to_max_length=pad_to_max_length
        )
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1)
    return _finalize_preprocessed_dataset(
        ds2, keep_cols=("input_ids", "attention_mask", "labels")
    )


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


def preprocess_task(
    task: str,
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    if task in ["wgs", "wgm"]:
        return preprocess_wg(ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length)
    if task in ["arc-c", "arc-e"]:
        return preprocess_arc(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task == "obqa":
        return preprocess_obqa(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task == "boolq":
        return preprocess_boolq(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task == "sciq":
        return preprocess_sciq(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task == "hellaswag":
        return preprocess_hellaswag(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task in ["gpqa", "gpqa_main"]:
        return preprocess_gpqa(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task == "agieval":
        return preprocess_agieval(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if task in [SCIENCEQA_CURRIC_TASK_NAME, SCIENCEQA_GRADE12_TASK_NAME]:
        return preprocess_scienceqa_curriculum(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    if _is_mmlu_eval_task(task):
        return preprocess_mmlu_subset(
            ds, tokenizer, max_len, pad_to_max_length=pad_to_max_length
        )
    raise ValueError(f"Unknown task: {task}")


__all__ = [
    "_finalize_preprocessed_dataset",
    "_tokenize_prompts",
    "preprocess_4choice_dataset",
    "preprocess_agieval",
    "preprocess_arc",
    "preprocess_boolq",
    "preprocess_gpqa",
    "preprocess_hellaswag",
    "preprocess_mmlu_subset",
    "preprocess_multiple_choice_dataset",
    "preprocess_obqa",
    "preprocess_scienceqa_curriculum",
    "preprocess_sciq",
    "preprocess_task",
    "preprocess_wg",
]
