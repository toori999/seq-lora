from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from datasets import Dataset

from evaluation.common import (
    get_choice_token_ids,
    get_task_num_classes,
    preprocess_task,
)

Tensor = torch.Tensor

SUPPORTED_EVAL_PROTOCOLS = {"default", "bayesian_peft"}
BAYESIAN_PEFT_TASKS = {"arc-c", "arc-e", "obqa"}

_BP_PROMPT_ARC = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

_BP_PROMPT_OBQA = """Return the label of the correct answer for the question below.

Question: {question}
Chioces:
{choices}
Answer:"""

def normalize_eval_protocol(protocol: str) -> str:
    value = str(protocol).strip().lower()
    if value not in SUPPORTED_EVAL_PROTOCOLS:
        raise ValueError(f"Unknown eval protocol: {protocol}")
    return value


def is_bayesian_peft_protocol(protocol: str) -> bool:
    return normalize_eval_protocol(protocol) == "bayesian_peft"


def require_bayesian_peft_task(task: str) -> None:
    if task not in BAYESIAN_PEFT_TASKS:
        raise ValueError(
            f"Task '{task}' is not supported under eval_protocol=bayesian_peft. "
            f"Supported tasks: {sorted(BAYESIAN_PEFT_TASKS)}"
        )


def get_num_classes_for_protocol(task: str, protocol: str) -> int:
    task = task.lower().strip()
    if is_bayesian_peft_protocol(protocol):
        require_bayesian_peft_task(task)
        if task == "obqa":
            return 4
        if task in {"arc-c", "arc-e"}:
            return 5
    return get_task_num_classes(task)


def bayesian_peft_label_strings(task: str, add_space: bool) -> List[str]:
    task = task.lower().strip()
    require_bayesian_peft_task(task)
    spc = " " if add_space else ""
    if task == "obqa":
        return [f"{spc}A", f"{spc}B", f"{spc}C", f"{spc}D"]
    if task in {"arc-c", "arc-e"}:
        return [f"{spc}A", f"{spc}B", f"{spc}C", f"{spc}D", f"{spc}E"]
    raise ValueError(f"Unsupported bayesian-peft task: {task}")


def get_target_token_ids_for_protocol(
    tokenizer,
    task: str,
    protocol: str,
    device: torch.device,
    add_space: bool,
) -> Tensor:
    if is_bayesian_peft_protocol(protocol):
        labels = bayesian_peft_label_strings(task, add_space=add_space)
        enc = tokenizer(labels, return_tensors="pt", add_special_tokens=False).input_ids[:, -1]
        ids = enc.to(device=device, dtype=torch.long)
        print(f"[Target token ids][bayesian_peft] task={task} ids={dict(zip(labels, ids.tolist()))}")
        return ids

    num_classes = get_task_num_classes(task)
    return get_choice_token_ids(tokenizer, device, num_classes)


def _tokenize_prompts_no_padding(tokenizer, prompts: List[str], max_len: int) -> Dict[str, List[List[int]]]:
    return tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_len,
    )


def _finalize_protocol_dataset(ds: Dataset, keep_cols: Sequence[str]) -> Dataset:
    keep = set(keep_cols)
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


def _preprocess_bayesian_peft_obqa(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    keep_extra = [c for c in ["slice_id"] if c in ds.column_names]

    def _fn(batch: Dict) -> Dict:
        prompts, labels = [], []
        for qstem, choices, answer_key in zip(
            batch["question_stem"], batch["choices"], batch["answerKey"]
        ):
            try:
                choice_lines = "\n".join(
                    [f"{l}) {c}" for l, c in zip(choices["text"], choices["label"])]
                )
                prompts.append(_BP_PROMPT_OBQA.format(question=qstem, choices=choice_lines))
                labels.append(ord(str(answer_key).strip()) - ord("A"))
            except Exception:
                prompts.append("")
                labels.append(-1)
        enc = _tokenize_prompts_no_padding(tokenizer, prompts, max_len)
        enc["labels"] = labels
        enc["num_choices"] = [4] * len(labels)
        for key in keep_extra:
            enc[key] = batch[key]
        return enc

    ds2 = ds.map(_fn, batched=True).filter(lambda ex: ex["labels"] != -1)
    return _finalize_protocol_dataset(
        ds2,
        keep_cols=("input_ids", "attention_mask", "labels", "num_choices", *keep_extra),
    )


def _preprocess_bayesian_peft_arc(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    keep_extra = [c for c in ["slice_id"] if c in ds.column_names]

    def _fn(batch: Dict) -> Dict:
        prompts, labels, num_choices = [], [], []
        for question, choices, answer_key in zip(
            batch["question"], batch["choices"], batch["answerKey"]
        ):
            try:
                choice_lines = "\n".join(
                    [f"{l}) {c}" for l, c in zip(choices["text"], choices["label"])]
                )
                prompts.append(_BP_PROMPT_ARC.format(question=question, choices=choice_lines))
                answer_text = str(answer_key).strip()
                class_alpha = ord(answer_text) - ord("A")
                cls = class_alpha if class_alpha >= 0 else int(answer_text) - 1
                labels.append(cls)
                num_choices.append(len(choices["label"]))
            except Exception:
                prompts.append("")
                labels.append(-1)
                num_choices.append(-1)
        enc = _tokenize_prompts_no_padding(tokenizer, prompts, max_len)
        enc["labels"] = labels
        enc["num_choices"] = num_choices
        for key in keep_extra:
            enc[key] = batch[key]
        return enc

    ds2 = ds.map(_fn, batched=True).filter(
        lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= 5
    )
    return _finalize_protocol_dataset(
        ds2,
        keep_cols=("input_ids", "attention_mask", "labels", "num_choices", *keep_extra),
    )


def preprocess_task_for_protocol(
    task: str,
    ds: Dataset,
    tokenizer,
    max_len: int,
    protocol: str,
    bayesian_peft_add_space: bool,
    pad_to_max_length: bool = True,
) -> Dataset:
    task = task.lower().strip()
    if not is_bayesian_peft_protocol(protocol):
        return preprocess_task(
            task,
            ds,
            tokenizer,
            max_len,
            pad_to_max_length=pad_to_max_length,
        )

    require_bayesian_peft_task(task)
    if bayesian_peft_add_space:
        print("[Protocol] bayesian_peft_add_space only affects target label tokens; prompts are unchanged.")
    if task == "obqa":
        return _preprocess_bayesian_peft_obqa(ds, tokenizer, max_len)
    if task in {"arc-c", "arc-e"}:
        return _preprocess_bayesian_peft_arc(ds, tokenizer, max_len)
    raise ValueError(f"Unsupported bayesian-peft task: {task}")
