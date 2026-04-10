from __future__ import annotations

from typing import Dict, List

from datasets import Dataset

from refactor.seq_lora.prompts import (
    answer_key_to_index,
    get_choice_labels,
    make_prompt_from_choices,
)

DEFAULT_MAX_CHOICES = 4


def coerce_choice_texts(choices_obj) -> List[str]:
    if hasattr(choices_obj, "tolist"):
        values = choices_obj.tolist()
    elif isinstance(choices_obj, (list, tuple)):
        values = list(choices_obj)
    else:
        values = []
    return [str(value).strip() for value in values if str(value).strip()]


def preprocess_scienceqa_closedchoice(
    ds: Dataset,
    tokenizer,
    max_len: int,
    *,
    keep_slice_id: bool = False,
    max_choices: int = DEFAULT_MAX_CHOICES,
) -> Dataset:
    keep_extra = [
        column
        for column in ["slice_id", "grade_num", "num_choices"]
        if keep_slice_id and column in ds.column_names
    ]
    if not keep_slice_id and "num_choices" in ds.column_names:
        keep_extra = ["num_choices"]

    def _fn(batch: Dict) -> Dict:
        prompts: List[str] = []
        labels: List[int] = []
        valid_num_choices: List[int] = []

        for idx in range(len(batch["question"])):
            try:
                choices = coerce_choice_texts(batch["choices"][idx])
                num_choices = len(choices)
                if num_choices < 2 or num_choices > max_choices:
                    raise ValueError(f"unsupported num_choices={num_choices}")
                label_order = get_choice_labels(num_choices)
                mapping = {
                    label_order[choice_idx]: choices[choice_idx]
                    for choice_idx in range(num_choices)
                }
                answer = answer_key_to_index(batch["answer"][idx], label_order)
                prompt = make_prompt_from_choices(
                    str(batch["question"][idx]),
                    mapping,
                    label_order=label_order,
                )
                prompts.append(prompt)
                labels.append(answer)
                valid_num_choices.append(num_choices)
            except Exception:
                prompts.append("")
                labels.append(-1)
                valid_num_choices.append(-1)

        enc = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = labels
        enc["num_choices"] = valid_num_choices
        for column in keep_extra:
            if column != "num_choices":
                enc[column] = batch[column]
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(
        lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= max_choices
    )
    keep_cols = {"input_ids", "attention_mask", "labels", "num_choices"} | set(keep_extra)
    return ds2.remove_columns([column for column in ds2.column_names if column not in keep_cols])


__all__ = [
    "DEFAULT_MAX_CHOICES",
    "coerce_choice_texts",
    "preprocess_scienceqa_closedchoice",
]
