from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from transformers import AutoTokenizer

from .constants import DEFAULT_CHOICE_LABELS


def _choices_obj_to_mapping(choices_obj) -> Dict[str, str]:
    if isinstance(choices_obj, dict):
        labels = choices_obj.get("label", [])
        texts = choices_obj.get("text", [])
        return {str(lab): str(txt) for lab, txt in zip(labels, texts)}
    if isinstance(choices_obj, list):
        out = {}
        for choice in choices_obj:
            if isinstance(choice, dict) and "label" in choice and "text" in choice:
                out[str(choice["label"])] = str(choice["text"])
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


def get_single_token_id(tokenizer: AutoTokenizer, token_text: str) -> int:
    ids = tokenizer.encode(token_text, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids_with_space = tokenizer.encode(" " + token_text, add_special_tokens=False)
    if len(ids_with_space) == 1:
        return int(ids_with_space[0])
    raise ValueError(
        f'"{token_text}" is not a single token: '
        f"ids={ids}, ids_with_space={ids_with_space}"
    )


def answer_key_to_index(answer, label_order: Sequence[str]) -> int:
    labels = [str(label).strip() for label in label_order]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    if isinstance(answer, int):
        idx = int(answer)
        if 0 <= idx < len(labels):
            return idx
        raise ValueError(f"answer int not in 0..{len(labels) - 1}")

    answer_text = str(answer).strip()
    answer_upper = answer_text.upper()
    if answer_text in label_to_idx:
        return label_to_idx[answer_text]
    if answer_upper in label_to_idx:
        return label_to_idx[answer_upper]
    if answer_text.isdigit():
        idx = int(answer_text)
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


__all__ = [
    "_choices_obj_to_mapping",
    "_ordered_choice_labels",
    "answer_index_to_key",
    "answer_key_to_index",
    "get_choice_labels",
    "get_single_token_id",
    "make_prompt_from_4choices",
    "make_prompt_from_choices",
]
