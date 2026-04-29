from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors import safe_open
except Exception:
    safe_open = None

from common_eval_utils import (
    DynamicEvalCollator,
    PROMPT_BOOLQ,
    PROMPT_WG,
    SCIENCEQA_CURRIC_TASK_NAME,
    answer_index_to_key,
    answer_key_to_index,
    get_choice_labels,
    get_choice_token_ids,
    get_task_num_classes,
    load_task_dataset,
    make_prompt_from_choices,
)

_BAYESIAN_PEFT_TASKS = {"wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq"}
_TASK_ALIASES = {
    "wgs": "wgs",
    "wg_s": "wgs",
    "winogrande_s": "wgs",
    "winogrande-small": "wgs",
    "winogrande_s": "wgs",
    "wgm": "wgm",
    "wgms": "wgm",
    "wg_m": "wgm",
    "winogrande_m": "wgm",
    "winogrande-medium": "wgm",
    "arcc": "arc-c",
    "arc-c": "arc-c",
    "arc_challenge": "arc-c",
    "arc-challenge": "arc-c",
    "arcchallenge": "arc-c",
    "arce": "arc-e",
    "arc-e": "arc-e",
    "arc_easy": "arc-e",
    "arc-easy": "arc-e",
    "arceasy": "arc-e",
    "obqa": "obqa",
    "openbookqa": "obqa",
    "open_book_qa": "obqa",
    "boolq": "boolq",
    "booq": "boolq",
    "bool": "boolq",
    "sciq": "sciq",
    "scienceqa": SCIENCEQA_CURRIC_TASK_NAME,
    "scienceqa_grade2_11": SCIENCEQA_CURRIC_TASK_NAME,
    "scienceqa_closedchoice_grade2_11": SCIENCEQA_CURRIC_TASK_NAME,
}
_SUPPORTED_TASK_INPUTS = sorted(_TASK_ALIASES.keys())

DEFAULT_OBQA_BASE_MODEL_NAME = "Qwen/Qwen3-8B-Base"
DEFAULT_OBQA_RUN_TAG = "obqa_qv_lmhead_leftpad"
DEFAULT_OBQA_OUTPUT_DIR = "./iid_qwen35_8b_obqa_lora_map_leftpad"
DEFAULT_OBQA_SCORE_WITH = "base"
DEFAULT_OBQA_SEED = 0
DEFAULT_OBQA_NUM_SLICES = 10
DEFAULT_OBQA_MAX_SEQ_LEN = 300
DEFAULT_OBQA_BATCH_SIZE = 32
DEFAULT_OBQA_NUM_WORKERS = 0
DEFAULT_OBQA_TOKENIZER_PADDING_SIDE = "left"
DEFAULT_OBQA_TRUST_REMOTE_CODE = False
DEFAULT_OBQA_LOCAL_FILES_ONLY = True
DEFAULT_OBQA_ATTN_IMPLEMENTATION = "sdpa"

DEFAULT_MODEL_FAMILY = "qwen_obqa"
DEFAULT_LLAMA2_BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DEFAULT_LLAMA2_CHECKPOINT_ROOT = "./bayesian-peft/checkpoints/mle/meta-llama/Llama-2-7b-hf"
_LLAMA2_BAYESIAN_PEFT_TASK = {
    "wgs": "winogrande_s",
    "wgm": "winogrande_m",
    "arc-c": "ARC-Challenge",
    "arc-e": "ARC-Easy",
    "obqa": "obqa",
    "boolq": "boolq",
}

_BP_PROMPT_WG = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

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

_BP_PROMPT_BOOLQ = """Read the passage below and answer the question with the words 'true' or 'false'.

Passage: {passage}
Question: {question}
Answer (true or false):"""


@dataclass(frozen=True)
class ScoringExample:
    raw_idx: int
    prompt: str
    label: int
    num_choices: int


@dataclass(frozen=True)
class AdapterHeadInfo:
    mode: str
    rows: Optional[int]
    source_key: str


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _parse_num_slices_values(value: str, fallback: int) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return [int(fallback)]

    out: List[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        sep = "-" if "-" in item else (":" if ":" in item else "")
        if sep:
            left, right = [x.strip() for x in item.split(sep, 1)]
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(item))

    deduped: List[int] = []
    seen = set()
    for n in out:
        if n <= 0:
            raise ValueError(f"num_slices values must be positive, got {n}")
        if n not in seen:
            deduped.append(int(n))
            seen.add(int(n))
    if not deduped:
        raise ValueError("--num_slices_list did not contain any valid values.")
    return deduped


def _normalize_task_name(task: str) -> str:
    key = str(task).strip().lower().replace(" ", "_")
    if key not in _TASK_ALIASES:
        raise ValueError(
            f"Unsupported dataset_name={task!r}. Supported inputs: {_SUPPORTED_TASK_INPUTS}"
        )
    return _TASK_ALIASES[key]


def _safe_str(value) -> str:
    return "" if value is None else str(value).strip()


def _choices_obj_to_mapping(choices_obj) -> Dict[str, str]:
    if isinstance(choices_obj, dict):
        labels = choices_obj.get("label", None)
        texts = choices_obj.get("text", None)
        if isinstance(labels, list) and isinstance(texts, list):
            return {str(label): str(text) for label, text in zip(labels, texts)}
        return {
            str(key): str(value)
            for key, value in choices_obj.items()
            if str(key).strip().upper() in get_choice_labels(5)
        }
    if isinstance(choices_obj, list):
        out: Dict[str, str] = {}
        for choice in choices_obj:
            if isinstance(choice, dict):
                if "label" in choice and "text" in choice:
                    out[str(choice["label"])] = str(choice["text"])
                elif "key" in choice and "value" in choice:
                    out[str(choice["key"])] = str(choice["value"])
        return out
    return {}


def _default_obqa_map_dir(seed: int) -> str:
    return os.path.join(
        DEFAULT_OBQA_OUTPUT_DIR,
        DEFAULT_OBQA_RUN_TAG,
        f"seed_{int(seed)}",
        "map_step_2000",
    )


def _slice_strategy_tag(strategy: str) -> str:
    value = str(strategy).strip().lower()
    if value == "stratified_mix":
        return "stratmix"
    return "easyhard"


def _default_obqa_out_dir(seed: int, *, uses_map: bool, strategy: str = "quantile") -> str:
    score_tag = "map_loss" if bool(uses_map) else "base_loss"
    return os.path.join(
        "./slice_data",
        f"{DEFAULT_OBQA_RUN_TAG}_{score_tag}_{_slice_strategy_tag(strategy)}_seed{int(seed)}",
    )


def _default_llama2_map_dir(task: str, seed: int) -> str:
    task = _normalize_task_name(task)
    if task not in _LLAMA2_BAYESIAN_PEFT_TASK:
        raise ValueError(f"Llama2 Bayesian-PEFT MAP defaults are not defined for task={task!r}.")
    source_task = _LLAMA2_BAYESIAN_PEFT_TASK[task]
    return os.path.join(
        DEFAULT_LLAMA2_CHECKPOINT_ROOT,
        source_task,
        f"lora-{source_task}-lr5e-5-bs4-drop0.1-step2000-seed{int(seed)}",
    )


def _default_llama2_out_dir(task: str, seed: int, *, uses_map: bool, strategy: str) -> str:
    task = _normalize_task_name(task)
    score_tag = "map_loss" if bool(uses_map) else "base_loss"
    task_tag = task.replace("-", "_")
    return os.path.join(
        "./slice_data",
        f"{task_tag}_llama2_{score_tag}_{_slice_strategy_tag(strategy)}_seed{int(seed)}",
    )


def _ordered_choice_labels(mapping: Dict[str, str], max_choices: int = 4) -> List[str]:
    labels = [str(label).strip() for label in mapping.keys() if str(label).strip()]
    alpha_rank = {label: idx for idx, label in enumerate(get_choice_labels(max_choices))}
    if labels and all(label.upper() in alpha_rank for label in labels):
        return sorted(labels, key=lambda label: alpha_rank[label.upper()])
    return labels


def _get_q_and_choices(ex: Dict) -> Tuple[str, object]:
    question = ex.get("question", ex.get("question_stem", ""))
    choices = ex.get("choices", None)
    if isinstance(question, dict):
        q_text = _safe_str(question.get("stem", question.get("text", "")))
        nested_choices = question.get("choices", choices)
        return q_text, nested_choices
    return _safe_str(question), choices


def _labels_texts_from_choices(choices_obj) -> Tuple[List[str], List[str]]:
    if isinstance(choices_obj, dict):
        labels = choices_obj.get("label", None)
        texts = choices_obj.get("text", None)
        if isinstance(labels, list) and isinstance(texts, list):
            return [str(x).strip() for x in labels], [str(x) for x in texts]
    mapping = _choices_obj_to_mapping(choices_obj)
    labels = _ordered_choice_labels(mapping, max_choices=5)
    return labels, [mapping[label] for label in labels if label in mapping]


def _bayesian_peft_answer_to_index(answer) -> int:
    answer_text = str(answer).strip()
    class_alpha = ord(answer_text.upper()) - ord("A") if answer_text else -1
    if class_alpha >= 0:
        return int(class_alpha)
    return int(answer_text) - 1


def _format_bayesian_peft_choice_lines(labels: Sequence[str], texts: Sequence[str]) -> str:
    # Match bayesian-peft/dataset/utils/dsets.py exactly: it zips text first,
    # then label, so the prompt lines look like "<choice text>) <label>".
    return "\n".join(f"{text}) {label}" for text, label in zip(texts, labels))


def _build_bayesian_peft_arc_or_obqa_example(ex: Dict, raw_idx: int, task: str) -> Optional[ScoringExample]:
    if task == "obqa":
        question = _safe_str(ex.get("question_stem", ""))
        choices = ex.get("choices", None)
        prompt_tmpl = _BP_PROMPT_OBQA
        max_choices = 4
    else:
        question, choices = _get_q_and_choices(ex)
        prompt_tmpl = _BP_PROMPT_ARC
        max_choices = 5

    labels, texts = _labels_texts_from_choices(choices)
    if not question or len(labels) < 2 or len(labels) > max_choices or len(labels) != len(texts):
        return None

    try:
        label = _bayesian_peft_answer_to_index(ex.get("answerKey"))
    except Exception:
        return None
    if label < 0 or label >= max_choices or label >= len(labels):
        return None

    return ScoringExample(
        raw_idx=raw_idx,
        prompt=prompt_tmpl.format(
            question=question,
            choices=_format_bayesian_peft_choice_lines(labels, texts),
        ),
        label=int(label),
        num_choices=len(labels),
    )


def _build_arc_or_obqa_example(
    ex: Dict,
    raw_idx: int,
    task: str,
    *,
    max_choices: int = 4,
) -> Optional[ScoringExample]:
    if task == "obqa":
        question = _safe_str(ex.get("question_stem", ""))
        choices = ex.get("choices", None)
        expected_num_choices = 4
    else:
        question, choices = _get_q_and_choices(ex)
        expected_num_choices = None

    mapping = _choices_obj_to_mapping(choices)
    label_order = get_choice_labels(4) if expected_num_choices == 4 else _ordered_choice_labels(mapping)
    if len(label_order) < 2 or len(label_order) > int(max_choices):
        return None
    if expected_num_choices is not None and len(label_order) != expected_num_choices:
        return None
    if not question or not all(label in mapping for label in label_order):
        return None

    try:
        label = answer_key_to_index(ex.get("answerKey"), label_order)
    except Exception:
        return None

    return ScoringExample(
        raw_idx=raw_idx,
        prompt=make_prompt_from_choices(question, mapping, label_order=label_order),
        label=int(label),
        num_choices=len(label_order),
    )


def _build_bayesian_peft_wg_example(ex: Dict, raw_idx: int) -> Optional[ScoringExample]:
    sentence = _safe_str(ex.get("sentence", ""))
    option1 = _safe_str(ex.get("option1", ""))
    option2 = _safe_str(ex.get("option2", ""))
    answer = str(ex.get("answer", "")).strip()
    if not sentence or not option1 or not option2 or answer not in {"1", "2"}:
        return None
    return ScoringExample(
        raw_idx=raw_idx,
        prompt=_BP_PROMPT_WG.format(
            question=sentence,
            choices=f"A) {option1}\nB) {option2}",
        ),
        label=0 if answer == "1" else 1,
        num_choices=2,
    )


def _build_wg_example(ex: Dict, raw_idx: int) -> Optional[ScoringExample]:
    sentence = _safe_str(ex.get("sentence", ""))
    option1 = _safe_str(ex.get("option1", ""))
    option2 = _safe_str(ex.get("option2", ""))
    answer = str(ex.get("answer", "")).strip()
    if not sentence or not option1 or not option2 or answer not in {"1", "2"}:
        return None
    return ScoringExample(
        raw_idx=raw_idx,
        prompt=PROMPT_WG.format(question=sentence, option1=option1, option2=option2),
        label=0 if answer == "1" else 1,
        num_choices=2,
    )


def _build_bayesian_peft_boolq_example(ex: Dict, raw_idx: int) -> Optional[ScoringExample]:
    question = _safe_str(ex.get("question", ""))
    passage = _safe_str(ex.get("passage", ""))
    if not question or not passage:
        return None
    answer = ex.get("answer", ex.get("label", 0))
    return ScoringExample(
        raw_idx=raw_idx,
        prompt=_BP_PROMPT_BOOLQ.format(question=question, passage=passage[:1024]),
        label=int(answer),
        num_choices=2,
    )


def _build_boolq_example(ex: Dict, raw_idx: int) -> Optional[ScoringExample]:
    question = _safe_str(ex.get("question", ""))
    passage = _safe_str(ex.get("passage", ""))
    if not question or not passage:
        return None
    return ScoringExample(
        raw_idx=raw_idx,
        prompt=PROMPT_BOOLQ.format(question=question, passage=passage),
        label=int(ex.get("label", 0)),
        num_choices=2,
    )


def _build_sciq_example(
    ex: Dict,
    raw_idx: int,
    *,
    shuffle_choices: bool,
    seed: int,
) -> Optional[ScoringExample]:
    question = _safe_str(ex.get("question", ""))
    correct = _safe_str(ex.get("correct_answer", ""))
    distractors = [
        _safe_str(ex.get("distractor1", "")),
        _safe_str(ex.get("distractor2", "")),
        _safe_str(ex.get("distractor3", "")),
    ]
    choices = [*distractors, correct]
    if not question or any(not choice for choice in choices):
        return None
    if shuffle_choices:
        rng = random.Random(int(seed) + int(raw_idx))
        rng.shuffle(choices)
    label_order = get_choice_labels(4)
    mapping = {label_order[i]: str(choices[i]) for i in range(4)}
    answer_key = answer_index_to_key(choices.index(correct), label_order)
    return ScoringExample(
        raw_idx=raw_idx,
        prompt=make_prompt_from_choices(question, mapping, label_order=label_order),
        label=answer_key_to_index(answer_key, label_order),
        num_choices=4,
    )


def _scienceqa_choice_texts(choices_obj) -> List[str]:
    if hasattr(choices_obj, "tolist"):
        values = choices_obj.tolist()
    elif isinstance(choices_obj, (list, tuple)):
        values = list(choices_obj)
    else:
        values = []
    return [str(value).strip() for value in values if str(value).strip()]


def _build_scienceqa_example(ex: Dict, raw_idx: int) -> Optional[ScoringExample]:
    question = _safe_str(ex.get("question", ""))
    choices = _scienceqa_choice_texts(ex.get("choices", None))
    if not question or len(choices) < 2 or len(choices) > 4:
        return None

    label_order = get_choice_labels(len(choices))
    mapping = {label_order[i]: str(choices[i]) for i in range(len(choices))}
    try:
        label = answer_key_to_index(ex.get("answer"), label_order)
    except Exception:
        return None

    return ScoringExample(
        raw_idx=raw_idx,
        prompt=make_prompt_from_choices(question, mapping, label_order=label_order),
        label=int(label),
        num_choices=len(label_order),
    )


def build_scoring_examples(
    ds: Dataset,
    task: str,
    *,
    scoring_protocol: str,
    sciq_shuffle_choices: bool,
    seed: int,
) -> List[ScoringExample]:
    examples: List[ScoringExample] = []
    dropped = 0
    for raw_idx in range(len(ds)):
        ex = ds[int(raw_idx)]
        if scoring_protocol == "bayesian_peft":
            if task in {"arc-c", "arc-e", "obqa"}:
                item = _build_bayesian_peft_arc_or_obqa_example(ex, int(raw_idx), task)
            elif task in {"wgs", "wgm"}:
                item = _build_bayesian_peft_wg_example(ex, int(raw_idx))
            elif task == "boolq":
                item = _build_bayesian_peft_boolq_example(ex, int(raw_idx))
            else:
                raise ValueError(f"Task {task!r} does not support bayesian_peft scoring.")
        elif task in {"arc-c", "arc-e", "obqa"}:
            item = _build_arc_or_obqa_example(ex, int(raw_idx), task, max_choices=4)
        elif task in {"wgs", "wgm"}:
            item = _build_wg_example(ex, int(raw_idx))
        elif task == "boolq":
            item = _build_boolq_example(ex, int(raw_idx))
        elif task == "sciq":
            item = _build_sciq_example(
                ex,
                int(raw_idx),
                shuffle_choices=bool(sciq_shuffle_choices),
                seed=int(seed),
            )
        elif task == SCIENCEQA_CURRIC_TASK_NAME:
            item = _build_scienceqa_example(ex, int(raw_idx))
        else:
            raise ValueError(f"Unsupported task for loss slicing: {task}")

        if item is None:
            dropped += 1
        else:
            examples.append(item)

    print(f"[Prep] valid={len(examples)} dropped={dropped}")
    if not examples:
        raise RuntimeError("No valid examples after prompt/label formatting.")
    return examples


def _resolve_scoring_protocol(args) -> str:
    requested = str(args.scoring_protocol).strip().lower()
    if requested not in {"auto", "default", "bayesian_peft"}:
        raise ValueError(f"Unknown scoring protocol: {args.scoring_protocol}")
    task = str(args.dataset_name).strip().lower()
    if requested != "auto":
        if requested == "bayesian_peft" and task not in _BAYESIAN_PEFT_TASKS:
            raise ValueError(f"Task {task!r} does not support bayesian_peft scoring.")
        return requested

    map_dir = str(args.map_dir or "")
    base_model = str(args.base_model or "")
    if task in _BAYESIAN_PEFT_TASKS and (
        "bayesian-peft" in map_dir
        or "meta-llama/Llama-2-7b-hf" in base_model
    ):
        return "bayesian_peft"
    return "default"


def _num_classes_for_scoring(task: str, scoring_protocol: str) -> int:
    task = str(task).strip().lower()
    if scoring_protocol == "bayesian_peft":
        if task in {"wgs", "wgm", "boolq"}:
            return 2
        if task == "obqa":
            return 4
        if task in {"arc-c", "arc-e"}:
            return 5
        raise ValueError(f"Task {task!r} does not support bayesian_peft scoring.")
    return int(get_task_num_classes(task))


def _bayesian_peft_label_strings(task: str, add_space: bool) -> List[str]:
    task = str(task).strip().lower()
    spc = " " if bool(add_space) else ""
    if task in {"wgs", "wgm"}:
        return [f"{spc}A", f"{spc}B"]
    if task == "obqa":
        return [f"{spc}A", f"{spc}B", f"{spc}C", f"{spc}D"]
    if task in {"arc-c", "arc-e"}:
        return [f"{spc}A", f"{spc}B", f"{spc}C", f"{spc}D", f"{spc}E"]
    if task == "boolq":
        return [f"{spc}True", f"{spc}False"]
    raise ValueError(f"Task {task!r} does not support bayesian_peft scoring.")


def _choice_token_ids_for_scoring(tokenizer, device: torch.device, task: str, scoring_protocol: str, add_space: bool) -> torch.Tensor:
    if scoring_protocol == "bayesian_peft":
        labels = _bayesian_peft_label_strings(task, add_space=bool(add_space))
        ids = tokenizer(labels, return_tensors="pt", add_special_tokens=False).input_ids[:, -1]
        ids = ids.to(device=device, dtype=torch.long)
        print(f"[Target token ids][bayesian_peft] task={task} ids={dict(zip(labels, ids.tolist()))}")
        return ids

    num_classes = _num_classes_for_scoring(task, scoring_protocol)
    return get_choice_token_ids(tokenizer, device, num_classes)


def build_scoring_dataset(
    examples: Sequence[ScoringExample],
    tokenizer,
    *,
    max_seq_len: int,
) -> Dataset:
    enc = tokenizer(
        [ex.prompt for ex in examples],
        padding=False,
        truncation=True,
        max_length=int(max_seq_len),
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [int(ex.label) for ex in examples],
            "num_choices": [int(ex.num_choices) for ex in examples],
            "raw_idx": [int(ex.raw_idx) for ex in examples],
        }
    )


def _mask_invalid_choices(logits: torch.Tensor, num_choices: Optional[Sequence[int]]) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


@torch.inference_mode()
def compute_choice_losses(
    model,
    loader: DataLoader,
    *,
    choice_token_ids: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    apply_choice_mask: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    losses: List[torch.Tensor] = []
    raw_indices: List[int] = []

    for batch in tqdm(loader, desc="LOSS", file=None):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(device.type == "cuda"),
        ):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[:, -1, :].index_select(-1, choice_token_ids)

        logits = logits.float()
        if apply_choice_mask:
            logits = _mask_invalid_choices(logits, num_choices)
        loss = F.cross_entropy(logits, labels, reduction="none")
        losses.append(loss.detach().cpu())
        raw_indices.extend([int(idx) for idx in batch["raw_idx"]])

    if not losses:
        raise RuntimeError("Loss loader produced no batches.")
    return torch.cat(losses).numpy(), np.asarray(raw_indices, dtype=np.int64)


def assign_loss_quantile_slice_ids(losses: np.ndarray, num_slices: int) -> np.ndarray:
    n = int(losses.shape[0])
    if n < int(num_slices):
        raise ValueError(f"Need at least num_slices examples, got n={n}, num_slices={num_slices}")
    order = np.argsort(losses, kind="stable")
    slice_ids = np.empty(n, dtype=np.int32)
    for rank, pos in enumerate(order.tolist()):
        slice_ids[int(pos)] = min(int(rank * int(num_slices) // n), int(num_slices) - 1)
    return slice_ids


def _ipf_count_matrix(
    weights: np.ndarray,
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    *,
    max_iter: int = 200,
) -> np.ndarray:
    target = np.asarray(weights, dtype=np.float64).clip(min=1e-12)
    row_sums = np.asarray(row_sums, dtype=np.float64)
    col_sums = np.asarray(col_sums, dtype=np.float64)
    if int(row_sums.sum()) != int(col_sums.sum()):
        raise ValueError("row_sums and col_sums must have the same total.")

    for _ in range(int(max_iter)):
        target *= row_sums[:, None] / np.maximum(target.sum(axis=1, keepdims=True), 1e-12)
        target *= col_sums[None, :] / np.maximum(target.sum(axis=0, keepdims=True), 1e-12)

    counts = np.floor(target).astype(np.int64)
    row_remaining = row_sums.astype(np.int64) - counts.sum(axis=1)
    col_remaining = col_sums.astype(np.int64) - counts.sum(axis=0)
    frac = target - counts

    remaining = int(row_remaining.sum())
    while remaining > 0:
        mask = (row_remaining[:, None] > 0) & (col_remaining[None, :] > 0)
        if not bool(mask.any()):
            raise RuntimeError("Could not round stratified mix count matrix.")
        scores = np.where(mask, frac, -np.inf)
        flat_idx = int(np.argmax(scores))
        row, col = np.unravel_index(flat_idx, scores.shape)
        counts[row, col] += 1
        row_remaining[row] -= 1
        col_remaining[col] -= 1
        frac[row, col] = -np.inf
        remaining -= 1

    return counts


def assign_stratified_mix_slice_ids(
    losses: np.ndarray,
    num_slices: int,
    *,
    num_bands: int,
    mix_strength: float,
    seed: int,
) -> np.ndarray:
    n = int(losses.shape[0])
    t = int(num_slices)
    if n < t:
        raise ValueError(f"Need at least num_slices examples, got n={n}, num_slices={num_slices}")
    b = max(1, min(int(num_bands), n))

    order = np.argsort(losses, kind="stable")
    bands = [np.asarray(x, dtype=np.int64) for x in np.array_split(order, b)]
    col_sums = np.asarray([len(x) for x in bands], dtype=np.int64)
    row_sums = np.asarray([len(x) for x in np.array_split(np.arange(n), t)], dtype=np.int64)

    if t == 1 or b == 1:
        weights = np.ones((t, b), dtype=np.float64)
    else:
        slice_axis = np.linspace(-1.0, 1.0, t, dtype=np.float64)
        band_axis = np.linspace(-1.0, 1.0, b, dtype=np.float64)
        weights = np.exp(float(mix_strength) * np.outer(slice_axis, band_axis))

    counts = _ipf_count_matrix(weights, row_sums=row_sums, col_sums=col_sums)
    rng = np.random.default_rng(int(seed))
    slice_ids = np.empty(n, dtype=np.int32)

    for band_id, band_indices in enumerate(bands):
        shuffled = band_indices.copy()
        rng.shuffle(shuffled)
        cursor = 0
        for slice_id in range(t):
            take_n = int(counts[slice_id, band_id])
            if take_n <= 0:
                continue
            take = shuffled[cursor : cursor + take_n]
            slice_ids[take] = int(slice_id)
            cursor += take_n
        if cursor != len(shuffled):
            raise RuntimeError("Internal mismatch while assigning stratified mix slices.")

    # Relabel by actual mean loss so the exported slice ids remain easy-to-hard on average.
    means = np.asarray([float(np.mean(losses[slice_ids == sid])) for sid in range(t)])
    order_slices = np.argsort(means, kind="stable")
    remap = np.empty(t, dtype=np.int32)
    for new_sid, old_sid in enumerate(order_slices.tolist()):
        remap[int(old_sid)] = int(new_sid)
    return remap[slice_ids]


def assign_slice_ids(
    losses: np.ndarray,
    num_slices: int,
    *,
    strategy: str,
    mix_num_bands: int,
    mix_strength: float,
    seed: int,
) -> np.ndarray:
    strategy = str(strategy).strip().lower()
    if strategy == "quantile":
        return assign_loss_quantile_slice_ids(losses, int(num_slices))
    if strategy == "stratified_mix":
        return assign_stratified_mix_slice_ids(
            losses,
            int(num_slices),
            num_bands=int(mix_num_bands),
            mix_strength=float(mix_strength),
            seed=int(seed),
        )
    raise ValueError(f"Unknown slice strategy: {strategy}")


def _summarize_slices(losses: np.ndarray, slice_ids: np.ndarray, num_slices: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for sid in range(int(num_slices)):
        vals = losses[slice_ids == sid]
        if vals.size == 0:
            rows.append({"slice_id": sid, "n": 0})
            continue
        rows.append(
            {
                "slice_id": sid,
                "n": int(vals.size),
                "loss_min": float(np.min(vals)),
                "loss_mean": float(np.mean(vals)),
                "loss_max": float(np.max(vals)),
            }
        )
    return rows


def _load_split(task: str, split: str) -> Dataset:
    train, validation, test = load_task_dataset(task)
    split_map = {
        "train": train,
        "validation": validation,
        "val": validation,
        "test": test,
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split={split!r}; expected train/validation/test.")
    return split_map[split]


def _iter_adapter_tensor_shapes(adapter_dir: str):
    st_path = os.path.join(str(adapter_dir), "adapter_model.safetensors")
    if os.path.exists(st_path):
        if safe_open is None:
            raise RuntimeError(
                "adapter_model.safetensors is present, but safetensors could not be imported."
            )
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, tuple(int(x) for x in f.get_tensor(key).shape)
        return

    bin_path = os.path.join(str(adapter_dir), "adapter_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                yield key, tuple(int(x) for x in value.shape)
        return

    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors or adapter_model.bin under {adapter_dir}"
    )


def _inspect_adapter_head(adapter_dir: str, num_classes: int) -> AdapterHeadInfo:
    lm_head_rows: List[Tuple[str, int]] = []
    has_lm_head_tensor = False
    for key, shape in _iter_adapter_tensor_shapes(adapter_dir):
        if "lm_head" not in key:
            continue
        has_lm_head_tensor = True
        if len(shape) < 2:
            continue
        if key.endswith("lm_head.base_layer.weight") or ".lm_head.base_layer.weight" in key:
            lm_head_rows.insert(0, (key, int(shape[0])))
        elif ".lm_head.lora_B." in key and key.endswith(".weight"):
            lm_head_rows.append((key, int(shape[0])))

    if not has_lm_head_tensor:
        return AdapterHeadInfo(mode="no_lm_head_adapter", rows=None, source_key="")
    if not lm_head_rows:
        return AdapterHeadInfo(mode="unknown_lm_head_adapter", rows=None, source_key="")

    source_key, rows = lm_head_rows[0]
    if rows == int(num_classes):
        return AdapterHeadInfo(mode="trimmed_head", rows=rows, source_key=source_key)
    if rows > int(num_classes):
        return AdapterHeadInfo(mode="full_vocab_head", rows=rows, source_key=source_key)
    raise ValueError(
        f"Adapter lm_head rows={rows} from {source_key!r}, but task has "
        f"{int(num_classes)} classes. Check --dataset_name/--scoring_protocol."
    )


def _trim_lm_head_to_choice_tokens(model: torch.nn.Module, choice_token_ids: torch.Tensor) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "lm_head"):
        raise RuntimeError("Could not locate lm_head on base model for trimming.")

    lm_head = base.lm_head
    weight = lm_head.weight.index_select(0, choice_token_ids).detach()
    bias = (
        lm_head.bias.index_select(0, choice_token_ids).detach()
        if getattr(lm_head, "bias", None) is not None
        else None
    )
    new_head = nn.Linear(
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        bias=(bias is not None),
        device=weight.device,
        dtype=weight.dtype,
    )
    new_head.weight.data.copy_(weight)
    if bias is not None:
        new_head.bias.data.copy_(bias)

    base.lm_head = new_head
    if hasattr(base, "set_output_embeddings"):
        base.set_output_embeddings(new_head)
    if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
        base.config.vocab_size = int(choice_token_ids.numel())


def _load_model_and_tokenizer(args, device: torch.device, scoring_protocol: str):
    if args.map_dir:
        peft_cfg = PeftConfig.from_pretrained(args.map_dir)
        base_name = str(args.base_model or peft_cfg.base_model_name_or_path)
        model_desc = f"map:{args.map_dir}"
    else:
        if not args.base_model:
            raise ValueError("--base_model is required when --map_dir is not provided.")
        base_name = str(args.base_model)
        model_desc = f"base:{base_name}"

    tokenizer_name = str(args.tokenizer_name or base_name)
    if not args.tokenizer_name and args.map_dir:
        adapter_tokenizer_config = os.path.join(str(args.map_dir), "tokenizer_config.json")
        if os.path.exists(adapter_tokenizer_config):
            tokenizer_name = str(args.map_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    tokenizer.padding_side = str(args.tokenizer_padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation=str(args.attn_implementation),
        local_files_only=bool(args.local_files_only),
    ).to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()

    logits_are_choice_space = False
    if args.map_dir:
        num_classes = _num_classes_for_scoring(str(args.dataset_name), str(scoring_protocol))
        adapter_head = _inspect_adapter_head(str(args.map_dir), num_classes)
        print(
            "[Adapter head] "
            f"mode={adapter_head.mode} rows={adapter_head.rows} "
            f"source={adapter_head.source_key or 'n/a'}"
        )
        if adapter_head.mode == "trimmed_head":
            choice_token_ids = _choice_token_ids_for_scoring(
                tokenizer,
                device,
                str(args.dataset_name),
                str(scoring_protocol),
                bool(args.bayesian_peft_add_space),
            )
            _trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
            logits_are_choice_space = True
            print(f"[Head] trimmed base lm_head to {num_classes} choice logits before adapter load")

        model = PeftModel.from_pretrained(
            base_model,
            args.map_dir,
            local_files_only=bool(args.local_files_only),
        ).to(device)
    else:
        model = base_model

    model.eval()
    return model, tokenizer, amp_dtype, model_desc, logits_are_choice_space


def _save_outputs(
    *,
    ds_raw: Dataset,
    out_dir: str,
    valid_raw_indices: np.ndarray,
    losses: np.ndarray,
    slice_ids: np.ndarray,
    rng: np.random.Generator,
    num_slices: int,
    kfac_per_slice: int,
    save_full_train: bool,
    save_kfac_balanced: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rank = np.empty_like(slice_ids)
    rank[np.argsort(losses, kind="stable")] = np.arange(len(losses), dtype=np.int32)
    ds_scored = ds_raw.select(valid_raw_indices.tolist())
    if "slice_id" in ds_scored.column_names:
        if "source_slice_id" in ds_scored.column_names:
            ds_scored = ds_scored.remove_columns(["slice_id"])
        else:
            ds_scored = ds_scored.rename_column("slice_id", "source_slice_id")
    for col in ["loss_rank", "loss_value"]:
        if col in ds_scored.column_names:
            ds_scored = ds_scored.remove_columns([col])

    ds_scored = (
        ds_scored
        .add_column("slice_id", slice_ids.astype(np.int32).tolist())
        .add_column("loss_rank", rank.astype(np.int32).tolist())
        .add_column("loss_value", losses.astype(np.float32).tolist())
    )

    if save_full_train:
        order = np.lexsort((rank, slice_ids)).tolist()
        out_full = os.path.join(out_dir, "full_train")
        DatasetDict({"train": ds_scored.select(order)}).save_to_disk(out_full)
        print(f"[Save] full_train -> {out_full}")

    if save_kfac_balanced:
        per = int(kfac_per_slice)
        if per <= 0:
            kept = np.lexsort((rank, slice_ids)).astype(int).tolist()
        else:
            kept: List[int] = []
            for sid in range(int(num_slices)):
                idxs = np.where(slice_ids == sid)[0]
                if idxs.size < per:
                    raise RuntimeError(
                        f"Slice {sid} has only {idxs.size} valid examples, "
                        f"but kfac_per_slice={per}. Lower --kfac_per_slice or --num_slices."
                    )
                kept.extend(rng.choice(idxs, size=per, replace=False).astype(int).tolist())
            kept = sorted(kept, key=lambda idx: (int(slice_ids[idx]), int(rank[idx])))
        out_kfac = os.path.join(out_dir, "kfac_balanced")
        DatasetDict({"train": ds_scored.select(kept)}).save_to_disk(out_kfac)
        print(f"[Save] kfac_balanced -> {out_kfac}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MCQA Seq-LoRA slices from MAP/base loss quantiles."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="obqa",
        choices=_SUPPORTED_TASK_INPUTS,
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "val", "test"])
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. For default OBQA, this is derived from score_with and seed.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_OBQA_SEED)
    parser.add_argument("--num_slices", type=int, default=DEFAULT_OBQA_NUM_SLICES)
    parser.add_argument(
        "--num_slices_list",
        type=str,
        default="",
        help=(
            "Optional comma/range list of slice counts, e.g. '1-20' or '3,5,10'. "
            "Losses are computed once, then every T is saved under out_dir/T<T>."
        ),
    )
    parser.add_argument(
        "--kfac_per_slice",
        type=int,
        default=0,
        help="Examples sampled per slice for kfac_balanced; <=0 saves every scored training example.",
    )
    parser.add_argument(
        "--slice_strategy",
        type=str,
        default="quantile",
        choices=["quantile", "stratified_mix"],
        help=(
            "quantile makes pure easy-to-hard loss buckets. stratified_mix mixes "
            "difficulty bands into every slice while increasing average slice difficulty."
        ),
    )
    parser.add_argument(
        "--mix_num_bands",
        type=int,
        default=10,
        help="Number of loss-ranked difficulty bands used by --slice_strategy stratified_mix.",
    )
    parser.add_argument(
        "--mix_strength",
        type=float,
        default=2.0,
        help="Difficulty tilt for stratified_mix. 0 is uniform mixing; larger is closer to pure curriculum.",
    )

    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument(
        "--model_family",
        type=str,
        default=DEFAULT_MODEL_FAMILY,
        choices=["qwen_obqa", "llama2", "custom"],
        help=(
            "Preset model/checkpoint family. llama2 uses meta-llama/Llama-2-7b-hf "
            "and Bayesian-PEFT prompts/classes for WinoGrande, ARC, OBQA, and BoolQ."
        ),
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        default="",
        help="LoRA adapter used for scoring. Use with --score_with map.",
    )
    parser.add_argument(
        "--score_with",
        type=str,
        default=DEFAULT_OBQA_SCORE_WITH,
        choices=["base", "map"],
        help="Choose base-model loss or a MAP/MLE LoRA checkpoint loss.",
    )
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--trust_remote_code", type=_parse_bool, default=DEFAULT_OBQA_TRUST_REMOTE_CODE)
    parser.add_argument("--local_files_only", type=_parse_bool, default=DEFAULT_OBQA_LOCAL_FILES_ONLY)
    parser.add_argument(
        "--tokenizer_padding_side",
        type=str,
        default=DEFAULT_OBQA_TOKENIZER_PADDING_SIDE,
        choices=["left", "right"],
    )
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_OBQA_MAX_SEQ_LEN)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_OBQA_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_OBQA_NUM_WORKERS)
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_OBQA_ATTN_IMPLEMENTATION)
    parser.add_argument(
        "--scoring_protocol",
        type=str,
        default="auto",
        choices=["auto", "default", "bayesian_peft"],
        help=(
            "Prompt/target protocol used for loss scoring. auto uses bayesian_peft "
            "when the MAP path looks like a bayesian-peft checkpoint."
        ),
    )
    parser.add_argument(
        "--bayesian_peft_add_space",
        type=_parse_bool,
        default=False,
        help="Prepend a space to bayesian-peft target label strings when scoring.",
    )
    parser.add_argument("--sciq_shuffle_choices", type=_parse_bool, default=False)
    parser.add_argument("--save_full_train", type=_parse_bool, default=True)
    parser.add_argument("--save_kfac_balanced", type=_parse_bool, default=True)

    args = parser.parse_args()
    args.dataset_name = _normalize_task_name(args.dataset_name)
    num_slices_values = _parse_num_slices_values(args.num_slices_list, int(args.num_slices))
    model_family = str(args.model_family).strip().lower()
    score_with = str(args.score_with).strip().lower()

    if score_with == "base" and args.map_dir:
        raise ValueError(
            "--score_with base cannot be combined with --map_dir. "
            "Remove --map_dir for base-loss slices, or use --score_with map."
        )

    if model_family == "llama2":
        if args.dataset_name not in _LLAMA2_BAYESIAN_PEFT_TASK:
            raise ValueError(
                f"--model_family llama2 supports {sorted(_LLAMA2_BAYESIAN_PEFT_TASK)}; "
                f"got dataset_name={args.dataset_name!r}."
            )
        if not args.base_model:
            args.base_model = DEFAULT_LLAMA2_BASE_MODEL_NAME
        if not args.map_dir and score_with == "map":
            args.map_dir = _default_llama2_map_dir(str(args.dataset_name), int(args.seed))
        if not args.out_dir:
            args.out_dir = _default_llama2_out_dir(
                str(args.dataset_name),
                int(args.seed),
                uses_map=bool(args.map_dir),
                strategy=str(args.slice_strategy),
            )
    elif str(args.dataset_name).strip().lower() == "obqa":
        if not args.base_model:
            args.base_model = DEFAULT_OBQA_BASE_MODEL_NAME
        if not args.map_dir and score_with == "map":
            args.map_dir = _default_obqa_map_dir(int(args.seed))
        if not args.out_dir:
            args.out_dir = _default_obqa_out_dir(
                int(args.seed),
                uses_map=bool(args.map_dir),
                strategy=str(args.slice_strategy),
            )
    elif score_with == "map" and not args.map_dir:
        raise ValueError(
            "--score_with map requires --map_dir unless --model_family llama2 "
            "or default OBQA can infer the checkpoint path."
        )
    elif not args.out_dir:
        raise ValueError("--out_dir is required for non-OBQA tasks.")
    scoring_protocol = _resolve_scoring_protocol(args)
    apply_choice_mask = scoring_protocol != "bayesian_peft"
    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if any(int(n) <= 0 for n in num_slices_values):
        raise ValueError("--num_slices/--num_slices_list values must be positive.")
    if int(args.kfac_per_slice) < 0:
        print("[KFAC] kfac_per_slice < 0; saving all scored examples to kfac_balanced.")
    if str(args.slice_strategy).strip().lower() == "stratified_mix":
        if int(args.mix_num_bands) <= 0:
            raise ValueError("--mix_num_bands must be positive.")
        if float(args.mix_strength) < 0:
            raise ValueError("--mix_strength must be non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(
        f"[Protocol] scoring_protocol={scoring_protocol} "
        f"apply_choice_mask={bool(apply_choice_mask)}"
    )
    print(f"[Model Family] {model_family}")
    print(f"[Score With] {score_with}")
    print(f"[Base Model] {args.base_model}")
    print(f"[Map Dir] {args.map_dir or '<none>'}")
    print(f"[Out Dir] {args.out_dir}")
    print(f"[Slice Counts] {num_slices_values}")
    print(
        f"[Slice Strategy] {args.slice_strategy} "
        f"mix_num_bands={int(args.mix_num_bands)} mix_strength={float(args.mix_strength):.3f}"
    )
    print(f"[Load] dataset_name={args.dataset_name} split={args.split}")
    ds_raw = _load_split(str(args.dataset_name), str(args.split))
    print(f"[Load] raw size={len(ds_raw)}")

    model, tokenizer, amp_dtype, model_desc, logits_are_choice_space = _load_model_and_tokenizer(
        args,
        device,
        scoring_protocol,
    )
    print(f"[Scoring Model] {model_desc}")

    examples = build_scoring_examples(
        ds_raw,
        str(args.dataset_name),
        scoring_protocol=scoring_protocol,
        sciq_shuffle_choices=bool(args.sciq_shuffle_choices),
        seed=int(args.seed),
    )
    score_ds = build_scoring_dataset(examples, tokenizer, max_seq_len=int(args.max_seq_len))
    collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )
    loader = DataLoader(
        score_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    num_classes = _num_classes_for_scoring(str(args.dataset_name), scoring_protocol)
    if logits_are_choice_space:
        choice_token_ids = torch.arange(int(num_classes), device=device, dtype=torch.long)
        print(f"[Target token ids] using trimmed choice-space ids={choice_token_ids.tolist()}")
    else:
        choice_token_ids = _choice_token_ids_for_scoring(
            tokenizer,
            device,
            str(args.dataset_name),
            scoring_protocol,
            bool(args.bayesian_peft_add_space),
        )
    losses, raw_indices = compute_choice_losses(
        model,
        loader,
        choice_token_ids=choice_token_ids,
        device=device,
        amp_dtype=amp_dtype,
        apply_choice_mask=bool(apply_choice_mask),
    )
    if raw_indices.shape[0] != losses.shape[0]:
        raise RuntimeError("Internal mismatch between scored losses and raw indices.")

    multi_t = len(num_slices_values) > 1
    for num_slices in num_slices_values:
        slice_ids = assign_slice_ids(
            losses,
            int(num_slices),
            strategy=str(args.slice_strategy),
            mix_num_bands=int(args.mix_num_bands),
            mix_strength=float(args.mix_strength),
            seed=int(args.seed) + int(num_slices) * 1009,
        )
        summary = _summarize_slices(losses, slice_ids, int(num_slices))
        out_dir = (
            os.path.join(str(args.out_dir), f"T{int(num_slices)}")
            if multi_t
            else str(args.out_dir)
        )
        print(f"\n[Loss Quantile Slices] T={int(num_slices)} easy-to-hard")
        for row in summary:
            print(
                f"  slice={int(row['slice_id'])} n={int(row['n'])} "
                f"loss=[{row.get('loss_min', float('nan')):.4f}, "
                f"{row.get('loss_mean', float('nan')):.4f}, "
                f"{row.get('loss_max', float('nan')):.4f}]"
            )

        _save_outputs(
            ds_raw=ds_raw,
            out_dir=out_dir,
            valid_raw_indices=raw_indices,
            losses=losses,
            slice_ids=slice_ids,
            rng=np.random.default_rng(int(args.seed)),
            num_slices=int(num_slices),
            kfac_per_slice=int(args.kfac_per_slice),
            save_full_train=bool(args.save_full_train),
            save_kfac_balanced=bool(args.save_kfac_balanced),
        )

        meta = {
            "method": "loss_quantile_easy_to_hard",
            "dataset_name": str(args.dataset_name),
            "split": str(args.split),
            "seed": int(args.seed),
            "num_slices": int(num_slices),
            "num_slices_values": [int(x) for x in num_slices_values],
            "losses_computed_once": bool(multi_t),
            "slice_strategy": str(args.slice_strategy),
            "mix_num_bands": int(args.mix_num_bands),
            "mix_strength": float(args.mix_strength),
            "kfac_per_slice": int(args.kfac_per_slice),
            "base_model": str(args.base_model),
            "model_family": str(args.model_family),
            "map_dir": str(args.map_dir),
            "tokenizer_name": str(args.tokenizer_name),
            "scoring_model": model_desc,
            "scoring_protocol": scoring_protocol,
            "num_classes": int(num_classes),
            "apply_choice_mask": bool(apply_choice_mask),
            "logits_are_choice_space": bool(logits_are_choice_space),
            "bayesian_peft_add_space": bool(args.bayesian_peft_add_space),
            "attn_implementation": str(args.attn_implementation),
            "max_seq_len": int(args.max_seq_len),
            "summary": summary,
        }
        meta_path = os.path.join(out_dir, "loss_slices_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"[Save] metadata -> {meta_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
