from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import time

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..collators import DynamicEvalCollator
from ..datasets import get_task_num_classes, load_eval_dataset, load_task_dataset
from ..preprocessing import preprocess_task


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return mem_gb(torch.cuda.max_memory_allocated())


def peak_reserved_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return mem_gb(torch.cuda.max_memory_reserved())


class StageTimer:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0: Optional[float] = None

    def __enter__(self):
        reset_cuda_peak()
        cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        cuda_sync()
        assert self.t0 is not None
        dt = time.perf_counter() - self.t0
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(
            f"[PEAK] {self.tag}: alloc={peak_alloc_gb():.2f} GB  "
            f"reserved={peak_reserved_gb():.2f} GB"
        )


def resolve_device_amp_dtype() -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    return device, amp_dtype


def parse_eval_tasks(spec: str, default_task: str) -> List[str]:
    if not spec or not spec.strip():
        return [default_task]

    expanded: List[str] = []
    for raw in spec.split(","):
        task = raw.strip().lower()
        if not task:
            continue
        if task == "iid":
            expanded.append(default_task)
        elif task == "arc":
            expanded.extend(["arc-c", "arc-e"])
        elif task == "mmlu":
            expanded.extend(["mmlu_science_high", "mmlu_science_college"])
        else:
            expanded.append(task)

    deduped: List[str] = []
    seen = set()
    for task in expanded:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


def add_seq_len(ds: Dataset) -> Dataset:
    if "seq_len" in ds.column_names:
        return ds
    return ds.add_column("seq_len", [len(x) for x in ds["input_ids"]])


def make_eval_loader(
    proc: Dataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
) -> DataLoader:
    if "seq_len" in proc.column_names:
        proc = proc.sort("seq_len").remove_columns(["seq_len"])
    collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )
    return DataLoader(
        proc,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def mask_invalid_choices(
    logits: torch.Tensor,
    num_choices: Optional[Sequence[int]],
) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor(
        [int(x) for x in num_choices],
        device=logits.device,
        dtype=torch.long,
    )
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


@dataclass(frozen=True)
class EvalTask:
    name: str
    split_name: str
    num_classes: int


@dataclass
class PreparedEvalTask:
    spec: EvalTask
    raw_dataset: Dataset
    processed_dataset: Dataset
    loader: DataLoader


@dataclass(frozen=True)
class EvalRunContext:
    source_task: str
    max_seq_len: int
    eval_batch_size: int
    device: torch.device
    amp_dtype: torch.dtype


def load_raw_eval_task(source_task: str, eval_task: str) -> tuple[Dataset, str]:
    if eval_task == source_task:
        _, _, raw = load_task_dataset(eval_task)
        return raw, "test"
    return load_eval_dataset(eval_task), "ood"


def prepare_eval_task(
    source_task: str,
    eval_task: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    eval_batch_size: int,
    device: torch.device,
    expected_num_classes: Optional[int] = None,
) -> PreparedEvalTask:
    num_classes = get_task_num_classes(eval_task)
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ValueError(
            f"Eval task '{eval_task}' has {num_classes} classes, "
            f"but source task '{source_task}' has {expected_num_classes} classes."
        )
    raw_dataset, split_name = load_raw_eval_task(source_task, eval_task)
    processed_dataset = add_seq_len(
        preprocess_task(
            eval_task,
            raw_dataset,
            tokenizer,
            max_seq_len,
            pad_to_max_length=False,
        )
    )
    loader = make_eval_loader(
        processed_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=eval_batch_size,
    )
    return PreparedEvalTask(
        spec=EvalTask(name=eval_task, split_name=split_name, num_classes=num_classes),
        raw_dataset=raw_dataset,
        processed_dataset=processed_dataset,
        loader=loader,
    )


def prepare_eval_tasks(
    source_task: str,
    eval_tasks: Sequence[str],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    eval_batch_size: int,
    device: torch.device,
    expected_num_classes: Optional[int] = None,
) -> List[PreparedEvalTask]:
    return [
        prepare_eval_task(
            source_task=source_task,
            eval_task=eval_task,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            eval_batch_size=eval_batch_size,
            device=device,
            expected_num_classes=expected_num_classes,
        )
        for eval_task in eval_tasks
    ]


__all__ = [
    "EvalRunContext",
    "EvalTask",
    "PreparedEvalTask",
    "StageTimer",
    "add_seq_len",
    "cuda_sync",
    "make_eval_loader",
    "mask_invalid_choices",
    "parse_eval_tasks",
    "peak_alloc_gb",
    "peak_reserved_gb",
    "prepare_eval_task",
    "prepare_eval_tasks",
    "reset_cuda_peak",
    "resolve_device_amp_dtype",
]
