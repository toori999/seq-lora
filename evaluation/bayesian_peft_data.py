from __future__ import annotations

import os
import sys
from typing import Optional

import datasets as hf_datasets
from torch.utils.data import DataLoader

from evaluation.protocols import is_bayesian_peft_protocol


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
BAYESIAN_PEFT_ROOT = os.path.join(PROJECT_ROOT, "baselines", "bayesian-peft")
if BAYESIAN_PEFT_ROOT not in sys.path:
    sys.path.append(BAYESIAN_PEFT_ROOT)

from dataset.utils import dsets as bayesian_peft_dsets  # noqa: E402


def bayesian_peft_dataset_name(task: str) -> Optional[str]:
    mapping = {
        "obqa": "obqa",
        "arc-e": "ARC-Easy",
        "arc-c": "ARC-Challenge",
        "mmlu-chem": "MMLU-chem",
        "mmlu-phy": "MMLU-phy",
    }
    return mapping.get(task)


def uses_direct_bayesian_peft_data(task: str, eval_protocol: str) -> bool:
    return is_bayesian_peft_protocol(eval_protocol) and bayesian_peft_dataset_name(task) is not None


def build_bayesian_peft_task_dataset(
    tokenizer,
    task: str,
    *,
    add_space: bool,
    max_seq_len: int,
):
    dataset_name = bayesian_peft_dataset_name(task)
    if dataset_name is None:
        raise ValueError(f"Task '{task}' does not have a direct bayesian-peft dataset wrapper.")
    if dataset_name.startswith("ARC"):
        return bayesian_peft_dsets.arc(
            tokenizer,
            add_space=add_space,
            name=dataset_name,
            max_seq_len=max_seq_len,
        )
    if dataset_name == "obqa":
        return bayesian_peft_dsets.obqa(
            tokenizer,
            add_space=add_space,
            max_seq_len=max_seq_len,
        )
    if dataset_name.startswith("MMLU-"):
        return bayesian_peft_dsets.mmlu(
            tokenizer,
            add_space=add_space,
            name=dataset_name[5:],
            max_seq_len=max_seq_len,
        )
    raise ValueError(f"Unhandled direct bayesian-peft dataset name: {dataset_name}")


def get_direct_bayesian_peft_eval_dataset(task_dataset, split: str):
    dset = task_dataset.dset
    if isinstance(dset, hf_datasets.Dataset):
        return dset
    if isinstance(dset, hf_datasets.DatasetDict):
        if split in dset:
            return dset[split]
        if split == "test" and "validation" in dset:
            return dset["validation"]
        if split == "validation" and "test" in dset:
            return dset["test"]
        raise KeyError(f"Split '{split}' not found in bayesian-peft dataset. Available splits: {list(dset.keys())}")
    if hasattr(dset, "keys"):
        if split in dset:
            return dset[split]
        if split == "test" and "validation" in dset:
            return dset["validation"]
        if split == "validation" and "test" in dset:
            return dset["test"]
    return dset


class BayesianPeftCLMCollator:
    def __init__(self, task_dataset):
        self.task_dataset = task_dataset

    def __call__(self, batch):
        prompts, classes, _targets = self.task_dataset.clm_collate_fn(batch)
        return {
            "input_ids": prompts["input_ids"],
            "attention_mask": prompts["attention_mask"],
            "labels": classes.to(dtype=torch.long),
        }


def make_direct_bayesian_peft_loader(
    raw_dataset,
    *,
    collate_fn,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(raw_dataset, **kwargs)
