import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.utils.datasetbase import DatasetBase


def _load_seq_lora_helpers():
    try:
        import common_eval_utils as ceu  # type: ignore

        return ceu
    except Exception:
        pass

    candidates = []
    env_root = os.getenv("SEQ_LORA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())

    for root in candidates:
        root = Path(root)
        if (root / "common_eval_utils.py").exists():
            sys.path.insert(0, str(root))
            import common_eval_utils as ceu  # type: ignore

            return ceu

    raise ImportError(
        "Could not import common_eval_utils. Run from the Seq-LoRA workspace root "
        "or set SEQ_LORA_ROOT=/path/to/Seq_LoRA."
    )


ceu = _load_seq_lora_helpers()


def _subset_dataset(ds: Dataset, subset_size: int, seed: int) -> Dataset:
    subset_size = int(subset_size)
    if subset_size <= 0 or subset_size >= len(ds):
        return ds
    return ds.shuffle(seed=int(seed)).select(range(subset_size))


def _resolve_source_anchor_and_eval(
    source_task: str,
    testing_set: str,
    anchor_size: int,
    seed: int,
) -> Tuple[Dataset, str, Dataset, str]:
    train_ds, val_ds, test_ds = ceu.load_task_dataset(source_task)
    testing_set = str(testing_set or "").strip().lower()

    if testing_set in {"", "val"}:
        return _subset_dataset(val_ds, anchor_size, seed), "validation", val_ds, "validation"
    if testing_set == "train_train_val":
        return _subset_dataset(train_ds, anchor_size, seed), "train", val_ds, "validation"
    if testing_set == "train_val_val":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", val_ds, "validation"
    if testing_set == "train_train_train":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", _subset_dataset(train_ds, anchor_size, seed), "train"
    if testing_set == "train_train_test":
        return _subset_dataset(train_ds, anchor_size, seed), "train", test_ds, "test"
    if testing_set == "train_val_test":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", test_ds, "test"

    raise ValueError(
        f"Unsupported testing_set={testing_set!r}. Expected one of: "
        "val, train_train_val, train_val_val, train_train_train, "
        "train_train_test, train_val_test"
    )


class _BenchmarkMCCollator:
    def __init__(self, tokenizer, target_ids: torch.Tensor):
        self.tokenizer = tokenizer
        self.target_ids = target_ids.clone().detach().cpu()

    def __call__(self, batch):
        inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in batch
        ]
        padded = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        labels = torch.tensor([int(sample["labels"]) for sample in batch], dtype=torch.long)
        targets = self.target_ids[labels].squeeze(-1)
        if "num_choices" in batch[0]:
            padded["num_choices"] = torch.tensor([int(sample["num_choices"]) for sample in batch], dtype=torch.long)
        return padded, labels, targets


class BenchmarkMCDataset(DatasetBase):
    NAME = "benchmark_mcdataset"

    def __init__(self, accelerator, args):
        super().__init__()
        self.args = args
        self.accelerator = accelerator

        accelerator.wait_for_everyone()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.bos_token or self.tokenizer.eos_token

        self.source_task = str(args.dataset).strip()
        self.eval_task = str(getattr(args, "eval_dataset", "") or "").strip().lower()
        self.anchor_split_name = ""
        self.source_eval_split_name = ""
        self.eval_task_name = self.source_task
        self.eval_split_name = "validation"

        self.num_labels = ceu.get_task_num_classes(self.source_task)
        self.choice_token_ids = ceu.get_choice_token_ids(self.tokenizer, torch.device("cpu"), self.num_labels).view(-1, 1)
        # benchmark_mcdataset uses a trimmed lm_head, so class ids are 0..C-1.
        self.target_ids = torch.arange(self.num_labels, dtype=torch.long).view(-1, 1)

        print("=====================================")
        print(f"Loaded benchmark_mcdataset source={self.source_task} eval={self.eval_task or 'iid'}")
        print("=====================================")

    def _prep(self, task: str, raw_ds: Dataset) -> Dataset:
        proc = ceu.preprocess_task(
            task,
            raw_ds,
            self.tokenizer,
            self.args.max_seq_len,
            pad_to_max_length=False,
        )
        keep_cols = {"input_ids", "attention_mask", "labels", "num_choices"}
        extra_cols = [c for c in proc.column_names if c not in keep_cols]
        if extra_cols:
            proc = proc.remove_columns(extra_cols)
        return proc

    def _loader(self, ds: Dataset, batch_size: int, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            collate_fn=_BenchmarkMCCollator(self.tokenizer, self.target_ids),
            num_workers=0,
        )

    def get_loaders(self):
        train_raw, _, _ = ceu.load_task_dataset(self.source_task)
        anchor_raw, anchor_split_name, source_eval_raw, source_eval_split_name = _resolve_source_anchor_and_eval(
            source_task=self.source_task,
            testing_set=self.args.testing_set,
            anchor_size=self.args.anchor_size,
            seed=self.args.seed,
        )

        eval_task = self.source_task if self.eval_task in {"", "iid"} else self.eval_task
        if eval_task == self.source_task:
            test_raw = source_eval_raw
            test_split_name = source_eval_split_name
        else:
            test_raw = ceu.load_eval_dataset(eval_task)
            test_split_name = "ood"

        self.anchor_split_name = anchor_split_name
        self.source_eval_split_name = source_eval_split_name
        self.eval_task_name = eval_task
        self.eval_split_name = test_split_name

        train_proc = self._prep(self.source_task, train_raw)
        anchor_proc = self._prep(self.source_task, anchor_raw)
        test_proc = self._prep(eval_task, test_raw)

        self.num_samples = len(train_proc)
        self.train_dataloader = self._loader(train_proc, self.args.batch_size, drop_last=False)
        self.anchor_dataloader = self._loader(anchor_proc, self.args.batch_size, drop_last=False)
        self.test_dataloader = self._loader(test_proc, self.args.batch_size, drop_last=False)

        print(
            f"[benchmark_mcdataset] train={len(train_proc)} "
            f"anchor={len(anchor_proc)}({anchor_split_name}) "
            f"test={len(test_proc)}({eval_task}:{test_split_name})"
        )
