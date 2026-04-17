import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, concatenate_datasets
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


def _parse_eval_tasks(spec: str, default_task: str) -> List[str]:
    alias_map = {
        "iid": default_task,
        "grade12": getattr(ceu, "SCIENCEQA_GRADE12_TASK_NAME", "scienceqa_closedchoice_grade12"),
        "scienceqa_closedchoice_grade12": getattr(ceu, "SCIENCEQA_GRADE12_TASK_NAME", "scienceqa_closedchoice_grade12"),
        "obqa": "obqa",
        "arc-c": "arc-c",
        "arcc": "arc-c",
        "arc_c": "arc-c",
        "mmlu-h": "mmlu_science_high",
        "mmlu_high": "mmlu_science_high",
        "mmlu_science_high": "mmlu_science_high",
        "mmlu-c": "mmlu_science_college",
        "mmlu_college": "mmlu_science_college",
        "mmlu_science_college": "mmlu_science_college",
        "gpqa": "gpqa_main",
        "gpqa_main": "gpqa_main",
    }
    if not spec or not spec.strip():
        return [default_task]

    out: List[str] = []
    seen = set()
    for raw in str(spec).split(","):
        key = raw.strip().lower()
        if not key:
            continue
        task = alias_map.get(key, key)
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out or [default_task]


def _subset_dataset(ds: Dataset, subset_size: int, seed: int) -> Dataset:
    subset_size = int(subset_size)
    if subset_size <= 0 or subset_size >= len(ds):
        return ds
    return ds.shuffle(seed=int(seed)).select(range(subset_size))


def _order_scienceqa_train_by_grade(train_ds: Dataset, seed: int) -> Dataset:
    if "grade_num" not in train_ds.column_names:
        return train_ds

    grade_min = int(getattr(ceu, "SCIENCEQA_GRADE_MIN", 2))
    grade_max = int(getattr(ceu, "SCIENCEQA_GRADE_MAX", 11))
    parts = []
    for grade_num in range(grade_min, grade_max + 1):
        idxs = [i for i, g in enumerate(train_ds["grade_num"]) if int(g) == grade_num]
        if not idxs:
            continue
        ds_g = train_ds.select(idxs).shuffle(seed=int(seed) + grade_num)
        parts.append(ds_g)

    if not parts:
        raise RuntimeError("No ScienceQA training examples left after grade ordering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _resolve_source_eval(
    source_task: str,
    testing_set: str,
    seed: int,
) -> Tuple[Dataset, str, Dataset, str]:
    train_ds, val_ds, test_ds = ceu.load_task_dataset(source_task)
    testing_set = str(testing_set or "").strip().lower()

    # c_lora uses validation during training and a configurable source split for
    # final IID evaluation.
    train_eval_ds = val_ds
    train_eval_name = "validation"

    if testing_set in {"", "val", "train_train_val", "train_val_val"}:
        final_eval_ds, final_eval_name = val_ds, "validation"
    elif testing_set in {"test", "train_train_test", "train_val_test"}:
        final_eval_ds, final_eval_name = test_ds, "test"
    elif testing_set == "train_train_train":
        final_eval_ds, final_eval_name = _subset_dataset(train_ds, len(val_ds), seed), "train"
    else:
        raise ValueError(
            f"Unsupported testing_set={testing_set!r}. Expected one of: "
            "val, test, train_train_val, train_val_val, train_train_train, "
            "train_train_test, train_val_test"
        )

    return train_eval_ds, train_eval_name, final_eval_ds, final_eval_name


class _BenchmarkMCCollator:
    def __init__(self, tokenizer, target_ids: torch.Tensor, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.target_ids = target_ids.clone().detach().cpu()
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        inputs = [
            {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
            }
            for sample in batch
        ]
        padded = self.tokenizer.pad(
            inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.bos_token or self.tokenizer.eos_token

        self.source_task = str(args.dataset).strip()
        self.eval_task = str(getattr(args, "eval_dataset", "") or "").strip().lower()
        self.eval_tasks: List[str] = []
        self.eval_loaders: Dict[str, DataLoader] = {}
        self.eval_split_name_by_task: Dict[str, str] = {}
        self._lazy_eval_loaders: Dict[str, DataLoader] = {}
        self.eval_task_name = self.source_task
        self.eval_split_name = "validation"
        self.train_eval_split_name = "validation"
        self.source_eval_split_name = "validation"

        self.num_labels = ceu.get_task_num_classes(self.source_task)
        self.choice_token_ids = ceu.get_choice_token_ids(self.tokenizer, torch.device("cpu"), self.num_labels).view(-1, 1)
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
        if "seq_len" not in proc.column_names:
            proc = proc.add_column("seq_len", [len(x) for x in proc["input_ids"]])
        keep_cols = {"input_ids", "attention_mask", "labels", "num_choices"}
        extra_cols = [c for c in proc.column_names if c not in keep_cols | {"seq_len"}]
        if extra_cols:
            proc = proc.remove_columns(extra_cols)
        return proc

    def _loader(self, ds: Dataset, batch_size: int, drop_last: bool = False, sort_by_len: bool = False) -> DataLoader:
        proc_ds = ds
        if sort_by_len and "seq_len" in proc_ds.column_names:
            proc_ds = proc_ds.sort("seq_len")
        if "seq_len" in proc_ds.column_names:
            proc_ds = proc_ds.remove_columns(["seq_len"])
        return DataLoader(
            proc_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            collate_fn=_BenchmarkMCCollator(
                self.tokenizer,
                self.target_ids,
                pad_to_multiple_of=(8 if self.accelerator.device.type == "cuda" else None),
            ),
            num_workers=0,
        )

    def get_eval_loader_for_task(self, eval_task: str) -> DataLoader:
        eval_task = str(eval_task)
        if eval_task == self.source_task:
            return self.test_dataloader
        if eval_task in self._lazy_eval_loaders:
            return self._lazy_eval_loaders[eval_task]

        eval_num_labels = int(ceu.get_task_num_classes(eval_task))
        if eval_num_labels != self.num_labels:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_labels} classes, "
                f"but source task '{self.source_task}' has {self.num_labels} classes."
            )

        print(f"[benchmark_mcdataset] lazy loading eval task={eval_task}")
        eval_raw = ceu.load_eval_dataset(eval_task)
        eval_proc = self._prep(eval_task, eval_raw)
        eval_batch_size = int(getattr(self.args, "eval_batch_size", 48))
        loader = self._loader(
            eval_proc,
            eval_batch_size,
            drop_last=False,
            sort_by_len=True,
        )
        self._lazy_eval_loaders[eval_task] = loader
        self.eval_split_name_by_task[eval_task] = "ood"
        return loader

    def get_loaders(self):
        train_raw, _, _ = ceu.load_task_dataset(self.source_task)
        if self.source_task == getattr(ceu, "SCIENCEQA_CURRIC_TASK_NAME", "scienceqa_closedchoice_grade2_11"):
            train_raw = _order_scienceqa_train_by_grade(train_raw, self.args.seed)
        train_eval_raw, train_eval_name, source_eval_raw, source_eval_name = _resolve_source_eval(
            source_task=self.source_task,
            testing_set=self.args.testing_set,
            seed=self.args.seed,
        )
        self.train_eval_split_name = train_eval_name
        self.source_eval_split_name = source_eval_name

        train_proc = self._prep(self.source_task, train_raw)
        train_eval_proc = self._prep(self.source_task, train_eval_raw)
        source_test_proc = self._prep(self.source_task, source_eval_raw)

        self.num_samples = len(train_proc)
        eval_batch_size = int(getattr(self.args, "eval_batch_size", 48))
        sort_train_by_len = self.source_task != getattr(ceu, "SCIENCEQA_CURRIC_TASK_NAME", "scienceqa_closedchoice_grade2_11")
        self.train_dataloader = self._loader(train_proc, self.args.batch_size, drop_last=True, sort_by_len=sort_train_by_len)
        self.val_dataloader = self._loader(train_eval_proc, eval_batch_size, drop_last=False, sort_by_len=True)
        self.valid_dataloader = self.val_dataloader
        self.test_dataloader = self._loader(source_test_proc, eval_batch_size, drop_last=False, sort_by_len=True)

        eval_tasks = _parse_eval_tasks(self.eval_task, self.source_task)
        self.eval_tasks = eval_tasks
        self.eval_loaders = {}
        self.eval_split_name_by_task = {}
        for eval_task in eval_tasks:
            eval_num_labels = int(ceu.get_task_num_classes(eval_task))
            if eval_num_labels != self.num_labels:
                raise ValueError(
                    f"Eval task '{eval_task}' has {eval_num_labels} classes, "
                    f"but source task '{self.source_task}' has {self.num_labels} classes."
                )
            if eval_task == self.source_task:
                self.eval_loaders[eval_task] = self.test_dataloader
                self.eval_split_name_by_task[eval_task] = source_eval_name
                continue
            self.eval_split_name_by_task[eval_task] = "ood"

        primary_eval_task = eval_tasks[0] if eval_tasks else self.source_task
        self.eval_task_name = primary_eval_task
        self.eval_split_name = self.eval_split_name_by_task.get(primary_eval_task, source_eval_name)

        print(
            f"[benchmark_mcdataset] train={len(train_proc)} "
            f"val={len(train_eval_proc)}({train_eval_name}) "
            f"source_eval={len(source_test_proc)}({self.source_task}:{source_eval_name}) "
            f"eval_tasks={eval_tasks} "
            f"train_bsz={int(self.args.batch_size)} eval_bsz={eval_batch_size}"
        )
