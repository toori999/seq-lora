from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common_eval_utils import (
    get_task_num_classes,
    get_transformer_and_lm_head,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    preprocess_task,
)
from map_eval import (
    EVAL_BSZ,
    MAX_SEQ_LEN,
    SEED,
    _add_seq_len,
    _cuda_sync,
    _load_base_and_adapter,
    _make_eval_loader,
    _mask_invalid_choices,
    _multiclass_brier_score,
    _parse_eval_tasks,
    _peak_alloc_gb,
    _peak_reserved_gb,
    _reset_cuda_peak,
)


def _format_seconds(dt: float) -> str:
    return f"{dt:.2f} sec ({dt/60:.2f} min)"


@torch.inference_mode()
def _predict_probs_one_dataset(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformer, lm_head = get_transformer_and_lm_head(model)
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = lm_head(out.last_hidden_state[:, -1, :]).float()
        logits = _mask_invalid_choices(logits, batch.get("num_choices"))
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_all = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, lm_head.out_features), dtype=torch.float32)
    labels_all = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    return probs_all, labels_all


def _metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    device = torch.device("cpu")
    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 15)
    acc_m.reset()
    ece_m.reset()
    acc_m.update(probs, labels)
    ece_m.update(probs, labels)
    eps = 1e-12
    nll = float((-torch.log(probs[torch.arange(labels.numel()), labels].clamp_min(eps))).mean().item()) if labels.numel() > 0 else float("nan")
    return {
        "nll": nll,
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": _multiclass_brier_score(probs, labels) if labels.numel() > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Deep Ensemble formed by multiple MAP LoRA adapters.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", "scienceqa_closedchoice_grade2_11"],
    )
    parser.add_argument("--map_adapter_dir", action="append", required=True, help="Repeat this flag once per ensemble member.")
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--eval_bsz", type=int, default=EVAL_BSZ)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    adapter_dirs = [str(x) for x in args.map_adapter_dir if str(x).strip()]
    if len(adapter_dirs) < 2:
        raise ValueError("Deep Ensemble evaluation expects at least two --map_adapter_dir values.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print("Using device:", device, "amp_dtype:", amp_dtype)
    print(f"[Ensemble] members={len(adapter_dirs)}")

    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    task_to_loader: Dict[str, object] = {}
    task_to_split: Dict[str, str] = {}
    task_prob_sums: Dict[str, Optional[torch.Tensor]] = {task: None for task in eval_tasks}
    task_labels: Dict[str, Optional[torch.Tensor]] = {task: None for task in eval_tasks}
    task_time_sec: Dict[str, float] = {task: 0.0 for task in eval_tasks}
    task_peak_alloc_gb: Dict[str, float] = {task: 0.0 for task in eval_tasks}
    task_peak_reserved_gb: Dict[str, float] = {task: 0.0 for task in eval_tasks}

    num_classes: Optional[int] = None
    tokenizer = None

    for member_idx, adapter_dir in enumerate(adapter_dirs):
        print("\n========================")
        print(f" ENSEMBLE MEMBER {member_idx + 1}/{len(adapter_dirs)} ")
        print("========================")
        print(f"[Load] adapter={adapter_dir}")
        tokenizer_i, model, member_num_classes = _load_base_and_adapter(
            task=args.task,
            map_adapter_dir=adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
        )
        if tokenizer is None:
            tokenizer = tokenizer_i
        if num_classes is None:
            num_classes = int(member_num_classes)
            for eval_task in eval_tasks:
                eval_num_classes = get_task_num_classes(eval_task)
                if eval_num_classes != num_classes:
                    raise ValueError(
                        f"Eval task '{eval_task}' has {eval_num_classes} classes, but source task '{args.task}' has {num_classes} classes."
                    )
                if eval_task == args.task:
                    _, _, eval_raw = load_task_dataset(eval_task)
                    split_name = "test"
                else:
                    eval_raw = load_eval_dataset(eval_task)
                    split_name = "ood"
                eval_proc = _add_seq_len(preprocess_task(eval_task, eval_raw, tokenizer, args.max_seq_len, pad_to_max_length=False))
                task_to_loader[eval_task] = _make_eval_loader(eval_proc, tokenizer, device, bsz=args.eval_bsz)
                task_to_split[eval_task] = split_name
        elif int(member_num_classes) != int(num_classes):
            raise ValueError(f"Ensemble member {adapter_dir} has {member_num_classes} classes, expected {num_classes}.")

        for eval_task in eval_tasks:
            split_name = task_to_split[eval_task]
            tag = f"{eval_task}({split_name})"
            _reset_cuda_peak()
            _cuda_sync()
            t0 = time.perf_counter()
            probs, labels = _predict_probs_one_dataset(model, task_to_loader[eval_task], device, amp_dtype)
            _cuda_sync()
            dt = time.perf_counter() - t0
            task_time_sec[eval_task] += float(dt)
            task_peak_alloc_gb[eval_task] = max(task_peak_alloc_gb[eval_task], _peak_alloc_gb())
            task_peak_reserved_gb[eval_task] = max(task_peak_reserved_gb[eval_task], _peak_reserved_gb())
            print(
                f"[Member {member_idx + 1}] {tag}: "
                f"time={_format_seconds(dt)} alloc={_peak_alloc_gb():.2f}GB reserved={_peak_reserved_gb():.2f}GB"
            )

            if task_prob_sums[eval_task] is None:
                task_prob_sums[eval_task] = probs
                task_labels[eval_task] = labels
            else:
                if task_labels[eval_task] is None or not torch.equal(task_labels[eval_task], labels):
                    raise RuntimeError(f"Label ordering mismatch while building ensemble for task {eval_task}.")
                task_prob_sums[eval_task] = task_prob_sums[eval_task] + probs

        del model
        del tokenizer_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert num_classes is not None
    print("\n========================")
    print("     ENSEMBLE ONLY      ")
    print("========================")

    for eval_task in eval_tasks:
        split_name = task_to_split[eval_task]
        tag = f"{eval_task}({split_name})"
        probs_sum = task_prob_sums[eval_task]
        labels = task_labels[eval_task]
        if probs_sum is None or labels is None:
            raise RuntimeError(f"Missing ensemble accumulation for task {eval_task}.")
        probs = probs_sum / float(len(adapter_dirs))
        metrics = _metrics_from_probs(probs, labels, num_classes)
        dt = task_time_sec[eval_task]
        print(f"[TIME] INFER Ensemble on {tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(
            f"[PEAK] INFER Ensemble on {tag}: "
            f"alloc={task_peak_alloc_gb[eval_task]:.2f} GB  reserved={task_peak_reserved_gb[eval_task]:.2f} GB"
        )
        print(f"\n[{tag}][ENSEMBLE]")
        print(
            f"  NLL={metrics['nll']:.4f}  ACC={metrics['acc']*100:.2f}%  "
            f"ECE={metrics['ece']*100:.2f}%  Brier={metrics['brier']:.4f}  members={len(adapter_dirs)}"
        )


if __name__ == "__main__":
    main()
