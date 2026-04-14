from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    DynamicEvalCollator,
    get_choice_token_ids,
    get_task_num_classes,
    get_transformer_and_lm_head,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    preprocess_task,
)

SEED = 0
TRUST_REMOTE_CODE = False
MAX_SEQ_LEN = 300
EVAL_BSZ = 32
MC_SAMPLES = 32
TEMP = 1.0


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_allocated())


def _peak_reserved_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_reserved())


class _StageTimer:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0 = None

    def __enter__(self):
        _reset_cuda_peak()
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _cuda_sync()
        dt = time.perf_counter() - self.t0
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(f"[PEAK] {self.tag}: alloc={_peak_alloc_gb():.2f} GB  reserved={_peak_reserved_gb():.2f} GB")


def _parse_eval_tasks(spec: str, default_task: str) -> List[str]:
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
    out: List[str] = []
    seen = set()
    for task in expanded:
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out


def _add_seq_len(ds):
    if "seq_len" in ds.column_names:
        return ds
    return ds.add_column("seq_len", [len(x) for x in ds["input_ids"]])


def trim_lm_head_to_choice_tokens(model: nn.Module, choice_token_ids: torch.Tensor) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    _, lm_head = get_transformer_and_lm_head(base)
    weight = lm_head.weight.index_select(0, choice_token_ids).detach()
    bias = (
        lm_head.bias.index_select(0, choice_token_ids).detach()
        if getattr(lm_head, "bias", None) is not None
        else None
    )
    new_head = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=(bias is not None),
        device=weight.device,
        dtype=weight.dtype,
    )
    new_head.weight.data.copy_(weight)
    if bias is not None:
        new_head.bias.data.copy_(bias)
    if hasattr(base, "lm_head"):
        base.lm_head = new_head
    else:
        raise RuntimeError("Could not locate lm_head on base model for trimming.")
    if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
        base.config.vocab_size = int(choice_token_ids.numel())


def _force_lora_fp32(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype=torch.float32)


def _mask_invalid_choices(logits: torch.Tensor, num_choices) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


def _load_base_and_adapter(task: str, map_adapter_dir: str, amp_dtype: torch.dtype, device: torch.device):
    if not os.path.isdir(map_adapter_dir):
        raise RuntimeError(f"Adapter dir not found: {map_adapter_dir}")

    peft_cfg = PeftConfig.from_pretrained(map_adapter_dir)
    base_name = peft_cfg.base_model_name_or_path
    print(f"[Load] base_model = {base_name}")
    print(f"[Load] adapter    = {map_adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "left"

    num_classes = get_task_num_classes(task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa",
    ).to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    model = PeftModel.from_pretrained(base_model, map_adapter_dir).to(device)
    model.eval()
    _force_lora_fp32(model)
    return tokenizer, model, num_classes


def _make_eval_loader(proc, tokenizer, device: torch.device, bsz: int) -> DataLoader:
    if "seq_len" in proc.column_names:
        proc = proc.sort("seq_len").remove_columns(["seq_len"])
    collator = DynamicEvalCollator(tokenizer=tokenizer, pad_to_multiple_of=(8 if device.type == "cuda" else None))
    return DataLoader(
        proc,
        batch_size=bsz,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


@torch.inference_mode()
def eval_mcdrop_one_dataset(model, loader, device, amp_dtype, mc_samples: int, temp: float):
    model.eval()
    transformer, lm_head = get_transformer_and_lm_head(model)
    num_classes = lm_head.out_features

    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 10)
    acc_m.reset()
    ece_m.reset()

    total = 0
    nll_sum = 0.0
    brier_sum = 0.0
    std_sum = 0.0
    eps = 1e-12
    inv_temp = 1.0 / float(temp) if float(temp) != 1.0 else 1.0
    dropouts = [mod for mod in model.modules() if isinstance(mod, nn.Dropout)]

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        bsz = labels.size(0)
        logits_list = []

        old_training = dropouts[0].training if dropouts else False
        for dropout in dropouts:
            dropout.training = True
        for _ in range(int(mc_samples)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = lm_head(out.last_hidden_state[:, -1, :]).float()
            logits = _mask_invalid_choices(logits, batch.get("num_choices"))
            if inv_temp != 1.0:
                logits = logits * inv_temp
            logits_list.append(logits)
        for dropout in dropouts:
            dropout.training = old_training

        logits = torch.stack(logits_list, dim=1)
        probs = torch.softmax(logits, dim=-1).mean(dim=1)
        std_sum += float(torch.softmax(logits, dim=-1).std(dim=1).mean().item()) * bsz
        p_y = probs[torch.arange(bsz, device=device), labels].clamp_min(eps)
        nll_sum += float((-torch.log(p_y)).sum().item())
        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )
        total += bsz

    model.eval()
    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": brier_sum / max(total, 1),
        "std": std_sum / max(total, 1),
        "mc_samples": int(mc_samples),
    }


def main():
    ap = argparse.ArgumentParser(description="Run MC Dropout evaluation on IID/OOD tasks.")
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
    )
    ap.add_argument("--map_adapter_dir", type=str, required=True)
    ap.add_argument("--eval_tasks", type=str, default="")
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--eval_bsz", type=int, default=EVAL_BSZ)
    ap.add_argument("--mc_samples", type=int, default=MC_SAMPLES)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    requested_seed = int(args.seed)
    effective_seed = int(SEED)
    if requested_seed != effective_seed:
        print(
            f"[Seed] requested --seed={requested_seed}, but MCDrop internal seed is fixed to {effective_seed} for benchmark consistency."
        )
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    random.seed(effective_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with _StageTimer(f"LOAD-STAGE MCDrop on {args.task}"):
        tokenizer, model, num_classes = _load_base_and_adapter(
            task=args.task,
            map_adapter_dir=args.map_adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
        )
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)

    print("\n========================")
    print("      MCDROP ONLY       ")
    print("========================")
    print(f"[Config] MC_SAMPLES={args.mc_samples} TEMP={args.temp}")

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
        loader = _make_eval_loader(eval_proc, tokenizer, device, bsz=args.eval_bsz)
        with _StageTimer(f"INFER MCDrop on {eval_task}({split_name})"):
            m = eval_mcdrop_one_dataset(model, loader, device, amp_dtype, args.mc_samples, args.temp)
        print(f"\n[{eval_task}({split_name})][MCDROP]")
        print(
            f"  NLL={m['nll']:.4f}  ACC={m['acc']*100:.2f}%  ECE={m['ece']*100:.2f}%"
            f"  Brier={m['brier']:.4f}  std={m['std']:.4f}  mc={m['mc_samples']}"
        )


if __name__ == "__main__":
    main()
