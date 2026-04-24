from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file as _load_safetensors_file
except Exception:
    _load_safetensors_file = None

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
EVAL_BSZ = 256


def _load_adapter_checkpoint(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        if _load_safetensors_file is None:
            raise RuntimeError("safetensors is required to load adapter_model.safetensors")
        return _load_safetensors_file(safetensors_path)
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected adapter checkpoint object type: {type(state)}")
        return state
    raise FileNotFoundError(
        f"Could not find adapter checkpoint under {adapter_dir}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def _remap_legacy_peft_keys(state_dict: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], int]:
    remapped: Dict[str, torch.Tensor] = {}
    num_changed = 0
    old_prefix = "base_model.model.base_model.model."
    new_prefix = "base_model.model."

    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith(old_prefix):
            new_key = new_prefix + new_key[len(old_prefix):]
        if new_key != key:
            num_changed += 1
        remapped[new_key] = value
    return remapped, num_changed


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


def _mask_invalid_choices(logits: torch.Tensor, num_choices: Optional[Sequence[int]]) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


def _multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def _load_base_and_adapter(
    task: str,
    map_adapter_dir: str,
    amp_dtype: torch.dtype,
    device: torch.device,
):
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
    b_norms = [float(p.data.norm()) for n, p in model.named_parameters() if "lora_B" in n]
    if b_norms and all(v == 0.0 for v in b_norms):
        print("[Load] Standard PEFT adapter load left all LoRA-B tensors at zero; retrying with legacy key remap.")
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
        model = get_peft_model(base_model, peft_cfg, adapter_name="default").to(device)
        adapter_state = _load_adapter_checkpoint(map_adapter_dir)
        adapter_state, num_remapped = _remap_legacy_peft_keys(adapter_state)
        if num_remapped > 0:
            print(f"[Load] Remapped {num_remapped} adapter tensors for legacy checkpoint compatibility.")
        incompat = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
        missing_lora = [k for k in incompat.missing_keys if "lora_" in k]
        unexpected_lora = [k for k in incompat.unexpected_keys if "lora_" in k]
        if missing_lora or unexpected_lora:
            raise RuntimeError(
                "Legacy adapter load left unresolved LoRA tensors. "
                f"missing={missing_lora[:8]} unexpected={unexpected_lora[:8]}"
            )
        trim_lm_head_to_choice_tokens(model.base_model.model, choice_token_ids)
        print(f"[Head] trimmed lm_head to {num_classes} choice logits after legacy adapter load")
    model.eval()
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
def eval_map_one_dataset(model, loader, device, amp_dtype):
    _, lm_head = get_transformer_and_lm_head(model)
    transformer, _ = get_transformer_and_lm_head(model)
    num_classes = lm_head.out_features
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 10)
    acc_m.reset()
    ece_m.reset()

    total = 0
    nll_sum = 0.0
    brier_sum = 0.0

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

        nll_sum += float(ce_sum(logits, labels).item())
        probs = torch.softmax(logits, dim=-1)
        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )
        total += labels.size(0)

    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": (brier_sum / max(total, 1) if total > 0 else float("nan")),
    }


def main():
    parser = argparse.ArgumentParser(description="Run deterministic MAP-only evaluation on IID/OOD tasks.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
    )
    parser.add_argument("--map_adapter_dir", type=str, required=True)
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--eval_bsz", type=int, default=EVAL_BSZ)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with _StageTimer(f"LOAD-STAGE MAP on {args.task}"):
        tokenizer, model, num_classes = _load_base_and_adapter(
            task=args.task,
            map_adapter_dir=args.map_adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
        )

    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    print("\n========================")
    print("        MAP ONLY        ")
    print("========================")

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
        with _StageTimer(f"INFER MAP on {eval_task}({split_name})"):
            m = eval_map_one_dataset(model, loader, device, amp_dtype)

        print(f"\n[{eval_task}({split_name})][MAP]")
        print(
            f"  NLL={m['nll']:.4f}  ACC={m['acc']*100:.2f}%  "
            f"ECE={m['ece']*100:.2f}%  Brier={m['brier']:.4f}"
        )


if __name__ == "__main__":
    main()
