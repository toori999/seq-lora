from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os

import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..datasets import get_task_num_classes
from ..metrics import get_choice_token_ids
from ..peft_utils import get_transformer_and_lm_head


@dataclass
class LoadedAdapterModel:
    tokenizer: AutoTokenizer
    model: PeftModel
    num_classes: int
    base_model_name: str


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


def force_lora_fp32(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(dtype=torch.float32)


def _configure_tokenizer(base_name: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.bos_token
            if tokenizer.bos_token is not None
            else tokenizer.eos_token
        )
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(
    base_name: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: str,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation=attn_implementation,
    ).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return model


def load_base_and_adapter(
    task: str,
    adapter_dir: str,
    amp_dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
    attn_implementation: str = "sdpa",
) -> LoadedAdapterModel:
    if not os.path.isdir(adapter_dir):
        raise RuntimeError(f"Adapter dir not found: {adapter_dir}")

    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    base_name = peft_cfg.base_model_name_or_path
    print(f"[Load] base_model = {base_name}")
    print(f"[Load] adapter    = {adapter_dir}")

    tokenizer = _configure_tokenizer(base_name, trust_remote_code=trust_remote_code)
    num_classes = get_task_num_classes(task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = _load_base_model(
        base_name=base_name,
        device=device,
        amp_dtype=amp_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()
    force_lora_fp32(model)
    return LoadedAdapterModel(
        tokenizer=tokenizer,
        model=model,
        num_classes=num_classes,
        base_model_name=base_name,
    )


def load_base_and_adapters(
    task: str,
    adapter_dirs: List[str],
    amp_dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
    attn_implementation: str = "sdpa",
) -> LoadedAdapterModel:
    if len(adapter_dirs) == 0:
        raise ValueError("adapter_dirs must not be empty")
    for adapter_dir in adapter_dirs:
        if not os.path.isdir(adapter_dir):
            raise RuntimeError(f"Adapter dir not found: {adapter_dir}")

    peft_cfg = PeftConfig.from_pretrained(adapter_dirs[0])
    base_name = peft_cfg.base_model_name_or_path
    print(f"[Load] base_model = {base_name}")
    for adapter_dir in adapter_dirs:
        print(f"[Load] adapter = {adapter_dir}")

    tokenizer = _configure_tokenizer(base_name, trust_remote_code=trust_remote_code)
    num_classes = get_task_num_classes(task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = _load_base_model(
        base_name=base_name,
        device=device,
        amp_dtype=amp_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    model = PeftModel.from_pretrained(base_model, adapter_dirs[0]).to(device)
    for idx, adapter_dir in enumerate(adapter_dirs[1:], start=1):
        model.load_adapter(adapter_dir, adapter_name=f"adapter_{idx}")
    force_lora_fp32(model)
    model.eval()
    return LoadedAdapterModel(
        tokenizer=tokenizer,
        model=model,
        num_classes=num_classes,
        base_model_name=base_name,
    )


def peft_set_adapter(model: PeftModel, name: str) -> None:
    if hasattr(model, "set_adapter"):
        model.set_adapter(name)
        return
    if hasattr(model, "active_adapter"):
        model.active_adapter = name
        return
    raise RuntimeError("PeftModel does not support set_adapter/active_adapter.")


__all__ = [
    "LoadedAdapterModel",
    "force_lora_fp32",
    "load_base_and_adapter",
    "load_base_and_adapters",
    "peft_set_adapter",
    "trim_lm_head_to_choice_tokens",
]
