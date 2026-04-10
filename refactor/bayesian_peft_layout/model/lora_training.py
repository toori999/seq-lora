from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import re

import torch
import torch.nn as nn

from refactor.seq_lora.eval.adapter_loading import (
    force_lora_fp32,
    trim_lm_head_to_choice_tokens,
)

DEFAULT_ATTENTION_TARGET_MODULES = ("q_proj", "v_proj")
DEFAULT_LM_HEAD_TARGET_MODULES = ("lm_head",)

_LORA_ADAPTER_PLACEHOLDER = "__adapter__"
_LORA_ADAPTER_RE = re.compile(r"(\.lora_(?:A|B)\.)([^.]+)(\.)")


def freeze_base_enable_lora(model: nn.Module) -> None:
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True


def resolve_qv_lm_head_target_modules(
    model: nn.Module,
    attention_target_modules: Sequence[str] = DEFAULT_ATTENTION_TARGET_MODULES,
    lm_head_target_modules: Sequence[str] = DEFAULT_LM_HEAD_TARGET_MODULES,
) -> list[str]:
    wanted_attention = set(attention_target_modules)
    wanted_lm_head = set(lm_head_target_modules)
    resolved = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if (
            ("layers." in name or ".layers." in name)
            and ("self_attn." in name or ".self_attn." in name)
            and suffix in wanted_attention
        ):
            resolved.append(name)
            continue
        if name in wanted_lm_head or suffix in wanted_lm_head:
            resolved.append(name)
    if not resolved:
        raise RuntimeError("Could not resolve any q/v attention or lm_head LoRA target modules")
    return sorted(set(resolved))


def get_lora_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if "lora_" in key
    }


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(lora_state, strict=False)


def _normalize_lora_key(key: str) -> str:
    return _LORA_ADAPTER_RE.sub(rf"\1{_LORA_ADAPTER_PLACEHOLDER}\3", key)


def _denormalize_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(f".{_LORA_ADAPTER_PLACEHOLDER}.", f".{adapter_name}.")


def get_normalized_lora_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "lora_" not in key or "lora_A_rho" in key:
            continue
        out[_normalize_lora_key(key)] = value.detach().cpu().clone()
    return out


def load_normalized_lora_state_dict(
    model: nn.Module,
    lora_state: Dict[str, torch.Tensor],
    adapter_name: str,
) -> None:
    mapped = {
        _denormalize_lora_key(key, adapter_name): value
        for key, value in lora_state.items()
    }
    model.load_state_dict(mapped, strict=False)


def sync_or_create_shared_lora_init(
    model: nn.Module,
    init_path: str | Path,
    adapter_name: str = "default",
) -> Path:
    init_path = Path(init_path)
    if init_path.exists():
        saved = torch.load(init_path, map_location="cpu")
        load_normalized_lora_state_dict(model, saved, adapter_name=adapter_name)
        print(f"[Init LoRA] loaded shared init from {init_path}")
    else:
        init_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(get_normalized_lora_state_dict_cpu(model), init_path)
        print(f"[Init LoRA] saved shared init to {init_path}")
    return init_path


__all__ = [
    "DEFAULT_ATTENTION_TARGET_MODULES",
    "DEFAULT_LM_HEAD_TARGET_MODULES",
    "force_lora_fp32",
    "freeze_base_enable_lora",
    "get_lora_state_dict_cpu",
    "get_normalized_lora_state_dict_cpu",
    "load_lora_state_dict",
    "load_normalized_lora_state_dict",
    "resolve_qv_lm_head_target_modules",
    "sync_or_create_shared_lora_init",
    "trim_lm_head_to_choice_tokens",
]
