from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from seq_lora.algorithm.pseudo_observations import get_param_weight


@dataclass
class LoraACache:
    name: str
    weight: nn.Parameter
    U_fp32: Tensor
    B_fp32: Tensor
    scaling: float
    offset: int
    L: int
    shape: Tuple[int, ...]
    numel: int


def resolve_lora_parent_and_adapter(module_path: str) -> Tuple[str, str]:
    parts = module_path.split(".")
    if len(parts) < 3 or parts[-2] != "lora_A":
        raise RuntimeError(f"Unexpected LoRA-A module path: {module_path}")
    return ".".join(parts[:-2]), parts[-1]


def build_loraA_cache(model: nn.Module, module_specs: List[Dict], device: torch.device) -> List[LoraACache]:
    caches: List[LoraACache] = []
    for spec in module_specs:
        module_name = spec["name"]
        weight = get_param_weight(model, module_name)
        if weight.dtype != torch.float32:
            weight.data = weight.data.to(dtype=torch.float32)
        parent_path, adapter_name = resolve_lora_parent_and_adapter(module_name)
        parent = model.get_submodule(parent_path)
        if not hasattr(parent, "lora_B") or adapter_name not in parent.lora_B:
            raise RuntimeError(f"Could not resolve lora_B for {module_name}")
        if not hasattr(parent, "scaling") or adapter_name not in parent.scaling:
            raise RuntimeError(f"Could not resolve scaling for {module_name}")
        B_weight = parent.lora_B[adapter_name].weight
        if B_weight.dtype != torch.float32:
            B_weight = B_weight.to(dtype=torch.float32)
        caches.append(
            LoraACache(
                name=module_name,
                weight=weight,
                U_fp32=spec["subspace_info"]["U_lora"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).contiguous(),
                B_fp32=B_weight.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).contiguous(),
                scaling=float(parent.scaling[adapter_name]),
                offset=int(spec["offset"]),
                L=int(spec["L"]),
                shape=tuple(weight.shape),
                numel=weight.numel(),
            )
        )
    return caches


@torch.inference_mode()
def compute_deltas_for_one_sample(lora_cache: List[LoraACache], xs: Tensor, scale: float) -> List[Tensor]:
    return [
        (spec.U_fp32 @ xs[spec.offset: spec.offset + spec.L] * float(scale)).view(spec.shape)
        for spec in lora_cache
    ]

