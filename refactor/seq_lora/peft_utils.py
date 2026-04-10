from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn


def get_active_adapter_name(model: nn.Module) -> str:
    if hasattr(model, "active_adapter"):
        active_adapter = model.active_adapter
        if isinstance(active_adapter, str):
            return active_adapter
        if isinstance(active_adapter, (list, tuple)) and len(active_adapter) > 0:
            return str(active_adapter[0])
    return "default"


def pick_adapter_module(maybe_mod, adapter_name: str):
    if isinstance(maybe_mod, (nn.ModuleDict, dict)):
        if adapter_name in maybe_mod:
            return maybe_mod[adapter_name]
        try:
            return next(iter(maybe_mod.values()))
        except StopIteration:
            return None
    return maybe_mod


def pick_scaling(maybe_scaling, adapter_name: str):
    if isinstance(maybe_scaling, dict):
        if adapter_name in maybe_scaling:
            return maybe_scaling[adapter_name]
        try:
            return next(iter(maybe_scaling.values()))
        except StopIteration:
            return 1.0
    if isinstance(maybe_scaling, (list, tuple)):
        return maybe_scaling[0] if len(maybe_scaling) > 0 else 1.0
    return maybe_scaling


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


def init_blob_rho_(rho: torch.Tensor, eps: float) -> torch.Tensor:
    if eps < 0:
        nn.init.uniform_(rho, eps - 1.0, eps)
    else:
        nn.init.uniform_(rho, eps / math.sqrt(2.0), eps)
    return rho


def blob_sigma_from_rho(rho: torch.Tensor) -> torch.Tensor:
    return rho.square()


def blob_kl_div_stable(
    mu_q: torch.Tensor,
    rho_q: torch.Tensor,
    mu_p: float = 0.0,
    sigma_p: float = 0.2,
) -> torch.Tensor:
    eps = 1e-6
    sigma_q = blob_sigma_from_rho(rho_q)
    kl = (
        math.log(float(sigma_p) + eps)
        - torch.log(sigma_q.to(torch.float64) + eps)
        + (
            sigma_q.to(torch.float64) ** 2
            + (mu_q.to(torch.float64) - float(mu_p)) ** 2
        )
        / (2 * (float(sigma_p) ** 2) + eps)
        - 0.5
    )
    return kl.sum()


def blob_sample_lora_noise(
    x: torch.Tensor,
    lora_a_weight: torch.Tensor,
    lora_b_weight: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    sigma_a = blob_sigma_from_rho(rho).to(dtype=lora_a_weight.dtype)
    if x.dim() == 2:
        r_a = torch.empty(
            (x.size(0), lora_a_weight.size(1)),
            device=x.device,
            dtype=x.dtype,
        ).uniform_(-1, 1).sign()
        s_a = torch.empty(
            (x.size(0), lora_a_weight.size(0)),
            device=x.device,
            dtype=x.dtype,
        ).uniform_(-1, 1).sign()
    elif x.dim() == 3:
        r_a = torch.empty(
            (x.size(0), x.size(1), lora_a_weight.size(1)),
            device=x.device,
            dtype=x.dtype,
        ).uniform_(-1, 1).sign()
        s_a = torch.empty(
            (x.size(0), x.size(1), lora_a_weight.size(0)),
            device=x.device,
            dtype=x.dtype,
        ).uniform_(-1, 1).sign()
    else:
        raise ValueError(f"Unsupported BLoB input rank {x.dim()}, expected 2 or 3.")

    lora_noise_a = sigma_a * torch.randn_like(lora_a_weight)
    return (((x * r_a) @ lora_noise_a.transpose(0, 1)) * s_a) @ lora_b_weight.transpose(0, 1)


def get_transformer_and_lm_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "model"):
        transformer = base.model
    elif hasattr(base, "transformer"):
        transformer = base.transformer
    else:
        raise RuntimeError("Cannot locate transformer body.")

    if hasattr(base, "lm_head"):
        lm_head = base.lm_head
    else:
        lm_head = base.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Cannot locate lm_head.")
    return transformer, lm_head


def lm_head_has_lora(lm_head: nn.Module) -> bool:
    return hasattr(lm_head, "lora_A") and hasattr(lm_head, "lora_B")


def get_lm_head_lora_scaling(lm_head: nn.Module, adapter: str) -> float:
    if hasattr(lm_head, "scaling"):
        scaling = pick_scaling(getattr(lm_head, "scaling"), adapter)
        if isinstance(scaling, (float, int)):
            return float(scaling)

    rank = getattr(lm_head, "r", None)
    if isinstance(rank, dict):
        rank = rank.get(adapter, None)
    alpha = getattr(lm_head, "lora_alpha", None)
    if isinstance(alpha, dict):
        alpha = alpha.get(adapter, None)
    if rank and alpha and float(rank) != 0:
        return float(alpha) / float(rank)
    return 1.0


def get_lm_head_dropout(lm_head: nn.Module, adapter: str) -> Optional[nn.Module]:
    dropout = getattr(lm_head, "lora_dropout", None)
    if isinstance(dropout, (nn.ModuleDict, dict)) and adapter in dropout:
        return dropout[adapter]
    if isinstance(dropout, nn.Module) and not isinstance(dropout, nn.ModuleDict):
        return dropout
    return None


def get_lm_head_lora_A_weight(
    lm_head: nn.Module, adapter: str
) -> Optional[torch.Tensor]:
    if not lm_head_has_lora(lm_head):
        return None
    a_module = pick_adapter_module(getattr(lm_head, "lora_A", None), adapter)
    return a_module.weight if hasattr(a_module, "weight") else None


def get_lm_head_lora_B_choice_fp32(
    lm_head: nn.Module,
    adapter: str,
    choice_token_ids: torch.Tensor,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    if not lm_head_has_lora(lm_head):
        return None
    if adapter in cache:
        return cache[adapter]
    b_module = pick_adapter_module(getattr(lm_head, "lora_B", None), adapter)
    if b_module is None or not hasattr(b_module, "weight"):
        return None
    choice_weights = (
        b_module.weight.index_select(0, choice_token_ids)
        .detach()
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    cache[adapter] = choice_weights
    return choice_weights


def set_inference_fast(model: nn.Module):
    if hasattr(model, "base_model") and hasattr(
        model.base_model, "gradient_checkpointing_disable"
    ):
        model.base_model.gradient_checkpointing_disable()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "config")
        and hasattr(model.base_model.config, "use_cache")
    ):
        model.base_model.config.use_cache = False


__all__ = [
    "blob_kl_div_stable",
    "blob_sample_lora_noise",
    "blob_sigma_from_rho",
    "get_active_adapter_name",
    "get_lm_head_dropout",
    "get_lm_head_lora_A_weight",
    "get_lm_head_lora_B_choice_fp32",
    "get_lm_head_lora_scaling",
    "get_transformer_and_lm_head",
    "init_blob_rho_",
    "lm_head_has_lora",
    "pick_adapter_module",
    "pick_scaling",
    "set_inference_fast",
    "softplus",
]
