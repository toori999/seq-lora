from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from .peft_utils import (
    blob_sample_lora_noise,
    get_active_adapter_name,
    get_lm_head_dropout,
    get_lm_head_lora_A_weight,
    get_lm_head_lora_B_choice_fp32,
    get_lm_head_lora_scaling,
    get_transformer_and_lm_head,
    lm_head_has_lora,
)


@dataclass
class ChoiceHeadCache:
    W_choice_fp32: torch.Tensor
    b_choice_fp32: Optional[torch.Tensor]
    transformer: nn.Module
    lm_head: nn.Module
    choice_token_ids: torch.Tensor
    num_classes: int
    B_choice_fp32_by_adapter: Dict[str, torch.Tensor] = field(default_factory=dict)
    bayes_eps: float = 0.0


def build_choice_head_cache(
    model: nn.Module,
    choice_token_ids: torch.Tensor,
    device: torch.device,
    bayes_eps: float = 0.0,
) -> ChoiceHeadCache:
    transformer, lm_head = get_transformer_and_lm_head(model)
    w_choice = lm_head.weight.index_select(0, choice_token_ids).detach()
    b_choice = (
        lm_head.bias.index_select(0, choice_token_ids).detach()
        if getattr(lm_head, "bias", None) is not None
        else None
    )
    return ChoiceHeadCache(
        W_choice_fp32=w_choice.to(
            device=device, dtype=torch.float32, non_blocking=True
        ).contiguous(),
        b_choice_fp32=(
            None
            if b_choice is None
            else b_choice.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        ),
        transformer=transformer,
        lm_head=lm_head,
        choice_token_ids=choice_token_ids,
        num_classes=len(choice_token_ids),
        bayes_eps=float(bayes_eps),
    )


def restricted_choice_logits_last_token(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_cache: ChoiceHeadCache,
    amp_dtype: torch.dtype,
    last_idx: Optional[torch.Tensor] = None,
    batch_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = input_ids.device
    batch_size = input_ids.size(0)

    if last_idx is None:
        last_token_is_valid = attention_mask[:, -1].to(dtype=torch.bool)
        last_idx = torch.empty((batch_size,), device=device, dtype=torch.long)
        last_idx[last_token_is_valid] = attention_mask.size(1) - 1
        if (~last_token_is_valid).any():
            last_idx[~last_token_is_valid] = (
                attention_mask[~last_token_is_valid].sum(dim=1) - 1
            )
    if batch_idx is None:
        batch_idx = torch.arange(batch_size, device=device)

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        out = choice_cache.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        h_last = out.last_hidden_state[batch_idx, last_idx, :]

    h_last_fp32 = h_last.float()
    logits = h_last_fp32 @ choice_cache.W_choice_fp32.t()
    if choice_cache.b_choice_fp32 is not None:
        logits += choice_cache.b_choice_fp32.view(1, choice_cache.num_classes)

    lm_head = choice_cache.lm_head
    if lm_head_has_lora(lm_head):
        adapter = get_active_adapter_name(model)
        a_weight = get_lm_head_lora_A_weight(lm_head, adapter)
        b_choice_fp32 = get_lm_head_lora_B_choice_fp32(
            lm_head=lm_head,
            adapter=adapter,
            choice_token_ids=choice_cache.choice_token_ids,
            device=device,
            cache=choice_cache.B_choice_fp32_by_adapter,
        )

        if a_weight is not None and b_choice_fp32 is not None:
            if torch.is_grad_enabled():
                h_last = h_last.clone()
                a_weight = a_weight.clone()
                b_choice_fp32 = b_choice_fp32.clone()

            drop = get_lm_head_dropout(lm_head, adapter)
            h_for_lora = h_last if drop is None else drop(h_last)
            h_for_lora = h_for_lora.to(dtype=a_weight.dtype)
            z_td = h_for_lora @ a_weight.to(dtype=a_weight.dtype).t()
            scaling = float(get_lm_head_lora_scaling(lm_head, adapter))
            lora_logits = (z_td.float() @ b_choice_fp32.t()) * scaling

            rho = getattr(lm_head, f"blob_rho_{adapter}", None)
            if isinstance(rho, nn.Parameter) and bool(
                getattr(lm_head, f"blob_sample_{adapter}", True)
            ):
                lora_logits += (
                    blob_sample_lora_noise(
                        x=h_for_lora,
                        lora_a_weight=a_weight.to(dtype=a_weight.dtype),
                        lora_b_weight=b_choice_fp32.to(dtype=a_weight.dtype),
                        rho=rho.to(dtype=a_weight.dtype),
                    ).float()
                    * scaling
                )

            logits += lora_logits

    return logits


def logits_via_lm_head_last_token_for_kfac(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    choice_cache: ChoiceHeadCache,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_ids.device
    batch_size = input_ids.size(0)
    last_token_is_valid = attention_mask[:, -1].to(dtype=torch.bool)
    last_idx = torch.empty((batch_size,), device=device, dtype=torch.long)
    last_idx[last_token_is_valid] = attention_mask.size(1) - 1
    if (~last_token_is_valid).any():
        last_idx[~last_token_is_valid] = (
            attention_mask[~last_token_is_valid].sum(dim=1) - 1
        )
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        out = choice_cache.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits_v = choice_cache.lm_head(
            out.last_hidden_state[torch.arange(batch_size, device=device), last_idx, :]
        )
    return logits_v.index_select(-1, choice_cache.choice_token_ids)


__all__ = [
    "ChoiceHeadCache",
    "build_choice_head_cache",
    "logits_via_lm_head_last_token_for_kfac",
    "restricted_choice_logits_last_token",
]
