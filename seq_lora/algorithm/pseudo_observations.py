from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from seq_lora.algorithm.asdl_backend import batch_gradient as asdl_batch_gradient
from seq_lora.algorithm.kfac import (
    AsdlForwardWrapper,
    iter_weight_block_module_names,
    set_dropout_modules_training,
    temporarily_select_lora_a_weights,
)
from seq_lora.algorithm.subspace import solve_xhat_from_grad


def get_param_weight(model: nn.Module, module_path: str) -> nn.Parameter:
    module = model.get_submodule(module_path)
    if not hasattr(module, "weight"):
        raise RuntimeError(f"[mu-obs] submodule has no .weight: {module_path}")
    weight = getattr(module, "weight")
    if not isinstance(weight, nn.Parameter):
        raise RuntimeError(f"[mu-obs] .weight is not nn.Parameter: {module_path}")
    return weight


def estimate_mu_global_list_from_slice_grads_asdl(
    model: nn.Module,
    slice_loaders: List[DataLoader],
    forward_call_for_kfac,
    module_names: List[str],
    module_subspace_info: Dict[str, Dict[str, Tensor]],
    module_R_lists: Dict[str, List[Tensor]],
    device: torch.device,
    n_batches_per_slice: int = 1,
    dtype: torch.dtype = torch.float64,
    *,
    disable_dropout: bool = False,
) -> List[Tensor]:
    wrapper = AsdlForwardWrapper(
        model,
        forward_call_for_kfac,
        force_fp32_output=False,
    ).to(device)
    wrapper.train()
    if disable_dropout:
        set_dropout_modules_training(wrapper, enabled=False)

    mu_global_list: List[Tensor] = []

    with temporarily_select_lora_a_weights(model):
        asdl_module_names = list(iter_weight_block_module_names(wrapper))
        missing = sorted(set(module_names) - set(asdl_module_names))
        if missing:
            extra = sorted(set(asdl_module_names) - set(module_names))
            raise RuntimeError(
                "ASDL gx module-name mismatch. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        extra = sorted(set(asdl_module_names) - set(module_names))
        if extra:
            print(f"[mu-obs-asdl] ignoring {len(extra)} extra ASDL module(s): {extra[:5]}")

        block_slices: Dict[str, slice] = {}
        offset = 0
        for name in asdl_module_names:
            numel = int(get_param_weight(model, name).numel())
            block_slices[name] = slice(offset, offset + numel)
            offset += numel

        for t, loader in enumerate(slice_loaders):
            g_x_parts = [
                torch.zeros(int(module_subspace_info[name]["U_lora"].shape[1]), device=device, dtype=dtype)
                for name in module_names
            ]
            n_seen = 0

            for batch in loader:
                if n_seen >= n_batches_per_slice:
                    break

                batch = {
                    key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                    for key, value in batch.items()
                }
                labels = batch["labels"].to(device=device, non_blocking=True)
                input_shape = tuple(batch["input_ids"].shape)

                def closure():
                    wrapper.zero_grad(set_to_none=True)
                    logits = wrapper(**batch)
                    loss = F.cross_entropy(logits.float(), labels, reduction="sum")
                    loss.backward()
                    return loss

                batch_grads, _ = asdl_batch_gradient(
                    wrapper,
                    closure,
                    input_shape,
                    return_outputs=True,
                )
                batch_mean_grad = batch_grads.mean(dim=0).to(device=device, dtype=dtype)

                for mi, name in enumerate(module_names):
                    grad_block = batch_mean_grad[block_slices[name]]
                    g_x_parts[mi] += (
                        module_subspace_info[name]["U_lora"].to(device=device, dtype=dtype).T @ grad_block
                    )
                n_seen += 1
                del batch, labels, batch_grads, batch_mean_grad, grad_block
                wrapper.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if n_seen == 0:
                raise RuntimeError(f"[mu-obs-asdl] slice {t} loader produced no batches")

            mu_parts: List[Tensor] = []
            for mi, name in enumerate(module_names):
                g_x_avg = g_x_parts[mi] / float(n_seen)
                mu_part = solve_xhat_from_grad(
                    module_R_lists[name][t].to(device=device, dtype=dtype),
                    g_x_avg,
                )
                mu_parts.append(mu_part)
            mu_global_list.append(torch.cat(mu_parts, dim=0).cpu())

    return mu_global_list

