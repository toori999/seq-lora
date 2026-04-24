from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import kfac as hook_kfac
import seq_eval_iid_constantq as base

Tensor = torch.Tensor


def _select_token_rows(tensor: Tensor) -> Tensor:
    if tensor.dim() == 3:
        bsz, seq_len, width = tensor.shape
        if str(getattr(hook_kfac, "_CURRENT_TOKEN_MODE", "last")) == "all_valid":
            token_mask = getattr(hook_kfac, "_CURRENT_TOKEN_MASK", None)
            if token_mask is None:
                return tensor.reshape(bsz * seq_len, width)
            token_mask = token_mask.to(device=tensor.device, dtype=torch.bool)
            if token_mask.shape != (bsz, seq_len):
                raise RuntimeError(
                    f"Token mask shape mismatch: expected {(bsz, seq_len)}, got {tuple(token_mask.shape)}"
                )
            rows = tensor[token_mask]
            return rows if rows.numel() > 0 else tensor[:, -1, :]

        idx = getattr(hook_kfac, "_CURRENT_LAST_IDX", None)
        if idx is None:
            return tensor[:, -1, :]
        idx = idx.to(device=tensor.device, dtype=torch.long).clamp(min=0, max=seq_len - 1)
        return tensor[torch.arange(bsz, device=tensor.device), idx, :]

    if tensor.dim() == 2:
        return tensor

    raise RuntimeError(f"Unexpected tensor dim for selected-token rows: {tuple(tensor.shape)}")


@dataclass
class _ModuleGradState:
    weight_shape: torch.Size
    inputs: List[Tensor]
    grad_outputs: List[Tensor]


class _SelectedTokenWeightGradCapture:
    def __init__(self, model: nn.Module, module_names: List[str]):
        self._states: Dict[str, _ModuleGradState] = {}
        self._handles = []

        for module_name in module_names:
            module = model.get_submodule(module_name)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(
                    f"Selected-token local grad capture currently expects nn.Linear, got {type(module)} for {module_name}"
                )
            self._states[module_name] = _ModuleGradState(
                weight_shape=module.weight.shape,
                inputs=[],
                grad_outputs=[],
            )
            self._handles.append(module.register_forward_pre_hook(self._make_input_hook(module_name)))
            self._handles.append(module.register_full_backward_hook(self._make_grad_hook(module_name)))

    def _make_input_hook(self, module_name: str):
        def _hook(_module: nn.Module, pos_args: tuple[Tensor, ...]) -> None:
            if not pos_args:
                raise RuntimeError(f"Missing forward inputs for module {module_name}")
            self._states[module_name].inputs.append(_select_token_rows(pos_args[0].detach()))

        return _hook

    def _make_grad_hook(self, module_name: str):
        def _hook(_module: nn.Module, _grad_input, grad_output: tuple[Tensor, ...]) -> None:
            if not grad_output:
                raise RuntimeError(f"Missing backward outputs for module {module_name}")
            self._states[module_name].grad_outputs.append(_select_token_rows(grad_output[0].detach()))

        return _hook

    def take_weight_grads(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        weight_grads: Dict[str, Tensor] = {}
        for module_name, state in self._states.items():
            grad_weight = torch.zeros(state.weight_shape, device=device, dtype=dtype)
            if len(state.inputs) != len(state.grad_outputs):
                raise RuntimeError(
                    f"Hook event mismatch for {module_name}: "
                    f"{len(state.inputs)} inputs vs {len(state.grad_outputs)} grad outputs"
                )
            for a_rows, s_rows in zip(state.inputs, state.grad_outputs):
                if a_rows.shape[0] != s_rows.shape[0]:
                    raise RuntimeError(
                        f"Selected-row mismatch for {module_name}: "
                        f"inputs={tuple(a_rows.shape)} grad_outputs={tuple(s_rows.shape)}"
                    )
                a_rows = a_rows.to(device=device, dtype=dtype)
                s_rows = s_rows.to(device=device, dtype=dtype)
                grad_weight.addmm_(s_rows.T, a_rows, beta=1.0, alpha=1.0)
            state.inputs.clear()
            state.grad_outputs.clear()
            weight_grads[module_name] = grad_weight
        return weight_grads

    def remove(self) -> None:
        while self._handles:
            self._handles.pop().remove()


def estimate_mu_global_list_from_slice_local_weight_grads(
    model: nn.Module,
    slice_loaders: List[base.DataLoader],
    forward_call_for_kfac,
    module_names: List[str],
    module_subspace_info: Dict[str, Dict[str, Tensor]],
    module_R_lists: Dict[str, List[Tensor]],
    device: torch.device,
    n_batches_per_slice: int = 1,
    dtype: torch.dtype = torch.float64,
) -> List[Tensor]:
    model.train()
    mu_global_list: List[Tensor] = []
    grad_capture = _SelectedTokenWeightGradCapture(model, module_names)

    try:
        for t, loader in enumerate(slice_loaders):
            g_x_parts = [
                torch.zeros(int(module_subspace_info[name]["U_lora"].shape[1]), device=device, dtype=dtype)
                for name in module_names
            ]
            n_seen = 0

            for batch in loader:
                if n_seen >= n_batches_per_slice:
                    break

                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    forward_call_for_kfac(model, batch),
                    batch["labels"].to(device=device, non_blocking=True),
                )
                loss.backward()

                local_weight_grads = grad_capture.take_weight_grads(device=device, dtype=dtype)
                for mi, name in enumerate(module_names):
                    g_x_parts[mi] += (
                        module_subspace_info[name]["U_lora"].to(device=device, dtype=dtype).T
                        @ local_weight_grads[name].reshape(-1)
                    )
                n_seen += 1

            if n_seen == 0:
                raise RuntimeError(f"[mu-obs-local] slice {t} loader produced no batches")

            mu_parts: List[Tensor] = []
            for mi, name in enumerate(module_names):
                g_x_avg = g_x_parts[mi] / float(n_seen)
                mu_part = base.solve_xhat_from_grad(
                    module_R_lists[name][t].to(device=device, dtype=dtype),
                    g_x_avg,
                )
                mu_parts.append(mu_part)
            mu_global_list.append(torch.cat(mu_parts, dim=0).cpu())

        return mu_global_list
    finally:
        grad_capture.remove()


def main() -> None:
    base.KFAC_BACKEND = "hook"
    base.KFAC_TOKEN_MODE = "last"
    base.estimate_mu_global_list_from_slice_grads = estimate_mu_global_list_from_slice_local_weight_grads
    base.main()


if __name__ == "__main__":
    main()
