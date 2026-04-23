from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Tuple
import inspect
import sys
import zlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from laplace.curvature.asdl import AsdlGGN

import seq_eval_iid_constantq as base


Tensor = torch.Tensor


def forward_call_for_kfac_factory(amp_dtype: torch.dtype):
    def _forward_call(model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        logits = base.compute_choice_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            amp_dtype=amp_dtype,
        )
        return base._mask_invalid_choices(logits, num_choices)

    return _forward_call


class _AsdlForwardWrapper(nn.Module):
    def __init__(self, peft_model: nn.Module, forward_call):
        super().__init__()
        self.peft_model = peft_model
        closure = inspect.getclosurevars(forward_call).nonlocals
        self.amp_dtype = closure.get("amp_dtype", torch.float16)

    @property
    def device(self) -> torch.device:
        return next(self.peft_model.parameters()).device

    def forward(self, **batch) -> Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        num_choices = batch.get("num_choices")

        logits = base.compute_choice_logits(
            model=self.peft_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            amp_dtype=self.amp_dtype,
        ).to(torch.float32)
        return base._mask_invalid_choices(logits, num_choices)


@contextmanager
def _temporarily_select_lora_a_weights(model: nn.Module):
    saved = []
    for name, param in model.named_parameters():
        saved.append((param, bool(param.requires_grad)))
        param.requires_grad = bool("lora_A." in name and name.endswith(".weight"))
    try:
        yield
    finally:
        for param, req_grad in saved:
            param.requires_grad = req_grad


def _has_trainable_local_weight(module: nn.Module) -> Tuple[bool, bool]:
    local_params = {
        name: param
        for name, param in module.named_parameters(recurse=False)
        if param.requires_grad
    }
    if not local_params:
        return False, False

    unsupported = [name for name in local_params if name not in {"weight", "bias"}]
    if unsupported:
        raise ValueError(
            f"Unsupported trainable local parameters for ASDL Kron extraction in "
            f"{module.__class__.__name__}: {unsupported}"
        )
    return ("weight" in local_params), ("bias" in local_params)


def _iter_weight_block_module_names(wrapper: nn.Module) -> Iterable[str]:
    for name, module in wrapper.named_modules():
        if not name:
            continue
        has_weight, has_bias = _has_trainable_local_weight(module)
        if not has_weight and not has_bias:
            continue

        normalized_name = name
        if normalized_name.startswith("peft_model."):
            normalized_name = normalized_name[len("peft_model."):]

        if has_weight:
            yield normalized_name
        if has_bias:
            raise RuntimeError(
                f"Unexpected trainable bias block in ASDL Kron extraction for module {normalized_name}. "
                "Seq-LoRA currently expects weight-only LoRA-A modules."
            )


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.T)


def _randomized_psd_low_rank_factor(
    matrix: Tensor,
    *,
    target_rank: int,
    tag: str,
    oversample: int = 8,
    n_power_iters: int = 2,
) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square PSD matrix, got shape={tuple(matrix.shape)}")

    side = int(matrix.shape[0])
    rank = min(int(target_rank), side)
    if rank <= 0:
        raise ValueError(f"target_rank must be positive, got {target_rank}")
    if rank >= side:
        return _symmetrize(matrix)

    sketch_dim = min(side, rank + max(int(oversample), 4))
    work_dtype = matrix.dtype if matrix.dtype in {torch.float32, torch.float64} else torch.float32
    work_matrix = _symmetrize(matrix.detach().to(dtype=work_dtype))

    base_seed = int(torch.initial_seed())
    tag_seed = int(zlib.crc32(tag.encode("utf-8")))
    omega_gen = torch.Generator(device="cpu")
    omega_gen.manual_seed((base_seed + tag_seed) % (2 ** 63 - 1))
    omega = torch.randn((side, sketch_dim), generator=omega_gen, dtype=work_dtype).to(
        device=work_matrix.device,
        non_blocking=True,
    )

    q_mat = torch.linalg.qr(work_matrix @ omega, mode="reduced").Q
    for _ in range(max(int(n_power_iters), 0)):
        q_mat = torch.linalg.qr(work_matrix @ q_mat, mode="reduced").Q

    small = _symmetrize(q_mat.T @ work_matrix @ q_mat)
    evals, evecs = torch.linalg.eigh(small)
    evals = evals[-rank:].clamp_min(0.0)
    evecs = evecs[:, -rank:]
    basis = q_mat @ evecs

    if evals.numel() == 0:
        return torch.zeros((side, 0), device=matrix.device, dtype=work_dtype)

    positive = evals > 0
    if not torch.any(positive):
        return torch.zeros((side, 1), device=matrix.device, dtype=work_dtype)

    evals = evals[positive]
    basis = basis[:, positive]
    factor = basis * torch.sqrt(evals).unsqueeze(0)
    return factor.to(device=matrix.device, dtype=work_dtype).contiguous()


def _compress_asdl_psd_factor(
    factor: Tensor,
    *,
    n_kfac: int | None,
    lr_threshold: int,
    tag: str,
) -> Tensor:
    if factor.ndim != 2:
        raise ValueError(f"Expected a 2D ASDL factor, got shape={tuple(factor.shape)}")

    side = int(factor.shape[0])
    if factor.shape[0] != factor.shape[1]:
        return factor.detach()
    if n_kfac is None or side < int(lr_threshold) or int(n_kfac) >= side:
        return _symmetrize(factor.detach())

    return _randomized_psd_low_rank_factor(
        factor,
        target_rank=int(n_kfac),
        tag=tag,
    )


def calculate_kronecker_factors(
    model: nn.Module,
    forward_call,
    loader: DataLoader,
    n_kfac: int | None = None,
    lr_threshold: int = 512,
    target_module_keywords: list[str] | None = None,
    exclude_bias: bool = False,
    use_tqdm: bool = False,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    del target_module_keywords, exclude_bias

    if not hasattr(loader, "dataset"):
        raise ValueError("ASDL Kron extraction requires loader.dataset to infer N.")

    device = next(model.parameters()).device
    total_examples = len(loader.dataset)
    if total_examples <= 0:
        raise ValueError("ASDL Kron extraction requires a non-empty loader.dataset.")

    wrapper = _AsdlForwardWrapper(model, forward_call).to(device)
    wrapper.eval()

    with _temporarily_select_lora_a_weights(model):
        module_names = list(_iter_weight_block_module_names(wrapper))
        if not module_names:
            raise RuntimeError("No trainable LoRA-A weight blocks found for ASDL Kron extraction.")

        backend = AsdlGGN(wrapper, likelihood="classification", last_layer=False)
        kron_total = None

        batch_iter = tqdm(loader, disable=not use_tqdm, file=sys.stdout)
        for batch in batch_iter:
            batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            wrapper.zero_grad(set_to_none=True)
            loss_batch, kron_batch, _ = backend.kron(batch, N=total_examples)
            kron_total = kron_batch if kron_total is None else (kron_total + kron_batch)
            del loss_batch, kron_batch

        if kron_total is None:
            raise RuntimeError("ASDL Kron extraction produced no factors.")
        if len(kron_total.kfacs) != len(module_names):
            raise RuntimeError(
                f"ASDL Kron block count mismatch: got {len(kron_total.kfacs)} blocks "
                f"for {len(module_names)} LoRA-A modules."
            )

        factors: Dict[str, Tuple[Tensor, Tensor]] = {}
        for module_name, block in zip(module_names, kron_total.kfacs):
            if len(block) != 2:
                raise RuntimeError(
                    f"Expected a 2-factor Kron block for {module_name}, got {len(block)} factors."
                )
            factors[module_name] = (
                _compress_asdl_psd_factor(
                    block[1],
                    n_kfac=n_kfac,
                    lr_threshold=lr_threshold,
                    tag=f"{module_name}:H",
                ),
                _compress_asdl_psd_factor(
                    block[0],
                    n_kfac=n_kfac,
                    lr_threshold=lr_threshold,
                    tag=f"{module_name}:G",
                ),
            )

        return factors


def main():
    base.forward_call_for_kfac_factory = forward_call_for_kfac_factory
    base.calculate_kronecker_factors = calculate_kronecker_factors
    base.main()


if __name__ == "__main__":
    main()
