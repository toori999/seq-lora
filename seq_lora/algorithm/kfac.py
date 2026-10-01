from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Tuple
import inspect
import sys
import zlib

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.logits import compute_choice_logits, mask_invalid_choices
from seq_lora.algorithm.asdl_backend import (
    add_kron_blocks as asdl_add_kron_blocks,
    exact_classification_kron_blocks as asdl_exact_classification_kron_blocks,
)


def forward_call_for_kfac_factory(
    amp_dtype: torch.dtype,
    choice_token_ids: Tensor,
    *,
    apply_choice_mask: bool,
):
    def _forward_call(model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        logits = compute_choice_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            amp_dtype=amp_dtype,
            choice_token_ids=choice_token_ids,
        )
        if apply_choice_mask:
            return mask_invalid_choices(logits, num_choices)
        return logits

    return _forward_call


class AsdlForwardWrapper(nn.Module):
    """Expose choice-logit forward as an nn.Module for ASDL."""

    def __init__(self, peft_model: nn.Module, forward_call, *, force_fp32_output: bool = True):
        super().__init__()
        self.peft_model = peft_model
        closure = inspect.getclosurevars(forward_call).nonlocals
        self.amp_dtype = closure.get("amp_dtype", torch.float16)
        self.choice_token_ids = closure.get("choice_token_ids")
        self.apply_choice_mask = bool(closure.get("apply_choice_mask", False))
        self.force_fp32_output = bool(force_fp32_output)

    @property
    def device(self) -> torch.device:
        return next(self.peft_model.parameters()).device

    def forward(self, **batch) -> Tensor:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        num_choices = batch.get("num_choices")

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.device.type == "cuda"),
        ):
            out = self.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]
            if self.choice_token_ids is not None and int(torch.max(self.choice_token_ids).item()) < logits.size(-1):
                logits = logits.index_select(-1, self.choice_token_ids.to(self.device))

        if self.force_fp32_output:
            logits = logits.to(torch.float32)
        if self.apply_choice_mask:
            logits = mask_invalid_choices(logits, num_choices)
        return logits


@contextmanager
def temporarily_select_lora_a_weights(model: nn.Module):
    saved = []
    for name, param in model.named_parameters():
        saved.append((param, bool(param.requires_grad)))
        param.requires_grad = bool("lora_A." in name and name.endswith(".weight"))
    try:
        yield
    finally:
        for param, req_grad in saved:
            param.requires_grad = req_grad


def set_dropout_modules_training(model: nn.Module, enabled: bool) -> None:
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train(enabled)


def effective_loader_num_samples(loader: DataLoader) -> int:
    if getattr(loader, "drop_last", False):
        batch_size = getattr(loader, "batch_size", None)
        if batch_size is not None:
            effective_n = len(loader) * int(batch_size)
            if effective_n > 0:
                return effective_n
    if hasattr(loader, "dataset"):
        effective_n = len(loader.dataset)
        if effective_n > 0:
            return effective_n
    raise ValueError("ASDL Kron extraction requires a non-empty effective sample set.")


def has_trainable_local_weight(module: nn.Module) -> Tuple[bool, bool]:
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


def iter_weight_block_module_names(wrapper: nn.Module) -> Iterable[str]:
    for name, module in wrapper.named_modules():
        if not name:
            continue
        has_weight, has_bias = has_trainable_local_weight(module)
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


def symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.T)


def randomized_psd_low_rank_factor(
    matrix: Tensor,
    *,
    target_rank: int,
    tag: str,
    oversample: int = 8,
    n_power_iters: int = 2,
) -> Tensor:
    """Return F such that matrix ~= F F^T using randomized subspace iteration."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square PSD matrix, got shape={tuple(matrix.shape)}")

    side = int(matrix.shape[0])
    rank = min(int(target_rank), side)
    if rank <= 0:
        raise ValueError(f"target_rank must be positive, got {target_rank}")
    if rank >= side:
        return symmetrize(matrix)

    sketch_dim = min(side, rank + max(int(oversample), 4))
    work_dtype = matrix.dtype if matrix.dtype in {torch.float32, torch.float64} else torch.float32
    work_matrix = symmetrize(matrix.detach().to(dtype=work_dtype))

    base_seed = int(torch.initial_seed())
    tag_seed = int(zlib.crc32(tag.encode("utf-8")))
    omega_gen = torch.Generator(device="cpu")
    omega_gen.manual_seed((base_seed + tag_seed) % (2 ** 63 - 1))
    omega = torch.randn((side, sketch_dim), generator=omega_gen, dtype=work_dtype).to(
        device=work_matrix.device,
        non_blocking=True,
    )

    Q = torch.linalg.qr(work_matrix @ omega, mode="reduced").Q
    for _ in range(max(int(n_power_iters), 0)):
        Q = torch.linalg.qr(work_matrix @ Q, mode="reduced").Q

    B = symmetrize(Q.T @ work_matrix @ Q)
    evals, evecs = torch.linalg.eigh(B)
    evals = evals[-rank:].clamp_min(0.0)
    evecs = evecs[:, -rank:]
    U = Q @ evecs

    if evals.numel() == 0:
        return torch.zeros((side, 0), device=matrix.device, dtype=work_dtype)

    positive = evals > 0
    if not torch.any(positive):
        return torch.zeros((side, 1), device=matrix.device, dtype=work_dtype)

    evals = evals[positive]
    U = U[:, positive]
    factor = U * torch.sqrt(evals).unsqueeze(0)
    return factor.to(device=matrix.device, dtype=work_dtype).contiguous()


def compress_asdl_psd_factor(
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
        return symmetrize(factor.detach())

    return randomized_psd_low_rank_factor(
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
    disable_dropout: bool = False,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """ASDL Kron extraction over the full wrapper graph."""
    del target_module_keywords, exclude_bias

    device = next(model.parameters()).device
    n_samples = effective_loader_num_samples(loader)
    if n_samples <= 0:
        raise ValueError("ASDL Kron extraction requires a non-empty effective sample set.")

    wrapper = AsdlForwardWrapper(model, forward_call).to(device)
    wrapper.train()
    if disable_dropout:
        set_dropout_modules_training(wrapper, enabled=False)

    with temporarily_select_lora_a_weights(model):
        module_names = list(iter_weight_block_module_names(wrapper))
        if not module_names:
            raise RuntimeError("No trainable LoRA-A weight blocks found for ASDL Kron extraction.")

        kron_total = None

        batch_iter = tqdm(loader, disable=not use_tqdm, file=sys.stdout)
        for batch in batch_iter:
            batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            wrapper.zero_grad(set_to_none=True)
            loss_batch, kron_batch, _ = asdl_exact_classification_kron_blocks(
                wrapper,
                batch,
                total_n=n_samples,
            )
            kron_total = asdl_add_kron_blocks(kron_total, kron_batch)
            del batch, loss_batch, kron_batch
            wrapper.zero_grad(set_to_none=True)

        if kron_total is None:
            raise RuntimeError("ASDL Kron extraction produced no factors.")
        if len(kron_total) != len(module_names):
            raise RuntimeError(
                f"ASDL Kron block count mismatch: got {len(kron_total)} blocks "
                f"for {len(module_names)} LoRA-A modules."
            )

        factors: Dict[str, Tuple[Tensor, Tensor]] = {}
        for module_name, block in zip(module_names, kron_total):
            if len(block) != 2:
                raise RuntimeError(
                    f"Expected a 2-factor Kron block for {module_name}, got {len(block)} factors."
                )
            factors[module_name] = (
                compress_asdl_psd_factor(
                    block[1],
                    n_kfac=n_kfac,
                    lr_threshold=lr_threshold,
                    tag=f"{module_name}:H",
                ),
                compress_asdl_psd_factor(
                    block[0],
                    n_kfac=n_kfac,
                    lr_threshold=lr_threshold,
                    tag=f"{module_name}:G",
                ),
            )

        return factors

