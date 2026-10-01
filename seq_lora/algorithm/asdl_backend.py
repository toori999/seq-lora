from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asdl.core import extend
from asdl.fisher import FisherConfig, get_fisher_maker
from asdl.grad_maker import LOSS_CROSS_ENTROPY
from asdl.matrices import FISHER_EXACT, SHAPE_KRON
from asdl.operations import OP_BATCH_GRADS
import asdl.operations.linear as asdl_linear_ops

Tensor = torch.Tensor
KronBlock = List[Tensor]


def patch_asdl_linear_batch_grads_weight_dtype() -> None:
    current = getattr(asdl_linear_ops.Linear.batch_grads_weight, "__name__", "")
    if current == "_seq_lora_safe_batch_grads_weight":
        return

    def _seq_lora_safe_batch_grads_weight(
        module: nn.Module,
        in_data: Tensor,
        out_grads: Tensor,
    ) -> Tensor:
        del module
        if in_data.dtype != out_grads.dtype:
            common_dtype = torch.promote_types(in_data.dtype, out_grads.dtype)
            in_data = in_data.to(common_dtype)
            out_grads = out_grads.to(common_dtype)
        return torch.bmm(
            out_grads.unsqueeze(2),
            in_data.unsqueeze(1),
        )

    asdl_linear_ops.Linear.batch_grads_weight = staticmethod(_seq_lora_safe_batch_grads_weight)


patch_asdl_linear_batch_grads_weight_dtype()


def batch_gradient(
    model: nn.Module,
    closure: Callable[[], object],
    input_shape: Tuple[int, ...],
    *,
    return_outputs: bool = False,
):
    """Sequence-aware per-example gradient wrapper around ASDL batch grads.

    ASDL exposes per-token batch gradients for sequence-shaped inputs. Seq-LoRA
    needs one gradient vector per example, so sequence dimensions are summed
    before concatenating module-local gradients.
    """
    with extend(model, OP_BATCH_GRADS) as cxt:
        outputs = closure()
        grads = []
        batch_size = int(input_shape[0])
        seq_len = int(input_shape[-1])
        for module in model.modules():
            g = cxt.batch_grads(module, flatten=True)
            if g is None:
                continue
            if len(input_shape) == 2:
                if g.shape[0] > batch_size:
                    if g.shape[0] % batch_size != 0:
                        raise RuntimeError(
                            f"Per-example gradient shape mismatch for module "
                            f"{module.__class__.__name__}: leading dim {g.shape[0]} "
                            f"is not divisible by batch size {batch_size}."
                        )
                    seq_steps = g.shape[0] // batch_size
                    grads.append(g.reshape(batch_size, seq_steps, -1).sum(-2))
                else:
                    grads.append(g)
            else:
                if g.shape[0] > batch_size * seq_len:
                    grads.append(g.reshape(*input_shape, -1).sum(-2).sum(-2))
                elif g.shape[0] > batch_size:
                    grads.append(g.reshape(*input_shape[:-1], -1).sum(-2))
                else:
                    grads.append(g)
        grads_out = torch.cat(grads, dim=-1)
    if return_outputs:
        return grads_out, outputs
    return grads_out


def _trainable_local_param_flags(module: nn.Module) -> Tuple[bool, bool]:
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


def _module_kron_blocks(model: nn.Module, total_n: int) -> List[KronBlock]:
    blocks: List[KronBlock] = []
    for module in model.modules():
        stats = getattr(module, "fisher", None)
        if stats is None:
            continue
        has_weight, has_bias = _trainable_local_param_flags(module)
        if not has_weight and not has_bias:
            continue

        kron_stats = getattr(stats, "kron", None)
        if kron_stats is None:
            raise RuntimeError(f"Module {module.__class__.__name__} is missing ASDL Kron statistics.")
        if has_weight:
            if not hasattr(module, "weight") or module.weight is None:
                raise ValueError(f"Module {module} has trainable weight but no weight attribute.")
            p = int(kron_stats.B.numel())
            q = int(kron_stats.A.numel())
            if p == q == 1:
                blocks.append([(kron_stats.B * kron_stats.A).detach().clone()])
            else:
                blocks.append([
                    kron_stats.B.detach().clone(),
                    kron_stats.A.detach().clone() / float(total_n),
                ])
        if has_bias:
            if not hasattr(module, "bias") or module.bias is None:
                raise ValueError(f"Module {module} has trainable bias but no bias attribute.")
            blocks.append([kron_stats.B.detach().clone()])
    return blocks


def exact_classification_kron_blocks(
    model: nn.Module,
    batch: dict,
    *,
    total_n: int,
) -> Tuple[Tensor, List[KronBlock], Tensor]:
    """Extract exact-GGN Kron blocks with the official ASDL Fisher maker."""
    labels = batch["labels"]
    cfg = FisherConfig(
        fisher_type=FISHER_EXACT,
        loss_type=LOSS_CROSS_ENTROPY,
        fisher_shapes=[SHAPE_KRON],
        data_size=1,
    )
    fisher_maker = get_fisher_maker(model, cfg)
    fisher_maker.setup_model_call(model, **batch)
    outputs, _ = fisher_maker.forward_and_backward()
    loss = F.cross_entropy(outputs.detach().float(), labels)
    return loss, _module_kron_blocks(model, total_n=total_n), outputs.detach()


def add_kron_blocks(
    total: Optional[List[KronBlock]],
    batch_blocks: List[KronBlock],
) -> List[KronBlock]:
    if total is None:
        return [[factor.detach().clone() for factor in block] for block in batch_blocks]
    if len(total) != len(batch_blocks):
        raise RuntimeError(
            f"ASDL Kron block count mismatch while accumulating: "
            f"total={len(total)} batch={len(batch_blocks)}."
        )
    for total_block, batch_block in zip(total, batch_blocks):
        if len(total_block) != len(batch_block):
            raise RuntimeError(
                f"ASDL Kron factor count mismatch: total={len(total_block)} batch={len(batch_block)}."
            )
        for idx, factor in enumerate(batch_block):
            if tuple(total_block[idx].shape) != tuple(factor.shape):
                raise RuntimeError(
                    f"ASDL Kron factor shape mismatch: "
                    f"total={tuple(total_block[idx].shape)} batch={tuple(factor.shape)}."
                )
            total_block[idx] = total_block[idx] + factor
    return total
