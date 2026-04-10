from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


def set_seed(seed: int, fast_but_nondeterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast_but_nondeterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device_amp_dtype() -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    return device, amp_dtype


def enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def build_training_tokenizer(
    base_model_name: str,
    *,
    trust_remote_code: bool = False,
    padding_side: str = "left",
):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
        )
    tokenizer.padding_side = padding_side
    return tokenizer


def load_causal_lm_with_attn_fallback(
    *,
    base_model_name: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    trust_remote_code: bool = False,
    attn_implementation: str = "sdpa",
    fallback_attn_implementation: str = "sdpa",
):
    load_kwargs = dict(
        pretrained_model_name_or_path=base_model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
    )
    attn_impl_used = fallback_attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            **load_kwargs,
            attn_implementation=attn_implementation,
        ).to(device)
        attn_impl_used = attn_implementation
    except Exception as exc:
        print(
            f"[Model] attn_implementation={attn_implementation} unavailable, "
            f"falling back to {fallback_attn_implementation}: {exc}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            **load_kwargs,
            attn_implementation=fallback_attn_implementation,
        ).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, attn_impl_used


__all__ = [
    "build_training_tokenizer",
    "enable_gradient_checkpointing",
    "load_causal_lm_with_attn_fallback",
    "resolve_device_amp_dtype",
    "seed_worker",
    "set_seed",
]
