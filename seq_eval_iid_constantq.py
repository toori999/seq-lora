from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Dict, Iterable, List, Tuple, Optional, Sequence
import inspect
import os
import random
import math
import time
import argparse
import gc
import sys
import zlib
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets as hf_datasets
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from tqdm import tqdm

from laplace.curvature.asdl import AsdlGGN, batch_gradient as asdl_batch_gradient
import asdl.operations.linear as asdl_linear_ops
from lssm_ffbs_obs import kalman_filter
from seq_lora_subspace_obs import (
    build_global_kronecker_eigenspace,
    materialize_mean_psd_from_factors,
    project_curvature_factors_to_subspace,
    prepare_lgssm_observations,
    solve_xhat_from_grad,
)

try:
    from torchmetrics import Accuracy, CalibrationError
except Exception:
    Accuracy = None
    CalibrationError = None
    from torchmetrics.classification import MulticlassAccuracy as _MulticlassAccuracy
    from torchmetrics.classification import MulticlassCalibrationError as _MulticlassCalibrationError

Tensor = torch.Tensor

warnings.filterwarnings(
    "ignore",
    message=(
        "Using a non-full backward hook when the forward contains multiple autograd "
        "Nodes is deprecated and will be removed in future versions.*"
    ),
    category=FutureWarning,
)


def _patch_asdl_linear_batch_grads_weight_dtype() -> None:
    current = getattr(asdl_linear_ops.Linear.batch_grads_weight, "__name__", "")
    if current == "_seq_lora_safe_batch_grads_weight":
        return

    def _seq_lora_safe_batch_grads_weight(
        module: nn.Module,
        in_data: torch.Tensor,
        out_grads: torch.Tensor,
    ):
        if in_data.dtype != out_grads.dtype:
            common_dtype = torch.promote_types(in_data.dtype, out_grads.dtype)
            in_data = in_data.to(common_dtype)
            out_grads = out_grads.to(common_dtype)
        return torch.bmm(
            out_grads.unsqueeze(2),
            in_data.unsqueeze(1),
        )

    asdl_linear_ops.Linear.batch_grads_weight = staticmethod(_seq_lora_safe_batch_grads_weight)


_patch_asdl_linear_batch_grads_weight_dtype()

_POSTERIOR_STATS_CACHE_FORMAT = "seq_lora_constantq_posterior_stats_v1"


def _cache_norm_path(path: str) -> str:
    path = str(path or "").strip()
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def _serialize_module_specs_cpu(module_specs: List[Dict]) -> List[Dict[str, object]]:
    specs_cpu: List[Dict[str, object]] = []
    for spec in module_specs:
        subspace_info_cpu = {}
        for key, value in spec["subspace_info"].items():
            if torch.is_tensor(value):
                subspace_info_cpu[key] = value.detach().cpu()
            else:
                subspace_info_cpu[key] = value
        specs_cpu.append(
            {
                "name": str(spec["name"]),
                "offset": int(spec["offset"]),
                "L": int(spec["L"]),
                "subspace_info": subspace_info_cpu,
            }
        )
    return specs_cpu


def _build_posterior_stats_cache_snapshot(
    args: argparse.Namespace,
    *,
    slice_ids: Sequence[int],
    train_raw_fingerprint: str,
    train_proc_fingerprint: str,
) -> Dict[str, object]:
    return {
        "task": str(args.task),
        "map_dir": _cache_norm_path(str(args.map_dir)),
        "slices_dir": _cache_norm_path(str(args.slices_dir)),
        "seed": int(args.seed),
        "kfac_backend": str(args.kfac_backend),
        "random_num_slices": int(args.random_num_slices),
        "slice_order": str(args.slice_order),
        "slice_order_seed": int(args.slice_order_seed),
        "slice_ids": [int(x) for x in slice_ids],
        "train_raw_fingerprint": str(train_raw_fingerprint),
        "train_proc_fingerprint": str(train_proc_fingerprint),
        "subspace_dim_per_module": int(args.subspace_dim_per_module),
        "max_seq_len": int(args.max_seq_len),
        "kfac_bsz": int(args.kfac_bsz),
        "n_kfac": int(args.n_kfac),
        "lr_threshold": int(args.lr_threshold),
        "max_kfac_samples_per_slice": int(args.max_kfac_samples_per_slice),
        "kfac_tail_policy": "keep",
        "mu_obs_batches": int(args.mu_obs_batches),
        "disable_dropout_during_kfac_mu": bool(args.disable_dropout_during_kfac_mu),
        "tokenizer_padding_side": str(args.tokenizer_padding_side),
        "keep_full_vocab_lm_head": bool(args.keep_full_vocab_lm_head),
    }


def _build_posterior_stats_cache_payload(
    *,
    args: argparse.Namespace,
    snapshot: Dict[str, object],
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    T: int,
) -> Dict[str, object]:
    return {
        "format": _POSTERIOR_STATS_CACHE_FORMAT,
        "args_snapshot": dict(snapshot),
        "module_specs": _serialize_module_specs_cpu(module_specs),
        "module_R_lists": {
            str(name): [tensor.detach().cpu() for tensor in tensors]
            for name, tensors in module_R_lists.items()
        },
        "mu_global_list_raw": [mu_t.detach().cpu() for mu_t in mu_global_list_raw],
        "T": int(T),
        "num_modules": int(len(module_specs)),
        "L_total": int(sum(int(spec["L"]) for spec in module_specs)),
        "created_by": os.path.basename(__file__),
        "seed": int(args.seed),
    }


def _validate_cache_snapshot(
    *,
    cache_path: str,
    snapshot: Dict[str, object],
    expected: Dict[str, object],
) -> None:
    mismatches = [
        f"{key}: cache={snapshot.get(key)!r} current={expected[key]!r}"
        for key in expected
        if snapshot.get(key) != expected[key]
    ]
    if mismatches:
        mismatch_text = "\n".join(mismatches[:16])
        raise RuntimeError(
            f"Posterior-stats cache at {cache_path} does not match this run:\n"
            f"{mismatch_text}\n"
            "Use a matching cache path or pass --force_rebuild_posterior_stats_cache."
        )


def _load_posterior_stats_cache(
    cache_path: str,
    expected_snapshot: Dict[str, object],
) -> Tuple[List[Dict], Dict[str, List[Tensor]], List[Tensor], int]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != _POSTERIOR_STATS_CACHE_FORMAT:
        raise RuntimeError(
            f"Unsupported posterior-stats cache format in {cache_path}. "
            f"Expected format={_POSTERIOR_STATS_CACHE_FORMAT!r}."
        )

    snapshot = dict(payload.get("args_snapshot") or {})
    _validate_cache_snapshot(
        cache_path=cache_path,
        snapshot=snapshot,
        expected=expected_snapshot,
    )

    module_specs = payload.get("module_specs")
    module_R_lists = payload.get("module_R_lists")
    mu_global_list_raw = payload.get("mu_global_list_raw")
    T = int(payload.get("T"))
    if (
        not isinstance(module_specs, list)
        or not isinstance(module_R_lists, dict)
        or not isinstance(mu_global_list_raw, list)
    ):
        raise RuntimeError(f"Posterior-stats cache at {cache_path} is missing required tensors.")
    return module_specs, module_R_lists, mu_global_list_raw, T

# =========================
# Config Defaults
# =========================

SEED = 0
TRUST_REMOTE_CODE = True

MAX_SEQ_LEN = 300
EVAL_BSZ = 64
KFAC_BSZ = 4
SLICE_ORDER = "sorted"
SLICE_ORDER_SEED = 0

# KFAC / train-slice loaders remain conservative
NUM_WORKERS = 0

# Eval loader gets its own workers for dynamic padding pipeline
EVAL_NUM_WORKERS = 0
EVAL_PREFETCH_FACTOR = 4

N_KFAC = 8
LR_THRESHOLD = 256
MAX_KFAC_SAMPLES_PER_SLICE = -1
KFAC_BACKEND = "asdl"

MU_OBS_SCALE = 2.0
MU_OBS_BATCHES = 32
S_Q = 1.0
P1_VAR = 1.0
Q_MODE = "module_constant"
MODULE_Q_CLIP_MIN = 0.5
MODULE_Q_CLIP_MAX = 2
GAP_Q_CLIP_MIN = 0.5
GAP_Q_CLIP_MAX = 2.0
MODULE_Q_SHRINK_EXPONENT = 0.05
GAP_Q_SHRINK_EXPONENT = 0.05

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 10
POSTERIOR_EVAL_MODE = "lgssm_final"
INDEPENDENT_SLICE_MC_SAMPLES_PER_SLICE = 3

POSTERIOR_TAU = 1
TEMP_BAYES = 1.05
DISABLE_DROPOUT_DURING_KFAC_MU = True
TAU_MODE = "fixed"
TAU_SEARCH_MAX = 1.0
TAU_SEARCH_ITERS = 6
TAU_ANCHOR_SIZE = 500
TAU_ANCHOR_BSZ = EVAL_BSZ
TAU_ANCHOR_N_SAMPLES = 32
TAU_ACC_TOLERANCE = 0.01
TAU_KL_TARGET_LOW = 0.05
TAU_KL_TARGET_HIGH = 0.0525

TOKENIZER_PADDING_SIDE = "left"
HF_DATASETS_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_datasets")

from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    DynamicEvalCollator,
    get_choice_token_ids,
    get_transformer_and_lm_head,
    get_task_num_classes,
    load_eval_dataset,
    load_iid_test_set,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    preprocess_task,
    set_inference_fast as _set_inference_fast,
)
from seq_eval_iid import (
    _assign_random_slice_ids,
)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def _reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_allocated())


def _peak_reserved_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mem_gb(torch.cuda.max_memory_reserved())

def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _forecast_from_final_posterior(
    x_T: Tensor,
    P_T: Tensor,
    Q_list: List[Tensor],
    horizon: int,
) -> Tuple[Tensor, Tensor]:
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if horizon == 0:
        return x_T, P_T
    if len(Q_list) == 0:
        raise ValueError("Q_list must be non-empty for forecasting")

    # Random-walk transition: x_{t+1} = x_t + u_{t+1}, so the predictive mean
    # stays at the final filtered mean. We do not have an explicit Q_{T+1},
    # so we roll forward with the last estimated process noise as a proxy.
    Q_ref = Q_list[-1].to(device=P_T.device, dtype=P_T.dtype)
    x_fore = x_T.clone()
    P_fore = P_T + float(horizon) * Q_ref
    return x_fore, 0.5 * (P_fore + P_fore.T)


def _multiclass_brier_score(probs: Tensor, labels: Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def _multiclass_brier_sum(probs: Tensor, labels: Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).sum().item())


def _mask_invalid_choices(logits: Tensor, num_choices: Optional[Sequence[int]]) -> Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


def _left_padded_last_idx(input_ids: Tensor) -> Tensor:
    return torch.full(
        (input_ids.size(0),),
        input_ids.size(1) - 1,
        device=input_ids.device,
        dtype=torch.long,
    )


def trim_lm_head_to_choice_tokens(model: nn.Module, choice_token_ids: Tensor) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    _, lm_head = get_transformer_and_lm_head(base)
    weight = lm_head.weight.index_select(0, choice_token_ids).detach()
    bias = (
        lm_head.bias.index_select(0, choice_token_ids).detach()
        if getattr(lm_head, "bias", None) is not None
        else None
    )

    new_head = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=(bias is not None),
        device=weight.device,
        dtype=weight.dtype,
    )
    new_head.weight.data.copy_(weight)
    if bias is not None:
        new_head.bias.data.copy_(bias)

    if hasattr(base, "lm_head"):
        base.lm_head = new_head
    else:
        raise RuntimeError("Could not locate lm_head on base model for trimming.")
    if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
        base.config.vocab_size = int(choice_token_ids.numel())


def compute_choice_logits(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    amp_dtype: torch.dtype,
    choice_token_ids: Optional[Tensor] = None,
) -> Tensor:
    device = input_ids.device
    transformer, lm_head = get_transformer_and_lm_head(model)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = lm_head(out.last_hidden_state[:, -1, :])
        if choice_token_ids is not None and int(torch.max(choice_token_ids).item()) < logits.size(-1):
            logits = logits.index_select(-1, choice_token_ids)
    return logits.to(torch.float32)


def _resolve_bayes_module_names(factors: Dict[str, Tuple[Tensor, Tensor]]) -> List[str]:
    return sorted([name for name in factors.keys() if "lora_A" in name])


def _ensure_slice_ids_for_seq(task: str, train_raw: Dataset) -> Dataset:
    if "slice_id" in train_raw.column_names:
        return train_raw
    if task == SCIENCEQA_CURRIC_TASK_NAME and "grade_num" in train_raw.column_names:
        grade_min = min(int(x) for x in train_raw["grade_num"])
        return train_raw.map(
            lambda ex: {"slice_id": int(ex["grade_num"]) - grade_min}
        )
    raise ValueError(
        "Seq-LoRA requires slice ids. Provide --slices_dir, or use a task whose "
        "training set already includes slice_id/grade_num metadata."
    )


def _canonicalize_proc_columns(ds: Dataset) -> Dataset:
    preferred = [
        "grade_num",
        "slice_id",
        "num_choices",
        "input_ids",
        "attention_mask",
        "labels",
        "seq_len",
        "source_subset",
    ]
    ordered = [c for c in preferred if c in ds.column_names]
    ordered.extend(c for c in ds.column_names if c not in set(ordered))
    if ds.column_names == ordered:
        return ds
    return ds.select_columns(ordered)

class _StageTimer:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0 = None

    def __enter__(self):
        _reset_cuda_peak()
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _cuda_sync()
        dt = time.perf_counter() - self.t0
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)", flush=True)
        print(f"[PEAK] {self.tag}: alloc={_peak_alloc_gb():.2f} GB  reserved={_peak_reserved_gb():.2f} GB", flush=True)


@contextmanager
def _temporarily_disable_dropout_modules(model: nn.Module):
    touched: List[Tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            touched.append((module, bool(module.training)))
            module.train(False)
    try:
        yield
    finally:
        for module, was_training in touched:
            module.train(was_training)


def _parse_eval_tasks(spec: str, default_task: str) -> List[str]:
    if not spec or not spec.strip():
        return [default_task]

    expanded: List[str] = []
    for raw in spec.split(","):
        task = raw.strip().lower()
        if not task:
            continue
        if task == "iid":
            expanded.append(default_task)
        elif task == "arc":
            expanded.extend(["arc-c", "arc-e"])
        elif task == "mmlu":
            expanded.extend(["mmlu_science_high", "mmlu_science_college"])
        else:
            expanded.append(task)

    out: List[str] = []
    seen = set()
    for task in expanded:
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out


# =========================
# KFAC forward
# =========================

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
            return _mask_invalid_choices(logits, num_choices)
        return logits

    return _forward_call


class _AsdlForwardWrapper(nn.Module):
    """Expose Laplace-style choice-logit forward as an nn.Module for ASDL."""

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
            logits = _mask_invalid_choices(logits, num_choices)
        return logits


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


def _set_dropout_modules_training(model: nn.Module, enabled: bool) -> None:
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


def _effective_loader_num_samples(loader: DataLoader) -> int:
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
    """Return F such that matrix ~= F F^T using randomized subspace iteration."""
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

    Q = torch.linalg.qr(work_matrix @ omega, mode="reduced").Q
    for _ in range(max(int(n_power_iters), 0)):
        Q = torch.linalg.qr(work_matrix @ Q, mode="reduced").Q

    B = _symmetrize(Q.T @ work_matrix @ Q)
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
    disable_dropout: bool = False,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """ASDL Kron extraction over the full wrapper graph."""
    del target_module_keywords, exclude_bias

    device = next(model.parameters()).device
    N = _effective_loader_num_samples(loader)
    if N <= 0:
        raise ValueError("ASDL Kron extraction requires a non-empty effective sample set.")

    wrapper = _AsdlForwardWrapper(model, forward_call).to(device)
    wrapper.train()
    if disable_dropout:
        _set_dropout_modules_training(wrapper, enabled=False)

    with _temporarily_select_lora_a_weights(model):
        module_names = list(_iter_weight_block_module_names(wrapper))
        if not module_names:
            raise RuntimeError("No trainable LoRA-A weight blocks found for ASDL Kron extraction.")

        backend = AsdlGGN(wrapper, likelihood="classification", last_layer=False, stochastic=False)
        kron_total = None

        batch_iter = tqdm(loader, disable=not use_tqdm, file=sys.stdout)
        for batch in batch_iter:
            batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            wrapper.zero_grad(set_to_none=True)
            loss_batch, kron_batch, _ = backend.kron(batch, N=N)
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

# =========================
# μ-observation & Math
# =========================

def _get_param_weight(model: nn.Module, module_path: str) -> nn.Parameter:
    m = model.get_submodule(module_path)
    if not hasattr(m, "weight"):
        raise RuntimeError(f"[mu-obs] submodule has no .weight: {module_path}")
    w = getattr(m, "weight")
    if not isinstance(w, nn.Parameter):
        raise RuntimeError(f"[mu-obs] .weight is not nn.Parameter: {module_path}")
    return w

@torch.no_grad()
def materialize_scalar_Q_list(
    var_list: Sequence[float],
    L: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    I = torch.eye(L, device=device, dtype=dtype)
    return [float(var) * I for var in var_list]


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
    wrapper = _AsdlForwardWrapper(
        model,
        forward_call_for_kfac,
        force_fp32_output=False,
    ).to(device)
    wrapper.train()
    if disable_dropout:
        _set_dropout_modules_training(wrapper, enabled=False)

    mu_global_list: List[Tensor] = []

    with _temporarily_select_lora_a_weights(model):
        asdl_module_names = list(_iter_weight_block_module_names(wrapper))
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
            numel = int(_get_param_weight(model, name).numel())
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


def _move_subspace_info(
    subspace_info: Dict[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Tensor]:
    moved: Dict[str, Tensor] = {}
    for key, value in subspace_info.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _report_scalar_constant_q_results(s_q: float) -> None:
    print("\n=== Scalar-Constant Q Report ===")
    print(f"[Constant Q] mode=constant  shared Q_t = s_Q * I with s_Q={float(s_q):.6f}")


def _normalize_and_clip_scales(
    values: Sequence[float],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float = 1.0,
) -> List[float]:
    values = [float(v) for v in values]
    positive = [v for v in values if math.isfinite(v) and v > 0.0]
    if not positive:
        return [1.0 for _ in values]
    denom = float(torch.tensor(positive, dtype=torch.float64).median().item())
    if not math.isfinite(denom) or denom <= 0.0:
        denom = sum(positive) / max(len(positive), 1)
    if not math.isfinite(denom) or denom <= 0.0:
        return [1.0 for _ in values]
    beta = float(min(max(float(shrink_exponent), 0.0), 1.0))
    scales = []
    for v in values:
        if math.isfinite(v) and v > 0.0:
            ratio = max(v / denom, 1e-12)
            scale = math.exp(beta * math.log(ratio))
        else:
            scale = 1.0
        scales.append(scale)
    return [float(min(max(s, clip_min), clip_max)) for s in scales]


def _median_positive_or_default(values: Tensor, default: float = 0.0) -> float:
    values = values.detach().reshape(-1).to(dtype=torch.float64)
    values = values[torch.isfinite(values) & (values > 0.0)]
    if int(values.numel()) == 0:
        return float(default)
    return float(values.median().item())


def _build_module_constant_q_scales(
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float,
) -> Dict[str, float]:
    if len(mu_global_list_raw) < 2:
        return {str(spec["name"]): 1.0 for spec in module_specs}

    raw_vals: List[float] = []
    names: List[str] = []
    for spec in module_specs:
        name = str(spec["name"])
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        R_list = module_R_lists.get(name, [])
        if not R_list:
            raw_vals.append(1.0)
            names.append(name)
            continue

        curvature_diag = torch.zeros(Lm, dtype=torch.float64)
        n_curv = 0
        for R_t in R_list:
            R_cpu = R_t.detach().to(device=curvature_diag.device, dtype=torch.float64)
            if tuple(R_cpu.shape) != (Lm, Lm):
                raise ValueError(
                    f"R_list for module {name} has shape {tuple(R_cpu.shape)}, expected {(Lm, Lm)}"
                )
            curvature_diag.add_(R_cpu.pow(2).sum(dim=0))
            n_curv += 1
        curvature_diag.div_(max(n_curv, 1)).clamp_min_(0.0)

        gap_vals: List[float] = []
        for t in range(1, len(mu_global_list_raw)):
            delta = (
                mu_global_list_raw[t][offset : offset + Lm].to(dtype=torch.float64)
                - mu_global_list_raw[t - 1][offset : offset + Lm].to(dtype=torch.float64)
            )
            coord_energy = delta.pow(2) * curvature_diag
            gap_vals.append(_median_positive_or_default(coord_energy, default=0.0))

        raw_vals.append(
            _median_positive_or_default(
                torch.tensor(gap_vals, dtype=torch.float64),
                default=1.0,
            )
        )
        names.append(name)

    scales = _normalize_and_clip_scales(
        raw_vals,
        clip_min=clip_min,
        clip_max=clip_max,
        shrink_exponent=shrink_exponent,
    )
    return {name: scale for name, scale in zip(names, scales)}


def _build_gap_q_scales(
    mu_global_list_raw: List[Tensor],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float,
) -> List[float]:
    T = len(mu_global_list_raw)
    if T <= 1:
        return [1.0 for _ in range(T)]

    raw_vals = [1.0]
    for t in range(1, T):
        delta = mu_global_list_raw[t] - mu_global_list_raw[t - 1]
        raw_vals.append(float(delta.pow(2).mean().item()))
    return _normalize_and_clip_scales(
        raw_vals,
        clip_min=clip_min,
        clip_max=clip_max,
        shrink_exponent=shrink_exponent,
    )


def _report_module_constant_q_results(
    module_scales: Dict[str, float],
    *,
    s_q: float,
    shrink_exponent: float,
) -> None:
    if not module_scales:
        print("[Module Q] No module scales available.")
        return

    vals = [float(v) for v in module_scales.values()]
    print("\n=== Module-Constant Q Report ===")
    print(
        f"[Module Q] mode=module_constant  estimator=curvature_normalized_mu_drift_prior  "
        f"exposed scale s_Q={float(s_q):.6f}  "
        f"beta={float(min(max(float(shrink_exponent), 0.0), 1.0)):.3f}  "
        "drift_m=median_t median_l ((delta_mu_l)^2 diag(Hbar_m)_l)  Q_t^(m)=s_Q*q_m*I"
    )
    print(
        f"[Module Q Summary] scale(min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f})"
    )
    print("[Module Q Per Module]")
    for name, scale in module_scales.items():
        print(f"  {name}: q_scale={float(scale):.6f} q_diag={float(s_q) * float(scale):.6f}")


def _report_module_gap_q_results(
    module_scales: Dict[str, float],
    gap_scales: List[float],
    *,
    s_q: float,
    module_shrink_exponent: float,
    gap_shrink_exponent: float,
) -> None:
    print("\n=== Module-Gap Q Report ===")
    print(
        f"[Module Q] mode=module_gap  module_estimator=curvature_normalized_mu_drift_prior  "
        f"exposed scale s_Q={float(s_q):.6f}  "
        f"module_beta={float(min(max(float(module_shrink_exponent), 0.0), 1.0)):.3f}  "
        f"gap_beta={float(min(max(float(gap_shrink_exponent), 0.0), 1.0)):.3f}  "
        "Q_t^(m)=s_Q*rho_t*q_m*I"
    )
    if module_scales:
        vals = [float(v) for v in module_scales.values()]
        print(
            f"[Module Q Summary] module_scale(min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f})"
        )
    if gap_scales:
        print(
            f"[Gap Q Summary] gap_scale(min={min(gap_scales):.6f} "
            f"mean={sum(gap_scales)/len(gap_scales):.6f} max={max(gap_scales):.6f})"
        )
    print("[Gap Q Per Slice]")
    for t, scale in enumerate(gap_scales):
        print(f"  t={t}: gap_scale={float(scale):.6f}")


# =========================
# Fast Bayesian eval
# =========================

@dataclass
class _LoraACache:
    name: str
    weight: nn.Parameter
    U_fp32: Tensor
    B_fp32: Tensor
    scaling: float
    offset: int
    L: int
    shape: Tuple[int, ...]
    numel: int

def _resolve_lora_parent_and_adapter(module_path: str) -> Tuple[str, str]:
    parts = module_path.split(".")
    if len(parts) < 3 or parts[-2] != "lora_A":
        raise RuntimeError(f"Unexpected LoRA-A module path: {module_path}")
    return ".".join(parts[:-2]), parts[-1]

def build_loraA_cache(model: nn.Module, module_specs: List[Dict], device: torch.device) -> List[_LoraACache]:
    caches: List[_LoraACache] = []
    for spec in module_specs:
        module_name = spec["name"]
        w = _get_param_weight(model, module_name)
        if w.dtype != torch.float32:
            w.data = w.data.to(dtype=torch.float32)
        parent_path, adapter_name = _resolve_lora_parent_and_adapter(module_name)
        parent = model.get_submodule(parent_path)
        if not hasattr(parent, "lora_B") or adapter_name not in parent.lora_B:
            raise RuntimeError(f"Could not resolve lora_B for {module_name}")
        if not hasattr(parent, "scaling") or adapter_name not in parent.scaling:
            raise RuntimeError(f"Could not resolve scaling for {module_name}")
        B_weight = parent.lora_B[adapter_name].weight
        if B_weight.dtype != torch.float32:
            B_weight = B_weight.to(dtype=torch.float32)
        caches.append(
            _LoraACache(
                name=module_name,
                weight=w,
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
                shape=tuple(w.shape),
                numel=w.numel(),
            )
        )
    return caches


def _sample_posterior_from_stats(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    T: int,
    device: torch.device,
    cpu_device: torch.device,
) -> Tensor:
    # Make posterior MC samples independent of cache hit/miss and KFAC execution path.
    sample_seed = int(args.seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]
    use_module_constant_q = str(args.q_mode) in {"module_constant", "module_gap"}
    use_module_gap_q = str(args.q_mode) == "module_gap"
    gap_q_scales = (
        _build_gap_q_scales(
            mu_global_list_raw,
            clip_min=float(args.gap_q_clip_min),
            clip_max=float(args.gap_q_clip_max),
            shrink_exponent=float(args.gap_q_shrink_exponent),
        )
        if use_module_gap_q
        else [1.0 for _ in range(T)]
    )
    module_q_scales = (
        _build_module_constant_q_scales(
            module_specs,
            module_R_lists,
            mu_global_list_raw,
            clip_min=float(args.module_q_clip_min),
            clip_max=float(args.module_q_clip_max),
            shrink_exponent=float(args.module_q_shrink_exponent),
        )
        if use_module_constant_q
        else {}
    )

    print(f"\n=== Kalman Filter Only (module-wise) ===")
    print(f"[Kalman] modules={len(module_specs)} L_total={sum(int(spec['L']) for spec in module_specs)}")

    if args.forecast_horizon > 0:
        print(
            f"\nDirectly sampling forecast posterior (t=T+{args.forecast_horizon} | T): "
            f"S={args.mc_eval_samples}"
        )
    else:
        print(f"\nDirectly sampling final posterior (t=T): S={args.mc_eval_samples}")

    x_sample_parts: List[Tensor] = []
    for spec in module_specs:
        name = spec["name"]
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        mu_module_list = [
            mu_t[offset : offset + Lm].to(device=cpu_device, dtype=torch.float64)
            for mu_t in mu_global_list
        ]
        H_obs_list, y_list = prepare_lgssm_observations(
            module_R_lists[name],
            mu_list=mu_module_list,
        )
        m1 = torch.zeros(Lm, device=cpu_device, dtype=torch.float64)
        P1 = float(args.p1_var) * torch.eye(Lm, device=cpu_device, dtype=torch.float64)
        if use_module_constant_q:
            module_scale = float(module_q_scales.get(name, 1.0))
            q_var_list = [float(args.s_q) * module_scale * gap_scale for gap_scale in gap_q_scales]
        else:
            q_var_list = [float(args.s_q) for _ in range(T)]
        Q_list = materialize_scalar_Q_list(
            q_var_list,
            L=Lm,
            device=cpu_device,
            dtype=torch.float64,
        )

        x_filt_m, P_filt_m, _, _ = kalman_filter(
            H_list=H_obs_list,
            y_list=y_list,
            Q_list=Q_list,
            m1=m1,
            P1=P1,
        )
        mu_T_m, cov_T_m = _forecast_from_final_posterior(
            x_T=x_filt_m[-1],
            P_T=P_filt_m[-1],
            Q_list=Q_list,
            horizon=int(args.forecast_horizon),
        )
        cov_T_stable = cov_T_m + torch.eye(
            cov_T_m.shape[0],
            device=cov_T_m.device,
            dtype=cov_T_m.dtype,
        ) * 1e-6

        dist_m = torch.distributions.MultivariateNormal(
            mu_T_m,
            covariance_matrix=cov_T_stable,
        )
        x_sample_parts.append(dist_m.sample((int(args.mc_eval_samples),)))

        del (
            H_obs_list,
            y_list,
            Q_list,
            x_filt_m,
            P_filt_m,
            mu_T_m,
            cov_T_m,
            cov_T_stable,
            dist_m,
        )
        gc.collect()

    if use_module_gap_q:
        _report_module_gap_q_results(
            module_q_scales,
            gap_q_scales,
            s_q=float(args.s_q),
            module_shrink_exponent=float(args.module_q_shrink_exponent),
            gap_shrink_exponent=float(args.gap_q_shrink_exponent),
        )
    elif use_module_constant_q:
        _report_module_constant_q_results(
            module_q_scales,
            s_q=float(args.s_q),
            shrink_exponent=float(args.module_q_shrink_exponent),
        )
    else:
        _report_scalar_constant_q_results(args.s_q)
    del mu_global_list
    return torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)


def _sample_independent_slice_ensemble_from_stats(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    T: int,
    cpu_device: torch.device,
) -> Tensor:
    sample_seed = int(args.seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    samples_per_slice = max(int(args.independent_slice_mc_samples_per_slice), 1)
    total_samples = int(T) * samples_per_slice
    mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]

    print("\n=== Independent Slice Ensemble Posterior ===", flush=True)
    print(
        f"[Independent Ensemble] modules={len(module_specs)} T={int(T)} "
        f"samples_per_slice={samples_per_slice} total_samples={total_samples} "
        f"prior=P1={float(args.p1_var):.6g} no_Kalman no_Q",
        flush=True,
    )

    x_sample_parts: List[Tensor] = []
    for spec in module_specs:
        name = spec["name"]
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        mu_module_list = [
            mu_t[offset : offset + Lm].to(device=cpu_device, dtype=torch.float64)
            for mu_t in mu_global_list
        ]
        H_obs_list, y_list = prepare_lgssm_observations(
            module_R_lists[name],
            mu_list=mu_module_list,
        )

        eye = torch.eye(Lm, device=cpu_device, dtype=torch.float64)
        prior_precision = (1.0 / max(float(args.p1_var), 1e-12)) * eye
        module_samples: List[Tensor] = []
        for t in range(int(T)):
            R_t = H_obs_list[t].to(device=cpu_device, dtype=torch.float64)
            y_t = y_list[t].to(device=cpu_device, dtype=torch.float64)
            precision = prior_precision + R_t.T @ R_t
            precision = 0.5 * (precision + precision.T) + 1e-6 * eye
            rhs = R_t.T @ y_t
            chol = torch.linalg.cholesky(precision)
            mean_t = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
            cov_t = torch.cholesky_inverse(chol)
            cov_t = 0.5 * (cov_t + cov_t.T) + 1e-6 * eye
            dist_t = torch.distributions.MultivariateNormal(
                mean_t,
                covariance_matrix=cov_t,
            )
            module_samples.append(dist_t.sample((samples_per_slice,)))
            del R_t, y_t, precision, rhs, chol, mean_t, cov_t, dist_t

        x_sample_parts.append(torch.cat(module_samples, dim=0))
        del H_obs_list, y_list, module_samples
        gc.collect()

    del mu_global_list
    return torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)


def _set_inference_fast(model: nn.Module):
    if hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_disable"):
        model.base_model.gradient_checkpointing_disable()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "base_model") and hasattr(model.base_model, "config") and hasattr(model.base_model.config, "use_cache"):
        model.base_model.config.use_cache = False

@torch.inference_mode()
def _compute_deltas_for_one_sample(lora_cache: List[_LoraACache], xs: torch.Tensor, scale: float) -> List[torch.Tensor]:
    return [
        (spec.U_fp32 @ xs[spec.offset: spec.offset + spec.L] * float(scale)).view(spec.shape)
        for spec in lora_cache
    ]


@torch.inference_mode()
def eval_bayes_fast_restricted_4way_probmean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    choice_token_ids: Tensor,
    lora_cache: List[_LoraACache],
    x_samples_T: Tensor,
    posterior_scale_tau: float = 0.8,
    temp_bayes: float = 1.0,
    max_mc_samples: int = 32,
    mc_eval_chunk: int = 0,
    progress_desc: Optional[str] = None,
    apply_choice_mask: bool = True,
) -> Dict[str, float]:
    model.eval()
    _set_inference_fast(model)

    #scale = float(posterior_scale_tau) / math.sqrt(max(len(lora_cache), 1))
    scale = float(posterior_scale_tau)
    S = min(int(max_mc_samples), int(x_samples_T.shape[0]))
    if S <= 0:
        raise ValueError("max_mc_samples must be positive.")
    chunk_size = S if int(mc_eval_chunk) <= 0 else min(int(mc_eval_chunk), S)

    weight_tensors = [spec.weight.data for spec in lora_cache]
    eps = 1e-12
    acc_bay_m = _make_accuracy(device, num_classes=num_classes)
    acc_bay_m.reset()
    ece_bay_m = _make_ece(device, num_classes=num_classes, n_bins=10)
    ece_bay_m.reset()
    total_samples = 0
    nll_sum = 0.0
    brier_sum = 0.0
    kl_map_to_bayes_sum = 0.0

    bayes_t0 = time.perf_counter()
    for batch in loader:
        lengths_cpu = batch["attention_mask"].sum(dim=1)
        Lmax = max(int(lengths_cpu.max().item()), 1)

        ids = batch["input_ids"][:, -Lmax:].to(device, non_blocking=True)
        attn = batch["attention_mask"][:, -Lmax:].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")
        bsz = int(labels.size(0))
        total_samples += bsz

        logits_map = compute_choice_logits(
            model=model,
            input_ids=ids,
            attention_mask=attn,
            amp_dtype=amp_dtype,
            choice_token_ids=choice_token_ids,
        )
        if apply_choice_mask:
            logits_map = _mask_invalid_choices(logits_map, num_choices)
        probs_map_batch = torch.softmax(logits_map, dim=-1)

        probs_acc_batch = torch.zeros((bsz, num_classes), device=device, dtype=torch.float32)

        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            x_chunk = x_samples_T[chunk_start:chunk_end].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ).contiguous()
            for local_idx in range(x_chunk.shape[0]):
                deltas_s = _compute_deltas_for_one_sample(lora_cache, x_chunk[local_idx], scale)
                torch._foreach_add_(weight_tensors, deltas_s)

                logits = compute_choice_logits(
                    model=model,
                    input_ids=ids,
                    attention_mask=attn,
                    amp_dtype=amp_dtype,
                    choice_token_ids=choice_token_ids,
                )
                if apply_choice_mask:
                    logits = _mask_invalid_choices(logits, num_choices)
                probs_acc_batch.add_(torch.softmax(logits, dim=-1))

                torch._foreach_sub_(weight_tensors, deltas_s)
            del x_chunk

        probs_bayes_batch = probs_acc_batch / float(S)

        if temp_bayes != 1.0:
            p = probs_bayes_batch.clamp_min(eps) ** (1.0 / float(temp_bayes))
            probs_bayes_batch = p / p.sum(dim=-1, keepdim=True)

        idx = torch.arange(bsz, device=device)
        nll_sum += float((-torch.log(probs_bayes_batch[idx, labels].clamp_min(eps))).sum().item())
        brier_sum += _multiclass_brier_sum(probs_bayes_batch, labels)
        kl_map_to_bayes_sum += float(
            (
                probs_map_batch.clamp_min(eps)
                * (
                    torch.log(probs_map_batch.clamp_min(eps))
                    - torch.log(probs_bayes_batch.clamp_min(eps))
                )
            ).sum(dim=-1).sum().item()
        )

        acc_bay_m.update(probs_bayes_batch, labels)
        ece_bay_m.update(probs_bayes_batch, labels)
        del (
            probs_acc_batch,
            probs_bayes_batch,
            probs_map_batch,
            logits_map,
            ids,
            attn,
            labels,
        )

    bayes_extra_time = time.perf_counter() - bayes_t0

    metrics = {
        "nll_bayes": nll_sum / max(total_samples, 1),
        "brier_bayes": brier_sum / max(total_samples, 1),
        "ece_bayes": float(ece_bay_m.compute().item()),
        "acc_bayes": float(acc_bay_m.compute().item()),
        "kl_map_to_bayes": kl_map_to_bayes_sum / max(total_samples, 1),
        "mc_samples_used": float(S),
        "mc_chunk_used": float(chunk_size),
        "posterior_scale_factor": float(scale),
        "time_bayes_sec": float(bayes_extra_time),
    }
    return metrics


def _subset_dataset(ds: Dataset, subset_size: int, seed: int) -> Dataset:
    if int(subset_size) <= 0 or int(subset_size) >= len(ds):
        return ds
    return ds.shuffle(seed=int(seed)).select(range(int(subset_size)))


def _tau_accuracy_ok(
    metrics: Dict[str, float],
    *,
    acc_floor: float,
) -> bool:
    return float(metrics["acc_bayes"]) >= float(acc_floor)


def _tau_kl_in_window(metrics: Dict[str, float], *, kl_low: float, kl_high: float) -> bool:
    kl = float(metrics["kl_map_to_bayes"])
    return float(kl_low) <= kl <= float(kl_high)


def _tau_kl_window_distance(metrics: Dict[str, float], *, kl_low: float, kl_high: float) -> float:
    kl = float(metrics["kl_map_to_bayes"])
    if kl < float(kl_low):
        return float(kl_low) - kl
    if kl > float(kl_high):
        return kl - float(kl_high)
    return 0.0


@torch.inference_mode()
def fit_seq_lora_tau_anchor_kl_direct(
    model: nn.Module,
    anchor_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    choice_token_ids: Tensor,
    lora_cache: List[_LoraACache],
    x_samples_T: Tensor,
    tau_max: float,
    search_iters: int,
    anchor_n_samples: int,
    temp_bayes: float,
    mc_eval_chunk: int,
    apply_choice_mask: bool,
    acc_tolerance: float,
    kl_target_low: float,
    kl_target_high: float,
) -> Dict[str, float]:
    tau_max_arg = float(tau_max)
    if tau_max_arg <= 0.0:
        raise ValueError(f"tau_search_max must be positive, got {tau_max_arg}")
    tau_max = tau_max_arg
    kl_target_low = float(kl_target_low)
    kl_target_high = float(kl_target_high)
    if not (0.0 <= kl_target_low <= kl_target_high):
        raise ValueError(
            f"Expected 0 <= tau_kl_target_low <= tau_kl_target_high, "
            f"got {kl_target_low}, {kl_target_high}"
        )

    def _eval_tau(tau: float, *, n_samples: int, desc: str) -> Dict[str, float]:
        return eval_bayes_fast_restricted_4way_probmean(
            model=model,
            loader=anchor_loader,
            device=device,
            amp_dtype=amp_dtype,
            num_classes=num_classes,
            choice_token_ids=choice_token_ids,
            lora_cache=lora_cache,
            x_samples_T=x_samples_T,
            posterior_scale_tau=float(tau),
            temp_bayes=temp_bayes,
            max_mc_samples=max(int(n_samples), 1),
            mc_eval_chunk=mc_eval_chunk,
            progress_desc=desc,
            apply_choice_mask=apply_choice_mask,
        )

    baseline = _eval_tau(0.0, n_samples=1, desc="TAU-AUTO ref tau=0")
    acc0 = float(baseline["acc_bayes"])
    nll0 = float(baseline["nll_bayes"])
    acc_floor = acc0 - float(acc_tolerance)
    records: List[Tuple[float, Dict[str, float], str]] = [(0.0, baseline, "baseline")]
    kl_target = kl_target_low
    probe_tau = min(0.5, tau_max)
    eps = 1e-12

    print(
        "\n=== Auto-selecting posterior_tau by direct anchor KL estimate ===\n"
        f"[Tau auto] baseline tau=0.000000 "
        f"Acc0={acc0*100:.2f}% NLL0={nll0:.6f} KL0={baseline['kl_map_to_bayes']:.6f}\n"
        f"[Tau auto] target: "
        f"{kl_target_low:.6f} <= KL_tau <= {kl_target_high:.6f} "
        f"(direct_target={kl_target:.6f}, acc floor logged only: {acc_floor*100:.2f}%) "
        f"using KL(tau) ~= C * tau^2 within [0, {tau_max:.6f}]",
        flush=True,
    )

    def _select(
        tau: float,
        metrics: Dict[str, float],
        source: str,
    ) -> Dict[str, float]:
        print(
            f"[Tau auto] selected posterior_tau={float(tau):.6f} "
            f"source={source} "
            f"anchor_acc={metrics['acc_bayes']*100:.2f}% "
            f"anchor_nll={metrics['nll_bayes']:.6f} "
            f"anchor_kl={metrics['kl_map_to_bayes']:.6f} "
            f"kl_window=[{kl_target_low:.6f}, {kl_target_high:.6f}]",
            flush=True,
        )
        return {
            "optimal_posterior_tau": float(tau),
            "baseline_acc": float(acc0),
            "baseline_nll": float(nll0),
            "selected_acc": float(metrics["acc_bayes"]),
            "selected_nll": float(metrics["nll_bayes"]),
            "selected_kl_map_to_bayes": float(metrics["kl_map_to_bayes"]),
            "acc_tolerance": float(acc_tolerance),
            "kl_target": float(kl_target),
            "kl_target_low": float(kl_target_low),
            "kl_target_high": float(kl_target_high),
            "tau_max": float(tau_max),
            "anchor_n_samples": int(anchor_n_samples),
        }

    probe_metrics = _eval_tau(
        probe_tau,
        n_samples=int(anchor_n_samples),
        desc=f"TAU-AUTO probe tau={probe_tau:.4f}",
    )
    probe_kl = float(probe_metrics["kl_map_to_bayes"])
    probe_accuracy_ok = _tau_accuracy_ok(probe_metrics, acc_floor=acc_floor)
    probe_in_window = _tau_kl_in_window(
        probe_metrics,
        kl_low=kl_target_low,
        kl_high=kl_target_high,
    )
    records.append((float(probe_tau), probe_metrics, "probe"))
    print(
        f"[Tau auto] probe tau={probe_tau:.6f} "
        f"acc={probe_metrics['acc_bayes']*100:.2f}% "
        f"nll={probe_metrics['nll_bayes']:.6f} "
        f"kl={probe_kl:.6f} "
        f"accuracy_ok={int(probe_accuracy_ok)} kl_in_window={int(probe_in_window)}",
        flush=True,
    )
    if probe_in_window:
        return _select(probe_tau, probe_metrics, "probe_kl_window")

    if probe_kl <= eps:
        tau_hat = tau_max
    else:
        tau_hat = probe_tau * math.sqrt(max(kl_target, 0.0) / max(probe_kl, eps))
        tau_hat = min(max(float(tau_hat), 0.0), tau_max)
    print(
        f"[Tau auto] direct estimate from probe: "
        f"tau={tau_hat:.6f} = {probe_tau:.6f} * sqrt({kl_target:.6f} / {max(probe_kl, eps):.6f})",
        flush=True,
    )

    if abs(float(tau_hat) - float(probe_tau)) <= 1e-8:
        tau_hat_metrics = probe_metrics
    else:
        tau_hat_metrics = _eval_tau(
            tau_hat,
            n_samples=int(anchor_n_samples),
            desc=f"TAU-AUTO direct tau={tau_hat:.4f}",
        )
        records.append((float(tau_hat), tau_hat_metrics, "direct"))
    tau_hat_accuracy_ok = _tau_accuracy_ok(tau_hat_metrics, acc_floor=acc_floor)
    tau_hat_in_window = _tau_kl_in_window(
        tau_hat_metrics,
        kl_low=kl_target_low,
        kl_high=kl_target_high,
    )
    print(
        f"[Tau auto] direct tau={tau_hat:.6f} "
        f"acc={tau_hat_metrics['acc_bayes']*100:.2f}% "
        f"nll={tau_hat_metrics['nll_bayes']:.6f} "
        f"kl={tau_hat_metrics['kl_map_to_bayes']:.6f} "
        f"accuracy_ok={int(tau_hat_accuracy_ok)} kl_in_window={int(tau_hat_in_window)}",
        flush=True,
    )
    if tau_hat_in_window:
        return _select(tau_hat, tau_hat_metrics, "direct_kl_window")

    tau_hat_kl = float(tau_hat_metrics["kl_map_to_bayes"])
    if tau_hat_kl > eps and tau_hat > 0.0:
        tau_refined = tau_hat * math.sqrt(max(kl_target, 0.0) / max(tau_hat_kl, eps))
        tau_refined = min(max(float(tau_refined), 0.0), tau_max)
        print(
            f"[Tau auto] one-step KL refinement: "
            f"tau={tau_refined:.6f} = {tau_hat:.6f} * sqrt({kl_target:.6f} / {tau_hat_kl:.6f})",
            flush=True,
        )
        if abs(float(tau_refined) - float(tau_hat)) > 1e-8:
            refined_metrics = _eval_tau(
                tau_refined,
                n_samples=int(anchor_n_samples),
                desc=f"TAU-AUTO refined tau={tau_refined:.4f}",
            )
            records.append((float(tau_refined), refined_metrics, "refined"))
            refined_accuracy_ok = _tau_accuracy_ok(refined_metrics, acc_floor=acc_floor)
            refined_in_window = _tau_kl_in_window(
                refined_metrics,
                kl_low=kl_target_low,
                kl_high=kl_target_high,
            )
            print(
                f"[Tau auto] refined tau={tau_refined:.6f} "
                f"acc={refined_metrics['acc_bayes']*100:.2f}% "
                f"nll={refined_metrics['nll_bayes']:.6f} "
                f"kl={refined_metrics['kl_map_to_bayes']:.6f} "
                f"accuracy_ok={int(refined_accuracy_ok)} kl_in_window={int(refined_in_window)}",
                flush=True,
            )
            if refined_in_window:
                return _select(tau_refined, refined_metrics, "refined_kl_window")

            refined_kl = float(refined_metrics["kl_map_to_bayes"])
            direct_kl = float(tau_hat_metrics["kl_map_to_bayes"])
            if (
                min(direct_kl, refined_kl) <= kl_target_low
                and max(direct_kl, refined_kl) >= kl_target_high
                and abs(direct_kl - refined_kl) > eps
            ):
                if direct_kl <= refined_kl:
                    tau_low, kl_low_actual = float(tau_hat), direct_kl
                    tau_high, kl_high_actual = float(tau_refined), refined_kl
                else:
                    tau_low, kl_low_actual = float(tau_refined), refined_kl
                    tau_high, kl_high_actual = float(tau_hat), direct_kl
                tau_interp = tau_low + (
                    (kl_target_low - kl_low_actual)
                    / max(kl_high_actual - kl_low_actual, eps)
                    * (tau_high - tau_low)
                )
                tau_interp = min(max(float(tau_interp), 0.0), tau_max)
                print(
                    f"[Tau auto] linear KL interpolation: tau={tau_interp:.6f} "
                    f"from ({tau_low:.6f}, kl={kl_low_actual:.6f}) "
                    f"to ({tau_high:.6f}, kl={kl_high_actual:.6f}) "
                    f"target_kl={kl_target_low:.6f}; selecting without extra eval",
                    flush=True,
                )
                interp_metrics = dict(refined_metrics if refined_kl < direct_kl else tau_hat_metrics)
                interp_metrics["kl_map_to_bayes"] = float(kl_target_low)
                return _select(tau_interp, interp_metrics, "interp_linear_kl_no_eval")

    selected_tau, selected_metrics, selected_source = min(
        records,
        key=lambda item: _tau_kl_window_distance(
            item[1],
            kl_low=kl_target_low,
            kl_high=kl_target_high,
        ),
    )
    return _select(float(selected_tau), selected_metrics, f"{selected_source}_closest_kl")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Seq-LoRA with ASDL Kron curvature and process noise.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--trust_remote_code",
        type=_parse_bool,
        default=TRUST_REMOTE_CODE,
        help="Whether to enable trust_remote_code when loading the base model/tokenizer.",
    )
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="Maximum sequence length.")
    parser.add_argument("--eval_bsz", type=int, default=EVAL_BSZ, help="Evaluation batch size.")
    parser.add_argument("--kfac_bsz", type=int, default=KFAC_BSZ, help="KFAC slice batch size.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Num workers for KFAC/train slice loaders.")
    parser.add_argument("--eval_num_workers", type=int, default=EVAL_NUM_WORKERS, help="Num workers for eval loaders.")
    parser.add_argument(
        "--eval_prefetch_factor",
        type=int,
        default=EVAL_PREFETCH_FACTOR,
        help="Prefetch factor used only when eval_num_workers > 0.",
    )
    parser.add_argument("--n_kfac", type=int, default=N_KFAC, help="Target number of KFAC factors/eigendirections.")
    parser.add_argument("--lr_threshold", type=int, default=LR_THRESHOLD, help="Low-rank threshold used in subspace construction.")
    parser.add_argument(
        "--kfac_backend",
        type=str,
        default=KFAC_BACKEND,
        choices=["asdl"],
        help="Curvature backend. Only ASDL Kron is supported.",
    )
    parser.add_argument(
        "--max_kfac_samples_per_slice",
        type=int,
        default=MAX_KFAC_SAMPLES_PER_SLICE,
        help="Maximum number of KFAC samples per slice. Set to a negative value to disable the cap.",
    )
    parser.add_argument("--mu_obs_scale", type=float, default=MU_OBS_SCALE, help="Scale factor applied to mu observations.")
    parser.add_argument("--mu_obs_batches", type=int, default=MU_OBS_BATCHES, help="Number of batches per slice used for mu observations.")
    parser.add_argument(
        "--disable_dropout_during_kfac_mu",
        type=_parse_bool,
        default=DISABLE_DROPOUT_DURING_KFAC_MU,
        help=(
            "Temporarily disable nn.Dropout modules while building KFAC and mu observations. "
            "Useful to isolate train-mode dropout effects."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
        help="Unified tasks",
    )
    parser.add_argument(
        "--slices_dir",
        type=str,
        default="",
        help=(
            "Path to the KFAC slices dataset directory. If omitted for "
            "scienceqa_closedchoice_grade2_11, the script will read the task "
            "training split directly and use grade-based slice ids."
        ),
    )
    parser.add_argument(
        "--random_num_slices",
        type=int,
        default=0,
        help="If > 0 and --slices_dir is omitted, assign balanced random slice ids to the source-task training split.",
    )
    parser.add_argument(
        "--slice_order",
        type=str,
        default=SLICE_ORDER,
        choices=["sorted", "reverse", "shuffle"],
        help=(
            "Order in which slice ids are fed to the LGSSM/Kalman chain. "
            "Use shuffle for slice-order ablations."
        ),
    )
    parser.add_argument(
        "--slice_order_seed",
        type=int,
        default=SLICE_ORDER_SEED,
        help="Random seed used only when --slice_order shuffle.",
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        required=True,
        help="Path to the MAP adapter directory.",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="Comma-separated eval tasks. Supports iid, arc, arc-c, arc-e, sciq, hellaswag, gpqa, gpqa_main, agieval, mmlu, mmlu_science_high, mmlu_science_college, scienceqa_closedchoice_grade12.",
    )
    parser.add_argument(
        "--s_q",
        type=float,
        default=float(S_Q),
        help=(
            "Global process-noise scale. In constant mode it sets Q_t = s_Q * I. "
            "In module_constant/module_gap modes it scales the learned module/gap multipliers."
        ),
    )
    parser.add_argument(
        "--constant_q_var",
        dest="s_q",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--q_mode",
        type=str,
        default=Q_MODE,
        choices=["constant", "module_constant", "module_gap"],
        help=(
            "Process-noise construction. constant uses shared Q_t = s_Q * I; "
            "module_constant uses per-module q_m from curvature-normalized adjacent-mu drift; "
            "module_gap additionally modulates Q_t by adjacent slice-gap scales."
        ),
    )
    parser.add_argument(
        "--module_q_clip_min",
        type=float,
        default=MODULE_Q_CLIP_MIN,
        help="Lower clip for curvature-normalized per-module Q scales.",
    )
    parser.add_argument(
        "--module_q_clip_max",
        type=float,
        default=MODULE_Q_CLIP_MAX,
        help="Upper clip for curvature-normalized per-module Q scales.",
    )
    parser.add_argument(
        "--module_q_shrink_exponent",
        type=float,
        default=MODULE_Q_SHRINK_EXPONENT,
        help=(
            "Log-normal shrinkage exponent beta for module_constant/module_gap Q. "
            "q_m = clip((curvature_normalized_drift_m / median_drift) ** beta). "
            "0 makes all modules equal; 1 preserves the normalized drift heuristic."
        ),
    )
    parser.add_argument(
        "--gap_q_clip_min",
        type=float,
        default=GAP_Q_CLIP_MIN,
        help="Lower clip for per-slice gap Q scales in module_gap mode.",
    )
    parser.add_argument(
        "--gap_q_clip_max",
        type=float,
        default=GAP_Q_CLIP_MAX,
        help="Upper clip for per-slice gap Q scales in module_gap mode.",
    )
    parser.add_argument(
        "--gap_q_shrink_exponent",
        type=float,
        default=GAP_Q_SHRINK_EXPONENT,
        help=(
            "Log-normal shrinkage exponent beta for per-slice gap Q scales in module_gap mode. "
            "rho_t = clip((gap_drift_t / median_gap_drift) ** beta). "
            "0 makes all gaps equal; 1 preserves the previous raw ratio behavior."
        ),
    )
    parser.add_argument("--p1_var", type=float, default=P1_VAR, help="Initial state covariance scale P1.")
    parser.add_argument(
        "--subspace_dim_per_module",
        type=int,
        default=SUBSPACE_DIM_PER_MODULE,
        help="Subspace dimension retained per module.",
    )
    parser.add_argument("--mc_eval_samples", type=int, default=MC_EVAL_SAMPLES, help="Number of MC posterior samples during evaluation.")
    parser.add_argument(
        "--posterior_eval_mode",
        type=str,
        default=POSTERIOR_EVAL_MODE,
        choices=["lgssm_final", "independent_slice_ensemble"],
        help=(
            "Posterior predictive mode. lgssm_final samples the final Kalman-filtered "
            "state. independent_slice_ensemble samples each slice-local posterior "
            "independently and averages their posterior predictive probabilities."
        ),
    )
    parser.add_argument(
        "--independent_slice_mc_samples_per_slice",
        type=int,
        default=INDEPENDENT_SLICE_MC_SAMPLES_PER_SLICE,
        help="MC samples per slice when --posterior_eval_mode independent_slice_ensemble.",
    )
    parser.add_argument(
        "--mc_eval_chunk",
        type=int,
        default=0,
        help="Optional chunk size for MC samples during evaluation. <=0 disables chunking.",
    )
    parser.add_argument("--posterior_tau", type=float, default=POSTERIOR_TAU, help="Posterior scale multiplier used at evaluation.")
    parser.add_argument("--temp_bayes", type=float, default=TEMP_BAYES, help="Temperature applied to Bayesian mean probabilities.")
    parser.add_argument(
        "--tau_mode",
        type=str,
        default=TAU_MODE,
        choices=["fixed", "auto"],
        help=(
            "fixed uses --posterior_tau. auto estimates tau from anchor KL using "
            "KL(tau) ~= C * tau^2."
        ),
    )
    parser.add_argument(
        "--auto_tau",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Alias for --tau_mode auto.",
    )
    parser.add_argument(
        "--tau_search_max",
        type=float,
        default=TAU_SEARCH_MAX,
        help="Upper bound for automatic posterior_tau estimation.",
    )
    parser.add_argument(
        "--tau_search_iters",
        type=int,
        default=TAU_SEARCH_ITERS,
        help="Deprecated compatibility option; direct KL tau estimation does not use binary search.",
    )
    parser.add_argument(
        "--tau_anchor_size",
        type=int,
        default=TAU_ANCHOR_SIZE,
        help="Maximum number of source-train anchor examples for tau estimation. <=0 uses the full train split.",
    )
    parser.add_argument(
        "--tau_anchor_bsz",
        type=int,
        default=TAU_ANCHOR_BSZ,
        help="Batch size for source-train anchor tau estimation.",
    )
    parser.add_argument(
        "--tau_anchor_n_samples",
        type=int,
        default=TAU_ANCHOR_N_SAMPLES,
        help="MC samples per nonzero tau during automatic tau estimation.",
    )
    parser.add_argument(
        "--tau_acc_tolerance",
        type=float,
        default=TAU_ACC_TOLERANCE,
        help="Accuracy-drop reference from tau=0, logged during direct KL automatic tau estimation.",
    )
    parser.add_argument(
        "--tau_kl_target_low",
        type=float,
        default=TAU_KL_TARGET_LOW,
        help="Lower edge of target anchor KL(MAP || Bayes) window for automatic tau estimation.",
    )
    parser.add_argument(
        "--tau_kl_target_high",
        type=float,
        default=TAU_KL_TARGET_HIGH,
        help="Upper edge of target anchor KL(MAP || Bayes) window for automatic tau estimation.",
    )
    parser.add_argument(
        "--posterior_stats_cache_path",
        type=str,
        default="",
        help=(
            "Optional path for caching KFAC/subspace/mu stats. Matching caches skip "
            "the expensive KFAC and mu stages, while still allowing q/tau/temp/MC sweeps."
        ),
    )
    parser.add_argument(
        "--force_rebuild_posterior_stats_cache",
        action="store_true",
        help="Rebuild and overwrite --posterior_stats_cache_path even if it already exists.",
    )
    parser.add_argument(
        "--tokenizer_padding_side",
        type=str,
        default=TOKENIZER_PADDING_SIDE,
        choices=["left", "right"],
        help="Padding side used by the tokenizer.",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=0,
        help="Forecast horizon h. 0 uses x_{T|T}; 1 uses one-step-ahead x_{T+1|T}.",
    )
    parser.add_argument(
        "--keep_full_vocab_lm_head",
        type=_parse_bool,
        default=False,
        help=(
            "Keep the checkpoint's full-vocab lm_head and slice choice-token logits dynamically. "
            "Useful for local full-vocab checkpoints trained under the default prompt protocol "
            "(for example train_closedchoice_llama2_7b_lora_map_leftpad.py outputs)."
        ),
    )
    args = parser.parse_args()
    if bool(args.auto_tau):
        args.tau_mode = "auto"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(HF_DATASETS_CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE_DIR
    try:
        hf_datasets.config.HF_DATASETS_CACHE = HF_DATASETS_CACHE_DIR
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print("Using device:", device)
    apply_choice_mask = True
    keep_full_vocab_lm_head = bool(args.keep_full_vocab_lm_head)
    print(
        "[Curvature backend] Using ASDL Kron on the full wrapper graph "
        "with randomized PSD low-rank compression for large blocks."
    )
    print(
        f"[KFAC] disable_dropout_during_kfac_mu={bool(args.disable_dropout_during_kfac_mu)}"
    )

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    pin_memory = (device.type == "cuda")

    peft_cfg = PeftConfig.from_pretrained(args.map_dir)
    base_name = peft_cfg.base_model_name_or_path
    print(f"\n[Load] base_model = {base_name}\n[Load] adapter    = {args.map_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.padding_side = args.tokenizer_padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token

    num_classes = get_task_num_classes(args.task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    if keep_full_vocab_lm_head:
        print("[Head] keeping full-vocab lm_head and slicing choice-token logits dynamically")
        model = PeftModel.from_pretrained(base_model, args.map_dir).to(device)
    else:
        trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
        print(f"[Head] trimmed lm_head to {num_classes} choice logits")
        model = PeftModel.from_pretrained(base_model, args.map_dir).to(device)
    model.eval()

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype=torch.float32)
            p.requires_grad = True

    if args.slices_dir:
        ds_slices = load_from_disk(args.slices_dir)
        train_raw = ds_slices["train"]
        slice_source = f"slices_dir={args.slices_dir}"
    else:
        train_raw, _, _ = load_task_dataset(args.task)
        if int(args.random_num_slices) > 0:
            train_raw = _assign_random_slice_ids(train_raw, int(args.random_num_slices), int(args.seed))
            slice_source = f"task_train_split[random_{int(args.random_num_slices)}_slices_seed_{int(args.seed)}]"
        else:
            train_raw = _ensure_slice_ids_for_seq(args.task, train_raw)
            slice_source = "task_train_split"

    # -------------------------
    # Eval task validation only. Actual eval datasets are loaded lazily later.
    # -------------------------
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    for eval_task in eval_tasks:
        eval_num_classes = get_task_num_classes(eval_task)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, "
                f"but source task '{args.task}' has {num_classes} classes."
            )

    # -------------------------
    # KFAC/train slices: dynamic padding to reduce wasted compute on long MMLU inputs
    # -------------------------
    train_proc = preprocess_task(
        args.task,
        train_raw,
        tokenizer,
        args.max_seq_len,
        pad_to_max_length=False,
    )
    if "slice_id" not in train_proc.column_names:
        train_proc = train_proc.map(
            lambda ex, idx: {"slice_id": int(train_raw[idx]["slice_id"])},
            with_indices=True,
        )
    if "seq_len" not in train_proc.column_names:
        train_proc = train_proc.add_column("seq_len", [len(x) for x in train_proc["input_ids"]])
    train_proc = _canonicalize_proc_columns(train_proc)

    slice_ids = sorted(set(int(x) for x in train_raw["slice_id"]))
    natural_slice_ids = list(slice_ids)
    if str(args.slice_order) == "reverse":
        slice_ids = list(reversed(slice_ids))
    elif str(args.slice_order) == "shuffle":
        rng = random.Random(int(args.slice_order_seed))
        rng.shuffle(slice_ids)
    T = len(slice_ids)
    if slice_ids != natural_slice_ids:
        print(
            f"[Slices] source={slice_source} order={str(args.slice_order)} "
            f"order_seed={int(args.slice_order_seed)} run_ids={slice_ids}",
            flush=True,
        )

    kfac_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    slice_loaders: List[DataLoader] = []
    total_kfac_samples = 0
    total_kfac_batches = 0
    max_kfac_samples_per_slice = (
        None if int(args.max_kfac_samples_per_slice) < 0 else int(args.max_kfac_samples_per_slice)
    )
    for prep_idx, sid in enumerate(slice_ids):
        ds_t = train_proc.filter(lambda ex, sid=sid: int(ex["slice_id"]) == sid)
        if max_kfac_samples_per_slice is not None and len(ds_t) > max_kfac_samples_per_slice:
            ds_t = ds_t.shuffle(seed=42).select(range(max_kfac_samples_per_slice))
        ds_t = ds_t.sort("seq_len")
        eff_samples = len(ds_t)
        eff_batches = math.ceil(eff_samples / max(int(args.kfac_bsz), 1))
        total_kfac_samples += eff_samples
        total_kfac_batches += eff_batches
        ds_loader = ds_t.remove_columns(["seq_len"]) if "seq_len" in ds_t.column_names else ds_t
        slice_loaders.append(
            DataLoader(
                ds_loader,
                batch_size=args.kfac_bsz,
                shuffle=False,
                drop_last=False,
                collate_fn=kfac_collator,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
        )
    print(
        f"[KFAC] prepared {len(slice_loaders)} slices from {slice_source}; "
        f"samples={total_kfac_samples} batches={total_kfac_batches} drop_last=False",
        flush=True,
    )
    forward_call_for_kfac = forward_call_for_kfac_factory(
        amp_dtype,
        choice_token_ids,
        apply_choice_mask=apply_choice_mask,
    )
    stats_cache_path = str(args.posterior_stats_cache_path).strip()
    stats_cache_snapshot = _build_posterior_stats_cache_snapshot(
        args,
        slice_ids=slice_ids,
        train_raw_fingerprint=str(getattr(train_raw, "_fingerprint", "")),
        train_proc_fingerprint=str(getattr(train_proc, "_fingerprint", "")),
    )
    stats_cache_loaded = False

    if stats_cache_path and os.path.exists(stats_cache_path) and not bool(args.force_rebuild_posterior_stats_cache):
        module_specs, module_R_lists, mu_global_list_raw, cached_T = _load_posterior_stats_cache(
            stats_cache_path,
            stats_cache_snapshot,
        )
        if int(cached_T) != int(T):
            raise RuntimeError(
                f"Posterior-stats cache at {stats_cache_path} has T={cached_T}, current T={T}."
        )
        lora_cache = build_loraA_cache(model, module_specs, device=device)
        stats_cache_loaded = True

    if stats_cache_loaded:
        with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior sample from cached stats on {args.task}"):
            if str(args.posterior_eval_mode) == "independent_slice_ensemble":
                x_samples_T = _sample_independent_slice_ensemble_from_stats(
                    args=args,
                    module_specs=module_specs,
                    module_R_lists=module_R_lists,
                    mu_global_list_raw=mu_global_list_raw,
                    T=T,
                    cpu_device=cpu_device,
                )
            else:
                x_samples_T = _sample_posterior_from_stats(
                    args=args,
                    module_specs=module_specs,
                    module_R_lists=module_R_lists,
                    mu_global_list_raw=mu_global_list_raw,
                    T=T,
                    device=device,
                    cpu_device=cpu_device,
                )
    else:
        H_factor_per_module, G_factor_per_module, module_names = {}, {}, None

        dropout_ctx = (
            _temporarily_disable_dropout_modules(model)
            if bool(args.disable_dropout_during_kfac_mu)
            else nullcontext()
        )

        with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior build on {args.task}"), dropout_ctx:
            print("[KFAC] posterior stage ready", flush=True)
            for t_idx, loader_t in enumerate(slice_loaders):
                print(
                    f"[KFAC] slice {t_idx + 1}/{len(slice_loaders)} "
                    f"sid={slice_ids[t_idx]} batches={len(loader_t)}",
                    flush=True,
                )
                autocast_ctx = (
                    torch.amp.autocast(device_type="cuda", enabled=False)
                    if device.type == "cuda"
                    else type("NoOp", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: False})()
                )

                with autocast_ctx:
                    factors = calculate_kronecker_factors(
                        model=model,
                        forward_call=forward_call_for_kfac,
                        loader=loader_t,
                        n_kfac=args.n_kfac,
                        lr_threshold=args.lr_threshold,
                        target_module_keywords=["lora_A"],
                        exclude_bias=False,
                        use_tqdm=False,
                        disable_dropout=bool(args.disable_dropout_during_kfac_mu),
                    )
                print(f"[KFAC] slice {t_idx + 1}/{len(slice_loaders)} done", flush=True)

                if module_names is None:
                    module_names = _resolve_bayes_module_names(factors)
                    for n in module_names:
                        H_factor_per_module[n], G_factor_per_module[n] = [], []

                for name in module_names:
                    A_t, S_t = factors[name]
                    H_factor_per_module[name].append(
                        A_t.detach().to(dtype=torch.float64, device=cpu_device)
                    )
                    G_factor_per_module[name].append(
                        S_t.detach().to(dtype=torch.float64, device=cpu_device)
                    )
                    del A_t, S_t

                factors.clear()
                del factors
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            module_subspace_info, module_R_lists = {}, {}
            for name in module_names:
                H_factors = H_factor_per_module[name]
                G_factors = G_factor_per_module[name]

                H_bar_bal = materialize_mean_psd_from_factors(
                    H_factors,
                    matrix_scale=1.0,
                    device=device,
                    dtype=torch.float64,
                )
                G_bar_bal = materialize_mean_psd_from_factors(
                    G_factors,
                    matrix_scale=1.0,
                    device=device,
                    dtype=torch.float64,
                )

                subspace_info_gpu = build_global_kronecker_eigenspace(
                    H_list=[H_bar_bal],
                    G_B_list=[G_bar_bal],
                    subspace_dim=args.subspace_dim_per_module,
                    eps_eig=1e-6,
                )
                H_x_list, R_list = project_curvature_factors_to_subspace(
                    H_factors=H_factors,
                    G_B_factors=G_factors,
                    subspace_info=subspace_info_gpu,
                    lambda_damp=1e-4,
                    H_matrix_scale=1.0,
                    G_matrix_scale=1.0,
                    work_device=device,
                    out_device=cpu_device,
                    dtype=torch.float64,
                )
                module_subspace_info[name] = _move_subspace_info(
                    subspace_info_gpu,
                    device=cpu_device,
                    dtype=torch.float64,
                )
                module_R_lists[name] = R_list

                H_factors.clear()
                G_factors.clear()
                del H_bar_bal, G_bar_bal, subspace_info_gpu, H_x_list
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            H_factor_per_module.clear()
            G_factor_per_module.clear()

            module_specs, offset = [], 0
            for name in module_names:
                Lm = int(module_subspace_info[name]["U_lora"].shape[1])
                module_specs.append(
                    {
                        "name": name,
                        "subspace_info": module_subspace_info[name],
                        "offset": offset,
                        "L": Lm,
                    }
                )
                offset += Lm
            lora_cache = build_loraA_cache(model, module_specs, device=device)

            mu_global_list_raw = estimate_mu_global_list_from_slice_grads_asdl(
                model,
                slice_loaders,
                forward_call_for_kfac,
                module_names,
                module_subspace_info,
                module_R_lists,
                device,
                args.mu_obs_batches,
                torch.float64,
                disable_dropout=bool(args.disable_dropout_during_kfac_mu),
            )

            if stats_cache_path:
                stats_cache_dir = os.path.dirname(stats_cache_path)
                if stats_cache_dir:
                    os.makedirs(stats_cache_dir, exist_ok=True)
                torch.save(
                    _build_posterior_stats_cache_payload(
                        args=args,
                        snapshot=stats_cache_snapshot,
                        module_specs=module_specs,
                        module_R_lists=module_R_lists,
                        mu_global_list_raw=mu_global_list_raw,
                        T=T,
                    ),
                    stats_cache_path,
                )
                print(f"[Posterior Stats Cache] saved to {stats_cache_path}", flush=True)

            if str(args.posterior_eval_mode) == "independent_slice_ensemble":
                x_samples_T = _sample_independent_slice_ensemble_from_stats(
                    args=args,
                    module_specs=module_specs,
                    module_R_lists=module_R_lists,
                    mu_global_list_raw=mu_global_list_raw,
                    T=T,
                    cpu_device=cpu_device,
                )
            else:
                x_samples_T = _sample_posterior_from_stats(
                    args=args,
                    module_specs=module_specs,
                    module_R_lists=module_R_lists,
                    mu_global_list_raw=mu_global_list_raw,
                    T=T,
                    device=device,
                    cpu_device=cpu_device,
                )

    eval_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    eval_loader_kwargs = {
        "batch_size": args.eval_bsz,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": eval_collator,
        "num_workers": args.eval_num_workers,
        "pin_memory": pin_memory,
    }
    if args.eval_num_workers > 0:
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = args.eval_prefetch_factor

    effective_posterior_tau = float(args.posterior_tau)
    if str(args.tau_mode) == "auto":
        tau_anchor_proc = _subset_dataset(
            train_proc,
            subset_size=int(args.tau_anchor_size),
            seed=int(args.seed),
        )
        if "seq_len" in tau_anchor_proc.column_names:
            tau_anchor_proc = tau_anchor_proc.sort("seq_len")
            tau_anchor_eval = tau_anchor_proc.remove_columns(["seq_len"])
        else:
            tau_anchor_eval = tau_anchor_proc
        tau_loader_kwargs = dict(eval_loader_kwargs)
        tau_loader_kwargs["batch_size"] = max(int(args.tau_anchor_bsz), 1)
        tau_anchor_loader = DataLoader(tau_anchor_eval, **tau_loader_kwargs)
        print(
            f"[Tau auto] anchor source=train rows={len(tau_anchor_proc)} "
            f"batch_size={tau_loader_kwargs['batch_size']} "
            f"mc_samples={int(args.tau_anchor_n_samples)}",
            flush=True,
        )
        with _StageTimer(f"FIT Seq-LoRA tau on {args.task}(train_anchor)"):
            tau_fit_info = fit_seq_lora_tau_anchor_kl_direct(
                model=model,
                anchor_loader=tau_anchor_loader,
                device=device,
                amp_dtype=amp_dtype,
                num_classes=num_classes,
                choice_token_ids=choice_token_ids,
                lora_cache=lora_cache,
                x_samples_T=x_samples_T,
                tau_max=float(args.tau_search_max),
                search_iters=int(args.tau_search_iters),
                anchor_n_samples=int(args.tau_anchor_n_samples),
                temp_bayes=float(args.temp_bayes),
                mc_eval_chunk=int(args.mc_eval_chunk),
                apply_choice_mask=apply_choice_mask,
                acc_tolerance=float(args.tau_acc_tolerance),
                kl_target_low=float(args.tau_kl_target_low),
                kl_target_high=float(args.tau_kl_target_high),
            )
        effective_posterior_tau = float(tau_fit_info["optimal_posterior_tau"])
        del tau_anchor_loader, tau_anchor_eval, tau_anchor_proc, tau_fit_info
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[Tau] mode={args.tau_mode} effective_posterior_tau={effective_posterior_tau:.6f}", flush=True)

    def _load_eval_proc(eval_task: str):
        eval_raw = load_iid_test_set(eval_task) if eval_task == args.task else load_eval_dataset(eval_task)
        eval_proc = preprocess_task(
            eval_task,
            eval_raw,
            tokenizer,
            args.max_seq_len,
            pad_to_max_length=False,
        )
        eval_proc = eval_proc.add_column("seq_len", [len(x) for x in eval_proc["input_ids"]])
        return eval_proc.sort("seq_len")

    def eval_one(tag: str, eval_task: str):
        proc_or_loader = _load_eval_proc(eval_task)
        proc_eval = (
            proc_or_loader.remove_columns(["seq_len"])
            if "seq_len" in proc_or_loader.column_names
            else proc_or_loader
        )
        loader = DataLoader(proc_eval, **eval_loader_kwargs)

        with _StageTimer(f"INFER Seq-LoRA on {tag}"):
            metrics = eval_bayes_fast_restricted_4way_probmean(
                model=model,
                loader=loader,
                device=device,
                amp_dtype=amp_dtype,
                num_classes=num_classes,
                choice_token_ids=choice_token_ids,
                lora_cache=lora_cache,
                x_samples_T=x_samples_T,
                posterior_scale_tau=effective_posterior_tau,
                temp_bayes=args.temp_bayes,
                max_mc_samples=args.mc_eval_samples,
                mc_eval_chunk=args.mc_eval_chunk,
                progress_desc=f"SEQ {tag}",
                apply_choice_mask=apply_choice_mask,
            )

        print(f"\n[{tag}]\n  ===== Bayesian (Seq-LoRA) Only =====")
        print(f"  nll_bayes: {metrics['nll_bayes']:.4f}")
        print(f"  brier_bayes: {metrics['brier_bayes']:.4f}")
        print(f"  ece_bayes: {metrics['ece_bayes']*100:.2f}%")
        print(f"  acc_bayes: {metrics['acc_bayes']*100:.2f}%")
        print(f"  kl_map_to_bayes: {metrics['kl_map_to_bayes']:.6f}")
        print(f"  posterior_tau: {effective_posterior_tau:.6f}")
        if "past_rate" in metrics:
            print(f"  past_rate: {metrics['past_rate']*100:.2f}%")
            print(f"  future_rate: {metrics['future_rate']*100:.2f}%")
            print(f"  irrelevant_rate: {metrics['irrelevant_rate']*100:.2f}%")
        print(f"  [Timing] Bayes sampling: {metrics['time_bayes_sec']:.3f}s")

        del loader, proc_eval, proc_or_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Evaluation: source={args.task} | targets={eval_tasks} ===")
    for eval_task in eval_tasks:
        split_name = "iid" if eval_task == args.task else "ood"
        eval_one(f"{eval_task}_{split_name}", eval_task)
    print(f"\n[Done] Evaluation complete for source task {args.task}.")

if __name__ == "__main__":
    main()
