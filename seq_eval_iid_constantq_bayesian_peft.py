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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import datasets as hf_datasets
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm

from laplace.curvature.asdl import AsdlGGN
from lssm_ffbs_obs import kalman_filter, lag_one_smoothed_covariances, rts_smoother
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

# =========================
# Config Defaults
# =========================

SEED = 0
TRUST_REMOTE_CODE = False

MAX_SEQ_LEN = 300
EVAL_BSZ = 48
KFAC_BSZ = 4

# KFAC / train-slice loaders remain conservative
NUM_WORKERS = 0

# Eval loader gets its own workers for dynamic padding pipeline
EVAL_NUM_WORKERS = 0
EVAL_PREFETCH_FACTOR = 4

N_KFAC = 8
LR_THRESHOLD = 256
MAX_KFAC_SAMPLES_PER_SLICE = -1

MU_OBS_SCALE = 2
MU_OBS_BATCHES = 32
S_Q = 1.0
Q_MODE = "module_constant"
P1_VAR = 1.0

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 32

ADAPTIVE_Q_WARMSTART_VAR = 1.0
ADAPTIVE_Q_EIG_FLOOR = 1e-8

POSTERIOR_TAU = 1
TEMP_BAYES = 1.0
DISABLE_DROPOUT_DURING_KFAC_MU = False

TOKENIZER_PADDING_SIDE = "left"
BAYESIAN_PEFT_ADD_EOS = False
IID_EVAL_SPLIT = "validation"
BAYESIAN_PEFT_PERTURB_LM_HEAD = True
HF_DATASETS_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_datasets")

from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    SCIENCEQA_GRADE12_TASK_NAME,
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
    _get_num_classes_for_protocol,
    _get_target_token_ids_for_protocol,
    _is_bayesian_peft_protocol,
    _load_adapter_checkpoint,
    _normalize_eval_protocol,
    _preprocess_task_for_protocol,
    _remap_bayesian_peft_adapter_keys,
)

_BAYESIAN_PEFT_ROOT = os.path.join(os.path.dirname(__file__), "bayesian-peft")
if _BAYESIAN_PEFT_ROOT not in sys.path:
    sys.path.append(_BAYESIAN_PEFT_ROOT)

from dataset.utils import dsets as bayesian_peft_dsets


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


def _filter_bayes_module_names(
    module_names: List[str],
    *,
    eval_protocol: str,
    perturb_lm_head: bool,
) -> List[str]:
    if not _is_bayesian_peft_protocol(eval_protocol):
        return module_names
    if perturb_lm_head:
        return module_names
    filtered = [name for name in module_names if ".lm_head." not in name]
    if not filtered:
        raise RuntimeError(
            "All Bayesian modules were filtered out after disabling lm_head posterior perturbation."
        )
    removed = len(module_names) - len(filtered)
    print(
        f"[Posterior] Keeping {len(filtered)} Bayesian modules after excluding "
        f"{removed} lm_head module(s) from posterior perturbation."
    )
    return filtered

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
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(f"[PEAK] {self.tag}: alloc={_peak_alloc_gb():.2f} GB  reserved={_peak_reserved_gb():.2f} GB")


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


def _bayesian_peft_dataset_name(task: str) -> Optional[str]:
    mapping = {
        "wgs": "winogrande_s",
        "wgm": "winogrande_m",
        "boolq": "boolq",
        "obqa": "obqa",
        "arc-e": "ARC-Easy",
        "arc-c": "ARC-Challenge",
    }
    return mapping.get(task)


def _uses_direct_bayesian_peft_data(task: str, eval_protocol: str) -> bool:
    return _is_bayesian_peft_protocol(eval_protocol) and _bayesian_peft_dataset_name(task) is not None


def _build_bayesian_peft_task_dataset(
    tokenizer,
    task: str,
    *,
    add_space: bool,
    max_seq_len: int,
):
    dataset_name = _bayesian_peft_dataset_name(task)
    if dataset_name is None:
        raise ValueError(f"Task '{task}' does not have a direct bayesian-peft dataset wrapper.")
    if dataset_name.startswith("winogrande"):
        return bayesian_peft_dsets.winogrande(
            tokenizer,
            add_space=add_space,
            name=dataset_name,
            max_seq_len=max_seq_len,
        )
    if dataset_name.startswith("ARC"):
        return bayesian_peft_dsets.arc(
            tokenizer,
            add_space=add_space,
            name=dataset_name,
            max_seq_len=max_seq_len,
        )
    if dataset_name == "obqa":
        return bayesian_peft_dsets.obqa(
            tokenizer,
            add_space=add_space,
            max_seq_len=max_seq_len,
        )
    if dataset_name == "boolq":
        return bayesian_peft_dsets.boolq(
            tokenizer,
            add_space=add_space,
            max_seq_len=max_seq_len,
        )
    raise ValueError(f"Unhandled direct bayesian-peft dataset name: {dataset_name}")


class _BayesianPeftCLMCollator:
    def __init__(self, task_dataset):
        self.task_dataset = task_dataset

    def __call__(self, batch):
        prompts, classes, _targets = self.task_dataset.clm_collate_fn(batch)
        return {
            "input_ids": prompts["input_ids"],
            "attention_mask": prompts["attention_mask"],
            "labels": classes.to(dtype=torch.long),
        }


def _make_direct_bayesian_peft_loader(
    raw_dataset,
    *,
    collate_fn,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(raw_dataset, **kwargs)

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

    def __init__(self, peft_model: nn.Module, forward_call):
        super().__init__()
        self.peft_model = peft_model
        closure = inspect.getclosurevars(forward_call).nonlocals
        self.amp_dtype = closure.get("amp_dtype", torch.float16)
        self.choice_token_ids = closure.get("choice_token_ids")
        self.apply_choice_mask = bool(closure.get("apply_choice_mask", False))

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
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """ASDL Kron extraction over the full wrapper graph."""
    del target_module_keywords, exclude_bias

    if not hasattr(loader, "dataset"):
        raise ValueError("ASDL Kron extraction requires loader.dataset to infer N.")

    device = next(model.parameters()).device
    N = len(loader.dataset)
    if N <= 0:
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


def _adaptive_q_eps(dtype: torch.dtype) -> float:
    return 1e-8 if dtype == torch.float64 else 1e-6


def _symmetrize(M: Tensor) -> Tensor:
    return 0.5 * (M + M.T)


def _relative_floor_psd_eigs(evals: Tensor, eps_rel: float) -> Tensor:
    evals = evals.clamp_min(0.0)
    if eps_rel <= 0.0:
        return evals
    scale = torch.amax(evals)
    return evals.clamp_min(scale * float(eps_rel))


@torch.no_grad()
def _build_module_q_basis(
    H_x_list: List[Tensor],
    *,
    eps_rel: float,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    if len(H_x_list) == 0:
        raise ValueError("H_x_list must be non-empty")
    H_mean = torch.zeros_like(H_x_list[0], dtype=dtype)
    for H_x in H_x_list:
        H_mean.add_(H_x.to(dtype=dtype))
    H_mean = _symmetrize(H_mean / float(len(H_x_list)))
    nu_bar, U_q = torch.linalg.eigh(H_mean)
    nu_bar = _relative_floor_psd_eigs(nu_bar, eps_rel)
    return U_q.to(dtype=dtype), nu_bar.to(dtype=dtype)


@torch.no_grad()
def _estimate_module_q_diag(
    x_smooth: List[Tensor],
    P_smooth: List[Tensor],
    lag_covariances: List[Tensor],
    U_q: Tensor,
    nu_bar: Tensor,
    *,
    eps: float,
) -> Tuple[Tensor, Dict[str, float]]:
    T = len(x_smooth)
    L = x_smooth[0].numel()
    dtype = x_smooth[0].dtype
    device = x_smooth[0].device

    nu_safe = nu_bar.to(device=device, dtype=dtype).clamp_min(float(eps))

    if T <= 1:
        q_prior = torch.ones(L, device=device, dtype=dtype)
        return q_prior, {
            "alpha_mix": 0.0,
            "q_em_min": 1.0,
            "q_em_mean": 1.0,
            "q_em_max": 1.0,
            "q_prior_min": 1.0,
            "q_prior_mean": 1.0,
            "q_prior_max": 1.0,
            "q_diag_min": 1.0,
            "q_diag_mean": 1.0,
            "q_diag_max": 1.0,
            "r_eff": float(L),
        }

    U_q_t = U_q.to(device=device, dtype=dtype)

    mu_basis = [U_q_t.T @ x_t.to(device=device, dtype=dtype) for x_t in x_smooth]
    diag_basis = []
    cross_diag = [torch.zeros(L, device=device, dtype=dtype)]

    for P_t in P_smooth:
        P_basis = U_q_t.T @ P_t.to(device=device, dtype=dtype) @ U_q_t
        diag_basis.append(torch.diagonal(_symmetrize(P_basis)))
    for t in range(1, T):
        C_basis = U_q_t.T @ lag_covariances[t].to(device=device, dtype=dtype) @ U_q_t
        cross_diag.append(torch.diagonal(C_basis))

    q_em_terms: List[Tensor] = []
    for t in range(1, T):
        delta_mu = mu_basis[t] - mu_basis[t - 1]
        term_t = (
            delta_mu.square()
            + diag_basis[t]
            + diag_basis[t - 1]
            - 2.0 * cross_diag[t]
        )
        q_em_terms.append(term_t)

    q_em = torch.stack(q_em_terms, dim=0).mean(dim=0).clamp_min(float(eps))
    kappa = torch.median((nu_safe * q_em).clamp_min(float(eps)))
    q_prior = (kappa / nu_safe).clamp_min(float(eps))

    r_eff = (nu_safe.sum().square() / nu_safe.square().sum().clamp_min(float(eps))).clamp_min(1.0)
    alpha_mix = float((T - 1) / ((T - 1) + float(r_eff.item())))
    q_diag = (alpha_mix * q_em + (1.0 - alpha_mix) * q_prior).clamp_min(float(eps))

    def _summ(x: Tensor, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_min": float(x.min().item()),
            f"{prefix}_mean": float(x.mean().item()),
            f"{prefix}_max": float(x.max().item()),
        }

    stats: Dict[str, float] = {
        "alpha_mix": float(alpha_mix),
        "r_eff": float(r_eff.item()),
    }
    stats.update(_summ(q_em, "q_em"))
    stats.update(_summ(q_prior, "q_prior"))
    stats.update(_summ(q_diag, "q_diag"))
    return q_diag, stats


@torch.no_grad()
def materialize_constant_module_Q_list(
    U_q: Tensor,
    q_diag: Tensor,
    num_steps: int,
    *,
    s_q: float,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    U_t = U_q.to(device=device, dtype=dtype)
    q_t = q_diag.to(device=device, dtype=dtype)
    base_Q = (U_t * (float(s_q) * q_t).unsqueeze(0)) @ U_t.T
    base_Q = _symmetrize(base_Q)
    return [base_Q.clone() for _ in range(num_steps)]

def estimate_mu_global_list_from_slice_grads(
    model: nn.Module,
    slice_loaders: List[DataLoader],
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

            for mi, name in enumerate(module_names):
                w = _get_param_weight(model, name)
                g_x_parts[mi] += (
                    module_subspace_info[name]["U_lora"].to(device=device, dtype=dtype).T
                    @ w.grad.detach().to(dtype=dtype).reshape(-1)
                )
            n_seen += 1

        if n_seen == 0:
            raise RuntimeError(f"[mu-obs] slice {t} loader produced no batches")

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

def _report_module_constant_q_results(module_stats: Dict[str, Dict[str, float]], s_q: float) -> None:
    if not module_stats:
        print("[Module Q] No module statistics available.")
        return

    def _summ(key: str) -> str:
        vals = [float(stats[key]) for stats in module_stats.values()]
        return f"min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f}"

    print("\n=== Module-Constant Q Report ===")
    print(f"[Module Q] exposed scale s_Q={float(s_q):.6f}")
    print(
        f"[Module Q Summary] "
        f"alpha({_summ('alpha_mix')}) | "
        f"qdiag({_summ('q_diag_mean')})"
    )
    print("[Module Q Per Module]")
    for name, stats in module_stats.items():
        print(
            f"  {name}: "
            f"alpha={stats['alpha_mix']:.6f} "
            f"r_eff={stats['r_eff']:.3f} "
            f"nu=[{stats['nu_min']:.6f}, {stats['nu_mean']:.6f}, {stats['nu_max']:.6f}] "
            f"qdiag=[{stats['q_diag_min']:.6f}, {stats['q_diag_mean']:.6f}, {stats['q_diag_max']:.6f}]"
        )


def _report_scalar_constant_q_results(s_q: float) -> None:
    print("\n=== Scalar-Constant Q Report ===")
    print(f"[Constant Q] mode=constant  shared Q_t = s_Q * I with s_Q={float(s_q):.6f}")

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

    scale = float(posterior_scale_tau) / math.sqrt(max(len(lora_cache), 1))
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


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Seq-LoRA on various tasks with selectable process-noise Q modes.")
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
        help="Global process-noise scale. In module_constant mode it scales the learned per-module Q_m; in constant mode it sets Q_t = s_Q * I.",
    )
    parser.add_argument(
        "--q_mode",
        type=str,
        choices=["module_constant", "constant"],
        default=Q_MODE,
        help="Process-noise mode: learned per-module constant Q_m, or a shared scalar constant Q_t = s_Q * I.",
    )
    parser.add_argument(
        "--constant_q_var",
        dest="s_q",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
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
        "--mc_eval_chunk",
        type=int,
        default=0,
        help="Optional chunk size for MC samples during evaluation. <=0 disables chunking.",
    )
    parser.add_argument("--posterior_tau", type=float, default=POSTERIOR_TAU, help="Posterior scale multiplier used at evaluation.")
    parser.add_argument("--temp_bayes", type=float, default=TEMP_BAYES, help="Temperature applied to Bayesian mean probabilities.")
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
        "--eval_protocol",
        type=str,
        default="bayesian_peft",
        choices=["default", "bayesian_peft"],
        help="Evaluation protocol. bayesian_peft matches the original bayesian-peft prompt/target-id setup.",
    )
    parser.add_argument(
        "--bayesian_peft_add_space",
        type=_parse_bool,
        default=False,
        help="Match bayesian-peft's add_space flag when eval_protocol=bayesian_peft.",
    )
    parser.add_argument(
        "--bayesian_peft_add_eos",
        type=_parse_bool,
        default=BAYESIAN_PEFT_ADD_EOS,
        help="Match bayesian-peft tokenizer.add_eos_token handling when eval_protocol=bayesian_peft.",
    )
    parser.add_argument(
        "--bayesian_peft_perturb_lm_head",
        type=_parse_bool,
        default=BAYESIAN_PEFT_PERTURB_LM_HEAD,
        help=(
            "Whether Seq posterior sampling should perturb lm_head LoRA-A when "
            "eval_protocol=bayesian_peft. Defaults to true to match the full-vocab tfb/blob behavior."
        ),
    )
    parser.add_argument(
        "--iid_eval_split",
        type=str,
        default=IID_EVAL_SPLIT,
        choices=["validation", "test"],
        help="Split used for source-task IID evaluation when eval_protocol=bayesian_peft.",
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
    eval_protocol = _normalize_eval_protocol(args.eval_protocol)
    apply_choice_mask = not _is_bayesian_peft_protocol(eval_protocol)
    keep_full_vocab_lm_head = bool(args.keep_full_vocab_lm_head) or _is_bayesian_peft_protocol(eval_protocol)
    print(f"[Protocol] eval_protocol={eval_protocol}")
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

    use_direct_source_bayesian_peft = _uses_direct_bayesian_peft_data(args.task, eval_protocol)
    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.padding_side = args.tokenizer_padding_side
    if use_direct_source_bayesian_peft:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    if _is_bayesian_peft_protocol(eval_protocol) and hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = bool(args.bayesian_peft_add_eos)
        print(f"[Protocol] tokenizer.add_eos_token={bool(args.bayesian_peft_add_eos)}")

    source_bayesian_peft_dataset = None
    if use_direct_source_bayesian_peft:
        source_bayesian_peft_dataset = _build_bayesian_peft_task_dataset(
            tokenizer,
            args.task,
            add_space=bool(args.bayesian_peft_add_space),
            max_seq_len=args.max_seq_len,
        )
        num_classes = int(source_bayesian_peft_dataset.n_labels)
        choice_token_ids = source_bayesian_peft_dataset.target_ids.view(-1).to(device=device, dtype=torch.long)
        print(f"[Protocol] using direct bayesian-peft dataset wrapper for source task {args.task}")
    else:
        num_classes = _get_num_classes_for_protocol(args.task, eval_protocol)
        choice_token_ids = _get_target_token_ids_for_protocol(
            tokenizer,
            task=args.task,
            protocol=eval_protocol,
            device=device,
            add_space=bool(args.bayesian_peft_add_space),
        )

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
    if _is_bayesian_peft_protocol(eval_protocol):
        print("[Head] keeping full-vocab lm_head (bayesian_peft protocol)")
        model = get_peft_model(base_model, peft_cfg).to(device)
        adapter_state = _load_adapter_checkpoint(args.map_dir)
        adapter_state, num_remapped = _remap_bayesian_peft_adapter_keys(adapter_state)
        print(f"[Adapter] loaded legacy checkpoint keys={len(adapter_state)} remapped={num_remapped}")
        incompat = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
        missing_lora = [k for k in incompat.missing_keys if "lora_" in k]
        unexpected_lora = [k for k in incompat.unexpected_keys if "lora_" in k]
        if missing_lora or unexpected_lora:
            raise RuntimeError(
                f"LoRA load mismatch for {args.map_dir}: "
                f"missing_lora={missing_lora[:8]} unexpected_lora={unexpected_lora[:8]}"
            )
    elif keep_full_vocab_lm_head:
        print("[Head] keeping full-vocab lm_head and slicing choice-token logits dynamically")
        model = PeftModel.from_pretrained(base_model, args.map_dir).to(device)
    else:
        trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
        print(f"[Head] trimmed lm_head to {num_classes} choice logits")
        model = PeftModel.from_pretrained(base_model, args.map_dir).to(device)
    model.eval()

    print("\n[Setup] Casting all LoRA params to float32 for numerical stability...")
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype=torch.float32)
            p.requires_grad = True

    if args.slices_dir:
        ds_slices = load_from_disk(args.slices_dir)
        train_raw = ds_slices["train"]
        slice_source = f"slices_dir={args.slices_dir}"
    else:
        if use_direct_source_bayesian_peft:
            train_raw = source_bayesian_peft_dataset.dset["train"]
        else:
            train_raw, _, _ = load_task_dataset(args.task)
        if int(args.random_num_slices) > 0:
            train_raw = _assign_random_slice_ids(train_raw, int(args.random_num_slices), int(args.seed))
            slice_source = f"task_train_split[random_{int(args.random_num_slices)}_slices_seed_{int(args.seed)}]"
            print(
                f"[Slices] No --slices_dir provided; using balanced random slice ids "
                f"with K={int(args.random_num_slices)} and seed={int(args.seed)}."
            )
        else:
            train_raw = _ensure_slice_ids_for_seq(args.task, train_raw)
            slice_source = "task_train_split"
            if args.task == SCIENCEQA_CURRIC_TASK_NAME:
                print("[Slices] No --slices_dir provided; using ScienceQA train split with grade-based slice ids.")
    print(f"\n[Slices] train={len(train_raw)} source={slice_source} (used for KFAC only)")

    # -------------------------
    # Eval set: dynamic padding + length sorting
    # -------------------------
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    eval_task_to_proc: Dict[str, object] = {}
    for eval_task in eval_tasks:
        use_direct_eval_bayesian_peft = _uses_direct_bayesian_peft_data(eval_task, eval_protocol)
        if use_direct_eval_bayesian_peft:
            eval_task_dataset = _build_bayesian_peft_task_dataset(
                tokenizer,
                eval_task,
                add_space=bool(args.bayesian_peft_add_space),
                max_seq_len=args.max_seq_len,
            )
            eval_num_classes = int(eval_task_dataset.n_labels)
        else:
            eval_num_classes = _get_num_classes_for_protocol(eval_task, eval_protocol)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, "
                f"but source task '{args.task}' has {num_classes} classes."
            )
        if use_direct_eval_bayesian_peft:
            eval_split = str(args.iid_eval_split) if eval_task == args.task else "validation"
            print(f"[Eval split] task={eval_task} split={eval_split}")
            eval_task_to_proc[eval_task] = _make_direct_bayesian_peft_loader(
                eval_task_dataset.dset[eval_split],
                collate_fn=_BayesianPeftCLMCollator(eval_task_dataset),
                batch_size=args.eval_bsz,
                shuffle=False,
                drop_last=True,
                num_workers=args.eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.eval_prefetch_factor,
            )
        else:
            eval_raw = load_iid_test_set(eval_task) if eval_task == args.task else load_eval_dataset(eval_task)
            eval_proc = _preprocess_task_for_protocol(
                eval_task,
                eval_raw,
                tokenizer,
                args.max_seq_len,
                protocol=eval_protocol,
                bayesian_peft_add_space=bool(args.bayesian_peft_add_space),
                pad_to_max_length=False,
            )
            eval_proc = eval_proc.add_column("seq_len", [len(x) for x in eval_proc["input_ids"]])
            eval_task_to_proc[eval_task] = eval_proc.sort("seq_len")

    # -------------------------
    # KFAC/train slices: dynamic padding to reduce wasted compute on long MMLU inputs
    # -------------------------
    direct_bayesian_peft_collator = None
    train_proc = None
    if use_direct_source_bayesian_peft:
        direct_bayesian_peft_collator = _BayesianPeftCLMCollator(source_bayesian_peft_dataset)
    else:
        train_proc = _preprocess_task_for_protocol(
            args.task,
            train_raw,
            tokenizer,
            args.max_seq_len,
            protocol=eval_protocol,
            bayesian_peft_add_space=bool(args.bayesian_peft_add_space),
            pad_to_max_length=False,
        )
        if "slice_id" not in train_proc.column_names:
            train_proc = train_proc.map(
                lambda ex, idx: {"slice_id": int(train_raw[idx]["slice_id"])},
                with_indices=True,
            )
        if "seq_len" not in train_proc.column_names:
            train_proc = train_proc.add_column("seq_len", [len(x) for x in train_proc["input_ids"]])

    slice_ids = sorted(set(int(x) for x in train_raw["slice_id"]))
    T = len(slice_ids)
    print(f"\n[Curvature] unique slice_ids={slice_ids} (T={T})")

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
    for sid in slice_ids:
        if use_direct_source_bayesian_peft:
            slice_indices = [idx for idx, ex in enumerate(train_raw) if int(ex["slice_id"]) == sid]
            if max_kfac_samples_per_slice is not None and len(slice_indices) > max_kfac_samples_per_slice:
                rng = random.Random(42)
                rng.shuffle(slice_indices)
                slice_indices = slice_indices[:max_kfac_samples_per_slice]
            eff_batches = len(slice_indices) // args.kfac_bsz
            eff_samples = eff_batches * args.kfac_bsz
            total_kfac_samples += eff_samples
            total_kfac_batches += eff_batches
            print(
                f"[KFAC slice] sid={sid} raw={len(slice_indices)} "
                f"eff_samples={eff_samples} batches={eff_batches}"
            )
            ds_loader = Subset(train_raw, slice_indices[:eff_samples])
            slice_loaders.append(
                _make_direct_bayesian_peft_loader(
                    ds_loader,
                    collate_fn=direct_bayesian_peft_collator,
                    batch_size=args.kfac_bsz,
                    shuffle=False,
                    drop_last=True,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                    prefetch_factor=args.eval_prefetch_factor,
                )
            )
            continue

        ds_t = train_proc.filter(lambda ex, sid=sid: int(ex["slice_id"]) == sid)
        if max_kfac_samples_per_slice is not None and len(ds_t) > max_kfac_samples_per_slice:
            ds_t = ds_t.shuffle(seed=42).select(range(max_kfac_samples_per_slice))
        ds_t = ds_t.sort("seq_len")
        eff_batches = len(ds_t) // args.kfac_bsz
        eff_samples = eff_batches * args.kfac_bsz
        total_kfac_samples += eff_samples
        total_kfac_batches += eff_batches
        print(
            f"[KFAC slice] sid={sid} raw={len(ds_t)} "
            f"eff_samples={eff_samples} batches={eff_batches}"
        )
        ds_loader = ds_t.remove_columns(["seq_len"]) if "seq_len" in ds_t.column_names else ds_t
        slice_loaders.append(
            DataLoader(
                ds_loader,
                batch_size=args.kfac_bsz,
                shuffle=False,
                drop_last=True,
                collate_fn=kfac_collator,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
        )
    print(
        f"[KFAC] effective samples after per-slice cap/drop_last = {total_kfac_samples} "
        f"across {total_kfac_batches} batches"
    )

    forward_call_for_kfac = forward_call_for_kfac_factory(
        amp_dtype,
        choice_token_ids,
        apply_choice_mask=apply_choice_mask,
    )
    H_factor_per_module, G_factor_per_module, module_names = {}, {}, None

    dropout_ctx = (
        _temporarily_disable_dropout_modules(model)
        if bool(args.disable_dropout_during_kfac_mu)
        else nullcontext()
    )

    with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior build on {args.task}"), dropout_ctx:
        print("\n=== Running KFAC on seq slices (targets=all LoRA-A modules in adapter) ===")
        for t_idx, loader_t in enumerate(slice_loaders):
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
                    use_tqdm=True,
                )

            if module_names is None:
                module_names = _resolve_bayes_module_names(factors)
                module_names = _filter_bayes_module_names(
                    module_names,
                    eval_protocol=eval_protocol,
                    perturb_lm_head=bool(args.bayesian_peft_perturb_lm_head),
                )
                print(f"[Seq-LoRA] Resolved Bayesian modules: {len(module_names)}")
                for name in module_names:
                    print(f"  - {name}")
                for n in module_names:
                    H_factor_per_module[n], G_factor_per_module[n] = [], []

            for name in module_names:
                A_t, S_t = factors[name]
                # Retain the compact KFAC factors instead of materializing
                # full H_t / G_t on the CPU. We only expand/project later when
                # compressing into the small Seq-LoRA subspace.
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

        module_subspace_info, module_R_lists, module_Hx_lists = {}, {}, {}
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
            module_Hx_lists[name] = H_x_list
            module_R_lists[name] = R_list

            H_factors.clear()
            G_factors.clear()
            del H_bar_bal, G_bar_bal, subspace_info_gpu
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
        L_total = offset
        lora_cache = build_loraA_cache(model, module_specs, device=device)

        mu_global_list_raw = estimate_mu_global_list_from_slice_grads(
            model,
            slice_loaders,
            forward_call_for_kfac,
            module_names,
            module_subspace_info,
            module_R_lists,
            device,
            args.mu_obs_batches,
            torch.float64,
        )
        mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]

        print(f"\n=== Kalman Filter Only (module-wise) ===")
        print(f"[Kalman] modules={len(module_specs)} L_total={L_total}")

        if args.forecast_horizon > 0:
            print(
                f"\nDirectly sampling forecast posterior (t=T+{args.forecast_horizon} | T): "
                f"S={args.mc_eval_samples}"
            )
        else:
            print(f"\nDirectly sampling final posterior (t=T): S={args.mc_eval_samples}")

        x_sample_parts: List[Tensor] = []
        module_constant_q_stats: Dict[str, Dict[str, float]] = {}
        use_module_constant_q = args.q_mode == "module_constant"
        for spec in module_specs:
            name = spec["name"]
            offset = int(spec["offset"])
            Lm = int(spec["L"])
            mu_module_list = [
                mu_t[offset : offset + Lm].to(device=cpu_device, dtype=torch.float64)
                for mu_t in mu_global_list
            ]
            H_x_list = module_Hx_lists[name]
            H_obs_list, y_list = prepare_lgssm_observations(
                module_R_lists[name],
                mu_list=mu_module_list,
            )
            m1 = torch.zeros(Lm, device=cpu_device, dtype=torch.float64)
            P1 = float(args.p1_var) * torch.eye(Lm, device=cpu_device, dtype=torch.float64)

            if use_module_constant_q:
                U_q, nu_bar = _build_module_q_basis(
                    H_x_list,
                    eps_rel=ADAPTIVE_Q_EIG_FLOOR,
                    dtype=torch.float64,
                )

                warm_Q_list = materialize_scalar_Q_list(
                    [ADAPTIVE_Q_WARMSTART_VAR for _ in range(T)],
                    L=Lm,
                    device=cpu_device,
                    dtype=torch.float64,
                )
                x_filt_w, P_filt_w, x_pred_w, P_pred_w = kalman_filter(
                    H_list=H_obs_list,
                    y_list=y_list,
                    Q_list=warm_Q_list,
                    m1=m1,
                    P1=P1,
                )
                x_smooth_w, P_smooth_w, J_w = rts_smoother(
                    x_filt_w,
                    P_filt_w,
                    x_pred_w,
                    P_pred_w,
                    warm_Q_list,
                )
                lag_cov_w = lag_one_smoothed_covariances(P_smooth_w, J_w)
                q_diag, q_stats = _estimate_module_q_diag(
                    x_smooth_w,
                    P_smooth_w,
                    lag_cov_w,
                    U_q,
                    nu_bar,
                    eps=_adaptive_q_eps(torch.float64),
                )
                Q_list = materialize_constant_module_Q_list(
                    U_q,
                    q_diag,
                    num_steps=T,
                    s_q=float(args.s_q),
                    device=cpu_device,
                    dtype=torch.float64,
                )
                module_constant_q_stats[name] = {
                    **q_stats,
                    "nu_min": float(nu_bar.min().item()),
                    "nu_mean": float(nu_bar.mean().item()),
                    "nu_max": float(nu_bar.max().item()),
                }
            else:
                Q_list = materialize_scalar_Q_list(
                    [float(args.s_q) for _ in range(T)],
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
            if use_module_constant_q:
                del (
                    warm_Q_list,
                    x_filt_w,
                    P_filt_w,
                    x_pred_w,
                    P_pred_w,
                    x_smooth_w,
                    P_smooth_w,
                    J_w,
                    lag_cov_w,
                    q_diag,
                )
            gc.collect()

        if use_module_constant_q:
            _report_module_constant_q_results(module_constant_q_stats, args.s_q)
        else:
            _report_scalar_constant_q_results(args.s_q)
        del mu_global_list, mu_global_list_raw
        x_samples_T = torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)
        del x_sample_parts

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

    def eval_one(tag: str, proc_or_loader):
        if isinstance(proc_or_loader, DataLoader):
            loader = proc_or_loader
        else:
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
        print(f"  posterior_tau_used: {effective_posterior_tau:.8f}")
        print(
            "  [bayesian-peft style] "
            f"val_acc: {metrics['acc_bayes']}, "
            f"val_ece: {metrics['ece_bayes']}, "
            f"val_nll: {metrics['nll_bayes']}, "
            f"val_brier: {metrics['brier_bayes']}"
        )
        if "past_rate" in metrics:
            print(f"  past_rate: {metrics['past_rate']*100:.2f}%")
            print(f"  future_rate: {metrics['future_rate']*100:.2f}%")
            print(f"  irrelevant_rate: {metrics['irrelevant_rate']*100:.2f}%")
        print(f"  [Timing] Bayes sampling: {metrics['time_bayes_sec']:.3f}s")

    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Evaluation: source={args.task} | targets={eval_tasks} ===")
    for eval_task in eval_tasks:
        split_name = "iid" if eval_task == args.task else "ood"
        eval_one(f"{eval_task}_{split_name}", eval_task_to_proc[eval_task])
    print(f"\n[Done] Evaluation complete for source task {args.task}.")

if __name__ == "__main__":
    main()
