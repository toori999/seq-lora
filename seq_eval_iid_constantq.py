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
from torch.utils.data import DataLoader, Subset

import datasets as hf_datasets
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm

from laplace.curvature.asdl import AsdlGGN, batch_gradient as asdl_batch_gradient
import asdl.operations.linear as asdl_linear_ops
import kfac as hook_kfac
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


_POSTERIOR_CACHE_FORMAT = "seq_lora_posterior_cache_v1"
_POSTERIOR_STATS_CACHE_FORMAT = "seq_lora_posterior_stats_cache_v1"


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


def _serialize_module_specs_cpu(module_specs: List[Dict]) -> List[Dict[str, object]]:
    cpu_module_specs: List[Dict[str, object]] = []
    for spec in module_specs:
        subspace_info_cpu = {}
        for key, value in spec["subspace_info"].items():
            if torch.is_tensor(value):
                subspace_info_cpu[key] = value.detach().cpu()
            else:
                subspace_info_cpu[key] = value
        cpu_module_specs.append(
            {
                "name": str(spec["name"]),
                "offset": int(spec["offset"]),
                "L": int(spec["L"]),
                "subspace_info": subspace_info_cpu,
            }
        )
    return cpu_module_specs


def _build_posterior_stats_snapshot(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "task": str(args.task),
        "map_dir": os.path.abspath(str(args.map_dir)),
        "slices_dir": os.path.abspath(str(args.slices_dir)) if str(args.slices_dir) else "",
        "seed": int(args.seed),
        "kfac_backend": str(args.kfac_backend),
        "random_num_slices": int(args.random_num_slices),
        "subspace_dim_per_module": int(args.subspace_dim_per_module),
        "max_seq_len": int(args.max_seq_len),
        "kfac_bsz": int(args.kfac_bsz),
        "n_kfac": int(args.n_kfac),
        "lr_threshold": int(args.lr_threshold),
        "max_kfac_samples_per_slice": int(args.max_kfac_samples_per_slice),
        "mu_obs_batches": int(args.mu_obs_batches),
        "disable_dropout_during_kfac_mu": bool(args.disable_dropout_during_kfac_mu),
        "tokenizer_padding_side": str(args.tokenizer_padding_side),
        "kfac_token_mode": str(args.kfac_token_mode),
        "eval_protocol": str(args.eval_protocol),
        "bayesian_peft_add_space": bool(args.bayesian_peft_add_space),
        "bayesian_peft_add_eos": bool(args.bayesian_peft_add_eos),
        "bayesian_peft_perturb_lm_head": bool(args.bayesian_peft_perturb_lm_head),
    }


def _build_posterior_cache_snapshot(args: argparse.Namespace) -> Dict[str, object]:
    args_snapshot = _build_posterior_stats_snapshot(args)
    args_snapshot.update(
        {
            "q_mode": str(args.q_mode),
            "module_q_estimator": "lognormal_mu_drift_inverse_curvature_shape_v1",
            "module_q_shrink_exponent": float(
                min(max(float(args.module_q_shrink_exponent), 0.0), 1.0)
            ),
            "module_q_clip_min": float(args.module_q_clip_min),
            "module_q_clip_max": float(args.module_q_clip_max),
            "module_q_shape_power": float(min(max(float(args.module_q_shape_power), 0.0), 1.0)),
            "module_q_shape_eps_rel": float(max(float(args.module_q_shape_eps_rel), 0.0)),
            "gap_q_clip_min": float(args.gap_q_clip_min),
            "gap_q_clip_max": float(args.gap_q_clip_max),
            "s_q": float(args.s_q),
            "mu_obs_scale": float(args.mu_obs_scale),
            "forecast_horizon": int(args.forecast_horizon),
            "mc_eval_samples": int(args.mc_eval_samples),
            "p1_var": float(args.p1_var),
        }
    )
    return args_snapshot


def _build_posterior_stats_cache_payload(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    l_total: int,
    num_modules: int,
) -> Dict[str, object]:
    return {
        "format": _POSTERIOR_STATS_CACHE_FORMAT,
        "args_snapshot": _build_posterior_stats_snapshot(args),
        "module_specs": _serialize_module_specs_cpu(module_specs),
        "module_R_lists": {
            str(name): [tensor.detach().cpu() for tensor in tensors]
            for name, tensors in module_R_lists.items()
        },
        "mu_global_list_raw": [mu_t.detach().cpu() for mu_t in mu_global_list_raw],
        "l_total": int(l_total),
        "num_modules": int(num_modules),
    }


def _build_posterior_cache_payload(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    x_samples_T: Tensor,
    l_total: int,
    num_modules: int,
) -> Dict[str, object]:
    return {
        "format": _POSTERIOR_CACHE_FORMAT,
        "args_snapshot": _build_posterior_cache_snapshot(args),
        "module_specs": _serialize_module_specs_cpu(module_specs),
        "x_samples_T": x_samples_T.detach().cpu(),
        "l_total": int(l_total),
        "num_modules": int(num_modules),
    }


def _validate_cache_snapshot(
    *,
    cache_path: str,
    snapshot: Dict[str, object],
    expected: Dict[str, object],
    kind: str,
) -> None:
    mismatches = [
        f"{key}: cache={snapshot.get(key)!r} current={expected[key]!r}"
        for key in expected
        if snapshot.get(key) != expected[key]
    ]
    if mismatches:
        mismatch_text = "\n".join(mismatches[:12])
        raise RuntimeError(
            f"{kind} cache at {cache_path} does not match current run configuration:\n{mismatch_text}"
        )


def _load_posterior_stats_cache(
    cache_path: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict[str, List[Tensor]], List[Tensor], int, int]:
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
        expected=_build_posterior_stats_snapshot(args),
        kind="Posterior-stats",
    )

    module_specs = payload.get("module_specs")
    module_R_lists = payload.get("module_R_lists")
    mu_global_list_raw = payload.get("mu_global_list_raw")
    l_total = int(payload.get("l_total"))
    num_modules = int(payload.get("num_modules"))
    if (
        not isinstance(module_specs, list)
        or not isinstance(module_R_lists, dict)
        or not isinstance(mu_global_list_raw, list)
    ):
        raise RuntimeError(f"Posterior-stats cache at {cache_path} is missing required tensors.")
    return module_specs, module_R_lists, mu_global_list_raw, l_total, num_modules


def _load_posterior_cache(
    cache_path: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Tensor, int, int]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != _POSTERIOR_CACHE_FORMAT:
        raise RuntimeError(
            f"Unsupported posterior cache format in {cache_path}. "
            f"Expected format={_POSTERIOR_CACHE_FORMAT!r}."
        )

    snapshot = dict(payload.get("args_snapshot") or {})
    _validate_cache_snapshot(
        cache_path=cache_path,
        snapshot=snapshot,
        expected=_build_posterior_cache_snapshot(args),
        kind="Posterior",
    )

    module_specs = payload.get("module_specs")
    x_samples_T = payload.get("x_samples_T")
    l_total = int(payload.get("l_total"))
    num_modules = int(payload.get("num_modules"))
    if not isinstance(module_specs, list) or not torch.is_tensor(x_samples_T):
        raise RuntimeError(f"Posterior cache at {cache_path} is missing required tensors.")
    return module_specs, x_samples_T, l_total, num_modules

# =========================
# Config Defaults
# =========================

SEED = 0
TRUST_REMOTE_CODE = True

MAX_SEQ_LEN = 300
EVAL_BSZ = 64
KFAC_BSZ = 4

# KFAC / train-slice loaders remain conservative
NUM_WORKERS = 0

# Eval loader gets its own workers for dynamic padding pipeline
EVAL_NUM_WORKERS = 0
EVAL_PREFETCH_FACTOR = 4

N_KFAC = 8
LR_THRESHOLD = 256
MAX_KFAC_SAMPLES_PER_SLICE = -1
KFAC_BACKEND = "hook"
KFAC_TOKEN_MODE = "all_valid"

MU_OBS_SCALE = 1.0
MU_OBS_BATCHES = 32
S_Q = 1.0
P1_VAR = 1.0
Q_MODE = "module_constant"
MODULE_Q_CLIP_MIN = 0.5
MODULE_Q_CLIP_MAX = 2.0
GAP_Q_CLIP_MIN = 0.8
GAP_Q_CLIP_MAX = 1.2
MODULE_Q_SHRINK_EXPONENT = 0.05
MODULE_Q_SHAPE_POWER = 0.25
MODULE_Q_SHAPE_EPS_REL = 1e-3

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 32

POSTERIOR_TAU = 0.65
TEMP_BAYES = 1.0
DISABLE_DROPOUT_DURING_KFAC_MU = False

TOKENIZER_PADDING_SIDE = "left"
BAYESIAN_PEFT_ADD_EOS = False
IID_EVAL_SPLIT = "validation"
BAYESIAN_PEFT_PERTURB_LM_HEAD = True
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
    kfac_token_mode: str,
):
    def _forward_call(model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")
        hook_kfac._CURRENT_LAST_IDX = _left_padded_last_idx(input_ids)
        if str(kfac_token_mode) == "all_valid":
            hook_kfac._CURRENT_TOKEN_MODE = "all_valid"
            hook_kfac._CURRENT_TOKEN_MASK = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        else:
            hook_kfac._CURRENT_TOKEN_MODE = "last"
            hook_kfac._CURRENT_TOKEN_MASK = None

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


def calculate_kronecker_factors_hook(
    model: nn.Module,
    forward_call,
    loader: DataLoader,
    n_kfac: int | None = None,
    lr_threshold: int = 512,
    target_module_keywords: list[str] | None = None,
    exclude_bias: bool = False,
    use_tqdm: bool = False,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    return hook_kfac.calculate_kronecker_factors(
        model=model,
        forward_call=forward_call,
        loader=loader,
        n_kfac=n_kfac,
        lr_threshold=lr_threshold,
        target_module_keywords=(target_module_keywords or [""]),
        exclude_bias=exclude_bias,
        use_tqdm=use_tqdm,
    )


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


@torch.no_grad()
def materialize_shaped_Q_list(
    var_list: Sequence[float],
    q_shape: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    S = q_shape.to(device=device, dtype=dtype)
    S = 0.5 * (S + S.T)
    return [float(var) * S for var in var_list]


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
        if set(asdl_module_names) != set(module_names):
            missing = sorted(set(module_names) - set(asdl_module_names))
            extra = sorted(set(asdl_module_names) - set(module_names))
            raise RuntimeError(
                "ASDL gx module-name mismatch. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

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


def _build_module_delta_q_scales(
    module_specs: List[Dict],
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
        diffs = []
        for t in range(1, len(mu_global_list_raw)):
            delta = mu_global_list_raw[t][offset : offset + Lm] - mu_global_list_raw[t - 1][offset : offset + Lm]
            diffs.append(float(delta.pow(2).mean().item()))
        raw_vals.append(float(torch.tensor(diffs, dtype=torch.float64).median().item()) if diffs else 1.0)
        names.append(name)

    scales = _normalize_and_clip_scales(
        raw_vals,
        clip_min=clip_min,
        clip_max=clip_max,
        shrink_exponent=shrink_exponent,
    )
    return {name: scale for name, scale in zip(names, scales)}


def _build_module_constant_q_scales(
    module_specs: List[Dict],
    mu_global_list_raw: List[Tensor],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float,
) -> Dict[str, float]:
    return _build_module_delta_q_scales(
        module_specs,
        mu_global_list_raw,
        clip_min=clip_min,
        clip_max=clip_max,
        shrink_exponent=shrink_exponent,
    )


def _identity_q_shape(L: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(int(L), device=device, dtype=dtype)


@torch.no_grad()
def _build_inverse_curvature_q_shape(
    R_list: List[Tensor],
    *,
    power: float,
    eps_rel: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Dict[str, float]]:
    if not R_list:
        raise ValueError("R_list must be non-empty when building inverse-curvature Q shape.")

    L = int(R_list[0].shape[0])
    alpha = float(min(max(float(power), 0.0), 1.0))
    eps_rel = max(float(eps_rel), 0.0)

    H_bar = torch.zeros((L, L), device=device, dtype=dtype)
    for R_t in R_list:
        R = R_t.to(device=device, dtype=dtype)
        H_bar = H_bar + R.T @ R
    H_bar = H_bar / float(len(R_list))
    H_bar = 0.5 * (H_bar + H_bar.T)

    try:
        eigvals, eigvecs = torch.linalg.eigh(H_bar)
    except RuntimeError:
        shape = _identity_q_shape(L, device=device, dtype=dtype)
        return shape, {
            "shape_eig_min": 1.0,
            "shape_eig_mean": 1.0,
            "shape_eig_max": 1.0,
            "shape_cond": 1.0,
            "curv_eig_min": float("nan"),
            "curv_eig_mean": float("nan"),
            "curv_eig_max": float("nan"),
            "shape_eps": float("nan"),
        }

    eigvals = eigvals.clamp_min(0.0)
    curv_mean = float(eigvals.mean().item())
    curv_max = float(eigvals.max().item()) if eigvals.numel() else 0.0
    curv_min = float(eigvals.min().item()) if eigvals.numel() else 0.0
    if not math.isfinite(curv_mean) or curv_mean <= 0.0:
        shape = _identity_q_shape(L, device=device, dtype=dtype)
        return shape, {
            "shape_eig_min": 1.0,
            "shape_eig_mean": 1.0,
            "shape_eig_max": 1.0,
            "shape_cond": 1.0,
            "curv_eig_min": curv_min,
            "curv_eig_mean": curv_mean,
            "curv_eig_max": curv_max,
            "shape_eps": 0.0,
        }

    if alpha <= 0.0:
        shape_eigs = torch.ones_like(eigvals)
        eps_abs = 0.0
    else:
        eps_abs = max(eps_rel * curv_mean, torch.finfo(dtype).eps)
        shape_eigs = (eigvals + eps_abs).pow(-alpha)
        eig_sum = shape_eigs.sum().clamp_min(torch.finfo(dtype).eps)
        shape_eigs = shape_eigs * (float(L) / eig_sum)

    shape = (eigvecs * shape_eigs.unsqueeze(0)) @ eigvecs.T
    shape = 0.5 * (shape + shape.T)
    shape_min = float(shape_eigs.min().item())
    shape_mean = float(shape_eigs.mean().item())
    shape_max = float(shape_eigs.max().item())
    shape_cond = shape_max / max(shape_min, torch.finfo(dtype).eps)
    return shape, {
        "shape_eig_min": shape_min,
        "shape_eig_mean": shape_mean,
        "shape_eig_max": shape_max,
        "shape_cond": float(shape_cond),
        "curv_eig_min": curv_min,
        "curv_eig_mean": curv_mean,
        "curv_eig_max": curv_max,
        "shape_eps": float(eps_abs),
    }


def _build_module_q_shapes(
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    *,
    power: float,
    eps_rel: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Tensor], Dict[str, Dict[str, float]]]:
    shapes: Dict[str, Tensor] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for spec in module_specs:
        name = str(spec["name"])
        shape, shape_stats = _build_inverse_curvature_q_shape(
            module_R_lists[name],
            power=power,
            eps_rel=eps_rel,
            device=device,
            dtype=dtype,
        )
        shapes[name] = shape
        stats[name] = shape_stats
    return shapes, stats


def _build_gap_q_scales(
    mu_global_list_raw: List[Tensor],
    *,
    clip_min: float,
    clip_max: float,
) -> List[float]:
    T = len(mu_global_list_raw)
    if T <= 1:
        return [1.0 for _ in range(T)]

    raw_vals = [1.0]
    for t in range(1, T):
        delta = mu_global_list_raw[t] - mu_global_list_raw[t - 1]
        raw_vals.append(float(delta.pow(2).mean().item()))
    return _normalize_and_clip_scales(raw_vals, clip_min=clip_min, clip_max=clip_max)


def _report_module_constant_q_results(
    module_scales: Dict[str, float],
    module_shape_stats: Dict[str, Dict[str, float]],
    *,
    s_q: float,
    shrink_exponent: float,
    shape_power: float,
    shape_eps_rel: float,
) -> None:
    if not module_scales:
        print("[Module Q] No module scales available.")
        return

    vals = [float(v) for v in module_scales.values()]
    print("\n=== Module-Constant Q Report ===")
    print(
        f"[Module Q] mode=module_constant  estimator=lognormal_mu_drift_inverse_curvature_shape  "
        f"exposed scale s_Q={float(s_q):.6f}  "
        f"beta={float(min(max(float(shrink_exponent), 0.0), 1.0)):.3f}  "
        f"shape_power={float(min(max(float(shape_power), 0.0), 1.0)):.3f}  "
        f"shape_eps_rel={float(max(float(shape_eps_rel), 0.0)):.3e}"
    )
    print(
        f"[Module Q Summary] scale(min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f})"
    )
    if module_shape_stats:
        shape_mins = [float(v["shape_eig_min"]) for v in module_shape_stats.values()]
        shape_maxs = [float(v["shape_eig_max"]) for v in module_shape_stats.values()]
        shape_conds = [float(v["shape_cond"]) for v in module_shape_stats.values()]
        print(
            f"[Module Q Shape Summary] eig_min(min={min(shape_mins):.6f} mean={sum(shape_mins)/len(shape_mins):.6f}) "
            f"eig_max(mean={sum(shape_maxs)/len(shape_maxs):.6f} max={max(shape_maxs):.6f}) "
            f"cond(mean={sum(shape_conds)/len(shape_conds):.3f} max={max(shape_conds):.3f})"
        )
    print("[Module Q Per Module]")
    for name, scale in module_scales.items():
        base_q = float(s_q) * float(scale)
        shape_stats = module_shape_stats.get(name, {})
        shape_min = float(shape_stats.get("shape_eig_min", 1.0))
        shape_max = float(shape_stats.get("shape_eig_max", 1.0))
        print(
            f"  {name}: q_scale={float(scale):.6f} q_base={base_q:.6f} "
            f"q_eig_min={base_q * shape_min:.6f} q_eig_max={base_q * shape_max:.6f}"
        )


def _report_module_gap_q_results(
    module_scales: Dict[str, float],
    module_shape_stats: Dict[str, Dict[str, float]],
    gap_scales: List[float],
    *,
    s_q: float,
    shrink_exponent: float,
    shape_power: float,
    shape_eps_rel: float,
) -> None:
    print("\n=== Module-Gap Q Report ===")
    print(
        f"[Module Q] mode=module_gap  estimator=lognormal_mu_drift_inverse_curvature_shape  "
        f"exposed scale s_Q={float(s_q):.6f}  "
        f"beta={float(min(max(float(shrink_exponent), 0.0), 1.0)):.3f}  "
        f"shape_power={float(min(max(float(shape_power), 0.0), 1.0)):.3f}  "
        f"shape_eps_rel={float(max(float(shape_eps_rel), 0.0)):.3e}"
    )
    if module_scales:
        vals = [float(v) for v in module_scales.values()]
        print(
            f"[Module Q Summary] module_scale(min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f})"
        )
    if module_shape_stats:
        shape_mins = [float(v["shape_eig_min"]) for v in module_shape_stats.values()]
        shape_maxs = [float(v["shape_eig_max"]) for v in module_shape_stats.values()]
        shape_conds = [float(v["shape_cond"]) for v in module_shape_stats.values()]
        print(
            f"[Module Q Shape Summary] eig_min(min={min(shape_mins):.6f} mean={sum(shape_mins)/len(shape_mins):.6f}) "
            f"eig_max(mean={sum(shape_maxs)/len(shape_maxs):.6f} max={max(shape_maxs):.6f}) "
            f"cond(mean={sum(shape_conds)/len(shape_conds):.3f} max={max(shape_conds):.3f})"
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
    mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]
    use_module_constant_q = str(args.q_mode) in {"module_constant", "module_gap"}
    use_module_gap_q = str(args.q_mode) == "module_gap"
    gap_q_scales = (
        _build_gap_q_scales(
            mu_global_list_raw,
            clip_min=float(args.gap_q_clip_min),
            clip_max=float(args.gap_q_clip_max),
        )
        if use_module_gap_q
        else [1.0 for _ in range(T)]
    )
    module_q_scales = (
        _build_module_constant_q_scales(
            module_specs,
            mu_global_list_raw,
            clip_min=float(args.module_q_clip_min),
            clip_max=float(args.module_q_clip_max),
            shrink_exponent=float(args.module_q_shrink_exponent),
        )
        if use_module_constant_q
        else {}
    )
    module_q_shapes, module_q_shape_stats = (
        _build_module_q_shapes(
            module_specs,
            module_R_lists,
            power=float(args.module_q_shape_power),
            eps_rel=float(args.module_q_shape_eps_rel),
            device=cpu_device,
            dtype=torch.float64,
        )
        if use_module_constant_q
        else ({}, {})
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
            Q_list = materialize_shaped_Q_list(
                q_var_list,
                module_q_shapes.get(name, _identity_q_shape(Lm, device=cpu_device, dtype=torch.float64)),
                device=cpu_device,
                dtype=torch.float64,
            )
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
            module_q_shape_stats,
            gap_q_scales,
            s_q=float(args.s_q),
            shrink_exponent=float(args.module_q_shrink_exponent),
            shape_power=float(args.module_q_shape_power),
            shape_eps_rel=float(args.module_q_shape_eps_rel),
        )
    elif use_module_constant_q:
        _report_module_constant_q_results(
            module_q_scales,
            module_q_shape_stats,
            s_q=float(args.s_q),
            shrink_exponent=float(args.module_q_shrink_exponent),
            shape_power=float(args.module_q_shape_power),
            shape_eps_rel=float(args.module_q_shape_eps_rel),
        )
    else:
        _report_scalar_constant_q_results(args.s_q)
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


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Seq-LoRA on various tasks with configurable KFAC backend and process noise.")
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
        choices=["hook", "asdl"],
        help="KFAC backend. 'hook' restores the legacy hook-based KFAC path; 'asdl' uses the ASDL Kron backend.",
    )
    parser.add_argument(
        "--kfac_token_mode",
        type=str,
        default=KFAC_TOKEN_MODE,
        choices=["last", "all_valid"],
        help="Token-selection mode for the legacy hook KFAC backend.",
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
            "module_constant uses per-module q_m from adjacent-mu drift and inverse-curvature Q shape; "
            "module_gap additionally modulates Q_t by adjacent slice-gap scales."
        ),
    )
    parser.add_argument(
        "--module_q_clip_min",
        type=float,
        default=MODULE_Q_CLIP_MIN,
        help="Lower clip for per-module Q scales from adjacent-mu drift.",
    )
    parser.add_argument(
        "--module_q_clip_max",
        type=float,
        default=MODULE_Q_CLIP_MAX,
        help="Upper clip for per-module Q scales from adjacent-mu drift.",
    )
    parser.add_argument(
        "--module_q_shrink_exponent",
        type=float,
        default=MODULE_Q_SHRINK_EXPONENT,
        help=(
            "Log-normal shrinkage exponent beta for module_constant/module_gap Q. "
            "q_m/s_Q = clip((drift_m / median_drift) ** beta). "
            "0 makes all modules equal; 1 recovers the raw adjacent-mu drift heuristic."
        ),
    )
    parser.add_argument(
        "--module_q_shape_power",
        type=float,
        default=MODULE_Q_SHAPE_POWER,
        help=(
            "Power alpha for inverse-curvature Q shape S_m = normalize((Hbar_x + eps I)^(-alpha)). "
            "0 gives identity shape; 0.5 is a conservative half-inverse curvature shape; 1 is full inverse curvature."
        ),
    )
    parser.add_argument(
        "--module_q_shape_eps_rel",
        type=float,
        default=MODULE_Q_SHAPE_EPS_REL,
        help="Relative curvature floor eps = module_q_shape_eps_rel * mean_eig(Hbar_x) for inverse-curvature Q shape.",
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
        "--posterior_stats_cache_path",
        type=str,
        default="",
        help=(
            "Optional path to cache KFAC/subspace/mu_raw stats before Kalman. "
            "Matching caches let S_Q and mu_obs_scale sweep without recomputing the expensive KFAC stage."
        ),
    )
    parser.add_argument(
        "--force_rebuild_posterior_stats_cache",
        action="store_true",
        help="Rebuild the posterior-stats cache even if --posterior_stats_cache_path already exists.",
    )
    parser.add_argument(
        "--posterior_cache_path",
        type=str,
        default="",
        help=(
            "Optional path to cache everything up to pre-tau posterior samples. "
            "When present, matching caches are loaded and posterior build is skipped."
        ),
    )
    parser.add_argument(
        "--force_rebuild_posterior_cache",
        action="store_true",
        help="Rebuild the posterior cache even if --posterior_cache_path already exists.",
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
        "--eval_protocol",
        type=str,
        default="default",
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
        else:
            train_raw = _ensure_slice_ids_for_seq(args.task, train_raw)
            slice_source = "task_train_split"

    # -------------------------
    # Eval task validation only. Actual eval datasets are loaded lazily later.
    # -------------------------
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    for eval_task in eval_tasks:
        eval_num_classes = _get_num_classes_for_protocol(eval_task, eval_protocol)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, "
                f"but source task '{args.task}' has {num_classes} classes."
            )

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
        if eff_samples < len(ds_t):
            ds_t = ds_t.select(range(eff_samples))
        total_kfac_samples += eff_samples
        total_kfac_batches += eff_batches
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
    forward_call_for_kfac = forward_call_for_kfac_factory(
        amp_dtype,
        choice_token_ids,
        apply_choice_mask=apply_choice_mask,
        kfac_token_mode=str(args.kfac_token_mode),
    )
    stats_cache_path = str(args.posterior_stats_cache_path).strip()
    cache_path = str(args.posterior_cache_path).strip()
    cache_loaded = False
    stats_cache_loaded = False
    module_specs: List[Dict] | None = None
    module_R_lists: Dict[str, List[Tensor]] | None = None
    mu_global_list_raw: List[Tensor] | None = None
    L_total: int | None = None

    if cache_path and os.path.exists(cache_path) and not bool(args.force_rebuild_posterior_cache):
        try:
            module_specs, x_samples_T, L_total, cached_num_modules = _load_posterior_cache(cache_path, args)
            lora_cache = build_loraA_cache(model, module_specs, device=device)
            cache_loaded = True
            print(f"\n[Posterior Cache] loaded from {cache_path}")
            print(f"[Posterior Cache] modules={cached_num_modules} L_total={L_total} samples={tuple(x_samples_T.shape)}")
        except Exception as exc:
            print(f"\n[Posterior Cache] ignoring {cache_path}: {exc}")

    if (
        not cache_loaded
        and stats_cache_path
        and os.path.exists(stats_cache_path)
        and not bool(args.force_rebuild_posterior_stats_cache)
    ):
        try:
            (
                module_specs,
                module_R_lists,
                mu_global_list_raw,
                L_total,
                cached_num_modules,
            ) = _load_posterior_stats_cache(stats_cache_path, args)
            lora_cache = build_loraA_cache(model, module_specs, device=device)
            stats_cache_loaded = True
            print(f"\n[Posterior Stats Cache] loaded from {stats_cache_path}")
            print(
                f"[Posterior Stats Cache] modules={cached_num_modules} "
                f"L_total={L_total} slices={len(mu_global_list_raw)}"
            )
        except Exception as exc:
            print(f"\n[Posterior Stats Cache] ignoring {stats_cache_path}: {exc}")

    if not cache_loaded and not stats_cache_loaded:
        H_factor_per_module, G_factor_per_module, module_names = {}, {}, None

        dropout_ctx = (
            _temporarily_disable_dropout_modules(model)
            if bool(args.disable_dropout_during_kfac_mu)
            else nullcontext()
        )

        with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior build on {args.task}"), dropout_ctx:
            for t_idx, loader_t in enumerate(slice_loaders):
                print(
                    f"[KFAC] slice {t_idx + 1}/{len(slice_loaders)} "
                    f"sid={slice_ids[t_idx]} batches={len(loader_t)}"
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
                    ) if str(args.kfac_backend) == "asdl" else calculate_kronecker_factors_hook(
                        model=model,
                        forward_call=forward_call_for_kfac,
                        loader=loader_t,
                        n_kfac=args.n_kfac,
                        lr_threshold=args.lr_threshold,
                        target_module_keywords=["lora_A"],
                        exclude_bias=False,
                        use_tqdm=False,
                    )
                print(f"[KFAC] slice {t_idx + 1}/{len(slice_loaders)} done")

                if module_names is None:
                    module_names = _resolve_bayes_module_names(factors)
                    module_names = _filter_bayes_module_names(
                        module_names,
                        eval_protocol=eval_protocol,
                        perturb_lm_head=bool(args.bayesian_peft_perturb_lm_head),
                    )
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
            L_total = offset
            lora_cache = build_loraA_cache(model, module_specs, device=device)

            mu_estimator = (
                estimate_mu_global_list_from_slice_grads_asdl
                if str(args.kfac_backend) == "asdl"
                else estimate_mu_global_list_from_slice_grads
            )
            mu_kwargs = (
                {"disable_dropout": bool(args.disable_dropout_during_kfac_mu)}
                if str(args.kfac_backend) == "asdl"
                else {}
            )
            mu_global_list_raw = mu_estimator(
                model,
                slice_loaders,
                forward_call_for_kfac,
                module_names,
                module_subspace_info,
                module_R_lists,
                device,
                args.mu_obs_batches,
                torch.float64,
                **mu_kwargs,
            )

            if stats_cache_path:
                stats_cache_dir = os.path.dirname(stats_cache_path)
                if stats_cache_dir:
                    os.makedirs(stats_cache_dir, exist_ok=True)
                torch.save(
                    _build_posterior_stats_cache_payload(
                        args=args,
                        module_specs=module_specs,
                        module_R_lists=module_R_lists,
                        mu_global_list_raw=mu_global_list_raw,
                        l_total=L_total,
                        num_modules=len(module_specs),
                    ),
                    stats_cache_path,
                )
                print(f"[Posterior Stats Cache] saved to {stats_cache_path}")

            x_samples_T = _sample_posterior_from_stats(
                args=args,
                module_specs=module_specs,
                module_R_lists=module_R_lists,
                mu_global_list_raw=mu_global_list_raw,
                T=T,
                device=device,
                cpu_device=cpu_device,
            )

    if not cache_loaded and stats_cache_loaded:
        with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior build on {args.task} [Kalman-only]"):
            x_samples_T = _sample_posterior_from_stats(
                args=args,
                module_specs=module_specs,
                module_R_lists=module_R_lists,
                mu_global_list_raw=mu_global_list_raw,
                T=T,
                device=device,
                cpu_device=cpu_device,
            )

    if not cache_loaded:
        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(
                _build_posterior_cache_payload(
                    args=args,
                    module_specs=module_specs,
                    x_samples_T=x_samples_T,
                    l_total=L_total,
                    num_modules=len(module_specs),
                ),
                cache_path,
            )
            print(f"[Posterior Cache] saved to {cache_path}")

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

    def _load_eval_proc_or_loader(eval_task: str):
        if _uses_direct_bayesian_peft_data(eval_task, eval_protocol):
            eval_task_dataset = _build_bayesian_peft_task_dataset(
                tokenizer,
                eval_task,
                add_space=bool(args.bayesian_peft_add_space),
                max_seq_len=args.max_seq_len,
            )
            eval_split = str(args.iid_eval_split) if eval_task == args.task else "validation"
            return _make_direct_bayesian_peft_loader(
                eval_task_dataset.dset[eval_split],
                collate_fn=_BayesianPeftCLMCollator(eval_task_dataset),
                batch_size=args.eval_bsz,
                shuffle=False,
                drop_last=True,
                num_workers=args.eval_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=args.eval_prefetch_factor,
            )

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
        return eval_proc.sort("seq_len")

    def eval_one(tag: str, eval_task: str):
        proc_or_loader = _load_eval_proc_or_loader(eval_task)
        if isinstance(proc_or_loader, DataLoader):
            loader = proc_or_loader
            proc_eval = None
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
