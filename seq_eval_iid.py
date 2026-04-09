from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence
import os
import random
import math
import time
import argparse
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from kfac import calculate_kronecker_factors
from lssm_ffbs_obs import kalman_filter
from seq_lora_subspace_obs import (
    build_global_kronecker_eigenspace,
    materialize_mean_psd_from_factors,
    project_curvature_to_subspace,
    project_curvature_factors_to_subspace,
    prepare_lgssm_observations,
    solve_xhat_from_grad,
    trace_psd_factor,
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
TRUST_REMOTE_CODE = True

MAX_SEQ_LEN = 300
EVAL_BSZ = 32
KFAC_BSZ = 8

# KFAC / train-slice loaders remain conservative
NUM_WORKERS = 0

# Eval loader gets its own workers for dynamic padding pipeline
EVAL_NUM_WORKERS = 0
EVAL_PREFETCH_FACTOR = 4

N_KFAC = 16
LR_THRESHOLD = 256
MAX_KFAC_SAMPLES_PER_SLICE = 256

MU_OBS_SCALE = 3
MU_OBS_BATCHES = 16
Q_CONST_VAR = 1
P1_VAR = 1.0

Q_BASE_VAR = float(Q_CONST_VAR)
Q_C = 0.5
Q_VAR_MIN = 0.5
Q_VAR_MAX = 5
Q_SMOOTH_ALPHA = 0.5
USE_PRECISION_DRIFT = True

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 32

MIN_MEAN_TRACE_G = 1e-12
PI_MAX = 1e6

POSTERIOR_TAU = 1.0
TEMP_BAYES = 1.0

# retained for call compatibility; step-1 eval path no longer micro-batches
BAYES_MICRO_BSZ = 32
DELTA_CHUNK_SIZE = 8
TOKENIZER_PADDING_SIDE = "left"

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

def _format_eta(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


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
    return logits.to(torch.float32)


def _resolve_bayes_module_names(factors: Dict[str, Tuple[Tensor, Tensor]]) -> List[str]:
    return sorted([name for name in factors.keys() if "lora_A" in name])

def _build_slice_label_map(train_raw: Dataset) -> Dict[int, str]:
    if "slice_id" not in train_raw.column_names:
        return {}

    labels: Dict[int, str] = {}
    has_grade_num = "grade_num" in train_raw.column_names
    has_grade = "grade" in train_raw.column_names
    has_block_name = "block_name" in train_raw.column_names
    for ex in train_raw:
        sid = int(ex["slice_id"])
        if sid in labels:
            continue
        if has_grade_num and ex.get("grade_num") is not None:
            labels[sid] = f"grade{int(ex['grade_num'])}"
            continue
        if has_grade and ex.get("grade") is not None:
            labels[sid] = str(ex["grade"])
            continue
        block_name = str(ex["block_name"]) if has_block_name and ex.get("block_name") is not None else ""
        labels[sid] = f"sid={sid} [{block_name}]" if block_name else f"sid={sid}"
    return labels


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

def _report_q_results(
    slice_ids: Sequence[int],
    q_stats: Dict[str, List[float]],
    L_total: int,
    slice_label_map: Optional[Dict[int, str]] = None,
) -> None:
    d_list = [float(x) for x in q_stats.get("d_list", [])]
    sigma_list = [float(x) for x in q_stats.get("sigma_list", [])]
    var_list = [float(x) for x in q_stats.get("var_list", [])]
    if not var_list:
        print("[Q] No Q statistics available.")
        return

    mode = str(q_stats.get("mode", "drift"))
    constant_q_var = q_stats.get("constant_q_var", None)

    def _summ(vals: Sequence[float]) -> str:
        return f"min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f}"

    print("\n=== Q Process Noise Report ===")
    print(
        f"[Q Config] mode={mode} base_var={Q_BASE_VAR:.6f} c={Q_C:.6f} "
        f"var_min={Q_VAR_MIN:.6f} var_max={Q_VAR_MAX:.6f} "
        f"smooth_alpha={Q_SMOOTH_ALPHA:.6f} use_precision={USE_PRECISION_DRIFT}"
    )
    if constant_q_var is not None:
        print(f"[Q Config] constant_q_var={float(constant_q_var):.6f}")
    print(f"[Q Summary] T={len(var_list)} | drift({_summ(d_list)}) | sigma({_summ(sigma_list)}) | var({_summ(var_list)})")
    print("[Q Per Slice]")
    for idx, sid in enumerate(slice_ids):
        label = slice_label_map.get(int(sid), f"sid={sid}") if slice_label_map else f"sid={sid}"
        drift = d_list[idx] if idx < len(d_list) else float("nan")
        sigma = sigma_list[idx] if idx < len(sigma_list) else float("nan")
        var = var_list[idx] if idx < len(var_list) else float("nan")
        trace_q = float(L_total) * var
        print(f"  {label}: drift={drift:.6f} sigma={sigma:.6f} var={var:.6f} trace(Q_t)={trace_q:.6f}")


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

def forward_call_for_kfac_factory(amp_dtype: torch.dtype):
    def _forward_call(model: nn.Module, batch: Dict[str, Tensor]) -> Tensor:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        import kfac as kfac_mod
        kfac_mod._CURRENT_LAST_IDX = _left_padded_last_idx(input_ids)

        logits = compute_choice_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            amp_dtype=amp_dtype,
        )
        return _mask_invalid_choices(logits, num_choices)

    return _forward_call

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
def build_Q_list_from_R_drift(
    R_big_list: List[Tensor],
    L: int,
    base_var: float = 1.0,
    c: float = 0.5,
    eps: float = 1e-12,
    var_min: float = 1e-6,
    var_max: float = 1e2,
    use_precision: bool = True,
    smooth_alpha: float = 0.8,
) -> Tuple[List[Tensor], dict]:
    T = len(R_big_list)
    device = R_big_list[0].device
    dtype = R_big_list[0].dtype
    I = torch.eye(L, device=device, dtype=dtype)
    d_list = [0.0]
    prev_M = None

    import numpy as np

    for t in range(T):
        M_t = R_big_list[t].transpose(-1, -2) @ R_big_list[t] if use_precision else R_big_list[t]
        if t == 0:
            prev_M = M_t
            continue

        denom = (0.5 * (torch.linalg.norm(M_t.reshape(-1), ord=2) + torch.linalg.norm(prev_M.reshape(-1), ord=2))).clamp_min(eps)
        drift = torch.linalg.norm((M_t - prev_M).reshape(-1), ord=2) / denom
        d_list.append(math.log1p(float(drift.item())))
        prev_M = M_t

    d_arr = np.array(d_list, dtype=np.float64)
    if len(d_arr) > 1:
        cap = float(np.quantile(d_arr[1:], 0.95))
        d_list = [d_list[0]] + [min(float(x), cap) for x in d_list[1:]]

    base_sigma = float(base_var) ** 0.5
    sigma_list, var_list = [], []

    for t in range(T):
        var_t = max(
            float(var_min),
            min(float(var_max), float((base_sigma * (1.0 + float(c) * d_list[t])) ** 2)),
        )
        sigma_list.append(float(var_t) ** 0.5)
        var_list.append(float(var_t))

    if smooth_alpha > 0.0:
        sm = sigma_list[0]
        for t in range(1, T):
            sm = float(smooth_alpha) * sm + (1.0 - float(smooth_alpha)) * sigma_list[t]
            sigma_list[t], var_list[t] = float(sm), float(sm * sm)

    return [float(var_list[t]) * I for t in range(T)], {
        "mode": "drift",
        "d_list": d_list,
        "sigma_list": sigma_list,
        "var_list": var_list,
    }

@torch.no_grad()
def build_constant_Q_list(
    T: int,
    L: int,
    q_var: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[Tensor], dict]:
    q_var = float(q_var)
    I = torch.eye(L, device=device, dtype=dtype)
    q_t = q_var * I
    sigma = math.sqrt(max(q_var, 0.0))
    return [q_t.clone() for _ in range(T)], {
        "mode": "constant",
        "constant_q_var": q_var,
        "d_list": [0.0 for _ in range(T)],
        "sigma_list": [sigma for _ in range(T)],
        "var_list": [q_var for _ in range(T)],
    }


def build_Q_var_schedule_from_module_R_drift(
    module_R_lists: Dict[str, List[Tensor]],
    module_names: List[str],
    base_var: float = 1.0,
    c: float = 0.5,
    eps: float = 1e-12,
    var_min: float = 1e-6,
    var_max: float = 1e2,
    use_precision: bool = True,
    smooth_alpha: float = 0.8,
) -> Tuple[List[float], dict]:
    if not module_names:
        raise ValueError("module_names must be non-empty")

    T = len(module_R_lists[module_names[0]])
    d_list = [0.0]
    prev_blocks: Optional[Dict[str, Tensor]] = None

    import numpy as np

    for t in range(T):
        curr_blocks: Dict[str, Tensor] = {}
        curr_norm_sq = 0.0
        prev_norm_sq = 0.0
        diff_norm_sq = 0.0

        for name in module_names:
            R_t = module_R_lists[name][t]
            M_t = R_t.transpose(-1, -2) @ R_t if use_precision else R_t
            curr_blocks[name] = M_t
            curr_norm_sq += float(torch.sum(M_t * M_t).item())

            if prev_blocks is not None:
                prev_M = prev_blocks[name]
                prev_norm_sq += float(torch.sum(prev_M * prev_M).item())
                delta = M_t - prev_M
                diff_norm_sq += float(torch.sum(delta * delta).item())

        if prev_blocks is None:
            prev_blocks = curr_blocks
            continue

        denom = max(
            0.5 * (math.sqrt(curr_norm_sq) + math.sqrt(prev_norm_sq)),
            float(eps),
        )
        drift = math.sqrt(diff_norm_sq) / denom
        d_list.append(math.log1p(float(drift)))
        prev_blocks = curr_blocks

    d_arr = np.array(d_list, dtype=np.float64)
    if len(d_arr) > 1:
        cap = float(np.quantile(d_arr[1:], 0.95))
        d_list = [d_list[0]] + [min(float(x), cap) for x in d_list[1:]]

    base_sigma = float(base_var) ** 0.5
    sigma_list, var_list = [], []

    for t in range(T):
        var_t = max(
            float(var_min),
            min(float(var_max), float((base_sigma * (1.0 + float(c) * d_list[t])) ** 2)),
        )
        sigma_list.append(float(var_t) ** 0.5)
        var_list.append(float(var_t))

    if smooth_alpha > 0.0:
        sm = sigma_list[0]
        for t in range(1, T):
            sm = float(smooth_alpha) * sm + (1.0 - float(smooth_alpha)) * sigma_list[t]
            sigma_list[t], var_list[t] = float(sm), float(sm * sm)

    return var_list, {
        "mode": "drift",
        "d_list": d_list,
        "sigma_list": sigma_list,
        "var_list": var_list,
    }


@torch.no_grad()
def materialize_scalar_Q_list(
    var_list: Sequence[float],
    L: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    I = torch.eye(L, device=device, dtype=dtype)
    return [float(var) * I for var in var_list]

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

        mu_parts = [
            solve_xhat_from_grad(
                module_R_lists[name][t].to(device=device, dtype=dtype),
                g_x_parts[mi] / float(n_seen),
            )
            for mi, name in enumerate(module_names)
        ]
        mu_global_list.append(torch.cat(mu_parts, dim=0).cpu())

    return mu_global_list

def balance_h_g_scales(H_list: List[Tensor], G_list: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], float, Dict[str, float]]:
    mean_trace_H = torch.stack([torch.trace(H_t) / H_list[0].shape[0] for H_t in H_list]).mean()
    mean_trace_G = torch.stack([torch.trace(G_t) / G_list[0].shape[0] for G_t in G_list]).mean()
    denom = (torch.tensor(MIN_MEAN_TRACE_G, dtype=mean_trace_G.dtype) if mean_trace_G < MIN_MEAN_TRACE_G else mean_trace_G + 1e-20)
    pi = torch.sqrt(mean_trace_H / denom)
    if pi > PI_MAX:
        pi = torch.tensor(PI_MAX, dtype=pi.dtype)
    H_bal = [H_t / pi for H_t in H_list]
    G_bal = [G_t * pi for G_t in G_list]
    stats = {
        "mean_trace_H": float(mean_trace_H.item()),
        "mean_trace_G": float(mean_trace_G.item()),
        "pi": float(pi.item()),
        "mean_trace_H_bal": float(torch.stack([torch.trace(H_t) / H_bal[0].shape[0] for H_t in H_bal]).mean().item()),
        "mean_trace_G_bal": float(torch.stack([torch.trace(G_t) / G_bal[0].shape[0] for G_t in G_bal]).mean().item()),
    }
    return H_bal, G_bal, float(pi.item()), stats


def balance_h_g_factor_scales(
    H_factors: List[Tensor],
    G_factors: List[Tensor],
) -> Tuple[float, Dict[str, float]]:
    mean_trace_H = torch.stack([
        trace_psd_factor(H_t) / H_factors[0].shape[0] for H_t in H_factors
    ]).mean()
    mean_trace_G = torch.stack([
        trace_psd_factor(G_t) / G_factors[0].shape[0] for G_t in G_factors
    ]).mean()
    denom = (
        torch.tensor(MIN_MEAN_TRACE_G, dtype=mean_trace_G.dtype)
        if mean_trace_G < MIN_MEAN_TRACE_G
        else mean_trace_G + 1e-20
    )
    pi = torch.sqrt(mean_trace_H / denom)
    if pi > PI_MAX:
        pi = torch.tensor(PI_MAX, dtype=pi.dtype)
    stats = {
        "mean_trace_H": float(mean_trace_H.item()),
        "mean_trace_G": float(mean_trace_G.item()),
        "pi": float(pi.item()),
        "mean_trace_H_bal": float((mean_trace_H / pi).item()),
        "mean_trace_G_bal": float((mean_trace_G * pi).item()),
    }
    return float(pi.item()), stats


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

def _report_h_g_pi_results(module_stats: Dict[str, Dict[str, float]]) -> None:
    if not module_stats:
        print("[H/G/pi] No module statistics available.")
        return

    def _summ(key: str) -> str:
        vals = [float(stats[key]) for stats in module_stats.values()]
        return f"min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f}"

    print("\n=== H/G/pi Report ===")
    print(
        f"[H/G/pi Summary] "
        f"H({_summ('mean_trace_H')}) | "
        f"G({_summ('mean_trace_G')}) | "
        f"pi({_summ('pi')})"
    )
    print("[H/G/pi Per Module]")
    for name, stats in module_stats.items():
        print(
            f"  {name}: "
            f"H={stats['mean_trace_H']:.6f} "
            f"G={stats['mean_trace_G']:.6f} "
            f"pi={stats['pi']:.6f} "
            f"H_bal={stats['mean_trace_H_bal']:.6f} "
            f"G_bal={stats['mean_trace_G_bal']:.6f}"
        )

# =========================
# Fast Bayesian eval
# =========================

@dataclass
class _LoraACache:
    name: str
    weight: nn.Parameter
    U_fp32: Tensor
    offset: int
    L: int
    shape: Tuple[int, ...]
    numel: int

def build_loraA_cache(model: nn.Module, module_specs: List[Dict], device: torch.device) -> List[_LoraACache]:
    caches: List[_LoraACache] = []
    for spec in module_specs:
        w = _get_param_weight(model, spec["name"])
        if w.dtype != torch.float32:
            w.data = w.data.to(dtype=torch.float32)
        caches.append(
            _LoraACache(
                name=spec["name"],
                weight=w,
                U_fp32=spec["subspace_info"]["U_lora"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).contiguous(),
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
    lora_cache: List[_LoraACache],
    x_samples_T: Tensor,
    posterior_scale_tau: float = 0.8,
    temp_bayes: float = 1.0,
    max_mc_samples: int = 32,
    progress_desc: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    model.eval()
    _set_inference_fast(model)

    scale = float(posterior_scale_tau) / math.sqrt(max(len(lora_cache), 1))
    S = min(int(max_mc_samples), int(x_samples_T.shape[0]))
    xS = x_samples_T[:S].contiguous()

    weight_tensors = [spec.weight.data for spec in lora_cache]
    eps = 1e-12

    # 1. 预先将测试集完全加载到 GPU 显存中，避免反复的 Host->Device 拷贝
    cached_batches = []
    total_samples = 0
    cache_t0 = time.perf_counter()
    for batch in loader:
        lengths_cpu = batch["attention_mask"].sum(dim=1)
        Lmax = max(int(lengths_cpu.max().item()), 1)
        
        # Left padding keeps real tokens on the right, so trim from the right edge.
        ids = batch["input_ids"][:, -Lmax:].to(device, non_blocking=True)
        attn = batch["attention_mask"][:, -Lmax:].to(device, non_blocking=True)
        labs = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        cached_batches.append((ids, attn, labs, num_choices))
        total_samples += labs.size(0)
        
    cache_time_sec = time.perf_counter() - cache_t0

    # 预分配全局概率张量和标签
    probs_acc_global = torch.zeros((total_samples, num_classes), device=device, dtype=torch.float32)
    labels_global = torch.zeros(total_samples, device=device, dtype=torch.long)
    
    idx = 0
    for (_, _, labs, _) in cached_batches:
        bsz = labs.size(0)
        labels_global[idx : idx + bsz] = labs
        idx += bsz

    bayes_t0 = time.perf_counter()
    progress_total = max(total_samples * S, 1)
    progress_bar = tqdm(
        total=progress_total,
        desc=(progress_desc or "Seq eval"),
        unit="sample",
        leave=False,
    )

    # 2. 核心优化：Sample 在外层，Batch 在内层！只修改 S 次权重
    for s in range(S):
        # 计算第 s 个样本的微调增量，并物理注入到 7B 模型中
        deltas_s = _compute_deltas_for_one_sample(lora_cache, xS[s], scale)
        torch._foreach_add_(weight_tensors, deltas_s)

        idx = 0
        # 拿着注入噪声的模型，一口气推完整个缓存的测试集
        for (ids_mb, attn_mb, _, num_choices_mb) in cached_batches:
            bsz = ids_mb.size(0)
            logits = compute_choice_logits(
                model=model,
                input_ids=ids_mb,
                attention_mask=attn_mb,
                amp_dtype=amp_dtype,
            )
            logits = _mask_invalid_choices(logits, num_choices_mb)
            probs_acc_global[idx : idx + bsz].add_(torch.softmax(logits, dim=-1))
            idx += bsz
            progress_bar.update(bsz)
            elapsed = time.perf_counter() - bayes_t0
            avg_sec_per_sample = elapsed / max(progress_bar.n, 1)
            eta = _format_eta(avg_sec_per_sample * max(progress_total - progress_bar.n, 0))
            progress_bar.set_postfix(
                mc=f"{s+1}/{S}",
                avg_s_per_sample=f"{avg_sec_per_sample:.3f}",
                eta=eta,
                refresh=False,
            )

        # 恢复权重，准备下一次采样
        torch._foreach_sub_(weight_tensors, deltas_s)

    progress_bar.close()
    bayes_extra_time = time.perf_counter() - bayes_t0

    # 3. 计算最终的贝叶斯平均概率 (BMA)
    probs_bayes_global = probs_acc_global / float(S)

    if temp_bayes != 1.0:
        p = probs_bayes_global.clamp_min(eps) ** (1.0 / float(temp_bayes))
        probs_bayes_global = p / p.sum(dim=-1, keepdim=True)

    p_y = probs_bayes_global[torch.arange(total_samples, device=device), labels_global].clamp_min(eps)
    nll_bayes = float((-torch.log(p_y)).sum().item()) / max(total_samples, 1)
    brier_bayes = _multiclass_brier_score(probs_bayes_global, labels_global)

    acc_bay_m = _make_accuracy(device, num_classes=num_classes)
    acc_bay_m.reset()
    ece_bay_m = _make_ece(device, num_classes=num_classes, n_bins=15)
    ece_bay_m.reset()

    acc_bay_m.update(probs_bayes_global, labels_global)
    ece_bay_m.update(probs_bayes_global, labels_global)

    metrics = {
        "nll_bayes": nll_bayes,
        "brier_bayes": brier_bayes,
        "ece_bayes": float(ece_bay_m.compute().item()),
        "acc_bayes": float(acc_bay_m.compute().item()),
        "mc_samples_used": float(S),
        "posterior_scale_factor": float(scale),
        "time_bayes_sec": float(bayes_extra_time),
    }
    return metrics
# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Seq-LoRA on various tasks.")
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
        "--use_constant_q",
        action="store_true",
        help="Use a constant process-noise Q_t = q_var * I for all slices instead of constructing Q from R-drift.",
    )
    parser.add_argument(
        "--constant_q_var",
        type=float,
        default=float(Q_CONST_VAR),
        help="Constant Q variance used when --use_constant_q is enabled.",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=0,
        help="Forecast horizon h. 0 uses x_{T|T}; 1 uses one-step-ahead x_{T+1|T}.",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print("Using device:", device)

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    pin_memory = (device.type == "cuda")

    peft_cfg = PeftConfig.from_pretrained(args.map_dir)
    base_name = peft_cfg.base_model_name_or_path
    print(f"\n[Load] base_model = {base_name}\n[Load] adapter    = {args.map_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = TOKENIZER_PADDING_SIDE

    num_classes = get_task_num_classes(args.task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa",
    ).to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
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
        train_raw, _, _ = load_task_dataset(args.task)
        train_raw = _ensure_slice_ids_for_seq(args.task, train_raw)
        slice_source = "task_train_split"
        if args.task == SCIENCEQA_CURRIC_TASK_NAME:
            print("[Slices] No --slices_dir provided; using ScienceQA train split with grade-based slice ids.")
    print(f"\n[Slices] train={len(train_raw)} source={slice_source} (used for KFAC only)")
    slice_label_map = _build_slice_label_map(train_raw)

    # -------------------------
    # Eval set: dynamic padding + length sorting
    # -------------------------
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    eval_task_to_proc: Dict[str, Dataset] = {}
    for eval_task in eval_tasks:
        eval_num_classes = get_task_num_classes(eval_task)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, "
                f"but source task '{args.task}' has {num_classes} classes."
            )
        eval_raw = load_iid_test_set(eval_task) if eval_task == args.task else load_eval_dataset(eval_task)
        eval_proc = preprocess_task(
            eval_task,
            eval_raw,
            tokenizer,
            MAX_SEQ_LEN,
            pad_to_max_length=False,
        )
        eval_proc = eval_proc.add_column("seq_len", [len(x) for x in eval_proc["input_ids"]])
        eval_task_to_proc[eval_task] = eval_proc.sort("seq_len")

    # -------------------------
    # KFAC/train slices: dynamic padding to reduce wasted compute on long MMLU inputs
    # -------------------------
    train_proc = preprocess_task(
        args.task,
        train_raw,
        tokenizer,
        MAX_SEQ_LEN,
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
    for sid in slice_ids:
        ds_t = train_proc.filter(lambda ex, sid=sid: int(ex["slice_id"]) == sid)
        if MAX_KFAC_SAMPLES_PER_SLICE is not None and len(ds_t) > MAX_KFAC_SAMPLES_PER_SLICE:
            ds_t = ds_t.shuffle(seed=42).select(range(MAX_KFAC_SAMPLES_PER_SLICE))
        ds_t = ds_t.sort("seq_len")
        eff_batches = len(ds_t) // KFAC_BSZ
        eff_samples = eff_batches * KFAC_BSZ
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
                batch_size=KFAC_BSZ,
                shuffle=False,
                drop_last=True,
                collate_fn=kfac_collator,
                num_workers=NUM_WORKERS,
                pin_memory=pin_memory,
            )
        )
    print(
        f"[KFAC] effective samples after per-slice cap/drop_last = {total_kfac_samples} "
        f"across {total_kfac_batches} batches"
    )

    forward_call_for_kfac = forward_call_for_kfac_factory(amp_dtype)
    H_factor_per_module, G_factor_per_module, module_names = {}, {}, None

    with _StageTimer(f"TRAIN-STAGE Seq-LoRA posterior build on {args.task}"):
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
                    n_kfac=N_KFAC,
                    lr_threshold=LR_THRESHOLD,
                    target_module_keywords=["lora_A"],
                    exclude_bias=False,
                    use_tqdm=True,
                )

            if module_names is None:
                module_names = _resolve_bayes_module_names(factors)
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

        module_subspace_info, module_R_lists = {}, {}
        module_hgpi_stats: Dict[str, Dict[str, float]] = {}
        for name in module_names:
            H_factors = H_factor_per_module[name]
            G_factors = G_factor_per_module[name]
            pi, hgpi_stats = balance_h_g_factor_scales(H_factors, G_factors)
            module_hgpi_stats[name] = hgpi_stats

            H_bar_bal = materialize_mean_psd_from_factors(
                H_factors,
                matrix_scale=(1.0 / pi),
                device=device,
                dtype=torch.float64,
            )
            G_bar_bal = materialize_mean_psd_from_factors(
                G_factors,
                matrix_scale=pi,
                device=device,
                dtype=torch.float64,
            )

            subspace_info_gpu = build_global_kronecker_eigenspace(
                H_list=[H_bar_bal],
                G_B_list=[G_bar_bal],
                subspace_dim=SUBSPACE_DIM_PER_MODULE,
                eps_eig=1e-6,
            )
            _, R_list = project_curvature_factors_to_subspace(
                H_factors=H_factors,
                G_B_factors=G_factors,
                subspace_info=subspace_info_gpu,
                lambda_damp=1e-4,
                H_matrix_scale=(1.0 / pi),
                G_matrix_scale=pi,
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
            del H_bar_bal, G_bar_bal, subspace_info_gpu
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        H_factor_per_module.clear()
        G_factor_per_module.clear()
        _report_h_g_pi_results(module_hgpi_stats)

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

        mu_global_list = estimate_mu_global_list_from_slice_grads(
            model,
            slice_loaders,
            forward_call_for_kfac,
            module_names,
            module_subspace_info,
            module_R_lists,
            device,
            MU_OBS_BATCHES,
            torch.float64,
        )
        mu_global_list = [MU_OBS_SCALE * mu_t for mu_t in mu_global_list]

        if args.use_constant_q:
            _, q_stats = build_constant_Q_list(
                T=T,
                L=1,
                q_var=float(args.constant_q_var),
                device=cpu_device,
                dtype=torch.float64,
            )
            q_var_list = [float(args.constant_q_var) for _ in range(T)]
        else:
            q_var_list, q_stats = build_Q_var_schedule_from_module_R_drift(
                module_R_lists=module_R_lists,
                module_names=module_names,
                base_var=Q_BASE_VAR,
                c=Q_C,
                var_min=Q_VAR_MIN,
                var_max=Q_VAR_MAX,
                use_precision=USE_PRECISION_DRIFT,
                smooth_alpha=Q_SMOOTH_ALPHA,
            )
        _report_q_results(
            slice_ids=slice_ids,
            q_stats=q_stats,
            L_total=L_total,
            slice_label_map=slice_label_map,
        )

        print(f"\n=== Kalman Filter Only (module-wise) ===")
        print(f"[Kalman] modules={len(module_specs)} L_total={L_total}")

        if args.forecast_horizon > 0:
            print(
                f"\nDirectly sampling forecast posterior (t=T+{args.forecast_horizon} | T): "
                f"S={MC_EVAL_SAMPLES}"
            )
        else:
            print(f"\nDirectly sampling final posterior (t=T): S={MC_EVAL_SAMPLES}")

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
            P1 = P1_VAR * torch.eye(Lm, device=cpu_device, dtype=torch.float64)
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
            x_sample_parts.append(dist_m.sample((MC_EVAL_SAMPLES,)))

            del H_obs_list, y_list, Q_list, x_filt_m, P_filt_m, mu_T_m, cov_T_m, cov_T_stable, dist_m
            gc.collect()

        del mu_global_list
        x_samples_T = torch.cat(x_sample_parts, dim=1).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        del x_sample_parts

    lora_cache = build_loraA_cache(model, module_specs, device=device)

    eval_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    eval_loader_kwargs = {
        "batch_size": EVAL_BSZ,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": eval_collator,
        "num_workers": EVAL_NUM_WORKERS,
        "pin_memory": pin_memory,
    }
    if EVAL_NUM_WORKERS > 0:
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = EVAL_PREFETCH_FACTOR

    def eval_one(tag: str, proc: Dataset):
        proc_eval = proc.remove_columns(["seq_len"]) if "seq_len" in proc.column_names else proc
        loader = DataLoader(proc_eval, **eval_loader_kwargs)

        with _StageTimer(f"INFER Seq-LoRA on {tag}"):
            metrics = eval_bayes_fast_restricted_4way_probmean(
                model=model,
                loader=loader,
                device=device,
                amp_dtype=amp_dtype,
                num_classes=num_classes,
                lora_cache=lora_cache,
                x_samples_T=x_samples_T,
                posterior_scale_tau=POSTERIOR_TAU,
                temp_bayes=TEMP_BAYES,
                max_mc_samples=MC_EVAL_SAMPLES,
                progress_desc=f"SEQ {tag}",
                bayes_micro_bsz=BAYES_MICRO_BSZ,
                delta_chunk_size=DELTA_CHUNK_SIZE,
            )

        print(f"\n[{tag}]\n  ===== Bayesian (Seq-LoRA) Only =====")
        print(f"  nll_bayes: {metrics['nll_bayes']:.4f}")
        print(f"  brier_bayes: {metrics['brier_bayes']:.4f}")
        print(f"  ece_bayes: {metrics['ece_bayes']*100:.2f}%")
        print(f"  acc_bayes: {metrics['acc_bayes']*100:.2f}%")
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
