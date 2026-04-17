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

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from kfac import calculate_kronecker_factors
from lssm_ffbs_obs import kalman_filter, lag_one_smoothed_covariances, rts_smoother
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
EVAL_BSZ = 48
KFAC_BSZ = 8

# KFAC / train-slice loaders remain conservative
NUM_WORKERS = 0

# Eval loader gets its own workers for dynamic padding pipeline
EVAL_NUM_WORKERS = 0
EVAL_PREFETCH_FACTOR = 4

N_KFAC = 16
LR_THRESHOLD = 256
MAX_KFAC_SAMPLES_PER_SLICE = 2048

MU_OBS_SCALE = 2
MU_OBS_BATCHES = 32
S_Q = 1.0
Q_MODE = "module_constant"
P1_VAR = 1.0

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 32

ADAPTIVE_Q_WARMSTART_VAR = 1.0
ADAPTIVE_Q_EIG_FLOOR = 1e-8

MIN_MEAN_TRACE_G = 1e-12
PI_MAX = 1e8
HG_DIAG_SHRINKAGE = 1e-3

POSTERIOR_TAU = 1.0
TEMP_BAYES = 1.0

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

        mu_parts = [
            solve_xhat_from_grad(
                module_R_lists[name][t].to(device=device, dtype=dtype),
                g_x_parts[mi] / float(n_seen),
            )
            for mi, name in enumerate(module_names)
        ]
        mu_global_list.append(torch.cat(mu_parts, dim=0).cpu())

    return mu_global_list

def summarize_h_g_factor_stats(
    H_factors: List[Tensor],
    G_factors: List[Tensor],
) -> Dict[str, float]:
    mean_trace_H = torch.stack([
        trace_psd_factor(H_t) / H_factors[0].shape[0] for H_t in H_factors
    ]).mean()
    mean_trace_G = torch.stack([
        trace_psd_factor(G_t) / G_factors[0].shape[0] for G_t in G_factors
    ]).mean()
    stats = {
        "mean_trace_H": float(mean_trace_H.item()),
        "mean_trace_G": float(mean_trace_G.item()),
        "pi": 1.0,
        "mean_trace_H_bal": float(mean_trace_H.item()),
        "mean_trace_G_bal": float(mean_trace_G.item()),
    }
    return stats


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

    def _fmt_value(key: str, value: float) -> str:
        if key in {"mean_trace_G", "mean_trace_G_bal", "robust_scale_G"}:
            return f"{value:.6e}"
        return f"{value:.6f}"

    def _summ(key: str) -> str:
        vals = [float(stats[key]) for stats in module_stats.values()]
        return (
            f"min={_fmt_value(key, min(vals))} "
            f"mean={_fmt_value(key, sum(vals)/len(vals))} "
            f"max={_fmt_value(key, max(vals))}"
        )

    print("\n=== H/G/pi Report ===")
    print("[H/G/pi] Relative eig floors are enabled, so pi is fixed to 1.0.")
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
            f"G={stats['mean_trace_G']:.6e} "
            f"pi={stats['pi']:.6f} "
            f"H_bal={stats['mean_trace_H_bal']:.6f} "
            f"G_bal={stats['mean_trace_G_bal']:.6e}"
        )


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
    mc_eval_chunk: int = 0,
    progress_desc: Optional[str] = None,
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
                )
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
        acc_bay_m.update(probs_bayes_batch, labels)
        ece_bay_m.update(probs_bayes_batch, labels)
        del probs_acc_batch, probs_bayes_batch, ids, attn, labels

    bayes_extra_time = time.perf_counter() - bayes_t0

    metrics = {
        "nll_bayes": nll_sum / max(total_samples, 1),
        "brier_bayes": brier_sum / max(total_samples, 1),
        "ece_bayes": float(ece_bay_m.compute().item()),
        "acc_bayes": float(acc_bay_m.compute().item()),
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
    parser.add_argument(
        "--min_mean_trace_g",
        type=float,
        default=MIN_MEAN_TRACE_G,
        help="Deprecated no-op retained for CLI compatibility after removing explicit H/G pi balancing.",
    )
    parser.add_argument(
        "--pi_max",
        type=float,
        default=PI_MAX,
        help="Deprecated no-op retained for CLI compatibility after removing explicit H/G pi balancing.",
    )
    parser.add_argument(
        "--hg_diag_shrinkage",
        type=float,
        default=HG_DIAG_SHRINKAGE,
        help="Deprecated no-op retained for CLI compatibility after removing explicit H/G pi balancing.",
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
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = args.tokenizer_padding_side

    num_classes = get_task_num_classes(args.task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=bool(args.trust_remote_code),
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
            args.max_seq_len,
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
                    n_kfac=args.n_kfac,
                    lr_threshold=args.lr_threshold,
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

        module_subspace_info, module_R_lists, module_Hx_lists = {}, {}, {}
        module_hgpi_stats: Dict[str, Dict[str, float]] = {}
        for name in module_names:
            H_factors = H_factor_per_module[name]
            G_factors = G_factor_per_module[name]
            hgpi_stats = summarize_h_g_factor_stats(H_factors, G_factors)
            module_hgpi_stats[name] = hgpi_stats

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
            args.mu_obs_batches,
            torch.float64,
        )
        mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list]

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
        del mu_global_list
        x_samples_T = torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)
        del x_sample_parts

    lora_cache = build_loraA_cache(model, module_specs, device=device)

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
                posterior_scale_tau=args.posterior_tau,
                temp_bayes=args.temp_bayes,
                max_mc_samples=args.mc_eval_samples,
                mc_eval_chunk=args.mc_eval_chunk,
                progress_desc=f"SEQ {tag}",
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
