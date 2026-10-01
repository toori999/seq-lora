from __future__ import annotations
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Tuple, Optional, Sequence
import os
import random
import math
import time
import argparse
import gc
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional for headless runs.
    _tqdm = None

import datasets as hf_datasets
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from seq_lora.algorithm.kfac import (
    calculate_kronecker_factors,
    forward_call_for_kfac_factory,
)
from seq_lora.algorithm.lora_state import (
    LoraACache as _LoraACache,
    build_loraA_cache,
    compute_deltas_for_one_sample as _compute_deltas_for_one_sample,
)
from seq_lora.algorithm.pseudo_observations import (
    estimate_mu_global_list_from_slice_grads_asdl,
)
from seq_lora.algorithm.posterior import (
    move_subspace_info as _move_subspace_info,
    sample_independent_slice_ensemble_from_stats as _sample_independent_slice_ensemble_from_stats,
    sample_lgssm_posterior_from_stats as _sample_posterior_from_stats,
)
from seq_lora.algorithm.subspace import (
    build_global_kronecker_eigenspace,
    materialize_mean_psd_from_factors,
    project_curvature_factors_to_subspace,
)
from seq_lora.algorithm.tau import (
    fit_posterior_scale_from_anchor_kl as _fit_posterior_scale_from_anchor_kl,
)
from evaluation.bayesian_peft_data import (
    BayesianPeftCLMCollator as _BayesianPeftCLMCollator,
    build_bayesian_peft_task_dataset as _build_bayesian_peft_task_dataset,
    get_direct_bayesian_peft_eval_dataset as _get_direct_bayesian_peft_eval_dataset,
    make_direct_bayesian_peft_loader as _make_direct_bayesian_peft_loader,
    uses_direct_bayesian_peft_data as _uses_direct_bayesian_peft_data,
)
from seq_lora.eval.cache import (
    build_posterior_stats_cache_payload as _build_posterior_stats_cache_payload,
    build_posterior_stats_cache_snapshot as _build_posterior_stats_cache_snapshot,
    dataset_fingerprint as _dataset_fingerprint,
    load_posterior_stats_cache as _load_posterior_stats_cache,
)
from evaluation.logits import (
    compute_choice_logits,
    mask_invalid_choices as _mask_invalid_choices,
    trim_lm_head_to_choice_tokens,
)
from evaluation.runtime import (
    StageTimer as _StageTimer,
    multiclass_brier_sum as _multiclass_brier_sum,
    normalize_posterior_stats_dtype as _normalize_posterior_stats_dtype,
    parse_bool as _parse_bool,
    torch_dtype_from_name as _torch_dtype_from_name,
)

Tensor = torch.Tensor

warnings.filterwarnings(
    "ignore",
    message=(
        "Using a non-full backward hook when the forward contains multiple autograd "
        "Nodes is deprecated and will be removed in future versions.*"
    ),
    category=FutureWarning,
)


# =========================
# Config Defaults
# =========================

SEED = 0
TRUST_REMOTE_CODE = False

MAX_SEQ_LEN = 300
EVAL_BSZ = 96
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
MAX_KFAC_SAMPLES_PER_SLICE = 256
KFAC_BACKEND = "asdl"
POSTERIOR_STATS_DTYPE = "float32"

MU_OBS_SCALE = 2
MU_OBS_BATCHES = 32
S_Q = 1.0
Q_MODE = "module_constant"
MODULE_Q_CLIP_MIN = 0.5
MODULE_Q_CLIP_MAX = 2.0
MODULE_Q_SHRINK_EXPONENT = 0.05
P1_VAR = 1.0

SUBSPACE_DIM_PER_MODULE = 64
MC_EVAL_SAMPLES = 10
POSTERIOR_EVAL_MODE = "lgssm_final"
INDEPENDENT_SLICE_MC_SAMPLES_PER_SLICE = 3

POSTERIOR_TAU = 0.625
POSTERIOR_PROB_SMOOTHING_POWER = 1.0 / 1.05
DISABLE_DROPOUT_DURING_KFAC_MU = True
TAU_MODE = "fixed"
TAU_SEARCH_MAX = 2.0
TAU_ANCHOR_SIZE = 500
TAU_ANCHOR_BSZ = EVAL_BSZ
TAU_ANCHOR_N_SAMPLES = 32
TAU_KL_TARGET_LOW = 0.05
TAU_KL_TARGET_HIGH = 0.0525

TOKENIZER_PADDING_SIDE = "left"
BAYESIAN_PEFT_ADD_EOS = False
IID_EVAL_SPLIT = "validation"
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
HF_DATASETS_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".hf_datasets")

from evaluation.common import (
    SCIENCEQA_CURRIC_TASK_NAME,
    DynamicEvalCollator,
    load_eval_dataset,
    load_iid_test_set,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    preprocess_task,
    set_inference_fast as _set_inference_fast,
)
from evaluation.checkpoints import (
    load_adapter_checkpoint as _load_adapter_checkpoint,
    remap_bayesian_peft_adapter_keys as _remap_bayesian_peft_adapter_keys,
)
from evaluation.protocols import (
    get_num_classes_for_protocol as _get_num_classes_for_protocol,
    get_target_token_ids_for_protocol as _get_target_token_ids_for_protocol,
    is_bayesian_peft_protocol as _is_bayesian_peft_protocol,
    normalize_eval_protocol as _normalize_eval_protocol,
    preprocess_task_for_protocol as _preprocess_task_for_protocol,
)
from seq_lora.data.slices import (
    assign_random_slice_ids as _assign_random_slice_ids,
)


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
        elif task in {"mmlu-chem", "mmlu_chem", "mmlu-chemistry", "mmlu_chemistry"}:
            expanded.append("mmlu-chem")
        elif task in {"mmlu-phy", "mmlu_phy", "mmlu-physics", "mmlu_physics"}:
            expanded.append("mmlu-phy")
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
# Fast Bayesian eval
# =========================

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
    max_mc_samples: int = 32,
    mc_eval_chunk: int = 0,
    progress_desc: Optional[str] = None,
    progress_every: int = 0,
    progress_bar: bool = False,
    apply_choice_mask: bool = True,
) -> Dict[str, float]:
    model.eval()
    _set_inference_fast(model)

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
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    progress_every = max(int(progress_every), 0)
    iterator = loader
    tqdm_iter = None
    if bool(progress_bar) and _tqdm is not None:
        tqdm_iter = _tqdm(
            loader,
            total=total_batches,
            desc=progress_desc or "Seq-LoRA eval",
            dynamic_ncols=True,
            leave=False,
        )
        iterator = tqdm_iter
    for batch_idx, batch in enumerate(iterator, start=1):
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

        if POSTERIOR_PROB_SMOOTHING_POWER != 1.0:
            p = probs_bayes_batch.clamp_min(eps) ** float(POSTERIOR_PROB_SMOOTHING_POWER)
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
        if progress_every and batch_idx % progress_every == 0:
            elapsed = time.perf_counter() - bayes_t0
            if total_batches is None:
                where = f"batch={batch_idx}"
            else:
                where = f"batch={batch_idx}/{total_batches}"
            label = progress_desc or "Seq-LoRA eval"
            print(
                f"[Eval progress] {label} {where} "
                f"elapsed={elapsed:.1f}s avg_batch={elapsed / batch_idx:.2f}s",
                flush=True,
            )
        if tqdm_iter is not None and batch_idx % 5 == 0:
            elapsed = time.perf_counter() - bayes_t0
            tqdm_iter.set_postfix_str(
                f"avg_batch={elapsed / batch_idx:.2f}s",
                refresh=False,
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



# =========================
# Main
# =========================

def main(default_eval_protocol: str = "default", argv: Optional[Sequence[str]] = None):
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
        "--kfac_backend",
        type=str,
        default=KFAC_BACKEND,
        choices=["asdl"],
        help="KFAC backend. Only ASDL Kron is supported.",
    )
    parser.add_argument(
        "--posterior_stats_dtype",
        type=_normalize_posterior_stats_dtype,
        default=POSTERIOR_STATS_DTYPE,
        help=(
            "Floating dtype for Seq-LoRA posterior statistics during KFAC projection "
            "and cache storage. Use float32 to reduce memory; float64 keeps legacy precision."
        ),
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
        choices=["obqa", SCIENCEQA_CURRIC_TASK_NAME],
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
        help="Order in which slice ids are fed to the LGSSM/Kalman chain.",
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
        help="Comma-separated eval tasks. Supports iid, arc, arc-c, arc-e, mmlu-chem, mmlu-phy, mmlu, mmlu_science_high, mmlu_science_college, gpqa_main, scienceqa_closedchoice_grade12.",
    )
    parser.add_argument(
        "--s_q",
        type=float,
        default=float(S_Q),
        help=(
            "Global process-noise scale. In constant mode it sets Q_t = s_Q * I. "
            "In module_constant mode it scales the learned module multipliers."
        ),
    )
    parser.add_argument(
        "--q_mode",
        type=str,
        choices=["constant", "module_constant"],
        default=Q_MODE,
        help=(
            "Process-noise construction. constant uses shared Q_t = s_Q * I; "
            "module_constant uses per-module q_m from adjacent-mu drift."
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
            "Log-normal shrinkage exponent beta for module_constant Q. "
            "q_m/s_Q = clip((drift_m / median_drift) ** beta). "
            "0 makes all modules equal; 1 recovers the raw adjacent-mu drift heuristic."
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
    parser.add_argument(
        "--eval_progress_every",
        type=int,
        default=0,
        help="Print eval-loop progress every N batches. 0 disables progress logging.",
    )
    parser.add_argument(
        "--eval_progress_bar",
        type=_parse_bool,
        default=True,
        help="Show a tqdm progress bar for Seq-LoRA evaluation loops.",
    )
    parser.add_argument("--posterior_tau", type=float, default=POSTERIOR_TAU, help="Posterior scale multiplier used at evaluation.")
    parser.add_argument(
        "--tau_mode",
        type=str,
        default=TAU_MODE,
        choices=["fixed", "auto"],
        help="fixed uses --posterior_tau. auto estimates tau from anchor KL using KL(tau) ~= C * tau^2.",
    )
    parser.add_argument(
        "--tau_search_max",
        type=float,
        default=TAU_SEARCH_MAX,
        help="Upper bound for automatic posterior_tau estimation.",
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
            "the expensive KFAC and mu stages, while still allowing q/tau/MC sweeps."
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
        "--eval_protocol",
        type=str,
        default=default_eval_protocol,
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
            "Useful for local full-vocab checkpoints trained under the default prompt protocol."
        ),
    )
    args = parser.parse_args(argv)
    args.posterior_stats_dtype = _normalize_posterior_stats_dtype(args.posterior_stats_dtype)
    posterior_stats_dtype = _torch_dtype_from_name(args.posterior_stats_dtype)

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
    args.eval_protocol = eval_protocol
    apply_choice_mask = not _is_bayesian_peft_protocol(eval_protocol)
    keep_full_vocab_lm_head = bool(args.keep_full_vocab_lm_head) or _is_bayesian_peft_protocol(eval_protocol)
    args.keep_full_vocab_lm_head = bool(keep_full_vocab_lm_head)
    print(f"[Protocol] eval_protocol={eval_protocol}")
    print(
        "[Curvature backend] Using ASDL Kron on the full wrapper graph "
        "with randomized PSD low-rank compression for large blocks."
    )
    print(
        f"[KFAC] disable_dropout_during_kfac_mu={bool(args.disable_dropout_during_kfac_mu)}"
    )
    print(f"[Seq-LoRA] posterior_stats_dtype={args.posterior_stats_dtype}")

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
    # Eval set: dynamic padding + length sorting
    # -------------------------
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    eval_task_to_proc: Dict[str, object] = {}
    eval_task_to_num_classes: Dict[str, int] = {}
    eval_task_to_choice_token_ids: Dict[str, Tensor] = {}
    eval_task_to_apply_choice_mask: Dict[str, bool] = {}
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
            eval_choice_token_ids = eval_task_dataset.target_ids.view(-1).to(device=device, dtype=torch.long)
            eval_apply_choice_mask = False
        else:
            eval_num_classes = _get_num_classes_for_protocol(eval_task, eval_protocol)
            eval_choice_token_ids = _get_target_token_ids_for_protocol(
                tokenizer,
                eval_task,
                eval_protocol,
                device,
                add_space=bool(args.bayesian_peft_add_space),
            )
            eval_apply_choice_mask = apply_choice_mask
        eval_task_to_num_classes[eval_task] = int(eval_num_classes)
        eval_task_to_choice_token_ids[eval_task] = eval_choice_token_ids
        eval_task_to_apply_choice_mask[eval_task] = bool(eval_apply_choice_mask)
        if use_direct_eval_bayesian_peft:
            eval_split = str(args.iid_eval_split) if eval_task == args.task else "test"
            eval_task_to_proc[eval_task] = _make_direct_bayesian_peft_loader(
                _get_direct_bayesian_peft_eval_dataset(eval_task_dataset, eval_split),
                collate_fn=_BayesianPeftCLMCollator(eval_task_dataset),
                batch_size=args.eval_bsz,
                shuffle=False,
                drop_last=False,
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

    natural_slice_ids = sorted(set(int(x) for x in train_raw["slice_id"]))
    slice_ids = list(natural_slice_ids)
    if str(args.slice_order) == "reverse":
        slice_ids = list(reversed(slice_ids))
    elif str(args.slice_order) == "shuffle":
        rng = random.Random(int(args.slice_order_seed))
        rng.shuffle(slice_ids)
    print(
        f"[Slices] source={slice_source} natural_ids={natural_slice_ids} "
        f"order={args.slice_order} order_seed={int(args.slice_order_seed)} run_ids={slice_ids}",
        flush=True,
    )
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
            eff_samples = len(slice_indices)
            eff_batches = math.ceil(eff_samples / max(int(args.kfac_bsz), 1))
            total_kfac_samples += eff_samples
            total_kfac_batches += eff_batches
            ds_loader = Subset(train_raw, slice_indices)
            slice_loaders.append(
                _make_direct_bayesian_peft_loader(
                    ds_loader,
                    collate_fn=direct_bayesian_peft_collator,
                    batch_size=args.kfac_bsz,
                    shuffle=False,
                    drop_last=False,
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
    train_raw_fingerprint = _dataset_fingerprint(train_raw)
    train_proc_fingerprint = (
        _dataset_fingerprint(train_proc)
        if train_proc is not None
        else f"direct_bayesian_peft:{_dataset_fingerprint(train_raw)}"
    )
    stats_cache_snapshot = _build_posterior_stats_cache_snapshot(
        args,
        slice_ids=slice_ids,
        train_raw_fingerprint=train_raw_fingerprint,
        train_proc_fingerprint=train_proc_fingerprint,
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
        stats_cache_loaded = True
        print(f"[Posterior Stats Cache] loaded from {stats_cache_path} (modules={len(module_specs)} T={cached_T})")

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
                        A_t.detach().to(dtype=posterior_stats_dtype, device=cpu_device)
                    )
                    G_factor_per_module[name].append(
                        S_t.detach().to(dtype=posterior_stats_dtype, device=cpu_device)
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
                    dtype=posterior_stats_dtype,
                )
                G_bar_bal = materialize_mean_psd_from_factors(
                    G_factors,
                    matrix_scale=1.0,
                    device=device,
                    dtype=posterior_stats_dtype,
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
                    dtype=posterior_stats_dtype,
                    return_h_x=False,
                )
                module_subspace_info[name] = _move_subspace_info(
                    subspace_info_gpu,
                    device=cpu_device,
                    dtype=posterior_stats_dtype,
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
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[KFAC] all slices done; projecting curvature into LoRA subspaces", flush=True)

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
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[mu-obs] estimating slice-wise pseudo-observations", flush=True)
            mu_global_list_raw = estimate_mu_global_list_from_slice_grads_asdl(
                model,
                slice_loaders,
                forward_call_for_kfac,
                module_names,
                module_subspace_info,
                module_R_lists,
                device,
                args.mu_obs_batches,
                posterior_stats_dtype,
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

    effective_posterior_tau = float(args.posterior_tau)
    if str(args.tau_mode) == "auto":
        if use_direct_source_bayesian_peft:
            tau_anchor_raw = _subset_dataset(
                train_raw,
                subset_size=int(args.tau_anchor_size),
                seed=int(args.seed),
            )
            tau_loader_kwargs = {
                "batch_size": max(int(args.tau_anchor_bsz), 1),
                "shuffle": False,
                "drop_last": False,
                "collate_fn": direct_bayesian_peft_collator,
                "num_workers": args.eval_num_workers,
                "pin_memory": pin_memory,
            }
            if args.eval_num_workers > 0:
                tau_loader_kwargs["persistent_workers"] = True
                tau_loader_kwargs["prefetch_factor"] = args.eval_prefetch_factor
            tau_anchor_loader = DataLoader(tau_anchor_raw, **tau_loader_kwargs)
            tau_anchor_rows = len(tau_anchor_raw)
        else:
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
            tau_anchor_rows = len(tau_anchor_proc)
        print(
            f"[Tau auto] anchor source=train rows={tau_anchor_rows} "
            f"batch_size={tau_loader_kwargs['batch_size']} "
            f"mc_samples={int(args.tau_anchor_n_samples)}",
            flush=True,
        )
        with _StageTimer(f"FIT Seq-LoRA tau on {args.task}(train_anchor)"):
            def _eval_tau_anchor(tau: float, n_samples: int, desc: str) -> Dict[str, float]:
                return eval_bayes_fast_restricted_4way_probmean(
                    model=model,
                    loader=tau_anchor_loader,
                    device=device,
                    amp_dtype=amp_dtype,
                    num_classes=num_classes,
                    choice_token_ids=choice_token_ids,
                    lora_cache=lora_cache,
                    x_samples_T=x_samples_T,
                    posterior_scale_tau=float(tau),
                    max_mc_samples=max(int(n_samples), 1),
                    mc_eval_chunk=int(args.mc_eval_chunk),
                    progress_desc=desc,
                    progress_every=int(args.eval_progress_every),
                    progress_bar=bool(args.eval_progress_bar),
                    apply_choice_mask=apply_choice_mask,
                )

            tau_fit_info = _fit_posterior_scale_from_anchor_kl(
                _eval_tau_anchor,
                tau_max=float(args.tau_search_max),
                anchor_n_samples=int(args.tau_anchor_n_samples),
                kl_target_low=float(args.tau_kl_target_low),
                kl_target_high=float(args.tau_kl_target_high),
            )
        effective_posterior_tau = float(tau_fit_info["optimal_posterior_tau"])
        del tau_anchor_loader, tau_fit_info
        if "tau_anchor_eval" in locals():
            del tau_anchor_eval
        if "tau_anchor_proc" in locals():
            del tau_anchor_proc
        if "tau_anchor_raw" in locals():
            del tau_anchor_raw
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[Tau] mode={args.tau_mode} effective_posterior_tau={effective_posterior_tau:.6f}", flush=True)

    def eval_one(
        tag: str,
        proc_or_loader,
        *,
        eval_num_classes: int,
        eval_choice_token_ids: Tensor,
        eval_apply_choice_mask: bool,
    ):
        if isinstance(proc_or_loader, DataLoader):
            loader = proc_or_loader
        else:
            proc_eval = (
                proc_or_loader.remove_columns(["seq_len"])
                if "seq_len" in proc_or_loader.column_names
                else proc_or_loader
            )
            loader = DataLoader(proc_eval, **eval_loader_kwargs)

        print(
            f"[Eval] start tag={tag} rows={len(loader.dataset)} "
            f"batches={len(loader)} mc_samples={int(args.mc_eval_samples)} "
            f"mc_chunk={int(args.mc_eval_chunk)} tau={float(effective_posterior_tau):.6f}",
            flush=True,
        )
        with _StageTimer(f"INFER Seq-LoRA on {tag}"):
            metrics = eval_bayes_fast_restricted_4way_probmean(
                model=model,
                loader=loader,
                device=device,
                amp_dtype=amp_dtype,
                num_classes=int(eval_num_classes),
                choice_token_ids=eval_choice_token_ids,
                lora_cache=lora_cache,
                x_samples_T=x_samples_T,
                posterior_scale_tau=effective_posterior_tau,
                max_mc_samples=args.mc_eval_samples,
                mc_eval_chunk=args.mc_eval_chunk,
                progress_desc=f"SEQ {tag}",
                progress_every=int(args.eval_progress_every),
                progress_bar=bool(args.eval_progress_bar),
                apply_choice_mask=bool(eval_apply_choice_mask),
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

    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Evaluation: source={args.task} | targets={eval_tasks} ===", flush=True)
    for eval_task in eval_tasks:
        split_name = "iid" if eval_task == args.task else "ood"
        eval_one(
            f"{eval_task}_{split_name}",
            eval_task_to_proc[eval_task],
            eval_num_classes=eval_task_to_num_classes[eval_task],
            eval_choice_token_ids=eval_task_to_choice_token_ids[eval_task],
            eval_apply_choice_mask=eval_task_to_apply_choice_mask[eval_task],
        )
    print(f"\n[Done] Evaluation complete for source task {args.task}.")

if __name__ == "__main__":
    main()
