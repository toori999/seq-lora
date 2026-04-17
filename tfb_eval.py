from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import Dataset

from peft import PeftConfig, PeftModel
from peft.tuners.lora import LoraLayer, Linear as LoraLinear

try:
    from peft.tuners.lora.bnb import Linear8bitLt
except Exception:
    Linear8bitLt = None

from transformers import AutoModelForCausalLM, AutoTokenizer

from blob_eval_iid_official import (
    _iter_active_adapters,
    _iter_lora_linear_modules,
    _official_blob_8bitlinear_forward,
    _official_blob_linear_forward,
    _official_sample,
    compute_class_logits,
    set_blob_sampling,
)
from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    DynamicEvalCollator,
    get_choice_token_ids,
    get_task_num_classes,
    get_transformer_and_lm_head,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy,
    make_ece,
    preprocess_task,
)


SEED = 0
TRUST_REMOTE_CODE = False
MAX_SEQ_LEN = 300
EVAL_BSZ = 32
ANCHOR_BSZ = 32
TFB_ANCHOR_N_SAMPLES = 10
TFB_EVAL_N_SAMPLES = 10
TFB_BETA_MAX = 0.015
TFB_THRESHOLD = 0.003
TFB_BINARY_SEARCH_ITERS = 5
TFB_BAYES_EPS = 0.05


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


def _add_seq_len(ds):
    if "seq_len" in ds.column_names:
        return ds
    return ds.add_column("seq_len", [len(x) for x in ds["input_ids"]])


def trim_lm_head_to_choice_tokens(model: nn.Module, choice_token_ids: torch.Tensor) -> None:
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


def _force_lora_fp32(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(dtype=torch.float32)


def _load_base_and_adapter(
    task: str,
    map_adapter_dir: str,
    amp_dtype: torch.dtype,
    device: torch.device,
):
    if not os.path.isdir(map_adapter_dir):
        raise RuntimeError(f"Adapter dir not found: {map_adapter_dir}")

    peft_cfg = PeftConfig.from_pretrained(map_adapter_dir)
    base_name = peft_cfg.base_model_name_or_path
    print(f"[Load] base_model = {base_name}")
    print(f"[Load] adapter    = {map_adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "left"

    num_classes = get_task_num_classes(task)
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

    model = PeftModel.from_pretrained(base_model, map_adapter_dir).to(device)
    model.eval()
    _force_lora_fp32(model)
    return tokenizer, model, num_classes


def _make_eval_loader(proc, tokenizer, device: torch.device, bsz: int) -> DataLoader:
    if "seq_len" in proc.column_names:
        proc = proc.sort("seq_len").remove_columns(["seq_len"])
    collator = DynamicEvalCollator(tokenizer=tokenizer, pad_to_multiple_of=(8 if device.type == "cuda" else None))
    return DataLoader(
        proc,
        batch_size=bsz,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def _std_to_rho(std: torch.Tensor, bayes_eps: float) -> torch.Tensor:
    std = std.clamp_min(1e-12)
    if float(bayes_eps) < 0:
        return torch.log(torch.expm1(std))
    return torch.sqrt(std)


def _reparameterize_tfb_layer(layer: nn.Module, adapter_name: str, bayes_beta: float, bayes_eps: float) -> None:
    dtype_A = layer.lora_A[adapter_name].weight.dtype
    dtype_B = layer.lora_B[adapter_name].weight.dtype
    lora_A = layer.lora_A[adapter_name].weight.float()
    lora_B = layer.lora_B[adapter_name].weight.float()

    U, S, V = torch.svd(lora_B)
    std = float(bayes_beta) / (S.reshape(-1, 1).expand(-1, layer.in_features) + 1e-6)

    if not hasattr(layer, "lora_A_rho") or not isinstance(layer.lora_A_rho, nn.ParameterDict):
        layer.lora_A_rho = nn.ParameterDict({})
    layer.lora_A_rho[adapter_name] = nn.Parameter(_std_to_rho(std, bayes_eps).to(dtype=torch.float32))

    layer.lora_B[adapter_name].weight = nn.Parameter((U @ torch.diag(S)).to(dtype=dtype_B))
    layer.lora_A[adapter_name].weight = nn.Parameter((V.transpose(0, 1) @ lora_A).to(dtype=dtype_A))


def _update_tfb_beta_layer(layer: nn.Module, adapter_name: str, bayes_beta: float, bayes_eps: float) -> None:
    lora_B = layer.lora_B[adapter_name].weight.float()
    _, S, _ = torch.svd(lora_B)
    std = float(bayes_beta) / (S.reshape(-1, 1).expand(-1, layer.in_features) + 1e-6)
    rho = layer.lora_A_rho[adapter_name]
    layer.lora_A_rho[adapter_name] = nn.Parameter(
        _std_to_rho(std, bayes_eps).to(device=rho.device, dtype=rho.dtype)
    )
    layer.bayes_beta = float(bayes_beta)


def wrap_tfb_lora_layers(model: PeftModel, adapter_name: str, bayes_eps: float, bayes_beta: float) -> int:
    wrapped = 0
    for _, mod in _iter_lora_linear_modules(model):
        if not isinstance(mod, LoraLayer):
            continue
        if not isinstance(mod, LoraLinear) and not (Linear8bitLt is not None and isinstance(mod, Linear8bitLt)):
            continue
        A_dict = getattr(mod, "lora_A", None)
        if not isinstance(A_dict, nn.ModuleDict) or adapter_name not in A_dict:
            continue
        mod.bayes_eps = float(bayes_eps)
        mod.bayes_beta = float(bayes_beta)
        mod.blobsample = True
        mod.sample = _official_sample.__get__(mod, mod.__class__)
        if Linear8bitLt is not None and isinstance(mod, Linear8bitLt):
            mod.forward = _official_blob_8bitlinear_forward.__get__(mod, mod.__class__)
        else:
            mod.forward = _official_blob_linear_forward.__get__(mod, mod.__class__)
        _reparameterize_tfb_layer(mod, adapter_name, bayes_beta, bayes_eps)
        wrapped += 1
    print(f"[TFB] patched and reparameterized {wrapped} LoRA layers (adapter={adapter_name}).")
    if wrapped == 0:
        raise RuntimeError("No LoRA layers were patched for TFB.")
    return wrapped


def update_tfb_beta(model: nn.Module, adapter_name: str, bayes_beta: float, bayes_eps: float) -> None:
    for _, mod in _iter_lora_linear_modules(model):
        if not isinstance(mod, LoraLayer):
            continue
        if not hasattr(mod, "lora_A_rho") or adapter_name not in mod.lora_A_rho:
            continue
        _update_tfb_beta_layer(mod, adapter_name, bayes_beta, bayes_eps)


@torch.inference_mode()
def _collect_probs(
    model: PeftModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    n_samples: int,
    sample: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    if sample:
        set_blob_sampling(model, "default", True)
    else:
        set_blob_sampling(model, "default", False)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")

        if not sample:
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            if num_choices is not None:
                num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
                col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
                invalid = col_idx >= num_choices_t.view(-1, 1)
                logits = logits.masked_fill(invalid, -1e9)
            probs = torch.softmax(logits, dim=-1)
        else:
            probs_acc = torch.zeros((labels.size(0), num_classes), device=device, dtype=torch.float32)
            for _ in range(int(n_samples)):
                logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
                if num_choices is not None:
                    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
                    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
                    invalid = col_idx >= num_choices_t.view(-1, 1)
                    logits = logits.masked_fill(invalid, -1e9)
                probs_acc.add_(torch.softmax(logits, dim=-1))
            probs = probs_acc / float(n_samples)

        all_probs.append(probs.detach())
        all_labels.append(labels.detach())

    set_blob_sampling(model, "default", True)
    probs_all = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, num_classes), device=device)
    labels_all = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long, device=device)
    return probs_all, labels_all


def _multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


@torch.inference_mode()
def evaluate_tfb_official_style(
    model: PeftModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    n_samples: int,
    progress_desc: str,
) -> Dict[str, float]:
    model.eval()
    model.set_adapter("default")
    acc_metric = make_accuracy(device, num_classes)
    ece_metric = make_ece(device, num_classes, 10)
    acc_metric.reset()
    ece_metric.reset()

    nll_sum = 0.0
    total = 0
    flip_count = 0
    total_count = 0
    brier_sum = 0.0
    last_std = 0.0

    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress_start = time.perf_counter()
    progress = tqdm(
        total=(total_samples if total_samples is not None else len(loader)),
        desc=progress_desc,
        unit=("sample" if total_samples is not None else "batch"),
        leave=False,
    )

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")
        bsz = labels.size(0)
        total += bsz
        total_count += bsz

        set_blob_sampling(model, "default", True)
        stochastic_logits: List[torch.Tensor] = []
        for _ in range(int(n_samples)):
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            if num_choices is not None:
                num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
                col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
                invalid = col_idx >= num_choices_t.view(-1, 1)
                logits = logits.masked_fill(invalid, -1e9)
            stochastic_logits.append(logits)
        logits_stochastic = torch.stack(stochastic_logits, dim=1)

        set_blob_sampling(model, "default", False)
        logits_deterministic = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
        if num_choices is not None:
            num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits_deterministic.device, dtype=torch.long)
            col_idx = torch.arange(logits_deterministic.size(-1), device=logits_deterministic.device).view(1, -1)
            invalid = col_idx >= num_choices_t.view(-1, 1)
            logits_deterministic = logits_deterministic.masked_fill(invalid, -1e9)
        set_blob_sampling(model, "default", True)

        probs_stochastic = torch.softmax(logits_stochastic, dim=-1).mean(dim=1)
        probs_deterministic = torch.softmax(logits_deterministic, dim=-1)

        pred_stochastic = probs_stochastic.argmax(dim=-1)
        pred_deterministic = probs_deterministic.argmax(dim=-1)
        flip_count += int((pred_stochastic != pred_deterministic).sum().item())
        last_std = float(torch.softmax(logits_stochastic, dim=-1).std(dim=1).mean().item()) if int(n_samples) > 1 else 0.0

        acc_metric.update(probs_stochastic, labels)
        ece_metric.update(probs_stochastic, labels)
        nll_sum += float((-torch.log(probs_stochastic[torch.arange(bsz, device=device), labels].clamp_min(1e-12))).sum().item())
        brier_sum += float(
            (probs_stochastic - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )

        elapsed = time.perf_counter() - progress_start
        avg_sec_per_sample = elapsed / max(total, 1)
        if total_samples is not None:
            remaining_samples = max(int(total_samples) - total, 0)
            eta_seconds = avg_sec_per_sample * remaining_samples
            progress.update(bsz)
        else:
            remaining_batches = max(len(loader) - progress.n - 1, 0)
            eta_seconds = (elapsed / max(progress.n + 1, 1)) * remaining_batches
            progress.update(1)
        progress.set_postfix(avg_s_per_sample=f"{avg_sec_per_sample:.3f}", eta=_format_eta(eta_seconds), refresh=False)

    progress.close()
    set_blob_sampling(model, "default", True)

    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_metric.compute().item()),
        "ece": float(ece_metric.compute().item()),
        "brier": (brier_sum / max(total, 1) if total > 0 else float("nan")),
        "std": last_std,
        "flip_ratio": (float(flip_count) / float(total_count) if total_count > 0 else 0.0),
    }


@torch.inference_mode()
def fit_tfb_beta(
    model: PeftModel,
    anchor_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    bayes_beta_max: float,
    threshold: float,
    search_iters: int,
    anchor_n_samples: int,
    bayes_eps: float,
) -> Dict[str, float]:
    model.eval()
    model.set_adapter("default")
    update_tfb_beta(model, "default", 0.0, bayes_eps)

    ref_probs, _ = _collect_probs(
        model=model,
        loader=anchor_loader,
        device=device,
        amp_dtype=amp_dtype,
        num_classes=num_classes,
        n_samples=1,
        sample=True,
    )
    ref_preds = ref_probs.argmax(dim=-1)
    ref_nll = F.nll_loss(torch.log(ref_probs.clamp_min(1e-12)), ref_preds)

    low, high = 0.001, float(bayes_beta_max)
    best_beta = high

    for step in range(int(search_iters)):
        mid = (low + high) / 2.0
        update_tfb_beta(model, "default", mid, bayes_eps)
        probs_mid, _ = _collect_probs(
            model=model,
            loader=anchor_loader,
            device=device,
            amp_dtype=amp_dtype,
            num_classes=num_classes,
            n_samples=anchor_n_samples,
            sample=True,
        )
        cur_nll = F.nll_loss(torch.log(probs_mid.clamp_min(1e-12)), ref_preds)
        ratio = (abs(cur_nll.item() - ref_nll.item()) / max(ref_nll.item(), 1e-12)) / max(ref_preds.numel(), 1)
        print(
            f"[TFB fit] iter={step+1}/{int(search_iters)} beta={mid:.6f} "
            f"ref_nll={ref_nll.item():.6f} cur_nll={cur_nll.item():.6f} ratio={ratio:.8f}"
        )
        if ratio > float(threshold):
            best_beta = mid
            high = mid
        else:
            low = mid

    update_tfb_beta(model, "default", best_beta, bayes_eps)
    print(f"[TFB fit] selected beta={best_beta:.6f}")
    return {
        "optimal_bayes_beta": float(best_beta),
        "reference_nll": float(ref_nll.item()),
        "threshold": float(threshold),
        "search_iters": int(search_iters),
        "anchor_n_samples": int(anchor_n_samples),
    }


def _get_source_split(task: str, split: str):
    train_ds, val_ds, test_ds = load_task_dataset(task)
    if split == "train":
        return train_ds
    if split in {"val", "validation"}:
        return val_ds
    if split == "test":
        return test_ds
    raise ValueError(f"Unsupported split: {split}")


def _subset_dataset(ds: Dataset, subset_size: int, seed: int) -> Dataset:
    if int(subset_size) <= 0 or int(subset_size) >= len(ds):
        return ds
    shuffled = ds.shuffle(seed=int(seed))
    return shuffled.select(range(min(int(subset_size), len(shuffled))))


def _resolve_source_anchor_and_eval(
    task: str,
    anchor_split: str,
    testing_set: str,
    anchor_size: int,
    seed: int,
) -> Tuple[Dataset, str, Dataset, str]:
    train_ds, val_ds, test_ds = load_task_dataset(task)
    testing_set = str(testing_set).strip().lower()
    if not testing_set:
        return _get_source_split(task, anchor_split), str(anchor_split).strip().lower(), test_ds, "test"

    if testing_set == "train_train_val":
        return _subset_dataset(train_ds, anchor_size, seed), "train", val_ds, "validation"
    if testing_set == "train_val_val":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", val_ds, "validation"
    if testing_set == "train_train_train":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", _subset_dataset(train_ds, anchor_size, seed), "train"
    if testing_set == "train_train_test":
        return _subset_dataset(train_ds, anchor_size, seed), "train", test_ds, "test"
    if testing_set == "train_val_test":
        return _subset_dataset(val_ds, anchor_size, seed), "validation", test_ds, "test"

    raise ValueError(
        "Unsupported testing_set={!r}. Expected one of: train_train_val, "
        "train_val_val, train_train_train, train_train_test, train_val_test".format(testing_set)
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate training-free Bayesianization (TFB) on ScienceQA MAP LoRA and OOD tasks.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
    )
    parser.add_argument("--map_adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--anchor_split", type=str, default="val")
    parser.add_argument("--testing-set", type=str, default="")
    parser.add_argument("--anchor-size", type=int, default=500)
    parser.add_argument("--anchor_bsz", type=int, default=ANCHOR_BSZ)
    parser.add_argument("--eval_bsz", type=int, default=EVAL_BSZ)
    parser.add_argument("--anchor_n_samples", type=int, default=TFB_ANCHOR_N_SAMPLES)
    parser.add_argument("--eval_n_samples", type=int, default=TFB_EVAL_N_SAMPLES)
    parser.add_argument("--bayes_beta_max", type=float, default=TFB_BETA_MAX)
    parser.add_argument("--threshold", type=float, default=TFB_THRESHOLD)
    parser.add_argument("--search_iters", type=int, default=TFB_BINARY_SEARCH_ITERS)
    parser.add_argument("--bayes_eps", type=float, default=TFB_BAYES_EPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with _StageTimer(f"LOAD-STAGE TFB on {args.task}"):
        tokenizer, model, num_classes = _load_base_and_adapter(
            task=args.task,
            map_adapter_dir=args.map_adapter_dir,
            amp_dtype=amp_dtype,
            device=device,
        )
        model.set_adapter("default")
        wrap_tfb_lora_layers(
            model=model,
            adapter_name="default",
            bayes_eps=float(args.bayes_eps),
            bayes_beta=float(args.bayes_beta_max),
        )

    anchor_raw, anchor_split_name, source_eval_raw, source_eval_split_name = _resolve_source_anchor_and_eval(
        task=args.task,
        anchor_split=args.anchor_split,
        testing_set=args.testing_set,
        anchor_size=int(args.anchor_size),
        seed=int(args.seed),
    )
    anchor_proc = _add_seq_len(
        preprocess_task(args.task, anchor_raw, tokenizer, args.max_seq_len, pad_to_max_length=False)
    )
    anchor_loader = _make_eval_loader(anchor_proc, tokenizer, device, bsz=args.anchor_bsz)

    with _StageTimer(f"FIT TFB on {args.task}({anchor_split_name})"):
        fit_info = fit_tfb_beta(
            model=model,
            anchor_loader=anchor_loader,
            device=device,
            amp_dtype=amp_dtype,
            num_classes=num_classes,
            bayes_beta_max=float(args.bayes_beta_max),
            threshold=float(args.threshold),
            search_iters=int(args.search_iters),
            anchor_n_samples=int(args.anchor_n_samples),
            bayes_eps=float(args.bayes_eps),
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tfb_fit.json").write_text(json.dumps(fit_info, indent=2) + "\n", encoding="utf-8")
        print(f"[Save] TFB fit info -> {out_dir / 'tfb_fit.json'}")

    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    print("\n========================")
    print("          TFB           ")
    print("========================")

    for eval_task in eval_tasks:
        eval_num_classes = get_task_num_classes(eval_task)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, but source task '{args.task}' has {num_classes} classes."
            )

        if eval_task == args.task:
            eval_raw = source_eval_raw
            split_name = source_eval_split_name
        else:
            eval_raw = load_eval_dataset(eval_task)
            split_name = "ood"

        eval_proc = _add_seq_len(preprocess_task(eval_task, eval_raw, tokenizer, args.max_seq_len, pad_to_max_length=False))
        loader = _make_eval_loader(eval_proc, tokenizer, device, bsz=args.eval_bsz)

        with _StageTimer(f"INFER TFB on {eval_task}({split_name})"):
            m = evaluate_tfb_official_style(
                model=model,
                loader=loader,
                device=device,
                amp_dtype=amp_dtype,
                num_classes=num_classes,
                n_samples=int(args.eval_n_samples),
                progress_desc=f"TFB {eval_task}",
            )

        print(f"\n[{eval_task}({split_name})][TFB]")
        print(
            f"  NLL={m['nll']:.4f}  ACC={m['acc']*100:.2f}%  "
            f"ECE={m['ece']*100:.2f}%  Brier={m['brier']:.4f}  "
            f"mc={int(args.eval_n_samples)}  beta={fit_info['optimal_bayes_beta']:.6f}  "
            f"std={m['std']:.6f}  flip_ratio={m['flip_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
