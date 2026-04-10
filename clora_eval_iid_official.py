from __future__ import annotations

import argparse
import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft.tuners.lora.bnb import Linear8bitLt
except Exception:
    Linear8bitLt = None

from common_eval_utils import (
    DynamicEvalCollator,
    SCIENCEQA_CURRIC_TASK_NAME,
    get_choice_token_ids,
    get_task_num_classes,
    get_transformer_and_lm_head,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    preprocess_task,
)


DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES_SPEC = "auto_qv_lmhead"
FULL_ATTENTION_TARGET_MODULES = ["q_proj", "v_proj"]
LM_HEAD_TARGET_MODULES = ["lm_head"]
TOKENIZER_PADDING_SIDE = "left"

CLORA_ADAPTER_NAME = "clora"
CLORA_EXTRA_FILENAME = "clora_extra.pt"

_LORA_ADAPTER_PLACEHOLDER = "__adapter__"
_LORA_ADAPTER_RE = re.compile(r"(\.lora_(?:A|B)\.)([^.]+)(\.)")


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(value: int) -> float:
    return float(value) / (1024 ** 3)


def _reset_cuda_peak() -> None:
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


class _Timer:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0: Optional[float] = None

    def __enter__(self):
        _reset_cuda_peak()
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _cuda_sync()
        assert self.t0 is not None
        dt = time.perf_counter() - self.t0
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(f"[PEAK] {self.tag}: alloc={_peak_alloc_gb():.2f} GB  reserved={_peak_reserved_gb():.2f} GB")


def _add_seq_len(ds: Dataset) -> Dataset:
    if "seq_len" in ds.column_names:
        return ds
    return ds.add_column("seq_len", [len(x) for x in ds["input_ids"]])


def _order_scienceqa_train_by_grade(train_raw: Dataset, seed: int) -> Dataset:
    if "grade_num" not in train_raw.column_names:
        return train_raw

    grade_values = sorted({int(grade) for grade in train_raw["grade_num"]})
    parts: List[Dataset] = []
    for grade_num in grade_values:
        idxs = [i for i, value in enumerate(train_raw["grade_num"]) if int(value) == grade_num]
        if not idxs:
            continue
        ds_grade = train_raw.select(idxs).shuffle(seed=seed + grade_num)
        parts.append(ds_grade)
    if not parts:
        raise RuntimeError("No ScienceQA training examples left after grade ordering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def resolve_all_layer_target_modules(model: nn.Module) -> List[str]:
    wanted_attention = set(FULL_ATTENTION_TARGET_MODULES)
    wanted_lm_head = set(LM_HEAD_TARGET_MODULES)
    resolved = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if (
            ".layers." in name
            and ".self_attn." in name
            and suffix in wanted_attention
        ):
            resolved.append(name)
            continue
        if name in wanted_lm_head or suffix in wanted_lm_head:
            resolved.append(name)
    if not resolved:
        raise RuntimeError("Could not resolve any q/v attention or lm_head LoRA target modules")
    return sorted(set(resolved))


def _multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def _mask_invalid_choices(logits: torch.Tensor, num_choices: Optional[Sequence[int]]) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(value) for value in num_choices], device=logits.device, dtype=torch.long)
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


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


def _normalize_lora_key(key: str) -> str:
    return _LORA_ADAPTER_RE.sub(rf"\1{_LORA_ADAPTER_PLACEHOLDER}\3", key)


def _denormalize_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(f".{_LORA_ADAPTER_PLACEHOLDER}.", f".{adapter_name}.")


def load_normalized_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor], adapter_name: str) -> None:
    mapped = {_denormalize_lora_key(key, adapter_name): value for key, value in lora_state.items()}
    model.load_state_dict(mapped, strict=False)


def force_lora_fp32(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()


def _iter_lora_linear_modules(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def _iter_active_adapters(layer: nn.Module) -> List[str]:
    active = getattr(layer, "active_adapters", None)
    if active is None:
        active = getattr(layer, "_active_adapter", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return list(active)


class ContextualE(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, device=None, dtype=None):
        super().__init__()
        self.e1 = nn.Linear(in_feat, 64, device=device, dtype=dtype)
        self.e2 = nn.Linear(64, out_feat, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.e2(self.relu(self.e1(x)))


def _sigma_from_context(rho_like: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log1p(torch.exp(rho_like)) if float(eps) < 0 else rho_like.square()


def _kl_div_stable(mu_q: torch.Tensor, sigma_q: torch.Tensor, mu_p: float, sigma_p: float) -> torch.Tensor:
    eps = 1e-6
    kl = (
        math.log(float(sigma_p) + eps)
        - torch.log(sigma_q.to(torch.float64) + eps)
        + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - float(mu_p)) ** 2)
        / (2 * (float(sigma_p) ** 2) + eps)
        - 0.5
    )
    return kl.sum()


def _postprocess_context_scale(raw_scale: torch.Tensor, eps: float) -> torch.Tensor:
    if 0.0 < float(eps) < 1.0:
        return torch.sigmoid(raw_scale)
    if 1.0 < float(eps) < 2.0:
        return torch.tanh(raw_scale)
    if float(eps) > 2.0:
        return torch.clamp(raw_scale, min=-1.0, max=1.0)
    return raw_scale


def _compute_contextual_mean_terms(layer: nn.Module, x: torch.Tensor, active_adapter: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    lora_A = layer.lora_A[active_adapter]
    lora_E = layer.lora_E[active_adapter]
    dropout = layer.lora_dropout[active_adapter]
    rank = int(layer.r[active_adapter])

    x_det = x.to(lora_A.weight.dtype)
    oA = lora_A(dropout(x_det))
    E = lora_E(oA)
    Em, Eg = E.split(rank * rank, dim=-1)
    Eg = _postprocess_context_scale(Eg, layer.clora_eps)
    layer.E_m[active_adapter] = Em
    layer.E_g[active_adapter] = Eg
    return x_det, oA, Em, rank


def _clora_linear_forward(self, x: torch.Tensor, *args, **kwargs):
    previous_dtype = x.dtype
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A or active_adapter not in self.lora_E:
            continue
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]
        _, oA, Em, rank = _compute_contextual_mean_terms(self, x, active_adapter)
        Emm = Em.contiguous().view(*oA.shape[:-1], rank, rank)
        mean_hidden = torch.matmul(Emm, oA.unsqueeze(-1)).squeeze(-1)
        result = result + lora_B(mean_hidden) * scaling

    if getattr(self, "clorasample", True):
        for active_adapter in _iter_active_adapters(self):
            if active_adapter not in self.lora_A or active_adapter not in self.lora_E:
                continue
            if active_adapter not in self.E_g:
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            rank = int(self.r[active_adapter])
            x_noise = x.to(lora_A.weight.dtype)

            if x_noise.dim() == 2:
                r_e = torch.empty((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_e = torch.empty((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            elif x_noise.dim() == 3:
                r_e = torch.empty((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_e = torch.empty((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            else:
                raise ValueError(f"Unsupported C-LoRA input rank {x_noise.dim()}, expected 2 or 3.")

            x_noise = dropout(x_noise)
            sigma_e = _sigma_from_context(self.E_g[active_adapter], self.clora_eps).to(dtype=lora_A.weight.dtype)
            noise_e = sigma_e * torch.randn_like(sigma_e)
            noise_e = noise_e.contiguous().view(*noise_e.shape[:-1], rank, rank)
            oA_noise = (x_noise @ lora_A.weight.transpose(0, 1)) * r_e
            contextual_noise = torch.matmul(noise_e, oA_noise.unsqueeze(-1)).squeeze(-1)
            noise = (contextual_noise * s_e) @ lora_B.weight.transpose(0, 1)
            result = result + noise * scaling

    return result.to(previous_dtype)


def _clora_8bitlinear_forward(self, x: torch.Tensor, *args, **kwargs):
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A or active_adapter not in self.lora_E:
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        scaling = self.scaling[active_adapter]
        requires_conversion = not torch.is_autocast_enabled()
        x_det = x
        if requires_conversion:
            expected_dtype = result.dtype
            compute_dtype = lora_A.weight.dtype
            if x_det.dtype != compute_dtype:
                x_det = x_det.to(compute_dtype)
        _, oA, Em, rank = _compute_contextual_mean_terms(self, x_det, active_adapter)
        Emm = Em.contiguous().view(*oA.shape[:-1], rank, rank)
        mean_hidden = torch.matmul(Emm, oA.unsqueeze(-1)).squeeze(-1)
        output = lora_B(mean_hidden)
        if requires_conversion:
            output = output.to(expected_dtype)
        result = result + output * scaling

    if getattr(self, "clorasample", True):
        for active_adapter in _iter_active_adapters(self):
            if active_adapter not in self.lora_A or active_adapter not in self.lora_E:
                continue
            if active_adapter not in self.E_g:
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            rank = int(self.r[active_adapter])
            requires_conversion = not torch.is_autocast_enabled()
            x_noise = x
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x_noise.dtype != compute_dtype:
                    x_noise = x_noise.to(compute_dtype)

            if x_noise.dim() == 2:
                r_e = torch.empty((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_e = torch.empty((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            elif x_noise.dim() == 3:
                r_e = torch.empty((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_e = torch.empty((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            else:
                raise ValueError(f"Unsupported C-LoRA input rank {x_noise.dim()}, expected 2 or 3.")

            x_noise = dropout(x_noise)
            sigma_e = _sigma_from_context(self.E_g[active_adapter], self.clora_eps).to(dtype=lora_A.weight.dtype)
            noise_e = sigma_e * torch.randn_like(sigma_e)
            noise_e = noise_e.contiguous().view(*noise_e.shape[:-1], rank, rank)
            oA_noise = (x_noise @ lora_A.weight.transpose(0, 1)) * r_e
            contextual_noise = torch.matmul(noise_e, oA_noise.unsqueeze(-1)).squeeze(-1)
            noise = (contextual_noise * s_e) @ lora_B.weight.transpose(0, 1)
            if requires_conversion:
                noise = noise.to(expected_dtype)
            result = result + noise * scaling

    return result


def _clora_div_posterior_prior(self) -> torch.Tensor:
    kl = 0.0
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A or active_adapter not in self.lora_E:
            continue
        if active_adapter not in self.E_m or active_adapter not in self.E_g:
            continue
        sigma_weight = _sigma_from_context(self.E_g[active_adapter], self.clora_eps)
        kl = kl + _kl_div_stable(self.E_m[active_adapter], sigma_weight, 0.0, self.clora_beta)
    return kl


def _clora_sample(self, status: bool = True):
    if self.training and not bool(status):
        raise ValueError("clorasample should be set to True only during training.")
    self.clorasample = bool(status)


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


def wrap_clora_lora_layers(model: PeftModel, adapter_name: str, eps: float, beta: float) -> None:
    wrapped = 0
    for _, module in _iter_lora_linear_modules(model):
        if not isinstance(module, LoraLayer):
            continue
        if not isinstance(module, LoraLinear) and not (Linear8bitLt is not None and isinstance(module, Linear8bitLt)):
            continue
        A_dict = getattr(module, "lora_A", None)
        if not isinstance(A_dict, nn.ModuleDict) or adapter_name not in A_dict:
            continue

        if not hasattr(module, "lora_E") or not isinstance(module.lora_E, nn.ModuleDict):
            module.lora_E = nn.ModuleDict({})
        if not hasattr(module, "E_m") or not isinstance(module.E_m, dict):
            module.E_m = {}
        if not hasattr(module, "E_g") or not isinstance(module.E_g, dict):
            module.E_g = {}

        module.clora_eps = float(eps)
        module.clora_beta = float(beta)
        module.clorasample = True

        if adapter_name not in module.lora_E:
            rank = int(module.r[adapter_name])
            device = module.lora_A[adapter_name].weight.device
            dtype = module.lora_A[adapter_name].weight.dtype
            module.lora_E[adapter_name] = ContextualE(
                in_feat=rank,
                out_feat=rank * rank * 2,
                device=device,
                dtype=dtype,
            )

        module.div_posterior_prior = _clora_div_posterior_prior.__get__(module, module.__class__)
        module.sample = _clora_sample.__get__(module, module.__class__)
        if Linear8bitLt is not None and isinstance(module, Linear8bitLt):
            module.forward = _clora_8bitlinear_forward.__get__(module, module.__class__)
        else:
            module.forward = _clora_linear_forward.__get__(module, module.__class__)
        wrapped += 1

    print(f"[C-LoRA official] patched {wrapped} LoRA layers (adapter={adapter_name}).")


def set_clora_sampling(model: nn.Module, status: bool) -> None:
    if model.training and not bool(status):
        raise ValueError("clorasample should be set to True only during training.")
    for _, module in _iter_lora_linear_modules(model):
        if isinstance(module, LoraLayer) and hasattr(module, "sample"):
            module.sample(status)


def collect_clora_kl(model: nn.Module, adapter_name: str) -> torch.Tensor:
    kl_terms = []
    for _, module in _iter_lora_linear_modules(model):
        if not isinstance(module, LoraLayer):
            continue
        if not hasattr(module, "div_posterior_prior"):
            continue
        if not hasattr(module, "lora_E") or adapter_name not in module.lora_E:
            continue
        if not hasattr(module, "E_m") or adapter_name not in module.E_m:
            continue
        if not hasattr(module, "E_g") or adapter_name not in module.E_g:
            continue
        kl_terms.append(module.div_posterior_prior())
    if not kl_terms:
        raise RuntimeError("No C-LoRA layers found while collecting KL.")
    return torch.stack(kl_terms).sum()


def save_clora_extra_state(model: nn.Module, save_dir: str) -> str:
    extra_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_E." in key
    }
    if not extra_state:
        raise RuntimeError("No lora_E weights found while saving C-LoRA extra state.")
    path = os.path.join(save_dir, CLORA_EXTRA_FILENAME)
    torch.save(extra_state, path)
    return path


def resolve_clora_paths(clora_dir: str) -> Tuple[str, str]:
    adapter_dir = clora_dir
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        subdir = os.path.join(clora_dir, CLORA_ADAPTER_NAME)
        if os.path.exists(os.path.join(subdir, "adapter_config.json")):
            adapter_dir = subdir
        else:
            raise FileNotFoundError(
                f"Could not find adapter_config.json in '{clora_dir}' or '{subdir}'"
            )

    extra_candidates = [
        os.path.join(clora_dir, CLORA_EXTRA_FILENAME),
        os.path.join(adapter_dir, CLORA_EXTRA_FILENAME),
    ]
    extra_path = next((path for path in extra_candidates if os.path.exists(path)), None)
    if extra_path is None:
        raise FileNotFoundError(f"Missing C-LoRA extra file. Tried: {extra_candidates}")
    return adapter_dir, extra_path


def load_clora_extra_state(model: nn.Module, extra_path: str) -> None:
    saved = torch.load(extra_path, map_location="cpu")
    if not isinstance(saved, dict):
        raise RuntimeError(f"Malformed C-LoRA extra state: {extra_path}")
    model.load_state_dict(saved, strict=False)
    print(f"[C-LoRA official] loaded contextual state from: {extra_path}")


def compute_class_logits(
    model: PeftModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_ids.device
    transformer, lm_head = get_transformer_and_lm_head(model)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        feats = out.last_hidden_state[:, -1, :]
        logits = lm_head(feats)
    return logits.to(torch.float32)


@torch.inference_mode()
def eval_adapter(
    model: PeftModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    num_classes: int,
    adapter_name: str,
    mode: str,
    n_samples: int,
    clora_adapter_name: str,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    model.set_adapter(adapter_name)
    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 15)
    acc_m.reset()
    ece_m.reset()

    total = 0
    nll_sum = 0.0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress_total = total_samples if total_samples is not None else len(loader)
    progress_unit = "sample" if total_samples is not None else "batch"
    progress_start = time.perf_counter()
    batch_iter = tqdm(
        total=progress_total,
        desc=(progress_desc or f"{adapter_name}:{mode}"),
        unit=progress_unit,
        leave=False,
    )

    if mode in {"det", "clora_mean"} and adapter_name == clora_adapter_name:
        set_clora_sampling(model, False)
    if mode == "clora_sample":
        set_clora_sampling(model, True)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_choices = batch.get("num_choices")
        batch_size = labels.size(0)
        total += batch_size

        if mode in {"det", "clora_mean"}:
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            logits = _mask_invalid_choices(logits, num_choices)
            probs = torch.softmax(logits, dim=-1)
            nll_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
        else:
            probs_acc = torch.zeros((batch_size, num_classes), device=device, dtype=torch.float32)
            for _ in range(int(n_samples)):
                logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
                logits = _mask_invalid_choices(logits, num_choices)
                probs_acc.add_(torch.softmax(logits, dim=-1))
            probs = probs_acc / float(n_samples)
            nll_sum += float((-torch.log(probs[torch.arange(batch_size, device=device), labels].clamp_min(1e-12))).sum().item())

        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        elapsed = time.perf_counter() - progress_start
        avg_sec_per_sample = elapsed / max(total, 1)
        if total_samples is not None:
            remaining_samples = max(int(total_samples) - total, 0)
            eta = _format_eta(avg_sec_per_sample * remaining_samples)
            batch_iter.update(batch_size)
        else:
            remaining_batches = max(len(loader) - batch_iter.n - 1, 0)
            eta = _format_eta((elapsed / max(batch_iter.n + 1, 1)) * remaining_batches)
            batch_iter.update(1)
        batch_iter.set_postfix(avg_s_per_sample=f"{avg_sec_per_sample:.3f}", eta=eta, refresh=False)

    batch_iter.close()
    if adapter_name == clora_adapter_name:
        set_clora_sampling(model, True)

    probs_all = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, num_classes), dtype=torch.float32)
    labels_all = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": _multiclass_brier_score(probs_all, labels_all) if total > 0 else float("nan"),
    }


def train_clora(
    model: PeftModel,
    train_loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float,
    warmup_ratio: float,
    batch_size: int,
    grad_accum: int,
    clora_adapter: str,
    bayes_beta: float,
    bayes_kllr: float,
    bayes_gamma: float,
    bayes_klreweighting: bool,
    bayes_datasetrescaling: bool,
    bayes_train_n_samples: int,
    clora_kl_scale: float,
    clora_kl_divisor: float,
    amp_dtype: torch.dtype,
) -> None:
    del bayes_beta
    model.train()
    model.set_adapter(clora_adapter)
    set_clora_sampling(model, True)

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    context_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and ".lora_E." in name
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for C-LoRA training.")
    if not context_params:
        raise RuntimeError("No lora_E parameters found for C-LoRA training.")

    opt = torch.optim.AdamW(trainable_params, lr=lr)
    opt2 = torch.optim.SGD(context_params, lr=bayes_kllr)
    sched = get_linear_schedule_with_warmup(opt, int(steps * warmup_ratio), steps)
    sched2 = get_linear_schedule_with_warmup(opt2, int(steps * warmup_ratio), steps)

    grad_accum = max(int(grad_accum), 1)
    if grad_accum != 1:
        print(f"[Warn] grad_accum={grad_accum} uses a combined-loss approximation in this first C-LoRA port.")

    M = max(
        int(100 * (len(train_loader.dataset) ** (math.pi / float(bayes_gamma))) / max(int(batch_size), 1))
        if bayes_datasetrescaling
        else len(train_loader),
        1,
    )
    print(
        f"[KL schedule] datasetrescaling={bool(bayes_datasetrescaling)} "
        f"M={M} warmup_steps={int(steps * warmup_ratio)}"
    )

    opt.zero_grad(set_to_none=True)
    opt2.zero_grad(set_to_none=True)
    global_step = 0
    step_i = 1
    loader_iter = iter(train_loader)
    last_pi = 0.0
    last_kl_value = 0.0

    while global_step < steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_choices = batch.get("num_choices")

        logits_samples = []
        for _ in range(max(int(bayes_train_n_samples), 1)):
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            logits = _mask_invalid_choices(logits, num_choices)
            logits_samples.append(logits)

        mean_logits = torch.stack(logits_samples, dim=1).mean(dim=1)
        log_probs = torch.log_softmax(mean_logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction="mean")

        kl_loss = collect_clora_kl(model, clora_adapter)
        if bayes_klreweighting:
            pi = (2 ** ((step_i % M) or M)) / (2 ** (M + 1) - 1)
        else:
            pi = 1.0 / float(M)
        step_i += 1
        last_pi = float(pi)
        kl_term = (kl_loss / float(clora_kl_divisor)) * float(pi) * float(clora_kl_scale)
        last_kl_value = float(kl_term.detach().item())

        total_loss = (nll_loss + kl_term) / float(grad_accum)
        total_loss.backward()

        if (global_step + 1) % grad_accum == 0:
            opt.step()
            opt2.step()
            opt.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            sched.step()
            sched2.step()

        if global_step % 50 == 0:
            print(
                f"[train] step={global_step:5d}/{steps} "
                f"nll={float(nll_loss.item()):.4f} "
                f"kl={last_kl_value:.6g} "
                f"kl_lr={float(opt2.param_groups[0]['lr']):.6g} "
                f"pi={last_pi:.6g}"
            )
        global_step += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
        help="Unified tasks",
    )
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--map_adapter_dir", type=str, default="")
    ap.add_argument("--max_seq_len", type=int, default=300)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--train_bsz", type=int, default=32)
    ap.add_argument("--eval_bsz", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument(
        "--target_modules",
        type=str,
        default=DEFAULT_TARGET_MODULES_SPEC,
        help="Comma-separated target module names/suffixes, or 'auto_qv_lmhead'.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bayes_eps", type=float, default=0.05)
    ap.add_argument("--bayes_beta", type=float, default=0.2)
    ap.add_argument("--bayes_kllr", type=float, default=0.01)
    ap.add_argument("--bayes_gamma", type=float, default=8.0)
    ap.add_argument("--bayes_klreweighting", action="store_true")
    ap.add_argument("--bayes_datasetrescaling", action="store_true")
    ap.add_argument("--bayes_train_n_samples", type=int, default=1)
    ap.add_argument("--clora_eval_n", type=int, default=10)
    ap.add_argument("--clora_kl_scale", type=float, default=1e-6)
    ap.add_argument("--clora_kl_divisor", type=float, default=65.0)
    ap.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="Comma-separated eval tasks. Supports iid, arc, arc-c, arc-e, gpqa_main, mmlu, scienceqa_closedchoice_grade12, etc.",
    )
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_eval", action="store_true")
    ap.add_argument("--save_clora_dir", type=str, default="")
    ap.add_argument("--load_clora_dir", type=str, default="")
    ap.add_argument("--shared_init_lora_path", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n[C-LoRA official-style] [Task] {args.task} | [Device] {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.bos_token if tok.bos_token is not None else tok.eos_token
    tok.padding_side = TOKENIZER_PADDING_SIDE

    num_classes = get_task_num_classes(args.task)
    choice_token_ids = get_choice_token_ids(tok, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    if str(args.target_modules).strip().lower() == DEFAULT_TARGET_MODULES_SPEC:
        target_modules = resolve_all_layer_target_modules(base_model)
    else:
        target_modules = [module.strip() for module in str(args.target_modules).split(",") if module.strip()]
        if not target_modules:
            raise ValueError("target_modules must contain at least one module name")
    print(
        f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} target_modules={target_modules}"
    )

    if args.load_clora_dir and os.path.isdir(args.load_clora_dir):
        clora_adapter_dir, clora_extra_path = resolve_clora_paths(args.load_clora_dir)
        print(f"[Load C-LoRA adapter] -> name={CLORA_ADAPTER_NAME} from {clora_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            clora_adapter_dir,
            adapter_name=CLORA_ADAPTER_NAME,
            is_trainable=bool(args.do_train),
        ).to(device)
        wrap_clora_lora_layers(
            model,
            adapter_name=CLORA_ADAPTER_NAME,
            eps=args.bayes_eps,
            beta=args.bayes_beta,
        )
        load_clora_extra_state(model, clora_extra_path)
    elif args.do_train:
        if args.map_adapter_dir and os.path.isdir(args.map_adapter_dir):
            print(f"[Init from MAP adapter] -> name={CLORA_ADAPTER_NAME} from {args.map_adapter_dir}")
            model = PeftModel.from_pretrained(
                base_model,
                args.map_adapter_dir,
                adapter_name=CLORA_ADAPTER_NAME,
                is_trainable=True,
            ).to(device)
        else:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            model = get_peft_model(base_model, lora_cfg, adapter_name=CLORA_ADAPTER_NAME).to(device)
            if args.shared_init_lora_path:
                if not os.path.exists(args.shared_init_lora_path):
                    raise FileNotFoundError(f"Missing shared init LoRA file: {args.shared_init_lora_path}")
                saved_init = torch.load(args.shared_init_lora_path, map_location="cpu")
                load_normalized_lora_state_dict(model, saved_init, adapter_name=CLORA_ADAPTER_NAME)
                print(f"[Init LoRA] loaded shared init from {args.shared_init_lora_path}")

        wrap_clora_lora_layers(
            model,
            adapter_name=CLORA_ADAPTER_NAME,
            eps=args.bayes_eps,
            beta=args.bayes_beta,
        )
    else:
        raise ValueError("Eval-only mode needs --load_clora_dir. Otherwise pass --do_train to fit a new C-LoRA adapter.")

    force_lora_fp32(model)
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name)

    print(f"\n=== Loading {args.task} ===")
    train_raw, _, test_raw = load_task_dataset(args.task)
    if args.task == SCIENCEQA_CURRIC_TASK_NAME:
        train_raw = _order_scienceqa_train_by_grade(train_raw, args.seed)
    task_train = _add_seq_len(preprocess_task(args.task, train_raw, tok, args.max_seq_len, pad_to_max_length=False))
    task_test = _add_seq_len(preprocess_task(args.task, test_raw, tok, args.max_seq_len, pad_to_max_length=False))
    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)

    dyn_collator = DynamicEvalCollator(
        tokenizer=tok,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    def _make_loader(proc: Dataset, batch_size: int, shuffle: bool, drop_last: bool, sort_by_len: bool = True) -> DataLoader:
        proc_loader = proc
        if sort_by_len and not shuffle and "seq_len" in proc_loader.column_names:
            proc_loader = proc_loader.sort("seq_len")
        if "seq_len" in proc_loader.column_names:
            proc_loader = proc_loader.remove_columns(["seq_len"])
        return DataLoader(
            proc_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=dyn_collator,
            pin_memory=(device.type == "cuda"),
        )

    train_shuffle = args.task != SCIENCEQA_CURRIC_TASK_NAME
    train_sort_by_len = False if args.task == SCIENCEQA_CURRIC_TASK_NAME else True
    train_loader = _make_loader(
        task_train,
        args.train_bsz,
        shuffle=train_shuffle,
        drop_last=True,
        sort_by_len=train_sort_by_len,
    )
    eval_loaders: Dict[str, DataLoader] = {
        args.task: _make_loader(task_test, args.eval_bsz, shuffle=False, drop_last=False)
    }
    for eval_task in eval_tasks:
        if eval_task == args.task:
            continue
        eval_num_classes = get_task_num_classes(eval_task)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, "
                f"but source task '{args.task}' has {num_classes} classes."
            )
        eval_raw = load_eval_dataset(eval_task)
        eval_proc = _add_seq_len(preprocess_task(eval_task, eval_raw, tok, args.max_seq_len, pad_to_max_length=False))
        eval_loaders[eval_task] = _make_loader(eval_proc, args.eval_bsz, shuffle=False, drop_last=False)

    if args.do_train:
        print(f"\n=== TRAIN: C-LoRA on {args.task}(train) ===")
        with _Timer(f"C-LoRA TRAIN ({args.task})"):
            train_clora(
                model=model,
                train_loader=train_loader,
                device=device,
                steps=args.train_steps,
                lr=args.lr,
                warmup_ratio=args.warmup_ratio,
                batch_size=args.train_bsz,
                grad_accum=args.grad_accum,
                clora_adapter=CLORA_ADAPTER_NAME,
                bayes_beta=args.bayes_beta,
                bayes_kllr=args.bayes_kllr,
                bayes_gamma=args.bayes_gamma,
                bayes_klreweighting=bool(args.bayes_klreweighting),
                bayes_datasetrescaling=bool(args.bayes_datasetrescaling),
                bayes_train_n_samples=args.bayes_train_n_samples,
                clora_kl_scale=args.clora_kl_scale,
                clora_kl_divisor=args.clora_kl_divisor,
                amp_dtype=amp_dtype,
            )
        if args.save_clora_dir:
            os.makedirs(args.save_clora_dir, exist_ok=True)
            model.set_adapter(CLORA_ADAPTER_NAME)
            model.save_pretrained(args.save_clora_dir)
            extra_path = save_clora_extra_state(model, args.save_clora_dir)
            print(f"[Save] saved C-LoRA adapter to: {args.save_clora_dir}")
            print(f"[Save] saved C-LoRA contextual state to: {extra_path}")

    if args.do_eval:
        for eval_task in eval_tasks:
            eval_loader = eval_loaders[eval_task]
            split_name = "test" if eval_task == args.task else "ood"
            print(f"\n=== EVAL: source={args.task} -> target={eval_task} ({split_name}) ===")
            with _Timer(f"EVAL clora_mean on {eval_task}"):
                m_mean = eval_adapter(
                    model,
                    eval_loader,
                    device,
                    amp_dtype,
                    num_classes,
                    CLORA_ADAPTER_NAME,
                    "clora_mean",
                    1,
                    CLORA_ADAPTER_NAME,
                    progress_desc=f"C-LoRA mean {eval_task}",
                )
            with _Timer(f"EVAL clora_sample(N={args.clora_eval_n}) on {eval_task}"):
                m_samp = eval_adapter(
                    model,
                    eval_loader,
                    device,
                    amp_dtype,
                    num_classes,
                    CLORA_ADAPTER_NAME,
                    "clora_sample",
                    args.clora_eval_n,
                    CLORA_ADAPTER_NAME,
                    progress_desc=f"C-LoRA samp {eval_task}",
                )

            print(f"\n[{eval_task}({split_name}) Results]")
            print(
                f"  C-LoRA mean : NLL={m_mean['nll']:.4f}  ACC={m_mean['acc']*100:.2f}%  "
                f"ECE={m_mean['ece']*100:.2f}%  Brier={m_mean['brier']:.4f} (N=0)"
            )
            print(
                f"  C-LoRA samp : NLL={m_samp['nll']:.4f}  ACC={m_samp['acc']*100:.2f}%  "
                f"ECE={m_samp['ece']*100:.2f}%  Brier={m_samp['brier']:.4f} (N={args.clora_eval_n})"
            )


if __name__ == "__main__":
    main()
