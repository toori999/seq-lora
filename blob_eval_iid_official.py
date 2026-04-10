from __future__ import annotations
import os, math, time, random, argparse
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer, Linear as LoraLinear

try:
    from peft.tuners.lora.bnb import Linear8bitLt
except Exception:
    Linear8bitLt = None

# torchmetrics
try:
    from torchmetrics import Accuracy, CalibrationError
except Exception:
    Accuracy = None
    CalibrationError = None
    from torchmetrics.classification import MulticlassAccuracy as _MulticlassAccuracy
    from torchmetrics.classification import MulticlassCalibrationError as _MulticlassCalibrationError

# =========================
# Timing + Peak GPU memory helpers
# =========================
def _cuda_sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)

def _reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def _peak_alloc_gb() -> float:
    if not torch.cuda.is_available(): return 0.0
    return _mem_gb(torch.cuda.max_memory_allocated())

def _peak_reserved_gb() -> float:
    if not torch.cuda.is_available(): return 0.0
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


from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    blob_kl_div_stable,
    DynamicEvalCollator,
    get_active_adapter_name as _get_active_adapter_name,
    get_choice_token_ids,
    get_task_num_classes,
    get_transformer_and_lm_head,
    init_blob_rho_,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    pick_adapter_module as _pick_adapter_module,
    pick_scaling as _pick_scaling,
    preprocess_task,
)

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES_SPEC = "auto_qv_lmhead"
FULL_ATTENTION_TARGET_MODULES = ["q_proj", "v_proj"]
LM_HEAD_TARGET_MODULES = ["lm_head"]

def _add_seq_len(ds: Dataset) -> Dataset:
    if "seq_len" in ds.column_names:
        return ds
    return ds.add_column("seq_len", [len(x) for x in ds["input_ids"]])


def _order_scienceqa_train_by_grade(train_raw: Dataset, seed: int) -> Dataset:
    if "grade_num" not in train_raw.column_names:
        return train_raw

    grade_values = sorted({int(g) for g in train_raw["grade_num"]})
    parts: List[Dataset] = []
    for grade_num in grade_values:
        idxs = [i for i, g in enumerate(train_raw["grade_num"]) if int(g) == grade_num]
        if not idxs:
            continue
        ds_g = train_raw.select(idxs).shuffle(seed=seed + grade_num)
        parts.append(ds_g)
    if not parts:
        raise RuntimeError("No ScienceQA training examples left after grade ordering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def resolve_all_layer_target_modules(model: nn.Module) -> List[str]:
    wanted_full_attention = set(FULL_ATTENTION_TARGET_MODULES)
    wanted_lm_head = set(LM_HEAD_TARGET_MODULES)
    resolved = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if (
            ".layers." in name
            and ".self_attn." in name
            and suffix in wanted_full_attention
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
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
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

# =========================
# Official BLoB components adapted to the local train/eval pipeline
# =========================

TOKENIZER_PADDING_SIDE = "left"


class BLoBNLLScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_ratio: float,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            factor = max(
                0.0,
                (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps),
            )
        return [base_lr * factor for base_lr in self.base_lrs]


class BLoBKLScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_ratio: float,
        total_steps: int,
        num_samples: int,
        num_batches: int,
        batch_size: int,
        gamma: float,
        dataset_rescaling: bool = False,
        use_exponential: bool = True,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = int(total_steps)
        if bool(dataset_rescaling):
            self.M = max(
                int(100 * (num_samples ** (math.pi / float(gamma))) / max(int(batch_size), 1)),
                1,
            )
        else:
            self.M = max(int(num_batches), 1)
        self.use_exponential = bool(use_exponential)
        self._denom = 2 ** (self.M + 1) - 1
        self.last_pi = 0.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lin = step / max(1, self.warmup_steps)
        else:
            lin = max(
                0.0,
                (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps),
            )
        if self.use_exponential:
            i = (step % self.M) + 1
            pi = 2 ** i / self._denom
        else:
            pi = 1.0 / self.M
        self.last_pi = pi
        scale = lin * pi
        return [base_lr * scale for base_lr in self.base_lrs]

def _iter_lora_linear_modules(model: nn.Module):
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            yield name, mod


_LORA_ADAPTER_PLACEHOLDER = "__adapter__"
_LORA_ADAPTER_RE = re.compile(r"(\.lora_(?:A|B)\.)([^.]+)(\.)")


def _normalize_lora_key(key: str) -> str:
    return _LORA_ADAPTER_RE.sub(rf"\1{_LORA_ADAPTER_PLACEHOLDER}\3", key)


def _denormalize_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(f".{_LORA_ADAPTER_PLACEHOLDER}.", f".{adapter_name}.")


def load_normalized_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor], adapter_name: str) -> None:
    mapped = {_denormalize_lora_key(k, adapter_name): v for k, v in lora_state.items()}
    model.load_state_dict(mapped, strict=False)


def _iter_active_adapters(layer: nn.Module) -> List[str]:
    active = getattr(layer, "active_adapters", None)
    if active is None:
        active = getattr(layer, "_active_adapter", None)
    if active is None:
        return []
    if isinstance(active, str):
        return [active]
    return list(active)


def _official_sigma_from_rho(rho: torch.Tensor, bayes_eps: float) -> torch.Tensor:
    return torch.log1p(torch.exp(rho)) if float(bayes_eps) < 0 else rho.square()


def _official_kl_div_stable(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_p: float,
    sigma_p: float,
) -> torch.Tensor:
    eps = 1e-6
    kl = (
        math.log(float(sigma_p) + eps)
        - torch.log(sigma_q.to(torch.float64) + eps)
        + (sigma_q.to(torch.float64) ** 2 + (mu_q.to(torch.float64) - float(mu_p)) ** 2)
        / (2 * (float(sigma_p) ** 2) + eps)
        - 0.5
    )
    return kl.sum()


def _official_blob_linear_forward(self, x: torch.Tensor, *args, **kwargs):
    previous_dtype = x.dtype
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x_det = x.to(lora_A.weight.dtype)
        result += lora_B(lora_A(dropout(x_det))) * scaling

    if getattr(self, "blobsample", True):
        for active_adapter in _iter_active_adapters(self):
            if active_adapter not in self.lora_A.keys():
                continue
            if not hasattr(self, "lora_A_rho") or active_adapter not in self.lora_A_rho:
                continue

            lora_A = self.lora_A[active_adapter]
            sigma_a = _official_sigma_from_rho(self.lora_A_rho[active_adapter], self.bayes_eps)
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            rank = int(lora_A.weight.shape[0])
            x_noise = x.to(lora_A.weight.dtype)

            if x_noise.dim() == 2:
                r_a = torch.ones((x_noise.size(0), self.in_features), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_a = torch.ones((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            elif x_noise.dim() == 3:
                r_a = torch.ones((x_noise.size(0), x_noise.size(1), self.in_features), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_a = torch.ones((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            else:
                raise ValueError(f"Unsupported BLoB input rank {x_noise.dim()}, expected 2 or 3.")

            x_noise = dropout(x_noise)
            lora_noise_a = sigma_a.to(dtype=lora_A.weight.dtype) * torch.randn_like(lora_A.weight)
            noise = (((x_noise * r_a) @ lora_noise_a.transpose(0, 1)) * s_a) @ self.lora_B[active_adapter].weight.transpose(0, 1)
            result += noise * scaling

    return result.to(previous_dtype)


def _official_blob_8bitlinear_forward(self, x: torch.Tensor, *args, **kwargs):
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        return self.base_layer(x, *args, **kwargs)
    if self.merged:
        return self.base_layer(x, *args, **kwargs)

    result = self.base_layer(x, *args, **kwargs)
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        requires_conversion = not torch.is_autocast_enabled()
        x_det = x
        if requires_conversion:
            expected_dtype = result.dtype
            compute_dtype = lora_A.weight.dtype
            if x_det.dtype != compute_dtype:
                x_det = x_det.to(compute_dtype)
        output = lora_B(lora_A(dropout(x_det)))
        if requires_conversion:
            output = output.to(expected_dtype)
        result += output * scaling

    if getattr(self, "blobsample", True):
        for active_adapter in _iter_active_adapters(self):
            if active_adapter not in self.lora_A.keys():
                continue
            if not hasattr(self, "lora_A_rho") or active_adapter not in self.lora_A_rho:
                continue

            lora_A = self.lora_A[active_adapter]
            sigma_a = _official_sigma_from_rho(self.lora_A_rho[active_adapter], self.bayes_eps)
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            rank = int(lora_A.weight.shape[0])
            requires_conversion = not torch.is_autocast_enabled()
            x_noise = x
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x_noise.dtype != compute_dtype:
                    x_noise = x_noise.to(compute_dtype)

            if x_noise.dim() == 2:
                r_a = torch.ones((x_noise.size(0), self.in_features), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_a = torch.ones((x_noise.size(0), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            elif x_noise.dim() == 3:
                r_a = torch.ones((x_noise.size(0), x_noise.size(1), self.in_features), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
                s_a = torch.ones((x_noise.size(0), x_noise.size(1), rank), device=x_noise.device, dtype=x_noise.dtype).uniform_(-1, 1).sign()
            else:
                raise ValueError(f"Unsupported BLoB input rank {x_noise.dim()}, expected 2 or 3.")

            x_noise = dropout(x_noise)
            lora_noise_a = sigma_a.to(dtype=lora_A.weight.dtype) * torch.randn_like(lora_A.weight)
            noise = (((x_noise * r_a) @ lora_noise_a.transpose(0, 1)) * s_a) @ self.lora_B[active_adapter].weight.transpose(0, 1)
            if requires_conversion:
                noise = noise.to(expected_dtype)
            result += noise * scaling

    return result


def _official_div_posterior_prior(self) -> torch.Tensor:
    kl = 0.0
    for active_adapter in _iter_active_adapters(self):
        if active_adapter not in self.lora_A.keys():
            continue
        if not hasattr(self, "lora_A_rho") or active_adapter not in self.lora_A_rho:
            continue
        sigma_weight = _official_sigma_from_rho(self.lora_A_rho[active_adapter], self.bayes_eps)
        kl = kl + _official_kl_div_stable(
            self.lora_A[active_adapter].weight,
            sigma_weight,
            0.0,
            self.bayes_beta,
        )
    return kl


def _official_sample(self, status: bool = True):
    if self.training and not bool(status):
        raise ValueError("blobsample should be set to True only during training.")
    self.blobsample = bool(status)


def _left_padded_last_idx(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.full(
        (input_ids.size(0),),
        input_ids.size(1) - 1,
        device=input_ids.device,
        dtype=torch.long,
    )


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


def wrap_blob_lora_layers(model: PeftModel, adapter_name: str, eps: float, beta: float) -> None:
    wrapped = 0
    for _, mod in _iter_lora_linear_modules(model):
        if not isinstance(mod, LoraLayer):
            continue
        if not isinstance(mod, LoraLinear) and not (Linear8bitLt is not None and isinstance(mod, Linear8bitLt)):
            continue
        A_dict = getattr(mod, "lora_A", None)
        if not isinstance(A_dict, nn.ModuleDict) or adapter_name not in A_dict:
            continue
        if not hasattr(mod, "lora_A_rho") or not isinstance(mod.lora_A_rho, nn.ParameterDict):
            mod.lora_A_rho = nn.ParameterDict({})
        mod.bayes_eps = float(eps)
        mod.bayes_beta = float(beta)
        mod.blobsample = True
        if adapter_name not in mod.lora_A_rho:
            rho = nn.Parameter(A_dict[adapter_name].weight.new_zeros(mod.r[adapter_name], mod.in_features))
            init_blob_rho_(rho, float(eps))
            mod.lora_A_rho[adapter_name] = rho
        mod.div_posterior_prior = _official_div_posterior_prior.__get__(mod, mod.__class__)
        mod.sample = _official_sample.__get__(mod, mod.__class__)
        if Linear8bitLt is not None and isinstance(mod, Linear8bitLt):
            mod.forward = _official_blob_8bitlinear_forward.__get__(mod, mod.__class__)
        else:
            mod.forward = _official_blob_linear_forward.__get__(mod, mod.__class__)
        wrapped += 1
    print(f"[BLoB official] patched {wrapped} LoRA layers (adapter={adapter_name}).")


def patch_blob_forward(model: PeftModel, blob_adapter: str) -> None:
    print(f"[BLoB official] forward is patched via wrap_blob_lora_layers(adapter={blob_adapter}).")


def resolve_blob_paths(blob_dir: str) -> Tuple[str, str]:
    adapter_dir = blob_dir
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        subdir = os.path.join(blob_dir, "blob")
        if os.path.exists(os.path.join(subdir, "adapter_config.json")):
            adapter_dir = subdir
        else:
            raise FileNotFoundError(
                f"Could not find adapter_config.json in '{blob_dir}' or '{subdir}'"
            )

    rho_candidates = [
        os.path.join(blob_dir, "blob_rho.pt"),
        os.path.join(adapter_dir, "blob_rho.pt"),
    ]
    rho_path = next((p for p in rho_candidates if os.path.exists(p)), None)
    if rho_path is None:
        raise FileNotFoundError(f"Missing blob rho file. Tried: {rho_candidates}")
    return adapter_dir, rho_path


def load_blob_rho(model: nn.Module, adapter_name: str, rho_path: str) -> None:
    saved = torch.load(rho_path, map_location="cpu")
    loaded = 0
    for i, (_, mod) in enumerate(_iter_lora_linear_modules(model)):
        if not isinstance(mod, LoraLayer) or not hasattr(mod, "lora_A_rho") or adapter_name not in mod.lora_A_rho:
            continue
        key = f"{i}::{type(mod).__name__}"
        if key not in saved:
            raise KeyError(f"Missing rho tensor for key '{key}' in {rho_path}")
        rho = mod.lora_A_rho[adapter_name]
        rho.data.copy_(saved[key].to(device=rho.device, dtype=rho.dtype))
        loaded += 1
    print(f"[BLoB official] loaded rho for {loaded} modules from: {rho_path}")


def set_blob_sampling(model: nn.Module, adapter_name: str, status: bool) -> None:
    if model.training and not bool(status):
        raise ValueError("blobsample should be set to True only during training.")
    for _, mod in _iter_lora_linear_modules(model):
        if isinstance(mod, LoraLayer) and hasattr(mod, "sample"):
            mod.sample(status)


def collect_blob_kl(model: nn.Module, adapter_name: str) -> torch.Tensor:
    kl_terms = []
    for _, mod in _iter_lora_linear_modules(model):
        if (
            isinstance(mod, LoraLayer)
            and hasattr(mod, "div_posterior_prior")
            and hasattr(mod, "lora_A_rho")
            and adapter_name in mod.lora_A_rho
        ):
            kl_terms.append(mod.div_posterior_prior())
    if not kl_terms:
        raise RuntimeError("No official BLoB LoRA layers found while collecting KL.")
    return torch.stack(kl_terms).sum()


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


# =========================
# Evaluation
# =========================

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
    blob_adapter_name: str,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    model.set_adapter(adapter_name)
    acc_m, ece_m = _make_accuracy(device, num_classes), _make_ece(device, num_classes, 15)
    acc_m.reset(); ece_m.reset()
    total, nll_sum = 0, 0.0
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress_total = total_samples if total_samples is not None else len(loader)
    progress_unit = "sample" if total_samples is not None else "batch"
    progress_start = time.perf_counter()
    batch_iter = tqdm(total=progress_total, desc=(progress_desc or f"{adapter_name}:{mode}"), unit=progress_unit, leave=False)

    if mode in ["det", "blob_mean"] and adapter_name == blob_adapter_name:
        set_blob_sampling(model, blob_adapter_name, False)
    if mode == "blob_sample":
        set_blob_sampling(model, blob_adapter_name, True)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_choices = batch.get("num_choices")
        bsz = labels.size(0)
        total += bsz

        if mode in ["det", "blob_mean"]:
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            logits = _mask_invalid_choices(logits, num_choices)
            probs = torch.softmax(logits, dim=-1)
            nll_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
        else:
            probs_acc = torch.zeros((bsz, num_classes), device=device, dtype=torch.float32)
            for _ in range(int(n_samples)):
                logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
                logits = _mask_invalid_choices(logits, num_choices)
                probs_acc.add_(torch.softmax(logits, dim=-1))
            probs = probs_acc / float(n_samples)
            nll_sum += float((-torch.log(probs[torch.arange(bsz, device=device), labels].clamp_min(1e-12))).sum().item())

        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        elapsed = time.perf_counter() - progress_start
        avg_sec_per_sample = elapsed / max(total, 1)
        if total_samples is not None:
            remaining_samples = max(int(total_samples) - total, 0)
            eta = _format_eta(avg_sec_per_sample * remaining_samples)
            batch_iter.update(bsz)
        else:
            remaining_batches = max(len(loader) - batch_iter.n - 1, 0)
            eta = _format_eta((elapsed / max(batch_iter.n + 1, 1)) * remaining_batches)
            batch_iter.update(1)
        batch_iter.set_postfix(avg_s_per_sample=f"{avg_sec_per_sample:.3f}", eta=eta, refresh=False)

    batch_iter.close()
    if adapter_name == blob_adapter_name: set_blob_sampling(model, blob_adapter_name, True)
    probs_all = torch.cat(all_probs, dim=0) if all_probs else torch.empty((0, num_classes), dtype=torch.float32)
    labels_all = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    metrics = {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": _multiclass_brier_score(probs_all, labels_all) if total > 0 else float("nan"),
    }
    return metrics


# =========================
# Training BLoB from BASE
# =========================

def train_blob_from_base(
    model: PeftModel,
    train_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    steps: int,
    lr: float,
    warmup_ratio: float,
    batch_size: int,
    grad_accum: int,
    blob_adapter: str,
    bayes_beta: float,
    bayes_kllr: float,
    bayes_gamma: float,
    bayes_klreweighting: bool,
    bayes_datasetrescaling: bool,
    bayes_train_n_samples: int,
    amp_dtype: torch.dtype,
):
    model.train()
    model.set_adapter(blob_adapter)

    rho_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A_rho" in n]
    mean_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A_rho" not in n]
    kl_params = [p for p in model.parameters() if p.requires_grad]
    if not rho_params:
        raise RuntimeError("No rho parameters found for BLoB training.")
    if not mean_params:
        raise RuntimeError("No mean parameters found for BLoB training.")
    if not kl_params:
        raise RuntimeError("No trainable parameters found for KL optimizer.")

    opt = torch.optim.AdamW(mean_params, lr=lr)
    opt2 = torch.optim.SGD(kl_params, lr=bayes_kllr)
    sched = get_linear_schedule_with_warmup(opt, int(steps * warmup_ratio), steps)
    sched2 = get_linear_schedule_with_warmup(opt2, int(steps * warmup_ratio), steps)
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

    global_step, loader_iter = 0, iter(train_loader)
    step_i = 1
    last_pi = 0.0
    set_blob_sampling(model, blob_adapter, True)

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
        bsz = int(labels.size(0))

        logits_samples = []
        for _ in range(max(int(bayes_train_n_samples), 1)):
            logits = compute_class_logits(model, input_ids, attention_mask, amp_dtype)
            logits = _mask_invalid_choices(logits, num_choices)
            logits_samples.append(logits)
        logits_stack = torch.stack(logits_samples, dim=1)
        mean_logits = logits_stack.mean(dim=1)
        output = torch.log_softmax(mean_logits, dim=-1)
        nll_loss = F.nll_loss(output, labels, reduction="mean")
        (nll_loss / float(grad_accum)).backward()

        if (global_step + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

            kl_terms = [collect_blob_kl(model, blob_adapter) for _ in range(max(int(bayes_train_n_samples), 1))]
            kl_loss = torch.mean(torch.stack(kl_terms), dim=0)
            pi = (2 ** ((step_i % M) or M)) / (2 ** (M + 1) - 1) if bayes_klreweighting else 1.0 / float(M)
            last_pi = float(pi)
            step_i += 1
            (kl_loss * float(pi)).backward()
            opt2.step()
            opt2.zero_grad(set_to_none=True)
            sched2.step()

        if global_step % 50 == 0:
            print(
                f"[train] step={global_step:5d}/{steps} "
                f"nll={float(nll_loss.item()):.4f} "
                f"kl_lr={float(opt2.param_groups[0]['lr']):.6g} "
                f"pi={last_pi:.6g}"
            )
        global_step += 1


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", "mmlu_mix", SCIENCEQA_CURRIC_TASK_NAME],
        help="Unified tasks",
    )
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--map_adapter_dir", type=str, default="")
    ap.add_argument("--max_seq_len", type=int, default=300)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--train_bsz", type=int, default=32)
    ap.add_argument("--eval_bsz", type=int, default=32)
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
        help="Comma-separated target module names/suffixes, or 'auto_qv_lmhead' to match the current ScienceQA MAP setup.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bayes_eps", type=float, default=0.05)
    ap.add_argument("--bayes_beta", type=float, default=0.2)
    ap.add_argument("--bayes_kllr", type=float, default=0.01)
    ap.add_argument("--bayes_gamma", type=float, default=8.0)
    ap.add_argument("--bayes_klreweighting", action="store_true")
    ap.add_argument("--bayes_datasetrescaling", action="store_true")
    ap.add_argument("--bayes_train_n_samples", type=int, default=1)
    ap.add_argument("--blob_eval_n", type=int, default=10)
    ap.add_argument("--eval_tasks", type=str, default="", help="Comma-separated eval tasks. Supports iid, arc, arc-c, arc-e, sciq, hellaswag, gpqa, gpqa_main, mmlu, mmlu_science_high, mmlu_science_college, scienceqa_closedchoice_grade12")
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_eval", action="store_true")
    ap.add_argument("--save_blob_dir", type=str, default="")
    ap.add_argument("--load_blob_dir", type=str, default="")
    ap.add_argument("--shared_init_lora_path", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n[BLoB official-style] [Task] {args.task} | [Device] {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.bos_token if tok.bos_token is not None else tok.eos_token
    tok.padding_side = TOKENIZER_PADDING_SIDE

    num_classes = get_task_num_classes(args.task)
    choice_token_ids = get_choice_token_ids(tok, device, num_classes)

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=(amp_dtype if device.type == "cuda" else None), attn_implementation="sdpa", trust_remote_code=False).to(device)
    if hasattr(base_model.config, "use_cache"): base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"): base_model.gradient_checkpointing_disable()
    trim_lm_head_to_choice_tokens(base_model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    if str(args.target_modules).strip().lower() == DEFAULT_TARGET_MODULES_SPEC:
        target_modules = resolve_all_layer_target_modules(base_model)
    else:
        target_modules = [m.strip() for m in str(args.target_modules).split(",") if m.strip()]
        if not target_modules:
            raise ValueError("target_modules must contain at least one module name")
    print(
        f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={args.lora_dropout} target_modules={target_modules}"
    )

    if args.load_blob_dir and os.path.isdir(args.load_blob_dir):
        blob_adapter_dir, blob_rho_path = resolve_blob_paths(args.load_blob_dir)
        print(f"[Load BLoB adapter] -> name=blob from {blob_adapter_dir}")
        model = PeftModel.from_pretrained(base_model, blob_adapter_dir, adapter_name="blob", is_trainable=bool(args.do_train)).to(device)
    else:
        blob_adapter_dir, blob_rho_path = "", ""
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_cfg, adapter_name="blob").to(device)
        if args.shared_init_lora_path:
            if not os.path.exists(args.shared_init_lora_path):
                raise FileNotFoundError(f"Missing shared init LoRA file: {args.shared_init_lora_path}")
            saved_init = torch.load(args.shared_init_lora_path, map_location="cpu")
            load_normalized_lora_state_dict(model, saved_init, adapter_name="blob")
            print(f"[Init LoRA] loaded shared init from {args.shared_init_lora_path}")

    wrap_blob_lora_layers(model, adapter_name="blob", eps=args.bayes_eps, beta=args.bayes_beta)
    patch_blob_forward(model, blob_adapter="blob")

    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)

    # Dataset Loading & Preprocessing
    print(f"\n=== Loading {args.task} ===")
    train_raw, val_raw, test_raw = load_task_dataset(args.task)
    if args.task == SCIENCEQA_CURRIC_TASK_NAME:
        train_raw = _order_scienceqa_train_by_grade(train_raw, args.seed)
    task_train = _add_seq_len(preprocess_task(args.task, train_raw, tok, args.max_seq_len, pad_to_max_length=False))
    task_test  = _add_seq_len(preprocess_task(args.task, test_raw, tok, args.max_seq_len, pad_to_max_length=False))
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

    train_shuffle = (args.task != SCIENCEQA_CURRIC_TASK_NAME)
    train_sort_by_len = False if args.task == SCIENCEQA_CURRIC_TASK_NAME else True
    train_loader = _make_loader(task_train, args.train_bsz, shuffle=train_shuffle, drop_last=True, sort_by_len=train_sort_by_len)
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

    if args.load_blob_dir and os.path.isdir(args.load_blob_dir):
        load_blob_rho(model, adapter_name="blob", rho_path=blob_rho_path)

    if args.do_train:
        print(f"\n=== TRAIN: BLoB on {args.task}(train) ===")
        with _Timer(f"BLoB TRAIN ({args.task})"):
            train_blob_from_base(
                model=model,
                train_loader=train_loader,
                device=device,
                num_classes=num_classes,
                steps=args.train_steps,
                lr=args.lr,
                warmup_ratio=args.warmup_ratio,
                batch_size=args.train_bsz,
                grad_accum=max(1, int(args.grad_accum)),
                blob_adapter="blob",
                bayes_beta=args.bayes_beta,
                bayes_kllr=args.bayes_kllr,
                bayes_gamma=args.bayes_gamma,
                bayes_klreweighting=bool(args.bayes_klreweighting),
                bayes_datasetrescaling=bool(args.bayes_datasetrescaling),
                bayes_train_n_samples=args.bayes_train_n_samples,
                amp_dtype=amp_dtype,
            )
        if args.save_blob_dir:
            os.makedirs(args.save_blob_dir, exist_ok=True)
            model.set_adapter("blob")
            model.save_pretrained(args.save_blob_dir)
            torch.save(
                {
                    f"{i}::{type(mod).__name__}": mod.lora_A_rho["blob"].detach().cpu()
                    for i, (_, mod) in enumerate(_iter_lora_linear_modules(model))
                    if isinstance(mod, LoraLayer) and hasattr(mod, "lora_A_rho") and "blob" in mod.lora_A_rho
                },
                os.path.join(args.save_blob_dir, "blob_rho.pt"),
            )
            print(f"[Save] saved BLoB adapter and rho to: {args.save_blob_dir}")

    if args.do_eval:
        for eval_task in eval_tasks:
            eval_loader = eval_loaders[eval_task]
            split_name = "test" if eval_task == args.task else "ood"
            print(f"\n=== EVAL: source={args.task} -> target={eval_task} ({split_name}) ===")
            with _Timer(f"EVAL blob_mean on {eval_task}"):
                m_mean = eval_adapter(model, eval_loader, device, amp_dtype, num_classes, "blob", "blob_mean", 1, "blob", progress_desc=f"BLoB mean {eval_task}")
            with _Timer(f"EVAL blob_sample(N={args.blob_eval_n}) on {eval_task}"):
                m_samp = eval_adapter(model, eval_loader, device, amp_dtype, num_classes, "blob", "blob_sample", args.blob_eval_n, "blob", progress_desc=f"BLoB samp {eval_task}")

            print(f"\n[{eval_task}({split_name}) Results]")
            print(f"  BLoB mean  : NLL={m_mean['nll']:.4f}  ACC={m_mean['acc']*100:.2f}%  ECE={m_mean['ece']*100:.2f}%  Brier={m_mean['brier']:.4f} (N=0)")
            print(f"  BLoB samp  : NLL={m_samp['nll']:.4f}  ACC={m_samp['acc']*100:.2f}%  ECE={m_samp['ece']*100:.2f}%  Brier={m_samp['brier']:.4f} (N={args.blob_eval_n})")

if __name__ == "__main__":
    main()
