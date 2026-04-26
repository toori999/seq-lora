from __future__ import annotations

import argparse
import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from laplace import Laplace
import laplace.utils.matrix as laplace_matrix_utils
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import concatenate_datasets

from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    DynamicEvalCollator,
    answer_key_to_index,
    get_active_adapter_name,
    get_choice_labels,
    get_choice_token_ids,
    get_lm_head_dropout,
    get_lm_head_lora_scaling,
    get_task_num_classes,
    get_transformer_and_lm_head,
    load_eval_dataset,
    load_task_dataset,
    make_accuracy as _make_accuracy,
    make_ece as _make_ece,
    pick_adapter_module,
    preprocess_task,
    _tokenize_prompts,
)

try:
    from asdl.operations.linear import Linear as _AsdlLinearOp
except Exception:
    _AsdlLinearOp = None

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


SUPPORTED_TASKS = [
    "wgs",
    "wgm",
    "arc-c",
    "arc-e",
    "obqa",
    "boolq",
    "sciq",
    SCIENCEQA_CURRIC_TASK_NAME,
]

TASK_ALIASES = {
    "winogrande_s": "wgs",
    "winogrande-small": "wgs",
    "winogrande_m": "wgm",
    "winogrande-medium": "wgm",
    "arc-challenge": "arc-c",
    "arc_challenge": "arc-c",
    "arc-c": "arc-c",
    "arc_easy": "arc-e",
    "arc-easy": "arc-e",
    "arc-e": "arc-e",
    "openbookqa": "obqa",
    "open_book_qa": "obqa",
    "obqa": "obqa",
    "boolq": "boolq",
    "sciq": "sciq",
    SCIENCEQA_CURRIC_TASK_NAME: SCIENCEQA_CURRIC_TASK_NAME,
}

_LAPLACE_FIT_CACHE_FORMAT = "official_source_laplace_fit_cache_v1"
FIXED_INTERNAL_SEED = 0


def _patch_asdl_linear_dtype_mismatch() -> None:
    if _AsdlLinearOp is None:
        return

    @staticmethod
    def _batch_grads_weight_safe(module: nn.Module, in_data: torch.Tensor, out_grads: torch.Tensor):
        if in_data.dtype != out_grads.dtype:
            in_data = in_data.to(dtype=out_grads.dtype)
        return torch.bmm(out_grads.unsqueeze(2), in_data.unsqueeze(1))

    _AsdlLinearOp.batch_grads_weight = _batch_grads_weight_safe


_patch_asdl_linear_dtype_mismatch()


def _patch_laplace_kron_dtype_mismatch() -> None:
    kron_cls = getattr(laplace_matrix_utils, "KronDecomposed", None)
    if kron_cls is None:
        return

    orig_inv_square_form = kron_cls.inv_square_form

    def _infer_target_dtype(self):
        for attr_name in ("eigenvectors", "eigenvalues"):
            groups = getattr(self, attr_name, None)
            if groups is None:
                continue
            for group in groups:
                for tensor in group:
                    if isinstance(tensor, torch.Tensor):
                        return tensor.dtype
        deltas = getattr(self, "deltas", None)
        if isinstance(deltas, torch.Tensor):
            return deltas.dtype
        return None

    def _inv_square_form_safe(self, W: torch.Tensor) -> torch.Tensor:
        target_dtype = _infer_target_dtype(self)
        if target_dtype is not None and W.dtype != target_dtype:
            W = W.to(dtype=target_dtype)
        return orig_inv_square_form(self, W)

    kron_cls.inv_square_form = _inv_square_form_safe


_patch_laplace_kron_dtype_mismatch()


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


def _release_laplace_fit_state(la) -> None:
    # KronLaplace.fit() keeps both H_facs and the decomposed H. Evaluation only
    # needs the decomposed posterior precision, so release fit-only state here.
    if hasattr(la, "H_facs"):
        la.H_facs = None
    if hasattr(la, "_backend"):
        la._backend = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_base_model_name(adapter_dir: str, model_name_or_path: str) -> str:
    if model_name_or_path:
        return model_name_or_path
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    return str(peft_cfg.base_model_name_or_path)


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


def _multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def _split_train_for_testing_set(train_raw, testing_set: str):
    if testing_set == "val":
        return train_raw, None
    if testing_set != "train_val":
        raise ValueError(f"Unsupported testing_set={testing_set!r}")
    split = train_raw.train_test_split(test_size=0.2, seed=42, shuffle=False)
    return split["train"], split["test"]


def _normalize_task_name(task_name: str) -> str:
    raw = str(task_name or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("/", "-")
    normalized = TASK_ALIASES.get(key, TASK_ALIASES.get(raw, raw))
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task_name={task_name!r}. Supported canonical tasks: "
            f"{', '.join(SUPPORTED_TASKS)}"
        )
    return normalized


def _infer_task_name_from_adapter_dir(adapter_dir: str) -> str:
    parts = [p.lower() for p in os.path.normpath(adapter_dir).split(os.sep) if p]
    joined = "/".join(parts)
    ordered_aliases = [
        ("scienceqa_text_closedchoice_grade2_11", SCIENCEQA_CURRIC_TASK_NAME),
        (SCIENCEQA_CURRIC_TASK_NAME, SCIENCEQA_CURRIC_TASK_NAME),
        ("arc-challenge", "arc-c"),
        ("arc_challenge", "arc-c"),
        ("arc-easy", "arc-e"),
        ("arc_easy", "arc-e"),
        ("openbookqa", "obqa"),
        ("obqa", "obqa"),
        ("boolq", "boolq"),
        ("winogrande_s", "wgs"),
        ("winogrande_m", "wgm"),
        ("sciq", "sciq"),
    ]
    for needle, task in ordered_aliases:
        if needle in joined:
            return task
    raise ValueError(
        "Could not infer --task_name from adapter path. Pass one of: "
        f"{', '.join(SUPPORTED_TASKS)}"
    )


@dataclass(frozen=True)
class AdapterHeadInfo:
    mode: str
    rows: Optional[int]
    source_key: str


def _iter_adapter_tensor_shapes(adapter_dir: str):
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(st_path):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, tuple(int(x) for x in f.get_tensor(key).shape)
        return

    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                yield key, tuple(int(x) for x in value.shape)
        return

    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors or adapter_model.bin under {adapter_dir}"
    )


def _inspect_adapter_head(adapter_dir: str, num_classes: int) -> AdapterHeadInfo:
    lm_head_rows: List[tuple[str, int]] = []
    has_lm_head_tensor = False
    for key, shape in _iter_adapter_tensor_shapes(adapter_dir):
        if "lm_head" not in key:
            continue
        has_lm_head_tensor = True
        if len(shape) < 2:
            continue
        if key.endswith("lm_head.base_layer.weight") or ".lm_head.base_layer.weight" in key:
            lm_head_rows.insert(0, (key, int(shape[0])))
        elif ".lm_head.lora_B." in key and key.endswith(".weight"):
            lm_head_rows.append((key, int(shape[0])))

    if not has_lm_head_tensor:
        return AdapterHeadInfo(mode="no_lm_head_adapter", rows=None, source_key="")
    if not lm_head_rows:
        return AdapterHeadInfo(mode="unknown_lm_head_adapter", rows=None, source_key="")

    source_key, rows = lm_head_rows[0]
    if rows == int(num_classes):
        return AdapterHeadInfo(mode="trimmed_head", rows=rows, source_key=source_key)
    if rows > int(num_classes):
        return AdapterHeadInfo(mode="full_vocab_head", rows=rows, source_key=source_key)
    raise ValueError(
        f"Adapter lm_head rows={rows} from {source_key!r}, but task has "
        f"{int(num_classes)} classes. Check --task_name."
    )


def _is_arc_task(task_name: str) -> bool:
    return str(task_name).lower().strip() in {"arc-c", "arc-e"}


def _is_bayesian_peft_arc_adapter(task_name: str, adapter_dir: str, base_model_name: str) -> bool:
    if not _is_arc_task(task_name):
        return False
    path = os.path.normpath(adapter_dir).lower()
    return "llama" in str(base_model_name).lower() and (
        "arc-challenge" in path or "arc-easy" in path
    )


def _task_num_classes(task_name: str, bayesian_peft_arc_style: bool) -> int:
    if bayesian_peft_arc_style and _is_arc_task(task_name):
        return 5
    return get_task_num_classes(task_name)


def preprocess_bayesian_peft_arc(
    ds,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
):
    """Match bayesian-peft's mcdataset ARC setup: A-E logits, no 4-choice filter."""

    def _fn(batch: Dict) -> Dict:
        prompts: List[str] = []
        labels: List[int] = []
        for question, choices, answer_key in zip(
            batch["question"],
            batch["choices"],
            batch["answerKey"],
        ):
            try:
                label_order = [str(x) for x in choices["label"]]
                text_by_label = {
                    str(label): str(text)
                    for label, text in zip(choices["label"], choices["text"])
                }
                choice_lines = "\n".join(
                    f"{label}) {text_by_label[label]}" for label in label_order
                )
                prompts.append(
                    "Return the label of the correct answer for the question below.\n\n"
                    f"Question: {question}\n"
                    f"Choices:\n{choice_lines}\n"
                    "Answer:"
                )

                answer = str(answer_key).strip()
                if answer.isdigit():
                    y = int(answer) - 1
                else:
                    y = answer_key_to_index(answer, get_choice_labels(5))
                if y < 0 or y >= 5:
                    raise ValueError(f"ARC label out of range: {answer_key!r}")
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = _tokenize_prompts(
            tokenizer,
            prompts,
            max_len,
            pad_to_max_length=pad_to_max_length,
        )
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1)
    return ds2.remove_columns(
        [c for c in ds2.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    )


def _preprocess_laplace_task(
    task_name: str,
    ds,
    tokenizer: AutoTokenizer,
    max_len: int,
    *,
    bayesian_peft_arc_style: bool,
):
    if bayesian_peft_arc_style and _is_arc_task(task_name):
        return preprocess_bayesian_peft_arc(
            ds,
            tokenizer,
            max_len,
            pad_to_max_length=False,
        )
    return preprocess_task(
        task_name,
        ds,
        tokenizer,
        max_len,
        pad_to_max_length=False,
    )


def _tensor_groups_to_cpu(groups):
    return [[tensor.detach().cpu() for tensor in group] for group in groups]


def _tensor_groups_to_device(groups, device: torch.device):
    return [[tensor.to(device=device) for tensor in group] for group in groups]


def _serialize_laplace_hessian(H) -> Dict[str, object]:
    if isinstance(H, laplace_matrix_utils.KronDecomposed):
        return {
            "kind": "KronDecomposed",
            "eigenvectors": _tensor_groups_to_cpu(H.eigenvectors),
            "eigenvalues": _tensor_groups_to_cpu(H.eigenvalues),
            "deltas": H.deltas.detach().cpu(),
            "damping": bool(H.damping),
        }
    if isinstance(H, laplace_matrix_utils.Kron):
        return {
            "kind": "Kron",
            "kfacs": _tensor_groups_to_cpu(H.kfacs),
        }
    raise TypeError(f"Unsupported Laplace Hessian state type: {type(H)!r}")


def _deserialize_laplace_hessian(payload: MutableMapping[str, object], device: torch.device):
    kind = str(payload.get("kind", ""))
    if kind == "KronDecomposed":
        return laplace_matrix_utils.KronDecomposed(
            eigenvectors=_tensor_groups_to_device(payload["eigenvectors"], device),
            eigenvalues=_tensor_groups_to_device(payload["eigenvalues"], device),
            deltas=payload["deltas"].to(device=device),
            damping=bool(payload.get("damping", False)),
        )
    if kind == "Kron":
        return laplace_matrix_utils.Kron(
            _tensor_groups_to_device(payload["kfacs"], device)
        )
    raise ValueError(f"Unknown Laplace Hessian cache kind: {kind!r}")


def _build_laplace_fit_cache_payload(
    la,
    *,
    base_model_name: str,
    map_adapter_dir: str,
    task_name: str,
    subset_tag: str,
    testing_set: str,
    max_length: int,
    selected_param_names: List[str],
) -> Dict[str, object]:
    return {
        "format": _LAPLACE_FIT_CACHE_FORMAT,
        "base_model_name": base_model_name,
        "map_adapter_dir": os.path.abspath(map_adapter_dir),
        "task_name": task_name,
        "subset_tag": subset_tag,
        "testing_set": testing_set,
        "max_length": int(max_length),
        "selected_param_names": list(selected_param_names),
        "n_params": int(la.n_params),
        "n_layers": int(la.n_layers),
        "mean": la.mean.detach().cpu(),
        "loss": float(torch.as_tensor(la.loss).detach().cpu().item()),
        "n_data": int(la.n_data),
        "n_outputs": int(la.n_outputs),
        "H": _serialize_laplace_hessian(la.H),
    }


def _restore_laplace_fit_cache(
    la,
    payload: MutableMapping[str, object],
    *,
    base_model_name: str,
    map_adapter_dir: str,
    task_name: str,
    subset_tag: str,
    testing_set: str,
    max_length: int,
    selected_param_names: List[str],
    expected_n_data: int,
) -> None:
    if payload.get("format") != _LAPLACE_FIT_CACHE_FORMAT:
        raise ValueError(f"Unsupported cache format: {payload.get('format')!r}")
    if str(payload.get("base_model_name")) != base_model_name:
        raise ValueError("Base model name does not match fit cache.")
    if str(payload.get("map_adapter_dir")) != os.path.abspath(map_adapter_dir):
        raise ValueError("Adapter directory does not match fit cache.")
    if str(payload.get("task_name")) != task_name:
        raise ValueError("Task name does not match fit cache.")
    if str(payload.get("subset_tag")) != subset_tag:
        raise ValueError("Laplace subset does not match fit cache.")
    if str(payload.get("testing_set")) != testing_set:
        raise ValueError("testing_set does not match fit cache.")
    if int(payload.get("max_length", -1)) != int(max_length):
        raise ValueError("max_length does not match fit cache.")

    cached_param_names = list(payload.get("selected_param_names", []))
    if cached_param_names != list(selected_param_names):
        raise ValueError("Selected Laplace parameter names do not match fit cache.")

    if int(payload.get("n_params", -1)) != int(la.n_params):
        raise ValueError("Laplace parameter count does not match fit cache.")
    if int(payload.get("n_layers", -1)) != int(la.n_layers):
        raise ValueError("Laplace layer count does not match fit cache.")
    if int(payload.get("n_data", -1)) != int(expected_n_data):
        raise ValueError("Fit dataset size does not match fit cache.")

    mean = payload["mean"]
    if not isinstance(mean, torch.Tensor) or mean.numel() != la.n_params:
        raise ValueError("Cached posterior mean has invalid shape.")

    H = _deserialize_laplace_hessian(payload["H"], la._device)
    if isinstance(H, laplace_matrix_utils.Kron):
        H = H.decompose(damping=getattr(la, "damping", False))

    la.mean = mean.to(device=la._device)
    la.loss = float(payload["loss"])
    la.n_data = int(payload["n_data"])
    la.n_outputs = int(payload["n_outputs"])
    la.H = H
    if hasattr(la, "H_facs"):
        la.H_facs = None
    setattr(la.model, "output_size", la.n_outputs)


def _order_scienceqa_train_by_grade(train_raw, seed: int):
    if "grade_num" not in train_raw.column_names:
        return train_raw

    grade_values = sorted({int(g) for g in train_raw["grade_num"]})
    parts: List[object] = []
    for grade_num in grade_values:
        idxs = [i for i, g in enumerate(train_raw["grade_num"]) if int(g) == grade_num]
        if not idxs:
            continue
        ds_g = train_raw.select(idxs).shuffle(seed=seed + grade_num)
        parts.append(ds_g)
    if not parts:
        raise RuntimeError("No ScienceQA training examples left after grade ordering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


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


class ChoiceRestrictedLoraLMHead(nn.Module):
    """Full-vocab PEFT lm_head restricted to the task's choice-token rows."""

    def __init__(
        self,
        original_lm_head: nn.Module,
        choice_token_ids: torch.Tensor,
        adapter_name: str,
    ):
        super().__init__()
        device = choice_token_ids.device
        ids = choice_token_ids.to(device=device, dtype=torch.long)

        base_weight = getattr(original_lm_head, "weight", None)
        if base_weight is None and hasattr(original_lm_head, "base_layer"):
            base_weight = getattr(original_lm_head.base_layer, "weight", None)
        if base_weight is None:
            raise RuntimeError("Could not locate lm_head base weight for full-vocab adapter restriction.")

        base_bias = getattr(original_lm_head, "bias", None)
        if base_bias is None and hasattr(original_lm_head, "base_layer"):
            base_bias = getattr(original_lm_head.base_layer, "bias", None)

        weight = base_weight.index_select(0, ids).detach().to(device=device, dtype=torch.float32)
        bias = (
            None
            if base_bias is None
            else base_bias.index_select(0, ids).detach().to(device=device, dtype=torch.float32)
        )

        self.linear = nn.Linear(
            in_features=int(weight.shape[1]),
            out_features=int(weight.shape[0]),
            bias=(bias is not None),
            device=device,
            dtype=torch.float32,
        )
        self.linear.weight.data.copy_(weight)
        self.linear.weight.requires_grad = False
        if bias is not None:
            self.linear.bias.data.copy_(bias)
            self.linear.bias.requires_grad = False

        A_mod = pick_adapter_module(getattr(original_lm_head, "lora_A", None), adapter_name)
        B_mod = pick_adapter_module(getattr(original_lm_head, "lora_B", None), adapter_name)
        if A_mod is None or B_mod is None or not hasattr(A_mod, "weight") or not hasattr(B_mod, "weight"):
            self.lora_A = None
            self.lora_B = None
        else:
            A_w = A_mod.weight.detach().to(device=device, dtype=torch.float32)
            B_w = B_mod.weight.index_select(0, ids).detach().to(device=device, dtype=torch.float32)
            self.lora_A = nn.Linear(
                in_features=int(A_w.shape[1]),
                out_features=int(A_w.shape[0]),
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.lora_B = nn.Linear(
                in_features=int(B_w.shape[1]),
                out_features=int(B_w.shape[0]),
                bias=False,
                device=device,
                dtype=torch.float32,
            )
            self.lora_A.weight.data.copy_(A_w)
            self.lora_B.weight.data.copy_(B_w)

        dropout = get_lm_head_dropout(original_lm_head, adapter_name)
        self.lora_dropout = dropout if dropout is not None else nn.Identity()
        self.scaling = float(get_lm_head_lora_scaling(original_lm_head, adapter_name))

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        h = hidden_states.to(dtype=torch.float32)
        logits = self.linear(h)
        if self.lora_A is not None and self.lora_B is not None:
            logits = logits + self.lora_B(self.lora_A(self.lora_dropout(h))) * self.scaling
        return logits


def restrict_full_vocab_lm_head_to_choice_tokens(
    peft_model: PeftModel,
    choice_token_ids: torch.Tensor,
) -> None:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    _, lm_head = get_transformer_and_lm_head(peft_model)
    adapter_name = get_active_adapter_name(peft_model)
    restricted = ChoiceRestrictedLoraLMHead(
        original_lm_head=lm_head,
        choice_token_ids=choice_token_ids,
        adapter_name=adapter_name,
    )
    if hasattr(base, "lm_head"):
        base.lm_head = restricted
    elif hasattr(base, "set_output_embeddings"):
        base.set_output_embeddings(restricted)
    else:
        raise RuntimeError("Could not replace lm_head with restricted choice head.")
    if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
        base.config.vocab_size = int(choice_token_ids.numel())


class SourceStyleLaplaceWrapper(nn.Module):
    def __init__(
        self,
        peft_model: PeftModel,
        amp_dtype: torch.dtype,
        choice_token_ids: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.peft_model = peft_model
        self.amp_dtype = amp_dtype
        self.choice_token_ids = choice_token_ids
        self.num_classes = num_classes

    @property
    def device(self) -> torch.device:
        return next(self.peft_model.parameters()).device

    def forward(self, **batch) -> torch.Tensor:
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
            logits = out.logits
            if logits.ndim == 3:
                logits = logits[:, -1, :]
            elif logits.ndim != 2:
                raise RuntimeError(f"Expected 2D or 3D logits, got shape={tuple(logits.shape)}")
            if (
                self.choice_token_ids is not None
                and self.num_classes is not None
                and logits.size(-1) != int(self.num_classes)
            ):
                logits = logits.index_select(-1, self.choice_token_ids.to(device=logits.device))
            logits = logits.to(dtype=torch.float32)
        return _mask_invalid_choices(logits, num_choices)


def _configure_laplace_trainable_params(model: nn.Module, laplace_sub: str) -> List[str]:
    selected: List[str] = []
    for name, param in model.named_parameters():
        use_param = False
        if ".lm_head." in name and "lora_" in name:
            use_param = True
        elif laplace_sub == "all" and ("lora_" in name) and (".lm_head." not in name):
            use_param = True
        param.requires_grad = use_param
        if use_param:
            selected.append(name)
    if not selected:
        raise RuntimeError("No Laplace parameters selected.")
    return selected


def _gaussian_mc_prob_mean(
    f_mu: torch.Tensor,
    f_var: torch.Tensor,
    mc_samples: int,
    mc_chunk: int,
) -> torch.Tensor:
    if mc_samples <= 0:
        raise ValueError(f"mc_samples must be positive, got {mc_samples}.")

    total_prob = None
    samples_done = 0
    chunk_size = mc_samples if mc_chunk <= 0 else mc_chunk
    eye = torch.eye(f_var.shape[-1], device=f_var.device, dtype=f_var.dtype)
    L = torch.linalg.cholesky(f_var + eye * 1e-6)

    while samples_done < mc_samples:
        cur = min(chunk_size, mc_samples - samples_done)
        f_mu_chunk = f_mu.unsqueeze(0).expand(cur, -1, -1)
        noise = torch.randn((cur,) + tuple(f_mu.shape), device=f_mu.device, dtype=f_mu.dtype).unsqueeze(-1)
        logits = f_mu_chunk + (L.unsqueeze(0) @ noise).squeeze(-1)
        prob_sum = torch.softmax(logits, dim=-1).sum(dim=0)
        total_prob = prob_sum if total_prob is None else (total_prob + prob_sum)
        samples_done += cur

    return total_prob / float(mc_samples)


@torch.no_grad()
def eval_map_source_style(
    model: SourceStyleLaplaceWrapper,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 10)
    acc_m.reset()
    ece_m.reset()
    total = 0
    nll_sum = 0.0
    eps = 1e-12
    brier_sum = 0.0

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        probs = torch.softmax(model(**batch), dim=-1)
        bsz = int(labels.size(0))
        total += bsz
        nll_sum += float((-torch.log(probs[torch.arange(bsz, device=device), labels].clamp_min(eps))).sum().item())
        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )

    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": (brier_sum / max(total, 1) if total > 0 else float("nan")),
    }


def eval_laplace_source_mc_corr(
    la,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    mc_samples: int,
    mc_chunk: int,
    progress_desc: str,
) -> Dict[str, float]:
    acc_m = _make_accuracy(device, num_classes)
    ece_m = _make_ece(device, num_classes, 10)
    acc_m.reset()
    ece_m.reset()

    total = 0
    nll_sum = 0.0
    eps = 1e-12
    brier_sum = 0.0

    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress_total = total_samples if total_samples is not None else len(loader)
    progress_unit = "sample" if total_samples is not None else "batch"
    progress_start = time.perf_counter()
    batch_iter = tqdm(total=progress_total, desc=progress_desc, unit=progress_unit, leave=False)

    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        labels = batch["labels"]
        bsz = int(labels.size(0))
        with torch.enable_grad():
            f_mu, f_var = la._glm_predictive_distribution(batch)
        probs = _gaussian_mc_prob_mean(
            f_mu=f_mu.to(device=device, dtype=torch.float32),
            f_var=f_var.to(device=device, dtype=torch.float32),
            mc_samples=int(mc_samples),
            mc_chunk=int(mc_chunk),
        ).to(device=device, dtype=torch.float32)

        nll_sum += float((-torch.log(probs[torch.arange(bsz, device=device), labels].clamp_min(eps))).sum().item())
        total += bsz
        acc_m.update(probs, labels)
        ece_m.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )
        del f_mu, f_var, probs, batch

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
    return {
        "nll": nll_sum / max(total, 1),
        "acc": float(acc_m.compute().item()),
        "ece": float(ece_m.compute().item()),
        "brier": (brier_sum / max(total, 1) if total > 0 else float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Official-source-style Laplace-LoRA evaluation for MCQA.")
    ap.add_argument(
        "--task_name",
        type=str,
        default="",
        help=(
            "Source task. If omitted, inferred from --map_adapter_dir when possible. "
            f"Canonical tasks: {', '.join(SUPPORTED_TASKS)}"
        ),
    )
    ap.add_argument("--map_adapter_dir", type=str, required=True)
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Optional override. By default this is read from adapter_config.json.",
    )
    ap.add_argument("--output_dir", type=str, default="./outputs_laplace_official_source")
    ap.add_argument("--eval_tasks", type=str, default="iid")
    ap.add_argument("--max_length", type=int, default=300)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--fit_bsz", type=int, default=32)
    ap.add_argument("--laplace_bsz", type=int, default=16)
    ap.add_argument("--laplace_hessian", type=str, default="kron", choices=["kron"])
    ap.add_argument("--laplace_sub", type=str, default="all", choices=["last_layer", "all"])
    ap.add_argument("--testing_set", type=str, default="val", choices=["val", "train_val"])
    ap.add_argument("--prior_var", type=float, default=1.0)
    ap.add_argument("--prior_opt_lr", type=float, default=1e-1)
    ap.add_argument("--prior_optim_step", type=int, default=1000)
    ap.add_argument("--laplace_mc_samples", type=int, default=1000)
    ap.add_argument("--laplace_mc_chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=FIXED_INTERNAL_SEED)
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--force_refit", action="store_true", help="Ignore saved KFAC/Hessian cache and recompute la.fit().")
    ap.add_argument("--force_reprior", action="store_true", help="Ignore saved prior precision cache and re-optimize prior.")
    args = ap.parse_args()
    args.map_adapter_dir = os.path.abspath(args.map_adapter_dir)
    args.task_name = (
        _normalize_task_name(args.task_name)
        if str(args.task_name).strip()
        else _infer_task_name_from_adapter_dir(args.map_adapter_dir)
    )

    requested_seed = int(args.seed)
    effective_seed = int(FIXED_INTERNAL_SEED)
    if requested_seed != effective_seed:
        print(
            f"[Seed] requested --seed={requested_seed}, but Laplace internal seed is fixed to {effective_seed} for benchmark consistency."
        )
    set_seed(effective_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    pin_memory = (device.type == "cuda")

    base_model_name = _resolve_base_model_name(args.map_adapter_dir, args.model_name_or_path)
    print(f"[Official source Laplace] task={args.task_name} sub={args.laplace_sub} testing_set={args.testing_set}")
    print(f"[Base model] {base_model_name}")
    print(f"[Adapter] {args.map_adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
    ).to(device)
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    bayesian_peft_arc_style = _is_bayesian_peft_arc_adapter(
        args.task_name,
        args.map_adapter_dir,
        base_model_name,
    )
    if bayesian_peft_arc_style:
        print("[Dataset style] bayesian-peft ARC mcdataset: using 5 A-E logits without 4-choice filtering")

    num_classes = _task_num_classes(args.task_name, bayesian_peft_arc_style)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)
    adapter_head = _inspect_adapter_head(args.map_adapter_dir, num_classes)
    print(
        "[Adapter head] "
        f"mode={adapter_head.mode} rows={adapter_head.rows} source={adapter_head.source_key or 'n/a'}"
    )
    if adapter_head.mode in {"trimmed_head", "no_lm_head_adapter", "unknown_lm_head_adapter"}:
        trim_lm_head_to_choice_tokens(base, choice_token_ids)
        print(f"[Head] trimmed base lm_head to {num_classes} choice logits before adapter load")
        model = PeftModel.from_pretrained(base, args.map_adapter_dir, is_trainable=True).to(device)
        head_cache_tag = "trimmedhead"
        wrapper_choice_ids = None
    elif adapter_head.mode == "full_vocab_head":
        model = PeftModel.from_pretrained(base, args.map_adapter_dir, is_trainable=True).to(device)
        restrict_full_vocab_lm_head_to_choice_tokens(model, choice_token_ids)
        print(
            f"[Head] loaded full-vocab adapter then restricted lm_head to "
            f"{num_classes} choice logits"
        )
        head_cache_tag = "fullvocab_choicehead"
        wrapper_choice_ids = choice_token_ids
    else:
        raise RuntimeError(f"Unsupported adapter head mode: {adapter_head.mode}")
    model.eval()

    selected_param_names = _configure_laplace_trainable_params(model, args.laplace_sub)
    print(f"[Laplace subset] enabled {len(selected_param_names)} tensors")
    for name in selected_param_names[:8]:
        print(f"  {name}")

    laplace_model = SourceStyleLaplaceWrapper(
        model,
        amp_dtype,
        choice_token_ids=wrapper_choice_ids,
        num_classes=num_classes,
    ).to(device)

    train_raw_full, val_raw, test_raw = load_task_dataset(args.task_name)
    if args.task_name == SCIENCEQA_CURRIC_TASK_NAME:
        train_raw_fit = _order_scienceqa_train_by_grade(train_raw_full, effective_seed)
        prior_raw = val_raw
        print("[ScienceQA] fit data ordered by grade curriculum; prior tuning uses validation split.")
    else:
        train_raw_fit, train_raw_valsplit = _split_train_for_testing_set(train_raw_full, args.testing_set)
        prior_raw = val_raw if args.testing_set == "val" else train_raw_valsplit
        if prior_raw is None:
            raise RuntimeError("Prior validation split resolved to None.")

    iid_train = _add_seq_len(
        _preprocess_laplace_task(
            args.task_name,
            train_raw_fit,
            tokenizer,
            args.max_length,
            bayesian_peft_arc_style=bayesian_peft_arc_style,
        )
    )
    iid_prior = _add_seq_len(
        _preprocess_laplace_task(
            args.task_name,
            prior_raw,
            tokenizer,
            args.max_length,
            bayesian_peft_arc_style=bayesian_peft_arc_style,
        )
    )
    iid_test = _add_seq_len(
        _preprocess_laplace_task(
            args.task_name,
            test_raw,
            tokenizer,
            args.max_length,
            bayesian_peft_arc_style=bayesian_peft_arc_style,
        )
    )

    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task_name)
    eval_task_to_proc: Dict[str, object] = {args.task_name: iid_test}
    for eval_task in eval_tasks:
        if eval_task == args.task_name:
            continue
        eval_num_classes = _task_num_classes(
            eval_task,
            bayesian_peft_arc_style and _is_arc_task(eval_task),
        )
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task {eval_task!r} has {eval_num_classes} classes, "
                f"but source task {args.task_name!r} has {num_classes}."
            )
        eval_raw = load_eval_dataset(eval_task)
        eval_task_to_proc[eval_task] = _add_seq_len(
            _preprocess_laplace_task(
                eval_task,
                eval_raw,
                tokenizer,
                args.max_length,
                bayesian_peft_arc_style=(
                    bayesian_peft_arc_style and _is_arc_task(eval_task)
                ),
            )
        )

    eval_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    def _make_loader(proc, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        proc_loader = proc
        if not shuffle and "seq_len" in proc_loader.column_names:
            proc_loader = proc_loader.sort("seq_len")
        if "seq_len" in proc_loader.column_names:
            proc_loader = proc_loader.remove_columns(["seq_len"])
        return DataLoader(
            proc_loader,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=eval_collator,
            num_workers=0,
            pin_memory=pin_memory,
        )

    fit_shuffle = (args.task_name != SCIENCEQA_CURRIC_TASK_NAME)
    fit_loader = _make_loader(iid_train, int(args.fit_bsz), shuffle=fit_shuffle, drop_last=False)
    prior_loader = _make_loader(iid_prior, int(args.per_device_eval_batch_size), shuffle=False, drop_last=False)

    dataset_style_tag = "bp_arc5" if bayesian_peft_arc_style else "seq_mcqa"
    subset_tag = f"official_source_{head_cache_tag}_{dataset_style_tag}_{args.laplace_sub}"
    prior_mode = "valsplit" if (args.task_name == SCIENCEQA_CURRIC_TASK_NAME or args.testing_set == "val") else "trainvalsplit"
    fit_cache_path = os.path.join(
        args.output_dir,
        f"laplace_fit_{subset_tag}_{args.task_name}_{prior_mode}_maxlen{int(args.max_length)}.pth",
    )
    prior_path = os.path.join(
        args.output_dir,
        (
            f"prior_precision_{subset_tag}_{args.task_name}_{prior_mode}"
            f"_psteps{int(args.prior_optim_step)}.pth"
        ),
    )

    with _StageTimer(f"OFFICIAL SOURCE Laplace fit on {args.task_name}"):
        la = Laplace(
            laplace_model,
            likelihood="classification",
            subset_of_weights="all",
            hessian_structure=args.laplace_hessian,
            prior_precision=(1.0 / max(float(args.prior_var), 1e-12)),
        )
        fit_cache_loaded = False
        if (not args.force_refit) and os.path.exists(fit_cache_path):
            try:
                saved_fit = torch.load(fit_cache_path, map_location="cpu")
                _restore_laplace_fit_cache(
                    la,
                    saved_fit,
                    base_model_name=base_model_name,
                    map_adapter_dir=args.map_adapter_dir,
                    task_name=args.task_name,
                    subset_tag=subset_tag,
                    testing_set=prior_mode,
                    max_length=int(args.max_length),
                    selected_param_names=selected_param_names,
                    expected_n_data=len(iid_train),
                )
                fit_cache_loaded = True
                print(f"[Laplace fit] Loaded KFAC/Hessian cache from {fit_cache_path}")
            except Exception as exc:
                print(f"[Laplace fit] Failed to load cache from {fit_cache_path}; recomputing la.fit(): {exc}")

        if not fit_cache_loaded:
            la.fit(fit_loader)
            try:
                torch.save(
                    _build_laplace_fit_cache_payload(
                        la,
                        base_model_name=base_model_name,
                        map_adapter_dir=args.map_adapter_dir,
                        task_name=args.task_name,
                        subset_tag=subset_tag,
                        testing_set=prior_mode,
                        max_length=int(args.max_length),
                        selected_param_names=selected_param_names,
                    ),
                    fit_cache_path,
                )
                print(f"[Laplace fit] Saved KFAC/Hessian cache to {fit_cache_path}")
            except Exception as exc:
                print(f"[Laplace fit] Warning: failed to save KFAC/Hessian cache to {fit_cache_path}: {exc}")

        if (not args.force_reprior) and os.path.exists(prior_path):
            saved = torch.load(prior_path, map_location="cpu")
            prior_payload = saved["prior_precision"] if isinstance(saved, dict) and "prior_precision" in saved else saved
            prior_precision = torch.as_tensor(prior_payload, device=device, dtype=torch.float32)
            la.prior_precision = prior_precision
            print(f"[Prior] Loaded prior precision from {prior_path}")
        elif args.task_name == SCIENCEQA_CURRIC_TASK_NAME or args.testing_set == "val":
            print(f"[Prior] optimizing with method=marglik, steps={int(args.prior_optim_step)}")
            la.optimize_prior_precision(
                method="marglik",
                n_steps=int(args.prior_optim_step),
                lr=float(args.prior_opt_lr),
            )
            torch.save(torch.as_tensor(la.prior_precision).detach().cpu(), prior_path)
        else:
            print(f"[Prior] optimizing with method=val_gd, steps={int(args.prior_optim_step)}")
            la.optimize_prior_precision(
                method="val_gd",
                val_loader=prior_loader,
                n_steps=int(args.prior_optim_step),
                lr=float(args.prior_opt_lr),
            )
            torch.save(torch.as_tensor(la.prior_precision).detach().cpu(), prior_path)

    _release_laplace_fit_state(la)
    prior_precision = torch.as_tensor(la.prior_precision, device=device, dtype=torch.float32)
    print(f"[Prior] precision={float(prior_precision.flatten()[0].item()):.6g}")

    def _eval_one(tag: str, proc) -> None:
        print("\n==============================")
        print(f"[{tag}] n={len(proc)}")
        print("==============================")

        lap_loader = _make_loader(proc, int(args.laplace_bsz), False, False)

        with _StageTimer(f"INFER Official-Source-Laplace on {tag}"):
            m_lap = eval_laplace_source_mc_corr(
                la=la,
                loader=lap_loader,
                device=device,
                num_classes=num_classes,
                mc_samples=int(args.laplace_mc_samples),
                mc_chunk=int(args.laplace_mc_chunk),
                progress_desc=f"OfficialSrc-LAP {tag}",
            )
        print(
            f"LAP:  NLL={m_lap['nll']:.4f}  ACC={m_lap['acc']*100:.2f}%  "
            f"ECE={m_lap['ece']*100:.2f}%  Brier={m_lap['brier']:.4f}"
        )

    print(f"\n=== Official-source evaluation: source={args.task_name} | targets={eval_tasks} ===")
    for eval_task in eval_tasks:
        split_name = "test" if eval_task == args.task_name else "ood"
        _eval_one(f"{eval_task}({split_name})", eval_task_to_proc[eval_task])
    print("\n[DONE]")


if __name__ == "__main__":
    main()
