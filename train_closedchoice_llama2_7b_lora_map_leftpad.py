from __future__ import annotations

import argparse
from typing import Dict, List, Sequence
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

from common_eval_utils import (
    DynamicEvalCollator,
    get_choice_token_ids,
    get_task_num_classes,
    load_task_dataset,
    preprocess_task,
)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TRUST_REMOTE_CODE = False
ATTN_IMPLEMENTATION = "sdpa"
FALLBACK_ATTN_IMPLEMENTATION = "sdpa"

DEFAULT_TASK_NAME = "obqa"
RUN_TAG_TEMPLATE = "{task}_qv_lmhead_leftpad_llama2_7b"
OUTPUT_DIR_TEMPLATE = "./iid_llama2_7b_{task}_lora_map_leftpad"
SLICE_OUT_DIR_TEMPLATE = "./slice_data/{run_tag}/kfac_balanced"

NUM_SLICES = 0
SLICE_PARTITION_SEED = 0

MAX_SEQ_LEN = 300
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

MAX_STEPS = 2_000
SAVE_EVERY = 1_000
EVAL_EVERY = 500
MAP_STEP_FOR_TABLE = 2_000

MICRO_BSZ = 4
GRAD_ACCUM = 2

EVAL_BSZ = 32
NUM_WORKERS = 0

USE_GRADIENT_CHECKPOINTING = False
FAST_BUT_NONDETERMINISTIC = True

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

FULL_ATTENTION_TARGET_MODULES = ["q_proj", "v_proj"]
LM_HEAD_TARGET_MODULES = ["lm_head"]
SOURCE_EVAL_SPLIT = "validation"

SEEDS = [0]
TOKENIZER_PADDING_SIDE = "left"

_TASK_ALIASES = {
    "wgs": "wgs",
    "wgm": "wgm",
    "arce": "arc-e",
    "arc-e": "arc-e",
    "arcc": "arc-c",
    "arc-c": "arc-c",
    "boolq": "boolq",
    "obqa": "obqa",
}

_BP_PROMPT_ARC = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


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


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if FAST_BUT_NONDETERMINISTIC and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()
    ece = torch.zeros(1, dtype=torch.float64)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)

    confidences = confidences.to(dtype=torch.float64).cpu()
    accuracies = accuracies.to(dtype=torch.float64).cpu()

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(acc_bin - conf_bin) * prop
    return float(ece.item())


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


def freeze_base_enable_lora(model: nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True


_LORA_ADAPTER_PLACEHOLDER = "__adapter__"
_LORA_ADAPTER_RE = re.compile(r"(\.lora_(?:A|B)\.)([^.]+)(\.)")


def _normalize_lora_key(key: str) -> str:
    return _LORA_ADAPTER_RE.sub(rf"\1{_LORA_ADAPTER_PLACEHOLDER}\3", key)


def _denormalize_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(f".{_LORA_ADAPTER_PLACEHOLDER}.", f".{adapter_name}.")


def get_lora_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(lora_state, strict=False)


def get_normalized_lora_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if "lora_" not in k or "lora_A_rho" in k:
            continue
        out[_normalize_lora_key(k)] = v.detach().cpu().clone()
    return out


def load_normalized_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor], adapter_name: str) -> None:
    mapped = {_denormalize_lora_key(k, adapter_name): v for k, v in lora_state.items()}
    model.load_state_dict(mapped, strict=False)


def sync_or_create_shared_lora_init(model: nn.Module, init_path: str, adapter_name: str) -> None:
    if os.path.exists(init_path):
        saved = torch.load(init_path, map_location="cpu")
        load_normalized_lora_state_dict(model, saved, adapter_name=adapter_name)
        print(f"[Init LoRA] loaded shared init from {init_path}")
    else:
        torch.save(get_normalized_lora_state_dict_cpu(model), init_path)
        print(f"[Init LoRA] saved shared init to {init_path}")


def assign_random_slice_ids(train_ds: Dataset, num_slices: int, seed: int) -> Dataset:
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    perm = np.random.default_rng(seed).permutation(len(train_ds))
    slice_ids = np.empty(len(train_ds), dtype=np.int32)
    for sid, idxs in enumerate(np.array_split(perm, num_slices)):
        slice_ids[idxs] = sid
    if "slice_id" in train_ds.column_names:
        train_ds = train_ds.remove_columns(["slice_id"])
    return train_ds.add_column("slice_id", slice_ids.tolist())


def save_kfac_balanced_dataset(train_ds: Dataset, slice_out_dir: str) -> None:
    os.makedirs(os.path.dirname(slice_out_dir), exist_ok=True)
    order = np.argsort(np.asarray(train_ds["slice_id"], dtype=np.int32)).tolist()
    ds_dict = DatasetDict({"train": train_ds.select(order)})
    ds_dict.save_to_disk(slice_out_dir)
    print(f"[Save] kfac_balanced slices -> {slice_out_dir}")


def compute_choice_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    amp_dtype: torch.dtype,
    choice_token_ids: torch.Tensor,
) -> torch.Tensor:
    device = input_ids.device
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[:, -1, choice_token_ids]
    return logits.float()


def _mask_invalid_choices(logits: torch.Tensor, num_choices: Sequence[int] | None) -> torch.Tensor:
    if num_choices is None:
        return logits
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=logits.device, dtype=torch.long)
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


def _print_split_summary(prefix: str, ds: Dataset) -> None:
    print(f"[{prefix}] total={len(ds)}")


def normalize_task_name(task_name: str) -> str:
    task = str(task_name).strip().lower()
    if task not in _TASK_ALIASES:
        raise ValueError(f"Unsupported task: {task_name}. Supported: {sorted(_TASK_ALIASES)}")
    return _TASK_ALIASES[task]


def build_run_tag(task_name: str) -> str:
    return RUN_TAG_TEMPLATE.format(task=task_name.replace("-", ""))


def build_output_dir(task_name: str) -> str:
    return OUTPUT_DIR_TEMPLATE.format(task=task_name.replace("-", ""))


def build_slice_out_dir(run_tag: str) -> str:
    return SLICE_OUT_DIR_TEMPLATE.format(run_tag=run_tag)


def load_train_val(task_name: str, num_slices: int) -> tuple[Dataset, Dataset]:
    train_raw, val_raw, _ = load_task_dataset(task_name)
    if int(num_slices) > 0:
        train_raw = assign_random_slice_ids(train_raw, int(num_slices), seed=SLICE_PARTITION_SEED)
    return train_raw, val_raw


def preprocess_arc_c_bayesian_peft(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    pad_to_max_length: bool = True,
) -> Dataset:
    keep_extra = [c for c in ds.column_names if c in ["slice_id"]]

    def _tokenize(prompts: List[str]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            prompts,
            padding=("max_length" if pad_to_max_length else False),
            truncation=True,
            max_length=max_len,
        )

    def _fn(batch: Dict) -> Dict:
        prompts, labels, num_choices = [], [], []
        for question, choices, answer_key in zip(
            batch["question"], batch["choices"], batch["answerKey"]
        ):
            try:
                choice_lines = "\n".join(
                    [f"{l}) {c}" for l, c in zip(choices["text"], choices["label"])]
                )
                prompts.append(_BP_PROMPT_ARC.format(question=question, choices=choice_lines))

                answer_text = str(answer_key).strip()
                class_alpha = ord(answer_text) - ord("A")
                if class_alpha >= 0:
                    cls = class_alpha
                else:
                    cls = int(answer_text) - 1

                labels.append(cls)
                num_choices.append(len(choices["label"]))
            except Exception:
                prompts.append("")
                labels.append(-1)
                num_choices.append(-1)

        enc = _tokenize(prompts)
        enc["labels"] = labels
        enc["num_choices"] = num_choices
        for key in keep_extra:
            enc[key] = batch[key]
        return enc

    ds2 = ds.map(_fn, batched=True).filter(
        lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= 5
    )
    keep_cols = ("input_ids", "attention_mask", "labels", "num_choices", *keep_extra)
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])


@torch.no_grad()
def eval_next_token(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    choice_token_ids: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total = 0
    total_correct = 0
    total_nll = 0.0
    all_probs = []
    all_labels = []
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        num_choices = batch.get("num_choices")
        bsz = input_ids.size(0)
        cand_logits = compute_choice_logits(
            model,
            input_ids,
            attention_mask,
            amp_dtype,
            choice_token_ids,
        )
        cand_logits = _mask_invalid_choices(cand_logits, num_choices)

        total_nll += float(loss_fct(cand_logits, labels).item())
        probs = torch.softmax(cand_logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        total_correct += int((pred == labels).sum().item())
        total += bsz
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    return {
        "nll": total_nll / max(total, 1),
        "acc": total_correct / max(total, 1),
        "ece": compute_ece(probs_all, labels_all, n_bins=10),
    }


def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    choice_token_ids: torch.Tensor,
    run_dir: str,
) -> Dict[str, torch.Tensor]:
    freeze_base_enable_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] total={total_params:,} trainable(LoRA)={trainable_params:,}")

    adamw_kwargs = dict(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    if device.type == "cuda":
        adamw_kwargs["fused"] = True

    optimizer = torch.optim.AdamW(**adamw_kwargs)

    warmup_steps = int(WARMUP_RATIO * MAX_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=MAX_STEPS,
    )

    lora_state_at_map_step: Dict[str, torch.Tensor] | None = None
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_cnt = 0
    seen = 0

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(GRAD_ACCUM):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            num_choices = batch.get("num_choices")
            bsz = input_ids.size(0)
            cand_logits = compute_choice_logits(
                model,
                input_ids,
                attention_mask,
                amp_dtype,
                choice_token_ids,
            )
            cand_logits = _mask_invalid_choices(cand_logits, num_choices)

            loss_sum = ce_sum(cand_logits, labels)
            loss = (loss_sum / bsz) / GRAD_ACCUM

            loss.backward()
            running_loss += float(loss.item() * GRAD_ACCUM)
            running_cnt += 1
            seen += bsz

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            avg = running_loss / max(running_cnt, 1)
            dt = time.time() - t0
            print(f"[Train] step={step:5d} avg_loss={avg:.4f} seen={seen} time={dt/60:.1f}m")

        if (step % SAVE_EVERY == 0) or (step == MAX_STEPS):
            ckpt_dir = os.path.join(run_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)

        if (step % EVAL_EVERY == 0) or (step == MAX_STEPS):
            m = eval_next_token(model, eval_loader, device, amp_dtype, choice_token_ids)
            print(f"[Eval] step={step} NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
            running_loss = 0.0
            running_cnt = 0

        if (step == MAP_STEP_FOR_TABLE) and (lora_state_at_map_step is None):
            lora_state_at_map_step = get_lora_state_dict_cpu(model)
            print(f"[MAP cached] step={MAP_STEP_FOR_TABLE} (LoRA-only state_dict cached)")

    if lora_state_at_map_step is None:
        raise RuntimeError("MAP LoRA state not cached.")
    return lora_state_at_map_step


def run_one(
    task_name: str,
    run_tag: str,
    output_dir: str,
    num_classes: int,
    seed: int,
    train_raw: Dataset,
    eval_raw: Dataset,
) -> Dict[str, float]:
    print("\n" + "=" * 90)
    print(f"[Run] dataset={run_tag} | task={task_name} | seed={seed}")
    print("=" * 90)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32

    print(
        f"[Closed-choice scoring] task={task_name} "
        f"left-padded {num_classes}-choice last-token classification"
    )
    print(f"[Source Eval Split] {SOURCE_EVAL_SPLIT}")
    print(
        f"[Batch Config] micro_bsz={MICRO_BSZ} grad_accum={GRAD_ACCUM} "
        f"effective_train_bsz={MICRO_BSZ * GRAD_ACCUM} eval_bsz={EVAL_BSZ}"
    )

    run_dir = os.path.join(output_dir, run_tag, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = TOKENIZER_PADDING_SIDE

    load_kwargs = dict(
        pretrained_model_name_or_path=BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
    )
    attn_impl_used = FALLBACK_ATTN_IMPLEMENTATION
    try:
        model = AutoModelForCausalLM.from_pretrained(
            **load_kwargs,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(device)
        attn_impl_used = ATTN_IMPLEMENTATION
    except Exception as exc:
        print(
            f"[Model] attn_implementation={ATTN_IMPLEMENTATION} unavailable, "
            f"falling back to {FALLBACK_ATTN_IMPLEMENTATION}: {exc}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            **load_kwargs,
            attn_implementation=FALLBACK_ATTN_IMPLEMENTATION,
        ).to(device)
    print(f"[Model] attn_implementation={attn_impl_used}")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if USE_GRADIENT_CHECKPOINTING:
        enable_gradient_checkpointing(model)

    if task_name == "arc-c":
        print("[Protocol] arc-c uses bayesian-peft-style prompt/choice formatting and up to 5 choices.")
        train_proc = preprocess_arc_c_bayesian_peft(
            train_raw, tokenizer, MAX_SEQ_LEN, pad_to_max_length=False
        )
        eval_proc = preprocess_arc_c_bayesian_peft(
            eval_raw, tokenizer, MAX_SEQ_LEN, pad_to_max_length=False
        )
    else:
        train_proc = preprocess_task(task_name, train_raw, tokenizer, MAX_SEQ_LEN, pad_to_max_length=False)
        eval_proc = preprocess_task(task_name, eval_raw, tokenizer, MAX_SEQ_LEN, pad_to_max_length=False)
    print(f"[Processed] train={len(train_proc)} eval={len(eval_proc)}")

    pin_memory = (device.type == "cuda")
    batch_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_proc,
        batch_size=MICRO_BSZ,
        shuffle=True,
        generator=train_generator,
        collate_fn=batch_collator,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(
        eval_proc,
        batch_size=EVAL_BSZ,
        shuffle=False,
        collate_fn=batch_collator,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )

    candidate_token_ids = get_choice_token_ids(tokenizer, device, num_classes)
    print(
        f"[Head] keeping full-vocab lm_head; selecting "
        f"{int(candidate_token_ids.numel())} choice logits dynamically"
    )
    m_base = eval_next_token(model, eval_loader, device, amp_dtype, candidate_token_ids)
    print(f"[Base Eval] seed={seed} | NLL={m_base['nll']:.4f} ACC={100*m_base['acc']:.2f}% ECE={100*m_base['ece']:.2f}%")

    target_modules = resolve_all_layer_target_modules(model)
    print(f"[PEFT] Resolved all-layer target modules: {len(target_modules)}")
    for name in target_modules:
        print(f"  - {name}")

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg).to(device)
    init_lora_path = os.path.join(run_dir, "init_lora.pt")
    sync_or_create_shared_lora_init(model, init_lora_path, adapter_name="default")
    if USE_GRADIENT_CHECKPOINTING:
        enable_gradient_checkpointing(model)

    print("[PEFT] Trainable parameters:")
    model.print_trainable_parameters()

    print(f"[Train] run_dir={run_dir}")
    with _StageTimer(f"TRAIN MAP on {run_tag}"):
        lora_state_map = train_map(
            model,
            train_loader,
            eval_loader,
            device,
            amp_dtype,
            candidate_token_ids,
            run_dir,
        )

    load_lora_state_dict(model, lora_state_map)
    map_dir = os.path.join(run_dir, f"map_step_{MAP_STEP_FOR_TABLE}")
    os.makedirs(map_dir, exist_ok=True)
    model.save_pretrained(map_dir)
    tokenizer.save_pretrained(map_dir)
    print(f"[Save] MAP adapter -> {map_dir}")

    m = eval_next_token(model, eval_loader, device, amp_dtype, candidate_token_ids)
    print(f"[Final] seed={seed} | NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA MAP with a full-vocab lm_head and dynamic choice-logit selection."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK_NAME,
        choices=sorted(_TASK_ALIASES),
        help="Task to train on. Aliases arce/arcc are accepted.",
    )
    parser.add_argument(
        "--num_slices",
        type=int,
        default=NUM_SLICES,
        help="If > 0, assign balanced random slice ids and export a kfac_balanced dataset for Seq-LoRA.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in SEEDS),
        help="Comma-separated random seeds.",
    )
    return parser.parse_args()


def parse_seeds(seed_text: str) -> List[int]:
    values = [part.strip() for part in str(seed_text).split(",") if part.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return [int(v) for v in values]


def main() -> None:
    args = parse_args()
    task_name = normalize_task_name(args.task)
    run_tag = build_run_tag(task_name)
    output_dir = build_output_dir(task_name)
    slice_out_dir = build_slice_out_dir(run_tag)
    num_classes = 5 if task_name == "arc-c" else get_task_num_classes(task_name)
    os.makedirs(output_dir, exist_ok=True)

    train_raw, eval_raw = load_train_val(task_name, args.num_slices)
    _print_split_summary("Train Raw", train_raw)
    _print_split_summary("Eval Raw", eval_raw)
    if int(args.num_slices) > 0:
        save_kfac_balanced_dataset(train_raw, slice_out_dir)
    else:
        print("[Slices] disabled (num_slices <= 0); skipping slice export.")

    summary = []
    for seed in parse_seeds(args.seeds):
        m = run_one(task_name, run_tag, output_dir, num_classes, seed, train_raw, eval_raw)
        summary.append((seed, m))

    print("\n" + "=" * 90)
    print(f"[Summary] dataset={run_tag}")
    for seed, m in summary:
        print(f"  seed={seed} | NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
