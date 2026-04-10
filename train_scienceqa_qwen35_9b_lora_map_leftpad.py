from __future__ import annotations

from typing import Dict, List, Sequence
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

from common_eval_utils import (
    DynamicEvalCollator,
    answer_key_to_index,
    get_choice_labels,
    get_choice_token_ids,
    get_transformer_and_lm_head,
    make_prompt_from_choices,
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

BASE_MODEL_NAME = "Qwen/Qwen3-8B-Base"
DATASET_NAME = "tcallens/scienceqa-text-only"
TRUST_REMOTE_CODE = False
ATTN_IMPLEMENTATION = "sdpa"
FALLBACK_ATTN_IMPLEMENTATION = "sdpa"

RUN_TAG = "scienceqa_text_closedchoice_grade2_11_curriculum_qv_lmhead_leftpad"
OUTPUT_DIR = "./iid_qwen35_8b_scienceqa_lora_map_leftpad"
SLICE_OUT_DIR = f"./slice_data/{RUN_TAG}/kfac_balanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRADE_MIN = 2
GRADE_MAX = 11
TASK_FILTER = "closed choice"

MAX_SEQ_LEN = 300
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

MAX_STEPS = 2_000
SAVE_EVERY = 1_000
EVAL_EVERY = 100
MAP_STEP_FOR_TABLE = 2_000

MICRO_BSZ = 8
GRAD_ACCUM = 1

EVAL_BSZ = 32
NUM_WORKERS = 0

USE_GRADIENT_CHECKPOINTING = False
FAST_BUT_NONDETERMINISTIC = True

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

FULL_ATTENTION_TARGET_MODULES = ["q_proj", "v_proj"]
LM_HEAD_TARGET_MODULES = ["lm_head"]
MAX_CHOICES = 4
SOURCE_EVAL_SPLIT = "test"

SEEDS = [0]
TOKENIZER_PADDING_SIDE = "left"


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


def parse_grade_num(grade_value) -> int:
    text = str(grade_value).strip().lower()
    if text.startswith("grade"):
        return int(text.replace("grade", ""))
    raise ValueError(f"Unexpected grade format: {grade_value}")


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


def force_lora_fp32(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()


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


def save_kfac_balanced_dataset(train_ds: Dataset) -> None:
    os.makedirs(os.path.dirname(SLICE_OUT_DIR), exist_ok=True)
    order = np.argsort(np.asarray(train_ds["slice_id"], dtype=np.int32)).tolist()
    ds_dict = DatasetDict({"train": train_ds.select(order)})
    ds_dict.save_to_disk(SLICE_OUT_DIR)
    print(f"[Save] kfac_balanced slices -> {SLICE_OUT_DIR}")


def _coerce_choices(choices_obj) -> List[str]:
    if isinstance(choices_obj, np.ndarray):
        values = choices_obj.tolist()
    elif isinstance(choices_obj, (list, tuple)):
        values = list(choices_obj)
    else:
        values = []
    return [str(x).strip() for x in values if str(x).strip()]


def _mask_invalid_choices(cand_logits: torch.Tensor, num_choices: Sequence[int]) -> torch.Tensor:
    num_choices_t = torch.tensor([int(x) for x in num_choices], device=cand_logits.device, dtype=torch.long)
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > cand_logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {cand_logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(cand_logits.size(-1), device=cand_logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return cand_logits.masked_fill(invalid, -1e9)


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


def compute_choice_logits(
    model: nn.Module,
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
        logits = lm_head(out.last_hidden_state[:, -1, :])
    return logits.float()


def _print_grade_summary(prefix: str, ds: Dataset) -> None:
    grade_counts: Dict[int, int] = {}
    choice_counts: Dict[int, int] = {}
    for grade_num in ds["grade_num"]:
        g = int(grade_num)
        grade_counts[g] = grade_counts.get(g, 0) + 1
    for num_choices in ds["num_choices"]:
        k = int(num_choices)
        choice_counts[k] = choice_counts.get(k, 0) + 1

    print(f"[{prefix}] total={len(ds)}")
    for grade_num in sorted(grade_counts):
        print(f"  grade{grade_num}: {grade_counts[grade_num]}")
    print(
        f"[{prefix}] choice-counts="
        + ", ".join(f"{k}-choice={choice_counts[k]}" for k in sorted(choice_counts))
    )


def load_scienceqa_train_val() -> tuple[Dataset, Dataset]:
    ds = load_dataset(DATASET_NAME)
    train_raw = ds["train"]
    eval_raw = ds[SOURCE_EVAL_SPLIT]

    def _keep(ex: Dict) -> bool:
        try:
            grade_num = parse_grade_num(ex["grade"])
        except Exception:
            return False
        return (
            str(ex.get("task", "")).strip().lower() == TASK_FILTER
            and GRADE_MIN <= grade_num <= GRADE_MAX
        )

    def _add_meta(ex: Dict) -> Dict:
        grade_num = parse_grade_num(ex["grade"])
        return {
            "grade_num": grade_num,
            "slice_id": grade_num - GRADE_MIN,
            "num_choices": len(_coerce_choices(ex["choices"])),
        }

    train_raw = train_raw.filter(_keep).map(_add_meta)
    eval_raw = eval_raw.filter(_keep).map(_add_meta)
    return train_raw, eval_raw


def order_train_by_grade(train_raw: Dataset, seed: int) -> Dataset:
    parts: List[Dataset] = []
    for grade_num in range(GRADE_MIN, GRADE_MAX + 1):
        idxs = [i for i, g in enumerate(train_raw["grade_num"]) if int(g) == grade_num]
        if not idxs:
            continue
        ds_g = train_raw.select(idxs).shuffle(seed=seed + grade_num)
        parts.append(ds_g)
    if not parts:
        raise RuntimeError("No training examples left after grade filtering.")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def preprocess_scienceqa(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    keep_slice_id: bool = False,
) -> Dataset:
    keep_extra = [c for c in ["slice_id", "grade_num", "num_choices"] if keep_slice_id and c in ds.column_names]
    if not keep_slice_id and "num_choices" in ds.column_names:
        keep_extra = ["num_choices"]

    def _fn(batch: Dict) -> Dict:
        prompts: List[str] = []
        labels: List[int] = []
        valid_num_choices: List[int] = []

        for i in range(len(batch["question"])):
            try:
                choices = _coerce_choices(batch["choices"][i])
                k = len(choices)
                if k < 2 or k > MAX_CHOICES:
                    raise ValueError(f"unsupported num_choices={k}")
                label_order = get_choice_labels(k)
                mapping = {label_order[j]: choices[j] for j in range(k)}
                answer = answer_key_to_index(batch["answer"][i], label_order)
                prompt = make_prompt_from_choices(str(batch["question"][i]), mapping, label_order=label_order)
                prompts.append(prompt)
                labels.append(answer)
                valid_num_choices.append(k)
            except Exception:
                prompts.append("")
                labels.append(-1)
                valid_num_choices.append(-1)

        enc = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = labels
        enc["num_choices"] = valid_num_choices
        for k in keep_extra:
            if k != "num_choices":
                enc[k] = batch[k]
        return enc

    ds2 = ds.map(_fn, batched=True)
    ds2 = ds2.filter(lambda ex: ex["labels"] != -1 and 2 <= int(ex["num_choices"]) <= MAX_CHOICES)
    keep_cols = {"input_ids", "attention_mask", "labels", "num_choices"} | set(keep_extra)
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])


@torch.no_grad()
def eval_next_token(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
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
        bsz = input_ids.size(0)
        cand_logits = compute_choice_logits(model, input_ids, attention_mask, amp_dtype)
        cand_logits = _mask_invalid_choices(cand_logits, batch["num_choices"])

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
        "ece": compute_ece(probs_all, labels_all, n_bins=15),
    }


def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
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
            bsz = input_ids.size(0)
            cand_logits = compute_choice_logits(model, input_ids, attention_mask, amp_dtype)
            cand_logits = _mask_invalid_choices(cand_logits, batch["num_choices"])

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
            m = eval_next_token(
                model,
                eval_loader,
                device,
                amp_dtype,
            )
            print(f"[Eval] step={step} NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
            running_loss = 0.0
            running_cnt = 0

        if (step == MAP_STEP_FOR_TABLE) and (lora_state_at_map_step is None):
            lora_state_at_map_step = get_lora_state_dict_cpu(model)
            print(f"[MAP cached] step={MAP_STEP_FOR_TABLE} (LoRA-only state_dict cached)")

    if lora_state_at_map_step is None:
        raise RuntimeError("MAP LoRA state not cached.")
    return lora_state_at_map_step


def run_one(seed: int, train_raw: Dataset, eval_raw: Dataset) -> Dict[str, float]:
    print("\n" + "=" * 90)
    print(f"[Run] dataset={RUN_TAG} | seed={seed}")
    print("=" * 90)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    print("[ScienceQA scoring] left-padded 2/3/4-choice last-token classification over A-D with masking")
    print(f"[Source Eval Split] {SOURCE_EVAL_SPLIT}")
    print(
        f"[Batch Config] micro_bsz={MICRO_BSZ} grad_accum={GRAD_ACCUM} "
        f"effective_train_bsz={MICRO_BSZ * GRAD_ACCUM} eval_bsz={EVAL_BSZ}"
    )

    run_dir = os.path.join(OUTPUT_DIR, RUN_TAG, f"seed_{seed}")
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

    train_ordered = order_train_by_grade(train_raw, seed)
    train_proc = preprocess_scienceqa(train_ordered, tokenizer, MAX_SEQ_LEN, keep_slice_id=True)
    eval_proc = preprocess_scienceqa(eval_raw, tokenizer, MAX_SEQ_LEN, keep_slice_id=False)
    print(f"[Processed] train={len(train_proc)} eval={len(eval_proc)}")

    pin_memory = (device.type == "cuda")
    batch_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )

    train_loader = DataLoader(
        train_proc,
        batch_size=MICRO_BSZ,
        shuffle=False,
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

    candidate_token_ids = get_choice_token_ids(tokenizer, device, MAX_CHOICES)
    trim_lm_head_to_choice_tokens(model, candidate_token_ids)
    print(f"[Head] trimmed lm_head to {int(candidate_token_ids.numel())} choice logits")
    m_base = eval_next_token(model, eval_loader, device, amp_dtype)
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
    with _StageTimer(f"TRAIN MAP on {RUN_TAG}"):
        lora_state_map = train_map(model, train_loader, eval_loader, device, amp_dtype, run_dir)

    load_lora_state_dict(model, lora_state_map)
    map_dir = os.path.join(run_dir, f"map_step_{MAP_STEP_FOR_TABLE}")
    os.makedirs(map_dir, exist_ok=True)
    model.save_pretrained(map_dir)
    tokenizer.save_pretrained(map_dir)
    print(f"[Save] MAP adapter -> {map_dir}")

    m = eval_next_token(model, eval_loader, device, amp_dtype)
    print(f"[Final] seed={seed} | NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
    return m


def main() -> None:
    train_raw, eval_raw = load_scienceqa_train_val()
    _print_grade_summary("Train Raw", train_raw)
    _print_grade_summary("Eval Raw", eval_raw)
    save_kfac_balanced_dataset(train_raw)

    summary = []
    for seed in SEEDS:
        m = run_one(seed, train_raw, eval_raw)
        summary.append((seed, m))

    print("\n" + "=" * 90)
    print(f"[Summary] dataset={RUN_TAG}")
    for seed, m in summary:
        print(f"  seed={seed} | NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
