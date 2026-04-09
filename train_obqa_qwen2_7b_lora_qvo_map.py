from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from common_eval_utils import (
    answer_key_to_index,
    get_choice_labels,
    get_choice_token_ids,
    make_prompt_from_choices,
)

# ------------------------ Config (paper-aligned) ------------------------ #

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

BASE_MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
TRUST_REMOTE_CODE = False

OUTPUT_DIR = "./iid_qwen3p5_9b_lora_qv_map"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEQ_LEN = 300
LR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.06

MAX_STEPS = 2_000
SAVE_EVERY = 2_000
EVAL_EVERY = 2_000
MAP_STEP_FOR_TABLE = 2_000

MICRO_BSZ = 2
GRAD_ACCUM = 2

EVAL_BSZ = 16
NUM_WORKERS = 4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj",]

SEEDS = [0,1,2]

# ------------------------ Prompts (Table 5-like) ------------------------ #

PROMPT_WG = (
    "Select one of the choices that answer the following question: {question}\n"
    "Choices: A. {option1}. B. {option2}. Answer:"
)

PROMPT_BOOLQ = (
    "Select one of the choices that answer the following question:\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Choices: A. False. B. True. Answer:"
)

# 新增 4分类 Prompt
# ------------------------ Reproducibility ------------------------ #

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------ Utils: ECE ------------------------ #

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

# ------------------------ Generic eval: restricted K-way ------------------------ #

@torch.no_grad()
def eval_next_token(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    candidate_token_ids: torch.Tensor,
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

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits

        bsz = input_ids.size(0)
        last_idx = torch.full((bsz,), input_ids.size(1) - 1, device=device, dtype=torch.long)

        logits_last = logits[torch.arange(bsz, device=device), last_idx, :]
        cand_logits = logits_last.index_select(dim=-1, index=candidate_token_ids)

        nll_sum = loss_fct(cand_logits, labels)
        total_nll += float(nll_sum.item())

        probs = torch.softmax(cand_logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)

        total_correct += int((pred == labels).sum().item())
        total += bsz

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    ece = compute_ece(probs_all, labels_all, n_bins=15)
    return {"nll": total_nll / max(total, 1), "acc": total_correct / max(total, 1), "ece": ece}

# ------------------------ LoRA helpers ------------------------ #

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

def get_lora_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items() if "lora_" in k}

def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(lora_state, strict=False)

# =========================================================================
# Task-specific Formatting & Dispatchers 
# =========================================================================

def preprocess_wg(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        sents, o1, o2, ans = batch["sentence"], batch["option1"], batch["option2"], batch["answer"]
        prompts, labels = [], []
        for i in range(len(sents)):
            prompt = PROMPT_WG.format(question=sents[i], option1=o1[i], option2=o2[i])
            y = 0 if str(ans[i]) == "1" else 1
            prompts.append(prompt)
            labels.append(y)
        enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_len)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])

def preprocess_boolq(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        qs, ps, ans = batch["question"], batch["passage"], batch["label"] 
        prompts, labels = [], []
        for i in range(len(qs)):
            prompt = PROMPT_BOOLQ.format(question=qs[i], passage=ps[i])
            y = int(ans[i])
            prompts.append(prompt)
            labels.append(y)
        enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_len)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])

def preprocess_arc(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _get_q_and_choices(ex_question, ex_choices):
        if isinstance(ex_question, dict):
            return ex_question.get("stem", ""), ex_question.get("choices", ex_choices)
        return str(ex_question), ex_choices

    def _fn(batch: Dict) -> Dict:
        questions = batch["question"]
        choices_col = batch.get("choices", None)
        answer_keys = batch["answerKey"]
        prompts, labels = [], []

        for i in range(len(answer_keys)):
            try:
                ex_choices = choices_col[i] if choices_col is not None else None
                qtext, ch = _get_q_and_choices(questions[i], ex_choices)
                if ch is None: raise ValueError("choices is None")
                labs, txts = ch["label"], ch["text"]
                if len(labs) < 2 or len(txts) < 2: raise ValueError("choices < 2")

                label_order = [str(x) for x in labs]
                mapping = {str(lab): str(txt) for lab, txt in zip(labs, txts)}
                y = answer_key_to_index(answer_keys[i], label_order)
                prompt = make_prompt_from_choices(qtext, mapping, label_order=label_order)
                prompts.append(prompt)
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_len)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True).filter(lambda ex: ex["labels"] != -1)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])

def preprocess_obqa(ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    def _fn(batch: Dict) -> Dict:
        questions = batch["question_stem"]
        choices_list = batch["choices"]
        answer_keys = batch["answerKey"]
        prompts, labels = [], []

        for i in range(len(questions)):
            try:
                mapping = {str(lab): str(txt) for lab, txt in zip(choices_list[i]["label"], choices_list[i]["text"])}
                label_order = [str(lab) for lab in choices_list[i]["label"]]
                if len(label_order) < 2:
                    raise ValueError("Choice labels missing")

                y = answer_key_to_index(answer_keys[i], label_order)
                prompt = make_prompt_from_choices(questions[i], mapping, label_order=label_order)
                prompts.append(prompt)
                labels.append(y)
            except Exception:
                prompts.append("")
                labels.append(-1)

        enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_len)
        enc["labels"] = labels
        return enc

    ds2 = ds.map(_fn, batched=True).filter(lambda ex: ex["labels"] != -1)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    return ds2.remove_columns([c for c in ds2.column_names if c not in keep_cols])

# =========================================================================
# Dispatchers
# =========================================================================

def load_task_dataset(task: str) -> Tuple[Dataset, Dataset]:
    if task == "wgs": return load_dataset("winogrande", "winogrande_s")["train"], load_dataset("winogrande", "winogrande_s")["validation"]
    if task == "wgm": return load_dataset("winogrande", "winogrande_m")["train"], load_dataset("winogrande", "winogrande_m")["validation"]
    if task == "boolq": return load_dataset("super_glue", "boolq")["train"], load_dataset("super_glue", "boolq")["validation"]
    if task == "arce": return load_dataset("ai2_arc", "ARC-Easy")["train"], load_dataset("ai2_arc", "ARC-Easy")["validation"]
    if task == "arcc": return load_dataset("ai2_arc", "ARC-Challenge")["train"], load_dataset("ai2_arc", "ARC-Challenge")["validation"]
    if task == "obqa": return load_dataset("openbookqa", "main")["train"], load_dataset("openbookqa", "main")["validation"]
    raise ValueError(f"Unknown task: {task}")

def preprocess_task(task: str, ds: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
    if task in ["wgs", "wgm"]: return preprocess_wg(ds, tokenizer, max_len)
    elif task == "boolq": return preprocess_boolq(ds, tokenizer, max_len)
    elif task in ["arce", "arcc"]: return preprocess_arc(ds, tokenizer, max_len)
    elif task == "obqa": return preprocess_obqa(ds, tokenizer, max_len)
    else: raise ValueError(f"Unknown task: {task}")

def task_candidates(task: str) -> List[str]:
    # 动态返回 2 分类或 4 分类候选头
    if task in ["wgs", "wgm", "boolq"]:
        return get_choice_labels(2)
    elif task in ["arce", "arcc", "obqa"]:
        return get_choice_labels(4)
    raise ValueError(f"Unknown task: {task}")


# ------------------------ Train loop (LoRA MAP) ------------------------ #

def train_map(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    candidate_token_ids: torch.Tensor,
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
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=MAX_STEPS
    )

    lora_state_at_map_step: Optional[Dict[str, torch.Tensor]] = None
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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                last_idx = torch.full((bsz,), input_ids.size(1) - 1, device=device, dtype=torch.long)
                bsz = input_ids.size(0)
                logits_last = logits[torch.arange(bsz, device=device), last_idx, :]
                cand_logits = logits_last.index_select(dim=-1, index=candidate_token_ids)

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
            m = eval_next_token(model, eval_loader, device, amp_dtype, candidate_token_ids)
            print(f"[Eval] step={step} NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
            running_loss = 0.0
            running_cnt = 0

        if (step == MAP_STEP_FOR_TABLE) and (lora_state_at_map_step is None):
            lora_state_at_map_step = get_lora_state_dict_cpu(model)
            print(f"[MAP cached] step={MAP_STEP_FOR_TABLE} (LoRA-only state_dict cached)")

    if lora_state_at_map_step is None:
        raise RuntimeError("MAP LoRA state not cached.")
    return lora_state_at_map_step

# ------------------------ Run one (task, seed) ------------------------ #

def run_one(task: str, seed: int) -> Dict[str, float]:
    print("\n" + "=" * 90)
    print(f"[Run] task={task} | seed={seed}")
    print("=" * 90)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32

    run_dir = os.path.join(OUTPUT_DIR, task, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa"
    ).to(device)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg).to(device)

    force_lora_fp32(model)

    print("[PEFT] Trainable parameters:")
    model.print_trainable_parameters()

    candidates = task_candidates(task)
    cand_token_ids = get_choice_token_ids(tokenizer, device, len(candidates))

    train_raw, eval_raw = load_task_dataset(task)
    print(f"[Data] task={task} train={len(train_raw)} eval(val)={len(eval_raw)}")

    train_proc = preprocess_task(task, train_raw, tokenizer, MAX_SEQ_LEN)
    eval_proc = preprocess_task(task, eval_raw, tokenizer, MAX_SEQ_LEN)

    print(f"[Processed] task={task} train={len(train_proc)} eval={len(eval_proc)}")

    cols = ["input_ids", "attention_mask", "labels"]
    train_proc.set_format(type="torch", columns=cols)
    eval_proc.set_format(type="torch", columns=cols)

    pin_memory = (device.type == "cuda")
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_proc,
        batch_size=MICRO_BSZ,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )
    eval_loader = DataLoader(
        eval_proc,
        batch_size=EVAL_BSZ,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker,
    )

    print(f"[Train] run_dir={run_dir}")
    lora_state_map = train_map(model, train_loader, eval_loader, device, amp_dtype, cand_token_ids, run_dir)

    load_lora_state_dict(model, lora_state_map)
    map_dir = os.path.join(run_dir, f"map_step_{MAP_STEP_FOR_TABLE}")
    os.makedirs(map_dir, exist_ok=True)
    model.save_pretrained(map_dir)
    tokenizer.save_pretrained(map_dir)
    print(f"[Save] MAP adapter -> {map_dir}")

    m = eval_next_token(model, eval_loader, device, amp_dtype, cand_token_ids)
    print(f"[Final] task={task} seed={seed} | NLL={m['nll']:.4f} ACC={100*m['acc']:.2f}% ECE={100*m['ece']:.2f}%")
    return m

# ------------------------ main ------------------------ #

def main():
    # 动态选择要跑的任务，目前加入了所有6个任务，你可以随意注释/取消注释
    tasks = ["obqa", "wgs", "wgm", "boolq", "arce", "arcc"]

    summary = {}
    for task in tasks:
        summary[task] = []
        for seed in SEEDS:
            m = run_one(task, seed)
            summary[task].append((seed, m))

    print("\n" + "=" * 90)
    print("[Summary]")
    for task in tasks:
        print(f"\n== {task} ==")
        for seed, m in summary[task]:
            print(f"seed={seed} | NLL={m['nll']:.4f} | ACC={100*m['acc']:.2f}% | ECE={100*m['ece']:.2f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
