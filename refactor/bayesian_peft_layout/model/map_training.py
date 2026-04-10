from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from refactor.seq_lora.collators import DynamicEvalCollator
from refactor.seq_lora.metrics import get_choice_token_ids

from ..datasets import (
    load_scienceqa_train_eval_split,
    order_scienceqa_train,
    preprocess_scienceqa_closedchoice,
    print_scienceqa_split_summary,
    save_kfac_balanced_dataset,
)
from ..utils.train_config import ScienceQAMapTrainConfig
from .lora_training import (
    freeze_base_enable_lora,
    get_lora_state_dict_cpu,
    load_lora_state_dict,
    resolve_qv_lm_head_target_modules,
    sync_or_create_shared_lora_init,
    trim_lm_head_to_choice_tokens,
)
from .train_runtime import (
    build_training_tokenizer,
    enable_gradient_checkpointing,
    load_causal_lm_with_attn_fallback,
    resolve_device_amp_dtype,
    seed_worker,
    set_seed,
)
from .training_eval import compute_choice_logits, eval_next_token, mask_invalid_choices


def build_scienceqa_dataloaders(
    config: ScienceQAMapTrainConfig,
    tokenizer,
    device: torch.device,
):
    train_raw, eval_raw = load_scienceqa_train_eval_split(config.source_eval_split)
    print_scienceqa_split_summary("Train Raw", train_raw)
    print_scienceqa_split_summary("Eval Raw", eval_raw)
    save_kfac_balanced_dataset(train_raw, config.slice_dir)

    train_ordered = order_scienceqa_train(train_raw, config.variant, config.seed)
    train_proc = preprocess_scienceqa_closedchoice(
        train_ordered,
        tokenizer,
        config.max_seq_len,
        keep_slice_id=True,
        max_choices=config.max_choices,
    )
    eval_proc = preprocess_scienceqa_closedchoice(
        eval_raw,
        tokenizer,
        config.max_seq_len,
        keep_slice_id=False,
        max_choices=config.max_choices,
    )
    print(f"[Processed] train={len(train_proc)} eval={len(eval_proc)}")

    pin_memory = device.type == "cuda"
    batch_collator = DynamicEvalCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=(8 if device.type == "cuda" else None),
    )
    train_loader = DataLoader(
        train_proc,
        batch_size=config.micro_bsz,
        shuffle=False,
        collate_fn=batch_collator,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(config.num_workers > 0),
        worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(
        eval_proc,
        batch_size=config.eval_bsz,
        shuffle=False,
        collate_fn=batch_collator,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(config.num_workers > 0),
        worker_init_fn=seed_worker,
    )
    return train_loader, eval_loader


def train_map_lora(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    config: ScienceQAMapTrainConfig,
    run_dir: Path,
):
    freeze_base_enable_lora(model)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"[Params] total={total_params:,} trainable(LoRA)={trainable_params:,}")

    adamw_kwargs = dict(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(**adamw_kwargs)

    warmup_steps = int(config.warmup_ratio * config.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.max_steps,
    )

    lora_state_at_map_step = None
    ce_sum = nn.CrossEntropyLoss(reduction="sum")
    train_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_cnt = 0
    seen = 0

    for step in range(1, config.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(config.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            batch_size = input_ids.size(0)
            logits = compute_choice_logits(model, input_ids, attention_mask, amp_dtype)
            logits = mask_invalid_choices(logits, batch["num_choices"])

            loss_sum = ce_sum(logits, labels)
            loss = (loss_sum / batch_size) / config.grad_accum

            loss.backward()
            running_loss += float(loss.item() * config.grad_accum)
            running_cnt += 1
            seen += batch_size

        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            1.0,
        )
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            avg = running_loss / max(running_cnt, 1)
            dt = time.time() - t0
            print(f"[Train] step={step:5d} avg_loss={avg:.4f} seen={seen} time={dt/60:.1f}m")

        if (step % config.save_every == 0) or (step == config.max_steps):
            ckpt_dir = run_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))

        if (step % config.eval_every == 0) or (step == config.max_steps):
            metrics = eval_next_token(model, eval_loader, device, amp_dtype)
            print(
                f"[Eval] step={step} "
                f"NLL={metrics['nll']:.4f} ACC={100*metrics['acc']:.2f}% "
                f"ECE={100*metrics['ece']:.2f}%"
            )
            running_loss = 0.0
            running_cnt = 0

        if (step == config.map_step_for_table) and (lora_state_at_map_step is None):
            lora_state_at_map_step = get_lora_state_dict_cpu(model)
            print(f"[MAP cached] step={config.map_step_for_table} (LoRA-only state_dict cached)")

    if lora_state_at_map_step is None:
        raise RuntimeError("MAP LoRA state not cached.")
    return lora_state_at_map_step


def run_refactor_scienceqa_map_training(config: ScienceQAMapTrainConfig):
    print("\n" + "=" * 90)
    print(f"[Run] dataset={config.run_tag} | seed={config.seed}")
    print("=" * 90)

    set_seed(config.seed, fast_but_nondeterministic=config.fast_but_nondeterministic)
    device, amp_dtype = resolve_device_amp_dtype()
    print("Using device:", device)
    print("[ScienceQA scoring] left-padded 2/3/4-choice last-token classification over A-D with masking")
    print(f"[Source Eval Split] {config.source_eval_split}")
    print(
        f"[Batch Config] micro_bsz={config.micro_bsz} grad_accum={config.grad_accum} "
        f"effective_train_bsz={config.effective_train_bsz} eval_bsz={config.eval_bsz}"
    )

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_training_tokenizer(
        config.base_model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side=config.tokenizer_padding_side,
    )
    model, attn_impl_used = load_causal_lm_with_attn_fallback(
        base_model_name=config.base_model_name,
        device=device,
        amp_dtype=amp_dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
        fallback_attn_implementation=config.fallback_attn_implementation,
    )
    print(f"[Model] attn_implementation={attn_impl_used}")
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)

    train_loader, eval_loader = build_scienceqa_dataloaders(config, tokenizer, device)

    candidate_token_ids = get_choice_token_ids(tokenizer, device, config.max_choices)
    trim_lm_head_to_choice_tokens(model, candidate_token_ids)
    print(f"[Head] trimmed lm_head to {int(candidate_token_ids.numel())} choice logits")
    base_metrics = eval_next_token(model, eval_loader, device, amp_dtype)
    print(
        f"[Base Eval] seed={config.seed} | "
        f"NLL={base_metrics['nll']:.4f} ACC={100*base_metrics['acc']:.2f}% "
        f"ECE={100*base_metrics['ece']:.2f}%"
    )

    target_modules = resolve_qv_lm_head_target_modules(model)
    print(f"[PEFT] Resolved all-layer target modules: {len(target_modules)}")
    for name in target_modules:
        print(f"  - {name}")

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg).to(device)
    sync_or_create_shared_lora_init(model, config.init_lora_path, adapter_name="default")
    if config.use_gradient_checkpointing:
        enable_gradient_checkpointing(model)

    print("[PEFT] Trainable parameters:")
    model.print_trainable_parameters()
    print(f"[Train] run_dir={run_dir}")
    lora_state_map = train_map_lora(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        amp_dtype=amp_dtype,
        config=config,
        run_dir=run_dir,
    )

    load_lora_state_dict(model, lora_state_map)
    map_dir = config.map_dir
    map_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(map_dir))
    tokenizer.save_pretrained(str(map_dir))
    print(f"[Save] MAP adapter -> {map_dir}")

    final_metrics = eval_next_token(model, eval_loader, device, amp_dtype)
    print(
        f"[Final] seed={config.seed} | "
        f"NLL={final_metrics['nll']:.4f} ACC={100*final_metrics['acc']:.2f}% "
        f"ECE={100*final_metrics['ece']:.2f}%"
    )
    return final_metrics


__all__ = [
    "build_scienceqa_dataloaders",
    "run_refactor_scienceqa_map_training",
    "train_map_lora",
]
