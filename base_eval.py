from __future__ import annotations

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_eval_utils import (
    SCIENCEQA_CURRIC_TASK_NAME,
    get_choice_token_ids,
    get_task_num_classes,
    load_eval_dataset,
    load_task_dataset,
    preprocess_task,
)
from map_eval import (
    EVAL_BSZ,
    MAX_SEQ_LEN,
    SEED,
    TRUST_REMOTE_CODE,
    _StageTimer,
    _add_seq_len,
    _make_eval_loader,
    _parse_eval_tasks,
    eval_map_one_dataset,
    trim_lm_head_to_choice_tokens,
)


def _load_base_model(
    task: str,
    base_model_name: str,
    amp_dtype: torch.dtype,
    device: torch.device,
):
    print(f"[Load] base_model = {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    tokenizer.padding_side = "left"

    num_classes = get_task_num_classes(task)
    choice_token_ids = get_choice_token_ids(tokenizer, device, num_classes)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        torch_dtype=(amp_dtype if device.type == "cuda" else None),
        attn_implementation="sdpa",
    ).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    trim_lm_head_to_choice_tokens(model, choice_token_ids)
    print(f"[Head] trimmed lm_head to {num_classes} choice logits")

    model.eval()
    return tokenizer, model, num_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base-model-only evaluation on IID/OOD tasks.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["wgs", "wgm", "arc-c", "arc-e", "obqa", "boolq", "sciq", SCIENCEQA_CURRIC_TASK_NAME],
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--eval_bsz", type=int, default=EVAL_BSZ)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print("Using device:", device, "amp_dtype:", amp_dtype)

    with _StageTimer(f"LOAD-STAGE BASE on {args.task}"):
        tokenizer, model, num_classes = _load_base_model(
            task=args.task,
            base_model_name=args.base_model,
            amp_dtype=amp_dtype,
            device=device,
        )

    eval_tasks = _parse_eval_tasks(args.eval_tasks, args.task)
    print("\n========================")
    print("       BASE ONLY        ")
    print("========================")

    for eval_task in eval_tasks:
        eval_num_classes = get_task_num_classes(eval_task)
        if eval_num_classes != num_classes:
            raise ValueError(
                f"Eval task '{eval_task}' has {eval_num_classes} classes, but source task '{args.task}' has {num_classes} classes."
            )

        if eval_task == args.task:
            _, _, eval_raw = load_task_dataset(eval_task)
            split_name = "test"
        else:
            eval_raw = load_eval_dataset(eval_task)
            split_name = "ood"

        eval_proc = _add_seq_len(preprocess_task(eval_task, eval_raw, tokenizer, args.max_seq_len, pad_to_max_length=False))
        loader = _make_eval_loader(eval_proc, tokenizer, device, bsz=args.eval_bsz)
        with _StageTimer(f"INFER BASE on {eval_task}({split_name})"):
            m = eval_map_one_dataset(model, loader, device, amp_dtype)

        print(f"\n[{eval_task}({split_name})][BASE]")
        print(
            f"  NLL={m['nll']:.4f}  ACC={m['acc']*100:.2f}%  "
            f"ECE={m['ece']*100:.2f}%  Brier={m['brier']:.4f}"
        )


if __name__ == "__main__":
    main()
