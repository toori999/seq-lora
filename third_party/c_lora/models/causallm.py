import torch
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    PeftConfig,
)


def _default_torch_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else None


def _is_bitsandbytes_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "bitsandbytes",
        "bnb8bitquantize",
        "cuda setup error",
        "libnvjitlink",
        "automatic conversion of the weights",
        "int8_vectorwise_quant",
    )
    return any(marker in message for marker in markers)


def _load_causal_lm(model_name_or_path: str, load_in_8bit: bool):
    load_kwargs = {}
    if load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        torch_dtype = _default_torch_dtype()
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs), load_in_8bit
    except Exception as exc:
        if not load_in_8bit or not _is_bitsandbytes_runtime_error(exc):
            raise
        error_summary = next(
            (line.strip() for line in str(exc).splitlines() if line.strip()),
            type(exc).__name__,
        )
        print(
            "[WARN] 8-bit model loading failed; retrying without bitsandbytes quantization. "
            f"Original error: {error_summary}",
            flush=True,
        )
        fallback_kwargs = {}
        torch_dtype = _default_torch_dtype()
        if torch_dtype is not None:
            fallback_kwargs["torch_dtype"] = torch_dtype
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **fallback_kwargs), False


def _get_single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids2 = tokenizer.encode(" " + s, add_special_tokens=False)
    if len(ids2) == 1:
        return int(ids2[0])
    raise ValueError(f'"{s}" is not a single token: ids={ids}, ids_with_space={ids2}')


def _get_choice_token_ids(model_name: str, num_classes: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    choices = [chr(ord("A") + i) for i in range(int(num_classes))]
    ids = [_get_single_token_id(tokenizer, c) for c in choices]
    return torch.tensor(ids, dtype=torch.long)


def _get_base_and_lm_head(model: nn.Module):
    base = model
    if hasattr(base, "lm_head"):
        return base, base.lm_head
    if hasattr(base, "get_output_embeddings"):
        lm_head = base.get_output_embeddings()
        if lm_head is not None:
            return base, lm_head
    raise RuntimeError("Cannot locate lm_head for trimming.")


def _trim_lm_head_to_choice_tokens(model: nn.Module, choice_token_ids: torch.Tensor) -> None:
    base, lm_head = _get_base_and_lm_head(model)
    old_weight = lm_head.weight.detach()
    keep = choice_token_ids.to(device=old_weight.device, dtype=torch.long)
    new_out = int(keep.numel())
    new_head = nn.Linear(
        old_weight.size(1),
        new_out,
        bias=(getattr(lm_head, "bias", None) is not None),
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    with torch.no_grad():
        new_head.weight.copy_(old_weight.index_select(0, keep))
        if getattr(lm_head, "bias", None) is not None:
            new_head.bias.copy_(lm_head.bias.detach().index_select(0, keep))
    if hasattr(base, "set_output_embeddings"):
        base.set_output_embeddings(new_head)
    else:
        base.lm_head = new_head
    if hasattr(base, "config") and hasattr(base.config, "vocab_size"):
        base.config.vocab_size = new_out


class CausalLM(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if args.load_checkpoint:
            load_path = getattr(args, "load_path", None)
            if load_path is None:
                load_path = f"checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}"
                args.load_path = load_path
            print("Loading model from:", load_path)
            peft_config = PeftConfig.from_pretrained(load_path, is_trainable=True)
            model, args.load_in_8bit = _load_causal_lm(
                peft_config.base_model_name_or_path,
                load_in_8bit=args.load_in_8bit,
            )
            if str(args.dataset_type).strip().lower() == "benchmark_mcdataset":
                num_classes = int(getattr(args, "outdim", 0) or 0)
                if num_classes <= 0:
                    raise ValueError("benchmark_mcdataset requires args.outdim > 0 before model construction.")
                tokenizer_ref = peft_config.base_model_name_or_path
                choice_token_ids = _get_choice_token_ids(tokenizer_ref, num_classes)
                _trim_lm_head_to_choice_tokens(model, choice_token_ids)
                print(f"[Head] trimmed lm_head to {num_classes} choice logits")
        else:
            if args.load_model_path is not None:
                model, args.load_in_8bit = _load_causal_lm(
                    args.load_model_path,
                    load_in_8bit=args.load_in_8bit,
                )
            else:
                model, args.load_in_8bit = _load_causal_lm(
                    args.model,
                    load_in_8bit=args.load_in_8bit,
                )

            if str(args.dataset_type).strip().lower() == "benchmark_mcdataset":
                num_classes = int(getattr(args, "outdim", 0) or 0)
                if num_classes <= 0:
                    raise ValueError("benchmark_mcdataset requires args.outdim > 0 before model construction.")
                tokenizer_ref = args.load_model_path if args.load_model_path is not None else args.model
                choice_token_ids = _get_choice_token_ids(tokenizer_ref, num_classes)
                _trim_lm_head_to_choice_tokens(model, choice_token_ids)
                print(f"[Head] trimmed lm_head to {num_classes} choice logits")

            if args.apply_classhead_lora:
                target_modules = ["q_proj", "v_proj", "lm_head"]
            elif args.apply_qkv_head_lora:
                target_modules = ["q_proj", "v_proj", "k_proj", "lm_head"]
            else:
                target_modules = ["q_proj", "v_proj"]

            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )

        self.model = model
        self.peft_config = peft_config
