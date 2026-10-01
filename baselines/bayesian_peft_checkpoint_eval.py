from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
BAYESIAN_PEFT_ROOT = REPO_ROOT / "baselines" / "bayesian-peft"
sys.path.insert(0, str(BAYESIAN_PEFT_ROOT))

from dataset.utils import dsets  # noqa: E402


DATASET_ALIASES = {
    "obqa": "obqa",
    "arc-c": "ARC-Challenge",
    "arcc": "ARC-Challenge",
    "arc-challenge": "ARC-Challenge",
    "ARC-Challenge": "ARC-Challenge",
    "arc-e": "ARC-Easy",
    "arce": "ARC-Easy",
    "arc-easy": "ARC-Easy",
    "ARC-Easy": "ARC-Easy",
    "mmlu-chem": "MMLU-chem",
    "MMLU-chem": "MMLU-chem",
    "mmlu-phy": "MMLU-phy",
    "MMLU-phy": "MMLU-phy",
}


def normalize_dataset(name: str) -> str:
    if name not in DATASET_ALIASES:
        key = str(name).strip()
        key_lower = key.lower()
        if key_lower in DATASET_ALIASES:
            return DATASET_ALIASES[key_lower]
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_ALIASES[name]


def default_eval_split(dataset: str) -> str:
    dataset = normalize_dataset(dataset)
    if dataset.startswith("ARC") or dataset == "obqa" or dataset.startswith("MMLU"):
        return "test"
    raise ValueError(f"No default eval split for dataset={dataset}")


def load_adapter_state(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        state = load_safetensors_file(safetensors_path)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Missing adapter_model.safetensors/bin under {adapter_dir}")

    old_prefix = "base_model.model.base_model.model."
    new_prefix = "base_model.model."
    return {
        (new_prefix + key[len(old_prefix) :] if key.startswith(old_prefix) else key): value
        for key, value in state.items()
    }


def build_bayesian_peft_dataset(dataset: str, tokenizer, add_space: bool, max_seq_len: int):
    dataset = normalize_dataset(dataset)
    if dataset.startswith("ARC"):
        return dsets.arc(tokenizer, add_space=add_space, name=dataset, max_seq_len=max_seq_len)
    if dataset.startswith("MMLU"):
        return dsets.mmlu(tokenizer, add_space=add_space, name=dataset[5:], max_seq_len=max_seq_len)
    return getattr(dsets, dataset)(tokenizer, add_space=add_space, max_seq_len=max_seq_len)


def make_loader(dataset_name: str, model_name: str, split: str, batch_size: int, add_space: bool, max_seq_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.bos_token
    bp_dataset = build_bayesian_peft_dataset(dataset_name, tokenizer, add_space, max_seq_len)
    loader = bp_dataset.loader(
        is_s2s=False,
        batch_size=batch_size,
        split=split,
        subset_size=-1,
        drop_last=False,
    )
    return tokenizer, bp_dataset, loader


def load_model_with_adapters(
    model_name: str,
    adapter_dirs: Sequence[str],
    device: torch.device,
    amp_dtype: torch.dtype,
    load_in_8bit: bool,
):
    if not adapter_dirs:
        raise ValueError("Expected at least one adapter dir")
    peft_cfg = PeftConfig.from_pretrained(adapter_dirs[0])
    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": True,
    }
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else "cpu"}
    else:
        load_kwargs["dtype"] = amp_dtype if device.type == "cuda" else None

    base_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if not load_in_8bit:
        base_model = base_model.to(device)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model = get_peft_model(base_model, peft_cfg, adapter_name="adapter_0")
    adapter_names = []
    for idx, adapter_dir in enumerate(adapter_dirs):
        adapter_name = f"adapter_{idx}"
        if idx > 0:
            model.add_adapter(adapter_name, peft_cfg)
        state = load_adapter_state(adapter_dir)
        incompat = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
        current_adapter_marker = f".{adapter_name}."
        missing_lora = [k for k in incompat.missing_keys if "lora_" in k and current_adapter_marker in k]
        unexpected_lora = [k for k in incompat.unexpected_keys if "lora_" in k]
        if missing_lora or unexpected_lora:
            raise RuntimeError(
                f"LoRA load mismatch for {adapter_dir}: "
                f"missing_lora={missing_lora[:8]} unexpected_lora={unexpected_lora[:8]}"
            )
        adapter_names.append(adapter_name)
    model.eval()
    return model, adapter_names


def ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int) -> float:
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    ece = torch.zeros((), device=probs.device)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    for idx in range(n_bins):
        lo, hi = boundaries[idx], boundaries[idx + 1]
        in_bin = confidences.gt(lo) & confidences.le(hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            ece = ece + (confidences[in_bin].mean() - accuracies[in_bin].mean()).abs() * prop
    return float(ece.item())


def batch_to_device(batch, device: torch.device):
    prompts, labels, _ = batch
    return prompts.to(device), labels.to(device)


@torch.inference_mode()
def evaluate_map(model, loader, target_ids: torch.Tensor, device: torch.device):
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    for batch in loader:
        inputs, labels = batch_to_device(batch, device)
        logits = model(**inputs).logits[:, -1, target_ids]
        probs = torch.softmax(logits.float(), dim=-1)
        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(probs_all, dim=0), torch.cat(labels_all, dim=0)


@torch.inference_mode()
def evaluate_mcdrop(model, loader, target_ids: torch.Tensor, device: torch.device, mc_samples: int):
    dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    for batch in loader:
        inputs, labels = batch_to_device(batch, device)
        logits_list = []
        old_training = dropouts[0].training if dropouts else False
        for dropout in dropouts:
            dropout.training = True
        for _ in range(int(mc_samples)):
            logits = model(**inputs).logits[:, -1, target_ids]
            logits_list.append(logits.float())
        for dropout in dropouts:
            dropout.training = old_training
        probs = torch.softmax(torch.stack(logits_list, dim=1), dim=-1).mean(dim=1)
        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(probs_all, dim=0), torch.cat(labels_all, dim=0)


@torch.inference_mode()
def evaluate_ensemble(
    model,
    adapter_names: Sequence[str],
    loader,
    target_ids: torch.Tensor,
    device: torch.device,
):
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    for batch in loader:
        inputs, labels = batch_to_device(batch, device)
        logits_list = []
        for adapter_name in adapter_names:
            model.set_adapter(adapter_name)
            logits = model(**inputs).logits[:, -1, target_ids]
            logits_list.append(logits.float())
        logits = torch.stack(logits_list, dim=1)
        probs = torch.softmax(logits, dim=-1).mean(dim=1)
        probs_all.append(probs.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return torch.cat(probs_all, dim=0), torch.cat(labels_all, dim=0)


def metrics_from_probs(probs: torch.Tensor, labels: torch.Tensor, n_bins: int) -> Dict[str, float]:
    labels = labels.long()
    pred = probs.argmax(dim=-1)
    nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels, reduction="mean")
    brier = (probs - F.one_hot(labels, num_classes=probs.size(-1))).pow(2).sum(dim=-1).mean()
    return {
        "acc": float(pred.eq(labels).float().mean().item()),
        "nll": float(nll.item()),
        "ece": ece_score(probs, labels, n_bins),
        "brier": float(brier.item()),
    }


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(
        f"{prefix} "
        f"NLL={metrics['nll']:.4f} ACC={metrics['acc']*100:.2f}% "
        f"ECE={metrics['ece']*100:.2f}% Brier={metrics['brier']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bayesian-peft LoRA checkpoints on bayesian-peft datasets.")
    parser.add_argument("--method", choices=["map", "mcdrop", "ens"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter_dir", action="append", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--eval_split", default="auto", choices=["auto", "validation", "test"])
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--mc_samples", type=int, default=10)
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = normalize_dataset(args.dataset)
    eval_split = default_eval_split(dataset) if args.eval_split == "auto" else args.eval_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    adapter_dirs = [str(Path(path)) for path in args.adapter_dir]
    if args.method in {"map", "mcdrop"} and len(adapter_dirs) != 1:
        raise ValueError(f"{args.method} expects exactly one --adapter_dir")
    if args.method == "ens" and len(adapter_dirs) < 2:
        raise ValueError("ens expects at least two --adapter_dir values")

    print(f"[Config] method={args.method} dataset={dataset} split={eval_split} adapters={len(adapter_dirs)}")
    print(f"[Config] model={args.model} device={device} amp_dtype={amp_dtype} load_in_8bit={args.load_in_8bit}")
    for adapter_dir in adapter_dirs:
        print(f"[Adapter] {adapter_dir}")

    tokenizer, bp_dataset, loader = make_loader(
        dataset,
        args.model,
        eval_split,
        args.eval_bsz,
        args.add_space,
        args.max_seq_len,
    )
    del tokenizer
    target_ids = bp_dataset.target_ids.squeeze(-1).to(device=device, dtype=torch.long)
    model, adapter_names = load_model_with_adapters(args.model, adapter_dirs, device, amp_dtype, args.load_in_8bit)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.method == "map":
        probs, labels = evaluate_map(model, loader, target_ids, device)
    elif args.method == "mcdrop":
        probs, labels = evaluate_mcdrop(model, loader, target_ids, device, args.mc_samples)
    else:
        probs, labels = evaluate_ensemble(model, adapter_names, loader, target_ids, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    metrics = metrics_from_probs(probs, labels, args.num_bins)
    print(f"[TIME] INFER {args.method} on {dataset}({eval_split}): {dt:.2f} sec ({dt/60:.2f} min)")
    print_metrics(f"[{dataset}({eval_split})][{args.method.upper()}]", metrics)


if __name__ == "__main__":
    main()
