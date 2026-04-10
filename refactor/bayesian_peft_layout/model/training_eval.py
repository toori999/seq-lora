from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from refactor.seq_lora.peft_utils import get_transformer_and_lm_head


def compute_multiclass_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == labels).float()
    ece = torch.zeros(1, dtype=torch.float64)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)

    confidences = confidences.to(dtype=torch.float64).cpu()
    accuracies = accuracies.to(dtype=torch.float64).cpu()

    for idx in range(n_bins):
        lo = bin_boundaries[idx]
        hi = bin_boundaries[idx + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(acc_bin - conf_bin) * prop
    return float(ece.item())


def mask_invalid_choices(
    logits: torch.Tensor,
    num_choices: Sequence[int],
) -> torch.Tensor:
    num_choices_t = torch.tensor(
        [int(value) for value in num_choices],
        device=logits.device,
        dtype=torch.long,
    )
    if int(num_choices_t.min().item()) < 2 or int(num_choices_t.max().item()) > logits.size(-1):
        raise ValueError(
            f"num_choices must be in [2, {logits.size(-1)}], got "
            f"min={int(num_choices_t.min().item())} max={int(num_choices_t.max().item())}"
        )
    col_idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    invalid = col_idx >= num_choices_t.view(-1, 1)
    return logits.masked_fill(invalid, -1e9)


def compute_choice_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    device = input_ids.device
    transformer, lm_head = get_transformer_and_lm_head(model)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=(device.type == "cuda"),
    ):
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = lm_head(out.last_hidden_state[:, -1, :])
    return logits.float()


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
        batch_size = input_ids.size(0)
        logits = compute_choice_logits(model, input_ids, attention_mask, amp_dtype)
        logits = mask_invalid_choices(logits, batch["num_choices"])

        total_nll += float(loss_fct(logits, labels).item())
        probs = torch.softmax(logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        total_correct += int((pred == labels).sum().item())
        total += batch_size
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_all = torch.cat(all_probs, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    return {
        "nll": total_nll / max(total, 1),
        "acc": total_correct / max(total, 1),
        "ece": compute_multiclass_ece(probs_all, labels_all, n_bins=15),
    }


__all__ = [
    "compute_choice_logits",
    "compute_multiclass_ece",
    "eval_next_token",
    "mask_invalid_choices",
]
