from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..metrics import make_accuracy, make_ece
from ..peft_utils import get_active_adapter_name, get_transformer_and_lm_head
from .adapter_loading import peft_set_adapter
from .common import mask_invalid_choices


@dataclass
class EvalMetrics:
    nll: float
    acc: float
    ece: float
    brier: float
    std: Optional[float] = None
    mc_samples: Optional[int] = None
    n_models: Optional[int] = None

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "nll": self.nll,
            "acc": self.acc,
            "ece": self.ece,
            "brier": self.brier,
        }
        if self.std is not None:
            out["std"] = self.std
        if self.mc_samples is not None:
            out["mc_samples"] = float(self.mc_samples)
        if self.n_models is not None:
            out["n_models"] = float(self.n_models)
        return out


def multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    one_hot = torch.nn.functional.one_hot(
        labels,
        num_classes=probs.size(-1),
    ).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> EvalMetrics:
    device = probs.device
    acc_metric = make_accuracy(device, num_classes)
    ece_metric = make_ece(device, num_classes, 15)
    acc_metric.reset()
    ece_metric.reset()
    acc_metric.update(probs, labels)
    ece_metric.update(probs, labels)
    eps = 1e-12
    nll = (
        float(
            -torch.log(probs[torch.arange(labels.numel(), device=device), labels].clamp_min(eps))
            .mean()
            .item()
        )
        if labels.numel() > 0
        else float("nan")
    )
    return EvalMetrics(
        nll=nll,
        acc=float(acc_metric.compute().item()),
        ece=float(ece_metric.compute().item()),
        brier=(multiclass_brier_score(probs, labels) if labels.numel() > 0 else float("nan")),
    )


@torch.inference_mode()
def predict_map_probabilities(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transformer, lm_head = get_transformer_and_lm_head(model)
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

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
            logits = lm_head(out.last_hidden_state[:, -1, :]).float()
        logits = mask_invalid_choices(logits, batch.get("num_choices"))
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.detach())
        all_labels.append(labels.detach())

    if all_probs:
        return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
    return (
        torch.empty((0, lm_head.out_features), device=device, dtype=torch.float32),
        torch.empty((0,), device=device, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate_map_dataset(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> EvalMetrics:
    probs, labels = predict_map_probabilities(model, loader, device, amp_dtype)
    return metrics_from_probs(probs, labels, num_classes=probs.size(-1))


@torch.inference_mode()
def evaluate_mc_dropout_dataset(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    mc_samples: int,
    temp: float = 1.0,
) -> EvalMetrics:
    model.eval()
    transformer, lm_head = get_transformer_and_lm_head(model)
    num_classes = lm_head.out_features
    acc_metric = make_accuracy(device, num_classes)
    ece_metric = make_ece(device, num_classes, 15)
    acc_metric.reset()
    ece_metric.reset()

    total = 0
    nll_sum = 0.0
    brier_sum = 0.0
    std_sum = 0.0
    eps = 1e-12
    inv_temp = 1.0 / float(temp) if float(temp) != 1.0 else 1.0
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        batch_size = labels.size(0)
        logits_list = []

        old_training = dropouts[0].training if dropouts else False
        for dropout in dropouts:
            dropout.training = True
        for _ in range(int(mc_samples)):
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
                logits = lm_head(out.last_hidden_state[:, -1, :]).float()
            logits = mask_invalid_choices(logits, batch.get("num_choices"))
            if inv_temp != 1.0:
                logits = logits * inv_temp
            logits_list.append(logits)
        for dropout in dropouts:
            dropout.training = old_training

        logits = torch.stack(logits_list, dim=1)
        probs = torch.softmax(logits, dim=-1).mean(dim=1)
        std_sum += float(torch.softmax(logits, dim=-1).std(dim=1).mean().item()) * batch_size
        p_y = probs[torch.arange(batch_size, device=device), labels].clamp_min(eps)
        nll_sum += float((-torch.log(p_y)).sum().item())
        acc_metric.update(probs, labels)
        ece_metric.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )
        total += batch_size

    model.eval()
    return EvalMetrics(
        nll=nll_sum / max(total, 1),
        acc=float(acc_metric.compute().item()),
        ece=float(ece_metric.compute().item()),
        brier=brier_sum / max(total, 1),
        std=std_sum / max(total, 1),
        mc_samples=int(mc_samples),
    )


@torch.inference_mode()
def evaluate_deep_ensemble_dataset(
    model,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    temp: float = 1.0,
) -> EvalMetrics:
    model.eval()
    transformer, lm_head = get_transformer_and_lm_head(model)
    num_classes = lm_head.out_features
    acc_metric = make_accuracy(device, num_classes)
    ece_metric = make_ece(device, num_classes, 15)
    acc_metric.reset()
    ece_metric.reset()

    eps = 1e-12
    inv_temp = 1.0 / float(temp) if float(temp) != 1.0 else 1.0
    total = 0
    nll_sum = 0.0
    brier_sum = 0.0
    std_sum = 0.0

    active0 = get_active_adapter_name(model)
    adapter_names = list(model.peft_config.keys())

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        batch_size = labels.size(0)
        total += batch_size

        logits_list = []
        for name in adapter_names:
            peft_set_adapter(model, name)
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
                logits = lm_head(out.last_hidden_state[:, -1, :]).float()
            logits = mask_invalid_choices(logits, batch.get("num_choices"))
            if inv_temp != 1.0:
                logits = logits * inv_temp
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=1)
        probs = torch.softmax(logits.mean(dim=1), dim=-1)
        std_sum += float(logits.std(dim=1).mean().item()) * batch_size
        p_y = probs[torch.arange(batch_size, device=device), labels].clamp_min(eps)
        nll_sum += float((-torch.log(p_y)).sum().item())
        acc_metric.update(probs, labels)
        ece_metric.update(probs, labels)
        brier_sum += float(
            (probs - torch.nn.functional.one_hot(labels, num_classes=num_classes))
            .pow(2)
            .sum(dim=-1)
            .sum()
            .item()
        )

    peft_set_adapter(model, active0)
    return EvalMetrics(
        nll=nll_sum / max(total, 1),
        acc=float(acc_metric.compute().item()),
        ece=float(ece_metric.compute().item()),
        brier=brier_sum / max(total, 1),
        std=std_sum / max(total, 1),
        n_models=len(adapter_names),
    )


def evaluate_probability_ensemble(
    probs_sum: torch.Tensor,
    labels: torch.Tensor,
    n_members: int,
) -> EvalMetrics:
    if n_members <= 0:
        raise ValueError("n_members must be > 0")
    probs = probs_sum / float(n_members)
    metrics = metrics_from_probs(probs, labels, num_classes=probs.size(-1))
    metrics.n_models = int(n_members)
    return metrics


__all__ = [
    "EvalMetrics",
    "evaluate_deep_ensemble_dataset",
    "evaluate_map_dataset",
    "evaluate_mc_dropout_dataset",
    "evaluate_probability_ensemble",
    "metrics_from_probs",
    "multiclass_brier_score",
    "predict_map_probabilities",
]
