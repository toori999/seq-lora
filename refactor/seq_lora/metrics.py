from __future__ import annotations

import torch

from .prompts import get_choice_labels, get_single_token_id

_MulticlassAccuracy = None
_MulticlassCalibrationError = None

try:
    from torchmetrics import Accuracy, CalibrationError
except Exception:
    Accuracy = None
    CalibrationError = None

try:
    from torchmetrics.classification import (
        MulticlassAccuracy as _MulticlassAccuracy,
        MulticlassCalibrationError as _MulticlassCalibrationError,
    )
except Exception:
    _MulticlassAccuracy = None
    _MulticlassCalibrationError = None


def get_choice_token_ids(
    tokenizer, device: torch.device, num_classes: int
) -> torch.Tensor:
    choices = get_choice_labels(num_classes)
    ids = [get_single_token_id(tokenizer, choice) for choice in choices]
    print(f"[Choice token ids] classes={num_classes}, ids={dict(zip(choices, ids))}")
    return torch.tensor(ids, device=device, dtype=torch.long)


def make_accuracy(device: torch.device, num_classes: int):
    if Accuracy is not None:
        try:
            return Accuracy(task="multiclass", num_classes=num_classes).to(device)
        except Exception:
            pass
    if _MulticlassAccuracy is None:
        raise RuntimeError("No usable torchmetrics Accuracy implementation found.")
    return _MulticlassAccuracy(num_classes=num_classes).to(device)


def make_ece(device: torch.device, num_classes: int, n_bins: int = 15):
    if CalibrationError is not None:
        try:
            return CalibrationError(
                task="multiclass",
                num_classes=num_classes,
                n_bins=n_bins,
                norm="l1",
            ).to(device)
        except Exception:
            pass
    if _MulticlassCalibrationError is None:
        raise RuntimeError("No usable torchmetrics CalibrationError implementation found.")
    return _MulticlassCalibrationError(
        num_classes=num_classes,
        n_bins=n_bins,
        norm="l1",
    ).to(device)


__all__ = ["get_choice_token_ids", "make_accuracy", "make_ece"]
