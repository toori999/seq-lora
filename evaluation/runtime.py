from __future__ import annotations

import argparse
import time
from typing import Dict

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_alloc_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return mem_gb(torch.cuda.max_memory_allocated())


def peak_reserved_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return mem_gb(torch.cuda.max_memory_reserved())


class StageTimer:
    def __init__(self, tag: str):
        self.tag = tag
        self.t0 = None

    def __enter__(self):
        reset_cuda_peak()
        cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        cuda_sync()
        dt = time.perf_counter() - self.t0
        print(f"[TIME] {self.tag}: {dt:.2f} sec ({dt/60:.2f} min)")
        print(f"[PEAK] {self.tag}: alloc={peak_alloc_gb():.2f} GB  reserved={peak_reserved_gb():.2f} GB")


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_posterior_stats_dtype(value: str) -> str:
    value_norm = str(value).strip().lower()
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
        "fp64": "float64",
        "double": "float64",
        "float64": "float64",
    }
    if value_norm not in aliases:
        raise argparse.ArgumentTypeError(
            f"Invalid posterior stats dtype: {value}. Expected float32 or float64."
        )
    return aliases[value_norm]


def torch_dtype_from_name(value: str) -> torch.dtype:
    value_norm = normalize_posterior_stats_dtype(value)
    if value_norm == "float32":
        return torch.float32
    if value_norm == "float64":
        return torch.float64
    raise AssertionError(f"Unhandled dtype name: {value_norm}")


def multiclass_brier_score(probs: Tensor, labels: Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


def multiclass_brier_sum(probs: Tensor, labels: Tensor) -> float:
    one_hot = F.one_hot(labels, num_classes=probs.size(-1)).to(dtype=probs.dtype)
    return float(((probs - one_hot) ** 2).sum(dim=-1).sum().item())


def tau_kl_in_window(metrics: Dict[str, float], *, kl_low: float, kl_high: float) -> bool:
    kl = float(metrics.get("kl_map_to_bayes", float("nan")))
    return kl_low <= kl <= kl_high


def tau_kl_window_distance(metrics: Dict[str, float], *, kl_low: float, kl_high: float) -> float:
    kl = float(metrics.get("kl_map_to_bayes", float("nan")))
    if kl_low <= kl <= kl_high:
        return 0.0
    return min(abs(kl - kl_low), abs(kl - kl_high))
