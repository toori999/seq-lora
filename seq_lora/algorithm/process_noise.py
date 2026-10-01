from __future__ import annotations

from typing import Dict, List, Sequence
import math

import torch
from torch import Tensor


@torch.no_grad()
def materialize_scalar_Q_list(
    var_list: Sequence[float],
    L: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tensor]:
    eye = torch.eye(L, device=device, dtype=dtype)
    return [float(var) * eye for var in var_list]


def normalize_and_clip_scales(
    values: Sequence[float],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float = 1.0,
) -> List[float]:
    values = [float(v) for v in values]
    positive = [v for v in values if math.isfinite(v) and v > 0.0]
    if not positive:
        return [1.0 for _ in values]
    denom = float(torch.tensor(positive, dtype=torch.float64).median().item())
    if not math.isfinite(denom) or denom <= 0.0:
        denom = sum(positive) / max(len(positive), 1)
    if not math.isfinite(denom) or denom <= 0.0:
        return [1.0 for _ in values]
    beta = float(min(max(float(shrink_exponent), 0.0), 1.0))
    scales = []
    for v in values:
        if math.isfinite(v) and v > 0.0:
            ratio = max(v / denom, 1e-12)
            scale = math.exp(beta * math.log(ratio))
        else:
            scale = 1.0
        scales.append(scale)
    return [float(min(max(s, clip_min), clip_max)) for s in scales]


def median_positive_or_default(values: Tensor, default: float = 0.0) -> float:
    values = values.detach().reshape(-1).to(dtype=torch.float64)
    values = values[torch.isfinite(values) & (values > 0.0)]
    if int(values.numel()) == 0:
        return float(default)
    return float(values.median().item())


def build_module_constant_q_scales(
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    *,
    clip_min: float,
    clip_max: float,
    shrink_exponent: float,
) -> Dict[str, float]:
    if len(mu_global_list_raw) < 2:
        return {str(spec["name"]): 1.0 for spec in module_specs}

    raw_vals: List[float] = []
    names: List[str] = []
    for spec in module_specs:
        name = str(spec["name"])
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        R_list = module_R_lists.get(name, [])
        if not R_list:
            raw_vals.append(1.0)
            names.append(name)
            continue

        curvature_diag = torch.zeros(Lm, dtype=torch.float64)
        n_curv = 0
        for R_t in R_list:
            R_cpu = R_t.detach().to(device=curvature_diag.device, dtype=torch.float64)
            if tuple(R_cpu.shape) != (Lm, Lm):
                raise ValueError(
                    f"R_list for module {name} has shape {tuple(R_cpu.shape)}, expected {(Lm, Lm)}"
                )
            curvature_diag.add_(R_cpu.pow(2).sum(dim=0))
            n_curv += 1
        curvature_diag.div_(max(n_curv, 1)).clamp_min_(0.0)

        gap_vals: List[float] = []
        for t in range(1, len(mu_global_list_raw)):
            delta = (
                mu_global_list_raw[t][offset : offset + Lm].to(dtype=torch.float64)
                - mu_global_list_raw[t - 1][offset : offset + Lm].to(dtype=torch.float64)
            )
            coord_energy = delta.pow(2) * curvature_diag
            gap_vals.append(median_positive_or_default(coord_energy, default=0.0))

        raw_vals.append(
            median_positive_or_default(
                torch.tensor(gap_vals, dtype=torch.float64),
                default=1.0,
            )
        )
        names.append(name)

    scales = normalize_and_clip_scales(
        raw_vals,
        clip_min=clip_min,
        clip_max=clip_max,
        shrink_exponent=shrink_exponent,
    )
    return {name: scale for name, scale in zip(names, scales)}


def report_scalar_constant_q_results(s_q: float) -> None:
    print("\n=== Scalar-Constant Q Report ===")
    print(f"[Constant Q] mode=constant  shared Q_t = s_Q * I with s_Q={float(s_q):.6f}")


def report_module_constant_q_results(
    module_scales: Dict[str, float],
    *,
    s_q: float,
    shrink_exponent: float,
) -> None:
    if not module_scales:
        print("[Module Q] No module scales available.")
        return

    vals = [float(v) for v in module_scales.values()]
    print("\n=== Module-Constant Q Report ===")
    print(
        f"[Module Q] mode=module_constant  estimator=curvature_normalized_mu_drift_prior  "
        f"exposed scale s_Q={float(s_q):.6f}  "
        f"beta={float(min(max(float(shrink_exponent), 0.0), 1.0)):.3f}  "
        "drift_m=median_t median_l ((delta_mu_l)^2 diag(Hbar_m)_l)  Q_t^(m)=s_Q*q_m*I"
    )
    print(
        f"[Module Q Summary] scale(min={min(vals):.6f} mean={sum(vals)/len(vals):.6f} max={max(vals):.6f})"
    )
    print("[Module Q Per Module]")
    for name, scale in module_scales.items():
        print(f"  {name}: q_scale={float(scale):.6f} q_diag={float(s_q) * float(scale):.6f}")

