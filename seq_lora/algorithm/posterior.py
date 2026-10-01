from __future__ import annotations

from typing import Dict, List
import argparse
import gc

import torch
from torch import Tensor

from seq_lora.algorithm.lgssm import kalman_filter
from seq_lora.algorithm.process_noise import (
    build_module_constant_q_scales,
    materialize_scalar_Q_list,
    report_module_constant_q_results,
    report_scalar_constant_q_results,
)
from seq_lora.algorithm.subspace import prepare_lgssm_observations


def move_subspace_info(
    subspace_info: Dict[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Tensor]:
    moved: Dict[str, Tensor] = {}
    for key, value in subspace_info.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def sample_lgssm_posterior_from_stats(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    T: int,
    device: torch.device,
    cpu_device: torch.device,
) -> Tensor:
    del device
    sample_seed = int(args.seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]
    use_module_constant_q = str(args.q_mode) == "module_constant"
    module_q_scales = (
        build_module_constant_q_scales(
            module_specs,
            module_R_lists,
            mu_global_list_raw,
            clip_min=float(args.module_q_clip_min),
            clip_max=float(args.module_q_clip_max),
            shrink_exponent=float(args.module_q_shrink_exponent),
        )
        if use_module_constant_q
        else {}
    )

    print(f"\n=== Kalman Filter Only (module-wise) ===")
    print(f"[Kalman] modules={len(module_specs)} L_total={sum(int(spec['L']) for spec in module_specs)}")
    print(f"\nDirectly sampling final posterior (t=T): S={args.mc_eval_samples}")

    x_sample_parts: List[Tensor] = []
    for spec in module_specs:
        name = spec["name"]
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        mu_module_list = [
            mu_t[offset : offset + Lm].to(device=cpu_device, dtype=torch.float64)
            for mu_t in mu_global_list
        ]
        H_obs_list, y_list = prepare_lgssm_observations(
            module_R_lists[name],
            mu_list=mu_module_list,
        )
        m1 = torch.zeros(Lm, device=cpu_device, dtype=torch.float64)
        P1 = float(args.p1_var) * torch.eye(Lm, device=cpu_device, dtype=torch.float64)
        if use_module_constant_q:
            module_scale = float(module_q_scales.get(name, 1.0))
            q_var_list = [float(args.s_q) * module_scale for _ in range(T)]
        else:
            q_var_list = [float(args.s_q) for _ in range(T)]
        Q_list = materialize_scalar_Q_list(
            q_var_list,
            L=Lm,
            device=cpu_device,
            dtype=torch.float64,
        )

        x_filt_m, P_filt_m, _, _ = kalman_filter(
            H_list=H_obs_list,
            y_list=y_list,
            Q_list=Q_list,
            m1=m1,
            P1=P1,
        )
        mu_T_m, cov_T_m = x_filt_m[-1], P_filt_m[-1]
        cov_T_stable = cov_T_m + torch.eye(
            cov_T_m.shape[0],
            device=cov_T_m.device,
            dtype=cov_T_m.dtype,
        ) * 1e-6

        dist_m = torch.distributions.MultivariateNormal(
            mu_T_m,
            covariance_matrix=cov_T_stable,
        )
        x_sample_parts.append(dist_m.sample((int(args.mc_eval_samples),)))

        del (
            H_obs_list,
            y_list,
            Q_list,
            x_filt_m,
            P_filt_m,
            mu_T_m,
            cov_T_m,
            cov_T_stable,
            dist_m,
        )
        gc.collect()

    if use_module_constant_q:
        report_module_constant_q_results(
            module_q_scales,
            s_q=float(args.s_q),
            shrink_exponent=float(args.module_q_shrink_exponent),
        )
    else:
        report_scalar_constant_q_results(args.s_q)
    del mu_global_list
    return torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)


def sample_independent_slice_ensemble_from_stats(
    *,
    args: argparse.Namespace,
    module_specs: List[Dict],
    module_R_lists: Dict[str, List[Tensor]],
    mu_global_list_raw: List[Tensor],
    T: int,
    cpu_device: torch.device,
) -> Tensor:
    sample_seed = int(args.seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    samples_per_slice = max(int(args.independent_slice_mc_samples_per_slice), 1)
    total_samples = int(T) * samples_per_slice
    mu_global_list = [float(args.mu_obs_scale) * mu_t for mu_t in mu_global_list_raw]

    print("\n=== Independent Slice Ensemble Posterior ===", flush=True)
    print(
        f"[Independent Ensemble] modules={len(module_specs)} T={int(T)} "
        f"samples_per_slice={samples_per_slice} total_samples={total_samples} "
        f"prior=P1={float(args.p1_var):.6g} no_Kalman no_Q",
        flush=True,
    )

    x_sample_parts: List[Tensor] = []
    for spec in module_specs:
        name = spec["name"]
        offset = int(spec["offset"])
        Lm = int(spec["L"])
        mu_module_list = [
            mu_t[offset : offset + Lm].to(device=cpu_device, dtype=torch.float64)
            for mu_t in mu_global_list
        ]
        H_obs_list, y_list = prepare_lgssm_observations(
            module_R_lists[name],
            mu_list=mu_module_list,
        )

        eye = torch.eye(Lm, device=cpu_device, dtype=torch.float64)
        prior_precision = (1.0 / max(float(args.p1_var), 1e-12)) * eye
        module_samples: List[Tensor] = []
        for t in range(int(T)):
            R_t = H_obs_list[t].to(device=cpu_device, dtype=torch.float64)
            y_t = y_list[t].to(device=cpu_device, dtype=torch.float64)
            precision = prior_precision + R_t.T @ R_t
            precision = 0.5 * (precision + precision.T) + 1e-6 * eye
            rhs = R_t.T @ y_t
            chol = torch.linalg.cholesky(precision)
            mean_t = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
            cov_t = torch.cholesky_inverse(chol)
            cov_t = 0.5 * (cov_t + cov_t.T) + 1e-6 * eye
            dist_t = torch.distributions.MultivariateNormal(
                mean_t,
                covariance_matrix=cov_t,
            )
            module_samples.append(dist_t.sample((samples_per_slice,)))
            del R_t, y_t, precision, rhs, chol, mean_t, cov_t, dist_t

        x_sample_parts.append(torch.cat(module_samples, dim=0))
        del H_obs_list, y_list, module_samples
        gc.collect()

    del mu_global_list
    return torch.cat(x_sample_parts, dim=1).to(dtype=torch.float32)

