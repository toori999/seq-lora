# seq_lora_subspace.py
"""
Subspace construction for Seq-LoRA using Kronecker KFAC curvature.

Inputs (per layer, per slice):
    - H_list:   [T] of H_t ∈ R^{k × k}       (activation-side curvature)
    - G_B_list: [T] of G_{t,B} ∈ R^{r × r}  (LoRA-down-projected output curvature)

Goal:
    - Construct global Kronecker curvature approximation:
          H_bar      = (1/T) ∑_t H_t
          G_B_bar    = (1/T) ∑_t G_{t,B}
          H_glob     = H_bar ⊗ G_B_bar
    - Compute eigendecomposition of H_bar, G_B_bar to get α_i, u_i and β_j, v_j.
    - Form Kronecker eigenpairs:
          λ_{ij} = α_i β_j,  w_{ij} = v_j ⊗ u_i
      and select the top-L indices (i_ℓ, j_ℓ).
    - Construct the subspace along these directions and compute slice-wise subspace curvature:
          H_t^(x)[ℓ,ℓ'] = (u_{i_ℓ}^T H_t u_{i_ℓ'}) * (v_{j_ℓ}^T G_{t,B} v_{j_ℓ'})
      (Project H_t, G_{t,B} onto their respective eigenspaces, then combine via Hadamard product).

Outputs:
    - subspace_info: Dictionary containing global eigenspace & index info.
    - H_x_list:  [T] of H_t^(x) ∈ R^{L × L}
    - R_list:    [T] of R_t ∈ R^{L × L}, such that R_t^T R_t = H_t^(x) + λI

LGSSM Integration:
    - Define the state as subspace coordinates x_t ∈ R^L.
    - Set the observation model:
          y_t^(x) = 0,   H_obs_t = R_t,   ε_t ~ N(0, I)
      The observation NLL is:
          0.5 ||y_t - R_t x_t||^2 = 0.5 x_t^T (R_t^T R_t) x_t
      which corresponds exactly to the 0.5 x_t^T H_t^(x) x_t curvature penalty.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import torch
from torch import Tensor

def solve_xhat_from_grad(R: Tensor, g_x: Tensor) -> Tensor:
    """Compute μ = -H^{-1} g in the subspace.

    Correct logic for H = R^T R where R is UPPER triangular.
    (This matches the output of _chol_upper in project_curvature_to_subspace)
    
    Step 1: Solve R^T z = -g  (R^T is Lower)
    Step 2: Solve R μ = z     (R is Upper)
    """
    if g_x.ndim != 1:
        raise ValueError(f"g_x must be 1D, got shape={tuple(g_x.shape)}")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square 2D, got shape={tuple(R.shape)}")
    if R.shape[0] != g_x.numel():
        raise ValueError(f"Shape mismatch: R is {tuple(R.shape)}, g_x is {tuple(g_x.shape)}")
    
    rhs = (-g_x).unsqueeze(-1)  # (L,1)
    
    # 1. Solve R^T z = rhs (R is Upper, so R.T is Lower)
    z = torch.linalg.solve_triangular(R.T, rhs, upper=False)
    
    # 2. Solve R μ = z (R is Upper)
    mu = torch.linalg.solve_triangular(R, z, upper=True)
    
    return mu.squeeze(-1)


# -------------------------------------------------------------------------
# 1) Build global Kronecker eigenspace
# -------------------------------------------------------------------------

def build_global_kronecker_eigenspace(
    H_list: List[Tensor],
    G_B_list: List[Tensor],
    subspace_dim: int,
    eps_eig: float = 1e-6,
) -> Dict[str, Tensor]:
    """
    Constructs the global Kronecker eigenspace for Seq-LoRA.

    Args
    ----
    H_list: [T] of H_t ∈ R^{k×k}
        Slice-wise activation-side curvature.
    G_B_list: [T] of G_{t,B} ∈ R^{r×r}
        Slice-wise output-side curvature projected into LoRA B-subspace.
    subspace_dim: int
        Target subspace dimension L (selecting top-L Kronecker eigen directions).
    eps_eig: float
        Lower bound for eigenvalue clamping to ensure numerical stability.

    Returns
    -------
    subspace_info: dict
        Contains U_H, alpha, U_G, beta, pair_i, pair_j, unique indices, and U_lora projection matrix.
    """
    assert len(H_list) == len(G_B_list), "H_list and G_B_list must have same length."
    T = len(H_list)

    device = H_list[0].device
    dtype = H_list[0].dtype
    k = H_list[0].shape[0]
    r = G_B_list[0].shape[0]

    # Compute global averages
    H_bar = torch.zeros_like(H_list[0])
    G_B_bar = torch.zeros_like(G_B_list[0])
    for H_t, G_B_t in zip(H_list, G_B_list):
        H_bar = H_bar + H_t
        G_B_bar = G_B_bar + G_B_t
    H_bar = H_bar / float(T)
    G_B_bar = G_B_bar / float(T)

    # Symmetrize to avoid numerical errors
    H_bar = 0.5 * (H_bar + H_bar.T)
    G_B_bar = 0.5 * (G_B_bar + G_B_bar.T)

    # Eigendecomposition (returns ascending eigenvalues α, eigenvectors U_H)
    alpha, U_H = torch.linalg.eigh(H_bar)   # α ∈ R^{k}
    beta, U_G = torch.linalg.eigh(G_B_bar)  # β ∈ R^{r}

    # Clamp negative eigenvalues caused by numerical instability
    alpha = torch.clamp(alpha, min=eps_eig)
    beta = torch.clamp(beta, min=eps_eig)

    # Kronecker eigenvalues λ_{ij} = α_i β_j
    lam_2d = alpha.unsqueeze(1) * beta.unsqueeze(0)  # (k, r)
    lam_flat = lam_2d.reshape(-1)                    # (k*r,)

    total_dim = lam_flat.numel()
    L = min(subspace_dim, total_dim)

    # Select top-L eigenvalues
    vals, idx = torch.topk(lam_flat, k=L, largest=True, sorted=True)

    # Map 1D indices back to 2D (i, j)
    pair_i = idx // r   # (L,)
    pair_j = idx % r    # (L,)

    # Get unique indices to minimize projection sub-blocks later
    unique_i, inv_i = torch.unique(pair_i, return_inverse=True)  # unique_i: K_H', inv_i: (L,)
    unique_j, inv_j = torch.unique(pair_j, return_inverse=True)  # unique_j: K_G', inv_j: (L,)

    d_a = k * r
    L = pair_i.shape[0]
    U_lora = torch.zeros(d_a, L, device=device, dtype=dtype)

    for ell in range(L):
        i = pair_i[ell].item()
        j = pair_j[ell].item()
        u_i = U_H[:, i]    # (k,)
        v_j = U_G[:, j]    # (r,)
        
        w_ell = torch.ger(v_j, u_i).reshape(-1)  # (d_a,)
        U_lora[:, ell] = w_ell

    subspace_info = {
        "U_H": U_H.to(device=device, dtype=dtype),
        "alpha": alpha.to(device=device, dtype=dtype),
        "U_G": U_G.to(device=device, dtype=dtype),
        "beta": beta.to(device=device, dtype=dtype),
        "pair_i": pair_i.to(device=device),
        "pair_j": pair_j.to(device=device),
        "unique_i": unique_i.to(device=device),
        "inv_i": inv_i.to(device=device),
        "unique_j": unique_j.to(device=device),
        "inv_j": inv_j.to(device=device),
        "U_lora": U_lora,
        "r": torch.tensor(r),
        "k": torch.tensor(k),
    }
    return subspace_info


# -------------------------------------------------------------------------
# 2) Project slice-wise curvature into the subspace
# -------------------------------------------------------------------------

def _chol_upper(A: Tensor) -> Tensor:
    """Return an upper-triangular Cholesky factor R such that R.T @ R = A."""
    A = 0.5 * (A + A.T)

    def _chol(mat: Tensor) -> Tensor:
        try:
            return torch.linalg.cholesky(mat, upper=True)
        except TypeError:
            L = torch.linalg.cholesky(mat)
            return L.transpose(-1, -2)

    try:
        return _chol(A)
    except torch.linalg.LinAlgError:
        pass

    eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    diag_mean = torch.mean(torch.abs(torch.diagonal(A))).clamp_min(torch.as_tensor(1.0, device=A.device, dtype=A.dtype))
    eps = torch.finfo(A.dtype).eps

    evals = torch.linalg.eigvalsh(A)
    min_eig = float(evals.min().item())
    base_jitter = max(1e-8 * float(diag_mean.item()), eps)
    jitter = max(base_jitter, -min_eig + base_jitter)

    for _ in range(8):
        try:
            return _chol(A + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10.0

    # Final fallback: clip the spectrum to a small positive floor and factorize that.
    evals_clipped = evals.clamp_min(base_jitter)
    evecs = torch.linalg.eigh(A).eigenvectors
    A_psd = (evecs * evals_clipped.unsqueeze(0)) @ evecs.T
    A_psd = 0.5 * (A_psd + A_psd.T)
    return _chol(A_psd)


def trace_psd_factor(factor: Tensor) -> Tensor:
    """
    Return trace(M) for a PSD matrix represented either as:
      - a full square matrix M, or
      - a low-rank factor F with M = F F^T.
    """
    if factor.ndim != 2:
        raise ValueError(f"factor must be 2D, got shape={tuple(factor.shape)}")
    if factor.shape[0] == factor.shape[1]:
        return torch.trace(factor)
    return torch.sum(factor * factor)


def materialize_mean_psd_from_factors(
    factors: List[Tensor],
    matrix_scale: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Materialize the mean PSD matrix from raw KFAC factors without expanding each
    slice on the CPU. A square factor is interpreted as the PSD matrix itself;
    a rectangular factor F is interpreted as F F^T.
    """
    if len(factors) == 0:
        raise ValueError("factors must be non-empty")

    target_device = device or factors[0].device
    target_dtype = dtype or factors[0].dtype
    side = factors[0].shape[0]
    mean_psd = torch.zeros((side, side), device=target_device, dtype=target_dtype)
    scale = float(matrix_scale)

    for factor in factors:
        factor_t = factor.to(device=target_device, dtype=target_dtype)
        if factor_t.shape[0] == factor_t.shape[1]:
            mean_psd.add_(scale * 0.5 * (factor_t + factor_t.T))
        else:
            mean_psd.addmm_(factor_t, factor_t.T, beta=1.0, alpha=scale)

    mean_psd = mean_psd / float(len(factors))
    return 0.5 * (mean_psd + mean_psd.T)


def _project_psd_factor_to_basis(
    factor: Tensor,
    basis: Tensor,
    matrix_scale: float = 1.0,
) -> Tensor:
    scale = torch.as_tensor(matrix_scale, device=factor.device, dtype=factor.dtype)
    if factor.shape[0] == factor.shape[1]:
        proj = basis.T @ (scale * (0.5 * (factor + factor.T))) @ basis
    else:
        z = factor.T @ basis
        proj = scale * (z.T @ z)
    return 0.5 * (proj + proj.T)


def project_curvature_factors_to_subspace(
    H_factors: List[Tensor],
    G_B_factors: List[Tensor],
    subspace_info: Dict[str, Tensor],
    lambda_damp: float = 1e-4,
    H_matrix_scale: float = 1.0,
    G_matrix_scale: float = 1.0,
    work_device: torch.device | None = None,
    out_device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Project raw KFAC factors directly into the Kronecker subspace. This keeps
    the large activation/output factors in their compact form until after the
    L x L compression is complete.
    """
    if len(H_factors) != len(G_B_factors):
        raise ValueError("H_factors and G_B_factors must have the same length.")
    if len(H_factors) == 0:
        raise ValueError("H_factors must be non-empty.")

    target_device = work_device or H_factors[0].device
    target_dtype = dtype or H_factors[0].dtype

    U_H: Tensor = subspace_info["U_H"].to(device=target_device, dtype=target_dtype)
    U_G: Tensor = subspace_info["U_G"].to(device=target_device, dtype=target_dtype)
    unique_i: Tensor = subspace_info["unique_i"].to(device=target_device)
    inv_i: Tensor = subspace_info["inv_i"].to(device=target_device)
    unique_j: Tensor = subspace_info["unique_j"].to(device=target_device)
    inv_j: Tensor = subspace_info["inv_j"].to(device=target_device)

    L = inv_i.shape[0]
    U_H_sel = U_H[:, unique_i]
    U_G_sel = U_G[:, unique_j]

    H_x_list: List[Tensor] = []
    R_list: List[Tensor] = []
    eye_L = torch.eye(L, device=target_device, dtype=target_dtype)

    for H_factor, G_factor in zip(H_factors, G_B_factors):
        H_factor_t = H_factor.to(device=target_device, dtype=target_dtype)
        G_factor_t = G_factor.to(device=target_device, dtype=target_dtype)

        H_proj_t = _project_psd_factor_to_basis(H_factor_t, U_H_sel, H_matrix_scale)
        G_proj_t = _project_psd_factor_to_basis(G_factor_t, U_G_sel, G_matrix_scale)

        H_rows = H_proj_t[inv_i, :]
        A_t = H_rows[:, inv_i]

        G_rows = G_proj_t[inv_j, :]
        B_t = G_rows[:, inv_j]

        H_x = A_t * B_t
        H_x = 0.5 * (H_x + H_x.T)
        R_t = _chol_upper(H_x + lambda_damp * eye_L)

        if out_device is not None:
            H_x = H_x.to(device=out_device, dtype=target_dtype)
            R_t = R_t.to(device=out_device, dtype=target_dtype)

        H_x_list.append(H_x)
        R_list.append(R_t)

    return H_x_list, R_list


def project_curvature_to_subspace(
    H_list: List[Tensor],
    G_B_list: List[Tensor],
    subspace_info: Dict[str, Tensor],
    lambda_damp: float = 1e-4,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Projects slice-wise curvature (H_t, G_{t,B}) into the Kronecker subspace:

        H_t^(x)  ∈ R^{L×L}
        R_t      ∈ R^{L×L}, R_t^T R_t = H_t^(x) + λI

    Projection formula:
        - H_proj_t = U_H_sel^T H_t U_H_sel   (K_H'×K_H')
        - G_proj_t = U_G_sel^T G_{t,B} U_G_sel (K_G'×K_G')
        - A_t[ℓ,ℓ'] = H_proj_t[ inv_i[ℓ], inv_i[ℓ'] ]
        - B_t[ℓ,ℓ'] = G_proj_t[ inv_j[ℓ], inv_j[ℓ'] ]
        - H_t^(x)   = A_t ∘ B_t (Hadamard product)

    Returns
    -------
    H_x_list: [T] of H_t^(x) ∈ R^{L×L}
    R_list:   [T] of R_t ∈ R^{L×L}  (Cholesky factors)
    """
    assert len(H_list) == len(G_B_list), "H_list and G_B_list must have same length."
    T = len(H_list)

    device = H_list[0].device
    dtype = H_list[0].dtype

    U_H: Tensor = subspace_info["U_H"]
    U_G: Tensor = subspace_info["U_G"]
    unique_i: Tensor = subspace_info["unique_i"]
    inv_i: Tensor = subspace_info["inv_i"]
    unique_j: Tensor = subspace_info["unique_j"]
    inv_j: Tensor = subspace_info["inv_j"]

    L = inv_i.shape[0]

    # Select sub-blocks of the global eigenspace
    U_H_sel = U_H[:, unique_i]  # (k, K_H')
    U_G_sel = U_G[:, unique_j]  # (r, K_G')

    H_x_list: List[Tensor] = []
    R_list: List[Tensor] = []

    eye_L = torch.eye(L, device=device, dtype=dtype)

    for t_idx, (H_t, G_B_t) in enumerate(zip(H_list, G_B_list)):
        H_t = 0.5 * (H_t + H_t.T)
        G_B_t = 0.5 * (G_B_t + G_B_t.T)

        # Project onto eigenspaces
        H_proj_t = U_H_sel.T @ H_t @ U_H_sel
        H_proj_t = 0.5 * (H_proj_t + H_proj_t.T)

        G_proj_t = U_G_sel.T @ G_B_t @ U_G_sel
        G_proj_t = 0.5 * (G_proj_t + G_proj_t.T)

        # Re-index to L x L
        H_rows = H_proj_t[inv_i, :]   # (L, K_H')
        A_t = H_rows[:, inv_i]        # (L, L)

        G_rows = G_proj_t[inv_j, :]   # (L, K_G')
        B_t = G_rows[:, inv_j]        # (L, L)

        # Subspace curvature via Hadamard product
        H_x = A_t * B_t               # (L, L)
        H_x = 0.5 * (H_x + H_x.T)

        H_x_damped = H_x + lambda_damp * eye_L

        # Upper-triangular Cholesky factor: R_t^T R_t = H_x_damped
        R_t = _chol_upper(H_x_damped) 

        H_x_list.append(H_x)
        R_list.append(R_t)

    return H_x_list, R_list


# -------------------------------------------------------------------------
# 3) Prepare LGSSM inputs for lssm_ffbs.py
# -------------------------------------------------------------------------

def prepare_lgssm_observations(
    R_list: List[Tensor],
    mu_list: List[Tensor] | None = None,
    y_list: List[Tensor] | None = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Prepare LGSSM observation matrices H_t and observations y_t.

    Observation model (curvature subspace):
        y_t = R_t x_t + eps_t,     eps_t ~ N(0, I_L)
    with upper-triangular R_t satisfying:
        R_t^T R_t = H_t^(x) + λ I.
    """
    T = len(R_list)
    if T == 0:
        raise ValueError("R_list is empty")
    L = R_list[0].shape[0]
    device = R_list[0].device
    dtype = R_list[0].dtype

    if mu_list is not None and y_list is not None:
        raise ValueError("Provide only one of mu_list or y_list (not both).")

    H_obs_list = [R_t for R_t in R_list]

    if y_list is not None:
        if len(y_list) != T:
            raise ValueError(f"y_list length {len(y_list)} != T={T}")
        y_out = []
        for t in range(T):
            y_t = y_list[t]
            if y_t.shape != (L,):
                raise ValueError(f"y_list[{t}] has shape {tuple(y_t.shape)} expected {(L,)}")
            y_out.append(y_t.to(device=device, dtype=dtype))
        return H_obs_list, y_out

    if mu_list is not None:
        if len(mu_list) != T:
            raise ValueError(f"mu_list length {len(mu_list)} != T={T}")
        y_out = []
        for t in range(T):
            mu_t = mu_list[t]
            if mu_t.shape != (L,):
                raise ValueError(f"mu_list[{t}] has shape {tuple(mu_t.shape)} expected {(L,)}")
            mu_t = mu_t.to(device=device, dtype=dtype)
            y_out.append(R_list[t] @ mu_t)
        return H_obs_list, y_out

    y_out = [torch.zeros(L, device=device, dtype=dtype) for _ in range(T)]
    return H_obs_list, y_out


# -------------------------------------------------------------------------
# 4) Simple test: Projection of random SPD matrices
# -------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cpu")
    dtype = torch.float64

    T = 3
    k = 8
    r = 4
    L = 10  # Subspace dimension, <= k*r

    H_list: List[Tensor] = []
    G_B_list: List[Tensor] = []

    # Generate random SPD matrices H_t, G_{t,B}
    for t in range(T):
        A = torch.randn(k, k, device=device, dtype=dtype)
        H_t = A @ A.T + 0.1 * torch.eye(k, device=device, dtype=dtype)

        B = torch.randn(r, r, device=device, dtype=dtype)
        G_t = B @ B.T + 0.1 * torch.eye(r, device=device, dtype=dtype)

        H_list.append(H_t)
        G_B_list.append(G_t)

    subspace_info = build_global_kronecker_eigenspace(
        H_list=H_list,
        G_B_list=G_B_list,
        subspace_dim=L,
        eps_eig=1e-6,
    )

    H_x_list, R_list = project_curvature_to_subspace(
        H_list=H_list,
        G_B_list=G_B_list,
        subspace_info=subspace_info,
        lambda_damp=1e-4,
    )

    H_obs_list, y_list = prepare_lgssm_observations(R_list)

    print("=== Seq-LoRA subspace test ===")
    print("Number of slices T:", len(H_x_list))
    print("Subspace dim L:", H_x_list[0].shape[0])
    print("H_x[0] shape:", H_x_list[0].shape)
    print("R_0 shape:", R_list[0].shape)
    print("H_obs_0 shape:", H_obs_list[0].shape)
    print("y_0 shape:", y_list[0].shape)
