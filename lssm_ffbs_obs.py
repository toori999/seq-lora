"""
Linear Gaussian State-Space Model (LGSSM) utilities:
- Kalman filtering
- RTS smoothing
- FFBS trajectory sampling
- EM estimation of scalar process noise Q_t = sigma * I

All operations are done in a low-dimensional state space R^L.
This module is completely independent of LoRA / KFAC; it just
implements the generic linear-Gaussian inference.
"""

from typing import List, Tuple
import torch
import math

Tensor = torch.Tensor


def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure tensor is 2D (vector -> (L,1) or (1,L) as needed)."""
    if x.dim() == 1:
        return x.unsqueeze(-1)  # (L,) -> (L,1)
    return x


def kalman_filter(
    H_list: List[Tensor],
    y_list: List[Tensor],
    Q_list: List[Tensor],
    m1: Tensor,
    P1: Tensor,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """
    Standard Kalman filter for:
        x_t = x_{t-1} + u_t,    u_t ~ N(0, Q_t)
        y_t = H_t x_t + eps_t,  eps_t ~ N(0, I)

    Args
    ----
    H_list : list of length T
        H_t \in R^{L x L} (observation matrix in reduced space).
    y_list : list of length T
        y_t \in R^{L} (observations in reduced space).
    Q_list : list of length T
        Q_t \in R^{L x L} (process noise covariances).
        You can also use a shared Q and repeat it.
    m1 : Tensor, shape (L,)
        Prior mean of x_1.
    P1 : Tensor, shape (L, L)
        Prior covariance of x_1.

    Returns
    -------
    x_filt : list of T tensors, shape (L,)
        Filtered means x_{t|t}.
    P_filt : list of T tensors, shape (L, L)
        Filtered covariances P_{t|t}.
    x_pred : list of T tensors, shape (L,)
        One-step predicted means x_{t|t-1}.
    P_pred : list of T tensors, shape (L, L)
        One-step predicted covariances P_{t|t-1}.
    """
    T = len(H_list)
    assert len(y_list) == T
    assert len(Q_list) == T

    L = m1.shape[0]
    device = m1.device
    dtype = m1.dtype

    x_filt: List[Tensor] = []
    P_filt: List[Tensor] = []
    x_pred: List[Tensor] = []
    P_pred: List[Tensor] = []

    # Initial prior
    m_prev = m1.to(device=device, dtype=dtype)          # (L,)
    P_prev = P1.to(device=device, dtype=dtype)          # (L,L)

    I_L = torch.eye(L, device=device, dtype=dtype)

    for t in range(T):
        H_t = H_list[t].to(device=device, dtype=dtype)  # (L,L)
        y_t = y_list[t].to(device=device, dtype=dtype)  # (L,)
        Q_t = Q_list[t].to(device=device, dtype=dtype)  # (L,L)

        # Prediction: x_{t|t-1}, P_{t|t-1}
        x_t_pred = m_prev.clone()                       # random walk: F = I
        P_t_pred = P_prev + Q_t

        # Innovation covariance: S_t = H P H^T + I
        HP = H_t @ P_t_pred
        S_t = HP @ H_t.T + I_L  # (L,L)
        # Symmetrize + jitter for numerical stability
        S_t = 0.5 * (S_t + S_t.T)
        S_t = S_t + 1e-6 * I_L

        # Kalman gain: K_t = P H^T S^{-1}
        # We solve S_t X = H_t P_t_pred for X, then transpose.
        K_t_T = torch.linalg.solve(S_t, HP).T
        K_t = K_t_T  # (L,L)

        # Innovation residual
        resid = y_t - (H_t @ x_t_pred)  # (L,)

        # Filtered mean and covariance
        x_t_filt = x_t_pred + K_t @ resid
        P_t_filt = (I_L - K_t @ H_t) @ P_t_pred
        # Symmetrize to avoid drift
        P_t_filt = 0.5 * (P_t_filt + P_t_filt.T)

        # Save
        x_pred.append(x_t_pred)
        P_pred.append(P_t_pred)
        x_filt.append(x_t_filt)
        P_filt.append(P_t_filt)

        # Update for next step
        m_prev = x_t_filt
        P_prev = P_t_filt

    return x_filt, P_filt, x_pred, P_pred


def rts_smoother(
    x_filt: List[Tensor],
    P_filt: List[Tensor],
    x_pred: List[Tensor],
    P_pred: List[Tensor],
    Q_list: List[Tensor],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Rauch–Tung–Striebel (RTS) smoother for the random-walk LGSSM.

    Args
    ----
    x_filt : list[T] of (L,)
        Filtered means x_{t|t}.
    P_filt : list[T] of (L,L)
        Filtered covariances P_{t|t}.
    x_pred : list[T] of (L,)
        One-step predicted means x_{t|t-1}.
    P_pred : list[T] of (L,L)
        One-step predicted covariances P_{t|t-1}.
    Q_list : list[T] of (L,L)
        Process noise covariances Q_t.
        (Not explicitly used for RTS gain in this implementation,
         but kept for API compatibility.)

    Returns
    -------
    x_smooth : list[T] of (L,)
        Smoothed means x_{t|T}.
    P_smooth : list[T] of (L,L)
        Smoothed covariances P_{t|T}.
    J_list : list[T-1] of (L,L)
        Smoother gains J_t for t=1..T-1.
    """
    T = len(x_filt)
    assert len(P_filt) == T
    assert len(x_pred) == T
    assert len(P_pred) == T
    assert len(Q_list) == T

    L = x_filt[0].shape[0]
    device = x_filt[0].device
    dtype = x_filt[0].dtype

    x_smooth: List[Tensor] = [torch.zeros_like(x) for x in x_filt]
    P_smooth: List[Tensor] = [torch.zeros_like(P) for P in P_filt]
    J_list: List[Tensor] = []

    # Initialize at T
    x_smooth[T - 1] = x_filt[T - 1].clone()
    P_smooth[T - 1] = P_filt[T - 1].clone()

    # Backward pass
    for t in reversed(range(T - 1)):
        P_t_t = P_filt[t]        # (L,L)
        P_tp1_t = P_pred[t + 1]  # (L,L)

        # RTS gain: J_t = P_{t|t} P_{t+1|t}^{-1}
        # Solve P_{t+1|t}^T X^T = P_{t|t}^T for X^T, then transpose.
        J_t = torch.linalg.solve(P_tp1_t.T, P_t_t.T).T  # (L,L)

        x_smooth[t] = x_filt[t] + J_t @ (
            x_smooth[t + 1] - x_pred[t + 1]
        )
        P_smooth[t] = P_t_t + J_t @ (
            P_smooth[t + 1] - P_tp1_t
        ) @ J_t.T

        J_list.insert(0, J_t)  # prepend to match index t

    return x_smooth, P_smooth, J_list


def lag_one_smoothed_covariances(
    P_smooth: List[Tensor],
    J_list: List[Tensor],
) -> List[Tensor]:
    """
    Return lag-one smoothed covariances P_{t,t-1|T}.

    For the random-walk model used here, the conditional mean of x_{t-1}
    given x_t and observations has affine gain J_{t-1}, so

        P_{t-1,t|T} = J_{t-1} P_{t|T}
        P_{t,t-1|T} = P_{t|T} J_{t-1}^T

    We return a length-T list aligned with the time index; entry 0 is a zero
    placeholder because there is no lag-one covariance for t=0.
    """
    if len(P_smooth) == 0:
        raise ValueError("P_smooth must be non-empty")
    if len(J_list) != max(len(P_smooth) - 1, 0):
        raise ValueError(
            f"Expected len(J_list)={max(len(P_smooth) - 1, 0)}, got {len(J_list)}"
        )

    L = P_smooth[0].shape[0]
    device = P_smooth[0].device
    dtype = P_smooth[0].dtype
    out: List[Tensor] = [torch.zeros((L, L), device=device, dtype=dtype)]
    for t in range(1, len(P_smooth)):
        out.append(P_smooth[t] @ J_list[t - 1].T)
    return out


def ffbs_sample(
    x_filt: List[Tensor],
    P_filt: List[Tensor],
    x_pred: List[Tensor],
    P_pred: List[Tensor],
    num_samples: int = 1,
) -> Tensor:
    """
    Forward-Filtering Backward-Sampling (FFBS) for random-walk LGSSM.

    This implementation follows the standard formulation using
    filtered moments and RTS-style gains, without explicitly constructing
    any block matrices.

    Args
    ----
    x_filt : list[T] of (L,)
        Filtered means x_{t|t}.
    P_filt : list[T] of (L,L)
        Filtered covariances P_{t|t}.
    x_pred : list[T] of (L,)
        One-step predicted means x_{t|t-1}.
    P_pred : list[T] of (L,L)
        One-step predicted covariances P_{t|t-1}.
    num_samples : int
        Number of trajectories to sample.

    Returns
    -------
    samples : Tensor of shape (num_samples, T, L)
        Sampled trajectories {x_{1:T}^{(s)}}_{s=1}^S.
    """
    T = len(x_filt)
    assert len(P_filt) == T
    assert len(x_pred) == T
    assert len(P_pred) == T

    L = x_filt[0].shape[0]
    device = x_filt[0].device
    dtype = x_filt[0].dtype

    samples = torch.zeros(
        num_samples, T, L, device=device, dtype=dtype
    )

    # Precompute smoother-like gains J_t
    J_list: List[Tensor] = []
    for t in range(T - 1):
        P_t_t = P_filt[t]        # (L,L)
        P_tp1_t = P_pred[t + 1]  # (L,L)
        J_t = torch.linalg.solve(P_tp1_t.T, P_t_t.T).T  # (L,L)
        J_list.append(J_t)

    # Cholesky at each time for conditional covariances C_t
    # C_t = P_{t|t} - J_t P_{t+1|t} J_t^T
    C_chol: List[Tensor] = [None] * (T - 1)
    for t in range(T - 1):
        J_t = J_list[t]
        P_t_t = P_filt[t]
        P_tp1_t = P_pred[t + 1]
        C_t = P_t_t - J_t @ P_tp1_t @ J_t.T
        # numerical stabilisation
        C_t = 0.5 * (C_t + C_t.T)
        jitter = 1e-6 * torch.eye(L, device=device, dtype=dtype)
        C_t = C_t + jitter
        C_chol[t] = torch.linalg.cholesky(C_t)

    # Terminal covariance Cholesky for x_T
    P_TT = P_filt[T - 1]
    P_TT = 0.5 * (P_TT + P_TT.T)
    jitter_T = 1e-6 * torch.eye(L, device=device, dtype=dtype)
    L_T = torch.linalg.cholesky(P_TT + jitter_T)

    # Sample trajectories
    for s in range(num_samples):
        # Step (i): sample x_T
        eps_T = torch.randn(L, device=device, dtype=dtype)
        x_T = x_filt[T - 1] + L_T @ eps_T
        samples[s, T - 1] = x_T

        # Step (ii): backward sampling
        x_next = x_T
        for t in reversed(range(T - 1)):
            J_t = J_list[t]
            C_t_chol = C_chol[t]

            m_t = x_filt[t] + J_t @ (x_next - x_pred[t + 1])
            eps_t = torch.randn(L, device=device, dtype=dtype)
            x_t = m_t + C_t_chol @ eps_t

            samples[s, t] = x_t
            x_next = x_t

    return samples
