"""
Linear Gaussian State-Space Model (LGSSM) utilities:
- Kalman filtering

All operations are done in a low-dimensional state space R^L.
This module is completely independent of LoRA / KFAC; it just
implements the generic linear-Gaussian inference.
"""

from typing import List, Tuple
import torch

Tensor = torch.Tensor


def kalman_filter(
    H_list: List[Tensor],
    y_list: List[Tensor],
    Q_list: List[Tensor],
    m1: Tensor,
    P1: Tensor,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    r"""
    Standard Kalman filter for:
        x_1 ~ N(m1, P1)
        x_t = x_{t-1} + u_t,    u_t ~ N(0, Q_t), t > 1
        y_t = H_t x_t + eps_t,  eps_t ~ N(0, I)

    Args
    ----
    H_list : list of length T
        H_t \in R^{L x L} (observation matrix in reduced space).
    y_list : list of length T
        y_t \in R^{L} (observations in reduced space).
    Q_list : list of length T
        Q_t \in R^{L x L} (process noise covariances for transitions into t).
        Q_list[0] is ignored because P1 is already the prior covariance for x_1.
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

        # Prediction: x_{t|t-1}, P_{t|t-1}. For t=0, P1 is already
        # the prior covariance of x_1, so do not add a process-noise term.
        x_t_pred = m_prev.clone()                       # random walk: F = I
        P_t_pred = P_prev if t == 0 else P_prev + Q_t

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
