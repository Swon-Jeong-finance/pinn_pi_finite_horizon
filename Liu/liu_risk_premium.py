"""Risk-premium specifications shared by Liu training and post-processing.

The paper's non-affine experiment uses

    lambda_eps(x) = lambda_0 + Lambda x
                    + eps * loading_scale * Lambda
                      tanh((x - xbar) / state_scale).

For the reported experiment ``state_scale`` is the positive vector ``eta``
(so D = diag(eta)) and the nonlinear loading is aligned with ``Lambda``
(so Psi = loading_scale * Lambda).  Keeping this calculation in one small,
pure module prevents the frozen PDE, greedy map, and plotting code from
silently solving different models.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # NumPy-only checks remain usable off-cluster.
    torch = None  # type: ignore[assignment]


RISK_PREMIUM_MODES = ("affine", "tanh")


def validate_risk_premium_config(
    mode: str,
    eps: float,
    loading_scale: float,
    state_scale: Any | None = None,
) -> None:
    """Validate a risk-premium configuration and fail on ignored settings."""

    mode = str(mode).strip().lower()
    if mode not in RISK_PREMIUM_MODES:
        raise ValueError(
            f"unknown risk-premium mode {mode!r}; expected one of {RISK_PREMIUM_MODES}"
        )
    if not math.isfinite(float(eps)) or float(eps) < 0.0:
        raise ValueError("nonaffine eps must be finite and non-negative")
    if not math.isfinite(float(loading_scale)):
        raise ValueError("nonaffine loading scale must be finite")
    if mode == "affine" and float(eps) != 0.0:
        raise ValueError(
            "--nonaffine-eps is nonzero while --risk-premium-mode=affine; "
            "use mode=tanh or set eps=0"
        )
    if state_scale is not None:
        scale = np.asarray(state_scale, dtype=np.float64)
        if scale.ndim != 1 or scale.size == 0:
            raise ValueError("state_scale must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("every state_scale entry must be finite and strictly positive")


def has_affine_reference(mode: str, eps: float) -> bool:
    """Whether the affine Riccati solution is an exact reference."""

    return str(mode).strip().lower() == "affine" or float(eps) == 0.0


def risk_premium_torch(
    x: torch.Tensor,
    lam0: torch.Tensor,
    Lam: torch.Tensor,
    *,
    mode: str,
    eps: float,
    xbar: torch.Tensor,
    state_scale: torch.Tensor,
    loading_scale: float = 1.0,
) -> torch.Tensor:
    """Evaluate the affine or tanh risk premium on a Torch batch.

    ``x`` has shape ``(..., M)``, ``lam0`` has shape ``(N,)``, and ``Lam``
    has shape ``(N, M)``.  No ``no_grad`` context is used: coefficients do
    not contain trainable parameters, but retaining the state Jacobian makes
    the model definition transparent to diagnostics and derivative tests.
    """

    if torch is None:
        raise RuntimeError("risk_premium_torch requires PyTorch")

    if x.shape[-1] != Lam.shape[-1]:
        raise ValueError(f"x/Lam state dimension mismatch: {x.shape} vs {Lam.shape}")
    if Lam.shape[0] != lam0.shape[-1]:
        raise ValueError(f"lam0/Lam asset dimension mismatch: {lam0.shape} vs {Lam.shape}")
    if xbar.shape[-1] != x.shape[-1] or state_scale.shape[-1] != x.shape[-1]:
        raise ValueError("xbar/state_scale dimension does not match x")

    affine = lam0 + torch.matmul(x, Lam.transpose(-1, -2))
    if str(mode).strip().lower() == "affine" or float(eps) == 0.0:
        return affine

    standardized = (x - xbar) / state_scale
    nonlinear = torch.matmul(torch.tanh(standardized), Lam.transpose(-1, -2))
    return affine + float(eps) * float(loading_scale) * nonlinear


def risk_premium_numpy(
    x: np.ndarray,
    lam0: np.ndarray,
    Lam: np.ndarray,
    *,
    mode: str,
    eps: float,
    xbar: np.ndarray,
    state_scale: np.ndarray,
    loading_scale: float = 1.0,
) -> np.ndarray:
    """NumPy counterpart of :func:`risk_premium_torch`."""

    x_arr = np.asarray(x)
    lam0_arr = np.asarray(lam0)
    Lam_arr = np.asarray(Lam)
    xbar_arr = np.asarray(xbar)
    scale_arr = np.asarray(state_scale)

    if x_arr.shape[-1] != Lam_arr.shape[-1]:
        raise ValueError(f"x/Lam state dimension mismatch: {x_arr.shape} vs {Lam_arr.shape}")
    affine = lam0_arr + np.matmul(x_arr, Lam_arr.T)
    if str(mode).strip().lower() == "affine" or float(eps) == 0.0:
        return affine

    standardized = (x_arr - xbar_arr) / scale_arr
    nonlinear = np.matmul(np.tanh(standardized), Lam_arr.T)
    return affine + float(eps) * float(loading_scale) * nonlinear
