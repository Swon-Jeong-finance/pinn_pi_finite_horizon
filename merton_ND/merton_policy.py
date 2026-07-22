"""Greedy-control helpers matching the Merton trainer's implemented map.

The current trainer does not import this module, so parity is enforced by
regression tests against its formulas.  Activation masks are returned so the
finite-difference post-processor never reports a guarded/clipped calculation
as the unmodified theoretical map.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def consumption_from_log_derivative(
    value_y: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float,
    vw_guard: float = 1e-8,
    kappa_min: Optional[float] = None,
    kappa_max: Optional[float] = None,
    consumption_min: Optional[float] = None,
    consumption_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    wealth = torch.exp(y)
    value_w = value_y / wealth
    guard_mask = value_w < float(vw_guard)
    value_w_safe = torch.clamp(value_w, min=float(vw_guard))
    consumption_raw = value_w_safe.pow(-1.0 / float(gamma))
    kappa_raw = consumption_raw / wealth
    kappa = kappa_raw
    kappa_low = torch.zeros_like(kappa_raw, dtype=torch.bool)
    kappa_high = torch.zeros_like(kappa_raw, dtype=torch.bool)
    if kappa_min is not None:
        kappa_low = kappa < float(kappa_min)
        kappa = torch.clamp(kappa, min=float(kappa_min))
    if kappa_max is not None:
        kappa_high = kappa > float(kappa_max)
        kappa = torch.clamp(kappa, max=float(kappa_max))
    consumption = kappa * wealth
    consumption_low = torch.zeros_like(consumption, dtype=torch.bool)
    consumption_high = torch.zeros_like(consumption, dtype=torch.bool)
    if consumption_min is not None:
        consumption_low = consumption < float(consumption_min)
        consumption = torch.clamp(consumption, min=float(consumption_min))
    if consumption_max is not None:
        consumption_high = consumption > float(consumption_max)
        consumption = torch.clamp(consumption, max=float(consumption_max))
    return consumption, {
        "vw_guard": guard_mask,
        "kappa_low_clip": kappa_low,
        "kappa_high_clip": kappa_high,
        "consumption_low_clip": consumption_low,
        "consumption_high_clip": consumption_high,
        "value_w": value_w,
        "kappa_raw": kappa_raw,
    }


def portfolio_from_log_derivatives(
    value_y: torch.Tensor,
    value_yy: torch.Tensor,
    y: torch.Tensor,
    sigma_inv_mu: torch.Tensor,
    *,
    guard_mode: str,
    numerator_guard: Optional[float] = None,
    denominator_guard: float = 1e-8,
    portfolio_min: Optional[float] = None,
    portfolio_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return portfolio weights and masks for the selected denominator rule.

    Modes:

    ``one-sided``
        The current trainer rule.  Clamp both the log-value numerator
        ``v_y`` and the positive curvature denominator ``v_y-v_yy`` from
        below by ``numerator_guard`` and ``denominator_guard``, respectively.
    ``legacy-signed``
        Preserve the sign and floor ``abs(v_yy-v_y)``.  This exactly matches
        the original Merton file in this repository.
    ``log-concavity``
        Clamp ``v_yy-v_y`` above by ``-denominator_guard``.
    ``wealth-concavity``
        Apply ``V_ww <= -denominator_guard`` before converting back to log
        coordinates, i.e. use ``min(v_yy-v_y, -eps*exp(2y))``.
    ``none``
        Use the raw denominator.  Non-concave/near-zero points then raise.
    """
    mode = str(guard_mode).replace("_", "-").lower()
    if mode in {"trainer-one-sided", "current-one-sided", "log-one-sided"}:
        mode = "one-sided"
    denominator = value_yy - value_y
    positive_curvature = denominator >= 0.0
    eps = float(denominator_guard)
    if eps <= 0.0:
        raise ValueError("denominator_guard must be positive")
    numerator_eps = eps if numerator_guard is None else float(numerator_guard)
    if numerator_eps <= 0.0:
        raise ValueError("numerator_guard must be positive")
    numerator_guard_mask = torch.zeros_like(value_y, dtype=torch.bool)
    if mode == "one-sided":
        # Trainer formula:
        #   d_safe = clamp(v_y-v_yy, min=eps)
        #   scalar = clamp(v_y, min=eps) / d_safe.
        # Keep the denominator in the historical (v_yy-v_y) orientation for
        # the frozen-PDE diagnostics returned below.
        positive_denominator = value_y - value_yy
        numerator_guard_mask = value_y < numerator_eps
        guard_mask = positive_denominator < eps
        numerator_safe = torch.clamp(value_y, min=numerator_eps)
        denominator_safe = -torch.clamp(positive_denominator, min=eps)
    elif mode == "legacy-signed":
        guard_mask = torch.abs(denominator) < eps
        sign = torch.sign(denominator)
        sign = torch.where(sign == 0.0, -torch.ones_like(sign), sign)
        denominator_safe = torch.where(guard_mask, sign * eps, denominator)
        numerator_safe = value_y
    elif mode == "log-concavity":
        threshold = -eps * torch.ones_like(denominator)
        guard_mask = denominator > threshold
        denominator_safe = torch.minimum(denominator, threshold)
        numerator_safe = value_y
    elif mode == "wealth-concavity":
        threshold = -eps * torch.exp(2.0 * y)
        guard_mask = denominator > threshold
        denominator_safe = torch.minimum(denominator, threshold)
        numerator_safe = value_y
    elif mode == "none":
        guard_mask = torch.zeros_like(denominator, dtype=torch.bool)
        if bool(torch.any(positive_curvature).item()) or bool(torch.any(torch.abs(denominator) <= eps).item()):
            raise ValueError("unmodified greedy portfolio is undefined at a non-concave/near-zero denominator")
        denominator_safe = denominator
        numerator_safe = value_y
    else:
        raise ValueError(f"unknown Merton denominator guard mode: {guard_mode}")

    scalar = -(numerator_safe / denominator_safe)
    portfolio_raw = scalar * sigma_inv_mu.reshape(1, -1)
    portfolio = portfolio_raw
    low_clip = torch.zeros_like(portfolio_raw, dtype=torch.bool)
    high_clip = torch.zeros_like(portfolio_raw, dtype=torch.bool)
    if portfolio_min is not None:
        low_clip = portfolio < float(portfolio_min)
        portfolio = torch.clamp(portfolio, min=float(portfolio_min))
    if portfolio_max is not None:
        high_clip = portfolio > float(portfolio_max)
        portfolio = torch.clamp(portfolio, max=float(portfolio_max))
    any_clip = torch.any(low_clip | high_clip, dim=1, keepdim=True)
    return portfolio, {
        "numerator_guard": numerator_guard_mask,
        "denominator_guard": guard_mask,
        "positive_curvature": positive_curvature,
        "portfolio_low_clip_components": low_clip,
        "portfolio_high_clip_components": high_clip,
        "portfolio_any_clip": any_clip,
        "denominator_raw": denominator,
        "denominator_safe": denominator_safe,
        "portfolio_raw": portfolio_raw,
    }
