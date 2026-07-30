"""Shared full-window evaluation metrics for the reduced Merton problem.

The trained value networks use ``(t, y=log(w))`` as inputs, but the paper's
reduced derivative bundle is expressed in the physical wealth coordinate:

    D V = (V_w, V_ww).

There is no factor/state coordinate in this Merton reduction, so no ``V_wx``
component is added.  Keeping these pure NumPy definitions in one module makes
the Direct PINN and PI-PINN final evaluators use exactly the same formulas.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


FULL_WINDOW_METRIC_NAMES = (
    "MSE_V",
    "MSE_c",
    "MSE_pi",
    "RelL2_V",
    "RelL2_D",
    "RelL2_c",
    "RelL2_pi",
    "MaxErr_V",
    "e_D_sup",
    "e_Xev",
    "MaxErr_c",
    "MaxErr_pi",
)


def crra_homothetic_scale(
    tau: np.ndarray,
    nu: float,
    gamma: float,
    epsilon_bequest: float,
) -> np.ndarray:
    """Return the finite-horizon CRRA consumption/value scale.

    For terminal utility ``epsilon_bequest * U(W_T)``, homogeneity gives the
    terminal scale ``epsilon_bequest**(1/gamma)``.  Writing the closed form in
    terms of that root avoids the common (and silent for epsilon=1) mistake of
    inserting ``epsilon_bequest`` itself into the denominator.

    The returned ``h(tau)`` satisfies ``c/W = 1/h`` and the value multiplier
    is ``h**gamma``.  The explicit ``nu -> 0`` limit prevents cancellation.
    """
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if not np.isfinite(epsilon_bequest) or epsilon_bequest <= 0.0:
        raise ValueError("epsilon_bequest must be finite and positive")
    tau64 = np.asarray(tau, dtype=np.float64)
    epsilon_root = float(epsilon_bequest) ** (1.0 / float(gamma))
    nu64 = float(nu)
    if abs(nu64) < 1e-10:
        return epsilon_root + tau64
    decay = np.exp(-nu64 * tau64)
    return epsilon_root * decay - np.expm1(-nu64 * tau64) / nu64


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Return ``||pred-ref||_2 / ||ref||_2`` with a float64 safe floor."""
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    if pred64.shape != ref64.shape:
        raise ValueError(
            f"relative-L2 arrays must have identical shapes: "
            f"{pred64.shape} != {ref64.shape}"
        )
    denominator = float(np.sum(np.square(ref64), dtype=np.float64))
    numerator = float(np.sum(np.square(pred64 - ref64), dtype=np.float64))
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny)))


def derivative_bundle_metrics(
    Vw_pred: np.ndarray,
    Vww_pred: np.ndarray,
    Vw_ref: np.ndarray,
    Vww_ref: np.ndarray,
) -> Dict[str, float]:
    """Metrics for the wealth-coordinate reduced bundle ``(V_w,V_ww)``.

    ``RelL2_D`` is the relative product-space L2 norm

    ``sqrt(sum(dVw^2+dVww^2) / sum(Vw_ref^2+Vww_ref^2))``.

    ``e_D_sup`` is the discrete vector-valued sup norm

    ``max sqrt(dVw^2+dVww^2)``.
    """
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (Vw_pred, Vww_pred, Vw_ref, Vww_ref)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise ValueError(
            "all derivative-bundle arrays must have identical shapes: "
            + ", ".join(str(value.shape) for value in arrays)
        )
    d_vw = arrays[0] - arrays[2]
    d_vww = arrays[1] - arrays[3]
    pointwise_squared_error = np.square(d_vw) + np.square(d_vww)
    denominator = float(np.sum(
        np.square(arrays[2]) + np.square(arrays[3]), dtype=np.float64
    ))
    numerator = float(np.sum(pointwise_squared_error, dtype=np.float64))
    return {
        "RelL2_D": float(np.sqrt(
            numerator / max(denominator, np.finfo(np.float64).tiny)
        )),
        "e_D_sup": float(np.sqrt(np.max(pointwise_squared_error))),
    }


def full_window_metrics(
    V_pred: np.ndarray,
    c_pred: np.ndarray,
    pi_pred: np.ndarray,
    Vw_pred: np.ndarray,
    Vww_pred: np.ndarray,
    V_ref: np.ndarray,
    c_ref: np.ndarray,
    pi_ref: np.ndarray,
    Vw_ref: np.ndarray,
    Vww_ref: np.ndarray,
) -> Dict[str, float]:
    """Return the common all-margin value/bundle/control metric schema."""
    V_pred64 = np.asarray(V_pred, dtype=np.float64)
    c_pred64 = np.asarray(c_pred, dtype=np.float64)
    pi_pred64 = np.asarray(pi_pred, dtype=np.float64)
    V_ref64 = np.asarray(V_ref, dtype=np.float64)
    c_ref64 = np.asarray(c_ref, dtype=np.float64)
    pi_ref64 = np.asarray(pi_ref, dtype=np.float64)
    for label, pred, ref in (
        ("V", V_pred64, V_ref64),
        ("c", c_pred64, c_ref64),
        ("pi", pi_pred64, pi_ref64),
    ):
        if pred.shape != ref.shape:
            raise ValueError(
                f"{label} prediction/reference shapes differ: {pred.shape} != {ref.shape}"
            )

    metrics = {
        "MSE_V": float(np.mean(np.square(V_pred64 - V_ref64))),
        "MSE_c": float(np.mean(np.square(c_pred64 - c_ref64))),
        "MSE_pi": float(np.mean(np.square(pi_pred64 - pi_ref64))),
        "RelL2_V": relative_l2(V_pred64, V_ref64),
        "RelL2_c": relative_l2(c_pred64, c_ref64),
        "RelL2_pi": relative_l2(pi_pred64, pi_ref64),
        "MaxErr_V": float(np.max(np.abs(V_pred64 - V_ref64))),
        "MaxErr_c": float(np.max(np.abs(c_pred64 - c_ref64))),
        "MaxErr_pi": float(np.max(np.abs(pi_pred64 - pi_ref64))),
    }
    metrics.update(derivative_bundle_metrics(
        Vw_pred, Vww_pred, Vw_ref, Vww_ref))
    # Manuscript X_ev norm: ||dV||_infinity plus the vector-valued sup norm
    # of the physical-wealth derivative bundle D V=(V_w,V_ww).
    metrics["e_Xev"] = metrics["MaxErr_V"] + metrics["e_D_sup"]
    # A fixed order is useful for deterministic logs/CSV snapshots.
    return {name: metrics[name] for name in FULL_WINDOW_METRIC_NAMES}
