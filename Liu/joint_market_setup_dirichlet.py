
"""
joint_market_setup.py
=====================

Utilities to sample *numerically stable* synthetic market parameters for
multi-asset (n) and optional state-factor (k) diffusion models.

What it gives you
-----------------
1) A (n+k)x(n+k) *joint correlation* matrix for Brownian shocks, with:
   - Positive definiteness (SPD)
   - Optional max |off-diagonal| constraint
   - Condition number control (limits precision blow-ups)
   - Optional minimum eigenvalue floor

2) Volatility scales for assets and states, and the corresponding covariance blocks.

3) (Optional, KO-style) simple OU dynamics for the state vector and a linear
   risk-premium loading matrix for assets.

This is model-agnostic: you can use it for Merton, Kim–Omberg, or any setting
where you need a stable joint Gaussian shock structure.

If k == 0, the generator reduces to "asset-only" parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Mapping

import numpy as np


RHO_CONVENTION = "identity_block_whitened_v1"
MARKET_SCHEMA_VERSION = 2


# ----------------------------
# Linear algebra helpers
# ----------------------------

def _to_correlation(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SPD matrix S to a correlation matrix with unit diagonal."""
    S = 0.5 * (S + S.T)
    d = np.sqrt(np.maximum(np.diag(S), eps))
    inv_d = 1.0 / d
    R = (S * inv_d[None, :]) * inv_d[:, None]
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _max_abs_offdiag(R: np.ndarray) -> float:
    """Max absolute off-diagonal entry."""
    A = np.abs(R.copy())
    np.fill_diagonal(A, 0.0)
    return float(A.max(initial=0.0))


def _eigvalsh(A: np.ndarray) -> np.ndarray:
    """Eigenvalues for symmetric matrix (sorted ascending)."""
    A = 0.5 * (A + A.T)
    return np.linalg.eigvalsh(A)


def _cond_spd(A: np.ndarray, eps: float = 1e-18) -> float:
    """Condition number for SPD matrix via eigenvalues."""
    w = _eigvalsh(A)
    w_min = float(np.maximum(w[0], eps))
    w_max = float(np.maximum(w[-1], eps))
    return w_max / w_min


def _find_alpha_for_kappa(R0: np.ndarray, kappa_max: float, tol: float = 1e-10) -> float:
    """
    Find minimal alpha in [0,1) such that cond((1-alpha)R0 + alpha I) <= kappa_max.
    Uses bisection; monotone because shrinkage to I improves conditioning.
    """
    if kappa_max <= 1.0:
        return 1.0 - 1e-12
    if _cond_spd(R0) <= kappa_max:
        return 0.0

    lo, hi = 0.0, 1.0 - 1e-12
    I = np.eye(R0.shape[0])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        R = (1.0 - mid) * R0 + mid * I
        if _cond_spd(R) <= kappa_max:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def _alpha_for_min_eig(R0: np.ndarray, eig_min: float) -> float:
    """
    Smallest alpha so that min_eig((1-alpha)R0 + alpha I) >= eig_min.
    For shrinkage to identity: lambda_min(alpha) = (1-alpha)lambda_min0 + alpha.
    """
    if eig_min <= 0:
        return 0.0
    w = _eigvalsh(R0)
    lam_min0 = float(w[0])
    if lam_min0 >= eig_min:
        return 0.0
    denom = max(1.0 - lam_min0, 1e-12)
    a = (eig_min - lam_min0) / denom
    return float(np.clip(a, 0.0, 1.0 - 1e-12))


def _ridge_for_target_kappa(lam_min: float, lam_max: float, kappa_target: Optional[float]) -> float:
    """
    Minimal delta >= 0 such that (lam_max + delta)/(lam_min + delta) <= kappa_target.
    Works for SPD where lam_min > 0. If already conditioned, returns 0.
    """
    if kappa_target is None or not np.isfinite(kappa_target):
        return 0.0
    if kappa_target <= 1.0:
        return float(max(lam_max - lam_min, 0.0))
    if (lam_max / lam_min) <= kappa_target:
        return 0.0

    num = lam_max - kappa_target * lam_min
    den = kappa_target - 1.0
    return float(max(num / den, 0.0))


def _symmetric_inverse_sqrt(A: np.ndarray, *, name: str) -> np.ndarray:
    """Return the symmetric inverse square root of a finite SPD matrix."""

    value = np.asarray(A, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape={value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(value, value.T, rtol=2.0e-12, atol=2.0e-13):
        raise ValueError(f"{name} must be symmetric")
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(
            f"{name} must be positive definite; min eigenvalue={eigenvalues[0]:.3e}"
        )
    inverse_root = (
        eigenvectors * np.power(eigenvalues, -0.5)[None, :]
    ) @ eigenvectors.T
    return 0.5 * (inverse_root + inverse_root.T)


def identity_block_correlation_diagnostics(rho: np.ndarray) -> Dict[str, float]:
    """Diagnose ``[[I, rho], [rho.T, I]]`` without forming the full matrix.

    Its extreme eigenvalues are ``1 +/- sigma_max(rho)``.  Strict positive
    definiteness is therefore equivalent to ``||rho||_2 < 1``.
    """

    value = np.asarray(rho, dtype=float)
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError(
            f"rho must be a non-empty asset-by-state matrix, got shape={value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("rho contains NaN or infinity")
    singular_values = np.linalg.svd(value, compute_uv=False)
    spectral_norm = float(singular_values[0])
    return {
        "rho_spectral_norm": spectral_norm,
        "min_eig": 1.0 - spectral_norm,
        "max_eig": 1.0 + spectral_norm,
    }


def canonicalize_cross_correlation(
    Psi: np.ndarray,
    rho_raw: np.ndarray,
    Phi_Z: np.ndarray,
) -> np.ndarray:
    """Whiten a cross block for the identity-block Brownian convention.

    ``sample_spd_correlation`` supplies a valid source correlation

        C = [[Psi, rho_raw], [rho_raw.T, Phi_Z]].

    The Liu HJB, however, writes the asset and state Brownian covariance
    blocks as identities.  Its compatible cross-correlation is therefore

        rho = Psi^{-1/2} rho_raw Phi_Z^{-1/2}.

    No clipping or jitter is applied.  If the source blocks do not imply a
    strictly positive identity-block correlation, generation fails rather
    than silently changing the model.
    """

    psi = np.asarray(Psi, dtype=float)
    phi = np.asarray(Phi_Z, dtype=float)
    raw = np.asarray(rho_raw, dtype=float)
    psi_inverse_root = _symmetric_inverse_sqrt(psi, name="Psi")
    phi_inverse_root = _symmetric_inverse_sqrt(phi, name="Phi_Z")
    expected_shape = (psi_inverse_root.shape[0], phi_inverse_root.shape[0])
    if raw.shape != expected_shape:
        raise ValueError(
            "rho_raw shape is incompatible with Psi/Phi_Z: "
            f"{raw.shape} vs {expected_shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("rho_raw contains NaN or infinity")
    canonical = psi_inverse_root @ raw @ phi_inverse_root
    diagnostics = identity_block_correlation_diagnostics(canonical)
    if diagnostics["min_eig"] <= 0.0:
        raise ValueError(
            "whitened cross-correlation is not strictly admissible for identity "
            f"Brownian blocks: ||rho||_2={diagnostics['rho_spectral_norm']:.16g}"
        )
    return canonical


def rho_snapshot_metadata(params: "JointMarketParams") -> Dict[str, np.ndarray]:
    """Return auditable metadata for the HJB cross-correlation snapshot."""

    if params.k <= 0 or params.Phi_Z is None or params.rho_Z is None:
        return {}
    diagnostics = identity_block_correlation_diagnostics(params.rho_canonical)
    return {
        "market_schema_version": np.asarray([MARKET_SCHEMA_VERSION], dtype=np.int64),
        "rho_convention": np.asarray([RHO_CONVENTION]),
        "Psi": np.asarray(params.Psi, dtype=float),
        "Phi_Z": np.asarray(params.Phi_Z, dtype=float),
        "rho_raw": np.asarray(params.rho_Z, dtype=float),
        "rho_spectral_norm": np.asarray(
            [diagnostics["rho_spectral_norm"]], dtype=float
        ),
        "min_eig_joint_innovation": np.asarray(
            [diagnostics["min_eig"]], dtype=float
        ),
    }


def validate_rho_snapshot(
    values: Mapping[str, Any],
    *,
    expected_rho: Optional[np.ndarray] = None,
    require_canonical_metadata: bool = False,
) -> Dict[str, float]:
    """Validate the rho convention and identity-block covariance in an NPZ.

    Legacy snapshots without convention metadata remain readable when
    ``require_canonical_metadata`` is false, but they must still have a
    strictly positive identity-block covariance.  Eval-only execution of the
    updated trainers sets the flag to true so an old checkpoint cannot be
    silently evaluated under newly whitened coefficients.
    """

    if "rho" not in values:
        raise ValueError("market snapshot is missing rho")
    rho = np.asarray(values["rho"], dtype=float)
    diagnostics = identity_block_correlation_diagnostics(rho)
    if diagnostics["min_eig"] <= 0.0:
        raise ValueError(
            "market snapshot has a non-positive identity-block innovation "
            f"covariance: min eigenvalue={diagnostics['min_eig']:.3e}"
        )

    metadata_keys = {
        "market_schema_version", "rho_convention", "Psi", "Phi_Z", "rho_raw",
        "rho_spectral_norm", "min_eig_joint_innovation",
    }
    present = metadata_keys.intersection(values.keys())
    if present and present != metadata_keys:
        missing = sorted(metadata_keys - present)
        raise ValueError(f"market snapshot has incomplete rho metadata: {missing}")
    if not present:
        if require_canonical_metadata:
            raise ValueError(
                "market snapshot predates canonical rho metadata; retraining is "
                "required before eval-only use with the updated solver"
            )
    else:
        version_array = np.asarray(values["market_schema_version"]).reshape(-1)
        convention_array = np.asarray(values["rho_convention"]).reshape(-1)
        if (
            version_array.size != 1
            or not np.issubdtype(version_array.dtype, np.number)
            or not np.isfinite(float(version_array[0]))
            or float(version_array[0]) != float(MARKET_SCHEMA_VERSION)
        ):
            raise ValueError(
                "unsupported market schema version: "
                f"{version_array.tolist()}"
            )
        if convention_array.size != 1 or str(convention_array[0]) != RHO_CONVENTION:
            raise ValueError(
                "unsupported rho convention: "
                f"{convention_array.tolist()}"
            )
        reconstructed = canonicalize_cross_correlation(
            np.asarray(values["Psi"], dtype=float),
            np.asarray(values["rho_raw"], dtype=float),
            np.asarray(values["Phi_Z"], dtype=float),
        )
        if not np.allclose(rho, reconstructed, rtol=2.0e-12, atol=2.0e-13):
            raise ValueError(
                "saved rho does not equal the canonical whitening of "
                "Psi/rho_raw/Phi_Z"
            )
        spectral_array = np.asarray(values["rho_spectral_norm"]).reshape(-1)
        minimum_array = np.asarray(values["min_eig_joint_innovation"]).reshape(-1)
        if spectral_array.size != 1 or minimum_array.size != 1:
            raise ValueError(
                "rho_spectral_norm and min_eig_joint_innovation must be scalars"
            )
        recorded_spectral = float(spectral_array[0])
        recorded_minimum = float(minimum_array[0])
        if not np.isfinite(recorded_spectral) or not np.isfinite(recorded_minimum):
            raise ValueError(
                "rho_spectral_norm and min_eig_joint_innovation must be finite"
            )
        if not np.isclose(
            recorded_spectral,
            diagnostics["rho_spectral_norm"],
            rtol=2.0e-12,
            atol=2.0e-13,
        ):
            raise ValueError("saved rho_spectral_norm is inconsistent with rho")
        if not np.isclose(
            recorded_minimum,
            diagnostics["min_eig"],
            rtol=2.0e-12,
            atol=2.0e-13,
        ):
            raise ValueError(
                "saved min_eig_joint_innovation is inconsistent with rho"
            )

    if expected_rho is not None and not np.allclose(
        rho, np.asarray(expected_rho, dtype=float), rtol=2.0e-12, atol=2.0e-13
    ):
        raise ValueError(
            "saved rho differs from the market generated by the current "
            "canonical convention"
        )
    return diagnostics


def validate_market_snapshot(
    values: Mapping[str, Any],
    *,
    expected: Optional[Mapping[str, Any]] = None,
    require_canonical_metadata: bool = False,
) -> Dict[str, Any]:
    """Validate all economic identities in a saved Liu market snapshot."""

    required = (
        "K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0", "lam0",
        "X_min", "X_max", "eta", "gamma", "r", "tau_max", "W_min", "W_max",
        "market_seed",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"market snapshot is missing fields {missing}")
    arrays = {
        key: np.asarray(values[key], dtype=float)
        for key in required
    }
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("market snapshot contains NaN or infinity")

    xbar = arrays["xbar"].reshape(-1)
    lam0 = arrays["lam0"].reshape(-1)
    eta = arrays["eta"].reshape(-1)
    n_assets = int(lam0.size)
    m_states = int(xbar.size)
    if n_assets <= 0 or m_states <= 0:
        raise ValueError("market snapshot must have positive asset/state dimensions")
    shapes = {
        "K": (m_states, m_states),
        "SigmaX": (m_states, m_states),
        "rho": (n_assets, m_states),
        "Lam": (n_assets, m_states),
        "Q": (m_states, m_states),
        "Gamma": (n_assets, m_states),
        "k0": (m_states,),
        "X_min": (m_states,),
        "X_max": (m_states,),
        "eta": (m_states,),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(
                f"market snapshot {key} has shape {arrays[key].shape}, expected {shape}"
            )
    for key in ("gamma", "r", "tau_max", "W_min", "W_max", "market_seed"):
        if arrays[key].size != 1:
            raise ValueError(f"market snapshot {key} must be scalar")

    if not np.allclose(
        arrays["Q"],
        arrays["SigmaX"] @ arrays["SigmaX"].T,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("market snapshot Q != SigmaX @ SigmaX.T")
    if not np.allclose(
        arrays["Gamma"],
        arrays["rho"] @ arrays["SigmaX"].T,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("market snapshot Gamma != rho @ SigmaX.T")
    if not np.allclose(
        arrays["k0"],
        arrays["K"] @ xbar,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("market snapshot k0 != K @ xbar")
    q_symmetric = 0.5 * (arrays["Q"] + arrays["Q"].T)
    min_eig_q = float(np.linalg.eigvalsh(q_symmetric)[0])
    if min_eig_q <= 0.0:
        raise ValueError(
            f"market snapshot Q is not positive definite: min eigenvalue={min_eig_q:.3e}"
        )
    if np.any(eta <= 0.0) or np.any(arrays["X_max"] <= arrays["X_min"]):
        raise ValueError("market snapshot has invalid state scales or bounds")
    gamma = float(arrays["gamma"].reshape(-1)[0])
    horizon = float(arrays["tau_max"].reshape(-1)[0])
    w_min = float(arrays["W_min"].reshape(-1)[0])
    w_max = float(arrays["W_max"].reshape(-1)[0])
    if gamma <= 0.0 or abs(gamma - 1.0) < 1.0e-12:
        raise ValueError("market snapshot requires CRRA gamma>0 and gamma!=1")
    if horizon <= 0.0 or w_min <= 0.0 or w_max <= w_min:
        raise ValueError("market snapshot has invalid horizon or wealth bounds")
    market_seed = float(arrays["market_seed"].reshape(-1)[0])
    if not market_seed.is_integer():
        raise ValueError("market snapshot market_seed must be integer-valued")

    rho_diagnostics = validate_rho_snapshot(
        values,
        expected_rho=(None if expected is None else np.asarray(expected["rho"])),
        require_canonical_metadata=require_canonical_metadata,
    )

    if expected is not None:
        missing_expected = [key for key in expected if key not in values]
        if missing_expected:
            raise ValueError(
                f"saved market snapshot is missing expected fields {missing_expected}"
            )
        for key, expected_value in expected.items():
            actual = np.asarray(values[key])
            target = np.asarray(expected_value)
            if actual.shape != target.shape:
                raise ValueError(
                    f"saved market field {key} has shape {actual.shape}, "
                    f"expected {target.shape}"
                )
            if actual.dtype.kind in {"U", "S"} or target.dtype.kind in {"U", "S"}:
                matches = np.array_equal(actual.astype(str), target.astype(str))
            else:
                matches = np.allclose(
                    np.asarray(actual, dtype=float),
                    np.asarray(target, dtype=float),
                    rtol=2.0e-12,
                    atol=2.0e-13,
                )
            if not matches:
                raise ValueError(
                    f"saved market field {key} differs from the current run configuration"
                )

    has_metadata = "rho_convention" in values
    return {
        **rho_diagnostics,
        "min_eig_Q": min_eig_q,
        "n_assets": n_assets,
        "m_states": m_states,
        "market_schema_version": (
            MARKET_SCHEMA_VERSION if has_metadata else None
        ),
        "rho_convention": (
            RHO_CONVENTION if has_metadata else "legacy_unlabeled"
        ),
    }


# ----------------------------
# Core sampling
# ----------------------------

def sample_spd_correlation(
    dim: int,
    rho_max: Optional[float] = 0.8,
    kappa_max: float = 30.0,
    eig_min: float = 1e-6,
    seed: Optional[int] = None,
    wishart_df: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Sample an SPD correlation matrix R (dim x dim), then apply SPD-preserving shrinkage:
        R = (1-alpha) R0 + alpha I
    to satisfy:
        - max |rho_ij| <= rho_max    (if rho_max is not None)
        - cond(R) <= kappa_max
        - min_eig(R) >= eig_min

    Returns R and a diagnostics dict.
    """
    rng = np.random.default_rng(seed)

    if wishart_df is None:
        wishart_df = max(dim + 5, int(1.5 * dim))

    X = rng.standard_normal(size=(wishart_df, dim))
    S0 = (X.T @ X) / float(wishart_df)
    R0 = _to_correlation(S0)

    alpha_kappa = _find_alpha_for_kappa(R0, kappa_max=kappa_max)

    alpha_rho = 0.0
    if rho_max is not None:
        max_rho0 = _max_abs_offdiag(R0)
        if max_rho0 > rho_max and max_rho0 > 1e-12:
            alpha_rho = 1.0 - (rho_max / max_rho0)
            alpha_rho = float(np.clip(alpha_rho, 0.0, 1.0 - 1e-12))

    alpha_eig = _alpha_for_min_eig(R0, eig_min=eig_min)

    alpha = max(alpha_kappa, alpha_rho, alpha_eig)
    I = np.eye(dim)
    R = (1.0 - alpha) * R0 + alpha * I
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    w = _eigvalsh(R)
    diag = {
        "alpha_used": float(alpha),
        "cond": float(w[-1] / max(w[0], 1e-18)),
        "min_eig": float(w[0]),
        "max_abs_rho": float(_max_abs_offdiag(R)),
        "wishart_df": float(wishart_df),
    }
    return R, diag


def generate_ou_state_params(
    k: int,
    *,
    # Mean reversion rates (diagonal OU by default)
    kappa_range: Tuple[float, float] = (0.5, 2.0),
    # Long-run mean
    theta_range: Tuple[float, float] = (-0.5, 0.5),
    # State diffusion scale (used if eta is not provided)
    eta_range: Tuple[float, float] = (0.05, 0.30),
    # If provided, uses sigma_Z = diag(eta)
    eta: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Simple OU state dynamics sampler:
        dZ = -K (Z - theta) dt + Sigma_Z dW_Z

    Returns:
        K      : (k,k) mean reversion matrix (diagonal)
        theta  : (k,)  long-run mean
        SigmaZ : (k,k) diffusion matrix (diagonal)
        eta    : (k,)  diagonal entries of SigmaZ
    """
    if k <= 0:
        raise ValueError("k must be positive for OU state params")

    rng = np.random.default_rng(seed)

    kap_lo, kap_hi = kappa_range
    if not (0.0 < kap_lo <= kap_hi):
        raise ValueError("kappa_range must satisfy 0 < lo <= hi")
    kappa_diag = rng.uniform(kap_lo, kap_hi, size=(k,)).astype(float)
    K = np.diag(kappa_diag)

    th_lo, th_hi = theta_range
    theta = rng.uniform(th_lo, th_hi, size=(k,)).astype(float)

    if eta is None:
        et_lo, et_hi = eta_range
        if not (0.0 < et_lo <= et_hi):
            raise ValueError("eta_range must satisfy 0 < lo <= hi")
        eta = rng.uniform(et_lo, et_hi, size=(k,)).astype(float)
    else:
        eta = np.asarray(eta, dtype=float).reshape(k,)

    SigmaZ = np.diag(eta)
    return {"K": K, "theta": theta, "SigmaZ": SigmaZ, "eta": eta}


def generate_alpha_loading(
    n: int,
    k: int,
    *,
    alpha_scale: float = 1.0,
    alpha_sparsity: float = 0.0,
    seed: Optional[int] = None,
    dist: str = "normal",
    dirichlet_concentration: float = 1.0,
) -> np.ndarray:
    """
    Sample a linear loading matrix alpha (n x k) used in KO-style models, e.g.

        (mu(t, X) - r*1) = diag(sigma) * (alpha @ X)

    where X is a k-dimensional state (often modeled as OU), and each asset i has
    loading vector alpha[i, :].

    This function supports two sampling modes:

    1) dist="normal" (default, backward-compatible)
       - Entries ~ Normal(0, alpha_scale / sqrt(k))
       - Optional hard sparsity by setting a fraction of entries to 0

    2) dist="dirichlet"
       - Each row is sampled from a Dirichlet distribution (nonnegative, sums to 1)
       - Then scaled so that row-sum == alpha_scale
       - Optional hard sparsity by zeroing random entries per row and renormalizing

       This is a very convenient way to *control the overall scale* of risk premia
       without needing extra post-hoc normalization: because alpha is nonnegative
       and each row has a fixed L1 norm (row-sum), alpha @ X cannot explode purely
       due to alpha's magnitude.

    Parameters
    ----------
    n, k : int
        Number of assets and state dimensions.

    alpha_scale : float
        Controls magnitude.
        - normal: standard deviation scale (alpha_scale / sqrt(k))
        - dirichlet: row-sum after scaling (each row sums to alpha_scale)

    alpha_sparsity : float in [0, 1)
        Fraction of entries set exactly to 0 (hard sparsity). For dirichlet mode,
        after masking we renormalize each row to keep the row-sum fixed.

    dist : {"normal", "dirichlet"}
        Sampling distribution for alpha.

    dirichlet_concentration : float > 0
        Concentration parameter for Dirichlet. Smaller (<1) yields sparser,
        more "one-hot-like" rows; larger (>1) yields more uniform rows.

    Returns
    -------
    alpha : ndarray, shape (n, k)
    """
    if k <= 0:
        raise ValueError("k must be positive to generate alpha loading")
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 <= alpha_sparsity < 1.0):
        raise ValueError("alpha_sparsity must be in [0,1)")

    rng = np.random.default_rng(seed)
    dist = str(dist).lower().strip()

    if dist in ("normal", "gaussian"):
        # Backward-compatible behavior
        scale = float(alpha_scale / max(np.sqrt(k), 1e-12))
        A = rng.normal(loc=0.0, scale=scale, size=(n, k)).astype(float)

        if alpha_sparsity > 0.0:
            mask = rng.uniform(size=(n, k)) < alpha_sparsity
            A[mask] = 0.0
        return A

    if dist in ("dirichlet", "dir"):
        if dirichlet_concentration <= 0.0:
            raise ValueError("dirichlet_concentration must be > 0")

        # Base Dirichlet: rows sum to 1
        conc = np.full((k,), float(dirichlet_concentration), dtype=float)
        A = rng.dirichlet(conc, size=n).astype(float)  # (n, k), nonnegative, rows sum to 1

        if alpha_sparsity > 0.0:
            # Hard-mask some entries and renormalize each row.
            mask = rng.uniform(size=(n, k)) < alpha_sparsity
            A = A.copy()
            A[mask] = 0.0

            row_sums = A.sum(axis=1)
            zero_rows = np.where(row_sums <= 1e-15)[0]
            if zero_rows.size > 0:
                # Ensure every row keeps at least one nonzero entry
                for i in zero_rows:
                    j = int(rng.integers(0, k))
                    A[i, j] = 1.0

            # Renormalize to make each row sum to 1 again
            A = A / A.sum(axis=1, keepdims=True)

        # Scale rows: row-sum becomes alpha_scale
        A = float(alpha_scale) * A
        return A

    raise ValueError(f"Unknown dist={dist!r}. Use 'normal' or 'dirichlet'.")
@dataclass(frozen=True)
class JointMarketParams:
    """
    Container for joint (assets + states) parameters.

    - n assets, k states
    - Joint Brownian correlation: C (n+k)x(n+k)
    - Blocks:
        Psi   = C[:n, :n]   (asset shock correlation)
        Phi_Z = C[n:, n:]   (state shock correlation)
        rho_Z = C[:n, n:]   (raw asset-state cross block)
        rho_canonical = Psi^{-1/2} rho_Z Phi_Z^{-1/2}, the cross-correlation
                        used when both Brownian covariance blocks are identity

    - Volatility scales:
        sigma (n,), eta (k,)

    - Covariance blocks (after scaling):
        Sigma_RR = diag(sigma) Psi diag(sigma)
        Sigma_ZZ = diag(eta)   Phi_Z diag(eta)
        Sigma_RZ = diag(sigma) rho_Z diag(eta)

    - Stabilized asset covariance (for safe precision usage):
        Sigma_RR_safe = Sigma_RR + delta_asset I

    Optional (KO-style):
      - OU state params: K (k,k), theta (k,), SigmaZ (k,k)
      - alpha loading: alpha (n,k)
    """
    n: int
    k: int

    # Correlation blocks
    C: np.ndarray
    Psi: np.ndarray
    Phi_Z: Optional[np.ndarray]
    rho_Z: Optional[np.ndarray]
    rho_canonical: Optional[np.ndarray]

    # Volatility scales
    sigma: np.ndarray
    eta: Optional[np.ndarray]

    # Covariance blocks
    Sigma_RR: np.ndarray
    Sigma_ZZ: Optional[np.ndarray]
    Sigma_RZ: Optional[np.ndarray]

    # Stabilized asset covariance + Cholesky factor
    Sigma_RR_safe: np.ndarray
    chol_Sigma_RR_safe: np.ndarray
    delta_asset: float

    # Optional OU + loadings
    K: Optional[np.ndarray]
    theta: Optional[np.ndarray]
    SigmaZ: Optional[np.ndarray]
    alpha: Optional[np.ndarray]

    # Diagnostics
    diag: Dict[str, Any]


def generate_joint_market_params(
    n: int,
    k: int = 0,
    *,
    # Joint correlation constraints
    rho_max: Optional[float] = 1.0,
    kappa_max: float = 30.0,
    eig_min: float = 1e-6,
    wishart_df: Optional[int] = None,
    # Volatility ranges
    sigma_range: Tuple[float, float] = (0.10, 0.50),
    eta_range: Tuple[float, float] = (0.3, 0.5),
    # Asset covariance stabilization
    asset_kappa_max: Optional[float] = 200.0,
    delta_rel: float = 1e-4,
    # Optional KO-style state dynamics + loadings
    sample_ou: bool = True,
    kappa_range: Tuple[float, float] = (0.5, 2.0),
    theta_range: Tuple[float, float] = (0.2, 0.4),
    sample_alpha: bool = True,
    alpha_scale: float = 1.0,
    alpha_sparsity: float = 0.0,
    alpha_dist: str = "normal",
    dirichlet_concentration: float = 1.0,
    # RNG
    seed: Optional[int] = None,
) -> JointMarketParams:
    """
    Generate numerically stable joint market parameters.

    - Samples a joint (n+k)x(n+k) correlation for Brownian shocks.
    - Samples asset vols sigma and (if k>0) state vols eta.
    - Builds covariance blocks.
    - Stabilizes the asset covariance with a ridge so that
      cond(Sigma_RR_safe) <= asset_kappa_max (if asset_kappa_max is not None).
    - Optionally samples OU state dynamics and alpha loadings.
      * alpha_dist='normal'   : Gaussian loadings (original behavior).
      * alpha_dist='dirichlet': nonnegative rows with fixed row-sum (scale control).

    If k == 0:
      - C == Psi (n x n)
      - Phi_Z, rho_Z, eta, Sigma_ZZ, Sigma_RZ, K, theta, SigmaZ, alpha are None
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0:
        raise ValueError("k must be >= 0")

    dim = n + k
    C, diag_C = sample_spd_correlation(
        dim=dim,
        rho_max=rho_max,
        kappa_max=kappa_max,
        eig_min=eig_min,
        seed=seed,
        wishart_df=wishart_df,
    )

    Psi = C[:n, :n].copy()

    Phi_Z = None
    rho_Z = None
    rho_canonical = None
    if k > 0:
        Phi_Z = C[n:, n:].copy()
        rho_Z = C[:n, n:].copy()
        rho_canonical = canonicalize_cross_correlation(Psi, rho_Z, Phi_Z)

    rng = np.random.default_rng(seed + 1 if seed is not None else None)

    sig_lo, sig_hi = sigma_range
    if not (0.0 < sig_lo <= sig_hi):
        raise ValueError("sigma_range must satisfy 0 < lo <= hi")
    sigma = rng.uniform(sig_lo, sig_hi, size=(n,)).astype(float)

    eta = None
    if k > 0:
        eta_lo, eta_hi = eta_range
        if not (0.0 < eta_lo <= eta_hi):
            raise ValueError("eta_range must satisfy 0 < lo <= hi")
        eta = rng.uniform(eta_lo, eta_hi, size=(k,)).astype(float)

    Dsig = np.diag(sigma)
    Sigma_RR = Dsig @ Psi @ Dsig

    Sigma_ZZ = None
    Sigma_RZ = None
    if k > 0:
        Deta = np.diag(eta)
        Sigma_ZZ = Deta @ Phi_Z @ Deta
        Sigma_RZ = Dsig @ rho_Z @ Deta

    # Stabilize asset covariance: Sigma_RR_safe = Sigma_RR + delta I
    Sigma_RR = 0.5 * (Sigma_RR + Sigma_RR.T)
    wR = _eigvalsh(Sigma_RR)
    lam_min, lam_max = float(wR[0]), float(wR[-1])

    avg_var = float(np.mean(np.diag(Sigma_RR)))
    delta_base = float(delta_rel * max(avg_var, 1e-12))

    delta_needed = 0.0
    if asset_kappa_max is not None:
        delta_needed = _ridge_for_target_kappa(lam_min=lam_min, lam_max=lam_max, kappa_target=asset_kappa_max)

    delta_asset = float(max(delta_base, delta_needed))

    Sigma_RR_safe = Sigma_RR + delta_asset * np.eye(n)
    Sigma_RR_safe = 0.5 * (Sigma_RR_safe + Sigma_RR_safe.T)

    # Cholesky factor (used for stable solves, avoids explicit inverse)
    try:
        chol = np.linalg.cholesky(Sigma_RR_safe)
    except np.linalg.LinAlgError:
        # As a last resort, increase ridge a bit
        bump = 10.0 * (delta_asset + 1e-12)
        Sigma_RR_safe = Sigma_RR_safe + bump * np.eye(n)
        chol = np.linalg.cholesky(Sigma_RR_safe)
        delta_asset += bump

    # Optional OU and alpha
    K = None
    theta = None
    SigmaZ = None
    alpha = None
    if k > 0 and sample_ou:
        ou = generate_ou_state_params(
            k,
            kappa_range=kappa_range,
            theta_range=theta_range,
            eta_range=eta_range,
            eta=eta,
            seed=(seed + 2 if seed is not None else None),
        )
        K, theta, SigmaZ = ou["K"], ou["theta"], ou["SigmaZ"]

    if k > 0 and sample_alpha:
        alpha = generate_alpha_loading(
            n,
            k,
            alpha_scale=alpha_scale,
            alpha_sparsity=alpha_sparsity,
            dist=alpha_dist,
            dirichlet_concentration=dirichlet_concentration,
            seed=(seed + 3 if seed is not None else None),
        )

    diag_out: Dict[str, Any] = {
        "joint_corr": diag_C,
        "identity_joint_corr": (
            identity_block_correlation_diagnostics(rho_canonical)
            if rho_canonical is not None else None
        ),
        "rho_convention": RHO_CONVENTION,
        "asset_sigma_range": sigma_range,
        "state_eta_range": (eta_range if k > 0 else None),
        "asset_cond_raw": float(lam_max / max(lam_min, 1e-18)),
        "asset_cond_safe": float(_cond_spd(Sigma_RR_safe)),
        "delta_asset": float(delta_asset),
    }

    return JointMarketParams(
        n=n,
        k=k,
        C=C,
        Psi=Psi,
        Phi_Z=Phi_Z,
        rho_Z=rho_Z,
        rho_canonical=rho_canonical,
        sigma=sigma,
        eta=eta,
        Sigma_RR=Sigma_RR,
        Sigma_ZZ=Sigma_ZZ,
        Sigma_RZ=Sigma_RZ,
        Sigma_RR_safe=Sigma_RR_safe,
        chol_Sigma_RR_safe=chol,
        delta_asset=delta_asset,
        K=K,
        theta=theta,
        SigmaZ=SigmaZ,
        alpha=alpha,
        diag=diag_out,
    )


def cholesky_solve(chol: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve (chol @ chol.T) x = b for x, where chol is lower-triangular Cholesky.
    Equivalent to SPD solve without forming inverse.
    """
    y = np.linalg.solve(chol, b)
    x = np.linalg.solve(chol.T, y)
    return x


def to_torch(params: JointMarketParams, *, device: str = "cpu", dtype: str = "float32") -> Dict[str, Any]:
    """
    Optional helper: convert numpy arrays to torch tensors.
    Keeps torch as an optional dependency.
    """
    import torch  # optional import

    dt = getattr(torch, dtype)
    out: Dict[str, Any] = {"n": params.n, "k": params.k, "diag": params.diag}

    def t(x):
        if x is None:
            return None
        return torch.as_tensor(x, dtype=dt, device=device)

    out.update(
        C=t(params.C),
        Psi=t(params.Psi),
        Phi_Z=t(params.Phi_Z),
        rho_Z=t(params.rho_Z),
        rho_canonical=t(params.rho_canonical),
        sigma=t(params.sigma),
        eta=t(params.eta),
        Sigma_RR=t(params.Sigma_RR),
        Sigma_ZZ=t(params.Sigma_ZZ),
        Sigma_RZ=t(params.Sigma_RZ),
        Sigma_RR_safe=t(params.Sigma_RR_safe),
        chol_Sigma_RR_safe=t(params.chol_Sigma_RR_safe),
        delta_asset=params.delta_asset,
        K=t(params.K),
        theta=t(params.theta),
        SigmaZ=t(params.SigmaZ),
        alpha=t(params.alpha),
    )
    return out
