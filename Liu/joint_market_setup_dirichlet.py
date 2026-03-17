
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
from typing import Optional, Tuple, Dict, Any

import numpy as np


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
        rho_Z = C[:n, n:]   (asset-state cross correlation)

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
    if k > 0:
        Phi_Z = C[n:, n:].copy()
        rho_Z = C[:n, n:].copy()

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
