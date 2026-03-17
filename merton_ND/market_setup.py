"""
Market Parameter Generator for Multi-Asset Merton Problem
==========================================================
Generates synthetic market parameters with numerical stability guarantees.

Features:
- SPD correlation matrix generation with shrinkage
- Condition number and correlation magnitude controls
- Stable Cholesky-based inverse computation
- Multiple mu_excess generation modes
"""

import numpy as np


def _to_correlation(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert SPD covariance-like matrix to correlation matrix."""
    d = np.sqrt(np.clip(np.diag(S), eps, None))
    Dinv = np.diag(1.0 / d)
    R = Dinv @ S @ Dinv
    # enforce exact symmetry and unit diagonal
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _max_abs_offdiag(A: np.ndarray) -> float:
    B = A.copy()
    np.fill_diagonal(B, 0.0)
    return float(np.max(np.abs(B)))


def _cond_spd(A: np.ndarray, eps: float = 1e-15) -> float:
    # SPD condition number via eigenvalues
    w = np.linalg.eigvalsh(A)
    w = np.clip(w, eps, None)
    return float(w[-1] / w[0])


def _find_alpha_for_kappa(R0: np.ndarray, kappa_max: float, tol: float = 1e-10) -> float:
    """Find minimal alpha in [0,1) such that cond((1-alpha)R0 + alpha I) <= kappa_max."""
    if kappa_max <= 1.0:
        return 1.0  # degenerate request; will return near-identity
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


def sample_spd_correlation(
    n: int,
    rho_max: float = 0.7,
    kappa_max: float = 50.0,
    seed: int | None = None,
    wishart_df: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Generate an SPD correlation matrix R with max |rho_ij| <= rho_max and cond(R) <= kappa_max
    using: R = (1-alpha)R0 + alpha I (SPD-preserving shrinkage).
    Returns (R, alpha_used).
    """
    rng = np.random.default_rng(seed)

    if wishart_df is None:
        # moderate df tends to avoid extremely ill-conditioned matrices
        wishart_df = max(n + 5, 2 * n)

    # Build SPD matrix via random Gaussian factor (Wishart-like)
    X = rng.standard_normal((wishart_df, n))
    S0 = (X.T @ X) / wishart_df  # SPD (with high probability)
    R0 = _to_correlation(S0)

    # alpha needed for correlation magnitude
    off0 = _max_abs_offdiag(R0)
    if off0 <= rho_max:
        alpha_rho = 0.0
    else:
        alpha_rho = 1.0 - (rho_max / off0)  # since offdiag scales by (1-alpha)

    # alpha needed for condition number
    alpha_kappa = _find_alpha_for_kappa(R0, kappa_max=kappa_max)

    alpha = max(alpha_rho, alpha_kappa)
    alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-12))

    I = np.eye(n)
    R = (1.0 - alpha) * R0 + alpha * I
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    # (Optional) sanity checks
    if _max_abs_offdiag(R) > rho_max + 1e-6:
        # if this ever triggers, increase alpha slightly
        slack = _max_abs_offdiag(R) / max(rho_max, 1e-12)
        alpha = min(1.0 - 1e-12, 1.0 - (1.0 - alpha) / slack)
        R = (1.0 - alpha) * R0 + alpha * I
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)

    return R, alpha


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = b with forward/backward solves."""
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x


def generate_synthetic_merton_market(
    n: int,
    gamma: float = 2.0,
    sigma_range: tuple[float, float] = (0.10, 0.30),
    rho_max: float = 0.7,
    kappa_max: float = 50.0,
    delta_rel: float = 1e-4,
    seed: int | None = None,
    mu_mode: str = "pi_target",
    pi_scale: float = 0.5,
    mu_noise_rel: float = 0.0,
) -> dict:
    """
    Generate synthetic parameters for unconstrained multi-asset Merton:
      - sigma (vols), correlation R, covariance Sigma, jittered Sigma_safe
      - mu_excess (excess return vector)
      - pi_star = (1/gamma) * Sigma_safe^{-1} mu_excess
    mu_mode:
      - "pi_target": sample pi_target and set mu_excess = gamma * Sigma_safe * pi_target (+ optional noise)
      - "sharpe": sample per-asset sharpe and set mu_excess_i = sharpe_i * sigma_i (less controlled)
    """
    rng = np.random.default_rng(seed)

    # 1) volatilities
    s_lo, s_hi = sigma_range
    if not (0.0 < s_lo < s_hi):
        raise ValueError("sigma_range must satisfy 0 < low < high")
    sigma = rng.uniform(s_lo, s_hi, size=n)

    # 2) SPD correlation with shrinkage to satisfy rho_max and kappa_max
    R, alpha_used = sample_spd_correlation(
        n=n, rho_max=rho_max, kappa_max=kappa_max, seed=seed
    )

    # 3) covariance
    D = np.diag(sigma)
    Sigma = D @ R @ D

    # 4) ridge/jitter
    avg_var = float(np.mean(np.diag(Sigma)))
    delta = float(delta_rel * avg_var)
    Sigma_safe = Sigma + delta * np.eye(n)

    # Cholesky (guaranteed SPD if delta>0 and Sigma SPD)
    L = np.linalg.cholesky(Sigma_safe)

    # 5) choose mu_excess
    if mu_mode == "pi_target":
        # Sample a reasonable target portfolio and back out mu so pi* is well-scaled.
        pi_target = rng.standard_normal(n)
        pi_target = (pi_scale / (np.linalg.norm(pi_target) + 1e-12)) * pi_target

        mu_excess = gamma * (Sigma_safe @ pi_target)

        if mu_noise_rel > 0.0:
            # Add small noise proportional to sigma to avoid being perfectly constructed.
            mu_excess = mu_excess + (mu_noise_rel * sigma) * rng.standard_normal(n)

    elif mu_mode == "sharpe":
        # Less controlled: per-asset Sharpe ratios (per unit time) times vol.
        # Use moderate Sharpe to avoid huge pi.
        sharpe = rng.uniform(0.05, 0.60, size=n)
        signs = rng.choice([-1.0, 1.0], size=n, p=[0.2, 0.8])  # mostly positive
        mu_excess = signs * sharpe * sigma
        pi_target = None
    else:
        raise ValueError("mu_mode must be 'pi_target' or 'sharpe'")

    # 6) compute pi* via stable solve (no explicit inverse)
    x = cholesky_solve(L, mu_excess)
    pi_star = (1.0 / gamma) * x

    # Diagnostics
    cond_R = _cond_spd(R)
    cond_S = _cond_spd(Sigma_safe)
    max_rho = _max_abs_offdiag(R)
    quad = float(mu_excess.T @ cholesky_solve(L, mu_excess))  # mu^T Sigma^{-1} mu using solve

    out = {
        "n": n,
        "gamma": gamma,
        "sigma": sigma,
        "R": R,
        "Sigma": Sigma,
        "delta": delta,
        "Sigma_safe": Sigma_safe,
        "L": L,
        "mu_excess": mu_excess,
        "pi_star": pi_star,
        "pi_target": pi_target,
        "alpha_used": alpha_used,
        "max_abs_rho": max_rho,
        "cond_R": cond_R,
        "cond_Sigma_safe": cond_S,
        "mu_SigmaInv_mu": quad,
        "pi_norm2": float(np.linalg.norm(pi_star)),
        "pi_norm1": float(np.linalg.norm(pi_star, ord=1)),
        "pi_maxabs": float(np.max(np.abs(pi_star))),
    }
    return out


# --- Example usage (prints diagnostics) ---
if __name__ == "__main__":
    params = generate_synthetic_merton_market(
        n=3,
        gamma=2.0,
        sigma_range=(0.10, 0.25),
        rho_max=0.7,
        kappa_max=30.0,
        delta_rel=1e-4,
        seed=123,
        mu_mode="pi_target",
        pi_scale=0.6,
        mu_noise_rel=0.02,
    )

    print("n =", params["n"])
    print("max_abs_rho =", params["max_abs_rho"])
    print("cond_R =", params["cond_R"])
    print("cond_Sigma_safe =", params["cond_Sigma_safe"])
    print("delta =", params["delta"])
    print("||pi||_2 =", params["pi_norm2"])
    print("max|pi_i| =", params["pi_maxabs"])
