"""Numerical core for the Merton exact policy-map diagnostic.

The module is deliberately free of PyTorch so the finite-difference reference
solver can be tested independently of the neural-network environment.  For a
fixed consumption/portfolio feedback, the log-wealth equation in remaining
time ``tau = T - t`` is

    u_tau = a u_yy + b u_y - rho u + U(c),

where ``a = 0.5*pi.T@Sigma@pi`` and
``b = r + pi.T@mu - c/exp(y) - a``.  The feedback supplied to
``solve_frozen_policy`` is sampled once from the source checkpoint and is
never updated from the finite-difference solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


Array = np.ndarray
PolicyFn = Callable[[float, Array], Tuple[Array, Array, Mapping[str, float]]]


@dataclass(frozen=True)
class MertonProblem:
    horizon: float
    y_min: float
    y_max: float
    gamma: float
    discount: float
    bequest: float
    risk_free: float
    mu_excess: Array
    sigma: Array

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu_excess, dtype=np.float64).reshape(-1)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        if sigma.shape != (mu.size, mu.size):
            raise ValueError("sigma must have shape (n_assets, n_assets)")
        if not np.allclose(sigma, sigma.T, rtol=1e-11, atol=1e-13):
            raise ValueError("sigma must be symmetric")
        if float(np.linalg.eigvalsh(sigma).min()) <= 0.0:
            raise ValueError("sigma must be positive definite")
        if not self.horizon > 0.0:
            raise ValueError("horizon must be positive")
        if not self.y_min < self.y_max:
            raise ValueError("y_min must be smaller than y_max")
        if not self.gamma > 0.0 or np.isclose(self.gamma, 1.0):
            raise ValueError("gamma must be positive and different from one")
        if not self.bequest > 0.0:
            raise ValueError("bequest must be positive")
        object.__setattr__(self, "mu_excess", mu)
        object.__setattr__(self, "sigma", sigma)

    @property
    def n_assets(self) -> int:
        return int(self.mu_excess.size)

    @property
    def sigma_inv_mu(self) -> Array:
        return np.linalg.solve(self.sigma, self.mu_excess)

    @property
    def theta(self) -> float:
        return float(self.mu_excess @ self.sigma_inv_mu)

    @property
    def nu(self) -> float:
        q = 1.0 - self.gamma
        return float(
            self.discount / self.gamma
            - q * (
                self.risk_free / self.gamma
                + self.theta / (2.0 * self.gamma * self.gamma)
            )
        )


@dataclass(frozen=True)
class FDGrid:
    y_min: float
    y_max: float
    ny: int
    nt: int

    def __post_init__(self) -> None:
        if self.ny < 7:
            raise ValueError("ny must be at least 7")
        if self.nt < 2:
            raise ValueError("nt must be at least 2")
        if not self.y_min < self.y_max:
            raise ValueError("FD y_min must be smaller than y_max")


@dataclass
class FDDiagnostics:
    # ``diffusion`` is the coefficient a=0.5*pi^T Sigma pi multiplying u_yy
    # in the transformed PDE.  Keep its historical field names for artifact
    # compatibility, but also expose the portfolio variance explicitly so the
    # ellipticity diagnostic agrees with the trainer/paper convention.
    min_diffusion: float = float("inf")
    max_diffusion: float = 0.0
    min_diffusion_variance: float = float("inf")
    max_diffusion_variance: float = 0.0
    max_peclet: float = 0.0
    upwind_points: int = 0
    coefficient_points: int = 0
    max_linear_residual: float = 0.0
    policy_sums: Dict[str, float] = field(default_factory=dict)
    policy_points: float = 0.0

    def update_policy(self, values: Mapping[str, float]) -> None:
        points = float(values.get("points", 0.0))
        self.policy_points += points
        for key, value in values.items():
            if key == "points":
                continue
            self.policy_sums[key] = self.policy_sums.get(key, 0.0) + float(value)

    def as_dict(self) -> Dict[str, float]:
        out = {
            # Backward-compatible aliases for the PDE coefficient.
            "min_diffusion": float(self.min_diffusion),
            "max_diffusion": float(self.max_diffusion),
            "min_diffusion_coefficient": float(self.min_diffusion),
            "max_diffusion_coefficient": float(self.max_diffusion),
            "min_diffusion_variance": float(self.min_diffusion_variance),
            "max_diffusion_variance": float(self.max_diffusion_variance),
            "max_peclet": float(self.max_peclet),
            "upwind_fraction": (
                float(self.upwind_points) / float(self.coefficient_points)
                if self.coefficient_points
                else 0.0
            ),
            "max_linear_residual": float(self.max_linear_residual),
            "policy_points": float(self.policy_points),
        }
        denom = self.policy_points if self.policy_points > 0.0 else 1.0
        out.update({f"policy_{key}": float(value) / denom for key, value in self.policy_sums.items()})
        return out


@dataclass
class FDSolution:
    tau: Array
    y: Array
    value: Array
    diagnostics: FDDiagnostics


def crra_closed_form(problem: MertonProblem, tau: Array, y: Array) -> Tuple[Array, Array, Array]:
    """Return ``(v, v_y, v_yy)`` for the finite-horizon Merton solution.

    ``bequest`` is the coefficient in ``g(w)=bequest*w^(1-gamma)/(1-gamma)``.
    The formula therefore uses ``bequest**(1/gamma)`` at the terminal point.
    """
    tau_arr, y_arr = np.broadcast_arrays(
        np.asarray(tau, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    q = 1.0 - problem.gamma
    nu = problem.nu
    if abs(nu) < 1e-12:
        scale = problem.bequest ** (1.0 / problem.gamma) + tau_arr
    else:
        scale = (
            1.0 / nu
            + (problem.bequest ** (1.0 / problem.gamma) - 1.0 / nu)
            * np.exp(-nu * tau_arr)
        )
    amplitude = np.power(scale, problem.gamma)
    exp_qy = np.exp(q * y_arr)
    value = amplitude * exp_qy / q
    value_y = amplitude * exp_qy
    value_yy = q * amplitude * exp_qy
    return value, value_y, value_yy


def crra_terminal(problem: MertonProblem, y: Array) -> Array:
    q = 1.0 - problem.gamma
    return problem.bequest * np.exp(q * np.asarray(y, dtype=np.float64)) / q


def crra_utility(c: Array, gamma: float) -> Array:
    c_arr = np.asarray(c, dtype=np.float64)
    if np.any(c_arr <= 0.0):
        raise ValueError("frozen consumption must be strictly positive")
    return np.power(c_arr, 1.0 - gamma) / (1.0 - gamma)


def analytic_optimal_policy(problem: MertonProblem) -> PolicyFn:
    pi_star = problem.sigma_inv_mu / problem.gamma

    def policy(tau: float, y: Array) -> Tuple[Array, Array, Mapping[str, float]]:
        tau_vec = np.full_like(np.asarray(y, dtype=np.float64), float(tau))
        _v, vy, _vyy = crra_closed_form(problem, tau_vec, y)
        wealth = np.exp(y)
        consumption = np.power(vy / wealth, -1.0 / problem.gamma)
        pi = np.broadcast_to(pi_star.reshape(1, -1), (wealth.size, pi_star.size)).copy()
        return consumption, pi, {"points": float(wealth.size)}

    return policy


def policy_coefficients(
    problem: MertonProblem,
    y: Array,
    consumption: Array,
    portfolio: Array,
) -> Tuple[Array, Array, Array]:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    c = np.asarray(consumption, dtype=np.float64).reshape(-1)
    pi = np.asarray(portfolio, dtype=np.float64)
    if pi.shape != (y_arr.size, problem.n_assets):
        raise ValueError("portfolio has an incompatible shape")
    if c.shape != (y_arr.size,):
        raise ValueError("consumption has an incompatible shape")
    variance = np.einsum("bi,ij,bj->b", pi, problem.sigma, pi, optimize=True)
    if float(variance.min()) < -1e-12:
        raise ValueError("pi.T Sigma pi became negative")
    variance = np.maximum(variance, 0.0)
    diffusion = 0.5 * variance
    drift = (
        problem.risk_free
        + pi @ problem.mu_excess
        - c / np.exp(y_arr)
        - diffusion
    )
    source = crra_utility(c, problem.gamma)
    return diffusion, drift, source


def stencil_coefficients(
    diffusion: Array,
    drift: Array,
    discount: float,
    dy: float,
    scheme: str,
    peclet_limit: float,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Return lower/diag/upper coefficients and Peclet/upwind indicators."""
    a = np.asarray(diffusion, dtype=np.float64)
    b = np.asarray(drift, dtype=np.float64)
    denom = np.maximum(2.0 * a, np.finfo(np.float64).tiny)
    peclet = np.abs(b) * dy / denom
    if scheme == "central":
        use_upwind = np.zeros_like(a, dtype=bool)
    elif scheme == "monotone":
        use_upwind = np.ones_like(a, dtype=bool)
    elif scheme == "adaptive":
        use_upwind = (a <= 0.0) | (peclet > float(peclet_limit))
    else:
        raise ValueError(f"unknown drift scheme: {scheme}")

    lower_c = a / (dy * dy) - b / (2.0 * dy)
    upper_c = a / (dy * dy) + b / (2.0 * dy)
    diag_c = -2.0 * a / (dy * dy) - float(discount)

    # Monotone upwind for u_tau = b*u_y: positive b uses the forward
    # difference and negative b the backward difference.
    lower_u = a / (dy * dy) + np.maximum(-b, 0.0) / dy
    upper_u = a / (dy * dy) + np.maximum(b, 0.0) / dy
    diag_u = -lower_u - upper_u - float(discount)

    lower = np.where(use_upwind, lower_u, lower_c)
    diag = np.where(use_upwind, diag_u, diag_c)
    upper = np.where(use_upwind, upper_u, upper_c)
    return lower, diag, upper, peclet, use_upwind


def _tridiagonal_matvec(lower: Array, diag: Array, upper: Array, x: Array) -> Array:
    out = diag * x
    out[1:] += lower[1:] * x[:-1]
    out[:-1] += upper[:-1] * x[1:]
    return out


def solve_tridiagonal(lower: Array, diag: Array, upper: Array, rhs: Array) -> Array:
    """Thomas solve with finite-pivot checks."""
    a = np.asarray(lower, dtype=np.float64).copy()
    b = np.asarray(diag, dtype=np.float64).copy()
    c = np.asarray(upper, dtype=np.float64).copy()
    d = np.asarray(rhs, dtype=np.float64).copy()
    n = b.size
    if any(arr.size != n for arr in (a, c, d)):
        raise ValueError("tridiagonal arrays must have equal lengths")
    tiny = 100.0 * np.finfo(np.float64).tiny
    for i in range(1, n):
        if not np.isfinite(b[i - 1]) or abs(b[i - 1]) <= tiny:
            raise np.linalg.LinAlgError(f"invalid tridiagonal pivot at {i - 1}")
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    if not np.isfinite(b[-1]) or abs(b[-1]) <= tiny:
        raise np.linalg.LinAlgError(f"invalid tridiagonal pivot at {n - 1}")
    x = np.empty(n, dtype=np.float64)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if not np.isfinite(b[i]) or abs(b[i]) <= tiny:
            raise np.linalg.LinAlgError(f"invalid tridiagonal pivot at {i}")
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("non-finite tridiagonal solution")
    return x


def _robin_eliminated_operator(
    lower: Array,
    diag: Array,
    upper: Array,
    dy: float,
    exponent: float,
) -> Tuple[Array, Array, Array]:
    """Eliminate second-order homogeneous Robin boundaries from L."""
    lo = np.asarray(lower, dtype=np.float64).copy()
    di = np.asarray(diag, dtype=np.float64).copy()
    up = np.asarray(upper, dtype=np.float64).copy()
    left_den = 3.0 + 2.0 * dy * exponent
    right_den = 3.0 - 2.0 * dy * exponent
    if abs(left_den) < 1e-10 or abs(right_den) < 1e-10:
        raise ValueError("Robin boundary elimination is singular on this grid")
    di[0] += lo[0] * (4.0 / left_den)
    up[0] += lo[0] * (-1.0 / left_den)
    lo[0] = 0.0
    di[-1] += up[-1] * (4.0 / right_den)
    lo[-1] += up[-1] * (-1.0 / right_den)
    up[-1] = 0.0
    return lo, di, up


def _apply_robin_boundaries(value: Array, dy: float, exponent: float) -> None:
    left_den = 3.0 + 2.0 * dy * exponent
    right_den = 3.0 - 2.0 * dy * exponent
    value[0] = (4.0 * value[1] - value[2]) / left_den
    value[-1] = (4.0 * value[-2] - value[-3]) / right_den


def solve_frozen_policy(
    problem: MertonProblem,
    policy: PolicyFn,
    grid: FDGrid,
    *,
    theta_method: float = 0.5,
    rannacher_steps: int = 2,
    drift_scheme: str = "adaptive",
    peclet_limit: float = 1.0,
    boundary: str = "robin",
) -> FDSolution:
    """Solve one frozen-policy equation on a uniform log-wealth grid.

    ``rannacher_steps`` is retained as the public argument name, but denotes
    full-``dt`` backward-Euler damping steps before the theta method resumes;
    it is not the classical two-half-step Rannacher construction.

    ``boundary='robin'`` imposes the homogeneous CRRA scaling relation
    ``u_y=(1-gamma)u`` without injecting the exact value amplitude.
    ``boundary='exact-dirichlet'`` is available as an audit closure.  The
    reported exact-map ratio should be checked across FD domain sizes (and,
    when practical, both closures).
    """
    if not 0.5 <= float(theta_method) <= 1.0:
        raise ValueError("theta_method must lie in [0.5, 1]")
    if rannacher_steps < 0:
        raise ValueError("rannacher_steps must be nonnegative")
    boundary = str(boundary).replace("_", "-").lower()
    if boundary not in {"robin", "exact-dirichlet"}:
        raise ValueError("boundary must be robin or exact-dirichlet")

    tau_grid = np.linspace(0.0, problem.horizon, grid.nt + 1, dtype=np.float64)
    y_grid = np.linspace(grid.y_min, grid.y_max, grid.ny, dtype=np.float64)
    dt = float(tau_grid[1] - tau_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    values = np.empty((grid.nt + 1, grid.ny), dtype=np.float64)
    values[0] = crra_terminal(problem, y_grid)
    diagnostics = FDDiagnostics()

    if boundary == "robin":
        _apply_robin_boundaries(values[0], dy, 1.0 - problem.gamma)

    for step in range(grid.nt):
        tau_old = float(tau_grid[step])
        tau_new = float(tau_grid[step + 1])
        tau_mid = 0.5 * (tau_old + tau_new)
        consumption, portfolio, policy_diag = policy(tau_mid, y_grid)
        diagnostics.update_policy(policy_diag)
        diffusion, drift, source = policy_coefficients(
            problem, y_grid, consumption, portfolio
        )
        if float(diffusion.min()) < -1e-12:
            raise ValueError("negative frozen-policy diffusion")
        diagnostics.min_diffusion = min(diagnostics.min_diffusion, float(diffusion.min()))
        diagnostics.max_diffusion = max(diagnostics.max_diffusion, float(diffusion.max()))
        variance = 2.0 * diffusion
        diagnostics.min_diffusion_variance = min(
            diagnostics.min_diffusion_variance, float(variance.min())
        )
        diagnostics.max_diffusion_variance = max(
            diagnostics.max_diffusion_variance, float(variance.max())
        )

        lower_all, diag_all, upper_all, peclet_all, upwind_all = stencil_coefficients(
            diffusion, drift, problem.discount, dy, drift_scheme, peclet_limit
        )
        diagnostics.max_peclet = max(diagnostics.max_peclet, float(np.max(peclet_all)))
        diagnostics.upwind_points += int(np.count_nonzero(upwind_all[1:-1]))
        diagnostics.coefficient_points += int(grid.ny - 2)

        # Unknowns are spatial nodes 1,...,ny-2.
        lower_l = lower_all[1:-1].copy()
        diag_l = diag_all[1:-1].copy()
        upper_l = upper_all[1:-1].copy()
        old = values[step].copy()
        theta = 1.0 if step < int(rannacher_steps) else float(theta_method)

        if boundary == "robin":
            _apply_robin_boundaries(old, dy, 1.0 - problem.gamma)
            lower_eff, diag_eff, upper_eff = _robin_eliminated_operator(
                lower_l, diag_l, upper_l, dy, 1.0 - problem.gamma
            )
            old_int = old[1:-1]
            l_old = _tridiagonal_matvec(lower_eff, diag_eff, upper_eff, old_int)
            rhs = old_int + dt * (1.0 - theta) * l_old + dt * source[1:-1]
            mat_lower = -dt * theta * lower_eff
            mat_diag = 1.0 - dt * theta * diag_eff
            mat_upper = -dt * theta * upper_eff
            new_int = solve_tridiagonal(mat_lower, mat_diag, mat_upper, rhs)
            new = np.empty(grid.ny, dtype=np.float64)
            new[1:-1] = new_int
            _apply_robin_boundaries(new, dy, 1.0 - problem.gamma)
        else:
            old_int = old[1:-1]
            l_old = (
                lower_l * old[:-2]
                + diag_l * old[1:-1]
                + upper_l * old[2:]
            )
            rhs = old_int + dt * (1.0 - theta) * l_old + dt * source[1:-1]
            left_new = float(crra_closed_form(problem, tau_new, grid.y_min)[0])
            right_new = float(crra_closed_form(problem, tau_new, grid.y_max)[0])
            rhs[0] += dt * theta * lower_l[0] * left_new
            rhs[-1] += dt * theta * upper_l[-1] * right_new
            mat_lower = -dt * theta * lower_l
            mat_diag = 1.0 - dt * theta * diag_l
            mat_upper = -dt * theta * upper_l
            mat_lower[0] = 0.0
            mat_upper[-1] = 0.0
            new_int = solve_tridiagonal(mat_lower, mat_diag, mat_upper, rhs)
            new = np.empty(grid.ny, dtype=np.float64)
            new[0] = left_new
            new[-1] = right_new
            new[1:-1] = new_int

        residual = _tridiagonal_matvec(mat_lower, mat_diag, mat_upper, new_int) - rhs
        scale = max(1.0, float(np.max(np.abs(rhs))))
        diagnostics.max_linear_residual = max(
            diagnostics.max_linear_residual,
            float(np.max(np.abs(residual))) / scale,
        )
        if not np.all(np.isfinite(new)):
            raise FloatingPointError(f"non-finite FD value at tau step {step + 1}")
        values[step + 1] = new

    return FDSolution(tau=tau_grid, y=y_grid, value=values, diagnostics=diagnostics)


def evaluate_fd_bundle(solution: FDSolution, tau_eval: Array, y_eval: Array) -> Tuple[Array, Array, Array]:
    """Evaluate ``u, u_y, u_yy`` on a fixed tensor grid.

    Evaluation times must be nested nodes of the FD time grid.  Spatial
    values and derivatives are obtained from one cubic spline per time slice,
    so all refinement/domain variants are compared on identical Q_ev nodes.
    """
    tau = np.asarray(tau_eval, dtype=np.float64).reshape(-1)
    y = np.asarray(y_eval, dtype=np.float64).reshape(-1)
    if tau.size == 0 or y.size == 0:
        raise ValueError("evaluation grids must be nonempty")
    if y[0] < solution.y[0] or y[-1] > solution.y[-1]:
        raise ValueError("evaluation y grid lies outside the FD domain")
    dt = float(solution.tau[1] - solution.tau[0])
    indices = np.rint((tau - solution.tau[0]) / dt).astype(int)
    if np.any(indices < 0) or np.any(indices >= solution.tau.size):
        raise ValueError("evaluation tau grid lies outside the FD time domain")
    if not np.allclose(solution.tau[indices], tau, rtol=0.0, atol=1e-11):
        raise ValueError("evaluation tau nodes must be nested in the FD time grid")
    slices = solution.value[indices]
    spline = CubicSpline(solution.y, slices, axis=1, bc_type="not-a-knot")
    return spline(y, 0), spline(y, 1), spline(y, 2)


def log_to_wealth_derivatives(
    value_y: Array,
    value_yy: Array,
    y: Array,
) -> Tuple[Array, Array]:
    """Convert log-wealth derivatives to ``(V_w, V_ww)``.

    If ``v(t, y) = V(t, exp(y))``, then

    ``V_w = exp(-y) v_y`` and
    ``V_ww = exp(-2y) (v_yy - v_y)``.

    ``y`` may be a scalar, a spatial vector, or a tensor grid as long as it
    broadcasts to the derivative bundle.  The two derivative arrays must
    already have identical shapes; this prevents an accidental cross-product
    broadcast between unrelated bundles.
    """
    value_y_arr = np.asarray(value_y, dtype=np.float64)
    value_yy_arr = np.asarray(value_yy, dtype=np.float64)
    if value_y_arr.shape != value_yy_arr.shape:
        raise ValueError("value_y and value_yy must have identical shapes")
    try:
        y_arr = np.broadcast_to(
            np.asarray(y, dtype=np.float64), value_y_arr.shape
        )
    except ValueError as exc:
        raise ValueError(
            "log-wealth coordinates must broadcast to the derivative bundle"
        ) from exc
    value_w = np.exp(-y_arr) * value_y_arr
    value_ww = np.exp(-2.0 * y_arr) * (value_yy_arr - value_y_arr)
    return value_w, value_ww


def x_norm_components(
    value: Array,
    value_y: Array,
    value_yy: Array,
    reference: Tuple[Array, Array, Array],
    y: Array,
) -> Dict[str, float]:
    """Return the manuscript Merton ``X_ev`` error components.

    Both the candidate and ``reference`` bundles are supplied in log-wealth
    derivatives ``(V, V_y, V_yy)``, since that is what the neural evaluator
    and FD spline produce.  The policy-relevant derivative error is measured
    in the original wealth coordinate:

    ``sup sqrt((Delta V_w)^2 + (Delta V_ww)^2)``.

    The total norm is the value sup-error plus that *joint* derivative sup.
    Raw log-derivative sup-errors remain in the returned dictionary as useful
    numerical diagnostics, but they do not enter ``x_norm``.
    """
    ref_v, ref_y, ref_yy = (np.asarray(item, dtype=np.float64) for item in reference)
    err_v = np.asarray(value, dtype=np.float64) - ref_v
    err_y = np.asarray(value_y, dtype=np.float64) - ref_y
    err_yy = np.asarray(value_yy, dtype=np.float64) - ref_yy
    if not (err_v.shape == err_y.shape == err_yy.shape):
        raise ValueError("value and derivative bundles must have identical shapes")
    err_w, err_ww = log_to_wealth_derivatives(err_y, err_yy, y)
    value_sup = float(np.max(np.abs(err_v)))
    vy_sup = float(np.max(np.abs(err_y)))
    vyy_sup = float(np.max(np.abs(err_yy)))
    vw_sup = float(np.max(np.abs(err_w)))
    vww_sup = float(np.max(np.abs(err_ww)))
    derivative_sup = float(np.max(np.sqrt(err_w * err_w + err_ww * err_ww)))
    return {
        "value_sup": value_sup,
        "vw_sup": vw_sup,
        "vww_sup": vww_sup,
        # Retained for FD/autograd diagnostics; excluded from x_norm.
        "vy_sup": vy_sup,
        "vyy_sup": vyy_sup,
        "derivative_sup": derivative_sup,
        "x_norm": value_sup + derivative_sup,
    }
