#!/usr/bin/env python3
"""Torch-free numerical core for the Liu/Kim--Omberg M=1 FD audits.

The policy supplied to :func:`solve_frozen_policy` is a *frozen* normalized
volatility feedback ``vartheta = theta / w``.  In log wealth ``y=log(w)`` the
remaining-time equation is

    u_tau = .5 |vartheta|^2 u_yy + (vartheta' Gamma) u_yx
            + .5 Q u_xx
            + (r + vartheta' lambda(x) - .5 |vartheta|^2) u_y
            + (k0 - K x) u_x.

There is no running source term in the terminal-wealth Liu experiment.  The
module intentionally has no PyTorch dependency so the two-dimensional mixed-
derivative finite-difference solver can be tested independently of checkpoint
loading and automatic differentiation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve


Array = np.ndarray
PolicyFn = Callable[[float, Array, Array], Tuple[Array, Mapping[str, float]]]
BoundaryValueFn = Callable[[float, Array, Array], Array]


@dataclass(frozen=True)
class LiuProblem:
    """One-factor multi-asset Liu problem in standardized-return units."""

    horizon: float
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    gamma: float
    risk_free: float
    K: float
    k0: float
    Q: float
    Gamma: Array
    lam0: Array
    Lam: Array

    def __post_init__(self) -> None:
        gamma_vec = np.asarray(self.Gamma, dtype=np.float64).reshape(-1)
        lam0_vec = np.asarray(self.lam0, dtype=np.float64).reshape(-1)
        lam_vec = np.asarray(self.Lam, dtype=np.float64).reshape(-1)
        if not (gamma_vec.size == lam0_vec.size == lam_vec.size and lam0_vec.size > 0):
            raise ValueError("Gamma, lam0, and Lam must be nonempty vectors of equal length")
        if not self.horizon > 0.0:
            raise ValueError("horizon must be positive")
        if not self.y_min < self.y_max or not self.x_min < self.x_max:
            raise ValueError("spatial lower bounds must be smaller than upper bounds")
        if not self.gamma > 0.0 or np.isclose(self.gamma, 1.0):
            raise ValueError("gamma must be positive and different from one")
        if not self.Q > 0.0:
            raise ValueError("Q must be strictly positive")
        if not self.K > 0.0:
            raise ValueError("K must be strictly positive")
        for value, name in ((self.risk_free, "risk_free"), (self.k0, "k0")):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "Gamma", gamma_vec)
        object.__setattr__(self, "lam0", lam0_vec)
        object.__setattr__(self, "Lam", lam_vec)

    @property
    def n_assets(self) -> int:
        return int(self.lam0.size)

    def risk_premium(self, x: Array) -> Array:
        x_arr = np.asarray(x, dtype=np.float64)
        return self.lam0 + x_arr[..., None] * self.Lam


@dataclass(frozen=True)
class FDGrid:
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    ny: int
    nx: int
    nt: int

    def __post_init__(self) -> None:
        if self.ny < 7 or self.nx < 7:
            raise ValueError("ny and nx must both be at least 7")
        if self.nt < 2:
            raise ValueError("nt must be at least 2")
        if not self.y_min < self.y_max or not self.x_min < self.x_max:
            raise ValueError("invalid FD bounds")


@dataclass
class FDDiagnostics:
    min_log_joint_eig: float = float("inf")
    max_log_joint_eig: float = 0.0
    min_original_joint_eig: float = float("inf")
    max_original_joint_eig: float = 0.0
    nonpositive_log_eig_points: int = 0
    coefficient_points: int = 0
    max_peclet_y: float = 0.0
    max_peclet_x: float = 0.0
    upwind_y_points: int = 0
    upwind_x_points: int = 0
    operator_points: int = 0
    max_linear_residual: float = 0.0
    policy_sums: Dict[str, float] = field(default_factory=dict)
    policy_minima: Dict[str, float] = field(default_factory=dict)
    policy_maxima: Dict[str, float] = field(default_factory=dict)
    policy_points: float = 0.0

    def update_policy(self, values: Mapping[str, float]) -> None:
        points = float(values.get("points", 0.0))
        self.policy_points += points
        for key, value in values.items():
            if key == "points":
                continue
            number = float(value)
            if key.endswith("_min"):
                self.policy_minima[key] = min(self.policy_minima.get(key, float("inf")), number)
            elif key.endswith("_max"):
                self.policy_maxima[key] = max(self.policy_maxima.get(key, -float("inf")), number)
            else:
                # Count-like values are accumulated and divided by the number
                # of policy sample points in ``as_dict``.  Extrema must never
                # be averaged: doing so can conceal clipping or degeneracy on
                # a single time slice of the frozen feedback.
                self.policy_sums[key] = self.policy_sums.get(key, 0.0) + number

    def as_dict(self) -> Dict[str, float]:
        denom = self.policy_points if self.policy_points > 0.0 else 1.0
        coefficient_points = max(self.coefficient_points, 1)
        operator_points = max(self.operator_points, 1)
        out = {
            "min_log_joint_eig": float(self.min_log_joint_eig),
            "max_log_joint_eig": float(self.max_log_joint_eig),
            "min_original_joint_eig": float(self.min_original_joint_eig),
            "max_original_joint_eig": float(self.max_original_joint_eig),
            "nonpositive_log_eig_fraction": float(self.nonpositive_log_eig_points) / coefficient_points,
            "max_peclet_y": float(self.max_peclet_y),
            "max_peclet_x": float(self.max_peclet_x),
            "upwind_y_fraction": float(self.upwind_y_points) / operator_points,
            "upwind_x_fraction": float(self.upwind_x_points) / operator_points,
            "max_linear_residual": float(self.max_linear_residual),
            "policy_points": float(self.policy_points),
        }
        out.update({f"policy_{key}": float(value) / denom for key, value in self.policy_sums.items()})
        out.update({f"policy_{key}": float(value) for key, value in self.policy_minima.items()})
        out.update({f"policy_{key}": float(value) for key, value in self.policy_maxima.items()})
        return out


@dataclass
class FDSolution:
    tau: Array
    y: Array
    x: Array
    value: Array
    diagnostics: FDDiagnostics


@dataclass(frozen=True)
class AffineClosedForm:
    """Dense Riccati solution for the affine one-factor benchmark."""

    problem: LiuProblem
    tau: Array
    coeff: Array  # rows: a, b, C

    def __post_init__(self) -> None:
        tau = np.asarray(self.tau, dtype=np.float64).reshape(-1)
        coeff = np.asarray(self.coeff, dtype=np.float64)
        if coeff.shape != (3, tau.size):
            raise ValueError("one-factor closed-form coeff must have shape (3, n_tau)")
        if tau.size < 2 or np.any(np.diff(tau) <= 0.0):
            raise ValueError("closed-form tau nodes must be strictly increasing")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "coeff", coeff)

    def coefficients(self, tau: Array) -> Tuple[Array, Array, Array]:
        t = np.asarray(tau, dtype=np.float64)
        return tuple(np.interp(t, self.tau, self.coeff[i]) for i in range(3))  # type: ignore[return-value]

    def value(self, tau: Array, y: Array, x: Array) -> Array:
        t, yy, xx = np.broadcast_arrays(
            np.asarray(tau, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(x, dtype=np.float64),
        )
        a, b, C = self.coefficients(t)
        q = 1.0 - self.problem.gamma
        log_phi = a + b * xx + 0.5 * C * xx * xx
        return np.exp(q * self.problem.risk_free * t + q * yy + log_phi) / q

    def wealth_bundle(self, tau: Array, y: Array, x: Array) -> Tuple[Array, Array, Array, Array]:
        t, yy, xx = np.broadcast_arrays(
            np.asarray(tau, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(x, dtype=np.float64),
        )
        a, b, C = self.coefficients(t)
        phi = np.exp(a + b * xx + 0.5 * C * xx * xx)
        g = self.problem.gamma
        D = np.exp((1.0 - g) * self.problem.risk_free * t)
        w = np.exp(yy)
        value = D * np.power(w, 1.0 - g) * phi / (1.0 - g)
        value_w = D * np.power(w, -g) * phi
        value_ww = -g * D * np.power(w, -g - 1.0) * phi
        value_wx = value_w * (b + C * xx)
        return value, value_w, value_ww, value_wx

    def optimal_vartheta(self, tau: Array, x: Array) -> Array:
        t, xx = np.broadcast_arrays(np.asarray(tau, dtype=np.float64), np.asarray(x, dtype=np.float64))
        _a, b, C = self.coefficients(t)
        lam = self.problem.risk_premium(xx)
        grad_log_phi = b + C * xx
        return (lam + grad_log_phi[..., None] * self.problem.Gamma) / self.problem.gamma


def solve_affine_closed_form(problem: LiuProblem, *, n_tau: int = 8001,
                             rtol: float = 1e-12, atol: float = 1e-14) -> AffineClosedForm:
    """Independently solve the one-factor affine Riccati system."""

    alpha = (1.0 - problem.gamma) / problem.gamma

    def rhs(_tau: float, state: Array) -> Array:
        a, b, C = (float(value) for value in state)
        A = problem.Lam + problem.Gamma * C
        m = problem.lam0 + problem.Gamma * b
        dC = -2.0 * problem.K * C + problem.Q * C * C + alpha * float(A @ A)
        db = C * problem.k0 - problem.K * b + C * problem.Q * b + alpha * float(A @ m)
        da = (problem.k0 * b + 0.5 * problem.Q * C + 0.5 * problem.Q * b * b
              + 0.5 * alpha * float(m @ m))
        return np.asarray([da, db, dC], dtype=np.float64)

    nodes = np.linspace(0.0, problem.horizon, int(n_tau), dtype=np.float64)
    sol = solve_ivp(rhs, (0.0, problem.horizon), np.zeros(3), t_eval=nodes,
                    rtol=float(rtol), atol=float(atol))
    if not sol.success or sol.y.shape != (3, nodes.size):
        raise RuntimeError(f"affine Riccati solve failed: {sol.message}")
    return AffineClosedForm(problem=problem, tau=sol.t, coeff=sol.y)


def crra_terminal(problem: LiuProblem, y: Array) -> Array:
    q = 1.0 - problem.gamma
    return np.exp(q * np.asarray(y, dtype=np.float64)) / q


def _interior_index(i: int, j: int, nx: int) -> int:
    return (i - 1) * (nx - 2) + (j - 1)


def _linearity_extension(grid: FDGrid) -> csr_matrix:
    """Map interior unknowns to all nodes using V_ww=0 and V_xx=0.

    ``V_ww=0`` becomes ``u_yy-u_y=0`` in log wealth.  Second-order one-sided
    formulas are used on each edge.  Corners use bilinear continuation of the
    already eliminated adjacent edges; corners are outside every admissible
    interior evaluation window and have no independent parabolic boundary
    datum.
    """

    ny, nx = grid.ny, grid.nx
    n_int, n_full = (ny - 2) * (nx - 2), ny * nx
    hy = (grid.y_max - grid.y_min) / (ny - 1)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(full_i: int, full_j: int, mapping: Mapping[Tuple[int, int], float]) -> None:
        row = full_i * nx + full_j
        for (ii, jj), value in mapping.items():
            rows.append(row)
            cols.append(_interior_index(ii, jj, nx))
            data.append(float(value))

    def ymap(side: int, j: int) -> Dict[Tuple[int, int], float]:
        if side == 0:
            denom = 4.0 + 3.0 * hy
            return {(1, j): (10.0 + 4.0 * hy) / denom,
                    (2, j): -(8.0 + hy) / denom,
                    (3, j): 2.0 / denom}
        denom = 4.0 - 3.0 * hy
        if abs(denom) < 1e-12:
            raise ValueError("upper log-wealth linearity closure is singular")
        return {(ny - 2, j): (10.0 - 4.0 * hy) / denom,
                (ny - 3, j): -(8.0 - hy) / denom,
                (ny - 4, j): 2.0 / denom}

    def xmap(i: int, side: int) -> Dict[Tuple[int, int], float]:
        if side == 0:
            return {(i, 1): 2.5, (i, 2): -2.0, (i, 3): 0.5}
        return {(i, nx - 2): 2.5, (i, nx - 3): -2.0, (i, nx - 4): 0.5}

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            add(i, j, {(i, j): 1.0})
    for j in range(1, nx - 1):
        add(0, j, ymap(0, j))
        add(ny - 1, j, ymap(1, j))
    for i in range(1, ny - 1):
        add(i, 0, xmap(i, 0))
        add(i, nx - 1, xmap(i, 1))

    # Bilinear corner continuation: u_corner=u_yedge+u_xedge-u_inner.
    for i_edge, i_inner, j_edge, j_inner, y_side, x_side in (
        (0, 1, 0, 1, 0, 0),
        (0, 1, nx - 1, nx - 2, 0, 1),
        (ny - 1, ny - 2, 0, 1, 1, 0),
        (ny - 1, ny - 2, nx - 1, nx - 2, 1, 1),
    ):
        mapping: Dict[Tuple[int, int], float] = {}
        for key, value in ymap(y_side, j_inner).items():
            mapping[key] = mapping.get(key, 0.0) + value
        for key, value in xmap(i_inner, x_side).items():
            mapping[key] = mapping.get(key, 0.0) + value
        mapping[(i_inner, j_inner)] = mapping.get((i_inner, j_inner), 0.0) - 1.0
        add(i_edge, j_edge, mapping)

    return coo_matrix((data, (rows, cols)), shape=(n_full, n_int)).tocsr()


def _dirichlet_extension(grid: FDGrid) -> csr_matrix:
    ny, nx = grid.ny, grid.nx
    rows, cols, data = [], [], []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            rows.append(i * nx + j)
            cols.append(_interior_index(i, j, nx))
            data.append(1.0)
    return coo_matrix((data, (rows, cols)), shape=(ny * nx, (ny - 2) * (nx - 2))).tocsr()


def _boundary_vector(grid: FDGrid, tau: float, y: Array, x: Array,
                     value_fn: BoundaryValueFn) -> Array:
    yy, xx = np.meshgrid(y, x, indexing="ij")
    values = np.asarray(value_fn(float(tau), yy, xx), dtype=np.float64)
    if values.shape != yy.shape:
        values = np.broadcast_to(values, yy.shape).copy()
    mask = np.zeros_like(values, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    out = np.zeros_like(values)
    out[mask] = values[mask]
    return out.reshape(-1)


def _joint_eigen_extremes(vartheta: Array, Gamma: Array, Q: float,
                          wealth: Optional[Array] = None) -> Tuple[Array, Array]:
    # The direct formula ``0.5 * (trace - discriminant)`` catastrophically
    # cancels for the very large policies that can occur when the V_ww guard
    # activates. Compute lambda_max from the stable plus branch and recover
    # lambda_min from det/lambda_max in extended precision instead.
    v = np.asarray(vartheta, dtype=np.longdouble)
    if wealth is not None:
        v = v * np.asarray(wealth, dtype=np.longdouble)[..., None]
    gamma = np.asarray(Gamma, dtype=np.longdouble)
    q = np.longdouble(Q)
    a = np.sum(v * v, axis=-1)
    c = np.sum(v * gamma, axis=-1)
    trace = a + q
    disc = np.hypot(a - q, 2.0 * c)
    hi = 0.5 * (trace + disc)
    determinant = a * q - c * c
    lo = np.divide(
        determinant,
        hi,
        out=np.zeros_like(determinant),
        where=hi != 0.0,
    )
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _spatial_operator(problem: LiuProblem, grid: FDGrid, y: Array, x: Array,
                      vartheta: Array, *, drift_scheme: str,
                      peclet_limit: float) -> Tuple[csr_matrix, Dict[str, Array]]:
    """Build the interior rows of the two-dimensional generator."""

    ny, nx = grid.ny, grid.nx
    hy = float(y[1] - y[0])
    hx = float(x[1] - x[0])
    expected = (ny, nx, problem.n_assets)
    v = np.asarray(vartheta, dtype=np.float64)
    if v.shape != expected:
        raise ValueError(f"vartheta must have shape {expected}, got {v.shape}")
    yy, xx = np.meshgrid(y, x, indexing="ij")
    lam = problem.risk_premium(xx)
    variance = np.sum(v * v, axis=-1)
    cross = np.einsum("ijn,n->ij", v, problem.Gamma, optimize=True)
    drift_y = problem.risk_free + np.sum(v * lam, axis=-1) - 0.5 * variance
    drift_x = problem.k0 - problem.K * xx
    diff_y = 0.5 * variance
    diff_x = 0.5 * problem.Q

    peclet_y = np.abs(drift_y) * hy / np.maximum(2.0 * diff_y, np.finfo(float).tiny)
    peclet_x = np.abs(drift_x) * hx / max(2.0 * diff_x, np.finfo(float).tiny)
    if drift_scheme == "central":
        up_y = np.zeros((ny, nx), dtype=bool)
        up_x = np.zeros((ny, nx), dtype=bool)
    elif drift_scheme == "monotone":
        up_y = np.ones((ny, nx), dtype=bool)
        up_x = np.ones((ny, nx), dtype=bool)
    elif drift_scheme == "adaptive":
        up_y = (diff_y <= 0.0) | (peclet_y > float(peclet_limit))
        up_x = peclet_x > float(peclet_limit)
    else:
        raise ValueError("drift_scheme must be central, monotone, or adaptive")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def put(row: int, i: int, j: int, value: float) -> None:
        rows.append(row)
        cols.append(i * nx + j)
        data.append(float(value))

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            row = _interior_index(i, j, nx)
            center = -2.0 * diff_y[i, j] / hy**2 - 2.0 * diff_x / hx**2
            ym = diff_y[i, j] / hy**2
            yp = diff_y[i, j] / hy**2
            xm = diff_x / hx**2
            xp = diff_x / hx**2

            by = drift_y[i, j]
            if up_y[i, j]:
                if by >= 0.0:
                    yp += by / hy
                    center -= by / hy
                else:
                    ym += -by / hy
                    center += by / hy
            else:
                ym -= by / (2.0 * hy)
                yp += by / (2.0 * hy)

            bx = drift_x[i, j]
            if up_x[i, j]:
                if bx >= 0.0:
                    xp += bx / hx
                    center -= bx / hx
                else:
                    xm += -bx / hx
                    center += bx / hx
            else:
                xm -= bx / (2.0 * hx)
                xp += bx / (2.0 * hx)

            put(row, i, j, center)
            put(row, i - 1, j, ym)
            put(row, i + 1, j, yp)
            put(row, i, j - 1, xm)
            put(row, i, j + 1, xp)

            mixed = cross[i, j] / (4.0 * hy * hx)
            put(row, i + 1, j + 1, mixed)
            put(row, i + 1, j - 1, -mixed)
            put(row, i - 1, j + 1, -mixed)
            put(row, i - 1, j - 1, mixed)

    operator = coo_matrix(
        (data, (rows, cols)),
        shape=((ny - 2) * (nx - 2), ny * nx),
    ).tocsr()
    return operator, {
        "peclet_y": peclet_y[1:-1, 1:-1],
        "peclet_x": peclet_x[1:-1, 1:-1],
        "upwind_y": up_y[1:-1, 1:-1],
        "upwind_x": up_x[1:-1, 1:-1],
        "wealth": np.exp(yy),
    }


def solve_frozen_policy(problem: LiuProblem, policy: PolicyFn, grid: FDGrid, *,
                        theta_method: float = 0.5, startup_be_steps: int = 2,
                        drift_scheme: str = "adaptive", peclet_limit: float = 1.0,
                        boundary: str = "linearity",
                        exact_boundary_value: Optional[BoundaryValueFn] = None,
                        ellipticity_tolerance: float = 0.0) -> FDSolution:
    """Solve one frozen-feedback equation on a 2-D log-wealth/factor grid.

    No ellipticity repair is performed.  The sampled minimum eigenvalue must
    be *strictly* larger than ``ellipticity_tolerance``.  Thus the default
    tolerance of zero rejects a degenerate frozen policy instead of silently
    solving a different (or merely parabolic) problem.
    """

    if not 0.5 <= float(theta_method) <= 1.0:
        raise ValueError("theta_method must lie in [0.5, 1]")
    if startup_be_steps < 0:
        raise ValueError("startup_be_steps must be nonnegative")
    boundary = str(boundary).replace("_", "-").lower()
    if boundary not in {"linearity", "exact-dirichlet"}:
        raise ValueError("boundary must be linearity or exact-dirichlet")
    if boundary == "exact-dirichlet" and exact_boundary_value is None:
        raise ValueError("exact-dirichlet requires exact_boundary_value")

    tau = np.linspace(0.0, problem.horizon, grid.nt + 1, dtype=np.float64)
    y = np.linspace(grid.y_min, grid.y_max, grid.ny, dtype=np.float64)
    x = np.linspace(grid.x_min, grid.x_max, grid.nx, dtype=np.float64)
    dt = float(tau[1] - tau[0])
    extension = _linearity_extension(grid) if boundary == "linearity" else _dirichlet_extension(grid)
    n_int = (grid.ny - 2) * (grid.nx - 2)
    ident = eye(n_int, format="csr", dtype=np.float64)
    z = np.broadcast_to(crra_terminal(problem, y[1:-1])[:, None],
                        (grid.ny - 2, grid.nx - 2)).copy().reshape(-1)
    values = np.empty((grid.nt + 1, grid.ny, grid.nx), dtype=np.float64)
    diagnostics = FDDiagnostics()

    def bvec(time_value: float) -> Array:
        if boundary == "linearity":
            return np.zeros(grid.ny * grid.nx, dtype=np.float64)
        assert exact_boundary_value is not None
        return _boundary_vector(grid, time_value, y, x, exact_boundary_value)

    values[0] = (extension @ z + bvec(0.0)).reshape(grid.ny, grid.nx)

    for step in range(grid.nt):
        old_tau = float(tau[step])
        new_tau = float(tau[step + 1])
        mid_tau = 0.5 * (old_tau + new_tau)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        vartheta, policy_diag = policy(mid_tau, yy, xx)
        vartheta = np.asarray(vartheta, dtype=np.float64)
        diagnostics.update_policy(policy_diag)

        log_lo, log_hi = _joint_eigen_extremes(vartheta, problem.Gamma, problem.Q)
        orig_lo, orig_hi = _joint_eigen_extremes(
            vartheta, problem.Gamma, problem.Q, wealth=np.exp(yy)
        )
        diagnostics.min_log_joint_eig = min(diagnostics.min_log_joint_eig, float(np.min(log_lo)))
        diagnostics.max_log_joint_eig = max(diagnostics.max_log_joint_eig, float(np.max(log_hi)))
        diagnostics.min_original_joint_eig = min(
            diagnostics.min_original_joint_eig, float(np.min(orig_lo))
        )
        diagnostics.max_original_joint_eig = max(
            diagnostics.max_original_joint_eig, float(np.max(orig_hi))
        )
        diagnostics.nonpositive_log_eig_points += int(np.count_nonzero(log_lo <= 0.0))
        diagnostics.coefficient_points += int(log_lo.size)
        if float(np.min(log_lo)) <= float(ellipticity_tolerance):
            raise ValueError(
                f"sampled log-coordinate ellipticity {float(np.min(log_lo)):.3e} "
                f"does not exceed tolerance {ellipticity_tolerance:.3e}"
            )

        full_operator, diag = _spatial_operator(
            problem, grid, y, x, vartheta,
            drift_scheme=drift_scheme, peclet_limit=peclet_limit,
        )
        reduced = (full_operator @ extension).tocsr()
        diagnostics.max_peclet_y = max(diagnostics.max_peclet_y, float(np.max(diag["peclet_y"])))
        diagnostics.max_peclet_x = max(diagnostics.max_peclet_x, float(np.max(diag["peclet_x"])))
        diagnostics.upwind_y_points += int(np.count_nonzero(diag["upwind_y"]))
        diagnostics.upwind_x_points += int(np.count_nonzero(diag["upwind_x"]))
        diagnostics.operator_points += int(diag["upwind_y"].size)

        theta = 1.0 if step < int(startup_be_steps) else float(theta_method)
        old_boundary = bvec(old_tau)
        new_boundary = bvec(new_tau)
        old_l = reduced @ z + full_operator @ old_boundary
        rhs = z + dt * (1.0 - theta) * old_l + dt * theta * (full_operator @ new_boundary)
        matrix = (ident - dt * theta * reduced).tocsr()
        z_new = np.asarray(spsolve(matrix, rhs), dtype=np.float64)
        residual = matrix @ z_new - rhs
        scale = max(1.0, float(np.max(np.abs(rhs))))
        diagnostics.max_linear_residual = max(
            diagnostics.max_linear_residual,
            float(np.max(np.abs(residual))) / scale,
        )
        if not np.all(np.isfinite(z_new)):
            raise FloatingPointError(f"non-finite FD value at tau step {step + 1}")
        z = z_new
        values[step + 1] = (extension @ z + new_boundary).reshape(grid.ny, grid.nx)

    return FDSolution(tau=tau, y=y, x=x, value=values, diagnostics=diagnostics)


def evaluate_fd_wealth_bundle(solution: FDSolution, tau_eval: Array,
                              y_eval: Array, x_eval: Array) -> Tuple[Array, Array, Array, Array]:
    """Evaluate ``(V,V_w,V_ww,V_wx)`` on a fixed tensor grid."""

    tau_values = np.asarray(tau_eval, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y_eval, dtype=np.float64).reshape(-1)
    x_values = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    if min(tau_values.size, y_values.size, x_values.size) == 0:
        raise ValueError("evaluation grids must be nonempty")
    if y_values[0] < solution.y[0] or y_values[-1] > solution.y[-1]:
        raise ValueError("evaluation y grid lies outside FD domain")
    if x_values[0] < solution.x[0] or x_values[-1] > solution.x[-1]:
        raise ValueError("evaluation x grid lies outside FD domain")
    dt = float(solution.tau[1] - solution.tau[0])
    indices = np.rint((tau_values - solution.tau[0]) / dt).astype(int)
    if np.any(indices < 0) or np.any(indices >= solution.tau.size):
        raise ValueError("evaluation tau grid lies outside FD time domain")
    if not np.allclose(solution.tau[indices], tau_values, rtol=0.0, atol=1e-11):
        raise ValueError("evaluation tau nodes must be nested in the FD time grid")

    shape = (tau_values.size, y_values.size, x_values.size)
    out = [np.empty(shape, dtype=np.float64) for _ in range(4)]
    exp_minus_y = np.exp(-y_values)[:, None]
    for out_index, time_index in enumerate(indices):
        spline = RectBivariateSpline(
            solution.y, solution.x, solution.value[time_index],
            kx=min(3, solution.y.size - 1), ky=min(3, solution.x.size - 1), s=0.0,
        )
        u = spline(y_values, x_values, dx=0, dy=0, grid=True)
        uy = spline(y_values, x_values, dx=1, dy=0, grid=True)
        uyy = spline(y_values, x_values, dx=2, dy=0, grid=True)
        uyx = spline(y_values, x_values, dx=1, dy=1, grid=True)
        out[0][out_index] = u
        out[1][out_index] = exp_minus_y * uy
        out[2][out_index] = exp_minus_y**2 * (uyy - uy)
        out[3][out_index] = exp_minus_y * uyx
    return tuple(out)  # type: ignore[return-value]


def x_norm_components(value: Array, value_w: Array, value_ww: Array,
                      value_wx: Array,
                      reference: Sequence[Array]) -> Dict[str, float]:
    """Policy-relevant Liu norm, shared by numerator and denominator."""

    arrays = [np.asarray(item, dtype=np.float64) for item in (value, value_w, value_ww, value_wx)]
    refs = [np.asarray(item, dtype=np.float64) for item in reference]
    if len(refs) != 4 or any(item.shape != arrays[0].shape for item in arrays + refs):
        raise ValueError("value and reduced-bundle arrays must have identical shapes")
    if any(not np.all(np.isfinite(item)) for item in arrays + refs):
        raise FloatingPointError("value/reference bundle contains non-finite entries")
    errors = [item - ref for item, ref in zip(arrays, refs)]
    value_sup = float(np.max(np.abs(errors[0])))
    vw_sup = float(np.max(np.abs(errors[1])))
    vww_sup = float(np.max(np.abs(errors[2])))
    vwx_sup = float(np.max(np.abs(errors[3])))
    bundle_sup = float(np.max(np.sqrt(errors[1] ** 2 + errors[2] ** 2 + errors[3] ** 2)))
    return {
        "value_sup": value_sup,
        "vw_sup": vw_sup,
        "vww_sup": vww_sup,
        "vwx_sup": vwx_sup,
        "bundle_sup": bundle_sup,
        "x_norm": value_sup + bundle_sup,
    }
