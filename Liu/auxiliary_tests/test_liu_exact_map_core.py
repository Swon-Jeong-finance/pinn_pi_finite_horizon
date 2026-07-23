"""CPU regression tests for the independent Liu M=1 FD reference."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import liu_exact_map_core as exact_map_core
from liu_exact_map_core import (
    FDDiagnostics,
    FDGrid,
    FDSolution,
    LiuProblem,
    _boundary_elimination_diagnostics,
    _joint_eigen_extremes,
    _linearity_extension,
    _spatial_operator,
    evaluate_fd_wealth_bundle,
    solve_affine_closed_form,
    solve_frozen_policy,
    x_norm_components,
)


def problem() -> LiuProblem:
    return LiuProblem(
        horizon=0.5,
        y_min=float(np.log(0.2)),
        y_max=float(np.log(3.0)),
        x_min=-0.7,
        x_max=0.7,
        gamma=3.0,
        risk_free=0.02,
        K=0.8,
        k0=0.1,
        Q=0.04,
        Gamma=np.asarray([0.04, -0.02]),
        lam0=np.asarray([0.08, 0.04]),
        Lam=np.asarray([0.05, -0.03]),
    )


class LiuExactMapCoreTests(unittest.TestCase):
    @staticmethod
    def _constant_elliptic_policy(p: LiuProblem):
        vector = np.asarray([0.20, -0.10], dtype=np.float64)
        if vector.size != p.n_assets:
            raise AssertionError("test policy dimension must match the test problem")

        def policy(_tau: float, _y: np.ndarray, x: np.ndarray):
            values = np.broadcast_to(vector, (*x.shape, p.n_assets)).copy()
            return values, {"points": float(x.size)}

        return policy

    def test_joint_minimum_eigenvalue_is_stable_for_guard_scale_policy(self) -> None:
        low, high = _joint_eigen_extremes(
            np.asarray([[1.0e8, 0.0]]),
            np.asarray([0.04, -0.02]),
            0.04,
        )
        self.assertGreater(float(low[0]), 0.0)
        self.assertAlmostEqual(float(low[0]), 0.0384, places=11)
        self.assertGreater(float(high[0]), 1.0e15 - 1.0)

    def test_default_ellipticity_gate_rejects_degenerate_zero_policy(self) -> None:
        p = problem()

        def zero_policy(_tau: float, _y: np.ndarray, x: np.ndarray):
            return np.zeros((*x.shape, p.n_assets)), {"points": float(x.size)}

        with self.assertRaisesRegex(ValueError, "ellipticity"):
            solve_frozen_policy(
                p,
                zero_policy,
                FDGrid(p.y_min, p.y_max, p.x_min, p.x_max, ny=9, nx=9, nt=4),
            )

    def test_mixed_derivative_stencil_has_the_generator_sign(self) -> None:
        p = problem()
        grid = FDGrid(-0.8, 0.9, -0.6, 0.65, ny=9, nx=10, nt=4)
        y = np.linspace(grid.y_min, grid.y_max, grid.ny)
        x = np.linspace(grid.x_min, grid.x_max, grid.nx)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        vartheta = np.empty((grid.ny, grid.nx, p.n_assets))
        vartheta[..., 0] = 0.20
        vartheta[..., 1] = -0.10
        operator, _diag = _spatial_operator(
            p, grid, y, x, vartheta, drift_scheme="central", peclet_limit=1.0
        )
        # u=y*x has u_y=x, u_x=y, u_yx=1 and zero pure second derivatives.
        numerical = np.asarray(operator @ (yy * xx).reshape(-1)).reshape(grid.ny - 2, grid.nx - 2)
        variance = float(np.sum(vartheta[0, 0] ** 2))
        cross = float(vartheta[0, 0] @ p.Gamma)
        lam = p.risk_premium(xx)
        drift_y = p.risk_free + np.sum(vartheta * lam, axis=-1) - 0.5 * variance
        drift_x = p.k0 - p.K * xx
        expected = cross + drift_y * xx + drift_x * yy
        np.testing.assert_allclose(numerical, expected[1:-1, 1:-1], rtol=0.0, atol=2e-13)

    def test_linearity_extension_enforces_both_one_sided_conditions(self) -> None:
        grid = FDGrid(-1.1, 0.8, -0.7, 0.6, ny=9, nx=10, nt=4)
        rng = np.random.default_rng(123)
        interior = rng.normal(size=(grid.ny - 2) * (grid.nx - 2))
        full = np.asarray(_linearity_extension(grid) @ interior).reshape(grid.ny, grid.nx)
        hy = (grid.y_max - grid.y_min) / (grid.ny - 1)

        lower_uyy = (2 * full[0] - 5 * full[1] + 4 * full[2] - full[3]) / hy**2
        lower_uy = (-3 * full[0] + 4 * full[1] - full[2]) / (2 * hy)
        upper_uyy = (2 * full[-1] - 5 * full[-2] + 4 * full[-3] - full[-4]) / hy**2
        upper_uy = (3 * full[-1] - 4 * full[-2] + full[-3]) / (2 * hy)
        np.testing.assert_allclose((lower_uyy - lower_uy)[1:-1], 0.0, atol=2e-13)
        np.testing.assert_allclose((upper_uyy - upper_uy)[1:-1], 0.0, atol=2e-13)

        lower_xx = 2 * full[:, 0] - 5 * full[:, 1] + 4 * full[:, 2] - full[:, 3]
        upper_xx = 2 * full[:, -1] - 5 * full[:, -2] + 4 * full[:, -3] - full[:, -4]
        np.testing.assert_allclose(lower_xx[1:-1], 0.0, atol=2e-13)
        np.testing.assert_allclose(upper_xx[1:-1], 0.0, atol=2e-13)

    def test_boundary_elimination_diagnostics_are_full_rank_and_recorded(self) -> None:
        p = problem()
        grid = FDGrid(p.y_min, p.y_max, p.x_min, p.x_max, ny=9, nx=10, nt=4)
        expected_size = 2 * grid.ny + 2 * grid.nx - 4

        boundary_diag = _boundary_elimination_diagnostics(grid, "linearity")
        self.assertEqual(int(boundary_diag["size"]), expected_size)
        self.assertEqual(int(boundary_diag["rank"]), expected_size)
        self.assertTrue(np.isfinite(boundary_diag["condition_inf"]))
        self.assertGreaterEqual(float(boundary_diag["condition_inf"]), 1.0)

        solution = solve_frozen_policy(
            p,
            self._constant_elliptic_policy(p),
            grid,
            boundary="linearity",
        )
        recorded = solution.diagnostics.as_dict()
        self.assertEqual(recorded["boundary_elimination_size"], expected_size)
        self.assertEqual(recorded["boundary_elimination_rank"], expected_size)
        self.assertAlmostEqual(
            recorded["boundary_elimination_cond_inf"],
            float(boundary_diag["condition_inf"]),
        )

    def test_rank_deficient_linearity_boundary_is_rejected_before_fd_solve(self) -> None:
        p = problem()
        # h_y = 4/3 makes the upper u_yy-u_y boundary pivot 4-3*h_y zero.
        grid = FDGrid(0.0, 8.0, p.x_min, p.x_max, ny=7, nx=7, nt=2)
        boundary_diag = _boundary_elimination_diagnostics(grid, "linearity")
        self.assertLess(int(boundary_diag["rank"]), int(boundary_diag["size"]))
        self.assertFalse(np.isfinite(boundary_diag["condition_inf"]))

        with self.assertRaisesRegex(np.linalg.LinAlgError, "rank deficient"):
            solve_frozen_policy(
                p,
                self._constant_elliptic_policy(p),
                grid,
                boundary="linearity",
            )

    def test_normal_linearity_solve_residual_passes_hard_gate(self) -> None:
        p = problem()
        solution = solve_frozen_policy(
            p,
            self._constant_elliptic_policy(p),
            FDGrid(p.y_min, p.y_max, p.x_min, p.x_max, ny=9, nx=9, nt=4),
            boundary="linearity",
            linear_residual_tolerance=1.0e-8,
        )
        diagnostics = solution.diagnostics.as_dict()
        self.assertTrue(np.isfinite(diagnostics["max_linear_residual"]))
        self.assertLessEqual(diagnostics["max_linear_residual"], 1.0e-8)
        self.assertGreater(diagnostics["min_linear_system_lu_pivot_ratio"], 0.0)

    def test_hard_residual_gate_rejects_inaccurate_finite_lu_solution(self) -> None:
        p = problem()
        grid = FDGrid(p.y_min, p.y_max, p.x_min, p.x_max, ny=9, nx=9, nt=2)

        class FakeU:
            def __init__(self, size: int):
                self._size = size

            def diagonal(self) -> np.ndarray:
                return np.ones(self._size, dtype=np.float64)

        class InaccurateLU:
            def __init__(self, size: int):
                self.U = FakeU(size)

            @staticmethod
            def solve(rhs: np.ndarray) -> np.ndarray:
                # Finite and correctly shaped, but deliberately not A^{-1} rhs.
                return np.zeros_like(rhs)

        def inaccurate_splu(matrix):
            return InaccurateLU(int(matrix.shape[0]))

        with patch.object(exact_map_core, "splu", side_effect=inaccurate_splu):
            with self.assertRaisesRegex(FloatingPointError, "linear residual gate failed"):
                solve_frozen_policy(
                    p,
                    self._constant_elliptic_policy(p),
                    grid,
                    boundary="linearity",
                    linear_residual_tolerance=1.0e-8,
                )

    def test_spline_bundle_is_converted_to_original_wealth_derivatives(self) -> None:
        tau = np.linspace(0.0, 0.5, 5)
        y = np.linspace(-0.8, 0.9, 17)
        x = np.linspace(-0.6, 0.7, 18)
        tt, yy, xx = np.meshgrid(tau, y, x, indexing="ij")
        values = (1.0 + tt) * (yy**3 + yy * xx + xx**2)
        solution = FDSolution(tau=tau, y=y, x=x, value=values, diagnostics=FDDiagnostics())
        tau_ev = tau[1:]
        y_ev = y[2:-2]
        x_ev = x[2:-2]
        out = evaluate_fd_wealth_bundle(solution, tau_ev, y_ev, x_ev)
        tev, yev, xev = np.meshgrid(tau_ev, y_ev, x_ev, indexing="ij")
        scale = 1.0 + tev
        expected = (
            scale * (yev**3 + yev * xev + xev**2),
            np.exp(-yev) * scale * (3.0 * yev**2 + xev),
            np.exp(-2.0 * yev) * scale * (6.0 * yev - 3.0 * yev**2 - xev),
            np.exp(-yev) * scale,
        )
        for observed, target in zip(out, expected):
            np.testing.assert_allclose(observed, target, rtol=2e-11, atol=2e-11)

    def test_x_norm_uses_joint_original_derivative_bundle(self) -> None:
        shape = (2, 3, 4)
        zeros = np.zeros(shape)
        metric = x_norm_components(
            np.full(shape, 2.0), np.full(shape, 3.0), np.full(shape, 4.0),
            np.full(shape, 12.0), (zeros, zeros, zeros, zeros),
        )
        self.assertAlmostEqual(metric["value_sup"], 2.0)
        self.assertAlmostEqual(metric["bundle_sup"], 13.0)
        self.assertAlmostEqual(metric["x_norm"], 15.0)

    def test_optimal_frozen_policy_recovers_affine_solution_under_refinement(self) -> None:
        p = problem()
        exact = solve_affine_closed_form(p)

        def policy(tau_value: float, _y: np.ndarray, x: np.ndarray):
            values = exact.optimal_vartheta(np.full_like(x, tau_value), x)
            return values, {"points": float(x.size)}

        errors = []
        for n in (25, 49):
            solution = solve_frozen_policy(
                p, policy,
                FDGrid(p.y_min, p.y_max, p.x_min, p.x_max, ny=n, nx=n, nt=2 * (n - 1)),
                boundary="exact-dirichlet",
                exact_boundary_value=lambda t, y, x: exact.value(t, y, x),
                drift_scheme="central",
            )
            tau_ev = np.linspace(0.0, p.horizon, 9)[1:]
            y_ev = np.linspace(np.log(0.4), np.log(1.5), 15)
            x_ev = np.linspace(-0.35, 0.35, 15)
            tt, yy, xx = np.meshgrid(tau_ev, y_ev, x_ev, indexing="ij")
            metric = x_norm_components(
                *evaluate_fd_wealth_bundle(solution, tau_ev, y_ev, x_ev),
                exact.wealth_bundle(tt, yy, xx),
            )
            errors.append(metric["x_norm"])
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1], 0.35 * errors[0])


if __name__ == "__main__":
    unittest.main()
