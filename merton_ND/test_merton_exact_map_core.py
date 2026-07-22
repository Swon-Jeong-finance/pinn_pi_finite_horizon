"""CPU-only regression tests for the Merton finite-difference reference."""
from __future__ import annotations

import unittest

import numpy as np

from merton_exact_map_core import (
    FDGrid,
    MertonProblem,
    analytic_optimal_policy,
    crra_closed_form,
    evaluate_fd_bundle,
    log_to_wealth_derivatives,
    solve_frozen_policy,
    x_norm_components,
)


def problem() -> MertonProblem:
    return MertonProblem(
        horizon=1.0,
        y_min=float(np.log(0.05)),
        y_max=float(np.log(4.0)),
        gamma=2.0,
        discount=0.04,
        bequest=1.0,
        risk_free=0.03,
        mu_excess=np.asarray([0.08]),
        sigma=np.asarray([[0.04]]),
    )


def error_for_grid(ny: int, boundary: str = "robin") -> float:
    p = problem()
    solution = solve_frozen_policy(
        p,
        analytic_optimal_policy(p),
        FDGrid(p.y_min, p.y_max, ny=ny, nt=ny - 1),
        boundary=boundary,
        drift_scheme="adaptive",
    )
    tau = np.linspace(0.0, p.horizon, 41)
    y = np.linspace(np.log(0.1), np.log(2.0), 61)
    tt, yy = np.meshgrid(tau, y, indexing="ij")
    bundle = evaluate_fd_bundle(solution, tau, y)
    metrics = x_norm_components(*bundle, crra_closed_form(p, tt, yy), yy)
    return float(metrics["x_norm"])


class ExactMapCoreTests(unittest.TestCase):
    def test_closed_form_is_recovered_under_optimal_frozen_policy(self) -> None:
        coarse = error_for_grid(81)
        fine = error_for_grid(161)
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.4 * coarse)

    def test_dirichlet_audit_closure_also_converges(self) -> None:
        coarse = error_for_grid(81, boundary="exact-dirichlet")
        fine = error_for_grid(161, boundary="exact-dirichlet")
        self.assertLess(fine, 0.4 * coarse)

    def test_ratio_norm_uses_value_and_wealth_derivatives(self) -> None:
        shape = (2, 3)
        zeros = np.zeros(shape)
        reference = (zeros, zeros, zeros)
        value = np.full(shape, 2.0)
        value_y = np.full(shape, 3.0)
        value_yy = np.full(shape, 4.0)
        y = np.log(np.asarray([1.0, 2.0, 4.0]))
        metrics = x_norm_components(value, value_y, value_yy, reference, y)
        self.assertAlmostEqual(metrics["value_sup"], 2.0)
        self.assertAlmostEqual(metrics["vy_sup"], 3.0)
        self.assertAlmostEqual(metrics["vyy_sup"], 4.0)
        self.assertAlmostEqual(metrics["vw_sup"], 3.0)
        self.assertAlmostEqual(metrics["vww_sup"], 1.0)
        self.assertAlmostEqual(metrics["derivative_sup"], np.sqrt(10.0))
        self.assertAlmostEqual(metrics["x_norm"], 2.0 + np.sqrt(10.0))

    def test_log_to_wealth_derivatives_broadcasts_spatial_y(self) -> None:
        value_y = np.asarray([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
        value_yy = value_y + np.asarray([[1.0, 4.0, 16.0], [2.0, 8.0, 32.0]])
        y = np.log(np.asarray([1.0, 2.0, 4.0]))
        value_w, value_ww = log_to_wealth_derivatives(value_y, value_yy, y)
        np.testing.assert_allclose(value_w, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        np.testing.assert_allclose(value_ww, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "broadcast"):
            log_to_wealth_derivatives(value_y, value_yy, np.zeros(2))

    def test_diffusion_diagnostic_separates_variance_from_pde_coefficient(self) -> None:
        p = problem()
        solution = solve_frozen_policy(
            p,
            analytic_optimal_policy(p),
            FDGrid(p.y_min, p.y_max, ny=21, nt=20),
        )
        diagnostics = solution.diagnostics.as_dict()
        expected_variance = float(
            (p.sigma_inv_mu / p.gamma)
            @ p.sigma
            @ (p.sigma_inv_mu / p.gamma)
        )
        self.assertAlmostEqual(
            diagnostics["min_diffusion_variance"], expected_variance
        )
        self.assertAlmostEqual(
            diagnostics["max_diffusion_variance"], expected_variance
        )
        self.assertAlmostEqual(
            diagnostics["min_diffusion_coefficient"], 0.5 * expected_variance
        )
        self.assertAlmostEqual(
            diagnostics["max_diffusion_coefficient"], 0.5 * expected_variance
        )
        self.assertEqual(
            diagnostics["min_diffusion"], diagnostics["min_diffusion_coefficient"]
        )

    def test_ratio_is_never_clipped(self) -> None:
        zeros = np.zeros((2, 2))
        denominator = x_norm_components(
            np.ones((2, 2)), zeros, zeros, (zeros, zeros, zeros), zeros
        )["x_norm"]
        numerator = x_norm_components(
            np.full((2, 2), 2.0), zeros, zeros, (zeros, zeros, zeros), zeros
        )["x_norm"]
        self.assertEqual(numerator / denominator, 2.0)


if __name__ == "__main__":
    unittest.main()
