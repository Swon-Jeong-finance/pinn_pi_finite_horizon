from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import check_residual_substitution as gate


class ResidualSubstitutionGateTests(unittest.TestCase):
    def test_numpy_affine_substitution_is_machine_precision(self):
        batch = gate.make_substitution_batch(seed=727, batch_size=32)
        result = gate.numpy_substitution(batch, tolerance=5.0e-12)
        self.assertEqual(result.status, "pass")
        self.assertIsNotNone(result.max_scaled_residual)
        self.assertLess(result.max_scaled_residual, 1.0e-12)

    def test_nonaffine_requests_are_rejected_exactly(self):
        with self.assertRaisesRegex(gate.GateError, "affine-only"):
            gate.validate_affine_request("tanh", 0.0)
        with self.assertRaisesRegex(gate.GateError, "affine-only"):
            gate.validate_affine_request("affine", 1.0e-16)
        gate.validate_affine_request("affine", 0.0)

    def test_current_source_contract_includes_residual_dependencies_and_ode_defaults(self):
        direct = gate.inspect_source_contract(gate.SOURCE_BY_SOLVER["pinn"], "pinn")
        pi = gate.inspect_source_contract(gate.SOURCE_BY_SOLVER["pipinn"], "pipinn")
        self.assertIn("safe_concave_vww", direct.residual_calls)
        self.assertIn("safe_concave_vww", pi.residual_calls)
        self.assertIn("actual_risk_premium_torch", pi.residual_calls)
        self.assertIsNone(direct.linear_residual_lineno)
        self.assertEqual(direct.linear_residual_calls, ())
        self.assertIsNotNone(pi.linear_residual_lineno)
        self.assertIn("compute_derivatives_nd", pi.linear_residual_calls)
        self.assertIn("actual_risk_premium_torch", pi.linear_residual_calls)
        self.assertNotIn("safe_concave_vww", pi.linear_residual_calls)
        self.assertNotIn("clamp", pi.linear_residual_calls)
        self.assertNotIn("clip", pi.linear_residual_calls)
        for contract in (direct, pi):
            self.assertEqual(contract.ode_rtol, 1.0e-12)
            self.assertEqual(contract.ode_atol, 1.0e-14)
            self.assertEqual(contract.ode_nodes, 8001)

    def test_extracted_function_fails_without_helpers_and_passes_with_them(self):
        source = """
def residual(x):
    return actual_risk_premium_torch(safe_concave_vww(x))
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dummy.py"
            path.write_text(source, encoding="utf-8")
            missing = gate.compile_named_functions(path, ("residual",), {})
            with self.assertRaises(NameError):
                missing["residual"](2)
            supplied = gate.compile_named_functions(
                path,
                ("residual",),
                {
                    "safe_concave_vww": lambda value: value + 3,
                    "actual_risk_premium_torch": lambda value: value * 2,
                },
            )
            self.assertEqual(supplied["residual"](2), 10)

    def test_torch_current_residual_when_available(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        batch = gate.make_substitution_batch(seed=727, batch_size=16)
        for solver in ("pinn", "pipinn"):
            result = gate.torch_substitution(
                batch,
                gate.SOURCE_BY_SOLVER[solver],
                solver,
                tolerance=5.0e-11,
            )
            self.assertEqual(result.status, "pass", result)

    def test_linear_policy_conditions_are_raw_unclipped_and_guard_inactive(self):
        batch = gate.make_substitution_batch(seed=727, batch_size=16)
        conditions = gate.linear_policy_conditions(batch)
        self.assertEqual(
            conditions["policy_representation"],
            "raw_theta_not_theta_over_w",
        )
        self.assertFalse(conditions["theta_clipping_applied"])
        self.assertFalse(conditions["vww_guard_applied"])
        self.assertTrue(conditions["vww_guard_inactive_on_reference"])
        self.assertLessEqual(
            conditions["analytic_vww_max"],
            -conditions["vww_guard_threshold"],
        )

    def test_torch_current_linear_residual_when_available(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        batch = gate.make_substitution_batch(seed=727, batch_size=16)
        result = gate.torch_linear_substitution(
            batch,
            gate.SOURCE_BY_SOLVER["pipinn"],
            tolerance=5.0e-11,
        )
        self.assertEqual(result.status, "pass", result)
        self.assertIsNotNone(result.conditions)
        self.assertTrue(result.conditions["vww_guard_inactive_on_reference"])
        self.assertFalse(result.conditions["vww_guard_applied"])
        self.assertFalse(result.conditions["theta_clipping_applied"])
        self.assertEqual(
            result.conditions["policy_representation"],
            "raw_theta_not_theta_over_w",
        )
        self.assertLessEqual(
            result.conditions["max_scaled_foc_vs_closed_form_theta"],
            result.tolerance,
        )

    def test_pipinn_run_reports_separate_linear_stage_even_on_torch_skip(self):
        args = argparse.Namespace(
            solver="pipinn",
            risk_premium_mode="affine",
            nonaffine_eps=0.0,
            seed=727,
            batch_size=8,
            numpy_tol=5.0e-12,
            torch_tol=5.0e-11,
            require_torch=False,
        )
        payload, code = gate.run(args)
        rows = [
            row for row in payload["results"]
            if row["stage"] == "torch_current_linear_residual_pipinn"
        ]
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]["status"], {"pass", "skip"})
        self.assertEqual(
            rows[0]["conditions"]["policy_representation"],
            "raw_theta_not_theta_over_w",
        )
        self.assertTrue(
            rows[0]["conditions"]["vww_guard_inactive_on_reference"]
        )
        self.assertFalse(rows[0]["conditions"]["vww_guard_applied"])
        self.assertFalse(rows[0]["conditions"]["theta_clipping_applied"])
        self.assertEqual(code, 0)

    def test_require_torch_turns_a_skip_into_failure(self):
        args = argparse.Namespace(
            solver="pipinn",
            risk_premium_mode="affine",
            nonaffine_eps=0.0,
            seed=727,
            batch_size=8,
            numpy_tol=5.0e-12,
            torch_tol=5.0e-11,
            require_torch=True,
        )
        payload, code = gate.run(args)
        torch_rows = [row for row in payload["results"] if row["stage"].startswith("torch_")]
        if torch_rows[0]["status"] == "skip":
            self.assertEqual(code, 1)
            self.assertEqual(payload["overall_status"], "fail")
        else:
            self.assertEqual(code, 0)
            self.assertEqual(payload["overall_status"], "pass")


if __name__ == "__main__":
    unittest.main()
