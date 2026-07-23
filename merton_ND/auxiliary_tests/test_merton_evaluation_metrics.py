from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from auxiliary_tests._paths import SOURCE_ROOT
import merton_evaluation_metrics as mem


class DerivativeBundleMetricTests(unittest.TestCase):
    def test_known_product_space_bundle_norms(self) -> None:
        Vw_ref = np.array([[1.0], [2.0]])
        Vww_ref = np.array([[-1.0], [-2.0]])
        Vw_pred = np.array([[2.0], [2.0]])
        Vww_pred = np.array([[-1.0], [-1.0]])

        result = mem.derivative_bundle_metrics(
            Vw_pred, Vww_pred, Vw_ref, Vww_ref)

        self.assertAlmostEqual(result["RelL2_D"], np.sqrt(2.0 / 10.0))
        self.assertAlmostEqual(result["e_D_sup"], 1.0)

    def test_full_schema_and_zero_error(self) -> None:
        V = np.array([[1.0], [2.0]])
        c = np.array([[0.4], [0.5]])
        pi = np.array([[0.1, -0.2], [0.1, -0.2]])
        Vw = np.array([[2.0], [3.0]])
        Vww = np.array([[-3.0], [-4.0]])
        result = mem.full_window_metrics(
            V, c, pi, Vw, Vww, V, c, pi, Vw, Vww)
        self.assertEqual(tuple(result), mem.FULL_WINDOW_METRIC_NAMES)
        self.assertTrue(all(value == 0.0 for value in result.values()))

    def test_bundle_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            mem.derivative_bundle_metrics(
                np.ones((2, 1)), np.ones((3, 1)),
                np.ones((2, 1)), np.ones((2, 1)),
            )


class ClosedFormScaleTests(unittest.TestCase):
    def test_nonunit_bequest_uses_gamma_root_at_terminal_time(self) -> None:
        scale = mem.crra_homothetic_scale(
            np.asarray([0.0, 0.25]), nu=0.03, gamma=2.0,
            epsilon_bequest=4.0,
        )
        self.assertAlmostEqual(float(scale[0]), 2.0)
        self.assertAlmostEqual(float(scale[0] ** 2), 4.0)
        self.assertAlmostEqual(float(1.0 / scale[0]), 0.5)

    def test_zero_nu_limit_is_continuous(self) -> None:
        tau = np.asarray([0.0, 0.2, 1.0])
        at_zero = mem.crra_homothetic_scale(tau, 0.0, 2.0, 4.0)
        near_zero = mem.crra_homothetic_scale(tau, 1e-9, 2.0, 4.0)
        np.testing.assert_allclose(at_zero, 2.0 + tau)
        np.testing.assert_allclose(near_zero, at_zero, rtol=1e-8, atol=1e-10)

    def test_invalid_bequest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "epsilon_bequest"):
            mem.crra_homothetic_scale(np.asarray([0.0]), 0.1, 2.0, 0.0)


class E9AggregationTests(unittest.TestCase):
    def test_aggregate_seeds_accepts_outer_20_and_30_as_separate_groups(self) -> None:
        repo = SOURCE_ROOT
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for model in ("pinn", "pipinn"):
                for outer_iters in (20, 30):
                    run = root / f"{model}_outer{outer_iters}_seed1"
                    run.mkdir()
                    (run / "config.json").write_text(json.dumps({"args": {
                        "model_type": model, "n_assets": 10, "m_states": 1,
                        "seed": 1, "market_seed": 12,
                        "outer_iters": outer_iters,
                        "eval_margin": "0.10",
                    }}), encoding="utf-8")
                    (run / "_SUCCESS").write_text("", encoding="utf-8")
                    sigma = np.eye(10, dtype=np.float64) * 0.04
                    np.savez(
                        run / "market_params.npz",
                        mu_excess=np.full(10, 0.05, dtype=np.float64),
                        Sigma_safe=sigma,
                        chol=np.linalg.cholesky(sigma),
                        pi_star=np.full(10, 0.1, dtype=np.float64),
                        Theta=np.asarray(0.625), nu=np.asarray(0.02),
                        gamma=np.asarray(2.0), r=np.asarray(0.03),
                        rho_discount=np.asarray(0.04), epsilon=np.asarray(1.0),
                        T=np.asarray(1.0), w_min=np.asarray(0.1),
                        w_max=np.asarray(2.0), n_assets=np.asarray(10),
                        market_seed=np.asarray(12),
                    )
                    with (run / "metrics.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=[
                            "scope", "eval_margin", "metric", "value"])
                        writer.writeheader()
                        for metric in (
                            "RelL2_V", "RelL2_D", "e_D_sup", "RelL2_pi", "RelL2_c"
                        ):
                            writer.writerow({
                                "scope": "fulldim", "eval_margin": 0.10,
                                "metric": metric, "value": outer_iters / 1000.0,
                            })

            subprocess.run([
                sys.executable, str(repo / "aggregate_seeds.py"),
                "--out-root", str(root),
                "--expected-n-assets", "10",
                "--expected-m-states", "1",
                "--expected-models", "pinn,pipinn",
                "--expected-seeds", "1",
                "--strict-market-snapshots",
            ], cwd=repo, check=True, capture_output=True, text=True)
            groups = json.loads(
                (root / "seed_summary" / "groups.json").read_text(encoding="utf-8"))
            self.assertEqual(len(groups), 4)

    def test_aggregate_seeds_writes_all_requested_nested_windows(self) -> None:
        repo = SOURCE_ROOT
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run_seed1"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({"args": {
                "model_type": "pinn", "n_assets": 10, "m_states": 1,
                "seed": 1, "eval_margin": "0.05,0.10,0.20,0.30",
            }}), encoding="utf-8")
            (run / "_SUCCESS").write_text("", encoding="utf-8")
            with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "scope", "eval_margin", "metric", "value"])
                writer.writeheader()
                for margin in (0.05, 0.10, 0.20, 0.30):
                    for index, metric in enumerate((
                        "RelL2_V", "RelL2_D", "e_D_sup", "RelL2_pi", "RelL2_c"
                    ), start=1):
                        writer.writerow({
                            "scope": "fulldim", "eval_margin": margin,
                            "metric": metric, "value": margin + index,
                        })

            subprocess.run([
                sys.executable, str(repo / "aggregate_seeds.py"),
                "--out-root", str(root),
                "--e9-margins", "0.05,0.10,0.20,0.30",
            ], cwd=repo, check=True, capture_output=True, text=True)

            with (root / "seed_summary" / "summary_e9.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(row["eval_margin"]) for row in rows],
                             [0.05, 0.10, 0.20, 0.30])
            self.assertTrue(all(row["RelL2_D_mean"] for row in rows))
            self.assertTrue(all(row["e_D_sup_mean"] for row in rows))


class TrainerE9ContractTests(unittest.TestCase):
    def test_both_trainers_use_wealth_bundle_and_shared_all_margin_schema(self) -> None:
        root = SOURCE_ROOT
        for filename in (
            "merton_nd_consumption_pinn.py",
            "merton_nd_consumption_pi_pinn.py",
        ):
            source = (root / filename).read_text(encoding="utf-8")
            self.assertIn("import merton_evaluation_metrics as mem", source)
            self.assertIn("V_ww = (V_yy - V_y) / (W ** 2)", source)
            self.assertIn("return mem.full_window_metrics(", source)
            # Definition plus random-point and deterministic-grid call sites.
            self.assertGreaterEqual(source.count("eval_metrics_on_points("), 3)

    def test_both_trainers_share_nonunit_bequest_closed_form_scale(self) -> None:
        root = SOURCE_ROOT
        for filename in (
            "merton_nd_consumption_pinn.py",
            "merton_nd_consumption_pi_pinn.py",
        ):
            source = (root / filename).read_text(encoding="utf-8")
            self.assertGreaterEqual(source.count("mem.crra_homothetic_scale("), 3)

    def test_timing_mode_forces_the_persisted_skip_figures_flag(self) -> None:
        root = SOURCE_ROOT
        expected = "ARGS.skip_figures or ARGS.skip_plots or ARGS.timing_mode"
        for filename in (
            "merton_nd_consumption_pinn.py",
            "merton_nd_consumption_pi_pinn.py",
        ):
            source = (root / filename).read_text(encoding="utf-8")
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
