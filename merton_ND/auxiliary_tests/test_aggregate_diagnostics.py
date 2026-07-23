from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from aggregate_diagnostics import aggregate_diagnostics, metric_specs
from auxiliary_tests._paths import SOURCE_ROOT


class AggregateDiagnosticsTests(unittest.TestCase):
    def test_direct_trainer_persists_fixed_qev_scope_and_point_count(self) -> None:
        source = (SOURCE_ROOT / "merton_nd_consumption_pinn.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"control_metric_scope": "fixed_qev" if diag_res else ""', source)
        self.assertIn('"control_metric_points": (', source)
        self.assertIn('"control_metric_scope",', source)
        self.assertIn('"control_metric_points",', source)

    def test_direct_policy_mapping_uses_greedy_without_duplicate_aliases(self) -> None:
        specs = metric_specs("pinn")
        self.assertEqual(
            specs["diffusion_covariance_lambda_min"].source,
            "diffusion_var_min_greedy",
        )
        self.assertEqual(specs["vartheta_component_min"].source,
                         "pi_component_min_greedy")
        self.assertNotIn("greedy_vartheta_component_min", specs)

    def _run(
        self,
        root: str,
        seed: int,
        *,
        missing: str = "",
        duplicate: bool = False,
        margin: str = "0.10,0.0",
        suffix: str = "",
    ) -> str:
        run_dir = os.path.join(root, f"pipinn_n_assets10_seed{seed}{suffix}")
        os.makedirs(run_dir)
        config = {
            "model_type": "pipinn",
            "n_assets": 10,
            "m_states": 1,
            "seed": seed,
            "outer_iters": 2,
            "diag_points": 64,
            "diag_every": 1,
            "eval_margin": margin,
            "policy_bounds_mode": "box",
            "timing_mode": False,
        }
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"args": config}, handle)
        open(os.path.join(run_dir, "_SUCCESS"), "w", encoding="utf-8").close()

        sources = sorted({spec.source for spec in metric_specs("pipinn").values()})
        fields = [
            "model_type", "outer_iter", "control_metric_scope",
            "control_metric_points", *sources,
        ]
        with open(os.path.join(run_dir, "outer_history.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            outer_values = [1, 1] if duplicate else [1, 2]
            for outer in outer_values:
                row = {
                    "model_type": "pipinn",
                    "outer_iter": outer,
                    "control_metric_scope": "fixed_qev",
                    "control_metric_points": 64,
                }
                for source in sources:
                    row[source] = "" if source == missing else seed * 10 + outer
                writer.writerow(row)
        return run_dir

    @staticmethod
    def _read(path: str):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_seed_is_inference_unit_and_global_extreme_is_separate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1)
            self._run(root, 2)
            paths = aggregate_diagnostics(root, min_seeds=2)
            rows = self._read(paths["summary_long"])
            by_metric = {row["metric"]: row for row in rows}

            # Each seed is first reduced over its two iterations: minima are
            # 11 and 21.  The seed-level mean is 16, whereas the paper-wide
            # diagnostic extreme is 11.  Iterations are not pooled as n=4.
            row = by_metric["diffusion_covariance_lambda_min"]
            self.assertEqual(int(row["n_seeds"]), 2)
            self.assertAlmostEqual(float(row["seed_extrema_mean"]), 16.0)
            self.assertAlmostEqual(float(row["paper_extreme_across_seed_outer"]), 11.0)
            self.assertAlmostEqual(float(row["seed_extrema_std"]), math.sqrt(50.0))

            # Max-reduced fields use per-seed maxima 12 and 22.
            row = by_metric["M_y"]
            self.assertAlmostEqual(float(row["seed_extrema_mean"]), 17.0)
            self.assertAlmostEqual(float(row["paper_extreme_across_seed_outer"]), 22.0)
            self.assertTrue(math.isfinite(float(row["seed_extrema_ci95_lo"])))
            self.assertTrue(math.isfinite(float(row["seed_extrema_ci95_hi"])))

    def test_missing_applicable_metric_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1, missing="m_y")
            with self.assertRaisesRegex(ValueError, "incomplete E1 diagnostics"):
                aggregate_diagnostics(root)

    def test_duplicate_outer_iteration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate outer_iter"):
                aggregate_diagnostics(root)

    def test_merton_coverage_does_not_fabricate_kim_omberg_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1)
            paths = aggregate_diagnostics(root)
            rows = self._read(paths["coverage"])
            by_concept = {row["concept"]: row for row in rows}
            self.assertEqual(by_concept["m_ww"]["status"], "not_applicable_merton")
            self.assertEqual(by_concept["M_num"]["status"], "not_applicable_merton")
            self.assertEqual(
                by_concept["M_num_over_w_min_m_ww"]["status"],
                "not_applicable_merton",
            )
            self.assertEqual(by_concept["vartheta_component_min"]["source"],
                             "pi_component_min_frozen")

    def test_primary_diag_margin_splits_groups_but_later_margins_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1, margin="0.10,0.0", suffix="_a")
            self._run(root, 2, margin="0.20,0.0", suffix="_b")
            paths = aggregate_diagnostics(root)
            rows = self._read(paths["table"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["group"] for row in rows}), 2)

        with tempfile.TemporaryDirectory() as root:
            self._run(root, 1, margin="0.10,0.0", suffix="_a")
            self._run(root, 2, margin="0.10,0.30", suffix="_b")
            paths = aggregate_diagnostics(root, min_seeds=2)
            rows = self._read(paths["table"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["n_seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
