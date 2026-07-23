#!/usr/bin/env python3
"""Auxiliary tests for Merton postprocess_contraction.py (no torch needed)."""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import postprocess_contraction as contraction


class MertonFigureFixture:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _market(path: Path, *, shift: float = 0.0) -> None:
        n = 10
        sigma = np.eye(n, dtype=np.float64) * 0.04
        np.savez(
            path,
            mu_excess=np.full(n, 0.05 + shift, dtype=np.float64),
            Sigma_safe=sigma,
            chol=np.linalg.cholesky(sigma),
            pi_star=np.full(n, 0.1, dtype=np.float64),
            Theta=np.asarray(0.625, dtype=np.float64),
            nu=np.asarray(0.02, dtype=np.float64),
            gamma=np.asarray(2.0, dtype=np.float64),
            r=np.asarray(0.03, dtype=np.float64),
            rho_discount=np.asarray(0.04, dtype=np.float64),
            epsilon=np.asarray(1.0, dtype=np.float64),
            T=np.asarray(1.0, dtype=np.float64),
            w_min=np.asarray(0.1, dtype=np.float64),
            w_max=np.asarray(2.0, dtype=np.float64),
            n_assets=np.asarray(n, dtype=np.int64),
            market_seed=np.asarray(12, dtype=np.int64),
        )

    def add_run(
        self,
        seed: int,
        *,
        outer_iters: int = 20,
        rows: int | None = None,
        diag_every: int = 1,
        market_shift: float = 0.0,
        batch_size: int = 10000,
        duplicate_outer: bool = False,
    ) -> Path:
        run_dir = self.root / f"pipinn_n_assets10_seed{seed}"
        run_dir.mkdir(parents=True)
        args = {
            "model_type": "pipinn",
            "run_tag": run_dir.name,
            "n_assets": 10,
            "m_states": 1,
            "seed": seed,
            "market_seed": 12,
            "outer_iters": outer_iters,
            "diag_every": diag_every,
            "diag_points": 8192,
            "eval_margin": "0.10,0.0,0.05",
            "pi_init_method": "myopic",
            "pi_init_scale": 1.0,
            "policy_bounds_mode": "none",
            "batch_size": batch_size,
        }
        (run_dir / "config.json").write_text(
            json.dumps({"args": args}), encoding="utf-8"
        )
        (run_dir / "status.json").write_text(
            json.dumps({"status": "success", "updated_at": f"2026-07-22T00:00:{seed:02d}"}),
            encoding="utf-8",
        )
        (run_dir / "_SUCCESS").touch()
        self._market(run_dir / "market_params.npz", shift=market_shift)

        count = outer_iters if rows is None else rows
        history_rows = []
        for index in range(count):
            outer = index + 1
            history_rows.append(
                {
                    "outer_iter": outer,
                    "diag_RelL2_V": 0.20 * (0.8 ** index) * (1.0 + 0.05 * seed),
                    "diag_RelL2_pi": 0.30 * (0.75 ** index) * (1.0 + 0.03 * seed),
                    "diag_RelL2_c": 0.10 * (0.70 ** index) * (1.0 + 0.02 * seed),
                    "e_Xev": 1000.0,  # Must be ignored by the postprocessor.
                }
            )
        if duplicate_outer:
            history_rows.append(dict(history_rows[-1]))
        with (run_dir / "outer_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "outer_iter",
                    "diag_RelL2_V",
                    "diag_RelL2_pi",
                    "diag_RelL2_c",
                    "e_Xev",
                ],
            )
            writer.writeheader()
            writer.writerows(history_rows)
        return run_dir


class MertonFigure2Tests(unittest.TestCase):
    @staticmethod
    def _run(root: Path, *extra: str) -> Path:
        output = root / "derived"
        code = contraction.main(
            [
                "--out-root", str(root),
                "--output", str(output),
                "--n-assets", "10",
                "--expected-seeds", "1,2",
                "--min-seeds", "2",
                "--no-plots",
                *extra,
            ]
        )
        if code != 0:
            raise AssertionError(f"postprocessor returned {code}")
        return output

    @staticmethod
    def _complete(root: Path) -> MertonFigureFixture:
        fixture = MertonFigureFixture(root)
        fixture.add_run(1)
        fixture.add_run(2)
        return fixture

    def test_default_rms_is_within_seed_and_raw_components_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = self._run(root)

            with (output / "figure2_trajectories.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            metrics = {row["metric"] for row in rows}
            self.assertEqual(metrics, set(contraction.EXPORTED_METRICS))
            rms = next(
                row for row in rows
                if row["metric"] == contraction.POLICY_RMS_METRIC
                and row["seed"] == "1" and row["outer_iter"] == "1"
            )
            expected = math.sqrt(((0.30 * 1.03) ** 2 + (0.10 * 1.02) ** 2) / 2.0)
            self.assertAlmostEqual(float(rms["value"]), expected)

            metadata = json.loads(
                (output / "figure2_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["plotted_metrics"],
                [contraction.VALUE_METRIC, contraction.POLICY_RMS_METRIC],
            )
            self.assertFalse(metadata["e_Xev_used"])
            self.assertFalse(metadata["one_step_ratio_used"])
            self.assertFalse(metadata["floor_filter_used"])
            self.assertIn(
                "sqrt((diag_RelL2_pi^2 + diag_RelL2_c^2)/2)",
                metadata["metric_definitions"][contraction.POLICY_RMS_METRIC],
            )

    def test_separate_policy_choice_changes_plot_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = self._run(root, "--policy-curve", "separate")
            metadata = json.loads(
                (output / "figure2_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["plotted_metrics"],
                [
                    contraction.VALUE_METRIC,
                    contraction.PI_METRIC,
                    contraction.CONSUMPTION_METRIC,
                ],
            )
            self.assertIn(
                contraction.POLICY_RMS_METRIC, metadata["exported_metrics"]
            )

    def test_endpoint_is_ratio_of_seed_means_at_outer_1_and_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = self._run(root)
            with (output / "figure2_endpoint_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            value = next(row for row in rows if row["metric"] == contraction.VALUE_METRIC)
            self.assertEqual(value["outer_start"], "1")
            self.assertEqual(value["outer_end"], "20")
            self.assertAlmostEqual(
                float(value["seed_mean_reduction_factor"]), 1.0 / (0.8 ** 19)
            )

    def test_incomplete_outer_history_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1)
            fixture.add_run(2, rows=19)
            with self.assertRaisesRegex(ValueError, "must cover outer 1..20 exactly"):
                self._run(root)

    def test_duplicate_outer_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1)
            fixture.add_run(2, duplicate_outer=True)
            with self.assertRaisesRegex(ValueError, "duplicate outer_iter=20"):
                self._run(root)

    def test_diag_every_must_be_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1, diag_every=2)
            fixture.add_run(2, diag_every=2)
            with self.assertRaisesRegex(ValueError, "Figure 2 requires diag_every=1"):
                self._run(root)

    def test_distinct_markets_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1)
            fixture.add_run(2, market_shift=0.001)
            with self.assertRaisesRegex(ValueError, "distinct canonical Merton markets"):
                self._run(root)

    def test_training_config_mismatch_requires_explicit_group_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1, batch_size=10000)
            fixture.add_run(2, batch_size=5000)
            with self.assertRaisesRegex(ValueError, "exactly one eligible Merton"):
                self._run(root)

    def test_outer_iters_filter_selects_main_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = MertonFigureFixture(root)
            fixture.add_run(1, outer_iters=20)
            fixture.add_run(2, outer_iters=20)
            fixture.add_run(3, outer_iters=30)
            fixture.add_run(4, outer_iters=30)
            output = self._run(root, "--outer-iters", "20")
            with (output / "figure2_trajectories.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["seed"]) for row in rows}, {1, 2})
            self.assertEqual(max(int(row["outer_iter"]) for row in rows), 20)

    def test_deprecated_m_states_rejects_non_merton_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            with self.assertRaisesRegex(ValueError, "must be 1"):
                self._run(root, "--m-states", "3")

    def test_skip_decay_fits_omits_only_optional_fit_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = self._run(root, "--skip-decay-fits")
            self.assertFalse((output / "figure2_seed_decay_fits.csv").exists())
            self.assertFalse((output / "figure2_decay_summary.csv").exists())
            self.assertTrue((output / "figure2_trajectories.csv").is_file())

    def test_png_and_eps_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = root / "figures"
            code = contraction.main(
                [
                    "--out-root", str(root),
                    "--output", str(output),
                    "--n-assets", "10",
                    "--expected-seeds", "1,2",
                    "--min-seeds", "2",
                    "--formats", "png,eps",
                    "--dpi", "72",
                ]
            )
            self.assertEqual(code, 0)
            self.assertGreater(
                (output / "figure2_empirical_convergence.png").stat().st_size, 0
            )
            self.assertGreater(
                (output / "figure2_empirical_convergence.eps").stat().st_size, 0
            )


if __name__ == "__main__":
    unittest.main()
