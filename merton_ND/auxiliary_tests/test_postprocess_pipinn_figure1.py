#!/usr/bin/env python3
"""Auxiliary tests for postprocess_pipinn_figure1.py (no torch needed)."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import postprocess_pipinn_figure1 as figure1


class Figure1Fixture:
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
        outer_iters: int = 4,
        rows: int | None = None,
        batch_size: int = 10000,
        market_shift: float = 0.0,
        scope: str = "fixed_qev",
        duplicate_outer: bool = False,
        diag_points: int = 8192,
        metric_points: int = 8281,
        metric_points_last: int | None = None,
        zero_metrics: tuple[str, ...] = (),
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
            "diag_points": diag_points,
            "diag_every": 1,
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
            json.dumps(
                {
                    "status": "success",
                    "updated_at": f"2026-07-23T00:00:{seed:02d}",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "_SUCCESS").touch()
        self._market(run_dir / "market_params.npz", shift=market_shift)

        count = outer_iters if rows is None else rows
        history_rows = []
        for index in range(count):
            factor = 0.5 ** index
            history_rows.append(
                {
                    "outer_iter": index + 1,
                    "c_diff": (
                        0.0 if "c_diff" in zero_metrics else 0.10 * factor * seed
                    ),
                    "pi_diff": (
                        0.0 if "pi_diff" in zero_metrics else 0.20 * factor * seed
                    ),
                    "c_vs_closed_form": (
                        0.0
                        if "c_vs_closed_form" in zero_metrics
                        else 0.30 * factor * seed
                    ),
                    "pi_vs_closed_form": (
                        0.0
                        if "pi_vs_closed_form" in zero_metrics
                        else 0.40 * factor * seed
                    ),
                    "control_metric_scope": scope,
                    "control_metric_points": (
                        metric_points_last
                        if metric_points_last is not None and index == count - 1
                        else metric_points
                    ),
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
                    *figure1.METRICS,
                    "control_metric_scope",
                    "control_metric_points",
                ],
            )
            writer.writeheader()
            writer.writerows(history_rows)
        return run_dir


class Figure1PostprocessorTests(unittest.TestCase):
    @staticmethod
    def _complete(root: Path) -> Figure1Fixture:
        fixture = Figure1Fixture(root)
        fixture.add_run(1)
        fixture.add_run(2)
        return fixture

    @staticmethod
    def _run(root: Path, *extra: str) -> Path:
        output = root / "derived"
        code = figure1.main(
            [
                "--out-root",
                str(root),
                "--output",
                str(output),
                "--n-assets",
                "10",
                "--outer-iters",
                "4",
                "--expected-seeds",
                "1,2",
                "--min-seeds",
                "2",
                "--no-plots",
                *extra,
            ]
        )
        if code != 0:
            raise AssertionError(f"postprocessor returned {code}")
        return output

    def test_pointwise_mean_and_sample_sd_use_seeds_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = self._run(root)
            with (output / "figure1_pointwise_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            row = next(
                item
                for item in rows
                if item["metric"] == "c_diff" and item["outer_iter"] == "1"
            )
            self.assertEqual(row["n_seeds"], "2")
            self.assertAlmostEqual(float(row["mean"]), 0.15)
            self.assertAlmostEqual(float(row["sample_sd"]), np.std([0.1, 0.2], ddof=1))

            metadata = json.loads(
                (output / "figure1_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["source_metrics"], list(figure1.METRICS))
            self.assertEqual(metadata["control_metric_scope"], "fixed_qev")
            self.assertEqual(metadata["diag_points_requested"], 8192)
            self.assertEqual(metadata["diag_points_actual"], 8281)
            self.assertIn("outer_iter=k", metadata["indexing_contract"])
            self.assertIn("outer iterations are not replicates", metadata["aggregation"])

            with (output / "figure1_runs_used.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                run_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["diag_points_requested"] for row in run_rows}, {"8192"}
            )
            self.assertEqual(
                {row["diag_points_actual"] for row in run_rows}, {"8281"}
            )

    def test_realized_grid_points_must_be_positive_and_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, metric_points=8281)
            fixture.add_run(2, metric_points=8200)
            with self.assertRaisesRegex(ValueError, "uses 8200 fixed-Q_ev points"):
                self._run(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, metric_points_last=8200)
            fixture.add_run(2)
            with self.assertRaisesRegex(
                ValueError, "changed across outer iterations"
            ):
                self._run(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, metric_points=0)
            fixture.add_run(2, metric_points=0)
            with self.assertRaisesRegex(ValueError, "invalid control_metric_points"):
                self._run(root)

    def test_exact_zero_metrics_are_preserved_and_plot_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, zero_metrics=("c_diff",))
            fixture.add_run(2, zero_metrics=("c_diff",))
            output = root / "figures"
            code = figure1.main(
                [
                    "--out-root",
                    str(root),
                    "--output",
                    str(output),
                    "--n-assets",
                    "10",
                    "--outer-iters",
                    "4",
                    "--expected-seeds",
                    "1,2",
                    "--min-seeds",
                    "2",
                    "--formats",
                    "png",
                    "--dpi",
                    "72",
                ]
            )
            self.assertEqual(code, 0)
            self.assertGreater(
                (output / f"{figure1.OUTPUT_BASENAME}.png").stat().st_size, 0
            )
            with (output / "figure1_control_trajectories.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            c_rows = [row for row in rows if row["metric"] == "c_diff"]
            self.assertTrue(c_rows)
            self.assertTrue(all(float(row["value"]) == 0.0 for row in c_rows))

            metadata = json.loads(
                (output / "figure1_metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(metadata["raw_zero_values_preserved"])
            self.assertGreater(metadata["log_plot_floor"], 0.0)
            self.assertEqual(
                metadata["zero_mean_points_plot_floored"]["c_diff"], 4
            )

    def test_exact_seed_set_and_complete_outer_grid_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1)
            fixture.add_run(2, rows=3)
            with self.assertRaisesRegex(ValueError, "must cover outer 1..4 exactly"):
                self._run(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1)
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                self._run(root)

    def test_duplicate_outer_and_non_fixed_scope_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1)
            fixture.add_run(2, duplicate_outer=True)
            with self.assertRaisesRegex(ValueError, "duplicate outer_iter=4"):
                self._run(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, scope="training_batch_fallback")
            fixture.add_run(2, scope="training_batch_fallback")
            with self.assertRaisesRegex(ValueError, "requires fixed_qev"):
                self._run(root)

    def test_configuration_and_market_mixing_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1, batch_size=10000)
            fixture.add_run(2, batch_size=5000)
            with self.assertRaisesRegex(ValueError, "exactly one eligible Merton"):
                self._run(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = Figure1Fixture(root)
            fixture.add_run(1)
            fixture.add_run(2, market_shift=0.001)
            with self.assertRaisesRegex(ValueError, "distinct canonical Merton markets"):
                self._run(root)

    def test_png_and_eps_are_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._complete(root)
            output = root / "figures"
            code = figure1.main(
                [
                    "--out-root",
                    str(root),
                    "--output",
                    str(output),
                    "--n-assets",
                    "10",
                    "--outer-iters",
                    "4",
                    "--expected-seeds",
                    "1,2",
                    "--min-seeds",
                    "2",
                    "--formats",
                    "png,eps",
                    "--dpi",
                    "72",
                ]
            )
            self.assertEqual(code, 0)
            self.assertGreater(
                (output / f"{figure1.OUTPUT_BASENAME}.png").stat().st_size, 0
            )
            self.assertGreater(
                (output / f"{figure1.OUTPUT_BASENAME}.eps").stat().st_size, 0
            )


if __name__ == "__main__":
    unittest.main()
