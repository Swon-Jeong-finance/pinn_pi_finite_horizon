#!/usr/bin/env python3
"""Auxiliary PyTorch-free regression tests for strict Merton E8 aggregation."""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import aggregate_compute as e8


class ComputeFixture:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def write_market(path: Path, *, shift: float = 0.0) -> None:
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
        method: str,
        seed: int,
        *,
        timing_mode: bool = True,
        skip_figures: bool = True,
        market_shift: float = 0.0,
        batch_size: int = 10000,
        omit_error: str = "",
        add_outer_e_xev: bool = False,
        duplicate_error: str = "",
        outer_iters: int = 20,
    ) -> Path:
        run = self.root / method / (
            f"{method}_n_assets10_outer_iters{outer_iters}_seed{seed}")
        run.mkdir(parents=True)
        args = {
            "model_type": method,
            "run_tag": run.name,
            "n_assets": 10,
            "m_states": 1,
            "seed": seed,
            "market_seed": 12,
            "timing_mode": timing_mode,
            "skip_figures": skip_figures,
            "skip_plots": False,
            "skip_eval": False,
            "eval_only": False,
            "test_points": 100000,
            "eval_margin": "0.10,0.0,0.05",
            "batch_size": batch_size,
            "outer_iters": outer_iters,
            "eval_epochs": 2000,
            "device": f"cuda:{seed}",
            "output_root": str(self.root),
            "weight_root": str(self.root / "weights" / run.name),
        }
        (run / "config.json").write_text(json.dumps({"args": args}), encoding="utf-8")
        scale = 1.0 if method == "pinn" else 1.5
        status = {
            "status": "success",
            "model_type": method,
            "run_tag": run.name,
            "updated_at": f"2026-07-22T00:00:{seed:02d}",
            "primary_margin": 0.10,
            "eval_margins": [0.10, 0.0, 0.05],
            "train_wall_sec": 100.0 * scale + seed,
            "total_optimizer_steps": 40000,
            "train_gpu_peak_mem_bytes": int((1000 + 10 * seed) * 2**20),
            "eval_gpu_peak_mem_bytes": int((500 + 5 * seed) * 2**20),
        }
        (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (run / "_SUCCESS").touch()
        self.write_market(run / "market_params.npz", shift=market_shift)

        rows = []
        values = {
            "RelL2_V": 0.01 * scale * seed,
            "RelL2_D": 0.015 * scale * seed,
            "RelL2_pi": 0.02 * scale * seed,
            "RelL2_c": 0.03 * scale * seed,
            "e_Xev": 0.04 * scale * seed,
        }
        for margin in (0.10, 0.0, 0.05):
            for name, value in values.items():
                if name == omit_error:
                    continue
                rows.append({
                    "scope": "fulldim",
                    "eval_margin": margin,
                    "metric": name,
                    "value": value + margin,
                })
                if name == duplicate_error and math.isclose(margin, 0.10):
                    rows.append(dict(rows[-1]))
        with (run / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["scope", "eval_margin", "metric", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)
        if add_outer_e_xev:
            (run / "outer_history.csv").write_text("outer_iter,e_Xev\n20,0.001\n", encoding="utf-8")
        return run

    def complete(self) -> None:
        for method in ("pinn", "pipinn"):
            self.add_run(method, 1)
            self.add_run(method, 2)


class AggregateComputeTests(unittest.TestCase):
    @staticmethod
    def run_ok(root: Path, *extra: str) -> Path:
        output = root / "derived"
        result = e8.main([
            "--out-root", str(root),
            "--output", str(output),
            "--expected-seeds", "1,2",
            "--expected-n-assets", "10",
            "--min-seeds", "2",
            "--require-sample-sd",
            "--no-plots",
            *extra,
        ])
        if result != 0:
            raise AssertionError(result)
        return output

    def test_strict_panel_writes_cost_error_and_scatter_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ComputeFixture(root).complete()
            output = self.run_ok(root)
            with (output / "e8_compute_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            row = next(
                x for x in summary
                if x["model_type"] == "pinn" and x["metric"] == "train_wall_sec"
            )
            self.assertEqual(row["n"], "2")
            self.assertAlmostEqual(float(row["mean"]), 101.5)
            self.assertAlmostEqual(float(row["std"]), math.sqrt(0.5))
            self.assertIn("+/-", row["mean_plus_minus_sample_sd"])

            with (output / "e8_per_run.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                per_run = list(csv.DictReader(handle))
            self.assertEqual(len(per_run), 4)
            self.assertEqual(float(per_run[0]["train_gpu_peak_mem_mib"]), 1010.0)
            with (output / "e8_error_vs_compute.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                points = list(csv.DictReader(handle))
            self.assertEqual(len(points), 4 * 4 * 4)
            metadata = json.loads((output / "e8_metadata.json").read_text(encoding="utf-8"))
            self.assertIn("no cross-margin", metadata["error_semantics"]["RelL2_V"])
            self.assertTrue(metadata["eligibility"]["timing_mode"])

    def test_n_equals_one_is_explicitly_na_unless_sd_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1)
            fixture.add_run("pipinn", 1)
            output = root / "single"
            self.assertEqual(e8.main([
                "--out-root", str(root), "--output", str(output),
                "--expected-seeds", "1", "--expected-n-assets", "10",
                "--no-plots",
            ]), 0)
            with (output / "e8_compute_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["std"] == "nan" for row in rows))
            self.assertTrue(all("n=1" in row["mean_plus_minus_sample_sd"] for row in rows))

    def test_non_timing_run_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1, timing_mode=False)
            fixture.add_run("pipinn", 1)
            with self.assertRaisesRegex(ValueError, "timing_mode=true"):
                e8.main(["--out-root", str(root), "--no-plots"])

    def test_eval_figures_are_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1, skip_figures=False)
            fixture.add_run("pipinn", 1)
            with self.assertRaisesRegex(ValueError, "contaminate eval_gpu"):
                e8.main(["--out-root", str(root), "--no-plots"])

    def test_e_xev_is_never_backfilled_from_outer_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1, omit_error="e_Xev", add_outer_e_xev=True)
            fixture.add_run("pipinn", 1, omit_error="e_Xev", add_outer_e_xev=True)
            with self.assertRaisesRegex(ValueError, "No outer_history fallback"):
                e8.main([
                    "--out-root", str(root), "--error-metrics", "e_Xev", "--no-plots"
                ])

    def test_explicit_metrics_csv_e_xev_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1)
            fixture.add_run("pipinn", 1)
            output = root / "e_xev"
            self.assertEqual(e8.main([
                "--out-root", str(root), "--output", str(output),
                "--error-metrics", "e_Xev", "--no-plots",
            ]), 0)
            with (output / "e8_error_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["metric"] for row in rows}, {"e_Xev"})

    def test_duplicate_official_metric_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1, duplicate_error="RelL2_V")
            fixture.add_run("pipinn", 1)
            with self.assertRaisesRegex(ValueError, "found 2"):
                e8.main(["--out-root", str(root), "--no-plots"])

    def test_market_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1)
            fixture.add_run("pipinn", 1, market_shift=0.001)
            with self.assertRaisesRegex(ValueError, "distinct canonical Merton markets"):
                e8.main(["--out-root", str(root), "--no-plots"])

    def test_method_config_change_across_seeds_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1)
            fixture.add_run("pinn", 2, batch_size=5000)
            fixture.add_run("pipinn", 1)
            fixture.add_run("pipinn", 2)
            with self.assertRaisesRegex(ValueError, "multiple method configurations"):
                e8.main([
                    "--out-root", str(root), "--expected-seeds", "1,2", "--no-plots"
                ])

    def test_outer_20_and_30_are_distinct_valid_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            for method in ("pinn", "pipinn"):
                fixture.add_run(method, 1, outer_iters=20)
                fixture.add_run(method, 1, outer_iters=30)
            output = root / "budgets"
            self.assertEqual(e8.main([
                "--out-root", str(root), "--output", str(output),
                "--expected-seeds", "1", "--expected-n-assets", "10",
                "--no-plots",
            ]), 0)
            with (output / "e8_per_run.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({int(row["outer_iters"]) for row in rows}, {20, 30})
            self.assertEqual(len({row["setting_id"] for row in rows}), 2)

    def test_compute_and_error_figures_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ComputeFixture(root)
            fixture.add_run("pinn", 1)
            fixture.add_run("pipinn", 1)
            output = root / "figures"
            self.assertEqual(e8.main([
                "--out-root", str(root),
                "--output", str(output),
                "--expected-seeds", "1",
                "--expected-n-assets", "10",
                "--error-metrics", "RelL2_V",
                "--formats", "png",
                "--dpi", "72",
            ]), 0)
            self.assertGreater((output / "e8_compute_costs.png").stat().st_size, 0)
            self.assertGreater(
                (output / "e8_error_vs_compute_RelL2_V.png").stat().st_size, 0
            )


if __name__ == "__main__":
    unittest.main()
