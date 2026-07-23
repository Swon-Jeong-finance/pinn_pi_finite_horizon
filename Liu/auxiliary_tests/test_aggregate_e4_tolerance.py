"""Focused regression tests for residual-tolerance E4 aggregation."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from aggregate_e4_tolerance import _canonical_market_hash, _commit_stage, aggregate
from aggregate_liu_exact_map import sha256_file
from test_aggregate_liu_exact_map import make_result, refresh_artifact_hashes, write_csv


def _rewrite_csv(path: Path, update) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        update(row)
    write_csv(path, rows)


def configure_cell(root: Path, seed: int, tolerance: float, achieved: float) -> Path:
    directory = make_result(root, seed)
    run_dir = directory.parent

    # Every training seed uses the same actual market snapshot.  Only the
    # excluded network seed differs.
    market_path = run_dir / "market_params.npz"
    np.savez(
        market_path,
        seed=np.asarray(seed),
        market_seed=np.asarray(123),
        K=np.asarray([[0.25]]),
        SigmaX=np.asarray([[0.15]]),
    )
    market_hash = _canonical_market_hash(market_path)

    source_config_path = run_dir / "config.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_config["args"].update(
        {
            "seed": seed,
            "model_type": "pipinn",
            "m_states": 1,
            "outer_iters": 2,
            "pres_target": tolerance,
        }
    )
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "updated_at": f"2026-07-23T00:00:{seed:02d}+00:00",
                "pres_max": achieved,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "_SUCCESS").touch()

    for name in ("exact_map_ratios.csv", "e4_approximation_errors.csv"):
        _rewrite_csv(directory / name, lambda row: row.update(market_sha256=market_hash))
    # Refinement CSVs are independently hashed and use the same identity.
    for name in ("exact_map_refinement.csv", "e4_approximation_refinement.csv"):
        _rewrite_csv(directory / name, lambda row: row.update(market_sha256=market_hash))

    config_path = directory / "exact_map_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "run_dir": str(run_dir),
            "config_sha256": sha256_file(source_config_path),
            "market_file_sha256": sha256_file(market_path),
            "market_sha256": market_hash,
            "training_protocol_args": {
                "outer_iters": 2,
                "pres_target": tolerance,
                "lr": 1.0e-3,
            },
            "grid": {
                "base_ny": 11,
                "base_nx": 11,
                "base_nt": 20,
                "eval_ny": 11,
                "eval_nx": 11,
                "grid_factors": [1, 2],
                "domain_factors": [1.5, 2.0],
                "boundaries": ["linearity", "exact-dirichlet"],
                "verify_checkpoints": "all",
                "drift_scheme": "adaptive",
                "peclet_limit": 1.0,
                "theta_method": 0.5,
                "startup_be_steps": 2,
            },
            "refinement_abs_tolerance": 1.0e-2,
            "refinement_rel_tolerance": 2.0e-2,
            "denominator_tolerance": 1.0e-12,
            "norm": "test X norm",
            "indexing": "test shifted E4 indexing",
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    refresh_artifact_hashes(directory)
    return directory


class AggregateE4ToleranceTests(unittest.TestCase):
    @staticmethod
    def _run(result_dirs, output, **overrides):
        kwargs = {
            "expected_seeds": [1, 3],
            "expected_tolerances": [1.0e-2, 1.0e-3],
            "min_runs_per_tolerance": 2,
            "checkpoints": None,
            "make_plot": False,
            "plot_metrics": [],
            "formats": ["png"],
            "figure_size": (6.4, 4.5),
            "dpi": 100,
            "font_size": 10.0,
            "overwrite": False,
        }
        kwargs.update(overrides)
        return aggregate(result_dirs, output, **kwargs)

    def test_balanced_cells_form_seedwise_worst_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dirs = []
            for tolerance, achieved in ((1.0e-2, 8.0e-3), (1.0e-3, 8.0e-4)):
                for seed in (1, 3):
                    result_dirs.append(
                        configure_cell(
                            root / f"tol-{tolerance:g}", seed, tolerance, achieved
                        )
                    )
            output = root / "paper"
            status = self._run(result_dirs, output)
            self.assertEqual(status["seeds"], [1, 3])
            self.assertEqual(status["tolerances"], [1.0e-3, 1.0e-2])
            self.assertTrue((output / "_SUCCESS_E4_TOLERANCE_AGG").is_file())
            with (output / "e4_tolerance_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            target = next(
                row
                for row in rows
                if math.isclose(float(row["pres_target"]), 1.0e-2)
                and row["metric"] == "max_e_approx_X"
            )
            # Each seed is maximized over outers first: 0.3 and 0.5.
            self.assertAlmostEqual(float(target["mean"]), 0.4)
            self.assertEqual(int(target["n"]), 2)
            self.assertTrue(math.isfinite(float(target["ci95_high"])))

    def test_commit_failure_restores_previous_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            stage = root / "stage"
            output.mkdir()
            stage.mkdir()
            (output / "e4_tolerance_per_seed.csv").write_text(
                "old\n", encoding="utf-8"
            )
            (output / "_SUCCESS_E4_TOLERANCE_AGG").write_text(
                "old-success\n", encoding="utf-8"
            )
            (stage / "e4_tolerance_per_seed.csv").write_text(
                "new\n", encoding="utf-8"
            )
            (stage / "e4_tolerance_summary.csv").write_text(
                "new-summary\n", encoding="utf-8"
            )
            (stage / "_SUCCESS_E4_TOLERANCE_AGG").write_text(
                "new-success\n", encoding="utf-8"
            )
            import aggregate_e4_tolerance as module
            real_replace = module.os.replace

            def fail_second_stage_move(source, destination):
                source_path = Path(source)
                if source_path.parent == stage and source_path.name == "e4_tolerance_summary.csv":
                    raise OSError("synthetic E4 commit failure")
                return real_replace(source, destination)

            with mock.patch.object(
                module.os, "replace", side_effect=fail_second_stage_move
            ):
                with self.assertRaisesRegex(OSError, "synthetic E4 commit failure"):
                    _commit_stage(stage, output)
            self.assertEqual(
                (output / "e4_tolerance_per_seed.csv").read_text(encoding="utf-8"),
                "old\n",
            )
            self.assertEqual(
                (output / "_SUCCESS_E4_TOLERANCE_AGG").read_text(encoding="utf-8"),
                "old-success\n",
            )
            self.assertFalse((output / "e4_tolerance_summary.csv").exists())

    def test_pres_target_is_only_removed_training_protocol_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = configure_cell(root / "a", 1, 1.0e-2, 8.0e-3)
            second = configure_cell(root / "b", 1, 1.0e-3, 8.0e-4)
            config_path = second / "exact_map_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["training_protocol_args"]["lr"] = 2.0e-3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            refresh_artifact_hashes(second)
            with self.assertRaisesRegex(ValueError, "canonical protocol mismatch"):
                self._run(
                    [first, second],
                    root / "paper",
                    expected_seeds=[1],
                    min_runs_per_tolerance=1,
                )

    def test_exact_tolerance_and_seed_sets_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = configure_cell(root / "a", 1, 1.0e-2, 8.0e-3)
            with self.assertRaisesRegex(ValueError, "residual-tolerance set mismatch"):
                self._run(
                    [result],
                    root / "paper",
                    expected_seeds=[1],
                    min_runs_per_tolerance=1,
                )

    def test_canonical_market_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = configure_cell(root / "a", 1, 1.0e-2, 8.0e-3)
            second = configure_cell(root / "b", 1, 1.0e-3, 8.0e-4)
            market_path = second.parent / "market_params.npz"
            np.savez(
                market_path,
                seed=np.asarray(1),
                market_seed=np.asarray(999),
                K=np.asarray([[0.25]]),
                SigmaX=np.asarray([[0.15]]),
            )
            changed_hash = _canonical_market_hash(market_path)
            for name in (
                "exact_map_ratios.csv",
                "e4_approximation_errors.csv",
                "exact_map_refinement.csv",
                "e4_approximation_refinement.csv",
            ):
                _rewrite_csv(
                    second / name,
                    lambda row: row.update(market_sha256=changed_hash),
                )
            config_path = second / "exact_map_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["market_sha256"] = changed_hash
            config["market_file_sha256"] = sha256_file(market_path)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            refresh_artifact_hashes(second)
            with self.assertRaisesRegex(ValueError, "market snapshot mismatch"):
                self._run(
                    [first, second],
                    root / "paper",
                    expected_seeds=[1],
                    min_runs_per_tolerance=1,
                )

    def test_discovered_failed_attempt_cannot_revive_old_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = configure_cell(root / "old", 1, 1.0e-2, 8.0e-3)
            failed = root / "new" / "liu_exact_map_fd"
            failed.mkdir(parents=True)
            (failed / "exact_map_status.json").write_text(
                json.dumps({"status": "failed", "error": "new failure"}),
                encoding="utf-8",
            )
            (failed / "_FAILED_EXACT_MAP").touch()
            newer = old.stat().st_mtime_ns + 10_000_000
            os.utime(failed / "exact_map_status.json", ns=(newer, newer))
            with self.assertRaisesRegex(ValueError, "attempt is not successful"):
                self._run(
                    [old, failed],
                    root / "paper",
                    expected_seeds=[1],
                    expected_tolerances=[1.0e-2],
                    min_runs_per_tolerance=1,
                )

    def test_single_seed_sd_and_ci_are_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = configure_cell(root / "a", 1, 1.0e-2, 8.0e-3)
            output = root / "paper"
            self._run(
                [result],
                output,
                expected_seeds=[1],
                expected_tolerances=[1.0e-2],
                min_runs_per_tolerance=1,
            )
            with (output / "e4_tolerance_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertTrue(math.isnan(float(row["std"])))
            self.assertTrue(math.isnan(float(row["ci95_low"])))

    def test_optional_plot_and_transactional_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = configure_cell(root / "a", 1, 1.0e-2, 8.0e-3)
            output = root / "paper"
            common = {
                "expected_seeds": [1],
                "expected_tolerances": [1.0e-2],
                "min_runs_per_tolerance": 1,
            }
            self._run(
                [result],
                output,
                **common,
                make_plot=True,
                plot_metrics=["max_e_approx_value", "max_e_approx_X"],
            )
            plot = output / "e4_tolerance_errors.png"
            self.assertTrue(plot.is_file())
            old_status = (output / "e4_tolerance_aggregate_status.json").read_bytes()

            # A failing replacement must leave the prior successful output
            # (including its plot) byte-for-byte intact.
            with self.assertRaisesRegex(ValueError, "tolerance set mismatch"):
                self._run(
                    [result],
                    output,
                    **{
                        **common,
                        "overwrite": True,
                        "expected_tolerances": [1.0e-2, 1.0e-3],
                    },
                )
            self.assertEqual(
                (output / "e4_tolerance_aggregate_status.json").read_bytes(), old_status
            )
            self.assertTrue(plot.is_file())


if __name__ == "__main__":
    unittest.main()
