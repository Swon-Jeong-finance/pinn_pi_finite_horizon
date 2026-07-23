import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import aggregate_e6 as e6
from aggregate_seeds import MARKET_HASH_KEYS


def _write_run(
    root: Path,
    name: str,
    *,
    seed: int,
    tolerance: float,
    status: str = "success",
    updated_at: str = "2026-07-23T00:00:00",
    market_value: float = 1.0,
    eval_margin: float = 0.10,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    args = {
        "model_type": "pipinn",
        "n_assets": 30,
        "m_states": 3,
        "seed": seed,
        "market_seed": 12,
        "pres_target": tolerance,
        "eval_margin": f"{eval_margin:g},0.30",
        "outer_iters": 20,
        "batch_size": 64,
        "output_root": str(run),
        "weight_root": str(run / "weights"),
        "run_tag": name,
        "device": "cuda:0",
    }
    (run / "config.json").write_text(
        json.dumps({"args": args}), encoding="utf-8"
    )
    payload = {
        "status": status,
        "updated_at": updated_at,
        "pres_max": tolerance * (0.8 + 0.01 * seed),
        "total_inner_steps": 1000 + seed,
    }
    (run / "status.json").write_text(json.dumps(payload), encoding="utf-8")
    marker = {
        "success": "_SUCCESS",
        "failed": "_FAILED",
        "stopped_early": "_STOPPED_EARLY",
    }[status]
    (run / marker).touch()
    with (run / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["scope", "eval_margin", "metric", "value"]
        )
        writer.writeheader()
        writer.writerow({
            "scope": "fulldim", "eval_margin": eval_margin,
            "metric": "RelL2_V", "value": tolerance ** 0.5 * (1 + seed / 100),
        })
        writer.writerow({
            "scope": "fulldim", "eval_margin": eval_margin,
            "metric": "RelL2_theta", "value": tolerance ** 0.4 * (1 + seed / 100),
        })
    market = {key: np.asarray([market_value]) for key in MARKET_HASH_KEYS}
    np.savez(run / "market_params.npz", **market)
    return run


class AggregateE6Tests(unittest.TestCase):
    def test_one_seed_has_undefined_sample_uncertainty(self):
        stats = e6._summary_stats([2.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertTrue(math.isnan(stats["std"]))
        self.assertTrue(math.isnan(stats["ci95_lo"]))

    def test_complete_exact_design_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for tolerance in (1e-2, 1e-3):
                for seed in (1, 2):
                    _write_run(
                        root, f"tol{tolerance}_seed{seed}",
                        seed=seed, tolerance=tolerance,
                    )
            output = root / "derived" / "e6"
            metadata = e6.main([
                "--out-root", str(root),
                "--output", str(output),
                "--expected-seeds", "1,2",
                "--expected-tolerances", "1e-2,1e-3",
                "--min-runs-per-tolerance", "2",
                "--expected-n-assets", "30",
                "--expected-m-states", "3",
                "--skip-plot",
            ])
            self.assertEqual(metadata["n_point_rows"], 8)
            for name in (
                "points.csv", "per_target.csv", "fit.csv",
                "e6_metadata.json", "_SUCCESS_E6",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "per_target.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["error_ci95_lo"] for row in rows))

    def test_newer_failure_invalidates_older_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(
                root, "old_success", seed=1, tolerance=1e-3,
                status="success", updated_at="2026-07-23T00:00:00",
            )
            _write_run(
                root, "new_failure", seed=1, tolerance=1e-3,
                status="failed", updated_at="2026-07-23T01:00:00",
            )
            with self.assertRaisesRegex(ValueError, "accepted seeds|tolerances"):
                e6.main([
                    "--out-root", str(root), "--expected-seeds", "1",
                    "--expected-tolerances", "1e-3", "--skip-plot",
                ])

    def test_mixed_primary_margins_cannot_fragment_balanced_seed_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "seed1", seed=1, tolerance=1e-3, eval_margin=0.10)
            _write_run(root, "seed2", seed=2, tolerance=1e-3, eval_margin=0.20)
            with self.assertRaisesRegex(ValueError, "mixed primary eval margins"):
                e6.main([
                    "--out-root", str(root), "--expected-seeds", "1,2",
                    "--expected-tolerances", "1e-3",
                    "--min-runs-per-tolerance", "2", "--skip-plot",
                ])

    def test_exact_tolerance_set_and_market_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "a", seed=1, tolerance=1e-2)
            _write_run(root, "b", seed=1, tolerance=1e-3, market_value=2.0)
            with self.assertRaisesRegex(ValueError, "market snapshot"):
                e6.main([
                    "--out-root", str(root), "--expected-seeds", "1",
                    "--expected-tolerances", "1e-2,1e-3", "--skip-plot",
                ])
            with self.assertRaisesRegex(ValueError, "tolerances"):
                e6.main([
                    "--out-root", str(root), "--expected-seeds", "1",
                    "--expected-tolerances", "1e-2", "--skip-plot",
                ])

    def test_failed_validation_does_not_replace_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "only_seed1", seed=1, tolerance=1e-3)
            output = root / "derived"
            output.mkdir()
            old = output / "points.csv"
            old.write_text("old-good-output\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                e6.main([
                    "--out-root", str(root), "--output", str(output),
                    "--expected-seeds", "1,2", "--expected-tolerances", "1e-3",
                    "--skip-plot", "--overwrite",
                ])
            self.assertEqual(old.read_text(encoding="utf-8"), "old-good-output\n")

    def test_overwrite_preserves_unrelated_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "run", seed=1, tolerance=1e-3)
            output = root / "derived"
            output.mkdir()
            (output / "points.csv").write_text("old\n", encoding="utf-8")
            unknown = output / ".ipynb_checkpoints"
            unknown.mkdir()
            e6.main([
                "--out-root", str(root), "--output", str(output),
                "--expected-seeds", "1", "--expected-tolerances", "1e-3",
                "--skip-plot", "--overwrite",
            ])
            self.assertTrue(unknown.is_dir())
            self.assertNotEqual((output / "points.csv").read_text(), "old\n")

    def test_commit_failure_rolls_back_previous_complete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            stage = root / "stage"
            output.mkdir()
            stage.mkdir()
            (output / "points.csv").write_text("old-points\n", encoding="utf-8")
            (output / "_SUCCESS_E6").write_text("old-success\n", encoding="utf-8")
            (stage / "points.csv").write_text("new-points\n", encoding="utf-8")
            (stage / "per_target.csv").write_text("new-summary\n", encoding="utf-8")
            (stage / "_SUCCESS_E6").write_text("new-success\n", encoding="utf-8")
            real_replace = e6.os.replace

            def fail_second_stage_move(source, destination):
                source_path = Path(source)
                if source_path.parent == stage and source_path.name == "points.csv":
                    raise OSError("synthetic commit failure")
                return real_replace(source, destination)

            with mock.patch.object(e6.os, "replace", side_effect=fail_second_stage_move):
                with self.assertRaisesRegex(OSError, "synthetic commit failure"):
                    e6._commit_e6_outputs(stage, output, overwrite=True)
            self.assertEqual(
                (output / "points.csv").read_text(encoding="utf-8"), "old-points\n"
            )
            self.assertEqual(
                (output / "_SUCCESS_E6").read_text(encoding="utf-8"), "old-success\n"
            )
            self.assertFalse((output / "per_target.csv").exists())


if __name__ == "__main__":
    unittest.main()
