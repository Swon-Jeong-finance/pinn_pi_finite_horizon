"""Regression tests for strict Liu exact-map seed aggregation."""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aggregate_liu_exact_map import (
    E4_METRICS,
    EXACT_METRICS,
    HASHED_INPUTS,
    _stats,
    _commit_staged_output,
    aggregate,
    sha256_file,
)


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh_artifact_hashes(directory: Path) -> None:
    status_path = directory / "exact_map_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifact_sha256"] = {
        name: sha256_file(directory / name) for name in HASHED_INPUTS
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")


def update_exact_rows(directory: Path, updates) -> None:
    """Apply ``updates(row)`` to primary exact rows and refresh the manifest."""

    path = directory / "exact_map_ratios.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        updates(row)
    write_csv(path, rows)
    refresh_artifact_hashes(directory)


def make_result(root: Path, seed: int, *, undefined: bool = False) -> Path:
    analysis_mode = "exact_map_and_e4"
    policy_extension = "boundary-projection"
    map_definition = "finite_domain_boundary_projected_policy_extension"
    whole_space_map_claim = "not_a_whole_space_map"
    run_dir = root / f"seed{seed}"
    directory = run_dir / "liu_exact_map_fd"
    directory.mkdir(parents=True)
    weight_dir = run_dir / "weights"
    (weight_dir / "iterates").mkdir(parents=True)
    source_config = run_dir / "config.json"
    source_config.write_text(json.dumps({
        "cwd": str(run_dir), "weight_dir": str(weight_dir), "args": {"seed": seed},
    }), encoding="utf-8")
    market_path = run_dir / "market_params.npz"
    market_path.write_bytes(f"market-{seed}".encode())
    checkpoint_paths = {}
    checkpoint_hashes = {}
    for outer in (1, 2):
        path = weight_dir / "iterates" / f"value_net_iter{outer:04d}.pt"
        path.write_bytes(f"checkpoint-{seed}-{outer}".encode())
        checkpoint_paths[outer] = path
        checkpoint_hashes[str(outer)] = sha256_file(path)
    common = {
        "problem": "liu", "group": "same-group", "protocol_hash": "same-protocol",
        "market_sha256": "same-market", "seed": str(seed), "is_primary": "1",
        "analysis_mode": analysis_mode, "policy_extension": policy_extension,
        "map_definition": map_definition,
        "refinement_status": "pass",
    }
    exact_rows = []
    e4_rows = []
    for outer in (1, 2):
        exact = {
            **common, "source_outer_iter": str(outer),
            "frozen_policy_iter": str(outer - 1), "greedy_policy_iter": str(outer),
            "target_value_outer_iter": str(outer + 1),
            "checkpoint": str(checkpoint_paths[outer]),
            "checkpoint_sha256": checkpoint_hashes[str(outer)],
            "policy_hash": f"policy-{seed}-{outer}",
            "denominator_defined": "0" if undefined and outer == 1 else "1",
            "checkpoint_selection": "all",
            "local_map_unmodified_on_xfd": "1",
            "local_greedy_unmodified_on_policy_support": "1",
            "map_variant": "locally_unmodified_on_sampled_xfd",
            "whole_space_map_claim": whole_space_map_claim,
            "linear_residual_tolerance": "1e-8",
            "boundary_elimination_size": "2",
            "boundary_elimination_rank": "2",
        }
        exact.update({metric: str(float(seed + outer)) for metric in EXACT_METRICS})
        exact["nonpositive_log_eig_fraction"] = "0.0"
        exact["outside_collocation_fraction_fd"] = "0.0"
        exact["outside_collocation_y_fraction_fd"] = "0.0"
        exact["outside_collocation_x_fraction_fd"] = "0.0"
        exact["min_linear_system_lu_pivot_ratio"] = "1.0"
        if undefined and outer == 1:
            exact["rho_exact"] = "nan"
        exact_rows.append(exact)
        source = outer - 1
        e4 = {
            **common, "target_outer_iter": str(outer),
            "frozen_policy_iter": str(source), "policy_source_outer_iter": str(source),
            "checkpoint": str(checkpoint_paths[outer]),
            "checkpoint_sha256": checkpoint_hashes[str(outer)],
            "source_policy_hash": (f"alpha0-{seed}" if source == 0 else f"policy-{seed}-{source}"),
            "fd_reference_source": ("analytic_alpha0_fd_solve" if source == 0
                                    else f"reused_exact_map_source_outer_{source}"),
        }
        e4.update({metric: str(float(seed + outer) / 10.0) for metric in E4_METRICS})
        e4["source_min_log_joint_eig"] = "0.01"
        e4["source_min_original_joint_eig"] = "0.001"
        e4["source_nonpositive_log_eig_fraction"] = "0.0"
        e4["source_outside_collocation_fraction_fd"] = "0.0"
        e4_rows.append(e4)
    write_csv(directory / "exact_map_ratios.csv", exact_rows)
    write_csv(directory / "e4_approximation_errors.csv", e4_rows)
    write_csv(directory / "exact_map_refinement.csv", exact_rows)
    write_csv(directory / "e4_approximation_refinement.csv", e4_rows)
    undefined_outers = [1] if undefined else []
    (directory / "exact_map_status.json").write_text(json.dumps({
        "status": "success", "analysis_mode": analysis_mode,
        "paper_aggregation_eligible": True,
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "n_exact_rows": 2, "n_e4_rows": 2,
        "n_refinement_rows": 2, "n_e4_refinement_rows": 2,
        "e4_status": "computed",
        "all_denominators_defined": not undefined,
        "undefined_denominator_outers": undefined_outers,
        "all_refinement_pass": True,
        "exact_refinement_failures": [], "e4_refinement_failures": [],
        "all_e4_source_policies_elliptic": True,
        "nonelliptic_e4_targets": [],
    }), encoding="utf-8")
    here = Path(__file__).resolve().parent
    (directory / "exact_map_config.json").write_text(json.dumps({
        "protocol_hash": "same-protocol",
        "analysis_mode": analysis_mode,
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "whole_space_map_claim": whole_space_map_claim,
        "implementation_hashes": {
            "driver": sha256_file(here / "liu_exact_map_fd.py"),
            "core": sha256_file(here / "liu_exact_map_core.py"),
        },
        "config_path": str(source_config), "config_sha256": sha256_file(source_config),
        "market_path": str(market_path), "market_file_sha256": sha256_file(market_path),
        "weight_dir": str(weight_dir), "checkpoint_selection": "all",
        "training_protocol_args": {"outer_iters": 2},
        "ellipticity_tolerance": 0.0,
        "checkpoint_schedule": [1, 2], "checkpoint_file_hashes": checkpoint_hashes,
    }), encoding="utf-8")
    refresh_artifact_hashes(directory)
    (directory / "_SUCCESS_EXACT_MAP").touch()
    return directory


class AggregateLiuExactMapTests(unittest.TestCase):
    @staticmethod
    def _aggregate(result_dirs, output, *, min_seeds=1):
        return aggregate(
            result_dirs, output,
            expected_seeds=sorted(int(path.parent.name.removeprefix("seed")) for path in result_dirs),
            min_seeds=min_seeds, allow_undefined_denominators=False,
            allow_partial_sensitivity=False, require_locally_unmodified=True,
            overwrite=False,
        )

    def test_common_seed_sample_is_aggregated_with_t_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dirs = [make_result(root, 1), make_result(root, 3)]
            output = root / "paper"
            status = aggregate(
                result_dirs, output, expected_seeds=[1, 3], min_seeds=2,
                allow_undefined_denominators=False, allow_partial_sensitivity=False,
                require_locally_unmodified=True, overwrite=False,
            )
            self.assertEqual(status["paper_summary_status"], "complete")
            self.assertTrue((output / "_SUCCESS_EXACT_MAP_AGG").is_file())
            with (output / "exact_map_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rho_outer1 = next(row for row in rows if row["source_outer_iter"] == "1"
                              and row["metric"] == "rho_exact")
            self.assertEqual(int(rho_outer1["n"]), 2)
            self.assertAlmostEqual(float(rho_outer1["mean"]), 3.0)
            self.assertGreater(float(rho_outer1["ci95_high"]), float(rho_outer1["mean"]))

    def test_commit_failure_restores_previous_complete_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            stage = root / "stage"
            output.mkdir()
            stage.mkdir()
            (output / "exact_map_per_seed.csv").write_text("old\n", encoding="utf-8")
            (output / "_SUCCESS_EXACT_MAP_AGG").write_text(
                "old-success\n", encoding="utf-8"
            )
            (stage / "exact_map_per_seed.csv").write_text("new\n", encoding="utf-8")
            (stage / "exact_map_summary.csv").write_text("new-summary\n", encoding="utf-8")
            (stage / "_SUCCESS_EXACT_MAP_AGG").write_text(
                "new-success\n", encoding="utf-8"
            )
            import aggregate_liu_exact_map as module
            real_replace = module.os.replace

            def fail_second_stage_move(source, destination):
                source_path = Path(source)
                if source_path.parent == stage and source_path.name == "exact_map_summary.csv":
                    raise OSError("synthetic exact commit failure")
                return real_replace(source, destination)

            with mock.patch.object(
                module.os, "replace", side_effect=fail_second_stage_move
            ):
                with self.assertRaisesRegex(OSError, "synthetic exact commit failure"):
                    _commit_staged_output(stage, output)
            self.assertEqual(
                (output / "exact_map_per_seed.csv").read_text(encoding="utf-8"),
                "old\n",
            )
            self.assertEqual(
                (output / "_SUCCESS_EXACT_MAP_AGG").read_text(encoding="utf-8"),
                "old-success\n",
            )
            self.assertFalse((output / "exact_map_summary.csv").exists())

    def test_paper_worst_summary_records_seedwise_and_global_argmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make_result(root, 1)
            second = make_result(root, 3)
            values = {
                1: {1: (0.30, 0.35), 2: (0.40, 0.45)},
                3: {1: (0.50, 0.55), 2: (0.20, 0.25)},
            }
            for seed, directory in ((1, first), (3, second)):
                update_exact_rows(directory, lambda row, seed=seed: row.update({
                    "rho_exact": str(values[seed][int(row["source_outer_iter"])][0]),
                    "rho_sensitivity_envelope": str(
                        values[seed][int(row["source_outer_iter"])][1]
                    ),
                }))
            output = root / "paper"
            status = self._aggregate([first, second], output, min_seeds=2)

            with (output / "exact_map_worst_per_seed.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                per_seed = {int(row["seed"]): row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(per_seed[1]["max_rho_exact"]), 0.40)
            self.assertEqual(int(per_seed[1]["max_rho_exact_outer"]), 2)
            self.assertAlmostEqual(
                float(per_seed[3]["max_rho_sensitivity_envelope"]), 0.55
            )
            self.assertEqual(
                int(per_seed[3]["max_rho_sensitivity_envelope_outer"]), 1
            )

            with (output / "exact_map_worst_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                summary = {row["metric"]: row for row in csv.DictReader(handle)}
            envelope = summary["max_rho_sensitivity_envelope"]
            self.assertEqual(envelope["summary_status"], "complete_common_seed_sample")
            self.assertAlmostEqual(float(envelope["mean"]), 0.50)
            self.assertAlmostEqual(float(envelope["global_max"]), 0.55)
            self.assertEqual(int(envelope["global_max_seed"]), 3)
            self.assertEqual(int(envelope["global_max_outer"]), 1)
            self.assertTrue(status["finite_domain_all_tested_ratios_below_one"])
            self.assertEqual(
                status["finite_domain_ratio_claim_status"],
                "supported_on_tested_sampled_fd_audit",
            )
            self.assertIn("finite-domain", status["finite_domain_ratio_claim_text"])
            self.assertIn("no whole-space", status["interpretation"])

    def test_below_one_claim_is_blocked_when_sampled_map_was_modified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            update_exact_rows(directory, lambda row: row.update({
                "rho_exact": "0.4", "rho_sensitivity_envelope": "0.5",
                "local_map_unmodified_on_xfd": (
                    "0" if row["source_outer_iter"] == "2" else "1"
                ),
            }))
            status = aggregate(
                [directory], root / "paper", expected_seeds=[1], min_seeds=1,
                allow_undefined_denominators=False,
                allow_partial_sensitivity=False,
                require_locally_unmodified=False,
                overwrite=False,
            )
            self.assertFalse(status["all_seed_outer_locally_unmodified"])
            self.assertFalse(status["finite_domain_all_tested_ratios_below_one"])
            self.assertIsNone(status["finite_domain_ratio_claim_text"])
            self.assertIn(
                "guard_or_clip_modified_map_on_sampled_fd_domain",
                status["finite_domain_ratio_claim_blockers"],
            )

    def test_below_one_claim_is_blocked_by_global_sensitivity_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            update_exact_rows(directory, lambda row: row.update({
                "rho_exact": "0.8",
                "rho_sensitivity_envelope": (
                    "1.01" if row["source_outer_iter"] == "2" else "0.9"
                ),
            }))
            status = self._aggregate([directory], root / "paper")
            self.assertFalse(status["finite_domain_all_tested_ratios_below_one"])
            self.assertEqual(
                status["worst_case"]["max_rho_sensitivity_envelope"]["outer"], 2
            )
            self.assertIn(
                "global_sensitivity_envelope_not_below_one",
                status["finite_domain_ratio_claim_blockers"],
            )

    def test_below_one_claim_also_requires_primary_exact_ratio_below_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            update_exact_rows(directory, lambda row: row.update({
                "rho_exact": (
                    "1.01" if row["source_outer_iter"] == "2" else "0.8"
                ),
                "rho_sensitivity_envelope": (
                    "1.05" if row["source_outer_iter"] == "2" else "0.9"
                ),
            }))
            status = self._aggregate([directory], root / "paper")
            self.assertFalse(status["finite_domain_all_tested_ratios_below_one"])
            self.assertEqual(status["worst_case"]["max_rho_exact"]["outer"], 2)
            self.assertIn(
                "global_exact_ratio_not_below_one",
                status["finite_domain_ratio_claim_blockers"],
            )

    def test_sensitivity_envelope_cannot_be_below_primary_exact_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            update_exact_rows(directory, lambda row: row.update({
                "rho_exact": "1.2", "rho_sensitivity_envelope": "0.8",
            }))
            with self.assertRaisesRegex(ValueError, "envelope is below"):
                self._aggregate([directory], root / "paper")

    def test_undefined_denominator_fails_instead_of_dropping_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dirs = [make_result(root, 1, undefined=True), make_result(root, 3)]
            output = root / "paper"
            with self.assertRaisesRegex(ValueError, "cannot be silently omitted"):
                aggregate(
                    result_dirs, output, expected_seeds=[1, 3], min_seeds=2,
                    allow_undefined_denominators=False, allow_partial_sensitivity=False,
                    require_locally_unmodified=False, overwrite=False,
                )
            self.assertTrue((output / "_FAILED_EXACT_MAP_AGG").is_file())

    def test_seed_set_mismatch_is_not_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dirs = [make_result(root, 1), make_result(root, 3)]
            with self.assertRaisesRegex(ValueError, "seed set mismatch"):
                aggregate(
                    result_dirs, root / "paper", expected_seeds=[1, 2, 3], min_seeds=2,
                    allow_undefined_denominators=False, allow_partial_sensitivity=False,
                    require_locally_unmodified=False, overwrite=False,
                )

    def test_identically_truncated_csvs_cannot_shorten_paper_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dirs = [make_result(root, 1), make_result(root, 3)]
            for directory in result_dirs:
                for name in ("exact_map_ratios.csv", "e4_approximation_errors.csv"):
                    path = directory / name
                    with path.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    write_csv(path, rows[:1])
                refresh_artifact_hashes(directory)
            with self.assertRaisesRegex(ValueError, "status n_exact_rows mismatch"):
                self._aggregate(result_dirs, root / "paper", min_seeds=2)

    def test_tampered_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            status_path = directory / "exact_map_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["undefined_denominator_outers"] = [2]
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "undefined_denominator_outers mismatch"):
                self._aggregate([directory], root / "paper")

    def test_checkpoint_hash_key_set_must_match_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            config_path = directory / "exact_map_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["checkpoint_file_hashes"]["2"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            refresh_artifact_hashes(directory)
            with self.assertRaisesRegex(ValueError, "checkpoint hash key set"):
                self._aggregate([directory], root / "paper")

    def test_schedule_must_cover_training_outer_iters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            config_path = directory / "exact_map_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["training_protocol_args"]["outer_iters"] = 3
            config_path.write_text(json.dumps(config), encoding="utf-8")
            refresh_artifact_hashes(directory)
            with self.assertRaisesRegex(ValueError, "does not cover all training outer iterations"):
                self._aggregate([directory], root / "paper")

    def test_explicit_checkpoint_subset_is_not_paper_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            config_path = directory / "exact_map_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["checkpoint_selection"] = "explicit_subset"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            refresh_artifact_hashes(directory)
            with self.assertRaisesRegex(ValueError, "explicit checkpoint subsets"):
                self._aggregate([directory], root / "paper")

    def test_tampered_hashed_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            path = directory / "e4_approximation_errors.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                self._aggregate([directory], root / "paper")

    def test_failed_overwrite_keeps_previous_completed_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = make_result(root, 1)
            output = root / "paper"
            output.mkdir()
            prior = output / "exact_map_aggregate_status.json"
            success = output / "_SUCCESS_EXACT_MAP_AGG"
            prior.write_text("previous-success", encoding="utf-8")
            success.touch()
            path = directory / "exact_map_ratios.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                aggregate(
                    [directory], output, expected_seeds=[1], min_seeds=1,
                    allow_undefined_denominators=False,
                    allow_partial_sensitivity=False,
                    require_locally_unmodified=True,
                    overwrite=True,
                )
            self.assertEqual(prior.read_text(encoding="utf-8"), "previous-success")
            self.assertTrue(success.is_file())
            self.assertFalse((output / "_FAILED_EXACT_MAP_AGG").exists())

    def test_one_seed_sample_sd_and_t_interval_are_undefined(self) -> None:
        stats = _stats([2.5])
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["mean"], 2.5)
        for field in ("std", "sem", "ci95_low", "ci95_high"):
            self.assertTrue(math.isnan(stats[field]), field)


if __name__ == "__main__":
    unittest.main()
