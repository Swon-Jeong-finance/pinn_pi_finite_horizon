"""Import-safe driver tests that do not require PyTorch."""
from __future__ import annotations

import csv
import json
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from liu_exact_map_core import LiuProblem
from liu_exact_map_fd import (
    _assess,
    _e4_source_outer,
    _metric_to_row,
    _policy_extension_coordinates,
    _variant_schedule,
    _verification_set,
    _resolve_weight_dir,
    _training_protocol_args,
    _prepare_output,
    load_run,
    main,
    write_csv,
)


class LiuExactMapDriverTests(unittest.TestCase):
    @staticmethod
    def _write_preflight_run(root: Path, *, snapshot_seed: int = 1,
                             ode_success: int = 0) -> Path:
        args = {
            "model_type": "pipinn", "m_states": 1, "n_assets": 2,
            "seed": 1, "market_seed": 9, "risk_premium_mode": "affine",
            "nonaffine_eps": 0.0, "gamma": 3.0, "r": 0.02,
            "tau_max": 0.5, "w_min": 0.2, "w_max": 3.0,
        }
        (root / "config.json").write_text(json.dumps({"args": args}), encoding="utf-8")
        (root / "status.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
        (root / "_SUCCESS").touch()
        np.savez(
            root / "market_params.npz",
            K=np.asarray([[0.8]]), xbar=np.asarray([0.125]),
            SigmaX=np.asarray([[0.2]]), rho=np.asarray([[0.2], [-0.1]]),
            Lam=np.asarray([[0.05], [-0.03]]), Q=np.asarray([[0.04]]),
            Gamma=np.asarray([[0.04], [-0.02]]), k0=np.asarray([0.1]),
            lam0=np.asarray([0.08, 0.04]), X_min=np.asarray([-0.7]),
            X_max=np.asarray([0.7]), eta=np.asarray([0.2]),
            gamma=np.asarray([3.0]), r=np.asarray([0.02]),
            tau_max=np.asarray([0.5]), W_min=np.asarray([0.2]), W_max=np.asarray([3.0]),
            seed=np.asarray([snapshot_seed]), market_seed=np.asarray([9]),
        )
        np.savez(
            root / "closed_form_ode.npz",
            t=np.asarray([0.0, 0.5]), y=np.zeros((3, 2)),
            success=np.asarray([ode_success]),
        )
        return root

    def test_load_run_requires_explicit_success_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "market_params.npz").write_bytes(b"placeholder")
            (root / "_SUCCESS").touch()
            with self.assertRaisesRegex(FileNotFoundError, "status.json"):
                load_run(root)

            (root / "status.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status is not success"):
                load_run(root)

    def test_load_run_crosschecks_snapshot_seed_before_fd_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_preflight_run(Path(tmp), snapshot_seed=2)
            with self.assertRaisesRegex(ValueError, "training seeds disagree"):
                load_run(root)

    def test_load_run_rejects_unsuccessful_saved_riccati_solve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_preflight_run(Path(tmp), ode_success=0)
            with self.assertRaisesRegex(ValueError, "ODE solve was unsuccessful"):
                load_run(root)

    def test_all_verification_includes_alpha0_for_first_e4_target(self) -> None:
        checkpoints = [(1, Path("iter1.pt")), (2, Path("iter2.pt")), (3, Path("iter3.pt"))]
        verification = _verification_set(checkpoints, "all")
        self.assertEqual(verification, {0, 1, 2, 3})
        variants = _variant_schedule(
            0, verification, finest=2, largest_domain=2.0,
            primary_boundary="linearity", grid_factors=[1, 2],
            domain_factors=[1.5, 2.0], boundaries=["linearity", "exact-dirichlet"],
        )
        self.assertEqual(len(variants), 8)
        self.assertIn((2, 2.0, "linearity"), variants)
        self.assertIn((1, 1.5, "exact-dirichlet"), variants)

    def test_sparse_exact_map_verification_does_not_inject_alpha0(self) -> None:
        checkpoints = [
            (1, Path("iter1.pt")),
            (5, Path("iter5.pt")),
            (10, Path("iter10.pt")),
            (15, Path("iter15.pt")),
            (20, Path("iter20.pt")),
        ]
        self.assertEqual(
            _verification_set(checkpoints, "all", include_alpha0=False),
            {1, 5, 10, 15, 20},
        )

    def test_boundary_projection_clips_both_policy_coordinates(self) -> None:
        p = LiuProblem(
            horizon=0.5, y_min=-1.0, y_max=1.0, x_min=-0.5, x_max=0.5,
            gamma=3.0, risk_free=0.02, K=0.8, k0=0.1, Q=0.04,
            Gamma=np.asarray([0.04, -0.02]),
            lam0=np.asarray([0.08, 0.04]),
            Lam=np.asarray([0.05, -0.03]),
        )
        y = np.asarray([-2.0, 0.0, 2.0, 0.0])
        x = np.asarray([0.0, -1.0, 0.0, 1.0])
        projected_y, projected_x, diag = _policy_extension_coordinates(
            p, y, x, "boundary-projection"
        )
        np.testing.assert_array_equal(projected_y, [-1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(projected_x, [0.0, -0.5, 0.0, 0.5])
        self.assertEqual(diag["outside_collocation_count"], 4.0)
        self.assertEqual(diag["outside_collocation_y_count"], 2.0)
        self.assertEqual(diag["outside_collocation_x_count"], 2.0)

        raw_y, raw_x, raw_diag = _policy_extension_coordinates(
            p, y, x, "neural-extrapolation"
        )
        np.testing.assert_array_equal(raw_y, y)
        np.testing.assert_array_equal(raw_x, x)
        self.assertEqual(raw_diag, diag)

    def test_x_norm_components_are_mapped_to_declared_csv_fields(self) -> None:
        metric = {
            "value_sup": 1.0,
            "vw_sup": 2.0,
            "vww_sup": 3.0,
            "vwx_sup": 4.0,
            "bundle_sup": 5.0,
            "x_norm": 6.0,
        }
        mapped = _metric_to_row("e_input", metric)
        self.assertEqual(
            mapped,
            {
                "e_input_value": 1.0,
                "e_input_vw": 2.0,
                "e_input_vww": 3.0,
                "e_input_vwx": 4.0,
                "e_input_bundle": 5.0,
                "e_input_X": 6.0,
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "components.csv"
            write_csv(path, [mapped], list(mapped))
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertTrue(all(row[key] != "" for key in mapped))

    def test_csv_writer_rejects_unknown_component_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict.csv"
            with self.assertRaises(ValueError):
                write_csv(
                    path,
                    [{"e_input_value": 1.0, "e_input_value_sup": 1.0}],
                    ["e_input_value"],
                )
            self.assertFalse(path.exists())

    def test_e4_source_target_shift_is_explicit(self) -> None:
        self.assertEqual(_e4_source_outer(1), 0)
        self.assertEqual(_e4_source_outer(2), 1)
        self.assertEqual(_e4_source_outer(20), 19)
        with self.assertRaises(ValueError):
            _e4_source_outer(0)

    def test_no_verification_keeps_only_primary_alpha0_solve(self) -> None:
        variants = _variant_schedule(
            0, set(), finest=2, largest_domain=2.0, primary_boundary="linearity",
            grid_factors=[1, 2], domain_factors=[1.5, 2.0],
            boundaries=["linearity", "exact-dirichlet"],
        )
        self.assertEqual(variants, [(2, 2.0, "linearity")])

    def test_alpha0_e4_primary_gets_same_refinement_assessment(self) -> None:
        verification = {0}
        variants = _variant_schedule(
            0, verification, finest=2, largest_domain=2.0,
            primary_boundary="linearity", grid_factors=[1, 2],
            domain_factors=[1.5, 2.0], boundaries=["linearity", "exact-dirichlet"],
        )
        rows = [{
            "target_outer_iter": 1, "grid_factor": factor,
            "domain_factor": domain, "boundary": boundary,
            "e_approx_X": 0.5, "is_primary": int((factor, domain, boundary) == (2, 2.0, "linearity")),
            "is_verification": 1,
        } for factor, domain, boundary in variants]
        _assess(
            rows, key_name="target_outer_iter", value_name="e_approx_X",
            finest=2, largest_domain=2.0, primary_boundary="linearity",
            grid_factors=[1, 2], domain_factors=[1.5, 2.0],
            boundaries=["linearity", "exact-dirichlet"],
            envelope_name="approx_sensitivity_envelope",
            abs_tolerance=1e-3, rel_tolerance=1e-3,
        )
        primary = next(row for row in rows if row["is_primary"])
        self.assertEqual(primary["refinement_status"], "pass")
        self.assertEqual(primary["approx_sensitivity_envelope"], 0.0)

    def test_cartesian_interaction_cannot_hide_above_one_ratio(self) -> None:
        variants = _variant_schedule(
            1, {1}, finest=2, largest_domain=2.0,
            primary_boundary="linearity", grid_factors=[1, 2],
            domain_factors=[1.5, 2.0], boundaries=["linearity", "exact-dirichlet"],
        )
        rows = []
        for factor, domain, boundary in variants:
            one_axis = sum((factor != 2, domain != 2.0, boundary != "linearity")) <= 1
            rho = 0.5 if one_axis else 1.5
            rows.append({
                "source_outer_iter": 1, "grid_factor": factor,
                "domain_factor": domain, "boundary": boundary,
                "rho_exact": rho, "is_primary": int((factor, domain, boundary) == (2, 2.0, "linearity")),
                "is_verification": 1, "denominator_defined": 1,
                "local_map_unmodified_on_xfd": 1,
            })
        _assess(
            rows, key_name="source_outer_iter", value_name="rho_exact",
            finest=2, largest_domain=2.0, primary_boundary="linearity",
            grid_factors=[1, 2], domain_factors=[1.5, 2.0],
            boundaries=["linearity", "exact-dirichlet"],
            envelope_name="rho_sensitivity_envelope",
            abs_tolerance=1e-2, rel_tolerance=2e-2,
        )
        primary = next(row for row in rows if row["is_primary"])
        self.assertEqual(primary["refinement_status"], "fail")
        self.assertGreaterEqual(primary["rho_sensitivity_envelope"], 1.5)
        self.assertNotEqual(primary["contraction_status"], "sensitivity_stable_below_one")

    def test_relative_weight_path_is_remapped_from_launcher_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "moved" / "pi-pinn" / "tag"
            weight_dir = root / "moved" / "weights" / "pi-pinn" / "tag"
            run_dir.mkdir(parents=True)
            (weight_dir / "iterates").mkdir(parents=True)
            config = {
                "cwd": str(root / "missing_old_cwd"),
                "output_dir": "outputs/main/pi-pinn/tag",
                "weight_dir": "outputs/main/weights/pi-pinn/tag",
            }
            self.assertEqual(_resolve_weight_dir(config, run_dir, None), weight_dir.resolve())

    def test_recorded_cwd_weight_path_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "copy" / "pi-pinn" / "tag"
            cwd_weight = root / "training" / "weights" / "tag"
            run_dir.mkdir(parents=True)
            (cwd_weight / "iterates").mkdir(parents=True)
            config = {
                "cwd": str(root / "training"),
                "output_dir": "outputs/tag",
                "weight_dir": "weights/tag",
            }
            self.assertEqual(_resolve_weight_dir(config, run_dir, None), cwd_weight.resolve())

    def test_training_protocol_excludes_locations_but_keeps_theta_initialization(self) -> None:
        base = {
            "seed": 1, "device": "cuda:0", "run_tag": "seed1",
            "output_root": "outputs/a", "weight_root": "outputs/a/weights",
            "stop_flag_path": "outputs/a/stop", "theta_init_method": "myopic",
            "theta_init_scale": 1.0, "outer_iters": 20,
        }
        moved = {**base, "seed": 7, "device": "cuda:6", "run_tag": "seed7",
                 "output_root": "elsewhere", "weight_root": "elsewhere/weights"}
        self.assertEqual(_training_protocol_args(base), _training_protocol_args(moved))
        changed = {**moved, "theta_init_scale": 1.5}
        self.assertNotEqual(_training_protocol_args(base), _training_protocol_args(changed))

    def test_output_overwrite_is_explicit_and_preserves_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            status = output / "exact_map_status.json"
            notebook = output / "notes.ipynb"
            status.write_text("old", encoding="utf-8")
            notebook.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_output(output, False)
            self.assertEqual(status.read_text(encoding="utf-8"), "old")
            _prepare_output(output, True)
            self.assertFalse(status.exists())
            self.assertEqual(notebook.read_text(encoding="utf-8"), "keep")

    def test_failed_overwrite_keeps_previous_completed_driver_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "derived"
            output.mkdir()
            status = output / "exact_map_status.json"
            success = output / "_SUCCESS_EXACT_MAP"
            status.write_text("previous-success", encoding="utf-8")
            success.touch()
            with mock.patch(
                "liu_exact_map_fd.load_run", side_effect=ValueError("preflight failure")
            ):
                with self.assertRaisesRegex(ValueError, "preflight failure"):
                    main([
                        "--run-dir", str(root / "input"),
                        "--output", str(output),
                        "--overwrite",
                    ])
            self.assertEqual(status.read_text(encoding="utf-8"), "previous-success")
            self.assertTrue(success.is_file())
            self.assertFalse((output / "_FAILED_EXACT_MAP").exists())


if __name__ == "__main__":
    unittest.main()
