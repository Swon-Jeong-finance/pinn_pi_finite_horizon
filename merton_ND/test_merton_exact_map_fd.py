"""Artifact/interface tests that do not require PyTorch."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merton_exact_map_fd import (
    ACTIVATION_FIELDS,
    RATIO_FIELDS,
    aggregate_exact_map,
    assess_refinement,
    load_run_spec,
)


class ExactMapInterfaceTests(unittest.TestCase):
    def test_refinement_envelope_is_labelled_as_sensitivity_not_bound(self) -> None:
        rows = []
        ratios = {
            (1, -1.0, "robin"): 0.51,
            (2, -1.0, "robin"): 0.50,
            (2, -0.5, "robin"): 0.52,
            (2, -1.0, "exact-dirichlet"): 0.49,
        }
        for (factor, margin, boundary), ratio in ratios.items():
            rows.append({
                "source_iter": 1,
                "grid_factor": factor,
                "fd_margin": margin,
                "boundary": boundary,
                "rho_exact": ratio,
                "is_verification": 1,
                "local_map_unmodified_on_xfd": 1,
            })
        assess_refinement(
            rows,
            grid_factors=[1, 2],
            fd_margins=[-1.0, -0.5],
            boundaries=["robin", "exact-dirichlet"],
            abs_tolerance=0.05,
            rel_tolerance=0.0,
        )
        primary = next(
            row
            for row in rows
            if row["grid_factor"] == 2
            and row["fd_margin"] == -1.0
            and row["boundary"] == "robin"
        )
        self.assertEqual(primary["refinement_status"], "pass")
        self.assertAlmostEqual(primary["rho_sensitivity_envelope"], 0.54)
        self.assertEqual(primary["contraction_status"], "sensitivity_stable_below_one")
        primary["local_map_unmodified_on_xfd"] = 0
        assess_refinement(
            rows,
            grid_factors=[1, 2],
            fd_margins=[-1.0, -0.5],
            boundaries=["robin", "exact-dirichlet"],
            abs_tolerance=0.05,
            rel_tolerance=0.0,
        )
        self.assertEqual(
            primary["contraction_status"],
            "sampled_modified_map_sensitivity_stable_below_one",
        )

    def test_structured_run_requires_and_loads_explicit_map_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            iterate = run / "weights" / "iterates"
            iterate.mkdir(parents=True)
            (iterate / "value_net_iter0001.pt").write_bytes(b"placeholder")
            config = {
                "args": {
                    "model_type": "pipinn",
                    "seed": 1,
                    "market_seed": 9,
                    "n_assets": 1,
                    "tau_max": 1.0,
                    "w_min": 0.1,
                    "w_max": 2.0,
                    "gamma": 2.0,
                    "rho_discount": 0.04,
                    "epsilon": 1.0,
                    "r": 0.03,
                    "eval_margin": "0.10,0.30",
                    "policy_guard_mode": "legacy-signed",
                    "network_time_coordinate": "t",
                    "network_input_order": "t,y",
                    "weight_dir": "weights",
                }
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "_SUCCESS_TRAINING").touch()
            np.savez(
                run / "market_params.npz",
                mu_excess=np.asarray([0.08]),
                Sigma_safe=np.asarray([[0.04]]),
            )
            spec = load_run_spec(run)
            self.assertEqual(spec.seed, 1)
            self.assertEqual(spec.market_seed, 9)
            self.assertEqual(spec.network.time_coordinate, "t")
            self.assertEqual(spec.policy.guard_mode, "legacy-signed")
            self.assertEqual([outer for outer, _path in spec.checkpoints], [1])
            self.assertAlmostEqual(spec.eval_margin, 0.10)

    def test_current_trainer_contract_is_derived_without_ambiguous_legacy_fields(self) -> None:
        def make_run(root: Path, *, pi_init_scale: float) -> Path:
            run = root
            iterate = run / "weights" / "iterates"
            iterate.mkdir(parents=True)
            (iterate / "value_net_iter0001.pt").write_bytes(b"placeholder")
            config = {
                "args": {
                    "model_type": "pipinn",
                    "n_assets": 1,
                    "tau_max": 1.0,
                    "w_min": 0.1,
                    "w_max": 2.0,
                    "gamma": 2.0,
                    "rho_discount": 0.04,
                    "epsilon_bequest": 1.25,
                    "r": 0.03,
                    "eval_margin": "0.10,0.0",
                    # Market conditioning cap and consumption-ratio cap are
                    # deliberately different.
                    "kappa_max": 30.0,
                    "kappa_max_bound": 3.0,
                    "utility_cap": 1e3,
                    "pi_clip_abs": 1.75,
                    "pi_init_scale": pi_init_scale,
                    "policy_guard_mode": "trainer-one-sided",
                    "policy_guard_version": "merton-logw-v1",
                    "network_time_coordinate": "t",
                    "network_input_order": "t,y",
                    "weight_dir": "weights",
                }
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "_SUCCESS").touch()
            np.savez(
                run / "market_params.npz",
                mu_excess=np.asarray([0.08]),
                Sigma_safe=np.asarray([[0.04]]),
                T=np.asarray([1.0]), w_min=np.asarray([0.1]), w_max=np.asarray([2.0]),
                gamma=np.asarray([2.0]), rho_discount=np.asarray([0.04]),
                epsilon=np.asarray([1.25]), r=np.asarray([0.03]),
                seed=np.asarray([4]), market_seed=np.asarray([9]),
            )
            return run

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = load_run_spec(make_run(root / "scale05", pi_init_scale=0.5))
            self.assertEqual(spec.policy.guard_mode, "one-sided")
            self.assertEqual(spec.network.time_coordinate, "t")
            self.assertEqual(spec.network.input_order, "t,y")
            self.assertEqual(spec.network.dtype, "float32")
            self.assertAlmostEqual(spec.problem.bequest, 1.25)
            self.assertAlmostEqual(spec.policy.kappa_min, 0.01)
            self.assertAlmostEqual(spec.policy.kappa_max, 3.0)
            self.assertAlmostEqual(spec.policy.consumption_min, 0.001)
            self.assertAlmostEqual(spec.policy.consumption_max, 2.0)
            self.assertAlmostEqual(spec.policy.portfolio_min, -1.75)
            self.assertAlmostEqual(spec.policy.portfolio_max, 1.75)
            self.assertAlmostEqual(spec.policy.numerator_guard, 1e-8)
            self.assertAlmostEqual(spec.policy.denominator_guard, 1e-8)
            self.assertEqual(spec.seed, 4)
            self.assertEqual(spec.market_seed, 9)
            self.assertEqual(
                spec.metadata_provenance["policy_guard_mode"],
                "config:policy_guard_mode",
            )
            other = load_run_spec(make_run(root / "scale15", pi_init_scale=1.5))
            self.assertNotEqual(spec.group, other.group)

    def test_policy_bounds_mode_none_disables_every_finite_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            iterate = run / "weights" / "iterates"
            iterate.mkdir(parents=True)
            (iterate / "value_net_iter0001.pt").write_bytes(b"placeholder")
            config = {
                "args": {
                    "model_type": "pipinn", "seed": 1, "market_seed": 9,
                    "n_assets": 1, "tau_max": 1.0,
                    "w_min": 0.1, "w_max": 2.0, "gamma": 2.0,
                    "rho_discount": 0.04, "epsilon_bequest": 1.0, "r": 0.03,
                    "eval_margin": "0.10", "weight_dir": "weights",
                    "utility_cap": 1e3, "kappa_max_bound": 3.0,
                    # Raw stabilized defaults may remain in args; the resolved
                    # global mode must take precedence over every finite box.
                    "pi_clip_abs": 2.0, "policy_bounds_mode": "none",
                    "policy_guard_mode": "trainer-one-sided",
                    "policy_guard_version": "merton-logw-v1",
                    "network_time_coordinate": "t",
                    "network_input_order": "t,y",
                },
                "policy_kappa_min": None,
                "policy_kappa_max": None,
                "policy_c_min": None,
                "policy_c_max": None,
                "policy_pi_min": None,
                "policy_pi_max": None,
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "_SUCCESS").touch()
            np.savez(
                run / "market_params.npz",
                mu_excess=np.asarray([0.08]), Sigma_safe=np.asarray([[0.04]]),
                T=np.asarray([1.0]), w_min=np.asarray([0.1]), w_max=np.asarray([2.0]),
                gamma=np.asarray([2.0]), rho_discount=np.asarray([0.04]),
                epsilon=np.asarray([1.0]), r=np.asarray([0.03]),
                seed=np.asarray([1]), market_seed=np.asarray([9]),
            )
            spec = load_run_spec(run)
            self.assertIsNone(spec.policy.portfolio_min)
            self.assertIsNone(spec.policy.portfolio_max)
            self.assertIsNone(spec.policy.kappa_min)
            self.assertIsNone(spec.policy.kappa_max)
            self.assertIsNone(spec.policy.consumption_min)
            self.assertIsNone(spec.policy.consumption_max)
            self.assertEqual(
                spec.metadata_provenance["policy_bounds_mode"],
                "config:policy_bounds_mode",
            )

    def test_seed_defaults_match_independent_trainer_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            iterate = run / "weights" / "iterates"
            iterate.mkdir(parents=True)
            (iterate / "value_net_iter0001.pt").write_bytes(b"placeholder")
            (run / "config.json").write_text(json.dumps({"args": {
                "model_type": "pipinn", "n_assets": 1, "tau_max": 1.0,
                "w_min": 0.1, "w_max": 2.0, "gamma": 2.0,
                "rho_discount": 0.04, "epsilon_bequest": 1.0, "r": 0.03,
                "policy_guard_mode": "trainer-one-sided",
                "weight_dir": "weights",
            }}), encoding="utf-8")
            (run / "_SUCCESS").touch()
            np.savez(
                run / "market_params.npz",
                mu_excess=np.asarray([0.08]), Sigma_safe=np.asarray([[0.04]]),
            )
            spec = load_run_spec(run)
            self.assertEqual(spec.seed, 12)
            self.assertEqual(spec.market_seed, 12)

    def test_seed_aggregation_uses_seedwise_exact_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_dirs = []
            for seed, ratios in ((1, (0.5, 0.6, 0.8)), (2, (0.7, 0.8, 0.9))):
                result = root / f"seed{seed}" / "exact_map_fd"
                result.mkdir(parents=True)
                result_dirs.append(result)
                rows = []
                for checkpoint_outer, (ratio, error) in enumerate(
                    zip(ratios, (0.5, 0.2, 0.01)), start=1
                ):
                    source_iter = checkpoint_outer - 1
                    locally_unmodified = int(not (seed == 1 and checkpoint_outer == 1))
                    row = {field: "" for field in RATIO_FIELDS}
                    row.update({
                        "problem": "merton",
                        "group": "samegroup",
                        "protocol_hash": "sameprotocol",
                        "model_type": "pipinn",
                        "n_assets": 50,
                        "seed": seed,
                        "market_seed": 11,
                        "horizon": 1.0,
                        "gamma": 2.0,
                        "discount": 0.04,
                        "bequest": 1.0,
                        "risk_free": 0.03,
                        "network_dtype": "float32",
                        "eval_margin": 0.10,
                        "fd_margin": -1.0,
                        "market_sha256": "same-market",
                        "checkpoint_outer_iter": checkpoint_outer,
                        "source_iter": source_iter,
                        "target_policy_iter": checkpoint_outer,
                        "e_input_X": error,
                        "e_map_X": error * ratio,
                        "rho_exact": ratio,
                        "rho_sensitivity_envelope": ratio + 0.02,
                        "min_diffusion_variance": 0.10 + 0.01 * seed,
                        "max_diffusion_variance": 0.30 + 0.01 * seed,
                        "min_diffusion_variance_ev": 0.12 + 0.01 * seed,
                        "max_diffusion_variance_ev": 0.28 + 0.01 * seed,
                        "local_map_unmodified_on_xfd": locally_unmodified,
                        "whole_space_map_claim": "not_verified_by_finite_domain",
                        "checkpoint_selection": "all",
                        "is_verification": 1,
                        "refinement_status": "pass",
                        "contraction_status": (
                            "sensitivity_stable_below_one"
                            if locally_unmodified
                            else "sampled_modified_map_sensitivity_stable_below_one"
                        ),
                        "map_variant": (
                            "locally_unmodified_on_sampled_xfd"
                            if locally_unmodified
                            else "sampled_guarded_clipped"
                        ),
                    })
                    row.update({field: 0.0 for field in ACTIVATION_FIELDS})
                    if not locally_unmodified:
                        row["consumption_clip_frac_fd"] = 0.25
                    rows.append(row)
                with (result / "exact_map_ratios.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RATIO_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                (result / "_SUCCESS_EXACT_MAP").touch()

            output = root / "paper"
            aggregate_exact_map(
                result_dirs,
                output,
                expected_seeds=[1, 2],
                floor_multiple=10.0,
                allow_incomplete=False,
                allow_unverified=False,
                require_locally_unmodified_map=False,
                make_plot=True,
                plot_format="png",
                dpi=100,
            )
            with (output / "figure2_exact_map_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue((output / "figure2_exact_map.png").is_file())
            self.assertEqual([int(row["source_iter"]) for row in summary], [0, 1])
            self.assertAlmostEqual(float(summary[0]["rho_exact_mean"]), 0.6)
            self.assertAlmostEqual(float(summary[1]["rho_exact_mean"]), 0.7)
            self.assertAlmostEqual(
                float(summary[0]["rho_sensitivity_envelope_mean"]), 0.62
            )
            self.assertAlmostEqual(
                float(summary[1]["rho_sensitivity_envelope_max"]), 0.82
            )
            self.assertEqual(
                int(summary[0]["all_seed_sensitivity_envelope_below_one"]), 1
            )
            self.assertEqual(int(summary[0]["all_seed_locally_unmodified_map"]), 0)
            self.assertEqual(
                summary[0]["whole_space_map_claim"],
                "not_verified_by_finite_domain",
            )
            self.assertEqual(summary[0]["map_status"], "sampled_modification_active")
            self.assertAlmostEqual(float(summary[0]["consumption_clip_frac_fd_max"]), 0.25)
            self.assertAlmostEqual(
                float(summary[0]["diffusion_variance_min_across_seeds"]), 0.11
            )
            self.assertEqual(
                int(summary[0]["all_seed_diffusion_variance_above_tolerance"]), 1
            )
            with self.assertRaisesRegex(ValueError, "locally unmodified G was required"):
                aggregate_exact_map(
                    result_dirs,
                    root / "strict-paper",
                    expected_seeds=[1, 2],
                    floor_multiple=10.0,
                    allow_incomplete=False,
                    allow_unverified=False,
                    require_locally_unmodified_map=True,
                    make_plot=False,
                    plot_format="png",
                    dpi=100,
                )

            all_output = root / "paper-all-finite"
            aggregate_exact_map(
                result_dirs,
                all_output,
                expected_seeds=[1, 2],
                floor_multiple=0.0,
                allow_incomplete=False,
                allow_unverified=False,
                require_locally_unmodified_map=False,
                make_plot=False,
                plot_format="png",
                dpi=100,
            )
            with (all_output / "figure2_exact_map_summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                all_rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["source_iter"]) for row in all_rows], [0, 1, 2])

            first_csv = result_dirs[0] / "exact_map_ratios.csv"
            with first_csv.open("r", encoding="utf-8", newline="") as handle:
                degenerate_rows = list(csv.DictReader(handle))
            degenerate_rows[0]["min_diffusion_variance"] = "0.0"
            with first_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=RATIO_FIELDS)
                writer.writeheader()
                writer.writerows(degenerate_rows)
            with self.assertRaisesRegex(ValueError, "frozen diffusion variance"):
                aggregate_exact_map(
                    result_dirs,
                    root / "degenerate-paper",
                    expected_seeds=[1, 2],
                    floor_multiple=0.0,
                    allow_incomplete=False,
                    allow_unverified=False,
                    require_locally_unmodified_map=False,
                    make_plot=False,
                    plot_format="png",
                    dpi=100,
                )

    def test_inconsistent_time_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            iterate = run / "weights" / "iterates"
            iterate.mkdir(parents=True)
            (iterate / "value_net_iter0001.pt").write_bytes(b"placeholder")
            config = {
                "args": {
                    "model_type": "pipinn", "seed": 1, "market_seed": 9,
                    "n_assets": 1, "tau_max": 1.0, "w_min": 0.1, "w_max": 2.0,
                    "gamma": 2.0, "rho_discount": 0.04, "epsilon": 1.0, "r": 0.03,
                    "eval_margin": "0.10", "policy_guard_mode": "legacy-signed",
                    "network_time_coordinate": "t", "network_input_order": "tau,y",
                    "weight_dir": "weights",
                }
            }
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "_SUCCESS_TRAINING").touch()
            np.savez(
                run / "market_params.npz",
                mu_excess=np.asarray([0.08]), Sigma_safe=np.asarray([[0.04]]),
                market_seed=np.asarray([9]),
            )
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_run_spec(run)


if __name__ == "__main__":
    unittest.main()
