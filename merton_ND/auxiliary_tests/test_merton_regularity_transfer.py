"""Auxiliary regression tests for regularity-transfer post-processing."""
from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from merton_exact_map_fd import load_outer_residuals
from postprocess_regularity_transfer import (
    build_summaries,
    exact_protocol_group,
    official_residual,
    required_refinement_iterations,
    validate_defect_refinement_evidence,
    validate_panel,
)


class RegularityTransferTests(unittest.TestCase):
    def test_fd_protocol_changes_create_distinct_e4_groups(self) -> None:
        first, first_hash = exact_protocol_group(
            "training", {"grid": {"base_ny": 101}, "norm": "X"}
        )
        second, second_hash = exact_protocol_group(
            "training", {"grid": {"base_ny": 201}, "norm": "X"}
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first_hash, second_hash)

    def test_primary_eval_window_and_producer_hash_split_e4_groups(self) -> None:
        base = {
            "protocol_hash": "producer-a",
            "eval_margin": 0.10,
            "problem": {"w_min": 0.1, "w_max": 2.0},
            "grid": {
                "base_ny": 101,
                "evaluation_tau_min": 0.01,
                "evaluation_tau_max": 1.0,
                "evaluation_y_min": -2.0,
                "evaluation_y_max": 0.4,
            },
            "norm": "X",
        }
        first, first_hash = exact_protocol_group("training", base)
        changed_window = {
            **base,
            "eval_margin": 0.20,
            "grid": {
                **base["grid"],
                "evaluation_y_min": -1.8,
                "evaluation_y_max": 0.2,
            },
        }
        second, second_hash = exact_protocol_group(
            "training", changed_window
        )
        changed_producer = {**base, "protocol_hash": "producer-b"}
        third, third_hash = exact_protocol_group(
            "training", changed_producer
        )
        self.assertNotEqual((first, first_hash), (second, second_hash))
        self.assertNotEqual((first, first_hash), (third, third_hash))

    def test_e4_requires_complete_defect_level_fd_evidence(self) -> None:
        exact_cfg = {
            "protocol_hash": "protocol",
            "grid": {
                "grid_factors": [1, 2],
                "fd_margins": [-1.0, -0.5],
                "boundaries": ["robin", "exact-dirichlet"],
            },
        }
        primary_deltas = {0: 1.0, 1: 2.0, 2: 5.0, 3: 3.0, 4: 2.5}
        required = [0, 1, 2, 4]
        defects = [
            {
                "defect_iter": defect_iter,
                "delta_X": delta,
                "refinement_status": (
                    "pass" if defect_iter in required else "not_checked"
                ),
            }
            for defect_iter, delta in primary_deltas.items()
        ]
        self.assertEqual(required_refinement_iterations(defects), required)

        fields = [
            "protocol_hash", "defect_iter", "grid_factor", "fd_margin",
            "boundary", "is_primary", "is_verification", "delta_X",
            "defect_grid_abs_change", "defect_domain_abs_change",
            "defect_boundary_abs_change", "defect_sensitivity_envelope",
            "refinement_status",
        ]
        refinement_rows = []
        for defect_iter in required:
            primary_delta = primary_deltas[defect_iter]
            for factor in (1, 2):
                for margin in (-1.0, -0.5):
                    for boundary in ("robin", "exact-dirichlet"):
                        is_primary = (
                            factor, margin, boundary
                        ) == (2, -1.0, "robin")
                        refinement_rows.append({
                            "protocol_hash": "protocol",
                            "defect_iter": defect_iter,
                            "grid_factor": factor,
                            "fd_margin": margin,
                            "boundary": boundary,
                            "is_primary": int(is_primary),
                            "is_verification": 1,
                            "delta_X": primary_delta if is_primary else (
                                primary_delta + 0.01
                            ),
                            "defect_grid_abs_change": (
                                0.01 if is_primary else ""
                            ),
                            "defect_domain_abs_change": (
                                0.01 if is_primary else ""
                            ),
                            "defect_boundary_abs_change": (
                                0.01 if is_primary else ""
                            ),
                            "defect_sensitivity_envelope": (
                                primary_delta + 0.03 if is_primary else ""
                            ),
                            "refinement_status": (
                                "pass" if is_primary else "variant"
                            ),
                        })

        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            path = result / "exact_map_defect_refinement.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(refinement_rows)
            self.assertEqual(
                validate_defect_refinement_evidence(
                    result, exact_cfg, defects
                ),
                required,
            )

            # A set comparison alone would miss duplicate rows.  E4 must
            # reject both duplicates and missing Cartesian variants.
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(refinement_rows[0])
            with self.assertRaisesRegex(ValueError, "refinement rows"):
                validate_defect_refinement_evidence(
                    result, exact_cfg, defects
                )

    def test_post_restore_residual_is_required_and_maximized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            path = run / "outer_history.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "outer_iter", "val_pres", "val_pres_at_stop",
                        "val_pres_post_restore",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "outer_iter": 1, "val_pres": 0.03,
                        "val_pres_at_stop": 0.01,
                        "val_pres_post_restore": 0.025,
                    },
                    {
                        "outer_iter": 2, "val_pres": 0.02,
                        "val_pres_at_stop": 0.009,
                        "val_pres_post_restore": 0.015,
                    },
                ])
            values, semantics = load_outer_residuals(run)
            self.assertEqual(semantics, "official_post_restore")
            self.assertEqual(values, {1: 0.025, 2: 0.015})
            achieved, n = official_residual(run)
            self.assertAlmostEqual(achieved, 0.025)
            self.assertEqual(n, 2)

    def test_summary_uses_seedwise_points_and_upper_envelope(self) -> None:
        rows = []
        for target, scale in ((1e-2, 1.0), (1e-3, 0.1), (1e-4, 0.01)):
            for seed, multiplier in ((1, 2.0), (2, 3.0)):
                p_res = target * (0.8 + 0.05 * seed)
                p_hat = multiplier * p_res
                rows.append({
                    "group": "g", "pres_target": target, "seed": seed,
                    "market_hash": "same", "achieved_pres_post_restore": p_res,
                    "p_hat_X": p_hat, "C_num_run": p_hat / p_res,
                })
        validate_panel(rows, {1, 2}, 2)
        per_target, fits = build_summaries(rows)
        self.assertEqual(len(per_target), 3)
        self.assertEqual(len(fits), 1)
        self.assertAlmostEqual(float(fits[0]["slope"]), 1.0, delta=0.01)
        self.assertAlmostEqual(float(fits[0]["C_num_empirical_upper"]), 3.0)
        self.assertTrue(math.isfinite(float(fits[0]["r2"])))


if __name__ == "__main__":
    unittest.main()
