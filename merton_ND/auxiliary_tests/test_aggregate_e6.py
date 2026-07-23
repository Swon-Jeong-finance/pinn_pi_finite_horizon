"""Auxiliary regression tests for Merton E6 aggregation."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import aggregate_e6 as e6


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_e6_run(
    root: Path,
    *,
    seed: int,
    target: float,
    achieved: float,
    outer_iters: int = 20,
) -> Path:
    run = root / f"pipinn_N10_K{outer_iters}_p{target:g}_seed{seed}"
    run.mkdir(parents=True)
    error_scale = (achieved ** 0.6) * (1.0 + 0.03 * seed)
    config = {
        "args": {
            "model_type": "pipinn",
            "seed": seed,
            "market_seed": 12,
            "n_assets": 10,
            "outer_iters": outer_iters,
            "pres_target": target,
            "eval_margin": "0.10,0.0",
            "lr": 5e-4,
        }
    }
    with (run / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle)
    with (run / "status.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "success",
                "updated_at": f"2026-07-23T00:00:{seed:02d}Z",
                "final_outer_iter": 2,
                "pres_max": achieved,
                "pres_max_semantics": "max_outer_post_restore_fixed_qres",
                "target_reached": achieved <= target,
                "target_reached_semantics": (
                    "all_outer_post_restore_fixed_qres_at_or_below_target"
                ),
                "total_inner_steps": 4000,
            },
            handle,
        )
    _write_csv(
        run / "outer_history.csv",
        ["outer_iter", "val_pres_post_restore", "inner_epochs_used", "e_Xev"],
        [
            {
                "outer_iter": 1,
                "val_pres_post_restore": achieved * 0.9,
                "inner_epochs_used": 2000,
                "e_Xev": error_scale * 1.2,
            },
            {
                "outer_iter": 2,
                "val_pres_post_restore": achieved,
                "inner_epochs_used": 2000,
                "e_Xev": error_scale,
            },
        ],
    )
    _write_csv(
        run / "metrics.csv",
        ["scope", "eval_margin", "metric", "value"],
        [
            {
                "scope": "fulldim",
                "eval_margin": 0.1,
                "metric": "RelL2_pi",
                "value": error_scale * 0.7,
            },
            {
                "scope": "fulldim",
                "eval_margin": 0.1,
                "metric": "RelL2_c",
                "value": error_scale * 0.5,
            },
        ],
    )
    (run / "_SUCCESS").write_text("", encoding="utf-8")
    return run


def _add_successful_eval_overlay(
    run: Path,
    *,
    eval_margin: str,
    test_points: int = 12345,
) -> None:
    with (run / "config_eval.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": {
                    "model_type": "pipinn",
                    "eval_margin": eval_margin,
                    "test_points": test_points,
                    "n_tau": 77,
                    "n_x": 88,
                }
            },
            handle,
        )
    with (run / "status_eval.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "success"}, handle)
    (run / "_SUCCESS_EVAL").write_text("", encoding="utf-8")
    # The guarded overlay contract requires metrics.csv to be no older than
    # config_eval.json.
    config_mtime = (run / "config_eval.json").stat().st_mtime
    os.utime(run / "metrics.csv", (config_mtime + 1.0, config_mtime + 1.0))


class AggregateE6Tests(unittest.TestCase):
    def test_group_key_collapses_target_but_not_outer_budget(self):
        base = {
            "model_type": "pipinn",
            "seed": 1,
            "n_assets": 10,
            "outer_iters": 20,
            "pres_target": 0.1,
        }
        another_target = {**base, "seed": 2, "pres_target": 0.01}
        another_budget = {**base, "outer_iters": 30}
        self.assertEqual(e6.e6_group_key(base), e6.e6_group_key(another_target))
        self.assertNotEqual(e6.e6_group_key(base), e6.e6_group_key(another_budget))

    def test_single_seed_sample_sd_is_missing_not_zero(self):
        mean, sample_sd, n = e6.mean_std([3.0])
        self.assertEqual(mean, 3.0)
        self.assertTrue(math.isnan(sample_sd))
        self.assertEqual(n, 1)

    def test_post_restore_residual_is_required_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _write_csv(
                run / "outer_history.csv",
                ["outer_iter", "val_pres"],
                [{"outer_iter": 1, "val_pres": 0.2}],
            )
            value, semantics = e6.achieved_pres_from_outer_history(str(run))
            self.assertIsNone(value)
            self.assertEqual(semantics, "missing_post_restore_residual")

            value, semantics = e6.achieved_pres_from_outer_history(
                str(run), allow_legacy=True
            )
            self.assertEqual(value, 0.2)
            self.assertEqual(semantics, "legacy_val_pres_explicitly_allowed")

    def test_post_restore_max_ignores_training_crossing_column(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _write_csv(
                run / "outer_history.csv",
                ["outer_iter", "val_pres_at_stop", "val_pres_post_restore"],
                [
                    {
                        "outer_iter": 1,
                        "val_pres_at_stop": 0.01,
                        "val_pres_post_restore": 0.04,
                    },
                    {
                        "outer_iter": 2,
                        "val_pres_at_stop": 0.02,
                        "val_pres_post_restore": 0.03,
                    },
                ],
            )
            value, semantics = e6.achieved_pres_from_outer_history(str(run))
            self.assertEqual(value, 0.04)
            self.assertEqual(semantics, "max_outer_post_restore_fixed_qres")

    def test_e_xev_uses_largest_outer_not_smallest_error(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _write_csv(
                run / "outer_history.csv",
                ["outer_iter", "e_Xev"],
                [
                    {"outer_iter": 2, "e_Xev": 0.1},
                    {"outer_iter": 1, "e_Xev": 0.01},
                    {"outer_iter": 3, "e_Xev": 0.2},
                ],
            )
            with (run / "config.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "args": {
                            "model_type": "pipinn",
                            "eval_margin": "0.10,0.0",
                        }
                    },
                    handle,
                )
            value = e6.pick_outer_metric_value(str(run), "e_Xev")
            self.assertEqual(value, (0.2, 0.1))

    def test_e_xev_does_not_fall_back_when_final_outer_is_blank(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _write_csv(
                run / "outer_history.csv",
                ["outer_iter", "e_Xev"],
                [
                    {"outer_iter": 1, "e_Xev": 0.1},
                    {"outer_iter": 2, "e_Xev": ""},
                ],
            )
            with (run / "config.json").open("w", encoding="utf-8") as handle:
                json.dump({"args": {"eval_margin": "0.10"}}, handle)
            self.assertIsNone(e6.pick_outer_metric_value(str(run), "e_Xev"))

    def test_e_xev_primary_margin_ignores_successful_eval_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_e6_run(
                root,
                seed=1,
                target=0.1,
                achieved=0.08,
            )
            _add_successful_eval_overlay(run, eval_margin="0.25,0.10")
            value, margin = e6.pick_outer_metric_value(str(run), "e_Xev")
            self.assertGreater(value, 0.0)
            self.assertEqual(margin, 0.1)

    def test_final_metric_primary_margin_uses_successful_eval_overlay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_e6_run(
                root,
                seed=1,
                target=0.1,
                achieved=0.08,
            )
            _write_csv(
                run / "metrics.csv",
                ["scope", "eval_margin", "metric", "value"],
                [
                    {
                        "scope": "fulldim",
                        "eval_margin": 0.1,
                        "metric": "RelL2_pi",
                        "value": 1.0,
                    },
                    {
                        "scope": "fulldim",
                        "eval_margin": 0.25,
                        "metric": "RelL2_pi",
                        "value": 2.0,
                    },
                ],
            )
            _add_successful_eval_overlay(run, eval_margin="0.25,0.10")
            effective = e6.load_config_args(str(run))
            self.assertIsNotNone(effective)
            self.assertEqual(
                e6.pick_metric_value(
                    str(run),
                    "RelL2_pi",
                    effective_config=effective,
                ),
                (2.0, 0.25),
            )

    def test_declared_overlay_primary_margin_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = _make_e6_run(
                root,
                seed=1,
                target=0.1,
                achieved=0.08,
            )
            _add_successful_eval_overlay(run, eval_margin="0.25,0.10")
            effective = e6.load_config_args(str(run))
            self.assertIsNotNone(effective)
            # The helper fixture has final metrics only at margin 0.10. A
            # declared overlay primary of 0.25 must not relabel that row.
            self.assertIsNone(
                e6.pick_metric_value(
                    str(run),
                    "RelL2_pi",
                    effective_config=effective,
                )
            )

    def test_mixed_final_metric_provenance_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "outputs"
            root.mkdir()
            run_overlay = _make_e6_run(
                root,
                seed=1,
                target=0.1,
                achieved=0.08,
            )
            _add_successful_eval_overlay(
                run_overlay,
                eval_margin="0.10,0.0",
            )
            _make_e6_run(
                root,
                seed=2,
                target=0.1,
                achieved=0.081,
            )
            output = Path(temp) / "summary"
            with self.assertRaisesRegex(SystemExit, "validation failed"):
                e6.main([
                    "--out-root", str(root),
                    "--output", str(output),
                    "--metrics", "RelL2_pi",
                    "--no-plots",
                ])
            errors = (output / "validation_errors.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("mixed metric provenance", errors)

    def test_e_xev_group_and_margin_use_raw_training_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "outputs"
            root.mkdir()
            run = _make_e6_run(
                root,
                seed=1,
                target=0.1,
                achieved=0.08,
            )
            _add_successful_eval_overlay(
                run,
                eval_margin="0.25,0.10",
                test_points=999,
            )
            raw = e6.load_config_args_raw(str(run))
            effective = e6.load_config_args(str(run))
            self.assertIsNotNone(raw)
            self.assertIsNotNone(effective)
            self.assertNotEqual(
                e6.e6_group_key(raw),
                e6.e6_group_key(effective),
            )

            output = Path(temp) / "summary"
            self.assertEqual(
                e6.main([
                    "--out-root", str(root),
                    "--output", str(output),
                    "--metrics", "e_Xev",
                    "--no-plots",
                ]),
                0,
            )
            with (output / "points.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                points = list(csv.DictReader(handle))
            self.assertEqual(len(points), 1)
            self.assertEqual(points[0]["group"], e6.e6_group_key(raw))
            self.assertEqual(float(points[0]["eval_margin"]), 0.1)
            self.assertEqual(
                points[0]["metric_provenance_source"],
                "training_trajectory",
            )

    def test_end_to_end_writes_nonpooled_paper_figure_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "outputs"
            root.mkdir()
            for seed in (1, 2):
                _make_e6_run(
                    root, seed=seed, target=0.1, achieved=0.08 + seed * 0.001
                )
                _make_e6_run(
                    root, seed=seed, target=0.02, achieved=0.016 + seed * 0.0002
                )
            # This valid but scientifically different budget must be excluded
            # by the explicit paper-panel filter, rather than pooled.
            _make_e6_run(
                root, seed=1, target=0.1, achieved=0.07, outer_iters=30
            )

            output = Path(temp) / "summary"
            rc = e6.main([
                "--out-root", str(root),
                "--output", str(output),
                "--outer-iters", "20",
                "--formats", "png",
                "--dpi", "72",
            ])
            self.assertEqual(rc, 0)

            with (output / "settings.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                settings = list(csv.DictReader(handle))
            self.assertEqual(len(settings), 1)
            self.assertEqual(settings[0]["model_type"], "pipinn")
            self.assertEqual(settings[0]["n_assets"], "10")
            self.assertEqual(settings[0]["outer_iters"], "20")
            self.assertIn("PI-PINN, N=10, K=20", settings[0]["setting_label"])

            with (output / "points.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                points = list(csv.DictReader(handle))
            self.assertEqual(len(points), 12)  # 2 targets x 2 seeds x 3 metrics
            self.assertEqual({row["outer_iters"] for row in points}, {"20"})
            self.assertTrue(any(
                float(row["achieved_pres"]) != float(row["pres_target"])
                for row in points
            ))
            e_xev_rows = [row for row in points if row["metric"] == "e_Xev"]
            final_metric_rows = [
                row for row in points if row["metric"] == "RelL2_pi"
            ]
            self.assertEqual(
                {row["metric_provenance_source"] for row in e_xev_rows},
                {"training_trajectory"},
            )
            self.assertEqual(
                {
                    row["metric_provenance_source"]
                    for row in final_metric_rows
                },
                {"raw_training_evaluation"},
            )

            figures = list(output.glob("e6_residual_error_scaling_*.png"))
            self.assertEqual(len(figures), 2)
            self.assertEqual(
                sum(
                    path.name.startswith(
                        "e6_residual_error_scaling_mean_sd_"
                    )
                    for path in figures
                ),
                1,
            )
            for figure in figures:
                self.assertGreater(figure.stat().st_size, 1000)
            with (output / "e6_metadata.json").open(
                "r", encoding="utf-8"
            ) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["outer_iters_filter"], [20])
            self.assertEqual(
                set(metadata["figures"]),
                {figure.name for figure in figures},
            )
            self.assertEqual(len(metadata["scatter_fit_figures"]), 1)
            self.assertEqual(len(metadata["mean_sd_figures"]), 1)
            self.assertIn(
                "sample SD",
                metadata["paper_summary_figure"],
            )
            self.assertIn(
                "raw config/status",
                metadata["metric_provenance_contract"],
            )
            self.assertIn("never nominal", metadata["fit_x_axis"])

            with self.assertRaises(FileExistsError):
                e6.main([
                    "--out-root", str(root),
                    "--output", str(output),
                    "--outer-iters", "20",
                    "--formats", "png",
                ])
            self.assertEqual(
                e6.main([
                    "--out-root", str(root),
                    "--output", str(output),
                    "--outer-iters", "20",
                    "--formats", "png",
                    "--dpi", "72",
                    "--overwrite",
                ]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
