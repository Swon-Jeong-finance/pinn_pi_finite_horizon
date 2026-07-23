#!/usr/bin/env python3
"""Regression tests for the Liu Figure-2 convergence post-processor."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "liu_mplconfig"))

import postprocess_contraction as pc  # noqa: E402
from aggregate_seeds import MARKET_HASH_KEYS  # noqa: E402


def synthetic_histories(include_val: bool = False):
    seeds = [1, 2, 3]
    value_scales = [100.0, 1.0, 5.0]
    value_rhos = [0.4, 0.8, 0.9]
    theta_rhos = [0.5, 0.6, 0.7]
    histories = {}
    for seed, scale, rho_v, rho_theta in zip(
        seeds, value_scales, value_rhos, theta_rhos
    ):
        value = {outer: scale * rho_v ** (outer - 1) for outer in range(1, 5)}
        theta = {outer: rho_theta ** (outer - 1) for outer in range(1, 5)}
        # A plateau after the primary window makes the 1-5 sensitivity fit
        # genuinely different from the primary 1-4 result.
        value[5] = value[4]
        value[6] = value[4]
        theta[5] = theta[4]
        theta[6] = theta[4]
        history = {
            "diag_RelL2_V": value,
            "diag_RelL2_vartheta": theta,
            # Legacy raw-theta diagnostic must not feed Figure 2.
            "diag_RelL2_theta": {
                outer: 1.0e4 * seed * outer for outer in range(1, 7)
            },
            # Must be ignored even when present and wildly scaled.
            "e_Xev": {outer: 1.0e6 * seed * (7 - outer) for outer in range(1, 7)},
        }
        if include_val:
            history["val_pres"] = {outer: 0.75 ** (outer - 1) for outer in range(1, 7)}
        histories[seed] = history
    return seeds, histories


def meta():
    return {
        "group": "synthetic",
        "model_type": "pipinn",
        "n_assets": 10,
        "m_states": 3,
    }


def write_market(path: Path) -> None:
    payload = {}
    for index, key in enumerate(MARKET_HASH_KEYS):
        if key == "market_seed":
            payload[key] = np.asarray(2718, dtype=np.int64)
        else:
            payload[key] = np.asarray([index + 0.25], dtype=np.float64)
    np.savez(path, **payload)


def write_run(root: Path, seed: int, e_scale: float, include_val: bool = False) -> None:
    run_dir = root / f"run_seed{seed}"
    run_dir.mkdir(parents=True)
    config = {
        "args": {
            "model_type": "pipinn",
            "m_states": 3,
            "n_assets": 10,
            "seed": seed,
            "eval_margin": "0.10,0.30",
            "risk_premium_mode": "affine",
            "theta_init_method": "myopic",
            "theta_init_scale": 1.0,
            "diag_every": 1,
            "outer_iters": 6,
            "run_tag": f"seed{seed}",
            "output_root": str(root),
        }
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "_SUCCESS").write_text("", encoding="utf-8")
    write_market(run_dir / "market_params.npz")

    fields = [
        "outer_iter", "diag_RelL2_V", "diag_RelL2_theta",
        "diag_RelL2_vartheta", "e_Xev",
    ]
    if include_val:
        fields.append("val_pres")
    value_rho = {1: 0.4, 2: 0.8, 3: 0.9}[seed]
    theta_rho = {1: 0.5, 2: 0.6, 3: 0.7}[seed]
    value_scale = {1: 100.0, 2: 1.0, 3: 5.0}[seed]
    with (run_dir / "outer_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for outer in range(1, 7):
            fitted_outer = min(outer, 4)
            row = {
                "outer_iter": outer,
                "diag_RelL2_V": value_scale * value_rho ** (fitted_outer - 1),
                "diag_RelL2_theta": 1.0e4 * seed * outer,
                "diag_RelL2_vartheta": theta_rho ** (fitted_outer - 1),
                "e_Xev": e_scale * seed * (outer**2 + 1),
            }
            if include_val:
                row["val_pres"] = 0.75 ** (outer - 1)
            writer.writerow(row)


class Figure2CoreTests(unittest.TestCase):
    def test_seedwise_fit_precedes_aggregation(self):
        seeds, histories = synthetic_histories()
        windows = [(1, 3), (1, 4), (1, 5)]
        trajectories, pointwise, fits, decay = pc.build_tables(
            meta(), histories, seeds, list(pc.MAIN_METRICS), windows, (1, 4)
        )
        primary_v = next(
            row
            for row in decay
            if row["metric"] == "diag_RelL2_V" and row["is_primary"] == 1
        )
        self.assertAlmostEqual(primary_v["rho_mean"], 0.7, places=12)
        self.assertAlmostEqual(primary_v["rho_std"], np.std([0.4, 0.8, 0.9], ddof=1), places=12)

        mean_v = {
            int(row["outer_iter"]): float(row["mean"])
            for row in pointwise
            if row["metric"] == "diag_RelL2_V"
        }
        fitted_mean_rho = float(pc.fit_log_linear(mean_v, (1, 4))["rho"])
        self.assertGreater(abs(fitted_mean_rho - primary_v["rho_mean"]), 0.1)

        first = next(
            row
            for row in pointwise
            if row["metric"] == "diag_RelL2_V" and row["outer_iter"] == 1
        )
        self.assertAlmostEqual(first["mean"], np.mean([100.0, 1.0, 5.0]), places=12)
        self.assertAlmostEqual(first["std"], np.std([100.0, 1.0, 5.0], ddof=1), places=12)
        self.assertEqual(len(trajectories), 3 * 6 * 2)
        self.assertTrue(all(row["metric"] != "e_Xev" for row in fits))

    def test_single_panel_value_policy_and_optional_third_curve(self):
        seeds, histories = synthetic_histories(include_val=True)
        primary = (1, 4)
        windows = [primary]
        tables = pc.build_tables(
            meta(), histories, seeds, list(pc.MAIN_METRICS), windows, primary
        )
        fig = pc.create_figure(
            *tables[:2], tables[3], list(pc.MAIN_METRICS), primary, outer_end=6
        )
        self.assertEqual(len(fig.axes), 1)
        self.assertTrue(all(ax.get_yscale() == "log" for ax in fig.axes))
        self.assertIsNone(getattr(fig, "_suptitle", None))
        self.assertEqual(len(fig.axes[0].lines), 2)
        self.assertEqual(len(fig.axes[0].collections), 2)
        self.assertTrue(np.allclose(fig.axes[0].get_xlim(), [1.0, 6.0]))
        self.assertEqual(
            [text.get_text() for text in fig.axes[0].get_legend().texts],
            ["Value", "Policy"],
        )
        self.assertEqual(len(fig.axes[0].texts), 0)
        self.assertEqual(len(fig.axes[0].patches), 0)

        fig_hidden = pc.create_figure(
            *tables[:2],
            tables[3],
            list(pc.MAIN_METRICS),
            primary,
            show_seed_trajectories=False,
            figure_size=(8.0, 4.0),
            font_size=13.0,
            outer_end=6,
        )
        self.assertTrue(np.allclose(fig_hidden.get_size_inches(), [8.0, 4.0]))
        self.assertEqual(len(fig_hidden.axes[0].lines), 2)
        self.assertEqual(
            [text.get_text() for text in fig_hidden.axes[0].get_legend().texts],
            ["Value", "Policy"],
        )
        self.assertTrue(
            all(ax.xaxis.label.get_fontsize() == 13.0 for ax in fig_hidden.axes)
        )

        metrics = [*pc.MAIN_METRICS, pc.OPTIONAL_METRIC]
        tables = pc.build_tables(meta(), histories, seeds, metrics, windows, primary)
        fig_three = pc.create_figure(
            *tables[:2], tables[3], metrics, primary, outer_end=6
        )
        self.assertEqual(len(fig_three.axes), 1)
        self.assertEqual(len(fig_three.axes[0].lines), 3)
        self.assertTrue(all(ax.get_yscale() == "log" for ax in fig_three.axes))
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(fig_hidden)
        plt.close(fig_three)

    def test_log_sd_band_masks_nonpositive_lower_endpoint_without_clipping(self):
        lower, upper = pc.log_sd_band(
            np.asarray([2.0, 1.0, 4.0]),
            np.asarray([0.5, 1.0, 5.0]),
        )
        self.assertAlmostEqual(lower[0], 1.5)
        self.assertTrue(np.isnan(lower[1]))
        self.assertTrue(np.isnan(lower[2]))
        np.testing.assert_allclose(upper, [2.5, 2.0, 9.0])
        with self.assertRaises(ValueError):
            pc.log_sd_band(np.asarray([1.0]), np.asarray([-0.1]))

    def test_window_validation_and_invalid_fit_values(self):
        self.assertEqual(
            pc.parse_windows("1-4", "1-3,1-5"),
            ((1, 4), [(1, 3), (1, 4), (1, 5)]),
        )
        for invalid in ("4-1", "1-2", "bad"):
            with self.assertRaises(ValueError):
                pc.parse_window(invalid)
        with self.assertRaises(ValueError):
            pc.parse_windows("1-4", "1-3,1-3")
        with self.assertRaises(ValueError):
            pc.fit_log_linear({1: 1.0, 2: 0.0, 3: 0.5}, (1, 3))
        constant = pc.fit_log_linear({1: 2.0, 2: 2.0, 3: 2.0}, (1, 3))
        self.assertAlmostEqual(constant["rho"], 1.0, places=12)
        self.assertTrue(np.isnan(constant["r_squared"]))

    def test_endpoint_summary_uses_ratio_of_seed_means(self):
        seeds, histories = synthetic_histories()
        primary = (1, 4)
        tables = pc.build_tables(
            meta(), histories, seeds, list(pc.MAIN_METRICS), [primary], primary
        )
        endpoints = pc.build_endpoint_summary(
            meta(), tables[1], list(pc.MAIN_METRICS), 1, 6
        )
        value = next(row for row in endpoints if row["metric"] == "diag_RelL2_V")
        start = np.mean([100.0, 1.0, 5.0])
        end = np.mean([100.0 * 0.4**3, 1.0 * 0.8**3, 5.0 * 0.9**3])
        self.assertAlmostEqual(value["mean_start"], start, places=12)
        self.assertAlmostEqual(value["mean_end"], end, places=12)
        self.assertAlmostEqual(value["seed_mean_reduction_factor"], start / end, places=12)


class Figure2EndToEndTests(unittest.TestCase):
    def test_e_xev_changes_do_not_change_outputs_and_val_is_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = [base / "a", base / "b"]
            outputs = [base / "out_a", base / "out_b"]
            for root, e_scale in zip(roots, (1.0, 1.0e12)):
                for seed in (1, 2, 3):
                    write_run(root, seed, e_scale=e_scale, include_val=False)
            common = [
                "--expected-seeds", "1,2,3", "--min-seeds", "3",
                "--endpoint-outer", "6",
            ]
            pc.main(
                [
                    "--out-root", str(roots[0]), "--output", str(outputs[0]), *common,
                    "--hide-seed-trajectories", "--fig-width", "8", "--fig-height", "4",
                    "--font-size", "12", "--dpi", "150", "--bbox-inches", "standard",
                    "--formats", "png,pdf,eps",
                ]
            )
            pc.main(
                [
                    "--out-root", str(roots[1]), "--output", str(outputs[1]),
                    *common, "--no-plots",
                ]
            )
            self.assertTrue((outputs[0] / "figure2_empirical_convergence.png").is_file())
            self.assertTrue((outputs[0] / "figure2_empirical_convergence.pdf").is_file())
            self.assertTrue((outputs[0] / "figure2_empirical_convergence.eps").is_file())
            self.assertTrue(
                (outputs[0] / "figure2_empirical_convergence.eps")
                .read_bytes()
                .startswith(b"%!PS-Adobe")
            )
            from PIL import Image

            with Image.open(outputs[0] / "figure2_empirical_convergence.png") as image:
                self.assertEqual(image.size, (1200, 600))
            for filename in (
                "figure2_trajectories.csv",
                "figure2_pointwise_summary.csv",
                "figure2_endpoint_summary.csv",
                "figure2_seed_decay_fits.csv",
                "figure2_decay_summary.csv",
            ):
                self.assertEqual(
                    (outputs[0] / filename).read_bytes(),
                    (outputs[1] / filename).read_bytes(),
                )
            for legacy in (
                "figure2_ratios.csv",
                "figure2_worst_summary.csv",
                "figure2_contraction.png",
            ):
                self.assertFalse((outputs[0] / legacy).exists())

            metadata = json.loads(
                (outputs[0] / "figure2_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["metrics"], list(pc.MAIN_METRICS))
            self.assertFalse(metadata["e_Xev_used"])
            self.assertFalse(metadata["show_individual_seed_trajectories"])
            self.assertEqual(metadata["figure_layout"], "single Liu panel with Value and Policy curves")
            self.assertFalse(metadata["figure_shows_decay_fit_or_early_window"])
            self.assertEqual(metadata["endpoint_summary"]["outer_end"], 6)
            self.assertEqual(metadata["figure_style"]["width_inches"], 8.0)
            self.assertEqual(metadata["figure_style"]["height_inches"], 4.0)
            self.assertEqual(metadata["figure_style"]["dpi"], 150)
            self.assertIn("PostScript", metadata["figure_style"]["eps_rendering"])

            with self.assertRaises(FileExistsError):
                pc.main(
                    [
                        "--out-root", str(roots[0]), "--output", str(outputs[0]),
                        *common, "--no-plots",
                    ]
                )
            (outputs[0] / "figure2_ratios.csv").write_text("legacy", encoding="utf-8")
            checkpoint_dir = outputs[0] / ".ipynb_checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "notes-checkpoint.txt").write_text(
                "unrelated", encoding="utf-8"
            )
            pc.main(
                [
                    "--out-root", str(roots[0]), "--output", str(outputs[0]),
                    *common, "--no-plots", "--overwrite",
                ]
            )
            self.assertFalse((outputs[0] / "figure2_ratios.csv").exists())
            self.assertFalse((outputs[0] / "figure2_empirical_convergence.png").exists())
            self.assertFalse((outputs[0] / "figure2_empirical_convergence.pdf").exists())
            self.assertFalse((outputs[0] / "figure2_empirical_convergence.eps").exists())
            self.assertEqual(
                (checkpoint_dir / "notes-checkpoint.txt").read_text(encoding="utf-8"),
                "unrelated",
            )

            with self.assertRaisesRegex(ValueError, "missing requested columns"):
                pc.main(
                    [
                        "--out-root", str(roots[0]), "--output", str(base / "with_val"),
                        "--expected-seeds", "1,2,3", "--min-seeds", "3",
                        "--include-val-pres", "--no-plots",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
