#!/usr/bin/env python3
"""Regression tests for check_contraction_pilot.py (no torch required)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest

import numpy as np

import check_contraction_pilot as pilot


def decay(first: float, last: float, count: int = 20) -> np.ndarray:
    return np.geomspace(first, last, count)


def flat(value: float, count: int = 20) -> np.ndarray:
    return np.full(count, value, dtype=float)


class PilotFixture:
    def __init__(self, root: str):
        self.root = root

    def add_run(self, seed: int, scale: float, *, kind: str = "pass",
                success: bool = True, outer_iters: int = 20,
                protocol_patch=None, row_patch=None) -> str:
        run_dir = os.path.join(self.root, f"seed{seed}_scale{scale:g}")
        os.makedirs(run_dir, exist_ok=True)
        args = {
            "model_type": "pipinn",
            "run_tag": f"pilot_seed{seed}_scale{scale:g}",
            "n_assets": 30,
            "m_states": 3,
            "seed": seed,
            "market_seed": 12,
            "outer_iters": outer_iters,
            "theta_init_method": "myopic",
            "theta_init_scale": scale,
            "theta_clip_abs": None,
            "diag_every": 1,
            "diag_points": 8192,
            "val_points": 50000,
            "timing_mode": False,
            "eval_margin": "0.10,0.0",
        }
        if protocol_patch:
            args.update(protocol_patch)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"args": args}, handle)
        status = "success" if success else "running"
        with open(os.path.join(run_dir, "status.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": status}, handle)
        if success:
            open(os.path.join(run_dir, "_SUCCESS"), "a", encoding="utf-8").close()
        np.savez(
            os.path.join(run_dir, "market_params.npz"),
            K=np.eye(3),
            Q=np.eye(3),
            seed=np.asarray([seed]),
            market_seed=np.asarray([12]),
        )

        if kind == "pass":
            bundle = decay(8.0, 1.0, outer_iters)
            value = decay(12.0, 1.0, outer_iters)
            rel_v = decay(0.20, 0.01, outer_iters)
            rel_theta = decay(0.30, 0.015, outer_iters)
            val_pres = decay(1.0, 0.01, outer_iters)
        elif kind == "norm":
            bundle = flat(10.0, outer_iters)
            value = flat(40.0, outer_iters)
            rel_v = decay(0.20, 0.01, outer_iters)
            rel_theta = decay(0.30, 0.015, outer_iters)
            val_pres = decay(1.0, 0.01, outer_iters)
        elif kind == "val_only":
            bundle = flat(10.0, outer_iters)
            value = flat(40.0, outer_iters)
            rel_v = flat(0.20, outer_iters)
            rel_theta = flat(0.30, outer_iters)
            val_pres = decay(1.0, 0.01, outer_iters)
        else:
            raise ValueError(kind)

        fields = ["outer_iter", *pilot.MANDATORY_SERIES]
        rows = []
        for index in range(outer_iters):
            row = {
                "outer_iter": index + 1,
                "lam_min_sigma_frozen": 0.1,
                "lam_max_sigma_frozen": 1.0,
                "e_V_sup": value[index],
                "e_bundle_sup": bundle[index],
                "e_Xev": value[index] + bundle[index],
                "e_Vw_sup": 0.5 * bundle[index],
                "e_Vww_sup": 0.7 * bundle[index],
                "e_Vwx_sup": 0.6 * bundle[index],
                "diag_RelL2_V": rel_v[index],
                "diag_RelL2_theta": rel_theta[index],
                "val_pres": val_pres[index],
                "guard_frac_ev": 0.0,
            }
            if row_patch:
                row_patch(index, row)
            rows.append(row)
        with open(os.path.join(run_dir, "outer_history.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return run_dir


class ContractionPilotTests(unittest.TestCase):
    def run_main(self, root: str, *extra: str, seeds: str = "1,2"):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = pilot.main(["--out-root", root, "--seeds", seeds, *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def complete_grid(self, fixture: PilotFixture, kind: str = "pass", seeds=(1, 2)):
        for seed in seeds:
            for scale in (0.5, 1.5):
                fixture.add_run(seed, scale, kind=kind)

    def test_complete_grid_passes(self):
        with tempfile.TemporaryDirectory() as root:
            self.complete_grid(PilotFixture(root), "pass")
            code, output, _ = self.run_main(root)
            self.assertEqual(code, 0)
            self.assertIn("GLOBAL VERDICT: PASS", output)

    def test_arbitrary_seed_count_passes(self):
        with tempfile.TemporaryDirectory() as root:
            seeds = (1, 2, 3)
            self.complete_grid(PilotFixture(root), "pass", seeds=seeds)
            code, output, _ = self.run_main(root, seeds="1,2,3")
            self.assertEqual(code, 0)
            self.assertIn("runs=6", output)
            self.assertIn("GLOBAL VERDICT: PASS", output)

    def test_consistent_norm_proxy_is_global_only(self):
        with tempfile.TemporaryDirectory() as root:
            self.complete_grid(PilotFixture(root), "norm")
            code, output, _ = self.run_main(root)
            self.assertEqual(code, 1)
            self.assertIn("GLOBAL VERDICT: NORM-PROXY", output)

    def test_val_pres_alone_is_not_norm_proxy(self):
        with tempfile.TemporaryDirectory() as root:
            self.complete_grid(PilotFixture(root), "val_only")
            code, output, _ = self.run_main(root)
            self.assertEqual(code, 1)
            self.assertIn("GLOBAL VERDICT: FLAT", output)
            self.assertNotIn("GLOBAL VERDICT: NORM-PROXY", output)

    def test_missing_bundle_value_is_invalid_not_pass(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = PilotFixture(root)
            for seed in (1, 2):
                for scale in (0.5, 1.5):
                    patch = (lambda index, row: row.update({"e_bundle_sup": ""})) \
                        if (seed, scale) == (1, 0.5) else None
                    fixture.add_run(seed, scale, row_patch=patch)
            code, _, error = self.run_main(root)
            self.assertEqual(code, 2)
            self.assertIn("mandatory column 'e_bundle_sup' is blank", error)

    def test_missing_scale_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = PilotFixture(root)
            fixture.add_run(1, 0.5)
            fixture.add_run(2, 0.5)
            code, _, error = self.run_main(root)
            self.assertEqual(code, 2)
            self.assertIn("incomplete seed x scale pilot grid", error)

    def test_partial_run_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = PilotFixture(root)
            self.complete_grid(fixture)
            path = os.path.join(root, "seed1_scale0.5", "outer_history.csv")
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            with open(path, "w", encoding="utf-8") as handle:
                handle.writelines(lines[:-1])
            code, _, error = self.run_main(root)
            self.assertEqual(code, 2)
            self.assertIn("expected 20 outer rows, found 19", error)

    def test_protocol_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = PilotFixture(root)
            for seed in (1, 2):
                for scale in (0.5, 1.5):
                    patch = {"batch_size": 999} if (seed, scale) == (2, 1.5) else None
                    fixture.add_run(seed, scale, protocol_patch=patch)
            code, _, error = self.run_main(root)
            self.assertEqual(code, 2)
            self.assertIn("training protocol mismatch", error)

    def test_degenerate_run_blocks_global_pass(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = PilotFixture(root)
            for seed in (1, 2):
                for scale in (0.5, 1.5):
                    def patch(index, row, bad=(seed, scale) == (1, 0.5)):
                        if bad and index == 0:
                            row["lam_min_sigma_frozen"] = 0.0
                    fixture.add_run(seed, scale, row_patch=patch)
            code, output, _ = self.run_main(root)
            self.assertEqual(code, 1)
            self.assertIn("GLOBAL VERDICT: DEGENERATE", output)

    def test_tail_threshold_straddle_is_mixed(self):
        series = np.asarray([10.0] + [1.0] * 14 + [3.0, 3.0, 3.0, 1.0, 1.0])
        item = pilot.span(series, threshold=5.0, tail_fraction=0.10, min_tail_points=3)
        self.assertIsNone(item.decision)


if __name__ == "__main__":
    unittest.main()
