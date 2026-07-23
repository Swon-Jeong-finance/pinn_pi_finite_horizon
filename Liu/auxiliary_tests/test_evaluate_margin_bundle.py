#!/usr/bin/env python3
"""Regression tests for the read-only Liu E9 margin-bundle evaluator."""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_margin_bundle as e9

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


def simple_closed_form() -> e9.ClosedFormData:
    # M=1 rows are a, b, C.  Linear interpolation gives nontrivial state
    # derivatives while retaining the exact terminal state y(0)=0.
    return e9.ClosedFormData(
        t=np.array([0.0, 1.0]),
        y=np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.05]]),
        m_states=1,
    )


def write_market(path: Path, *, seed: int = 1, break_gamma: bool = False) -> None:
    K = np.array([[0.5]])
    xbar = np.array([0.2])
    SigmaX = np.array([[0.3]])
    rho = np.array([[0.1], [0.2]])
    Lam = np.array([[0.4], [-0.2]])
    eta = np.array([0.4])
    scale = 2.0
    np.savez(
        path,
        K=K,
        xbar=xbar,
        SigmaX=SigmaX,
        rho=rho,
        Lam=Lam,
        Q=SigmaX @ SigmaX.T,
        Gamma=rho @ SigmaX.T,
        k0=K @ xbar,
        lam0=np.array([0.1, -0.05]),
        X_min=xbar - scale * eta,
        X_max=xbar + scale * eta,
        eta=eta,
        gamma=np.array([1.0 if break_gamma else 3.0]),
        r=np.array([0.02]),
        tau_max=np.array([1.0]),
        W_min=np.array([0.1]),
        W_max=np.array([2.0]),
        seed=np.array([seed]),
        market_seed=np.array([20260718]),
    )


class PureHelperTests(unittest.TestCase):
    def test_main_early_selection_failure_is_not_masked_by_stage_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SystemExit, "no Liu training attempts"):
                e9.main(["--out-root", directory])

    def test_e9_commit_failure_restores_previous_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            stage = root / "stage"
            output.mkdir()
            stage.mkdir()
            (output / e9.PER_RUN_FILE).write_text("old\n", encoding="utf-8")
            (output / e9.SUCCESS_MARKER).write_text(
                "old-success\n", encoding="utf-8"
            )
            for name in e9.OUTPUT_FILES:
                (stage / name).write_text(f"new-{name}\n", encoding="utf-8")
            real_replace = e9.os.replace

            def fail_late_stage_move(source, destination):
                source_path = Path(source)
                if source_path.parent == stage and source_path.name == e9.SUMMARY_FILE:
                    raise OSError("synthetic E9 commit failure")
                return real_replace(source, destination)

            with mock.patch.object(e9.os, "replace", side_effect=fail_late_stage_move):
                with self.assertRaisesRegex(OSError, "synthetic E9 commit failure"):
                    e9.commit_staged_output(stage, output)
            self.assertEqual(
                (output / e9.PER_RUN_FILE).read_text(encoding="utf-8"), "old\n"
            )
            self.assertEqual(
                (output / e9.SUCCESS_MARKER).read_text(encoding="utf-8"),
                "old-success\n",
            )
            self.assertFalse((output / e9.SUMMARY_FILE).exists())

    def test_paper_design_defaults_match_main_metric_evaluation(self) -> None:
        parsed = e9.build_parser().parse_args(["--out-root", "/tmp/example"])
        self.assertEqual(parsed.n_points, 100000)
        self.assertEqual(parsed.base_seed, 727)

    def test_expected_seed_option_is_distinct_from_exploratory_subset(self):
        parsed = e9.build_parser().parse_args([
            "--out-root", "/tmp/example", "--expected-seeds", "1,7,42",
        ])
        self.assertEqual(parsed.expected_seeds, "1,7,42")
        self.assertEqual(parsed.seeds, "")

    def test_seed_and_margin_parsing_are_arbitrary_but_unambiguous(self) -> None:
        self.assertEqual(e9.parse_int_list("1,3,7,11-12"), [1, 3, 7, 11, 12])
        self.assertEqual(e9.parse_int_list(""), [])
        self.assertEqual(e9.parse_margins("0.05,0.10,0.20,0.30"), [0.05, 0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            e9.parse_margins("0.1,0.10")
        with self.assertRaises(ValueError):
            e9.parse_margins("1.0")

    def test_same_base_points_are_corresponding_across_nested_windows(self) -> None:
        unit = np.array([
            [0.0, 0.0, 0.25, 0.75],
            [0.5, 0.75, 0.9, 0.1],
        ])
        kwargs = dict(
            w_min=0.0, w_max=4.0,
            x_min=np.array([-2.0, 10.0]), x_max=np.array([2.0, 14.0]),
            tau_max=1.0, tau_epsilon=0.1,
        )
        tau_a, wealth_a, state_a = e9.map_base_design(unit, margin=0.1, **kwargs)
        tau_b, wealth_b, state_b = e9.map_base_design(unit, margin=0.3, **kwargs)
        np.testing.assert_allclose(tau_a, tau_b)
        # Standardizing each mapped coordinate by its own shrunken bounds
        # recovers exactly the same unit-cube coordinates.
        for margin, wealth, state in ((0.1, wealth_a, state_a), (0.3, wealth_b, state_b)):
            w_lo, w_hi = e9.shrink_bounds(0.0, 4.0, margin)
            x_lo, x_hi = e9.shrink_bounds(kwargs["x_min"], kwargs["x_max"], margin)
            np.testing.assert_allclose((wealth - w_lo) / (w_hi - w_lo), unit[:, 1])
            np.testing.assert_allclose((state - x_lo) / (x_hi - x_lo), unit[:, 2:])

    def test_analytic_reference_derivatives_and_normalized_control(self) -> None:
        closed = simple_closed_form()
        tau = np.array([0.4, 0.7])
        wealth = np.array([0.6, 1.2])
        state = np.array([[-0.1], [0.3]])
        params = dict(
            closed_form=closed, gamma=3.0, r=0.02,
            lam0=np.array([0.1, -0.05]),
            Lam=np.array([[0.4], [-0.2]]),
            Gamma=np.array([[0.03], [0.06]]),
        )
        result = e9.analytic_reference(tau, wealth, state, **params)
        self.assertEqual(result.value_wx.shape, (2, 1))
        self.assertEqual(result.vartheta.shape, (2, 2))
        self.assertTrue(np.all(result.value_ww < 0.0))

        h = 1.0e-5
        plus_w = e9.analytic_reference(tau, wealth + h, state, **params)
        minus_w = e9.analytic_reference(tau, wealth - h, state, **params)
        numerical_ww = (plus_w.value_w - minus_w.value_w) / (2.0 * h)
        np.testing.assert_allclose(result.value_ww, numerical_ww, rtol=2.0e-9, atol=2.0e-9)
        plus_x = e9.analytic_reference(tau, wealth, state + h, **params)
        minus_x = e9.analytic_reference(tau, wealth, state - h, **params)
        numerical_wx = (plus_x.value_w - minus_x.value_w) / (2.0 * h)
        np.testing.assert_allclose(result.value_wx[:, 0], numerical_wx, rtol=2.0e-9, atol=2.0e-9)

    def test_bundle_metrics_use_flat_rel_l2_and_pointwise_vector_sup(self) -> None:
        reference = e9.ReferenceValues(
            value=np.array([1.0, 2.0]),
            value_w=np.array([2.0, 4.0]),
            value_ww=np.array([-2.0, -4.0]),
            value_wx=np.array([[1.0], [2.0]]),
            vartheta=np.array([[1.0, 0.0], [0.0, 2.0]]),
        )
        metrics = e9.compute_error_metrics(
            value=reference.value + np.array([0.0, 1.0]),
            value_w=reference.value_w + np.array([3.0, 0.0]),
            value_ww=reference.value_ww.copy(),
            value_wx=reference.value_wx + np.array([[4.0], [0.0]]),
            vartheta=reference.vartheta + np.array([[3.0, 4.0], [0.0, 0.0]]),
            reference=reference,
        )
        self.assertAlmostEqual(metrics["RelL2_V"], 1.0 / math.sqrt(5.0))
        self.assertAlmostEqual(metrics["Sup_bundle"], 5.0)
        self.assertAlmostEqual(metrics["Sup_vartheta"], 5.0)
        self.assertEqual(metrics["guard_frac"], 0.0)

    def test_sample_sd_and_student_t_interval(self) -> None:
        summary = e9.mean_std_t_ci([1.0, 2.0])
        self.assertAlmostEqual(summary["mean"], 1.5)
        self.assertAlmostEqual(summary["std"], math.sqrt(0.5))
        # df=1 gives t_0.975=12.706..., SEM=0.5.
        self.assertAlmostEqual(summary["ci95_hi"] - 1.5, 12.706204736 * 0.5, places=5)
        one = e9.mean_std_t_ci([2.5])
        self.assertTrue(math.isnan(one["std"]))

    def test_market_and_closed_form_semantic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            market_path = root / "market_params.npz"
            write_market(market_path)
            market = e9.load_market(market_path)
            self.assertEqual((market.n_assets, market.m_states), (2, 1))
            self.assertGreater(market.min_eig_Q, 0.0)
            self.assertGreater(market.min_eig_joint, 0.0)

            closed_path = root / "closed_form_ode.npz"
            closed = simple_closed_form()
            np.savez(closed_path, t=closed.t, y=closed.y, success=np.array([1]))
            loaded = e9.load_closed_form(closed_path, 1, 1.0)
            np.testing.assert_array_equal(loaded.y, closed.y)

            bad_market = root / "bad_market.npz"
            write_market(bad_market, break_gamma=True)
            with self.assertRaises(ValueError):
                e9.load_market(bad_market)
            bad_closed = root / "bad_closed.npz"
            np.savez(bad_closed, t=closed.t, y=closed.y, success=np.array([0]))
            with self.assertRaises(ValueError):
                e9.load_closed_form(bad_closed, 1, 1.0)

    def test_discovery_accepts_user_selected_seed_set_and_skips_nonaffine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 3, 7):
                run = root / f"affine_seed{seed}"
                run.mkdir()
                args = {
                    "model_type": "pipinn", "n_assets": 2, "m_states": 1,
                    "seed": seed, "market_seed": 9, "risk_premium_mode": "affine",
                    "nonaffine_eps": 0.0,
                }
                (run / "config.json").write_text(
                    json.dumps({"args": args, "weight_dir": str(run / "weights")}),
                    encoding="utf-8",
                )
                (run / "status.json").write_text(
                    json.dumps({"status": "success", "updated_at": f"2026-01-{seed:02d}"}),
                    encoding="utf-8",
                )
                (run / "_SUCCESS").touch()
            # Give the non-affine configuration the complete requested seed
            # set as well: discovery must filter it, not merely discard it for
            # a missing seed.
            for seed in (1, 7):
                nonaffine = root / f"nonaffine_seed{seed}"
                nonaffine.mkdir()
                bad_args = {
                    "model_type": "pipinn", "n_assets": 2, "m_states": 1,
                    "seed": seed, "market_seed": 9, "risk_premium_mode": "tanh",
                    "nonaffine_eps": 1.0,
                }
                (nonaffine / "config.json").write_text(
                    json.dumps({"args": bad_args}), encoding="utf-8"
                )
                (nonaffine / "status.json").write_text(
                    json.dumps({"status": "success"}), encoding="utf-8"
                )
                (nonaffine / "_SUCCESS").touch()

            selected = e9.discover_runs(
                root, models=["pipinn"], n_assets=[2], m_states=[1],
                seeds=[1, 7], min_seeds=2,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(
                [record.seed for records in selected.values() for record in records],
                [1, 7],
            )

    def test_newer_failed_attempt_invalidates_older_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_args = {
                "model_type": "pipinn", "n_assets": 30, "m_states": 3,
                "seed": 7, "market_seed": 20260718,
                "risk_premium_mode": "affine", "nonaffine_eps": 0.0,
            }
            older = root / "older_success"
            newer = root / "newer_failure"
            for run, status, updated_at in (
                (older, "success", "2026-07-22T00:00:00+00:00"),
                (newer, "failed", "2026-07-23T00:00:00+00:00"),
            ):
                run.mkdir()
                (run / "config.json").write_text(
                    json.dumps({"args": common_args}), encoding="utf-8"
                )
                (run / "status.json").write_text(
                    json.dumps({"status": status, "updated_at": updated_at}),
                    encoding="utf-8",
                )
                (run / ("_SUCCESS" if status == "success" else "_FAILED")).touch()

            with self.assertRaisesRegex(
                ValueError,
                r"newest attempt\(s\) are not successful.*seed=7 status=failed",
            ):
                e9.discover_runs(
                    root, models=["pipinn"], n_assets=[30], m_states=[3],
                    seeds=[7], min_seeds=1,
                )

    def test_strict_crosscheck_rejects_every_nonpass_status(self) -> None:
        base = {
            "model_type": "pipinn", "m_states": 3, "training_seed": 1,
            "eval_margin": 0.1, "legacy_metric": "RelL2_V",
        }
        e9.enforce_strict_crosschecks([{**base, "status": "pass"}])
        for status in ("mismatch", "not_available", "not_comparable_design"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, status):
                    e9.enforce_strict_crosschecks([{**base, "status": status}])

    def test_numpy_provenance_resolves_only_official_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "pi-pinn" / "run_seed17"
            weights = root / "weights" / "pi-pinn" / "run_seed17"
            run.mkdir(parents=True)
            weights.mkdir(parents=True)
            args = {
                "model_type": "pipinn", "n_assets": 2, "m_states": 1,
                "seed": 17, "market_seed": 20260718,
                "risk_premium_mode": "affine", "nonaffine_eps": 0.0,
                "eval_only": False, "timing_mode": False,
                "gamma": 3.0, "r": 0.02, "tau_max": 1.0,
                "w_min": 0.1, "w_max": 2.0, "x_range_scale": 2.0,
                "value_hidden": 4, "value_depth": 2, "test_points": 20000,
            }
            (run / "config.json").write_text(
                json.dumps({"args": args, "cwd": str(root), "weight_dir": str(weights)}),
                encoding="utf-8",
            )
            (run / "status.json").write_text(
                json.dumps({"status": "success", "updated_at": "2026-07-22T00:00:00Z"}),
                encoding="utf-8",
            )
            (run / "_SUCCESS").touch()
            write_market(run / "market_params.npz", seed=17)
            closed = simple_closed_form()
            np.savez(
                run / "closed_form_ode.npz", t=closed.t, y=closed.y,
                success=np.array([1]),
            )
            (weights / "value_net_final.pt").write_bytes(b"official-final")
            # A diagnostic checkpoint must never be selected ahead of final.
            (weights / "value_net_best.pt").write_bytes(b"diagnostic-best")

            discovered = e9.discover_runs(
                root, models=["pipinn"], n_assets=[2], m_states=[1],
                seeds=[17], min_seeds=1,
            )
            validated, markets, closed_forms = e9.validate_run_provenance(discovered, root)
            record = next(iter(validated.values()))[0]
            self.assertEqual(record.checkpoint, (weights / "value_net_final.pt").resolve())
            self.assertEqual(record.checkpoint_sha256, e9.sha256_file(weights / "value_net_final.pt"))
            self.assertIn(record.run_dir, markets)
            self.assertIn(record.run_dir, closed_forms)

    def test_aggregation_uses_arbitrary_seeds_and_sample_sd(self) -> None:
        rows = [
            {
                "group": "abc", "model_type": "pipinn", "n_assets": 30,
                "m_states": 3, "eval_margin": 0.1, "metric": "RelL2_V",
                "training_seed": seed, "value": value,
            }
            for seed, value in ((1, 1.0), (7, 2.0), (101, 3.0))
        ]
        summary = e9.aggregate_rows(rows)[0]
        self.assertEqual(summary["seeds"], "1,7,101")
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["std"], 1.0)


@unittest.skipUnless(torch is not None, "PyTorch is not installed; optional checkpoint tests skipped")
class TorchCheckpointTests(unittest.TestCase):
    def test_torch_design_matches_training_rng_and_final_loader_is_strict(self) -> None:
        design = e9.torch_base_design(torch, 5, 1, 727)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(727)
        expected = torch.rand(5, 3, generator=generator).numpy().astype(np.float64)
        np.testing.assert_array_equal(design, expected)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "value_net_final.pt"
            source = e9.build_value_network(torch, nn, 1, 4, 2)
            torch.save(source.state_dict(), checkpoint)
            record = e9.RunRecord(
                run_dir=root, model_type="pipinn", n_assets=2, m_states=1,
                seed=17, group="g", updated_at="", config_doc={},
                config_args={"value_hidden": 4, "value_depth": 2},
                effective_eval_args={}, checkpoint=checkpoint,
            )
            loaded = e9.load_final_model(record, torch, nn, torch.device("cpu"))
            for left, right in zip(source.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(left, right))
            with self.assertRaises(ValueError):
                e9.load_final_model(
                    e9.replace(record, checkpoint=root / "value_net_best.pt"),
                    torch, nn, torch.device("cpu"),
                )

    def test_autograd_bundle_has_expected_shapes_and_finite_values(self) -> None:
        model = e9.build_value_network(torch, nn, 1, 4, 2)
        market = e9.MarketData(
            K=np.array([[0.5]]), xbar=np.array([0.2]), SigmaX=np.array([[0.3]]),
            rho=np.array([[0.1], [0.2]]), Lam=np.array([[0.4], [-0.2]]),
            Q=np.array([[0.09]]), Gamma=np.array([[0.03], [0.06]]),
            k0=np.array([0.1]), lam0=np.array([0.1, -0.05]),
            X_min=np.array([-0.6]), X_max=np.array([1.0]), eta=np.array([0.4]),
            gamma=3.0, r=0.02, tau_max=1.0, W_min=0.1, W_max=2.0,
            training_seed=1, market_seed=9, min_eig_Q=0.09, min_eig_joint=0.7,
        )
        result = e9.evaluate_model_bundle(
            model, tau=np.array([0.2, 0.8]), wealth=np.array([0.5, 1.0]),
            state=np.array([[0.0], [0.3]]), market=market,
            torch=torch, device=torch.device("cpu"), chunk_size=1,
        )
        self.assertEqual(result["value"].shape, (2,))
        self.assertEqual(result["value_wx"].shape, (2, 1))
        self.assertEqual(result["vartheta"].shape, (2, 2))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
