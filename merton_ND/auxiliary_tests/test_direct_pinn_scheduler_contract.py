"""Static contract tests for the Direct-PINN held-out LR scheduler.

The training module performs eager market/Torch initialization on import, so
these checks deliberately inspect its AST and remain runnable on lightweight
post-processing hosts where torch is unavailable.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from auxiliary_tests._paths import SOURCE_ROOT

ROOT = SOURCE_ROOT
SOURCE_PATH = ROOT / "merton_nd_consumption_pinn.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
UTIL_SOURCE = (ROOT / "merton_experiment_utils.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _load_pure_helper(name: str):
    """Compile one pure helper without importing eager Torch code."""
    node = next(
        item for item in TREE.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    namespace = {"math": __import__("math")}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


class DirectPinnSchedulerContractTests(unittest.TestCase):
    def test_plateau_step_uses_only_qsel_score(self) -> None:
        calls = []
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "step"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "scheduler"):
                calls.append(node)

        self.assertEqual(len(calls), 1, "Direct PINN must have one scheduler.step site")
        self.assertEqual(len(calls[0].args), 1)
        score_expr = ast.unparse(calls[0].args[0])
        self.assertEqual(score_expr, "float(scheduler_value[2])")
        self.assertNotIn("total_loss", score_expr)
        self.assertNotIn("pde_loss", score_expr)

    def test_qres_and_qsel_are_separate_deterministic_streams(self) -> None:
        self.assertIn("DIRECT_PINN_QSEL_SEED_OFFSET = 1_000_003", SOURCE)
        self.assertIn("sel_seed = int(val_seed) + DIRECT_PINN_QSEL_SEED_OFFSET", SOURCE)
        self.assertIn("value_net, sel_set", SOURCE)
        self.assertIn("value_net, val_set", SOURCE)
        self.assertIn('ARGS.q_res_role = "pres_target_and_official_residual"', SOURCE)
        self.assertIn('"lr_scheduler_and_qsel_rollback"', SOURCE)

    def test_machine_readable_score_and_patience_semantics_are_recorded(self) -> None:
        required = (
            '"sel_pde_rms"',
            '"sel_terminal_rms"',
            '"sel_pres"',
            '"scheduler_score"',
            '"scheduler_patience_unit": "q_sel_checks"',
            '"scheduler_metric": (',
            "ARGS.scheduler_check_every_optimizer_steps",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, SOURCE)

    def test_qsel_check_cadence_is_global_optimizer_steps(self) -> None:
        self.assertIn(
            "total_optimizer_steps % max(1, int(val_every)) == 0", SOURCE)
        self.assertNotIn("scheduler.step(float(total_loss", SOURCE)


class DirectPinnResamplingContractTests(unittest.TestCase):
    def test_zero_keeps_outer_start_batch_for_entire_inner_loop(self) -> None:
        should_refresh = _load_pure_helper("_resample_before_inner_step")
        self.assertFalse(any(should_refresh(step, 0) for step in range(1, 2001)))

    def test_positive_interval_refreshes_before_k_plus_one(self) -> None:
        should_refresh = _load_pure_helper("_resample_before_inner_step")
        observed = [
            step for step in range(1, 2001) if should_refresh(step, 200)
        ]
        self.assertEqual(observed, list(range(201, 2000, 200)))
        self.assertEqual(len(observed), 9)
        self.assertFalse(should_refresh(200, 200))
        self.assertTrue(should_refresh(201, 200))

    def test_negative_interval_is_rejected(self) -> None:
        should_refresh = _load_pure_helper("_resample_before_inner_step")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            should_refresh(1, -1)
        self.assertIn("if ARGS.resample_every < 0:", SOURCE)
        self.assertIn("if int(resample_every) < 0:", SOURCE)

    def test_batch_and_terminal_target_are_refreshed_together(self) -> None:
        required = (
            "t_int, y_int, t_term, y_term, V_T_target = _sample_training_batch()",
            "training_batch_resamples_outer += 1",
            '"training_batch_resamples_total"',
            '"training_batches_sampled_total"',
            'ARGS.training_batch_refresh_at_outer_start = True',
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, SOURCE)


class DirectPinnQselRollbackContractTests(unittest.TestCase):
    def test_only_qsel_ratio_drives_emergency_trigger(self) -> None:
        triggered = _load_pure_helper("_qsel_rollback_triggered")
        self.assertFalse(triggered(9.99, 1.0, 10.0))
        self.assertFalse(triggered(10.0, 1.0, 10.0))
        self.assertTrue(triggered(10.01, 1.0, 10.0))
        self.assertTrue(triggered(float("inf"), 1.0, 10.0))
        self.assertNotIn("guard_frac_denom", ast.unparse(next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_qsel_rollback_triggered")))

    def test_emergency_lr_never_increases_and_respects_floor(self) -> None:
        rollback_lr = _load_pure_helper("_qsel_rollback_lr")
        self.assertEqual(rollback_lr(1e-4, 5e-4, 1e-8, 0.5), 5e-5)
        self.assertEqual(rollback_lr(1e-8, 5e-4, 1e-8, 0.5), 1e-8)
        self.assertEqual(rollback_lr(1e-4, 5e-4, 1e-8, 1.0), 1e-4)

    def test_model_and_adam_restore_scheduler_reset_and_batch_refresh(self) -> None:
        required = (
            "value_net.load_state_dict(last_admissible_model_state)",
            "optimizer.load_state_dict(",
            "last_admissible_optimizer_state = _cpu_clone_tree(",
            "scheduler = _make_scheduler()",
            "qsel_rollback_scheduler_resets += 1",
            "steps_since_batch_refresh = 0",
            'best_loss = float("inf")',
            "best_state = None",
            '"qsel_rollback_rescue_limit_exceeded"',
            "recorder.rotate_training_checkpoints()",
            '"qsel_rollback_last_admissible_epoch"',
            '"qsel_rollback_last_admissible_pres"',
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, SOURCE)
        self.assertIn('"value_net_failed_last_admissible.pt"', UTIL_SOURCE)

    def test_every_official_outer_iterate_must_be_qsel_checked(self) -> None:
        self.assertIn(
            "ARGS.eval_epochs % ARGS.val_every != 0", SOURCE)
        self.assertIn(
            "int(eval_epochs) % int(val_every) != 0", SOURCE)

    def test_epoch_zero_baseline_does_not_advance_plateau_patience(self) -> None:
        baseline = next(
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "train_pinn_hybrid_reduced_logw_multi"
        )
        nested = next(
            node for node in baseline.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_establish_qsel_rollback_baseline"
        )
        text = ast.unparse(nested)
        self.assertNotIn("scheduler.step", text)
        self.assertNotIn("scheduler_checks +=", text)
        self.assertIn("optimizer.state_dict()", text)

    def test_discarded_qres_success_is_rechecked_after_restore(self) -> None:
        self.assertIn(
            "A Q_res success observed on the discarded state is", SOURCE)
        self.assertIn("row_val = _heldout(tau_vy_now, tau_denom_now)", SOURCE)
        self.assertIn("target_reached = bool(", SOURCE)

    def test_failure_is_not_promoted_to_official_final(self) -> None:
        self.assertIn('"run_failed": True', SOURCE)
        self.assertIn("if bool(stop_info.get(\"run_failed\", False)):", SOURCE)
        self.assertIn("raise RuntimeError(", SOURCE)
        self.assertIn("value_net_failed_last_admissible.pt", SOURCE)


if __name__ == "__main__":
    unittest.main()
