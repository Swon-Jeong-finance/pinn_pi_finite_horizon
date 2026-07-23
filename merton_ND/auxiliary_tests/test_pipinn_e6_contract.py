"""Torch-free structural checks for the PI-PINN E6 restore contract."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from auxiliary_tests._paths import SOURCE_ROOT

SOURCE = (SOURCE_ROOT / "merton_nd_consumption_pi_pinn.py").read_text(
    encoding="utf-8"
)


class PiPinnE6ContractTests(unittest.TestCase):
    def test_qsel_candidates_are_filtered_by_same_state_qres(self) -> None:
        start = SOURCE.index("def run_sel_check")
        end = SOURCE.index("if val_fn is not None and pres_target is not None:", start)
        body = SOURCE[start:end]
        self.assertIn("qres_value = run_val_check", body)
        self.assertIn("eligible = bool(qres_value[2] <= float(pres_target))", body)
        self.assertLess(body.index("if not eligible:"), body.index("if value[2] < best_sel_pres:"))
        self.assertIn('"optimizer": copy.deepcopy(self.optimizer.state_dict())', body)

    def test_restore_loads_model_optimizer_and_uses_nonincreasing_carry_lr(self) -> None:
        start = SOURCE.index("# Capture the true end-of-inner LR")
        end = SOURCE.index("if val_fn is not None and last_val_epoch != epochs_used:", start)
        body = SOURCE[start:end]
        self.assertIn('self.value_net.load_state_dict(best_sel_state["model"])', body)
        self.assertIn('self.optimizer.load_state_dict(best_sel_state["optimizer"])', body)
        self.assertIn("floor_lr = self._effective_min_lr()", body)
        self.assertIn('group["lr"] = max(floor_lr, min(best_lr, end_lr))', body)
        self.assertLess(
            body.index('self.optimizer.load_state_dict(best_sel_state["optimizer"])'),
            body.index('group["lr"] = max(floor_lr, min(best_lr, end_lr))'),
        )
        self.assertIn("lr_carried_next = min(", body)
        self.assertLess(body.index("end_lrs = ["), body.index("if restore_best"))
        self.assertIn(
            '"lr_carry=max(effective_floor,min(lr_best,lr_inner_end));"',
            SOURCE,
        )

    def test_lr_carried_next_is_logged_even_when_no_checkpoint_is_restored(self) -> None:
        start = SOURCE.index("# Capture the true end-of-inner LR")
        end = SOURCE.index("if val_fn is not None and last_val_epoch != epochs_used:", start)
        body = SOURCE[start:end]
        restore_end = body.index("lr_carried_next = \"\"")
        self.assertGreater(restore_end, body.index("if restore_best"))
        self.assertIn('float(self.optimizer.param_groups[0]["lr"])', body[restore_end:])

    def test_qres_is_run_fixed_and_qsel_is_outer_specific_inner_fixed(self) -> None:
        run_start = SOURCE.index("def run_policy_iteration")
        loop_start = SOURCE.index("for it in range(1, outer_iters + 1):", run_start)
        setup = SOURCE[run_start:loop_start]
        loop = SOURCE[loop_start:SOURCE.index("self.prepare_optimizer_for_outer()", loop_start)]

        self.assertIn("val_set = build_validation_set", setup)
        self.assertNotIn("q_sel_seed = qsel_seed_for_outer", setup)
        self.assertIn("q_sel_seed = qsel_seed_for_outer(val_seed, it)", loop)
        self.assertIn("sel_set = build_validation_set", loop)
        self.assertIn('"q_sel_seed": q_sel_seed', SOURCE)
        self.assertIn('"q_res_lifetime": "run-fixed"', SOURCE)
        self.assertIn('"q_sel_lifetime": "outer-specific-and-inner-fixed"', SOURCE)

    def test_qsel_seed_schedule_is_deterministic_and_outer_distinct(self) -> None:
        tree = ast.parse(SOURCE)
        wanted = {
            "PI_PINN_QSEL_MARKET_MULTIPLIER",
            "PI_PINN_QSEL_SEED_OFFSET",
            "PI_PINN_QSEL_OUTER_STRIDE",
            "PI_PINN_QSEL_SEED_MODULUS",
        }
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in node.targets
            ):
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == "qsel_seed_for_outer":
                nodes.append(node)
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<qsel>", "exec"), namespace)
        seed_fn = namespace["qsel_seed_for_outer"]
        self.assertEqual(seed_fn(12, 1), 12 * 7_919 + 101)
        self.assertEqual(seed_fn(12, 4), seed_fn(12, 4))
        self.assertNotEqual(seed_fn(12, 1), seed_fn(12, 2))
        self.assertEqual(seed_fn.__code__.co_argcount, 2)

    def test_full_adam_reset_under_carry_plateau_has_explicit_warning(self) -> None:
        self.assertIn("carry_plateau with adam_reset=full discards", SOURCE)

    def test_official_residual_is_post_restore_and_training_crossing_is_separate(self) -> None:
        required = (
            '"val_pres_post_restore": last_val[2] if last_val else ""',
            '"target_reached_post_restore": official_target_reached',
            '"training_target_crossed": bool(training_target_crossed)',
            'pres_max_semantics="max_outer_post_restore_fixed_qres"',
            'target_reached_semantics="all_outer_post_restore_fixed_qres_at_or_below_target"',
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, SOURCE)

    def test_restored_eligible_state_has_a_deterministic_invariant(self) -> None:
        self.assertIn(
            "target-eligible Q_sel checkpoint failed deterministic",
            SOURCE,
        )

    def test_degenerate_initial_portfolio_warning_and_status_are_present(self) -> None:
        self.assertIn("[warning] Degenerate initial portfolio:", SOURCE)
        self.assertIn("initial_policy_degenerate=ARGS.initial_policy_degenerate", SOURCE)
        self.assertIn("diffusion_var_min_init", SOURCE)


if __name__ == "__main__":
    unittest.main()
