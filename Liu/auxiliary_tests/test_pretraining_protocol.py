#!/usr/bin/env python3
"""Static regression tests for Liu's pre-training paper protocol.

``Liu_nd_pi_pinn.py`` performs substantial market/model setup at import time,
so these tests inspect its syntax tree instead of importing the training
script.  That keeps the checks fast, CPU-only, and safe on machines without
PyTorch while still pinning the launch-critical contracts.
"""
from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent
SOLVER = SOURCE_ROOT / "Liu_nd_pi_pinn.py"
DIRECT_SOLVER = SOURCE_ROOT / "Liu_nd_pinn.py"
LAUNCHER = SOURCE_ROOT / "tune_pipinn.sh"

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from experiment_utils import normalized_control_stats


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _assigned_literal_list(function: ast.AST, target_name: str) -> list[str]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list):
            raise AssertionError(f"{target_name} must be a literal list")
        return value
    raise AssertionError(f"assignment to {target_name} not found")


def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(root)
        for child in ast.iter_child_nodes(parent)
    }


class LiuPretrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOLVER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(SOLVER))
        cls.direct_source = DIRECT_SOLVER.read_text(encoding="utf-8")
        cls.direct_tree = ast.parse(
            cls.direct_source, filename=str(DIRECT_SOLVER)
        )
        cls.launcher_source = LAUNCHER.read_text(encoding="utf-8")
        cls.build_parser = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_arg_parser"
        )
        cls.run_policy_iteration = next(
            node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_policy_iteration"
        )
        cls.train_direct = next(
            node
            for node in cls.direct_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_pinn_nd"
        )

    def test_both_solvers_use_canonical_market_cross_correlation(self) -> None:
        for source in (self.source, self.direct_source):
            self.assertIn("rho = params.rho_canonical", source)
            self.assertNotIn("rho = params.rho_Z ", source)
            self.assertIn("Gamma = rho @ SigmaX.T", source)
            self.assertIn("rho_snapshot_metadata(params)", source)
            self.assertIn("require_canonical_metadata=True", source)

    def test_launcher_namespaces_canonical_market_runs(self) -> None:
        self.assertIn('MARKET_SCHEMA_TAG="rho_canonical_v1"', self.launcher_source)
        self.assertIn(
            'echo "${model}_${MARKET_SCHEMA_TAG}_baseline"',
            self.launcher_source,
        )
        self.assertIn(
            'printf "%s_%s_%s" "$model" "$MARKET_SCHEMA_TAG"',
            self.launcher_source,
        )
        self.assertEqual(
            self.launcher_source.count('local variant="rho:${MARKET_SCHEMA_TAG};'),
            2,
        )

    def parser_argument(self, flag: str) -> ast.Call:
        for node in ast.walk(self.build_parser):
            if not isinstance(node, ast.Call) or _call_name(node) != "add_argument":
                continue
            if node.args and _literal_string(node.args[0]) == flag:
                return node
        self.fail(f"parser argument {flag} not found")

    def test_direct_python_defaults_match_the_paper_launcher(self) -> None:
        init_default = _keyword(self.parser_argument("--theta-init-method"), "default")
        clip_default = _keyword(self.parser_argument("--theta-clip-abs"), "default")
        self.assertIsInstance(init_default, ast.Constant)
        self.assertEqual(init_default.value, "myopic")
        self.assertIsInstance(clip_default, ast.Constant)
        self.assertIsNone(clip_default.value)

    def test_e3b_saves_every_completed_outer_iteration(self) -> None:
        """The FD audit needs iterations 11--19 as well as the old schedule."""

        matching_branches: list[ast.If] = []
        for node in ast.walk(self.run_policy_iteration):
            if not isinstance(node, ast.If):
                continue
            if not isinstance(node.test, ast.Name) or node.test.id != "e3b_checkpoints":
                continue
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "_save_this_iter"
                    for target in statement.targets
                ):
                    matching_branches.append(node)

        self.assertEqual(
            len(matching_branches), 1,
            "expected one e3b checkpoint decision branch",
        )
        assignment = next(
            statement
            for statement in matching_branches[0].body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_save_this_iter"
                for target in statement.targets
            )
        )
        self.assertIsInstance(assignment.value, ast.Constant)
        self.assertIs(
            assignment.value.value, True,
            "--e3b-checkpoints must save every completed outer iteration",
        )

    def test_outer_history_contains_normalized_policy_ranges(self) -> None:
        required = {
            "vartheta_l2_min",
            "vartheta_l2_max",
            "vartheta_component_min",
            "vartheta_component_max",
            "vartheta_abs_max",
        }
        outer_fields = set(
            _assigned_literal_list(self.run_policy_iteration, "outer_fields")
        )
        self.assertTrue(
            required <= outer_fields,
            f"outer_fields missing normalized-policy columns: {sorted(required - outer_fields)}",
        )

        outer_row = next(
            node.value
            for node in ast.walk(self.run_policy_iteration)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "outer_row"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        )
        row_values = {
            key.value: value
            for key, value in zip(outer_row.keys, outer_row.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertTrue(
            required <= row_values.keys(),
            f"outer_row missing normalized-policy values: {sorted(required - row_values.keys())}",
        )
        for field in required:
            value = row_values[field]
            self.assertFalse(
                isinstance(value, ast.Constant) and value.value in (None, ""),
                f"{field} is permanently blank instead of using the frozen-policy diagnostic",
            )

    def test_direct_pinn_outer_rows_contain_normalized_policy_ranges(self) -> None:
        """Both the ordinary block row and pres-target stop row need the audit."""

        required = {
            "vartheta_l2_min",
            "vartheta_l2_max",
            "vartheta_component_min",
            "vartheta_component_max",
            "vartheta_abs_max",
        }
        outer_fields = set(_assigned_literal_list(self.train_direct, "outer_fields"))
        self.assertTrue(
            required <= outer_fields,
            f"direct-PINN outer_fields missing columns: {sorted(required - outer_fields)}",
        )

        diagnostic_rows: list[ast.Dict] = []
        for node in ast.walk(self.train_direct):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if {"outer_iter", "target_reached", "e_V_sup"} <= keys:
                diagnostic_rows.append(node)

        self.assertGreaterEqual(
            len(diagnostic_rows), 2,
            "expected both regular and pres-target-stop direct-PINN outer rows",
        )
        for index, row in enumerate(diagnostic_rows, start=1):
            row_values = {
                key.value: value
                for key, value in zip(row.keys, row.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            self.assertTrue(
                required <= row_values.keys(),
                f"direct-PINN outer row {index} missing values: "
                f"{sorted(required - row_values.keys())}",
            )
            for field in required:
                value = row_values[field]
                self.assertFalse(
                    isinstance(value, ast.Constant) and value.value in (None, ""),
                    f"direct-PINN row {index} leaves {field} permanently blank",
                )

    def test_per_outer_policy_error_records_raw_and_normalized_controls(self) -> None:
        """Figure 2 must not reuse the wealth-weighted raw-theta diagnostic."""

        required = {"diag_RelL2_theta", "diag_RelL2_vartheta"}
        for tree, source, training_function in (
            (self.tree, self.source, self.run_policy_iteration),
            (self.direct_tree, self.direct_source, self.train_direct),
        ):
            eval_diag = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "eval_diag_metrics"
            )
            function_source = ast.get_source_segment(source, eval_diag) or ""
            self.assertIn("vartheta_hat = theta_hat / w_col", function_source)
            self.assertIn("vartheta_cf = theta_cf / w_col", function_source)
            self.assertIn('"diag_RelL2_vartheta": rel_l2_vartheta', function_source)
            self.assertIn('"diag_RelL2_theta": rel_l2_theta', function_source)

            outer_fields = set(
                _assigned_literal_list(training_function, "outer_fields")
            )
            self.assertTrue(required <= outer_fields)
            diagnostic_rows = [
                node
                for node in ast.walk(training_function)
                if isinstance(node, ast.Dict)
                and required
                <= {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            ]
            self.assertTrue(
                diagnostic_rows,
                "outer-history row does not persist both policy diagnostics",
            )

    def test_ellipticity_diagnostic_does_not_depend_on_selection_set(self) -> None:
        calls = [
            node
            for node in ast.walk(self.run_policy_iteration)
            if isinstance(node, ast.Call) and _call_name(node) == "sigma_eig_extremes_batch"
        ]
        self.assertEqual(len(calls), 1, "expected one frozen-policy ellipticity call")

        parents = _parent_map(self.run_policy_iteration)
        ancestor = parents.get(calls[0])
        while ancestor is not None and ancestor is not self.run_policy_iteration:
            if isinstance(ancestor, ast.If):
                condition = ast.unparse(ancestor.test)
                self.assertNotIn(
                    "sel_set", condition,
                    "frozen-policy ellipticity must be recorded even when inner-best selection is off",
                )
            ancestor = parents.get(ancestor)

    def test_both_trainers_publish_a_separate_core_timing_field(self) -> None:
        """E8 must not substitute checkpoint-I/O elapsed time for core time."""

        for label, tree, source in (
            ("PINN", self.direct_tree, self.direct_source),
            ("PI-PINN", self.tree, self.source),
        ):
            mark_success_calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and _call_name(node) == "mark_success"
            ]
            self.assertTrue(mark_success_calls, f"{label}: mark_success call missing")
            self.assertTrue(
                any(_keyword(call, "core_train_wall_sec") is not None
                    for call in mark_success_calls),
                f"{label}: status does not publish core_train_wall_sec",
            )
            self.assertIn("torch.cuda.synchronize", source)
            self.assertIn("time.perf_counter()", source)


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class NormalizedControlStatsTests(unittest.TestCase):
    def test_reports_global_component_and_row_l2_ranges(self) -> None:
        theta = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float64)
        wealth = torch.tensor([[2.0], [0.5]], dtype=torch.float64)
        stats = normalized_control_stats(theta, wealth)
        expected = {
            "vartheta_l2_min": math.sqrt(1.25),
            "vartheta_l2_max": math.sqrt(20.0),
            "vartheta_component_min": -2.0,
            "vartheta_component_max": 4.0,
            "vartheta_abs_max": 4.0,
        }
        self.assertEqual(stats.keys(), expected.keys())
        for key, value in expected.items():
            self.assertAlmostEqual(stats[key], value, places=12)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            normalized_control_stats(torch.ones(2, 3), torch.ones(3, 1))


if __name__ == "__main__":
    unittest.main()
