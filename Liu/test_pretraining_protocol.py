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
SOLVER = HERE / "Liu_nd_pi_pinn.py"
DIRECT_SOLVER = HERE / "Liu_nd_pinn.py"

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
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
