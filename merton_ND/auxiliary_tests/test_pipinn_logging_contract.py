"""Torch-free checks for PI-PINN outer-loop logging semantics."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from auxiliary_tests._paths import SOURCE_ROOT

SOURCE = (SOURCE_ROOT / "merton_nd_consumption_pi_pinn.py").read_text(
    encoding="utf-8"
)


def _load_should_print_outer():
    tree = ast.parse(SOURCE)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "should_print_outer"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<should_print_outer>", "exec"),
        namespace,
    )
    return namespace["should_print_outer"]


class PiPinnLoggingContractTests(unittest.TestCase):
    def test_zero_disables_periodic_logging_but_keeps_first_three(self) -> None:
        should_print_outer = _load_should_print_outer()
        self.assertEqual(
            [should_print_outer(outer, 0) for outer in range(1, 8)],
            [True, True, True, False, False, False, False],
        )

    def test_positive_interval_preserves_existing_semantics(self) -> None:
        should_print_outer = _load_should_print_outer()
        self.assertTrue(should_print_outer(1, 10))
        self.assertTrue(should_print_outer(2, 10))
        self.assertTrue(should_print_outer(3, 10))
        self.assertFalse(should_print_outer(4, 10))
        self.assertTrue(should_print_outer(10, 10))
        self.assertTrue(should_print_outer(20, 10))

    def test_negative_interval_is_rejected(self) -> None:
        should_print_outer = _load_should_print_outer()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            should_print_outer(1, -1)

    def test_cli_and_runner_validate_nonnegative_interval(self) -> None:
        self.assertIn("if ARGS.print_every_outer < 0:", SOURCE)
        run_start = SOURCE.index("def run_policy_iteration")
        loop_start = SOURCE.index("for it in range(1, outer_iters + 1):", run_start)
        setup = SOURCE[run_start:loop_start]
        self.assertIn("if print_every_outer < 0:", setup)
        self.assertIn(
            "verbose = should_print_outer(it, print_every_outer)",
            SOURCE[loop_start:],
        )


if __name__ == "__main__":
    unittest.main()
