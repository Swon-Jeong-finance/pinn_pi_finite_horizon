#!/usr/bin/env python3
"""CPU-only regression tests for Liu run-artifact reuse safety."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent


def _load_experiment_utils():
    """Load the recorder without requiring PyTorch on the audit machine."""

    module_name = "liu_experiment_utils_recorder_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    restore_torch = "torch" in sys.modules
    original_torch = sys.modules.get("torch")
    if not restore_torch:
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object
        sys.modules["torch"] = fake_torch
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, SOURCE_ROOT / "experiment_utils.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load experiment_utils.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if not restore_torch:
            sys.modules.pop("torch", None)
        elif original_torch is not None:
            sys.modules["torch"] = original_torch


class ExperimentRecorderReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.utils = _load_experiment_utils()

    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(run_tag="reuse_test", model_type="pipinn")

    def test_new_training_quarantines_all_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_recorder_") as tmp:
            root = Path(tmp)
            output = root / "output"
            weights = root / "weights"
            recorder = self.utils.ExperimentRecorder(str(output), str(weights), self._args())

            output_payloads = {
                "train_history.csv": "train-old",
                "outer_history.csv": "outer-old",
                "metrics.csv": "metrics-old",
                "metrics.csv.eval_tmp": "eval-tmp-old",
                "config.json": "config-old",
                "status.json": "status-old",
                "config_eval.json": "config-eval-old",
                "status_eval.json": "status-eval-old",
                "market_params.npz": "market-old",
                "closed_form_ode.npz": "closed-form-old",
                "pi_pinn_convergence.png": "figure-old",
            }
            for name, payload in output_payloads.items():
                (output / name).write_text(payload, encoding="utf-8")
            (output / "plots" / "policy.png").write_text("plot-old", encoding="utf-8")
            (output / "_SUCCESS").touch()
            (output / "_SUCCESS_EVAL").touch()

            weight_payloads = {
                "value_net_final.pt": "final-old",
                "value_net_last.pt": "last-old",
                "value_net_best.pt": "best-old",
                "value_net_best_legacy.pt": "legacy-old",
            }
            for name, payload in weight_payloads.items():
                (weights / name).write_text(payload, encoding="utf-8")
            (weights / "iterates").mkdir()
            (weights / "iterates" / "value_net_iter0011.pt").write_text(
                "iterate-old", encoding="utf-8"
            )

            recorder.rotate_training_logs()

            for name in output_payloads:
                self.assertFalse((output / name).exists(), name)
                archived = list(output.glob(f"{name}.old.*"))
                self.assertEqual(len(archived), 1, name)
                self.assertEqual(archived[0].read_text(encoding="utf-8"), output_payloads[name])
            self.assertTrue((output / "plots").is_dir())
            self.assertEqual(list((output / "plots").iterdir()), [])
            archived_plots = list(output.glob("plots.old.*"))
            self.assertEqual(len(archived_plots), 1)
            self.assertEqual(
                (archived_plots[0] / "policy.png").read_text(encoding="utf-8"), "plot-old"
            )

            for name in weight_payloads:
                self.assertFalse((weights / name).exists(), name)
                archived = list(weights.glob(f"{name}.old.*"))
                self.assertEqual(len(archived), 1, name)
                self.assertEqual(archived[0].read_text(encoding="utf-8"), weight_payloads[name])
            self.assertFalse((weights / "iterates").exists())
            archived_iterates = list(weights.glob("iterates.old.*"))
            self.assertEqual(len(archived_iterates), 1)
            self.assertEqual(
                (archived_iterates[0] / "value_net_iter0011.pt").read_text(encoding="utf-8"),
                "iterate-old",
            )
            self.assertFalse((output / "_SUCCESS").exists())
            self.assertFalse((output / "_SUCCESS_EVAL").exists())
            self.assertEqual(len(list(output.glob("rerun_archive.*.json"))), 1)

            recorder.save_config()
            self.assertTrue((output / "config.json").is_file())
            self.assertEqual(list(output.rglob("config.json")), [output / "config.json"])

    def test_eval_only_writes_separate_config_without_touching_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_recorder_eval_") as tmp:
            root = Path(tmp)
            output = root / "output"
            weights = root / "weights"
            recorder = self.utils.ExperimentRecorder(str(output), str(weights), self._args())
            (output / "config.json").write_text("training-config", encoding="utf-8")
            (weights / "value_net_final.pt").write_text("training-final", encoding="utf-8")

            recorder.save_config_eval()

            self.assertEqual((output / "config.json").read_text(encoding="utf-8"), "training-config")
            self.assertEqual(
                (weights / "value_net_final.pt").read_text(encoding="utf-8"), "training-final"
            )
            self.assertTrue((output / "config_eval.json").is_file())

    def test_eval_only_config_rejects_changed_model_arguments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_eval_config_") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "args": {
                            "model_type": "pipinn",
                            "alpha_scale": 0.25,
                            "value_hidden": 256,
                        }
                    }
                ),
                encoding="utf-8",
            )
            matching = argparse.Namespace(
                model_type="pipinn", alpha_scale=0.25, value_hidden=256
            )
            self.utils.validate_eval_only_config(
                str(path),
                matching,
                ("model_type", "alpha_scale", "value_hidden"),
            )
            changed = argparse.Namespace(
                model_type="pipinn", alpha_scale=0.5, value_hidden=256
            )
            with self.assertRaisesRegex(ValueError, "alpha_scale"):
                self.utils.validate_eval_only_config(
                    str(path),
                    changed,
                    ("model_type", "alpha_scale", "value_hidden"),
                )


if __name__ == "__main__":
    unittest.main()
