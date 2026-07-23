#!/usr/bin/env python3
"""Synthetic CPU-only tests for Liu post-training artifact diagnostics."""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aggregate_diagnostics as diagnostics
import audit_run_artifacts as audit


MARKET_VALUES: Dict[str, Any] = {
    "K": np.array([[0.8]], dtype=np.float64),
    "xbar": np.array([0.1], dtype=np.float64),
    "SigmaX": np.array([[0.2]], dtype=np.float64),
    "rho": np.array([[0.2], [0.1]], dtype=np.float64),
    "Lam": np.array([[0.2], [0.3]], dtype=np.float64),
    "Q": np.array([[0.04]], dtype=np.float64),
    "Gamma": np.array([[0.04], [0.02]], dtype=np.float64),
    "k0": np.array([0.08], dtype=np.float64),
    "lam0": np.array([0.1, 0.2], dtype=np.float64),
    "X_min": np.array([-0.5], dtype=np.float64),
    "X_max": np.array([0.5], dtype=np.float64),
    "eta": np.array([0.2], dtype=np.float64),
    "gamma": np.array(2.0, dtype=np.float64),
    "r": np.array(0.03, dtype=np.float64),
    "tau_max": np.array(3.0, dtype=np.float64),
    "W_min": np.array([0.1], dtype=np.float64),
    "W_max": np.array([2.0], dtype=np.float64),
    "market_seed": np.array([12], dtype=np.int64),
}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_checkpoint(path: Path, state: Mapping[str, np.ndarray], *, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **state)
        else:
            np.savez(handle, **state)


def npz_state_loader(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def base_args(seed: int, run_tag: str, *, model_type: str = "pipinn") -> Dict[str, Any]:
    return {
        "run_tag": run_tag,
        "model_type": model_type,
        "seed": seed,
        "market_seed": 12,
        "n_assets": 2,
        "m_states": 1,
        "outer_iters": 2,
        "w_min": 0.1,
        "w_max": 2.0,
        "eval_margin": "0.10,0.05,0.20,0.30",
        "diag_points": 64,
        "diag_every": 1,
        "timing_mode": False,
        "save_iterate_every": 0,
        "e3b_checkpoints": model_type == "pipinn",
        "theta_clip_abs": None,
        "risk_premium_mode": "affine",
        "nonaffine_eps": 0.0,
        "value_hidden": 8,
        "value_depth": 2,
        "batch_size": 16,
        "eval_epochs": 5,
        "lr": 3e-4,
    }


def diagnostic_row(
    outer: int,
    level: float,
    *,
    run_tag: str,
    model_type: str = "pipinn",
    m_ww: float | None = None,
) -> Dict[str, Any]:
    comp = 0.2 + 0.01 * level + 0.01 * outer
    row: Dict[str, Any] = {
        "timestamp": "2026-07-22T00:00:00",
        "model_type": model_type,
        "run_tag": run_tag,
        "outer_iter": outer,
        "m_ww": level + 0.5 if m_ww is None else m_ww,
        "M_num": 2.0 * level + outer,
        "guard_frac_ev": 0.0,
        "vartheta_l2_min": 0.1,
        "vartheta_l2_max": comp + 0.5,
        "vartheta_component_min": -comp,
        "vartheta_component_max": comp,
        "vartheta_abs_max": comp,
        "clip_frac_frozen": "",
    }
    if model_type == "pipinn":
        row.update(
            frozen_policy_iter=outer - 1,
            improved_policy_iter=outer,
            lam_min_sigma_frozen=level + (0.5 if outer == 1 else 0.0),
            lam_max_sigma_frozen=level + outer + 2.0,
        )
    else:
        row.update(
            lam_min_sigma_greedy=level + (0.5 if outer == 1 else 0.0),
            lam_max_sigma_greedy=level + outer + 2.0,
        )
    return row


def make_run(
    root: Path,
    *,
    seed: int,
    level: float,
    model_type: str = "pipinn",
    market_shift: float = 0.0,
    with_checkpoints: bool = False,
) -> tuple[Path, Path]:
    method_folder = "pi-pinn" if model_type == "pipinn" else "pinn"
    run_tag = f"{model_type}_seed{seed}"
    run_dir = root / method_folder / run_tag
    weight_dir = root / "weights" / method_folder / run_tag
    run_dir.mkdir(parents=True)
    weight_dir.mkdir(parents=True)
    args = base_args(seed, run_tag, model_type=model_type)
    write_json(
        run_dir / "config.json",
        {"args": args, "output_dir": str(run_dir), "weight_dir": str(weight_dir)},
    )
    write_json(
        run_dir / "status.json",
        {
            "status": "success", "run_tag": run_tag, "model_type": model_type,
            "updated_at": f"2026-07-22T00:00:{seed % 60:02d}",
        },
    )
    (run_dir / "_SUCCESS").touch()
    market = {key: np.asarray(value).copy() for key, value in MARKET_VALUES.items()}
    market["lam0"] = market["lam0"] + market_shift
    np.savez(run_dir / "market_params.npz", **market)
    rows = [
        diagnostic_row(1, level, run_tag=run_tag, model_type=model_type),
        diagnostic_row(2, level, run_tag=run_tag, model_type=model_type),
    ]
    fields = list(rows[0])
    with (run_dir / "outer_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if with_checkpoints:
        state_1 = {
            "layer.weight": np.array([[level, 2.0]], dtype=np.float32),
            "layer.bias": np.array([0.5], dtype=np.float32),
        }
        state_2 = {
            "layer.weight": np.array([[level + 1.0, 2.0]], dtype=np.float32),
            "layer.bias": np.array([0.5], dtype=np.float32),
        }
        write_checkpoint(weight_dir / "value_net_final.pt", state_2)
        # Different NPZ container bytes, identical semantic tensor state.
        write_checkpoint(weight_dir / "value_net_last.pt", state_2, compressed=True)
        write_checkpoint(weight_dir / "iterates" / "value_net_iter0001.pt", state_1)
        write_checkpoint(
            weight_dir / "iterates" / "value_net_iter0002.pt", state_2, compressed=True
        )
    return run_dir, weight_dir


class CanonicalCheckpointTests(unittest.TestCase):
    def test_tensor_hash_ignores_mapping_order_but_not_semantics(self) -> None:
        first = {
            "b": np.array([1, 2], dtype=np.float32),
            "a": np.array([[3]], dtype=np.int64),
        }
        reordered = {"a": first["a"].copy(), "b": first["b"].copy()}
        self.assertEqual(audit.canonical_tensor_hash(first), audit.canonical_tensor_hash(reordered))

        changed_value = dict(reordered)
        changed_value["b"] = np.array([1, 3], dtype=np.float32)
        self.assertNotEqual(
            audit.canonical_tensor_hash(first), audit.canonical_tensor_hash(changed_value)
        )
        changed_dtype = dict(reordered)
        changed_dtype["b"] = np.array([1, 2], dtype=np.float64)
        self.assertNotEqual(
            audit.canonical_tensor_hash(first), audit.canonical_tensor_hash(changed_dtype)
        )

    def test_market_hash_normalizes_byte_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_market_hash_") as tmp:
            root = Path(tmp)
            little = root / "little.npz"
            big = root / "big.npz"
            np.savez(little, **MARKET_VALUES)
            big_values = {
                key: (
                    np.asarray(value).astype(np.asarray(value).dtype.newbyteorder(">"))
                    if np.asarray(value).dtype.kind in "fiu" else value
                )
                for key, value in MARKET_VALUES.items()
            }
            np.savez(big, **big_values)
            self.assertEqual(audit.canonical_npz_hash(little), audit.canonical_npz_hash(big))


class ArtifactAuditTests(unittest.TestCase):
    def test_complete_e3b_run_and_semantic_checkpoint_equality(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_artifact_audit_") as tmp:
            run_dir, _ = make_run(
                Path(tmp), seed=7, level=1.0, with_checkpoints=True
            )
            observed = audit.inspect_run(run_dir, state_loader=npz_state_loader)
            checkpoint = observed["checkpoint_audit"]
            self.assertEqual(checkpoint["observed_iterate_indices"], [1, 2])
            self.assertTrue(checkpoint["final_equals_last"])
            self.assertTrue(checkpoint["last_iterate_matches_final"])
            self.assertNotEqual(
                checkpoint["checkpoints"]["final"]["file_sha256"],
                checkpoint["checkpoints"]["last"]["file_sha256"],
                "fixture should prove raw bytes need not match",
            )
            self.assertEqual(
                checkpoint["checkpoints"]["final"]["tensor_sha256"],
                checkpoint["checkpoints"]["last"]["tensor_sha256"],
            )

            manifest = audit.build_manifest([run_dir], state_loader=npz_state_loader)
            self.assertTrue(manifest["valid"])
            self.assertEqual(manifest["failure_count"], 0)

    def test_missing_iterate_and_final_last_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_artifact_bad_") as tmp:
            root = Path(tmp)
            run_dir, weight_dir = make_run(
                root, seed=1, level=1.0, with_checkpoints=True
            )
            (weight_dir / "iterates" / "value_net_iter0001.pt").unlink()
            with self.assertRaisesRegex(audit.AuditError, "iterate schedule mismatch"):
                audit.inspect_run(run_dir, state_loader=npz_state_loader)

            write_checkpoint(
                weight_dir / "iterates" / "value_net_iter0001.pt",
                {"layer.weight": np.array([[1.0, 2.0]], np.float32)},
            )
            write_checkpoint(
                weight_dir / "value_net_last.pt",
                {"layer.weight": np.array([[99.0, 2.0]], np.float32)},
            )
            with self.assertRaisesRegex(audit.AuditError, "final and last checkpoints differ"):
                audit.inspect_run(run_dir, state_loader=npz_state_loader)

    def test_conflicting_completion_markers_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_artifact_marker_") as tmp:
            run_dir, _ = make_run(
                Path(tmp), seed=1, level=1.0, with_checkpoints=True
            )
            (run_dir / "_FAILED").touch()
            with self.assertRaisesRegex(audit.AuditError, "exactly one completion marker"):
                audit.inspect_run(run_dir, state_loader=npz_state_loader)


class DiagnosticsAggregationTests(unittest.TestCase):
    def _three_seed_fixture(self, root: Path) -> Sequence[int]:
        seeds = (1, 7, 42)
        for level, seed in enumerate(seeds, start=1):
            make_run(root, seed=seed, level=float(level))
        return seeds

    def test_single_seed_does_not_report_zero_sample_sd(self) -> None:
        stats = diagnostics.summarize_seed_values([2.5])
        self.assertEqual(stats["mean"], 2.5)
        self.assertIsNone(stats["std"])
        self.assertIsNone(stats["ci95_lo"])

    def test_arbitrary_seeds_seed_first_extrema_and_student_t_ci(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_") as tmp:
            root = Path(tmp)
            seeds = self._three_seed_fixture(root)
            raw, run_index, groups = diagnostics.collect_diagnostics(
                root,
                models=("pipinn",),
                m_states=(1,),
                expected_n_assets=2,
                expected_seeds=seeds,
                min_seeds=3,
            )
            self.assertEqual(len(run_index), 3)
            self.assertEqual(len(raw), 6)
            per_seed = diagnostics.per_seed_extrema(raw)
            self.assertEqual([row["seed"] for row in per_seed], list(seeds))
            self.assertEqual(
                [row["lambda_min_sigma"] for row in per_seed], [1.0, 2.0, 3.0]
            )
            self.assertTrue(all(row["clip_frac"] is None for row in per_seed))

            setting = diagnostics.setting_summaries(per_seed)
            lambda_row = next(row for row in setting if row["metric"] == "lambda_min_sigma")
            self.assertEqual(lambda_row["outer_reduction"], "min")
            self.assertAlmostEqual(lambda_row["mean"], 2.0)
            self.assertAlmostEqual(lambda_row["std"], 1.0)
            expected_half_width = diagnostics.t_crit_95(2) / math.sqrt(3.0)
            self.assertAlmostEqual(lambda_row["ci95_lo"], 2.0 - expected_half_width)
            self.assertAlmostEqual(lambda_row["ci95_hi"], 2.0 + expected_half_width)
            clip_row = next(row for row in setting if row["metric"] == "clip_frac")
            self.assertEqual(clip_row["applicability"], "not_applicable")
            self.assertEqual(clip_row["n"], 0)

            output = root / "posthoc" / "diagnostics"
            diagnostics.write_outputs(
                output,
                raw,
                per_seed,
                setting,
                diagnostics.assumption_summaries(per_seed),
                groups,
                run_index,
            )
            for name in diagnostics.OUTPUT_FILES:
                self.assertTrue((output / name).is_file(), name)

    def test_direct_pinn_uses_greedy_pseudo_outer_semantics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_direct_") as tmp:
            root = Path(tmp)
            make_run(root, seed=23, level=1.0, model_type="pinn")
            raw, _, _ = diagnostics.collect_diagnostics(
                root,
                models=("pinn",),
                m_states=(1,),
                expected_n_assets=2,
                expected_seeds=(23,),
            )
            self.assertEqual({row["policy_kind"] for row in raw}, {"greedy"})
            self.assertEqual(
                [row["lambda_policy_iter"] for row in raw],
                [row["outer_iter"] for row in raw],
            )
            self.assertEqual({row["outer_semantics"] for row in raw}, {"direct_training_block"})
            self.assertTrue(all(row["clip_frac"] is None for row in raw))

    def test_enabled_clipping_requires_observed_fraction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_clip_") as tmp:
            root = Path(tmp)
            run_dir, _ = make_run(root, seed=1, level=1.0)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["args"]["theta_clip_abs"] = 3.0
            write_json(run_dir / "config.json", config)
            with self.assertRaisesRegex(diagnostics.DiagnosticsError, "clip_frac_frozen"):
                diagnostics.collect_diagnostics(
                    root, models=("pipinn",), m_states=(1,), expected_seeds=(1,)
                )

    def test_missing_expected_seed_and_market_mismatch_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_bad_") as tmp:
            root = Path(tmp)
            make_run(root, seed=1, level=1.0)
            make_run(root, seed=7, level=2.0, market_shift=0.5)
            with self.assertRaisesRegex(diagnostics.DiagnosticsError, "multiple training/config groups|market mismatch"):
                diagnostics.collect_diagnostics(
                    root,
                    models=("pipinn",),
                    m_states=(1,),
                    expected_n_assets=2,
                    expected_seeds=(1, 7, 42),
                )

    def test_wrong_pi_policy_indices_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_index_") as tmp:
            root = Path(tmp)
            run_dir, _ = make_run(root, seed=1, level=1.0)
            with (run_dir / "outer_history.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[1]["frozen_policy_iter"] = "0"
            with (run_dir / "outer_history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(diagnostics.DiagnosticsError, "invalid selected runs"):
                diagnostics.collect_diagnostics(
                    root, models=("pipinn",), m_states=(1,), expected_seeds=(1,)
                )

    def test_nonpositive_margin_is_observed_assumption_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="liu_diag_assumption_") as tmp:
            root = Path(tmp)
            run_dir, _ = make_run(root, seed=1, level=1.0)
            with (run_dir / "outer_history.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["m_ww"] = "-0.5"
            with (run_dir / "outer_history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            raw, _, _ = diagnostics.collect_diagnostics(
                root, models=("pipinn",), m_states=(1,), expected_seeds=(1,)
            )
            per_seed = diagnostics.per_seed_extrema(raw)
            self.assertEqual(per_seed[0]["all_concave_margin"], 0)
            self.assertTrue(math.isinf(per_seed[0]["implied_control_bound"]))
            summary = diagnostics.setting_summaries(per_seed)
            bound = next(row for row in summary if row["metric"] == "implied_control_bound")
            self.assertEqual(bound["finite_n"], 0)
            self.assertIsNone(bound["mean"])


if __name__ == "__main__":
    unittest.main()
