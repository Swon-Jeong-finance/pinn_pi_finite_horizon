#!/usr/bin/env python3
"""Regression tests for historical Liu normalized-policy reconstruction."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import postprocess_contraction as pc  # noqa: E402
import reconstruct_vartheta_trajectory as rv  # noqa: E402
from evaluate_margin_bundle import RunRecord, sha256_file  # noqa: E402

try:
    import torch
except ModuleNotFoundError:
    torch = None


def record(run_dir: Path, final: Path) -> RunRecord:
    return RunRecord(
        run_dir=run_dir,
        model_type="pipinn",
        n_assets=30,
        m_states=3,
        seed=1,
        group="training-group",
        updated_at="2026-07-23T00:00:00",
        config_doc={},
        config_args={"value_hidden": 64, "value_depth": 2},
        effective_eval_args={},
        checkpoint=final,
    )


class ReconstructionContractTests(unittest.TestCase):
    def test_group_key_matches_figure2_postprocessor(self) -> None:
        config = {
            "model_type": "pipinn",
            "n_assets": 30,
            "m_states": 3,
            "seed": 1,
            "eval_margin": "0.10,0.30",
        }
        training_group, _ = pc.group_key(dict(config))
        self.assertEqual(
            rv.convergence_group_key(training_group, 0.10),
            pc.convergence_group_key(config),
        )

    def test_missing_middle_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            final = base / "value_net_final.pt"
            final.touch()
            iterate_dir = base / "iterates"
            iterate_dir.mkdir()
            (iterate_dir / "value_net_iter0001.pt").touch()
            (iterate_dir / "value_net_iter0003.pt").touch()
            with self.assertRaisesRegex(FileNotFoundError, "iter0002"):
                rv.resolve_iterate_checkpoints(record(base, final), [1, 2, 3])

    def test_extra_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            final = base / "value_net_final.pt"
            final.touch()
            iterate_dir = base / "iterates"
            iterate_dir.mkdir()
            for outer in (1, 2, 3, 4):
                (iterate_dir / f"value_net_iter{outer:04d}.pt").touch()
            with self.assertRaisesRegex(ValueError, "outside the completed"):
                rv.resolve_iterate_checkpoints(record(base, final), [1, 2, 3])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_final_outer_tensor_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            final = base / "value_net_final.pt"
            last = base / "value_net_iter0002.pt"
            torch.save({"weight": torch.tensor([1.0, 2.0])}, final)
            torch.save({"weight": torch.tensor([1.0, 3.0])}, last)
            with self.assertRaisesRegex(ValueError, "does not equal"):
                rv.verify_final_outer_state(final, last, torch=torch)
            torch.save({"weight": torch.tensor([1.0, 2.0])}, last)
            self.assertEqual(
                rv.verify_final_outer_state(final, last, torch=torch),
                rv.canonical_state_hash({"weight": torch.tensor([1.0, 2.0])}),
            )


class OverlayProvenanceTests(unittest.TestCase):
    def make_overlay(self, base: Path):
        run_dir = base / "run"
        run_dir.mkdir()
        sources = {}
        for name, content in (
            ("config.json", b"config"),
            ("outer_history.csv", b"outer"),
            ("market_params.npz", b"market"),
            ("closed_form_ode.npz", b"closed"),
        ):
            path = run_dir / name
            path.write_bytes(content)
            sources[name] = path
        checkpoint = base / "weights" / "iterates" / "value_net_iter0001.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        group = "figure-group"
        csv_path = base / rv.PER_OUTER_FILE
        fields = (
            "group", "training_seed", "outer_iter", "metric", "value", "run_dir",
            "config_sha256", "outer_history_sha256", "market_params",
            "market_params_file_sha256", "market_hash", "closed_form",
            "closed_form_file_sha256", "closed_form_hash", "checkpoint",
            "checkpoint_sha256",
        )
        row = {
            "group": group,
            "training_seed": 1,
            "outer_iter": 1,
            "metric": pc.POLICY_METRIC,
            "value": 0.25,
            "run_dir": str(run_dir),
            "config_sha256": sha256_file(sources["config.json"]),
            "outer_history_sha256": sha256_file(sources["outer_history.csv"]),
            "market_params": str(sources["market_params.npz"]),
            "market_params_file_sha256": sha256_file(sources["market_params.npz"]),
            "market_hash": "canonical-market",
            "closed_form": str(sources["closed_form_ode.npz"]),
            "closed_form_file_sha256": sha256_file(sources["closed_form_ode.npz"]),
            "closed_form_hash": "canonical-closed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        provenance = {
            "artifact_sha256": {csv_path.name: sha256_file(csv_path)}
        }
        (base / rv.PROVENANCE_FILE).write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        (base / rv.SUCCESS_MARKER).touch()
        meta = {
            "group": group,
            "runs": {1: run_dir},
            "market_hashes": {1: "canonical-market"},
        }
        return csv_path, meta, sources

    def test_overlay_binds_all_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            csv_path, meta, sources = self.make_overlay(Path(temp))
            series, provenance = pc.load_vartheta_overlay(csv_path, meta)
            self.assertEqual(series, {1: {1: 0.25}})
            self.assertEqual(provenance["csv_sha256"], sha256_file(csv_path))

            sources["closed_form_ode.npz"].write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "source SHA-256 changed"):
                pc.load_vartheta_overlay(csv_path, meta)


if __name__ == "__main__":
    unittest.main()
