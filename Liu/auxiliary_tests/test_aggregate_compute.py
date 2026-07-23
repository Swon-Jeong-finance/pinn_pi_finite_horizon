import csv
import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import aggregate_compute as ac
from aggregate_seeds import MARKET_HASH_KEYS


def _write_run(root: Path, model: str, seed: int, *, timing: bool = True) -> Path:
    run = root / model / f"seed{seed}"
    run.mkdir(parents=True)
    args = {
        "model_type": model,
        "n_assets": 30,
        "m_states": 3,
        "seed": seed,
        "market_seed": 12,
        "timing_mode": timing,
        "batch_size": 100,
        "outer_iters": 2,
        "eval_epochs": 3,
        "device": "cuda:0",
        "output_root": str(run),
        "weight_root": str(run / "weights"),
        "run_tag": f"{model}_{seed}",
    }
    (run / "config.json").write_text(
        json.dumps({
            "args": args,
            "runtime_environment": {
                "gpu_name": "Synthetic GPU",
                "gpu_total_memory_bytes": 24 * 1024**3,
                "gpu_compute_capability": "9.0",
                "torch_version": "test",
                "cuda_runtime_version": "test",
                "cudnn_version": 1,
                "python_version": "test",
                "numpy_version": "test",
                "platform": "test",
                "cuda_available": True,
                "effective_device": "cuda:0",
            },
        }),
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps({
            "status": "success",
            "train_wall_sec": 10.0 + seed,
            "core_train_wall_sec": 9.0 + seed,
            "total_optimizer_steps": 100 + seed,
            "train_gpu_peak_mem_bytes": 1024**3 * (2 + seed),
            "eval_gpu_peak_mem_bytes": 1024**3,
            "timing_mode": timing,
        }),
        encoding="utf-8",
    )
    (run / "_SUCCESS").touch()
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scope", "eval_margin", "metric", "value"],
        )
        writer.writeheader()
        writer.writerow({
            "scope": "fulldim", "eval_margin": 0.10,
            "metric": "RelL2_V", "value": 0.01 * seed,
        })
        writer.writerow({
            "scope": "fulldim", "eval_margin": 0.10,
            "metric": "RelL2_theta", "value": 0.02 * seed,
        })
    market = {key: np.asarray([1.0]) for key in MARKET_HASH_KEYS}
    np.savez(run / "market_params.npz", **market)
    return run


class AggregateComputeTests(unittest.TestCase):
    def test_summarize_single_seed_has_no_fake_uncertainty(self):
        stats = ac.summarize([3.0])
        self.assertEqual(stats["n"], 1)
        self.assertEqual(stats["mean"], 3.0)
        self.assertTrue(math.isnan(stats["std"]))
        self.assertTrue(math.isnan(stats["ci95_lo"]))

    def test_collect_validate_and_write_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("pinn", "pipinn"):
                _write_run(root, model, 1)
                _write_run(root, model, 2)
            observations, groups, skipped = ac.collect_observations(
                str(root),
                headline_margin=0.10,
                expected_models=("pinn", "pipinn"),
                expected_m_states=(3,),
                expected_n_assets=30,
            )
            self.assertEqual(len(observations), 4)
            self.assertEqual(len(groups), 2)
            self.assertEqual(skipped, [])
            ac.validate_support(observations, expected_seeds=(1, 2), min_runs=2)
            ac.validate_expected_cells(
                observations, models=("pinn", "pipinn"), m_states=(3,)
            )
            run_rows, long_rows, table_rows = ac.build_output_rows(observations)
            self.assertEqual(len(run_rows), 4)
            self.assertEqual(len(table_rows), 2)
            self.assertEqual({row["measure"] for row in long_rows}, set(ac.MEASURES))
            for row in table_rows:
                self.assertEqual(row["core_train_wall_sec_n"], 2)
                self.assertTrue(math.isfinite(float(row["core_train_wall_sec_std"])))

    def test_non_timing_runs_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _write_run(root, "pinn", 1, timing=False)
            observations, _groups, skipped = ac.collect_observations(
                str(root), headline_margin=0.10
            )
            self.assertEqual(observations, [])
            self.assertTrue(
                any(str(run) in item and "not a timing_mode run" in item for item in skipped)
            )

    def test_missing_runtime_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _write_run(root, "pinn", 1)
            payload = json.loads((run / "config.json").read_text(encoding="utf-8"))
            payload.pop("runtime_environment")
            (run / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime metadata"):
                ac.collect_observations(str(root), headline_margin=0.10)

    def test_gpu_memory_is_part_of_hardware_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "pinn", 1)
            run = _write_run(root, "pinn", 2)
            path = run / "config.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_environment"]["gpu_total_memory_bytes"] = 48 * 1024**3
            path.write_text(json.dumps(payload), encoding="utf-8")
            _observations, groups, _skipped = ac.collect_observations(
                str(root), headline_margin=0.10
            )
            self.assertEqual(len(groups), 2)

    def test_success_marker_and_status_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _write_run(root, "pinn", 1)
            path = run / "status.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status"] = "failed"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagreement"):
                ac.collect_observations(str(root), headline_margin=0.10)

    def test_eval_gpu_peak_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _write_run(root, "pinn", 1)
            path = run / "status.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("eval_gpu_peak_mem_bytes")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "eval_gpu_peak_mem_bytes"):
                ac.collect_observations(str(root), headline_margin=0.10)

    def test_bad_invocation_does_not_delete_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "derived"
            output.mkdir()
            old = output / "compute_table.csv"
            old.write_text("old-good-output\n", encoding="utf-8")
            args = argparse.Namespace(
                out_root=str(root), output=str(output), expected_seeds="1",
                min_runs=1, models="invalid", m_states="3", n_assets=30,
                headline_margin=0.10, include_main_runs=False, overwrite=True,
                allow_unknown_runtime=False, skip_figure=True, formats="png",
                fig_width=10.0, fig_height=6.0, font_size=10.0, dpi=150,
            )
            with self.assertRaisesRegex(ValueError, "--models"):
                ac.run(args)
            self.assertEqual(old.read_text(encoding="utf-8"), "old-good-output\n")

    def test_missing_method_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "pinn", 1)
            observations, _groups, _skipped = ac.collect_observations(
                str(root), headline_margin=0.10
            )
            with self.assertRaisesRegex(ValueError, "method/M cells"):
                ac.validate_expected_cells(
                    observations, models=("pinn", "pipinn"), m_states=(3,)
                )

    def test_mixed_market_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "pinn", 1)
            second = _write_run(root, "pinn", 2)
            with np.load(second / "market_params.npz") as data:
                market = {key: np.asarray(data[key]) for key in data.files}
            market[MARKET_HASH_KEYS[0]] = np.asarray([2.0])
            np.savez(second / "market_params.npz", **market)
            observations, _groups, _skipped = ac.collect_observations(
                str(root), headline_margin=0.10
            )
            with self.assertRaisesRegex(ValueError, "mixed market"):
                ac.validate_support(observations, expected_seeds=(1, 2), min_runs=2)

    def test_end_to_end_writes_derived_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("pinn", "pipinn"):
                _write_run(root, model, 1)
            output = root / "derived"
            metadata = ac.run(argparse.Namespace(
                out_root=str(root), output=str(output), expected_seeds="1",
                min_runs=1, models="pinn,pipinn", m_states="3", n_assets=30,
                headline_margin=0.10, include_main_runs=False, overwrite=False,
                allow_unknown_runtime=False,
                skip_figure=True, formats="png,pdf", fig_width=10.0,
                fig_height=6.0, font_size=10.0, dpi=150,
            ))
            self.assertEqual(metadata["n_observations"], 2)
            self.assertTrue((output / "compute_runs.csv").is_file())
            self.assertTrue((output / "compute_table.csv").is_file())
            self.assertTrue((output / "compute_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
