"""Regression tests for the opt-in cross-configuration seed merge."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from auxiliary_tests._paths import SOURCE_ROOT

REPO = SOURCE_ROOT
SCRIPT = REPO / "aggregate_seeds.py"
METRICS = ("RelL2_V", "RelL2_D", "e_D_sup", "RelL2_pi", "RelL2_c")


def write_run(
    root: Path,
    name: str,
    *,
    seed: object,
    variant: str,
    value: float,
    updated_at: str,
    status: str = "success",
    model_type: str = "pinn",
    n_assets: int = 2,
    m_states: int = 1,
    market_shift: float = 0.0,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(
            {
                "args": {
                    "model_type": model_type,
                    "n_assets": n_assets,
                    "m_states": m_states,
                    "seed": seed,
                    "market_seed": 12,
                    "variant": variant,
                    "eval_margin": "0.10",
                }
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "success": "_SUCCESS",
        "failed": "_FAILED",
        "stopped_early": "_STOPPED_EARLY",
    }[status]
    (run / marker).write_text("", encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"status": status, "updated_at": updated_at}),
        encoding="utf-8",
    )

    sigma = np.eye(n_assets, dtype=np.float64) * 0.04
    np.savez(
        run / "market_params.npz",
        mu_excess=np.full(
            n_assets, 0.05 + market_shift, dtype=np.float64
        ),
        Sigma_safe=sigma,
        chol=np.linalg.cholesky(sigma),
        pi_star=np.full(n_assets, 0.1, dtype=np.float64),
        Theta=np.asarray(0.625),
        nu=np.asarray(0.02),
        gamma=np.asarray(2.0),
        r=np.asarray(0.03),
        rho_discount=np.asarray(0.04),
        epsilon=np.asarray(1.0),
        T=np.asarray(1.0),
        w_min=np.asarray(0.1),
        w_max=np.asarray(2.0),
        n_assets=np.asarray(n_assets),
        market_seed=np.asarray(12),
    )
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["scope", "eval_margin", "metric", "value"]
        )
        writer.writeheader()
        for metric in METRICS:
            writer.writerow(
                {
                    "scope": "fulldim",
                    "eval_margin": 0.10,
                    "metric": metric,
                    "value": value,
                }
            )
    return run


def run_aggregate(
    root: Path,
    *,
    expected_seeds: str = "",
    merge: bool = False,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--out-root", str(root)]
    if expected_seeds:
        command.extend(["--expected-seeds", expected_seeds])
    if merge:
        command.append("--merge-config-groups")
    command.extend(extra)
    return subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


class MergeConfigurationGroupTests(unittest.TestCase):
    def test_default_is_strict_but_opt_in_merge_combines_seed_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(
                root, "a_seed1", seed=1, variant="a", value=1.0,
                updated_at="2026-07-23T00:00:01+00:00",
            )
            write_run(
                root, "a_seed2", seed=2, variant="a", value=2.0,
                updated_at="2026-07-23T00:00:02+00:00",
            )
            write_run(
                root, "b_seed3", seed=3, variant="b", value=3.0,
                updated_at="2026-07-23T00:00:03+00:00",
            )

            strict = run_aggregate(root, expected_seeds="1,2,3")
            self.assertNotEqual(strict.returncode, 0)
            self.assertIn("paper aggregation validation failed", strict.stderr)

            merged = run_aggregate(root, expected_seeds="1,2,3", merge=True)
            self.assertEqual(merged.returncode, 0, merged.stdout + merged.stderr)
            self.assertIn("--merge-config-groups is active", merged.stdout)

            with (root / "seed_summary" / "summary_long.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            value_row = next(row for row in rows if row["metric"] == "RelL2_V")
            self.assertEqual(value_row["n"], "3")
            self.assertEqual(value_row["seeds"], "1;2;3")
            self.assertAlmostEqual(float(value_row["mean"]), 2.0)
            self.assertTrue(value_row["group"].startswith("panel_"))

            with (root / "seed_summary" / "runs_index.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                index = list(csv.DictReader(handle))
            self.assertEqual({row["group"] for row in index}, {value_row["group"]})
            self.assertEqual(len({row["source_group"] for row in index}), 2)
            self.assertTrue(all(row["used"] == "1" for row in index))

            groups = json.loads(
                (root / "seed_summary" / "groups.json").read_text(
                    encoding="utf-8"
                )
            )
            panel = groups[value_row["group"]]
            self.assertEqual(panel["aggregation_mode"], "model_type+n_assets+m_states")
            self.assertEqual(len(panel["source_training_groups"]), 2)
            self.assertEqual(set(panel["source_group_configs"]),
                             set(panel["source_training_groups"]))
            self.assertTrue(groups["_aggregation"]["merge_config_groups"])

    def test_merge_never_crosses_method_n_or_m_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = (
                ("pinn_n2_m1", "pinn", 2, 1),
                ("pipinn_n2_m1", "pipinn", 2, 1),
                ("pinn_n3_m1", "pinn", 3, 1),
                ("pinn_n2_m2", "pinn", 2, 2),
            )
            for index, (name, method, n_assets, m_states) in enumerate(cases):
                write_run(
                    root, name, seed=1, variant=f"v{index}", value=index + 1.0,
                    updated_at=f"2026-07-23T00:00:0{index + 1}+00:00",
                    model_type=method, n_assets=n_assets, m_states=m_states,
                )

            result = run_aggregate(root, merge=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / "seed_summary" / "success_rates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            with (root / "seed_summary" / "summary_long.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                long_rows = [
                    row for row in csv.DictReader(handle)
                    if row["metric"] == "RelL2_V"
                ]
            self.assertEqual(
                {
                    (row["model_type"], int(row["n_assets"]), int(row["m_states"]))
                    for row in long_rows
                },
                {(method, n_assets, m_states)
                 for _name, method, n_assets, m_states in cases},
            )

    def test_panel_wide_newest_run_wins_and_string_seed_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(
                root, "old_seed1", seed=1, variant="old", value=1.0,
                updated_at="2026-07-23T00:00:01+00:00",
            )
            write_run(
                root, "new_seed1", seed="1", variant="new", value=9.0,
                updated_at="2026-07-23T00:00:09+00:00",
            )
            write_run(
                root, "seed2", seed=2, variant="old", value=3.0,
                updated_at="2026-07-23T00:00:02+00:00",
            )

            result = run_aggregate(root, expected_seeds="1,2", merge=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / "seed_summary" / "summary_long.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(
                    row for row in csv.DictReader(handle)
                    if row["metric"] == "RelL2_V"
                )
            self.assertEqual(row["n"], "2")
            self.assertEqual(row["seeds"], "1;2")
            self.assertAlmostEqual(float(row["mean"]), 6.0)

            with (root / "seed_summary" / "runs_index.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                index = {row["run_dir"]: row for row in csv.DictReader(handle)}
            self.assertEqual(index["old_seed1"]["used"], "0")
            self.assertEqual(index["new_seed1"]["used"], "1")
            self.assertEqual(index["seed2"]["used"], "1")

    def test_newer_failed_run_is_not_backfilled_from_older_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(
                root, "old_success", seed=1, variant="old", value=1.0,
                updated_at="2026-07-23T00:00:01+00:00",
            )
            write_run(
                root, "new_failed", seed=1, variant="new", value=9.0,
                updated_at="2026-07-23T00:00:09+00:00", status="failed",
            )

            result = run_aggregate(root, expected_seeds="1", merge=True)
            self.assertNotEqual(result.returncode, 0)
            errors = (
                root / "seed_summary" / "validation_errors.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("successful seeds=[]", errors)
            self.assertIn("missing=[1]", errors)

    def test_merge_mode_still_rejects_mixed_market_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(
                root, "a_seed1", seed=1, variant="a", value=1.0,
                updated_at="2026-07-23T00:00:01+00:00",
                market_shift=0.0,
            )
            write_run(
                root, "b_seed2", seed=2, variant="b", value=2.0,
                updated_at="2026-07-23T00:00:02+00:00",
                market_shift=0.01,
            )

            # Merge mode enables canonical-market validation even without an
            # explicit expected seed panel.
            result = run_aggregate(root, merge=True)
            self.assertNotEqual(result.returncode, 0)
            errors = (
                root / "seed_summary" / "validation_errors.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("expected one canonical hash", errors)


if __name__ == "__main__":
    unittest.main()
