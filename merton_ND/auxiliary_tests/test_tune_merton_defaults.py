"""Regression lock for the user-approved Merton sweep baseline."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from auxiliary_tests._paths import SOURCE_ROOT

SCRIPT = SOURCE_ROOT / "tune_merton.sh"


def parse_array(source: str, name: str) -> dict[str, str]:
    match = re.search(
        rf"declare -A {re.escape(name)}=\(\n(?P<body>.*?)\n\)",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing Bash array {name}")
    values: dict[str, str] = {}
    for key, value in re.findall(
        r"^\s*\[([^]]+)\]=(.*)$", match.group("body"), flags=re.MULTILINE
    ):
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        values[key] = value
    return values


class TuneMertonBaselineTests(unittest.TestCase):
    def test_attached_pinn_baseline_is_locked(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        expected = {
            "n_assets": "50", "seed": "12", "market_seed": "12",
            "gamma": "2.0", "rho_discount": "0.04", "r": "0.03",
            "epsilon_bequest": "1.0", "tau_max": "1.0",
            "w_min": "0.1", "w_max": "2.0", "sigma_lo": "0.10",
            "sigma_hi": "0.25", "rho_max": "1.0", "kappa_max": "30.0",
            "pi_scale": "0.6", "mu_noise_rel": "0.02",
            "value_hidden": "256", "value_depth": "3",
            "batch_size": "10000", "terminal_frac": "0.25",
            "lr": "5e-4", "outer_iters": "20", "eval_epochs": "2000",
            "resample_every": "200", "scheduler_patience": "25",
            "scheduler_factor": "0.5", "scheduler_min_lr": "1e-8",
            "lr_schedule": "plateau", "qsel_rollback_factor": "10.0",
            "qsel_rollback_lr_factor": "1.0",
            "qsel_rollback_max_rescues": "2",
            "hjb_guard_mode": "softplus",
            "hjb_vy_guard_eps": "1e-8", "hjb_denom_guard_eps": "1e-8",
            "hjb_vy_guard_tau": "0.05", "hjb_denom_guard_tau": "0.05",
            "hjb_vy_guard_tau_min": "1e-4",
            "hjb_denom_guard_tau_min": "1e-4",
            "hjb_guard_anneal_every": "0", "hjb_guard_anneal_factor": "0.5",
            "w_terminal": "10.0", "w_shape": "1.0", "w_eta": "1.5",
            "eta_clip": "10.0", "policy_bounds_mode": "stabilized",
            "pi_clip_abs": "${PI_CLIP_ABS:-2.0}", "pres_target": "none",
            "val_points": "50000", "val_terminal_points": "10000",
            "val_every": "25", "save_iterate_every": "0",
            "diag_points": "8192", "diag_every": "1", "print_every": "1000",
            "eval_margin": "0.10,0.0,0.05,0.15,0.20,0.25,0.30",
            "test_points": "100000", "n_tau": "100", "n_x": "100",
        }
        self.assertEqual(parse_array(source, "BASE_PINN"), expected)

    def test_attached_pipinn_baseline_is_locked(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        expected = {
            "n_assets": "50", "seed": "12", "market_seed": "12",
            "gamma": "2.0", "rho_discount": "0.04", "r": "0.03",
            "epsilon_bequest": "1.0", "tau_max": "1.0",
            "w_min": "0.1", "w_max": "2.0", "sigma_lo": "0.10",
            "sigma_hi": "0.25", "rho_max": "1.0", "kappa_max": "30.0",
            "pi_scale": "0.6", "mu_noise_rel": "0.02",
            "policy_bounds_mode": "stabilized",
            "pi_clip_abs": "${PI_CLIP_ABS:-2.0}",
            "kappa_max_bound": "3.0", "utility_cap": "1e3",
            "value_hidden": "256", "value_depth": "3",
            "batch_size": "10000", "terminal_frac": "0.5",
            "lr": "5e-4", "outer_iters": "20", "eval_epochs": "2000",
            "scheduler_patience": "20", "scheduler_factor": "0.5",
            "scheduler_min_lr": "1e-8", "lr_schedule": "carry_plateau",
            "adam_reset": "keep", "carry_lr_min": "1e-8",
            "carry_lr_max": "5e-4", "pi_init_method": "myopic",
            "theta_init_scale": "0.5", "c_init_method": "proportional",
            "w_terminal": "10.0", "w_shape": "1.0", "w_eta": "20.0",
            "eta_clip": "10.0", "eta_focus_w": "none",
            "pres_target": "none", "val_points": "50000",
            "val_terminal_points": "10000", "val_every": "25",
            "inner_best_restore": "1", "sel_points": "10000",
            "sel_terminal_points": "2000", "sel_every": "50",
            "sel_patience": "0", "pe_resample_every": "0",
            "save_iterate_every": "0", "diag_points": "8192",
            "diag_every": "1", "print_every_outer": "1",
            "print_every_eval": "0",
            "eval_margin": "0.10,0.0,0.05,0.15,0.20,0.25,0.30",
            "test_points": "100000", "n_tau": "100", "n_x": "100",
        }
        self.assertEqual(parse_array(source, "BASE_PIPINN"), expected)

    def test_active_run_list_and_absolute_output_contract_are_locked(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("export PYTHONUNBUFFERED=1", source)
        active_runs = [
            line.strip()
            for line in source.splitlines()
            if re.match(r"^run_(?:pinn|pipinn)\s+", line)
        ]
        self.assertEqual(active_runs, [
            "run_pinn   n_assets=10",
            "run_pinn   n_assets=50",
        ])
        self.assertIn('OUT_ROOT="$(realpath -m "$OUT_ROOT")"', source)

    def test_cross_config_aggregation_is_explicit_opt_in(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'MERGE_CONFIG_GROUPS="${MERGE_CONFIG_GROUPS:-0}"',
            source,
        )
        self.assertIn(
            'aggregate_args+=(--merge-config-groups)',
            source,
        )


if __name__ == "__main__":
    unittest.main()
