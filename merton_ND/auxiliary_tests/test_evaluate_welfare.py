"""PyTorch-free tests for Merton lifetime-welfare post-processing."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from auxiliary_tests._paths import SOURCE_ROOT
from evaluate_welfare import (
    MarketData,
    PolicyEvaluation,
    PolicyContract,
    RunRecord,
    analytic_optimal_value,
    discover_paper_runs,
    mean_std_ci,
    network_contract,
    network_policy_callable,
    optimal_policy_callable,
    optimal_consumption_ratio,
    parse_args,
    paired_welfare_statistics,
    policy_contract,
    simulate_policy,
    total_utility_statistics,
)


def deterministic_market(*, bequest: float = 1.0) -> MarketData:
    gamma = 2.0
    risk_free = 0.03
    discount = 0.04
    mu = np.asarray([0.0])
    sigma = np.asarray([[0.04]])
    solved = np.asarray([0.0])
    theta = 0.0
    nu = discount / gamma - (1.0 - gamma) * risk_free / gamma
    return MarketData(
        mu_excess=mu,
        sigma=sigma,
        chol=np.asarray([[0.2]]),
        sigma_inv_mu=solved,
        pi_star=solved,
        theta=theta,
        nu=nu,
        gamma=gamma,
        risk_free=risk_free,
        discount=discount,
        bequest=bequest,
        horizon=1.0,
        w_min=0.1,
        w_max=2.0,
        n_assets=1,
        market_seed=12,
    )


class WelfareRunSelectionTests(unittest.TestCase):
    def _make_run(self, root: Path, seed: int) -> None:
        run_dir = root / f"pinn_seed{seed}"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({
                "args": {
                    "model_type": "pinn",
                    "n_assets": 10,
                    "m_states": 1,
                    "seed": seed,
                    "outer_iters": 20,
                }
            }),
            encoding="utf-8",
        )
        (run_dir / "_SUCCESS").touch()

    def test_seed_cli_defaults_to_discovery_with_explicit_minimum(self) -> None:
        args = parse_args(["--out-root", "/tmp/not-used"])
        self.assertEqual(args.expected_seeds, "")
        self.assertEqual(args.min_seeds, 1)

    def test_empty_expected_seeds_uses_all_successes_and_enforces_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_run(root, 1)
            self._make_run(root, 2)
            with mock.patch(
                "evaluate_welfare.canonical_market_hash", return_value="same-market"
            ):
                selected = discover_paper_runs(
                    root, ["pinn"], [10], [], min_seeds=2
                )
                self.assertEqual(
                    [record.seed for record in selected[("pinn", 10)]], [1, 2]
                )
                with self.assertRaisesRegex(ValueError, "min_seeds=3"):
                    discover_paper_runs(
                        root, ["pinn"], [10], [], min_seeds=3
                    )

    def test_explicit_seed_set_remains_strict_and_minimum_survives_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_run(root, 1)
            self._make_run(root, 2)
            with mock.patch(
                "evaluate_welfare.canonical_market_hash", return_value="same-market"
            ):
                with self.assertRaisesRegex(ValueError, "expected=\\[1, 2, 3\\]"):
                    discover_paper_runs(
                        root, ["pinn"], [10], [1, 2, 3], min_seeds=1
                    )
                with self.assertRaisesRegex(ValueError, "min_seeds=3"):
                    discover_paper_runs(
                        root,
                        ["pinn"],
                        [10],
                        [1, 2, 3],
                        min_seeds=3,
                        allow_incomplete=True,
                    )

    def test_outer_iters_cli_selects_one_training_budget(self) -> None:
        args = parse_args([
            "--out-root", "/tmp/not-used", "--outer-iters", "20",
        ])
        self.assertEqual(args.outer_iters, 20)
        source = (SOURCE_ROOT / "evaluate_welfare.py").read_text(
            encoding="utf-8")
        self.assertIn(
            'if outer_iters is not None and _as_int(cfg.get("outer_iters")) != outer_iters:',
            source,
        )


class WelfareStatisticsTests(unittest.TestCase):
    def test_identical_paired_objectives_have_zero_loss_and_se(self) -> None:
        optimal = np.asarray([-2.0, -2.2, -1.8, -2.1])
        stats = paired_welfare_statistics(optimal, optimal, gamma=2.0, w0=0.5)
        self.assertEqual(stats.q, 1.0)
        self.assertEqual(stats.ce0, 0.5)
        self.assertEqual(stats.wl, 0.0)
        self.assertEqual(stats.se_wl, 0.0)
        self.assertEqual(stats.utility_gap, 0.0)
        self.assertEqual(stats.se_utility_gap, 0.0)

    def test_crra_ratio_and_paired_influence_match_manual_formula(self) -> None:
        optimal = np.asarray([-1.0, -2.0, -3.0, -4.0])
        learned = np.asarray([-2.0, -2.0, -4.0, -4.0])
        stats = paired_welfare_statistics(learned, optimal, gamma=2.0, w0=0.5)
        ratio = learned.mean() / optimal.mean()
        expected_q = ratio ** -1.0
        influence = expected_q * -1.0 * (
            (learned - learned.mean()) / learned.mean()
            - (optimal - optimal.mean()) / optimal.mean()
        )
        self.assertAlmostEqual(stats.q, expected_q)
        self.assertAlmostEqual(stats.se_q, np.std(influence, ddof=1) / 2.0)
        self.assertAlmostEqual(stats.ce0, 0.5 * expected_q)
        self.assertAlmostEqual(stats.wl, 1.0 - expected_q)

    def test_total_utility_se_and_seed_ci(self) -> None:
        values = np.asarray([-1.0, -2.0, -3.0, -4.0])
        stats = total_utility_statistics(values)
        self.assertEqual(stats.expected_total_utility, -2.5)
        self.assertAlmostEqual(stats.se_expected_total_utility, np.std(values, ddof=1) / 2)
        mean, std, sem, low, high = mean_std_ci([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)
        self.assertGreater(high, mean)
        self.assertLess(low, mean)
        self.assertAlmostEqual(sem, 1.0 / math.sqrt(3.0))


class ClosedFormAndSimulationTests(unittest.TestCase):
    def test_one_step_total_objective_matches_independent_oracle(self) -> None:
        market = deterministic_market(bequest=1.5)
        kappa = 0.2

        def policy(_t: float, y: np.ndarray) -> PolicyEvaluation:
            return PolicyEvaluation(
                consumption=kappa * np.exp(y),
                portfolio=np.zeros((y.size, 1)),
                masks={},
            )

        result = simulate_policy(
            market, policy=policy, n_paths=3, n_steps=1, w0=0.5,
            mc_seed=3, path_batch=2,
        )
        gamma = market.gamma
        utility = lambda amount: amount ** (1.0 - gamma) / (1.0 - gamma)
        terminal_wealth = 0.5 * math.exp(market.risk_free - kappa)
        expected = (
            utility(kappa * 0.5)
            + math.exp(-market.discount) * market.bequest
            * utility(terminal_wealth)
        )
        np.testing.assert_allclose(result.pathwise_total_utility, expected)

    def test_optimal_discretization_converges_to_analytic_value(self) -> None:
        market = deterministic_market()
        exact = analytic_optimal_value(market, 0.5)
        coarse = simulate_policy(
            market, policy=optimal_policy_callable(market), n_paths=2,
            n_steps=40, w0=0.5, mc_seed=5, path_batch=1,
        ).pathwise_total_utility.mean()
        fine = simulate_policy(
            market, policy=optimal_policy_callable(market), n_paths=2,
            n_steps=160, w0=0.5, mc_seed=5, path_batch=1,
        ).pathwise_total_utility.mean()
        self.assertLess(abs(fine - exact), abs(coarse - exact))

    def test_terminal_bequest_uses_epsilon_to_one_over_gamma(self) -> None:
        market = deterministic_market(bequest=4.0)
        terminal_kappa = float(optimal_consumption_ratio(market, market.horizon))
        self.assertAlmostEqual(terminal_kappa, 4.0 ** (-1.0 / market.gamma))
        expected_scale = (
            (1.0 + (market.nu * 2.0 - 1.0) * math.exp(-market.nu)) / market.nu
        ) ** market.gamma
        expected_value = expected_scale * 0.5 ** (1.0 - market.gamma) / (1.0 - market.gamma)
        self.assertAlmostEqual(analytic_optimal_value(market, 0.5), expected_value)

    def test_log_euler_does_not_project_wealth_to_training_window(self) -> None:
        market = deterministic_market()

        def aggressive(_t: float, y: np.ndarray) -> PolicyEvaluation:
            # c/W=100 deterministically sends log wealth far below w_min.
            wealth = np.exp(y)
            return PolicyEvaluation(
                consumption=100.0 * wealth,
                portfolio=np.zeros((y.size, 1)),
                masks={},
            )

        result = simulate_policy(
            market,
            policy=aggressive,
            n_paths=8,
            n_steps=2,
            w0=0.5,
            mc_seed=7,
            path_batch=4,
        )
        self.assertEqual(result.diagnostics.wealth_outside_path_frac, 1.0)
        self.assertGreater(result.diagnostics.wealth_below_path_time_frac, 0.0)
        self.assertLess(result.diagnostics.min_log_wealth, math.log(market.w_min))

    def test_common_random_numbers_are_reproducible_across_batch_runs(self) -> None:
        market = deterministic_market()

        def policy(_t: float, y: np.ndarray) -> PolicyEvaluation:
            return PolicyEvaluation(
                consumption=0.1 * np.exp(y),
                portfolio=np.full((y.size, 1), 0.25),
                masks={},
            )

        first = simulate_policy(
            market, policy=policy, n_paths=12, n_steps=5, w0=0.5,
            mc_seed=19, path_batch=4,
        )
        second = simulate_policy(
            market, policy=policy, n_paths=12, n_steps=5, w0=0.5,
            mc_seed=19, path_batch=4,
        )
        np.testing.assert_array_equal(
            first.pathwise_total_utility, second.pathwise_total_utility
        )
        np.testing.assert_array_equal(first.terminal_log_wealth, second.terminal_log_wealth)


class PolicyContractTests(unittest.TestCase):
    def record(self, model: str, config_doc: dict, args: dict) -> RunRecord:
        return RunRecord(
            run_dir=Path("/tmp/run"), model_type=model, n_assets=10, seed=1,
            group="g", updated_at="", status="success", config_args=args,
            config_doc=config_doc,
        )

    def test_pipinn_none_mode_rejects_stray_resolved_bound(self) -> None:
        doc = {
            "policy_guard_mode": "trainer-one-sided",
            "policy_guard_version": "merton-logw-v1",
            "policy_bounds_mode": "none",
            "policy_kappa_min": None, "policy_kappa_max": None,
            "policy_c_min": None, "policy_c_max": None,
            "policy_pi_min": None, "policy_pi_max": None,
        }
        contract = policy_contract(self.record("pipinn", doc, {}))
        self.assertEqual(contract.bounds_mode, "none")
        self.assertIsNone(contract.portfolio_max)
        doc["policy_pi_max"] = 2.0
        with self.assertRaisesRegex(ValueError, "finite bounds remain"):
            policy_contract(self.record("pipinn", doc, {}))

    def test_direct_pinn_uses_recorded_hard_evaluation_map_not_hjb_guard(self) -> None:
        doc = {
            "policy_bounds_mode": "stabilized",
            "policy_kappa_min": 0.01, "policy_kappa_max": 3.0,
            "policy_c_min": 0.001, "policy_c_max": 2.0,
            "policy_pi_min": -2.0, "policy_pi_max": 2.0,
        }
        args = {
            "evaluation_policy_guard_mode": "one-sided-hard-clamp",
            "hjb_guard_mode": "softplus",
        }
        contract = policy_contract(self.record("pinn", doc, args))
        self.assertEqual(contract.numerator_guard, 1e-8)
        self.assertEqual(contract.denominator_guard, 1e-8)
        self.assertEqual(contract.portfolio_min, -2.0)

    def test_stabilized_mode_allows_explicitly_unbounded_portfolio_only(self) -> None:
        doc = {
            "policy_guard_mode": "trainer-one-sided",
            "policy_guard_version": "merton-logw-v1",
            "policy_bounds_mode": "stabilized",
            "policy_kappa_min": 0.01, "policy_kappa_max": 3.0,
            "policy_c_min": 0.001, "policy_c_max": 2.0,
            "policy_pi_min": None, "policy_pi_max": None,
        }
        contract = policy_contract(self.record("pipinn", doc, {}))
        self.assertEqual(contract.kappa_max, 3.0)
        self.assertIsNone(contract.portfolio_min)

    def test_network_contract_rejects_ambiguous_checkpoint_interpretation(self) -> None:
        doc = {
            "network_time_coordinate": "t",
            "network_input_order": "t,y",
            "network_input_transform": "identity",
            "network_activation": "tanh",
            "network_dtype": "float32",
            "trainer_source_marker": (
                "merton-pipinn-logw-trainer-one-sided-selection-v2"
            ),
            "trainer_source_sha256": "a" * 64,
        }
        record = self.record(
            "pipinn", doc, {"value_hidden": 256, "value_depth": 3}
        )
        resolved = network_contract(record)
        self.assertEqual(resolved["network_input_order"], "t,y")
        doc["network_time_coordinate"] = "tau"
        with self.assertRaisesRegex(ValueError, "network_time_coordinate"):
            network_contract(record)

    def test_policy_contract_rejects_reversed_bounds(self) -> None:
        doc = {
            "policy_guard_mode": "trainer-one-sided",
            "policy_guard_version": "merton-logw-v1",
            "policy_bounds_mode": "stabilized",
            "policy_kappa_min": 2.0, "policy_kappa_max": 1.0,
            "policy_c_min": 0.001, "policy_c_max": 2.0,
            "policy_pi_min": -2.0, "policy_pi_max": 2.0,
        }
        with self.assertRaisesRegex(ValueError, "policy_kappa_min"):
            policy_contract(self.record("pipinn", doc, {}))


try:
    import torch as _torch
except Exception:
    _torch = None


@unittest.skipUnless(_torch is not None, "PyTorch is not installed")
class TorchGreedyPolicyParityTests(unittest.TestCase):
    def test_autograd_policy_matches_closed_form_crra_derivatives(self) -> None:
        torch = _torch
        assert torch is not None

        class HomogeneousValue(torch.nn.Module):
            def forward(self, t: object, y: object) -> object:
                del t
                return -torch.exp(-y)

        market = replace(
            deterministic_market(),
            mu_excess=np.asarray([0.04]),
            sigma_inv_mu=np.asarray([1.0]),
            pi_star=np.asarray([0.5]),
            theta=0.04,
        )
        contract = PolicyContract(
            bounds_mode="none", vw_guard=1e-8, numerator_guard=1e-8,
            denominator_guard=1e-8, kappa_min=None, kappa_max=None,
            consumption_min=None, consumption_max=None,
            portfolio_min=None, portfolio_max=None,
        )
        policy = network_policy_callable(
            torch=torch, network=HomogeneousValue(), device=torch.device("cpu"),
            market=market, contract=contract, policy_chunk=2,
        )
        y = np.log(np.asarray([0.25, 0.5, 1.0]))
        evaluated = policy(0.3, y)
        np.testing.assert_allclose(evaluated.consumption, np.exp(y), rtol=2e-6)
        np.testing.assert_allclose(evaluated.portfolio[:, 0], 0.5, rtol=2e-6)
        self.assertFalse(np.any(evaluated.masks["vw_guard"]))
        self.assertFalse(np.any(evaluated.masks["denominator_guard"]))


if __name__ == "__main__":
    unittest.main()
