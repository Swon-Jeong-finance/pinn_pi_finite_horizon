"""Auxiliary end-to-end checkpoint/G-map tests for the training environment."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in lightweight CI
    TORCH_AVAILABLE = False

from merton_exact_map_fd import (
    TorchCheckpointEvaluator,
    canonical_checkpoint_state_hash,
    load_run_spec,
    sha256_file,
)

if TORCH_AVAILABLE:
    from merton_policy import portfolio_from_log_derivatives


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TorchCheckpointParityTests(unittest.TestCase):
    def _make_run(self, root: Path) -> tuple[Path, "nn.Module"]:
        torch.manual_seed(7)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 8), nn.Tanh(),
                    nn.Linear(8, 8), nn.Tanh(),
                    nn.Linear(8, 1),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.net(inputs)

        network = Net().float().eval()
        run = root / "run"
        iterate = run / "weights" / "iterates"
        iterate.mkdir(parents=True)
        checkpoint = iterate / "value_net_iter0001.pt"
        final = run / "weights" / "value_net_final.pt"
        last = run / "weights" / "value_net_last.pt"
        torch.save(network.state_dict(), checkpoint)
        # Independently serialized files need not have identical container
        # bytes; provenance equality is defined on canonical tensor state.
        torch.save(network.state_dict(), final)
        torch.save(network.state_dict(), last)
        state_digest = canonical_checkpoint_state_hash(final)
        config = {
            "args": {
                "model_type": "pipinn", "seed": 1, "market_seed": 3,
                "n_assets": 1, "tau_max": 1.0, "w_min": 0.1, "w_max": 2.0,
                "gamma": 2.0, "rho_discount": 0.04, "epsilon_bequest": 1.0, "r": 0.03,
                "eval_margin": "0.10", "eval_margin_coordinate": "y",
                "utility_cap": 1e3, "kappa_max": 30.0, "kappa_max_bound": 3.0,
                "pi_clip_abs": 2.0, "pi_init_scale": 0.5,
                "policy_guard_mode": "trainer-one-sided",
                "policy_guard_version": "merton-logw-v1",
                "network_time_coordinate": "t", "network_input_order": "t,y",
                "network_input_transform": "identity", "network_dtype": "float32",
                "activation": "tanh",
                "value_hidden": 8, "value_depth": 2,
                "outer_iters": 1, "e3b_checkpoints": True, "weight_dir": "weights",
            }
        }
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        np.savez(
            run / "market_params.npz",
            mu_excess=np.asarray([0.08]), Sigma_safe=np.asarray([[0.04]]),
            Sigma_inv_mu=np.asarray([2.0]),
            T=np.asarray([1.0]), w_min=np.asarray([0.1]), w_max=np.asarray([2.0]),
            gamma=np.asarray([2.0]), rho_discount=np.asarray([0.04]),
            epsilon=np.asarray([1.0]), r=np.asarray([0.03]),
            seed=np.asarray([1]), market_seed=np.asarray([3]),
        )
        status_payload = {
            "status": "success", "outer_iters": 1,
            "final_weight_path": str(final),
            "final_checkpoint_file_sha256": sha256_file(final),
            "final_checkpoint_state_sha256": state_digest,
            "last_checkpoint_file_sha256": sha256_file(last),
            "last_checkpoint_state_sha256": state_digest,
            "final_iterate_file_sha256": sha256_file(checkpoint),
            "final_iterate_state_sha256": state_digest,
        }
        artifacts = {}
        for label, path in (("final", final), ("last", last), ("iterate", checkpoint)):
            artifacts[label] = {
                "path": str(path.relative_to(run / "weights")),
                "file_sha256": sha256_file(path),
                "state_sha256": canonical_checkpoint_state_hash(path),
            }
        manifest_path = run / "weights" / "checkpoint_manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "status": "complete",
            "requested_outer_iters": 1,
            "checkpoint_policy": "e3b",
            "indexing": "checkpoint_outer_i_contains_v_(i-1)",
            "checkpoints": [{
                "checkpoint_outer_iter": 1,
                "source_iter": 0,
                "target_policy_iter": 1,
                "path": "iterates/value_net_iter0001.pt",
                "reasons": ["early", "final"],
                "state_sha256": canonical_checkpoint_state_hash(checkpoint),
                "file_sha256": sha256_file(checkpoint),
            }],
            "official_final": {
                "outer_iter": 1,
                "state_sha256": state_digest,
                "artifacts": artifacts,
            },
        }), encoding="utf-8")
        status_payload["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
        (run / "status.json").write_text(
            json.dumps(status_payload), encoding="utf-8"
        )
        (run / "_SUCCESS").touch()
        return run, network

    def test_inferred_mlp_derivatives_policy_and_time_orientation_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, direct = self._make_run(Path(temp))
            spec = load_run_spec(run_dir)
            self.assertEqual(spec.policy.guard_mode, "one-sided")
            self.assertEqual(spec.checkpoint_schedule, [1])
            self.assertEqual(
                spec.final_checkpoint_state_hash,
                canonical_checkpoint_state_hash(spec.checkpoints[0][1]),
            )
            evaluator = TorchCheckpointEvaluator(spec.checkpoints[0][1], spec, "cpu")
            tau = np.asarray([0.0, 0.35, 1.0], dtype=np.float64)
            y = np.asarray([-1.2, -0.3, 0.4], dtype=np.float64)
            inferred = evaluator.bundle_at_points(tau, y)

            t_t = torch.tensor((1.0 - tau)[:, None], dtype=torch.float32)
            y_t = torch.tensor(y[:, None], dtype=torch.float32, requires_grad=True)
            value = direct(torch.cat([t_t, y_t], dim=1))
            value_y = torch.autograd.grad(
                value, y_t, torch.ones_like(value), create_graph=True, retain_graph=True
            )[0]
            value_yy = torch.autograd.grad(value_y, y_t, torch.ones_like(value_y))[0]
            direct_bundle = (
                value.detach().numpy().reshape(-1),
                value_y.detach().numpy().reshape(-1),
                value_yy.detach().numpy().reshape(-1),
            )
            for actual, expected in zip(inferred, direct_bundle):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

            y_grid = np.linspace(-1.0, 0.3, 9)
            direct_c, direct_pi, direct_diag = evaluator.policy(0.25, y_grid)

            t_policy = torch.full((y_grid.size, 1), 0.75, dtype=torch.float32)
            y_policy = torch.tensor(y_grid[:, None], dtype=torch.float32, requires_grad=True)
            value_policy = direct(torch.cat([t_policy, y_policy], dim=1))
            vy_policy = torch.autograd.grad(
                value_policy, y_policy, torch.ones_like(value_policy),
                create_graph=True, retain_graph=True,
            )[0]
            vyy_policy = torch.autograd.grad(
                vy_policy, y_policy, torch.ones_like(vy_policy)
            )[0]
            wealth = torch.exp(y_policy)
            c_expected = torch.clamp(vy_policy / wealth, min=1e-8).pow(-0.5)
            kappa_expected = torch.clamp(c_expected / wealth, min=0.01, max=3.0)
            c_expected = torch.clamp(kappa_expected * wealth, min=0.001, max=2.0)
            d_safe = torch.clamp(vy_policy - vyy_policy, min=1e-8)
            pi_expected = torch.clamp(
                torch.clamp(vy_policy, min=1e-8) / d_safe * 2.0,
                min=-2.0, max=2.0,
            )
            np.testing.assert_allclose(
                direct_c, c_expected.detach().numpy().reshape(-1), rtol=2e-6, atol=2e-7
            )
            np.testing.assert_allclose(
                direct_pi, pi_expected.detach().numpy(), rtol=2e-6, atol=2e-7
            )
            frozen, _digest, _aggregate = evaluator.precompute_policy(
                np.asarray([0.25]), y_grid
            )
            cached_c, cached_pi, cached_diag = frozen(0.25, y_grid)
            np.testing.assert_allclose(cached_c, direct_c, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(cached_pi, direct_pi, rtol=2e-6, atol=2e-7)
            for key in direct_diag:
                self.assertAlmostEqual(float(cached_diag[key]), float(direct_diag[key]))

    def test_one_sided_portfolio_guard_matches_trainer_formula(self) -> None:
        value_y = torch.tensor([[-0.5], [0.4], [0.4]], dtype=torch.float64)
        value_yy = torch.tensor([[-0.6], [0.5], [0.4 - 1e-10]], dtype=torch.float64)
        y = torch.zeros_like(value_y)
        sigma_inv_mu = torch.tensor([2.0], dtype=torch.float64)
        actual, diag = portfolio_from_log_derivatives(
            value_y, value_yy, y, sigma_inv_mu,
            guard_mode="one-sided", numerator_guard=1e-8,
            denominator_guard=1e-8,
            portfolio_min=-2.0, portfolio_max=2.0,
        )
        expected = torch.clamp(
            torch.clamp(value_y, min=1e-8)
            / torch.clamp(value_y - value_yy, min=1e-8)
            * sigma_inv_mu.reshape(1, -1),
            min=-2.0, max=2.0,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(diag["numerator_guard"].reshape(-1).tolist(), [True, False, False])
        self.assertEqual(diag["denominator_guard"].reshape(-1).tolist(), [False, True, True])

    def test_manifest_state_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _network = self._make_run(Path(temp))
            manifest_path = run_dir / "weights" / "checkpoint_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoints"][0]["state_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state_sha256 mismatch"):
                load_run_spec(run_dir)

    def test_manifest_file_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _network = self._make_run(Path(temp))
            manifest_path = run_dir / "weights" / "checkpoint_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["updated_at"] = "tampered"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance hash"):
                load_run_spec(run_dir)

    def test_manifest_completion_status_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir, _network = self._make_run(Path(temp))
            manifest_path = run_dir / "weights" / "checkpoint_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest is not complete"):
                load_run_spec(run_dir)


if __name__ == "__main__":
    unittest.main()
