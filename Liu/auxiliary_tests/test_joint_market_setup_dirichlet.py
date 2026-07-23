#!/usr/bin/env python3
"""Regression tests for Liu's identity-block market correlation convention."""
from __future__ import annotations

import unittest

import numpy as np

from joint_market_setup_dirichlet import (
    MARKET_SCHEMA_VERSION,
    RHO_CONVENTION,
    canonicalize_cross_correlation,
    generate_joint_market_params,
    identity_block_correlation_diagnostics,
    rho_snapshot_metadata,
    validate_market_snapshot,
    validate_rho_snapshot,
)


def _inverse_root(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return (eigenvectors * np.power(eigenvalues, -0.5)[None, :]) @ eigenvectors.T


class CanonicalCrossCorrelationTests(unittest.TestCase):
    def test_identity_source_blocks_leave_cross_block_unchanged(self) -> None:
        raw = np.array([[0.2, -0.1], [0.05, 0.3], [-0.1, 0.15]])
        actual = canonicalize_cross_correlation(
            np.eye(3), raw, np.eye(2)
        )
        np.testing.assert_allclose(actual, raw, rtol=0.0, atol=2.0e-15)

    def test_whitening_is_the_identity_block_congruence(self) -> None:
        rng = np.random.default_rng(20260723)
        sample = rng.standard_normal((12, 5))
        covariance = sample.T @ sample + 0.5 * np.eye(5)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / scale[:, None] / scale[None, :]
        psi = correlation[:3, :3]
        phi = correlation[3:, 3:]
        raw = correlation[:3, 3:]
        canonical = canonicalize_cross_correlation(psi, raw, phi)

        transform = np.block([
            [_inverse_root(psi), np.zeros((3, 2))],
            [np.zeros((2, 3)), _inverse_root(phi)],
        ])
        transformed = transform @ correlation @ transform.T
        expected = np.block([
            [np.eye(3), canonical],
            [canonical.T, np.eye(2)],
        ])
        np.testing.assert_allclose(transformed, expected, rtol=2.0e-12, atol=2.0e-13)
        self.assertGreater(float(np.linalg.eigvalsh(expected)[0]), 0.0)

    def test_paper_market_seed_12_is_strictly_admissible(self) -> None:
        expected_minimum = {
            1: 0.3428230354018328,
            3: 0.1223650279361440,
            5: 0.1684356417950524,
        }
        for m_states, expected in expected_minimum.items():
            with self.subTest(m_states=m_states):
                params = generate_joint_market_params(
                    n=30,
                    k=m_states,
                    seed=12,
                    sample_alpha=True,
                    alpha_dist="dirichlet",
                )
                diagnostics = identity_block_correlation_diagnostics(
                    params.rho_canonical
                )
                self.assertAlmostEqual(diagnostics["min_eig"], expected, places=12)
                self.assertLess(diagnostics["rho_spectral_norm"], 1.0)

        raw_m5 = generate_joint_market_params(
            n=30,
            k=5,
            seed=12,
            sample_alpha=True,
            alpha_dist="dirichlet",
        ).rho_Z
        self.assertLess(
            identity_block_correlation_diagnostics(raw_m5)["min_eig"], 0.0
        )

    def test_generation_is_deterministic_and_admissible(self) -> None:
        for n_assets, m_states in ((1, 1), (2, 3), (5, 2), (30, 5)):
            for seed in (0, 7, 19):
                with self.subTest(n_assets=n_assets, m_states=m_states, seed=seed):
                    first = generate_joint_market_params(n_assets, m_states, seed=seed)
                    second = generate_joint_market_params(n_assets, m_states, seed=seed)
                    np.testing.assert_array_equal(
                        first.rho_canonical, second.rho_canonical
                    )
                    diagnostics = identity_block_correlation_diagnostics(
                        first.rho_canonical
                    )
                    joint = np.block([
                        [np.eye(n_assets), first.rho_canonical],
                        [first.rho_canonical.T, np.eye(m_states)],
                    ])
                    self.assertGreater(float(np.linalg.eigvalsh(joint)[0]), 0.0)
                    self.assertAlmostEqual(
                        float(np.linalg.eigvalsh(joint)[0]),
                        diagnostics["min_eig"],
                        places=12,
                    )

    def test_invalid_inputs_fail_without_silent_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive definite"):
            canonicalize_cross_correlation(
                np.array([[1.0, 1.0], [1.0, 1.0]]),
                np.zeros((2, 1)),
                np.eye(1),
            )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            canonicalize_cross_correlation(
                np.eye(2), np.zeros((3, 1)), np.eye(1)
            )
        with self.assertRaisesRegex(ValueError, "NaN"):
            identity_block_correlation_diagnostics(np.array([[np.nan]]))


class SnapshotConventionTests(unittest.TestCase):
    def _snapshot(self) -> dict[str, np.ndarray]:
        params = generate_joint_market_params(5, 3, seed=12)
        return {
            "rho": np.asarray(params.rho_canonical),
            **rho_snapshot_metadata(params),
        }

    def test_schema_two_snapshot_reconstructs_canonical_rho(self) -> None:
        values = self._snapshot()
        diagnostics = validate_rho_snapshot(
            values,
            expected_rho=values["rho"],
            require_canonical_metadata=True,
        )
        self.assertEqual(
            int(np.asarray(values["market_schema_version"]).reshape(-1)[0]),
            MARKET_SCHEMA_VERSION,
        )
        self.assertEqual(
            str(np.asarray(values["rho_convention"]).reshape(-1)[0]),
            RHO_CONVENTION,
        )
        self.assertGreater(diagnostics["min_eig"], 0.0)

    def test_metadata_tampering_is_detected(self) -> None:
        values = self._snapshot()
        values["rho"] = np.asarray(values["rho"]).copy()
        values["rho"][0, 0] += 1.0e-4
        with self.assertRaisesRegex(ValueError, "canonical whitening"):
            validate_rho_snapshot(values)

    def test_schema_and_scalar_metadata_are_strict(self) -> None:
        values = self._snapshot()
        values["market_schema_version"] = np.array([2.5])
        with self.assertRaisesRegex(ValueError, "schema version"):
            validate_rho_snapshot(values)
        values = self._snapshot()
        values["rho_spectral_norm"] = np.array([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "must be scalars"):
            validate_rho_snapshot(values)

    def test_asymmetric_source_block_is_rejected(self) -> None:
        values = self._snapshot()
        values["Psi"] = np.asarray(values["Psi"]).copy()
        values["Psi"][0, 1] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_rho_snapshot(values)

    def test_legacy_snapshot_is_readable_but_not_eval_only_compatible(self) -> None:
        legacy = {"rho": np.array([[0.1], [0.2]])}
        self.assertGreater(validate_rho_snapshot(legacy)["min_eig"], 0.0)
        with self.assertRaisesRegex(ValueError, "predates canonical rho"):
            validate_rho_snapshot(
                legacy,
                require_canonical_metadata=True,
            )

    def test_nonelliptic_legacy_snapshot_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-positive"):
            validate_rho_snapshot({"rho": np.array([[1.01]])})

    def test_full_market_validator_detects_derived_coefficient_tampering(self) -> None:
        params = generate_joint_market_params(5, 3, seed=12)
        sigma_x = params.SigmaZ
        rho = params.rho_canonical
        xbar = params.theta
        eta = params.eta
        market = {
            "K": params.K,
            "xbar": xbar,
            "SigmaX": sigma_x,
            "rho": rho,
            "Lam": params.alpha,
            "Q": sigma_x @ sigma_x.T,
            "Gamma": rho @ sigma_x.T,
            "k0": params.K @ xbar,
            "lam0": np.ones(5) * 0.1,
            "X_min": xbar - eta,
            "X_max": xbar + eta,
            "eta": eta,
            "gamma": np.array([2.0]),
            "r": np.array([0.03]),
            "tau_max": np.array([3.0]),
            "W_min": np.array([0.1]),
            "W_max": np.array([2.0]),
            "market_seed": np.array([12]),
            **rho_snapshot_metadata(params),
        }
        self.assertGreater(
            validate_market_snapshot(
                market, require_canonical_metadata=True
            )["min_eig"],
            0.0,
        )
        market["Gamma"] = np.asarray(market["Gamma"]).copy()
        market["Gamma"][0, 0] += 1.0e-4
        with self.assertRaisesRegex(ValueError, "Gamma"):
            validate_market_snapshot(market)


if __name__ == "__main__":
    unittest.main()
