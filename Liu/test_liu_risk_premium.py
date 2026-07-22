from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from liu_risk_premium import (
        has_affine_reference,
        risk_premium_numpy,
        risk_premium_torch,
        validate_risk_premium_config,
    )
except ModuleNotFoundError:  # Allow `python -m unittest Liu/test_...py` from repo root.
    from Liu.liu_risk_premium import (
        has_affine_reference,
        risk_premium_numpy,
        risk_premium_torch,
        validate_risk_premium_config,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TorchRiskPremiumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64
        self.lam0 = torch.tensor([0.1, -0.2], dtype=self.dtype)
        self.Lam = torch.tensor([[0.3, -0.1], [0.2, 0.4]], dtype=self.dtype)
        self.xbar = torch.tensor([0.25, -0.5], dtype=self.dtype)
        self.scale = torch.tensor([0.5, 0.25], dtype=self.dtype)

    def evaluate(self, x: torch.Tensor, *, mode: str, eps: float) -> torch.Tensor:
        return risk_premium_torch(
            x,
            self.lam0,
            self.Lam,
            mode=mode,
            eps=eps,
            xbar=self.xbar,
            state_scale=self.scale,
            loading_scale=1.0,
        )

    def test_tanh_eps_zero_equals_affine(self) -> None:
        x = torch.tensor([[0.1, -0.2], [0.6, -0.8]], dtype=self.dtype)
        self.assertTrue(torch.equal(
            self.evaluate(x, mode="affine", eps=0.0),
            self.evaluate(x, mode="tanh", eps=0.0),
        ))
        self.assertTrue(has_affine_reference("tanh", 0.0))

    def test_perturbation_vanishes_at_xbar(self) -> None:
        affine = self.evaluate(self.xbar, mode="affine", eps=0.0)
        for eps in (0.1, 1.0, 5.0):
            self.assertTrue(torch.allclose(
                affine, self.evaluate(self.xbar, mode="tanh", eps=eps)
            ))

    def test_jacobian_at_xbar(self) -> None:
        eps = 1.7
        x = self.xbar.detach().clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(
            lambda z: self.evaluate(z, mode="tanh", eps=eps), x
        )
        expected = self.Lam + eps * self.Lam / self.scale.unsqueeze(0)
        self.assertTrue(torch.allclose(jac, expected, rtol=1e-12, atol=1e-12))

    def test_numpy_torch_parity_for_paper_shapes(self) -> None:
        rng = np.random.default_rng(123)
        for m in (1, 3):
            n = 30
            x = rng.normal(size=(7, m))
            lam0 = rng.normal(size=n)
            Lam = rng.normal(size=(n, m))
            xbar = rng.normal(size=m)
            scale = rng.uniform(0.1, 1.0, size=m)
            got_np = risk_premium_numpy(
                x, lam0, Lam, mode="tanh", eps=2.0,
                xbar=xbar, state_scale=scale, loading_scale=0.75,
            )
            got_t = risk_premium_torch(
                torch.tensor(x, dtype=self.dtype),
                torch.tensor(lam0, dtype=self.dtype),
                torch.tensor(Lam, dtype=self.dtype),
                mode="tanh", eps=2.0,
                xbar=torch.tensor(xbar, dtype=self.dtype),
                state_scale=torch.tensor(scale, dtype=self.dtype),
                loading_scale=0.75,
            ).numpy()
            self.assertEqual(got_np.shape, (7, n))
            np.testing.assert_allclose(got_np, got_t, rtol=1e-12, atol=1e-12)

    def test_invalid_config_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            validate_risk_premium_config("affine", 0.1, 1.0)
        with self.assertRaises(ValueError):
            validate_risk_premium_config("tanh", -0.1, 1.0)
        with self.assertRaises(ValueError):
            validate_risk_premium_config("tanh", 0.1, 1.0, [1.0, 0.0])


class NumpyRiskPremiumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lam0 = np.array([0.1, -0.2])
        self.Lam = np.array([[0.3, -0.1], [0.2, 0.4]])
        self.xbar = np.array([0.25, -0.5])
        self.scale = np.array([0.5, 0.25])

    def evaluate(self, x: np.ndarray, *, mode: str, eps: float) -> np.ndarray:
        return risk_premium_numpy(
            x, self.lam0, self.Lam,
            mode=mode, eps=eps, xbar=self.xbar,
            state_scale=self.scale, loading_scale=1.0,
        )

    def test_tanh_eps_zero_equals_affine_exactly(self) -> None:
        x = np.array([[0.1, -0.2], [0.6, -0.8]])
        np.testing.assert_array_equal(
            self.evaluate(x, mode="affine", eps=0.0),
            self.evaluate(x, mode="tanh", eps=0.0),
        )

    def test_perturbation_vanishes_at_xbar(self) -> None:
        affine = self.evaluate(self.xbar, mode="affine", eps=0.0)
        for eps in (0.1, 1.0, 5.0):
            np.testing.assert_allclose(
                self.evaluate(self.xbar, mode="tanh", eps=eps), affine,
                rtol=0.0, atol=0.0,
            )

    def test_jacobian_at_xbar(self) -> None:
        eps = 1.7
        step = 1.0e-6
        jacobian = np.column_stack([
            (
                self.evaluate(self.xbar + step * np.eye(2)[j], mode="tanh", eps=eps)
                - self.evaluate(self.xbar - step * np.eye(2)[j], mode="tanh", eps=eps)
            ) / (2.0 * step)
            for j in range(2)
        ])
        expected = self.Lam + eps * self.Lam / self.scale[None, :]
        np.testing.assert_allclose(jacobian, expected, rtol=1.0e-9, atol=1.0e-9)

    def test_paper_shapes(self) -> None:
        rng = np.random.default_rng(123)
        for m in (1, 3):
            n = 30
            result = risk_premium_numpy(
                rng.normal(size=(7, m)), rng.normal(size=n), rng.normal(size=(n, m)),
                mode="tanh", eps=2.0, xbar=rng.normal(size=m),
                state_scale=rng.uniform(0.1, 1.0, size=m), loading_scale=0.75,
            )
            self.assertEqual(result.shape, (7, n))
            self.assertTrue(np.all(np.isfinite(result)))

    def test_invalid_config_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            validate_risk_premium_config("affine", 0.1, 1.0)
        with self.assertRaises(ValueError):
            validate_risk_premium_config("tanh", -0.1, 1.0)
        with self.assertRaises(ValueError):
            validate_risk_premium_config("tanh", 0.1, 1.0, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
