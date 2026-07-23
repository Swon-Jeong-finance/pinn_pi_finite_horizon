import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_welfare as ew
from aggregate_seeds import MARKET_HASH_KEYS


def _make_run(
    root: Path,
    name: str,
    *,
    n_assets: int,
    epsilon: float,
    mode: str | None = None,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    args = {
        "model_type": "pinn",
        "n_assets": n_assets,
        "m_states": 1,
        "seed": 1,
        "market_seed": 12,
        "risk_premium_mode": mode or ("affine" if epsilon == 0.0 else "tanh"),
        "nonaffine_eps": epsilon,
        "run_tag": name,
        "output_root": str(run),
        "weight_root": str(run / "weights"),
    }
    (run / "config.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    (run / "_SUCCESS").touch()
    market = {key: np.asarray([1.0]) for key in MARKET_HASH_KEYS}
    np.savez(run / "market_params.npz", **market)
    return run


class WelfarePreflightTests(unittest.TestCase):
    def test_expected_asset_dimension_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_run(root, "wrong_n", n_assets=10, epsilon=0.0)
            with self.assertRaisesRegex(ValueError, "asset-dimension mismatch"):
                ew.discover_paper_runs(
                    root,
                    models=("pinn",),
                    m_states=(1,),
                    expected_seeds=(1,),
                    expected_n_assets=30,
                )

    def test_nonaffine_run_is_rejected_for_closed_form_welfare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_run(root, "eps1", n_assets=30, epsilon=1.0)
            with self.assertRaisesRegex(ValueError, "non-affine runs"):
                ew.discover_paper_runs(
                    root,
                    models=("pinn",),
                    m_states=(1,),
                    expected_seeds=(1,),
                    expected_n_assets=30,
                )

    def test_tiny_nonzero_epsilon_is_not_rounded_to_affine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_run(root, "tiny_eps", n_assets=30, epsilon=1.0e-16)
            with self.assertRaisesRegex(ValueError, "non-affine runs"):
                ew.discover_paper_runs(
                    root,
                    models=("pinn",),
                    m_states=(1,),
                    expected_seeds=(1,),
                    expected_n_assets=30,
                )

    def test_unknown_risk_premium_mode_is_rejected_even_at_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_run(root, "unknown", n_assets=30, epsilon=0.0, mode="mystery")
            with self.assertRaisesRegex(ValueError, "non-affine runs"):
                ew.discover_paper_runs(
                    root,
                    models=("pinn",),
                    m_states=(1,),
                    expected_seeds=(1,),
                    expected_n_assets=30,
                )

    def test_tanh_zero_matches_training_affine_reference_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _make_run(root, "tanh_zero", n_assets=30, epsilon=0.0, mode="tanh")
            selected = ew.discover_paper_runs(
                root,
                models=("pinn",),
                m_states=(1,),
                expected_seeds=(1,),
                expected_n_assets=30,
            )
            self.assertEqual(selected[("pinn", 1)][0].run_dir, run)

    def test_unrelated_nonaffine_run_does_not_poison_valid_affine_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = _make_run(root, "affine", n_assets=30, epsilon=0.0)
            _make_run(root, "eps1", n_assets=30, epsilon=1.0)
            selected = ew.discover_paper_runs(
                root,
                models=("pinn",),
                m_states=(1,),
                expected_seeds=(1,),
                expected_n_assets=30,
            )
            self.assertEqual(selected[("pinn", 1)][0].run_dir, affine)


if __name__ == "__main__":
    unittest.main()
