"""
Multi-Asset Merton Portfolio (with Consumption) - PI-PINN (FOC-based) with log-wealth transform
============================================================================================

Goal
----
Extend your *log-wealth* PI-PINN implementation (single-asset, with consumption) to the
multi-asset Merton setting with N risky assets.

State variable
--------------
Wealth W only (still 1D). We use y = log(W).

Controls
--------
- c(t,W)  : consumption (scalar)
- pi(t,W) : portfolio weights in risky assets (N-vector)

HJB (in W)
----------
    rho V = V_t + max_{c,pi} { U(c)
             + (W(r + pi^T mu_excess) - c) V_W
             + 0.5 * W^2 * (pi^T Sigma pi) V_WW }

FOC
---
    c*  = V_W^{-1/gamma}
    pi* = -(V_W / (W V_WW)) * Sigma^{-1} mu_excess

Log-wealth transform (y=log W)
------------------------------
Let \tilde V(t,y) = V(t, e^y). Then
    V_W  = (1/W) V_y
    V_WW = (1/W^2)(V_yy - V_y)

So the PDE becomes (linear under fixed controls):
    0 = rho V - V_t - U(c)
        - (r + pi^T mu_excess - c/W) V_y
        - 0.5 * (pi^T Sigma pi) (V_yy - V_y)

FOC in (t,y):
    pi* = -(V_y / (V_yy - V_y)) * Sigma^{-1} mu_excess
    c*  from V_W = V_y/W.

Market parameters
-----------------
We keep your request: `import market_setup` is used 그대로.

Notes
-----
- This is the *direct* multi-asset analogue of your single-asset log-W solver.
- We keep the same stability tricks: kappa=c/W bounds + optional eta/shape penalties.
"""

import os
import sys
import time
import copy
import math
import shutil
import json
import hashlib
import argparse
import numpy as np
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import market_setup  # <-- keep as requested
import merton_experiment_utils as mxu
import merton_evaluation_metrics as mem


# =============================================================================
# Command-line configuration (Merton PI-PINN). --seed drives network/
# collocation/optimizer; --market-seed drives ONLY the market draw so a
# training-seed sweep reuses the same market. Module-level constants below are
# populated from ARGS (minimal-diff refactor; the global-reference structure
# of the original script is preserved).
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-asset Merton (with consumption) PI-PINN [log-wealth].")
    # Reproducibility / device
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--market-seed", type=int, default=12)
    p.add_argument("--device", type=str, default=None)
    # Market generation
    p.add_argument("--n-assets", type=int, default=50)
    p.add_argument("--sigma-lo", type=float, default=0.10)
    p.add_argument("--sigma-hi", type=float, default=0.25)
    p.add_argument("--rho-max", type=float, default=1.0)
    p.add_argument("--kappa-max", type=float, default=30.0)
    p.add_argument("--delta-rel", type=float, default=1e-4)
    p.add_argument("--pi-scale", type=float, default=0.6)
    p.add_argument("--mu-noise-rel", type=float, default=0.02)
    p.add_argument("--mu-mode", type=str, default="pi_target", choices=["pi_target", "sharpe"])
    # Preferences / market scalars
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho-discount", type=float, default=0.04)
    p.add_argument("--r", type=float, default=0.03)
    p.add_argument("--epsilon-bequest", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=1.0, help="Horizon T.")
    # Domain
    p.add_argument("--w-min", type=float, default=0.1)
    p.add_argument("--w-max", type=float, default=2.0)
    # Control bounds (PI-PINN-specific).  The canonical symmetric interface
    # mirrors Liu's theta_clip_abs: 2.0 means [-2,2], ``none`` is genuinely
    # unconstrained for the portfolio coordinate.
    p.add_argument("--pi-clip-abs", type=mxu.none_or_float, default=2.0)
    p.add_argument(
        "--policy-bounds-mode", type=str, default="stabilized",
        choices=["stabilized", "none"],
        help=(
            "stabilized applies the recorded portfolio/kappa/consumption projections; "
            "none disables every finite action projection (FOC sign guards remain)."
        ),
    )
    p.add_argument("--kappa-max-bound", type=float, default=3.0,
                   help="Upper bound on kappa = c/W.")
    p.add_argument("--utility-cap", type=float, default=1e3,
                   help="M in the CRRA floor c_floor = ((gamma-1) M)^(-1/(gamma-1)).")
    # Network / optimization
    p.add_argument("--value-hidden", type=int, default=256)
    p.add_argument("--value-depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--terminal-frac", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--outer-iters", type=int, default=500)
    p.add_argument("--eval-epochs", type=int, default=200)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    p.add_argument("--lr-schedule", type=str, default="carry_plateau",
                   choices=["inner_plateau", "fixed", "carry_plateau"])
    p.add_argument("--adam-reset", type=str, default="keep", choices=["keep", "full"])
    p.add_argument("--carry-lr-min", type=float, default=1e-5)
    p.add_argument("--carry-lr-max", type=float, default=5e-4)
    # Policy init
    p.add_argument("--pi-init-method", type=str, default="myopic")
    p.add_argument("--pi-init-scale", type=float, default=1.0,
                   help="Positive multiplier a in pi_0=a*pi_myopic.")
    p.add_argument("--c-init-method", type=str, default="proportional")
    # Loss weights
    p.add_argument("--w-terminal", type=float, default=10.0)
    p.add_argument("--w-shape", type=float, default=1.0)
    p.add_argument("--w-eta", type=float, default=3.0)
    p.add_argument("--eta-clip", type=str, default="10.0",
                   help="Optional |.|-clip for the eta penalty (none = off).")
    p.add_argument("--eta-focus-w", type=str, default="none",
                   help="Optional wealth focus for the eta penalty (none = off).")
    # Held-out residual / inner-selection protocol (same roles as Liu).
    p.add_argument("--pres-target", type=mxu.none_or_float, default=None)
    p.add_argument("--val-points", type=int, default=100000)
    p.add_argument("--val-terminal-points", type=int, default=10000)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--inner-best-restore", type=int, default=1, choices=[0, 1])
    p.add_argument("--sel-points", type=int, default=10000)
    p.add_argument("--sel-terminal-points", type=int, default=2000)
    p.add_argument("--sel-every", type=int, default=50)
    p.add_argument("--sel-patience", type=int, default=6)
    p.add_argument("--pe-resample-every", type=int, default=0,
                   help="Within-frozen-PDE batch refresh; 0 keeps one inner batch.")
    # E6 common-warm-up protocol.  A warm-up run produces v~_0 from the
    # analytic initial policy with p_res <= 1.  Every target branch then starts
    # from that identical seed-specific model+Adam+RNG state and records only
    # the target-dependent policy evaluations n >= 1.
    p.add_argument(
        "--e6-role",
        type=str,
        default="standard",
        choices=["standard", "warmup", "target_branch"],
    )
    p.add_argument(
        "--e6-warm-start",
        type=str,
        default=None,
        help="Warm-up bundle to restore for --e6-role target_branch.",
    )
    p.add_argument(
        "--e6-warmup-bundle",
        type=str,
        default=None,
        help="Output bundle written by --e6-role warmup.",
    )
    # Checkpoints / diagnostics.
    p.add_argument("--save-iterate-every", type=int, default=0)
    p.add_argument("--e3b-checkpoints", action="store_true")
    p.add_argument("--diag-points", type=int, default=4096)
    p.add_argument("--diag-every", type=int, default=1)
    # Evaluation window(s): only the log-wealth axis is shrunk.  Time keeps
    # the full [0,T) range, matching Q_ev=(0,T)xOmega_ev.
    p.add_argument("--eval-margin", type=str, default="0.10,0.0,0.05,0.15,0.20")
    p.add_argument(
        "--eval-w-min",
        type=mxu.none_or_float,
        default=None,
        help=(
            "Optional one-sided lower wealth endpoint for final/eval-only "
            "evaluation. eval-margin still determines the upper endpoint."
        ),
    )
    p.add_argument("--test-points", type=int, default=100000)
    p.add_argument("--n-tau", type=int, default=100)
    p.add_argument("--n-x", type=int, default=100)
    # Logging / output
    p.add_argument(
        "--print-every-outer",
        type=int,
        default=10,
        help="Periodic outer-loop logging interval; 0 disables it after the first three outers.",
    )
    p.add_argument("--print-every-eval", type=int, default=0)
    p.add_argument("--output-root", type=str, default="outputs_merton")
    p.add_argument("--weight-root", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="merton_pipinn")
    p.add_argument("--model-type", type=str, default="pipinn", choices=["pinn", "pipinn"])
    # Merton state dimension is 1 (wealth only). Recorded as m_states so the
    # shared make_figure2_contraction reader (which filters on m_states) selects
    # Merton runs with --merton-m-states 1. Not a tunable; fixed by the problem.
    p.add_argument("--m-states", type=int, default=1,
                   help="Fixed = 1 for Merton (wealth-only state); used for Figure-2 selection.")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-plots", action="store_true",
                   help="Back-compat alias for --skip-figures.")
    # Infrastructure (wired with the recorder)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--allow-legacy-best-eval", action="store_true",
                   help="Opt in to diagnostic-best fallback when final/last checkpoints are absent.")
    p.add_argument("--timing-mode", action="store_true")
    p.add_argument("--stop-flag-path", type=str, default=None)
    p.add_argument("--pde-stop-threshold", type=float, default=None)
    p.add_argument("--pde-stop-start-outer", type=int, default=0)
    p.add_argument("--pde-stop-patience", type=int, default=1)
    return p


ARGS = build_arg_parser().parse_args()

if ARGS.m_states != 1:
    raise ValueError("Merton has one PDE state (log wealth): --m-states must equal 1")
if ARGS.pi_init_scale <= 0.0 or not math.isfinite(ARGS.pi_init_scale):
    raise ValueError("--pi-init-scale must be finite and strictly positive")
if not 0.0 < ARGS.terminal_frac:
    raise ValueError("--terminal-frac must be strictly positive")
if not (0.0 < ARGS.carry_lr_min <= ARGS.carry_lr_max):
    raise ValueError("require 0 < carry_lr_min <= carry_lr_max")
if ARGS.scheduler_min_lr <= 0.0:
    raise ValueError("--scheduler-min-lr must be strictly positive")
if ARGS.val_every < 1 or ARGS.sel_every < 1:
    raise ValueError("--val-every and --sel-every must be positive")
if ARGS.val_points < 0 or ARGS.val_terminal_points < 0:
    raise ValueError("validation point counts must be nonnegative")
if ARGS.sel_points < 0 or ARGS.sel_terminal_points < 0:
    raise ValueError("selection point counts must be nonnegative")
if ARGS.sel_patience < 0 or ARGS.pe_resample_every < 0:
    raise ValueError("--sel-patience and --pe-resample-every must be nonnegative")
if ARGS.diag_points < 0 or ARGS.diag_every < 1:
    raise ValueError("require --diag-points >= 0 and --diag-every >= 1")
if ARGS.print_every_outer < 0:
    raise ValueError("--print-every-outer must be nonnegative")
if ARGS.eval_epochs < 0 or ARGS.outer_iters < 1 or ARGS.batch_size < 1:
    raise ValueError("require eval_epochs >= 0, outer_iters >= 1, and batch_size >= 1")
if ARGS.e3b_checkpoints and ARGS.timing_mode:
    raise ValueError("--e3b-checkpoints is incompatible with --timing-mode")
if not math.isfinite(ARGS.utility_cap) or ARGS.utility_cap <= 0.0:
    raise ValueError("--utility-cap must be finite and positive")
if (ARGS.policy_bounds_mode == "stabilized" and
        (not math.isfinite(ARGS.kappa_max_bound) or ARGS.kappa_max_bound <= 0.0)):
    raise ValueError("--kappa-max-bound must be finite and positive")
if ARGS.pi_clip_abs is not None and (
        not math.isfinite(ARGS.pi_clip_abs) or ARGS.pi_clip_abs <= 0.0):
    raise ValueError("--pi-clip-abs must be finite and positive, or none")
if ARGS.e6_role == "standard":
    if ARGS.e6_warm_start or ARGS.e6_warmup_bundle:
        raise ValueError(
            "--e6-warm-start/--e6-warmup-bundle require a non-standard --e6-role"
        )
elif ARGS.e6_role == "warmup":
    if ARGS.e6_warm_start:
        raise ValueError("--e6-role warmup cannot use --e6-warm-start")
    if not ARGS.e6_warmup_bundle:
        raise ValueError("--e6-role warmup requires --e6-warmup-bundle")
    if ARGS.pres_target is None or not math.isclose(
            float(ARGS.pres_target), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("--e6-role warmup requires --pres-target 1")
    if int(ARGS.outer_iters) != 1:
        raise ValueError("--e6-role warmup requires --outer-iters 1")
    if ARGS.eval_only or ARGS.timing_mode:
        raise ValueError("--e6-role warmup is incompatible with eval/timing mode")
else:
    if not ARGS.eval_only and not ARGS.e6_warm_start:
        raise ValueError("--e6-role target_branch requires --e6-warm-start")
    if ARGS.e6_warmup_bundle:
        raise ValueError(
            "--e6-role target_branch cannot write --e6-warmup-bundle")
    if ARGS.pres_target is None:
        raise ValueError("--e6-role target_branch requires --pres-target")
    if ARGS.timing_mode:
        raise ValueError(
            "--e6-role target_branch is incompatible with timing mode")


E6_WARMUP_BUNDLE_SCHEMA_VERSION = 1
E6_WARMUP_BUNDLE_KIND = "merton-pipinn-e6-common-warmup-v1"
E6_WARM_START_PROTOCOL = "merton-e6-common-warm-start-v1"
E6_WARM_START_POLICY_SOURCE = "warm_start_value_net"

# Only state-transition settings belong to this compatibility contract.
# Target, branch length, persistence, plotting, diagnostics, paths, and device
# index may differ without changing the warm-started optimization problem.
_E6_COMPAT_IGNORE = {
    "pres_target",
    "outer_iters",
    "run_tag",
    "device",
    "output_root",
    "weight_root",
    "eval_only",
    "allow_legacy_best_eval",
    "timing_mode",
    "skip_figures",
    "skip_figures_requested",
    "skip_eval",
    "skip_plots",
    "print_every",
    "print_every_outer",
    "print_every_eval",
    "verbose_detail",
    "save_iterate_every",
    "e3b_checkpoints",
    "diag_points",
    "diag_every",
    "eval_margin",
    "eval_w_min",
    "test_points",
    "n_tau",
    "n_x",
    "stop_flag_path",
    "pde_stop_threshold",
    "pde_stop_start_outer",
    "pde_stop_patience",
}


def _e6_compatibility_payload(args: argparse.Namespace) -> Dict:
    payload = {}
    for key, value in vars(args).items():
        if key in _E6_COMPAT_IGNORE or key.startswith("e6_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        payload[key] = value
    return {key: payload[key] for key in sorted(payload)}


def _canonical_json_sha256(payload: Dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=mxu.json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_tree_sha256(value) -> str:
    """Hash nested optimizer/RNG state independently of torch serialization."""
    digest = hashlib.sha256()

    def emit(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, byteorder="big", signed=False))
        digest.update(data)

    def visit(item) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            emit(b"torch.Tensor")
            emit(str(tensor.dtype).encode("utf-8"))
            emit(json.dumps(list(tensor.shape)).encode("ascii"))
            emit(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            emit(b"numpy.ndarray")
            emit(array.dtype.str.encode("ascii"))
            emit(json.dumps(list(array.shape)).encode("ascii"))
            emit(array.tobytes(order="C"))
            return
        if isinstance(item, np.generic):
            visit(item.item())
            return
        if isinstance(item, dict):
            emit(b"dict")
            ordered = sorted(
                item.items(),
                key=lambda pair: (
                    type(pair[0]).__module__,
                    type(pair[0]).__qualname__,
                    repr(pair[0]),
                ),
            )
            emit(str(len(ordered)).encode("ascii"))
            for key, child in ordered:
                visit(key)
                visit(child)
            return
        if isinstance(item, list):
            emit(b"list")
            emit(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            return
        if isinstance(item, tuple):
            emit(b"tuple")
            emit(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
            return
        if isinstance(item, bytes):
            emit(b"bytes")
            emit(item)
            return
        if item is None:
            emit(b"none")
            return
        if isinstance(item, (bool, int, float, str)):
            emit(
                (
                    f"{type(item).__module__}.{type(item).__qualname__}:"
                    f"{repr(item)}"
                ).encode("utf-8")
            )
            return
        raise TypeError(
            "unsupported value in canonical optimizer/RNG hash: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    visit(value)
    return digest.hexdigest()


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _torch_save_atomic(payload: Dict, path: str) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _archive_e6_warmup_bundle(path: str) -> Optional[str]:
    """Move an old bundle aside before a new warm-up attempt starts."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    archived = f"{path}.old.{time.time_ns()}.{os.getpid()}"
    os.replace(path, archived)
    print(f"[E6] previous warm-up bundle archived: {archived}")
    return archived


def _load_e6_warmup_bundle(path: str) -> Dict:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"E6 warm-up bundle not found: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("E6 warm-up bundle must be a mapping")
    if int(bundle.get("schema_version", -1)) != E6_WARMUP_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported E6 warm-up schema: {bundle.get('schema_version')!r}")
    if bundle.get("kind") != E6_WARMUP_BUNDLE_KIND:
        raise ValueError(f"unexpected E6 warm-up bundle kind: {bundle.get('kind')!r}")
    if bundle.get("warm_start_protocol") != E6_WARM_START_PROTOCOL:
        raise ValueError(
            "unexpected E6 warm-start protocol: "
            f"{bundle.get('warm_start_protocol')!r}"
        )
    if not str(bundle.get("warm_start_source", "")).strip():
        raise ValueError("E6 warm-up bundle is missing warm_start_source")
    if not isinstance(bundle.get("model_state"), dict):
        raise ValueError("E6 warm-up bundle is missing model_state")
    if not isinstance(bundle.get("optimizer_state"), dict):
        raise ValueError("E6 warm-up bundle is missing optimizer_state")
    if int(bundle.get("outer_count", -1)) != 1:
        raise ValueError("E6 warm-up bundle must end after exactly one outer solve")
    warmup_target = float(bundle.get("warmup_pres_target", float("nan")))
    if not math.isclose(warmup_target, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "E6 warm-up bundle must use warmup_pres_target=1.0, got "
            f"{warmup_target!r}"
        )
    achieved = float(bundle.get("warmup_achieved_pres", float("inf")))
    if (
        not math.isfinite(achieved)
        or achieved <= 0.0
        or achieved > 1.0 * (1.0 + 1e-9)
    ):
        raise ValueError(
            "E6 warm-up bundle is not admissible: "
            f"post-restore p_res={achieved!r} must lie in (0,1]")
    current_lrs = bundle.get("current_lrs")
    if (
        not isinstance(current_lrs, (list, tuple))
        or not current_lrs
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in current_lrs
        )
    ):
        raise ValueError("E6 warm-up bundle has invalid optimizer LR state")
    expected_compat = _e6_compatibility_payload(ARGS)
    expected_hash = _canonical_json_sha256(expected_compat)
    if bundle.get("compatibility_sha256") != expected_hash:
        stored = bundle.get("compatibility", {})
        differing = sorted({
            key for key in set(stored) | set(expected_compat)
            if stored.get(key) != expected_compat.get(key)
        })
        raise ValueError(
            "E6 warm-up bundle is incompatible with this target branch; "
            f"differing settings={differing}")
    actual_model_hash = mxu.canonical_state_dict_sha256(bundle["model_state"])
    if bundle.get("model_state_sha256") != actual_model_hash:
        raise ValueError("E6 warm-up model-state hash mismatch")
    actual_optimizer_hash = _canonical_tree_sha256(bundle["optimizer_state"])
    if bundle.get("optimizer_state_sha256") != actual_optimizer_hash:
        raise ValueError("E6 warm-up optimizer-state hash mismatch")
    rng_payload = {
        "torch_cpu_rng_state": bundle.get("torch_cpu_rng_state"),
        "torch_cuda_rng_state": bundle.get("torch_cuda_rng_state"),
        "numpy_rng_state": bundle.get("numpy_rng_state"),
        "rng_device_type": bundle.get("rng_device_type"),
    }
    actual_rng_hash = _canonical_tree_sha256(rng_payload)
    if bundle.get("rng_state_sha256") != actual_rng_hash:
        raise ValueError("E6 warm-up RNG-state hash mismatch")
    expected_warm_start_id = _canonical_json_sha256({
        "protocol": E6_WARM_START_PROTOCOL,
        "compatibility_sha256": bundle["compatibility_sha256"],
        "seed": int(bundle["seed"]),
        "market_seed": int(bundle["market_seed"]),
        "model_sha256": actual_model_hash,
        "optimizer_sha256": actual_optimizer_hash,
        "rng_sha256": actual_rng_hash,
    })
    if bundle.get("warm_start_id") != expected_warm_start_id:
        raise ValueError("E6 warm-up ID does not match the bundled state")
    if bundle.get("trainer_source_sha256") != TRAINER_METADATA["trainer_source_sha256"]:
        raise ValueError(
            "E6 warm-up trainer source differs from the target-branch trainer")
    stored_device_type = str(bundle.get("rng_device_type", ""))
    if stored_device_type != device.type:
        raise ValueError(
            "E6 warm-up and target branch must use the same device type for "
            f"exact RNG continuation: stored={stored_device_type!r}, "
            f"current={device.type!r}"
        )
    return bundle


# Q_res is fixed for the entire run.  Q_sel is instead regenerated once per
# frozen PDE and then held fixed throughout that inner policy-evaluation solve.
# Keeping outer 1 on the historical stream (market_seed * 7919 + 101) makes the
# protocol change minimally disruptive while giving every later outer its own
# deterministic, training-seed-independent selection set.
PI_PINN_QSEL_MARKET_MULTIPLIER = 7_919
PI_PINN_QSEL_SEED_OFFSET = 101
PI_PINN_QSEL_OUTER_STRIDE = 104_729
PI_PINN_QSEL_SEED_MODULUS = 2**63 - 1


def qsel_seed_for_outer(market_seed: int, outer_iter: int) -> int:
    """Deterministic seed for the outer-specific, inner-fixed Q_sel set."""
    outer_iter = int(outer_iter)
    if outer_iter < 1:
        raise ValueError("outer_iter must use one-based indexing")
    return int(
        (
            int(market_seed) * PI_PINN_QSEL_MARKET_MULTIPLIER
            + PI_PINN_QSEL_SEED_OFFSET
            + (outer_iter - 1) * PI_PINN_QSEL_OUTER_STRIDE
        )
        % PI_PINN_QSEL_SEED_MODULUS
    )


def should_print_outer(outer_iter: int, print_every_outer: int) -> bool:
    """Return whether the PI outer iteration should emit its summary."""
    outer_iter = int(outer_iter)
    print_every_outer = int(print_every_outer)
    if outer_iter < 1:
        raise ValueError("outer_iter must use one-based indexing")
    if print_every_outer < 0:
        raise ValueError("print_every_outer must be nonnegative")
    return bool(
        outer_iter <= 3
        or (
            print_every_outer > 0
            and outer_iter % print_every_outer == 0
        )
    )


# =============================================================================
# 0) Reproducibility + Device
# =============================================================================
SEED = int(ARGS.seed)
MARKET_SEED = int(ARGS.market_seed)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Cap intra/inter-op CPU threads (parallel sweep workers otherwise
# oversubscribe cores); TORCH_NUM_THREADS is exported by the tune script.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if ARGS.device is not None:
    device = torch.device(ARGS.device)
else:
    GPU_ID = int(os.environ.get("GPU_ID", "0"))
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device=device)

# =============================================================================
# 1) Problem Parameters
# =============================================================================
T_FINAL = float(ARGS.tau_max)
t_min, t_max = 0.0, T_FINAL

# Wealth domain (W)
x_min, x_max = float(ARGS.w_min), float(ARGS.w_max)
# Alias kept so recorder/eval helpers can use w_min/w_max like the PINN.
w_min, w_max = x_min, x_max

# Log-wealth domain (y = log W)
y_min, y_max = float(np.log(x_min)), float(np.log(x_max))

# Preferences
gamma_risk = float(ARGS.gamma)
rho_discount = float(ARGS.rho_discount)
epsilon = float(ARGS.epsilon_bequest)  # bequest weight

# Risk-free rate
r_rate = float(ARGS.r)

# Multi-asset market dimension
N_ASSETS = int(ARGS.n_assets)

# Synthetic market configuration. Drawn from MARKET_SEED (not SEED) so a
# training-seed sweep reuses the SAME market.
market_params = market_setup.generate_synthetic_merton_market(
    n=N_ASSETS,
    gamma=gamma_risk,
    sigma_range=(float(ARGS.sigma_lo), float(ARGS.sigma_hi)),
    rho_max=float(ARGS.rho_max),
    kappa_max=float(ARGS.kappa_max),
    delta_rel=float(ARGS.delta_rel),
    seed=MARKET_SEED,
    mu_mode=str(ARGS.mu_mode),
    pi_scale=float(ARGS.pi_scale),
    mu_noise_rel=float(ARGS.mu_noise_rel),
)

mu_excess_np = np.asarray(market_params["mu_excess"], dtype=np.float64).reshape(N_ASSETS)
Sigma_np = np.asarray(market_params["Sigma_safe"], dtype=np.float64).reshape(N_ASSETS, N_ASSETS)
chol_Sigma_np = np.asarray(market_params["L"], dtype=np.float64).reshape(N_ASSETS, N_ASSETS)

# Closed-form myopic/Merton portfolio (constant)
pi_star_np = np.asarray(market_params["pi_star"], dtype=np.float64).reshape(N_ASSETS)

# Precompute Sigma^{-1} mu_excess (stable solve, no explicit inverse)
Sigma_inv_mu_np = market_setup.cholesky_solve(chol_Sigma_np, mu_excess_np)

# Theta = mu^T Sigma^{-1} mu
Theta = float(market_params["mu_SigmaInv_mu"])

# nu for closed-form consumption ratio (multi-asset: replace (lam^2/sigma^2) by Theta)
nu = rho_discount / gamma_risk - (1.0 - gamma_risk) * (
    Theta / (2.0 * (gamma_risk**2)) + r_rate / gamma_risk
)

# A single audit switch can remove every finite action projection.  The
# one-sided derivative guards remain part of G and their activation continues
# to be recorded separately.
policy_bounds_mode = str(ARGS.policy_bounds_mode)
pi_clip_abs = None if policy_bounds_mode == "none" else ARGS.pi_clip_abs
pi_min_bound = -float(pi_clip_abs) if pi_clip_abs is not None else None
pi_max_bound = float(pi_clip_abs) if pi_clip_abs is not None else None

# Consumption bounds: use kappa=c/W bounds to avoid CRRA blow-up at tiny c
M_utility_cap = float(ARGS.utility_cap)
c_floor = ((gamma_risk - 1.0) * M_utility_cap) ** (-1.0 / (gamma_risk - 1.0))
kappa_min_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else c_floor / x_min
)
kappa_max_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else float(ARGS.kappa_max_bound)
)

# Optional level clamp for c (mainly for printing / extra safety)
c_min_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else c_floor
)
c_max_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else x_max
)


def _clamp_optional(
    value: torch.Tensor,
    minimum: Optional[float],
    maximum: Optional[float],
) -> torch.Tensor:
    """Apply only the finite action bounds that are actually configured."""
    if minimum is None and maximum is None:
        return value
    if minimum is None:
        return torch.clamp(value, max=float(maximum))
    if maximum is None:
        return torch.clamp(value, min=float(minimum))
    return torch.clamp(value, min=float(minimum), max=float(maximum))

# Torch constants
mu_excess = torch.tensor(mu_excess_np, device=device, dtype=torch.float32)          # (N,)
Sigma = torch.tensor(Sigma_np, device=device, dtype=torch.float32)                 # (N,N)
Sigma_inv_mu = torch.tensor(Sigma_inv_mu_np, device=device, dtype=torch.float32)   # (N,)
pi_star = torch.tensor(pi_star_np, device=device, dtype=torch.float32)             # (N,)

INITIAL_DIFFUSION_VARIANCE_TOL = 1e-12
_init_method = str(ARGS.pi_init_method).lower()
_initial_pi_for_audit: Optional[np.ndarray]
if _init_method == "myopic":
    _initial_pi_for_audit = float(ARGS.pi_init_scale) * pi_star_np
elif _init_method == "zero":
    _initial_pi_for_audit = np.zeros_like(pi_star_np)
else:
    _initial_pi_for_audit = None
if _initial_pi_for_audit is not None and pi_clip_abs is not None:
    _initial_pi_for_audit = np.clip(
        _initial_pi_for_audit, -float(pi_clip_abs), float(pi_clip_abs)
    )
if _initial_pi_for_audit is None:
    initial_policy_diffusion_variance = float("nan")
    initial_policy_degenerate = None
else:
    initial_policy_diffusion_variance = float(
        _initial_pi_for_audit @ Sigma_np @ _initial_pi_for_audit
    )
    initial_policy_degenerate = bool(
        initial_policy_diffusion_variance <= INITIAL_DIFFUSION_VARIANCE_TOL
    )
ARGS.initial_policy_diffusion_variance_analytic = (
    None if not math.isfinite(initial_policy_diffusion_variance)
    else initial_policy_diffusion_variance
)
ARGS.initial_policy_degenerate = initial_policy_degenerate
ARGS.initial_policy_degeneracy_tolerance = INITIAL_DIFFUSION_VARIANCE_TOL

# Machine-readable trainer contract consumed by the exact-map evaluator.  Keep
# this description literal: the current G implementation clamps both V_y and
# V_y-V_yy from below, so it is deliberately not labelled as an unguarded or
# log-concavity-only map.
TRAINER_METADATA = {
    "trainer_protocol": "merton-pipinn-heldout-selection-v2",
    "trainer_protocol_version": 2,
    "inner_selection_restore_contract": (
        "heldout-qsel-best-model-plus-optimizer;when-pres-target-is-set-"
        "only-same-state-qres-eligible-checkpoints"
    ),
    "carry_lr_restore_contract": (
        "restore-selected-model-and-adam-state;"
        "lr_carry=max(effective_floor,min(lr_best,lr_inner_end));"
        "ordinary-next-outer-carries-restored-inner-lr;"
        "pres-target-next-outer-restarts-at-carry-lr-max"
    ),
    "pres_target_lr_restart_contract": (
        "when-pres-target-is-set-and-lr-schedule-is-carry-plateau,"
        "every-outer-starts-at-carry-lr-max"
    ),
    "scheduler_reset_contract": (
        "within-frozen-pde-reduce-on-plateau-state-is-recreated-at-every-outer"
    ),
    "checkpoint_timing_contract": "post-policy-evaluation-after-optional-heldout-restore",
    "q_res_role": "pres_target_and_official_post_restore_residual",
    "q_res_seed": int(MARKET_SEED),
    "q_res_lifetime": "run-fixed",
    "q_sel_role": "inner_checkpoint_selection_and_carry_plateau_scheduler",
    "q_sel_lifetime": "outer-specific-and-inner-fixed",
    "q_sel_seed_source": "market_seed_and_one_based_outer_iter",
    "q_sel_seed_formula": (
        "(market_seed*7919+101+(outer_iter-1)*104729) mod (2^63-1)"
    ),
    "trainer_source": "merton_ND/merton_nd_consumption_pi_pinn.py",
    "trainer_source_marker": "merton-pipinn-logw-trainer-one-sided-selection-v2",
    "network_time_coordinate": "t",
    "network_input_order": "t,y",
    "network_input_transform": "identity",
    "network_activation": "tanh",
    "activation": "tanh",
    "network_dtype": "float32",
    "policy_guard_mode": "trainer-one-sided",
    "policy_guard_version": "merton-logw-v1",
    "policy_guard_eps": 1e-8,
    "policy_bounds_mode": policy_bounds_mode,
    "policy_numerator_expression": "V_y",
    "policy_numerator_guard": "clamp-min-eps",
    "policy_numerator_guard_eps": 1e-8,
    "policy_denominator_expression": "V_y-V_yy",
    "policy_denominator_coordinate": "log-wealth",
    "policy_denominator_guard": "clamp-min-eps",
    "policy_denominator_guard_eps": 1e-8,
    # Back-compatible scalar names used by the exact-map PolicySpec.
    "vw_guard": 1e-8,
    "denominator_guard": 1e-8,
    # Resolved bounds, rather than only the CLI inputs from which they arose.
    "policy_pi_min": pi_min_bound,
    "policy_pi_max": pi_max_bound,
    "policy_kappa_min": kappa_min_bound,
    "policy_kappa_max": kappa_max_bound,
    "policy_c_min": c_min_bound,
    "policy_c_max": c_max_bound,
    "eval_margin_coordinate": "y",
    "initial_policy_diffusion_variance_analytic": (
        None if not math.isfinite(initial_policy_diffusion_variance)
        else initial_policy_diffusion_variance
    ),
    "initial_policy_degenerate": initial_policy_degenerate,
    "initial_policy_degeneracy_tolerance": INITIAL_DIFFUSION_VARIANCE_TOL,
}
TRAINER_METADATA["trainer_source_sha256"] = mxu.sha256_file(os.path.abspath(__file__))

print(f"\n{'='*70}")
print("Multi-Asset Merton (with Consumption) - PI-PINN (FOC) [log W]")
print(f"{'='*70}")
print(f"  N_ASSETS={N_ASSETS}")
print(f"  gamma={gamma_risk}, rho={rho_discount}, r={r_rate}, epsilon={epsilon}")
print(f"  T={T_FINAL}, W∈[{x_min},{x_max}] -> y∈[{y_min:.3f},{y_max:.3f}]")
print(f"  Theta = mu^T Sigma^{-1} mu = {Theta:.6f}")
print(f"  nu = {nu:.6f}")
print(f"  policy bounds mode: {policy_bounds_mode}")
print(f"  pi clip abs: {pi_clip_abs}"
      + (f" (componentwise [{pi_min_bound},{pi_max_bound}])" if pi_clip_abs is not None else " (unconstrained)"))
print(f"  kappa=c/W bounds: [{kappa_min_bound},{kappa_max_bound}]")
print(f"  ||pi*||_2 = {np.linalg.norm(pi_star_np):.4f}, max|pi*_i|={np.max(np.abs(pi_star_np)):.4f}")
if initial_policy_degenerate:
    print(
        "[warning] Degenerate initial portfolio: "
        f"pi_0^T Sigma pi_0={initial_policy_diffusion_variance:.3e} <= "
        f"{INITIAL_DIFFUSION_VARIANCE_TOL:.1e}. This violates the intended "
        "nondegenerate frozen-PDE initialization; use myopic initialization "
        "with a positive, non-negligible pi_init_scale."
    )
elif initial_policy_degenerate is None:
    print(
        f"[warning] pi_init_method={_init_method!r} has no deterministic analytic "
        "ellipticity pre-check; inspect diffusion_var_min_init in status.json."
    )
if ARGS.lr_schedule == "carry_plateau" and ARGS.adam_reset == "full":
    print(
        "[warning] carry_plateau with adam_reset=full discards the restored "
        "held-out-best Adam moments at every outer boundary. The carried scalar "
        "LR is preserved, but adam_reset=keep is the model+optimizer carry "
        "protocol used for the paper runs."
    )
print(f"  cond(Sigma_safe) = {market_params['cond_Sigma_safe']:.2f}, max|rho_ij|={market_params['max_abs_rho']:.3f}")
print(f"{'='*70}\n")


# =============================================================================
# 2) Closed-form (for sanity check)
# =============================================================================
def closed_form_c(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Same functional form as 1D, with multi-asset nu."""
    tau = T_FINAL - t
    scale = mem.crra_homothetic_scale(tau, nu, gamma_risk, epsilon)
    return W / scale


def closed_form_V(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """CRRA value function V(t,W)=A(t) W^{1-gamma}/(1-gamma)."""
    tau = T_FINAL - t
    scale = mem.crra_homothetic_scale(tau, nu, gamma_risk, epsilon)
    A_t = scale ** gamma_risk
    return A_t * (W ** (1.0 - gamma_risk)) / (1.0 - gamma_risk)

def closed_form_pi(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Closed-form optimal portfolio (constant vector) broadcast on grid."""
    Nt, Nw = t.shape
    pi = np.broadcast_to(pi_star_np.reshape(1, 1, N_ASSETS), (Nt, Nw, N_ASSETS)).copy()
    return pi

def closed_form_numpy(t_grid: np.ndarray, W_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (V, c, pi) closed-form arrays on a grid."""
    V = closed_form_V(t_grid, W_grid)
    c = closed_form_c(t_grid, W_grid)
    pi = closed_form_pi(t_grid, W_grid)
    return V, c, pi
# =============================================================================
# 3) Value Network in (t, y)
# =============================================================================
class ValueNetLogW(nn.Module):
    def __init__(self, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, y], dim=1))


# =============================================================================
# 4) Sampling in (t, y)
# =============================================================================
def sample_interior(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    eps_t = 1e-3
    t = torch.rand(n, 1, device=device) * (T_FINAL - eps_t)
    y = y_min + torch.rand(n, 1, device=device) * (y_max - y_min)
    t.requires_grad_(True)
    y.requires_grad_(True)
    return t, y


def sample_terminal(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((n, 1), T_FINAL, device=device)
    y = y_min + torch.rand(n, 1, device=device) * (y_max - y_min)
    return t, y


# =============================================================================
# 5) Terminal and Utility
# =============================================================================
def V_terminal_from_y(y: torch.Tensor) -> torch.Tensor:
    W = torch.exp(y)
    return epsilon * W.pow(1.0 - gamma_risk) / (1.0 - gamma_risk)


def U_consumption(c: torch.Tensor) -> torch.Tensor:
    # Every implemented initialization/greedy path already returns c > 0.
    # An additional utility-only floor would silently change the frozen PDE
    # in policy_bounds_mode=none and break parity with the exact-map solver.
    return c.pow(1.0 - gamma_risk) / (1.0 - gamma_risk)


# =============================================================================
# 6) Derivatives in (t, y)
# =============================================================================
def compute_derivatives_log(
    value_net: nn.Module, t: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    V = value_net(t, y)
    ones = torch.ones_like(V)
    V_t = torch.autograd.grad(V, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_y = torch.autograd.grad(V, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_yy = torch.autograd.grad(V_y, y, grad_outputs=torch.ones_like(V_y), create_graph=True, retain_graph=True)[0]
    return V, V_t, V_y, V_yy


# =============================================================================
# 7) FOC-based policies in log space
# =============================================================================
def compute_c_from_foc_log(
    V_y: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
    kappa_min: Optional[float] = kappa_min_bound,
    kappa_max: Optional[float] = kappa_max_bound,
    c_min: Optional[float] = c_min_bound,
    c_max: Optional[float] = c_max_bound,
) -> torch.Tensor:
    """c* from U'(c)=V_W, with V_W = V_y/W and W=exp(y)."""
    W = torch.exp(y)
    V_w = V_y / W
    V_w_safe = torch.clamp(V_w, min=eps)
    c_raw = V_w_safe.pow(-1.0 / gamma_risk)

    kappa_raw = c_raw / W
    kappa = _clamp_optional(kappa_raw, kappa_min, kappa_max)
    c_new = kappa * W
    return _clamp_optional(c_new, c_min, c_max)


def compute_pi_from_foc_log_multi(
    V_y: torch.Tensor,
    V_yy: torch.Tensor,
    Sigma_inv_mu: torch.Tensor,
    eps: float = 1e-8,
    clip_abs: Optional[float] = pi_clip_abs,
    return_raw: bool = False,
) -> torch.Tensor:
    """
    Multi-asset pi* in y=log W coordinates:
        pi* = -(V_y / (V_yy - V_y)) * Sigma^{-1} mu_excess.
    """
    # d=V_y-V_yy=-W^2 V_ww must be positive.  A one-sided guard preserves
    # the concavity sign instead of turning a wrong-sign curvature into a
    # seemingly valid maximizer via abs().
    d_safe = torch.clamp(V_y - V_yy, min=eps)
    scalar = torch.clamp(V_y, min=eps) / d_safe  # (batch,1)
    pi_raw = scalar * Sigma_inv_mu.view(1, -1)  # (batch,N)
    if return_raw or clip_abs is None:
        return pi_raw
    return torch.clamp(pi_raw, min=-float(clip_abs), max=float(clip_abs))


# =============================================================================
# 8) LINEAR PDE residual in (t, y) for fixed (c_n, pi_n)
# =============================================================================
def linear_pde_residual_log_multi(
    value_net: nn.Module,
    c_n: torch.Tensor,      # (batch,1)
    pi_n: torch.Tensor,     # (batch,N)
    t: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Residual (linear in V under fixed controls):
        0 = rho V - V_t - U(c)
            - (r + pi^T mu_excess - c/W) V_y
            - 0.5 (pi^T Sigma pi) (V_yy - V_y)
    """
    V, V_t, V_y, V_yy = compute_derivatives_log(value_net, t, y)
    W = torch.exp(y)

    U_c = U_consumption(c_n)

    # pi^T mu_excess  (batch,1)
    pi_mu = (pi_n * mu_excess.view(1, -1)).sum(dim=1, keepdim=True)

    # pi^T Sigma pi (batch,1)
    Sigma_pi = pi_n @ Sigma  # (batch,N)
    pi_Sigma_pi = (Sigma_pi * pi_n).sum(dim=1, keepdim=True)

    drift_y = (r_rate + pi_mu) - (c_n / W)
    diff_coef = 0.5 * pi_Sigma_pi

    residual = rho_discount * V - V_t - U_c - drift_y * V_y - diff_coef * (V_yy - V_y)
    return residual, V, V_y, V_yy


def build_validation_set(
    n_int: int,
    n_term: int,
    target_device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Build a fixed held-out set without advancing the training RNG."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    t_int = torch.rand((n_int, 1), generator=gen) * (T_FINAL - 1e-3)
    y_int = y_min + torch.rand((n_int, 1), generator=gen) * (y_max - y_min)
    t_term = torch.full((n_term, 1), T_FINAL)
    y_term = y_min + torch.rand((n_term, 1), generator=gen) * (y_max - y_min)
    return {
        "t_int": t_int.to(target_device),
        "y_int": y_int.to(target_device),
        "t_term": t_term.to(target_device),
        "y_term": y_term.to(target_device),
        "V_term": V_terminal_from_y(y_term.to(target_device)).detach(),
    }


def build_diag_set(n_points: int, margin: float) -> Dict[str, torch.Tensor]:
    """Fixed complete tensor grid on Q_ev; shrink log wealth, never time."""
    n_points = max(4, int(n_points))
    n_t = max(2, int(round(math.sqrt(n_points))))
    n_y = max(2, int(math.ceil(n_points / n_t)))
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, float(margin))
    t_axis = np.linspace(t_min, t_max - 1e-3, n_t, dtype=np.float64)
    y_axis = np.linspace(y_lo, y_hi, n_y, dtype=np.float64)
    tt, yy = np.meshgrid(t_axis, y_axis, indexing="ij")
    t = torch.tensor(tt.reshape(-1, 1), device=device, dtype=torch.float32)
    y = torch.tensor(yy.reshape(-1, 1), device=device, dtype=torch.float32)
    return {"t": t, "y": y, "margin": float(margin)}


def closed_form_wealth_bundle(t_np: np.ndarray, w_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form (V,V_w,V_ww), used by the paper X_ev diagnostic."""
    tau = T_FINAL - t_np
    scale = mem.crra_homothetic_scale(tau, nu, gamma_risk, epsilon)
    A_t = scale ** gamma_risk
    V = A_t * w_np ** (1.0 - gamma_risk) / (1.0 - gamma_risk)
    V_w = A_t * w_np ** (-gamma_risk)
    V_ww = -gamma_risk * A_t * w_np ** (-gamma_risk - 1.0)
    return V, V_w, V_ww


def _relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return mem.relative_l2(pred, ref)


def _diffusion_variance(pi: torch.Tensor) -> torch.Tensor:
    return ((pi @ Sigma) * pi).sum(dim=1)


def _clip_fraction_pi(pi_raw: torch.Tensor) -> float:
    if pi_clip_abs is None:
        return 0.0
    active = torch.any(torch.abs(pi_raw) >= float(pi_clip_abs) - 1e-7, dim=1)
    return float(active.float().mean().item())


def _consumption_clip_fractions(comp: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Activation rates for the two-stage kappa then level projection."""
    kappa_raw = comp["kappa_raw"]
    c_level_raw = comp["c_level_raw"]
    return {
        "clip_frac_kappa_low": (0.0 if kappa_min_bound is None else float(
            (kappa_raw <= float(kappa_min_bound) + 1e-7).float().mean().item())),
        "clip_frac_kappa_high": (0.0 if kappa_max_bound is None else float(
            (kappa_raw >= float(kappa_max_bound) - 1e-7).float().mean().item())),
        "clip_frac_c_level_low": (0.0 if c_min_bound is None else float(
            (c_level_raw <= float(c_min_bound) + 1e-7).float().mean().item())),
        "clip_frac_c_level_high": (0.0 if c_max_bound is None else float(
            (c_level_raw >= float(c_max_bound) - 1e-7).float().mean().item())),
    }


def eval_diag_metrics(value_net: nn.Module, diag: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Merton-specific fixed-Q_ev X norm and policy/stability diagnostics."""
    was_training = value_net.training
    value_net.eval()
    t = diag["t"].detach().clone().requires_grad_(True)
    y = diag["y"].detach().clone().requires_grad_(True)
    V, _, V_y, V_yy = compute_derivatives_log(value_net, t, y)
    W = torch.exp(y)
    V_w = V_y / W
    V_ww = (V_yy - V_y) / (W ** 2)
    d = V_y - V_yy

    pi_raw = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu, return_raw=True)
    pi_eval = pi_raw if pi_clip_abs is None else torch.clamp(
        pi_raw, -float(pi_clip_abs), float(pi_clip_abs))

    V_w_safe = torch.clamp(V_w, min=1e-8)
    c_raw = V_w_safe.pow(-1.0 / gamma_risk)
    kappa_raw = c_raw / W
    kappa = _clamp_optional(kappa_raw, kappa_min_bound, kappa_max_bound)
    c_level_raw = kappa * W
    c_eval = _clamp_optional(c_level_raw, c_min_bound, c_max_bound)

    t_np = t.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()
    V_cf, Vw_cf, Vww_cf = closed_form_wealth_bundle(t_np, W_np)
    c_cf = closed_form_c(t_np, W_np)
    V_np = V.detach().cpu().numpy()
    Vw_np = V_w.detach().cpu().numpy()
    Vww_np = V_ww.detach().cpu().numpy()
    pi_np = pi_eval.detach().cpu().numpy()
    c_np = c_eval.detach().cpu().numpy()
    pi_cf = np.broadcast_to(pi_star_np.reshape(1, -1), pi_np.shape)

    e_V = float(np.max(np.abs(V_np - V_cf)))
    bundle_delta = np.sqrt((Vw_np - Vw_cf) ** 2 + (Vww_np - Vww_cf) ** 2)
    e_D = float(np.max(bundle_delta))
    a = _diffusion_variance(pi_eval)
    pi_l2 = torch.linalg.vector_norm(pi_eval, dim=1)
    chi_eval = c_eval / W
    out = {
        "e_V_sup": e_V,
        "e_bundle_sup": e_D,
        "e_Xev": e_V + e_D,
        "diag_RelL2_V": _relative_l2(V_np, V_cf),
        "diag_RelL2_pi": _relative_l2(pi_np, pi_cf),
        "diag_RelL2_c": _relative_l2(c_np, c_cf),
        "m_Vw": float(V_w.min().item()),
        "m_minus_Vww": float((-V_ww).min().item()),
        "m_curvature_y": float(d.min().item()),
        "m_y": float(V_y.min().item()),
        "M_y": float(V_y.max().item()),
        "m_c": float(d.min().item()),
        "pi_component_min_greedy": float(pi_eval.min().item()),
        "pi_component_max_greedy": float(pi_eval.max().item()),
        "pi_l2_min_greedy": float(pi_l2.min().item()),
        "pi_l2_max_greedy": float(pi_l2.max().item()),
        "chi_min_greedy": float(chi_eval.min().item()),
        "chi_max_greedy": float(chi_eval.max().item()),
        "guard_frac_Vw": float((V_w <= 1e-8).float().mean().item()),
        "guard_frac_curvature": float((d <= 1e-8).float().mean().item()),
        "clip_frac_pi_greedy": _clip_fraction_pi(pi_raw),
        "diffusion_var_min_greedy": float(a.min().item()),
        "diffusion_var_max_greedy": float(a.max().item()),
    }
    out.update(_consumption_clip_fractions({
        "kappa_raw": kappa_raw, "c_level_raw": c_level_raw}))
    if was_training:
        value_net.train()
    return out


# =============================================================================
# 9) PI-PINN Solver Class (FOC-based) in log W
# =============================================================================
class PIPINN_MultiAsset_Consumption_LogW:
    def __init__(
        self,
        value_hidden: int = 256,
        value_depth: int = 3,
        lr: float = 5e-4,
        scheduler_patience: int = 30,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
        clip_abs: Optional[float] = pi_clip_abs,
        c_min: Optional[float] = c_min_bound,
        c_max: Optional[float] = c_max_bound,
        lr_schedule: str = "carry_plateau",
        adam_reset: str = "keep",
        carry_lr_min: float = 1e-5,
        carry_lr_max: float = 5e-4,
        device: torch.device = device,
    ):
        self.device = device
        self.clip_abs = clip_abs
        self.c_min = c_min
        self.c_max = c_max

        self.value_net = ValueNetLogW(hidden=value_hidden, depth=value_depth).to(device)
        self.initial_lr = float(lr)
        self.lr_schedule = str(lr_schedule)
        self.adam_reset = str(adam_reset)
        self.carry_lr_min = float(carry_lr_min)
        self.carry_lr_max = float(carry_lr_max)
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_min_lr = float(scheduler_min_lr)
        self._outer_count = 0
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.scheduler = None if self.lr_schedule == "fixed" else self._make_scheduler()

    def _effective_min_lr(self) -> float:
        if self.lr_schedule == "carry_plateau":
            return max(self.scheduler_min_lr, self.carry_lr_min)
        return self.scheduler_min_lr

    def _make_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.scheduler_factor,
            patience=self.scheduler_patience, min_lr=self._effective_min_lr())

    def prepare_optimizer_for_outer(
        self,
        *,
        restart_carry_at_max: bool = False,
    ) -> float:
        """Prepare one frozen PDE and return its actual starting LR.

        ``restart_carry_at_max`` is reserved for residual-target training.
        With ``adam_reset=keep``, model parameters and Adam moments are
        retained, but every frozen PDE starts at ``carry_lr_max`` with a fresh
        plateau scheduler.  ``adam_reset=full`` keeps its explicit reset
        behavior. Ordinary PI-PINN runs keep the historical nonincreasing
        carry rule.
        """
        self._outer_count += 1
        if self.lr_schedule == "carry_plateau":
            if restart_carry_at_max:
                outer_lr = self.carry_lr_max
            elif self._outer_count == 1:
                outer_lr = self.initial_lr
            else:
                carried = float(self.optimizer.param_groups[0]["lr"])
                outer_lr = min(self.carry_lr_max, max(self._effective_min_lr(), carried))
            if self.adam_reset == "full":
                self.optimizer = optim.Adam(self.value_net.parameters(), lr=outer_lr)
            else:
                for group in self.optimizer.param_groups:
                    group["lr"] = outer_lr
            self.scheduler = self._make_scheduler()
        elif self.lr_schedule == "inner_plateau":
            if self.adam_reset == "full":
                self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.initial_lr)
            else:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.initial_lr
            self.scheduler = self._make_scheduler()
        return float(self.optimizer.param_groups[0]["lr"])

    def initialize_pi(
        self, t: torch.Tensor, y: torch.Tensor, method: str = "myopic", scale: float = 1.0,
        return_raw: bool = False,
    ) -> torch.Tensor:
        n = t.shape[0]
        if method == "zero":
            pi_raw = torch.zeros(n, N_ASSETS, device=self.device)
        elif method == "myopic":
            pi_raw = float(scale) * pi_star.view(1, -1).repeat(n, 1)
        elif method == "random":
            # Random initialization needs a finite scale even when the greedy
            # map itself is unconstrained.
            width = float(self.clip_abs) if self.clip_abs is not None else 2.0
            pi_raw = -width + 2.0 * width * torch.rand(n, N_ASSETS, device=self.device)
        else:
            raise ValueError(f"Unknown pi init method: {method}")
        if return_raw or self.clip_abs is None:
            return pi_raw
        return torch.clamp(pi_raw, -float(self.clip_abs), float(self.clip_abs))

    def initialize_c(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        method: str = "proportional",
        return_components: bool = False,
    ):
        n = t.shape[0]
        W = torch.exp(y.detach())
        if method == "zero":
            # Consumption must stay positive even without artificial box
            # bounds; this is an admissibility/numerical guard, not a level
            # projection.
            floor = 1e-8 if self.c_min is None else float(self.c_min)
            c_raw = torch.full((n, 1), floor, device=self.device)
        elif method == "proportional":
            c_raw = rho_discount * W
        elif method == "random":
            if self.c_min is None or self.c_max is None:
                c_raw = rho_discount * W * (
                    0.5 + torch.rand(n, 1, device=self.device))
            else:
                c_raw = float(self.c_min) + torch.rand(
                    n, 1, device=self.device) * (
                        float(self.c_max) - float(self.c_min))
        else:
            raise ValueError(f"Unknown c init method: {method}")
        kappa_raw = c_raw / W
        kappa = _clamp_optional(kappa_raw, kappa_min_bound, kappa_max_bound)
        c_level_raw = kappa * W
        c = _clamp_optional(c_level_raw, self.c_min, self.c_max)
        if return_components:
            return c, {
                "c_raw": c_raw.detach(),
                "kappa_raw": kappa_raw.detach(),
                "c_level_raw": c_level_raw.detach(),
            }
        return c

    def initialize_policy(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        pi_method: str,
        pi_scale: float,
        c_method: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate the analytic outer-zero policy and retain pre-clip controls."""
        c, comp = self.initialize_c(t, y, method=c_method, return_components=True)
        pi_raw = self.initialize_pi(
            t, y, method=pi_method, scale=pi_scale, return_raw=True)
        pi = pi_raw if self.clip_abs is None else torch.clamp(
            pi_raw, -float(self.clip_abs), float(self.clip_abs))
        comp["pi_raw"] = pi_raw.detach()
        return c.detach(), pi.detach(), comp

    def _policy_components(
        self, t: torch.Tensor, y: torch.Tensor, net: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        net = self.value_net if net is None else net
        t_eval = t.detach().clone().requires_grad_(True)
        y_eval = y.detach().clone().requires_grad_(True)
        V = net(t_eval, y_eval)
        V_y = torch.autograd.grad(V, y_eval, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        V_yy = torch.autograd.grad(V_y, y_eval, torch.ones_like(V_y), create_graph=False, retain_graph=True)[0]
        W = torch.exp(y_eval)
        V_w = V_y / W
        V_w_safe = torch.clamp(V_w, min=1e-8)
        c_raw = V_w_safe.pow(-1.0 / gamma_risk)
        kappa_raw = c_raw / W
        kappa = _clamp_optional(kappa_raw, kappa_min_bound, kappa_max_bound)
        c_level_raw = kappa * W
        c = _clamp_optional(c_level_raw, self.c_min, self.c_max)
        pi_raw = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu, return_raw=True)
        pi = pi_raw if self.clip_abs is None else torch.clamp(
            pi_raw, -float(self.clip_abs), float(self.clip_abs))
        return c.detach(), pi.detach(), {
            "c_raw": c_raw.detach(), "kappa_raw": kappa_raw.detach(),
            "c_level_raw": c_level_raw.detach(), "pi_raw": pi_raw.detach(),
            "V_w": V_w.detach(), "curvature_y": (V_y - V_yy).detach(),
        }

    def policy_improvement(
        self, t: torch.Tensor, y: torch.Tensor, net: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        own = net is None
        net = self.value_net if net is None else net
        if own:
            net.eval()
        c, pi, _ = self._policy_components(t, y, net=net)
        if own:
            net.train()
        return c, pi

    def policy_improvement_chunked(
        self, t: torch.Tensor, y: torch.Tensor, net: Optional[nn.Module] = None, chunk: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cs, pis = [], []
        for start in range(0, t.shape[0], chunk):
            c_b, pi_b = self.policy_improvement(t[start:start + chunk], y[start:start + chunk], net=net)
            cs.append(c_b); pis.append(pi_b)
        return torch.cat(cs, dim=0), torch.cat(pis, dim=0)

    def evaluate_heldout_pres(
        self, c_fixed: torch.Tensor, pi_fixed: torch.Tensor,
        val_set: Dict[str, torch.Tensor], chunk: int = 4096,
    ) -> Tuple[float, float, float]:
        was_training = self.value_net.training
        self.value_net.eval()
        sq_sum = 0.0
        count = 0
        for start in range(0, val_set["t_int"].shape[0], chunk):
            t_b = val_set["t_int"][start:start + chunk].detach().clone().requires_grad_(True)
            y_b = val_set["y_int"][start:start + chunk].detach().clone().requires_grad_(True)
            residual, _, _, _ = linear_pde_residual_log_multi(
                self.value_net, c_fixed[start:start + chunk], pi_fixed[start:start + chunk], t_b, y_b)
            sq_sum += float(torch.sum(residual.detach() ** 2).item())
            count += int(residual.numel())
        with torch.no_grad():
            pred_T = self.value_net(val_set["t_term"], val_set["y_term"])
            term_mse = float(torch.mean((pred_T - val_set["V_term"]) ** 2).item())
        pde_rms = math.sqrt(sq_sum / max(1, count))
        term_rms = math.sqrt(max(0.0, term_mse))
        if was_training:
            self.value_net.train()
        return pde_rms, term_rms, pde_rms + term_rms

    def policy_evaluation(
        self,
        c_n: torch.Tensor,
        pi_n: torch.Tensor,
        t_colloc: torch.Tensor,
        y_colloc: torch.Tensor,
        t_term: torch.Tensor,
        y_term: torch.Tensor,
        V_T_target: torch.Tensor,
        epochs: int = 200,
        w_terminal: float = 20.0,
        w_shape: float = 1.0,
        w_eta: float = 0.0,
        eta_focus_w: Optional[float] = None,
        eta_eps: float = 1e-8,
        eta_clip: Optional[float] = 10.0,
        print_every: int = 50,
        pres_target: Optional[float] = None,
        val_every: int = 1,
        val_fn=None,
        sel_fn=None,
        sel_every: int = 50,
        sel_patience: int = 6,
        restore_best: bool = True,
        resample_every: int = 0,
        resample_fn=None,
    ) -> Tuple[List[Dict], float, Optional[Dict[str, torch.Tensor]], int, float, Dict]:
        if pres_target is not None and val_fn is None:
            raise ValueError("pres_target requires a fixed Q_res validation function")
        if pres_target is not None and restore_best and sel_fn is None:
            raise ValueError(
                "target-eligible held-out restore requires a fixed Q_sel function")

        loss_history: List[Dict] = []
        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        c_fixed = c_n.detach()
        pi_fixed = pi_n.detach()

        # Keep the training-time crossing separate from the official outcome.
        # The former controls early stopping; the latter is computed only on
        # the state that remains after the optional held-out restore.
        training_target_crossed = False
        epochs_used = 0
        last_val = None
        last_val_epoch = -1
        val_pres_at_stop = ""
        val_pres_at_stop_epoch = ""
        n_resamples = 0

        best_sel_pres = float("inf")
        best_sel_state = None
        best_sel_epoch = -1
        sel_no_improve = 0
        sel_checks = 0
        sel_eligible_checks = 0
        sel_ineligible_checks = 0
        sel_stopped = False
        selection_requires_target = bool(restore_best and pres_target is not None)

        def run_val_check(epoch_idx: int, detect_training_crossing: bool = True):
            nonlocal last_val, last_val_epoch
            nonlocal training_target_crossed, val_pres_at_stop
            nonlocal val_pres_at_stop_epoch
            value = val_fn()
            last_val, last_val_epoch = value, int(epoch_idx)
            if (detect_training_crossing and not training_target_crossed
                    and pres_target is not None
                    and value[2] <= float(pres_target)):
                training_target_crossed = True
                val_pres_at_stop = float(value[2])
                val_pres_at_stop_epoch = int(epoch_idx)
            return value

        def run_sel_check(epoch_idx: int):
            nonlocal best_sel_pres, best_sel_state, best_sel_epoch
            nonlocal sel_no_improve, sel_checks, sel_stopped
            nonlocal sel_eligible_checks, sel_ineligible_checks
            value = sel_fn()
            sel_checks += 1
            if self.lr_schedule == "carry_plateau" and self.scheduler is not None:
                self.scheduler.step(float(value[2]))

            # Q_sel remains the ranking score.  For an E6 target run, however,
            # a restorable candidate must meet the Q_res target on this exact
            # same model state.  Because this is a genuine fixed-Q_res
            # evaluation, a crossing observed here is also a training-time
            # crossing and triggers the same early-stop contract.
            qres_value = None
            eligible = True
            if selection_requires_target:
                if last_val_epoch == int(epoch_idx) and last_val is not None:
                    qres_value = last_val
                else:
                    qres_value = run_val_check(
                        epoch_idx, detect_training_crossing=True)
                eligible = bool(qres_value[2] <= float(pres_target))

            if not eligible:
                sel_ineligible_checks += 1
                return value

            sel_eligible_checks += 1
            if value[2] < best_sel_pres:
                best_sel_pres = float(value[2])
                best_sel_epoch = int(epoch_idx)
                best_sel_state = {
                    "model": {k: tensor.detach().cpu().clone()
                              for k, tensor in self.value_net.state_dict().items()},
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "epoch": int(epoch_idx),
                    "pres": float(value[2]),
                    "qres_pres": (
                        float(qres_value[2]) if qres_value is not None else ""
                    ),
                }
                sel_no_improve = 0
            else:
                sel_no_improve += 1
                if sel_patience and sel_no_improve >= int(sel_patience):
                    sel_stopped = True
            return value

        if val_fn is not None and pres_target is not None:
            run_val_check(0)
        if sel_fn is not None:
            run_sel_check(0)

        for epoch in range(1, epochs + 1):
            if training_target_crossed or sel_stopped:
                break
            if (resample_fn is not None and resample_every > 0 and epoch > 1
                    and (epoch - 1) % int(resample_every) == 0):
                c_n, pi_n, t_colloc, y_colloc, t_term, y_term, V_T_target = resample_fn()
                c_fixed = c_n.detach(); pi_fixed = pi_n.detach()
                n_resamples += 1

            self.optimizer.zero_grad(set_to_none=True)

            t_int = t_colloc.detach().clone().requires_grad_(True)
            y_int = y_colloc.detach().clone().requires_grad_(True)

            residual, V, V_y, V_yy = linear_pde_residual_log_multi(
                self.value_net, c_fixed, pi_fixed, t_int, y_int
            )
            pde_loss = torch.mean(residual ** 2)

            V_T_pred = self.value_net(t_term, y_term)
            terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)

            # Shape penalties
            mono_penalty = torch.mean(torch.relu(-V_y) ** 2)  # V_y>0
            conc_indicator = (V_yy - V_y)  # should be <0
            conc_penalty = torch.mean(torch.relu(conc_indicator) ** 2)

            # eta penalty: eta = -(W V_WW)/V_W = 1 - V_yy/V_y
            V_y_safe = torch.clamp(V_y, min=eta_eps)
            eta = 1.0 - (V_yy / V_y_safe)
            if w_eta != 0.0 and eta_clip is not None:
                eta_err = torch.clamp(eta - gamma_risk, -eta_clip, eta_clip)
                eta_err_sq = eta_err ** 2
                if eta_focus_w is not None:
                    W_int = torch.exp(y_int)
                    mask = (W_int <= eta_focus_w).float()
                    eta_loss = (eta_err_sq * mask).sum() / (mask.sum() + 1e-12)
                else:
                    eta_loss = torch.mean(eta_err_sq)
            else:
                eta_loss = torch.zeros((), device=y_int.device)

            total_loss = pde_loss + w_terminal * terminal_loss + w_shape * (mono_penalty + conc_penalty) + w_eta * eta_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            epochs_used = epoch

            if self.lr_schedule == "inner_plateau" and self.scheduler is not None:
                self.scheduler.step(total_loss.detach().cpu())

            cur = float(total_loss.item())
            if cur < best_loss:
                best_loss = cur
                best_epoch = epoch

            loss_history.append(
                {
                    "inner_epoch": int(epoch),
                    "total": cur,
                    "pde": float(pde_loss.item()),
                    "terminal": float(terminal_loss.item()),
                    "mono": float(mono_penalty.item()),
                    "conc": float(conc_penalty.item()),
                    "eta": float(eta_loss.item()),
                    "train_pres": mxu.pres_from_mse(float(pde_loss.item()), float(terminal_loss.item())),
                    "val_pde_rms": "", "val_terminal_rms": "", "val_pres": "", "sel_pres": "",
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
            )

            if val_fn is not None and pres_target is not None and epoch % int(val_every) == 0:
                value = run_val_check(epoch)
                loss_history[-1]["val_pde_rms"] = value[0]
                loss_history[-1]["val_terminal_rms"] = value[1]
                loss_history[-1]["val_pres"] = value[2]
            if sel_fn is not None and epoch % int(sel_every) == 0:
                value = run_sel_check(epoch)
                loss_history[-1]["sel_pres"] = value[2]

            if print_every and print_every > 0 and epoch % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"      [Eval {epoch:4d}/{epochs}] Loss: {cur:.3e} | PDE: {pde_loss.item():.3e} | "
                    f"Terminal: {terminal_loss.item():.3e} | Eta: {eta_loss.item():.3e} | LR: {lr:.2e}"
                )

        # Target may be reached at epoch zero; downstream logging still needs
        # a finite row.  This row is diagnostic only (no optimizer step).
        if not loss_history:
            t_int = t_colloc.detach().clone().requires_grad_(True)
            y_int = y_colloc.detach().clone().requires_grad_(True)
            residual, _, V_y, V_yy = linear_pde_residual_log_multi(
                self.value_net, c_fixed, pi_fixed, t_int, y_int)
            pde = torch.mean(residual.detach() ** 2)
            with torch.no_grad():
                term = torch.mean((self.value_net(t_term, y_term) - V_T_target) ** 2)
            mono = torch.mean(torch.relu(-V_y.detach()) ** 2)
            conc = torch.mean(torch.relu(V_yy.detach() - V_y.detach()) ** 2)
            loss_history.append({
                "inner_epoch": 0,
                "total": float((pde + w_terminal * term + w_shape * (mono + conc)).item()),
                "pde": float(pde.item()), "terminal": float(term.item()),
                "mono": float(mono.item()), "conc": float(conc.item()), "eta": 0.0,
                "train_pres": mxu.pres_from_mse(float(pde.item()), float(term.item())),
                "val_pde_rms": last_val[0] if last_val else "",
                "val_terminal_rms": last_val[1] if last_val else "",
                "val_pres": last_val[2] if last_val else "", "sel_pres": "",
                "lr": float(self.optimizer.param_groups[0]["lr"]), "synthetic": True,
            })

        # If a target/cap stopped between regular selection checks, make the
        # actual terminal inner state eligible instead of accidentally
        # selecting only epoch zero.
        if (sel_fn is not None and epochs_used > 0
                and epochs_used % int(sel_every) != 0 and not sel_stopped):
            value = run_sel_check(epochs_used)
            loss_history[-1]["sel_pres"] = value[2]

        # Capture the true end-of-inner LR before any held-out restore.  A
        # restored Adam state contains the LR from the selected checkpoint, but
        # carry_plateau must never undo a later within-inner LR reduction.
        end_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        lr_end_before_restore = end_lrs[0]
        lr_best_checkpoint = ""
        lr_after_restore = end_lrs[0]
        if restore_best and best_sel_state is not None:
            self.value_net.load_state_dict(best_sel_state["model"])
            self.optimizer.load_state_dict(best_sel_state["optimizer"])
            lr_best_checkpoint = float(best_sel_state["lr"])
            if self.lr_schedule == "carry_plateau":
                # Liu carry rule: restore the selected model and Adam moments,
                # then carry the lower of the selected-checkpoint and inner-end
                # LRs, subject to the configured scheduler/carry floor.
                floor_lr = self._effective_min_lr()
                for group, end_lr in zip(self.optimizer.param_groups, end_lrs):
                    best_lr = float(group["lr"])
                    group["lr"] = max(floor_lr, min(best_lr, end_lr))
            lr_after_restore = float(self.optimizer.param_groups[0]["lr"])
            last_val_epoch = -1

        lr_carried_next = ""
        lr_next_outer_policy = "not-applicable"
        if self.lr_schedule == "carry_plateau":
            if pres_target is not None:
                # E6/p_res runs deliberately restart every frozen PDE at the
                # configured ceiling.  The selected model and Adam moments
                # remain restored; only the optimizer-group LR is replaced at
                # the next outer boundary.
                lr_carried_next = self.carry_lr_max
                lr_next_outer_policy = "restart-at-carry-lr-max"
            else:
                # Ordinary PI-PINN retains the Liu carry rule, including both
                # the configured floor and ceiling.
                lr_carried_next = min(
                    self.carry_lr_max,
                    max(
                        self._effective_min_lr(),
                        float(self.optimizer.param_groups[0]["lr"]),
                    ),
                )
                lr_next_outer_policy = "carry-restored-inner-lr"

        if val_fn is not None and last_val_epoch != epochs_used:
            # This is explicitly post-restore.  The training crossing remains
            # a separate sticky diagnostic and is never used as the official
            # residual achieved by the restored model.
            last_val = val_fn()
            last_val_epoch = epochs_used

        if (selection_requires_target and best_sel_state is not None
                and last_val is not None
                and float(last_val[2])
                > float(pres_target) * (1.0 + 1e-6)):
            raise RuntimeError(
                "target-eligible Q_sel checkpoint failed deterministic "
                "post-restore Q_res remeasurement: "
                f"selected={best_sel_state['qres_pres']}, "
                f"restored={float(last_val[2]):.6e}, "
                f"target={float(pres_target):.6e}"
            )

        official_target_reached = bool(
            pres_target is not None
            and last_val is not None
            and float(last_val[2]) <= float(pres_target)
        )

        info = {
            "epochs_used": int(epochs_used),
            "n_resamples": int(n_resamples),
            # Back-compatible name, now intentionally carrying the official
            # post-restore meaning required by E6.
            "target_reached": official_target_reached,
            "target_reached_post_restore": official_target_reached,
            "training_target_crossed": bool(training_target_crossed),
            "val_pres_at_stop": val_pres_at_stop,
            "val_pres_at_stop_epoch": val_pres_at_stop_epoch,
            "val_pde_rms_post_restore": last_val[0] if last_val else "",
            "val_terminal_rms_post_restore": last_val[1] if last_val else "",
            "val_pres_post_restore": last_val[2] if last_val else "",
            "sel_best_pres": best_sel_state["pres"] if best_sel_state else "",
            "sel_best_qres_pres": (
                best_sel_state["qres_pres"] if best_sel_state else ""
            ),
            "sel_best_epoch": best_sel_state["epoch"] if best_sel_state else "",
            "sel_best_lr": best_sel_state["lr"] if best_sel_state else "",
            "sel_checks": int(sel_checks), "sel_stopped": int(bool(sel_stopped)),
            "sel_eligible_checks": int(sel_eligible_checks),
            "sel_ineligible_checks": int(sel_ineligible_checks),
            "sel_restored": int(bool(restore_best and best_sel_state is not None)),
            "lr_end_before_restore": lr_end_before_restore,
            "lr_best_checkpoint": lr_best_checkpoint,
            "lr_after_restore": lr_after_restore,
            "lr_carried_next": lr_carried_next,
            "lr_next_outer_policy": lr_next_outer_policy,
        }
        return loss_history, best_loss, best_state, best_epoch, float(loss_history[-1]["total"]), info

    def run_policy_iteration(
        self,
        outer_iters: int = 200,
        eval_epochs: int = 200,
        batch_size: int = 2000,
        terminal_frac: float = 0.5,
        w_terminal: float = 20.0,
        w_shape: float = 1.0,
        w_eta: float = 0.0,
        eta_focus_w: Optional[float] = None,
        eta_clip: float = 10.0,
        pi_init_method: str = "myopic",
        pi_init_scale: float = 1.0,
        c_init_method: str = "proportional",
        print_every_outer: int = 10,
        print_every_eval: int = 200,
        verbose_detail: bool = False,
        inner_best_restore: bool = True,
        sel_points: int = 10000,
        sel_terminal_points: int = 2000,
        sel_every: int = 50,
        sel_patience: int = 6,
        pe_resample_every: int = 0,
        pres_target: Optional[float] = None,
        val_points: int = 100000,
        val_terminal_points: int = 10000,
        val_every: int = 1,
        val_seed: int = 0,
        diag_points: int = 4096,
        diag_margin: float = 0.1,
        diag_every: int = 1,
        save_iterate_every: int = 0,
        e3b_checkpoints: bool = False,
        timing_mode: bool = False,
        e6_role: str = "standard",
        e6_warm_start_provenance: Optional[Dict] = None,
        weight_dir: str = "weights",
        recorder=None,
        stopper=None,
    ) -> Dict:
        print_every_outer = int(print_every_outer)
        if print_every_outer < 0:
            raise ValueError("print_every_outer must be nonnegative")
        e6_role = str(e6_role)
        if e6_role not in {"standard", "warmup", "target_branch"}:
            raise ValueError(f"unsupported E6 role: {e6_role!r}")
        is_target_branch = e6_role == "target_branch"
        algorithm_outer_offset = 1 if is_target_branch else 0
        print(f"\n{'='*70}")
        print(f"PI-PINN Algorithm 2 (multi-asset, logW, with consumption): {outer_iters} outer iterations")
        print(f"  Eval epochs per iter: {eval_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  pi init: {pi_init_method} | c init: {c_init_method}")
        print(f"  Initial LR: {self.initial_lr:.2e}")
        if pres_target is not None and self.lr_schedule == "carry_plateau":
            print(
                "  p_res LR policy: every outer restarts at "
                f"carry_lr_max={self.carry_lr_max:.2e}; "
                "plateau best/patience resets at every outer"
            )
        if is_target_branch:
            print(
                "  E6 target branch: common warm-up v~_0 loaded; "
                f"running {outer_iters} target-phase evaluations n=1,...,{outer_iters}"
            )
        print(f"{'='*70}\n")

        results = {
            "pi_diff": [],
            "c_diff": [],
            "pi_vs_closed_form": [],
            "c_vs_closed_form": [],
            "eval_loss": [],
            "eta_loss": [],
            "pi_mean": [],
            "pi_std": [],
            "c_mean": [],
            "c_std": [],
            "lr": [],
            "loss_history": [],
            "outer_rows": [],
            "total_optimizer_steps": 0,
            "target_reached": False,
            "target_flags": [],
            "training_target_crossed": False,
            "training_target_flags": [],
            "val_pres": [],
            "inner_epochs_used": [],
        }

        best_eval_loss = float("inf")
        best_iter = 0
        best_diag_state = None

        os.makedirs(weight_dir, exist_ok=True)
        best_model_path = os.path.join(weight_dir, "value_net_best_diag.pt")
        last_model_path = os.path.join(weight_dir, "value_net_last.pt")
        final_model_path = os.path.join(weight_dir, "value_net_final.pt")
        iterate_dir = os.path.join(weight_dir, "iterates")
        manifest_path = os.path.join(weight_dir, "checkpoint_manifest.json")
        if e3b_checkpoints or save_iterate_every > 0:
            os.makedirs(iterate_dir, exist_ok=True)

        checkpoint_records: Dict[int, Dict] = {}
        checkpoint_manifest = {
            "schema_version": 1,
            "created_at": mxu.now_iso(),
            "updated_at": mxu.now_iso(),
            "status": "running",
            "e6_role": e6_role,
            "e6_phase": "target" if is_target_branch else e6_role,
            "algorithm_outer_index_offset": int(algorithm_outer_offset),
            "warmup_excluded_from_outer_history": bool(is_target_branch),
            "warmup_provenance": dict(e6_warm_start_provenance or {}),
            "trainer_protocol": TRAINER_METADATA["trainer_protocol"],
            "trainer_protocol_version": TRAINER_METADATA["trainer_protocol_version"],
            "trainer_source_marker": TRAINER_METADATA["trainer_source_marker"],
            "trainer_source_sha256": TRAINER_METADATA["trainer_source_sha256"],
            "policy_guard_mode": TRAINER_METADATA["policy_guard_mode"],
            "policy_guard_version": TRAINER_METADATA["policy_guard_version"],
            "policy_bounds_mode": TRAINER_METADATA["policy_bounds_mode"],
            "resolved_policy_bounds": {
                "portfolio_min": pi_min_bound,
                "portfolio_max": pi_max_bound,
                "kappa_min": kappa_min_bound,
                "kappa_max": kappa_max_bound,
                "consumption_min": c_min_bound,
                "consumption_max": c_max_bound,
            },
            "policy_numerator_guard_eps": TRAINER_METADATA[
                "policy_numerator_guard_eps"
            ],
            "policy_denominator_guard_eps": TRAINER_METADATA[
                "policy_denominator_guard_eps"
            ],
            "inner_selection_restore_contract": TRAINER_METADATA[
                "inner_selection_restore_contract"
            ],
            "carry_lr_restore_contract": TRAINER_METADATA[
                "carry_lr_restore_contract"
            ],
            "pres_target_lr_restart_contract": TRAINER_METADATA[
                "pres_target_lr_restart_contract"
            ],
            "scheduler_reset_contract": TRAINER_METADATA[
                "scheduler_reset_contract"
            ],
            "checkpoint_timing_contract": TRAINER_METADATA[
                "checkpoint_timing_contract"
            ],
            "q_res_role": TRAINER_METADATA["q_res_role"],
            "q_res_seed": TRAINER_METADATA["q_res_seed"],
            "q_res_lifetime": TRAINER_METADATA["q_res_lifetime"],
            "q_sel_role": TRAINER_METADATA["q_sel_role"],
            "q_sel_lifetime": TRAINER_METADATA["q_sel_lifetime"],
            "q_sel_seed_formula": TRAINER_METADATA["q_sel_seed_formula"],
            "requested_outer_iters": int(outer_iters),
            "checkpoint_policy": {
                "e3b_checkpoints": bool(e3b_checkpoints),
                "save_iterate_every": int(save_iterate_every),
                "timing_mode": bool(timing_mode),
            },
            "resolved_training_protocol": {
                "inner_best_restore": bool(inner_best_restore),
                "inner_selection": (
                    "heldout-residual" if inner_best_restore else "disabled"
                ),
                "inner_restore": (
                    "model-plus-optimizer"
                    if inner_best_restore else "final-inner-iterate"
                ),
                "target_eligible_selection": (
                    "same-state-fixed-qres-at-or-below-target"
                    if inner_best_restore and pres_target is not None
                    else "not-applied"
                ),
                "checkpoint_timing": (
                    "post-policy-evaluation-after-optional-heldout-restore"
                ),
                "q_res": "run-fixed-market-seed-stream",
                "q_sel": "outer-specific-market-seed-stream-fixed-within-inner",
                "q_sel_seed_formula": TRAINER_METADATA["q_sel_seed_formula"],
                "pres_target_lr_restart": bool(
                    pres_target is not None
                    and self.lr_schedule == "carry_plateau"
                ),
                "outer_start_lr_policy": (
                    "restart-at-carry-lr-max"
                    if (
                        pres_target is not None
                        and self.lr_schedule == "carry_plateau"
                    )
                    else "ordinary-schedule"
                ),
                "plateau_scheduler_state": (
                    "reset-at-every-outer"
                    if self.lr_schedule != "fixed"
                    else "not-applicable"
                ),
            },
            "indexing": {
                "checkpoint_outer_index_base": 1,
                "source_iter_offset_from_checkpoint_outer": (
                    0 if is_target_branch else -1
                ),
                "target_policy_iter_offset_from_checkpoint_outer": (
                    1 if is_target_branch else 0
                ),
                "description": (
                    (
                        "target-branch outer 1 freezes the policy induced by the "
                        "loaded common warm-up value iterate v~_0 and produces v~_1; "
                        "outer_history contains target-phase rows only"
                    )
                    if is_target_branch
                    else (
                        "outer 1 evaluates the analytic initial policy and produces "
                        "value iterate 0; value_net_iterNNNN therefore has exact-map "
                        "source_iter NNNN-1 and target_policy_iter NNNN"
                    )
                ),
            },
            "state_hash": {
                "algorithm": "sha256",
                "representation": "canonical-sorted-state-dict-tensors",
                "note": "file_sha256 is not used to decide model-state equality",
            },
            "checkpoints": [],
        }

        def snapshot_value_state() -> Dict[str, torch.Tensor]:
            return {
                key: tensor.detach().cpu().clone()
                for key, tensor in self.value_net.state_dict().items()
            }

        def relative_weight_path(path: str) -> str:
            return os.path.relpath(path, weight_dir)

        def write_checkpoint_manifest(status: Optional[str] = None) -> None:
            if status is not None:
                checkpoint_manifest["status"] = status
            checkpoint_manifest["updated_at"] = mxu.now_iso()
            checkpoint_manifest["checkpoints"] = [
                checkpoint_records[key] for key in sorted(checkpoint_records)
            ]
            mxu.save_json_atomic(manifest_path, checkpoint_manifest)

        def save_iterate_checkpoint(
            outer_iter: int,
            reason: str,
            state: Optional[Dict[str, torch.Tensor]] = None,
            *,
            update_manifest: bool = True,
        ) -> Dict:
            outer_iter = int(outer_iter)
            state = snapshot_value_state() if state is None else state
            path = os.path.join(iterate_dir, f"value_net_iter{outer_iter:04d}.pt")
            torch.save(state, path)
            previous = checkpoint_records.get(outer_iter, {})
            reasons = list(previous.get("reasons", []))
            if reason not in reasons:
                reasons.append(reason)
            record = {
                "checkpoint_outer_iter": outer_iter,
                "source_iter": (
                    outer_iter if is_target_branch else outer_iter - 1
                ),
                "target_policy_iter": (
                    outer_iter + 1 if is_target_branch else outer_iter
                ),
                "path": relative_weight_path(path),
                "reasons": reasons,
                "state_sha256": mxu.canonical_state_dict_sha256(state),
                "file_sha256": mxu.sha256_file(path),
            }
            checkpoint_records[outer_iter] = record
            if update_manifest:
                write_checkpoint_manifest()
            return record

        # Written before the first outer iteration so a failed/interrupted run
        # cannot be mistaken for a completed trajectory.
        write_checkpoint_manifest()

        val_set = None
        if val_points > 0 and not (timing_mode and pres_target is None):
            val_set = build_validation_set(val_points, max(1, val_terminal_points), self.device, val_seed)
        diag = None
        diag_col = None
        if diag_points > 0 and not timing_mode:
            diag = build_diag_set(diag_points, diag_margin)
            # E1 ellipticity uses a fixed dense tensor grid on all Q_col.
            diag_col = build_diag_set(diag_points, 0.0)

        # The historical initial-sample print consumes the global training RNG.
        # A target branch must instead begin at the exact RNG state stored after
        # warm-up, so it intentionally skips this diagnostic-only draw.
        if is_target_branch:
            print("Initial value iterate: restored common E6 warm-up v~_0")
        else:
            t_colloc, y_colloc = sample_interior(batch_size, self.device)
            t_term, y_term = sample_terminal(
                max(1, int(batch_size * terminal_frac)), self.device)
            V_T_target = V_terminal_from_y(y_term).detach()

            c_n, pi_n, _ = self.initialize_policy(
                t_colloc, y_colloc, pi_init_method, pi_init_scale, c_init_method)
            print(
                f"Initial c: mean={c_n.mean().item():.4f}, "
                f"std={c_n.std().item():.4f}")
            print(
                f"Initial pi: mean={pi_n.mean().item():.4f}, "
                f"std={pi_n.std().item():.4f}")
            print(
                f"pi* stats: mean={pi_star.mean().item():.4f}, "
                f"std={pi_star.std().item():.4f}, "
                f"||pi*||2={pi_star.norm().item():.4f}")

        for it in range(1, outer_iters + 1):
            algorithm_outer_iter = int(it + algorithm_outer_offset)
            if stopper is not None and stopper.shared_stop_exists():
                meta = stopper.mark_from_existing_flag(outer_iter=it, pde_loss=None)
                results["stopped_early"] = True
                results["stop_info"] = meta
                break
            verbose = should_print_outer(it, print_every_outer)
            if verbose:
                print(f"\n[Outer Iteration {it}/{outer_iters}]")
                print("-" * 40)

            # fresh samples
            t_colloc, y_colloc = sample_interior(batch_size, self.device)
            t_term, y_term = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device)
            V_T_target = V_terminal_from_y(y_term).detach()

            # recompute policy on new points
            policy_source = None
            if is_target_branch or it > 1:
                # Freeze the OUTER-start network so training/validation/
                # selection/resampling all see one identical frozen policy.
                policy_source = copy.deepcopy(self.value_net).eval()
                for parameter in policy_source.parameters():
                    parameter.requires_grad_(False)
                c_n, pi_n, _ = self._policy_components(
                    t_colloc, y_colloc, net=policy_source)
            else:
                c_n, pi_n, _ = self.initialize_policy(
                    t_colloc, y_colloc, pi_init_method, pi_init_scale, c_init_method)

            def frozen_policy(t_pts, y_pts, _src=policy_source):
                if _src is None:
                    c_f, pi_f, _ = self.initialize_policy(
                        t_pts, y_pts, pi_init_method, pi_init_scale, c_init_method)
                    return c_f, pi_f
                return self.policy_improvement_chunked(t_pts, y_pts, net=_src)

            val_fn = None
            c_val = pi_val = None
            if val_set is not None:
                c_val, pi_val = frozen_policy(val_set["t_int"], val_set["y_int"])
                val_fn = lambda _c=c_val, _p=pi_val: self.evaluate_heldout_pres(_c, _p, val_set)

            # Q_sel,n is independent across frozen PDEs and fixed throughout
            # this inner solve.  The dedicated CPU generator inside
            # build_validation_set means this does not advance the training
            # RNG and therefore depends on market_seed/outer_iter only.
            q_sel_seed = ""
            sel_set = None
            if inner_best_restore and sel_points > 0:
                q_sel_seed = qsel_seed_for_outer(
                    val_seed, algorithm_outer_iter)
                sel_set = build_validation_set(
                    sel_points,
                    max(1, sel_terminal_points),
                    self.device,
                    q_sel_seed,
                )
            sel_fn = None
            if sel_set is not None:
                c_sel, pi_sel = frozen_policy(sel_set["t_int"], sel_set["y_int"])
                sel_fn = lambda _c=c_sel, _p=pi_sel: self.evaluate_heldout_pres(_c, _p, sel_set)

            resample_fn = None
            if pe_resample_every > 0:
                def _resample():
                    t_c, y_c = sample_interior(batch_size, self.device)
                    t_T, y_T = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device)
                    c_f, pi_f = frozen_policy(t_c, y_c)
                    return c_f, pi_f, t_c, y_c, t_T, y_T, V_terminal_from_y(y_T).detach()
                resample_fn = _resample

            restart_carry_at_max = bool(
                pres_target is not None
                and self.lr_schedule == "carry_plateau"
            )
            lr_outer_start = self.prepare_optimizer_for_outer(
                restart_carry_at_max=restart_carry_at_max
            )

            # evaluation step
            eval_loss_hist, _, _, _, last_eval_loss, eval_info = self.policy_evaluation(
                c_n=c_n,
                pi_n=pi_n,
                t_colloc=t_colloc,
                y_colloc=y_colloc,
                t_term=t_term,
                y_term=y_term,
                V_T_target=V_T_target,
                epochs=eval_epochs,
                w_terminal=w_terminal,
                w_shape=w_shape,
                w_eta=w_eta,
                eta_focus_w=eta_focus_w,
                eta_clip=eta_clip,
                print_every=print_every_eval if (verbose and verbose_detail) else (eval_epochs + 1),
                pres_target=pres_target, val_every=val_every, val_fn=val_fn,
                sel_fn=sel_fn, sel_every=sel_every, sel_patience=sel_patience,
                restore_best=bool(inner_best_restore),
                resample_every=pe_resample_every, resample_fn=resample_fn,
            )
            results["total_optimizer_steps"] += int(eval_info["epochs_used"])
            results["target_reached"] = results["target_reached"] or bool(eval_info["target_reached"])
            results["target_flags"].append(bool(eval_info["target_reached"]))
            results["training_target_crossed"] = (
                results["training_target_crossed"]
                or bool(eval_info["training_target_crossed"])
            )
            results["training_target_flags"].append(
                bool(eval_info["training_target_crossed"])
            )
            results["inner_epochs_used"].append(int(eval_info["epochs_used"]))
            # E6's official achieved residual belongs to the exact model state
            # used for the outer checkpoint and downstream error evaluation.
            achieved_pres = eval_info["val_pres_post_restore"]
            if isinstance(achieved_pres, (int, float)):
                results["val_pres"].append(float(achieved_pres))
            for hist_row in eval_loss_hist:
                hist_row["outer_iter"] = it
            if timing_mode:
                results["loss_history"] = [eval_loss_hist[-1]]
            else:
                results["loss_history"].extend(eval_loss_hist)
            results["eval_loss"].append(last_eval_loss)
            results["eta_loss"].append(eval_loss_hist[-1]["eta"])

            # Figure-1 control trajectories use one fixed Q_ev grid across
            # every outer iteration and every training seed.  The frozen
            # policy is exactly the alpha_n used in this outer's linear PDE;
            # the current network generates alpha_{n+1}.
            if diag is not None:
                metric_t, metric_y = diag["t"], diag["y"]
                if policy_source is None:
                    c_frozen_metric, pi_frozen_metric, frozen_metric_comp = (
                        self.initialize_policy(
                            metric_t, metric_y, pi_init_method,
                            pi_init_scale, c_init_method
                        )
                    )
                else:
                    c_frozen_metric, pi_frozen_metric, frozen_metric_comp = (
                        self._policy_components(
                            metric_t, metric_y, net=policy_source
                        )
                    )
                c_new, pi_new = self.policy_improvement_chunked(
                    metric_t, metric_y)
                control_metric_scope = "fixed_qev"
            else:
                metric_t, metric_y = t_colloc, y_colloc
                if policy_source is None:
                    c_frozen_metric, pi_frozen_metric, frozen_metric_comp = (
                        self.initialize_policy(
                            metric_t, metric_y, pi_init_method,
                            pi_init_scale, c_init_method
                        )
                    )
                else:
                    c_frozen_metric, pi_frozen_metric, frozen_metric_comp = (
                        self._policy_components(
                            metric_t, metric_y, net=policy_source
                        )
                    )
                c_new, pi_new = self.policy_improvement(t_colloc, y_colloc)
                control_metric_scope = "training_batch_fallback"

            c_diff = ((c_new - c_frozen_metric) ** 2).mean().item()
            pi_diff = ((pi_new - pi_frozen_metric) ** 2).mean().item()
            results["c_diff"].append(c_diff)
            results["pi_diff"].append(pi_diff)

            # compare with closed form
            pi_star_rep = pi_star.view(1, -1).repeat(pi_new.shape[0], 1)
            pi_vs_cf = ((pi_new - pi_star_rep) ** 2).mean().item()
            results["pi_vs_closed_form"].append(pi_vs_cf)

            # closed form c
            t_np = metric_t.detach().cpu().numpy()
            W_np = np.exp(metric_y.detach().cpu().numpy())
            c_star_np = closed_form_c(t_np, W_np)
            c_star = torch.tensor(c_star_np, device=self.device, dtype=torch.float32)
            c_vs_cf = ((c_new - c_star) ** 2).mean().item()
            results["c_vs_closed_form"].append(c_vs_cf)

            results["pi_mean"].append(pi_new.mean().item())
            results["pi_std"].append(pi_new.std().item())
            results["c_mean"].append(c_new.mean().item())
            results["c_std"].append(c_new.std().item())

            frozen_pi_l2 = torch.linalg.vector_norm(pi_frozen_metric, dim=1)
            frozen_chi = c_frozen_metric / torch.exp(metric_y)
            frozen_control_ranges = {
                "pi_component_min_frozen": float(pi_frozen_metric.min().item()),
                "pi_component_max_frozen": float(pi_frozen_metric.max().item()),
                "pi_l2_min_frozen": float(frozen_pi_l2.min().item()),
                "pi_l2_max_frozen": float(frozen_pi_l2.max().item()),
                "chi_min_frozen": float(frozen_chi.min().item()),
                "chi_max_frozen": float(frozen_chi.max().item()),
            }

            cur_lr = self.optimizer.param_groups[0]["lr"]
            results["lr"].append(cur_lr)

            diagnostic_score = eval_info.get("sel_best_pres", "")
            if not isinstance(diagnostic_score, (int, float)):
                diagnostic_score = eval_info.get("val_pres_post_restore", "")
            if not isinstance(diagnostic_score, (int, float)):
                diagnostic_score = last_eval_loss
            if not timing_mode and float(diagnostic_score) < best_eval_loss:
                best_eval_loss = float(diagnostic_score)
                best_iter = it
                best_diag_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in self.value_net.state_dict().items()
                }

            diag_res = {}
            if diag is not None and (it == 1 or diag_every <= 1 or it % diag_every == 0 or it == outer_iters):
                diag_res = eval_diag_metrics(self.value_net, diag)

            # Uniform-ellipticity and projection diagnostics use one fixed
            # Q_col set for every outer iteration/training seed. This keeps
            # changes in the recorded range attributable to the policy, not
            # to a changing diagnostic sample.
            # Projection activation is reported on the same fixed Q_ev grid
            # as the E1 control ranges. Q_col below is reserved for the
            # uniform-ellipticity range required by the supplement.
            frozen_clip = _clip_fraction_pi(frozen_metric_comp["pi_raw"])
            frozen_c_clip = _consumption_clip_fractions(frozen_metric_comp)
            frozen_var_min = frozen_var_max = ""
            if diag_col is not None:
                if policy_source is None:
                    _, pi_diag_frozen, _ = self.initialize_policy(
                        diag_col["t"], diag_col["y"],
                        pi_init_method, pi_init_scale, c_init_method)
                else:
                    _, pi_diag_frozen, _ = self._policy_components(
                        diag_col["t"], diag_col["y"], net=policy_source)
                variance = _diffusion_variance(pi_diag_frozen)
                frozen_var_min = float(variance.min().item())
                frozen_var_max = float(variance.max().item())

            save_this = ((e3b_checkpoints and (it <= 10 or it % 10 == 0))
                         or (not e3b_checkpoints and save_iterate_every > 0
                             and it % save_iterate_every == 0))
            if save_this and not timing_mode:
                save_iterate_checkpoint(
                    it,
                    "e3b-schedule" if e3b_checkpoints else "periodic-schedule",
                )

            last = eval_loss_hist[-1]
            outer_row = {
                "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                "run_tag": ARGS.run_tag, "outer_iter": it,
                "algorithm_outer_iter": algorithm_outer_iter,
                "e6_phase": (
                    "target" if is_target_branch
                    else ("warmup" if e6_role == "warmup" else "standard")
                ),
                "warmup_excluded_from_achieved_pres": int(is_target_branch),
                "total_loss": last["total"], "pde_loss": last["pde"],
                "terminal_loss": last["terminal"], "monotonicity_loss": last["mono"],
                "concavity_loss": last["conc"], "eta_loss": last["eta"],
                "train_pres": last.get("train_pres", ""),
                "val_pde_rms": eval_info["val_pde_rms_post_restore"],
                "val_terminal_rms": eval_info["val_terminal_rms_post_restore"],
                "val_pres": eval_info["val_pres_post_restore"],
                "val_pres_at_stop": eval_info["val_pres_at_stop"],
                "val_pres_at_stop_epoch": eval_info["val_pres_at_stop_epoch"],
                "val_pres_post_restore": eval_info["val_pres_post_restore"],
                "achieved_pres": achieved_pres,
                "inner_epochs_used": eval_info["epochs_used"],
                "n_resamples": eval_info["n_resamples"],
                "target_reached": int(eval_info["target_reached"]),
                "target_reached_post_restore": int(
                    eval_info["target_reached_post_restore"]),
                "training_target_crossed": int(
                    eval_info["training_target_crossed"]),
                "sel_best_pres": eval_info["sel_best_pres"],
                "sel_best_qres_pres": eval_info["sel_best_qres_pres"],
                "sel_best_epoch": eval_info["sel_best_epoch"],
                "sel_best_lr": eval_info["sel_best_lr"],
                "sel_checks": eval_info["sel_checks"], "sel_stopped": eval_info["sel_stopped"],
                "sel_eligible_checks": eval_info["sel_eligible_checks"],
                "sel_ineligible_checks": eval_info["sel_ineligible_checks"],
                "sel_restored": eval_info["sel_restored"],
                "q_sel_seed": q_sel_seed,
                "lr_end_before_restore": eval_info["lr_end_before_restore"],
                "lr_best_checkpoint": eval_info["lr_best_checkpoint"],
                "lr_after_restore": eval_info["lr_after_restore"],
                "lr_carried_next": eval_info["lr_carried_next"],
                "lr_next_outer_policy": eval_info["lr_next_outer_policy"],
                "lr_outer_start": lr_outer_start,
                "lr_outer_restart_at_max": int(restart_carry_at_max),
                "scheduler_reset_at_outer_start": int(
                    self.lr_schedule != "fixed"
                ),
                "pi_diff": pi_diff, "c_diff": c_diff,
                "pi_vs_closed_form": pi_vs_cf, "c_vs_closed_form": c_vs_cf,
                "control_metric_scope": control_metric_scope,
                "control_metric_points": int(metric_t.shape[0]),
                "diffusion_var_min_frozen": frozen_var_min,
                "diffusion_var_max_frozen": frozen_var_max,
                "clip_frac_pi_frozen": frozen_clip,
                "clip_frac_kappa_low_frozen": frozen_c_clip["clip_frac_kappa_low"],
                "clip_frac_kappa_high_frozen": frozen_c_clip["clip_frac_kappa_high"],
                "clip_frac_c_level_low_frozen": frozen_c_clip["clip_frac_c_level_low"],
                "clip_frac_c_level_high_frozen": frozen_c_clip["clip_frac_c_level_high"],
                "lr": cur_lr,
            }
            outer_row.update(diag_res)
            outer_row.update(frozen_control_ranges)
            results["outer_rows"].append(outer_row)

            if verbose:
                print(
                    f"  c mean: {c_new.mean().item():.4f}, std: {c_new.std().item():.4f} | "
                    f"c diff: {c_diff:.2e} | vs cf: {c_vs_cf:.2e}"
                )
                print(
                    f"  pi mean: {pi_new.mean().item():.4f}, std: {pi_new.std().item():.4f} | "
                    f"pi diff: {pi_diff:.2e} | vs cf: {pi_vs_cf:.2e}"
                )
                print(
                    f"  Eval(last): {last_eval_loss:.3e} | PDE={last['pde']:.3e} | Term={last['terminal']:.3e} | "
                    f"Eta={last['eta']:.3e} | LR={cur_lr:.2e}"
                )

            if stopper is not None:
                stop, meta = stopper.update(it, float(last["pde"]))
                if stop:
                    results["stopped_early"] = True
                    results["stop_info"] = meta
                    break

        # Official result is the final outer iterate.  Diagnostic best is
        # written once and never restored.
        results["pres_max"] = max(results["val_pres"]) if results["val_pres"] else None
        results["total_inner_steps"] = int(sum(results["inner_epochs_used"]))
        results["all_targets_reached"] = bool(results["target_flags"]) and all(results["target_flags"])
        results["all_training_targets_crossed"] = (
            bool(results["training_target_flags"])
            and all(results["training_target_flags"])
        )
        final_state = None
        final_it = 0
        final_state_sha256 = None
        final_file_sha256 = None
        last_file_sha256 = None
        final_iterate_record = None
        if results["outer_rows"]:
            final_it = int(results["outer_rows"][-1]["outer_iter"])
            final_state = snapshot_value_state()
            final_state_sha256 = mxu.canonical_state_dict_sha256(final_state)
            torch.save(final_state, final_model_path)
            torch.save(final_state, last_model_path)
            final_file_sha256 = mxu.sha256_file(final_model_path)
            last_file_sha256 = mxu.sha256_file(last_model_path)
        if best_diag_state is not None and not timing_mode:
            torch.save(best_diag_state, best_model_path)
            print(
                f"Saved diagnostic best held-out state: outer={best_iter}, "
                f"score={best_eval_loss:.3e} -> {best_model_path}")
        if (e3b_checkpoints or save_iterate_every > 0) and final_state is not None:
            # The official final state is always present in an enabled iterate
            # schedule, even when the requested period does not divide the
            # number of completed outer iterations.
            final_iterate_record = save_iterate_checkpoint(
                final_it, "official-final", final_state, update_manifest=False)
        if results["outer_rows"]:
            print(f"\nSaved official FINAL iterate: {final_model_path}")

        if final_state is not None:
            final_artifacts = {
                "final": {
                    "path": relative_weight_path(final_model_path),
                    "state_sha256": final_state_sha256,
                    "file_sha256": final_file_sha256,
                },
                "last": {
                    "path": relative_weight_path(last_model_path),
                    "state_sha256": final_state_sha256,
                    "file_sha256": last_file_sha256,
                },
            }
            if final_iterate_record is not None:
                final_artifacts["iterate"] = {
                    "path": final_iterate_record["path"],
                    "state_sha256": final_iterate_record["state_sha256"],
                    "file_sha256": final_iterate_record["file_sha256"],
                }
            checkpoint_manifest["official_final"] = {
                "outer_iter": final_it,
                "state_sha256": final_state_sha256,
                "artifacts": final_artifacts,
            }
            manifest_status = (
                "stopped_early" if results.get("stopped_early", False) else "complete"
            )
        else:
            manifest_status = (
                "stopped_early" if results.get("stopped_early", False)
                else "no_completed_outer"
            )
        checkpoint_manifest["completed_outer_iters"] = len(results["outer_rows"])
        write_checkpoint_manifest(manifest_status)

        results["checkpoint_provenance"] = {
            "checkpoint_manifest_path": manifest_path,
            "checkpoint_manifest_sha256": mxu.sha256_file(manifest_path),
            "checkpoint_manifest_schema_version": checkpoint_manifest["schema_version"],
            "completed_outer_iters": len(results["outer_rows"]),
            "final_outer_iter": final_it,
            "final_checkpoint_state_sha256": final_state_sha256,
            "final_checkpoint_file_sha256": final_file_sha256,
            "last_checkpoint_state_sha256": final_state_sha256,
            "last_checkpoint_file_sha256": last_file_sha256,
            "final_iterate_state_sha256": (
                final_iterate_record["state_sha256"] if final_iterate_record is not None else None
            ),
            "final_iterate_file_sha256": (
                final_iterate_record["file_sha256"] if final_iterate_record is not None else None
            ),
        }

        return results


# =============================================================================
# 10) Evaluation helpers (grid is still 1D in W)
# =============================================================================
def eval_pinn_on_grid(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate PINN on a (t, W) grid.
    Returns:
      tt, ww, V_pinn, c_pinn, pi_pinn (Nt,Nw,N), pi_norm (Nt,Nw)
    """
    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.linspace(x_min, x_max, Nw)
    tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")
    V, c, pi = eval_model_on_points(value_net, tt, ww)
    V_grid = V.reshape(Nt, Nw)
    c_grid = c.reshape(Nt, Nw)
    pi_grid = pi.reshape(Nt, Nw, N_ASSETS)
    pi_norm_grid = np.linalg.norm(pi_grid, axis=2)
    return tt, ww, V_grid, c_grid, pi_grid, pi_norm_grid

def eval_pinn_on_grid_margin(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
    margin: float = 0.0,
    eval_w_min: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid evaluation on an eval window shrunk by HALF-WIDTH `margin`.

    Only y=log W is shrunk; time keeps the full [0,T) range.
    """
    window = mxu.resolve_eval_window(
        y_min, y_max, margin, eval_w_min=eval_w_min)
    y_lo, y_hi = window["ev_y_min"], window["ev_y_max"]
    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))
    tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")
    V, c, pi = eval_model_on_points(value_net, tt, ww)
    V_grid = V.reshape(Nt, Nw)
    c_grid = c.reshape(Nt, Nw)
    pi_grid = pi.reshape(Nt, Nw, N_ASSETS)
    pi_norm_grid = np.linalg.norm(pi_grid, axis=2)
    return tt, ww, V_grid, c_grid, pi_grid, pi_norm_grid


def compute_metrics(
    V_pinn, c_pinn, pi_pinn, V_cf, c_cf, pi_cf, *,
    Vw_pinn, Vww_pinn, Vw_cf, Vww_cf,
) -> Dict[str, float]:
    """Common all-margin value/bundle/control metrics.

    The bundle is the reduced wealth-coordinate pair ``(V_w,V_ww)``.  This
    Merton problem has no additional factor coordinate and hence no ``V_wx``.
    """
    return mem.full_window_metrics(
        V_pinn, c_pinn, pi_pinn, Vw_pinn, Vww_pinn,
        V_cf, c_cf, pi_cf, Vw_cf, Vww_cf,
    )


def eval_model_bundle_on_points(
    value_net: nn.Module, t_np: np.ndarray, w_np: np.ndarray, chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``V,V_w,V_ww,c,pi`` on arbitrary paired points."""
    was_training = value_net.training
    value_net.eval()
    V_parts, Vw_parts, Vww_parts, c_parts, pi_parts = [], [], [], [], []
    t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    w_np = np.asarray(w_np, dtype=np.float32).reshape(-1, 1)
    for start in range(0, len(t_np), chunk):
        t = torch.tensor(t_np[start:start + chunk], device=device)
        y = torch.tensor(
            np.log(w_np[start:start + chunk]), device=device, requires_grad=True)
        V = value_net(t, y)
        V_y = torch.autograd.grad(
            V, y, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        V_yy = torch.autograd.grad(
            V_y, y, torch.ones_like(V_y), create_graph=False, retain_graph=True)[0]
        W = torch.exp(y)
        V_w = V_y / W
        V_ww = (V_yy - V_y) / (W ** 2)
        c = compute_c_from_foc_log(V_y, y)
        pi = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu)
        V_parts.append(V.detach().cpu().numpy())
        Vw_parts.append(V_w.detach().cpu().numpy())
        Vww_parts.append(V_ww.detach().cpu().numpy())
        c_parts.append(c.detach().cpu().numpy())
        pi_parts.append(pi.detach().cpu().numpy())
    if was_training:
        value_net.train()
    return (
        np.concatenate(V_parts), np.concatenate(Vw_parts),
        np.concatenate(Vww_parts), np.concatenate(c_parts),
        np.concatenate(pi_parts),
    )


def eval_model_on_points(
    value_net: nn.Module, t_np: np.ndarray, w_np: np.ndarray, chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V,c,pi on arbitrary paired points."""
    V, _Vw, _Vww, c, pi = eval_model_bundle_on_points(
        value_net, t_np, w_np, chunk=chunk)
    return V, c, pi


def eval_metrics_on_points(
    value_net: nn.Module, t_np: np.ndarray, w_np: np.ndarray,
) -> Dict[str, float]:
    """Evaluate the common E9 metric schema on one fixed point set."""
    t_eval = np.asarray(t_np, dtype=np.float64).reshape(-1, 1)
    w_eval = np.asarray(w_np, dtype=np.float64).reshape(-1, 1)
    V, Vw, Vww, c, pi = eval_model_bundle_on_points(value_net, t_eval, w_eval)
    V_cf, Vw_cf, Vww_cf = closed_form_wealth_bundle(t_eval, w_eval)
    c_cf = closed_form_c(t_eval, w_eval)
    pi_cf = np.broadcast_to(pi_star_np.reshape(1, -1), pi.shape)
    return compute_metrics(
        V, c, pi, V_cf, c_cf, pi_cf,
        Vw_pinn=Vw, Vww_pinn=Vww, Vw_cf=Vw_cf, Vww_cf=Vww_cf,
    )


def eval_fulldim_test_metrics(
    value_net: nn.Module,
    n_points: int,
    margins: List[float],
    eval_w_min: Optional[float] = None,
) -> Dict[float, Dict[str, float]]:
    """Fixed corresponding (t,y) test points for every nested window."""
    rng = np.random.default_rng(104729 + int(MARKET_SEED))
    u_t = rng.random((int(n_points), 1))
    u_y = rng.random((int(n_points), 1))
    t_np = u_t * (T_FINAL - 1e-3)
    output: Dict[float, Dict[str, float]] = {}
    for margin in margins:
        window = mxu.resolve_eval_window(
            y_min, y_max, float(margin), eval_w_min=eval_w_min)
        y_lo, y_hi = window["ev_y_min"], window["ev_y_max"]
        y_np = y_lo + u_y * (y_hi - y_lo)
        w_np = np.exp(y_np)
        output[float(margin)] = eval_metrics_on_points(value_net, t_np, w_np)
    return output

def plot_convergence(results: Dict, save_path: str | None = None, show: bool = True):
    n_iters = len(results["pi_diff"])
    iters = np.arange(1, n_iters + 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # row1
    axes[0, 0].semilogy(iters, results["c_vs_closed_form"], "b-", lw=1.5)
    axes[0, 0].set_title("Consumption vs Closed-form c*")
    axes[0, 0].set_xlabel("Outer Iter")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(iters, results["pi_vs_closed_form"], "r-", lw=1.5)
    axes[0, 1].set_title("Portfolio vs Closed-form pi*")
    axes[0, 1].set_xlabel("Outer Iter")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].semilogy(iters, results["eval_loss"], "g-", lw=1.5)
    axes[0, 2].set_title("Policy Evaluation Loss")
    axes[0, 2].set_xlabel("Outer Iter")
    axes[0, 2].grid(True, alpha=0.3)

    # row2
    axes[1, 0].semilogy(iters, results["c_diff"], "b-", lw=1.5)
    axes[1, 0].set_title(r"$||c_{n+1}-c_n||^2$")
    axes[1, 0].set_xlabel("Outer Iter")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(iters, results["pi_diff"], "r-", lw=1.5)
    axes[1, 1].set_title(r"$||\pi_{n+1}-\pi_n||^2$")
    axes[1, 1].set_xlabel("Outer Iter")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].semilogy(iters, results["eta_loss"], "m-", lw=1.5)
    axes[1, 2].set_title(r"Eta loss: $(\eta-\gamma)^2$, $\eta=1-\frac{V_{yy}}{V_y}$")
    axes[1, 2].set_xlabel("Outer Iter")
    axes[1, 2].grid(True, alpha=0.3)

    # row3
    c_mean = np.asarray(results["c_mean"])
    c_std = np.asarray(results["c_std"])
    axes[2, 0].plot(iters, c_mean, "b-", lw=1.5)
    axes[2, 0].fill_between(iters, c_mean - c_std, c_mean + c_std, alpha=0.3)
    axes[2, 0].set_title("c mean ± std")
    axes[2, 0].set_xlabel("Outer Iter")
    axes[2, 0].grid(True, alpha=0.3)

    pi_mean = np.asarray(results["pi_mean"])
    pi_std = np.asarray(results["pi_std"])
    axes[2, 1].plot(iters, pi_mean, "r-", lw=1.5, label="pi mean")
    axes[2, 1].axhline(float(pi_star.mean().item()), color="k", ls="--", lw=1.5, label="pi*_mean")
    axes[2, 1].fill_between(iters, pi_mean - pi_std, pi_mean + pi_std, alpha=0.3)
    axes[2, 1].set_title("pi mean ± std (over points+assets)")
    axes[2, 1].set_xlabel("Outer Iter")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].axis("off")
    axes[2, 2].text(0.5, 0.85, f"iters: {n_iters}", ha="center")
    axes[2, 2].text(0.5, 0.70, f"final eval loss: {results['eval_loss'][-1]:.2e}", ha="center")
    axes[2, 2].text(0.5, 0.55, f"final c vs cf: {results['c_vs_closed_form'][-1]:.2e}", ha="center")
    axes[2, 2].text(0.5, 0.40, f"final pi vs cf: {results['pi_vs_closed_form'][-1]:.2e}", ha="center")
    axes[2, 2].text(0.5, 0.25, f"pi* ||.||2: {pi_star.norm().item():.3f}", ha="center")
    axes[2, 2].set_title("Summary")

    plt.suptitle("PI-PINN Convergence (Multi-Asset, logW, with Consumption)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_heatmaps(
    tt: np.ndarray,
    ww: np.ndarray,
    V_pinn: np.ndarray,
    c_pinn: np.ndarray,
    pi_norm_pinn: np.ndarray,
    save_path: str | None = None,
    show: bool = True,
):
    V_cf = closed_form_V(tt, ww)
    c_cf = closed_form_c(tt, ww)
    pi_norm_cf = np.linalg.norm(pi_star_np)
    pi_norm_cf_grid = np.full_like(pi_norm_pinn, pi_norm_cf)

    V_err = V_pinn - V_cf
    c_err = c_pinn - c_cf
    pi_err = pi_norm_pinn - pi_norm_cf_grid
    plot_extent = [float(np.min(ww)), float(np.max(ww)),
                   float(np.min(tt)), float(np.max(tt))]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    def heat(ax, Z, title, cmap="jet", vmin=None, vmax=None, div=False):
        if div:
            abs_max = np.percentile(np.abs(Z), 98)
            abs_max = max(abs_max, 1e-10)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=plot_extent,
                interpolation="bilinear",
                cmap="RdBu_r",
                norm=norm,
            )
        else:
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=plot_extent,
                interpolation="bilinear",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        ax.set_title(title)
        ax.set_xlabel("Wealth W")
        ax.set_ylabel("t")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Value row
    vmin, vmax = min(V_pinn.min(), V_cf.min()), max(V_pinn.max(), V_cf.max())
    heat(axes[0, 0], V_pinn, "V (PI-PINN)", vmin=vmin, vmax=vmax)
    heat(axes[0, 1], V_cf, "V (closed-form)", vmin=vmin, vmax=vmax)
    heat(axes[0, 2], V_err, "V error", div=True)

    # Consumption row
    vmin, vmax = min(c_pinn.min(), c_cf.min()), max(c_pinn.max(), c_cf.max())
    heat(axes[1, 0], c_pinn, "c (PI-PINN)", vmin=vmin, vmax=vmax)
    heat(axes[1, 1], c_cf, "c (closed-form)", vmin=vmin, vmax=vmax)
    heat(axes[1, 2], c_err, "c error", div=True)

    # Portfolio norm row (compact)
    vmin, vmax = min(pi_norm_pinn.min(), pi_norm_cf_grid.min()), max(pi_norm_pinn.max(), pi_norm_cf_grid.max())
    heat(axes[2, 0], pi_norm_pinn, r"||pi||_2 (PI-PINN)", vmin=vmin, vmax=vmax)
    heat(axes[2, 1], pi_norm_cf_grid, r"||pi*||_2 (closed)", vmin=vmin, vmax=vmax)
    heat(axes[2, 2], pi_err, r"||pi|| error", div=True)

    plt.suptitle("Multi-Asset Merton (with Consumption) - PI-PINN(logW) vs Closed-form")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def _save_e6_warmup_state(
    solver: PIPINN_MultiAsset_Consumption_LogW,
    results: Dict,
    path: str,
) -> Dict:
    """Persist the exact common v~_0 state used to fork all E6 targets."""
    rows = list(results.get("outer_rows", []))
    if len(rows) != 1:
        raise RuntimeError(
            f"E6 warm-up must complete exactly one outer solve, got {len(rows)}")
    achieved = rows[0].get("val_pres_post_restore")
    if not isinstance(achieved, (int, float)) or not math.isfinite(float(achieved)):
        raise RuntimeError("E6 warm-up has no finite post-restore p_res")
    achieved = float(achieved)
    if achieved <= 0.0:
        raise RuntimeError(
            "E6 warm-up post-restore p_res must be strictly positive")
    if achieved > 1.0 * (1.0 + 1e-9):
        raise RuntimeError(
            "E6 warm-up failed its admissibility target: "
            f"post-restore p_res={achieved:.6e} > 1")
    if not bool(rows[0].get("target_reached_post_restore", False)):
        raise RuntimeError(
            "E6 warm-up checkpoint is not marked target-reached post restore")

    model_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in solver.value_net.state_dict().items()
    }
    model_hash = mxu.canonical_state_dict_sha256(model_state)
    optimizer_state = _cpu_tree(solver.optimizer.state_dict())
    optimizer_hash = _canonical_tree_sha256(optimizer_state)
    compat = _e6_compatibility_payload(ARGS)
    compat_hash = _canonical_json_sha256(compat)
    current_lrs = [
        float(group["lr"]) for group in solver.optimizer.param_groups
    ]
    torch_cpu_rng_state = torch.get_rng_state().cpu().clone()
    cuda_rng_state = None
    if device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(device).cpu().clone()
    numpy_rng_state = copy.deepcopy(np.random.get_state())
    rng_payload = {
        "torch_cpu_rng_state": torch_cpu_rng_state,
        "torch_cuda_rng_state": cuda_rng_state,
        "numpy_rng_state": numpy_rng_state,
        "rng_device_type": device.type,
    }
    rng_hash = _canonical_tree_sha256(rng_payload)
    warm_start_id = _canonical_json_sha256({
        "protocol": E6_WARM_START_PROTOCOL,
        "compatibility_sha256": compat_hash,
        "seed": int(SEED),
        "market_seed": int(MARKET_SEED),
        "model_sha256": model_hash,
        "optimizer_sha256": optimizer_hash,
        "rng_sha256": rng_hash,
    })
    path = os.path.abspath(path)
    payload = {
        "schema_version": E6_WARMUP_BUNDLE_SCHEMA_VERSION,
        "kind": E6_WARMUP_BUNDLE_KIND,
        "created_at": mxu.now_iso(),
        "warm_start_protocol": E6_WARM_START_PROTOCOL,
        "warm_start_source": path,
        "warm_start_id": warm_start_id,
        "trainer_source_sha256": TRAINER_METADATA["trainer_source_sha256"],
        "compatibility": compat,
        "compatibility_sha256": compat_hash,
        "seed": int(SEED),
        "market_seed": int(MARKET_SEED),
        "n_assets": int(N_ASSETS),
        "model_state": model_state,
        "model_state_sha256": model_hash,
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_hash,
        "outer_count": int(solver._outer_count),
        "current_lrs": current_lrs,
        "scheduler_contract": (
            "not-restored;within-frozen-pde-scheduler-recreated-every-outer;"
            "target-branch-outer-start-lr-restarts-at-carry-lr-max;"
            "model-preserved;adam-moments-preserved-when-adam-reset-keep"
        ),
        "torch_cpu_rng_state": torch_cpu_rng_state,
        "torch_cuda_rng_state": cuda_rng_state,
        "rng_device_type": device.type,
        "numpy_rng_state": numpy_rng_state,
        "rng_state_sha256": rng_hash,
        "warmup_pres_target": 1.0,
        "warmup_achieved_pres": achieved,
        "warmup_inner_steps": int(results.get("total_inner_steps", 0)),
        "warmup_outer_row": copy.deepcopy(rows[0]),
        "warmup_residual_semantics": (
            "single_warmup_outer_post_restore_fixed_qres"
        ),
    }
    if int(payload["outer_count"]) != 1:
        raise RuntimeError(
            "E6 warm-up optimizer outer counter must equal one, got "
            f"{payload['outer_count']}")

    _torch_save_atomic(payload, path)
    bundle_hash = mxu.sha256_file(path)
    return {
        "e6_role": "warmup",
        "e6_warm_start_protocol": E6_WARM_START_PROTOCOL,
        "e6_warm_start_source": path,
        "e6_warm_start_id": warm_start_id,
        "e6_warm_start_model_sha256": model_hash,
        "e6_warm_start_optimizer_sha256": optimizer_hash,
        "e6_warm_start_rng_sha256": rng_hash,
        "e6_warm_start_bundle_sha256": bundle_hash,
        "e6_warm_start_loaded_bundle_sha256": bundle_hash,
        "e6_warmup_target": 1.0,
        "e6_warmup_post_restore_pres": achieved,
        "e6_warmup_optimizer_steps": payload["warmup_inner_steps"],
        "e6_warmup_bundle_path": path,
        "e6_warmup_bundle_file_sha256": bundle_hash,
        "e6_warmup_model_state_sha256": payload["model_state_sha256"],
        "e6_warmup_compatibility_sha256": compat_hash,
        "e6_warmup_achieved_pres": achieved,
        "e6_warmup_inner_steps": payload["warmup_inner_steps"],
        "e6_warmup_pres_target": 1.0,
    }


def _move_optimizer_state_to_device(
    optimizer: optim.Optimizer,
    target_device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(target_device)


def _restore_e6_warmup_state(
    solver: PIPINN_MultiAsset_Consumption_LogW,
    bundle: Dict,
    path: str,
) -> Dict:
    """Restore common model, Adam moments/LR, outer counter, and RNG state."""
    if bundle.get("trainer_source_sha256") != TRAINER_METADATA["trainer_source_sha256"]:
        raise ValueError(
            "E6 warm-up trainer source differs from the target-branch trainer")
    stored_device_type = str(bundle.get("rng_device_type", ""))
    if stored_device_type != device.type:
        raise ValueError(
            "E6 warm-up and target branch must use the same device type for "
            f"exact RNG continuation: stored={stored_device_type!r}, current={device.type!r}"
        )

    solver.value_net.load_state_dict(bundle["model_state"])
    solver.optimizer.load_state_dict(bundle["optimizer_state"])
    _move_optimizer_state_to_device(solver.optimizer, solver.device)
    solver._outer_count = int(bundle["outer_count"])
    stored_lrs = [float(value) for value in bundle.get("current_lrs", [])]
    if len(stored_lrs) != len(solver.optimizer.param_groups):
        raise ValueError("E6 warm-up LR group count does not match optimizer")
    for group, stored_lr in zip(solver.optimizer.param_groups, stored_lrs):
        group["lr"] = stored_lr

    torch.set_rng_state(bundle["torch_cpu_rng_state"].cpu())
    if device.type == "cuda":
        cuda_state = bundle.get("torch_cuda_rng_state")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("CUDA E6 warm-up bundle is missing CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)
    np.random.set_state(bundle["numpy_rng_state"])

    path = os.path.abspath(path)
    bundle_hash = mxu.sha256_file(path)
    return {
        "e6_role": "target_branch",
        "e6_warm_start_protocol": str(bundle["warm_start_protocol"]),
        "e6_warm_start_source": str(bundle["warm_start_source"]),
        "e6_warm_start_id": str(bundle["warm_start_id"]),
        "e6_warm_start_model_sha256": str(bundle["model_state_sha256"]),
        "e6_warm_start_optimizer_sha256": str(
            bundle["optimizer_state_sha256"]),
        "e6_warm_start_rng_sha256": str(bundle["rng_state_sha256"]),
        "e6_warm_start_bundle_sha256": bundle_hash,
        "e6_warm_start_loaded_bundle_sha256": bundle_hash,
        "e6_warmup_target": float(bundle["warmup_pres_target"]),
        "e6_warmup_post_restore_pres": float(bundle["warmup_achieved_pres"]),
        "e6_warmup_optimizer_steps": int(bundle.get("warmup_inner_steps", 0)),
        "e6_target_phase_outer_count": int(ARGS.outer_iters),
        "e6_target_phase_start_algorithm_iter": 2,
        "first_target_policy_source": E6_WARM_START_POLICY_SOURCE,
        "e6_warmup_bundle_path": path,
        "e6_warmup_bundle_file_sha256": bundle_hash,
        "e6_warmup_model_state_sha256": bundle["model_state_sha256"],
        "e6_warmup_compatibility_sha256": bundle["compatibility_sha256"],
        "e6_warmup_achieved_pres": float(bundle["warmup_achieved_pres"]),
        "e6_warmup_inner_steps": int(bundle.get("warmup_inner_steps", 0)),
        "e6_warmup_pres_target": float(bundle["warmup_pres_target"]),
        "e6_warmup_outer_count": int(bundle["outer_count"]),
        "e6_warmup_excluded_from_outer_history": True,
        "e6_warmup_excluded_from_pres_max": True,
    }


# =============================================================================
# 11) Main
# =============================================================================
def main():
    # Timing runs must suppress final figures automatically; persisting the
    # effective flag lets the E8 aggregator verify that evaluation peak memory
    # was not contaminated by plotting allocations.
    ARGS.skip_figures_requested = bool(ARGS.skip_figures)
    ARGS.skip_figures = bool(
        ARGS.skip_figures or ARGS.skip_plots or ARGS.timing_mode)
    ARGS.e6_phase = (
        "target" if ARGS.e6_role == "target_branch" else ARGS.e6_role
    )
    if ARGS.e6_warm_start:
        ARGS.e6_warm_start = os.path.abspath(ARGS.e6_warm_start)
    if ARGS.e6_warmup_bundle:
        ARGS.e6_warmup_bundle = os.path.abspath(ARGS.e6_warmup_bundle)

    # Validate a branch bundle before rotating any existing run artifacts.
    # This makes a typo or incompatible warm start fail without disturbing an
    # earlier successful run sharing the same tag.
    e6_loaded_bundle = None
    e6_warm_start_provenance: Dict = {}
    if ARGS.e6_role == "target_branch" and not ARGS.eval_only:
        e6_loaded_bundle = _load_e6_warmup_bundle(ARGS.e6_warm_start)
    out_dir = os.path.join(ARGS.output_root, ARGS.run_tag)
    weight_dir = ARGS.weight_root or os.path.join(out_dir, "weights")
    recorder = mxu.ExperimentRecorder(out_dir, weight_dir, ARGS)
    skip_figures = bool(ARGS.skip_figures)
    metrics_final_path = recorder.metrics_csv
    if ARGS.eval_only and not ARGS.skip_eval:
        recorder.metrics_csv = metrics_final_path + ".eval_tmp"
        if os.path.exists(recorder.metrics_csv):
            os.remove(recorder.metrics_csv)

    if ARGS.pres_target is not None and ARGS.val_points <= 0:
        raise ValueError("--pres-target requires --val-points > 0")
    if ARGS.inner_best_restore and ARGS.sel_points <= 0:
        raise ValueError("--inner-best-restore=1 requires --sel-points > 0")
    if ARGS.pi_clip_abs is not None and ARGS.pi_clip_abs <= 0.0:
        raise ValueError("--pi-clip-abs must be positive or none")

    stopper = None
    if not ARGS.eval_only and ARGS.pde_stop_threshold is not None:
        stopper = mxu.PDEEarlyStopper(
            threshold=float(ARGS.pde_stop_threshold),
            start_outer=int(ARGS.pde_stop_start_outer),
            patience=int(ARGS.pde_stop_patience),
            stop_flag_path=str(ARGS.stop_flag_path or ""), recorder=recorder,
            run_tag=ARGS.run_tag, model_type=ARGS.model_type)
        if stopper.shared_stop_exists():
            # A pre-existing shared flag means no new training attempt.  Check
            # it before rotating any same-tag checkpoints from an older run.
            info = stopper.mark_from_existing_flag(outer_iter=0, pde_loss=None)
            print(f"[early-stop] shared stop flag already exists; skipping run: {info}")
            return

    if ARGS.eval_only:
        recorder.save_config_eval(extra=TRAINER_METADATA)
    else:
        recorder.rotate_training_logs()
        recorder.rotate_training_checkpoints()
        recorder.save_config(extra=TRAINER_METADATA)
        recorder.save_market_snapshot(
            mu_excess=mu_excess_np, Sigma_safe=Sigma_np, chol=chol_Sigma_np,
            Sigma_inv_mu=Sigma_inv_mu_np, pi_star=pi_star_np,
            Theta=np.array([Theta]), nu=np.array([nu]),
            gamma=np.array([gamma_risk]), r=np.array([r_rate]),
            rho_discount=np.array([rho_discount]), epsilon=np.array([epsilon]),
            T=np.array([T_FINAL]), w_min=np.array([x_min]), w_max=np.array([x_max]),
            n_assets=np.array([N_ASSETS]), market_seed=np.array([MARKET_SEED]),
            seed=np.array([SEED]),
        )
        if ARGS.e6_role == "warmup":
            # A forced rerun must not leave a prior successful bundle at the
            # configured path if this new warm-up attempt later fails.
            _archive_e6_warmup_bundle(ARGS.e6_warmup_bundle)

    try:
        eta_clip = None if str(ARGS.eta_clip).lower() == "none" else float(ARGS.eta_clip)
        eta_focus_w = None if str(ARGS.eta_focus_w).lower() == "none" else float(ARGS.eta_focus_w)
        if eta_clip is not None and eta_clip <= 0.0:
            raise ValueError("--eta-clip must be positive or none")

        start = time.time()
        solver = PIPINN_MultiAsset_Consumption_LogW(
            value_hidden=int(ARGS.value_hidden),
            value_depth=int(ARGS.value_depth),
            lr=float(ARGS.lr),
            scheduler_patience=int(ARGS.scheduler_patience),
            scheduler_factor=float(ARGS.scheduler_factor),
            scheduler_min_lr=float(ARGS.scheduler_min_lr),
            clip_abs=pi_clip_abs,
            c_min=c_min_bound,
            c_max=c_max_bound,
            lr_schedule=str(ARGS.lr_schedule),
            adam_reset=str(ARGS.adam_reset),
            carry_lr_min=float(ARGS.carry_lr_min),
            carry_lr_max=float(ARGS.carry_lr_max),
            device=device,
        )
        if e6_loaded_bundle is not None:
            e6_warm_start_provenance = _restore_e6_warmup_state(
                solver, e6_loaded_bundle, ARGS.e6_warm_start)
            print(
                "[E6] restored common warm-up bundle: "
                f"{ARGS.e6_warm_start} "
                f"(p_res={e6_warm_start_provenance['e6_warmup_achieved_pres']:.6e})"
            )
        lr_protocol_fields = {
            "pres_target_lr_restart": bool(
                ARGS.pres_target is not None
                and ARGS.lr_schedule == "carry_plateau"
            ),
            "outer_start_lr_policy": (
                "restart-at-carry-lr-max"
                if (
                    ARGS.pres_target is not None
                    and ARGS.lr_schedule == "carry_plateau"
                )
                else "ordinary-schedule"
            ),
            "plateau_scheduler_state": (
                "reset-at-every-outer"
                if ARGS.lr_schedule != "fixed"
                else "not-applicable"
            ),
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        results = None
        loaded_weight_path = None
        train_gpu_peak = None
        if not ARGS.eval_only:
            running_e6_fields = {
                "e6_role": str(ARGS.e6_role),
                "e6_phase": str(ARGS.e6_phase),
            }
            running_e6_fields.update(e6_warm_start_provenance)
            running_e6_fields.update(lr_protocol_fields)
            recorder.write_status("running", **running_e6_fields)
            results = solver.run_policy_iteration(
                outer_iters=int(ARGS.outer_iters),
                eval_epochs=int(ARGS.eval_epochs),
                batch_size=int(ARGS.batch_size),
                terminal_frac=float(ARGS.terminal_frac),
                w_terminal=float(ARGS.w_terminal),
                w_shape=float(ARGS.w_shape),
                w_eta=float(ARGS.w_eta),
                eta_focus_w=eta_focus_w,
                eta_clip=eta_clip,
                pi_init_method=str(ARGS.pi_init_method),
                pi_init_scale=float(ARGS.pi_init_scale),
                c_init_method=str(ARGS.c_init_method),
                print_every_outer=int(ARGS.print_every_outer),
                print_every_eval=int(ARGS.print_every_eval),
                verbose_detail=bool(ARGS.print_every_eval > 0),
                inner_best_restore=bool(ARGS.inner_best_restore),
                sel_points=int(ARGS.sel_points),
                sel_terminal_points=int(ARGS.sel_terminal_points),
                sel_every=int(ARGS.sel_every), sel_patience=int(ARGS.sel_patience),
                pe_resample_every=int(ARGS.pe_resample_every),
                pres_target=ARGS.pres_target,
                val_points=int(ARGS.val_points),
                val_terminal_points=int(ARGS.val_terminal_points),
                val_every=int(ARGS.val_every), val_seed=int(MARKET_SEED),
                diag_points=int(ARGS.diag_points),
                diag_margin=mxu.parse_eval_margins(ARGS.eval_margin)[0],
                diag_every=int(ARGS.diag_every),
                save_iterate_every=int(ARGS.save_iterate_every),
                e3b_checkpoints=bool(ARGS.e3b_checkpoints),
                timing_mode=bool(ARGS.timing_mode),
                e6_role=str(ARGS.e6_role),
                e6_warm_start_provenance=e6_warm_start_provenance,
                weight_dir=weight_dir, recorder=recorder, stopper=stopper,
            )
            elapsed = time.time() - start
            if device.type == "cuda":
                train_gpu_peak = int(torch.cuda.max_memory_allocated(device))
                # Evaluation receives its own peak-memory measurement.
                torch.cuda.reset_peak_memory_stats(device)
            h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
            print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")

            if not ARGS.timing_mode:
                train_fields = [
                    "outer_iter", "inner_epoch", "total", "pde", "terminal", "mono", "conc", "eta",
                    "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres", "sel_pres", "lr", "synthetic"]
                mxu.append_csv_rows(recorder.train_csv, results["loss_history"], train_fields)
                base_outer = [
                    "timestamp", "model_type", "run_tag", "outer_iter",
                    "algorithm_outer_iter", "e6_phase",
                    "warmup_excluded_from_achieved_pres",
                    "total_loss", "pde_loss",
                    "terminal_loss", "monotonicity_loss", "concavity_loss", "eta_loss", "train_pres",
                    "val_pde_rms", "val_terminal_rms", "val_pres", "val_pres_at_stop",
                    "val_pres_at_stop_epoch",
                    "val_pres_post_restore", "achieved_pres", "inner_epochs_used", "n_resamples", "target_reached",
                    "target_reached_post_restore", "training_target_crossed",
                    "sel_best_pres", "sel_best_qres_pres", "sel_best_epoch", "sel_best_lr",
                    "sel_checks", "sel_eligible_checks", "sel_ineligible_checks", "sel_stopped",
                    "sel_restored", "q_sel_seed", "lr_end_before_restore", "lr_best_checkpoint", "lr_after_restore",
                    "lr_carried_next", "lr_next_outer_policy", "lr_outer_start",
                    "lr_outer_restart_at_max", "scheduler_reset_at_outer_start",
                    "pi_diff", "c_diff", "pi_vs_closed_form", "c_vs_closed_form",
                    "control_metric_scope", "control_metric_points",
                    "e_V_sup", "e_bundle_sup", "e_Xev", "diag_RelL2_V", "diag_RelL2_pi",
                    "diag_RelL2_c", "m_Vw", "m_minus_Vww", "m_curvature_y",
                    "m_y", "M_y", "m_c", "pi_component_min_greedy",
                    "pi_component_max_greedy", "pi_l2_min_greedy", "pi_l2_max_greedy",
                    "chi_min_greedy", "chi_max_greedy",
                    "pi_component_min_frozen", "pi_component_max_frozen",
                    "pi_l2_min_frozen", "pi_l2_max_frozen",
                    "chi_min_frozen", "chi_max_frozen", "guard_frac_Vw",
                    "guard_frac_curvature", "clip_frac_pi_greedy", "clip_frac_kappa_low",
                    "clip_frac_kappa_high", "clip_frac_c_level_low", "clip_frac_c_level_high",
                    "diffusion_var_min_greedy", "diffusion_var_max_greedy",
                    "diffusion_var_min_frozen", "diffusion_var_max_frozen", "clip_frac_pi_frozen",
                    "clip_frac_kappa_low_frozen", "clip_frac_kappa_high_frozen",
                    "clip_frac_c_level_low_frozen", "clip_frac_c_level_high_frozen", "lr"]
                mxu.append_csv_rows(recorder.outer_csv, results["outer_rows"], base_outer)

            if ARGS.e6_role == "warmup":
                warmup_provenance = _save_e6_warmup_state(
                    solver, results, ARGS.e6_warmup_bundle)
                # The normal success marker lets the queue resume/skip this
                # producer.  E6 aggregation filters e6_role=warmup before it
                # deduplicates target branches.
                recorder.mark_success(
                    elapsed_sec=elapsed,
                    outer_iters=1,
                    total_optimizer_steps=results["total_optimizer_steps"],
                    total_inner_steps=results["total_inner_steps"],
                    pres_target=1.0,
                    pres_max=results["pres_max"],
                    pres_max_semantics=(
                        "single_warmup_outer_post_restore_fixed_qres"
                    ),
                    target_reached=True,
                    target_reached_semantics=(
                        "single_warmup_outer_post_restore_fixed_qres_at_or_below_one"
                    ),
                    final_weight_path=os.path.join(
                        weight_dir, "value_net_final.pt"),
                    train_gpu_peak_mem_bytes=train_gpu_peak,
                    e6_phase="warmup",
                    **warmup_provenance,
                    **lr_protocol_fields,
                    **results.get("checkpoint_provenance", {}),
                )
                print(
                    "[E6] common warm-up bundle saved: "
                    f"{warmup_provenance['e6_warmup_bundle_path']}"
                )
                return

            if ARGS.e6_role == "target_branch":
                pres_max_semantics = (
                    "max_target_phase_outer_post_restore_fixed_qres_excluding_warmup"
                )
                target_reached_semantics = (
                    "all_target_phase_outer_post_restore_fixed_qres_at_or_below_"
                    "target_excluding_warmup"
                )
            else:
                pres_max_semantics = "max_outer_post_restore_fixed_qres"
                target_reached_semantics = (
                    "all_outer_post_restore_fixed_qres_at_or_below_target"
                )

            terminal_e6_fields = {
                "e6_role": str(ARGS.e6_role),
                "e6_phase": str(ARGS.e6_phase),
            }
            terminal_e6_fields.update(e6_warm_start_provenance)
            terminal_e6_fields.update(lr_protocol_fields)
            if ARGS.e6_role == "target_branch":
                target_rows = list(results.get("outer_rows", []))
                terminal_e6_fields.update({
                    "e6_target_phase_outer_count": len(target_rows),
                    "e6_target_phase_start_algorithm_iter": (
                        int(target_rows[0]["algorithm_outer_iter"])
                        if target_rows else 2
                    ),
                    "first_target_policy_source": E6_WARM_START_POLICY_SOURCE,
                })
            if results.get("stopped_early", False):
                recorder.write_status(
                    "stopped_early", elapsed_sec=elapsed,
                    final_weight_path=os.path.join(weight_dir, "value_net_final.pt"),
                    train_gpu_peak_mem_bytes=train_gpu_peak,
                    pres_target=ARGS.pres_target, pres_max=results["pres_max"],
                    any_target_reached=bool(results["target_reached"]),
                    target_reached=bool(results["all_targets_reached"]),
                    target_reached_semantics=target_reached_semantics,
                    any_training_target_crossed=bool(
                        results["training_target_crossed"]),
                    training_target_crossed=bool(
                        results["all_training_targets_crossed"]),
                    training_target_crossed_semantics=(
                        "all_outer_training_time_fixed_qres_crossings_before_restore"
                    ),
                    pres_max_semantics=pres_max_semantics,
                    **terminal_e6_fields,
                    policy_bounds_mode=policy_bounds_mode,
                    initial_policy_diffusion_variance_analytic=(
                        ARGS.initial_policy_diffusion_variance_analytic),
                    initial_policy_degenerate=ARGS.initial_policy_degenerate,
                    initial_policy_degeneracy_tolerance=(
                        ARGS.initial_policy_degeneracy_tolerance),
                    **results.get("checkpoint_provenance", {}),
                    **results.get("stop_info", {}))
                print("[early-stop] training stopped; evaluation and success marker skipped.")
                return
        else:
            recorder.write_status_eval("running")
            elapsed = 0.0
            final_path = os.path.join(weight_dir, "value_net_final.pt")
            last_path = os.path.join(weight_dir, "value_net_last.pt")
            candidates = [final_path, last_path]
            if ARGS.allow_legacy_best_eval:
                candidates.extend([
                    os.path.join(weight_dir, "value_net_best_diag.pt"),
                    os.path.join(weight_dir, "value_net_best.pt"),
                ])
            load_path = next((path for path in candidates if os.path.exists(path)), None)
            if load_path is None:
                hint = " (use --allow-legacy-best-eval for diagnostic/legacy fallback)"
                raise FileNotFoundError(f"no official final/last checkpoint under {weight_dir}{hint}")
            solver.value_net.load_state_dict(torch.load(load_path, map_location=device))
            loaded_weight_path = load_path
            print(f"[eval-only] loaded {load_path}")
            if device.type == "cuda":
                # Exclude model construction/checkpoint loading from the
                # evaluation-only peak, matching the direct PINN protocol.
                torch.cuda.reset_peak_memory_stats(device)

        margins = mxu.parse_eval_margins(ARGS.eval_margin)
        eval_windows = {
            float(margin): mxu.resolve_eval_window(
                y_min,
                y_max,
                float(margin),
                eval_w_min=ARGS.eval_w_min,
            )
            for margin in margins
        }
        if ARGS.skip_eval:
            final_path = os.path.join(weight_dir, "value_net_final.pt")
            if ARGS.eval_only:
                recorder.mark_success_eval(
                    elapsed_sec=elapsed, skipped_eval=True,
                    loaded_weight_path=loaded_weight_path)
            else:
                recorder.mark_success(
                    elapsed_sec=elapsed, final_weight_path=final_path, skipped_eval=True,
                    best_weight_path=os.path.join(weight_dir, "value_net_best_diag.pt"),
                    outer_iters=len(results["outer_rows"]),
                    total_optimizer_steps=results["total_optimizer_steps"],
                    total_inner_steps=results["total_inner_steps"],
                    pres_target=ARGS.pres_target, pres_max=results["pres_max"],
                    any_target_reached=bool(results["target_reached"]),
                    target_reached=bool(results["all_targets_reached"]),
                    target_reached_semantics=target_reached_semantics,
                    any_training_target_crossed=bool(results["training_target_crossed"]),
                    training_target_crossed=bool(results["all_training_targets_crossed"]),
                    training_target_crossed_semantics=(
                        "all_outer_training_time_fixed_qres_crossings_before_restore"
                    ),
                    pres_max_semantics=pres_max_semantics,
                    **terminal_e6_fields,
                    train_gpu_peak_mem_bytes=train_gpu_peak,
                    policy_bounds_mode=policy_bounds_mode,
                    policy_pi_min=pi_min_bound, policy_pi_max=pi_max_bound,
                    policy_kappa_min=kappa_min_bound,
                    policy_kappa_max=kappa_max_bound,
                    policy_c_min=c_min_bound, policy_c_max=c_max_bound,
                    initial_policy_diffusion_variance_analytic=(
                        ARGS.initial_policy_diffusion_variance_analytic),
                    initial_policy_degenerate=ARGS.initial_policy_degenerate,
                    initial_policy_degeneracy_tolerance=(
                        ARGS.initial_policy_degeneracy_tolerance),
                    **results.get("checkpoint_provenance", {}))
            return

        # Full random test if requested; test_points=0 is the deterministic
        # grid fallback and uses the same metric definitions.
        Nt, Nw = int(ARGS.n_tau), int(ARGS.n_x)
        if int(ARGS.test_points) > 0:
            metrics_by_margin = eval_fulldim_test_metrics(
                solver.value_net,
                int(ARGS.test_points),
                margins,
                eval_w_min=ARGS.eval_w_min,
            )
        else:
            metrics_by_margin = {}
            for margin in margins:
                window = eval_windows[float(margin)]
                y_lo, y_hi = window["ev_y_min"], window["ev_y_max"]
                t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
                w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))
                tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")
                metrics_by_margin[float(margin)] = eval_metrics_on_points(
                    solver.value_net, tt, ww)

        metric_rows = []
        for margin in margins:
            metrics = metrics_by_margin[float(margin)]
            window = eval_windows[float(margin)]
            for key, val in metrics.items():
                metric_rows.append({
                    "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag, "scope": "fulldim", "eval_margin": margin,
                    "eval_window_mode": window["eval_window_mode"],
                    "eval_w_min_requested": window["eval_w_min_requested"],
                    "eval_w_min_symmetric": window["eval_w_min_symmetric"],
                    "ev_y_min": window["ev_y_min"],
                    "ev_y_max": window["ev_y_max"],
                    "ev_w_min": window["ev_w_min"],
                    "ev_w_max": window["ev_w_max"],
                    "metric": key, "value": val})
        mxu.append_csv_rows(recorder.metrics_csv, metric_rows,
                            [
                                "timestamp", "model_type", "run_tag", "scope",
                                "eval_margin", "eval_window_mode",
                                "eval_w_min_requested",
                                "eval_w_min_symmetric", "ev_y_min", "ev_y_max",
                                "ev_w_min", "ev_w_max", "metric", "value",
                            ])

        eval_source = (
            f"{int(ARGS.test_points)} fixed random points"
            if int(ARGS.test_points) > 0
            else f"{Nt}x{Nw} deterministic grid"
        )
        print(f"\nEvaluation metrics by margin ({eval_source}):")
        for margin in margins:
            is_primary = float(margin) == float(margins[0])
            window = eval_windows[float(margin)]
            print(
                f"\n--- eval_margin={float(margin):.2f}"
                f"{' (primary)' if is_primary else ''}; "
                f"W=[{window['ev_w_min']:.6g},{window['ev_w_max']:.6g}] ---"
            )
            for key, value in metrics_by_margin[float(margin)].items():
                print(f"  {key}: {value:.6e}")

        if not skip_figures and not ARGS.eval_only:
            plot_convergence(results, save_path=os.path.join(out_dir, "plots", "convergence.png"), show=False)
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                solver.value_net,
                Nt=Nt,
                Nw=Nw,
                margin=margins[0],
                eval_w_min=ARGS.eval_w_min,
            )
            plot_comparison_heatmaps(
                tt, ww, V_pinn, c_pinn, pi_norm,
                save_path=os.path.join(out_dir, "plots", "comparison_heatmap.png"), show=False)

        eval_gpu_peak = None
        if device.type == "cuda":
            eval_gpu_peak = int(torch.cuda.max_memory_allocated(device))

        if ARGS.eval_only:
            if os.path.exists(recorder.metrics_csv):
                backup_path = metrics_final_path + ".bak_train"
                if os.path.exists(metrics_final_path) and not os.path.exists(backup_path):
                    shutil.copyfile(metrics_final_path, backup_path)
                os.replace(recorder.metrics_csv, metrics_final_path)
                recorder.metrics_csv = metrics_final_path
            primary_window = eval_windows[float(margins[0])]
            recorder.mark_success_eval(
                elapsed_sec=elapsed, primary_margin=margins[0],
                loaded_weight_path=loaded_weight_path,
                eval_gpu_peak_mem_bytes=eval_gpu_peak, eval_margins=margins,
                **{
                    key: primary_window[key]
                    for key in (
                        "eval_window_mode", "eval_w_min_requested",
                        "eval_w_min_symmetric", "ev_y_min", "ev_y_max",
                        "ev_w_min", "ev_w_max",
                    )
                },
            )
        else:
            first_outer = results["outer_rows"][0] if results["outer_rows"] else {}
            primary_window = eval_windows[float(margins[0])]
            recorder.mark_success(
                elapsed_sec=elapsed, outer_iters=len(results["outer_rows"]),
                total_optimizer_steps=results["total_optimizer_steps"], train_wall_sec=elapsed,
                primary_margin=margins[0], final_weight_path=os.path.join(weight_dir, "value_net_final.pt"),
                best_weight_path=os.path.join(weight_dir, "value_net_best_diag.pt"),
                pres_target=ARGS.pres_target, pres_max=results["pres_max"],
                total_inner_steps=results["total_inner_steps"],
                any_target_reached=bool(results["target_reached"]),
                target_reached=bool(results["all_targets_reached"]),
                target_reached_semantics=target_reached_semantics,
                any_training_target_crossed=bool(results["training_target_crossed"]),
                training_target_crossed=bool(results["all_training_targets_crossed"]),
                training_target_crossed_semantics=(
                    "all_outer_training_time_fixed_qres_crossings_before_restore"
                ),
                pres_max_semantics=pres_max_semantics,
                **terminal_e6_fields,
                pi_init_scale=float(ARGS.pi_init_scale),
                pi_clip_abs=pi_clip_abs,
                policy_bounds_mode=policy_bounds_mode,
                policy_pi_min=pi_min_bound, policy_pi_max=pi_max_bound,
                policy_kappa_min=kappa_min_bound,
                policy_kappa_max=kappa_max_bound,
                policy_c_min=c_min_bound, policy_c_max=c_max_bound,
                diffusion_var_min_init=first_outer.get("diffusion_var_min_frozen", ""),
                diffusion_var_max_init=first_outer.get("diffusion_var_max_frozen", ""),
                initial_policy_diffusion_variance_analytic=(
                    ARGS.initial_policy_diffusion_variance_analytic),
                initial_policy_degenerate=ARGS.initial_policy_degenerate,
                initial_policy_degeneracy_tolerance=(
                    ARGS.initial_policy_degeneracy_tolerance),
                train_gpu_peak_mem_bytes=train_gpu_peak,
                eval_gpu_peak_mem_bytes=eval_gpu_peak, eval_margins=margins,
                **{
                    key: primary_window[key]
                    for key in (
                        "eval_window_mode", "eval_w_min_requested",
                        "eval_w_min_symmetric", "ev_y_min", "ev_y_max",
                        "ev_w_min", "ev_w_max",
                    )
                },
                **results.get("checkpoint_provenance", {}))
        print("\nDone.")
    except Exception as exc:
        if ARGS.eval_only:
            recorder.mark_failed_eval(reason=repr(exc))
        else:
            failed_e6_fields = {
                "e6_role": str(ARGS.e6_role),
                "e6_phase": str(ARGS.e6_phase),
            }
            failed_e6_fields.update(e6_warm_start_provenance)
            recorder.mark_failed(reason=repr(exc), **failed_e6_fields)
        raise


if __name__ == "__main__":
    main()
