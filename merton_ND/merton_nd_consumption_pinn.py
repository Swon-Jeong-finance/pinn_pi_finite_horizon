"""
Multi-Asset Merton Portfolio (with Consumption) - PINN (Reduced-form HJB) in log-wealth
====================================================================================

What this script is
-------------------
This is the "training-method swap" you requested:

- Start from your multi-asset log-wealth PI-PINN code:
    `merton_nd_consumption_pi_pinn_logw.py`
  but REMOVE policy-iteration (fixed-policy evaluation / improvement loops).

- Replace it with your single-asset log-wealth reduced-form PINN trainer idea from:
    `merton_pinn_logw.py`

So we train ONLY ONE network v(t,y)=V(t,exp(y)) with a *reduced nonlinear HJB* residual
(after substituting the FOC for (c*, pi*)).

Key equations (y = log W)
-------------------------
Let v(t,y) = V(t, e^y), W = exp(y).

FOC (implied controls from v):
    c*(t,y)  = (V_W)^(-1/gamma) = (v_y / W)^(-1/gamma)
    pi*(t,y) = -(v_y / (v_yy - v_y)) * Sigma^{-1} mu_excess   (N-vector)

Reduced-form HJB (after substituting FOCs):
    0 = v_t + r v_y - rho v
        + (gamma/(1-gamma)) * ( (v_y / W) ^ ((gamma-1)/gamma) )
        + 0.5 * Theta * v_y^2 / (v_y - v_yy)

where Theta = mu_excess^T Sigma^{-1} mu_excess.

Implementation notes
--------------------
- We keep `market_setup.py` as-is (requested) to generate stable SPD covariance.
- We keep your stability tricks:
  * terminal loss, monotonicity/concavity penalties in y,
  * optional eta penalty: eta = 1 - v_yy/v_y ~ gamma,
  * hybrid sampling: several optimizer steps per fixed pseudo-outer batch.
- For evaluation/plots we still compute (c, pi) via FOC, and compare to closed-form.

Author: derived from your uploaded scripts (no new theory introduced).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import market_setup  # keep as requested
import merton_experiment_utils as mxu
import merton_evaluation_metrics as mem


# Direct PINN uses two fixed held-out streams with deliberately separate roles:
# Q_res controls an optional p_res stopping target, while Q_sel alone drives
# ReduceLROnPlateau and the same-metric emergency rollback. The offset is fixed
# and recorded in config/status so the split is deterministic across training
# seeds for a fixed market seed.
DIRECT_PINN_QSEL_SEED_OFFSET = 1_000_003
HELDOUT_SCORE_FORMULA = "RMS(PDE_residual)+RMS(terminal_error)"
SCHEDULER_STATUS_FIELDS = (
    "q_res_role", "q_res_seed", "q_sel_role", "q_sel_seed", "q_sel_active",
    "heldout_score_formula", "scheduler_metric", "scheduler_patience_unit",
    "scheduler_patience_checks",
    "scheduler_check_every_optimizer_steps", "scheduler_checks",
    "scheduler_lr_reductions", "sel_pde_rms_last",
    "sel_terminal_rms_last", "sel_pres_last", "scheduler_score_last",
    "sel_pres_best", "sel_check_epoch_last",
)
QSEL_ROLLBACK_STATUS_FIELDS = (
    "qsel_rollback_active", "qsel_rollback_factor",
    "qsel_rollback_lr_factor", "qsel_rollback_max_rescues",
    "qsel_rollback_triggers", "qsel_rollback_rescues_used",
    "qsel_rollback_scheduler_resets", "qsel_rollback_last_epoch",
    "qsel_rollback_last_observed_pres",
    "qsel_rollback_baseline_epoch", "qsel_rollback_baseline_pres",
    "qsel_rollback_last_admissible_epoch",
    "qsel_rollback_last_admissible_pres",
    "qsel_rollback_last_restored_epoch",
    "qsel_rollback_last_restored_pres", "qsel_rollback_failed",
    "qsel_rollback_failure_reason",
)
RESAMPLING_STATUS_FIELDS = (
    "resample_every_optimizer_steps", "training_batch_resamples_total",
    "training_batches_sampled_total",
)


# =============================================================================
# Command-line configuration (Merton reduced-form PINN).
#   * --seed drives network init / collocation / optimizer randomness.
#   * --market-seed drives ONLY the synthetic market draw, so a fixed market
#     can be shared across a training-seed sweep (paper protocol).
# The module-level constants below are populated from ARGS so the existing
# global-reference structure is preserved (minimal-diff refactor).
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-asset Merton (with consumption) reduced-form PINN [log-wealth].")
    # Reproducibility / device
    p.add_argument("--seed", type=int, default=12,
                   help="Network init / collocation / optimizer seed.")
    p.add_argument("--market-seed", type=int, default=12,
                   help="Synthetic-market seed (fixed across a training-seed sweep).")
    p.add_argument("--device", type=str, default=None,
                   help="e.g. cuda:0 or cpu (default: GPU_ID env or auto).")
    # Market generation (market_setup.generate_synthetic_merton_market)
    p.add_argument("--n-assets", type=int, default=50)
    p.add_argument("--sigma-lo", type=float, default=0.10)
    p.add_argument("--sigma-hi", type=float, default=0.25)
    p.add_argument("--rho-max", type=float, default=1.0)
    p.add_argument("--kappa-max", type=float, default=30.0)
    p.add_argument("--delta-rel", type=float, default=1e-4)
    p.add_argument("--pi-scale", type=float, default=0.6)
    p.add_argument("--mu-noise-rel", type=float, default=0.02)
    p.add_argument("--mu-mode", type=str, default="pi_target",
                   choices=["pi_target", "sharpe"])
    # Preferences / market scalars
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho-discount", type=float, default=0.04)
    p.add_argument("--r", type=float, default=0.03)
    p.add_argument("--epsilon-bequest", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=1.0, help="Horizon T.")
    # Domain
    p.add_argument("--w-min", type=float, default=0.1)
    p.add_argument("--w-max", type=float, default=2.0)
    # Network / optimization
    p.add_argument("--value-hidden", type=int, default=256)
    p.add_argument("--value-depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--terminal-frac", type=float, default=0.5,
                   help="Terminal sample count = max(1, int(batch_size*terminal_frac)).")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--outer-iters", type=int, default=500,
                   help="Number of training blocks (kept name-compatible with Liu).")
    p.add_argument("--eval-epochs", type=int, default=200,
                   help="Optimizer steps per block.")
    p.add_argument(
        "--resample-every", type=int, default=0,
        help=(
            "Refresh the interior and terminal training batch after this many "
            "inner optimizer steps; 0 keeps the outer-start batch fixed for "
            "the entire block."
        ),
    )
    p.add_argument(
        "--scheduler-patience", type=int, default=5000,
        help="ReduceLROnPlateau patience measured in fixed-Q_sel checks (not optimizer steps).")
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    p.add_argument("--lr-schedule", type=str, default="plateau",
                   choices=["plateau", "fixed"])
    p.add_argument(
        "--qsel-rollback-factor", type=float, default=10.0,
        help=(
            "Restore the last admissible model+Adam state when fixed-Q_sel "
            "p_sel exceeds this multiple of its best prior value."
        ),
    )
    p.add_argument(
        "--qsel-rollback-lr-factor", type=float, default=1.0,
        help=(
            "Emergency LR multiplier in (0, 1] applied after a Q_sel "
            "rollback; 1 keeps the non-increasing rollback LR unchanged."
        ),
    )
    p.add_argument(
        "--qsel-rollback-max-rescues", type=int, default=2,
        help=(
            "Maximum successful Q_sel rollback rescues; 0 disables the "
            "rollback mechanism."
        ),
    )
    # Nonlinear-HJB positivity continuation.  ``softplus`` is the new paper
    # candidate; ``abs`` reproduces the historical PINN and ``hard``
    # reproduces the unsuccessful hard one-sided pilot.
    p.add_argument("--hjb-guard-mode", type=str, default="softplus",
                   choices=["abs", "hard", "softplus"])
    p.add_argument("--hjb-vy-guard-eps", type=float, default=1e-8)
    p.add_argument("--hjb-denom-guard-eps", type=float, default=1e-8)
    p.add_argument("--hjb-vy-guard-tau", type=float, default=1e-2)
    p.add_argument("--hjb-denom-guard-tau", type=float, default=1e-2)
    p.add_argument("--hjb-vy-guard-tau-min", type=float, default=1e-4)
    p.add_argument("--hjb-denom-guard-tau-min", type=float, default=1e-4)
    p.add_argument(
        "--hjb-guard-anneal-every", type=int, default=0,
        help="Outer blocks per tau reduction; 0 keeps tau fixed (recommended first pilot).")
    p.add_argument("--hjb-guard-anneal-factor", type=float, default=0.5)
    # Loss weights
    p.add_argument("--w-terminal", type=float, default=10.0)
    p.add_argument("--w-shape", type=float, default=1.0)
    p.add_argument("--w-eta", type=float, default=1.5,
                   help="Weight of the RRA/consumption-shape diagnostic penalty.")
    p.add_argument("--eta-clip", type=str, default="10.0",
                   help="Optional |.|-clip for the eta diagnostic penalty (none = off).")
    p.add_argument("--pi-clip-abs", type=mxu.none_or_float, default=2.0,
                   help="Symmetric componentwise portfolio bound; none disables clipping.")
    p.add_argument(
        "--policy-bounds-mode", type=str, default="stabilized",
        choices=["stabilized", "none"],
        help=(
            "stabilized applies the recorded portfolio/kappa/consumption projections; "
            "none disables every finite action projection (FOC sign guards remain)."
        ),
    )
    # Held-out residual target and fixed diagnostics (same roles as Liu).
    p.add_argument("--pres-target", type=mxu.none_or_float, default=None)
    p.add_argument(
        "--val-points", type=int, default=100000,
        help="Interior points in each fixed held-out set Q_res and Q_sel.")
    p.add_argument(
        "--val-terminal-points", type=int, default=10000,
        help="Terminal points in each fixed held-out set Q_res and Q_sel.")
    p.add_argument(
        "--val-every", type=int, default=1,
        help="Optimizer-step interval for Q_sel scheduler checks and active Q_res target checks.")
    p.add_argument("--save-iterate-every", type=int, default=0)
    p.add_argument("--diag-points", type=int, default=4096)
    p.add_argument("--diag-every", type=int, default=1)
    # Evaluation window(s): first = PRIMARY (diagnostic + representative
    # metric); the rest are the E9 window-sensitivity list, re-evaluated on
    # the same trained network. The margin shrinks only y=log W; time retains
    # its full range, as in Q_ev=(0,T)xOmega_ev.
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
    p.add_argument("--print-every", type=int, default=5000)
    p.add_argument("--output-root", type=str, default="outputs_merton")
    p.add_argument("--weight-root", type=str, default=None)
    p.add_argument("--run-tag", type=str, default="merton_pinn")
    p.add_argument("--model-type", type=str, default="pinn", choices=["pinn", "pipinn"])
    # Merton state dimension is 1 (wealth only); recorded for aggregation /
    # figure selection parity with the PI-PINN side. Not tunable.
    p.add_argument("--m-states", type=int, default=1)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-plots", action="store_true",
                   help="Back-compat alias for --skip-figures.")
    # Infrastructure (wired in Phase 2)
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
if not math.isfinite(ARGS.terminal_frac) or ARGS.terminal_frac <= 0.0:
    raise ValueError("--terminal-frac must be finite and strictly positive")
if ARGS.pi_clip_abs is not None and (
        not math.isfinite(ARGS.pi_clip_abs) or ARGS.pi_clip_abs <= 0.0):
    raise ValueError("--pi-clip-abs must be finite and positive, or none")
if ARGS.val_every < 1:
    raise ValueError("--val-every must be positive")
if ARGS.resample_every < 0:
    raise ValueError("--resample-every must be nonnegative")
if ARGS.scheduler_patience < 0 or not (0.0 < ARGS.scheduler_factor < 1.0):
    raise ValueError("require scheduler_patience >= 0 and 0 < scheduler_factor < 1")
if ARGS.scheduler_min_lr <= 0.0:
    raise ValueError("--scheduler-min-lr must be positive")
if ARGS.lr_schedule == "plateau" and ARGS.lr < ARGS.scheduler_min_lr:
    raise ValueError("--lr must not be below --scheduler-min-lr")
if (not math.isfinite(ARGS.qsel_rollback_factor)
        or ARGS.qsel_rollback_factor <= 1.0):
    raise ValueError("--qsel-rollback-factor must be finite and greater than 1")
if (not math.isfinite(ARGS.qsel_rollback_lr_factor)
        or not (0.0 < ARGS.qsel_rollback_lr_factor <= 1.0)):
    raise ValueError(
        "--qsel-rollback-lr-factor must lie in (0, 1]")
if ARGS.qsel_rollback_max_rescues < 0:
    raise ValueError("--qsel-rollback-max-rescues must be nonnegative")
if (not ARGS.eval_only and ARGS.lr_schedule == "plateau"
        and ARGS.qsel_rollback_max_rescues > 0
        and ARGS.eval_epochs % ARGS.val_every != 0):
    raise ValueError(
        "Q_sel rollback requires --eval-epochs divisible by --val-every "
        "so every official outer iterate is checked")
if ARGS.pres_target is not None and ARGS.val_points <= 0:
    raise ValueError("--pres-target requires --val-points > 0")
if (not ARGS.eval_only and ARGS.lr_schedule == "plateau"
        and ARGS.val_points <= 0):
    raise ValueError("--lr-schedule plateau requires --val-points > 0 for fixed Q_sel")
if (not ARGS.eval_only
        and (ARGS.pres_target is not None or ARGS.lr_schedule == "plateau")
        and ARGS.val_terminal_points <= 0):
    raise ValueError(
        "held-out Q_res/Q_sel scoring requires --val-terminal-points > 0")
for _name in ("hjb_vy_guard_eps", "hjb_denom_guard_eps",
              "hjb_vy_guard_tau", "hjb_denom_guard_tau",
              "hjb_vy_guard_tau_min", "hjb_denom_guard_tau_min"):
    _value = float(getattr(ARGS, _name))
    if not math.isfinite(_value) or _value <= 0.0:
        raise ValueError(f"--{_name.replace('_', '-')} must be finite and positive")
if ARGS.hjb_vy_guard_tau_min > ARGS.hjb_vy_guard_tau:
    raise ValueError("--hjb-vy-guard-tau-min must not exceed --hjb-vy-guard-tau")
if ARGS.hjb_denom_guard_tau_min > ARGS.hjb_denom_guard_tau:
    raise ValueError("--hjb-denom-guard-tau-min must not exceed --hjb-denom-guard-tau")
if ARGS.hjb_guard_anneal_every < 0:
    raise ValueError("--hjb-guard-anneal-every must be nonnegative")
if not (0.0 < ARGS.hjb_guard_anneal_factor < 1.0):
    raise ValueError("--hjb-guard-anneal-factor must lie strictly between 0 and 1")


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
    if torch.cuda.is_available() and GPU_ID < torch.cuda.device_count():
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
print(f"Device: {device} (cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()})")

if device.type == "cuda":
    torch.cuda.init()
    _ = torch.zeros(1, device=device)


# =============================================================================
# 1) Problem Parameters
# =============================================================================
T_FINAL = float(ARGS.tau_max)
t_min, t_max = 0.0, T_FINAL

# Wealth domain (W) and log-wealth (y)
w_min, w_max = float(ARGS.w_min), float(ARGS.w_max)
y_min, y_max = float(np.log(w_min)), float(np.log(w_max))

# Preferences
gamma_risk = float(ARGS.gamma)
rho_discount = float(ARGS.rho_discount)
epsilon_bequest = float(ARGS.epsilon_bequest)

# Risk-free
r_rate = float(ARGS.r)

# Multi-asset dimension
N_ASSETS = int(ARGS.n_assets)

# Synthetic market configuration. NOTE: drawn from MARKET_SEED (not SEED) so a
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

pi_star_np = np.asarray(market_params["pi_star"], dtype=np.float64).reshape(N_ASSETS)

# Stable precompute of Sigma^{-1} mu_excess
Sigma_inv_mu_np = market_setup.cholesky_solve(chol_Sigma_np, mu_excess_np)

# Theta = mu^T Sigma^{-1} mu
Theta = float(market_params["mu_SigmaInv_mu"])

# Closed-form "nu" parameter (multi-asset: replace sharpe^2 by Theta)
nu = rho_discount / gamma_risk - (1.0 - gamma_risk) * (
    Theta / (2.0 * (gamma_risk**2)) + r_rate / gamma_risk
)

# Torch constants
mu_excess = torch.tensor(mu_excess_np, device=device, dtype=torch.float32)          # (N,)
Sigma = torch.tensor(Sigma_np, device=device, dtype=torch.float32)                 # (N,N)
Sigma_inv_mu = torch.tensor(Sigma_inv_mu_np, device=device, dtype=torch.float32)   # (N,)
pi_star = torch.tensor(pi_star_np, device=device, dtype=torch.float32)             # (N,)
Theta_t = torch.tensor(Theta, device=device, dtype=torch.float32)

print(f"\n{'='*70}")
print("Multi-Asset Merton (with Consumption) - PINN (Reduced-form HJB) [log W]")
print(f"{'='*70}")
print(f"  N_ASSETS={N_ASSETS}")
print(f"  gamma={gamma_risk}, rho={rho_discount}, r={r_rate}, epsilon={epsilon_bequest}")
print(f"  T={T_FINAL}, W∈[{w_min},{w_max}] -> y∈[{y_min:.3f},{y_max:.3f}]")
print(f"  Theta = mu^T Sigma^{-1} mu = {Theta:.6f}")
print(f"  nu = {nu:.6f}")
print(f"  ||pi*||_2 = {np.linalg.norm(pi_star_np):.4f}, max|pi*_i|={np.max(np.abs(pi_star_np)):.4f}")
print(f"  cond(Sigma_safe) = {market_params['cond_Sigma_safe']:.2f}, max|rho_ij|={market_params['max_abs_rho']:.3f}")
print(f"{'='*70}\n")

# =============================================================================
# 2) Closed-form Solution (for sanity check)
# =============================================================================
def closed_form_c(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """c*(t,W) = m(t) W with multi-asset nu."""
    tau = T_FINAL - t
    scale = mem.crra_homothetic_scale(
        tau, nu, gamma_risk, epsilon_bequest)
    return W / scale


def closed_form_V(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """V(t,W)=A(t) W^{1-gamma}/(1-gamma)."""
    tau = T_FINAL - t
    scale = mem.crra_homothetic_scale(
        tau, nu, gamma_risk, epsilon_bequest)
    A_t = scale ** gamma_risk
    return A_t * (W ** (1.0 - gamma_risk)) / (1.0 - gamma_risk)


def closed_form_pi(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """pi*(t,W) is constant vector; broadcast on grid."""
    Nt, Nw = t.shape
    return np.broadcast_to(pi_star_np.reshape(1, 1, N_ASSETS), (Nt, Nw, N_ASSETS)).copy()


def closed_form_numpy(t_grid: np.ndarray, W_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = closed_form_V(t_grid, W_grid)
    c = closed_form_c(t_grid, W_grid)
    pi = closed_form_pi(t_grid, W_grid)
    return V, c, pi


# =============================================================================
# 3) Value Network v(t,y) = V(t, exp(y))
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


def _resample_before_inner_step(inner_epoch: int, resample_every: int) -> bool:
    """Return whether a fresh batch is due before ``inner_epoch``.

    The outer-start batch is used for steps 1 through K.  With K > 0, the
    first refresh therefore occurs immediately before step K+1.  K=0 fixes
    the outer-start batch for the entire inner loop.
    """
    every = int(resample_every)
    if every < 0:
        raise ValueError("resample_every must be nonnegative")
    epoch = int(inner_epoch)
    return every > 0 and epoch > 1 and (epoch - 1) % every == 0


def _qsel_rollback_triggered(
    current_pres: float,
    best_pres: float,
    factor: float,
) -> bool:
    """Apply the sole emergency criterion: current Q_sel versus prior best.

    A non-finite Q_sel score is treated as an infinite score.  The caller
    fails cleanly if this happens before any admissible baseline exists.
    """
    multiplier = float(factor)
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError("qsel rollback factor must be finite and greater than 1")
    if not math.isfinite(float(current_pres)):
        return True
    if not math.isfinite(float(best_pres)):
        return False
    return float(current_pres) > multiplier * float(best_pres)


def _qsel_rollback_lr(
    current_lr: float,
    restored_lr: float,
    min_lr: float,
    factor: float,
) -> float:
    """Return an emergency LR that never exceeds the pre-rollback LR."""
    current = float(current_lr)
    restored = float(restored_lr)
    floor = float(min_lr)
    multiplier = float(factor)
    if current <= 0.0 or restored <= 0.0 or floor <= 0.0:
        raise ValueError("rollback learning rates must be positive")
    if not (0.0 < multiplier <= 1.0):
        raise ValueError("rollback LR factor must lie in (0, 1]")
    base = min(current, restored)
    candidate = max(floor, multiplier * base)
    return min(current, candidate)


# =============================================================================
# 5) Terminal condition
# =============================================================================
def V_terminal_from_y(y: torch.Tensor) -> torch.Tensor:
    W = torch.exp(y)
    return epsilon_bequest * W.pow(1.0 - gamma_risk) / (1.0 - gamma_risk)


# =============================================================================
# 6) Derivatives
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
# 7) Reduced-form HJB residual (multi-asset) in log space
# =============================================================================
HJB_GUARD_VERSION = "merton-direct-hjb-positive-v1"


def smooth_positive(
    x: torch.Tensor,
    eps: float,
    tau: float,
) -> torch.Tensor:
    """Identity-preserving smooth approximation of ``max(x, eps)``.

    ``torch.nn.functional.softplus`` is used instead of a hand-written
    log/exp expression so large positive inputs remain numerically stable.
    """
    return float(eps) + float(tau) * F.softplus(
        (x - float(eps)) / float(tau))


def positive_hjb_guard(
    x: torch.Tensor,
    mode: str,
    eps: float,
    tau: float,
) -> torch.Tensor:
    """Apply the selected positivity continuation to one HJB derivative."""
    if mode == "abs":
        return torch.abs(x) + float(eps)
    if mode == "hard":
        return torch.clamp(x, min=float(eps))
    if mode == "softplus":
        return smooth_positive(x, float(eps), float(tau))
    raise ValueError(f"unsupported HJB guard mode: {mode!r}")


def hjb_guard_taus_for_outer(
    outer_iter: int,
    tau_vy_initial: float,
    tau_denom_initial: float,
    tau_vy_min: float,
    tau_denom_min: float,
    anneal_every: int,
    anneal_factor: float,
) -> Tuple[float, float, int]:
    """Resolve blockwise continuation temperatures.

    The first block always uses the initial temperatures.  With
    ``anneal_every=K>0``, both temperatures are multiplied by ``factor`` after
    every K completed blocks and are floored independently.  A fixed tau
    (anneal_every=0) is the default first-pilot protocol.
    """
    stage = 0 if int(anneal_every) == 0 else (int(outer_iter) - 1) // int(anneal_every)
    scale = float(anneal_factor) ** int(stage)
    tau_vy = max(float(tau_vy_min), float(tau_vy_initial) * scale)
    tau_denom = max(float(tau_denom_min), float(tau_denom_initial) * scale)
    return tau_vy, tau_denom, stage


def reduced_hjb_residual_log_multi(
    value_net: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    guard_mode: str = "softplus",
    eps_vy: float = 1e-8,
    eps_denom: float = 1e-8,
    tau_vy: float = 1e-2,
    tau_denom: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduced-form HJB residual in (t,y), after substituting FOCs for c and pi:

        0 = v_t + r v_y - rho v
            + (gamma/(1-gamma)) * ( (v_y / W) ^ ((gamma-1)/gamma) )
            + 0.5 * Theta * v_y^2 / (v_y - v_yy)

    The default continuation is one-sided softplus:

        g_{tau,eps}(x) = eps + tau softplus((x-eps)/tau).

    It is asymptotically the identity on the admissible branch and retains a
    recovery gradient near the sign boundary.  Raw one-sided shape penalties
    still enforce V_y > 0 and V_y - V_yy > 0.  Historical ``abs`` and hard
    clamp continuations remain available for controlled comparisons.
    """
    V, V_t, V_y, V_yy = compute_derivatives_log(value_net, t, y)
    W = torch.exp(y)

    # For CRRA exponent (gamma-1)/gamma in (0,1), the base must be positive.
    V_y_safe = positive_hjb_guard(
        V_y, guard_mode, eps=float(eps_vy), tau=float(tau_vy))

    denom = (V_y - V_yy)            # should be positive (concavity in W)
    denom_safe = positive_hjb_guard(
        denom, guard_mode, eps=float(eps_denom), tau=float(tau_denom))

    exp_c = (gamma_risk - 1.0) / gamma_risk
    term_consumption = (gamma_risk / (1.0 - gamma_risk)) * ( (V_y_safe / W).pow(exp_c) )

    term_portfolio = 0.5 * Theta_t * (V_y_safe.pow(2) / denom_safe)

    residual = V_t + r_rate * V_y - rho_discount * V + term_consumption + term_portfolio
    return residual, V, V_y, V_yy, denom


# =============================================================================
# 8) (Optional) policies from FOC (used for evaluation/plots only)
# =============================================================================
# ``policy_bounds_mode=none`` is the paper audit setting with no finite action
# projection.  The one-sided FOC guards remain because they define the
# implemented derivative-to-policy map and are diagnosed separately.
policy_bounds_mode = str(ARGS.policy_bounds_mode)
pi_clip_abs: Optional[float] = (
    None if policy_bounds_mode == "none" else ARGS.pi_clip_abs
)

M_utility_cap = 1e3
c_floor = ((gamma_risk - 1.0) * M_utility_cap) ** (-1.0 / (gamma_risk - 1.0))
kappa_min_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else c_floor / w_min
)
kappa_max_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else 3.0
)
c_min_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else c_floor
)
c_max_bound: Optional[float] = (
    None if policy_bounds_mode == "none" else w_max
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


POLICY_METADATA = {
    "trainer_protocol": "merton-direct-pinn-heldout-scheduler-v1",
    "trainer_protocol_version": 1,
    "trainer_source": "merton_ND/merton_nd_consumption_pinn.py",
    "trainer_source_marker": "merton-direct-pinn-logw-heldout-scheduler-v1",
    "trainer_source_sha256": mxu.sha256_file(os.path.abspath(__file__)),
    "network_time_coordinate": "t",
    "network_input_order": "t,y",
    "network_input_transform": "identity",
    "network_activation": "tanh",
    "activation": "tanh",
    "network_dtype": "float32",
    "policy_bounds_mode": policy_bounds_mode,
    "policy_pi_min": None if pi_clip_abs is None else -float(pi_clip_abs),
    "policy_pi_max": None if pi_clip_abs is None else float(pi_clip_abs),
    "policy_kappa_min": kappa_min_bound,
    "policy_kappa_max": kappa_max_bound,
    "policy_c_min": c_min_bound,
    "policy_c_max": c_max_bound,
}


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
    pi* in y=log W coordinates:
        pi* = -(V_y / (V_yy - V_y)) * Sigma^{-1} mu_excess
            =  (V_y / (V_y - V_yy)) * Sigma^{-1} mu_excess
    """
    denom = (V_y - V_yy)  # should be positive
    denom_safe = torch.clamp(denom, min=eps)

    scalar = (torch.clamp(V_y, min=eps) / denom_safe)  # (batch,1)
    pi_raw = scalar * Sigma_inv_mu.view(1, -1)  # (batch,N)
    if return_raw or clip_abs is None:
        return pi_raw
    return torch.clamp(pi_raw, min=-float(clip_abs), max=float(clip_abs))


def compute_policies_from_log_multi(
    value_net: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        V: (batch,1)
        c: (batch,1)
        pi: (batch,N)
        eta: (batch,1) where eta = 1 - V_yy/V_y  ~ gamma
    """
    V = value_net(t, y)
    ones = torch.ones_like(V)

    V_y = torch.autograd.grad(V, y, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_yy = torch.autograd.grad(V_y, y, grad_outputs=torch.ones_like(V_y), create_graph=create_graph, retain_graph=True)[0]

    c = compute_c_from_foc_log(V_y, y)
    pi = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu)

    eps = 1e-8
    V_y_safe = torch.clamp(V_y, min=eps)
    eta = 1.0 - (V_yy / V_y_safe)

    return V, c, pi, eta


def build_validation_set(
    n_int: int,
    n_term: int,
    target_device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Fixed held-out Q_col sample using a dedicated RNG stream."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    t_int = torch.rand((int(n_int), 1), generator=gen) * (T_FINAL - 1e-3)
    y_int = y_min + torch.rand((int(n_int), 1), generator=gen) * (y_max - y_min)
    t_term = torch.full((int(n_term), 1), T_FINAL)
    y_term = y_min + torch.rand((int(n_term), 1), generator=gen) * (y_max - y_min)
    y_term_dev = y_term.to(target_device)
    return {
        "t_int": t_int.to(target_device),
        "y_int": y_int.to(target_device),
        "t_term": t_term.to(target_device),
        "y_term": y_term_dev,
        "V_term": V_terminal_from_y(y_term_dev).detach(),
    }


def evaluate_heldout_pres_pinn(
    value_net: nn.Module,
    val_set: Dict[str, torch.Tensor],
    guard_mode: str = "softplus",
    eps_vy: float = 1e-8,
    eps_denom: float = 1e-8,
    tau_vy: float = 1e-2,
    tau_denom: float = 1e-2,
    chunk: int = 4096,
) -> Tuple[float, float, float]:
    """Held-out p_res = RMS nonlinear-HJB residual + RMS terminal error."""
    was_training = value_net.training
    value_net.eval()
    sq_sum = 0.0
    n = int(val_set["t_int"].shape[0])
    for start in range(0, n, chunk):
        t = val_set["t_int"][start:start + chunk].detach().clone().requires_grad_(True)
        y = val_set["y_int"][start:start + chunk].detach().clone().requires_grad_(True)
        residual, _, _, _, _ = reduced_hjb_residual_log_multi(
            value_net, t, y, guard_mode=guard_mode,
            eps_vy=eps_vy, eps_denom=eps_denom,
            tau_vy=tau_vy, tau_denom=tau_denom)
        sq_sum += float(torch.sum(residual.detach() ** 2).item())
    pde_rms = float(np.sqrt(sq_sum / max(n, 1)))
    with torch.no_grad():
        pred = value_net(val_set["t_term"], val_set["y_term"])
        term_rms = float(torch.sqrt(torch.mean((pred - val_set["V_term"]) ** 2)).item())
    if was_training:
        value_net.train()
    return pde_rms, term_rms, pde_rms + term_rms


def build_diag_set(n_points: int, margin: float) -> Dict[str, torch.Tensor]:
    """Fixed complete tensor grid; shrink log wealth, never time.

    ``n_points`` is an approximate budget.  Using complete axes avoids
    truncating the final time row, which is important for reproducible extrema.
    """
    n_points = max(4, int(n_points))
    n_t = max(2, int(round(math.sqrt(n_points))))
    n_y = max(2, int(math.ceil(n_points / n_t)))
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, float(margin))
    t_axis = np.linspace(t_min, t_max - 1e-3, n_t, dtype=np.float64)
    y_axis = np.linspace(y_lo, y_hi, n_y, dtype=np.float64)
    tt, yy = np.meshgrid(t_axis, y_axis, indexing="ij")
    return {
        "t": torch.tensor(tt.reshape(-1, 1), device=device, dtype=torch.float32),
        "y": torch.tensor(yy.reshape(-1, 1), device=device, dtype=torch.float32),
        "margin": float(margin),
    }


def closed_form_wealth_bundle(
    t_np: np.ndarray, w_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form (V,V_w,V_ww) for the paper X_ev norm."""
    tau = T_FINAL - t_np
    scale = mem.crra_homothetic_scale(
        tau, nu, gamma_risk, epsilon_bequest)
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


def eval_diag_metrics(
    value_net: nn.Module,
    diag: Dict[str, torch.Tensor],
    diag_col: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Fixed-Q_ev bundle/control diagnostics plus Q_col ellipticity."""
    was_training = value_net.training
    value_net.eval()
    t = diag["t"].detach().clone().requires_grad_(True)
    y = diag["y"].detach().clone().requires_grad_(True)
    V, _, V_y, V_yy = compute_derivatives_log(value_net, t, y)
    W = torch.exp(y)
    V_w = V_y / W
    V_ww = (V_yy - V_y) / (W ** 2)
    curvature_y = V_y - V_yy

    pi_raw_ev = compute_pi_from_foc_log_multi(
        V_y, V_yy, Sigma_inv_mu, return_raw=True)
    pi_ev = pi_raw_ev if pi_clip_abs is None else torch.clamp(
        pi_raw_ev, -float(pi_clip_abs), float(pi_clip_abs))

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
    pi_np = pi_ev.detach().cpu().numpy()
    c_np = c_eval.detach().cpu().numpy()
    pi_cf = np.broadcast_to(pi_star_np.reshape(1, -1), pi_np.shape)

    e_V = float(np.max(np.abs(V_np - V_cf)))
    e_D = float(np.max(np.sqrt(
        (Vw_np - Vw_cf) ** 2 + (Vww_np - Vww_cf) ** 2)))

    # Ellipticity is measured on Q_col, not the contracted Q_ev grid.
    pi_raw_col = pi_raw_ev
    pi_col = pi_ev
    if diag_col is not None:
        t_col = diag_col["t"].detach().clone().requires_grad_(True)
        y_col = diag_col["y"].detach().clone().requires_grad_(True)
        _, _, Vy_col, Vyy_col = compute_derivatives_log(value_net, t_col, y_col)
        pi_raw_col = compute_pi_from_foc_log_multi(
            Vy_col, Vyy_col, Sigma_inv_mu, return_raw=True)
        pi_col = pi_raw_col if pi_clip_abs is None else torch.clamp(
            pi_raw_col, -float(pi_clip_abs), float(pi_clip_abs))
    diffusion = _diffusion_variance(pi_col)
    pi_l2 = torch.linalg.vector_norm(pi_ev, dim=1)
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
        "m_curvature_y": float(curvature_y.min().item()),
        # Paper E1 derivative margins in the network's log-wealth coordinate.
        "m_y": float(V_y.min().item()),
        "M_y": float(V_y.max().item()),
        "m_c": float(curvature_y.min().item()),
        # Wealth-normalized controls on the fixed Q_ev diagnostic grid.
        "pi_component_min_greedy": float(pi_ev.min().item()),
        "pi_component_max_greedy": float(pi_ev.max().item()),
        "pi_l2_min_greedy": float(pi_l2.min().item()),
        "pi_l2_max_greedy": float(pi_l2.max().item()),
        "chi_min_greedy": float(chi_eval.min().item()),
        "chi_max_greedy": float(chi_eval.max().item()),
        "guard_frac_Vw": float((V_w <= 1e-8).float().mean().item()),
        "guard_frac_curvature": float((curvature_y <= 1e-8).float().mean().item()),
        "clip_frac_pi_greedy": _clip_fraction_pi(pi_raw_ev),
        "clip_frac_kappa_low": (0.0 if kappa_min_bound is None else
            float((kappa_raw <= float(kappa_min_bound) + 1e-7).float().mean().item())),
        "clip_frac_kappa_high": (0.0 if kappa_max_bound is None else
            float((kappa_raw >= float(kappa_max_bound) - 1e-7).float().mean().item())),
        "clip_frac_c_level_low": (0.0 if c_min_bound is None else
            float((c_level_raw <= float(c_min_bound) + 1e-7).float().mean().item())),
        "clip_frac_c_level_high": (0.0 if c_max_bound is None else
            float((c_level_raw >= float(c_max_bound) - 1e-7).float().mean().item())),
        "diffusion_var_min_greedy": float(diffusion.min().item()),
        "diffusion_var_max_greedy": float(diffusion.max().item()),
    }
    if was_training:
        value_net.train()
    return out


# =============================================================================
# 9) Training: Hybrid-sampling PINN on reduced nonlinear PDE
# =============================================================================
def train_pinn_hybrid_reduced_logw_multi(
    value_net: nn.Module,
    epochs: int = 100000,
    batch_size: int = 3000,
    terminal_frac: float = 0.5,
    lr: float = 5e-4,
    eval_epochs: int = 200,
    outer_iters: int = 500,
    resample_every: int = 0,
    w_terminal: float = 20.0,
    w_shape: float = 1.0,
    w_eta: float = 0.0,
    eta_clip: Optional[float] = 10.0,
    scheduler_patience: int = 5000,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-8,
    lr_schedule: str = "plateau",
    qsel_rollback_factor: float = 10.0,
    qsel_rollback_lr_factor: float = 1.0,
    qsel_rollback_max_rescues: int = 2,
    hjb_guard_mode: str = "softplus",
    hjb_vy_guard_eps: float = 1e-8,
    hjb_denom_guard_eps: float = 1e-8,
    hjb_vy_guard_tau: float = 1e-2,
    hjb_denom_guard_tau: float = 1e-2,
    hjb_vy_guard_tau_min: float = 1e-4,
    hjb_denom_guard_tau_min: float = 1e-4,
    hjb_guard_anneal_every: int = 0,
    hjb_guard_anneal_factor: float = 0.5,
    save_iterate_every: int = 0,
    pres_target: Optional[float] = None,
    val_points: int = 100000,
    val_terminal_points: int = 10000,
    val_every: int = 1,
    val_seed: int = 0,
    diag_points: int = 4096,
    diag_margin: float = 0.1,
    diag_every: int = 1,
    timing_mode: bool = False,
    print_every: int = 2000,
    weight_dir: str = "weights/merton_multiasset_consumption_reduced_logw",
    recorder: Optional[mxu.ExperimentRecorder] = None,
    stopper: Optional[mxu.PDEEarlyStopper] = None,
) -> Tuple[List[Dict[str, float]], optim.Optimizer, Dict[str, object]]:
    """Train the direct nonlinear PINN in Liu-style pseudo-outer blocks.

    Each block starts with a newly sampled training batch.  When
    ``resample_every=K>0``, both the interior and terminal samples are
    refreshed after each K inner optimizer steps; ``K=0`` keeps that batch
    fixed for the full block.  Fixed held-out Q_res supplies the optional
    stopping residual, while an independently seeded fixed Q_sel supplies the
    *only* ReduceLROnPlateau score.  Reaching pres_target ends the single
    nonlinear solve globally; it is a successful tolerance stop, not
    divergence.  When plateau scheduling is active, the optional Q_sel
    rollback restores the last Q_sel-admissible model and Adam state if and
    only if the raw current Q_sel p_sel exceeds its configured multiple of
    the prior best.
    """
    if int(val_every) < 1:
        raise ValueError("val_every must be positive")
    if int(resample_every) < 0:
        raise ValueError("resample_every must be nonnegative")
    if (not math.isfinite(float(qsel_rollback_factor))
            or float(qsel_rollback_factor) <= 1.0):
        raise ValueError(
            "qsel_rollback_factor must be finite and greater than 1")
    if (not math.isfinite(float(qsel_rollback_lr_factor))
            or not (0.0 < float(qsel_rollback_lr_factor) <= 1.0)):
        raise ValueError(
            "qsel_rollback_lr_factor must lie in (0, 1]")
    if int(qsel_rollback_max_rescues) < 0:
        raise ValueError("qsel_rollback_max_rescues must be nonnegative")
    if (lr_schedule == "plateau" and int(qsel_rollback_max_rescues) > 0
            and int(eval_epochs) % int(val_every) != 0):
        raise ValueError(
            "Q_sel rollback requires eval_epochs divisible by val_every")
    if lr_schedule == "plateau" and float(lr) < float(scheduler_min_lr):
        raise ValueError("lr must not be below scheduler_min_lr")
    if lr_schedule == "plateau" and (
            int(val_points) <= 0 or int(val_terminal_points) <= 0):
        raise ValueError(
            "plateau scheduling requires positive fixed-Q_sel point counts")
    os.makedirs(weight_dir, exist_ok=True)
    best_model_path = os.path.join(weight_dir, "value_net_best_diag.pt")
    last_model_path = os.path.join(weight_dir, "value_net_last.pt")
    final_model_path = os.path.join(weight_dir, "value_net_final.pt")
    iterate_dir = os.path.join(weight_dir, "iterates")
    if save_iterate_every > 0 and not timing_mode:
        os.makedirs(iterate_dir, exist_ok=True)

    optimizer = optim.Adam(value_net.parameters(), lr=lr)

    def _make_scheduler():
        if lr_schedule != "plateau":
            return None
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=scheduler_min_lr)

    scheduler = _make_scheduler()

    loss_history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_iter = 0
    total_optimizer_steps = 0
    completed_outers = 0
    scheduler_checks = 0
    scheduler_lr_reductions = 0
    training_batch_resamples_total = 0
    training_batches_sampled_total = 0
    last_sel: Optional[Tuple[float, float, float]] = None
    last_sel_epoch = -1
    best_sel_pres = float("inf")
    qsel_rollback_active = bool(
        scheduler is not None and int(qsel_rollback_max_rescues) > 0)
    qsel_rollback_triggers = 0
    qsel_rollback_rescues_used = 0
    qsel_rollback_scheduler_resets = 0
    qsel_rollback_last_epoch = -1
    qsel_rollback_last_observed_pres: Optional[float] = None
    qsel_rollback_baseline_epoch = -1
    qsel_rollback_baseline_pres: Optional[float] = None
    qsel_rollback_last_restored_epoch = -1
    qsel_rollback_last_restored_pres: Optional[float] = None
    qsel_rollback_failed = False
    qsel_rollback_failure_reason = ""
    last_admissible_model_state: Optional[Dict[str, torch.Tensor]] = None
    last_admissible_optimizer_state: Optional[Dict[str, object]] = None
    last_admissible_epoch = -1
    last_admissible_pres: Optional[float] = None

    # Dedicated held-out streams do not advance the training RNG. Q_res and
    # Q_sel have separate deterministic streams, and both are identical across
    # training seeds when the market seed is fixed.  Q_res is never passed to
    # the scheduler; Q_sel is never used to declare pres_target success.
    val_set = None
    if val_points > 0 and not (timing_mode and pres_target is None):
        val_set = build_validation_set(
            int(val_points), max(1, int(val_terminal_points)), device, int(val_seed))
    sel_seed = int(val_seed) + DIRECT_PINN_QSEL_SEED_OFFSET
    sel_set = None
    if scheduler is not None:
        sel_set = build_validation_set(
            int(val_points), max(1, int(val_terminal_points)), device, sel_seed)
    diag = None
    diag_col = None
    if diag_points > 0 and not timing_mode:
        diag = build_diag_set(int(diag_points), float(diag_margin))
        # E1 ellipticity uses a fixed dense tensor grid on all of Q_col.
        diag_col = build_diag_set(int(diag_points), 0.0)

    train_fields = [
        "timestamp", "model_type", "run_tag", "epoch", "outer_iter", "inner_epoch",
        "total_loss", "pde_loss", "terminal_loss", "monotonicity_loss",
        "concavity_loss", "eta_loss", "train_pres", "val_pde_rms",
        "val_terminal_rms", "val_pres", "sel_pde_rms",
        "sel_terminal_rms", "sel_pres", "scheduler_score",
        "scheduler_check", "qsel_rollback_triggered",
        "qsel_rollback_rescued", "qsel_rollback_failed",
        "qsel_rollback_observed_pres", "qsel_rollback_restored_pres",
        "qsel_rollback_restored_epoch", "qsel_rollback_rescues_used",
        "hjb_guard_mode",
        "hjb_guard_tau_vy", "hjb_guard_tau_denom",
        "hjb_guard_frac_vy", "hjb_guard_frac_denom",
        "lr", "best_loss", "elapsed_sec",
        "stopped", "stop_reason",
    ]
    outer_fields = [
        "timestamp", "model_type", "run_tag", "outer_iter", "epoch", "total_loss",
        "pde_loss", "terminal_loss", "monotonicity_loss", "concavity_loss",
        "eta_loss", "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres",
        "sel_pde_rms", "sel_terminal_rms", "sel_pres", "scheduler_score",
        "sel_check_epoch", "scheduler_checks",
        "qsel_rollback_triggers", "qsel_rollback_rescues_used",
        "qsel_rollback_scheduler_resets", "qsel_rollback_last_epoch",
        "qsel_rollback_last_observed_pres",
        "qsel_rollback_baseline_epoch", "qsel_rollback_baseline_pres",
        "qsel_rollback_last_admissible_epoch",
        "qsel_rollback_last_admissible_pres",
        "qsel_rollback_last_restored_epoch",
        "qsel_rollback_last_restored_pres", "qsel_rollback_failed",
        "qsel_rollback_failure_reason",
        "training_batch_resamples_outer", "training_batches_sampled_outer",
        "training_batch_resamples_total", "training_batches_sampled_total",
        "inner_epochs_used", "target_reached", "control_metric_scope",
        "control_metric_points", "e_V_sup", "e_bundle_sup", "e_Xev",
        "diag_RelL2_V", "diag_RelL2_pi", "diag_RelL2_c", "m_Vw",
        "m_minus_Vww", "m_curvature_y", "m_y", "M_y", "m_c",
        "pi_component_min_greedy", "pi_component_max_greedy",
        "pi_l2_min_greedy", "pi_l2_max_greedy",
        "chi_min_greedy", "chi_max_greedy",
        "guard_frac_Vw", "guard_frac_curvature",
        "clip_frac_pi_greedy", "clip_frac_kappa_low", "clip_frac_kappa_high",
        "clip_frac_c_level_low", "clip_frac_c_level_high",
        "diffusion_var_min_greedy", "diffusion_var_max_greedy",
        "hjb_guard_mode", "hjb_guard_tau_vy", "hjb_guard_tau_denom",
        "hjb_guard_frac_vy", "hjb_guard_frac_denom", "lr", "best_loss",
        "bad_count", "stop_active", "stop_is_bad", "stopped", "stop_reason",
        "elapsed_sec",
    ]
    diagnostic_fields = [
        "e_V_sup", "e_bundle_sup", "e_Xev", "diag_RelL2_V", "diag_RelL2_pi",
        "diag_RelL2_c", "m_Vw", "m_minus_Vww", "m_curvature_y",
        "m_y", "M_y", "m_c", "pi_component_min_greedy",
        "pi_component_max_greedy", "pi_l2_min_greedy", "pi_l2_max_greedy",
        "chi_min_greedy", "chi_max_greedy",
        "guard_frac_Vw", "guard_frac_curvature", "clip_frac_pi_greedy",
        "clip_frac_kappa_low", "clip_frac_kappa_high", "clip_frac_c_level_low",
        "clip_frac_c_level_high", "diffusion_var_min_greedy",
        "diffusion_var_max_greedy",
    ]

    print(f"\n{'='*70}")
    print("Training PINN (Reduced-form HJB, logW, hybrid sampling)")
    print(f"  outer_iters={outer_iters}, eval_epochs={eval_epochs}, epochs={epochs}")
    print(f"  batch={batch_size}, terminal_frac={terminal_frac}, lr={lr}")
    print(
        "  training batch="
        + ("fixed within each outer block" if int(resample_every) == 0 else
           f"refreshed every {int(resample_every)} inner optimizer steps")
    )
    print(f"  w_terminal={w_terminal}, w_shape={w_shape}, w_eta={w_eta}")
    print(
        f"  policy bounds={policy_bounds_mode}; pi=[{POLICY_METADATA['policy_pi_min']},"
        f"{POLICY_METADATA['policy_pi_max']}], kappa=[{kappa_min_bound},{kappa_max_bound}], "
        f"c=[{c_min_bound},{c_max_bound}]"
    )
    print(
        f"  HJB guard={hjb_guard_mode} ({HJB_GUARD_VERSION}); "
        f"eps=({hjb_vy_guard_eps:g},{hjb_denom_guard_eps:g}), "
        f"tau0=({hjb_vy_guard_tau:g},{hjb_denom_guard_tau:g}), "
        f"anneal_every={hjb_guard_anneal_every}"
    )
    if scheduler is not None:
        print(
            f"  scheduler=fixed Q_sel {HELDOUT_SCORE_FORMULA}; "
            f"every {val_every} optimizer steps; patience={scheduler_patience} Q_sel checks; "
            f"Q_res seed={int(val_seed)}, Q_sel seed={sel_seed}"
        )
        if qsel_rollback_active:
            print(
                "  Q_sel rollback="
                f"p_sel > {float(qsel_rollback_factor):g} * best; "
                f"lr_factor={float(qsel_rollback_lr_factor):g}; "
                f"max_rescues={int(qsel_rollback_max_rescues)}"
            )
    print(f"{'='*70}\n")
    start_time = time.time()
    last_val: Optional[Tuple[float, float, float]] = None
    current = {
        "total": float("nan"), "pde": float("nan"), "terminal": float("nan"),
        "mono": float("nan"), "conc": float("nan"), "eta": float("nan"),
        "guard_frac_vy": float("nan"), "guard_frac_denom": float("nan"),
    }
    stop_info: Dict[str, object] = {
        "stopped_early": False, "target_reached": False,
        "total_optimizer_steps": 0,
    }

    def _heldout(tau_vy_now: float, tau_denom_now: float) -> Optional[Tuple[float, float, float]]:
        if val_set is None:
            return None
        return evaluate_heldout_pres_pinn(
            value_net, val_set, guard_mode=hjb_guard_mode,
            eps_vy=hjb_vy_guard_eps, eps_denom=hjb_denom_guard_eps,
            tau_vy=tau_vy_now, tau_denom=tau_denom_now)

    def _cpu_clone_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: _cpu_clone_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_cpu_clone_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_cpu_clone_tree(item) for item in value)
        return copy.deepcopy(value)

    def _establish_qsel_rollback_baseline(
        tau_vy_now: float,
        tau_denom_now: float,
        epoch_idx: int,
    ) -> bool:
        """Snapshot epoch-0/stage-start safety state without stepping LR."""
        nonlocal best_sel_pres
        nonlocal last_admissible_model_state
        nonlocal last_admissible_optimizer_state
        nonlocal last_admissible_epoch, last_admissible_pres
        nonlocal qsel_rollback_baseline_epoch
        nonlocal qsel_rollback_baseline_pres
        nonlocal qsel_rollback_triggers, qsel_rollback_last_epoch
        nonlocal qsel_rollback_last_observed_pres
        nonlocal qsel_rollback_failed, qsel_rollback_failure_reason
        if not qsel_rollback_active or sel_set is None:
            return True
        baseline = evaluate_heldout_pres_pinn(
            value_net, sel_set, guard_mode=hjb_guard_mode,
            eps_vy=hjb_vy_guard_eps, eps_denom=hjb_denom_guard_eps,
            tau_vy=tau_vy_now, tau_denom=tau_denom_now)
        baseline_pres = float(baseline[2])
        qsel_rollback_baseline_epoch = int(epoch_idx)
        qsel_rollback_baseline_pres = baseline_pres
        if not math.isfinite(baseline_pres):
            qsel_rollback_triggers += 1
            qsel_rollback_last_epoch = int(epoch_idx)
            qsel_rollback_last_observed_pres = baseline_pres
            qsel_rollback_failed = True
            qsel_rollback_failure_reason = (
                "qsel_rollback_no_finite_admissible_checkpoint")
            return False
        best_sel_pres = baseline_pres
        last_admissible_model_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in value_net.state_dict().items()
        }
        last_admissible_optimizer_state = _cpu_clone_tree(
            optimizer.state_dict())
        last_admissible_epoch = int(epoch_idx)
        last_admissible_pres = baseline_pres
        return True

    def _scheduler_check(
        tau_vy_now: float,
        tau_denom_now: float,
        epoch_idx: int,
    ) -> Optional[Dict[str, object]]:
        """Evaluate fixed Q_sel, rescue if needed, then seed/step the scheduler."""
        nonlocal scheduler
        nonlocal scheduler_checks, scheduler_lr_reductions
        nonlocal last_sel, last_sel_epoch, best_sel_pres
        nonlocal qsel_rollback_triggers, qsel_rollback_rescues_used
        nonlocal qsel_rollback_scheduler_resets, qsel_rollback_last_epoch
        nonlocal qsel_rollback_last_observed_pres
        nonlocal qsel_rollback_last_restored_epoch
        nonlocal qsel_rollback_last_restored_pres
        nonlocal qsel_rollback_failed, qsel_rollback_failure_reason
        nonlocal last_admissible_model_state
        nonlocal last_admissible_optimizer_state
        nonlocal last_admissible_epoch, last_admissible_pres
        if scheduler is None or sel_set is None:
            return None
        observed = evaluate_heldout_pres_pinn(
            value_net, sel_set, guard_mode=hjb_guard_mode,
            eps_vy=hjb_vy_guard_eps, eps_denom=hjb_denom_guard_eps,
            tau_vy=tau_vy_now, tau_denom=tau_denom_now)
        observed_pres = float(observed[2])
        scheduler_checks += 1
        triggered = bool(
            qsel_rollback_active
            and _qsel_rollback_triggered(
                observed_pres, best_sel_pres, qsel_rollback_factor)
        )
        rescued = False
        restored: Optional[Tuple[float, float, float]] = None
        restored_from_epoch: Optional[int] = None

        if triggered:
            qsel_rollback_triggers += 1
            qsel_rollback_last_epoch = int(epoch_idx)
            qsel_rollback_last_observed_pres = observed_pres
            no_checkpoint = (
                last_admissible_model_state is None
                or last_admissible_optimizer_state is None
            )
            rescue_limit_hit = (
                qsel_rollback_rescues_used
                >= int(qsel_rollback_max_rescues)
            )
            if no_checkpoint or rescue_limit_hit:
                qsel_rollback_failed = True
                if no_checkpoint:
                    qsel_rollback_failure_reason = (
                        "qsel_rollback_no_finite_admissible_checkpoint")
                else:
                    qsel_rollback_failure_reason = (
                        "qsel_rollback_rescue_limit_exceeded")
                last_sel = observed
                last_sel_epoch = int(epoch_idx)
                return {
                    "observed": observed, "scheduler_value": observed,
                    "triggered": True, "rescued": False, "failed": True,
                    "restored": None, "restored_from_epoch": None,
                }

            lr_before_restore = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            value_net.load_state_dict(last_admissible_model_state)
            optimizer.load_state_dict(
                copy.deepcopy(last_admissible_optimizer_state))
            restored_from_epoch = int(last_admissible_epoch)
            for group_idx, group in enumerate(optimizer.param_groups):
                pre_lr = lr_before_restore[
                    min(group_idx, len(lr_before_restore) - 1)]
                group["lr"] = _qsel_rollback_lr(
                    current_lr=pre_lr,
                    restored_lr=float(group["lr"]),
                    min_lr=float(scheduler_min_lr),
                    factor=float(qsel_rollback_lr_factor),
                )

            # The plateau history belongs to the discarded trajectory.  Seed
            # a fresh scheduler with Q_sel re-evaluated at the restored state.
            scheduler = _make_scheduler()
            qsel_rollback_scheduler_resets += 1
            restored = evaluate_heldout_pres_pinn(
                value_net, sel_set, guard_mode=hjb_guard_mode,
                eps_vy=hjb_vy_guard_eps, eps_denom=hjb_denom_guard_eps,
                tau_vy=tau_vy_now, tau_denom=tau_denom_now)
            if not math.isfinite(float(restored[2])):
                qsel_rollback_failed = True
                qsel_rollback_failure_reason = (
                    "qsel_rollback_restored_state_nonfinite")
                last_sel = restored
                last_sel_epoch = int(epoch_idx)
                return {
                    "observed": observed, "scheduler_value": restored,
                    "triggered": True, "rescued": False, "failed": True,
                    "restored": restored,
                    "restored_from_epoch": restored_from_epoch,
                }
            qsel_rollback_rescues_used += 1
            qsel_rollback_last_restored_epoch = restored_from_epoch
            qsel_rollback_last_restored_pres = float(restored[2])
            rescued = True

        scheduler_value = restored if restored is not None else observed
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(float(scheduler_value[2]))
        lr_after = float(optimizer.param_groups[0]["lr"])
        scheduler_lr_reductions += int(lr_after < lr_before)
        last_sel = scheduler_value
        last_sel_epoch = int(epoch_idx)
        best_sel_pres = min(best_sel_pres, float(scheduler_value[2]))

        # This state is admissible under the sole Q_sel ratio criterion. Save
        # model and Adam together, after any scheduler LR update, so a future
        # restore never mixes parameter and moment/LR vintages.
        last_admissible_model_state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in value_net.state_dict().items()
        }
        last_admissible_optimizer_state = _cpu_clone_tree(
            optimizer.state_dict())
        last_admissible_epoch = int(epoch_idx)
        last_admissible_pres = float(scheduler_value[2])
        return {
            "observed": observed, "scheduler_value": scheduler_value,
            "triggered": triggered, "rescued": rescued, "failed": False,
            "restored": restored,
            "restored_from_epoch": restored_from_epoch,
        }

    def _sample_training_batch(
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample the coupled interior/terminal batch and terminal target."""
        t_batch, y_batch = sample_interior(int(batch_size), device)
        t_terminal, y_terminal = sample_terminal(
            max(1, int(batch_size * terminal_frac)), device)
        terminal_target = V_terminal_from_y(y_terminal).detach()
        return t_batch, y_batch, t_terminal, y_terminal, terminal_target

    def _rollback_status() -> Dict[str, object]:
        return {
            "qsel_rollback_active": bool(qsel_rollback_active),
            "qsel_rollback_factor": float(qsel_rollback_factor),
            "qsel_rollback_lr_factor": float(qsel_rollback_lr_factor),
            "qsel_rollback_max_rescues":
                int(qsel_rollback_max_rescues),
            "qsel_rollback_triggers": int(qsel_rollback_triggers),
            "qsel_rollback_rescues_used":
                int(qsel_rollback_rescues_used),
            "qsel_rollback_scheduler_resets":
                int(qsel_rollback_scheduler_resets),
            "qsel_rollback_last_epoch": (
                None if qsel_rollback_last_epoch < 0
                else int(qsel_rollback_last_epoch)),
            "qsel_rollback_last_observed_pres":
                qsel_rollback_last_observed_pres,
            "qsel_rollback_baseline_epoch": (
                None if qsel_rollback_baseline_epoch < 0
                else int(qsel_rollback_baseline_epoch)),
            "qsel_rollback_baseline_pres":
                qsel_rollback_baseline_pres,
            "qsel_rollback_last_admissible_epoch": (
                None if last_admissible_epoch < 0
                else int(last_admissible_epoch)),
            "qsel_rollback_last_admissible_pres":
                last_admissible_pres,
            "qsel_rollback_last_restored_epoch": (
                None if qsel_rollback_last_restored_epoch < 0
                else int(qsel_rollback_last_restored_epoch)),
            "qsel_rollback_last_restored_pres":
                qsel_rollback_last_restored_pres,
            "qsel_rollback_failed": bool(qsel_rollback_failed),
            "qsel_rollback_failure_reason":
                qsel_rollback_failure_reason,
        }

    previous_guard_taus: Optional[Tuple[float, float]] = None
    tau_vy_now = float(hjb_vy_guard_tau)
    tau_denom_now = float(hjb_denom_guard_tau)
    for outer_iter in range(1, int(outer_iters) + 1):
        tau_vy_now, tau_denom_now, guard_stage = hjb_guard_taus_for_outer(
            outer_iter=outer_iter,
            tau_vy_initial=hjb_vy_guard_tau,
            tau_denom_initial=hjb_denom_guard_tau,
            tau_vy_min=hjb_vy_guard_tau_min,
            tau_denom_min=hjb_denom_guard_tau_min,
            anneal_every=hjb_guard_anneal_every,
            anneal_factor=hjb_guard_anneal_factor,
        )
        guard_taus = (tau_vy_now, tau_denom_now)
        if (hjb_guard_mode == "softplus" and previous_guard_taus is not None
                and guard_taus != previous_guard_taus and scheduler is not None):
            # The surrogate PDE and therefore Q_sel's scale change with tau.
            # Never compare a new stage against a prior-stage rollback best.
            scheduler = _make_scheduler()
            last_sel = None
            last_sel_epoch = -1
            best_sel_pres = float("inf")
            last_admissible_model_state = None
            last_admissible_optimizer_state = None
            last_admissible_epoch = -1
            last_admissible_pres = None
            print(
                f"[guard] stage={guard_stage}: tau_vy={tau_vy_now:.3e}, "
                f"tau_denom={tau_denom_now:.3e}; "
                "plateau and Q_sel rollback baseline reset"
            )
        previous_guard_taus = guard_taus
        # Every block starts with a fresh batch.  Optional inner-loop
        # resampling below refreshes this same coupled interior/terminal set.
        t_int, y_int, t_term, y_term, V_T_target = _sample_training_batch()
        training_batches_sampled_total += 1
        training_batch_resamples_outer = 0
        training_batches_sampled_outer = 1
        pending_rows: List[Dict[str, object]] = []
        epochs_used = 0
        target_reached = False
        steps_since_batch_refresh = 0
        rollback_terminal_failure = False

        # Establish an epoch-0 admissible baseline before the first update of
        # the run (and after every tau-stage reset). This makes the very first
        # Q_sel interval rescuable and snapshots model+Adam on CPU.
        if qsel_rollback_active and last_admissible_model_state is None:
            rollback_terminal_failure = not _establish_qsel_rollback_baseline(
                tau_vy_now, tau_denom_now, total_optimizer_steps)
            if rollback_terminal_failure:
                print(
                    "[rollback:failed] unable to establish finite Q_sel "
                    f"baseline at step={total_optimizer_steps}: "
                    f"{qsel_rollback_failure_reason}"
                )

        # A state already at tolerance must not be perturbed by another step.
        if pres_target is not None and not rollback_terminal_failure:
            last_val = _heldout(tau_vy_now, tau_denom_now)
            target_reached = bool(last_val is not None and last_val[2] <= float(pres_target))
            if target_reached:
                # Epoch 0 can legitimately satisfy the target in any block.
                # Re-measure the current state on this block's fresh batch and
                # active tau, rather than carrying a loss/guard fraction from
                # the preceding block (especially after tau continuation).
                residual0, _, Vy0, Vyy0, denom0 = reduced_hjb_residual_log_multi(
                    value_net, t_int, y_int, guard_mode=hjb_guard_mode,
                    eps_vy=hjb_vy_guard_eps,
                    eps_denom=hjb_denom_guard_eps,
                    tau_vy=tau_vy_now, tau_denom=tau_denom_now)
                with torch.no_grad():
                    term0 = torch.mean((value_net(t_term, y_term) - V_T_target) ** 2)
                pde0 = torch.mean(residual0.detach() ** 2)
                mono0 = torch.mean(torch.relu(-Vy0.detach()) ** 2)
                conc0 = torch.mean(torch.relu(-denom0.detach()) ** 2)
                if w_eta != 0.0 and eta_clip is not None:
                    eta0 = 1.0 - Vyy0.detach() / torch.clamp(
                        torch.abs(Vy0.detach()), min=1e-8)
                    eta_loss0 = torch.mean(torch.clamp(
                        eta0 - gamma_risk, -float(eta_clip), float(eta_clip)) ** 2)
                else:
                    eta_loss0 = torch.zeros((), device=t_int.device)
                current = {
                    "pde": float(pde0.item()), "terminal": float(term0.item()),
                    "mono": float(mono0.item()), "conc": float(conc0.item()),
                    "eta": float(eta_loss0.item()),
                    "guard_frac_vy": float(
                        (Vy0.detach() <= float(hjb_vy_guard_eps)).float().mean().item()),
                    "guard_frac_denom": float(
                        (denom0.detach() <= float(hjb_denom_guard_eps)).float().mean().item()),
                }
                current["total"] = float(
                    current["pde"] + w_terminal * current["terminal"]
                    + w_shape * (current["mono"] + current["conc"])
                    + w_eta * current["eta"])
                if current["total"] < best_loss:
                    best_loss = current["total"]
                    best_iter = total_optimizer_steps
                    if not timing_mode:
                        best_state = {
                            key: val.detach().cpu().clone()
                            for key, val in value_net.state_dict().items()}
                synthetic_history = {
                    "total": current["total"], "pde": current["pde"],
                    "terminal": current["terminal"], "mono": current["mono"],
                    "conc": current["conc"], "eta": current["eta"],
                    "hjb_guard_mode": hjb_guard_mode,
                    "hjb_guard_tau_vy": tau_vy_now,
                    "hjb_guard_tau_denom": tau_denom_now,
                    "hjb_guard_frac_vy": current["guard_frac_vy"],
                    "hjb_guard_frac_denom": current["guard_frac_denom"],
                }
                if timing_mode and loss_history:
                    loss_history[-1] = synthetic_history
                else:
                    loss_history.append(synthetic_history)
                if not timing_mode:
                    pending_rows.append({
                        "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                        "run_tag": ARGS.run_tag, "epoch": total_optimizer_steps,
                        "outer_iter": outer_iter, "inner_epoch": 0,
                        "total_loss": current["total"], "pde_loss": current["pde"],
                        "terminal_loss": current["terminal"],
                        "monotonicity_loss": current["mono"],
                        "concavity_loss": current["conc"], "eta_loss": current["eta"],
                        "train_pres": mxu.pres_from_mse(
                            current["pde"], current["terminal"]),
                        "val_pde_rms": last_val[0],
                        "val_terminal_rms": last_val[1], "val_pres": last_val[2],
                        "sel_pde_rms": "", "sel_terminal_rms": "",
                        "sel_pres": "", "scheduler_score": "",
                        "scheduler_check": 0,
                        "qsel_rollback_triggered": 0,
                        "qsel_rollback_rescued": 0,
                        "qsel_rollback_failed": 0,
                        "qsel_rollback_observed_pres": "",
                        "qsel_rollback_restored_pres": "",
                        "qsel_rollback_restored_epoch": "",
                        "qsel_rollback_rescues_used":
                            qsel_rollback_rescues_used,
                        "hjb_guard_mode": hjb_guard_mode,
                        "hjb_guard_tau_vy": tau_vy_now,
                        "hjb_guard_tau_denom": tau_denom_now,
                        "hjb_guard_frac_vy": current["guard_frac_vy"],
                        "hjb_guard_frac_denom": current["guard_frac_denom"],
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "best_loss": best_loss, "elapsed_sec": time.time() - start_time,
                        "stopped": 0, "stop_reason": "pres_target_reached",
                    })

        for inner_epoch in range(1, int(eval_epochs) + 1):
            if target_reached or rollback_terminal_failure:
                break
            # The current batch receives exactly ``resample_every`` updates.
            # Refresh before K+1 (and 2K+1, ...) rather than after computing
            # that step's loss.  Zero intentionally disables inner refreshes.
            if _resample_before_inner_step(
                    steps_since_batch_refresh + 1, int(resample_every)):
                t_int, y_int, t_term, y_term, V_T_target = _sample_training_batch()
                steps_since_batch_refresh = 0
                training_batch_resamples_outer += 1
                training_batches_sampled_outer += 1
                training_batch_resamples_total += 1
                training_batches_sampled_total += 1
            optimizer.zero_grad(set_to_none=True)
            residual, _, V_y, V_yy, denom = reduced_hjb_residual_log_multi(
                value_net, t_int, y_int, guard_mode=hjb_guard_mode,
                eps_vy=hjb_vy_guard_eps,
                eps_denom=hjb_denom_guard_eps,
                tau_vy=tau_vy_now, tau_denom=tau_denom_now)
            pde_loss = torch.mean(residual ** 2)
            V_T_pred = value_net(t_term, y_term)
            terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)
            mono_penalty = torch.mean(torch.relu(-V_y) ** 2)
            conc_penalty = torch.mean(torch.relu(-denom) ** 2)
            if w_eta != 0.0 and eta_clip is not None:
                V_y_safe = torch.clamp(torch.abs(V_y), min=1e-8)
                eta = 1.0 - V_yy / V_y_safe
                eta_err = torch.clamp(
                    eta - gamma_risk, -float(eta_clip), float(eta_clip))
                eta_loss = torch.mean(eta_err ** 2)
            else:
                eta_loss = torch.zeros((), device=t_int.device)
            total_loss = (
                pde_loss + w_terminal * terminal_loss
                + w_shape * (mono_penalty + conc_penalty) + w_eta * eta_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            optimizer.step()
            total_optimizer_steps += 1
            steps_since_batch_refresh += 1
            epochs_used = inner_epoch

            current = {
                "total": float(total_loss.detach().cpu()),
                "pde": float(pde_loss.detach().cpu()),
                "terminal": float(terminal_loss.detach().cpu()),
                "mono": float(mono_penalty.detach().cpu()),
                "conc": float(conc_penalty.detach().cpu()),
                "eta": float(eta_loss.detach().cpu()),
                "guard_frac_vy": float(
                    (V_y.detach() <= float(hjb_vy_guard_eps)).float().mean().item()),
                "guard_frac_denom": float(
                    (denom.detach() <= float(hjb_denom_guard_eps)).float().mean().item()),
            }
            lr_now = float(optimizer.param_groups[0]["lr"])

            row_val: Optional[Tuple[float, float, float]] = None
            row_sel: Optional[Tuple[float, float, float]] = None
            rollback_event: Optional[Dict[str, object]] = None
            if (val_set is not None and pres_target is not None and
                    total_optimizer_steps % max(1, int(val_every)) == 0):
                row_val = _heldout(tau_vy_now, tau_denom_now)
                last_val = row_val
                if (pres_target is not None and row_val is not None and
                        row_val[2] <= float(pres_target)):
                    target_reached = True
            if (scheduler is not None and
                    total_optimizer_steps % max(1, int(val_every)) == 0):
                # Never use the stochastic training loss (or Q_res) here.
                rollback_event = _scheduler_check(
                    tau_vy_now, tau_denom_now, total_optimizer_steps)
                if rollback_event is not None:
                    row_sel = rollback_event["scheduler_value"]
                    rollback_terminal_failure = bool(
                        rollback_event["failed"])
                    if bool(rollback_event["rescued"]):
                        # The discarded stochastic batch belongs to the bad
                        # trajectory. Start a fresh cadence from a coupled
                        # interior/terminal/target batch.
                        t_int, y_int, t_term, y_term, V_T_target = (
                            _sample_training_batch())
                        steps_since_batch_refresh = 0
                        training_batch_resamples_outer += 1
                        training_batches_sampled_outer += 1
                        training_batch_resamples_total += 1
                        training_batches_sampled_total += 1

                        # Replace the normal training columns with metrics of
                        # the restored model on its fresh batch. The raw bad
                        # Q_sel value remains in dedicated rollback columns.
                        residual_r, _, vy_r, vyy_r, denom_r = (
                            reduced_hjb_residual_log_multi(
                                value_net, t_int, y_int,
                                guard_mode=hjb_guard_mode,
                                eps_vy=hjb_vy_guard_eps,
                                eps_denom=hjb_denom_guard_eps,
                                tau_vy=tau_vy_now,
                                tau_denom=tau_denom_now))
                        with torch.no_grad():
                            term_r = torch.mean(
                                (value_net(t_term, y_term) - V_T_target) ** 2)
                        pde_r = torch.mean(residual_r.detach() ** 2)
                        mono_r = torch.mean(
                            torch.relu(-vy_r.detach()) ** 2)
                        conc_r = torch.mean(
                            torch.relu(-denom_r.detach()) ** 2)
                        if w_eta != 0.0 and eta_clip is not None:
                            eta_r = 1.0 - vyy_r.detach() / torch.clamp(
                                torch.abs(vy_r.detach()), min=1e-8)
                            eta_loss_r = torch.mean(torch.clamp(
                                eta_r - gamma_risk, -float(eta_clip),
                                float(eta_clip)) ** 2)
                        else:
                            eta_loss_r = torch.zeros(
                                (), device=t_int.device)
                        current = {
                            "pde": float(pde_r.item()),
                            "terminal": float(term_r.item()),
                            "mono": float(mono_r.item()),
                            "conc": float(conc_r.item()),
                            "eta": float(eta_loss_r.item()),
                            "guard_frac_vy": float(
                                (vy_r.detach()
                                 <= float(hjb_vy_guard_eps)).float().mean().item()),
                            "guard_frac_denom": float(
                                (denom_r.detach()
                                 <= float(hjb_denom_guard_eps)).float().mean().item()),
                        }
                        current["total"] = float(
                            current["pde"]
                            + w_terminal * current["terminal"]
                            + w_shape * (current["mono"] + current["conc"])
                            + w_eta * current["eta"])

                        # Any diagnostic-best candidate collected after the
                        # restored checkpoint belongs to the discarded
                        # trajectory. Restart that diagnostic selection from
                        # the restored state; the common block below records
                        # this freshly measured loss/model together.
                        best_loss = float("inf")
                        best_iter = total_optimizer_steps
                        best_state = None

                        # A Q_res success observed on the discarded state is
                        # not sticky. Re-evaluate it at the restored model.
                        if val_set is not None and pres_target is not None:
                            row_val = _heldout(tau_vy_now, tau_denom_now)
                            last_val = row_val
                            target_reached = bool(
                                row_val is not None
                                and row_val[2] <= float(pres_target))
                        print(
                            f"[rollback] step={total_optimizer_steps}: "
                            f"raw p_sel={float(rollback_event['observed'][2]):.3e}, "
                            f"restored step={rollback_event['restored_from_epoch']}, "
                            f"restored p_sel={float(row_sel[2]):.3e}, "
                            f"lr={float(optimizer.param_groups[0]['lr']):.2e}, "
                            f"rescue={qsel_rollback_rescues_used}/"
                            f"{int(qsel_rollback_max_rescues)}"
                        )
                    elif rollback_terminal_failure:
                        target_reached = False
                        print(
                            f"[rollback:failed] step={total_optimizer_steps}: "
                            f"raw p_sel="
                            f"{float(rollback_event['observed'][2]):.3e}; "
                            f"{qsel_rollback_failure_reason}"
                        )

            # Training-loss best is diagnostic only, but it must never retain
            # a state that the Q_sel safety rule just discarded.
            if (not rollback_terminal_failure
                    and current["total"] < best_loss):
                best_loss = current["total"]
                best_iter = total_optimizer_steps
                if not timing_mode:
                    best_state = {
                        key: val.detach().cpu().clone()
                        for key, val in value_net.state_dict().items()}

            # A Q_sel check can reduce the LR, so capture it afterwards.
            lr_now = float(optimizer.param_groups[0]["lr"])

            history_row = {
                "total": current["total"], "pde": current["pde"],
                "terminal": current["terminal"], "mono": current["mono"],
                "conc": current["conc"], "eta": current["eta"],
                "hjb_guard_mode": hjb_guard_mode,
                "hjb_guard_tau_vy": tau_vy_now,
                "hjb_guard_tau_denom": tau_denom_now,
                "hjb_guard_frac_vy": current["guard_frac_vy"],
                "hjb_guard_frac_denom": current["guard_frac_denom"],
            }
            if timing_mode and loss_history:
                loss_history[-1] = history_row
            else:
                loss_history.append(history_row)
            if not timing_mode:
                pending_rows.append({
                    "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag, "epoch": total_optimizer_steps,
                    "outer_iter": outer_iter, "inner_epoch": inner_epoch,
                    "total_loss": current["total"], "pde_loss": current["pde"],
                    "terminal_loss": current["terminal"],
                    "monotonicity_loss": current["mono"],
                    "concavity_loss": current["conc"], "eta_loss": current["eta"],
                    "train_pres": mxu.pres_from_mse(current["pde"], current["terminal"]),
                    "val_pde_rms": "" if row_val is None else row_val[0],
                    "val_terminal_rms": "" if row_val is None else row_val[1],
                    "val_pres": "" if row_val is None else row_val[2],
                    "sel_pde_rms": "" if row_sel is None else row_sel[0],
                    "sel_terminal_rms": "" if row_sel is None else row_sel[1],
                    "sel_pres": "" if row_sel is None else row_sel[2],
                    "scheduler_score": "" if row_sel is None else row_sel[2],
                    "scheduler_check": int(row_sel is not None),
                    "qsel_rollback_triggered": int(
                        bool(rollback_event is not None
                             and rollback_event["triggered"])),
                    "qsel_rollback_rescued": int(
                        bool(rollback_event is not None
                             and rollback_event["rescued"])),
                    "qsel_rollback_failed": int(
                        bool(rollback_event is not None
                             and rollback_event["failed"])),
                    "qsel_rollback_observed_pres": (
                        "" if rollback_event is None
                        else rollback_event["observed"][2]),
                    "qsel_rollback_restored_pres": (
                        "" if rollback_event is None
                        or rollback_event["restored"] is None
                        else rollback_event["restored"][2]),
                    "qsel_rollback_restored_epoch": (
                        "" if rollback_event is None
                        or rollback_event["restored_from_epoch"] is None
                        else rollback_event["restored_from_epoch"]),
                    "qsel_rollback_rescues_used":
                        qsel_rollback_rescues_used,
                    "hjb_guard_mode": hjb_guard_mode,
                    "hjb_guard_tau_vy": tau_vy_now,
                    "hjb_guard_tau_denom": tau_denom_now,
                    "hjb_guard_frac_vy": current["guard_frac_vy"],
                    "hjb_guard_frac_denom": current["guard_frac_denom"],
                    "lr": lr_now, "best_loss": best_loss,
                    "elapsed_sec": time.time() - start_time,
                    "stopped": int(rollback_terminal_failure),
                    "stop_reason": (
                        qsel_rollback_failure_reason
                        if rollback_terminal_failure else
                        ("pres_target_reached" if target_reached else "")),
                })

            if print_every > 0 and total_optimizer_steps % print_every == 0:
                ptxt = "" if row_val is None else f" | p_res={row_val[2]:.3e}"
                stxt = "" if row_sel is None else f" | s_sel={row_sel[2]:.3e}"
                print(f"[{total_optimizer_steps:6d}/{epochs}] total={current['total']:.3e} | "
                      f"pde={current['pde']:.3e} | term={current['terminal']:.3e} | "
                      f"eta={current['eta']:.3e} | lr={lr_now:.2e} | "
                      f"guard=({current['guard_frac_vy']:.2%},"
                      f"{current['guard_frac_denom']:.2%}){ptxt}{stxt}")

            if t_int.grad is not None:
                t_int.grad = None
            if y_int.grad is not None:
                y_int.grad = None
            if target_reached or rollback_terminal_failure:
                break

        # Always report p_res for the official end-of-block state when a
        # validation set exists, even when no target is active.
        if val_set is not None and not rollback_terminal_failure:
            last_val = _heldout(tau_vy_now, tau_denom_now)
            if (pres_target is not None and last_val is not None and
                    last_val[2] <= float(pres_target)):
                target_reached = True
        diag_res: Dict[str, float] = {}
        if (not rollback_terminal_failure and diag is not None and
                (diag_every <= 1 or outer_iter == 1
                 or outer_iter % diag_every == 0 or outer_iter == outer_iters)):
            diag_res = eval_diag_metrics(value_net, diag, diag_col)

        stop_triggered = False
        stop_meta: Dict[str, object] = {
            "active": False, "is_bad": False, "bad_count": 0}
        if (not target_reached and not rollback_terminal_failure
                and stopper is not None and
                math.isfinite(current["pde"])):
            stop_triggered, stop_meta = stopper.update(outer_iter, current["pde"])

        completed_outers = outer_iter
        lr_now = float(optimizer.param_groups[0]["lr"])
        outer_row: Dict[str, object] = {
            "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
            "run_tag": ARGS.run_tag, "outer_iter": outer_iter,
            "epoch": total_optimizer_steps, "total_loss": current["total"],
            "pde_loss": current["pde"], "terminal_loss": current["terminal"],
            "monotonicity_loss": current["mono"], "concavity_loss": current["conc"],
            "eta_loss": current["eta"],
            "train_pres": (mxu.pres_from_mse(current["pde"], current["terminal"])
                           if math.isfinite(current["pde"]) else ""),
            "val_pde_rms": "" if last_val is None else last_val[0],
            "val_terminal_rms": "" if last_val is None else last_val[1],
            "val_pres": "" if last_val is None else last_val[2],
            "sel_pde_rms": "" if last_sel is None else last_sel[0],
            "sel_terminal_rms": "" if last_sel is None else last_sel[1],
            "sel_pres": "" if last_sel is None else last_sel[2],
            "scheduler_score": "" if last_sel is None else last_sel[2],
            "sel_check_epoch": "" if last_sel is None else last_sel_epoch,
            "scheduler_checks": scheduler_checks,
            "qsel_rollback_triggers": qsel_rollback_triggers,
            "qsel_rollback_rescues_used": qsel_rollback_rescues_used,
            "qsel_rollback_scheduler_resets":
                qsel_rollback_scheduler_resets,
            "qsel_rollback_last_epoch": (
                "" if qsel_rollback_last_epoch < 0
                else qsel_rollback_last_epoch),
            "qsel_rollback_last_observed_pres": (
                "" if qsel_rollback_last_observed_pres is None
                else qsel_rollback_last_observed_pres),
            "qsel_rollback_baseline_epoch": (
                "" if qsel_rollback_baseline_epoch < 0
                else qsel_rollback_baseline_epoch),
            "qsel_rollback_baseline_pres": (
                "" if qsel_rollback_baseline_pres is None
                else qsel_rollback_baseline_pres),
            "qsel_rollback_last_admissible_epoch": (
                "" if last_admissible_epoch < 0
                else last_admissible_epoch),
            "qsel_rollback_last_admissible_pres": (
                "" if last_admissible_pres is None
                else last_admissible_pres),
            "qsel_rollback_last_restored_epoch": (
                "" if qsel_rollback_last_restored_epoch < 0
                else qsel_rollback_last_restored_epoch),
            "qsel_rollback_last_restored_pres": (
                "" if qsel_rollback_last_restored_pres is None
                else qsel_rollback_last_restored_pres),
            "qsel_rollback_failed": int(qsel_rollback_failed),
            "qsel_rollback_failure_reason":
                qsel_rollback_failure_reason,
            "training_batch_resamples_outer": training_batch_resamples_outer,
            "training_batches_sampled_outer": training_batches_sampled_outer,
            "training_batch_resamples_total": training_batch_resamples_total,
            "training_batches_sampled_total": training_batches_sampled_total,
            "hjb_guard_mode": hjb_guard_mode,
            "hjb_guard_tau_vy": tau_vy_now,
            "hjb_guard_tau_denom": tau_denom_now,
            "hjb_guard_frac_vy": current["guard_frac_vy"],
            "hjb_guard_frac_denom": current["guard_frac_denom"],
            "inner_epochs_used": epochs_used, "target_reached": int(target_reached),
            # E1 consumes the same fixed tensor Q_ev grid used by the
            # diagnostics below.  Record the scope explicitly so the paper
            # aggregator never has to infer it from diag_points.
            "control_metric_scope": "fixed_qev" if diag_res else "",
            "control_metric_points": (
                int(diag["t"].shape[0]) if diag_res and diag is not None else ""),
            **{key: diag_res.get(key, "") for key in diagnostic_fields},
            "lr": lr_now, "best_loss": best_loss,
            "bad_count": stop_meta.get("bad_count", ""),
            "stop_active": int(bool(stop_meta.get("active", False))),
            "stop_is_bad": int(bool(stop_meta.get("is_bad", False))),
            "stopped": int(bool(stop_triggered or rollback_terminal_failure)),
            "stop_reason": (
                qsel_rollback_failure_reason
                if rollback_terminal_failure else
                ("pres_target_reached" if target_reached
                 else str(stop_meta.get("reason", "")))),
            "elapsed_sec": time.time() - start_time,
        }

        if recorder is not None and not timing_mode:
            mxu.append_csv_rows(recorder.train_csv, pending_rows, train_fields)
            mxu.append_csv_rows(recorder.outer_csv, [outer_row], outer_fields)
        if rollback_terminal_failure:
            # Never create value_net_final/last for a rescue-exhausted run.
            # A clearly non-official CPU snapshot is retained only for audit.
            failed_admissible_path = None
            if last_admissible_model_state is not None and not timing_mode:
                failed_admissible_path = os.path.join(
                    weight_dir, "value_net_failed_last_admissible.pt")
                torch.save(
                    last_admissible_model_state, failed_admissible_path)
            stop_info = {
                "run_failed": True,
                "stopped_early": False,
                "target_reached": False,
                "achieved_pres": None,
                "failure_reason": qsel_rollback_failure_reason,
                "failed_admissible_weight_path": failed_admissible_path,
                "q_res_role": "pres_target_and_official_residual",
                "q_res_seed": int(val_seed),
                "q_sel_role": (
                    "lr_scheduler_and_qsel_rollback"
                    if qsel_rollback_active else "lr_scheduler_only"),
                "q_sel_seed": int(sel_seed),
                "q_sel_active": bool(scheduler is not None),
                "heldout_score_formula": HELDOUT_SCORE_FORMULA,
                "scheduler_metric": (
                    "fixed_q_sel_pres"
                    if scheduler is not None else "none_fixed_lr"),
                "scheduler_patience_unit": "q_sel_checks",
                "scheduler_patience_checks": int(scheduler_patience),
                "scheduler_check_every_optimizer_steps": int(val_every),
                "scheduler_checks": int(scheduler_checks),
                "scheduler_lr_reductions": int(
                    scheduler_lr_reductions),
                "resample_every_optimizer_steps": int(resample_every),
                "training_batch_resamples_total": int(
                    training_batch_resamples_total),
                "training_batches_sampled_total": int(
                    training_batches_sampled_total),
                "sel_pde_rms_last": (
                    None if last_sel is None else float(last_sel[0])),
                "sel_terminal_rms_last": (
                    None if last_sel is None else float(last_sel[1])),
                "sel_pres_last": (
                    None if last_sel is None else float(last_sel[2])),
                "scheduler_score_last": (
                    None if last_sel is None else float(last_sel[2])),
                "sel_pres_best": (
                    None if not math.isfinite(best_sel_pres)
                    else float(best_sel_pres)),
                "sel_check_epoch_last": (
                    None if last_sel is None else int(last_sel_epoch)),
                "outer_iter": int(outer_iter),
                "epoch_at_stop": int(total_optimizer_steps),
                "total_optimizer_steps": int(total_optimizer_steps),
                "hjb_guard_mode": hjb_guard_mode,
                "hjb_guard_version": HJB_GUARD_VERSION,
                "hjb_guard_tau_vy_final": tau_vy_now,
                "hjb_guard_tau_denom_final": tau_denom_now,
                **_rollback_status(),
            }
            return loss_history, optimizer, stop_info
        if save_iterate_every > 0 and not timing_mode and outer_iter % save_iterate_every == 0:
            torch.save(value_net.state_dict(), os.path.join(
                iterate_dir, f"value_net_iter{outer_iter:04d}.pt"))
        if target_reached or stop_triggered:
            # The training-loss best is diagnostic only. Keep it in CPU
            # memory throughout training and write it once when the run ends.
            if best_state is not None and not timing_mode:
                torch.save(best_state, best_model_path)
            torch.save(value_net.state_dict(), last_model_path)
            torch.save(value_net.state_dict(), final_model_path)
            stop_info = {
                "stopped_early": bool(stop_triggered),
                "target_reached": bool(target_reached),
                "achieved_pres": None if last_val is None else float(last_val[2]),
                "q_res_role": "pres_target_and_official_residual",
                "q_res_seed": int(val_seed),
                "q_sel_role": (
                    "lr_scheduler_and_qsel_rollback"
                    if qsel_rollback_active else "lr_scheduler_only"),
                "q_sel_seed": int(sel_seed),
                "q_sel_active": bool(scheduler is not None),
                "heldout_score_formula": HELDOUT_SCORE_FORMULA,
                "scheduler_metric": (
                    "fixed_q_sel_pres" if scheduler is not None else "none_fixed_lr"),
                "scheduler_patience_unit": "q_sel_checks",
                "scheduler_patience_checks": int(scheduler_patience),
                "scheduler_check_every_optimizer_steps": int(val_every),
                "scheduler_checks": int(scheduler_checks),
                "scheduler_lr_reductions": int(scheduler_lr_reductions),
                "resample_every_optimizer_steps": int(resample_every),
                "training_batch_resamples_total": int(
                    training_batch_resamples_total),
                "training_batches_sampled_total": int(
                    training_batches_sampled_total),
                "sel_pde_rms_last": None if last_sel is None else float(last_sel[0]),
                "sel_terminal_rms_last": (
                    None if last_sel is None else float(last_sel[1])),
                "sel_pres_last": None if last_sel is None else float(last_sel[2]),
                "scheduler_score_last": (
                    None if last_sel is None else float(last_sel[2])),
                "sel_pres_best": (
                    None if not math.isfinite(best_sel_pres) else float(best_sel_pres)),
                "sel_check_epoch_last": (
                    None if last_sel is None else int(last_sel_epoch)),
                "outer_iter": outer_iter, "epoch_at_stop": total_optimizer_steps,
                "total_optimizer_steps": total_optimizer_steps,
                "hjb_guard_mode": hjb_guard_mode,
                "hjb_guard_version": HJB_GUARD_VERSION,
                "hjb_guard_tau_vy_final": tau_vy_now,
                "hjb_guard_tau_denom_final": tau_denom_now,
                **_rollback_status(),
            }
            reason = "p_res target" if target_reached else "divergence stopper"
            print(f"[stop] direct PINN ended at outer={outer_iter}, steps={total_optimizer_steps} ({reason})")
            return loss_history, optimizer, stop_info

    if best_state is not None and not timing_mode:
        torch.save(best_state, best_model_path)
    # Official model is always the final optimizer iterate; best is diagnostic.
    torch.save(value_net.state_dict(), last_model_path)
    torch.save(value_net.state_dict(), final_model_path)
    print(f"\nPINN finished. Official FINAL state -> {final_model_path}")
    if best_state is not None:
        print(f"  [diagnostic] best training loss={best_loss:.3e} at step {best_iter} -> {best_model_path}")
    stop_info.update({
        "outer_iters_completed": completed_outers,
        "total_optimizer_steps": total_optimizer_steps,
        "achieved_pres": None if last_val is None else float(last_val[2]),
        "q_res_role": "pres_target_and_official_residual",
        "q_res_seed": int(val_seed),
        "q_sel_role": (
            "lr_scheduler_and_qsel_rollback"
            if qsel_rollback_active else "lr_scheduler_only"),
        "q_sel_seed": int(sel_seed),
        "q_sel_active": bool(scheduler is not None),
        "heldout_score_formula": HELDOUT_SCORE_FORMULA,
        "scheduler_metric": (
            "fixed_q_sel_pres" if scheduler is not None else "none_fixed_lr"),
        "scheduler_patience_unit": "q_sel_checks",
        "scheduler_patience_checks": int(scheduler_patience),
        "scheduler_check_every_optimizer_steps": int(val_every),
        "scheduler_checks": int(scheduler_checks),
        "scheduler_lr_reductions": int(scheduler_lr_reductions),
        "resample_every_optimizer_steps": int(resample_every),
        "training_batch_resamples_total": int(training_batch_resamples_total),
        "training_batches_sampled_total": int(training_batches_sampled_total),
        "sel_pde_rms_last": None if last_sel is None else float(last_sel[0]),
        "sel_terminal_rms_last": None if last_sel is None else float(last_sel[1]),
        "sel_pres_last": None if last_sel is None else float(last_sel[2]),
        "scheduler_score_last": None if last_sel is None else float(last_sel[2]),
        "sel_pres_best": (
            None if not math.isfinite(best_sel_pres) else float(best_sel_pres)),
        "sel_check_epoch_last": None if last_sel is None else int(last_sel_epoch),
        "hjb_guard_mode": hjb_guard_mode,
        "hjb_guard_version": HJB_GUARD_VERSION,
        "hjb_guard_tau_vy_final": tau_vy_now,
        "hjb_guard_tau_denom_final": tau_denom_now,
        **_rollback_status(),
    })
    return loss_history, optimizer, stop_info


# =============================================================================
# 10) Evaluation helpers
# =============================================================================
@torch.no_grad()
def _no_grad_dummy():
    # placeholder to satisfy static analyzers; actual evaluation uses autograd for derivatives.
    return


def eval_pinn_on_grid(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
    chunk: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate PINN on a (t,W) grid, using y=log W as input.
    Returns:
      tt, ww, V_pinn, c_pinn, pi_pinn (Nt,Nw,N), pi_norm (Nt,Nw)
    """
    was_training = value_net.training
    value_net.eval()

    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.linspace(w_min, w_max, Nw)
    tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")

    t_flat = torch.tensor(tt.reshape(-1, 1), device=device, dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(np.log(ww.reshape(-1, 1)), device=device, dtype=torch.float32, requires_grad=True)

    n = t_flat.shape[0]
    V_out = []
    c_out = []
    pi_out = []
    pi_norm_out = []

    for i in range(0, n, chunk):
        t_b = t_flat[i:i+chunk]
        y_b = y_flat[i:i+chunk]

        # need autograd for derivatives
        V_b, c_b, pi_b, _eta_b = compute_policies_from_log_multi(value_net, t_b, y_b, create_graph=False)
        V_out.append(V_b.detach().cpu())
        c_out.append(c_b.detach().cpu())
        pi_out.append(pi_b.detach().cpu())
        pi_norm_out.append(torch.linalg.vector_norm(pi_b, ord=2, dim=1, keepdim=True).detach().cpu())

        if t_b.grad is not None:
            t_b.grad = None
        if y_b.grad is not None:
            y_b.grad = None

    V_pinn = torch.cat(V_out, dim=0).numpy().reshape(Nt, Nw)
    c_pinn = torch.cat(c_out, dim=0).numpy().reshape(Nt, Nw)
    pi_pinn = torch.cat(pi_out, dim=0).numpy().reshape(Nt, Nw, N_ASSETS)
    pi_norm = torch.cat(pi_norm_out, dim=0).numpy().reshape(Nt, Nw)

    if was_training:
        value_net.train()
    return tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm


def eval_pinn_on_grid_margin(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
    margin: float = 0.0,
    eval_w_min: Optional[float] = None,
    chunk: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid evaluation on an eval window shrunk by HALF-WIDTH `margin`.

    Only y=log W is shrunk. Time keeps the full [0,T) range, matching the
    paper convention Q_ev=(0,T)xOmega_ev. margin=0 reproduces the full grid.
    The W grid is exp of the shrunk y-range so points stay inside the trained
    log-wealth box.
    """
    was_training = value_net.training
    value_net.eval()
    window = mxu.resolve_eval_window(
        y_min, y_max, margin, eval_w_min=eval_w_min)
    y_lo, y_hi = window["ev_y_min"], window["ev_y_max"]
    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))
    tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")

    t_flat = torch.tensor(tt.reshape(-1, 1), device=device, dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(np.log(ww.reshape(-1, 1)), device=device, dtype=torch.float32, requires_grad=True)

    n = t_flat.shape[0]
    V_out, c_out, pi_out, pi_norm_out = [], [], [], []
    for i in range(0, n, chunk):
        t_b = t_flat[i:i+chunk]
        y_b = y_flat[i:i+chunk]
        V_b, c_b, pi_b, _eta_b = compute_policies_from_log_multi(value_net, t_b, y_b, create_graph=False)
        V_out.append(V_b.detach().cpu())
        c_out.append(c_b.detach().cpu())
        pi_out.append(pi_b.detach().cpu())
        pi_norm_out.append(torch.linalg.vector_norm(pi_b, ord=2, dim=1, keepdim=True).detach().cpu())
        if t_b.grad is not None:
            t_b.grad = None
        if y_b.grad is not None:
            y_b.grad = None

    V_pinn = torch.cat(V_out, dim=0).numpy().reshape(Nt, Nw)
    c_pinn = torch.cat(c_out, dim=0).numpy().reshape(Nt, Nw)
    pi_pinn = torch.cat(pi_out, dim=0).numpy().reshape(Nt, Nw, N_ASSETS)
    pi_norm = torch.cat(pi_norm_out, dim=0).numpy().reshape(Nt, Nw)
    if was_training:
        value_net.train()
    return tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm


def compute_metrics(
    V_pinn: np.ndarray, c_pinn: np.ndarray, pi_pinn: np.ndarray,
    V_cf: np.ndarray, c_cf: np.ndarray, pi_cf: np.ndarray,
    *,
    Vw_pinn: np.ndarray, Vww_pinn: np.ndarray,
    Vw_cf: np.ndarray, Vww_cf: np.ndarray,
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
    value_net: nn.Module,
    t_np: np.ndarray,
    w_np: np.ndarray,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``V,V_w,V_ww,c,pi`` on arbitrary paired points."""
    was_training = value_net.training
    value_net.eval()
    t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    w_np = np.asarray(w_np, dtype=np.float32).reshape(-1, 1)
    V_parts, Vw_parts, Vww_parts, c_parts, pi_parts = [], [], [], [], []
    for start in range(0, len(t_np), int(chunk)):
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
    value_net: nn.Module,
    t_np: np.ndarray,
    w_np: np.ndarray,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V,c,pi on arbitrary paired (t,W) points."""
    V, _Vw, _Vww, c, pi = eval_model_bundle_on_points(
        value_net, t_np, w_np, chunk=chunk)
    return V, c, pi


def eval_metrics_on_points(
    value_net: nn.Module,
    t_np: np.ndarray,
    w_np: np.ndarray,
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


def plot_loss_history(loss_history: List[Dict[str, float]], save_path: Optional[str] = None, show: bool = True):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.semilogy(epochs, [h["total"] for h in loss_history], label="Total", alpha=0.9)
    ax.semilogy(epochs, [h["pde"] for h in loss_history], label="PDE", alpha=0.7)
    ax.semilogy(epochs, [h["terminal"] for h in loss_history], label="Terminal", alpha=0.7)
    ax.semilogy(epochs, [h["mono"] for h in loss_history], label="Mono", alpha=0.6)
    ax.semilogy(epochs, [h["conc"] for h in loss_history], label="Conc", alpha=0.6)
    ax.semilogy(epochs, [h["eta"] for h in loss_history], label="Eta", alpha=0.6)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History (Reduced-form PINN, logW)")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
    save_path: Optional[str] = None,
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
    heat(axes[0, 0], V_pinn, "V (PINN)", vmin=vmin, vmax=vmax)
    heat(axes[0, 1], V_cf, "V (closed-form)", vmin=vmin, vmax=vmax)
    heat(axes[0, 2], V_err, "V error", div=True)

    # Consumption row
    vmin, vmax = min(c_pinn.min(), c_cf.min()), max(c_pinn.max(), c_cf.max())
    heat(axes[1, 0], c_pinn, "c (PINN via FOC)", vmin=vmin, vmax=vmax)
    heat(axes[1, 1], c_cf, "c (closed-form)", vmin=vmin, vmax=vmax)
    heat(axes[1, 2], c_err, "c error", div=True)

    # Portfolio norm row (compact)
    vmin, vmax = min(pi_norm_pinn.min(), pi_norm_cf_grid.min()), max(pi_norm_pinn.max(), pi_norm_cf_grid.max())
    heat(axes[2, 0], pi_norm_pinn, r"||pi||_2 (PINN via FOC)", vmin=vmin, vmax=vmax)
    heat(axes[2, 1], pi_norm_cf_grid, r"||pi*||_2 (closed)", vmin=vmin, vmax=vmax)
    heat(axes[2, 2], pi_err, r"||pi|| error", div=True)

    plt.suptitle("Multi-Asset Merton (with Consumption) - Reduced-form PINN(logW) vs Closed-form")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# 11) Main
# =============================================================================
def main():
    # Timing runs must never execute final plotting: otherwise the measured
    # evaluation peak can include Matplotlib/image allocations.  Persist the
    # effective flag so aggregate_compute can audit this invariant directly.
    ARGS.skip_figures_requested = bool(ARGS.skip_figures)
    ARGS.skip_figures = bool(
        ARGS.skip_figures or ARGS.skip_plots or ARGS.timing_mode)
    # Persist an explicit separation between the training HJB continuation,
    # the legacy eta auxiliary denominator, and evaluation-only FOC clamps.
    # This prevents downstream audits from mistaking a direct-PINN change for
    # a change to the PI-PINN/exact policy map.
    ARGS.hjb_guard_version = HJB_GUARD_VERSION
    ARGS.hjb_guard_formula = (
        "eps+tau*softplus((x-eps)/tau)" if ARGS.hjb_guard_mode == "softplus"
        else ("abs(x)+eps" if ARGS.hjb_guard_mode == "abs" else "clamp_min(x,eps)"))
    ARGS.hjb_guard_scope = "direct-nonlinear-hjb-only"
    ARGS.eta_aux_guard_mode = "legacy-abs"
    ARGS.evaluation_policy_guard_mode = "one-sided-hard-clamp"
    # Config-facing held-out contract. Q_res and Q_sel deliberately reuse the
    # requested set sizes, but never their samples or algorithmic roles.
    ARGS.q_res_role = "pres_target_and_official_residual"
    ARGS.q_res_seed = int(MARKET_SEED)
    ARGS.q_res_points = int(ARGS.val_points)
    ARGS.q_res_terminal_points = int(ARGS.val_terminal_points)
    ARGS.qsel_rollback_active = bool(
        not ARGS.eval_only
        and ARGS.lr_schedule == "plateau"
        and int(ARGS.qsel_rollback_max_rescues) > 0)
    ARGS.q_sel_role = (
        "lr_scheduler_and_qsel_rollback"
        if ARGS.qsel_rollback_active else "lr_scheduler_only")
    ARGS.q_sel_seed_offset = int(DIRECT_PINN_QSEL_SEED_OFFSET)
    ARGS.q_sel_seed = int(MARKET_SEED) + int(DIRECT_PINN_QSEL_SEED_OFFSET)
    ARGS.q_sel_active = bool(not ARGS.eval_only and ARGS.lr_schedule == "plateau")
    ARGS.q_sel_points = int(ARGS.val_points)
    ARGS.q_sel_terminal_points = int(ARGS.val_terminal_points)
    ARGS.q_res_q_sel_independent_streams = True
    ARGS.scheduler_metric = (
        "fixed_q_sel_pres" if ARGS.lr_schedule == "plateau" else "none_fixed_lr")
    ARGS.scheduler_score_formula = HELDOUT_SCORE_FORMULA
    ARGS.scheduler_patience_unit = "q_sel_checks"
    ARGS.scheduler_patience_checks = int(ARGS.scheduler_patience)
    ARGS.scheduler_check_every_optimizer_steps = int(ARGS.val_every)
    ARGS.resample_every_semantics = (
        "inner_optimizer_steps; zero=fixed_within_outer; "
        "refresh_before_step_K_plus_1"
    )
    ARGS.training_batch_refresh_at_outer_start = True
    out_dir = os.path.join(ARGS.output_root, ARGS.run_tag)
    weight_dir = ARGS.weight_root or os.path.join(out_dir, "weights")
    recorder = mxu.ExperimentRecorder(out_dir, weight_dir, ARGS)
    skip_figures = bool(ARGS.skip_figures)
    loaded_weight_path = None

    if ARGS.eval_only:
        recorder.save_config_eval(extra=POLICY_METADATA)
    else:
        recorder.save_config(extra=POLICY_METADATA)
        # A NEW training run starts with FRESH per-run CSVs (appending onto a
        # previous same-tag run would interleave two experiments).
        recorder.rotate_training_logs()
        # Persist the exact market draw for reproducibility / eval-only reloads.
        recorder.save_market_snapshot(
            mu_excess=mu_excess_np, Sigma_safe=Sigma_np, chol=chol_Sigma_np,
            pi_star=pi_star_np, Theta=np.array([Theta]), nu=np.array([nu]),
            gamma=np.array([gamma_risk]), r=np.array([r_rate]),
            rho_discount=np.array([rho_discount]), epsilon=np.array([epsilon_bequest]),
            T=np.array([T_FINAL]), w_min=np.array([w_min]), w_max=np.array([w_max]),
            n_assets=np.array([N_ASSETS]), market_seed=np.array([MARKET_SEED]),
            seed=np.array([SEED]),
        )

    try:
        outer_iters = int(ARGS.outer_iters)
        eval_epochs = int(ARGS.eval_epochs)
        epochs = outer_iters * eval_epochs
        batch_size = int(ARGS.batch_size)
        lr = float(ARGS.lr)
        w_terminal = float(ARGS.w_terminal)
        w_shape = float(ARGS.w_shape)
        w_eta = float(ARGS.w_eta)
        eta_clip = None if str(ARGS.eta_clip).lower() == "none" else float(ARGS.eta_clip)
        if eta_clip is not None and eta_clip <= 0.0:
            raise ValueError("--eta-clip must be positive or none")
        print_every = int(ARGS.print_every)

        start = time.time()
        value_net = ValueNetLogW(hidden=int(ARGS.value_hidden), depth=int(ARGS.value_depth)).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        stopper = None
        if not ARGS.eval_only and ARGS.pde_stop_threshold is not None:
            stopper = mxu.PDEEarlyStopper(
                threshold=float(ARGS.pde_stop_threshold),
                start_outer=int(ARGS.pde_stop_start_outer),
                patience=int(ARGS.pde_stop_patience),
                stop_flag_path=str(ARGS.stop_flag_path or ""),
                recorder=recorder, run_tag=ARGS.run_tag, model_type=ARGS.model_type)
            if stopper.shared_stop_exists():
                info = stopper.mark_from_existing_flag(outer_iter=0, pde_loss=None)
                print(f"[early-stop] shared stop flag exists; skipping run: {info}")
                return

        train_gpu_peak = None
        if not ARGS.eval_only:
            # Do this only after the shared-stop early return above. A run
            # skipped by an existing stop flag must not archive otherwise
            # valid official checkpoints from an earlier attempt.
            recorder.rotate_training_checkpoints()
            recorder.write_status(
                "running",
                q_res_role=ARGS.q_res_role, q_res_seed=ARGS.q_res_seed,
                q_sel_role=ARGS.q_sel_role, q_sel_seed=ARGS.q_sel_seed,
                q_sel_active=ARGS.q_sel_active,
                heldout_score_formula=HELDOUT_SCORE_FORMULA,
                scheduler_metric=ARGS.scheduler_metric,
                scheduler_patience_unit=ARGS.scheduler_patience_unit,
                scheduler_patience_checks=int(ARGS.scheduler_patience),
                scheduler_check_every_optimizer_steps=int(ARGS.val_every),
                scheduler_checks=0,
                qsel_rollback_active=ARGS.qsel_rollback_active,
                qsel_rollback_factor=float(ARGS.qsel_rollback_factor),
                qsel_rollback_lr_factor=float(
                    ARGS.qsel_rollback_lr_factor),
                qsel_rollback_max_rescues=int(
                    ARGS.qsel_rollback_max_rescues),
                qsel_rollback_triggers=0,
                qsel_rollback_rescues_used=0,
                resample_every_optimizer_steps=int(ARGS.resample_every),
                training_batch_resamples_total=0,
                training_batches_sampled_total=0,
            )
            loss_history, _opt, stop_info = train_pinn_hybrid_reduced_logw_multi(
                value_net=value_net,
                epochs=epochs,
                batch_size=batch_size,
                terminal_frac=float(ARGS.terminal_frac),
                lr=lr,
                eval_epochs=eval_epochs,
                outer_iters=outer_iters,
                resample_every=int(ARGS.resample_every),
                w_terminal=w_terminal,
                w_shape=w_shape,
                w_eta=w_eta,
                eta_clip=eta_clip,
                scheduler_patience=int(ARGS.scheduler_patience),
                scheduler_factor=float(ARGS.scheduler_factor),
                scheduler_min_lr=float(ARGS.scheduler_min_lr),
                lr_schedule=str(ARGS.lr_schedule),
                qsel_rollback_factor=float(ARGS.qsel_rollback_factor),
                qsel_rollback_lr_factor=float(
                    ARGS.qsel_rollback_lr_factor),
                qsel_rollback_max_rescues=int(
                    ARGS.qsel_rollback_max_rescues),
                hjb_guard_mode=str(ARGS.hjb_guard_mode),
                hjb_vy_guard_eps=float(ARGS.hjb_vy_guard_eps),
                hjb_denom_guard_eps=float(ARGS.hjb_denom_guard_eps),
                hjb_vy_guard_tau=float(ARGS.hjb_vy_guard_tau),
                hjb_denom_guard_tau=float(ARGS.hjb_denom_guard_tau),
                hjb_vy_guard_tau_min=float(ARGS.hjb_vy_guard_tau_min),
                hjb_denom_guard_tau_min=float(ARGS.hjb_denom_guard_tau_min),
                hjb_guard_anneal_every=int(ARGS.hjb_guard_anneal_every),
                hjb_guard_anneal_factor=float(ARGS.hjb_guard_anneal_factor),
                save_iterate_every=int(ARGS.save_iterate_every),
                pres_target=ARGS.pres_target,
                val_points=int(ARGS.val_points),
                val_terminal_points=int(ARGS.val_terminal_points),
                val_every=int(ARGS.val_every),
                val_seed=int(MARKET_SEED),
                diag_points=int(ARGS.diag_points),
                diag_margin=mxu.parse_eval_margins(ARGS.eval_margin)[0],
                diag_every=int(ARGS.diag_every),
                timing_mode=bool(ARGS.timing_mode),
                print_every=print_every,
                weight_dir=weight_dir,
                recorder=recorder,
                stopper=stopper,
            )
            elapsed = time.time() - start
            if device.type == "cuda":
                train_gpu_peak = int(torch.cuda.max_memory_allocated(device))
                # Evaluation receives an independent peak-memory window.
                torch.cuda.reset_peak_memory_stats(device)
            h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
            print(f"\nElapsed time: {h:02d}:{m:02d}:{s:05.2f}")
            if bool(stop_info.get("run_failed", False)):
                raise RuntimeError(str(
                    stop_info.get(
                        "failure_reason", "qsel rollback failed")))
            if bool(stop_info.get("stopped_early", False)):
                recorder.write_status(
                    "stopped_early", elapsed_sec=elapsed,
                    final_weight_path=os.path.join(weight_dir, "value_net_final.pt"),
                    train_gpu_peak_mem_bytes=train_gpu_peak,
                    policy_bounds_mode=policy_bounds_mode,
                    policy_pi_min=POLICY_METADATA["policy_pi_min"],
                    policy_pi_max=POLICY_METADATA["policy_pi_max"],
                    policy_kappa_min=kappa_min_bound,
                    policy_kappa_max=kappa_max_bound,
                    policy_c_min=c_min_bound, policy_c_max=c_max_bound,
                    **stop_info)
                return
        else:
            recorder.write_status_eval("running")
            elapsed = 0.0
            loss_history = []
            stop_info = {"target_reached": False, "total_optimizer_steps": 0}
            candidates = [
                os.path.join(weight_dir, "value_net_final.pt"),
                os.path.join(weight_dir, "value_net_last.pt"),
            ]
            if ARGS.allow_legacy_best_eval:
                candidates.extend([
                    os.path.join(weight_dir, "value_net_best_diag.pt"),
                    # Legacy fallback for runs produced before diagnostic-best
                    # checkpoints were renamed.
                    os.path.join(weight_dir, "value_net_best.pt"),
                ])
            load_path = next((path for path in candidates if os.path.exists(path)), None)
            if load_path is None:
                hint = " (use --allow-legacy-best-eval for diagnostic/legacy fallback)"
                raise FileNotFoundError(f"no official final/last checkpoint under {weight_dir}{hint}")
            value_net.load_state_dict(torch.load(load_path, map_location=device))
            loaded_weight_path = load_path
            print(f"[eval-only] loaded checkpoint: {load_path}")
            if device.type == "cuda":
                # Exclude construction/checkpoint loading from eval peak.
                torch.cuda.reset_peak_memory_stats(device)

        Nt, Nw = int(ARGS.n_tau), int(ARGS.n_x)
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
        final_weight_path = os.path.join(weight_dir, "value_net_final.pt")
        best_weight_path = os.path.join(weight_dir, "value_net_best_diag.pt")

        if ARGS.skip_eval:
            if ARGS.eval_only:
                recorder.mark_success_eval(
                    elapsed_sec=elapsed, final_weight_path=final_weight_path,
                    loaded_weight_path=loaded_weight_path, skipped_eval=True)
            else:
                recorder.mark_success(
                    elapsed_sec=elapsed, final_weight_path=final_weight_path,
                    best_weight_path=best_weight_path, skipped_eval=True,
                    target_reached=bool(stop_info.get("target_reached", False)),
                    achieved_pres=stop_info.get("achieved_pres"),
                    total_optimizer_steps=stop_info.get("total_optimizer_steps"),
                    hjb_guard_mode=stop_info.get("hjb_guard_mode"),
                    hjb_guard_version=stop_info.get("hjb_guard_version"),
                    hjb_guard_tau_vy_final=stop_info.get("hjb_guard_tau_vy_final"),
                    hjb_guard_tau_denom_final=stop_info.get("hjb_guard_tau_denom_final"),
                    policy_bounds_mode=policy_bounds_mode,
                    policy_pi_min=POLICY_METADATA["policy_pi_min"],
                    policy_pi_max=POLICY_METADATA["policy_pi_max"],
                    policy_kappa_min=kappa_min_bound,
                    policy_kappa_max=kappa_max_bound,
                    policy_c_min=c_min_bound, policy_c_max=c_max_bound,
                    train_gpu_peak_mem_bytes=train_gpu_peak,
                    timing_mode=bool(ARGS.timing_mode),
                    **{key: stop_info.get(key) for key in (
                        SCHEDULER_STATUS_FIELDS
                        + QSEL_ROLLBACK_STATUS_FIELDS
                        + RESAMPLING_STATUS_FIELDS)})
            return

        # Eval-only uses a temporary metrics file. A failed reevaluation never
        # destroys the training-run table.
        metrics_final_path = recorder.metrics_csv
        if ARGS.eval_only:
            recorder.metrics_csv = metrics_final_path + ".eval_tmp"
            if os.path.exists(recorder.metrics_csv):
                os.remove(recorder.metrics_csv)

        if int(ARGS.test_points) > 0:
            metrics_by_margin = eval_fulldim_test_metrics(
                value_net,
                int(ARGS.test_points),
                margins,
                eval_w_min=ARGS.eval_w_min,
            )
        else:
            # Deterministic 2D-grid fallback (also useful for a cheap smoke).
            metrics_by_margin: Dict[float, Dict[str, float]] = {}
            for margin in margins:
                window = eval_windows[float(margin)]
                y_lo, y_hi = window["ev_y_min"], window["ev_y_max"]
                t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
                w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))
                tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")
                metrics_by_margin[float(margin)] = eval_metrics_on_points(
                    value_net, tt, ww)

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
            plot_loss_history(loss_history, save_path=os.path.join(out_dir, "plots", "loss_history.png"), show=False)
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                value_net,
                Nt=Nt,
                Nw=Nw,
                margin=margins[0],
                eval_w_min=ARGS.eval_w_min,
                chunk=4000,
            )
            plot_comparison_heatmaps(
                tt, ww, V_pinn, c_pinn, pi_norm,
                save_path=os.path.join(out_dir, "plots", "comparison_heatmap.png"), show=False)

        eval_gpu_peak = None
        if device.type == "cuda":
            eval_gpu_peak = int(torch.cuda.max_memory_allocated(device))
        if ARGS.eval_only:
            if os.path.exists(recorder.metrics_csv):
                backup = metrics_final_path + ".bak_train"
                if os.path.exists(metrics_final_path) and not os.path.exists(backup):
                    import shutil
                    shutil.copyfile(metrics_final_path, backup)
                os.replace(recorder.metrics_csv, metrics_final_path)
                recorder.metrics_csv = metrics_final_path
            primary_window = eval_windows[float(margins[0])]
            recorder.mark_success_eval(
                elapsed_sec=elapsed, primary_margin=margins[0],
                final_weight_path=final_weight_path,
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
            primary_window = eval_windows[float(margins[0])]
            recorder.mark_success(
                elapsed_sec=elapsed, epochs=epochs,
                total_optimizer_steps=stop_info.get("total_optimizer_steps"),
                train_wall_sec=elapsed, primary_margin=margins[0],
                final_weight_path=final_weight_path, best_weight_path=best_weight_path,
                target_reached=bool(stop_info.get("target_reached", False)),
                achieved_pres=stop_info.get("achieved_pres"), pi_clip_abs=pi_clip_abs,
                policy_bounds_mode=policy_bounds_mode,
                policy_pi_min=POLICY_METADATA["policy_pi_min"],
                policy_pi_max=POLICY_METADATA["policy_pi_max"],
                policy_kappa_min=kappa_min_bound,
                policy_kappa_max=kappa_max_bound,
                policy_c_min=c_min_bound, policy_c_max=c_max_bound,
                hjb_guard_mode=stop_info.get("hjb_guard_mode"),
                hjb_guard_version=stop_info.get("hjb_guard_version"),
                hjb_guard_tau_vy_final=stop_info.get("hjb_guard_tau_vy_final"),
                hjb_guard_tau_denom_final=stop_info.get("hjb_guard_tau_denom_final"),
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
                **{key: stop_info.get(key) for key in (
                    SCHEDULER_STATUS_FIELDS
                    + QSEL_ROLLBACK_STATUS_FIELDS
                    + RESAMPLING_STATUS_FIELDS)})
        print("\nDone.")
    except Exception as exc:
        if ARGS.eval_only:
            recorder.mark_failed_eval(reason=repr(exc))
        else:
            failure_meta = {}
            if isinstance(locals().get("stop_info"), dict):
                failure_meta = {
                    key: stop_info.get(key)
                    for key in (
                        SCHEDULER_STATUS_FIELDS
                        + QSEL_ROLLBACK_STATUS_FIELDS
                        + RESAMPLING_STATUS_FIELDS
                        + ("total_optimizer_steps", "failure_reason",
                           "failed_admissible_weight_path"))
                    if key in stop_info
                }
            recorder.mark_failed(reason=repr(exc), **failure_meta)
        raise


if __name__ == "__main__":
    main()
