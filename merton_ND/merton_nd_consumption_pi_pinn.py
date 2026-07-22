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
    # unconstrained.
    p.add_argument("--pi-clip-abs", type=mxu.none_or_float, default=2.0)
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
    # Checkpoints / diagnostics.
    p.add_argument("--save-iterate-every", type=int, default=0)
    p.add_argument("--e3b-checkpoints", action="store_true")
    p.add_argument("--diag-points", type=int, default=4096)
    p.add_argument("--diag-every", type=int, default=1)
    # Evaluation window(s): only the log-wealth axis is shrunk.  Time keeps
    # the full [0,T) range, matching Q_ev=(0,T)xOmega_ev.
    p.add_argument("--eval-margin", type=str, default="0.10,0.0,0.05,0.15,0.20")
    p.add_argument("--test-points", type=int, default=100000)
    p.add_argument("--n-tau", type=int, default=100)
    p.add_argument("--n-x", type=int, default=100)
    # Logging / output
    p.add_argument("--print-every-outer", type=int, default=10)
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
if ARGS.eval_epochs < 0 or ARGS.outer_iters < 1 or ARGS.batch_size < 1:
    raise ValueError("require eval_epochs >= 0, outer_iters >= 1, and batch_size >= 1")

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

# Symmetric componentwise portfolio safety bound.  None means that the raw
# FOC control is returned without a projection.
pi_clip_abs = ARGS.pi_clip_abs
pi_min_bound = -float(pi_clip_abs) if pi_clip_abs is not None else None
pi_max_bound = float(pi_clip_abs) if pi_clip_abs is not None else None

# Consumption bounds: use kappa=c/W bounds to avoid CRRA blow-up at tiny c
M_utility_cap = float(ARGS.utility_cap)
c_floor = ((gamma_risk - 1.0) * M_utility_cap) ** (-1.0 / (gamma_risk - 1.0))
kappa_min_bound = c_floor / x_min
kappa_max_bound = float(ARGS.kappa_max_bound)

# Optional level clamp for c (mainly for printing / extra safety)
c_min_bound = c_floor
c_max_bound = x_max

# Torch constants
mu_excess = torch.tensor(mu_excess_np, device=device, dtype=torch.float32)          # (N,)
Sigma = torch.tensor(Sigma_np, device=device, dtype=torch.float32)                 # (N,N)
Sigma_inv_mu = torch.tensor(Sigma_inv_mu_np, device=device, dtype=torch.float32)   # (N,)
pi_star = torch.tensor(pi_star_np, device=device, dtype=torch.float32)             # (N,)

print(f"\n{'='*70}")
print("Multi-Asset Merton (with Consumption) - PI-PINN (FOC) [log W]")
print(f"{'='*70}")
print(f"  N_ASSETS={N_ASSETS}")
print(f"  gamma={gamma_risk}, rho={rho_discount}, r={r_rate}, epsilon={epsilon}")
print(f"  T={T_FINAL}, W∈[{x_min},{x_max}] -> y∈[{y_min:.3f},{y_max:.3f}]")
print(f"  Theta = mu^T Sigma^{-1} mu = {Theta:.6f}")
print(f"  nu = {nu:.6f}")
print(f"  pi clip abs: {pi_clip_abs}"
      + (f" (componentwise [{pi_min_bound},{pi_max_bound}])" if pi_clip_abs is not None else " (unconstrained)"))
print(f"  kappa=c/W bounds: [{kappa_min_bound:.4g},{kappa_max_bound}]")
print(f"  ||pi*||_2 = {np.linalg.norm(pi_star_np):.4f}, max|pi*_i|={np.max(np.abs(pi_star_np)):.4f}")
print(f"  cond(Sigma_safe) = {market_params['cond_Sigma_safe']:.2f}, max|rho_ij|={market_params['max_abs_rho']:.3f}")
print(f"{'='*70}\n")


# =============================================================================
# 2) Closed-form (for sanity check)
# =============================================================================
def closed_form_c(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Same functional form as 1D, with multi-asset nu."""
    tau = T_FINAL - t
    exp_term = np.exp(-nu * tau)
    denom = 1.0 + (nu * epsilon - 1.0) * exp_term
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return (nu / denom) * W


def closed_form_V(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """CRRA value function V(t,W)=A(t) W^{1-gamma}/(1-gamma)."""
    tau = T_FINAL - t
    exp_term = np.exp(-nu * tau)
    denom = 1.0 + (nu * epsilon - 1.0) * exp_term
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    A_t = (denom / nu) ** gamma_risk
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
    c_safe = torch.clamp(c, min=1e-8)
    return c_safe.pow(1.0 - gamma_risk) / (1.0 - gamma_risk)


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
    kappa_min: float = kappa_min_bound,
    kappa_max: float = kappa_max_bound,
    c_min: float = c_min_bound,
    c_max: float = c_max_bound,
) -> torch.Tensor:
    """c* from U'(c)=V_W, with V_W = V_y/W and W=exp(y)."""
    W = torch.exp(y)
    V_w = V_y / W
    V_w_safe = torch.clamp(V_w, min=eps)
    c_raw = V_w_safe.pow(-1.0 / gamma_risk)

    kappa_raw = c_raw / W
    kappa = torch.clamp(kappa_raw, min=kappa_min, max=kappa_max)
    c_new = kappa * W
    return torch.clamp(c_new, min=c_min, max=c_max)


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
    """Fixed dense tensor grid on Q_ev; shrink log wealth, never time."""
    n_points = max(4, int(n_points))
    n_t = max(2, int(round(math.sqrt(n_points))))
    n_y = max(2, int(math.ceil(n_points / n_t)))
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, float(margin))
    t_axis = np.linspace(t_min, t_max - 1e-3, n_t, dtype=np.float64)
    y_axis = np.linspace(y_lo, y_hi, n_y, dtype=np.float64)
    tt, yy = np.meshgrid(t_axis, y_axis, indexing="ij")
    t = torch.tensor(tt.reshape(-1, 1)[:n_points], device=device, dtype=torch.float32)
    y = torch.tensor(yy.reshape(-1, 1)[:n_points], device=device, dtype=torch.float32)
    return {"t": t, "y": y, "margin": float(margin)}


def closed_form_wealth_bundle(t_np: np.ndarray, w_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form (V,V_w,V_ww), used by the paper X_ev diagnostic."""
    tau = T_FINAL - t_np
    exp_term = np.exp(-nu * tau)
    denom = 1.0 + (nu * epsilon - 1.0) * exp_term
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    A_t = (denom / nu) ** gamma_risk
    V = A_t * w_np ** (1.0 - gamma_risk) / (1.0 - gamma_risk)
    V_w = A_t * w_np ** (-gamma_risk)
    V_ww = -gamma_risk * A_t * w_np ** (-gamma_risk - 1.0)
    return V, V_w, V_ww


def _relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    den = float(np.sum(np.asarray(ref, dtype=np.float64) ** 2))
    num = float(np.sum((np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)) ** 2))
    return float(np.sqrt(num / max(den, np.finfo(np.float64).tiny)))


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
        "clip_frac_kappa_low": float(
            (kappa_raw <= kappa_min_bound + 1e-7).float().mean().item()),
        "clip_frac_kappa_high": float(
            (kappa_raw >= kappa_max_bound - 1e-7).float().mean().item()),
        "clip_frac_c_level_low": float(
            (c_level_raw <= c_min_bound + 1e-7).float().mean().item()),
        "clip_frac_c_level_high": float(
            (c_level_raw >= c_max_bound - 1e-7).float().mean().item()),
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
    kappa = torch.clamp(kappa_raw, min=kappa_min_bound, max=kappa_max_bound)
    c_level_raw = kappa * W
    c_eval = torch.clamp(c_level_raw, min=c_min_bound, max=c_max_bound)

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
        c_min: float = c_min_bound,
        c_max: float = c_max_bound,
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

    def prepare_optimizer_for_outer(self) -> None:
        """Reset only the within-frozen-PDE scheduler; never compare PDEs."""
        self._outer_count += 1
        if self.lr_schedule == "carry_plateau":
            if self._outer_count == 1:
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
            c_raw = torch.full((n, 1), self.c_min, device=self.device)
        elif method == "proportional":
            c_raw = rho_discount * W
        elif method == "random":
            c_raw = self.c_min + torch.rand(n, 1, device=self.device) * (self.c_max - self.c_min)
        else:
            raise ValueError(f"Unknown c init method: {method}")
        kappa_raw = c_raw / W
        kappa = torch.clamp(kappa_raw, min=kappa_min_bound, max=kappa_max_bound)
        c_level_raw = kappa * W
        c = torch.clamp(c_level_raw, self.c_min, self.c_max)
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
        kappa = torch.clamp(kappa_raw, min=kappa_min_bound, max=kappa_max_bound)
        c_level_raw = kappa * W
        c = torch.clamp(c_level_raw, min=self.c_min, max=self.c_max)
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
        loss_history: List[Dict] = []
        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        c_fixed = c_n.detach()
        pi_fixed = pi_n.detach()

        target_reached = False
        epochs_used = 0
        last_val = None
        last_val_epoch = -1
        val_pres_at_stop = ""
        n_resamples = 0

        best_sel_pres = float("inf")
        best_sel_state = None
        best_sel_epoch = -1
        sel_no_improve = 0
        sel_checks = 0
        sel_stopped = False

        def run_val_check(epoch_idx: int):
            nonlocal last_val, last_val_epoch, target_reached, val_pres_at_stop
            value = val_fn()
            last_val, last_val_epoch = value, int(epoch_idx)
            if (not target_reached and pres_target is not None
                    and value[2] <= float(pres_target)):
                target_reached = True
                val_pres_at_stop = float(value[2])
            return value

        def run_sel_check(epoch_idx: int):
            nonlocal best_sel_pres, best_sel_state, best_sel_epoch
            nonlocal sel_no_improve, sel_checks, sel_stopped
            value = sel_fn()
            sel_checks += 1
            if self.lr_schedule == "carry_plateau" and self.scheduler is not None:
                self.scheduler.step(float(value[2]))
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
            if target_reached or sel_stopped:
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

        lr_end_before_restore = ""
        lr_best_checkpoint = ""
        lr_after_restore = ""
        if restore_best and best_sel_state is not None:
            end_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            self.value_net.load_state_dict(best_sel_state["model"])
            self.optimizer.load_state_dict(best_sel_state["optimizer"])
            floor_lr = self._effective_min_lr()
            for group, end_lr in zip(self.optimizer.param_groups, end_lrs):
                group["lr"] = max(floor_lr, min(float(group["lr"]), end_lr))
            lr_end_before_restore = end_lrs[0]
            lr_best_checkpoint = float(best_sel_state["lr"])
            lr_after_restore = float(self.optimizer.param_groups[0]["lr"])
            last_val_epoch = -1

        if val_fn is not None and last_val_epoch != epochs_used:
            # This is explicitly post-restore.  target_reached remains the
            # sticky training-stop fact requested by the paper protocol.
            last_val = val_fn()
            last_val_epoch = epochs_used

        info = {
            "epochs_used": int(epochs_used),
            "n_resamples": int(n_resamples),
            "target_reached": bool(target_reached),
            "val_pres_at_stop": val_pres_at_stop,
            "val_pde_rms_post_restore": last_val[0] if last_val else "",
            "val_terminal_rms_post_restore": last_val[1] if last_val else "",
            "val_pres_post_restore": last_val[2] if last_val else "",
            "sel_best_pres": best_sel_state["pres"] if best_sel_state else "",
            "sel_best_epoch": best_sel_state["epoch"] if best_sel_state else "",
            "sel_best_lr": best_sel_state["lr"] if best_sel_state else "",
            "sel_checks": int(sel_checks), "sel_stopped": int(bool(sel_stopped)),
            "sel_restored": int(bool(restore_best and best_sel_state is not None)),
            "lr_end_before_restore": lr_end_before_restore,
            "lr_best_checkpoint": lr_best_checkpoint,
            "lr_after_restore": lr_after_restore,
            "lr_carried_next": min(self.carry_lr_max, lr_after_restore)
                if self.lr_schedule == "carry_plateau" and lr_after_restore != "" else "",
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
        weight_dir: str = "weights",
        recorder=None,
        stopper=None,
    ) -> Dict:
        print(f"\n{'='*70}")
        print(f"PI-PINN Algorithm 2 (multi-asset, logW, with consumption): {outer_iters} outer iterations")
        print(f"  Eval epochs per iter: {eval_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  pi init: {pi_init_method} | c init: {c_init_method}")
        print(f"  Initial LR: {self.initial_lr:.2e}")
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
        if e3b_checkpoints or save_iterate_every > 0:
            os.makedirs(iterate_dir, exist_ok=True)

        val_set = None
        if val_points > 0 and not (timing_mode and pres_target is None):
            val_set = build_validation_set(val_points, max(1, val_terminal_points), self.device, val_seed)
        sel_set = None
        if inner_best_restore and sel_points > 0:
            sel_set = build_validation_set(
                sel_points, max(1, sel_terminal_points), self.device,
                int(val_seed) * 7919 + 101)
        diag = None
        diag_col = None
        if diag_points > 0 and not timing_mode:
            diag = build_diag_set(diag_points, diag_margin)
            diag_col = build_validation_set(
                diag_points, 1, self.device, int(val_seed) + 104729)

        # initial sample
        t_colloc, y_colloc = sample_interior(batch_size, self.device)
        t_term, y_term = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device)
        V_T_target = V_terminal_from_y(y_term).detach()

        c_n, pi_n, _ = self.initialize_policy(
            t_colloc, y_colloc, pi_init_method, pi_init_scale, c_init_method)
        print(f"Initial c: mean={c_n.mean().item():.4f}, std={c_n.std().item():.4f}")
        print(f"Initial pi: mean={pi_n.mean().item():.4f}, std={pi_n.std().item():.4f}")
        print(f"pi* stats: mean={pi_star.mean().item():.4f}, std={pi_star.std().item():.4f}, ||pi*||2={pi_star.norm().item():.4f}")

        for it in range(1, outer_iters + 1):
            if stopper is not None and stopper.shared_stop_exists():
                meta = stopper.mark_from_existing_flag(outer_iter=it, pde_loss=None)
                results["stopped_early"] = True
                results["stop_info"] = meta
                break
            verbose = (it % print_every_outer == 0) or (it <= 3)
            if verbose:
                print(f"\n[Outer Iteration {it}/{outer_iters}]")
                print("-" * 40)

            # fresh samples
            t_colloc, y_colloc = sample_interior(batch_size, self.device)
            t_term, y_term = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device)
            V_T_target = V_terminal_from_y(y_term).detach()

            # recompute policy on new points
            policy_source = None
            if it > 1:
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

            self.prepare_optimizer_for_outer()

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
            results["inner_epochs_used"].append(int(eval_info["epochs_used"]))
            achieved_pres = eval_info["val_pres_at_stop"]
            if not isinstance(achieved_pres, (int, float)):
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

            # policy improvement for metrics
            c_new, pi_new = self.policy_improvement(t_colloc, y_colloc)

            c_diff = ((c_new - c_n) ** 2).mean().item()
            pi_diff = ((pi_new - pi_n) ** 2).mean().item()
            results["c_diff"].append(c_diff)
            results["pi_diff"].append(pi_diff)

            # compare with closed form
            pi_star_rep = pi_star.view(1, -1).repeat(pi_new.shape[0], 1)
            pi_vs_cf = ((pi_new - pi_star_rep) ** 2).mean().item()
            results["pi_vs_closed_form"].append(pi_vs_cf)

            # closed form c
            t_np = t_colloc.detach().cpu().numpy()
            W_np = np.exp(y_colloc.detach().cpu().numpy())
            c_star_np = closed_form_c(t_np, W_np)
            c_star = torch.tensor(c_star_np, device=self.device, dtype=torch.float32)
            c_vs_cf = ((c_new - c_star) ** 2).mean().item()
            results["c_vs_closed_form"].append(c_vs_cf)

            results["pi_mean"].append(pi_new.mean().item())
            results["pi_std"].append(pi_new.std().item())
            results["c_mean"].append(c_new.mean().item())
            results["c_std"].append(c_new.std().item())

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
            frozen_var_min = frozen_var_max = frozen_clip = ""
            frozen_c_clip = {
                "clip_frac_kappa_low": "", "clip_frac_kappa_high": "",
                "clip_frac_c_level_low": "", "clip_frac_c_level_high": "",
            }
            if diag_col is not None:
                if policy_source is None:
                    _, pi_diag_frozen, frozen_diag_comp = self.initialize_policy(
                        diag_col["t_int"], diag_col["y_int"],
                        pi_init_method, pi_init_scale, c_init_method)
                else:
                    _, pi_diag_frozen, frozen_diag_comp = self._policy_components(
                        diag_col["t_int"], diag_col["y_int"], net=policy_source)
                variance = _diffusion_variance(pi_diag_frozen)
                frozen_var_min = float(variance.min().item())
                frozen_var_max = float(variance.max().item())
                frozen_clip = _clip_fraction_pi(frozen_diag_comp["pi_raw"])
                frozen_c_clip = _consumption_clip_fractions(frozen_diag_comp)

            save_this = ((e3b_checkpoints and (it <= 10 or it % 10 == 0))
                         or (not e3b_checkpoints and save_iterate_every > 0
                             and it % save_iterate_every == 0))
            if save_this and not timing_mode:
                torch.save(self.value_net.state_dict(),
                           os.path.join(iterate_dir, f"value_net_iter{it:04d}.pt"))

            last = eval_loss_hist[-1]
            outer_row = {
                "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                "run_tag": ARGS.run_tag, "outer_iter": it,
                "total_loss": last["total"], "pde_loss": last["pde"],
                "terminal_loss": last["terminal"], "monotonicity_loss": last["mono"],
                "concavity_loss": last["conc"], "eta_loss": last["eta"],
                "train_pres": last.get("train_pres", ""),
                "val_pde_rms": eval_info["val_pde_rms_post_restore"],
                "val_terminal_rms": eval_info["val_terminal_rms_post_restore"],
                "val_pres": eval_info["val_pres_post_restore"],
                "val_pres_at_stop": eval_info["val_pres_at_stop"],
                "val_pres_post_restore": eval_info["val_pres_post_restore"],
                "achieved_pres": achieved_pres,
                "inner_epochs_used": eval_info["epochs_used"],
                "n_resamples": eval_info["n_resamples"],
                "target_reached": int(eval_info["target_reached"]),
                "sel_best_pres": eval_info["sel_best_pres"],
                "sel_best_epoch": eval_info["sel_best_epoch"],
                "sel_best_lr": eval_info["sel_best_lr"],
                "sel_checks": eval_info["sel_checks"], "sel_stopped": eval_info["sel_stopped"],
                "sel_restored": eval_info["sel_restored"],
                "lr_end_before_restore": eval_info["lr_end_before_restore"],
                "lr_best_checkpoint": eval_info["lr_best_checkpoint"],
                "lr_after_restore": eval_info["lr_after_restore"],
                "lr_carried_next": eval_info["lr_carried_next"],
                "pi_diff": pi_diff, "c_diff": c_diff,
                "pi_vs_closed_form": pi_vs_cf, "c_vs_closed_form": c_vs_cf,
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
        if results["outer_rows"]:
            torch.save(self.value_net.state_dict(), final_model_path)
            torch.save(self.value_net.state_dict(), last_model_path)
        if best_diag_state is not None and not timing_mode:
            torch.save(best_diag_state, best_model_path)
            print(
                f"Saved diagnostic best held-out state: outer={best_iter}, "
                f"score={best_eval_loss:.3e} -> {best_model_path}")
        if e3b_checkpoints and results["outer_rows"]:
            final_it = int(results["outer_rows"][-1]["outer_iter"])
            torch.save(self.value_net.state_dict(),
                       os.path.join(iterate_dir, f"value_net_iter{final_it:04d}.pt"))
        if results["outer_rows"]:
            print(f"\nSaved official FINAL iterate: {final_model_path}")

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid evaluation on an eval window shrunk by HALF-WIDTH `margin`.

    Only y=log W is shrunk; time keeps the full [0,T) range.
    """
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, margin)
    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))
    tt, ww = np.meshgrid(t_vals, w_vals, indexing="ij")
    V, c, pi = eval_model_on_points(value_net, tt, ww)
    V_grid = V.reshape(Nt, Nw)
    c_grid = c.reshape(Nt, Nw)
    pi_grid = pi.reshape(Nt, Nw, N_ASSETS)
    pi_norm_grid = np.linalg.norm(pi_grid, axis=2)
    return tt, ww, V_grid, c_grid, pi_grid, pi_norm_grid


def compute_metrics(V_pinn, c_pinn, pi_pinn, V_cf, c_cf, pi_cf) -> Dict[str, float]:
    """Paper metrics: standard relative L2, with MSE/max diagnostics."""
    mse_V = np.mean((V_pinn - V_cf) ** 2)
    mse_c = np.mean((c_pinn - c_cf) ** 2)
    mse_pi = np.mean((pi_pinn - pi_cf) ** 2)

    rel_l2_V = _relative_l2(V_pinn, V_cf)
    rel_l2_c = _relative_l2(c_pinn, c_cf)
    rel_l2_pi = _relative_l2(pi_pinn, pi_cf)

    max_V_err = np.max(np.abs(V_pinn - V_cf))
    max_c_err = np.max(np.abs(c_pinn - c_cf))
    max_pi_err = np.max(np.abs(pi_pinn - pi_cf))

    return {
        "MSE_V": float(mse_V),
        "MSE_c": float(mse_c),
        "MSE_pi": float(mse_pi),
        "RelL2_V": float(rel_l2_V),
        "RelL2_c": float(rel_l2_c),
        "RelL2_pi": float(rel_l2_pi),
        "MaxErr_V": float(max_V_err),
        "MaxErr_c": float(max_c_err),
        "MaxErr_pi": float(max_pi_err),
    }


def eval_model_on_points(
    value_net: nn.Module, t_np: np.ndarray, w_np: np.ndarray, chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate V,c,pi on arbitrary paired points."""
    was_training = value_net.training
    value_net.eval()
    V_parts, c_parts, pi_parts = [], [], []
    t_np = np.asarray(t_np, dtype=np.float32).reshape(-1, 1)
    w_np = np.asarray(w_np, dtype=np.float32).reshape(-1, 1)
    for start in range(0, len(t_np), chunk):
        t = torch.tensor(t_np[start:start + chunk], device=device, requires_grad=True)
        y = torch.tensor(np.log(w_np[start:start + chunk]), device=device, requires_grad=True)
        V = value_net(t, y)
        V_y = torch.autograd.grad(V, y, torch.ones_like(V), create_graph=True, retain_graph=True)[0]
        V_yy = torch.autograd.grad(V_y, y, torch.ones_like(V_y), create_graph=False, retain_graph=True)[0]
        c = compute_c_from_foc_log(V_y, y)
        pi = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu)
        V_parts.append(V.detach().cpu().numpy())
        c_parts.append(c.detach().cpu().numpy())
        pi_parts.append(pi.detach().cpu().numpy())
    if was_training:
        value_net.train()
    return np.concatenate(V_parts), np.concatenate(c_parts), np.concatenate(pi_parts)


def eval_fulldim_test_metrics(
    value_net: nn.Module, n_points: int, margins: List[float],
) -> Dict[float, Dict[str, float]]:
    """Fixed corresponding (t,y) test points for every nested window."""
    rng = np.random.default_rng(104729 + int(MARKET_SEED))
    u_t = rng.random((int(n_points), 1))
    u_y = rng.random((int(n_points), 1))
    t_np = u_t * (T_FINAL - 1e-3)
    output: Dict[float, Dict[str, float]] = {}
    for margin in margins:
        y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, float(margin))
        y_np = y_lo + u_y * (y_hi - y_lo)
        w_np = np.exp(y_np)
        V, c, pi = eval_model_on_points(value_net, t_np, w_np)
        V_cf = closed_form_V(t_np, w_np)
        c_cf = closed_form_c(t_np, w_np)
        pi_cf = np.broadcast_to(pi_star_np.reshape(1, -1), pi.shape)
        output[float(margin)] = compute_metrics(V, c, pi, V_cf, c_cf, pi_cf)
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


# =============================================================================
# 11) Main
# =============================================================================
def main():
    out_dir = os.path.join(ARGS.output_root, ARGS.run_tag)
    weight_dir = ARGS.weight_root or os.path.join(out_dir, "weights")
    recorder = mxu.ExperimentRecorder(out_dir, weight_dir, ARGS)
    skip_figures = bool(ARGS.skip_figures or ARGS.skip_plots)
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

    if ARGS.eval_only:
        recorder.save_config_eval()
    else:
        recorder.save_config()
        recorder.rotate_training_logs()
        recorder.save_market_snapshot(
            mu_excess=mu_excess_np, Sigma_safe=Sigma_np, chol=chol_Sigma_np,
            pi_star=pi_star_np, Theta=np.array([Theta]), nu=np.array([nu]),
            gamma=np.array([gamma_risk]), r=np.array([r_rate]),
            rho_discount=np.array([rho_discount]), epsilon=np.array([epsilon]),
            T=np.array([T_FINAL]), w_min=np.array([x_min]), w_max=np.array([x_max]),
            n_assets=np.array([N_ASSETS]), market_seed=np.array([MARKET_SEED]),
            seed=np.array([SEED]),
        )

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

        stopper = None
        if not ARGS.eval_only and ARGS.pde_stop_threshold is not None:
            stopper = mxu.PDEEarlyStopper(
                threshold=float(ARGS.pde_stop_threshold),
                start_outer=int(ARGS.pde_stop_start_outer),
                patience=int(ARGS.pde_stop_patience),
                stop_flag_path=str(ARGS.stop_flag_path or ""), recorder=recorder,
                run_tag=ARGS.run_tag, model_type=ARGS.model_type)

        results = None
        loaded_weight_path = None
        if not ARGS.eval_only:
            recorder.write_status("running")
            if stopper is not None and stopper.shared_stop_exists():
                info = stopper.mark_from_existing_flag(outer_iter=0, pde_loss=None)
                print(f"[early-stop] shared stop flag already exists; skipping run: {info}")
                return
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
                weight_dir=weight_dir, recorder=recorder, stopper=stopper,
            )
            elapsed = time.time() - start
            h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
            print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")

            if not ARGS.timing_mode:
                train_fields = [
                    "outer_iter", "inner_epoch", "total", "pde", "terminal", "mono", "conc", "eta",
                    "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres", "sel_pres", "lr", "synthetic"]
                mxu.append_csv_rows(recorder.train_csv, results["loss_history"], train_fields)
                base_outer = [
                    "timestamp", "model_type", "run_tag", "outer_iter", "total_loss", "pde_loss",
                    "terminal_loss", "monotonicity_loss", "concavity_loss", "eta_loss", "train_pres",
                    "val_pde_rms", "val_terminal_rms", "val_pres", "val_pres_at_stop",
                    "val_pres_post_restore", "achieved_pres", "inner_epochs_used", "n_resamples", "target_reached",
                    "sel_best_pres", "sel_best_epoch", "sel_best_lr", "sel_checks", "sel_stopped",
                    "sel_restored", "lr_end_before_restore", "lr_best_checkpoint", "lr_after_restore",
                    "lr_carried_next", "pi_diff", "c_diff", "pi_vs_closed_form", "c_vs_closed_form",
                    "e_V_sup", "e_bundle_sup", "e_Xev", "diag_RelL2_V", "diag_RelL2_pi",
                    "diag_RelL2_c", "m_Vw", "m_minus_Vww", "m_curvature_y", "guard_frac_Vw",
                    "guard_frac_curvature", "clip_frac_pi_greedy", "clip_frac_kappa_low",
                    "clip_frac_kappa_high", "clip_frac_c_level_low", "clip_frac_c_level_high",
                    "diffusion_var_min_greedy", "diffusion_var_max_greedy",
                    "diffusion_var_min_frozen", "diffusion_var_max_frozen", "clip_frac_pi_frozen",
                    "clip_frac_kappa_low_frozen", "clip_frac_kappa_high_frozen",
                    "clip_frac_c_level_low_frozen", "clip_frac_c_level_high_frozen", "lr"]
                mxu.append_csv_rows(recorder.outer_csv, results["outer_rows"], base_outer)
            if results.get("stopped_early", False):
                recorder.write_status(
                    "stopped_early", elapsed_sec=elapsed,
                    final_weight_path=os.path.join(weight_dir, "value_net_final.pt"),
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

        margins = mxu.parse_eval_margins(ARGS.eval_margin)
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
                    total_optimizer_steps=results["total_optimizer_steps"],
                    total_inner_steps=results["total_inner_steps"],
                    pres_target=ARGS.pres_target, pres_max=results["pres_max"],
                    any_target_reached=bool(results["target_reached"]),
                    target_reached=bool(results["all_targets_reached"]),
                    target_reached_semantics="all_outer_training_validation_crossings")
            return

        # Full random test if requested; test_points=0 is the deterministic
        # grid fallback and uses the same metric definitions.
        Nt, Nw = int(ARGS.n_tau), int(ARGS.n_x)
        if int(ARGS.test_points) > 0:
            metrics_by_margin = eval_fulldim_test_metrics(
                solver.value_net, int(ARGS.test_points), margins)
        else:
            metrics_by_margin = {}
            for margin in margins:
                tt, ww, V_pred, c_pred, pi_pred, _ = eval_pinn_on_grid_margin(
                    solver.value_net, Nt=Nt, Nw=Nw, margin=margin)
                V_cf, c_cf, pi_cf = closed_form_numpy(tt, ww)
                metrics_by_margin[float(margin)] = compute_metrics(
                    V_pred, c_pred, pi_pred, V_cf, c_cf, pi_cf)

        metric_rows = []
        for margin in margins:
            metrics = metrics_by_margin[float(margin)]
            for key, val in metrics.items():
                metric_rows.append({
                    "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag, "scope": "fulldim", "eval_margin": margin,
                    "metric": key, "value": val})
        mxu.append_csv_rows(recorder.metrics_csv, metric_rows,
                            ["timestamp", "model_type", "run_tag", "scope", "eval_margin", "metric", "value"])

        eval_source = (
            f"{int(ARGS.test_points)} fixed random points"
            if int(ARGS.test_points) > 0
            else f"{Nt}x{Nw} deterministic grid"
        )
        print(f"\nEvaluation metrics by margin ({eval_source}):")
        for margin in margins:
            is_primary = float(margin) == float(margins[0])
            print(
                f"\n--- eval_margin={float(margin):.2f}"
                f"{' (primary)' if is_primary else ''} ---"
            )
            for key, value in metrics_by_margin[float(margin)].items():
                print(f"  {key}: {value:.6e}")

        if not skip_figures and not ARGS.eval_only:
            plot_convergence(results, save_path=os.path.join(out_dir, "plots", "convergence.png"), show=False)
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                solver.value_net, Nt=Nt, Nw=Nw, margin=margins[0])
            plot_comparison_heatmaps(
                tt, ww, V_pinn, c_pinn, pi_norm,
                save_path=os.path.join(out_dir, "plots", "comparison_heatmap.png"), show=False)

        if ARGS.eval_only:
            if os.path.exists(recorder.metrics_csv):
                backup_path = metrics_final_path + ".bak_train"
                if os.path.exists(metrics_final_path) and not os.path.exists(backup_path):
                    shutil.copyfile(metrics_final_path, backup_path)
                os.replace(recorder.metrics_csv, metrics_final_path)
                recorder.metrics_csv = metrics_final_path
            recorder.mark_success_eval(
                elapsed_sec=elapsed, primary_margin=margins[0],
                loaded_weight_path=loaded_weight_path)
        else:
            first_outer = results["outer_rows"][0] if results["outer_rows"] else {}
            recorder.mark_success(
                elapsed_sec=elapsed, outer_iters=len(results["outer_rows"]),
                total_optimizer_steps=results["total_optimizer_steps"], train_wall_sec=elapsed,
                primary_margin=margins[0], final_weight_path=os.path.join(weight_dir, "value_net_final.pt"),
                best_weight_path=os.path.join(weight_dir, "value_net_best_diag.pt"),
                pres_target=ARGS.pres_target, pres_max=results["pres_max"],
                total_inner_steps=results["total_inner_steps"],
                any_target_reached=bool(results["target_reached"]),
                target_reached=bool(results["all_targets_reached"]),
                target_reached_semantics="all_outer_training_validation_crossings",
                pi_init_scale=float(ARGS.pi_init_scale),
                pi_clip_abs=pi_clip_abs,
                diffusion_var_min_init=first_outer.get("diffusion_var_min_frozen", ""),
                diffusion_var_max_init=first_outer.get("diffusion_var_max_frozen", ""))
        print("\nDone.")
    except Exception as exc:
        if ARGS.eval_only:
            recorder.mark_failed_eval(reason=repr(exc))
        else:
            recorder.mark_failed(reason=repr(exc))
        raise


if __name__ == "__main__":
    main()
