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
  * hybrid sampling: resample collocation points every `resample_every` steps.
- For evaluation/plots we still compute (c, pi) via FOC, and compare to closed-form.

Author: derived from your uploaded scripts (no new theory introduced).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import market_setup  # keep as requested
import merton_experiment_utils as mxu


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
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--outer-iters", type=int, default=500,
                   help="Number of training blocks (kept name-compatible with Liu).")
    p.add_argument("--eval-epochs", type=int, default=200,
                   help="Optimizer steps per block.")
    p.add_argument("--resample-every", type=int, default=200,
                   help="Redraw collocation every K steps (0 = fixed batch).")
    # Loss weights
    p.add_argument("--w-terminal", type=float, default=10.0)
    p.add_argument("--w-shape", type=float, default=1.0)
    p.add_argument("--w-eta", type=float, default=1.5,
                   help="Weight of the RRA/consumption-shape diagnostic penalty.")
    p.add_argument("--eta-clip", type=str, default="10.0",
                   help="Optional |.|-clip for the eta diagnostic penalty (none = off).")
    # Evaluation window(s): first = PRIMARY (diagnostic + representative
    # metric); the rest are the E9 window-sensitivity list, re-evaluated on
    # the same trained network. Half-width margin m shrinks BOTH the t-axis
    # and the y=log W axis inward. 0.0 = full window.
    p.add_argument("--eval-margin", type=str, default="0.10,0.0,0.05,0.15,0.20")
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
    p.add_argument("--skip-plots", action="store_true")
    # Infrastructure (wired in Phase 2)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--timing-mode", type=int, default=0)
    p.add_argument("--stop-flag-path", type=str, default=None)
    p.add_argument("--pde-stop-threshold", type=float, default=None)
    p.add_argument("--pde-stop-start-outer", type=int, default=0)
    p.add_argument("--pde-stop-patience", type=int, default=1)
    return p


ARGS = build_arg_parser().parse_args()


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
    exp_term = np.exp(-nu * tau)
    denom = 1.0 + (nu * epsilon_bequest - 1.0) * exp_term
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    return (nu / denom) * W


def closed_form_V(t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """V(t,W)=A(t) W^{1-gamma}/(1-gamma)."""
    tau = T_FINAL - t
    exp_term = np.exp(-nu * tau)
    denom = 1.0 + (nu * epsilon_bequest - 1.0) * exp_term
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    A_t = (denom / nu) ** gamma_risk
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
def reduced_hjb_residual_log_multi(
    value_net: nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduced-form HJB residual in (t,y), after substituting FOCs for c and pi:

        0 = v_t + r v_y - rho v
            + (gamma/(1-gamma)) * ( (v_y / W) ^ ((gamma-1)/gamma) )
            + 0.5 * Theta * v_y^2 / (v_y - v_yy)

    We add "safe" abs/clamps to avoid NaNs early; shape penalties enforce correct signs.
    """
    V, V_t, V_y, V_yy = compute_derivatives_log(value_net, t, y)
    W = torch.exp(y)

    # For CRRA exponent (gamma-1)/gamma in (0,1), we need positive base
    V_y_safe = torch.abs(V_y) + eps

    denom = (V_y - V_yy)            # should be positive (concavity in W)
    denom_safe = torch.abs(denom) + eps

    exp_c = (gamma_risk - 1.0) / gamma_risk
    term_consumption = (gamma_risk / (1.0 - gamma_risk)) * ( (V_y_safe / W).pow(exp_c) )

    term_portfolio = 0.5 * Theta_t * (V_y_safe.pow(2) / denom_safe)

    residual = V_t + r_rate * V_y - rho_discount * V + term_consumption + term_portfolio
    return residual, V, V_y, V_yy, denom


# =============================================================================
# 8) (Optional) policies from FOC (used for evaluation/plots only)
# =============================================================================
# Bounds: keep same "safety" design as your PI-PINN code
pi_min_bound = -2.0
pi_max_bound = 2.0

M_utility_cap = 1e3
c_floor = ((gamma_risk - 1.0) * M_utility_cap) ** (-1.0 / (gamma_risk - 1.0))
kappa_min_bound = c_floor / w_min
kappa_max_bound = 3.0

c_min_bound = c_floor
c_max_bound = w_max


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
    pi_min: float = pi_min_bound,
    pi_max: float = pi_max_bound,
) -> torch.Tensor:
    """
    pi* in y=log W coordinates:
        pi* = -(V_y / (V_yy - V_y)) * Sigma^{-1} mu_excess
            =  (V_y / (V_y - V_yy)) * Sigma^{-1} mu_excess
    """
    denom = (V_y - V_yy)  # should be positive
    denom_sign = torch.sign(denom)
    denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
    denom_safe = torch.where(torch.abs(denom) < eps, denom_sign * eps, denom)

    scalar = (V_y / denom_safe)  # (batch,1)
    pi_raw = scalar * Sigma_inv_mu.view(1, -1)  # (batch,N)
    return torch.clamp(pi_raw, min=pi_min, max=pi_max)


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
    V_y_safe = torch.clamp(torch.abs(V_y), min=eps)
    eta = 1.0 - (V_yy / V_y_safe)

    return V, c, pi, eta


# =============================================================================
# 9) Training: Hybrid-sampling PINN on reduced nonlinear PDE
# =============================================================================
def train_pinn_hybrid_reduced_logw_multi(
    value_net: nn.Module,
    epochs: int = 100000,
    batch_size: int = 3000,
    lr: float = 5e-4,
    resample_every: int = 200,
    w_terminal: float = 20.0,
    w_shape: float = 1.0,
    w_eta: float = 0.0,
    eta_clip: float = 10.0,
    print_every: int = 2000,
    weight_dir: str = "weights/merton_multiasset_consumption_reduced_logw",
) -> Tuple[List[Dict[str, float]], optim.Optimizer]:
    os.makedirs(weight_dir, exist_ok=True)
    best_model_path = os.path.join(
        weight_dir, f"value_net_reduced_logw_N({N_ASSETS})_batch({batch_size})_Wmin({w_min})_T({T_FINAL})_weta({w_eta})_wterm({w_terminal}).pt"
    )

    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5000, min_lr=1e-6
    )

    loss_history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    best_iter = 0

    t_int = y_int = None
    t_term = y_term = None
    V_T_target = None

    print(f"\n{'='*70}")
    print("Training PINN (Reduced-form HJB, logW, hybrid sampling)")
    print(f"  epochs={epochs}, batch={batch_size}, lr={lr}, resample_every={resample_every}")
    print(f"  w_terminal={w_terminal}, w_shape={w_shape}, w_eta={w_eta}")
    print(f"{'='*70}\n")

    for epoch in range(1, epochs + 1):
        # resample_every == 0 -> fixed batch: draw once at the first epoch and
        # never resample (matches the argparse default / legacy single-batch).
        if epoch == 1 or (resample_every > 0 and (epoch - 1) % resample_every == 0):
            t_int, y_int = sample_interior(batch_size, device)
            t_term, y_term = sample_terminal(max(1, batch_size // 4), device)
            V_T_target = V_terminal_from_y(y_term).detach()

        optimizer.zero_grad(set_to_none=True)

        residual, V, V_y, V_yy, denom = reduced_hjb_residual_log_multi(value_net, t_int, y_int)
        pde_loss = torch.mean(residual ** 2)

        V_T_pred = value_net(t_term, y_term)
        terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)

        # Shape penalties in y
        mono_penalty = torch.mean(torch.relu(-V_y) ** 2)            # V_y >= 0
        conc_penalty = torch.mean(torch.relu(-denom) ** 2)          # denom=V_y - V_yy >= 0

        # Eta penalty (optional): eta = 1 - V_yy/V_y ~ gamma. Skipped entirely
        # when disabled (w_eta == 0) or when no clip is set (eta_clip is None),
        # so a None clip never reaches the arithmetic.
        if w_eta != 0.0 and eta_clip is not None:
            eps = 1e-8
            V_y_safe = torch.clamp(torch.abs(V_y), min=eps)
            eta = 1.0 - (V_yy / V_y_safe)
            eta_err = torch.clamp(eta - gamma_risk, -eta_clip, eta_clip)
            eta_loss = torch.mean(eta_err ** 2)
        else:
            eta_loss = torch.zeros((), device=t_int.device)

        total_loss = pde_loss + w_terminal * terminal_loss + w_shape * (mono_penalty + conc_penalty) + w_eta * eta_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        cur = float(total_loss.detach().cpu())
        if cur < best_loss:
            best_loss = cur
            best_state = {k: v.detach().cpu().clone() for k, v in value_net.state_dict().items()}
            best_iter = epoch
            torch.save(value_net.state_dict(), best_model_path)

        loss_history.append({
            "total": cur,
            "pde": float(pde_loss.detach().cpu()),
            "terminal": float(terminal_loss.detach().cpu()),
            "mono": float(mono_penalty.detach().cpu()),
            "conc": float(conc_penalty.detach().cpu()),
            "eta": float(eta_loss.detach().cpu()),
        })

        if print_every > 0 and epoch % print_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:6d}/{epochs}] total={cur:.3e} | pde={pde_loss.item():.3e} | "
                f"term={terminal_loss.item():.3e} | eta={eta_loss.item():.3e} | lr={lr_now:.2e}"
            )

        # Clear leaf grads (memory)
        if t_int.grad is not None:
            t_int.grad = None
        if y_int.grad is not None:
            y_int.grad = None

    if best_state is not None:
        value_net.load_state_dict(best_state)
        print(f"\nRestored best model from iter {best_iter} (loss={best_loss:.3e})")
        print(f"Saved best weights to: {best_model_path}")

    return loss_history, optimizer


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

    value_net.train()
    return tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm


def eval_pinn_on_grid_margin(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
    margin: float = 0.0,
    chunk: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid evaluation on an eval window shrunk by HALF-WIDTH `margin`.

    Both the t-axis [t_min, t_max] and the y=log W axis [y_min, y_max] are
    shrunk inward (each side moves by margin*(hi-lo)/2), matching Liu's
    shrink_bounds convention. margin=0.0 reproduces the full-window grid.
    The W grid is exp of the shrunk y-range so points stay inside the trained
    log-wealth box.
    """
    value_net.eval()
    t_lo, t_hi = mxu.shrink_bounds(t_min, t_max - 1e-3, margin)
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, margin)
    t_vals = np.linspace(t_lo, t_hi, Nt)
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
    value_net.train()
    return tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm


def compute_metrics(
    V_pinn: np.ndarray, c_pinn: np.ndarray, pi_pinn: np.ndarray,
    V_cf: np.ndarray, c_cf: np.ndarray, pi_cf: np.ndarray
) -> Dict[str, float]:
    mse_V = float(np.mean((V_pinn - V_cf) ** 2))
    mse_c = float(np.mean((c_pinn - c_cf) ** 2))
    mse_pi = float(np.mean((pi_pinn - pi_cf) ** 2))

    rel_rmse_V = float(np.sqrt(mse_V) / (np.std(V_cf) + 1e-8))
    rel_rmse_c = float(np.sqrt(mse_c) / (np.std(c_cf) + 1e-8))
    rel_rmse_pi = float(np.sqrt(mse_pi) / (np.std(pi_cf) + 1e-8))

    max_V_err = float(np.max(np.abs(V_pinn - V_cf)))
    max_c_err = float(np.max(np.abs(c_pinn - c_cf)))
    max_pi_err = float(np.max(np.abs(pi_pinn - pi_cf)))

    return {
        "MSE_V": mse_V, "MSE_c": mse_c, "MSE_pi": mse_pi,
        "RelRMSE_V": rel_rmse_V, "RelRMSE_c": rel_rmse_c, "RelRMSE_pi": rel_rmse_pi,
        "MaxErr_V": max_V_err, "MaxErr_c": max_c_err, "MaxErr_pi": max_pi_err,
    }


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
                extent=[w_min, w_max, t_min, t_max],
                interpolation="bilinear",
                cmap="RdBu_r",
                norm=norm,
            )
        else:
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[w_min, w_max, t_min, t_max],
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
    # Output / weight directories (recorder-owned).
    out_dir = os.path.join(ARGS.output_root, ARGS.run_tag)
    weight_dir = ARGS.weight_root or os.path.join(out_dir, "weights")
    recorder = mxu.ExperimentRecorder(out_dir, weight_dir, ARGS)

    if ARGS.eval_only:
        recorder.save_config_eval()
    else:
        recorder.save_config()
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
        # Training schedule (blocks x steps; the trainer consumes a single
        # epoch budget = outer_iters * eval_epochs).
        outer_iters = int(ARGS.outer_iters)
        eval_epochs = int(ARGS.eval_epochs)
        epochs = outer_iters * eval_epochs
        resample_every = int(ARGS.resample_every)
        batch_size = int(ARGS.batch_size)
        lr = float(ARGS.lr)
        w_terminal = float(ARGS.w_terminal)
        w_shape = float(ARGS.w_shape)
        w_eta = float(ARGS.w_eta)
        eta_clip = None if str(ARGS.eta_clip).lower() == "none" else float(ARGS.eta_clip)
        print_every = int(ARGS.print_every)

        start = time.time()
        value_net = ValueNetLogW(hidden=int(ARGS.value_hidden), depth=int(ARGS.value_depth)).to(device)

        if not ARGS.eval_only:
            loss_history, _opt = train_pinn_hybrid_reduced_logw_multi(
                value_net=value_net,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                resample_every=resample_every,
                w_terminal=w_terminal,
                w_shape=w_shape,
                w_eta=w_eta,
                eta_clip=eta_clip,
                print_every=print_every,
            )
            elapsed = time.time() - start
            h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
            print(f"\nElapsed time: {h:02d}:{m:02d}:{s:05.2f}")

            # Persist the training-loss trajectory.
            train_rows = []
            for i, hrow in enumerate(loss_history, start=1):
                train_rows.append({"epoch": i, "total_loss": hrow["total"], "pde_loss": hrow["pde"],
                                   "terminal_loss": hrow["terminal"], "mono": hrow["mono"],
                                   "conc": hrow["conc"], "eta": hrow["eta"]})
            mxu.append_csv_rows(recorder.train_csv, train_rows,
                                ["epoch", "total_loss", "pde_loss", "terminal_loss", "mono", "conc", "eta"])
        else:
            elapsed = 0.0

        # Grid evaluation vs closed-form at every eval margin (first = primary).
        Nt, Nw = int(ARGS.n_tau), int(ARGS.n_x)
        margins = mxu.parse_eval_margins(ARGS.eval_margin)
        metric_rows = []
        primary_metrics = None
        for mi, margin in enumerate(margins):
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                value_net, Nt=Nt, Nw=Nw, margin=margin, chunk=4000)
            V_cf, c_cf, pi_cf = closed_form_numpy(tt, ww)
            metrics = compute_metrics(V_pinn, c_pinn, pi_pinn, V_cf, c_cf, pi_cf)
            if mi == 0:
                primary_metrics = metrics
            for key, val in metrics.items():
                metric_rows.append({
                    "timestamp": mxu.now_iso(), "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag, "scope": "fulldim", "eval_margin": margin,
                    "metric": key, "value": val})
        mxu.append_csv_rows(recorder.metrics_csv, metric_rows,
                            ["timestamp", "model_type", "run_tag", "scope", "eval_margin", "metric", "value"])

        print("\nMetrics (primary margin):")
        for k, v in primary_metrics.items():
            print(f"  {k}: {v:.6e}")

        if not ARGS.skip_plots and not ARGS.eval_only:
            plot_loss_history(loss_history, save_path=os.path.join(out_dir, "plots", "loss_history.png"), show=False)
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                value_net, Nt=Nt, Nw=Nw, margin=margins[0], chunk=4000)
            plot_comparison_heatmaps(
                tt, ww, V_pinn, c_pinn, pi_norm,
                save_path=os.path.join(out_dir, "plots", "comparison_heatmap.png"), show=False)

        if ARGS.eval_only:
            recorder.mark_success_eval(elapsed_sec=elapsed, primary_margin=margins[0])
        else:
            recorder.mark_success(elapsed_sec=elapsed, epochs=epochs,
                                  total_optimizer_steps=epochs, train_wall_sec=elapsed,
                                  primary_margin=margins[0])
        print("\nDone.")
    except Exception as exc:
        if ARGS.eval_only:
            recorder.mark_failed_eval(reason=repr(exc))
        else:
            recorder.mark_failed(reason=repr(exc))
        raise


if __name__ == "__main__":
    main()

