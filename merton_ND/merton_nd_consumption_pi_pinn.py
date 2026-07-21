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
import time
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
    # Control bounds (PI-PINN-specific)
    p.add_argument("--pi-min", type=float, default=-2.0)
    p.add_argument("--pi-max", type=float, default=2.0)
    p.add_argument("--kappa-max-bound", type=float, default=3.0,
                   help="Upper bound on kappa = c/W.")
    p.add_argument("--utility-cap", type=float, default=1e3,
                   help="M in the CRRA floor c_floor = ((gamma-1) M)^(-1/(gamma-1)).")
    # Network / optimization
    p.add_argument("--value-hidden", type=int, default=256)
    p.add_argument("--value-depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--outer-iters", type=int, default=500)
    p.add_argument("--eval-epochs", type=int, default=200)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    # Policy init
    p.add_argument("--pi-init-method", type=str, default="myopic")
    p.add_argument("--c-init-method", type=str, default="proportional")
    # Loss weights
    p.add_argument("--w-terminal", type=float, default=10.0)
    p.add_argument("--w-eta", type=float, default=3.0)
    p.add_argument("--eta-clip", type=str, default="10.0",
                   help="Optional |.|-clip for the eta penalty (none = off).")
    p.add_argument("--eta-focus-w", type=str, default="none",
                   help="Optional wealth focus for the eta penalty (none = off).")
    # Evaluation window(s): first = PRIMARY; half-width margin shrinks the
    # t-axis and y=log W axis inward. 0.0 = full window.
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
    p.add_argument("--skip-plots", action="store_true")
    # Infrastructure (wired with the recorder)
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

# Control bounds (componentwise for pi)
pi_min_bound = float(ARGS.pi_min)
pi_max_bound = float(ARGS.pi_max)

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
print(f"  pi bounds: [{pi_min_bound},{pi_max_bound}] (componentwise)")
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
    pi_min: float = pi_min_bound,
    pi_max: float = pi_max_bound,
) -> torch.Tensor:
    """
    Multi-asset pi* in y=log W coordinates:
        pi* = -(V_y / (V_yy - V_y)) * Sigma^{-1} mu_excess.
    """
    denom = (V_yy - V_y)  # (batch,1)
    denom_sign = torch.sign(denom)
    denom_sign = torch.where(denom_sign == 0, -torch.ones_like(denom_sign), denom_sign)
    denom_safe = torch.where(torch.abs(denom) < eps, denom_sign * eps, denom)

    scalar = -(V_y / denom_safe)  # (batch,1)
    pi_raw = scalar * Sigma_inv_mu.view(1, -1)  # (batch,N)
    return torch.clamp(pi_raw, min=pi_min, max=pi_max)


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
        pi_min: float = pi_min_bound,
        pi_max: float = pi_max_bound,
        c_min: float = c_min_bound,
        c_max: float = c_max_bound,
        device: torch.device = device,
    ):
        self.device = device
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.c_min = c_min
        self.c_max = c_max

        self.value_net = ValueNetLogW(hidden=value_hidden, depth=value_depth).to(device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            verbose=False,
        )
        self.initial_lr = lr

    def initialize_pi(self, t: torch.Tensor, y: torch.Tensor, method: str = "myopic") -> torch.Tensor:
        n = t.shape[0]
        if method == "zero":
            pi = torch.zeros(n, N_ASSETS, device=self.device)
        elif method == "myopic":
            # start from closed-form vector
            pi = pi_star.view(1, -1).repeat(n, 1)
        elif method == "random":
            pi = self.pi_min + torch.rand(n, N_ASSETS, device=self.device) * (self.pi_max - self.pi_min)
        else:
            raise ValueError(f"Unknown pi init method: {method}")
        return torch.clamp(pi, self.pi_min, self.pi_max)

    def initialize_c(self, t: torch.Tensor, y: torch.Tensor, method: str = "proportional") -> torch.Tensor:
        n = t.shape[0]
        W = torch.exp(y.detach())
        if method == "zero":
            c = torch.full((n, 1), self.c_min, device=self.device)
        elif method == "proportional":
            c = rho_discount * W
        elif method == "random":
            c = self.c_min + torch.rand(n, 1, device=self.device) * (self.c_max - self.c_min)
        else:
            raise ValueError(f"Unknown c init method: {method}")
        return torch.clamp(c, self.c_min, self.c_max)

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
        eta_clip: float = 10.0,
        print_every: int = 50,
    ) -> Tuple[List[Dict], float]:
        loss_history: List[Dict] = []
        best_loss = float("inf")
        best_state = None

        c_fixed = c_n.detach()
        pi_fixed = pi_n.detach()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

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
            eta_err = torch.clamp(eta - gamma_risk, -eta_clip, eta_clip)
            eta_err_sq = eta_err ** 2
            if eta_focus_w is not None:
                W_int = torch.exp(y_int)
                mask = (W_int <= eta_focus_w).float()
                eta_loss = (eta_err_sq * mask).sum() / (mask.sum() + 1e-12)
            else:
                eta_loss = torch.mean(eta_err_sq)

            total_loss = pde_loss + w_terminal * terminal_loss + w_shape * (mono_penalty + conc_penalty) + w_eta * eta_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            cur = float(total_loss.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {k: v.detach().cpu().clone() for k, v in self.value_net.state_dict().items()}

            loss_history.append(
                {
                    "total": cur,
                    "pde": float(pde_loss.item()),
                    "terminal": float(terminal_loss.item()),
                    "mono": float(mono_penalty.item()),
                    "conc": float(conc_penalty.item()),
                    "eta": float(eta_loss.item()),
                }
            )

            if epoch % print_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"      [Eval {epoch:4d}/{epochs}] Loss: {cur:.3e} | PDE: {pde_loss.item():.3e} | "
                    f"Terminal: {terminal_loss.item():.3e} | Eta: {eta_loss.item():.3e} | LR: {lr:.2e}"
                )

        if best_state is not None:
            self.value_net.load_state_dict(best_state)

        return loss_history, best_loss

    def policy_improvement(self, t: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.value_net.eval()
        t_eval = t.detach().clone().requires_grad_(True)
        y_eval = y.detach().clone().requires_grad_(True)

        V = self.value_net(t_eval, y_eval)
        ones = torch.ones_like(V)
        V_y = torch.autograd.grad(V, y_eval, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        V_yy = torch.autograd.grad(V_y, y_eval, grad_outputs=torch.ones_like(V_y), create_graph=True, retain_graph=True)[0]

        c_new = compute_c_from_foc_log(V_y, y_eval, c_min=self.c_min, c_max=self.c_max)
        pi_new = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu, pi_min=self.pi_min, pi_max=self.pi_max)

        self.value_net.train()
        return c_new.detach(), pi_new.detach()

    def run_policy_iteration(
        self,
        outer_iters: int = 200,
        eval_epochs: int = 200,
        batch_size: int = 2000,
        w_terminal: float = 20.0,
        w_eta: float = 0.0,
        eta_focus_w: Optional[float] = None,
        eta_clip: float = 10.0,
        pi_init_method: str = "myopic",
        c_init_method: str = "proportional",
        print_every_outer: int = 10,
        print_every_eval: int = 200,
        verbose_detail: bool = False,
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
        }

        best_eval_loss = float("inf")
        best_iter = 0

        # directories (local)
        weight_dir = "weights/merton_multiasset_consumption_logw"
        os.makedirs(weight_dir, exist_ok=True)
        best_model_path = os.path.join(
            weight_dir, f"value_net_logw_N({N_ASSETS})_batch({batch_size})_Wmin({w_min})_T({T_FINAL})_weta({w_eta})_wterm({w_terminal}).pt"
        )

        # initial sample
        t_colloc, y_colloc = sample_interior(batch_size, self.device)
        t_term, y_term = sample_terminal(batch_size // 2, self.device)
        V_T_target = V_terminal_from_y(y_term).detach()

        c_n = self.initialize_c(t_colloc, y_colloc, method=c_init_method)
        pi_n = self.initialize_pi(t_colloc, y_colloc, method=pi_init_method)
        print(f"Initial c: mean={c_n.mean().item():.4f}, std={c_n.std().item():.4f}")
        print(f"Initial pi: mean={pi_n.mean().item():.4f}, std={pi_n.std().item():.4f}")
        print(f"pi* stats: mean={pi_star.mean().item():.4f}, std={pi_star.std().item():.4f}, ||pi*||2={pi_star.norm().item():.4f}")

        for it in range(1, outer_iters + 1):
            verbose = (it % print_every_outer == 0) or (it <= 3)
            if verbose:
                print(f"\n[Outer Iteration {it}/{outer_iters}]")
                print("-" * 40)

            # fresh samples
            t_colloc, y_colloc = sample_interior(batch_size, self.device)
            t_term, y_term = sample_terminal(batch_size // 2, self.device)
            V_T_target = V_terminal_from_y(y_term).detach()

            # recompute policy on new points
            if it > 1:
                c_n, pi_n = self.policy_improvement(t_colloc, y_colloc)
            else:
                c_n = self.initialize_c(t_colloc, y_colloc, method=c_init_method)
                pi_n = self.initialize_pi(t_colloc, y_colloc, method=pi_init_method)

            # evaluation step
            eval_loss_hist, inner_best = self.policy_evaluation(
                c_n=c_n,
                pi_n=pi_n,
                t_colloc=t_colloc,
                y_colloc=y_colloc,
                t_term=t_term,
                y_term=y_term,
                V_T_target=V_T_target,
                epochs=eval_epochs,
                w_terminal=w_terminal,
                w_eta=w_eta,
                eta_focus_w=eta_focus_w,
                eta_clip=eta_clip,
                print_every=print_every_eval if (verbose and verbose_detail) else (eval_epochs + 1),
            )
            results["loss_history"].extend(eval_loss_hist)
            results["eval_loss"].append(inner_best)
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

            # scheduler
            self.scheduler.step(inner_best)
            cur_lr = self.optimizer.param_groups[0]["lr"]
            results["lr"].append(cur_lr)

            if inner_best < best_eval_loss:
                best_eval_loss = inner_best
                best_iter = it
                torch.save(self.value_net.state_dict(), best_model_path)

            if verbose:
                last = eval_loss_hist[-1]
                print(
                    f"  c mean: {c_new.mean().item():.4f}, std: {c_new.std().item():.4f} | "
                    f"c diff: {c_diff:.2e} | vs cf: {c_vs_cf:.2e}"
                )
                print(
                    f"  pi mean: {pi_new.mean().item():.4f}, std: {pi_new.std().item():.4f} | "
                    f"pi diff: {pi_diff:.2e} | vs cf: {pi_vs_cf:.2e}"
                )
                print(
                    f"  Eval(best): {inner_best:.3e} | PDE={last['pde']:.3e} | Term={last['terminal']:.3e} | "
                    f"Eta={last['eta']:.3e} | LR={cur_lr:.2e}"
                )

        if os.path.exists(best_model_path):
            self.value_net.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"\nRestored best model @ iter {best_iter} (loss={best_eval_loss:.3e})")

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
    value_net.eval()

    t_vals = np.linspace(t_min, t_max - 1e-3, Nt)
    w_vals = np.linspace(x_min, x_max, Nw)

    tt = np.zeros((Nt, Nw))
    ww = np.zeros((Nt, Nw))
    V_grid = np.zeros((Nt, Nw))
    c_grid = np.zeros((Nt, Nw))
    pi_grid = np.zeros((Nt, Nw, N_ASSETS))
    pi_norm_grid = np.zeros((Nt, Nw))

    for i, t_val in enumerate(t_vals):
        for j, w_val in enumerate(w_vals):
            tt[i, j] = t_val
            ww[i, j] = w_val

            t_t = torch.tensor([[t_val]], device=device, dtype=torch.float32).requires_grad_(True)
            y_t = torch.tensor([[np.log(w_val)]], device=device, dtype=torch.float32).requires_grad_(True)

            V = value_net(t_t, y_t)
            ones = torch.ones_like(V)
            V_y = torch.autograd.grad(V, y_t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
            V_yy = torch.autograd.grad(V_y, y_t, grad_outputs=torch.ones_like(V_y), create_graph=True, retain_graph=True)[0]

            c_val = compute_c_from_foc_log(V_y, y_t)
            pi_val = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu)  # (1, N)

            V_grid[i, j] = float(V.item())
            c_grid[i, j] = float(c_val.item())
            pi_np = pi_val.detach().cpu().numpy().reshape(-1)
            pi_grid[i, j, :] = pi_np
            pi_norm_grid[i, j] = float(np.linalg.norm(pi_np))

    value_net.train()
    return tt, ww, V_grid, c_grid, pi_grid, pi_norm_grid

def eval_pinn_on_grid_margin(
    value_net: nn.Module,
    Nt: int = 100,
    Nw: int = 100,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid evaluation on an eval window shrunk by HALF-WIDTH `margin`.

    Both the t-axis and the y=log W axis are shrunk inward (shrink_bounds
    convention); the W grid is exp of the shrunk y-range so points stay inside
    the trained log-wealth box. margin=0.0 reproduces the full-window grid.
    """
    value_net.eval()
    t_lo, t_hi = mxu.shrink_bounds(t_min, t_max - 1e-3, margin)
    y_lo, y_hi = mxu.shrink_bounds(y_min, y_max, margin)
    t_vals = np.linspace(t_lo, t_hi, Nt)
    w_vals = np.exp(np.linspace(y_lo, y_hi, Nw))

    tt = np.zeros((Nt, Nw)); ww = np.zeros((Nt, Nw))
    V_grid = np.zeros((Nt, Nw)); c_grid = np.zeros((Nt, Nw))
    pi_grid = np.zeros((Nt, Nw, N_ASSETS)); pi_norm_grid = np.zeros((Nt, Nw))

    for i, t_val in enumerate(t_vals):
        for j, w_val in enumerate(w_vals):
            tt[i, j] = t_val
            ww[i, j] = w_val
            t_t = torch.tensor([[t_val]], device=device, dtype=torch.float32).requires_grad_(True)
            y_t = torch.tensor([[np.log(w_val)]], device=device, dtype=torch.float32).requires_grad_(True)
            V = value_net(t_t, y_t)
            ones = torch.ones_like(V)
            V_y = torch.autograd.grad(V, y_t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
            V_yy = torch.autograd.grad(V_y, y_t, grad_outputs=torch.ones_like(V_y), create_graph=True, retain_graph=True)[0]
            c_val = compute_c_from_foc_log(V_y, y_t)
            pi_val = compute_pi_from_foc_log_multi(V_y, V_yy, Sigma_inv_mu)
            V_grid[i, j] = float(V.item())
            c_grid[i, j] = float(c_val.item())
            pi_np = pi_val.detach().cpu().numpy().reshape(-1)
            pi_grid[i, j, :] = pi_np
            pi_norm_grid[i, j] = float(np.linalg.norm(pi_np))

    value_net.train()
    return tt, ww, V_grid, c_grid, pi_grid, pi_norm_grid


def compute_metrics(V_pinn, c_pinn, pi_pinn, V_cf, c_cf, pi_cf) -> Dict[str, float]:
    """Compute MSE / RelRMSE / MaxErr for (V, c, pi)."""
    mse_V = np.mean((V_pinn - V_cf) ** 2)
    mse_c = np.mean((c_pinn - c_cf) ** 2)
    mse_pi = np.mean((pi_pinn - pi_cf) ** 2)

    rel_rmse_V = np.sqrt(mse_V) / (np.std(V_cf) + 1e-8)
    rel_rmse_c = np.sqrt(mse_c) / (np.std(c_cf) + 1e-8)
    rel_rmse_pi = np.sqrt(mse_pi) / (np.std(pi_cf) + 1e-8)

    max_V_err = np.max(np.abs(V_pinn - V_cf))
    max_c_err = np.max(np.abs(c_pinn - c_cf))
    max_pi_err = np.max(np.abs(pi_pinn - pi_cf))

    return {
        "MSE_V": float(mse_V),
        "MSE_c": float(mse_c),
        "MSE_pi": float(mse_pi),
        "RelRMSE_V": float(rel_rmse_V),
        "RelRMSE_c": float(rel_rmse_c),
        "RelRMSE_pi": float(rel_rmse_pi),
        "MaxErr_V": float(max_V_err),
        "MaxErr_c": float(max_c_err),
        "MaxErr_pi": float(max_pi_err),
    }

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
                extent=[x_min, x_max, t_min, t_max],
                interpolation="bilinear",
                cmap="RdBu_r",
                norm=norm,
            )
        else:
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[x_min, x_max, t_min, t_max],
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

        start = time.time()
        solver = PIPINN_MultiAsset_Consumption_LogW(
            value_hidden=int(ARGS.value_hidden),
            value_depth=int(ARGS.value_depth),
            lr=float(ARGS.lr),
            scheduler_patience=int(ARGS.scheduler_patience),
            scheduler_factor=float(ARGS.scheduler_factor),
            scheduler_min_lr=float(ARGS.scheduler_min_lr),
            pi_min=pi_min_bound,
            pi_max=pi_max_bound,
            c_min=c_min_bound,
            c_max=c_max_bound,
            device=device,
        )

        results = None
        if not ARGS.eval_only:
            results = solver.run_policy_iteration(
                outer_iters=int(ARGS.outer_iters),
                eval_epochs=int(ARGS.eval_epochs),
                batch_size=int(ARGS.batch_size),
                w_terminal=float(ARGS.w_terminal),
                w_eta=float(ARGS.w_eta),
                eta_focus_w=eta_focus_w,
                eta_clip=eta_clip,
                pi_init_method=str(ARGS.pi_init_method),
                c_init_method=str(ARGS.c_init_method),
                print_every_outer=int(ARGS.print_every_outer),
                print_every_eval=int(ARGS.print_every_eval),
                verbose_detail=bool(ARGS.print_every_eval > 0),
            )
            elapsed = time.time() - start
            h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
            print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")

            # Per-outer history for the E3-a / Figure-2 contraction analysis.
            # e_Xev is the closed-form-referenced convergence signal (here the
            # policy L2 gap to the closed form), matching the Liu column name
            # so make_figure2_contraction reads it unchanged.
            n_outer = len(results["eval_loss"])
            outer_rows = []
            for i in range(n_outer):
                pi_cf = results["pi_vs_closed_form"][i]
                c_cf = results["c_vs_closed_form"][i]
                outer_rows.append({
                    "outer_iter": i + 1,
                    "inner_epochs_used": int(ARGS.eval_epochs),
                    "eval_loss": results["eval_loss"][i],
                    "eta_loss": results["eta_loss"][i],
                    "pi_diff": results["pi_diff"][i],
                    "c_diff": results["c_diff"][i],
                    "pi_vs_closed_form": pi_cf,
                    "c_vs_closed_form": c_cf,
                    # sqrt(policy MSE-gap): an L2-type distance to the optimal
                    # controls; e_Xev = pi-part (primary contraction signal).
                    "e_Xev": float(np.sqrt(max(pi_cf, 0.0))),
                    "e_pi": float(np.sqrt(max(pi_cf, 0.0))),
                    "e_c": float(np.sqrt(max(c_cf, 0.0))),
                    "lr": results["lr"][i],
                    "pi_mean": results["pi_mean"][i], "pi_std": results["pi_std"][i],
                    "c_mean": results["c_mean"][i], "c_std": results["c_std"][i],
                })
            mxu.append_csv_rows(recorder.outer_csv, outer_rows,
                                ["outer_iter", "inner_epochs_used", "eval_loss", "eta_loss",
                                 "pi_diff", "c_diff", "pi_vs_closed_form", "c_vs_closed_form",
                                 "e_Xev", "e_pi", "e_c", "lr",
                                 "pi_mean", "pi_std", "c_mean", "c_std"])
        else:
            elapsed = 0.0

        # Grid evaluation vs closed-form at every eval margin (first = primary).
        Nt, Nw = int(ARGS.n_tau), int(ARGS.n_x)
        margins = mxu.parse_eval_margins(ARGS.eval_margin)
        metric_rows = []
        primary_metrics = None
        for mi, margin in enumerate(margins):
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                solver.value_net, Nt=Nt, Nw=Nw, margin=margin)
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
            plot_convergence(results, save_path=os.path.join(out_dir, "plots", "convergence.png"), show=False)
            tt, ww, V_pinn, c_pinn, pi_pinn, pi_norm = eval_pinn_on_grid_margin(
                solver.value_net, Nt=Nt, Nw=Nw, margin=margins[0])
            plot_comparison_heatmaps(
                tt, ww, V_pinn, c_pinn, pi_norm,
                save_path=os.path.join(out_dir, "plots", "comparison_heatmap.png"), show=False)

        if ARGS.eval_only:
            recorder.mark_success_eval(elapsed_sec=elapsed, primary_margin=margins[0])
        else:
            recorder.mark_success(elapsed_sec=elapsed, outer_iters=int(ARGS.outer_iters),
                                  total_optimizer_steps=int(ARGS.outer_iters) * int(ARGS.eval_epochs),
                                  train_wall_sec=elapsed, primary_margin=margins[0])
        print("\nDone.")
    except Exception as exc:
        if ARGS.eval_only:
            recorder.mark_failed_eval(reason=repr(exc))
        else:
            recorder.mark_failed(reason=repr(exc))
        raise


if __name__ == "__main__":
    main()