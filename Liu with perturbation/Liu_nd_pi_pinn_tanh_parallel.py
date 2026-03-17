"""
Multi-dimensional Kim-Omberg Portfolio Problem - PINN Solution
===============================================================
Dynamic nonmyopic portfolio optimization with N risky assets and M state variables.

State dynamics (OU process):
    dX_t = K(x̄ - X_t) dt + Σ_X dZ_t^X,    X_t ∈ ℝ^M

Wealth dynamics (normalized):
    dW_t = (rW_t + θ_t^⊤ λ(X_t)) dt + θ_t^⊤ dZ_t,    θ_t ∈ ℝ^N

Market price of risk (affine in state):
    λ(x) = λ_0 + Λx,    λ_0 ∈ ℝ^N, Λ ∈ ℝ^{N×M}

Correlation structure:
    𝔼[dZ_t dZ_t^{X⊤}] = ρ dt,    ρ ∈ ℝ^{N×M}

Key matrices:
    Γ := ρ Σ_X^⊤ ∈ ℝ^{N×M}
    Q := Σ_X Σ_X^⊤ ∈ ℝ^{M×M}

Fully nonlinear HJB PDE (in τ = T - t):
    0 = -V_τ + rwV_w + (k_0 - Kx)^⊤ V_x + ½tr(QV_xx) - ‖λ(x)V_w + ΓV_wx‖² / (2V_ww)

Terminal condition (τ = 0):
    V(0, w, x) = U(w) = w^{1-γ} / (1-γ)

Optimal portfolio:
    θ*(τ, w, x) = -(λ(x)V_w + ΓV_wx) / V_ww
                = (w/γ)(m(τ) + A(τ)x)    [closed-form structure]

Reference: Kim & Omberg (1996), "Dynamic Nonmyopic Portfolio Behavior", RFS

=== Parallel Execution Version ===
Usage:
    python Liu_nd_pi_pinn_tanh_parallel.py --eps 0.1 --gpu 0
    python Liu_nd_pi_pinn_tanh_parallel.py --eps 0.05 --gpu 1
"""

import os
import sys
import math
import argparse
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PI-PINN with tanh risk premium perturbation')
    parser.add_argument('--eps', type=float, required=True,
                        help='Epsilon value for tanh perturbation (e.g., 0.1, 0.05, 0.0)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (default: 0)')
    parser.add_argument('--seed', type=int, default=12,
                        help='Random seed (default: 12)')
    parser.add_argument('--n_assets', type=int, default=50,
                        help='Number of risky assets (default: 50)')
    parser.add_argument('--m_states', type=int, default=10,
                        help='Number of state variables (default: 10)')
    parser.add_argument('--outer_iters', type=int, default=200,
                        help='Number of outer PI iterations (default: 200)')
    parser.add_argument('--eval_epochs', type=int, default=200,
                        help='Number of evaluation epochs per PI iteration (default: 200)')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='Batch size for training (default: 3000)')
    parser.add_argument('--output_dir', type=str, default='outputs/pi-pinn',
                        help='Base output directory (default: outputs/pi-pinn)')
    parser.add_argument('--weight_dir', type=str, default='weights/pi-pinn',
                        help='Base weight directory (default: weights/pi-pinn)')
    return parser.parse_args()

args = parse_args()

# =============================================================================
# 0) Reproducibility + Device
# =============================================================================
SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"[EPS={args.eps:.4f}] Device: {device}")

if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device=device)

# Add path for joint_market_setup
sys.path.insert(0, '/mnt/user-data/uploads')
from joint_market_setup_dirichlet import generate_joint_market_params, JointMarketParams, cholesky_solve


def print_joint_market_report(params, gamma=2.0, Y_ref=None):
    n, k = params.n, params.k
    jc = params.diag["joint_corr"]

    print(f"\n[Joint Market Parameters] n={n}, k={k}")
    print(f"  sigma (asset vols)   : min={params.sigma.min():.4f}, max={params.sigma.max():.4f}, mean={params.sigma.mean():.4f}")
    if k > 0:
        print(f"  eta   (state vols)   : min={params.eta.min():.4f}, max={params.eta.max():.4f}, mean={params.eta.mean():.4f}")

    print(f"\n[Numerical Stability Diagnostics]")
    print(f"  joint cond(C)        : {jc['cond']:.2f}")
    print(f"  joint min eig(C)     : {jc['min_eig']:.2e}")
    print(f"  joint max|rho_ij|    : {jc['max_abs_rho']:.4f}")
    print(f"  joint shrink alpha   : {jc['alpha_used']:.4f}")
    print(f"  asset cond(raw Σ_RR) : {params.diag['asset_cond_raw']:.2f}")
    print(f"  asset cond(safe Σ)   : {params.diag['asset_cond_safe']:.2f}")
    print(f"  ridge delta_asset    : {params.diag['delta_asset']:.2e}")

    if k == 0 or params.alpha is None:
        print("\n[Myopic (reference-state) quantities] skipped (k==0 or alpha not sampled).")
        return

    if Y_ref is None:
        Y_ref = params.theta

    Y_ref = np.asarray(Y_ref, dtype=float).reshape(k,)
    risk = params.alpha @ Y_ref
    mu_excess = params.sigma * risk
    Sigma = params.Sigma_RR_safe

    Sigma_inv_mu = cholesky_solve(params.chol_Sigma_RR_safe, mu_excess)
    Theta = float(mu_excess @ Sigma_inv_mu)

    pi_myo = (1.0 / gamma) * Sigma_inv_mu

    print(f"\n[Reference State]")
    print(f"  Y_ref                : {Y_ref}")

    print(f"\n[Key Quantities @ Y_ref]")
    print(f"  Θ = μ⊤Σ⁻¹μ           : {Theta:.6f}")
    print(f"  √Θ (multi-asset SR)  : {np.sqrt(max(Theta,0.0)):.6f}")

    print(f"\n[Myopic Portfolio @ Y_ref]")
    print(f"  π_myo = (1/γ)Σ⁻¹μ     : (shape {pi_myo.shape})")
    print(f"  ||π_myo||₂           : {np.linalg.norm(pi_myo):.4f}")
    print(f"  ||π_myo||₁           : {np.linalg.norm(pi_myo, 1):.4f}")
    print(f"  max|π_myo,i|         : {np.max(np.abs(pi_myo)):.4f}")
    print(f"  sum π_myo            : {np.sum(pi_myo):.4f}")

# =============================================================================
# 1) Problem Parameters
# =============================================================================
N_ASSETS = args.n_assets
M_STATES = args.m_states

# Include epsilon in directory names for parallel runs
exp_name = f"kim_omberg_{N_ASSETS}asset-{M_STATES}state_eps{args.eps:.4f}"
weight_dir = os.path.join(args.weight_dir, exp_name)
output_dir = os.path.join(args.output_dir, exp_name)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print(f"[EPS={args.eps:.4f}] Output dir: {output_dir}")
print(f"[EPS={args.eps:.4f}] Weight dir: {weight_dir}")

# Time domain (τ = remaining horizon = T - t)
tau_max = 3.0
tau_min = 0.0

# Wealth domain
W_min, W_max = 0.1, 2.0

# State domain (will be set based on theta ± some range)
X_RANGE_SCALE = 1.0

# Model parameters
gamma = 2.0
r = 0.03

# Generate market parameters
params = generate_joint_market_params(
    n=N_ASSETS, k=M_STATES,
    seed=SEED,
    sample_alpha=True,
    alpha_dist="dirichlet",
    dirichlet_concentration=1.0,
    alpha_scale=0.25,
)

# Extract parameters
K = params.K
xbar = params.theta
SigmaX = params.SigmaZ
rho = params.rho_Z
Lam = params.alpha

# Derived quantities
Q = SigmaX @ SigmaX.T
Gamma = rho @ SigmaX.T
k0 = K @ xbar

# λ_0: baseline risk premium
lam0 = np.ones(N_ASSETS) * 0.1

# State domain based on long-run mean
eta = params.eta if params.eta is not None else np.diag(SigmaX)
X_min = xbar - X_RANGE_SCALE * eta
X_max = xbar + X_RANGE_SCALE * eta

# Print configuration
print_joint_market_report(params)
print(f"Minimum State domain: X ∈ {X_min}")
print(f"Maximum State domain: X ∈ {X_max}")
print(f"Wealth domain: W ∈ [{W_min}, {W_max}]")
print(f"\nK (mean reversion):\n{np.diag(K)}")
print(f"x̄ (long-run mean): {xbar}")


# =============================================================================
# 2) Closed-form Solution (ODE system)
# =============================================================================
def solve_closed_form_ode(T, gamma, K, xbar, SigmaX, rho, lam0, Lam,
                          method="RK45", t_eval=None):
    M = K.shape[0]
    N = lam0.shape[0]
    Q = SigmaX @ SigmaX.T
    Gamma = rho @ SigmaX.T
    k0 = K @ xbar
    alpha = (1 - gamma) / gamma
    
    def rhs(tau, y):
        a = y[0]
        b = y[1:1+M]
        C = y[1+M:].reshape(M, M)
        C = 0.5 * (C + C.T)
        
        A = Lam + Gamma @ C
        m = lam0 + Gamma @ b
        
        dC = -(K.T @ C + C @ K) + C @ Q @ C + alpha * (A.T @ A)
        db = C @ k0 - K.T @ b + C @ Q @ b + alpha * (A.T @ m)
        da = k0.T @ b + 0.5 * np.trace(Q @ C) + 0.5 * (b.T @ Q @ b) + 0.5 * alpha * (m.T @ m)
        
        return np.concatenate(([da], db, dC.reshape(-1)))
    
    y0 = np.zeros(1 + M + M * M)
    if t_eval is None:
        t_eval = np.linspace(0.0, T, 1001)
    
    sol = solve_ivp(rhs, (0.0, T), y0, t_eval=t_eval, method=method)
    return sol


def get_closed_form_at_tau(tau, sol, M):
    y_tau = np.array([np.interp(tau, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])
    a = y_tau[0]
    b = y_tau[1:1+M]
    C = y_tau[1+M:].reshape(M, M)
    C = 0.5 * (C + C.T)
    return a, b, C


def closed_form_V(tau, w, x, sol, M, gamma, r):
    a, b, C = get_closed_form_at_tau(tau, sol, M)
    phi = np.exp(a + b @ x + 0.5 * x @ C @ x)
    discount = np.exp((1 - gamma) * r * tau)
    U_w = np.power(w, 1 - gamma) / (1 - gamma)
    return discount * U_w * phi


def closed_form_decomposition(tau, w, x, sol, M, N, gamma, lam0, Lam, Gamma):
    a, b, C = get_closed_form_at_tau(tau, sol, M)
    lam_x = lam0 + Lam @ x
    grad_log_phi = b + C @ x
    myopic_norm = lam_x / gamma
    hedging_norm = (Gamma @ grad_log_phi) / gamma
    theta_norm = myopic_norm + hedging_norm
    theta = w * theta_norm
    return theta, theta_norm, myopic_norm, hedging_norm


# Solve ODE once
print("\nSolving closed-form ODE system...")
cf_sol = solve_closed_form_ode(tau_max, gamma, K, xbar, SigmaX, rho, lam0, Lam)
print(f"ODE solved: {len(cf_sol.t)} time points, success={cf_sol.success}")


# =============================================================================
# 3) Neural Network for V(τ, w, x)
# =============================================================================
class ValueNetND(nn.Module):
    def __init__(self, M, hidden=256, depth=4):
        super().__init__()
        self.M = M
        in_dim = M + 2
        
        layers = []
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
    
    def forward(self, w, x, tau):
        inp = torch.cat([w, x, tau], dim=1)
        return self.net(inp)


# =============================================================================
# 4) Sampling Functions
# =============================================================================
def sample_interior(n, device, M, X_min, X_max, W_min, W_max, tau_max):
    eps_tau = 1e-3
    eps_W = 1e-2
    
    tau = eps_tau + torch.rand(n, 1, device=device) * (tau_max - eps_tau)
    w = W_min + eps_W + torch.rand(n, 1, device=device) * (W_max - W_min - 2 * eps_W)
    
    X_min_t = torch.tensor(X_min, device=device, dtype=torch.float32)
    X_max_t = torch.tensor(X_max, device=device, dtype=torch.float32)
    x = X_min_t + torch.rand(n, M, device=device) * (X_max_t - X_min_t)
    
    tau.requires_grad_(True)
    w.requires_grad_(True)
    x.requires_grad_(True)
    
    return w, x, tau


def sample_terminal(n, device, M, X_min, X_max, W_min, W_max):
    eps_W = 1e-2
    
    tau = torch.zeros(n, 1, device=device)
    w = W_min + eps_W + torch.rand(n, 1, device=device) * (W_max - W_min - 2 * eps_W)
    
    X_min_t = torch.tensor(X_min, device=device, dtype=torch.float32)
    X_max_t = torch.tensor(X_max, device=device, dtype=torch.float32)
    x = X_min_t + torch.rand(n, M, device=device) * (X_max_t - X_min_t)
    
    return w, x, tau


def V_terminal(w, gamma):
    return torch.pow(w, 1.0 - gamma) / (1.0 - gamma)


# =============================================================================
# 4B) Risk premium perturbation (tanh)
# =============================================================================
USE_TANH_RISK_PREMIUM = True
EPS_NL = args.eps  # Set from command line argument
XBAR_T = None
D_INV_T = None
LAM_NL_T = None


def risk_premium_torch(x, lam0_t, Lam_t):
    with torch.no_grad():
        lam_lin = lam0_t.unsqueeze(0) + torch.einsum('ij,bj->bi', Lam_t, x)
        if (not USE_TANH_RISK_PREMIUM) or float(EPS_NL) == 0.0:
            return lam_lin

        if (XBAR_T is None) or (D_INV_T is None) or (LAM_NL_T is None):
            raise RuntimeError("Set XBAR_T, D_INV_T, LAM_NL_T in main before training.")

        z = (x - XBAR_T.unsqueeze(0)) * D_INV_T.unsqueeze(0)
        nl = torch.tanh(z)
        lam_nl = torch.einsum('ij,bj->bi', LAM_NL_T, nl)
        return lam_lin + float(EPS_NL) * lam_nl


# =============================================================================
# 5) HJB Residual (Fully Nonlinear PDE)
# =============================================================================
def compute_derivatives_nd(model, w, x, tau, M):
    V = model(w, x, tau)
    ones = torch.ones_like(V)
    
    V_tau = torch.autograd.grad(V, tau, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_x = torch.autograd.grad(V, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w), 
                               create_graph=True, retain_graph=True)[0]
    
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                               create_graph=True, retain_graph=True)[0]
    
    V_xx_rows = []
    for i in range(M):
        V_xi = V_x[:, i:i+1]
        V_xxi = torch.autograd.grad(V_xi, x, grad_outputs=torch.ones_like(V_xi),
                                    create_graph=True, retain_graph=True)[0]
        V_xx_rows.append(V_xxi)
    V_xx = torch.stack(V_xx_rows, dim=1)
    
    return V, V_tau, V_w, V_x, V_ww, V_xx, V_wx


def hjb_residual_nd(model, w, x, tau, M, N, gamma, r, K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t):
    V, V_tau, V_w, V_x, V_ww, V_xx, V_wx = compute_derivatives_nd(model, w, x, tau, M)
    
    batch = w.shape[0]
    
    term1 = -V_tau
    term2 = r * w * V_w
    
    drift = k0_t.unsqueeze(0) - torch.einsum('ij,bj->bi', K_t, x)
    term3 = torch.einsum('bi,bi->b', drift, V_x).unsqueeze(1)
    
    term4 = 0.5 * torch.einsum('ij,bij->b', Q_t, V_xx).unsqueeze(1)
    
    lam_x = risk_premium_torch(x, lam0_t, Lam_t)
    
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)
    
    combined = lam_x * V_w + Gamma_Vwx
    numerator = torch.sum(combined ** 2, dim=1, keepdim=True)
    
    denominator = 2.0 * V_ww
    denominator_safe = torch.where(
        torch.abs(denominator) < 1e-8,
        torch.sign(denominator) * 1e-8 + 1e-10,
        denominator
    )
    term5 = -numerator / denominator_safe
    
    residual = term1 + term2 + term3 + term4 + term5
    
    return residual, V, V_w, V_ww, V_wx


def compute_optimal_theta_nd(model, w, x, tau, M, N, gamma,
                              Gamma_t, lam0_t, Lam_t, create_graph=False):
    V = model(w, x, tau)
    ones = torch.ones_like(V)
    
    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=create_graph, retain_graph=True)[0]
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                               create_graph=create_graph, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w),
                               create_graph=create_graph, retain_graph=True)[0]
    
    lam_x = risk_premium_torch(x, lam0_t, Lam_t)
    
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)
    
    numerator = lam_x * V_w + Gamma_Vwx
    V_ww_safe = torch.where(
        torch.abs(V_ww) < 1e-8,
        torch.sign(V_ww) * 1e-8 + 1e-10,
        V_ww
    )
    theta = -numerator / V_ww_safe
    theta_norm = theta / w
    
    myopic_norm = lam_x / gamma
    hedging_norm = theta_norm - myopic_norm
    
    return V, theta, theta_norm, myopic_norm, hedging_norm


# =============================================================================
# 6) PI-PINN (No Policy Network) for ND Kim-Omberg
# =============================================================================
def linear_pde_residual_nd(
    value_net,
    theta_n,
    w, x, tau,
    M, N,
    gamma, r,
    K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t
):
    V, V_tau, V_w, V_x, V_ww, V_xx, V_wx = compute_derivatives_nd(value_net, w, x, tau, M)

    term1 = -V_tau
    term2 = r * w * V_w

    drift = k0_t.unsqueeze(0) - torch.einsum('ij,bj->bi', K_t, x)
    term3 = torch.einsum('bi,bi->b', drift, V_x).unsqueeze(1)

    term4 = 0.5 * torch.einsum('ij,bij->b', Q_t, V_xx).unsqueeze(1)

    lam_x = risk_premium_torch(x, lam0_t, Lam_t)

    theta_dot_lam = torch.sum(theta_n * lam_x, dim=1, keepdim=True)
    term5 = theta_dot_lam * V_w

    theta_sq = torch.sum(theta_n ** 2, dim=1, keepdim=True)
    term6 = 0.5 * theta_sq * V_ww

    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)
    theta_dot_GammaVwx = torch.sum(theta_n * Gamma_Vwx, dim=1, keepdim=True)
    term7 = theta_dot_GammaVwx

    residual = term1 + term2 + term3 + term4 + term5 + term6 + term7
    return residual, V, V_w, V_ww, V_wx


def compute_theta_from_foc_nd(
    value_net,
    w, x, tau,
    M, N,
    Gamma_t, lam0_t, Lam_t,
    theta_clip_abs=None
):
    V = value_net(w, x, tau)
    ones = torch.ones_like(V)

    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w), create_graph=True, retain_graph=True)[0]
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w), create_graph=True, retain_graph=True)[0]

    lam_x = risk_premium_torch(x, lam0_t, Lam_t)
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)

    numerator = lam_x * V_w + Gamma_Vwx

    V_ww_safe = torch.where(
        torch.abs(V_ww) < 1e-8,
        torch.sign(V_ww) * 1e-8 + 1e-10,
        V_ww
    )
    theta = -numerator / V_ww_safe

    if theta_clip_abs is not None:
        theta = torch.clamp(theta, -float(theta_clip_abs), float(theta_clip_abs))

    return theta, V_w, V_ww


class PIPINN_KimOmbergND:
    def __init__(
        self,
        M, N,
        gamma, r,
        K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t,
        value_hidden=256,
        value_depth=3,
        lr=5e-4,
        scheduler_patience=30,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        theta_clip_abs=5.0,
        device=device
    ):
        self.device = device
        self.M = M
        self.N = N
        self.gamma = gamma
        self.r = r

        self.K_t = K_t
        self.k0_t = k0_t
        self.Q_t = Q_t
        self.Gamma_t = Gamma_t
        self.lam0_t = lam0_t
        self.Lam_t = Lam_t

        self.theta_clip_abs = theta_clip_abs

        self.value_net = ValueNetND(M=M, hidden=value_hidden, depth=value_depth).to(device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=scheduler_factor, patience=scheduler_patience,
            min_lr=scheduler_min_lr, verbose=False
        )
        self.initial_lr = lr

    def initialize_theta(self, w, x, tau, method="myopic"):
        if method == "myopic":
            lam_x = risk_premium_torch(x, self.lam0_t, self.Lam_t)
            theta = (w / self.gamma) * lam_x
        elif method == "zero":
            theta = torch.zeros((w.shape[0], self.N), device=self.device, dtype=torch.float32)
        elif method == "closed_form":
            theta_list = []
            w_np = w.detach().cpu().numpy().reshape(-1)
            x_np = x.detach().cpu().numpy()
            tau_np = tau.detach().cpu().numpy().reshape(-1)
            for i in range(len(w_np)):
                th, _, _, _ = closed_form_decomposition(
                    float(tau_np[i]), float(w_np[i]), x_np[i],
                    cf_sol, self.M, self.N, self.gamma,
                    lam0, Lam, Gamma
                )
                theta_list.append(th)
            theta = torch.tensor(np.array(theta_list), device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown theta init method: {method}")

        if self.theta_clip_abs is not None:
            theta = torch.clamp(theta, -float(self.theta_clip_abs), float(self.theta_clip_abs))
        return theta

    def policy_evaluation(
        self,
        theta_n,
        w_colloc, x_colloc, tau_colloc,
        w_term, x_term, tau_term,
        V_T_target,
        epochs=200,
        w_terminal=10.0,
        w_shape=1.0,
        print_every=200
    ):
        loss_hist = []
        best_loss = float("inf")
        best_state = None

        theta_n_fixed = theta_n.detach()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            w_int = w_colloc.detach().clone().requires_grad_(True)
            x_int = x_colloc.detach().clone().requires_grad_(True)
            tau_int = tau_colloc.detach().clone().requires_grad_(True)

            residual, V, V_w, V_ww, V_wx = linear_pde_residual_nd(
                self.value_net, theta_n_fixed,
                w_int, x_int, tau_int,
                self.M, self.N, self.gamma, self.r,
                self.K_t, self.k0_t, self.Q_t,
                self.Gamma_t, self.lam0_t, self.Lam_t
            )
            pde_loss = torch.mean(residual ** 2)

            V_T_pred = self.value_net(w_term, x_term, tau_term)
            terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)

            mono_penalty = torch.mean(torch.relu(-V_w) ** 2)
            conc_penalty = torch.mean(torch.relu(V_ww) ** 2)

            total_loss = pde_loss + w_terminal * terminal_loss + w_shape * (mono_penalty + conc_penalty)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            cur = float(total_loss.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {k: v.detach().cpu().clone() for k, v in self.value_net.state_dict().items()}

            loss_hist.append({
                "total": cur,
                "pde": float(pde_loss.item()),
                "terminal": float(terminal_loss.item()),
                "mono": float(mono_penalty.item()),
                "conc": float(conc_penalty.item()),
            })

            if epoch % print_every == 0:
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(f"      [Eval {epoch:4d}/{epochs}] Loss={cur:.3e} | PDE={pde_loss.item():.3e} | "
                      f"Term={terminal_loss.item():.3e} | LR={lr_now:.2e}")

        if best_state is not None:
            self.value_net.load_state_dict(best_state)

        return loss_hist, best_loss

    def policy_improvement(self, w, x, tau):
        self.value_net.eval()

        w_e = w.detach().clone().requires_grad_(True)
        x_e = x.detach().clone().requires_grad_(True)
        tau_e = tau.detach().clone().requires_grad_(True)

        theta, V_w, V_ww = compute_theta_from_foc_nd(
            self.value_net,
            w_e, x_e, tau_e,
            self.M, self.N,
            self.Gamma_t, self.lam0_t, self.Lam_t,
            theta_clip_abs=self.theta_clip_abs
        )

        self.value_net.train()
        return theta.detach()

    def run_policy_iteration(
        self,
        outer_iters=50,
        eval_epochs=200,
        batch_size=3000,
        w_terminal=10.0,
        w_shape=1.0,
        theta_init_method="myopic",
        print_every_outer=5,
        print_every_eval=200,
        verbose_detail=False
    ):
        print(f"\n{'='*70}")
        print(f"PI-PINN ND (No Policy Net): N={self.N}, M={self.M}, EPS_NL={EPS_NL:.4f}")
        print(f"  outer_iters   : {outer_iters}")
        print(f"  eval_epochs   : {eval_epochs}")
        print(f"  batch_size    : {batch_size}")
        print(f"  θ init        : {theta_init_method}")
        print(f"  θ clip abs    : {self.theta_clip_abs}")
        print(f"  init LR       : {self.initial_lr:.2e}")
        print(f"{'='*70}\n")

        results = {
            "theta_diff": [],
            "eval_loss": [],
            "lr": [],
            'loss_history': []
        }

        best_eval_loss = float("inf")
        best_iter = 0

        w_colloc, x_colloc, tau_colloc = sample_interior(batch_size, self.device, self.M, X_min, X_max, W_min, W_max, tau_max)
        w_term, x_term, tau_term = sample_terminal(batch_size // 2, self.device, self.M, X_min, X_max, W_min, W_max)
        V_T_target = V_terminal(w_term, self.gamma).detach()

        theta_n = self.initialize_theta(w_colloc, x_colloc, tau_colloc, method=theta_init_method)
        print(f"Initial θ stats: mean={theta_n.mean().item():.4f}, std={theta_n.std().item():.4f}")

        for it in range(1, outer_iters + 1):
            verbose = (it % print_every_outer == 0) or (it <= 3)

            w_colloc, x_colloc, tau_colloc = sample_interior(batch_size, self.device, self.M, X_min, X_max, W_min, W_max, tau_max)
            w_term, x_term, tau_term = sample_terminal(batch_size // 2, self.device, self.M, X_min, X_max, W_min, W_max)
            V_T_target = V_terminal(w_term, self.gamma).detach()

            if it > 1:
                theta_n = self.policy_improvement(w_colloc, x_colloc, tau_colloc)
            else:
                theta_n = self.initialize_theta(w_colloc, x_colloc, tau_colloc, method=theta_init_method)

            eval_hist, inner_best_eval_loss = self.policy_evaluation(
                theta_n=theta_n,
                w_colloc=w_colloc, x_colloc=x_colloc, tau_colloc=tau_colloc,
                w_term=w_term, x_term=x_term, tau_term=tau_term,
                V_T_target=V_T_target,
                epochs=eval_epochs,
                w_terminal=w_terminal,
                w_shape=w_shape,
                print_every=print_every_eval if (verbose and verbose_detail) else (eval_epochs + 1)
            )
            
            results['loss_history'].extend(eval_hist)
            results['eval_loss'].append(inner_best_eval_loss)

            theta_new = self.policy_improvement(w_colloc, x_colloc, tau_colloc)
            theta_diff = torch.mean((theta_new - theta_n) ** 2).item()
            results["theta_diff"].append(theta_diff)

            self.scheduler.step(inner_best_eval_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]
            results["lr"].append(lr_now)

            if inner_best_eval_loss < best_eval_loss:
                best_eval_loss = inner_best_eval_loss
                best_iter = it
                torch.save(
                    self.value_net.state_dict(),
                    os.path.join(weight_dir, f"best_value_net_eps{EPS_NL:.4f}.pt")
                )

            if verbose:
                print(f"[Outer {it:3d}/{outer_iters}] EvalLoss={inner_best_eval_loss:.3e}, "
                      f"θ_diff={theta_diff:.3e}, LR={lr_now:.2e}")

        print(f"\n[Done] Best eval loss = {best_eval_loss:.4e} at iter {best_iter}")
        return results


# =============================================================================
# 7) Evaluation Functions
# =============================================================================
def eval_pinn_on_tau_X_grid(
    model, w_fixed, dimX, x_fixed,
    M, N, gamma,
    Gamma_t, lam0_t, Lam_t,
    X_min, X_max, tau_min, tau_max,
    N_tau=80, N_X=80
):
    model.eval()
    tau_grid = np.linspace(tau_min + 1e-3, tau_max, N_tau)
    X_grid = np.linspace(X_min[dimX], X_max[dimX], N_X)

    V_pinn = np.zeros((N_tau, N_X))
    theta_norm_pinn = np.zeros((N_tau, N_X, N))
    myopic_pinn = np.zeros((N_tau, N_X, N))
    hedging_pinn = np.zeros((N_tau, N_X, N))

    for i, tau_val in enumerate(tau_grid):
        for j, x_val in enumerate(X_grid):
            x_full = x_fixed.copy()
            x_full[dimX] = x_val

            tau_t = torch.tensor([[tau_val]], device=device, dtype=torch.float32)
            w_t = torch.tensor([[w_fixed]], device=device, dtype=torch.float32)
            x_t = torch.tensor([x_full], device=device, dtype=torch.float32)

            w_t.requires_grad_(True)
            x_t.requires_grad_(True)

            V_val, theta_val, theta_norm_val, myopic_norm_val, hedging_norm_val = \
                compute_optimal_theta_nd(
                    model, w_t, x_t, tau_t, M, N, gamma,
                    Gamma_t, lam0_t, Lam_t, create_graph=True
                )

            V_pinn[i, j] = V_val.item()
            theta_norm_pinn[i, j, :] = theta_norm_val.detach().cpu().numpy().flatten()
            myopic_pinn[i, j, :] = myopic_norm_val.detach().cpu().numpy().flatten()
            hedging_pinn[i, j, :] = hedging_norm_val.detach().cpu().numpy().flatten()

    model.train()
    return tau_grid, X_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn


def eval_closed_form_on_tau_X_grid(
    w_fixed, dimX, x_fixed,
    M, N, gamma, r,
    lam0, Lam, Gamma, sol,
    X_min, X_max, tau_min, tau_max,
    N_tau=80, N_X=80
):
    tau_grid = np.linspace(tau_min + 1e-3, tau_max, N_tau)
    X_grid = np.linspace(X_min[dimX], X_max[dimX], N_X)

    V_cf = np.zeros((N_tau, N_X))
    theta_norm_cf = np.zeros((N_tau, N_X, N))
    myopic_cf = np.zeros((N_tau, N_X, N))
    hedging_cf = np.zeros((N_tau, N_X, N))

    for i, tau_val in enumerate(tau_grid):
        for j, x_val in enumerate(X_grid):
            x_full = x_fixed.copy()
            x_full[dimX] = x_val

            V_cf[i, j] = closed_form_V(tau_val, w_fixed, x_full, sol, M, gamma, r)
            theta, theta_norm, myopic_norm, hedging_norm = \
                closed_form_decomposition(tau_val, w_fixed, x_full, sol, M, N, gamma, lam0, Lam, Gamma)

            theta_norm_cf[i, j, :] = theta_norm
            myopic_cf[i, j, :] = myopic_norm
            hedging_cf[i, j, :] = hedging_norm

    return tau_grid, X_grid, V_cf, theta_norm_cf, myopic_cf, hedging_cf


def compute_metrics(V_pinn, V_cf, theta_pinn, theta_cf, myopic_pinn, myopic_cf, hedging_pinn, hedging_cf):
    mse_V = np.mean((V_pinn - V_cf) ** 2)
    rel_rmse_V = np.sqrt(mse_V) / (np.abs(V_cf).mean() + 1e-10)

    mse_theta = np.mean((theta_pinn - theta_cf) ** 2)
    rel_rmse_theta = np.sqrt(mse_theta) / (np.abs(theta_cf).mean() + 1e-10)

    mse_myopic = np.mean((myopic_pinn - myopic_cf) ** 2)
    rel_rmse_myopic = np.sqrt(mse_myopic) / (np.abs(myopic_cf).mean() + 1e-10)

    mse_hedging = np.mean((hedging_pinn - hedging_cf) ** 2)
    rel_rmse_hedging = np.sqrt(mse_hedging) / (np.abs(hedging_cf).mean() + 1e-10)

    return {
        "MSE_V": mse_V,
        "RelRMSE_V": rel_rmse_V,
        "MSE_theta": mse_theta,
        "RelRMSE_theta": rel_rmse_theta,
        "MSE_myopic": mse_myopic,
        "RelRMSE_myopic": rel_rmse_myopic,
        "MSE_hedging": mse_hedging,
        "RelRMSE_hedging": rel_rmse_hedging,
    }


def plot_value_comparison_tauX(tau_grid, X_grid, V_pinn, V_cf, dimX, w_fixed,
                                save_path=None, show=True, xlabel="X[0]", ylabel=r"$\tau$"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    vmin = min(V_pinn.min(), V_cf.min())
    vmax = max(V_pinn.max(), V_cf.max())

    im0 = axes[0].imshow(V_pinn, origin='lower', aspect='auto',
                         extent=[X_grid.min(), X_grid.max(), tau_grid.min(), tau_grid.max()],
                         vmin=vmin, vmax=vmax, cmap='jet')
    axes[0].set_title(f'PI-PINN V (w={w_fixed})')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(V_cf, origin='lower', aspect='auto',
                         extent=[X_grid.min(), X_grid.max(), tau_grid.min(), tau_grid.max()],
                         vmin=vmin, vmax=vmax, cmap='jet')
    axes[1].set_title('Closed-form V')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    plt.colorbar(im1, ax=axes[1])

    diff = V_pinn - V_cf
    max_abs = np.abs(diff).max() + 1e-10
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    im2 = axes[2].imshow(diff, origin='lower', aspect='auto',
                         extent=[X_grid.min(), X_grid.max(), tau_grid.min(), tau_grid.max()],
                         norm=norm, cmap='RdBu_r')
    axes[2].set_title('PI-PINN - CF')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()

def heat_basic_tauX(ax, Z, title, extent, cmap='jet', vmin=None, vmax=None,
                    xlabel="X", ylabel=r"$\tau$"):
    """Basic heatmap for (x-axis=X, y-axis=tau)."""
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def heat_diverging_tauX(ax, Z, title, extent, cmap='RdBu_r', pct=98,
                        xlabel="X", ylabel=r"$\tau$"):
    """Diverging heatmap for (x-axis=X, y-axis=tau), centered at 0."""
    abs_max = np.percentile(np.abs(Z), pct)
    if abs_max < 1e-10:
        abs_max = max(np.abs(Z).max(), 1e-10)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

def plot_portfolio_comparison_tauX(tau_grid, X_grid,
                                   theta_pinn, theta_cf,
                                   myopic_pinn, myopic_cf,
                                   hedging_pinn, hedging_cf,
                                   dimX, w_fixed, N_ASSETS,
                                   save_path=None, show=True, only_hedge=True,
                                   xlabel="X", ylabel=r"$\tau$",
                                   max_assets=10, sort_by_range=True):
    """
    Portfolio comparison on (x-axis=X, y-axis=tau), one figure per asset.
    
    Args:
        max_assets: Maximum number of assets to plot (default: 10)
        sort_by_range: If True, sort assets by range of theta_cf (descending)
                       so that assets with larger variation are plotted first
    """
    extent = [X_grid.min(), X_grid.max(), tau_grid.min(), tau_grid.max()]
    xlab = xlabel if xlabel is not None else f'x[{dimX}]'
    ylab = ylabel

    # Compute range for each asset based on closed-form theta
    asset_ranges = []
    for asset_idx in range(N_ASSETS):
        theta_c = theta_cf[:, :, asset_idx]
        range_val = theta_c.max() - theta_c.min()
        asset_ranges.append((asset_idx, range_val))
    
    # Sort by range (descending) if requested
    if sort_by_range:
        asset_ranges.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to max_assets
    assets_to_plot = [idx for idx, _ in asset_ranges[:max_assets]]
    
    print(f"\n[Portfolio Plot] Plotting {len(assets_to_plot)} assets (sorted by range: {sort_by_range})")
    print(f"  Asset order: {assets_to_plot}")
    print(f"  Top ranges: {[f'{idx}:{r:.4f}' for idx, r in asset_ranges[:max_assets]]}")

    if only_hedge == False:
        for rank, asset_idx in enumerate(assets_to_plot):
            fig, axs = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    
            theta_p = theta_pinn[:, :, asset_idx]
            theta_c = theta_cf[:, :, asset_idx]
            myopic_p = myopic_pinn[:, :, asset_idx]
            myopic_c = myopic_cf[:, :, asset_idx]
            hedging_p = hedging_pinn[:, :, asset_idx]
            hedging_c = hedging_cf[:, :, asset_idx]
    
            # Row 0: Total
            vmin_t = min(theta_p.min(), theta_c.min())
            vmax_t = max(theta_p.max(), theta_c.max())
            heat_basic_tauX(axs[0, 0], theta_p, f'θ*[{asset_idx}]/w (PINN-PI)', extent,
                            vmin=vmin_t, vmax=vmax_t, xlabel=xlab, ylabel=ylab)
            heat_basic_tauX(axs[0, 1], theta_c, f'θ*[{asset_idx}]/w (Closed-form)', extent,
                            vmin=vmin_t, vmax=vmax_t, xlabel=xlab, ylabel=ylab)
            heat_diverging_tauX(axs[0, 2], theta_p - theta_c, f'θ*[{asset_idx}]/w Diff', extent,
                                xlabel=xlab, ylabel=ylab)
    
            # Row 1: Myopic
            vmin_m = min(myopic_p.min(), myopic_c.min())
            vmax_m = max(myopic_p.max(), myopic_c.max())
            heat_basic_tauX(axs[1, 0], myopic_p, f'Myopic[{asset_idx}] (PINN-PI)', extent,
                            vmin=vmin_m, vmax=vmax_m, xlabel=xlab, ylabel=ylab)
            heat_basic_tauX(axs[1, 1], myopic_c, f'Myopic[{asset_idx}] (Closed-form)', extent,
                            vmin=vmin_m, vmax=vmax_m, xlabel=xlab, ylabel=ylab)
            heat_diverging_tauX(axs[1, 2], myopic_p - myopic_c, f'Myopic[{asset_idx}] Diff', extent,
                                xlabel=xlab, ylabel=ylab)
    
            # Row 2: Hedging
            vmin_h = min(hedging_p.min(), hedging_c.min())
            vmax_h = max(hedging_p.max(), hedging_c.max())
            heat_basic_tauX(axs[2, 0], hedging_p, f'Hedging[{asset_idx}] (PINN-PI)', extent,
                            vmin=vmin_h, vmax=vmax_h, xlabel=xlab, ylabel=ylab)
            heat_basic_tauX(axs[2, 1], hedging_c, f'Hedging[{asset_idx}] (Closed-form)', extent,
                            vmin=vmin_h, vmax=vmax_h, xlabel=xlab, ylabel=ylab)
            heat_diverging_tauX(axs[2, 2], hedging_p - hedging_c, f'Hedging[{asset_idx}] Diff', extent,
                                xlabel=xlab, ylabel=ylab)

            plt.suptitle(f'Kim-Omberg ND: Asset {asset_idx} Portfolio [Rank {rank+1}/{len(assets_to_plot)}] (w={w_fixed:.2f})', fontsize=14)

            if save_path:
                path = save_path.replace('.png', f'_rank{rank+1:02d}_asset{asset_idx}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")

            if show:
                plt.show()
            else:
                plt.close()

    else:
        for rank, asset_idx in enumerate(assets_to_plot):
            fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

            hedging_p = hedging_pinn[:, :, asset_idx]
            hedging_c = hedging_cf[:, :, asset_idx]

            # Row: Hedging only
            vmin_h = min(hedging_p.min(), hedging_c.min())
            vmax_h = max(hedging_p.max(), hedging_c.max())
            heat_basic_tauX(axs[0], hedging_p, f'Hedging[{asset_idx}] (PINN-PI)', extent,
                            vmin=vmin_h, vmax=vmax_h, xlabel=xlab, ylabel=ylab)
            heat_basic_tauX(axs[1], hedging_c, f'Hedging[{asset_idx}] (Closed-form)', extent,
                            vmin=vmin_h, vmax=vmax_h, xlabel=xlab, ylabel=ylab)
            heat_diverging_tauX(axs[2], hedging_p - hedging_c, f'Hedging[{asset_idx}] Diff', extent,
                                xlabel=xlab, ylabel=ylab)

            plt.suptitle(f'Kim-Omberg ND: Asset {asset_idx} Hedging [Rank {rank+1}/{len(assets_to_plot)}] (w={w_fixed:.2f})', fontsize=14)

            if save_path:
                path = save_path.replace('.png', f'_rank{rank+1:02d}_asset{asset_idx}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")

            if show:
                plt.show()
            else:
                plt.close()


# =============================================================================
# 8) Hyperparameters
# =============================================================================
value_hidden = 256
value_depth = 3
lr = 5e-4
outer_iters = args.outer_iters
eval_epochs = args.eval_epochs
batch_size = args.batch_size
w_terminal = 20.0
w_shape = 1.0

theta_init_method = "zero"
theta_clip_abs = 3.0

print_every_outer = 20
print_every_eval = 200
verbose_detail = False


# =============================================================================
# 9) Main Execution
# =============================================================================
if __name__ == "__main__":

    # Torch tensors
    K_t = torch.tensor(K, device=device, dtype=torch.float32)
    k0_t = torch.tensor(k0, device=device, dtype=torch.float32)
    Q_t = torch.tensor(Q, device=device, dtype=torch.float32)
    Gamma_t = torch.tensor(Gamma, device=device, dtype=torch.float32)
    lam0_t = torch.tensor(lam0, device=device, dtype=torch.float32)
    Lam_t = torch.tensor(Lam, device=device, dtype=torch.float32)

    # tanh risk premium configuration
    xbar_t = torch.tensor(xbar, device=device, dtype=torch.float32)
    eta_t = torch.tensor(eta, device=device, dtype=torch.float32)
    D_inv_t = 1.0 / (eta_t + 1e-8)

    LAM_NL_SCALE = 1.0
    Lam_nl_t = LAM_NL_SCALE * Lam_t

    # Set globals
    XBAR_T = xbar_t
    D_INV_T = D_inv_t
    LAM_NL_T = Lam_nl_t

    print("\n" + "="*80)
    print(f"[Single Epsilon Run] EPS_NL = {EPS_NL:.4f}")
    print("="*80)

    # Create solver
    solver = PIPINN_KimOmbergND(
        M=M_STATES, N=N_ASSETS,
        gamma=gamma, r=r,
        K_t=K_t, k0_t=k0_t, Q_t=Q_t,
        Gamma_t=Gamma_t, lam0_t=lam0_t, Lam_t=Lam_t,
        value_hidden=value_hidden,
        value_depth=value_depth,
        lr=lr,
        scheduler_patience=30,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        theta_clip_abs=theta_clip_abs,
        device=device
    )

    # Run training
    results = solver.run_policy_iteration(
        outer_iters=outer_iters,
        eval_epochs=eval_epochs,
        batch_size=batch_size,
        w_terminal=w_terminal,
        w_shape=w_shape,
        theta_init_method=theta_init_method,
        print_every_outer=print_every_outer,
        print_every_eval=print_every_eval,
        verbose_detail=verbose_detail
    )

    model = solver.value_net
    model.load_state_dict(torch.load(os.path.join(weight_dir, f"best_value_net_eps{EPS_NL:.4f}.pt"), map_location=device))

    # Evaluation vs closed-form
    dimX = 0
    x_fixed = xbar.copy()
    W_levels = [0.5]
    N_tau, N_X = 100, 100

    for w_test in W_levels:
        tau_grid, X_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn = \
            eval_pinn_on_tau_X_grid(
                model, w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma,
                Gamma_t=Gamma_t, lam0_t=lam0_t, Lam_t=Lam_t,
                X_min=X_min, X_max=X_max, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )

        _, _, V_cf, theta_norm_cf, myopic_cf, hedging_cf = \
            eval_closed_form_on_tau_X_grid(
                w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma, r=r,
                lam0=lam0, Lam=Lam, Gamma=Gamma, sol=cf_sol,
                X_min=X_min, X_max=X_max, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )

        metrics = compute_metrics(
            V_pinn, V_cf,
            theta_norm_pinn, theta_norm_cf,
            myopic_pinn, myopic_cf,
            hedging_pinn, hedging_cf
        )

        print(f"\n[EPS={EPS_NL:.2f}] Metrics vs closed-form (baseline):")
        for k, v in metrics.items():
            if k.startswith('MSE_') or k.startswith('RelRMSE_'):
                print(f"  {k:14s}: {v:.6e}")

        # Save metrics to file
        metrics_path = os.path.join(output_dir, f"metrics_eps{EPS_NL:.2f}.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"EPS_NL={EPS_NL:.2f}\n")
            for k, v in metrics.items():
                f.write(f"{k}={v:.10e}\n")
        print(f"Saved metrics to: {metrics_path}")

        # Plot
        plot_value_comparison_tauX(
            tau_grid, X_grid, V_pinn, V_cf,
            dimX=dimX, w_fixed=w_test,
            save_path=os.path.join(output_dir, f"value_tauX_eps({EPS_NL:.2f})_w({w_test:.2f}).png"),
            show=False,
            xlabel=f"Risk premium X", ylabel=r"$\tau$"
        )

        plot_portfolio_comparison_tauX(
        tau_grid, X_grid,
        theta_norm_pinn, theta_norm_cf,
        myopic_pinn, myopic_cf,
        hedging_pinn, hedging_cf,
        dimX=dimX, w_fixed=w_test, N_ASSETS=N_ASSETS,
        save_path=os.path.join(output_dir, f"portfolio_tauX_eps({EPS_NL:.2f})_w({w_test:.2f}).png"),
        show=False, only_hedge=False,
        xlabel=f"risk premium X",
        ylabel=r"$\tau$",
        max_assets = 10, sort_by_range=True
        )

    print(f"\n[EPS={EPS_NL:.4f}] All done!")
