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
"""
import time
import os
import sys
import math
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Add path for joint_market_setup
sys.path.insert(0, '/mnt/user-data/uploads')
from joint_market_setup_dirichlet import generate_joint_market_params, JointMarketParams, cholesky_solve


# =============================================================================
# 0) Reproducibility + Device
# =============================================================================
SEED = 12
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device=device)

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

    # ---- "Merton-style" quantities at a reference state ----
    # KO/다요인에서는 mu_excess가 상태에 따라 변하므로, Y_ref에서 myopic을 찍는 게 자연스러움
    if k == 0 or params.alpha is None:
        print("\n[Myopic (reference-state) quantities] skipped (k==0 or alpha not sampled).")
        print("  - If you want Merton-style mu_excess+pi*, either:")
        print("    (i) reuse generate_synthetic_merton_market, or")
        print("    (ii) pick a constant mu_excess and solve pi* with Sigma_RR_safe.")
        return

    if Y_ref is None:
        Y_ref = params.theta  # 기본: 장기평균 상태에서 진단

    Y_ref = np.asarray(Y_ref, dtype=float).reshape(k,)
    risk = params.alpha @ Y_ref              # (n,)
    mu_excess = params.sigma * risk          # (n,)  (너 코드의 mu-r 구조와 동일)
    Sigma = params.Sigma_RR_safe

    # Σ^{-1} mu는 inverse 만들지 말고 cholesky_solve로
    Sigma_inv_mu = cholesky_solve(params.chol_Sigma_RR_safe, mu_excess)
    Theta = float(mu_excess @ Sigma_inv_mu)  # mu^T Σ^{-1} mu

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
# Dimensions
N_ASSETS = 10    # Number of risky assets
M_STATES = 1    # Number of state variables

weight_dir = f"weights/pinn/kim_omberg_{N_ASSETS}asset-{M_STATES}state"
os.makedirs(weight_dir, exist_ok=True)
output_dir = f"outputs/pinn/kim_omberg_{N_ASSETS}asset-{M_STATES}state"
os.makedirs(output_dir, exist_ok=True)

# Time domain (τ = remaining horizon = T - t)
tau_max = 3.0
tau_min = 0.0

# Wealth domain
W_min, W_max = 0.1, 2.0

# State domain (will be set based on theta ± some range)
X_RANGE_SCALE = 1.0  # x ∈ [θ - scale*η, θ + scale*η] roughly

# Model parameters
gamma = 2.0     # CRRA risk aversion
r = 0.03        # risk-free rate

# Generate market parameters
params = generate_joint_market_params(
    n=N_ASSETS, k=M_STATES,
    seed=SEED,
    sample_alpha=True,
    alpha_dist="dirichlet",
    dirichlet_concentration=1.0,  # 보통 1.0이 무난 (균등한 Dirichlet)
    alpha_scale=0.25,              # row-sum이 1이 되도록
)

# Extract parameters
K = params.K                    # (M, M) mean reversion
xbar = params.theta             # (M,) long-run mean
SigmaX = params.SigmaZ          # (M, M) state diffusion
rho = params.rho_Z              # (N, M) asset-state correlation
Lam = params.alpha              # (N, M) risk premium loading

# Derived quantities
Q = SigmaX @ SigmaX.T           # (M, M) state covariance
Gamma = rho @ SigmaX.T          # (N, M) cross term
k0 = K @ xbar                   # (M,) drift constant

# λ_0: baseline risk premium (set to small positive values)
lam0 = np.ones(N_ASSETS) * 0.1

# State domain based on long-run mean
eta = params.eta if params.eta is not None else np.diag(SigmaX)
X_min = xbar - X_RANGE_SCALE * eta
X_max = xbar + X_RANGE_SCALE * eta

# Print configuration
print_joint_market_report(params)
# print(f"\n{'='*60}")
# print("Multi-dimensional Kim-Omberg PINN")
# print(f"{'='*60}")
# print(f"Dimensions: N={N_ASSETS} assets, M={M_STATES} states")
# print(f"Parameters: γ={gamma}, r={r}, T={tau_max}")
print(f"Minimum State domain: X ∈ {X_min}")
print(f"Maxmimum State domain: X ∈ {X_max}")
print(f"Minimum Wealth domain: W ∈ [{W_min}, {W_max}]")
print(f"\nK (mean reversion):\n{np.diag(K)}")
print(f"x̄ (long-run mean): {xbar}")
# print(f"Σ_X (state diffusion):\n{SigmaX}")
# print(f"ρ (correlation):\n{rho}")
# print(f"Λ (loading):\n{Lam}")
# print(f"λ_0 (baseline): {lam0}")
# print(f"Γ = ρΣ_X^⊤:\n{Gamma}")
# print(f"Q = Σ_X Σ_X^⊤:\n{Q}")


# =============================================================================
# 2) Closed-form Solution (ODE system)
# =============================================================================
def solve_closed_form_ode(T, gamma, K, xbar, SigmaX, rho, lam0, Lam,
                          method="RK45", t_eval=None):
    """
    Solve normal-solution ODEs for:
      φ(t,x) = exp(a(τ) + b(τ)^T x + 0.5 x^T C(τ) x),  τ = T - t.
    
    Returns solve_ivp result with state y = [a, b(0..M-1), vec(C)]
    """
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
        C = 0.5 * (C + C.T)  # enforce symmetry
        
        A = Lam + Gamma @ C               # (N, M)
        m = lam0 + Gamma @ b              # (N,)
        
        # Riccati for C
        dC = -(K.T @ C + C @ K) + C @ Q @ C + alpha * (A.T @ A)
        
        # Linear ODE for b
        db = C @ k0 - K.T @ b + C @ Q @ b + alpha * (A.T @ m)
        
        # Scalar ODE for a
        da = k0.T @ b + 0.5 * np.trace(Q @ C) + 0.5 * (b.T @ Q @ b) + 0.5 * alpha * (m.T @ m)
        
        return np.concatenate(([da], db, dC.reshape(-1)))
    
    y0 = np.zeros(1 + M + M * M)
    if t_eval is None:
        t_eval = np.linspace(0.0, T, 1001)
    
    sol = solve_ivp(rhs, (0.0, T), y0, t_eval=t_eval, method=method)
    return sol


def get_closed_form_at_tau(tau, sol, M):
    """Interpolate a, b, C from ODE solution at given tau."""
    y_tau = np.array([np.interp(tau, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])
    a = y_tau[0]
    b = y_tau[1:1+M]
    C = y_tau[1+M:].reshape(M, M)
    C = 0.5 * (C + C.T)
    return a, b, C


def closed_form_V(tau, w, x, sol, M, gamma, r):
    """
    V(τ, w, x) = e^{(1-γ)rτ} · (w^{1-γ}/(1-γ)) · φ(τ, x)
    where φ = exp(a + b^T x + 0.5 x^T C x)
    """
    a, b, C = get_closed_form_at_tau(tau, sol, M)
    phi = np.exp(a + b @ x + 0.5 * x @ C @ x)
    discount = np.exp((1 - gamma) * r * tau)
    U_w = np.power(w, 1 - gamma) / (1 - gamma)
    return discount * U_w * phi


def closed_form_decomposition(tau, w, x, sol, M, N, gamma, lam0, Lam, Gamma):
    """
    Decompose θ* into myopic and hedging components.
    
    θ*/w = myopic + hedging
    myopic = λ(x)/γ = (λ_0 + Λx)/γ
    hedging = Γ∇_x log φ / γ = Γ(b + Cx)/γ
    
    Returns:
        theta: (N,) total optimal portfolio
        theta_norm: (N,) θ*/w
        myopic_norm: (N,) λ(x)/γ
        hedging_norm: (N,) Γ(b + Cx)/γ
    """
    a, b, C = get_closed_form_at_tau(tau, sol, M)
    
    # λ(x) = λ_0 + Λx
    lam_x = lam0 + Lam @ x  # (N,)
    
    # ∇_x log φ = b + Cx
    grad_log_phi = b + C @ x  # (M,)
    
    # Myopic: λ(x)/γ
    myopic_norm = lam_x / gamma  # (N,)
    
    # Hedging: Γ(b + Cx)/γ
    hedging_norm = (Gamma @ grad_log_phi) / gamma  # (N,)
    
    # Total
    theta_norm = myopic_norm + hedging_norm  # (N,)
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
    """
    Neural network for V(τ, w, x) in N-dimensional state space.
    
    Input: (w, x_1, ..., x_M, τ) ∈ ℝ^{M+2}
    Output: V ∈ ℝ
    """
    def __init__(self, M, hidden=256, depth=4):
        super().__init__()
        self.M = M
        in_dim = M + 2  # (w, x_1, ..., x_M, τ)
        
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, w, x, tau):
        """
        Forward pass.
        
        Args:
            w: (batch, 1) wealth
            x: (batch, M) state vector
            tau: (batch, 1) remaining time
        
        Returns:
            V: (batch, 1) value function
        """
        inp = torch.cat([w, x, tau], dim=1)  # (batch, M+2)
        return self.net(inp)


# =============================================================================
# 4) Sampling Functions
# =============================================================================
def sample_interior(n, device, M, X_min, X_max, W_min, W_max, tau_max):
    """Sample interior points (τ > 0)."""
    eps_tau = 1e-3
    eps_W = 1e-2
    
    tau = eps_tau + torch.rand(n, 1, device=device) * (tau_max - eps_tau)
    w = W_min + eps_W + torch.rand(n, 1, device=device) * (W_max - W_min - 2 * eps_W)
    
    # Sample x in [X_min, X_max] for each dimension
    X_min_t = torch.tensor(X_min, device=device, dtype=torch.float32)
    X_max_t = torch.tensor(X_max, device=device, dtype=torch.float32)
    x = X_min_t + torch.rand(n, M, device=device) * (X_max_t - X_min_t)
    
    tau.requires_grad_(True)
    w.requires_grad_(True)
    x.requires_grad_(True)
    
    return w, x, tau


def sample_terminal(n, device, M, X_min, X_max, W_min, W_max):
    """Sample terminal boundary points (τ = 0)."""
    eps_W = 1e-2
    
    tau = torch.zeros(n, 1, device=device)
    w = W_min + eps_W + torch.rand(n, 1, device=device) * (W_max - W_min - 2 * eps_W)
    
    X_min_t = torch.tensor(X_min, device=device, dtype=torch.float32)
    X_max_t = torch.tensor(X_max, device=device, dtype=torch.float32)
    x = X_min_t + torch.rand(n, M, device=device) * (X_max_t - X_min_t)
    
    return w, x, tau


def V_terminal(w, gamma):
    """Terminal condition: V(0, w, x) = U(w) = w^{1-γ}/(1-γ)"""
    return torch.pow(w, 1.0 - gamma) / (1.0 - gamma)


# =============================================================================
# 5) HJB Residual (Fully Nonlinear PDE)
# =============================================================================
def compute_derivatives_nd(model, w, x, tau, M):
    """
    Compute V and its partial derivatives for M-dimensional state.
    
    Returns:
        V: (batch, 1)
        V_tau: (batch, 1)
        V_w: (batch, 1)
        V_x: (batch, M)
        V_ww: (batch, 1)
        V_xx: (batch, M, M)
        V_wx: (batch, M)
    """
    V = model(w, x, tau)
    ones = torch.ones_like(V)
    
    # First derivatives
    V_tau = torch.autograd.grad(V, tau, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_x = torch.autograd.grad(V, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]  # (batch, M)
    
    # Second derivatives
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w), 
                               create_graph=True, retain_graph=True)[0]
    
    # V_wx: derivative of V_w w.r.t. x → (batch, M)
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                               create_graph=True, retain_graph=True)[0]
    
    # V_xx: Hessian of V w.r.t. x → need (batch, M, M)
    # Compute row by row
    V_xx_rows = []
    for i in range(M):
        V_xi = V_x[:, i:i+1]  # (batch, 1)
        V_xxi = torch.autograd.grad(V_xi, x, grad_outputs=torch.ones_like(V_xi),
                                    create_graph=True, retain_graph=True)[0]  # (batch, M)
        V_xx_rows.append(V_xxi)
    V_xx = torch.stack(V_xx_rows, dim=1)  # (batch, M, M)
    
    return V, V_tau, V_w, V_x, V_ww, V_xx, V_wx


def hjb_residual_nd(model, w, x, tau, M, N, gamma, r, K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t):
    """
    Compute HJB PDE residual for multi-dimensional Kim-Omberg problem.
    
    Fully nonlinear PDE (in τ = T - t):
        0 = -V_τ + rwV_w + (k_0 - Kx)^⊤ V_x + ½tr(QV_xx) - ‖λ(x)V_w + ΓV_wx‖² / (2V_ww)
    
    Args:
        model: neural network
        w: (batch, 1) wealth
        x: (batch, M) state
        tau: (batch, 1) remaining time
        M: number of states
        N: number of assets
        gamma: risk aversion
        r: risk-free rate
        K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t: torch tensors of parameters
    
    Returns:
        residual: (batch, 1)
        V, V_w, V_ww, V_wx for diagnostics
    """
    V, V_tau, V_w, V_x, V_ww, V_xx, V_wx = compute_derivatives_nd(model, w, x, tau, M)
    
    batch = w.shape[0]
    
    # Term 1: -V_τ
    term1 = -V_tau
    
    # Term 2: rwV_w
    term2 = r * w * V_w
    
    # Term 3: (k_0 - Kx)^⊤ V_x = k_0^⊤ V_x - x^⊤ K^⊤ V_x
    # drift = k_0 - K @ x  → (batch, M)
    drift = k0_t.unsqueeze(0) - torch.einsum('ij,bj->bi', K_t, x)  # (batch, M)
    term3 = torch.einsum('bi,bi->b', drift, V_x).unsqueeze(1)  # (batch, 1)
    
    # Term 4: ½ tr(Q V_xx)
    # tr(Q V_xx) = sum_{i,j} Q_ij (V_xx)_ij
    term4 = 0.5 * torch.einsum('ij,bij->b', Q_t, V_xx).unsqueeze(1)  # (batch, 1)
    
    # Term 5: -‖λ(x)V_w + Γ V_wx‖² / (2V_ww)
    # λ(x) = λ_0 + Λx → (batch, N)
    lam_x = lam0_t.unsqueeze(0) + torch.einsum('ij,bj->bi', Lam_t, x)  # (batch, N)
    
    # Γ V_wx → (batch, N)
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)  # (batch, N)
    
    # numerator = ‖λ(x) V_w + Γ V_wx‖²
    combined = lam_x * V_w + Gamma_Vwx  # (batch, N)
    numerator = torch.sum(combined ** 2, dim=1, keepdim=True)  # (batch, 1)
    
    # denominator = 2 V_ww (should be negative for concave V)
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
    """
    Compute optimal portfolio with myopic/hedging decomposition.
    
    θ*(τ, w, x) = -(λ(x)V_w + Γ V_wx) / V_ww
    θ*/w = myopic + hedging
    myopic = λ(x)/γ (closed-form, independent of V)
    hedging = θ*/w - myopic (PINN-derived)
    
    Returns:
        V: (batch, 1)
        theta: (batch, N) optimal portfolio
        theta_norm: (batch, N) θ*/w
        myopic_norm: (batch, N) λ(x)/γ
        hedging_norm: (batch, N) θ*/w - λ(x)/γ
    """
    V = model(w, x, tau)
    ones = torch.ones_like(V)
    
    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=create_graph, retain_graph=True)[0]
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w),
                               create_graph=create_graph, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w),
                               create_graph=create_graph, retain_graph=True)[0]
    
    # λ(x) = λ_0 + Λx
    lam_x = lam0_t.unsqueeze(0) + torch.einsum('ij,bj->bi', Lam_t, x)  # (batch, N)
    
    # Γ V_wx
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)  # (batch, N)
    
    # θ* = -(λ(x)V_w + Γ V_wx) / V_ww
    numerator = lam_x * V_w + Gamma_Vwx
    V_ww_safe = torch.where(
        torch.abs(V_ww) < 1e-8,
        torch.sign(V_ww) * 1e-8 + 1e-10,
        V_ww
    )
    theta = -numerator / V_ww_safe  # (batch, N)
    theta_norm = theta / w  # (batch, N)
    
    # Myopic (closed-form, independent of V)
    # myopic_norm = lam_x / gamma  # (batch, N)

    # Myopic (from network)
    pinn_coeff = -V_w / (w * V_ww_safe)  # (batch, 1)
    myopic_norm = pinn_coeff * lam_x  # (batch, N)
    
    # Hedging = Total - Myopic (PINN-derived)
    hedging_norm = theta_norm - myopic_norm  # (batch, N)
    
    return V, theta, theta_norm, myopic_norm, hedging_norm


# =============================================================================
# 6) Training
# =============================================================================
def train_pinn_nd(model, M, N, gamma, r, 
                  K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t,
                  X_min, X_max, W_min, W_max, tau_max,
                  epochs=50000, batch_size=2000, lr=5e-4,
                  resample_every=200, w_terminal=10.0, w_concavity=1.0,
                  print_every=2000):
    """Train PINN for multi-dimensional Kim-Omberg HJB."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000, min_lr=1e-8
    )
    
    loss_history = []
    best_loss = float('inf')
    
    # Initial sampling
    w_int, x_int, tau_int = sample_interior(batch_size, device, M, X_min, X_max, W_min, W_max, tau_max)
    w_term, x_term, tau_term = sample_terminal(batch_size, device, M, X_min, X_max, W_min, W_max)
    V_T_target = V_terminal(w_term, gamma).detach()
    
    print(f"\n{'='*60}")
    print(f"Training {M+2}D PINN (N={N} assets, M={M} states)")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        if epoch % resample_every == 0:
            w_int, x_int, tau_int = sample_interior(batch_size, device, M, X_min, X_max, W_min, W_max, tau_max)
            w_term, x_term, tau_term = sample_terminal(batch_size, device, M, X_min, X_max, W_min, W_max)
            V_T_target = V_terminal(w_term, gamma).detach()
        
        optimizer.zero_grad()
        
        # PDE loss
        residual, V, V_w, V_ww, V_wx = hjb_residual_nd(
            model, w_int, x_int, tau_int, M, N, gamma, r,
            K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t
        )
        pde_loss = torch.mean(residual ** 2)
        
        # Terminal condition loss
        V_T_pred = model(w_term, x_term, tau_term)
        terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)
        
        # Concavity loss: V_ww < 0
        concavity_violation = torch.relu(V_ww)
        concavity_loss = torch.mean(concavity_violation ** 2)
        
        # Total loss
        total_loss = pde_loss + w_terminal * terminal_loss + w_concavity * concavity_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss.detach().cpu())
        
        # Clear gradients
        for var in [w_int, x_int, tau_int]:
            if var.grad is not None:
                var.grad = None
        
        # Track best
        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), os.path.join(weight_dir, f"value_net_best_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).pt"))
        
        loss_history.append({
            'total': current_loss,
            'pde': pde_loss.item(),
            'terminal': terminal_loss.item(),
            'concavity': concavity_loss.item()
        })
        
        if epoch % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch:6d}/{epochs}] Total: {current_loss:.3e} | "
                  f"PDE: {pde_loss.item():.3e} | Term: {terminal_loss.item():.3e} | "
                  f"Conc: {concavity_loss.item():.3e} | LR: {current_lr:.2e}")
    
    # Restore best model
    model.load_state_dict(torch.load(os.path.join(weight_dir, f"value_net_best_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).pt"), map_location=device))
    print(f"\nRestored best model (loss: {best_loss:.3e})")
    
    return loss_history, optimizer


# =============================================================================
# 7) Evaluation
# =============================================================================
def eval_pinn_on_grid_2d_slice(model, tau_fixed, w_fixed, dim1, dim2, x_fixed, 
                                M, N, gamma, Gamma_t, lam0_t, Lam_t, X_min, X_max,
                                N_grid=50, chunk=2000):
    """
    Evaluate PINN on a 2D slice: varying x[dim1] and x[dim2], fixing others.
    
    Returns:
        x1_grid, x2_grid: meshgrid arrays (N_grid, N_grid)
        V_pinn: (N_grid, N_grid)
        theta_norm_pinn: (N_grid, N_grid, N)
        myopic_pinn: (N_grid, N_grid, N)
        hedging_pinn: (N_grid, N_grid, N)
    """
    model.eval()
    
    x1_vals = np.linspace(X_min[dim1], X_max[dim1], N_grid)
    x2_vals = np.linspace(X_min[dim2], X_max[dim2], N_grid)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals, indexing="ij")
    
    # Build full x array
    n_points = N_grid * N_grid
    x_full = np.tile(x_fixed, (n_points, 1))  # (n_points, M)
    x_full[:, dim1] = x1_grid.flatten()
    x_full[:, dim2] = x2_grid.flatten()
    
    w_flat = torch.full((n_points, 1), w_fixed, device=device, dtype=torch.float32, requires_grad=True)
    x_flat = torch.tensor(x_full, device=device, dtype=torch.float32, requires_grad=True)
    tau_flat = torch.full((n_points, 1), tau_fixed, device=device, dtype=torch.float32)
    
    V_list, theta_norm_list, myopic_list, hedging_list = [], [], [], []
    
    for i in range(0, n_points, chunk):
        w_b = w_flat[i:i+chunk]
        x_b = x_flat[i:i+chunk]
        tau_b = tau_flat[i:i+chunk]
        
        V_b, _, theta_norm_b, myopic_b, hedging_b = compute_optimal_theta_nd(
            model, w_b, x_b, tau_b, M, N, gamma, Gamma_t, lam0_t, Lam_t, create_graph=True
        )
        V_list.append(V_b.detach().cpu())
        theta_norm_list.append(theta_norm_b.detach().cpu())
        myopic_list.append(myopic_b.detach().cpu())
        hedging_list.append(hedging_b.detach().cpu())
    
    V_pinn = torch.cat(V_list, dim=0).numpy().reshape(N_grid, N_grid)
    theta_norm_pinn = torch.cat(theta_norm_list, dim=0).numpy().reshape(N_grid, N_grid, N)
    myopic_pinn = torch.cat(myopic_list, dim=0).numpy().reshape(N_grid, N_grid, N)
    hedging_pinn = torch.cat(hedging_list, dim=0).numpy().reshape(N_grid, N_grid, N)
    
    model.train()
    return x1_grid, x2_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn


def eval_closed_form_on_grid_2d_slice(tau_fixed, w_fixed, dim1, dim2, x_fixed,
                                       M, N, gamma, r, lam0, Lam, Gamma, sol,
                                       X_min, X_max, N_grid=100):
    """
    Evaluate closed-form on same 2D slice.
    
    Returns:
        x1_grid, x2_grid: meshgrid arrays
        V_cf: (N_grid, N_grid)
        theta_norm_cf: (N_grid, N_grid, N)
        myopic_cf: (N_grid, N_grid, N)
        hedging_cf: (N_grid, N_grid, N)
    """
    x1_vals = np.linspace(X_min[dim1], X_max[dim1], N_grid)
    x2_vals = np.linspace(X_min[dim2], X_max[dim2], N_grid)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals, indexing="ij")
    
    V_cf = np.zeros((N_grid, N_grid))
    theta_norm_cf = np.zeros((N_grid, N_grid, N))
    myopic_cf = np.zeros((N_grid, N_grid, N))
    hedging_cf = np.zeros((N_grid, N_grid, N))
    
    for i in range(N_grid):
        for j in range(N_grid):
            x = x_fixed.copy()
            x[dim1] = x1_grid[i, j]
            x[dim2] = x2_grid[i, j]
            
            V_cf[i, j] = closed_form_V(tau_fixed, w_fixed, x, sol, M, gamma, r)
            _, theta_norm_cf[i, j, :], myopic_cf[i, j, :], hedging_cf[i, j, :] = \
                closed_form_decomposition(tau_fixed, w_fixed, x, sol, M, N, gamma, lam0, Lam, Gamma)
    
    return x1_grid, x2_grid, V_cf, theta_norm_cf, myopic_cf, hedging_cf


def compute_metrics(V_pinn, V_cf, theta_pinn, theta_cf,
                    myopic_pinn, myopic_cf, hedging_pinn, hedging_cf):
    """
    Compute MSE and RelRMSE for V, θ, myopic, hedging.
    
    Returns dict with metrics for total and per-asset.
    """
    metrics = {}
    
    # Value function
    mse_V = np.mean((V_pinn - V_cf) ** 2)
    rel_V = np.sqrt(mse_V) / (np.std(V_cf) + 1e-8)
    metrics['MSE_V'] = mse_V
    metrics['RelRMSE_V'] = rel_V
    
    # Total portfolio (all assets combined)
    mse_theta = np.mean((theta_pinn - theta_cf) ** 2)
    rel_theta = np.sqrt(mse_theta) / (np.std(theta_cf) + 1e-8)
    metrics['MSE_theta'] = mse_theta
    metrics['RelRMSE_theta'] = rel_theta
    
    # Myopic (all assets combined)
    mse_myopic = np.mean((myopic_pinn - myopic_cf) ** 2)
    rel_myopic = np.sqrt(mse_myopic) / (np.std(myopic_cf) + 1e-8)
    metrics['MSE_myopic'] = mse_myopic
    metrics['RelRMSE_myopic'] = rel_myopic
    
    # Hedging (all assets combined)
    mse_hedging = np.mean((hedging_pinn - hedging_cf) ** 2)
    rel_hedging = np.sqrt(mse_hedging) / (np.std(hedging_cf) + 1e-8)
    metrics['MSE_hedging'] = mse_hedging
    metrics['RelRMSE_hedging'] = rel_hedging
    
    # Per-asset metrics
    N = theta_pinn.shape[-1]
    for asset_idx in range(N):
        theta_p = theta_pinn[..., asset_idx]
        theta_c = theta_cf[..., asset_idx]
        myopic_p = myopic_pinn[..., asset_idx]
        myopic_c = myopic_cf[..., asset_idx]
        hedging_p = hedging_pinn[..., asset_idx]
        hedging_c = hedging_cf[..., asset_idx]
        
        # metrics[f'MSE_theta_{asset_idx}'] = np.mean((theta_p - theta_c) ** 2)
        # metrics[f'MSE_myopic_{asset_idx}'] = np.mean((myopic_p - myopic_c) ** 2)
        metrics[f'MSE_hedging_{asset_idx}'] = np.mean((hedging_p - hedging_c) ** 2)
    
    return metrics


# =============================================================================
# 8) Visualization
# =============================================================================
def heat_basic(ax, Z, title, extent, cmap='jet', vmin=None, vmax=None,
               xlabel="x[0]", ylabel="x[1]"):
    """Basic heatmap with shared color range."""
    im = ax.imshow(Z.T, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def heat_diverging(ax, Z, title, extent, cmap='RdBu_r', pct=98,
                   xlabel="x[0]", ylabel="x[1]"):
    """Diverging heatmap centered at 0 for difference plots."""
    abs_max = np.percentile(np.abs(Z), pct)
    if abs_max < 1e-10:
        abs_max = max(np.abs(Z).max(), 1e-10)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im = ax.imshow(Z.T, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def plot_value_comparison(x1_grid, x2_grid, V_pinn, V_cf,
                          dim1, dim2, tau_fixed, w_fixed,
                          save_path=None, show=True):
    """
    Plot value function comparison: PINN vs Closed-form.
    Layout: 1 row × 3 columns (PINN, Closed-form, Difference)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    extent = [x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()]
    xlabel, ylabel = f'x[{dim1}]', f'x[{dim2}]'
    
    # Shared color range
    vmin_V = min(V_pinn.min(), V_cf.min())
    vmax_V = max(V_pinn.max(), V_cf.max())
    
    heat_basic(axs[0], V_pinn, 'V (PINN)', extent, vmin=vmin_V, vmax=vmax_V,
               xlabel=xlabel, ylabel=ylabel)
    heat_basic(axs[1], V_cf, 'V (Closed-form)', extent, vmin=vmin_V, vmax=vmax_V,
               xlabel=xlabel, ylabel=ylabel)
    heat_diverging(axs[2], V_pinn - V_cf, 'V Difference', extent,
                   xlabel=xlabel, ylabel=ylabel)
    
    plt.suptitle(f'Kim-Omberg ND: Value Function (τ={tau_fixed:.2f}, w={w_fixed:.2f})', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_portfolio_comparison(x1_grid, x2_grid,
                              theta_pinn, theta_cf,
                              myopic_pinn, myopic_cf,
                              hedging_pinn, hedging_cf,
                              dim1, dim2, tau_fixed, w_fixed, N_ASSETS,
                              save_path=None, show=True, only_hedge=False):
    """
    Plot portfolio comparison for all assets.
    
    Layout per asset: 3 rows × 3 columns
        Row 0: Total θ*/w (PINN, CF, Diff)
        Row 1: Myopic (PINN, CF, Diff)
        Row 2: Hedging (PINN, CF, Diff)
    
    Generates one figure per asset.
    """
    extent = [x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()]
    xlabel, ylabel = f'x[{dim1}]', f'x[{dim2}]'

    if only_hedge == False:
    
        for asset_idx in range(N_ASSETS):
            fig, axs = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
            
            # Extract data for this asset
            theta_p = theta_pinn[:, :, asset_idx]
            theta_c = theta_cf[:, :, asset_idx]
            myopic_p = myopic_pinn[:, :, asset_idx]
            myopic_c = myopic_cf[:, :, asset_idx]
            hedging_p = hedging_pinn[:, :, asset_idx]
            hedging_c = hedging_cf[:, :, asset_idx]
            
            # === Row 0: Total portfolio θ*/w ===
            vmin_t = min(theta_p.min(), theta_c.min())
            vmax_t = max(theta_p.max(), theta_c.max())
            
            heat_basic(axs[0, 0], theta_p, f'θ*[{asset_idx}]/w (PINN)', extent,
                       vmin=vmin_t, vmax=vmax_t, xlabel=xlabel, ylabel=ylabel)
            heat_basic(axs[0, 1], theta_c, f'θ*[{asset_idx}]/w (Closed-form)', extent,
                       vmin=vmin_t, vmax=vmax_t, xlabel=xlabel, ylabel=ylabel)
            heat_diverging(axs[0, 2], theta_p - theta_c, f'θ*[{asset_idx}]/w Diff', extent,
                           xlabel=xlabel, ylabel=ylabel)
            
            # === Row 1: Myopic component ===
            vmin_m = min(myopic_p.min(), myopic_c.min())
            vmax_m = max(myopic_p.max(), myopic_c.max())
            
            heat_basic(axs[1, 0], myopic_p, f'Myopic[{asset_idx}] (PINN)', extent,
                       vmin=vmin_m, vmax=vmax_m, xlabel=xlabel, ylabel=ylabel)
            heat_basic(axs[1, 1], myopic_c, f'Myopic[{asset_idx}] (Closed-form)', extent,
                       vmin=vmin_m, vmax=vmax_m, xlabel=xlabel, ylabel=ylabel)
            heat_diverging(axs[1, 2], myopic_p - myopic_c, f'Myopic[{asset_idx}] Diff', extent,
                           xlabel=xlabel, ylabel=ylabel)
            
            # === Row 2: Hedging component ===
            vmin_h = min(hedging_p.min(), hedging_c.min())
            vmax_h = max(hedging_p.max(), hedging_c.max())
            
            heat_basic(axs[2, 0], hedging_p, f'Hedging[{asset_idx}] (PINN)', extent,
                       vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
            heat_basic(axs[2, 1], hedging_c, f'Hedging[{asset_idx}] (Closed-form)', extent,
                       vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
            heat_diverging(axs[2, 2], hedging_p - hedging_c, f'Hedging[{asset_idx}] Diff', extent,
                           xlabel=xlabel, ylabel=ylabel)

            plt.suptitle(f'Kim-Omberg ND: Asset {asset_idx} Portfolio (τ={tau_fixed:.2f}, w={w_fixed:.2f})',
                 fontsize=14)
    
            if save_path:
                path = save_path.replace('.png', f'_asset{asset_idx}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    else:
        for asset_idx in range(N_ASSETS):
            fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            
            # Extract hedging demand for assets
            hedging_p = hedging_pinn[:, :, asset_idx]
            hedging_c = hedging_cf[:, :, asset_idx]

            # === Row 2: Hedging component ===
            vmin_h = min(hedging_p.min(), hedging_c.min())
            vmax_h = max(hedging_p.max(), hedging_c.max())
            
            heat_basic(axs[0], hedging_p, f'Hedging[{asset_idx}] (PINN)', extent,
                       vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
            heat_basic(axs[1], hedging_c, f'Hedging[{asset_idx}] (Closed-form)', extent,
                       vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
            heat_diverging(axs[2], hedging_p - hedging_c, f'Hedging[{asset_idx}] Diff', extent,
                           xlabel=xlabel, ylabel=ylabel)

            plt.suptitle(f'Kim-Omberg ND: Asset {asset_idx} Portfolio (τ={tau_fixed:.2f}, w={w_fixed:.2f})',
                 fontsize=14)
    
            if save_path:
                path = save_path.replace('.png', f'_asset{asset_idx}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")
            
            if show:
                plt.show()
            else:
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


def plot_value_comparison_tauX(tau_grid, X_grid, V_pinn, V_cf,
                               dimX, w_fixed,
                               save_path=None, show=True,
                               xlabel="X", ylabel=r"$\tau$"):
    """Value comparison on (x-axis=X, y-axis=tau)."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    extent = [X_grid.min(), X_grid.max(), tau_grid.min(), tau_grid.max()]
    xlab = xlabel if xlabel is not None else f'x[{dimX}]'
    ylab = ylabel

    vmin_V = min(V_pinn.min(), V_cf.min())
    vmax_V = max(V_pinn.max(), V_cf.max())

    heat_basic_tauX(axs[0], V_pinn, 'V (PINN)', extent, vmin=vmin_V, vmax=vmax_V,
                    xlabel=xlab, ylabel=ylab)
    heat_basic_tauX(axs[1], V_cf, 'V (Closed-form)', extent, vmin=vmin_V, vmax=vmax_V,
                    xlabel=xlab, ylabel=ylab)
    heat_diverging_tauX(axs[2], V_pinn - V_cf, 'V Difference', extent,
                        xlabel=xlab, ylabel=ylab)

    plt.suptitle(f'Kim-Omberg ND: Value Function (w={w_fixed:.2f})', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


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



def eval_pinn_on_tau_X_grid(
        model, w_fixed, dimX, x_fixed,
        M, N, gamma, Gamma_t, lam0_t, Lam_t,
        X_min, X_max, tau_min, tau_max,
        N_tau=60, N_X=60, chunk=4096):
    """Evaluate PINN on a 2D grid: x-axis = state x[dimX], y-axis = tau.
    Other state dimensions are held fixed at x_fixed.
    """
    model.eval()

    tau_vals = np.linspace(tau_min, tau_max, N_tau)
    X_vals = np.linspace(X_min[dimX], X_max[dimX], N_X)

    # rows=tau, cols=X  -> shape (N_tau, N_X)
    X_grid, tau_grid = np.meshgrid(X_vals, tau_vals, indexing="xy")

    n_points = N_tau * N_X
    x_full = np.tile(x_fixed, (n_points, 1))
    x_full[:, dimX] = X_grid.reshape(-1)

    w_flat = torch.full((n_points, 1), float(w_fixed), device=device,
                        dtype=torch.float32, requires_grad=True)
    x_flat = torch.tensor(x_full, device=device, dtype=torch.float32, requires_grad=True)
    tau_flat = torch.tensor(tau_grid.reshape(-1, 1), device=device, dtype=torch.float32)

    V_list, theta_norm_list, myopic_list, hedging_list = [], [], [], []
    for i in range(0, n_points, chunk):
        w_b = w_flat[i:i + chunk]
        x_b = x_flat[i:i + chunk]
        tau_b = tau_flat[i:i + chunk]

        V_b, _, theta_norm_b, myopic_b, hedging_b = compute_optimal_theta_nd(
            model, w_b, x_b, tau_b, M, N, gamma,
            Gamma_t, lam0_t, Lam_t, create_graph=True
        )
        V_list.append(V_b.detach().cpu())
        theta_norm_list.append(theta_norm_b.detach().cpu())
        myopic_list.append(myopic_b.detach().cpu())
        hedging_list.append(hedging_b.detach().cpu())

    V_pinn = torch.cat(V_list, dim=0).numpy().reshape(N_tau, N_X)
    theta_norm_pinn = torch.cat(theta_norm_list, dim=0).numpy().reshape(N_tau, N_X, N)
    myopic_pinn = torch.cat(myopic_list, dim=0).numpy().reshape(N_tau, N_X, N)
    hedging_pinn = torch.cat(hedging_list, dim=0).numpy().reshape(N_tau, N_X, N)

    model.train()
    return tau_grid, X_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn


def eval_closed_form_on_tau_X_grid(
        w_fixed, dimX, x_fixed,
        M, N, gamma, r, lam0, Lam, Gamma, sol,
        X_min, X_max, tau_min, tau_max,
        N_tau=60, N_X=60):
    """Evaluate closed-form on a 2D grid: x-axis = state x[dimX], y-axis = tau."""
    tau_vals = np.linspace(tau_min, tau_max, N_tau)
    X_vals = np.linspace(X_min[dimX], X_max[dimX], N_X)

    X_grid, tau_grid = np.meshgrid(X_vals, tau_vals, indexing="xy")

    V_cf = np.zeros((N_tau, N_X))
    theta_norm_cf = np.zeros((N_tau, N_X, N))
    myopic_cf = np.zeros((N_tau, N_X, N))
    hedging_cf = np.zeros((N_tau, N_X, N))

    for itau in range(N_tau):
        tau = float(tau_vals[itau])
        for jx in range(N_X):
            x = x_fixed.copy()
            x[dimX] = float(X_vals[jx])

            V_cf[itau, jx] = closed_form_V(tau, float(w_fixed), x, sol, M, gamma, r)
            _, theta_norm_cf[itau, jx, :], myopic_cf[itau, jx, :], hedging_cf[itau, jx, :] = \
                closed_form_decomposition(tau, float(w_fixed), x, sol, M, N, gamma, lam0, Lam, Gamma)

    return tau_grid, X_grid, V_cf, theta_norm_cf, myopic_cf, hedging_cf



def plot_loss_history(loss_history, save_path=None, show=True):
    """Plot training loss history."""
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = np.arange(1, len(loss_history) + 1)
    
    ax.semilogy(epochs, [h['total'] for h in loss_history], label='Total', alpha=0.8)
    ax.semilogy(epochs, [h['pde'] for h in loss_history], label='PDE', alpha=0.6)
    ax.semilogy(epochs, [h['terminal'] for h in loss_history], label='Terminal', alpha=0.6)
    ax.semilogy(epochs, [h['concavity'] for h in loss_history], label='Concavity', alpha=0.6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Kim-Omberg ND PINN: Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_slices(model, tau_fixed, save_path=None, show=True):
    """
    Plot V and θ* as functions of W for different X values.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    W_vals = np.linspace(W_min, W_max, 100)
    X_test_vals = [0.1, 0.3, 0.5]
    
    model.eval()
    
    for idx, X_fixed in enumerate(X_test_vals):
        V_pinn_list = []
        V_cf_list = []
        theta_pinn_list = []
        theta_cf_list = []
        
        for W_val in W_vals:
            # PINN
            W_t = torch.tensor([[W_val]], device=device, dtype=torch.float32, requires_grad=True)
            X_t = torch.tensor([[X_fixed]], device=device, dtype=torch.float32, requires_grad=True)
            tau_t = torch.tensor([[tau_fixed]], device=device, dtype=torch.float32)
            
            V_p, theta_p, _,_,_ = compute_optimal_theta(model, W_t, X_t, tau_t, create_graph=True)
            V_pinn_list.append(V_p.item())
            theta_pinn_list.append(theta_p.item())
            
            # Closed-form
            V_cf_list.append(closed_form_V(tau_fixed, W_val, X_fixed))
            theta_cf_list.append(closed_form_theta(tau_fixed, W_val, X_fixed))
        
        # Plot V
        axs[0, idx].plot(W_vals, V_pinn_list, 'b-', label='PINN', linewidth=2)
        axs[0, idx].plot(W_vals, V_cf_list, 'r--', label='Closed-form', linewidth=2)
        axs[0, idx].set_xlabel('Wealth W')
        axs[0, idx].set_ylabel('V')
        axs[0, idx].set_title(f'X = {X_fixed}')
        axs[0, idx].legend()
        axs[0, idx].grid(True, alpha=0.3)
        
        # Plot θ
        axs[1, idx].plot(W_vals, theta_pinn_list, 'b-', label='PINN', linewidth=2)
        axs[1, idx].plot(W_vals, theta_cf_list, 'r--', label='Closed-form', linewidth=2)
        axs[1, idx].set_xlabel('Wealth W')
        axs[1, idx].set_ylabel('θ*')
        axs[1, idx].set_title(f'X = {X_fixed}')
        axs[1, idx].legend()
        axs[1, idx].grid(True, alpha=0.3)
    
    model.train()
    
    plt.suptitle(f'Kim-Omberg ND: V and θ* vs W at τ={tau_fixed} [γ={gamma}, ρ={rho}]', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


value_hidden=256
value_depth=3

batch_size=2000
lr=5e-4
w_terminal=20.0
resample_every=200
epochs = resample_every * 1000
w_shape=1.0

# =============================================================================
# 9) Main Execution
# =============================================================================
start = time.time()
if __name__ == "__main__":
   
    # Convert parameters to torch
    K_t = torch.tensor(K, device=device, dtype=torch.float32)
    k0_t = torch.tensor(k0, device=device, dtype=torch.float32)
    Q_t = torch.tensor(Q, device=device, dtype=torch.float32)
    Gamma_t = torch.tensor(Gamma, device=device, dtype=torch.float32)
    lam0_t = torch.tensor(lam0, device=device, dtype=torch.float32)
    Lam_t = torch.tensor(Lam, device=device, dtype=torch.float32)
    
    # Initialize model
    model = ValueNetND(M=M_STATES, hidden=value_hidden, depth=value_depth).to(device)
    
    # Training
    loss_history, optimizer = train_pinn_nd(
        model, M_STATES, N_ASSETS, gamma, r,
        K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t,
        X_min, X_max, W_min, W_max, tau_max,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        resample_every=resample_every,
        w_terminal=w_terminal,
        w_concavity=w_shape,
        print_every=5000
    )

    end = time.time()
    elapsed = end - start
    
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    
    print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")

    print(f"\n{'='*60}")
    print("Evaluating PINN vs Closed-form...")
    print(f"{'='*60}")
    print(f"  hidden node : {value_hidden}")
    print(f"  num of hidden layers : {value_depth}")
    print(f"  epochs       : {epochs}")
    print(f"  batch size   : {batch_size}")
    print(f"  resample every    : {resample_every}")
    print(f"  initial lr   : {lr}")
    print(f"  T.C weight   : {w_terminal}")
    print(f"  Seed         : {SEED}")
    print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")
    print(f"{'='*70}")
    
    model = ValueNetND(M=M_STATES, hidden=value_hidden, depth=value_depth).to(device)
    model.load_state_dict(torch.load(os.path.join(weight_dir, f"value_net_best_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).pt"), map_location=device))
    
    # Plot loss history
    plot_loss_history(
        loss_history,
        save_path=os.path.join(output_dir, f"loss_history_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).png"),
        show=True
    )
    
    # Evaluation
    print(f"\n{'='*60}")
    print("Evaluating PINN vs Closed-form...")
    print(f"{'='*60}")
    
    
    # (x-axis = state X, y-axis = time-to-maturity tau), holding other states fixed
    dimX = 0
    x_fixed = xbar.copy()  # Fix other state dimensions at long-run mean
    
    W_levels = [0.5]
    N_tau, N_X = 100, 100
    # X_min_test = X_min.max().item()
    # X_max_test = X_max.min().item()
    
    X_min_test = X_min
    X_max_test = X_max
    
    for w_test in W_levels:
        print(f"\n--- Grid evaluation: w={w_test:.2f} ---")
    
        # PINN evaluation on (tau, X) grid
        tau_grid, X_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn = \
            eval_pinn_on_tau_X_grid(
                model, w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma,
                Gamma_t=Gamma_t, lam0_t=lam0_t, Lam_t=Lam_t,
                X_min=X_min_test, X_max=X_max_test, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )
    
        # Closed-form evaluation on (tau, X) grid
        _, _, V_cf, theta_norm_cf, myopic_cf, hedging_cf = \
            eval_closed_form_on_tau_X_grid(
                w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma, r=r,
                lam0=lam0, Lam=Lam, Gamma=Gamma, sol=cf_sol,
                X_min=X_min_test, X_max=X_max_test, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )
    
        # Metrics
        metrics = compute_metrics(
            V_pinn, V_cf,
            theta_norm_pinn, theta_norm_cf,
            myopic_pinn, myopic_cf,
            hedging_pinn, hedging_cf
        )
    
        print("Metrics:")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"  {k}: {v:.6e}")
    
        # Plots
        plot_value_comparison_tauX(
            tau_grid, X_grid, V_pinn, V_cf,
            dimX=dimX, w_fixed=w_test,
            save_path=os.path.join(output_dir, f"value_{w_test:.2f}_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).png"),
            show=True,
            xlabel=f"risk premium X",
            ylabel=r"$\tau$"
        )
    
        plot_portfolio_comparison_tauX(
            tau_grid, X_grid,
            theta_norm_pinn, theta_norm_cf,
            myopic_pinn, myopic_cf,
            hedging_pinn, hedging_cf,
            dimX=dimX, w_fixed=w_test, N_ASSETS=N_ASSETS,
            save_path=os.path.join(output_dir, f"portfolio_w{w_test:.2f}_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {resample_every}-eval epoch).png"),
            show=True,only_hedge=False,
            xlabel=f"risk premium X",
            ylabel=r"$\tau$",
            max_assets=10, sort_by_range=True
        )