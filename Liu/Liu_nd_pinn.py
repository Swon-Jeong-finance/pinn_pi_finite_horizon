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
import argparse
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn

# Cap intra/inter-op CPU threads (multi-worker sweeps oversubscribe
# cores otherwise); TORCH_NUM_THREADS is exported by tune_pipinn.sh.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Add local path for joint_market_setup and experiment utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, '/mnt/user-data/uploads')
from joint_market_setup_dirichlet import generate_joint_market_params, JointMarketParams, cholesky_solve
from experiment_utils import (
    add_common_experiment_args, parse_w_levels, resolve_device, set_reproducibility,
    ExperimentRecorder, PDEEarlyStopper, append_csv_rows, save_json, none_or_float,
    parse_eval_margins, shrink_bounds, pres_from_mse,
)


# =============================================================================
# 0) CLI + Reproducibility + Device
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Liu ND PINN experiment runner")
    add_common_experiment_args(parser, model_type_default="pinn")

    parser.add_argument("--print-every", type=int, default=2000)
    # Previously hardcoded inside train_pinn_nd; exposed so bash overrides take effect.
    parser.add_argument("--scheduler-patience", type=int, default=5000)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    parser.add_argument("--lr-schedule", type=str, default="plateau",
                        choices=["plateau", "fixed"],
                        help="plateau = ReduceLROnPlateau on the training loss (legacy default); "
                             "fixed = constant lr, no scheduler (mirrors PI-PINN's 'fixed').")
    return parser


ARGS = build_arg_parser().parse_args()
# Resolve skip flags. --skip-plots is a BACK-COMPAT alias for --skip-figures
# (figures-only); evaluation is skipped ONLY when --skip-eval is passed. This
# guarantees a main sweep always computes full-dim metrics.csv even when
# per-run figures are suppressed.
SKIP_FIGURES = bool(ARGS.skip_figures or ARGS.skip_plots)
SKIP_EVAL = bool(ARGS.skip_eval)
SEED = ARGS.seed
# Market seed decoupled from the training seed: a seed sweep varies network
# init / collocation / optimizer randomness while the benchmark market
# (K, xbar, SigmaX, rho, Lambda, ...) stays FIXED. None = legacy (use SEED).
MARKET_SEED = ARGS.market_seed if ARGS.market_seed is not None else ARGS.seed
device = resolve_device(ARGS.device)
set_reproducibility(SEED, device)
if torch.cuda.is_available() and str(device).startswith("cuda"):
    torch.cuda.reset_peak_memory_stats(device)
print(f"Device: {device}")


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
N_ASSETS = ARGS.n_assets    # Number of risky assets
M_STATES = ARGS.m_states    # Number of state variables

weight_dir = ARGS.weight_root or f"weights/pinn/kim_omberg_{N_ASSETS}asset-{M_STATES}state"
os.makedirs(weight_dir, exist_ok=True)
output_dir = ARGS.output_root or f"outputs/pinn/kim_omberg_{N_ASSETS}asset-{M_STATES}state"
os.makedirs(output_dir, exist_ok=True)
recorder = ExperimentRecorder(output_dir, weight_dir, ARGS)
if ARGS.eval_only:
    # Eval-only must NOT touch training-time provenance (config.json etc.).
    recorder.save_config_eval()
else:
    recorder.save_config()
    # A NEW training run must start with FRESH per-run CSVs: appending onto a
    # previous same-tag run interleaves two experiments in one file.
    recorder.rotate_training_logs()

# Config sanity: a residual target without a validation set cannot stop.
if ARGS.pres_target is not None and (not ARGS.val_points or ARGS.val_points <= 0):
    raise SystemExit("[config error] --pres-target requires --val-points > 0 (held-out set is the stopping rule).")

# Time domain (τ = remaining horizon = T - t)
tau_max = ARGS.tau_max
tau_min = 0.0

# Wealth domain
W_min, W_max = ARGS.w_min, ARGS.w_max

# State domain (will be set based on theta ± some range)
X_RANGE_SCALE = ARGS.x_range_scale  # x ∈ [θ - scale*η, θ + scale*η] roughly

# Model parameters
gamma = ARGS.gamma     # CRRA risk aversion
r = ARGS.r        # risk-free rate

# Generate market parameters
params = generate_joint_market_params(
    n=N_ASSETS, k=M_STATES,
    seed=MARKET_SEED,
    sample_alpha=True,
    alpha_dist="dirichlet",
    dirichlet_concentration=ARGS.dirichlet_concentration,  # 보통 1.0이 무난 (균등한 Dirichlet)
    alpha_scale=ARGS.alpha_scale,              # row-sum이 alpha_scale이 되도록
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
                          method="RK45", t_eval=None, rtol=1e-12, atol=1e-14):
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
        # Dense grid: get_closed_form_at_tau uses LINEAR interpolation between
        # the saved nodes, so the node spacing (not just the solver tolerance)
        # bounds the ground-truth accuracy. 8001 nodes -> interp error ~1e-9.
        t_eval = np.linspace(0.0, T, 8001)

    # Tight tolerances: the closed form is the ground truth for every RelL2 /
    # substitution comparison, so it must be accurate well below the smallest
    # reported error. (solve_ivp DEFAULTS are rtol=1e-3, atol=1e-6 -- runs
    # recorded before this fix carry ~1e-5-level ODE error in their npz.)
    sol = solve_ivp(rhs, (0.0, T), y0, t_eval=t_eval, method=method,
                    rtol=rtol, atol=atol)
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

# Persist market / closed-form objects needed for later standalone plotting.
if not ARGS.eval_only:
    recorder.save_market_snapshot(
        K=K, xbar=xbar, SigmaX=SigmaX, rho=rho, Lam=Lam, Q=Q, Gamma=Gamma,
        k0=k0, lam0=lam0, X_min=X_min, X_max=X_max, eta=eta,
        gamma=np.array([gamma]), r=np.array([r]), tau_max=np.array([tau_max]),
        W_min=np.array([W_min]), W_max=np.array([W_max]), seed=np.array([SEED]), market_seed=np.array([MARKET_SEED]),
    )
    recorder.save_closed_form_solution(cf_sol)


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


def build_validation_set(n_int, n_term, device, M, X_min, X_max, W_min, W_max, tau_max, seed):
    """Held-out validation set on Q_col, sampled ONCE with a dedicated RNG.

    Mirrors sample_interior / sample_terminal (same eps offsets, same uniform
    law) but draws on a CPU generator seeded independently of the global
    stream, so training reproducibility is unaffected and the set is fixed
    for the whole run (paper: held-out collocation set disjoint from training).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 1000003 + 20260718)

    eps_tau = 1e-3
    eps_W = 1e-2
    X_min_t = torch.tensor(X_min, dtype=torch.float32)
    X_max_t = torch.tensor(X_max, dtype=torch.float32)

    tau_i = eps_tau + torch.rand(n_int, 1, generator=gen) * (tau_max - eps_tau)
    w_i = W_min + eps_W + torch.rand(n_int, 1, generator=gen) * (W_max - W_min - 2 * eps_W)
    x_i = X_min_t + torch.rand(n_int, M, generator=gen) * (X_max_t - X_min_t)

    w_t = W_min + eps_W + torch.rand(n_term, 1, generator=gen) * (W_max - W_min - 2 * eps_W)
    x_t = X_min_t + torch.rand(n_term, M, generator=gen) * (X_max_t - X_min_t)
    tau_t = torch.zeros(n_term, 1)

    return {
        "w_int": w_i.to(device), "x_int": x_i.to(device), "tau_int": tau_i.to(device),
        "w_term": w_t.to(device), "x_term": x_t.to(device), "tau_term": tau_t.to(device),
    }


def evaluate_heldout_pres_pinn(model, val_set, M, N, gamma, r,
                               K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t, chunk=4096):
    """Held-out residual level for the DIRECT PINN baseline:
    p_res = RMS(nonlinear HJB residual on Q_col) + RMS(terminal mismatch)."""
    model.eval()
    n = val_set["w_int"].shape[0]
    sq_sum = 0.0
    for i in range(0, n, chunk):
        w_b = val_set["w_int"][i:i + chunk].detach().clone().requires_grad_(True)
        x_b = val_set["x_int"][i:i + chunk].detach().clone().requires_grad_(True)
        tau_b = val_set["tau_int"][i:i + chunk].detach().clone().requires_grad_(True)
        residual, _, _, _, _ = hjb_residual_nd(
            model, w_b, x_b, tau_b, M, N, gamma, r,
            K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t
        )
        sq_sum += float(torch.sum(residual.detach() ** 2).item())
    pde_rms = float(np.sqrt(sq_sum / max(n, 1)))

    with torch.no_grad():
        V_T_pred = model(val_set["w_term"], val_set["x_term"], val_set["tau_term"])
        V_T_true = V_terminal(val_set["w_term"], gamma)
        term_rms = float(torch.sqrt(torch.mean((V_T_pred - V_T_true) ** 2)).item())

    model.train()
    return pde_rms, term_rms, pde_rms + term_rms


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
    # denominator = 2 V_ww must be negative (concavity); clamp from above.
    denominator_safe = torch.clamp(denominator, max=-1e-8)
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
    # Concavity-respecting guard: V_ww must be negative for the FOC to be a
    # maximizer, so clamp from ABOVE at -1e-8. (The previous
    # sign(V_ww)*1e-8 + 1e-10 form gave +1e-10 at V_ww == 0: wrong sign and
    # 100x smaller than intended, letting the control blow up.)
    V_ww_safe = torch.clamp(V_ww, max=-1e-8)
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
                  epochs=50000, batch_size=2000, terminal_frac=0.5, lr=5e-4,
                  eval_epochs=200, outer_iters=1000,
                  w_terminal=10.0, w_concavity=1.0, w_rra=0.0,
                  scheduler_patience=5000, scheduler_factor=0.5, scheduler_min_lr=1e-8,
                  save_iterate_every=1,
                  pres_target=None, val_points=100000, val_terminal_points=10000,
                  val_every=1, val_seed=0,
                  diag_points=0, diag_margin=0.0, diag_every=1, timing_mode=False,
                  print_every=2000, recorder=None, stopper=None):
    """Train PINN for multi-dimensional Kim-Omberg HJB.

    eval_epochs is the old resample_every interval. One eval_epochs block is
    treated as one pseudo-outer iteration for PDE-loss early stopping.

    If pres_target is set, the held-out residual level p_res (nonlinear HJB
    RMS + terminal RMS on a fixed validation set) is checked every val_every
    epochs within a block, plus once BEFORE the block's first step. On the
    FIRST time p_res <= pres_target, training STOPS GLOBALLY: the direct
    PINN solves a single nonlinear HJB, so reaching the target means the
    equation is solved to tolerance and further blocks would only re-verify
    the same unchanged model. The stop epoch/block and achieved p_res are
    recorded; this is a SUCCESS stop, distinct from the divergence stopper.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if ARGS.lr_schedule == "fixed":
        scheduler = None  # constant lr throughout (scheduler-off test mode)
        print("[lr] schedule = fixed (no scheduler; constant lr)")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr
        )

    # Held-out validation set: sampled ONCE per run with a dedicated RNG
    # stream (training RNG untouched).
    val_set = None
    skip_val = timing_mode and pres_target is None  # audit-only val = diagnostic
    if val_points and val_points > 0 and not skip_val:
        val_set = build_validation_set(
            int(val_points), max(1, int(val_terminal_points)), device,
            M, X_min, X_max, W_min, W_max, tau_max, seed=val_seed,
        )

    if timing_mode:
        # Timing runs must not write iterate checkpoints regardless of caller.
        save_iterate_every = 0

    diag = None
    diag_col = None
    if diag_points and diag_points > 0 and not timing_mode:
        diag = build_diag_set(
            int(diag_points), float(diag_margin), M, N, gamma, r,
            X_min, X_max, W_min, W_max, tau_max,
            cf_sol, lam0, Lam, Gamma, market_seed=MARKET_SEED,
        )
        # Q_col subsample for the ellipticity of the implied GREEDY policy
        # (the direct method has no frozen policy; theta here is derived
        # from the current V, so the column is named ..._greedy).
        diag_col = build_validation_set(
            int(diag_points), 1, device, M,
            X_min, X_max, W_min, W_max, tau_max, seed=MARKET_SEED,
        )
        print(f"[diag] {diag_points} pts on Q_ev(margin={diag_margin}) for e_n / margins "
              f"+ {diag_points} Q_col pts for greedy-policy ellipticity")

    def _val_check():
        return evaluate_heldout_pres_pinn(
            model, val_set, M, N, gamma, r,
            K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t)

    loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    # Weight-saving policy (paper protocol): final = official, best = diagnostic.
    # iterates/value_net_iter{NNNN}.pt snapshots one state per pseudo-outer
    # (resampling) block, mirroring the PI-PINN per-iteration snapshots.
    best_path = os.path.join(weight_dir, "value_net_best.pt")
    last_path = os.path.join(weight_dir, "value_net_last.pt")
    final_path = os.path.join(weight_dir, "value_net_final.pt")
    iterate_dir = os.path.join(weight_dir, "iterates")
    if save_iterate_every and save_iterate_every > 0:
        os.makedirs(iterate_dir, exist_ok=True)
    legacy_best_path = os.path.join(
        weight_dir,
        f"value_net_best_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {eval_epochs}-eval epoch).pt"
    )

    train_fields = [
        "timestamp", "model_type", "run_tag", "epoch", "outer_iter", "inner_epoch",
        "total_loss", "pde_loss", "terminal_loss", "concavity_loss", "monotonicity_loss","rra_loss",
        "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres",
        "theta_diff", "eval_loss", "lr", "best_loss", "elapsed_sec", "stopped", "stop_reason",
    ]
    outer_fields = [
        "timestamp", "model_type", "run_tag", "outer_iter", "epoch", "total_loss", "pde_loss",
        "terminal_loss", "monotonicity_loss", "concavity_loss", "rra_loss",
        "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres",
        "inner_epochs_used", "target_reached",
        "e_V_sup", "e_bundle_sup", "e_Xev", "diag_RelL2_V", "diag_RelL2_theta",
        "m_ww", "M_num", "guard_frac_ev",
        "lam_min_sigma_greedy", "lam_max_sigma_greedy", "clip_frac_frozen",
        "lr", "best_loss", "bad_count", "stop_active",
        "stop_is_bad", "stopped", "stop_reason", "elapsed_sec",
    ]
    pending_train_rows = []

    # Diagnostic best cache (flushed to disk at block boundaries only).
    best_state_cpu = None
    best_dirty = False

    # Explicit optimizer-step counter (E8): with the target global stop the
    # executed steps cannot be recovered from epochs alone.
    total_opt_steps = 0

    # Per-block early-stop state (pres-target rule).
    block_stopped = False
    block_target_reached = False
    block_epochs_used = 0
    block_last_val = None        # (pde_rms, term_rms, pres) at the block's final state
    block_last_val_at_final = False

    # Initial sampling
    w_int, x_int, tau_int = sample_interior(batch_size, device, M, X_min, X_max, W_min, W_max, tau_max)
    # w_term, x_term, tau_term = sample_terminal(batch_size // 4, device, M, X_min, X_max, W_min, W_max)
    w_term, x_term, tau_term = sample_terminal(max(1, int(batch_size * terminal_frac)), device, M, X_min, X_max, W_min, W_max)
    V_T_target = V_terminal(w_term, gamma).detach()

    print(f"\n{'='*60}")
    print(f"Training {M+2}D PINN (N={N} assets, M={M} states)")
    print(f"  outer_iters : {outer_iters}")
    print(f"  eval_epochs : {eval_epochs}  (old resample_every)")
    print(f"  epochs      : {epochs}")
    print(f"{'='*60}")

    stop_info = {"stopped_early": False}
    # Safe defaults (referenced at block boundaries even if a block was fully
    # skipped by the pre-step target check before any training step ran).
    current_loss = current_pde = current_terminal = float("nan")
    current_conc = current_mono = current_rra = float("nan")
    current_lr = float(lr)
    elapsed = 0.0

    for epoch in range(1, epochs + 1):
        outer_iter_float = (epoch - 1) // eval_epochs + 1
        inner_epoch = (epoch - 1) % eval_epochs + 1

        # Block start. BUGFIX (off-by-one): the batch used to be refreshed at
        # `epoch % eval_epochs == 0`, i.e. on the LAST epoch of a block, so
        # each block's final step already trained on the NEXT block's batch.
        # Resampling now happens at the FIRST epoch of every block (block 1
        # uses the initial sample drawn above), so one block = one batch.
        if inner_epoch == 1:
            if epoch > 1:
                w_int, x_int, tau_int = sample_interior(batch_size, device, M, X_min, X_max, W_min, W_max, tau_max)
                w_term, x_term, tau_term = sample_terminal(max(1, int(batch_size * terminal_frac)), device, M, X_min, X_max, W_min, W_max)
                V_T_target = V_terminal(w_term, gamma).detach()

            # Reset the per-block pres-target state and run the pre-step
            # check (a state that already meets the target must not be
            # perturbed by further steps -- skip the whole block).
            block_stopped = False
            block_target_reached = False
            block_epochs_used = 0
            block_last_val = None
            block_last_val_at_final = False
            if val_set is not None and pres_target is not None:
                v = _val_check()
                block_last_val, block_last_val_at_final = v, True
                if v[2] <= float(pres_target):
                    block_stopped = True
                    block_target_reached = True
                    # GLOBAL STOP (pre-step): the model already satisfies the
                    # target, so training ends here. Run one diagnostic
                    # forward (no optimizer step) so current_* and one train
                    # row reflect the final state (mirrors the PI-PINN
                    # epoch-0 synthetic row), then finalize below.
                    w_d = w_int.detach().clone().requires_grad_(True)
                    x_d = x_int.detach().clone().requires_grad_(True)
                    tau_d = tau_int.detach().clone().requires_grad_(True)
                    residual_d, V_d, V_w_d, V_ww_d, _ = hjb_residual_nd(
                        model, w_d, x_d, tau_d, M, N, gamma, r,
                        K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t
                    )
                    with torch.no_grad():
                        V_T_pred_d = model(w_term, x_term, tau_term)
                        terminal_d = torch.mean((V_T_pred_d - V_T_target) ** 2)
                    pde_d = torch.mean(residual_d.detach() ** 2)
                    conc_d = torch.mean(torch.relu(V_ww_d.detach()) ** 2)
                    mono_d = torch.mean(torch.relu(-V_w_d.detach()) ** 2)
                    eta_d = -w_d.detach() * V_ww_d.detach() / torch.clamp(V_w_d.detach(), min=1e-8)
                    rra_d = torch.mean((eta_d - gamma) ** 2)
                    current_pde = float(pde_d.item())
                    current_terminal = float(terminal_d.item())
                    current_conc = float(conc_d.item())
                    current_mono = float(mono_d.item())
                    current_rra = float(rra_d.item())
                    current_loss = float(current_pde + w_terminal * current_terminal
                                         + w_concavity * (current_conc + current_mono)
                                         + w_rra * current_rra)
                    current_lr = float(optimizer.param_groups[0]['lr'])
                    elapsed = time.time() - start_time
                    if not timing_mode:
                     pending_train_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model_type": ARGS.model_type,
                        "run_tag": ARGS.run_tag,
                        "epoch": epoch,
                        "outer_iter": outer_iter_float,
                        "inner_epoch": 0,
                        "total_loss": current_loss,
                        "pde_loss": current_pde,
                        "terminal_loss": current_terminal,
                        "concavity_loss": current_conc,
                        "monotonicity_loss": current_mono,
                        "rra_loss": current_rra,
                        "train_pres": pres_from_mse(current_pde, current_terminal),
                        "val_pde_rms": v[0],
                        "val_terminal_rms": v[1],
                        "val_pres": v[2],
                        "theta_diff": "",
                        "eval_loss": "",
                        "lr": current_lr,
                        "best_loss": best_loss,
                        "elapsed_sec": elapsed,
                        "stopped": 0,
                        "stop_reason": "pres_target_reached",
                    })

        if not block_stopped:
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

            # shape constraints in wealth direction
            mono_loss = torch.mean(torch.relu(-V_w) ** 2)

            # CRRA homogeneity: local relative risk aversion eta = -w V_ww / V_w must equal gamma.
            # (Equivalently the myopic coefficient -V_w/(w V_ww) must equal 1/gamma.)
            eta = -w_int * V_ww / torch.clamp(V_w, min=1e-8)
            rra_loss = torch.mean((eta - gamma) ** 2)

            # Total loss
            total_loss = (pde_loss + w_terminal * terminal_loss + w_concavity * (concavity_loss + mono_loss) + w_rra * rra_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_opt_steps += 1
            if scheduler is not None:
                scheduler.step(total_loss.detach().cpu())

            # Clear gradients
            for var in [w_int, x_int, tau_int]:
                if var.grad is not None:
                    var.grad = None

            current_loss = float(total_loss.item())
            current_pde = float(pde_loss.item())
            current_terminal = float(terminal_loss.item())
            current_conc = float(concavity_loss.item())
            current_mono = float(mono_loss.item())
            current_rra = float(rra_loss.item())
            current_lr = float(optimizer.param_groups[0]['lr'])
            elapsed = time.time() - start_time
            block_epochs_used = inner_epoch
            block_last_val_at_final = False  # state changed since the last check

            # Track best (diagnostic only). Disk writes moved to block
            # boundaries: the old per-improvement double torch.save could
            # fire nearly every epoch and pollutes E8 wall-clock.
            if current_loss < best_loss:
                best_loss = current_loss
                if not timing_mode:
                    best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_dirty = True

            _lh_row = {
                'total': current_loss,
                'pde': current_pde,
                'terminal': current_terminal,
                'monotonicity': current_mono,
                'concavity': current_conc,
                "rra": current_rra,
            }
            if timing_mode and loss_history:
                loss_history[-1] = _lh_row  # timing: keep only the latest row
            else:
                loss_history.append(_lh_row)

            row_val = {"val_pde_rms": "", "val_terminal_rms": "", "val_pres": ""}

            # Post-step held-out check against the residual target.
            # First satisfaction triggers the GLOBAL stop below.
            if val_set is not None and pres_target is not None and (inner_epoch % max(1, int(val_every)) == 0):
                v = _val_check()
                block_last_val, block_last_val_at_final = v, True
                row_val = {"val_pde_rms": v[0], "val_terminal_rms": v[1], "val_pres": v[2]}
                if v[2] <= float(pres_target):
                    block_stopped = True
                    block_target_reached = True

            if not timing_mode:
             pending_train_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_type": ARGS.model_type,
                "run_tag": ARGS.run_tag,
                "epoch": epoch,
                "outer_iter": outer_iter_float,
                "inner_epoch": inner_epoch,
                "total_loss": current_loss,
                "pde_loss": current_pde,
                "terminal_loss": current_terminal,
                "concavity_loss": current_conc,
                "monotonicity_loss": current_mono,
                "rra_loss": current_rra,
                "train_pres": pres_from_mse(current_pde, current_terminal),
                **row_val,
                "theta_diff": "",
                "eval_loss": "",
                "lr": current_lr,
                "best_loss": best_loss,
                "elapsed_sec": elapsed,
                "stopped": 0,
                "stop_reason": "",
            })

            if epoch % print_every == 0:
                print(f"[{epoch:6d}/{epochs}] Total: {current_loss:.3e} | "
                      f"PDE: {current_pde:.3e} | Term: {current_terminal:.3e} | Mono: {current_mono:.3e} |"
                      f"Conc: {current_conc:.3e} | RRA(η-γ)²: {current_rra:.3e} | LR: {current_lr:.2e}")

        # GLOBAL STOP on first target satisfaction (pre-step or post-step).
        if block_target_reached:
            if best_dirty and best_state_cpu is not None:
                torch.save(best_state_cpu, best_path)
                torch.save(best_state_cpu, legacy_best_path)
                best_dirty = False
            diag_res = {}
            if diag is not None:
                diag_res = eval_diag_metrics(
                    model, diag, M, N, gamma,
                    Gamma_t, lam0_t, Lam_t, Gamma, lam0, Lam,
                )
            # Timing mode: no CSV I/O inside the timed loop (kept symmetric
            # with PI-PINN); the stop summary still goes to status.json.
            if recorder is not None and not timing_mode:
                if pending_train_rows:
                    append_csv_rows(recorder.train_csv, pending_train_rows, train_fields)
                    pending_train_rows = []
                _bv = block_last_val if block_last_val is not None else ("", "", "")
                append_csv_rows(recorder.outer_csv, [{
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag,
                    "outer_iter": outer_iter_float,
                    "epoch": epoch,
                    "total_loss": current_loss,
                    "pde_loss": current_pde,
                    "terminal_loss": current_terminal,
                    "monotonicity_loss": current_mono,
                    "concavity_loss": current_conc,
                    "rra_loss": current_rra,
                    "train_pres": pres_from_mse(current_pde, current_terminal) if current_pde == current_pde else "",
                    "val_pde_rms": _bv[0],
                    "val_terminal_rms": _bv[1],
                    "val_pres": _bv[2],
                    "inner_epochs_used": block_epochs_used,
                    "target_reached": 1,
                    "e_V_sup": diag_res.get("e_V_sup", ""),
                    "e_bundle_sup": diag_res.get("e_bundle_sup", ""),
                    "e_Xev": diag_res.get("e_Xev", ""),
                    "diag_RelL2_V": diag_res.get("diag_RelL2_V", ""),
                    "diag_RelL2_theta": diag_res.get("diag_RelL2_theta", ""),
                    "m_ww": diag_res.get("m_ww", ""),
                    "M_num": diag_res.get("M_num", ""),
                    "guard_frac_ev": diag_res.get("guard_frac_ev", ""),
                    "lam_min_sigma_greedy": "",
                    "lam_max_sigma_greedy": "",
                    "clip_frac_frozen": "",
                    "lr": current_lr,
                    "best_loss": best_loss,
                    "bad_count": "",
                    "stop_active": "",
                    "stop_is_bad": "",
                    "stopped": 0,
                    "stop_reason": "pres_target_reached",
                    "elapsed_sec": time.time() - start_time,
                }], outer_fields)
            torch.save(model.state_dict(), last_path)
            torch.save(model.state_dict(), final_path)
            _achieved = block_last_val[2] if block_last_val is not None else float("nan")
            print(f"\n[target-stop] PINN reached pres_target at epoch {epoch} "
                  f"(block {outer_iter_float}, {block_epochs_used} steps in block, "
                  f"achieved p_res={_achieved:.3e}). Training ends; model = FINAL state -> {final_path}")
            stop_info = {
                "stopped_early": False,
                "target_reached": True,
                "epoch_at_stop": int(epoch),
                "block_at_stop": int(outer_iter_float),
                "inner_epochs_in_stop_block": int(block_epochs_used),
                "achieved_pres": float(_achieved),
                "total_optimizer_steps": int(total_opt_steps),
            }
            return loss_history, optimizer, stop_info

        # One PINN pseudo-outer iteration = one eval_epochs block.
        if epoch % eval_epochs == 0:
            outer_iter = epoch // eval_epochs

            # Snapshot the state at the end of this pseudo-outer block
            # (kept symmetric with PI-PINN iterate snapshots).
            if save_iterate_every and save_iterate_every > 0 and (outer_iter % save_iterate_every == 0):
                torch.save(model.state_dict(), os.path.join(iterate_dir, f"value_net_iter{outer_iter:04d}.pt"))

            # Flush the diagnostic best cache once per block (not per epoch).
            if best_dirty and best_state_cpu is not None:
                torch.save(best_state_cpu, best_path)
                torch.save(best_state_cpu, legacy_best_path)
                best_dirty = False

            # End-of-block held-out measurement (the reported per-block
            # p_res). Reuse the last check if the state is unchanged since.
            if val_set is not None and not block_last_val_at_final:
                block_last_val = _val_check()
                block_last_val_at_final = True
            _bv = block_last_val if block_last_val is not None else ("", "", "")

            # Fixed-set diagnostics for this pseudo-outer state (E1-b/c
            # analogues + closed-form e_n components).
            diag_res = {}
            greedy_lam_min = ""
            greedy_lam_max = ""
            if diag is not None and (diag_every <= 1 or outer_iter % diag_every == 0 or outer_iter == 1):
                diag_res = eval_diag_metrics(
                    model, diag, M, N, gamma,
                    Gamma_t, lam0_t, Lam_t, Gamma, lam0, Lam,
                )
                # Ellipticity of the implied greedy policy on Q_col.
                _th_l = []
                _P = diag_col["w_int"].shape[0]
                for _i in range(0, _P, 4096):
                    _w = diag_col["w_int"][_i:_i + 4096].detach().clone().requires_grad_(True)
                    _x = diag_col["x_int"][_i:_i + 4096].detach().clone().requires_grad_(True)
                    _tau = diag_col["tau_int"][_i:_i + 4096]
                    _, _th, _, _, _ = compute_optimal_theta_nd(
                        model, _w, _x, _tau, M, N, gamma,
                        Gamma_t, lam0_t, Lam_t, create_graph=True)
                    _th_l.append(_th.detach().cpu())
                _th_np = torch.cat(_th_l, dim=0).numpy()
                _lmin, _lmax = sigma_eig_extremes_batch(_th_np, Gamma, Q)
                greedy_lam_min = float(np.min(_lmin))
                greedy_lam_max = float(np.max(_lmax))

            stop_triggered = False
            stop_meta = {"active": False, "is_bad": False, "bad_count": 0}
            if stopper is not None:
                stop_triggered, stop_meta = stopper.update(outer_iter, current_pde)

            outer_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_type": ARGS.model_type,
                "run_tag": ARGS.run_tag,
                "outer_iter": outer_iter,
                "epoch": epoch,
                "total_loss": current_loss,
                "pde_loss": current_pde,
                "terminal_loss": current_terminal,
                "monotonicity_loss": current_mono,
                "concavity_loss": current_conc,
                "rra_loss": current_rra,
                "train_pres": pres_from_mse(current_pde, current_terminal) if current_pde == current_pde else "",
                "val_pde_rms": _bv[0],
                "val_terminal_rms": _bv[1],
                "val_pres": _bv[2],
                "inner_epochs_used": block_epochs_used,
                "target_reached": int(bool(block_target_reached)),
                "e_V_sup": diag_res.get("e_V_sup", ""),
                "e_bundle_sup": diag_res.get("e_bundle_sup", ""),
                "e_Xev": diag_res.get("e_Xev", ""),
                    "diag_RelL2_V": diag_res.get("diag_RelL2_V", ""),
                    "diag_RelL2_theta": diag_res.get("diag_RelL2_theta", ""),
                "m_ww": diag_res.get("m_ww", ""),
                "M_num": diag_res.get("M_num", ""),
                "guard_frac_ev": diag_res.get("guard_frac_ev", ""),
                "lam_min_sigma_greedy": greedy_lam_min,
                "lam_max_sigma_greedy": greedy_lam_max,
                "clip_frac_frozen": "",
                "lr": current_lr,
                "best_loss": best_loss,
                "bad_count": stop_meta.get("bad_count", ""),
                "stop_active": int(bool(stop_meta.get("active", False))),
                "stop_is_bad": int(bool(stop_meta.get("is_bad", False))),
                "stopped": int(bool(stop_triggered)),
                "stop_reason": stop_meta.get("reason", ""),
                "elapsed_sec": elapsed,
            }

            # Timing mode: outer CSV writes are excluded from the timed loop
            # on BOTH methods (symmetric E8 accounting).
            if recorder is not None and not timing_mode:
                append_csv_rows(recorder.train_csv, pending_train_rows, train_fields)
                pending_train_rows = []
                append_csv_rows(recorder.outer_csv, [outer_row], outer_fields)
            elif timing_mode:
                pending_train_rows = []

            if stop_triggered:
                torch.save(model.state_dict(), last_path)
                torch.save(model.state_dict(), final_path)
                stop_info = {"stopped_early": True, **stop_meta, "outer_iter": outer_iter, "epoch": epoch,
                             "total_optimizer_steps": int(total_opt_steps)}
                print(f"\n[early-stop] PINN stopped at outer={outer_iter}, epoch={epoch}, PDE={current_pde:.4e}, reason={stop_meta.get('reason', '')}")
                return loss_history, optimizer, stop_info

    if recorder is not None and pending_train_rows:
        append_csv_rows(recorder.train_csv, pending_train_rows, train_fields)

    if best_dirty and best_state_cpu is not None:
        torch.save(best_state_cpu, best_path)
        torch.save(best_state_cpu, legacy_best_path)

    # FINAL-ITERATE POLICY: the reported model is the final training state.
    # best is saved during training but kept as a diagnostic artifact only
    # (no best-restore here).
    torch.save(model.state_dict(), last_path)
    torch.save(model.state_dict(), final_path)
    print(f"\nPINN finished. Reported model = FINAL state -> {final_path}")
    print(f"  [diagnostic] best checkpoint (loss={best_loss:.3e}) -> {best_path}")

    stop_info["total_optimizer_steps"] = int(total_opt_steps)
    return loss_history, optimizer, stop_info


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
    Compute MSE, RelL2 (headline) and StdNRMSE for V, θ, myopic, hedging.
    
    Returns dict with metrics for total and per-asset.
    """
    metrics = {}
    
    # Relative L2 error: ||f_hat - f*||_2 / ||f*||_2  (standard PINN convention)
    EPS = 1e-8
    
    mse_V = np.mean((V_pinn - V_cf) ** 2)
    rel_V = np.sqrt(mse_V) / (np.std(V_cf) + EPS)
    rel_V_l2 = np.sqrt(np.sum((V_pinn - V_cf) ** 2)) / (np.sqrt(np.sum(V_cf ** 2)) + EPS)
    metrics['MSE_V'] = mse_V
    metrics['StdNRMSE_V'] = rel_V
    metrics['RelL2_V'] = rel_V_l2
    
    # Total portfolio
    mse_theta = np.mean((theta_pinn - theta_cf) ** 2)
    rel_theta = np.sqrt(mse_theta) / (np.std(theta_cf) + EPS)
    rel_theta_l2 = np.sqrt(np.sum((theta_pinn - theta_cf) ** 2)) / (np.sqrt(np.sum(theta_cf ** 2)) + EPS)
    metrics['MSE_theta'] = mse_theta
    metrics['StdNRMSE_theta'] = rel_theta
    metrics['RelL2_theta'] = rel_theta_l2
    
    mse_myopic = np.mean((myopic_pinn - myopic_cf) ** 2)
    rel_myopic = np.sqrt(mse_myopic) / (np.std(myopic_cf) + EPS)
    rel_myopic_l2 = np.sqrt(np.sum((myopic_pinn - myopic_cf) ** 2)) / (np.sqrt(np.sum(myopic_cf ** 2)) + EPS)
    metrics['MSE_myopic'] = mse_myopic
    metrics['StdNRMSE_myopic'] = rel_myopic
    metrics['RelL2_myopic'] = rel_myopic_l2
    
    mse_hedging = np.mean((hedging_pinn - hedging_cf) ** 2)
    rel_hedging = np.sqrt(mse_hedging) / (np.std(hedging_cf) + EPS)
    rel_hedging_l2 = np.sqrt(np.sum((hedging_pinn - hedging_cf) ** 2)) / (np.sqrt(np.sum(hedging_cf ** 2)) + EPS)
    metrics['MSE_hedging'] = mse_hedging
    metrics['StdNRMSE_hedging'] = rel_hedging
    metrics['RelL2_hedging'] = rel_hedging_l2
    
    # Per-asset metrics
    N = theta_pinn.shape[-1]
    for asset_idx in range(N):
        theta_p = theta_pinn[..., asset_idx]
        theta_c = theta_cf[..., asset_idx]
        myopic_p = myopic_pinn[..., asset_idx]
        myopic_c = myopic_cf[..., asset_idx]
        hedging_p = hedging_pinn[..., asset_idx]
        hedging_c = hedging_cf[..., asset_idx]
        
        metrics[f'MSE_theta_{asset_idx}'] = np.mean((theta_p - theta_c) ** 2)
        metrics[f'MSE_myopic_{asset_idx}'] = np.mean((myopic_p - myopic_c) ** 2)
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



def build_diag_set(n_points, margin, M, N, gamma, r,
                   X_min, X_max, W_min, W_max, tau_max,
                   sol, lam0, Lam, Gamma, market_seed):
    """Fixed Q_ev diagnostic set for per-iteration e_n and stability margins.

    Points are uniform over (0, tau_max] x Omega_ev at the PRIMARY margin,
    drawn from a MARKET-seed-derived RNG so every training seed and both
    methods use the SAME set. Closed-form V and the reduced derivative
    bundle (V_w, V_ww, grad_x V_w) are precomputed here once:

        V    = D * w^{1-g}/(1-g) * phi,  phi = exp(a + b'x + x'Cx/2),
        V_w  = D * w^{-g} * phi,         D = exp((1-g) r tau),
        V_ww = -g * D * w^{-g-1} * phi,
        d_x V_w = D * w^{-g} * phi * (b + C x).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(market_seed) * 1000003 + 7)
    U = torch.rand(int(n_points), 2 + M, generator=gen).numpy().astype(np.float64)
    eps_tau = 1e-3
    W_lo, W_hi = shrink_bounds(float(W_min), float(W_max), float(margin))
    X_lo, X_hi = shrink_bounds(np.asarray(X_min, dtype=np.float64),
                               np.asarray(X_max, dtype=np.float64), float(margin))
    tau_np = eps_tau + U[:, 0] * (tau_max - eps_tau)
    w_np = W_lo + U[:, 1] * (W_hi - W_lo)
    x_np = X_lo[None, :] + U[:, 2:] * (X_hi - X_lo)[None, :]

    P = int(n_points)
    V_cf = np.zeros(P)
    Vw_cf = np.zeros(P)
    Vww_cf = np.zeros(P)
    Vwx_cf = np.zeros((P, M))
    for i in range(P):
        a, b, C = get_closed_form_at_tau(float(tau_np[i]), sol, M)
        phi = np.exp(a + b @ x_np[i] + 0.5 * x_np[i] @ C @ x_np[i])
        D = np.exp((1.0 - gamma) * r * tau_np[i])
        w_i = w_np[i]
        V_cf[i] = D * np.power(w_i, 1.0 - gamma) / (1.0 - gamma) * phi
        Vw_cf[i] = D * np.power(w_i, -gamma) * phi
        Vww_cf[i] = -gamma * D * np.power(w_i, -gamma - 1.0) * phi
        Vwx_cf[i] = D * np.power(w_i, -gamma) * phi * (b + C @ x_np[i])

    dev = device
    return {
        "w": torch.tensor(w_np.reshape(-1, 1), dtype=torch.float32, device=dev),
        "x": torch.tensor(x_np, dtype=torch.float32, device=dev),
        "tau": torch.tensor(tau_np.reshape(-1, 1), dtype=torch.float32, device=dev),
        "w_np": w_np, "x_np": x_np,
        "V_cf": V_cf, "Vw_cf": Vw_cf, "Vww_cf": Vww_cf, "Vwx_cf": Vwx_cf,
    }


def sigma_eig_extremes_batch(theta_np, Gamma_np, Q_np):
    """Vectorized extreme eigenvalues of the joint covariance Sigma(theta) =
    [[theta'theta, theta'Gamma], [Gamma'theta, Q]] for a batch of policies.

    theta_np: (P, N). Returns (lam_min, lam_max), each (P,), via one batched
    eigvalsh call. Both ends are needed to check the two-sided uniform
    ellipticity assumption nu*I <= Sigma^alpha <= Lambda*I."""
    P = theta_np.shape[0]
    Mq = Q_np.shape[0]
    Sig = np.zeros((P, 1 + Mq, 1 + Mq))
    Sig[:, 0, 0] = np.einsum("pn,pn->p", theta_np, theta_np)
    cross = theta_np @ Gamma_np                     # (P, M)
    Sig[:, 0, 1:] = cross
    Sig[:, 1:, 0] = cross
    Sig[:, 1:, 1:] = Q_np[None, :, :]
    ev = np.linalg.eigvalsh(Sig)
    return ev[:, 0], ev[:, -1]


def eval_diag_metrics(model, diag, M, N, gamma,
                      Gamma_t, lam0_t, Lam_t, Gamma_np, lam0_np, Lam_np,
                      chunk=4096):
    """One diagnostic pass on the fixed Q_EV set.

    Returns e_n components (sup-norms vs the closed form) plus the margins
    m_ww = min(-V_ww) and M_num = max||lam(x)V_w + Gamma V_wx|| and the
    V_ww guard-activation fraction ON OMEGA_EV. Ellipticity of the frozen
    policy (lambda_min, clip fraction) is measured separately on the
    Q_col held-out points -- see the caller.
    """
    was_training = model.training
    model.eval()
    P = diag["w"].shape[0]
    V_l, Vw_l, Vww_l, Vwx_l = [], [], [], []
    for i in range(0, P, chunk):
        w_b = diag["w"][i:i + chunk].detach().clone().requires_grad_(True)
        x_b = diag["x"][i:i + chunk].detach().clone().requires_grad_(True)
        tau_b = diag["tau"][i:i + chunk]
        V_b = model(w_b, x_b, tau_b)
        V_w = torch.autograd.grad(V_b.sum(), w_b, create_graph=True)[0]
        V_ww = torch.autograd.grad(V_w.sum(), w_b, create_graph=True)[0]
        V_wx = torch.autograd.grad(V_w.sum(), x_b, create_graph=True)[0]
        V_l.append(V_b.detach().cpu()); Vw_l.append(V_w.detach().cpu())
        Vww_l.append(V_ww.detach().cpu()); Vwx_l.append(V_wx.detach().cpu())
    V_m = torch.cat(V_l).numpy().reshape(-1)
    Vw_m = torch.cat(Vw_l).numpy().reshape(-1)
    Vww_m = torch.cat(Vww_l).numpy().reshape(-1)
    Vwx_m = torch.cat(Vwx_l).numpy()
    if was_training:
        model.train()

    # e_n on the fixed set: sup |V - V*| + sup ||bundle - bundle*||_2
    e_V = float(np.max(np.abs(V_m - diag["V_cf"])))
    bundle_err = np.concatenate([
        (Vw_m - diag["Vw_cf"]).reshape(-1, 1),
        (Vww_m - diag["Vww_cf"]).reshape(-1, 1),
        (Vwx_m - diag["Vwx_cf"]),
    ], axis=1)
    e_D = float(np.max(np.linalg.norm(bundle_err, axis=1)))

    # Stability margins (model side)
    m_ww = float(np.min(-Vww_m))
    guard_frac = float(np.mean(Vww_m > -1e-8))
    lam_x = lam0_np[None, :] + diag["x_np"] @ Lam_np.T            # (P, N)
    numer = lam_x * Vw_m[:, None] + Vwx_m @ Gamma_np.T            # (P, N)
    M_num = float(np.max(np.linalg.norm(numer, axis=1)))

    # Table-grade norms on the SAME diagnostic set (same RelL2 norm as the
    # full-dim Table metric, primary margin only): per-outer convergence
    # trajectory without saving per-iteration weights. theta is derived from
    # the ALREADY computed bundle (no extra autograd): the model side reuses
    # the FOC numerator with the training-side V_ww guard; the closed-form
    # side uses the exact (negative) V_ww*.
    rel_l2_V = float(np.linalg.norm(V_m - diag["V_cf"])
                     / max(np.linalg.norm(diag["V_cf"]), 1e-300))
    theta_hat = -numer / np.minimum(Vww_m, -1e-8)[:, None]
    numer_cf = lam_x * diag["Vw_cf"][:, None] + diag["Vwx_cf"] @ Gamma_np.T
    theta_cf = -numer_cf / diag["Vww_cf"][:, None]
    rel_l2_theta = float(np.linalg.norm(theta_hat - theta_cf)
                         / max(np.linalg.norm(theta_cf), 1e-300))

    return {
        "e_V_sup": e_V, "e_bundle_sup": e_D, "e_Xev": e_V + e_D,
        "diag_RelL2_V": rel_l2_V, "diag_RelL2_theta": rel_l2_theta,
        "m_ww": m_ww, "M_num": M_num, "guard_frac_ev": guard_frac,
    }



def eval_fulldim_test_metrics(model, n_points, margins,
                              M, N, gamma, r,
                              Gamma_t, lam0_t, Lam_t,
                              lam0, Lam, Gamma, sol,
                              X_min, X_max, W_min, W_max, tau_max,
                              chunk=4096, base_seed=727):
    """Independent FULL-DIMENSIONAL test evaluation on Omega_ev.

    Unlike the (tau, x_0) visualization slice (fixed w, other factors at
    xbar), this varies ALL coordinates: uniform points over
    (0, tau_max] x [W_ev] x prod_i [X_ev,i]. One base sample in the unit
    cube -- drawn from a FIXED dedicated RNG, identical across runs and
    seeds -- is affinely mapped into every requested evaluation window, so
    nested-window (E9) evaluations use corresponding points.

    Returns {margin: metrics_dict} via compute_metrics against the
    closed-form solution.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(base_seed))
    U = torch.rand(int(n_points), 2 + M, generator=gen).numpy().astype(np.float64)
    eps_tau = 1e-3

    out = {}
    was_training = model.training
    model.eval()
    for m in margins:
        W_lo, W_hi = shrink_bounds(float(W_min), float(W_max), float(m))
        X_lo, X_hi = shrink_bounds(np.asarray(X_min, dtype=np.float64),
                                   np.asarray(X_max, dtype=np.float64), float(m))
        tau_np = eps_tau + U[:, 0] * (tau_max - eps_tau)
        w_np = W_lo + U[:, 1] * (W_hi - W_lo)
        x_np = X_lo[None, :] + U[:, 2:] * (X_hi - X_lo)[None, :]

        P = int(n_points)
        w_t = torch.tensor(w_np.reshape(-1, 1), device=device, dtype=torch.float32, requires_grad=True)
        x_t = torch.tensor(x_np, device=device, dtype=torch.float32, requires_grad=True)
        tau_t = torch.tensor(tau_np.reshape(-1, 1), device=device, dtype=torch.float32)

        V_l, tn_l, my_l, he_l = [], [], [], []
        for i in range(0, P, chunk):
            V_b, _, tn_b, my_b, he_b = compute_optimal_theta_nd(
                model, w_t[i:i + chunk], x_t[i:i + chunk], tau_t[i:i + chunk],
                M, N, gamma, Gamma_t, lam0_t, Lam_t, create_graph=True
            )
            V_l.append(V_b.detach().cpu())
            tn_l.append(tn_b.detach().cpu())
            my_l.append(my_b.detach().cpu())
            he_l.append(he_b.detach().cpu())
        V_pinn = torch.cat(V_l, dim=0).numpy().reshape(-1)
        theta_pinn = torch.cat(tn_l, dim=0).numpy()
        myopic_pinn = torch.cat(my_l, dim=0).numpy()
        hedging_pinn = torch.cat(he_l, dim=0).numpy()

        V_cf = np.zeros(P)
        theta_cf = np.zeros((P, N))
        myopic_cf = np.zeros((P, N))
        hedging_cf = np.zeros((P, N))
        for i in range(P):
            V_cf[i] = closed_form_V(float(tau_np[i]), float(w_np[i]), x_np[i], sol, M, gamma, r)
            _, theta_cf[i], myopic_cf[i], hedging_cf[i] = closed_form_decomposition(
                float(tau_np[i]), float(w_np[i]), x_np[i], sol, M, N, gamma, lam0, Lam, Gamma
            )

        out[m] = compute_metrics(V_pinn, V_cf, theta_pinn, theta_cf,
                                 myopic_pinn, myopic_cf, hedging_pinn, hedging_cf)
    if was_training:
        model.train()
    return out


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

# (legacy plot_slices removed: unused; referenced undefined helper functions)


value_hidden = ARGS.value_hidden
value_depth = ARGS.value_depth

batch_size = ARGS.batch_size
terminal_frac = ARGS.terminal_frac

lr = ARGS.lr
w_terminal = ARGS.w_terminal
eval_epochs = ARGS.eval_epochs  # old resample_every
outer_iters = ARGS.outer_iters
epochs = eval_epochs * outer_iters
w_shape = ARGS.w_shape
w_rra = ARGS.w_rra

# =============================================================================
# 9) Main Execution
# =============================================================================
start = time.time()
if __name__ == "__main__":
    if ARGS.eval_only:
        # Evaluation is unrelated to training-divergence monitoring: no
        # "running" status on the TRAINING status file, no stopper, and the
        # shared stop flag is ignored entirely.
        recorder.write_status_eval("running")
        stopper = None
    else:
        recorder.write_status("running")
        stopper = PDEEarlyStopper(
            threshold=ARGS.pde_stop_threshold,
            start_outer=ARGS.pde_stop_start_outer,
            patience=ARGS.pde_stop_patience,
            stop_flag_path=ARGS.stop_flag_path,
            recorder=recorder,
            run_tag=ARGS.run_tag,
            model_type=ARGS.model_type,
        )
        if stopper.shared_stop_exists():
            info = stopper.mark_from_existing_flag(outer_iter=0, pde_loss=None)
            print(f"[early-stop] shared stop flag already exists. Skipping run. {info}")
            sys.exit(0)

    # Convert parameters to torch
    K_t = torch.tensor(K, device=device, dtype=torch.float32)
    k0_t = torch.tensor(k0, device=device, dtype=torch.float32)
    Q_t = torch.tensor(Q, device=device, dtype=torch.float32)
    Gamma_t = torch.tensor(Gamma, device=device, dtype=torch.float32)
    lam0_t = torch.tensor(lam0, device=device, dtype=torch.float32)
    Lam_t = torch.tensor(Lam, device=device, dtype=torch.float32)

    # Initialize model
    model = ValueNetND(M=M_STATES, hidden=value_hidden, depth=value_depth).to(device)

    if ARGS.eval_only:
        print("\n[eval-only] Skipping training. Loading saved weights for evaluation.")
        loss_history, optimizer, stop_info = [], None, {"stopped_early": False}
        elapsed = 0.0
    else:
        # Training
        loss_history, optimizer, stop_info = train_pinn_nd(
            model, M_STATES, N_ASSETS, gamma, r,
            K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t,
            X_min, X_max, W_min, W_max, tau_max,
            epochs=epochs,
            batch_size=batch_size,
            terminal_frac=terminal_frac,
            lr=lr,
            eval_epochs=eval_epochs,
            outer_iters=outer_iters,
            w_terminal=w_terminal,
            w_concavity=w_shape,
            w_rra=w_rra,
            scheduler_patience=ARGS.scheduler_patience,
            scheduler_factor=ARGS.scheduler_factor,
            scheduler_min_lr=ARGS.scheduler_min_lr,
            save_iterate_every=ARGS.save_iterate_every,
            pres_target=ARGS.pres_target,
            val_points=ARGS.val_points,
            val_terminal_points=ARGS.val_terminal_points,
            val_every=ARGS.val_every,
            # MARKET-seed-derived held-out set: identical across training
            # seeds and both methods.
            val_seed=MARKET_SEED,
            diag_points=ARGS.diag_points,
            diag_margin=parse_eval_margins(ARGS.eval_margin)[0],
            diag_every=ARGS.diag_every,
            timing_mode=ARGS.timing_mode,
            print_every=ARGS.print_every,
            recorder=recorder,
            stopper=stopper,
        )

    end = time.time()
    elapsed = end - start

    # E8: capture the TRAINING GPU peak before evaluation allocates memory.
    _train_gpu_peak = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        _train_gpu_peak = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60

    if stop_info.get("stopped_early", False):
        recorder.write_status("stopped_early", elapsed_sec=elapsed, **stop_info)
        print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")
        sys.exit(0)
    if stop_info.get("target_reached", False):
        print(f"[target-stop] pres target reached at epoch {stop_info.get('epoch_at_stop')} "
              f"(block {stop_info.get('block_at_stop')}); proceeding to evaluation.")

    print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")

    print(f"\n{'='*60}")
    print("Evaluating PINN vs Closed-form...")
    print(f"{'='*60}")
    print(f"  hidden node : {value_hidden}")
    print(f"  num of hidden layers : {value_depth}")
    print(f"  outer_iters  : {outer_iters}")
    print(f"  eval_epochs  : {eval_epochs}")
    print(f"  epochs       : {epochs}")
    print(f"  batch size   : {batch_size}")
    print(f"  initial lr   : {lr}")
    print(f"  T.C weight   : {w_terminal}")
    print(f"  Seed         : {SEED}")
    print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")
    print(f"{'='*70}")

    # FINAL-ITERATE POLICY: evaluation always uses the final training state.
    # best is loaded only as a legacy fallback (older runs without final).
    final_weight_path = os.path.join(weight_dir, "value_net_final.pt")
    best_weight_path = os.path.join(weight_dir, "value_net_best.pt")
    last_weight_path = os.path.join(weight_dir, "value_net_last.pt")
    if ARGS.eval_only:
        if os.path.exists(final_weight_path):
            model.load_state_dict(torch.load(final_weight_path, map_location=device))
            print(f"[eval-only] Loaded FINAL state: {final_weight_path}")
        elif os.path.exists(last_weight_path):
            model.load_state_dict(torch.load(last_weight_path, map_location=device))
            print(f"[eval-only][warn] final weight not found; loaded last: {last_weight_path}")
        elif os.path.exists(best_weight_path):
            model.load_state_dict(torch.load(best_weight_path, map_location=device))
            print(f"[eval-only][warn] final/last not found; loaded BEST (legacy run): {best_weight_path}")
        else:
            msg = f"no saved weights (final/last/best) under {weight_dir}"
            print(f"[eval-only][FATAL] {msg}")
            recorder.mark_failed_eval(reason=msg)
            sys.exit(1)
    # (Normal runs: the in-memory model already IS the final state.)

    if SKIP_EVAL:
        # Evaluation fully skipped (opt-in). No metrics.csv is produced.
        if ARGS.eval_only:
            recorder.mark_success_eval(elapsed_sec=elapsed, final_weight_path=final_weight_path,
                                       skipped_eval=True)
        else:
            recorder.mark_success(elapsed_sec=elapsed, final_weight_path=final_weight_path,
                                  best_weight_path=best_weight_path, skipped_eval=True,
                                  train_gpu_peak_mem_bytes=_train_gpu_peak,
                                  timing_mode=bool(ARGS.timing_mode))
        sys.exit(0)

    # Plot loss history (figure only; suppressed under --skip-figures/--skip-plots).
    if not SKIP_FIGURES:
        plot_loss_history(
            loss_history,
            save_path=os.path.join(output_dir, f"loss_history_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {eval_epochs}-eval epoch).png"),
            show=True
        )

    # Evaluation
    print(f"\n{'='*60}")
    print("Evaluating PINN vs Closed-form...")
    print(f"{'='*60}")

    # (x-axis = state X, y-axis = time-to-maturity tau), holding other states fixed
    dimX = 0
    x_fixed = xbar.copy()  # Fix other state dimensions at long-run mean

    W_levels = parse_w_levels(ARGS.w_levels)
    N_tau, N_X = ARGS.n_tau, ARGS.n_x

    metric_fields = ["timestamp", "model_type", "run_tag", "scope", "eval_margin", "metric", "value"]

    # Evaluation windows Omega_ev (per-side shrink of each SPATIAL axis; tau
    # keeps its full range). First margin = primary (headline metrics, plots);
    # every listed margin is re-evaluated with the SAME trained network and
    # recorded to metrics.csv -- the E9 window-sensitivity study for free.
    EVAL_MARGINS = parse_eval_margins(ARGS.eval_margin)
    primary_margin = EVAL_MARGINS[0]
    print(f"\nEvaluation windows (per-side margins): {EVAL_MARGINS} (primary={primary_margin})")
    for _m in EVAL_MARGINS:
        _w_lo, _w_hi = shrink_bounds(W_min, W_max, _m)
        for _w_test in W_levels:
            if not (_w_lo <= _w_test <= _w_hi):
                print(f"[warn] w_level {_w_test} lies OUTSIDE W_ev=[{_w_lo:.4f}, {_w_hi:.4f}] "
                      f"for eval_margin={_m}; that slice is not inside the evaluation window.")

    # Eval-only metrics are written ATOMICALLY: everything goes to a tmp
    # file and only replaces metrics.csv after the evaluation SUCCEEDS. A
    # mid-evaluation crash therefore leaves the existing (training) metrics
    # untouched instead of a deleted/partial file next to a training
    # _SUCCESS marker.
    _metrics_final_path = recorder.metrics_csv
    if ARGS.eval_only:
        recorder.metrics_csv = _metrics_final_path + ".eval_tmp"
        if os.path.exists(recorder.metrics_csv):
            os.remove(recorder.metrics_csv)  # stale tmp from an older failed eval
        print(f"[eval-only] Recording metrics to {recorder.metrics_csv} (atomic swap on success).")

    # Independent FULL-DIMENSIONAL test evaluation on Omega_ev (Table / E9).
    # All coordinates vary; the same base points are mapped into every margin
    # window so nested-window results are directly comparable. Printed BEFORE
    # the (tau, x_0) grid slices; per-asset metrics go to metrics.csv only.
    if ARGS.test_points and ARGS.test_points > 0:
        fulldim_metrics = eval_fulldim_test_metrics(
            model, ARGS.test_points, EVAL_MARGINS,
            M_STATES, N_ASSETS, gamma, r,
            Gamma_t, lam0_t, Lam_t,
            lam0, Lam, Gamma, cf_sol,
            X_min, X_max, W_min, W_max, tau_max,
        )
        for _m, mets in sorted(fulldim_metrics.items()):
            _is_primary = (_m == primary_margin)
            print(f"\n--- Full-dimensional Omega_ev test: {ARGS.test_points} points, "
                  f"eval_margin={_m:.2f}{' (primary)' if _is_primary else ''} ---")
            print("Metrics:")
            print("-" * 40)
            rows = []
            for k, v in mets.items():
                if not k.rsplit("_", 1)[-1].isdigit():
                    print(f"  {k}: {v:.6e}")
                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag,
                    "scope": "fulldim",
                    "eval_margin": _m,
                    "metric": k,
                    "value": float(v),
                })
            append_csv_rows(recorder.metrics_csv, rows, metric_fields)

    # The (tau, x_0) grid slice is retained for PLOTS ONLY, on the primary
    # evaluation window. Slice metrics were removed: all reported numbers come
    # from the full-dimensional Omega_ev test above. Under --skip-figures the
    # slice loop is skipped entirely (figures only; metrics already written).
    for w_test in ([] if SKIP_FIGURES else W_levels):
        X_ev_min, X_ev_max = shrink_bounds(X_min, X_max, primary_margin)
        print(f"\n--- Grid slice for plots: w={w_test:.2f}, eval_margin={primary_margin:.2f} ---")

        # PINN evaluation on (tau, X) grid restricted to Omega_ev
        tau_grid, X_grid, V_pinn, theta_norm_pinn, myopic_pinn, hedging_pinn = \
            eval_pinn_on_tau_X_grid(
                model, w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma,
                Gamma_t=Gamma_t, lam0_t=lam0_t, Lam_t=Lam_t,
                X_min=X_ev_min, X_max=X_ev_max, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )

        # Closed-form evaluation on the same Omega_ev grid
        _, _, V_cf, theta_norm_cf, myopic_cf, hedging_cf = \
            eval_closed_form_on_tau_X_grid(
                w_test, dimX=dimX, x_fixed=x_fixed,
                M=M_STATES, N=N_ASSETS, gamma=gamma, r=r,
                lam0=lam0, Lam=Lam, Gamma=Gamma, sol=cf_sol,
                X_min=X_ev_min, X_max=X_ev_max, tau_min=tau_min, tau_max=tau_max,
                N_tau=N_tau, N_X=N_X
            )

        # Plots
        plot_value_comparison_tauX(
            tau_grid, X_grid, V_pinn, V_cf,
            dimX=dimX, w_fixed=w_test,
            save_path=os.path.join(output_dir, f"value_{w_test:.2f}_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {eval_epochs}-eval epoch).png"),
            show=True,
            xlabel=f"risk premium X",
            ylabel=r"$	au$"
        )

        plot_portfolio_comparison_tauX(
            tau_grid, X_grid,
            theta_norm_pinn, theta_norm_cf,
            myopic_pinn, myopic_cf,
            hedging_pinn, hedging_cf,
            dimX=dimX, w_fixed=w_test, N_ASSETS=N_ASSETS,
            save_path=os.path.join(output_dir, f"portfolio_w{w_test:.2f}_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {eval_epochs}-eval epoch).png"),
            show=True, only_hedge=False,
            xlabel=f"risk premium X",
            ylabel=r"$	au$",
            max_assets=10, sort_by_range=True
        )


    _eval_gpu_peak = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        _eval_gpu_peak = int(torch.cuda.max_memory_allocated(device))
    if ARGS.eval_only:
        # Atomic commit: back up the training metrics once, then replace.
        if os.path.exists(recorder.metrics_csv):
            _bak = _metrics_final_path + ".bak_train"
            if os.path.exists(_metrics_final_path) and not os.path.exists(_bak):
                import shutil as _sh
                _sh.copyfile(_metrics_final_path, _bak)
                print(f"[eval-only] Backed up training metrics to {_bak}")
            os.replace(recorder.metrics_csv, _metrics_final_path)
            recorder.metrics_csv = _metrics_final_path
            print(f"[eval-only] Committed metrics -> {_metrics_final_path}")
        recorder.mark_success_eval(elapsed_sec=elapsed, final_weight_path=final_weight_path,
                                   eval_margins=EVAL_MARGINS,
                                   eval_gpu_peak_mem_bytes=_eval_gpu_peak)
    else:
        recorder.mark_success(elapsed_sec=elapsed, final_weight_path=final_weight_path,
                              best_weight_path=best_weight_path, skipped_figures=bool(SKIP_FIGURES),
                              pres_target=ARGS.pres_target,
                              target_reached=bool(stop_info.get("target_reached", False)),
                              epoch_at_stop=stop_info.get("epoch_at_stop"),
                              achieved_pres=stop_info.get("achieved_pres"),
                              total_optimizer_steps=stop_info.get("total_optimizer_steps"),
                              train_wall_sec=elapsed,
                              timing_mode=bool(ARGS.timing_mode),
                              train_gpu_peak_mem_bytes=_train_gpu_peak,
                              eval_gpu_peak_mem_bytes=_eval_gpu_peak,
                              eval_margins=EVAL_MARGINS)
