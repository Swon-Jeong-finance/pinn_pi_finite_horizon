"""
Multi-dimensional Liu Portfolio Problem - PINN Solution
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
         : Liu (2007), "Portfolio Selection in Stochastic Environments", RFS
"""

import time
import os
import sys
import math
import argparse
import copy
import csv
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
from joint_market_setup_dirichlet import (
    generate_joint_market_params,
    JointMarketParams,
    cholesky_solve,
    rho_snapshot_metadata,
    validate_market_snapshot,
)
from experiment_utils import (
    add_common_experiment_args, parse_w_levels, resolve_device, set_reproducibility,
    ExperimentRecorder, PDEEarlyStopper, append_csv_rows, save_json, none_or_float,
    parse_eval_margins, shrink_bounds, pres_from_mse, safe_concave_vww, VWW_GUARD,
    normalized_control_stats, validate_eval_only_config,
)
from liu_risk_premium import (
    RISK_PREMIUM_MODES,
    has_affine_reference,
    risk_premium_numpy,
    risk_premium_torch,
    validate_risk_premium_config,
)


# =============================================================================
# 0) CLI + Reproducibility + Device
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Liu ND PIPINN experiment runner")
    add_common_experiment_args(parser, model_type_default="pipinn")

    parser.add_argument("--theta-init-method", type=str, default="myopic", choices=["myopic", "zero", "closed_form"])
    parser.add_argument("--theta-init-scale", type=float, default=1.0,
                        help="theta_0 = scale * theta_init(method). Nondegenerate fixed-point "
                             "perturbation for the contraction pilot (e.g. 0.5, 1.5); 1.0 = unchanged.")
    parser.add_argument("--theta-clip-abs", type=none_or_float, default=None)
    parser.add_argument("--risk-premium-mode", choices=RISK_PREMIUM_MODES, default="affine",
                        help="affine benchmark or the paper's aligned tanh perturbation.")
    parser.add_argument("--nonaffine-eps", type=float, default=0.0,
                        help="epsilon >= 0 in lambda_eps; mode=tanh, eps=0 is the paired affine baseline.")
    parser.add_argument("--nonaffine-loading-scale", type=float, default=1.0,
                        help="Psi = scale * Lambda in the aligned tanh perturbation (paper default: 1).")
    # Previously hardcoded at the solver call site; exposed so bash overrides take effect.
    # LR schedule modes:
    #   inner_plateau (default): scheduler is re-created at the START of every
    #       outer iteration with the same initial LR and steps on every inner
    #       epoch (plateau over ONE frozen PDE solve).
    #   outer_plateau: legacy behavior -- one persistent scheduler stepped once
    #       per outer iteration on the last inner loss (cross-PDE plateau).
    #   fixed: constant LR, no scheduler.
    parser.add_argument("--lr-schedule", type=str, default="inner_plateau",
                        choices=["inner_plateau", "outer_plateau", "fixed", "carry_plateau"])
    parser.add_argument("--carry-lr-min", type=float, default=1e-5,
                        help="carry_plateau: lower clamp for the carried outer-start LR.")
    parser.add_argument("--carry-lr-max", type=float, default=5e-5,
                        help="carry_plateau: upper clamp for the carried outer-start LR (outer 1 uses --lr).")
    parser.add_argument("--adam-reset", type=str, default="keep", choices=["keep", "full"],
                        help="At each outer iter (inner_plateau mode): 'keep' resets only LR/scheduler, "
                             "'full' also re-creates the Adam optimizer (fresh moments).")
    parser.add_argument("--scheduler-patience", type=int, default=None,
                        help="Plateau patience. Default: 10 (inner epochs) for inner_plateau, 25 (outer iters) for outer_plateau.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-8)
    parser.add_argument("--print-every-outer", type=int, default=20)
    parser.add_argument("--print-every-eval", type=int, default=0,
                        help="Print inner (policy-evaluation) progress every k epochs; 0 = off. "
                             "1 shows every inner epoch, with p_res on check epochs.")
    parser.add_argument("--verbose-detail", action="store_true",
                        help="(deprecated for inner prints -- use --print-every-eval)")
    # Inner held-out BEST selection (within one frozen-PDE solve only; the
    # OUTER-level reported model remains the final PI iterate). A small
    # dedicated selection set is checked at inner epoch 0 and every
    # --sel-every epochs; the state with the lowest held-out p_res is stored
    # (post-update state matched with post-update validation) and RESTORED at
    # the end of the evaluation, BEFORE policy improvement and the big-audit
    # p_res measurement. --sel-patience consecutive non-improving checks end
    # the policy evaluation early. --inner-best-restore 0 = legacy final
    # inner state.
    parser.add_argument("--inner-best-restore", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sel-points", type=int, default=10000)
    parser.add_argument("--sel-terminal-points", type=int, default=2000)
    parser.add_argument("--sel-every", type=int, default=50)
    parser.add_argument("--sel-patience", type=int, default=6)
    parser.add_argument("--pe-resample-every", type=int, default=0,
                        help="Within-evaluation collocation resampling period (inner epochs); "
                             "the policy function stays frozen (source = previous iterate / "
                             "analytic init). 0 = single fixed batch per policy evaluation.")
    parser.add_argument("--e3b-checkpoints", action="store_true",
                        help="FD-reference schedule: save every completed outer iterate.")
    return parser


ARGS = build_arg_parser().parse_args()
RISK_PREMIUM_MODE = str(ARGS.risk_premium_mode)
NONAFFINE_EPS = float(ARGS.nonaffine_eps)
NONAFFINE_LOADING_SCALE = float(ARGS.nonaffine_loading_scale)
try:
    validate_risk_premium_config(
        RISK_PREMIUM_MODE, NONAFFINE_EPS, NONAFFINE_LOADING_SCALE
    )
except ValueError as exc:
    raise SystemExit(f"[config error] {exc}") from exc
HAS_AFFINE_REFERENCE = has_affine_reference(RISK_PREMIUM_MODE, NONAFFINE_EPS)
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
    hjb_joint = params.diag.get("identity_joint_corr")

    print(f"\n[Joint Market Parameters] n={n}, k={k}")
    print(f"  sigma (asset vols)   : min={params.sigma.min():.4f}, max={params.sigma.max():.4f}, mean={params.sigma.mean():.4f}")
    if k > 0:
        print(f"  eta   (state vols)   : min={params.eta.min():.4f}, max={params.eta.max():.4f}, mean={params.eta.mean():.4f}")

    print(f"\n[Numerical Stability Diagnostics]")
    print(f"  joint cond(C)        : {jc['cond']:.2f}")
    print(f"  joint min eig(C)     : {jc['min_eig']:.2e}")
    print(f"  joint max|rho_ij|    : {jc['max_abs_rho']:.4f}")
    print(f"  joint shrink alpha   : {jc['alpha_used']:.4f}")
    if hjb_joint is not None:
        print(f"  HJB rho spectral norm: {hjb_joint['rho_spectral_norm']:.6f}")
        print(f"  HJB joint min eig    : {hjb_joint['min_eig']:.2e}")
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

_default_exp_name = f"kim_omberg_{N_ASSETS}asset-{M_STATES}state"
if RISK_PREMIUM_MODE == "tanh":
    _eps_token = format(NONAFFINE_EPS, ".12g").replace("-", "m").replace(".", "p")
    _default_exp_name += f"_tanh_eps{_eps_token}"
weight_dir = ARGS.weight_root or f"weights/pi-pinn/{_default_exp_name}"
os.makedirs(weight_dir, exist_ok=True)
output_dir = ARGS.output_root or f"outputs/pi-pinn/{_default_exp_name}"
os.makedirs(output_dir, exist_ok=True)

# Validate launch-only invariants before quarantining an older run.  Invalid
# command lines must fail without moving valid checkpoints or provenance.
if ARGS.pres_target is not None and (not ARGS.val_points or ARGS.val_points <= 0):
    raise SystemExit("[config error] --pres-target requires --val-points > 0 (held-out set is the stopping rule).")
if (
    RISK_PREMIUM_MODE == "tanh"
    and NONAFFINE_EPS > 0.0
    and ARGS.theta_init_method == "closed_form"
):
    raise SystemExit(
        "[config error] --theta-init-method=closed_form is unavailable for "
        "non-affine eps>0; use myopic or zero."
    )
if (
    not ARGS.eval_only
    and ARGS.stop_flag_path
    and os.path.exists(ARGS.stop_flag_path)
):
    print(
        f"[early-stop] shared stop flag already exists; preserving the current "
        f"run artifacts unchanged: {ARGS.stop_flag_path}"
    )
    raise SystemExit(0)

recorder = ExperimentRecorder(output_dir, weight_dir, ARGS)
if ARGS.eval_only:
    # Delay writing config_eval.json until the immutable training config and
    # complete market snapshot have both passed provenance validation.
    pass
else:
    # Quarantine the complete previous attempt before creating the new
    # canonical config/checkpoint namespace.
    recorder.rotate_training_logs()
    recorder.save_config()

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
SigmaX = params.SigmaZ          # (M, M) state diffusion under identity shocks
# The source generator samples C=[[Psi,rho_raw],[rho_raw.T,Phi_Z]].  The HJB
# uses identity asset/state Brownian covariance blocks, so its compatible
# cross-correlation must be whitened rather than copied from C verbatim.
rho = params.rho_canonical       # (N, M) Psi^{-1/2} rho_raw Phi_Z^{-1/2}
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

if ARGS.eval_only:
    _market_path = os.path.join(output_dir, "market_params.npz")
    _config_path = os.path.join(output_dir, "config.json")
    _critical_eval_keys = (
        "model_type", "n_assets", "m_states", "seed", "market_seed",
        "tau_max", "w_min", "w_max", "gamma", "r", "x_range_scale",
        "dirichlet_concentration", "alpha_scale", "value_hidden", "value_depth",
        "risk_premium_mode", "nonaffine_eps", "nonaffine_loading_scale",
        "theta_clip_abs",
    )
    _expected_market = {
        "K": K, "xbar": xbar, "SigmaX": SigmaX, "rho": rho, "Lam": Lam,
        "Q": Q, "Gamma": Gamma, "k0": k0, "lam0": lam0,
        "X_min": X_min, "X_max": X_max, "eta": eta,
        "gamma": np.array([gamma]), "r": np.array([r]),
        "tau_max": np.array([tau_max]), "W_min": np.array([W_min]),
        "W_max": np.array([W_max]), "seed": np.array([SEED]),
        "market_seed": np.array([MARKET_SEED]),
        **rho_snapshot_metadata(params),
    }
    try:
        validate_eval_only_config(_config_path, ARGS, _critical_eval_keys)
        if not os.path.isfile(_market_path):
            raise ValueError(f"missing market snapshot: {_market_path}")
        with np.load(_market_path, allow_pickle=False) as _saved_market:
            validate_market_snapshot(
                _saved_market,
                expected=_expected_market,
                require_canonical_metadata=True,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"[eval-only provenance error] {exc}") from exc
    recorder.save_config_eval()

try:
    validate_risk_premium_config(
        RISK_PREMIUM_MODE,
        NONAFFINE_EPS,
        NONAFFINE_LOADING_SCALE,
        state_scale=eta,
    )
except ValueError as exc:
    raise SystemExit(f"[config error] {exc}") from exc

# Fixed coefficient tensors used by every actual-model path.  D=diag(eta)
# and Psi=loading_scale*Lambda are deterministic functions of the saved market
# snapshot, so market_params.npz remains identical across epsilon runs.
RISK_XBAR_T = torch.tensor(xbar, device=device, dtype=torch.float32)
RISK_STATE_SCALE_T = torch.tensor(eta, device=device, dtype=torch.float32)


def actual_risk_premium_torch(x, lam0_t, Lam_t):
    """Risk premium of the model being trained (affine or tanh)."""

    return risk_premium_torch(
        x,
        lam0_t,
        Lam_t,
        mode=RISK_PREMIUM_MODE,
        eps=NONAFFINE_EPS,
        xbar=RISK_XBAR_T,
        state_scale=RISK_STATE_SCALE_T,
        loading_scale=NONAFFINE_LOADING_SCALE,
    )


def actual_risk_premium_numpy(x, lam0_np=lam0, Lam_np=Lam):
    """NumPy counterpart used by model-only diagnostics."""

    return risk_premium_numpy(
        x,
        lam0_np,
        Lam_np,
        mode=RISK_PREMIUM_MODE,
        eps=NONAFFINE_EPS,
        xbar=xbar,
        state_scale=eta,
        loading_scale=NONAFFINE_LOADING_SCALE,
    )

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
print(
    f"Risk premium: mode={RISK_PREMIUM_MODE}, eps={NONAFFINE_EPS:g}, "
    f"Psi={NONAFFINE_LOADING_SCALE:g}*Lambda, D=diag(eta), "
    f"affine_reference={'yes' if HAS_AFFINE_REFERENCE else 'no'}"
)
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
# (Training runs only: eval-only must leave the training snapshots frozen.)
if not ARGS.eval_only:
    recorder.save_market_snapshot(
        K=K, xbar=xbar, SigmaX=SigmaX, rho=rho, Lam=Lam, Q=Q, Gamma=Gamma,
        k0=k0, lam0=lam0, X_min=X_min, X_max=X_max, eta=eta,
        gamma=np.array([gamma]), r=np.array([r]), tau_max=np.array([tau_max]),
        W_min=np.array([W_min]), W_max=np.array([W_max]), seed=np.array([SEED]), market_seed=np.array([MARKET_SEED]),
        **rho_snapshot_metadata(params),
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
    # Actual model coefficient: affine benchmark or aligned tanh perturbation.
    lam_x = actual_risk_premium_torch(x, lam0_t, Lam_t)  # (batch, N)
    
    # Γ V_wx → (batch, N)
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)  # (batch, N)
    
    # numerator = ‖λ(x) V_w + Γ V_wx‖²
    combined = lam_x * V_w + Gamma_Vwx  # (batch, N)
    numerator = torch.sum(combined ** 2, dim=1, keepdim=True)  # (batch, 1)
    
    # Guard V_ww itself so this path uses the same effective threshold as
    # control extraction (rather than clamping 2*V_ww at a different one).
    V_ww_safe = safe_concave_vww(V_ww)
    term5 = -numerator / (2.0 * V_ww_safe)
    
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
    
    lam_x = actual_risk_premium_torch(x, lam0_t, Lam_t)  # (batch, N)
    
    # Γ V_wx
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)  # (batch, N)
    
    # θ* = -(λ(x)V_w + Γ V_wx) / V_ww
    numerator = lam_x * V_w + Gamma_Vwx
    # Concavity-respecting guard: V_ww must be negative for the FOC to be a
    # maximizer, so clamp from ABOVE at -1e-8. (The previous
    # sign(V_ww)*1e-8 + 1e-10 form gave +1e-10 at V_ww == 0: wrong sign and
    # 100x smaller than intended, letting the control blow up.)
    V_ww_safe = safe_concave_vww(V_ww)
    theta = -numerator / V_ww_safe  # (batch, N)
    theta_norm = theta / w  # (batch, N)
    
    # Myopic (closed-form, independent of V)
    # myopic_norm = lam_x / gamma  # (batch, N)
    pinn_coeff = -V_w / (w * V_ww_safe)  # (batch, 1)
    myopic_norm = pinn_coeff * lam_x  # (batch, N)
    
    # Hedging = Total - Myopic (PINN-derived)
    hedging_norm = theta_norm - myopic_norm  # (batch, N)
    
    return V, theta, theta_norm, myopic_norm, hedging_norm


# =============================================================================
# 6) Training
# =============================================================================

# =============================================================================
# 6B) PI-PINN for ND Liu (2007)
# =============================================================================
# We convert the fully nonlinear HJB into a *linear* PDE by freezing the policy θ_n,
# then update θ_{n+1} via the FOC (Hamiltonian maximization):
#
#   θ*(τ,w,x) = - ( λ(x) V_w + Γ V_wx ) / V_ww
#
# Under fixed θ_n, the linear PDE residual is:
#   0 = -V_τ + r w V_w + (k0 - Kx)^T V_x + 1/2 tr(Q V_xx)
#       + (θ_n^T λ(x)) V_w + 1/2 (θ_n^T θ_n) V_ww + θ_n^T Γ V_wx


def linear_pde_residual_nd(
    value_net,
    theta_n,
    w, x, tau,
    M, N,
    gamma, r,
    K_t, k0_t, Q_t, Gamma_t, lam0_t, Lam_t
):
    """
    Linear PDE residual under fixed policy θ_n (batch, N).

    Residual:
        -V_τ + r w V_w + (k0 - Kx)^T V_x + 1/2 tr(Q V_xx)
        + (θ^T λ(x)) V_w + 1/2 (θ^T θ) V_ww + θ^T Γ V_wx = 0
    """
    V, V_tau, V_w, V_x, V_ww, V_xx, V_wx = compute_derivatives_nd(value_net, w, x, tau, M)

    # -V_tau
    term1 = -V_tau

    # r w V_w
    term2 = r * w * V_w

    # (k0 - Kx)^T V_x
    drift = k0_t.unsqueeze(0) - torch.einsum('ij,bj->bi', K_t, x)  # (batch, M)
    term3 = torch.einsum('bi,bi->b', drift, V_x).unsqueeze(1)  # (batch,1)

    # 1/2 tr(Q V_xx)
    term4 = 0.5 * torch.einsum('ij,bij->b', Q_t, V_xx).unsqueeze(1)  # (batch,1)

    lam_x = actual_risk_premium_torch(x, lam0_t, Lam_t)  # (batch, N)

    # (θ^T λ(x)) V_w
    theta_dot_lam = torch.sum(theta_n * lam_x, dim=1, keepdim=True)  # (batch,1)
    term5 = theta_dot_lam * V_w

    # 1/2 (θ^T θ) V_ww
    theta_sq = torch.sum(theta_n ** 2, dim=1, keepdim=True)  # (batch,1)
    term6 = 0.5 * theta_sq * V_ww

    # θ^T Γ V_wx
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)  # (batch,N)
    theta_dot_GammaVwx = torch.sum(theta_n * Gamma_Vwx, dim=1, keepdim=True)  # (batch,1)
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
    """
    FOC / Hamiltonian maximizer for ND case:

        θ*(τ,w,x) = - ( λ(x) V_w + Γ V_wx ) / V_ww

    Args:
        theta_clip_abs: if not None, clamp each component to [-theta_clip_abs, +theta_clip_abs]
    """
    V = value_net(w, x, tau)
    ones = torch.ones_like(V)

    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w), create_graph=True, retain_graph=True)[0]
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w), create_graph=True, retain_graph=True)[0]  # (batch,M)

    lam_x = actual_risk_premium_torch(x, lam0_t, Lam_t)  # (batch,N)
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)               # (batch,N)

    numerator = lam_x * V_w + Gamma_Vwx                                 # (batch,N)

    # Concavity-respecting guard: V_ww must be negative for the FOC to be a
    # maximizer, so clamp from ABOVE at -1e-8. (The previous
    # sign(V_ww)*1e-8 + 1e-10 form gave +1e-10 at V_ww == 0: wrong sign and
    # 100x smaller than intended, letting the control blow up.)
    V_ww_safe = safe_concave_vww(V_ww)
    theta = -numerator / V_ww_safe                                      # (batch,N)

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
        scheduler_patience=25,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-8,
        lr_schedule="outer_plateau",
        adam_reset="keep",
        carry_lr_min=1e-5,
        carry_lr_max=5e-5,
        theta_clip_abs=5.0,
        device=device
    ):
        self.device = device
        self.M = M
        self.N = N
        self.gamma = gamma
        self.r = r

        # problem tensors
        self.K_t = K_t
        self.k0_t = k0_t
        self.Q_t = Q_t
        self.Gamma_t = Gamma_t
        self.lam0_t = lam0_t
        self.Lam_t = Lam_t

        self.theta_clip_abs = theta_clip_abs

        self.value_net = ValueNetND(M=M, hidden=value_hidden, depth=value_depth).to(device)
        self.initial_lr = lr
        self.lr_schedule = str(lr_schedule)
        self.adam_reset = str(adam_reset)
        self.carry_lr_min = float(carry_lr_min)
        self.carry_lr_max = float(carry_lr_max)
        self._outer_count = 0
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr

        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        if self.lr_schedule == "fixed":
            self.scheduler = None
        else:
            self.scheduler = self._make_scheduler()

    def _effective_min_lr(self):
        """LR decay floor. Under carry_plateau the floor is
        max(scheduler_min_lr, carry_lr_min) -- carry_lr_min is a REAL floor
        again (a bare scheduler_min_lr=1e-8 would otherwise let the carried
        LR sink to 1e-8 across outers). Other schedules keep the legacy
        scheduler_min_lr. Used identically by the scheduler, the inner-best
        restore, and prepare_optimizer_for_outer so the three floors can
        never disagree."""
        if self.lr_schedule == "carry_plateau":
            return max(float(self.scheduler_min_lr), float(self.carry_lr_min))
        return float(self.scheduler_min_lr)

    def _make_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=self.scheduler_factor, patience=self.scheduler_patience,
            min_lr=self._effective_min_lr()
        )

    def prepare_optimizer_for_outer(self):
        """Called at the START of every outer iteration.

        inner_plateau: restart the LR protocol so every policy evaluation
        begins identically -- LR back to initial_lr and a FRESH plateau
        scheduler acting on inner-epoch losses. adam_reset='full' also
        re-creates the Adam optimizer (fresh moments); 'keep' retains the
        moment estimates as a warm start.
        carry_plateau: keep the Adam moments AND the learned LR level -- the
        outer-start LR is the CARRIED current LR clamped to
        [carry_lr_min, carry_lr_max] (outer 1 starts at initial_lr), and only
        the plateau-scheduler STATE restarts for the new frozen PDE. This
        avoids re-climbing from initial_lr at every outer while never
        inheriting a dead LR (e.g. min_lr) from a previous evaluation.
        outer_plateau / fixed: no per-outer action (legacy persistent
        scheduler / constant LR).
        """
        self._outer_count += 1
        if self.lr_schedule == "carry_plateau":
            if self._outer_count <= 1:
                outer_lr = self.initial_lr
                if self.adam_reset == "full":
                    print("[warn] carry_plateau is designed for adam_reset=keep; "
                          "'full' discards the restored best-checkpoint moments each outer.")
            else:
                # The previous evaluation's restore already set this LR to
                # max(scheduler_min_lr, min(LR_best, LR_end)), so the carried
                # value is non-increasing across outers by construction. Cap
                # from above only, never raise; the decay floor is
                # scheduler_min_lr.
                carried = float(self.optimizer.param_groups[0]["lr"])
                outer_lr = min(self.carry_lr_max,
                               max(self._effective_min_lr(), carried))
            if self.adam_reset == "full":
                self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=outer_lr)
            else:
                for g in self.optimizer.param_groups:
                    g["lr"] = outer_lr
            self.scheduler = self._make_scheduler()
            return
        if self.lr_schedule != "inner_plateau":
            return
        if self.adam_reset == "full":
            self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.initial_lr)
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = self.initial_lr
        self.scheduler = self._make_scheduler()

    def policy_improvement_chunked(self, w, x, tau, chunk=4096):
        """policy_improvement in chunks (for large held-out sets)."""
        outs = []
        n = w.shape[0]
        for i in range(0, n, chunk):
            outs.append(self.policy_improvement(w[i:i + chunk], x[i:i + chunk], tau[i:i + chunk]))
        return torch.cat(outs, dim=0)

    def evaluate_heldout_pres(self, theta_val, val_set, chunk=4096):
        """Held-out residual level p_res = RMS(frozen-PDE residual on Q_col)
        + RMS(terminal mismatch on Omega_col), for the CURRENT value_net and
        the frozen policy theta_val on the validation points."""
        self.value_net.eval()
        n = val_set["w_int"].shape[0]
        sq_sum = 0.0
        for i in range(0, n, chunk):
            w_b = val_set["w_int"][i:i + chunk].detach().clone().requires_grad_(True)
            x_b = val_set["x_int"][i:i + chunk].detach().clone().requires_grad_(True)
            tau_b = val_set["tau_int"][i:i + chunk].detach().clone().requires_grad_(True)
            residual, _, _, _, _ = linear_pde_residual_nd(
                self.value_net, theta_val[i:i + chunk].detach(),
                w_b, x_b, tau_b,
                self.M, self.N, self.gamma, self.r,
                self.K_t, self.k0_t, self.Q_t,
                self.Gamma_t, self.lam0_t, self.Lam_t
            )
            sq_sum += float(torch.sum(residual.detach() ** 2).item())
        pde_rms = float(np.sqrt(sq_sum / max(n, 1)))

        with torch.no_grad():
            V_T_pred = self.value_net(val_set["w_term"], val_set["x_term"], val_set["tau_term"])
            V_T_true = V_terminal(val_set["w_term"], self.gamma)
            term_rms = float(torch.sqrt(torch.mean((V_T_pred - V_T_true) ** 2)).item())

        self.value_net.train()
        return pde_rms, term_rms, pde_rms + term_rms

    def initialize_theta(self, w, x, tau, method="myopic"):
        """Initial policy θ_0.
        - myopic: θ = (w/γ) λ(x)
        - zero: θ = 0
        - closed_form: θ from ODE (expensive; for debugging)
        """
        if method == "myopic":
            lam_x = actual_risk_premium_torch(x, self.lam0_t, self.Lam_t)
            theta = (w / self.gamma) * lam_x
        elif method == "zero":
            theta = torch.zeros((w.shape[0], self.N), device=self.device, dtype=torch.float32)
        elif method == "closed_form":
            if not HAS_AFFINE_REFERENCE:
                raise ValueError(
                    "theta-init-method=closed_form is unavailable for non-affine eps>0; "
                    "use myopic (paper default) or zero"
                )
            # WARNING: slow (loop). Use only for sanity checks.
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

        # Contraction-pilot perturbation: theta_0 = scale * theta_init(method).
        # Apply before the numerical clip so the clip remains the outermost
        # safeguard and the non-affine myopic structure is preserved.
        _scale = float(getattr(self, "theta_init_scale", 1.0))
        if _scale != 1.0:
            theta = theta * _scale
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
        w_rra=0.0,
        print_every=200,
        pres_target=None,
        val_every=1,
        val_fn=None,
        sel_fn=None,
        sel_every=50,
        sel_patience=6,
        restore_best=False,
        resample_every=0,
        resample_fn=None,
    ):
        """Train V_n by solving the LINEAR PDE with fixed θ_n.

        If pres_target is set and val_fn is provided, the held-out residual
        level p_res is checked every val_every epochs (and once BEFORE the
        first gradient step); training stops as soon as p_res <= pres_target.
        In inner_plateau mode the (freshly reset) scheduler steps on every
        inner epoch. Returns an extra eval_info dict with the end-of-
        evaluation held-out measurement (the reported p_res,n)."""
        loss_hist = []
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        track_best = getattr(self, "_track_best", True)
        light_hist = getattr(self, "_timing_mode", False)  # keep only the last row

        theta_n_fixed = theta_n.detach()

        target_reached = False
        epochs_used = 0
        last_val = None          # (pde_rms, term_rms, pres)
        last_val_epoch = -1

        # Inner held-out BEST selection on a small dedicated selection set.
        # POST-update state copies are matched with the POST-update validation
        # value that selected them (the loss-based diagnostic best has a
        # one-step pre/post mismatch and is untouched).
        best_sel_pres = float("inf")
        best_sel_state = None
        best_sel_epoch = -1
        sel_no_improve = 0
        sel_checks = 0
        sel_stopped = False

        def _run_sel_check(epoch_idx):
            nonlocal best_sel_pres, best_sel_state, best_sel_epoch
            nonlocal sel_no_improve, sel_checks, sel_stopped
            v = sel_fn()
            sel_checks += 1
            # carry_plateau: the scheduler is driven by THIS held-out
            # selection residual (one step per check, patience in CHECKS),
            # never by the per-epoch noisy training loss.
            if self.lr_schedule == "carry_plateau" and self.scheduler is not None:
                self.scheduler.step(float(v[2]))
            if v[2] < best_sel_pres:
                best_sel_pres = v[2]
                best_sel_epoch = epoch_idx
                # Full checkpoint: MODEL + OPTIMIZER (Adam moments) + its LR.
                # Restoring only the weights while keeping end-of-inner Adam
                # moments would hand the next outer a mismatched
                # (parameters, moments) pair.
                best_sel_state = {
                    "model": {k: t.detach().cpu().clone()
                              for k, t in self.value_net.state_dict().items()},
                    "opt": copy.deepcopy(self.optimizer.state_dict()),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "epoch": int(epoch_idx),
                    "pres": float(v[2]),
                }
                sel_no_improve = 0
            else:
                sel_no_improve += 1
                if sel_patience and sel_no_improve >= int(sel_patience):
                    sel_stopped = True
            return v

        def _run_val_check(epoch_idx):
            nonlocal last_val, last_val_epoch, target_reached
            v = val_fn()
            last_val, last_val_epoch = v, epoch_idx
            if pres_target is not None and v[2] <= float(pres_target):
                target_reached = True
            return v

        # Epoch-0 pre-check: a warm-started iterate may already satisfy the
        # target; stop before any (freshly reset, large-LR) step perturbs it.
        if val_fn is not None and pres_target is not None:
            _run_val_check(0)
            if target_reached and print_every and print_every > 0:
                print(f"      [Eval] pres target already satisfied at inner epoch 0 "
                      f"(p_res={last_val[2]:.3e} <= {float(pres_target):g}); skipping this evaluation.")

        # Inner-epoch-0 selection baseline: the warm-started iterate itself.
        if sel_fn is not None:
            _run_sel_check(0)

        n_resamples = 0
        for epoch in range(1, epochs + 1):
            # Within-evaluation collocation resampling: every `resample_every`
            # epochs draw a FRESH batch and re-evaluate the SAME frozen policy
            # function on it (via resample_fn). The policy alpha_n itself never
            # changes here; only the points do. NOTE: the training loss jumps
            # at each refresh (new batch), which the inner-plateau scheduler
            # simply treats as a non-improving step.
            if (resample_fn is not None and resample_every and resample_every > 0
                    and epoch > 1 and (epoch - 1) % int(resample_every) == 0):
                (theta_n, w_colloc, x_colloc, tau_colloc,
                 w_term, x_term, tau_term, V_T_target) = resample_fn()
                theta_n_fixed = theta_n.detach()
                n_resamples += 1
                if print_every and print_every > 0:
                    print(f"      [Eval {epoch:4d}] resampled collocation batch "
                          f"(#{n_resamples}) under the frozen policy")
            if target_reached:
                break
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

            # shape constraints in wealth direction
            mono_penalty = torch.mean(torch.relu(-V_w) ** 2)    # V_w >= 0
            conc_penalty = torch.mean(torch.relu(V_ww) ** 2)    # V_ww <= 0

            # CRRA homogeneity: local relative risk aversion eta = -w V_ww / V_w must equal gamma.
            # (Equivalently the myopic coefficient -V_w/(w V_ww) must equal 1/gamma.)
            eta = -w_int * V_ww / torch.clamp(V_w, min=1e-8)
            rra_penalty = torch.mean((eta - self.gamma) ** 2)

            total_loss = (pde_loss + w_terminal * terminal_loss + w_shape * (mono_penalty + conc_penalty) + w_rra * rra_penalty)

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
                if track_best:  # diagnostic-only copy; skipped in timing mode
                    best_state = {k: v.detach().cpu().clone() for k, v in self.value_net.state_dict().items()}

            row = {
                "total": cur,
                "pde": float(pde_loss.item()),
                "terminal": float(terminal_loss.item()),
                "mono": float(mono_penalty.item()),
                "conc": float(conc_penalty.item()),
                "rra": float(rra_penalty.item()),
                "train_pres": pres_from_mse(float(pde_loss.item()), float(terminal_loss.item())),
                "val_pde_rms": "",
                "val_terminal_rms": "",
                "val_pres": "",
                # LR right after this epoch's (possible) scheduler step, so the
                # per-epoch scheduler path is reconstructible from the CSV.
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }

            # Held-out check against the residual target (post-step).
            if val_fn is not None and pres_target is not None and (epoch % max(1, int(val_every)) == 0):
                v = _run_val_check(epoch)
                row["val_pde_rms"], row["val_terminal_rms"], row["val_pres"] = v

            # Selection check (post-step; the stored state is the one that
            # produced this validation value).
            if sel_fn is not None and (epoch % max(1, int(sel_every)) == 0):
                sv = _run_sel_check(epoch)
                row["sel_pres"] = sv[2]

            if light_hist and loss_hist:
                loss_hist[-1] = row
            else:
                loss_hist.append(row)

            if print_every and print_every > 0 and (epoch % print_every == 0):
                lr_now = self.optimizer.param_groups[0]["lr"]
                extra = f" | p_res={row['val_pres']:.3e}" if isinstance(row["val_pres"], float) else ""
                print(f"      [Eval {epoch:4d}/{epochs}] Loss={cur:.3e} | PDE={row['pde']:.3e} | "
                      f"Term={row['terminal']:.3e} | Mono={row['mono']:.3e} | Conc={row['conc']:.3e} | "
                      f"RRA(η-γ)²={row['rra']:.3e}{extra} | LR={lr_now:.2e}")

            if target_reached:
                if print_every and print_every > 0:
                    print(f"      [Eval] pres target reached at inner epoch {epoch} "
                          f"(p_res={last_val[2]:.3e} <= {float(pres_target):g}); stopping this evaluation.")
                break

            if sel_stopped:
                if print_every and print_every > 0:
                    print(f"      [Eval] selection plateau: no held-out improvement over "
                          f"{sel_patience} checks (best p_res={best_sel_pres:.3e} "
                          f"@ epoch {best_sel_epoch}); stopping this evaluation.")
                break

        # Epoch-0 stop: no gradient step was taken, but downstream logging and
        # the divergence stopper still expect one loss row -- compute it once
        # (forward only, no optimizer step).
        if epochs_used == 0 and not loss_hist:
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
            pde_loss = torch.mean(residual.detach() ** 2)
            with torch.no_grad():
                V_T_pred = self.value_net(w_term, x_term, tau_term)
                terminal_loss = torch.mean((V_T_pred - V_T_target) ** 2)
            mono_penalty = torch.mean(torch.relu(-V_w.detach()) ** 2)
            conc_penalty = torch.mean(torch.relu(V_ww.detach()) ** 2)
            eta = -w_int.detach() * V_ww.detach() / torch.clamp(V_w.detach(), min=1e-8)
            rra_penalty = torch.mean((eta - self.gamma) ** 2)
            total0 = float((pde_loss + w_terminal * terminal_loss
                            + w_shape * (mono_penalty + conc_penalty)
                            + w_rra * rra_penalty).item())
            loss_hist.append({
                "total": total0,
                "pde": float(pde_loss.item()),
                "terminal": float(terminal_loss.item()),
                "mono": float(mono_penalty.item()),
                "conc": float(conc_penalty.item()),
                "rra": float(rra_penalty.item()),
                "train_pres": pres_from_mse(float(pde_loss.item()), float(terminal_loss.item())),
                "val_pde_rms": last_val[0] if last_val is not None else "",
                "val_terminal_rms": last_val[1] if last_val is not None else "",
                "val_pres": last_val[2] if last_val is not None else "",
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "synthetic": True,  # epoch-0 diagnostic row: no optimizer step
            })

        # End-of-evaluation held-out measurement: this is the reported
        # p_res,n. If the last check already happened at the final state
        # (stop epoch or cap epoch divisible by val_every), reuse it.
        # Restore the inner held-out best BEFORE policy improvement and the
        # big-audit measurement: within one outer everything solves the SAME
        # frozen PDE, so held-out selection here is legitimate (the OUTER
        # level still reports the final PI iterate, never a cross-outer best).
        lr_end_before_restore = ""
        lr_best_checkpoint = ""
        lr_after_restore = ""
        lr_carried_next = ""
        if restore_best and best_sel_state is not None:
            # Restore model weights + Adam moments from the inner best. The LR
            # handed to the next outer is NOT the raw best-checkpoint LR:
            #     LR_{n+1} = max(scheduler_min_lr, min(LR_best, LR_end)).
            # If the schedule already decayed past the best checkpoint, keep
            # the LOWER end-of-inner LR (so an epoch-0 best still retries at a
            # lower LR, and the LR is non-increasing across outer boundaries,
            # true to the carry_plateau name). LR is not part of the Adam
            # moments, so pairing best moments with a lower LR is consistent.
            end_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
            self.value_net.load_state_dict(best_sel_state["model"])
            self.optimizer.load_state_dict(best_sel_state["opt"])
            floor_lr = self._effective_min_lr()
            carried = []
            for g, end_lr in zip(self.optimizer.param_groups, end_lrs):
                best_lr = float(g["lr"])  # load_state_dict restored the best LR
                g["lr"] = max(floor_lr, min(best_lr, end_lr))
                carried.append(float(g["lr"]))
            lr_end_before_restore = end_lrs[0]
            lr_best_checkpoint = float(best_sel_state["lr"])
            # after_restore: the LR now sitting in the optimizer.
            # carried_next: the ACTUAL next-outer start LR (carry_lr_max cap
            # applied); only meaningful under carry_plateau -- other schedules
            # reset the LR at the next outer anyway.
            lr_after_restore = carried[0]
            if self.lr_schedule == "carry_plateau":
                lr_carried_next = min(float(self.carry_lr_max), carried[0])
            else:
                lr_carried_next = ""
            last_val_epoch = -1  # force the audit re-measurement below

        if val_fn is not None and last_val_epoch != epochs_used:
            _run_val_check(epochs_used)

        last_loss = float(loss_hist[-1]["total"]) if loss_hist else float("inf")

        eval_info = {
            "n_resamples": n_resamples,
            "epochs_used": int(epochs_used),
            "target_reached": bool(target_reached),
            "sel_best_pres": best_sel_state["pres"] if best_sel_state is not None else "",
            "sel_best_epoch": best_sel_state["epoch"] if best_sel_state is not None else "",
            "sel_best_lr": best_sel_state["lr"] if best_sel_state is not None else "",
            "lr_end_before_restore": lr_end_before_restore,
            "lr_best_checkpoint": lr_best_checkpoint,
            "lr_after_restore": lr_after_restore,
            "lr_carried_next": lr_carried_next,
            "sel_checks": int(sel_checks),
            "sel_stopped": int(bool(sel_stopped)),
            "sel_restored": int(bool(restore_best and best_sel_state is not None)),
            "val_pde_rms": last_val[0] if last_val is not None else "",
            "val_terminal_rms": last_val[1] if last_val is not None else "",
            "val_pres": last_val[2] if last_val is not None else "",
            "train_pres": loss_hist[-1].get("train_pres", "") if loss_hist else "",
        }

        return loss_hist, best_loss, best_state, best_epoch, last_loss, eval_info

        

    def policy_improvement(self, w, x, tau, net=None):
        """θ from the FOC of `net` (default: current value_net).

        Passing a FROZEN copy of a previous iterate lets the same policy
        FUNCTION be re-evaluated at freshly sampled collocation points
        (within-evaluation resampling) without touching the live network.
        """
        own = net is None or net is self.value_net
        net = self.value_net if net is None else net
        if own:
            net.eval()

        w_e = w.detach().clone().requires_grad_(True)
        x_e = x.detach().clone().requires_grad_(True)
        tau_e = tau.detach().clone().requires_grad_(True)

        theta, V_w, V_ww = compute_theta_from_foc_nd(
            net,
            w_e, x_e, tau_e,
            self.M, self.N,
            self.Gamma_t, self.lam0_t, self.Lam_t,
            theta_clip_abs=self.theta_clip_abs
        )

        if own:
            net.train()
        return theta.detach()

    @torch.no_grad()
    def closed_form_theta_on_points(self, w, x, tau):
        """Closed-form θ*(τ,w,x) on the given collocation points (vectorized).

        θ*/w = (λ(x) + Γ ∇_x log φ) / γ,  with ∇_x log φ = b(τ) + C(τ) x.
        Returns a (B, N) tensor (raw θ, NOT normalized by w), unclipped,
        on the same points used by policy_improvement so it is directly
        comparable to theta_diff.
        """
        w_np   = w.detach().cpu().numpy().reshape(-1)          # (B,)
        x_np   = x.detach().cpu().numpy()                      # (B, M)
        tau_np = tau.detach().cpu().numpy().reshape(-1)         # (B,)

        # Interpolate ODE state [a, b(0..M-1), vec(C)] at each tau.
        Y = np.stack(
            [np.interp(tau_np, cf_sol.t, cf_sol.y[i]) for i in range(cf_sol.y.shape[0])],
            axis=1,
        )                                                      # (B, 1+M+M*M)
        b = Y[:, 1:1 + self.M]                                 # (B, M)
        C = Y[:, 1 + self.M:].reshape(-1, self.M, self.M)      # (B, M, M)
        C = 0.5 * (C + np.transpose(C, (0, 2, 1)))

        lam_x = lam0[None, :] + x_np @ Lam.T                   # (B, N)
        grad_log_phi = b + np.einsum('bij,bj->bi', C, x_np)    # (B, M)
        theta_norm = (lam_x + grad_log_phi @ Gamma.T) / self.gamma  # (B, N)
        theta = w_np[:, None] * theta_norm                     # (B, N)
        return torch.tensor(theta, device=self.device, dtype=torch.float32)

    def run_policy_iteration(
        self,
        outer_iters=50,
        eval_epochs=200,
        batch_size=3000,
        terminal_frac=0.5,
        w_terminal=10.0,
        w_shape=1.0,
        w_rra=0.0,
        theta_init_method="myopic",
        theta_init_scale=1.0,
        print_every_outer=5,
        print_every_eval=200,
        verbose_detail=False,
        save_iterate_every=1,
        e3b_checkpoints=False,
        pe_resample_every=0,
        inner_best_restore=True,
        sel_points=10000,
        sel_terminal_points=2000,
        sel_every=50,
        sel_patience=6,
        pres_target=None,
        val_points=100000,
        val_terminal_points=10000,
        val_every=1,
        val_seed=0,
        diag_points=0,
        diag_margin=0.0,
        diag_every=1,
        timing_mode=False,
        recorder=None,
        stopper=None
    ):
        print(f"\n{'='*70}")
        print(f"PI-PINN ND (No Policy Net): N={self.N}, M={self.M}")
        print(f"  outer_iters   : {outer_iters}")
        print(f"  eval_epochs   : {eval_epochs}")
        print(f"  batch_size    : {batch_size}")
        self.theta_init_scale = float(theta_init_scale)
        print(f"  θ init        : {theta_init_method} (scale={self.theta_init_scale:g})")
        print(f"  θ clip abs    : {self.theta_clip_abs}")
        print(f"  risk premium : {RISK_PREMIUM_MODE} (eps={NONAFFINE_EPS:g})")
        print(f"  init LR       : {self.initial_lr:.2e}")
        print(f"  lr schedule   : {self.lr_schedule} (adam_reset={self.adam_reset}, patience={self.scheduler_patience})")
        print(f"  pres target   : {pres_target}  (val: {val_points} int + {val_terminal_points} term pts, check every {val_every} inner epochs)")
        print(f"{'='*70}\n")

        results = {
            "theta_diff": [],
            "theta_cf_diff": [],
            "eval_loss": [],
            "val_pde_rms": [],
            "val_terminal_rms": [],
            "val_pres": [],
            "e_Xev": [],
            "inner_epochs_used": [],
            "lr": [],
            "loss_history": [],
            "stopped_early": False,
            "stop_info": {},
        }

        # Held-out validation set: sampled ONCE per run with a dedicated RNG
        # stream (training RNG untouched). Used for the pres-target stopping
        # rule and for the reported per-outer residual level p_res,n.
        val_set = None
        skip_val = timing_mode and pres_target is None  # audit-only val = diagnostic
        if val_points and val_points > 0 and not skip_val:
            val_set = build_validation_set(
                int(val_points), max(1, int(val_terminal_points)), self.device,
                self.M, X_min, X_max, W_min, W_max, tau_max, seed=val_seed,
            )

        # Small dedicated SELECTION set for the inner held-out best (distinct
        # RNG stream from the big audit set so selection never sees the audit
        # points). Part of the training protocol, so NOT gated by timing_mode.
        sel_set = None
        if inner_best_restore and sel_points and sel_points > 0:
            sel_set = build_validation_set(
                int(sel_points), max(1, int(sel_terminal_points)), self.device,
                self.M, X_min, X_max, W_min, W_max, tau_max,
                seed=int(val_seed) * 7919 + 101,
            )
            print(f"  selection set : {sel_points} interior / {sel_terminal_points} terminal "
                  f"(check every {sel_every} epochs, patience {sel_patience}, restore ON)")

        # Fixed Q_ev diagnostics plus a dedicated full-window Q_col design for
        # frozen-policy ellipticity.  Both use the MARKET seed so all training
        # seeds and both methods share exactly the same points, independently
        # of val_points and inner-best selection settings.
        diag = None
        diag_col = None
        if diag_points and diag_points > 0 and not timing_mode:
            diag = build_diag_set(
                int(diag_points), float(diag_margin), self.M, self.N, self.gamma, self.r,
                X_min, X_max, W_min, W_max, tau_max,
                cf_sol, lam0, Lam, Gamma, market_seed=MARKET_SEED,
                include_affine_reference=HAS_AFFINE_REFERENCE,
            )
            diag_col = build_validation_set(
                int(diag_points), 1, self.device, self.M,
                X_min, X_max, W_min, W_max, tau_max, seed=MARKET_SEED,
            )
            _diag_kind = "e_n / margins" if HAS_AFFINE_REFERENCE else "model-side margins only"
            print(f"  diag set      : {diag_points} pts on Q_ev(margin={diag_margin}) for {_diag_kind} "
                  f"+ {diag_points} pts on Q_col for frozen-policy ellipticity")

        best_eval_loss = float("inf")
        best_iter = 0
        best_inner_epoch = 0
        global_step_base = 0
        self._track_best = not timing_mode
        self._timing_mode = bool(timing_mode)
        if timing_mode:
            # Timing runs must not write iterate checkpoints regardless of
            # what the caller passed.
            save_iterate_every = 0
            e3b_checkpoints = False
        start_time = time.time()

        # Weight-saving policy (paper protocol):
        #   value_net_final.pt   -> FINAL PI iterate; the official reported model.
        #   value_net_best.pt    -> lowest inner eval loss; DIAGNOSTIC ONLY.
        #   value_net_last.pt    -> alias of the final in-memory state (back-compat).
        #   iterates/value_net_iter{NNNN}.pt -> per-outer-iteration snapshots
        #       (every save_iterate_every iters) for post-hoc rho_n analyses.
        best_path = os.path.join(weight_dir, "value_net_best.pt")
        last_path = os.path.join(weight_dir, "value_net_last.pt")
        final_path = os.path.join(weight_dir, "value_net_final.pt")
        iterate_dir = os.path.join(weight_dir, "iterates")
        if e3b_checkpoints or (save_iterate_every and save_iterate_every > 0):
            os.makedirs(iterate_dir, exist_ok=True)
        legacy_best_path = os.path.join(
            weight_dir,
            f"value_net_best_{N_ASSETS}-asset_{M_STATES}-state({batch_size}-batch, {eval_epochs}-eval epoch).pt"
        )

        train_fields = [
            "timestamp", "model_type", "run_tag", "global_step", "outer_iter", "inner_epoch",
            "total_loss", "pde_loss", "terminal_loss", "concavity_loss", "monotonicity_loss","rra_loss",
            "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres", "sel_pres",
            "theta_diff", "eval_loss", "lr", "best_loss", "elapsed_sec", "stopped", "stop_reason",
        ]
        outer_fields = [
            "timestamp", "model_type", "run_tag", "outer_iter", "total_loss", "pde_loss",
            "terminal_loss", "monotonicity_loss", "concavity_loss", "rra_loss", "theta_diff", "theta_cf_diff", "eval_loss",
            "train_pres", "val_pde_rms", "val_terminal_rms", "val_pres",
            "inner_epochs_used", "target_reached",
            "sel_best_pres", "sel_best_epoch", "sel_best_lr",
            "lr_end_before_restore", "lr_best_checkpoint", "lr_after_restore", "lr_carried_next",
            "sel_checks", "sel_stopped", "sel_restored",
            "e_V_sup", "e_bundle_sup", "e_Xev",
            "e_Vw_sup", "e_Vww_sup", "e_Vwx_sup",
            "diag_RelL2_V", "diag_RelL2_theta", "diag_RelL2_vartheta",
            "m_ww", "M_num", "guard_frac_ev",
            "frozen_policy_iter", "improved_policy_iter",
            "lam_min_sigma_frozen", "lam_max_sigma_frozen", "clip_frac_frozen",
            "vartheta_l2_min", "vartheta_l2_max",
            "vartheta_component_min", "vartheta_component_max", "vartheta_abs_max",
            "lr", "best_eval_loss", "bad_count", "stop_active", "stop_is_bad",
            "stopped", "stop_reason", "elapsed_sec",
        ]

        # E8 core timer: measure the iterative PI training protocol while
        # excluding the final last/final checkpoint serialization.  Explicit
        # CUDA synchronization prevents asynchronous kernels from crossing a
        # timer boundary.
        if timing_mode and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        core_train_start = time.perf_counter()

        def _core_train_elapsed():
            if timing_mode and torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            return float(time.perf_counter() - core_train_start)

        # initial collocation points + theta_0
        w_colloc, x_colloc, tau_colloc = sample_interior(batch_size, self.device, self.M, X_min, X_max, W_min, W_max, tau_max)
        # w_term, x_term, tau_term = sample_terminal(batch_size // 2, self.device, self.M, X_min, X_max, W_min, W_max)
        w_term, x_term, tau_term = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device, self.M, X_min, X_max, W_min, W_max)
        V_T_target = V_terminal(w_term, self.gamma).detach()

        theta_n = self.initialize_theta(w_colloc, x_colloc, tau_colloc, method=theta_init_method)
        print(f"Initial θ stats: mean={theta_n.mean().item():.4f}, std={theta_n.std().item():.4f}")

        for it in range(1, outer_iters + 1):
            if stopper is not None and stopper.shared_stop_exists():
                info = stopper.mark_from_existing_flag(outer_iter=it, pde_loss=None)
                results["stopped_early"] = True
                results["stop_info"] = info
                print(f"[early-stop] shared stop flag detected before PI-PINN iter {it}. Skipping remaining work.")
                break

            verbose = (it % print_every_outer == 0) or (it <= 3)

            # resample points
            w_colloc, x_colloc, tau_colloc = sample_interior(batch_size, self.device, self.M, X_min, X_max, W_min, W_max, tau_max)
            # w_term, x_term, tau_term = sample_terminal(batch_size // 2, self.device, self.M, X_min, X_max, W_min, W_max)
            w_term, x_term, tau_term = sample_terminal(max(1, int(batch_size * terminal_frac)), self.device, self.M, X_min, X_max, W_min, W_max)
            V_T_target = V_terminal(w_term, self.gamma).detach()

            # θ_n from previous V (policy improvement), except first iter
            if it > 1:
                theta_n = self.policy_improvement(w_colloc, x_colloc, tau_colloc)
            else:
                theta_n = self.initialize_theta(w_colloc, x_colloc, tau_colloc, method=theta_init_method)
                with torch.no_grad():
                    _t0 = torch.norm(theta_n / torch.clamp(w_colloc, min=1e-8), dim=1)
                    results["theta0_norm_max"] = float(_t0.max().item())
                    results["theta0_norm_mean"] = float(_t0.mean().item())
                print(f"  [init policy] max||theta0/w|| = {results['theta0_norm_max']:.4e}, "
                      f"mean = {results['theta0_norm_mean']:.4e}")

            # Within-evaluation resampling machinery: freeze the policy
            # SOURCE (the previous iterate's network; analytic init at it=1)
            # so the same policy function can be re-evaluated on fresh
            # collocation points during this policy evaluation.
            resample_fn = None
            if pe_resample_every and pe_resample_every > 0:
                policy_source = None
                if it > 1:
                    policy_source = copy.deepcopy(self.value_net).eval()
                    for _p in policy_source.parameters():
                        _p.requires_grad_(False)

                def _resample(_src=policy_source):
                    w_c, x_c, t_c = sample_interior(
                        batch_size, self.device, self.M, X_min, X_max, W_min, W_max, tau_max)
                    w_t2, x_t2, t_t2 = sample_terminal(
                        max(1, int(batch_size * terminal_frac)), self.device, self.M,
                        X_min, X_max, W_min, W_max)
                    V_t2 = V_terminal(w_t2, self.gamma).detach()
                    if _src is not None:
                        th = self.policy_improvement(w_c, x_c, t_c, net=_src)
                    else:
                        # it == 1: the initial policy is analytic (state-free),
                        # so it re-evaluates identically on any points.
                        th = self.initialize_theta(w_c, x_c, t_c, method=theta_init_method)
                    return th, w_c, x_c, t_c, w_t2, x_t2, t_t2, V_t2

                resample_fn = _resample

            # Frozen policy on the held-out points (same rule, same clipping)
            # and the resulting per-outer validation evaluator.
            val_fn = None
            theta_val = None
            frozen_lam_min = ""
            frozen_lam_max = ""
            frozen_clip_frac = ""
            if val_set is not None:
                if it > 1:
                    theta_val = self.policy_improvement_chunked(
                        val_set["w_int"], val_set["x_int"], val_set["tau_int"])
                else:
                    theta_val = self.initialize_theta(
                        val_set["w_int"], val_set["x_int"], val_set["tau_int"],
                        method=theta_init_method)
                theta_val = theta_val.detach()
                val_fn = lambda: self.evaluate_heldout_pres(theta_val, val_set)

            # Ellipticity concerns the FROZEN policy used in this evaluation
            # (alpha_{it-1}) on a dedicated Q_col design.  It is independent
            # of held-out validation and inner-best selection settings.
            _do_frozen_diag = (
                diag_col is not None and not timing_mode
                and (it == 1 or diag_every <= 1 or it % diag_every == 0 or it == outer_iters)
            )
            if _do_frozen_diag:
                if it > 1:
                    _theta_diag = self.policy_improvement_chunked(
                        diag_col["w_int"], diag_col["x_int"], diag_col["tau_int"]
                    )
                else:
                    _theta_diag = self.initialize_theta(
                        diag_col["w_int"], diag_col["x_int"], diag_col["tau_int"],
                        method=theta_init_method,
                    )
                _theta_diag = _theta_diag.detach()
                _tv = _theta_diag.cpu().numpy()
                _lmin, _lmax = sigma_eig_extremes_batch(_tv, Gamma, Q)
                frozen_lam_min = float(np.min(_lmin))
                frozen_lam_max = float(np.max(_lmax))
                if self.theta_clip_abs is not None:
                    _c = float(self.theta_clip_abs)
                    frozen_clip_frac = float(
                        np.mean(np.any(np.abs(_tv) >= _c - 1e-12, axis=1))
                    )
                if it == 1:
                    results["theta0_lam_min_sigma"] = frozen_lam_min
                    results["theta0_lam_max_sigma"] = frozen_lam_max
                    print(f"  [init policy] lam(Sigma^theta0) on Q_col subsample: "
                          f"min = {frozen_lam_min:.4e}, max = {frozen_lam_max:.4e}")

            sel_fn = None
            if sel_set is None and self.lr_schedule == "carry_plateau" and it == 1:
                print("[warn] carry_plateau without a selection set: the scheduler "
                      "never steps (constant carried LR). Enable inner_best/sel_points.")
            if sel_set is not None:
                if it > 1:
                    theta_sel = self.policy_improvement_chunked(
                        sel_set["w_int"], sel_set["x_int"], sel_set["tau_int"])
                else:
                    theta_sel = self.initialize_theta(
                        sel_set["w_int"], sel_set["x_int"], sel_set["tau_int"],
                        method=theta_init_method)
                theta_sel = theta_sel.detach()
                sel_fn = lambda: self.evaluate_heldout_pres(theta_sel, sel_set)

            # LR protocol for this outer iteration (inner_plateau: reset).
            self.prepare_optimizer_for_outer()

            # === Policy Evaluation ===
            eval_hist, inner_best_eval_loss, inner_best_state, inner_best_epoch, last_eval_loss, eval_info = self.policy_evaluation(
                theta_n=theta_n,
                w_colloc=w_colloc, x_colloc=x_colloc, tau_colloc=tau_colloc,
                w_term=w_term, x_term=x_term, tau_term=tau_term,
                V_T_target=V_T_target,
                epochs=eval_epochs,
                w_terminal=w_terminal,
                w_shape=w_shape,
                w_rra=w_rra,
                print_every=print_every_eval,
                pres_target=pres_target,
                val_every=val_every,
                val_fn=val_fn,
                sel_fn=sel_fn,
                sel_every=sel_every,
                sel_patience=sel_patience,
                restore_best=bool(inner_best_restore),
                resample_every=pe_resample_every,
                resample_fn=resample_fn,
            )

            results["loss_history"].extend(eval_hist)
            results["eval_loss"].append(last_eval_loss)
            results["val_pde_rms"].append(eval_info.get("val_pde_rms", ""))
            results["val_terminal_rms"].append(eval_info.get("val_terminal_rms", ""))
            results["val_pres"].append(eval_info.get("val_pres", ""))
            results["inner_epochs_used"].append(eval_info.get("epochs_used", 0))

            # Fixed-set diagnostics for iterate v~_it: e_n components (E3-a)
            # and the stability margins (E1-b/c analogues).
            diag_res = {}
            if diag is not None and (
                diag_every <= 1 or it % diag_every == 0 or it == 1 or it == outer_iters
            ):
                diag_res = eval_diag_metrics(
                    self.value_net, diag, self.M, self.N, self.gamma,
                    self.Gamma_t, self.lam0_t, self.Lam_t,
                    Gamma, lam0, Lam,
                )
                if "e_Xev" in diag_res:
                    results["e_Xev"].append(diag_res["e_Xev"])
                results["last_diag"] = dict(diag_res)

            # Snapshot the current iterate v~_n (state right after policy
            # evaluation at outer iter `it`). Main runs keep this OFF; the
            # FD-reference mode retains every completed outer iterate.  The
            # approximation-hypothesis audit needs 11--19 as well as the old
            # sparse 1--10/every-10 schedule.
            _save_this_iter = False
            if e3b_checkpoints:
                _save_this_iter = True
            elif save_iterate_every and save_iterate_every > 0:
                _save_this_iter = (it % save_iterate_every == 0)
            if _save_this_iter:
                torch.save(
                    self.value_net.state_dict(),
                    os.path.join(iterate_dir, f"value_net_iter{it:04d}.pt"),
                )

            # === Policy Improvement ===
            # Diagnostic only: the next outer iteration recomputes theta on a
            # FRESH collocation batch, so theta_new is never consumed by the
            # algorithm. Skipped entirely in timing mode.
            if timing_mode:
                theta_new = None
                w_safe = None
                theta_diff = float("nan")
            else:
                theta_new = self.policy_improvement(w_colloc, x_colloc, tau_colloc)

                # Normalize by wealth so the metric matches the heatmaps' θ/w
                # (relative risky exposure) and is wealth-independent.
                # (θ_new - θ)/w_b == θ_new/w_b - θ/w_b since both share the same w_b.
                w_safe = w_colloc.detach()   # (batch, 1), w >= W_min = 0.1 so no zero-division

                theta_diff = torch.mean(((theta_new - theta_n) / w_safe) ** 2).item()
            results["theta_diff"].append(theta_diff)

            # Distance of the current iterate to the closed-form optimum,
            # on the same collocation points (for the convergence figure).
            if timing_mode or theta_new is None or not HAS_AFFINE_REFERENCE:
                theta_cf_diff = float("nan")  # closed-form oracle diagnostic: off in timing runs
            else:
                theta_star = self.closed_form_theta_on_points(w_colloc, x_colloc, tau_colloc)
                theta_cf_diff = torch.mean(((theta_new - theta_star) / w_safe) ** 2).item()
            results["theta_cf_diff"].append(theta_cf_diff)

            # === Scheduler ===
            # outer_plateau (legacy): one persistent scheduler stepped per outer.
            # inner_plateau: stepping already happened inside policy_evaluation.
            # fixed: no scheduler.
            if self.lr_schedule == "outer_plateau" and self.scheduler is not None:
                self.scheduler.step(last_eval_loss)
            lr_now = float(self.optimizer.param_groups[0]["lr"])
            results["lr"].append(lr_now)

            # === Track best checkpoint over all inner + outer epochs ===
            # The training trajectory remains at the last iterate, but the saved
            # checkpoint is the best inner state seen so far.
            if inner_best_state is not None and inner_best_eval_loss < best_eval_loss:
                best_eval_loss = inner_best_eval_loss
                best_iter = it
                best_inner_epoch = inner_best_epoch
                torch.save(inner_best_state, best_path)
                torch.save(inner_best_state, legacy_best_path)

            last = eval_hist[-1]
            _vp = eval_info.get("val_pres", "")
            _vp_s = f"{_vp:.3e}" if isinstance(_vp, float) else "n/a"
            _pi = f"p_res={_vp_s} | inner={eval_info.get('epochs_used', 0)}{'*' if eval_info.get('target_reached') else ''}"
            _cf_s = f"{theta_cf_diff:.4e}" if np.isfinite(theta_cf_diff) else "n/a"
            if verbose:
                print(f"[Iter {it:3d}] Loss={last['total']:.2e} | PDE Loss: {last['pde']:.4e} | Terminal Loss: {last['terminal']:.4e} | {_pi} | θ diff={theta_diff:.4e} | θ-θ* diff={_cf_s} | RRA(η-γ)²={last.get('rra', float('nan')):.4e} | LR={lr_now:.2e}")
            elif it % 20 == 0:
                print(f"[Iter {it:3d}] Loss={last['total']:.2e} | PDE Loss: {last['pde']:.4e} | Terminal Loss: {last['terminal']:.4e} | {_pi} | θ diff={theta_diff:.4e} | θ-θ* diff={_cf_s} | RRA(η-γ)²={last.get('rra', float('nan')):.4e} | LR={lr_now:.2e}")

            elapsed = time.time() - start_time
            stop_triggered = False
            stop_meta = {"active": False, "is_bad": False, "bad_count": 0}
            if stopper is not None:
                stop_triggered, stop_meta = stopper.update(it, float(last["pde"]))

            # CSV: inner training rows for this outer iteration
            # (skipped entirely in timing mode: per-epoch appends are not
            # part of the algorithm's core cost)
            if recorder is not None and not timing_mode:
                train_rows = []
                _real_j = 0
                for h in eval_hist:
                    if h.get("synthetic"):
                        _gs, _ie = "", 0  # no optimizer step: keep the global counter clean
                    else:
                        _real_j += 1
                        _gs, _ie = global_step_base + _real_j, _real_j
                    train_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model_type": ARGS.model_type,
                        "run_tag": ARGS.run_tag,
                        "global_step": _gs,
                        "outer_iter": it,
                        "inner_epoch": _ie,
                        "total_loss": h.get("total", ""),
                        "pde_loss": h.get("pde", ""),
                        "terminal_loss": h.get("terminal", ""),
                        "concavity_loss": h.get("conc", ""),
                        "monotonicity_loss": h.get("mono", ""),
                        "rra_loss": h.get("rra", ""),
                        "train_pres": h.get("train_pres", ""),
                        "val_pde_rms": h.get("val_pde_rms", ""),
                        "val_terminal_rms": h.get("val_terminal_rms", ""),
                        "val_pres": h.get("val_pres", ""),
                        "sel_pres": h.get("sel_pres", ""),
                        "theta_diff": "",
                        "eval_loss": last_eval_loss,
                        "lr": h.get("lr", lr_now),
                        "best_loss": best_eval_loss,
                        "elapsed_sec": elapsed,
                        "stopped": int(bool(stop_triggered)),
                        "stop_reason": stop_meta.get("reason", ""),
                    })
                append_csv_rows(recorder.train_csv, train_rows, train_fields)

                outer_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model_type": ARGS.model_type,
                    "run_tag": ARGS.run_tag,
                    "outer_iter": it,
                    "total_loss": last.get("total", ""),
                    "pde_loss": last.get("pde", ""),
                    "terminal_loss": last.get("terminal", ""),
                    "monotonicity_loss": last.get("mono", ""),
                    "concavity_loss": last.get("conc", ""),
                    "rra_loss": last.get("rra", ""),
                    "theta_diff": theta_diff,
                    "theta_cf_diff": theta_cf_diff if HAS_AFFINE_REFERENCE else "",
                    "eval_loss": last_eval_loss,
                    "train_pres": eval_info.get("train_pres", ""),
                    "val_pde_rms": eval_info.get("val_pde_rms", ""),
                    "val_terminal_rms": eval_info.get("val_terminal_rms", ""),
                    "val_pres": eval_info.get("val_pres", ""),
                    "inner_epochs_used": eval_info.get("epochs_used", ""),
                    "target_reached": int(bool(eval_info.get("target_reached", False))),
                    "sel_best_pres": eval_info.get("sel_best_pres", ""),
                    "sel_best_epoch": eval_info.get("sel_best_epoch", ""),
                    "sel_best_lr": eval_info.get("sel_best_lr", ""),
                    "lr_end_before_restore": eval_info.get("lr_end_before_restore", ""),
                    "lr_best_checkpoint": eval_info.get("lr_best_checkpoint", ""),
                    "lr_after_restore": eval_info.get("lr_after_restore", ""),
                    "lr_carried_next": eval_info.get("lr_carried_next", ""),
                    "sel_checks": eval_info.get("sel_checks", ""),
                    "sel_stopped": eval_info.get("sel_stopped", ""),
                    "sel_restored": eval_info.get("sel_restored", ""),
                    "e_V_sup": diag_res.get("e_V_sup", ""),
                    "e_bundle_sup": diag_res.get("e_bundle_sup", ""),
                    "e_Vw_sup": diag_res.get("e_Vw_sup", ""),
                    "e_Vww_sup": diag_res.get("e_Vww_sup", ""),
                    "e_Vwx_sup": diag_res.get("e_Vwx_sup", ""),
                    "e_Xev": diag_res.get("e_Xev", ""),
                    "diag_RelL2_V": diag_res.get("diag_RelL2_V", ""),
                    "diag_RelL2_theta": diag_res.get("diag_RelL2_theta", ""),
                    "diag_RelL2_vartheta": diag_res.get("diag_RelL2_vartheta", ""),
                    "m_ww": diag_res.get("m_ww", ""),
                    "M_num": diag_res.get("M_num", ""),
                    "guard_frac_ev": diag_res.get("guard_frac_ev", ""),
                    "frozen_policy_iter": it - 1,
                    "improved_policy_iter": it,
                    "lam_min_sigma_frozen": frozen_lam_min,
                    "lam_max_sigma_frozen": frozen_lam_max,
                    "clip_frac_frozen": frozen_clip_frac,
                    "vartheta_l2_min": diag_res.get("vartheta_l2_min", ""),
                    "vartheta_l2_max": diag_res.get("vartheta_l2_max", ""),
                    "vartheta_component_min": diag_res.get("vartheta_component_min", ""),
                    "vartheta_component_max": diag_res.get("vartheta_component_max", ""),
                    "vartheta_abs_max": diag_res.get("vartheta_abs_max", ""),
                    "lr": lr_now,
                    "best_eval_loss": best_eval_loss,
                    "bad_count": stop_meta.get("bad_count", ""),
                    "stop_active": int(bool(stop_meta.get("active", False))),
                    "stop_is_bad": int(bool(stop_meta.get("is_bad", False))),
                    "stopped": int(bool(stop_triggered)),
                    "stop_reason": stop_meta.get("reason", ""),
                    "elapsed_sec": elapsed,
                }
                append_csv_rows(recorder.outer_csv, [outer_row], outer_fields)

            if stop_triggered:
                results["core_train_wall_sec"] = _core_train_elapsed()
                torch.save(self.value_net.state_dict(), last_path)
                torch.save(self.value_net.state_dict(), final_path)
                results["stopped_early"] = True
                results["final_outer_iter"] = it
                results["stop_info"] = {**stop_meta, "outer_iter": it}
                print(f"\n[early-stop] PI-PINN stopped at iter={it}, PDE={float(last['pde']):.4e}, reason={stop_meta.get('reason', '')}")
                break

            # update θ for next iteration (functionally; actual eval uses policy_improvement anyway)
            if theta_new is not None:
                theta_n = theta_new
            results["final_outer_iter"] = it
            global_step_base += int(eval_info.get("epochs_used", 0))

        if "core_train_wall_sec" not in results:
            results["core_train_wall_sec"] = _core_train_elapsed()

        # p_res = max_n p_res,n over completed outer iterations.
        _vals = [v for v in results["val_pres"] if isinstance(v, float)]
        results["pres_max"] = max(_vals) if _vals else None
        results["total_inner_steps"] = int(sum(results["inner_epochs_used"]))

        # FINAL-ITERATE POLICY: the reported model is the final PI iterate.
        # We deliberately do NOT restore the best checkpoint here; the best
        # checkpoint is kept on disk only as a diagnostic artifact.
        torch.save(self.value_net.state_dict(), last_path)
        torch.save(self.value_net.state_dict(), final_path)
        if e3b_checkpoints:
            _it_last = int(results.get("final_outer_iter", 0))
            if _it_last > 0:
                torch.save(self.value_net.state_dict(),
                           os.path.join(iterate_dir, f"value_net_iter{_it_last:04d}.pt"))

        print(f"\n{'='*70}")
        status = "stopped early" if results.get("stopped_early", False) else "finished"
        print(
            f"PI-PINN {status}. Reported model = FINAL iterate "
            f"(outer iter {results.get('final_outer_iter', 0)}) -> {final_path}"
        )
        print(
            f"  [diagnostic] best inner checkpoint: outer iter {best_iter}, "
            f"inner epoch {best_inner_epoch} (eval_loss={best_eval_loss:.3e}) -> {best_path}"
        )
        print(f"{'='*70}")

        return results


def plot_pi_convergence_nd(results, save_path=None, show=True):
    """Convergence summary for ND PI-PINN."""
    iters = np.arange(1, len(results["eval_loss"]) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axs[0].semilogy(iters, results["eval_loss"], linewidth=1.6)
    axs[0].set_title("Policy Evaluation Loss")
    axs[0].set_xlabel("Outer Iter")
    axs[0].grid(True, alpha=0.3)

    axs[1].semilogy(iters, results["theta_diff"], linewidth=1.6)
    axs[1].set_title("θ diff (MSE)")
    axs[1].set_xlabel("Outer Iter")
    axs[1].grid(True, alpha=0.3)

    axs[2].semilogy(iters, results["lr"], linewidth=1.6)
    axs[2].set_title("Learning rate")
    axs[2].set_xlabel("Outer Iter")
    axs[2].grid(True, alpha=0.3)

    plt.suptitle("ND Liu: PI-PINN Convergence", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


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
    Plot value function comparison: PINN-PI vs Closed-form.
    Layout: 1 row × 3 columns (PINN, Closed-form, Difference)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    extent = [x1_grid.min(), x1_grid.max(), x2_grid.min(), x2_grid.max()]
    xlabel, ylabel = f'x[{dim1}]', f'x[{dim2}]'
    
    # Shared color range
    vmin_V = min(V_pinn.min(), V_cf.min())
    vmax_V = max(V_pinn.max(), V_cf.max())
    
    heat_basic(axs[0], V_pinn, 'V (PINN-PI)', extent, vmin=vmin_V, vmax=vmax_V,
               xlabel=xlabel, ylabel=ylabel)
    heat_basic(axs[1], V_cf, 'V (Closed-form)', extent, vmin=vmin_V, vmax=vmax_V,
               xlabel=xlabel, ylabel=ylabel)
    heat_diverging(axs[2], V_pinn - V_cf, 'V Difference', extent,
                   xlabel=xlabel, ylabel=ylabel)
    
    plt.suptitle(f'Liu ND: Value Function (τ={tau_fixed:.2f}, w={w_fixed:.2f})', fontsize=14)
    
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
                              save_path=None, show=True):
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
        
        heat_basic(axs[0, 0], theta_p, f'θ*[{asset_idx}]/w (PINN-PI)', extent,
                   vmin=vmin_t, vmax=vmax_t, xlabel=xlabel, ylabel=ylabel)
        heat_basic(axs[0, 1], theta_c, f'θ*[{asset_idx}]/w (Closed-form)', extent,
                   vmin=vmin_t, vmax=vmax_t, xlabel=xlabel, ylabel=ylabel)
        heat_diverging(axs[0, 2], theta_p - theta_c, f'θ*[{asset_idx}]/w Diff', extent,
                       xlabel=xlabel, ylabel=ylabel)
        
        # === Row 1: Myopic component ===
        vmin_m = min(myopic_p.min(), myopic_c.min())
        vmax_m = max(myopic_p.max(), myopic_c.max())
        
        heat_basic(axs[1, 0], myopic_p, f'Myopic[{asset_idx}] (PINN-PI)', extent,
                   vmin=vmin_m, vmax=vmax_m, xlabel=xlabel, ylabel=ylabel)
        heat_basic(axs[1, 1], myopic_c, f'Myopic[{asset_idx}] (Closed-form)', extent,
                   vmin=vmin_m, vmax=vmax_m, xlabel=xlabel, ylabel=ylabel)
        heat_diverging(axs[1, 2], myopic_p - myopic_c, f'Myopic[{asset_idx}] Diff', extent,
                       xlabel=xlabel, ylabel=ylabel)
        
        # === Row 2: Hedging component ===
        vmin_h = min(hedging_p.min(), hedging_c.min())
        vmax_h = max(hedging_p.max(), hedging_c.max())
        
        heat_basic(axs[2, 0], hedging_p, f'Hedging[{asset_idx}] (PINN-PI)', extent,
                   vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
        heat_basic(axs[2, 1], hedging_c, f'Hedging[{asset_idx}] (Closed-form)', extent,
                   vmin=vmin_h, vmax=vmax_h, xlabel=xlabel, ylabel=ylabel)
        heat_diverging(axs[2, 2], hedging_p - hedging_c, f'Hedging[{asset_idx}] Diff', extent,
                       xlabel=xlabel, ylabel=ylabel)
        
        plt.suptitle(f'Liu ND: Asset {asset_idx} Portfolio (τ={tau_fixed:.2f}, w={w_fixed:.2f})',
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

    plt.suptitle(f'Liu ND: Value Function (w={w_fixed:.2f})', fontsize=14)

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

            plt.suptitle(f'Liu ND: Asset {asset_idx} Portfolio [Rank {rank+1}/{len(assets_to_plot)}] (w={w_fixed:.2f})', fontsize=14)

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

            plt.suptitle(f'Liu ND: Asset {asset_idx} Hedging [Rank {rank+1}/{len(assets_to_plot)}] (w={w_fixed:.2f})', fontsize=14)

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
                   sol, lam0, Lam, Gamma, market_seed,
                   include_affine_reference=True):
    """Fixed Q_ev diagnostic set for per-iteration e_n and stability margins.

    Points are uniform over (0, tau_max] x Omega_ev at the PRIMARY margin,
    drawn from a MARKET-seed-derived RNG so every training seed and both
    methods use the SAME set. Closed-form V and the reduced derivative
    bundle (V_w, V_ww, grad_x V_w) are precomputed here once when the
    affine Riccati solution is an exact reference:

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

    dev = device
    out = {
        "w": torch.tensor(w_np.reshape(-1, 1), dtype=torch.float32, device=dev),
        "x": torch.tensor(x_np, dtype=torch.float32, device=dev),
        "tau": torch.tensor(tau_np.reshape(-1, 1), dtype=torch.float32, device=dev),
        "w_np": w_np, "x_np": x_np,
        "has_affine_reference": bool(include_affine_reference),
    }
    if include_affine_reference:
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
        out.update(V_cf=V_cf, Vw_cf=Vw_cf, Vww_cf=Vww_cf, Vwx_cf=Vwx_cf)
    return out


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

    Always returns model-side margins, guard fraction, and the improved
    normalized-control range on Q_ev.  The e_n and RelL2 fields are returned
    only when ``diag`` contains an exact affine reference; for tanh eps>0
    those fields would conflate deformation with solver error and are
    intentionally omitted.
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

    # Stability margins (model side)
    m_ww = float(np.min(-Vww_m))
    guard_frac = float(np.mean(Vww_m > -VWW_GUARD))
    lam_x = actual_risk_premium_numpy(diag["x_np"], lam0_np, Lam_np)  # (P, N)
    numer = lam_x * Vw_m[:, None] + Vwx_m @ Gamma_np.T            # (P, N)
    M_num = float(np.max(np.linalg.norm(numer, axis=1)))
    theta_hat = -numer / np.minimum(Vww_m, -VWW_GUARD)[:, None]
    vartheta_stats = normalized_control_stats(
        torch.from_numpy(theta_hat),
        torch.from_numpy(np.asarray(diag["w_np"]).reshape(-1, 1)),
    )

    out = {
        "m_ww": m_ww,
        "M_num": M_num,
        "guard_frac_ev": guard_frac,
        **vartheta_stats,
    }
    if not bool(diag.get("has_affine_reference", False)):
        return out

    # e_n on the fixed set: sup |V - V*| + sup ||bundle - bundle*||_2
    e_V = float(np.max(np.abs(V_m - diag["V_cf"])))
    bundle_err = np.concatenate([
        (Vw_m - diag["Vw_cf"]).reshape(-1, 1),
        (Vww_m - diag["Vww_cf"]).reshape(-1, 1),
        (Vwx_m - diag["Vwx_cf"]),
    ], axis=1)
    e_D = float(np.max(np.linalg.norm(bundle_err, axis=1)))
    e_Vw_sup = float(np.max(np.abs(Vw_m - diag["Vw_cf"])))
    e_Vww_sup = float(np.max(np.abs(Vww_m - diag["Vww_cf"])))
    e_Vwx_sup = float(np.max(np.linalg.norm(Vwx_m - diag["Vwx_cf"], axis=1)))

    # Per-outer norms on the SAME fixed diagnostic set (primary margin only).
    # Vartheta below uses the full-dimensional Table/E9 convention; the raw
    # theta norm is retained only for backward compatibility. Both controls
    # are derived from the ALREADY computed bundle (no extra autograd): the
    # model side reuses the training-side V_ww guard and the closed-form side
    # uses the exact (negative) V_ww*.
    rel_l2_V = float(np.linalg.norm(V_m - diag["V_cf"])
                     / max(np.linalg.norm(diag["V_cf"]), 1e-300))
    numer_cf = lam_x * diag["Vw_cf"][:, None] + diag["Vwx_cf"] @ Gamma_np.T
    theta_cf = -numer_cf / diag["Vww_cf"][:, None]
    rel_l2_theta = float(np.linalg.norm(theta_hat - theta_cf)
                         / max(np.linalg.norm(theta_cf), 1e-300))
    # The paper reports the volatility feedback vartheta = theta / w.  The
    # legacy raw-theta scalar is retained above, while this unweighted
    # normalized-control norm matches full-dimensional Table/E9 evaluation.
    w_col = np.asarray(diag["w_np"], dtype=np.float64).reshape(-1, 1)
    vartheta_hat = theta_hat / w_col
    vartheta_cf = theta_cf / w_col
    rel_l2_vartheta = float(
        np.linalg.norm(vartheta_hat - vartheta_cf)
        / max(np.linalg.norm(vartheta_cf), 1e-300)
    )

    out.update({
        "e_V_sup": e_V, "e_bundle_sup": e_D, "e_Xev": e_V + e_D,
        "e_Vw_sup": e_Vw_sup, "e_Vww_sup": e_Vww_sup, "e_Vwx_sup": e_Vwx_sup,
        "diag_RelL2_V": rel_l2_V,
        "diag_RelL2_theta": rel_l2_theta,
        "diag_RelL2_vartheta": rel_l2_vartheta,
    })
    return out


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
    ax.set_title('Liu ND PINN-PI: Training Loss History')
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


# -------------------------
# PI-PINN hyperparameters
# -------------------------
value_hidden = ARGS.value_hidden
value_depth  = ARGS.value_depth

outer_iters      = ARGS.outer_iters      # outer policy-iteration steps
eval_epochs      = ARGS.eval_epochs     # value-net epochs per outer step (linear PDE solve)
batch_size       = ARGS.batch_size
terminal_frac    = ARGS.terminal_frac

lr              = ARGS.lr
w_terminal      = ARGS.w_terminal     # terminal condition weight
w_shape         = ARGS.w_shape      # monotonicity/concavity penalty weight
w_rra           = ARGS.w_rra        # CRRA homogeneity penalty weight (0 disables)

theta_init_method = ARGS.theta_init_method  # {"myopic", "zero", "closed_form"}
theta_init_scale  = float(ARGS.theta_init_scale)  # theta_0 = scale * theta_init(method)
theta_clip_abs    = ARGS.theta_clip_abs       # None -> no clamp, else componentwise |θ_i| <= clip

print_every_outer = ARGS.print_every_outer
print_every_eval  = ARGS.print_every_eval
verbose_detail    = ARGS.verbose_detail


# =============================================================================
# 9) Main Execution
# =============================================================================
start = time.time()
if __name__ == "__main__":
    if ARGS.eval_only:
        # Evaluation is unrelated to training-divergence monitoring: no
        # "running" status on the TRAINING status file, no stopper, and the
        # shared stop flag is ignored entirely.
        recorder.prepare_eval_run()
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

    # Initialize PI-PINN solver
    # scheduler_patience default depends on the LR schedule scale:
    # inner_plateau counts inner epochs (10), outer_plateau counts outer iters (25).
    _sched_patience = ARGS.scheduler_patience
    if _sched_patience is None:
        _sched_patience = 10 if ARGS.lr_schedule == "inner_plateau" else 25
    solver = PIPINN_KimOmbergND(
        M=M_STATES, N=N_ASSETS,
        gamma=gamma, r=r,
        K_t=K_t, k0_t=k0_t, Q_t=Q_t,
        Gamma_t=Gamma_t, lam0_t=lam0_t, Lam_t=Lam_t,
        value_hidden=value_hidden,
        value_depth=value_depth,
        lr=lr,
        scheduler_patience=_sched_patience,
        scheduler_factor=ARGS.scheduler_factor,
        scheduler_min_lr=ARGS.scheduler_min_lr,
        lr_schedule=ARGS.lr_schedule,
        adam_reset=ARGS.adam_reset,
        carry_lr_min=ARGS.carry_lr_min,
        carry_lr_max=ARGS.carry_lr_max,
        theta_clip_abs=theta_clip_abs,
        device=device
    )

    # PI-PINN Training
    if ARGS.eval_only:
        print("\n[eval-only] Skipping training. Loading saved weights for evaluation.")
        results = {"stopped_early": False}
        elapsed = 0.0
    else:
        results = solver.run_policy_iteration(
            theta_init_scale=theta_init_scale,
            outer_iters=outer_iters,
            eval_epochs=eval_epochs,
            batch_size=batch_size,
            terminal_frac=terminal_frac,
            w_terminal=w_terminal,
            w_shape=w_shape,
            w_rra=w_rra,
            theta_init_method=theta_init_method,
            print_every_outer=print_every_outer,
            print_every_eval=print_every_eval,
            verbose_detail=verbose_detail,
            save_iterate_every=ARGS.save_iterate_every,
            e3b_checkpoints=ARGS.e3b_checkpoints,
            pe_resample_every=ARGS.pe_resample_every,
            inner_best_restore=bool(ARGS.inner_best_restore),
            sel_points=ARGS.sel_points,
            sel_terminal_points=ARGS.sel_terminal_points,
            sel_every=ARGS.sel_every,
            sel_patience=ARGS.sel_patience,
            pres_target=ARGS.pres_target,
            val_points=ARGS.val_points,
            val_terminal_points=ARGS.val_terminal_points,
            val_every=ARGS.val_every,
            # Held-out set is MARKET-seed derived: identical across training
            # seeds and both methods, so achieved p_res is directly comparable.
            val_seed=MARKET_SEED,
            diag_points=ARGS.diag_points,
            diag_margin=parse_eval_margins(ARGS.eval_margin)[0],
            diag_every=ARGS.diag_every,
            timing_mode=ARGS.timing_mode,
            recorder=recorder,
            stopper=stopper,
        )  
        elapsed = time.time() - start

    # E8: capture the TRAINING GPU peak before any evaluation allocates
    # memory, then reset so the evaluation peak is measured separately.
    _train_gpu_peak = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        _train_gpu_peak = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60

    if results.get("stopped_early", False):
        recorder.write_status(
            "stopped_early",
            elapsed_sec=elapsed,
            core_train_wall_sec=results.get("core_train_wall_sec"),
            **results.get("stop_info", {}),
        )
        print(f"Elapsed time: {h:02d}:{m:02d}:{s:05.2f}")
        sys.exit(0)

    print("\n" + "="*60)
    if HAS_AFFINE_REFERENCE:
        print("Evaluating PINN-PI vs Closed-form...")
    else:
        print("Evaluating PINN-PI non-affine residual and stability diagnostics...")
    print(f"{'='*60}")
    print(f"  hidden node      : {value_hidden}")
    print(f"  hidden layers    : {value_depth}")
    print(f"  outer_iters      : {outer_iters}")
    print(f"  eval_epochs      : {eval_epochs}")
    print(f"  batch size       : {batch_size}")
    print(f"  initial lr       : {lr}")
    print(f"  T.C weight       : {w_terminal}")
    print(f"  shape weight     : {w_shape}")
    print(f"  theta init       : {theta_init_method} (scale={theta_init_scale:g})")
    print(f"  theta clip abs   : {theta_clip_abs}")
    print(f"  risk premium     : {RISK_PREMIUM_MODE}, eps={NONAFFINE_EPS:g}")
    print(f"  Seed             : {SEED}")
    print(f"Elapsed time       : {h:02d}:{m:02d}:{s:05.2f}")
    print(f"{'='*70}")

    # FINAL-ITERATE POLICY: evaluation always uses the final PI iterate.
    # best is loaded only as a legacy fallback (older runs without final).
    final_weight_path = os.path.join(weight_dir, "value_net_final.pt")
    best_weight_path = os.path.join(weight_dir, "value_net_best.pt")
    last_weight_path = os.path.join(weight_dir, "value_net_last.pt")
    model = solver.value_net
    if ARGS.eval_only:
        if os.path.exists(final_weight_path):
            model.load_state_dict(torch.load(final_weight_path, map_location=device))
            print(f"[eval-only] Loaded FINAL iterate: {final_weight_path}")
        elif os.path.exists(last_weight_path):
            model.load_state_dict(torch.load(last_weight_path, map_location=device))
            print(f"[eval-only][warn] final weight not found; loaded last: {last_weight_path}")
        elif os.path.exists(best_weight_path):
            model.load_state_dict(torch.load(best_weight_path, map_location=device))
            print(f"[eval-only][warn] final/last not found; loaded BEST (legacy run): {best_weight_path}")
        else:
            # FAIL FAST: evaluating a freshly initialized network would
            # silently produce garbage paper numbers.
            msg = f"no saved weights (final/last/best) under {weight_dir}"
            print(f"[eval-only][FATAL] {msg}")
            recorder.mark_failed_eval(reason=msg)
            sys.exit(1)
    # (Normal runs: the in-memory model already IS the final iterate.)

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

    # Plot PI-PINN convergence (figure only; suppressed under --skip-figures/
    # --skip-plots, and in eval-only mode which has no training history).
    if not SKIP_FIGURES and not ARGS.eval_only:
        plot_pi_convergence_nd(
            results,
            save_path=os.path.join(output_dir, "pi_pinn_convergence.png"),
            show=True
        )
    elif ARGS.eval_only:
        print("[eval-only] Skipping pi_pinn_convergence plot (requires training history).")

    # Evaluation
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
    
    # For eps>0 there is no exact solution.  Record only quantities that remain
    # meaningful: the final held-out frozen-policy residual on Q_col and
    # model-side stability diagnostics.  The affine-reference error columns
    # below are deliberately not produced in this branch.
    if not HAS_AFFINE_REFERENCE:
        _last_outer = {}
        if os.path.exists(recorder.outer_csv):
            with open(recorder.outer_csv, newline="", encoding="utf-8") as _f:
                for _row in csv.DictReader(_f):
                    _last_outer = _row

        _nonaffine_metrics = [
            ("frozen_policy_heldout", "", "val_pde_rms"),
            ("frozen_policy_heldout", "", "val_terminal_rms"),
            ("frozen_policy_heldout", "", "val_pres"),
            ("model_diagnostic", primary_margin, "m_ww"),
            ("model_diagnostic", primary_margin, "M_num"),
            ("model_diagnostic", primary_margin, "guard_frac_ev"),
            ("frozen_policy_diagnostic", "", "lam_min_sigma_frozen"),
            ("frozen_policy_diagnostic", "", "lam_max_sigma_frozen"),
            ("frozen_policy_diagnostic", "", "clip_frac_frozen"),
        ]
        _rows = []
        for _scope, _margin, _metric in _nonaffine_metrics:
            _raw = _last_outer.get(_metric, "")
            if _raw in (None, ""):
                continue
            try:
                _value = float(_raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(_value):
                continue
            _rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_type": ARGS.model_type,
                "run_tag": ARGS.run_tag,
                "scope": _scope,
                "eval_margin": _margin,
                "metric": _metric,
                "value": _value,
            })
        append_csv_rows(recorder.metrics_csv, _rows, metric_fields)
        print("\n--- Non-affine final diagnostics (no closed-form accuracy claim) ---")
        for _row in _rows:
            print(f"  {_row['metric']}: {_row['value']:.6e}")

    # Independent FULL-DIMENSIONAL test evaluation on Omega_ev (Table / E9).
    # All coordinates vary; the same base points are mapped into every margin
    # window so nested-window results are directly comparable. Printed BEFORE
    # the (tau, x_0) grid slices; per-asset metrics go to metrics.csv only.
    if HAS_AFFINE_REFERENCE and ARGS.test_points and ARGS.test_points > 0:
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
    if not HAS_AFFINE_REFERENCE and not SKIP_FIGURES:
        print(
            "[non-affine] Skipping built-in affine comparison heatmaps; "
            "use postprocess_nonaffine.py for the paired homotopy figure."
        )
    for w_test in ([] if (SKIP_FIGURES or not HAS_AFFINE_REFERENCE) else W_levels):
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
            save_path=os.path.join(output_dir, f"value_tauX_w{w_test:.2f}.png"),
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
            save_path=os.path.join(output_dir, f"portfolio_tauX_w{w_test:.2f}.png"),
            show=True, only_hedge=True,
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
                              pres_max=results.get("pres_max"),
                              total_inner_steps=results.get("total_inner_steps"),
                              total_optimizer_steps=results.get("total_inner_steps"),
                              train_wall_sec=elapsed,
                              core_train_wall_sec=results.get("core_train_wall_sec"),
                              theta0_norm_max=results.get("theta0_norm_max"),
                              theta0_lam_min_sigma=results.get("theta0_lam_min_sigma"),
                              theta0_lam_max_sigma=results.get("theta0_lam_max_sigma"),
                              timing_mode=bool(ARGS.timing_mode),
                              train_gpu_peak_mem_bytes=_train_gpu_peak,
                              eval_gpu_peak_mem_bytes=_eval_gpu_peak,
                              eval_margins=EVAL_MARGINS)
