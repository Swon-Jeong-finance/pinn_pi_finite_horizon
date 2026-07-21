#!/usr/bin/env python3
"""
make_paper_figures.py
=====================

Standalone plotting and metrics-summary utility for saved Liu/Kim-Omberg
ND PINN / PI-PINN sweep outputs.

It does NOT train models. It reads saved outputs such as:

  run_dir/config.json
  run_dir/train_history.csv
  run_dir/outer_history.csv
  run_dir/metrics.csv
  run_dir/market_params.npz
  run_dir/closed_form_ode.npz
  weight_dir/value_net_best.pt

Typical sweep layout:

  OUT_ROOT/
    pinn/pinn_baseline/
    pinn/pinn_m_states1/
    pi-pinn/pipinn_baseline/
    pi-pinn/pipinn_m_states1/
    weights/pinn/pinn_baseline/
    weights/pi-pinn/pipinn_baseline/

Recommended example:

  python make_paper_figures.py \
    --sweep-root /workspace/outputs/my_run \
    --exp-list baseline,m_states1,m_states3,m_states5 \
    --models both \
    --summary-metrics \
    --plot-train \
    --plot-policy-convergence \
    --plot-value \
    --plot-portfolio \
    --format png --dpi 300 --font-size 14

Design choices:
  - Figure/subplot titles are intentionally omitted for paper use.
  - Console print() messages identify every figure that is saved.
  - PINN train curve uses per-pseudo-outer best loss from train_history.csv.
  - PI-PINN train curve uses outer_history.csv eval_loss for the default total loss.
  - Metrics are summarized into long, wide, and compact paper-table CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import torch
import torch.nn as nn


# =============================================================================
# Basic helpers
# =============================================================================


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            out.append(part)
    return out


def parse_float_list(text: Optional[str]) -> List[float]:
    return [float(x) for x in parse_list(text)]


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    vals = parse_list(text)
    if not vals:
        return None
    return [int(x) for x in vals]


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def to_float(x: Any, default: float = math.nan) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '' or s.lower() in {'nan', 'none', 'null'}:
        return default
    try:
        return float(s)
    except Exception:
        return default


def to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == '':
            return default
        return int(float(x))
    except Exception:
        return default


def cfg_get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Read config values from either config[key] or config['args'][key]."""
    if key in config:
        return config[key]
    args = config.get('args', {})
    if isinstance(args, dict) and key in args:
        return args[key]
    # Also support kebab spelling just in case.
    kebab = key.replace('_', '-')
    if isinstance(args, dict) and kebab in args:
        return args[kebab]
    return default


def get_first_np_scalar(npz: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in npz.files:
        return default
    arr = npz[key]
    try:
        return arr.reshape(-1)[0].item()
    except Exception:
        return default


def safe_label_for_model(model_key: str) -> str:
    if model_key == 'pinn':
        return 'PINN'
    if model_key == 'pipinn':
        return 'PINN-PI'
    return str(model_key)


def sci_fmt(x: Any, precision: int = 2) -> str:
    val = to_float(x)
    if math.isnan(val):
        return ''
    return f"{val:.{precision}e}"


def resolve_device(device_spec: str) -> torch.device:
    spec = str(device_spec or 'auto').strip()
    if spec.lower() in {'auto', ''}:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if spec.startswith('cuda') and not torch.cuda.is_available():
        print(f"[warn] requested device={spec}, but CUDA is unavailable; using CPU.")
        return torch.device('cpu')
    return torch.device(spec)


# =============================================================================
# Run discovery
# =============================================================================


@dataclass
class RunInfo:
    model_key: str       # 'pinn' or 'pipinn'
    suffix: str          # e.g. baseline, m_states1
    run_name: str        # e.g. pinn_baseline, pipinn_m_states1
    run_dir: Path
    weight_dir: Path

    @property
    def method_label(self) -> str:
        return safe_label_for_model(self.model_key)


def model_output_subdir(model_key: str) -> str:
    if model_key == 'pinn':
        return 'pinn'
    if model_key == 'pipinn':
        return 'pi-pinn'
    raise ValueError(f"Unknown model_key={model_key!r}")


def model_weight_subdir(model_key: str) -> str:
    if model_key == 'pinn':
        return 'pinn'
    if model_key == 'pipinn':
        return 'pi-pinn'
    raise ValueError(f"Unknown model_key={model_key!r}")


def infer_suffix_from_run_name(model_key: str, run_name: str) -> str:
    if model_key == 'pinn' and run_name.startswith('pinn_'):
        return run_name[len('pinn_'):]
    if model_key == 'pipinn' and run_name.startswith('pipinn_'):
        return run_name[len('pipinn_'):]
    return run_name


def run_name_from_suffix(model_key: str, suffix_or_name: str) -> Tuple[str, str]:
    """Return (run_name, suffix). Accepts either suffix or full run name."""
    s = suffix_or_name.strip()
    if model_key == 'pinn':
        if s.startswith('pinn_'):
            return s, s[len('pinn_'):]
        return f'pinn_{s}', s
    if model_key == 'pipinn':
        if s.startswith('pipinn_'):
            return s, s[len('pipinn_'):]
        return f'pipinn_{s}', s
    raise ValueError(model_key)


def resolve_run(sweep_root: Path, model_key: str, suffix_or_name: str) -> RunInfo:
    run_name, suffix = run_name_from_suffix(model_key, suffix_or_name)
    run_dir = sweep_root / model_output_subdir(model_key) / run_name
    weight_dir = sweep_root / 'weights' / model_weight_subdir(model_key) / run_name

    # Prefer the path saved in config.json if present.
    config_path = run_dir / 'config.json'
    if config_path.exists():
        try:
            cfg = read_json(config_path)
            cfg_weight_dir = cfg.get('weight_dir') or cfg_get(cfg, 'weight_root', None)
            if cfg_weight_dir:
                wd = Path(str(cfg_weight_dir))
                if wd.exists():
                    weight_dir = wd
        except Exception as e:
            print(f"[warn] could not read config for {run_name}: {e}")

    return RunInfo(model_key=model_key, suffix=suffix, run_name=run_name, run_dir=run_dir, weight_dir=weight_dir)


def resolve_runs_from_args(args: argparse.Namespace, *, for_heatmaps: bool = False) -> List[RunInfo]:
    sweep_root = Path(args.sweep_root).expanduser().resolve()
    models: List[str]
    if args.models == 'both':
        models = ['pinn', 'pipinn']
    else:
        models = [args.models]

    exp_text = args.heatmap_exp_list if for_heatmaps and args.heatmap_exp_list else args.exp_list
    suffixes = parse_list(exp_text)
    runs: List[RunInfo] = []
    missing: List[str] = []

    # Optional explicit full run names override suffix expansion.
    explicit_by_model = {
        'pinn': parse_list(args.pinn_runs),
        'pipinn': parse_list(args.pipinn_runs),
    }

    for model_key in models:
        explicit = explicit_by_model.get(model_key, [])
        entries = explicit if explicit else suffixes
        for item in entries:
            info = resolve_run(sweep_root, model_key, item)
            if not info.run_dir.exists():
                missing.append(str(info.run_dir))
                continue
            runs.append(info)

    if missing:
        msg = "\n".join(f"  - {m}" for m in missing)
        if args.strict:
            raise FileNotFoundError(f"Missing run directories:\n{msg}")
        print(f"[warn] missing run directories; skipping:\n{msg}")

    return runs


# =============================================================================
# Plot style / save helpers
# =============================================================================


def apply_plot_style(args: argparse.Namespace) -> None:
    plt.rcParams.update({
        'font.size': args.font_size,
        'axes.labelsize': args.label_size,
        'xtick.labelsize': args.tick_size,
        'ytick.labelsize': args.tick_size,
        'legend.fontsize': args.legend_size,
        'figure.titlesize': args.font_size,
        'axes.titlesize': args.font_size,
        'lines.linewidth': args.line_width,
    })
    if args.font_family:
        plt.rcParams['font.family'] = args.font_family


def common_fig_root(args: argparse.Namespace) -> Path:
    if args.fig_root:
        return Path(args.fig_root).expanduser().resolve()
    return Path(args.sweep_root).expanduser().resolve() / 'paper_figures'


def run_plot_dir(run: RunInfo, args: argparse.Namespace) -> Path:
    if args.fig_root:
        return common_fig_root(args) / run.model_key / run.run_name
    return run.run_dir / 'plots'


def summary_dir(args: argparse.Namespace) -> Path:
    return common_fig_root(args) / 'summary'


def train_fig_dir(args: argparse.Namespace) -> Path:
    return common_fig_root(args) / 'train'


def save_fig(fig: plt.Figure, path: Path, args: argparse.Namespace, *, kind: str, tight: bool = True) -> None:
    ensure_dir(path.parent)
    if tight:
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
    else:
        fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"[figure] {kind} saved: {path}")


# =============================================================================
# Neural net and closed-form utilities
# =============================================================================


class ValueNetND(nn.Module):
    """Same architecture as the training scripts."""
    def __init__(self, M: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.M = M
        in_dim = M + 2  # (w, x_1, ..., x_M, tau)
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, w: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([w, x, tau], dim=1))


def load_weight_path(run: RunInfo, args: argparse.Namespace) -> Path:
    candidates = []
    if args.weight_name:
        candidates.append(run.weight_dir / args.weight_name)
    candidates.extend([
        run.weight_dir / 'value_net_best.pt',
        run.weight_dir / 'value_net_last.pt',
    ])
    # Also search legacy descriptive best weights if needed.
    if run.weight_dir.exists():
        candidates.extend(sorted(run.weight_dir.glob('value_net_best_*.pt')))

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find weight file for {run.run_name} in {run.weight_dir}")


def load_run_objects(run: RunInfo, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    cfg_path = run.run_dir / 'config.json'
    market_path = run.run_dir / 'market_params.npz'
    cf_path = run.run_dir / 'closed_form_ode.npz'

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    if not market_path.exists():
        raise FileNotFoundError(f"Missing market_params.npz: {market_path}")
    if not cf_path.exists():
        raise FileNotFoundError(f"Missing closed_form_ode.npz: {cf_path}")

    cfg = read_json(cfg_path)
    market = np.load(market_path)
    cf = np.load(cf_path)

    xbar = market['xbar']
    lam0 = market['lam0']
    M = int(xbar.shape[0])
    N = int(lam0.shape[0])

    hidden = int(cfg_get(cfg, 'value_hidden', args.value_hidden_default))
    depth = int(cfg_get(cfg, 'value_depth', args.value_depth_default))
    gamma = float(get_first_np_scalar(market, 'gamma', cfg_get(cfg, 'gamma', 2.0)))
    r = float(get_first_np_scalar(market, 'r', cfg_get(cfg, 'r', 0.03)))
    tau_max = float(get_first_np_scalar(market, 'tau_max', cfg_get(cfg, 'tau_max', 3.0)))

    model = ValueNetND(M=M, hidden=hidden, depth=depth).to(device)
    weight_path = load_weight_path(run, args)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[load] {run.run_name}: weight={weight_path}, hidden={hidden}, depth={depth}, N={N}, M={M}")

    return {
        'cfg': cfg,
        'market': market,
        'cf': cf,
        'model': model,
        'weight_path': weight_path,
        'M': M,
        'N': N,
        'gamma': gamma,
        'r': r,
        'tau_max': tau_max,
    }


def get_closed_form_at_tau(tau: float, cf: np.lib.npyio.NpzFile, M: int) -> Tuple[float, np.ndarray, np.ndarray]:
    t_grid = cf['t']
    y = cf['y']
    y_tau = np.array([np.interp(float(tau), t_grid, y[i]) for i in range(y.shape[0])])
    a = float(y_tau[0])
    b = y_tau[1:1 + M]
    C = y_tau[1 + M:].reshape(M, M)
    C = 0.5 * (C + C.T)
    return a, b, C


def closed_form_V(tau: float, w: float, x: np.ndarray, cf: np.lib.npyio.NpzFile,
                  M: int, gamma: float, r: float) -> float:
    a, b, C = get_closed_form_at_tau(tau, cf, M)
    phi = np.exp(a + b @ x + 0.5 * x @ C @ x)
    discount = np.exp((1.0 - gamma) * r * tau)
    U_w = np.power(w, 1.0 - gamma) / (1.0 - gamma)
    return float(discount * U_w * phi)


def closed_form_decomposition(tau: float, w: float, x: np.ndarray,
                              cf: np.lib.npyio.NpzFile, market: np.lib.npyio.NpzFile,
                              M: int, N: int, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, b, C = get_closed_form_at_tau(tau, cf, M)
    lam0 = market['lam0']
    Lam = market['Lam']
    Gamma = market['Gamma']
    lam_x = lam0 + Lam @ x
    grad_log_phi = b + C @ x
    myopic_norm = lam_x / gamma
    hedging_norm = (Gamma @ grad_log_phi) / gamma
    theta_norm = myopic_norm + hedging_norm
    theta = w * theta_norm
    return theta, theta_norm, myopic_norm, hedging_norm


def compute_optimal_theta_nd(model: nn.Module, w: torch.Tensor, x: torch.Tensor, tau: torch.Tensor,
                             M: int, N: int, Gamma_t: torch.Tensor, lam0_t: torch.Tensor,
                             Lam_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V, theta, theta/w, myopic, hedging using FOC."""
    V = model(w, x, tau)
    ones = torch.ones_like(V)

    # First derivative must create graph so V_wx and V_ww can be computed.
    V_w = torch.autograd.grad(V, w, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    V_wx = torch.autograd.grad(V_w, x, grad_outputs=torch.ones_like(V_w), create_graph=False, retain_graph=True)[0]
    V_ww = torch.autograd.grad(V_w, w, grad_outputs=torch.ones_like(V_w), create_graph=False, retain_graph=True)[0]

    lam_x = lam0_t.unsqueeze(0) + torch.einsum('ij,bj->bi', Lam_t, x)
    Gamma_Vwx = torch.einsum('ij,bj->bi', Gamma_t, V_wx)
    numerator = lam_x * V_w + Gamma_Vwx
    V_ww_safe = torch.where(torch.abs(V_ww) < 1e-8,
                            torch.sign(V_ww) * 1e-8 + 1e-10,
                            V_ww)
    theta = -numerator / V_ww_safe
    theta_norm = theta / w

    # Preserve training/evaluation script convention: PINN-derived CRRA coefficient.
    pinn_coeff = -V_w / (w * V_ww_safe)
    myopic_norm = pinn_coeff * lam_x
    hedging_norm = theta_norm - myopic_norm
    return V, theta, theta_norm, myopic_norm, hedging_norm


# =============================================================================
# Grid evaluation
# =============================================================================


def eval_model_on_tau_X_grid(model: nn.Module, market: np.lib.npyio.NpzFile, *,
                             w_fixed: float, dimX: int, x_fixed: np.ndarray,
                             device: torch.device, N_tau: int, N_X: int,
                             tau_min: float, tau_max: float, chunk: int = 4096
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = int(market['xbar'].shape[0])
    N = int(market['lam0'].shape[0])
    X_min = market['X_min']
    X_max = market['X_max']

    tau_vals = np.linspace(tau_min, tau_max, N_tau)
    X_vals = np.linspace(X_min[dimX], X_max[dimX], N_X)
    X_grid, tau_grid = np.meshgrid(X_vals, tau_vals, indexing='xy')

    n_points = N_tau * N_X
    x_full = np.tile(x_fixed, (n_points, 1))
    x_full[:, dimX] = X_grid.reshape(-1)

    w_flat = torch.full((n_points, 1), float(w_fixed), device=device, dtype=torch.float32, requires_grad=True)
    x_flat = torch.tensor(x_full, device=device, dtype=torch.float32, requires_grad=True)
    tau_flat = torch.tensor(tau_grid.reshape(-1, 1), device=device, dtype=torch.float32)

    Gamma_t = torch.tensor(market['Gamma'], device=device, dtype=torch.float32)
    lam0_t = torch.tensor(market['lam0'], device=device, dtype=torch.float32)
    Lam_t = torch.tensor(market['Lam'], device=device, dtype=torch.float32)

    V_list: List[torch.Tensor] = []
    theta_list: List[torch.Tensor] = []
    myopic_list: List[torch.Tensor] = []
    hedging_list: List[torch.Tensor] = []

    for i in range(0, n_points, chunk):
        w_b = w_flat[i:i + chunk]
        x_b = x_flat[i:i + chunk]
        tau_b = tau_flat[i:i + chunk]
        V_b, _, theta_norm_b, myopic_b, hedging_b = compute_optimal_theta_nd(
            model, w_b, x_b, tau_b, M, N, Gamma_t, lam0_t, Lam_t
        )
        V_list.append(V_b.detach().cpu())
        theta_list.append(theta_norm_b.detach().cpu())
        myopic_list.append(myopic_b.detach().cpu())
        hedging_list.append(hedging_b.detach().cpu())

    V_pinn = torch.cat(V_list, dim=0).numpy().reshape(N_tau, N_X)
    theta_pinn = torch.cat(theta_list, dim=0).numpy().reshape(N_tau, N_X, N)
    myopic_pinn = torch.cat(myopic_list, dim=0).numpy().reshape(N_tau, N_X, N)
    hedging_pinn = torch.cat(hedging_list, dim=0).numpy().reshape(N_tau, N_X, N)

    return tau_grid, X_grid, V_pinn, theta_pinn, myopic_pinn, hedging_pinn


def eval_closed_form_on_tau_X_grid(market: np.lib.npyio.NpzFile, cf: np.lib.npyio.NpzFile, *,
                                   w_fixed: float, dimX: int, x_fixed: np.ndarray,
                                   N_tau: int, N_X: int, tau_min: float, tau_max: float,
                                   gamma: float, r: float
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = int(market['xbar'].shape[0])
    N = int(market['lam0'].shape[0])
    X_min = market['X_min']
    X_max = market['X_max']

    tau_vals = np.linspace(tau_min, tau_max, N_tau)
    X_vals = np.linspace(X_min[dimX], X_max[dimX], N_X)
    X_grid, tau_grid = np.meshgrid(X_vals, tau_vals, indexing='xy')

    V_cf = np.zeros((N_tau, N_X))
    theta_cf = np.zeros((N_tau, N_X, N))
    myopic_cf = np.zeros((N_tau, N_X, N))
    hedging_cf = np.zeros((N_tau, N_X, N))

    for itau, tau in enumerate(tau_vals):
        for jx, xv in enumerate(X_vals):
            x = x_fixed.copy()
            x[dimX] = float(xv)
            V_cf[itau, jx] = closed_form_V(float(tau), float(w_fixed), x, cf, M, gamma, r)
            _, theta_cf[itau, jx, :], myopic_cf[itau, jx, :], hedging_cf[itau, jx, :] = \
                closed_form_decomposition(float(tau), float(w_fixed), x, cf, market, M, N, gamma)

    return tau_grid, X_grid, V_cf, theta_cf, myopic_cf, hedging_cf


# =============================================================================
# Heatmap plotting: no titles by design
# =============================================================================


def heat_basic_tauX(ax: plt.Axes, Z: np.ndarray, extent: Sequence[float], args: argparse.Namespace,
                    *, cmap: str = 'jet', vmin: Optional[float] = None, vmax: Optional[float] = None,
                    xlabel: str = 'X', ylabel: str = r'$\tau$'):
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=args.tick_size)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=args.cbar_tick_size)
    # 작은 값일 때 scientific notation으로 표기해 cbar 라벨 폭을 고정
    cbar.formatter.set_powerlimits((-4, 4))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_x(3.5)
    return im


def heat_diverging_tauX(ax: plt.Axes, Z: np.ndarray, extent: Sequence[float], args: argparse.Namespace,
                        *, cmap: str = 'RdBu_r', pct: float = 98,
                        xlabel: str = 'X', ylabel: str = r'$\tau$'):
    abs_max = np.percentile(np.abs(Z), pct)
    if abs_max < 1e-10:
        abs_max = max(float(np.abs(Z).max()), 1e-10)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    im = ax.imshow(Z, origin='lower', aspect='auto', extent=extent,
                   interpolation='bilinear', cmap=cmap, norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=args.tick_size)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=args.cbar_tick_size)
    cbar.formatter.set_powerlimits((-4, 4))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    offset_text = cbar.ax.yaxis.get_offset_text()
    offset_text.set_x(3.5)
    return im


def plot_value_comparison_tauX(tau_grid: np.ndarray, X_grid: np.ndarray, V_model: np.ndarray, V_cf: np.ndarray,
                               args: argparse.Namespace, save_path: Path,
                               *, xlabel: str = 'risk premium X', ylabel: str = r'$\tau$') -> None:
    extent = [float(X_grid.min()), float(X_grid.max()), float(tau_grid.min()), float(tau_grid.max())]
    vmin = float(min(V_model.min(), V_cf.min()))
    vmax = float(max(V_model.max(), V_cf.max()))

    # -------------------------------------------------------------------------
    # Separate panel mode: save V_model, V_closed_form, and V_diff separately.
    # -------------------------------------------------------------------------
    if getattr(args, 'value_layout', 'grid') == 'separate':
        suffix = save_path.suffix
        base = save_path.with_suffix('')

        panel_specs = [
            ('model', V_model, False, vmin, vmax),
            ('closed_form', V_cf, False, vmin, vmax),
            ('diff', V_model - V_cf, True, None, None),
        ]

        for panel_name, Z, diverging, panel_vmin, panel_vmax in panel_specs:
            fig = plt.figure(figsize=(args.value_panel_fig_width, args.value_panel_fig_height))
            ax = fig.add_axes([0.18, 0.18, 0.62, 0.75])

            if diverging:
                heat_diverging_tauX(
                    ax,
                    Z,
                    extent,
                    args,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )
            else:
                heat_basic_tauX(
                    ax,
                    Z,
                    extent,
                    args,
                    vmin=panel_vmin,
                    vmax=panel_vmax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

            panel_path = base.parent / f"{base.name}_{panel_name}{suffix}"
            save_fig(fig, panel_path, args, kind=f'value tau-X single panel: {panel_name}', tight=False)

        return

    # -------------------------------------------------------------------------
    # Default grid mode: original 1x3 value figure.
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(
        1, 3,
        figsize=(args.value_fig_width, args.value_fig_height),
        constrained_layout=True,
    )

    heat_basic_tauX(
        axs[0],
        V_model,
        extent,
        args,
        vmin=vmin,
        vmax=vmax,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    heat_basic_tauX(
        axs[1],
        V_cf,
        extent,
        args,
        vmin=vmin,
        vmax=vmax,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    heat_diverging_tauX(
        axs[2],
        V_model - V_cf,
        extent,
        args,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    save_fig(fig, save_path, args, kind='value tau-X')


def choose_assets(theta_cf: np.ndarray, N: int, args: argparse.Namespace) -> List[int]:
    explicit = parse_int_list(args.assets)
    if explicit is not None:
        return [i for i in explicit if 0 <= i < N]
    ranges = []
    for asset_idx in range(N):
        tc = theta_cf[:, :, asset_idx]
        ranges.append((asset_idx, float(tc.max() - tc.min())))
    if args.sort_by_range:
        ranges.sort(key=lambda z: z[1], reverse=True)
    return [idx for idx, _ in ranges[:args.max_assets]]


def plot_portfolio_comparison_tauX(tau_grid: np.ndarray, X_grid: np.ndarray,
                                   theta_model: np.ndarray, theta_cf: np.ndarray,
                                   myopic_model: np.ndarray, myopic_cf: np.ndarray,
                                   hedging_model: np.ndarray, hedging_cf: np.ndarray,
                                   args: argparse.Namespace, save_dir: Path,
                                   *, prefix: str,
                                   xlabel: str = 'risk premium X', ylabel: str = r'$\tau$') -> List[Path]:
    N = int(theta_cf.shape[-1])
    assets_to_plot = choose_assets(theta_cf, N, args)
    extent = [float(X_grid.min()), float(X_grid.max()), float(tau_grid.min()), float(tau_grid.max())]
    ensure_dir(save_dir)
    saved: List[Path] = []

    print(f"[portfolio] {prefix}: plotting assets {assets_to_plot}")

    def save_single_panel(
        Z: np.ndarray,
        *,
        path: Path,
        kind: str,
        diverging: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Save one heatmap panel as an individual figure.
        Uses fixed-position axes so every panel has identical heatmap dimensions,
        regardless of cbar tick label width or offset text presence.
        """
        fig = plt.figure(figsize=(args.single_panel_fig_width, args.single_panel_fig_height))
        # [left, bottom, width, height] in figure-fraction coordinates.
        # Fix the heatmap rectangle so all panels share identical axes geometry.
        ax = fig.add_axes([0.18, 0.18, 0.62, 0.75])
    
        if diverging:
            heat_diverging_tauX(ax, Z, extent, args, xlabel=xlabel, ylabel=ylabel)
        else:
            heat_basic_tauX(ax, Z, extent, args, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel)
    
        save_fig(fig, path, args, kind=kind, tight=False)
        saved.append(path)

    # -------------------------------------------------------------------------
    # Hedge-only mode
    # -------------------------------------------------------------------------
    if args.portfolio_components == 'hedge':
        for rank, asset_idx in enumerate(assets_to_plot, start=1):
            hp = hedging_model[:, :, asset_idx]
            hc = hedging_cf[:, :, asset_idx]
            vmin = float(min(hp.min(), hc.min()))
            vmax = float(max(hp.max(), hc.max()))

            if args.portfolio_layout == 'separate':
                panel_specs = [
                    ('hedging_model', hp, False, vmin, vmax),
                    ('hedging_closed_form', hc, False, vmin, vmax),
                    ('hedging_diff', hp - hc, True, None, None),
                ]

                for panel_name, Z, diverging, panel_vmin, panel_vmax in panel_specs:
                    path = save_dir / f"{prefix}_rank{rank:02d}_asset{asset_idx}_{panel_name}.{args.format}"
                    save_single_panel(
                        Z,
                        path=path,
                        kind=f'portfolio hedge tau-X single panel: {panel_name}',
                        diverging=diverging,
                        vmin=panel_vmin,
                        vmax=panel_vmax,
                    )

            else:
                fig, axs = plt.subplots(
                    1, 3,
                    figsize=(args.portfolio_fig_width, args.single_row_fig_height),
                    constrained_layout=True,
                )
                heat_basic_tauX(axs[0], hp, extent, args, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel)
                heat_basic_tauX(axs[1], hc, extent, args, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel)
                heat_diverging_tauX(axs[2], hp - hc, extent, args, xlabel=xlabel, ylabel=ylabel)

                path = save_dir / f"{prefix}_rank{rank:02d}_asset{asset_idx}.{args.format}"
                save_fig(fig, path, args, kind='portfolio hedge tau-X')
                saved.append(path)

        return saved

    # -------------------------------------------------------------------------
    # All-components mode: theta / myopic / hedging
    # -------------------------------------------------------------------------
    for rank, asset_idx in enumerate(assets_to_plot, start=1):
        component_specs = [
            ('theta', theta_model[:, :, asset_idx], theta_cf[:, :, asset_idx]),
            ('myopic', myopic_model[:, :, asset_idx], myopic_cf[:, :, asset_idx]),
            ('hedging', hedging_model[:, :, asset_idx], hedging_cf[:, :, asset_idx]),
        ]

        if args.portfolio_layout == 'separate':
            for component_name, zp, zc in component_specs:
                vmin = float(min(zp.min(), zc.min()))
                vmax = float(max(zp.max(), zc.max()))

                panel_specs = [
                    (f'{component_name}_model', zp, False, vmin, vmax),
                    (f'{component_name}_closed_form', zc, False, vmin, vmax),
                    (f'{component_name}_diff', zp - zc, True, None, None),
                ]

                for panel_name, Z, diverging, panel_vmin, panel_vmax in panel_specs:
                    path = save_dir / f"{prefix}_rank{rank:02d}_asset{asset_idx}_{panel_name}.{args.format}"
                    save_single_panel(
                        Z,
                        path=path,
                        kind=f'portfolio tau-X single panel: {panel_name}',
                        diverging=diverging,
                        vmin=panel_vmin,
                        vmax=panel_vmax,
                    )

        else:
            fig, axs = plt.subplots(
                3, 3,
                figsize=(args.portfolio_fig_width, args.portfolio_fig_height),
                constrained_layout=True,
            )

            for row_idx, (component_name, zp, zc) in enumerate(component_specs):
                vmin = float(min(zp.min(), zc.min()))
                vmax = float(max(zp.max(), zc.max()))

                heat_basic_tauX(
                    axs[row_idx, 0],
                    zp,
                    extent,
                    args,
                    vmin=vmin,
                    vmax=vmax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )
                heat_basic_tauX(
                    axs[row_idx, 1],
                    zc,
                    extent,
                    args,
                    vmin=vmin,
                    vmax=vmax,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )
                heat_diverging_tauX(
                    axs[row_idx, 2],
                    zp - zc,
                    extent,
                    args,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

            path = save_dir / f"{prefix}_rank{rank:02d}_asset{asset_idx}.{args.format}"
            save_fig(fig, path, args, kind='portfolio tau-X')
            saved.append(path)

    return saved


# =============================================================================
# Training / convergence curves
# =============================================================================


def pinn_outer_best_curve(run: RunInfo, *, train_loss: str = 'total') -> Tuple[np.ndarray, np.ndarray]:
    path = run.run_dir / 'train_history.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv_rows(path)
    metric_col = {
        'total': 'total_loss',
        'pde': 'pde_loss',
        'terminal': 'terminal_loss',
        'concavity': 'concavity_loss',
    }.get(train_loss, 'total_loss')

    best_by_outer: Dict[int, float] = {}
    for row in rows:
        outer = to_int(row.get('outer_iter'))
        val = to_float(row.get(metric_col))
        if outer is None or math.isnan(val):
            continue
        if outer not in best_by_outer or val < best_by_outer[outer]:
            best_by_outer[outer] = val

    xs = np.array(sorted(best_by_outer.keys()), dtype=float)
    ys = np.array([best_by_outer[int(x)] for x in xs], dtype=float)
    return xs, ys


def pipinn_outer_curve(run: RunInfo, *, train_loss: str = 'total') -> Tuple[np.ndarray, np.ndarray]:
    outer_path = run.run_dir / 'outer_history.csv'
    if not outer_path.exists():
        raise FileNotFoundError(outer_path)

    if train_loss == 'total':
        rows = read_csv_rows(outer_path)
        xs, ys = [], []
        for row in rows:
            outer = to_int(row.get('outer_iter'))
            val = to_float(row.get('eval_loss'))
            if outer is None or math.isnan(val):
                continue
            xs.append(outer)
            ys.append(val)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    # For non-total components, compute per-outer best from inner train_history.csv.
    train_path = run.run_dir / 'train_history.csv'
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    rows = read_csv_rows(train_path)
    metric_col = {
        'pde': 'pde_loss',
        'terminal': 'terminal_loss',
        'concavity': 'concavity_loss',
        'monotonicity': 'monotonicity_loss',
    }.get(train_loss, 'total_loss')
    best_by_outer: Dict[int, float] = {}
    for row in rows:
        outer = to_int(row.get('outer_iter'))
        val = to_float(row.get(metric_col))
        if outer is None or math.isnan(val):
            continue
        if outer not in best_by_outer or val < best_by_outer[outer]:
            best_by_outer[outer] = val
    xs = np.array(sorted(best_by_outer.keys()), dtype=float)
    ys = np.array([best_by_outer[int(x)] for x in xs], dtype=float)
    return xs, ys


def group_runs_by_suffix(runs: Sequence[RunInfo]) -> Dict[str, Dict[str, RunInfo]]:
    grouped: Dict[str, Dict[str, RunInfo]] = {}
    for run in runs:
        grouped.setdefault(run.suffix, {})[run.model_key] = run
    return grouped


def plot_train_curves(runs: Sequence[RunInfo], args: argparse.Namespace) -> None:
    grouped = group_runs_by_suffix(runs)
    ensure_dir(train_fig_dir(args))

    curve_models = ['pinn', 'pipinn'] if args.curve_model == 'both' else [args.curve_model]

    for suffix, by_model in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(args.train_fig_width, args.train_fig_height))
        plotted = 0
        for model_key in curve_models:
            run = by_model.get(model_key)
            if run is None:
                continue
            try:
                if model_key == 'pinn':
                    xs, ys = pinn_outer_best_curve(run, train_loss=args.train_loss)
                else:
                    xs, ys = pipinn_outer_curve(run, train_loss=args.train_loss)
            except Exception as e:
                print(f"[warn] train curve skipped for {run.run_name}: {e}")
                continue
            if len(xs) == 0:
                print(f"[warn] empty train curve for {run.run_name}")
                continue
            ax.semilogy(xs, ys, label=run.method_label, linewidth=args.line_width)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=args.grid_alpha)
        if plotted > 1 or args.force_legend:
            ax.legend()
        path = train_fig_dir(args) / f"train_curve_{suffix}_{args.curve_model}_{args.train_loss}.{args.format}"
        save_fig(fig, path, args, kind=f'train curve ({suffix})')


def plot_policy_convergence(runs: Sequence[RunInfo], args: argparse.Namespace) -> None:
    for run in runs:
        if run.model_key != 'pipinn':
            continue

        path = run.run_dir / 'outer_history.csv'
        if not path.exists():
            print(f"[warn] policy convergence skipped; missing {path}")
            continue

        rows = read_csv_rows(path)
        xs, theta_diff, theta_cf_diff = [], [], []

        for row in rows:
            outer = to_int(row.get('outer_iter'))
            th = to_float(row.get('theta_diff'))
            th_cf = to_float(row.get('theta_cf_diff'))  # NaN if column absent

            if outer is None:
                continue
            if math.isnan(th):
                continue

            xs.append(outer)
            theta_diff.append(th)
            theta_cf_diff.append(th_cf)

        if not xs:
            print(f"[warn] policy convergence skipped; empty theta_diff data for {run.run_name}")
            continue

        # Closed-form distance is only plotted if the run actually logged it.
        has_cf = any(not math.isnan(v) for v in theta_cf_diff)

        label_diff = r'$\|\theta_{n+1}/w-\theta_n/w\|_2^2$'
        label_cf   = r'$\|\theta_n/w-\theta^*/w\|_2^2$'
        save_dir = run_plot_dir(run, args)

        def _new_ax():
            fig, ax = plt.subplots(
                1, 1,
                figsize=(args.policy_fig_width, args.policy_fig_height),
                constrained_layout=True
            )
            ax.set_xlabel('Iteration')
            ax.grid(True, alpha=args.grid_alpha)
            return fig, ax

        if args.policy_layout == 'separate' and has_cf:
            # Two independent figures so each curve keeps its own y-scale.
            for name, ys, lbl in (
                ('iterate_diff', theta_diff,    label_diff),
                ('closed_form',  theta_cf_diff, label_cf),
            ):
                fig, ax = _new_ax()
                ax.semilogy(xs, ys, linewidth=args.line_width, label=lbl)
                ax.legend(frameon=False)
                path_out = save_dir / f"paper_policy_convergence_{name}.{args.format}"
                save_fig(fig, path_out, args, kind=f'policy convergence {name} ({run.run_name})')
        else:
            # Combined (default), and the fallback when closed-form data is absent.
            fig, ax = _new_ax()
            ax.semilogy(xs, theta_diff, linewidth=args.line_width, label=label_diff)
            if has_cf:
                ax.semilogy(xs, theta_cf_diff, linewidth=args.line_width, label=label_cf)
            ax.legend(frameon=False)
            path_out = save_dir / f"paper_policy_convergence.{args.format}"
            save_fig(fig, path_out, args, kind=f'policy convergence ({run.run_name})')


# =============================================================================
# Value / portfolio plots from weights
# =============================================================================


def plot_heatmaps_for_run(run: RunInfo, args: argparse.Namespace, device: torch.device) -> None:
    obj = load_run_objects(run, args, device)
    model: nn.Module = obj['model']
    market = obj['market']
    cf = obj['cf']
    M = obj['M']
    N = obj['N']
    gamma = obj['gamma']
    r = obj['r']
    tau_max = float(args.tau_max if args.tau_max is not None else obj['tau_max'])
    tau_min = float(args.tau_min)

    dimX = int(args.dim_x)
    if dimX < 0 or dimX >= M:
        raise ValueError(f"dim-x={dimX} out of range for M={M}")

    x_fixed = market['xbar'].copy()
    w_levels = parse_float_list(args.w_levels)
    save_dir = run_plot_dir(run, args)
    ensure_dir(save_dir)

    for w_test in w_levels:
        print(f"[eval] {run.run_name}: w={w_test:.4g}, dimX={dimX}, grid=({args.n_tau},{args.n_x})")
        tau_grid, X_grid, V_model, theta_model, myopic_model, hedging_model = eval_model_on_tau_X_grid(
            model, market, w_fixed=w_test, dimX=dimX, x_fixed=x_fixed, device=device,
            N_tau=args.n_tau, N_X=args.n_x, tau_min=tau_min, tau_max=tau_max, chunk=args.chunk
        )
        _, _, V_cf, theta_cf, myopic_cf, hedging_cf = eval_closed_form_on_tau_X_grid(
            market, cf, w_fixed=w_test, dimX=dimX, x_fixed=x_fixed,
            N_tau=args.n_tau, N_X=args.n_x, tau_min=tau_min, tau_max=tau_max,
            gamma=gamma, r=r
        )

        w_tag = f"w{w_test:.2f}"
        prefix_base = f"paper_{run.model_key}_{run.suffix}_tauX_{w_tag}"

        if args.plot_value:
            value_path = save_dir / f"{prefix_base}_value.{args.format}"
            plot_value_comparison_tauX(
                tau_grid, X_grid, V_model, V_cf, args, value_path,
                xlabel=args.xlabel, ylabel=args.ylabel
            )

        if args.plot_portfolio:
            portfolio_prefix = f"{prefix_base}_portfolio"
            plot_portfolio_comparison_tauX(
                tau_grid, X_grid,
                theta_model, theta_cf,
                myopic_model, myopic_cf,
                hedging_model, hedging_cf,
                args, save_dir, prefix=portfolio_prefix,
                xlabel=args.xlabel, ylabel=args.ylabel
            )

    # Explicitly release GPU memory between runs.
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# =============================================================================
# Metrics summary
# =============================================================================


def infer_assets_states(run: RunInfo, config: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    assets = states = None
    if config is not None:
        assets = to_int(cfg_get(config, 'n_assets', None))
        states = to_int(cfg_get(config, 'm_states', None))
    if states is None:
        m = re.search(r'm_states(\d+)', run.run_name)
        if m:
            states = int(m.group(1))
    return assets, states


def collect_metrics(runs: Sequence[RunInfo], args: argparse.Namespace) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    target_w = str(args.metric_w).strip().lower()

    for run in runs:
        metrics_path = run.run_dir / 'metrics.csv'
        cfg_path = run.run_dir / 'config.json'
        if not metrics_path.exists():
            print(f"[warn] metrics skipped; missing {metrics_path}")
            continue
        cfg = read_json(cfg_path) if cfg_path.exists() else None
        assets, states = infer_assets_states(run, cfg)
        raw_rows = read_csv_rows(metrics_path)

        # Determine w to use for compact paper tables. Long/wide summaries keep all rows unless numeric filter is requested.
        available_w = sorted({str(r.get('w', '')).strip() for r in raw_rows if str(r.get('w', '')).strip()})
        if target_w == 'auto':
            selected_w = available_w[0] if available_w else ''
        elif target_w == 'all':
            selected_w = None
        else:
            selected_w = str(args.metric_w)

        for row in raw_rows:
            w_val = str(row.get('w', '')).strip()
            if selected_w is not None and w_val != selected_w:
                # Let exact string fail; try numeric equality.
                if not (abs(to_float(w_val) - to_float(selected_w)) < 1e-12):
                    continue
            metric = row.get('metric', '')
            value = to_float(row.get('value'))
            all_rows.append({
                'run_name': run.run_name,
                'suffix': run.suffix,
                'model_key': run.model_key,
                'method': run.method_label,
                'assets': assets if assets is not None else '',
                'states': states if states is not None else '',
                'w': w_val,
                'metric': metric,
                'value': value,
                'value_sci': sci_fmt(value, args.sci_precision),
            })

    return all_rows


def make_metric_wide_rows(long_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    metric_names: List[str] = []

    for row in long_rows:
        key = (row['run_name'], row['suffix'], row['model_key'], row['method'], row['assets'], row['states'], row['w'])
        g = groups.setdefault(key, {
            'run_name': row['run_name'],
            'suffix': row['suffix'],
            'model_key': row['model_key'],
            'method': row['method'],
            'assets': row['assets'],
            'states': row['states'],
            'w': row['w'],
        })

        metric = str(row['metric'])
        if metric and metric not in metric_names:
            metric_names.append(metric)
        g[metric] = row['value']

    def sort_key(r: Dict[str, Any]):
        assets = to_int(r.get('assets'), 10**9)
        states = to_int(r.get('states'), 10**9)
        method_order = 0 if r.get('model_key') == 'pinn' else 1
        return (assets, states, r.get('suffix', ''), method_order)

    def metric_order(metric: str) -> Tuple[int, int, int, str]:
        """
        Column order for selected_metrics_wide.csv.

        1) Main aggregate MSE metrics
        2) Main aggregate relative RMSE metrics
        3) Per-asset MSE_theta_i
        4) Per-asset MSE_myopic_i
        5) Per-asset MSE_hedging_i
        6) Any remaining metrics
        """
        preferred_first = [
            'MSE_V',
            'MSE_theta',
            'MSE_myopic',
            'MSE_hedging',
            'RelRMSE_V',
            'RelRMSE_theta',
            'RelRMSE_myopic',
            'RelRMSE_hedging',
            'RelL2_V',
            'RelL2_theta',
            'RelL2_myopic',
            'RelL2_hedging',
        ]

        if metric in preferred_first:
            return (0, preferred_first.index(metric), -1, metric)

        # Numeric-aware ordering for per-asset MSE columns:
        # MSE_theta_0, MSE_theta_1, ..., MSE_theta_10, ...
        m = re.fullmatch(r'MSE_(theta|myopic|hedging)_(\d+)', metric)
        if m:
            component = m.group(1)
            asset_idx = int(m.group(2))
            component_order = {
                'theta': 0,
                'myopic': 1,
                'hedging': 2,
            }[component]
            return (1, component_order, asset_idx, metric)

        # If future code creates per-asset RelRMSE columns, put them after per-asset MSEs.
        m = re.fullmatch(r'RelRMSE_(theta|myopic|hedging)_(\d+)', metric)
        if m:
            component = m.group(1)
            asset_idx = int(m.group(2))
            component_order = {
                'theta': 0,
                'myopic': 1,
                'hedging': 2,
            }[component]
            return (2, component_order, asset_idx, metric)

        # Everything else goes last.
        return (3, 0, 0, metric)

    rows = sorted(groups.values(), key=sort_key)

    base_fields = ['run_name', 'suffix', 'model_key', 'method', 'assets', 'states', 'w']
    ordered_metric_names = sorted(metric_names, key=metric_order)
    fields = base_fields + ordered_metric_names

    return rows, fields


def make_paper_table_rows(wide_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in wide_rows:
        value_metric = args.paper_value_metric
        portfolio_metric = args.paper_portfolio_metric
        v = row.get(value_metric, '')
        p = row.get(portfolio_metric, '')
        rows.append({
            'Assets': row.get('assets', ''),
            'States': row.get('states', ''),
            'Method': row.get('method', ''),
            'Value': sci_fmt(v, args.sci_precision),
            'Portfolio': sci_fmt(p, args.sci_precision),
            'Value_raw': v,
            'Portfolio_raw': p,
            'Run': row.get('run_name', ''),
            'w': row.get('w', ''),
        })

    def sort_key(r: Dict[str, Any]):
        assets = to_int(r.get('Assets'), 10**9)
        states = to_int(r.get('States'), 10**9)
        method_order = 0 if r.get('Method') == 'PINN' else 1
        return (assets, states, method_order, r.get('Run', ''))

    return sorted(rows, key=sort_key)


def write_latex_table(path: Path, table_rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('% Auto-generated by make_paper_figures.py\n')
        f.write('\\begin{tabular}{ccccc}\n')
        f.write('\\hline\n')
        f.write('Assets & States & Method & Value & Portfolio \\\\\n')
        f.write('\\hline\n')
        for r in table_rows:
            f.write(f"{r['Assets']} & {r['States']} & {r['Method']} & {r['Value']} & {r['Portfolio']} \\\\\n")
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
    print(f"[summary] LaTeX table saved: {path}")


def summarize_metrics(runs: Sequence[RunInfo], args: argparse.Namespace) -> None:
    out_dir = summary_dir(args)
    ensure_dir(out_dir)
    long_rows = collect_metrics(runs, args)
    if not long_rows:
        print('[warn] no metrics rows collected.')
        return

    long_fields = ['run_name', 'suffix', 'model_key', 'method', 'assets', 'states', 'w', 'metric', 'value', 'value_sci']
    long_path = out_dir / 'selected_metrics_long.csv'
    write_csv_rows(long_path, long_rows, long_fields)
    print(f"[summary] metrics long saved: {long_path}")

    wide_rows, wide_fields = make_metric_wide_rows(long_rows)
    wide_path = out_dir / 'selected_metrics_wide.csv'
    write_csv_rows(wide_path, wide_rows, wide_fields)
    print(f"[summary] metrics wide saved: {wide_path}")

    paper_rows = make_paper_table_rows(wide_rows, args)
    paper_fields = ['Assets', 'States', 'Method', 'Value', 'Portfolio', 'Value_raw', 'Portfolio_raw', 'Run', 'w']
    paper_path = out_dir / 'paper_relative_rmse_table.csv'
    write_csv_rows(paper_path, paper_rows, paper_fields)
    print(f"[summary] paper table CSV saved: {paper_path}")

    if args.write_latex:
        write_latex_table(out_dir / 'paper_relative_rmse_table.tex', paper_rows)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Generate paper-style figures and metrics summaries from saved PINN/PI-PINN runs.'
    )

    # Run selection
    p.add_argument('--sweep-root', type=str, required=True, help='Root directory containing pinn/, pi-pinn/, weights/.')
    p.add_argument('--exp-list', type=str, default='baseline',
                   help='Comma-separated experiment suffixes, e.g. baseline,m_states1,m_states3,m_states5.')
    p.add_argument('--heatmap-exp-list', type=str, default='',
                   help='Optional separate suffix list for value/portfolio heatmaps. Defaults to --exp-list.')
    p.add_argument('--models', choices=['pinn', 'pipinn', 'both'], default='both')
    p.add_argument('--pinn-runs', type=str, default='', help='Optional comma-separated full PINN run names.')
    p.add_argument('--pipinn-runs', type=str, default='', help='Optional comma-separated full PI-PINN run names.')
    p.add_argument('--strict', action='store_true', help='Fail if a selected run directory is missing.')

    # Actions
    p.add_argument('--summary-metrics', action='store_true', help='Collect metrics.csv from selected runs.')
    p.add_argument('--plot-train', action='store_true', help='Plot train curves using outer-iteration x-axis.')
    p.add_argument('--plot-policy-convergence', action='store_true', help='Plot PI-PINN eval loss and theta_diff.')
    p.add_argument('--plot-value', action='store_true', help='Plot value-function tau-X heatmaps from saved weights.')
    p.add_argument('--plot-portfolio', action='store_true', help='Plot portfolio tau-X heatmaps from saved weights.')
    p.add_argument('--plot-all', action='store_true', help='Enable all plot types and metrics summary.')

    # Train/convergence curves
    p.add_argument('--curve-model', choices=['pinn', 'pipinn', 'both'], default='both')
    p.add_argument('--train-loss', choices=['total', 'pde', 'terminal', 'concavity', 'monotonicity'], default='total',
                   help='Loss component to plot. Default total uses per-outer best total for PINN and eval_loss for PI-PINN.')
    p.add_argument('--force-legend', action='store_true', help='Show legend even for single-model train curves.')

    # Heatmap settings
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--weight-name', type=str, default='', help='Specific weight filename; default value_net_best.pt.')
    p.add_argument('--w-levels', type=str, default='0.5')
    p.add_argument('--dim-x', type=int, default=0)
    p.add_argument('--tau-min', type=float, default=0.0)
    p.add_argument('--tau-max', type=float, default=None, help='Override tau_max; default reads from market_params.npz.')
    p.add_argument('--n-tau', type=int, default=100)
    p.add_argument('--n-x', type=int, default=100)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--max-assets', type=int, default=1)
    p.add_argument('--assets', type=str, default='', help='Optional explicit asset indices, e.g. 0,3,7.')
    p.add_argument('--sort-by-range', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--portfolio-components', choices=['all', 'hedge'], default='all')
    p.add_argument(
        '--portfolio-layout',
        choices=['grid', 'separate'],
        default='grid',
        help='grid: save 3x3/1x3 portfolio figures; separate: save each portfolio panel as an individual figure.'
    )
    p.add_argument('--single-panel-fig-width', type=float, default=4.0)
    p.add_argument('--single-panel-fig-height', type=float, default=3.0)
    p.add_argument('--xlabel', type=str, default='X')
    p.add_argument('--ylabel', type=str, default=r'$\tau$')
    p.add_argument('--value-hidden-default', type=int, default=256)
    p.add_argument('--value-depth-default', type=int, default=3)

    # Figure style and output
    p.add_argument('--fig-root', type=str, default='',
                   help='Optional common output root. If omitted, heatmaps go to each run_dir/plots and shared outputs to sweep_root/paper_figures/.')
    p.add_argument('--format', choices=['png', 'pdf', 'eps', 'svg'], default='png')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--font-size', type=float, default=14)
    p.add_argument('--label-size', type=float, default=14)
    p.add_argument('--tick-size', type=float, default=12)
    p.add_argument('--legend-size', type=float, default=12)
    p.add_argument('--cbar-tick-size', type=float, default=11)
    p.add_argument('--line-width', type=float, default=1.8)
    p.add_argument('--grid-alpha', type=float, default=0.3)
    p.add_argument('--font-family', type=str, default='')
    p.add_argument('--train-fig-width', type=float, default=6.5)
    p.add_argument('--train-fig-height', type=float, default=4.2)
    p.add_argument('--policy-fig-width', type=float, default=6.5)
    p.add_argument('--policy-fig-height', type=float, default=4.2)
    p.add_argument(
        '--policy-layout',
        choices=['combined', 'separate'],
        default='combined',
        help='combined: both curves on one axis; separate: save iterate-diff and '
             'closed-form-distance as individual figures (each auto-scaled).'
    )
    p.add_argument('--value-fig-width', type=float, default=12.0)
    p.add_argument('--value-fig-height', type=float, default=3.8)
    p.add_argument(
        '--value-layout',
        choices=['grid', 'separate'],
        default='grid',
        help='grid: save the original 1x3 value figure; separate: save each value panel as an individual figure.'
    )
    p.add_argument('--value-panel-fig-width', type=float, default=4.0)
    p.add_argument('--value-panel-fig-height', type=float, default=3.0)
    
    p.add_argument('--portfolio-fig-width', type=float, default=12.0)
    p.add_argument('--portfolio-fig-height', type=float, default=9.0)
    p.add_argument('--single-row-fig-height', type=float, default=3.8)

    # Metrics summary
    p.add_argument('--metric-w', type=str, default='auto',
                   help='Which w to summarize from metrics.csv: auto, all, or a numeric value such as 0.5.')
    p.add_argument('--paper-value-metric', type=str, default='RelRMSE_V')
    p.add_argument('--paper-portfolio-metric', type=str, default='RelRMSE_theta')
    p.add_argument('--sci-precision', type=int, default=2)
    p.add_argument('--write-latex', action='store_true', help='Also write a compact LaTeX table.')

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.plot_all:
        args.summary_metrics = True
        args.plot_train = True
        args.plot_policy_convergence = True
        args.plot_value = True
        args.plot_portfolio = True

    # If the user runs the script with no action flags, do the cheap/default outputs.
    if not any([args.summary_metrics, args.plot_train, args.plot_policy_convergence, args.plot_value, args.plot_portfolio]):
        print('[info] no action flags given; defaulting to --summary-metrics --plot-train --plot-policy-convergence')
        args.summary_metrics = True
        args.plot_train = True
        args.plot_policy_convergence = True

    apply_plot_style(args)

    # General selected runs for train curves / policy convergence / summaries.
    runs = resolve_runs_from_args(args, for_heatmaps=False)
    if not runs:
        print('[error] no selected runs found.', file=sys.stderr)
        return 2

    print('[runs] selected:')
    for r in runs:
        print(f"  - {r.model_key:6s} {r.run_name} -> {r.run_dir}")

    if args.summary_metrics:
        summarize_metrics(runs, args)

    if args.plot_train:
        plot_train_curves(runs, args)

    if args.plot_policy_convergence:
        plot_policy_convergence(runs, args)

    if args.plot_value or args.plot_portfolio:
        heatmap_runs = resolve_runs_from_args(args, for_heatmaps=True)
        if not heatmap_runs:
            print('[warn] no heatmap runs found; skipping value/portfolio plots.')
        else:
            device = resolve_device(args.device)
            print(f"[device] heatmap evaluation device: {device}")
            for run in heatmap_runs:
                try:
                    plot_heatmaps_for_run(run, args, device)
                except Exception as e:
                    if args.strict:
                        raise
                    print(f"[warn] heatmap skipped for {run.run_name}: {e}")

    print('[done] make_paper_figures.py finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
