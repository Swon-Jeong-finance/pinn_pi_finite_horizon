#!/usr/bin/env python3
"""Post-process the Liu non-affine homotopy experiment.

For every training seed and perturbation strength epsilon, this script loads
the official saved value model and pairs it with the same-seed epsilon=0 run.
At the manuscript slice ``w=w0, x=xbar`` it computes

    Delta V_epsilon(tau) = V_epsilon(tau,w0,xbar) - V_0(tau,w0,xbar)

and

    Delta theta_epsilon(tau)
      = ||theta_epsilon(tau,w0,xbar)/w0
           - theta_0(tau,w0,xbar)/w0||_2.

Seed-wise curves are formed first and then summarized pointwise by mean and
sample standard deviation.  The final held-out residual components in
``outer_history.csv`` are summarized separately.  This is a read-only
postprocessor: it never modifies a training run.

Typical use::

    python3 postprocess_nonaffine.py \
      --out-root outputs/nonaffine \
      --n-assets 30 --m-states 1 \
      --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
      --eps 0,0.1,1,2,3,4,5

Run directories are discovered recursively from ``config.json``.  The
preferred launcher layout is

    OUT_ROOT/pi-pinn/<run-tag>/
    OUT_ROOT/weights/pi-pinn/<run-tag>/

but the recorded ``weight_dir`` is honored, so copied or custom layouts also
work.  Paper mode requires ``value_net_final.pt``.  The legacy
``final -> last -> best`` search is available only through the explicit
``--allow-checkpoint-fallback`` exploratory option.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

    class _UnavailableModule:
        pass

    class _UnavailableNN:
        Module = _UnavailableModule

    nn = _UnavailableNN()  # type: ignore[assignment]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from aggregate_seeds import GROUP_IGNORE_KEYS, canonical_market_hash, parse_seed_spec, run_status
from joint_market_setup_dirichlet import validate_market_snapshot
if torch is not None:
    from experiment_utils import VWW_GUARD, safe_concave_vww
else:
    VWW_GUARD = 1.0e-8

    def safe_concave_vww(_value):  # pragma: no cover - guarded in main
        raise RuntimeError("safe_concave_vww requires PyTorch")
from liu_risk_premium import risk_premium_torch


EPS_TOL = 1.0e-10
RESIDUAL_METRICS = ("val_pde_rms", "val_terminal_rms", "val_pres")
NONAFFINE_MODES = {"tanh", "nonaffine", "non-affine", "nonlinear"}
AFFINE_MODES = {"affine", "linear"}

# Epsilon and run-location values distinguish launcher bookkeeping, not the
# common training configuration that must be paired across epsilon and seed.
PAIR_IGNORE_KEYS = set(GROUP_IGNORE_KEYS) | {
    "nonaffine_eps",
    "non_affine_eps",
    "epsilon",
    "eps",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}={value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _optional_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_int(value: Any, *, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}={value!r}") from exc
    return out


def _parse_csv_floats(text: str) -> List[float]:
    values = [
        _finite_float(part, name="epsilon")
        for part in re.split(r"[\s,]+", str(text).strip())
        if part
    ]
    if len({_eps_key(value) for value in values}) != len(values):
        raise ValueError(f"duplicate epsilon values: {text!r}")
    return values


def _parse_csv_ints(text: str) -> set[int]:
    if not str(text or "").strip():
        return set()
    values = parse_seed_spec(str(text))
    return set(values)


def _eps_key(value: float) -> str:
    value = float(value)
    value = 0.0 if value == 0.0 else value  # normalize negative zero only
    return f"{value:.17g}"


def _is_zero_eps(value: float) -> bool:
    # This must match training's exact has_affine_reference rule.  A tiny
    # positive epsilon is still a non-affine model, never a baseline alias.
    return float(value) == 0.0


def _cfg_args(config: Mapping[str, Any]) -> Dict[str, Any]:
    args = config.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("config.json field 'args' must be an object")
    return dict(args)


def _cfg_get(config: Mapping[str, Any], args: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in args:
        return args[key]
    if key in config:
        return config[key]
    return default


def _extract_mode_eps(config: Mapping[str, Any], args: Mapping[str, Any]) -> Tuple[str, float]:
    mode = str(_cfg_get(config, args, "risk_premium_mode", "")).strip().lower().replace("_", "-")
    aliases = ("nonaffine_eps", "non_affine_eps", "epsilon", "eps")
    found: List[Tuple[str, float]] = []
    for key in aliases:
        value = _cfg_get(config, args, key, None)
        if value is not None and str(value).strip() != "":
            found.append((key, _finite_float(value, name=key)))

    if found:
        epsilon = found[0][1]
        for key, value in found[1:]:
            if value != epsilon:
                raise ValueError(f"conflicting epsilon aliases: {found[0]} vs {(key, value)}")
    else:
        raise KeyError("not a non-affine run (no epsilon field)")

    if epsilon < 0.0:
        raise ValueError(f"nonaffine epsilon must be nonnegative, got {epsilon}")
    if not mode:
        mode = "legacy-nonaffine"
    # The current parser records nonaffine_eps=0 even for every ordinary
    # affine main run.  Figure 4 deliberately uses mode=tanh, eps=0 as its
    # separately launched baseline, so ignore mode=affine here instead of
    # letting a main-sweep checkpoint silently replace that paired baseline.
    if mode in AFFINE_MODES:
        if not _is_zero_eps(epsilon):
            raise ValueError(f"risk_premium_mode={mode!r} is incompatible with epsilon={epsilon:g}")
        raise KeyError("ordinary affine run, not the tanh eps=0 baseline")
    if mode not in AFFINE_MODES and mode not in NONAFFINE_MODES and mode != "legacy-nonaffine":
        raise ValueError(f"unsupported risk_premium_mode={mode!r}")
    return mode, (0.0 if _is_zero_eps(epsilon) else float(epsilon))


def _canonical_signature(args: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    canonical = {key: value for key, value in args.items() if key not in PAIR_IGNORE_KEYS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return digest, canonical


def _updated_at(run_dir: Path, config: Mapping[str, Any]) -> Tuple[str, int]:
    try:
        status = _read_json(run_dir / "status.json")
        stamp = str(status.get("updated_at", ""))
    except Exception:
        stamp = ""
    if not stamp:
        stamp = str(config.get("created_at", ""))
    try:
        mtime = int((run_dir / "config.json").stat().st_mtime_ns)
    except OSError:
        mtime = 0
    return stamp, mtime


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    config: Dict[str, Any]
    args: Dict[str, Any]
    model_type: str
    n_assets: int
    m_states: int
    seed: int
    epsilon: float
    risk_premium_mode: str
    group_id: str
    group_config: Dict[str, Any]
    status: str
    updated_at: Tuple[str, int]


@dataclass
class LoadedRun:
    record: RunRecord
    checkpoint: Path
    checkpoint_kind: str
    market_hash: str
    tau_max: float
    w_min: float
    w_max: float
    values: np.ndarray
    theta_norm: np.ndarray
    guard_fraction_tau: float


def discover_runs(
    out_root: Path,
    *,
    model_type: str,
    n_assets: Optional[int],
    m_states: set[int],
    run_name_regex: str,
) -> Dict[str, Dict[Tuple[int, str], RunRecord]]:
    """Discover and newest-deduplicate runs by (group, seed, epsilon)."""
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int, str], RunRecord] = {}
    errors: List[str] = []

    for config_path in sorted(out_root.rglob("config.json")):
        run_dir = config_path.parent
        if pattern and not pattern.search(run_dir.name):
            continue
        try:
            config = _read_json(config_path)
            cfg_args = _cfg_args(config)
            mode, epsilon = _extract_mode_eps(config, cfg_args)
            current_model = str(_cfg_get(config, cfg_args, "model_type", "")).strip().lower().replace("-", "")
            requested_model = str(model_type).strip().lower().replace("-", "")
            if current_model != requested_model:
                continue
            n = _as_int(_cfg_get(config, cfg_args, "n_assets"), name="n_assets")
            m = _as_int(_cfg_get(config, cfg_args, "m_states"), name="m_states")
            seed = _as_int(_cfg_get(config, cfg_args, "seed"), name="seed")
            if n_assets is not None and n != n_assets:
                continue
            if m_states and m not in m_states:
                continue
            clip_raw = _cfg_get(config, cfg_args, "theta_clip_abs", None)
            if clip_raw is not None and str(clip_raw).strip().lower() not in {"", "none", "null"}:
                raise ValueError(
                    "Figure 4 uses the unconstrained greedy map, but this run has "
                    f"theta_clip_abs={clip_raw!r}"
                )
            group_id, group_config = _canonical_signature(cfg_args)
            record = RunRecord(
                run_dir=run_dir.resolve(), config=config, args=cfg_args,
                model_type=current_model, n_assets=n, m_states=m, seed=seed,
                epsilon=epsilon, risk_premium_mode=mode, group_id=group_id,
                group_config=group_config, status=run_status(str(run_dir)),
                updated_at=_updated_at(run_dir, config),
            )
            key = (group_id, seed, _eps_key(epsilon))
            previous = newest.get(key)
            if previous is None or record.updated_at >= previous.updated_at:
                newest[key] = record
        except KeyError:
            # Normal affine/main runs can share a broad output root; they are
            # intentionally ignored unless an epsilon field identifies them.
            continue
        except Exception as exc:
            errors.append(f"{config_path}: {exc}")

    if errors:
        message = "\n".join(f"  - {item}" for item in errors[:20])
        suffix = "\n  - ..." if len(errors) > 20 else ""
        raise ValueError(f"invalid candidate non-affine configs:\n{message}{suffix}")

    groups: Dict[str, Dict[Tuple[int, str], RunRecord]] = {}
    for (group_id, seed, epsilon), record in newest.items():
        groups.setdefault(group_id, {})[(seed, epsilon)] = record
    return groups


def _weight_dir_candidates(record: RunRecord, out_root: Path) -> List[Path]:
    raw_values = [record.config.get("weight_dir"), record.args.get("weight_root")]
    candidates: List[Path] = []
    for raw in raw_values:
        if raw is None or not str(raw).strip():
            continue
        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            cwd = record.config.get("cwd")
            if cwd:
                candidates.append(Path(str(cwd)).expanduser() / path)
            candidates.extend((record.run_dir / path, out_root / path, Path.cwd() / path))

    # Standard launcher layout inferred from the run directory itself.
    for parent in (record.run_dir, *record.run_dir.parents):
        if parent.name in {"pi-pinn", "pinn"}:
            candidates.append(parent.parent / "weights" / parent.name / record.run_dir.name)
            break
    candidates.append(out_root / "weights" / "pi-pinn" / record.run_dir.name)

    seen: set[str] = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return [path.resolve() if path.exists() else path for path in unique]


def _resolve_checkpoint(
    record: RunRecord,
    out_root: Path,
    *,
    allow_fallback: bool = False,
) -> Tuple[Path, str]:
    weight_dirs = _weight_dir_candidates(record, out_root)
    # Filename priority is global across all candidate directories: a copied
    # official final must outrank a legacy best file left at the recorded
    # training-machine path.
    names = [("value_net_final.pt", "final")]
    if allow_fallback:
        names.extend((
            ("value_net_last.pt", "last_fallback"),
            ("value_net_best.pt", "best_legacy_fallback"),
        ))
    for name, kind in names:
        for weight_dir in weight_dirs:
            path = weight_dir / name
            if path.is_file():
                return path, kind
    attempted = "\n".join(f"  - {path}" for path in weight_dirs)
    policy = (
        "no official value_net_final.pt"
        if not allow_fallback
        else "no value_net_final.pt/value_net_last.pt/value_net_best.pt"
    )
    suffix = " (pass --allow-checkpoint-fallback only for legacy exploratory runs)" if not allow_fallback else ""
    raise FileNotFoundError(f"{policy}{suffix} in candidate weight directories:\n{attempted}")


class ValueNetND(nn.Module):
    """Architecture shared by the current Liu training scripts."""

    def __init__(self, m_states: int, hidden: int, depth: int):
        super().__init__()
        in_dim = int(m_states) + 2
        layers: List[nn.Module] = []
        for _ in range(int(depth)):
            layers.extend((nn.Linear(in_dim, int(hidden)), nn.Tanh()))
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, w: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((w, x, tau), dim=1))


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        state: Any = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch<2.0 compatibility
        state = torch.load(path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint does not contain a state dict: {path}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module."):]: value for key, value in state.items()}
    return state


def _np_scalar(market: np.lib.npyio.NpzFile, key: str, fallback: Any) -> float:
    if key not in market.files:
        return _finite_float(fallback, name=key)
    array = np.asarray(market[key]).reshape(-1)
    if not array.size:
        raise ValueError(f"empty market field: {key}")
    return _finite_float(array[0], name=key)


def _evaluate_run(
    record: RunRecord,
    *,
    out_root: Path,
    tau_grid: np.ndarray,
    w0: float,
    device: torch.device,
    allow_checkpoint_fallback: bool = False,
) -> LoadedRun:
    market_path = record.run_dir / "market_params.npz"
    if not market_path.is_file():
        raise FileNotFoundError(f"missing market_params.npz: {market_path}")
    checkpoint, checkpoint_kind = _resolve_checkpoint(
        record, out_root, allow_fallback=allow_checkpoint_fallback
    )

    with np.load(market_path, allow_pickle=False) as market:
        required = ("xbar", "lam0", "Lam", "Gamma", "eta")
        missing = [key for key in required if key not in market.files]
        if missing:
            raise ValueError(f"{market_path}: missing fields {missing}")
        try:
            validate_market_snapshot(market)
        except ValueError as exc:
            raise ValueError(f"{market_path}: invalid market snapshot: {exc}") from exc
        xbar_np = np.asarray(market["xbar"], dtype=np.float32).reshape(-1)
        lam0_np = np.asarray(market["lam0"], dtype=np.float32).reshape(-1)
        loading_np = np.asarray(market["Lam"], dtype=np.float32)
        gamma_np = np.asarray(market["Gamma"], dtype=np.float32)
        eta_np = np.asarray(market["eta"], dtype=np.float32).reshape(-1)
        tau_max = _np_scalar(market, "tau_max", record.args.get("tau_max", 3.0))
        w_min = _np_scalar(market, "W_min", record.args.get("w_min", 0.1))
        w_max = _np_scalar(market, "W_max", record.args.get("w_max", 2.0))

    if xbar_np.shape != (record.m_states,):
        raise ValueError(f"{market_path}: xbar shape={xbar_np.shape}, expected {(record.m_states,)}")
    if lam0_np.shape != (record.n_assets,):
        raise ValueError(f"{market_path}: lam0 shape={lam0_np.shape}, expected {(record.n_assets,)}")
    expected_matrix = (record.n_assets, record.m_states)
    for name, array in (("Lam", loading_np), ("Gamma", gamma_np)):
        if array.shape != expected_matrix:
            raise ValueError(f"{market_path}: {name} shape={array.shape}, expected {expected_matrix}")
    if eta_np.shape != (record.m_states,):
        raise ValueError(f"{market_path}: eta shape={eta_np.shape}, expected {(record.m_states,)}")
    if not np.all(np.isfinite(eta_np)) or np.any(eta_np <= 0.0):
        raise ValueError(f"{market_path}: every eta entry must be finite and strictly positive")
    if not (w_min <= w0 <= w_max):
        raise ValueError(f"w0={w0:g} is outside [{w_min:g},{w_max:g}] for {record.run_dir}")
    if float(tau_grid[0]) < -EPS_TOL or float(tau_grid[-1]) > tau_max + EPS_TOL:
        raise ValueError(
            f"tau grid [{tau_grid[0]:g},{tau_grid[-1]:g}] exceeds [0,{tau_max:g}] for {record.run_dir}"
        )

    hidden = _as_int(record.args.get("value_hidden", 256), name="value_hidden")
    depth = _as_int(record.args.get("value_depth", 4), name="value_depth")
    model = ValueNetND(record.m_states, hidden, depth).to(device)
    model.load_state_dict(_load_state_dict(checkpoint, device), strict=True)
    model.eval()

    tau = torch.as_tensor(tau_grid, dtype=torch.float32, device=device).reshape(-1, 1)
    w = torch.full((len(tau_grid), 1), float(w0), dtype=torch.float32, device=device, requires_grad=True)
    x = torch.as_tensor(xbar_np, dtype=torch.float32, device=device).reshape(1, -1).repeat(len(tau_grid), 1)
    x.requires_grad_(True)

    value = model(w, x, tau)
    value_w = torch.autograd.grad(value, w, torch.ones_like(value), create_graph=True, retain_graph=True)[0]
    value_wx = torch.autograd.grad(
        value_w, x, torch.ones_like(value_w), create_graph=False, retain_graph=True
    )[0]
    value_ww = torch.autograd.grad(
        value_w, w, torch.ones_like(value_w), create_graph=False, retain_graph=False
    )[0]

    lam0_t = torch.as_tensor(lam0_np, dtype=torch.float32, device=device)
    loading_t = torch.as_tensor(loading_np, dtype=torch.float32, device=device)
    gamma_t = torch.as_tensor(gamma_np, dtype=torch.float32, device=device)
    xbar_t = torch.as_tensor(xbar_np, dtype=torch.float32, device=device)
    eta_t = torch.as_tensor(eta_np, dtype=torch.float32, device=device)
    nonlinear_scale = _finite_float(record.args.get("nonaffine_loading_scale", 1.0), name="nonaffine_loading_scale")
    helper_mode = "tanh" if record.risk_premium_mode in NONAFFINE_MODES | {"legacy-nonaffine"} else "affine"
    lam_x = risk_premium_torch(
        x, lam0_t, loading_t,
        mode=helper_mode, eps=record.epsilon,
        xbar=xbar_t, state_scale=eta_t, loading_scale=nonlinear_scale,
    )
    gamma_vwx = torch.einsum("ij,bj->bi", gamma_t, value_wx)
    value_ww_safe = safe_concave_vww(value_ww)
    theta = -(lam_x * value_w + gamma_vwx) / value_ww_safe
    theta_norm = theta / w
    guard_fraction = float(torch.mean((value_ww > -float(VWW_GUARD)).float()).item())

    return LoadedRun(
        record=record,
        checkpoint=checkpoint,
        checkpoint_kind=checkpoint_kind,
        market_hash=canonical_market_hash(str(market_path)),
        tau_max=tau_max,
        w_min=w_min,
        w_max=w_max,
        values=value.detach().cpu().numpy().reshape(-1).astype(float),
        theta_norm=theta_norm.detach().cpu().numpy().astype(float),
        guard_fraction_tau=guard_fraction,
    )


def _final_residuals(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "outer_history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing outer_history.csv: {path}")
    selected: Optional[Dict[str, str]] = None
    selected_outer = -math.inf
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            outer = _optional_float(row.get("outer_iter"))
            if not math.isfinite(outer):
                continue
            if outer >= selected_outer:
                selected_outer = outer
                selected = row
    if selected is None:
        raise ValueError(f"no valid outer rows in {path}")
    return {
        "outer_iter": int(selected_outer),
        **{metric: _optional_float(selected.get(metric)) for metric in RESIDUAL_METRICS},
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else float("nan")
    return mean, std


def _match_requested_eps(available: Iterable[float], requested: Sequence[float]) -> List[float]:
    available_values = sorted(set(float(value) for value in available))
    if not requested:
        return available_values
    matched: List[float] = []
    for target in requested:
        candidates = [value for value in available_values if value == target]
        if len(candidates) != 1:
            raise ValueError(f"requested epsilon={target:g} is not uniquely available; found {candidates}")
        matched.append(candidates[0])
    if not any(_is_zero_eps(value) for value in matched):
        baseline = [value for value in available_values if _is_zero_eps(value)]
        if len(baseline) != 1:
            raise ValueError("a unique epsilon=0 baseline is required")
        matched.insert(0, baseline[0])
    return sorted(set(matched))


def _create_curve_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
):
    """Build the paper-facing two-panel non-affine figure.

    Value and policy panels reuse exactly the same epsilon colors.  A single
    figure-level legend is placed to the right of both axes, matching the
    manuscript layout without duplicating epsilon entries in each panel.
    """
    eps_values = sorted({float(row["epsilon"]) for row in summary_rows if not _is_zero_eps(float(row["epsilon"]))})
    if not eps_values:
        raise ValueError("at least one epsilon>0 curve is required")
    cmap = plt.get_cmap(args.cmap)
    denom = max(max(eps_values) - min(eps_values), EPS_TOL)
    with plt.rc_context({"font.size": float(args.font_size)}):
        fig, axes = plt.subplots(
            1, 2, figsize=(float(args.fig_width), float(args.fig_height))
        )
        for epsilon in eps_values:
            rows = sorted(
                (row for row in summary_rows if _eps_key(row["epsilon"]) == _eps_key(epsilon)),
                key=lambda row: float(row["tau"]),
            )
            tau = np.asarray([float(row["tau"]) for row in rows])
            delta_v = np.asarray([float(row["delta_V_mean"]) for row in rows])
            delta_v_std = np.asarray([float(row["delta_V_std"]) for row in rows])
            delta_theta = np.asarray([float(row["delta_theta_l2_mean"]) for row in rows])
            delta_theta_std = np.asarray([float(row["delta_theta_l2_std"]) for row in rows])
            color = cmap((epsilon - min(eps_values)) / denom if len(eps_values) > 1 else 0.55)
            label = rf"$\varepsilon={epsilon:.2f}$"
            axes[0].plot(tau, delta_v, label=label, color=color, linewidth=1.5)
            axes[0].fill_between(
                tau, delta_v - delta_v_std, delta_v + delta_v_std,
                color=color, alpha=0.16, linewidth=0,
            )
            axes[1].plot(tau, delta_theta, label=label, color=color, linewidth=1.5)
            axes[1].fill_between(
                tau, np.maximum(delta_theta - delta_theta_std, 0.0),
                delta_theta + delta_theta_std,
                color=color, alpha=0.16, linewidth=0,
            )

        axes[0].axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Time to horizon")
        axes[1].set_xlabel("Time to horizon")
        for axis in axes:
            axis.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=int(args.x_max_ticks),
                    min_n_ticks=3,
                    steps=(1, 2, 2.5, 5, 10),
                )
            )
            axis.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=int(args.y_max_ticks),
                    min_n_ticks=3,
                    steps=(1, 2, 2.5, 5, 10),
                )
            )
            axis.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            frameon=False,
        )
        # Reserve the rightmost part of the canvas for the one shared legend.
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return fig


def _create_separate_curve_figures(
    summary_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
):
    """Build standalone Value and Policy figures from the same curve data.

    The Value file intentionally has no legend.  The Policy file carries the
    one shared epsilon legend on the right so the two exported files can be
    placed side by side without repeating legend entries.
    """
    eps_values = sorted({
        float(row["epsilon"])
        for row in summary_rows
        if not _is_zero_eps(float(row["epsilon"]))
    })
    if not eps_values:
        raise ValueError("at least one epsilon>0 curve is required")
    cmap = plt.get_cmap(args.cmap)
    denom = max(max(eps_values) - min(eps_values), EPS_TOL)
    shared_height = float(getattr(args, "single_fig_height", 4.0))
    value_height_arg = getattr(args, "value_fig_height", None)
    policy_height_arg = getattr(args, "policy_fig_height", None)
    value_height = (
        shared_height if value_height_arg is None else float(value_height_arg)
    )
    policy_height = (
        shared_height if policy_height_arg is None else float(policy_height_arg)
    )

    with plt.rc_context({"font.size": float(args.font_size)}):
        value_fig, value_axis = plt.subplots(
            figsize=(float(args.value_fig_width), value_height)
        )
        policy_fig, policy_axis = plt.subplots(
            figsize=(float(args.policy_fig_width), policy_height)
        )

        for epsilon in eps_values:
            rows = sorted(
                (
                    row
                    for row in summary_rows
                    if _eps_key(row["epsilon"]) == _eps_key(epsilon)
                ),
                key=lambda row: float(row["tau"]),
            )
            tau = np.asarray([float(row["tau"]) for row in rows])
            delta_v = np.asarray([float(row["delta_V_mean"]) for row in rows])
            delta_v_std = np.asarray([float(row["delta_V_std"]) for row in rows])
            delta_theta = np.asarray([
                float(row["delta_theta_l2_mean"]) for row in rows
            ])
            delta_theta_std = np.asarray([
                float(row["delta_theta_l2_std"]) for row in rows
            ])
            color = cmap(
                (epsilon - min(eps_values)) / denom
                if len(eps_values) > 1 else 0.55
            )
            label = rf"$\varepsilon={epsilon:.2f}$"

            value_axis.plot(
                tau, delta_v, label=label, color=color, linewidth=1.5
            )
            value_axis.fill_between(
                tau,
                delta_v - delta_v_std,
                delta_v + delta_v_std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            policy_axis.plot(
                tau, delta_theta, label=label, color=color, linewidth=1.5
            )
            policy_axis.fill_between(
                tau,
                np.maximum(delta_theta - delta_theta_std, 0.0),
                delta_theta + delta_theta_std,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        value_axis.axhline(
            0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5
        )
        for axis in (value_axis, policy_axis):
            axis.set_xlabel("Time to horizon")
            axis.xaxis.set_major_locator(
                MaxNLocator(
                    nbins=int(args.x_max_ticks),
                    min_n_ticks=3,
                    steps=(1, 2, 2.5, 5, 10),
                )
            )
            axis.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=int(args.y_max_ticks),
                    min_n_ticks=3,
                    steps=(1, 2, 2.5, 5, 10),
                )
            )
            axis.grid(True, alpha=0.3)

        handles, labels = policy_axis.get_legend_handles_labels()
        policy_fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.55, 0.6),
            frameon=False,
        )
        value_fig.tight_layout()
        policy_fig.tight_layout(rect=(0.0, 0.0, 0.61, 1.0))

    return value_fig, policy_fig


def _plot_curves(summary_rows: Sequence[Mapping[str, Any]], output_dir: Path, args: argparse.Namespace) -> List[Path]:
    combined_fig = _create_curve_figure(summary_rows, args)
    value_fig, policy_fig = _create_separate_curve_figures(summary_rows, args)

    paths: List[Path] = []
    try:
        for fmt in [
            part.strip().lower().lstrip(".")
            for part in args.formats.split(",")
            if part.strip()
        ]:
            for filename, fig in (
                (f"nonaffine_figure4.{fmt}", combined_fig),
                (f"V_diff_from_base.{fmt}", value_fig),
                (f"pi_diff_from_base.{fmt}", policy_fig),
            ):
                path = output_dir / filename
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                paths.append(path)
    finally:
        plt.close(combined_fig)
        plt.close(value_fig)
        plt.close(policy_fig)
    return paths


def _select_group_records(
    records: Mapping[Tuple[int, str], RunRecord],
    *,
    requested_eps: Sequence[float],
    expected_seeds: set[int],
    allow_incomplete: bool,
    min_seeds: int,
) -> Tuple[List[float], Dict[float, List[int]], Dict[Tuple[int, str], RunRecord]]:
    successful = {key: record for key, record in records.items() if record.status == "success"}
    # Include failed/stopped candidates in the default epsilon set.  Otherwise
    # an epsilon for which every job failed would disappear silently from the
    # paper figure instead of producing a missing-seed error.
    available_eps = sorted({record.epsilon for record in records.values()})
    eps_values = _match_requested_eps(available_eps, requested_eps)
    baseline = next((value for value in eps_values if _is_zero_eps(value)), None)
    if baseline is None:
        raise ValueError("epsilon=0 baseline is required")
    if not any(not _is_zero_eps(value) for value in eps_values):
        raise ValueError("at least one successful epsilon>0 run is required")

    all_latest = {(record.seed, _eps_key(record.epsilon)): record for record in records.values()}
    success_by_eps = {
        epsilon: {record.seed for record in successful.values() if _eps_key(record.epsilon) == _eps_key(epsilon)}
        for epsilon in eps_values
    }
    baseline_seeds = success_by_eps[baseline]
    target_by_eps: Dict[float, List[int]] = {}
    for epsilon in eps_values:
        available = success_by_eps[epsilon]
        if expected_seeds:
            missing = expected_seeds - available
            extra = available - expected_seeds
            if (missing or extra) and not allow_incomplete:
                details = []
                for seed in sorted(missing):
                    latest = all_latest.get((seed, _eps_key(epsilon)))
                    details.append(f"seed {seed}: {latest.status if latest else 'missing'}")
                raise ValueError(
                    f"epsilon={epsilon:g} seed set mismatch; missing/status={details}, extra={sorted(extra)}"
                )
            seeds = sorted(expected_seeds & available & baseline_seeds)
        elif allow_incomplete:
            seeds = sorted(available & baseline_seeds)
        else:
            if available != baseline_seeds:
                raise ValueError(
                    f"epsilon={epsilon:g} has seeds={sorted(available)}, but epsilon=0 has "
                    f"seeds={sorted(baseline_seeds)}; pass --allow-incomplete only for a pilot"
                )
            seeds = sorted(available)
        if not _is_zero_eps(epsilon) and len(seeds) < min_seeds:
            raise ValueError(
                f"epsilon={epsilon:g} has only {len(seeds)} paired seeds; --min-seeds={min_seeds}"
            )
        target_by_eps[epsilon] = seeds

    selected: Dict[Tuple[int, str], RunRecord] = {}
    needed_baseline_seeds: set[int] = set()
    for epsilon in eps_values:
        if _is_zero_eps(epsilon):
            continue
        for seed in target_by_eps[epsilon]:
            needed_baseline_seeds.add(seed)
            selected[(seed, _eps_key(epsilon))] = successful[(seed, _eps_key(epsilon))]
    for seed in sorted(needed_baseline_seeds):
        selected[(seed, _eps_key(baseline))] = successful[(seed, _eps_key(baseline))]
    return eps_values, target_by_eps, selected


def process_group(
    group_id: str,
    records: Mapping[Tuple[int, str], RunRecord],
    *,
    out_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    sample = next(iter(records.values()))
    requested_eps = _parse_csv_floats(args.eps) if args.eps else []
    expected_seeds = _parse_csv_ints(args.expected_seeds)
    eps_values, target_by_eps, selected = _select_group_records(
        records, requested_eps=requested_eps, expected_seeds=expected_seeds,
        allow_incomplete=args.allow_incomplete, min_seeds=args.min_seeds,
    )
    baseline_eps = next(value for value in eps_values if _is_zero_eps(value))

    tau_max_values: List[float] = []
    for record in selected.values():
        with np.load(record.run_dir / "market_params.npz", allow_pickle=False) as market:
            tau_max_values.append(_np_scalar(market, "tau_max", record.args.get("tau_max", 3.0)))
    if max(tau_max_values) - min(tau_max_values) > EPS_TOL:
        raise ValueError(f"group={group_id}: inconsistent tau_max values {sorted(set(tau_max_values))}")
    common_tau_max = tau_max_values[0]
    tau_min = _finite_float(args.tau_min, name="tau_min")
    tau_max = common_tau_max if args.tau_max is None else _finite_float(args.tau_max, name="tau_max")
    if not (0.0 <= tau_min < tau_max <= common_tau_max + EPS_TOL):
        raise ValueError(f"require 0 <= tau_min < tau_max <= {common_tau_max:g}")
    tau_grid = np.linspace(tau_min, tau_max, args.n_tau, dtype=float)

    loaded: Dict[Tuple[int, str], LoadedRun] = {}
    for key, record in sorted(selected.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        loaded[key] = _evaluate_run(
            record,
            out_root=out_root,
            tau_grid=tau_grid,
            w0=args.w0,
            device=device,
            allow_checkpoint_fallback=bool(args.allow_checkpoint_fallback),
        )
        if loaded[key].checkpoint_kind != "final":
            print(f"[warn] {record.run_dir}: using {loaded[key].checkpoint_kind} checkpoint {loaded[key].checkpoint}")

    market_hashes = {item.market_hash for item in loaded.values()}
    if len(market_hashes) != 1:
        detail = {str(item.record.run_dir): item.market_hash for item in loaded.values()}
        raise ValueError(
            "selected epsilon/seed runs do not share one canonical market snapshot; "
            f"paired deformations would be invalid: {detail}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for loaded_run in sorted(loaded.values(), key=lambda item: (item.record.epsilon, item.record.seed)):
        residual = _final_residuals(loaded_run.record.run_dir)
        if not args.allow_missing_residuals:
            missing = [metric for metric in RESIDUAL_METRICS if not math.isfinite(float(residual[metric]))]
            if missing:
                raise ValueError(f"{loaded_run.record.run_dir}: final outer row lacks {missing}")
        residual_rows.append({
            "group_id": group_id,
            "n_assets": loaded_run.record.n_assets,
            "m_states": loaded_run.record.m_states,
            "epsilon": loaded_run.record.epsilon,
            "seed": loaded_run.record.seed,
            "outer_iter": residual["outer_iter"],
            **{metric: residual[metric] for metric in RESIDUAL_METRICS},
            "guard_fraction_tau": loaded_run.guard_fraction_tau,
            "checkpoint_kind": loaded_run.checkpoint_kind,
            "checkpoint": str(loaded_run.checkpoint),
            "run_dir": str(loaded_run.record.run_dir),
        })
        run_rows.append({
            "group_id": group_id,
            "epsilon": loaded_run.record.epsilon,
            "seed": loaded_run.record.seed,
            "risk_premium_mode": loaded_run.record.risk_premium_mode,
            "status": loaded_run.record.status,
            "market_hash": loaded_run.market_hash,
            "checkpoint_kind": loaded_run.checkpoint_kind,
            "checkpoint": str(loaded_run.checkpoint),
            "run_dir": str(loaded_run.record.run_dir),
        })

    for epsilon in eps_values:
        if _is_zero_eps(epsilon):
            continue
        for seed in target_by_eps[epsilon]:
            current = loaded[(seed, _eps_key(epsilon))]
            baseline = loaded[(seed, _eps_key(baseline_eps))]
            delta_v = current.values - baseline.values
            delta_theta = np.linalg.norm(current.theta_norm - baseline.theta_norm, ord=2, axis=1)
            for index, tau in enumerate(tau_grid):
                per_seed_rows.append({
                    "group_id": group_id,
                    "n_assets": sample.n_assets,
                    "m_states": sample.m_states,
                    "epsilon": epsilon,
                    "seed": seed,
                    "tau": tau,
                    "w0": args.w0,
                    "delta_V": float(delta_v[index]),
                    "delta_theta_l2": float(delta_theta[index]),
                })

    summary_rows: List[Dict[str, Any]] = []
    for epsilon in [value for value in eps_values if not _is_zero_eps(value)]:
        seeds = target_by_eps[epsilon]
        by_seed = {
            seed: [
                row for row in per_seed_rows
                if row["seed"] == seed and _eps_key(row["epsilon"]) == _eps_key(epsilon)
            ]
            for seed in seeds
        }
        for index, tau in enumerate(tau_grid):
            delta_v_values = [float(by_seed[seed][index]["delta_V"]) for seed in seeds]
            delta_theta_values = [float(by_seed[seed][index]["delta_theta_l2"]) for seed in seeds]
            delta_v_mean, delta_v_std = _mean_std(delta_v_values)
            delta_theta_mean, delta_theta_std = _mean_std(delta_theta_values)
            summary_rows.append({
                "group_id": group_id,
                "n_assets": sample.n_assets,
                "m_states": sample.m_states,
                "epsilon": epsilon,
                "tau": tau,
                "w0": args.w0,
                "n": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "delta_V_mean": delta_v_mean,
                "delta_V_std": delta_v_std,
                "delta_theta_l2_mean": delta_theta_mean,
                "delta_theta_l2_std": delta_theta_std,
            })

    residual_summary_rows: List[Dict[str, Any]] = []
    for epsilon in eps_values:
        rows = [row for row in residual_rows if _eps_key(row["epsilon"]) == _eps_key(epsilon)]
        summary: Dict[str, Any] = {
            "group_id": group_id,
            "n_assets": sample.n_assets,
            "m_states": sample.m_states,
            "epsilon": epsilon,
            "n_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda row: int(row["seed"]))),
        }
        for metric in (*RESIDUAL_METRICS, "guard_fraction_tau"):
            values = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            mean, std = _mean_std(values)
            summary[f"n_{metric}"] = len(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        residual_summary_rows.append(summary)

    curve_fields = (
        "group_id", "n_assets", "m_states", "epsilon", "seed", "tau", "w0",
        "delta_V", "delta_theta_l2",
    )
    curve_summary_fields = (
        "group_id", "n_assets", "m_states", "epsilon", "tau", "w0", "n", "seeds",
        "delta_V_mean", "delta_V_std", "delta_theta_l2_mean", "delta_theta_l2_std",
    )
    residual_fields = (
        "group_id", "n_assets", "m_states", "epsilon", "seed", "outer_iter",
        *RESIDUAL_METRICS, "guard_fraction_tau", "checkpoint_kind", "checkpoint", "run_dir",
    )
    residual_summary_fields: List[str] = [
        "group_id", "n_assets", "m_states", "epsilon", "n_runs", "seeds",
    ]
    for metric in (*RESIDUAL_METRICS, "guard_fraction_tau"):
        residual_summary_fields.extend((f"n_{metric}", f"{metric}_mean", f"{metric}_std"))
    run_fields = (
        "group_id", "epsilon", "seed", "risk_premium_mode", "status", "market_hash",
        "checkpoint_kind", "checkpoint", "run_dir",
    )

    _write_csv(output_dir / "curves_per_seed.csv", per_seed_rows, curve_fields)
    _write_csv(output_dir / "curves_mean_std.csv", summary_rows, curve_summary_fields)
    _write_csv(output_dir / "final_residuals_per_seed.csv", residual_rows, residual_fields)
    _write_csv(output_dir / "final_residuals_mean_std.csv", residual_summary_rows, residual_summary_fields)
    _write_csv(output_dir / "runs_used.csv", run_rows, run_fields)
    figure_paths = _plot_curves(summary_rows, output_dir, args) if not args.no_plot else []

    metadata = {
        "group_id": group_id,
        "n_assets": sample.n_assets,
        "m_states": sample.m_states,
        "model_type": sample.model_type,
        "w0": args.w0,
        "x_slice": "xbar from market_params.npz",
        "tau_min": tau_min,
        "tau_max": tau_max,
        "n_tau": args.n_tau,
        "epsilon_values": eps_values,
        "baseline_epsilon": baseline_eps,
        "paired_seeds_by_epsilon": {str(epsilon): seeds for epsilon, seeds in target_by_eps.items()},
        "market_hash": next(iter(market_hashes)),
        "checkpoint_policy": (
            "official_final_only" if not args.allow_checkpoint_fallback
            else "legacy_fallback_final_last_best"
        ),
        "allow_checkpoint_fallback": bool(args.allow_checkpoint_fallback),
        "checkpoint_order": (
            ["value_net_final.pt"] if not args.allow_checkpoint_fallback
            else ["value_net_final.pt", "value_net_last.pt", "value_net_best.pt"]
        ),
        "curve_definition": {
            "delta_V": "V_epsilon(tau,w0,xbar)-V_0(tau,w0,xbar) (signed)",
            "delta_theta_l2": "L2 norm across assets of theta_epsilon/w0-theta_0/w0",
            "aggregation": (
                "pair within training seed, then pointwise mean and sample std; "
                "sample SD is undefined (NaN) for n=1"
            ),
        },
        "figure_style": {
            "layout": "combined_one_by_two_plus_standalone_value_and_policy",
            "fig_width_inches": float(args.fig_width),
            "fig_height_inches": float(args.fig_height),
            "standalone_value_width_inches": float(args.value_fig_width),
            "standalone_policy_width_inches": float(args.policy_fig_width),
            "standalone_value_height_inches": (
                float(args.single_fig_height)
                if args.value_fig_height is None
                else float(args.value_fig_height)
            ),
            "standalone_policy_height_inches": (
                float(args.single_fig_height)
                if args.policy_fig_height is None
                else float(args.policy_fig_height)
            ),
            "standalone_shared_height_fallback_inches": float(
                args.single_fig_height
            ),
            "font_size_pt": float(args.font_size),
            "dpi": int(args.dpi),
            "formats": [
                part.strip().lower().lstrip(".")
                for part in args.formats.split(",") if part.strip()
            ],
            "cmap": str(args.cmap),
            "line_width": 1.5,
            "uncertainty_band": "pointwise mean plus/minus one sample SD",
            "legend": (
                "combined: one shared legend on the right; standalone Value: "
                "none; standalone Policy: one external legend on the right "
                "at bbox_to_anchor=(0.63, 0.6)"
            ),
            "output_stems": [
                "nonaffine_figure4",
                "V_diff_from_base",
                "pi_diff_from_base",
            ],
            "epsilon_label_decimals": 2,
            "y_axis_labels": "omitted; definitions supplied in the manuscript caption",
            "x_max_major_intervals": int(args.x_max_ticks),
            "y_max_major_intervals": int(args.y_max_ticks),
        },
        "group_config": sample.group_config,
        "figures": [str(path) for path in figure_paths],
    }
    _write_json(output_dir / "postprocess_config.json", metadata)
    return metadata


def _resolve_device(spec: str) -> torch.device:
    text = str(spec or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={spec}, but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paired seed-averaged Liu non-affine deformation curves and residual summaries."
    )
    parser.add_argument(
        "--out-root", type=Path,
        help="Root containing recursively discoverable run config.json files.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output root (default: OUT_ROOT/nonaffine_postprocess).",
    )
    parser.add_argument("--model-type", default="pipinn", choices=("pipinn", "pi-pinn"))
    parser.add_argument(
        "--n-assets", type=int, default=None,
        help="Optional asset-dimension filter (paper baseline: 30).",
    )
    parser.add_argument("--m-states", default="", help="Optional comma/range state-dimension filter, e.g. 1 or 1,3.")
    parser.add_argument("--group-id", default="", help="Optional exact 12-character training-config group selector.")
    parser.add_argument("--run-name-regex", default="", help="Optional regex applied to the run directory name.")
    parser.add_argument(
        "--eps", default="",
        help="Requested epsilon list. Empty means every discovered epsilon; eps=0 is added.",
    )
    parser.add_argument("--expected-seeds", default="", help="Comma/space/range seed set, e.g. 1-10.")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Pilot only: use per-epsilon intersection with baseline seeds.",
    )
    parser.add_argument(
        "--allow-checkpoint-fallback", action="store_true",
        help="Exploratory legacy mode: allow final -> last -> best; paper mode requires final.",
    )
    parser.add_argument("--min-seeds", type=int, default=2, help="Minimum paired seeds per epsilon>0 (default: 2).")
    parser.add_argument(
        "--allow-missing-residuals", action="store_true",
        help="Keep curves if final held-out residual fields are missing.",
    )
    parser.add_argument("--w0", type=float, default=0.5)
    parser.add_argument("--tau-min", type=float, default=0.0)
    parser.add_argument("--tau-max", type=float, default=None, help="Default: the saved market horizon.")
    parser.add_argument("--n-tau", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated figure formats.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig-width", type=float, default=13.2,
        help="Combined two-panel figure width in inches (default: 13.2).",
    )
    parser.add_argument(
        "--fig-height", type=float, default=4.0,
        help="Combined two-panel figure height in inches (default: 4.0).",
    )
    parser.add_argument(
        "--value-fig-width", type=float, default=5.3,
        help="Standalone Value figure width in inches (default: 6.0).",
    )
    parser.add_argument(
        "--policy-fig-width", type=float, default=9.0,
        help="Standalone Policy figure width including its legend (default: 7.2).",
    )
    parser.add_argument(
        "--single-fig-height", type=float, default=4.5,
        help=(
            "Shared fallback height for standalone Value/Policy figures in "
            "inches (default: 4.0)."
        ),
    )
    parser.add_argument(
        "--value-fig-height", type=float, default=None,
        help=(
            "Standalone Value figure height in inches; overrides "
            "--single-fig-height for Value."
        ),
    )
    parser.add_argument(
        "--policy-fig-height", type=float, default=None,
        help=(
            "Standalone Policy figure height in inches; overrides "
            "--single-fig-height for Policy."
        ),
    )
    parser.add_argument(
        "--font-size", type=float, default=22.0,
        help="Base axis, tick, and shared-legend font size in points (default: 22).",
    )
    parser.add_argument(
        "--x-max-ticks", type=int, default=4,
        help="Approximate maximum number of major x-axis intervals per panel (default: 4).",
    )
    parser.add_argument(
        "--y-max-ticks", type=int, default=4,
        help="Approximate maximum number of major y-axis intervals per panel (default: 4).",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic end-to-end smoke test and exit.")
    return parser


def _make_self_test_run(root: Path, *, seed: int, epsilon: float) -> None:
    tag = f"pipinn_nonaffine_N2_M1_eps{epsilon:g}_seed{seed}"
    run_dir = root / "pi-pinn" / tag
    weight_dir = root / "weights" / "pi-pinn" / tag
    run_dir.mkdir(parents=True)
    weight_dir.mkdir(parents=True)
    cfg_args = {
        "model_type": "pipinn", "risk_premium_mode": "tanh", "nonaffine_eps": epsilon,
        "n_assets": 2, "m_states": 1, "seed": seed, "market_seed": 17,
        "value_hidden": 4, "value_depth": 1, "tau_max": 1.0, "w_min": 0.1, "w_max": 1.0,
        "batch_size": 8, "outer_iters": 2,
        "output_root": str(run_dir), "weight_root": str(weight_dir), "run_tag": tag,
    }
    _write_json(run_dir / "config.json", {"args": cfg_args, "weight_dir": str(weight_dir), "created_at": "test"})
    _write_json(run_dir / "status.json", {"status": "success", "updated_at": "test"})
    (run_dir / "_SUCCESS").touch()
    arrays = {
        "K": np.eye(1), "xbar": np.zeros(1), "SigmaX": np.eye(1), "rho": np.ones((2, 1)) * 0.02,
        "Lam": np.ones((2, 1)) * 0.1, "Q": np.eye(1), "Gamma": np.ones((2, 1)) * 0.02,
        "k0": np.zeros(1), "lam0": np.ones(2) * 0.1, "X_min": np.ones(1) * -1.0,
        "X_max": np.ones(1), "eta": np.ones(1), "gamma": np.array([2.0]), "r": np.array([0.03]),
        "tau_max": np.array([1.0]), "W_min": np.array([0.1]), "W_max": np.array([1.0]),
        "market_seed": np.array([17]), "seed": np.array([seed]),
    }
    np.savez(run_dir / "market_params.npz", **arrays)
    with (run_dir / "outer_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("outer_iter", *RESIDUAL_METRICS))
        writer.writeheader()
        writer.writerow({"outer_iter": 2, "val_pde_rms": 1 + epsilon, "val_terminal_rms": 2, "val_pres": 3 + epsilon})
    model = ValueNetND(1, 4, 1)
    torch.manual_seed(seed)
    for parameter in model.parameters():
        nn.init.uniform_(parameter, -0.03, 0.03)
    # Add a deterministic epsilon displacement so the paired curves exercise
    # both signed value differences and the portfolio norm.
    with torch.no_grad():
        model.net[-1].bias.add_(float(epsilon) * 0.01)
        model.net[0].weight.add_(float(epsilon) * 0.005)
    torch.save(model.state_dict(), weight_dir / "value_net_final.pt")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="liu_nonaffine_selftest_") as tmp:
        root = Path(tmp)
        for seed in (1, 2):
            for epsilon in (0.0, 0.5):
                _make_self_test_run(root, seed=seed, epsilon=epsilon)
        args = build_parser().parse_args([
            "--out-root", str(root), "--output", str(root / "result"),
            "--n-assets", "2", "--m-states", "1", "--expected-seeds", "1-2",
            "--eps", "0,0.5", "--n-tau", "7", "--device", "cpu", "--no-plot",
        ])
        result = run(args)
        group_dir = Path(result[0]["output_dir"])
        curve_rows = list(csv.DictReader((group_dir / "curves_mean_std.csv").open(encoding="utf-8")))
        residual_rows = list(csv.DictReader((group_dir / "final_residuals_mean_std.csv").open(encoding="utf-8")))
        assert len(curve_rows) == 7
        assert {int(row["n"]) for row in curve_rows} == {2}
        assert len(residual_rows) == 2
        used_rows = csv.DictReader((group_dir / "runs_used.csv").open(encoding="utf-8"))
        assert {row["checkpoint_kind"] for row in used_rows} == {"final"}
    print("[self-test] PASS: discovery, pairing, checkpoint load, derivatives, CSV summaries")


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.out_root is None:
        raise ValueError("--out-root is required")
    if args.n_tau < 2:
        raise ValueError("--n-tau must be >= 2")
    if args.min_seeds < 1:
        raise ValueError("--min-seeds must be >= 1")
    if args.w0 <= 0:
        raise ValueError("--w0 must be positive")
    for name in (
        "fig_width",
        "fig_height",
        "value_fig_width",
        "policy_fig_width",
        "single_fig_height",
        "font_size",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("value_fig_height", "policy_fig_height"):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and positive"
            )
    if args.x_max_ticks < 2 or args.y_max_ticks < 2:
        raise ValueError("--x-max-ticks and --y-max-ticks must be >= 2")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    out_root = args.out_root.expanduser().resolve()
    if not out_root.is_dir():
        raise FileNotFoundError(f"--out-root is not a directory: {out_root}")
    output_root = (args.output or (out_root / "nonaffine_postprocess")).expanduser().resolve()
    m_states = _parse_csv_ints(args.m_states)
    groups = discover_runs(
        out_root, model_type=args.model_type, n_assets=args.n_assets,
        m_states=m_states, run_name_regex=args.run_name_regex,
    )
    if args.group_id:
        groups = {key: value for key, value in groups.items() if key == args.group_id}
    if not groups:
        raise ValueError("no matching non-affine run groups found")

    device = _resolve_device(args.device)
    print(f"[info] device={device}; discovered groups={sorted(groups)}")
    results: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    for group_id, records in sorted(groups.items()):
        sample = next(iter(records.values()))
        group_output = output_root / f"group_{group_id}_N{sample.n_assets}_M{sample.m_states}"
        metadata = process_group(
            group_id, records, out_root=out_root, output_dir=group_output, args=args, device=device,
        )
        metadata["output_dir"] = str(group_output)
        results.append(metadata)
        index_rows.append({
            "group_id": group_id, "n_assets": sample.n_assets, "m_states": sample.m_states,
            "model_type": sample.model_type, "epsilon_values": ",".join(map(str, metadata["epsilon_values"])),
            "output_dir": str(group_output),
        })
        print(f"[done] group={group_id}, N={sample.n_assets}, M={sample.m_states}: {group_output}")
    _write_csv(
        output_root / "groups_index.csv", index_rows,
        ("group_id", "n_assets", "m_states", "model_type", "epsilon_values", "output_dir"),
    )
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if torch is None:
        parser.error(
            "PyTorch is required to load and differentiate Liu checkpoints; "
            "run this command in the training environment"
        )
    if args.self_test:
        run_self_test()
        return
    try:
        run(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
