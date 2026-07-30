#!/usr/bin/env python3
"""Independent Liu/Kim--Omberg M=1 exact-map and E4 FD evaluator.

For checkpoint ``k`` the training loop has produced

    v_tilde[k] ~= E(alpha[k-1]).

This program extracts ``alpha[k]=G(v_tilde[k])`` with the same concavity
guard and optional raw-theta clipping as training, freezes it, and solves the
two-dimensional policy-evaluation PDE.  Hence the exact-map object is
``E(alpha[k])=P(v_tilde[k])``.  The shifted result is also the independent FD
reference needed for the next checkpoint's E4 approximation error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from joint_market_setup_dirichlet import validate_market_snapshot
from liu_exact_map_core import (
    AffineClosedForm,
    FDGrid,
    LiuProblem,
    evaluate_fd_wealth_bundle,
    solve_affine_closed_form,
    solve_frozen_policy,
    x_norm_components,
)


CHECKPOINT_RE = re.compile(r"value_net_iter(\d+)\.pt$")
VWW_GUARD = 1.0e-8
POLICY_EXTENSIONS = ("boundary-projection", "neural-extrapolation")
BOUNDARY_VARIANTS = ("linearity", "crra-robin", "exact-dirichlet")
REFINEMENT_SCOPE = "grid_and_domain_within_primary_boundary"
BOUNDARY_SENSITIVITY_ROLE = "reported_separately_not_gated"

ERROR_PREFIXES = ("e_input", "e_map")
ERROR_COMPONENTS = ("value", "vw", "vww", "vwx", "bundle", "X")
RATIO_FIELDS = [
    "problem", "group", "protocol_hash", "model_type", "n_assets", "m_states",
    "seed", "market_seed", "source_outer_iter", "frozen_policy_iter",
    "greedy_policy_iter", "target_value_outer_iter", "checkpoint",
    "checkpoint_sha256", "market_sha256", "network_dtype", "eval_margin",
    "eval_x_margin", "eval_w_min_override", "eval_w_max_override",
    "ev_w_min", "ev_w_max", "ev_x_min", "ev_x_max",
    "domain_mode", "domain_factor", "wealth_domain_factor",
    "factor_domain_factor",
    "fd_y_min", "fd_y_max", "fd_w_min", "fd_w_max", "fd_x_min", "fd_x_max",
    "boundary", "drift_scheme", "grid_factor", "ny", "nx", "nt", "dy", "dx",
    "dt", "is_primary", "is_verification",
    *[f"{prefix}_{component}" for prefix in ERROR_PREFIXES for component in ERROR_COMPONENTS],
    "rho_exact", "denominator_defined", "support_status", "checkpoint_selection",
    "analysis_mode", "refinement_rule", "refinement_scope",
    "boundary_sensitivity_role", "min_paper_checkpoint",
    "policy_extension", "map_definition",
    "map_variant", "local_map_unmodified_on_xfd",
    "local_greedy_unmodified_on_policy_support", "whole_space_map_claim",
    "outside_collocation_fraction_fd", "outside_collocation_y_fraction_fd",
    "outside_collocation_x_fraction_fd",
    "guard_frac_fd", "positive_curvature_frac_fd", "theta_any_clip_frac_fd",
    "theta_component_clip_frac_fd", "guard_frac_ev", "positive_curvature_frac_ev",
    "theta_any_clip_frac_ev", "theta_component_clip_frac_ev",
    "vartheta_l2_min_fd", "vartheta_l2_max_fd", "vartheta_component_min_fd",
    "vartheta_component_max_fd", "min_log_joint_eig", "max_log_joint_eig",
    "min_original_joint_eig", "max_original_joint_eig",
    "nonpositive_log_eig_fraction", "max_peclet_y", "max_peclet_x",
    "upwind_y_fraction", "upwind_x_fraction", "max_linear_residual",
    "linear_residual_tolerance", "boundary_elimination_size",
    "boundary_elimination_rank", "boundary_elimination_cond_inf",
    "min_linear_system_lu_pivot_ratio", "policy_hash",
    "grid_abs_change", "grid_rel_change",
    "domain_abs_change", "domain_rel_change",
    "wealth_domain_abs_change", "wealth_domain_rel_change",
    "factor_domain_abs_change", "factor_domain_rel_change",
    "refinement_tolerance", "numerical_abs_change",
    "numerical_tolerance_ratio",
    "boundary_abs_change", "boundary_rel_change",
    "boundary_tolerance_ratio", "boundary_sensitivity_status",
    "rho_sensitivity_envelope", "refinement_status",
    "contraction_status",
]

E4_FIELDS = [
    "problem", "group", "protocol_hash", "model_type", "n_assets", "m_states",
    "seed", "market_seed", "target_outer_iter", "frozen_policy_iter",
    "policy_source_outer_iter", "checkpoint", "checkpoint_sha256", "market_sha256",
    "source_policy_hash", "fd_reference_source",
    "eval_margin", "eval_x_margin", "eval_w_min_override",
    "eval_w_max_override", "ev_w_min", "ev_w_max", "ev_x_min", "ev_x_max",
    "domain_mode", "domain_factor", "wealth_domain_factor",
    "factor_domain_factor",
    "fd_y_min", "fd_y_max", "fd_w_min", "fd_w_max", "fd_x_min", "fd_x_max",
    "boundary", "grid_factor", "ny", "nx", "nt",
    "is_primary",
    "is_verification", "analysis_mode", "refinement_rule",
    "refinement_scope", "boundary_sensitivity_role",
    "min_paper_checkpoint",
    "policy_extension", "map_definition",
    "e_approx_value", "e_approx_vw", "e_approx_vww",
    "e_approx_vwx", "e_approx_bundle", "e_approx_X", "support_status",
    "grid_abs_change", "grid_rel_change",
    "domain_abs_change", "domain_rel_change",
    "wealth_domain_abs_change", "wealth_domain_rel_change",
    "factor_domain_abs_change", "factor_domain_rel_change",
    "refinement_tolerance", "numerical_abs_change",
    "numerical_tolerance_ratio",
    "boundary_abs_change", "boundary_rel_change",
    "boundary_tolerance_ratio", "boundary_sensitivity_status",
    "approx_sensitivity_envelope", "refinement_status",
    "source_min_log_joint_eig", "source_max_log_joint_eig",
    "source_min_original_joint_eig", "source_max_original_joint_eig",
    "source_nonpositive_log_eig_fraction",
    "source_outside_collocation_fraction_fd",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tensor_state_hash(path: Path) -> str:
    """Hash state-dict tensor content, independent of ``torch.save`` metadata."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the driver needs torch anyway
        raise RuntimeError("PyTorch is required to validate checkpoint identity") from exc
    state = _state_dict(torch, path)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} in {path} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tuple(value.shape), dtype="<i8").tobytes())
        # Viewing as bytes also supports torch dtypes (notably bfloat16) that
        # NumPy cannot represent directly.
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_OPERATIONAL_ARG_KEYS = {
    "seed", "device", "run_tag", "output_root", "weight_root", "stop_flag_path",
}


def _training_protocol_args(args: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical training/evaluation choices, excluding per-seed locations."""

    return {str(key): args[key] for key in sorted(args) if key not in _OPERATIONAL_ARG_KEYS}


def canonical_array_hash(arrays: Mapping[str, Array]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


_MANAGED_OUTPUTS = (
    "exact_map_refinement.csv", "exact_map_ratios.csv",
    "e4_approximation_refinement.csv", "e4_approximation_errors.csv",
    "exact_map_config.json", "exact_map_status.json",
    "_SUCCESS_EXACT_MAP", "_FAILED_EXACT_MAP",
)


def _prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in _MANAGED_OUTPUTS if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"exact-map output already contains managed artifacts {existing}; "
            "pass --overwrite to replace them"
        )
    for name in _MANAGED_OUTPUTS:
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _check_output(output: Path, overwrite: bool) -> bool:
    """Check managed outputs without modifying an earlier completed audit."""

    existing = [name for name in _MANAGED_OUTPUTS if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"exact-map output already contains managed artifacts {existing}; "
            "pass --overwrite to replace them"
        )
    return bool(existing)


def _commit_staged_output(stage: Path, output: Path) -> None:
    """Replace managed files only after the complete FD audit succeeds."""

    _prepare_output(output, overwrite=True)
    for name in _MANAGED_OUTPUTS:
        source = stage / name
        if source.is_file() or source.is_symlink():
            os.replace(source, output / name)


def _pick(mapping: Mapping[str, Any], names: Sequence[str], *, default: Any = None,
          required: bool = False) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    if required:
        raise KeyError(f"missing required field; tried {list(names)}")
    return default


def _first_margin(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = list(np.asarray(value).reshape(-1))
        margin = float(pieces[0]) if pieces else 0.0
    else:
        text = str(value)
        pieces = [item.strip() for item in text.split(",") if item.strip()]
        margin = float(pieces[0]) if pieces else 0.0
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"eval margin must lie in [0,1), got {margin}")
    return margin


def _shrink(lo: float, hi: float, margin: float) -> Tuple[float, float]:
    delta = 0.5 * float(margin) * (float(hi) - float(lo))
    return float(lo) + delta, float(hi) - delta


def _resolve_evaluation_window(
    *,
    saved_w_min: float,
    saved_w_max: float,
    saved_x_min: float,
    saved_x_max: float,
    eval_margin: float,
    eval_w_min_override: Optional[float] = None,
    eval_w_max_override: Optional[float] = None,
    eval_x_margin_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve a one-sided wealth override on top of the legacy margin.

    ``eval_margin`` keeps its historical meaning: it symmetrically shrinks
    both saved wealth and factor intervals.  Explicit wealth endpoints then
    replace only their corresponding endpoint.  ``eval_x_margin_override`` is
    an optional way to keep the factor window fixed while wealth endpoints are
    varied.  None of these values changes the FD rectangle.
    """

    margin = _first_margin(eval_margin)
    x_margin = (
        margin
        if eval_x_margin_override is None
        else _first_margin(eval_x_margin_override)
    )
    base_w_min, base_w_max = _shrink(saved_w_min, saved_w_max, margin)
    ev_x_min, ev_x_max = _shrink(saved_x_min, saved_x_max, x_margin)

    def optional_endpoint(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    requested_w_min = optional_endpoint(eval_w_min_override, "eval_w_min")
    requested_w_max = optional_endpoint(eval_w_max_override, "eval_w_max")
    ev_w_min = base_w_min if requested_w_min is None else requested_w_min
    ev_w_max = base_w_max if requested_w_max is None else requested_w_max
    if not (math.isfinite(ev_w_min) and math.isfinite(ev_w_max)):
        raise ValueError("effective evaluation wealth bounds must be finite")
    if not (saved_w_min <= ev_w_min < ev_w_max <= saved_w_max):
        raise ValueError(
            "effective evaluation wealth bounds must satisfy "
            f"{saved_w_min} <= w_min < w_max <= {saved_w_max}; "
            f"got [{ev_w_min}, {ev_w_max}]"
        )
    if ev_w_min <= 0.0:
        raise ValueError("effective evaluation wealth lower bound must be positive")
    if not (saved_x_min <= ev_x_min < ev_x_max <= saved_x_max):
        raise ValueError("effective evaluation factor bounds are invalid")

    return {
        "definition": (
            "symmetric eval_margin baseline; optional eval_x_margin replaces "
            "the factor margin; explicit eval_w_min/eval_w_max replace only "
            "their corresponding wealth endpoints"
        ),
        "eval_margin": float(margin),
        "eval_x_margin": float(x_margin),
        "eval_w_min_override": requested_w_min,
        "eval_w_max_override": requested_w_max,
        "ev_w_min": float(ev_w_min),
        "ev_w_max": float(ev_w_max),
        "ev_x_min": float(ev_x_min),
        "ev_x_max": float(ev_x_max),
        "saved_w_min": float(saved_w_min),
        "saved_w_max": float(saved_w_max),
        "saved_x_min": float(saved_x_min),
        "saved_x_max": float(saved_x_max),
        "fd_domain_depends_on_evaluation_window": False,
    }


def _resolve_weight_dir(config: Mapping[str, Any], run_dir: Path,
                        override: Optional[Path]) -> Path:
    """Resolve a recorded weight path without assuming evaluation CWD.

    Launchers often record both ``output_dir`` and ``weight_dir`` as paths
    relative to the training ``cwd``.  Joining such a weight path to the run
    directory duplicates the output prefix.  Prefer the recorded CWD, then
    remap the recorded output/weight layout onto a moved run tree, and finally
    try the known launcher sibling layout.
    """

    if override is not None:
        candidate = override.expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"--weight-dir is not a directory: {candidate}")
        return candidate
    raw = config.get("weight_dir")
    candidates: List[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    if raw not in (None, ""):
        weight_raw = Path(str(raw)).expanduser()
        if weight_raw.is_absolute():
            add(weight_raw)
        else:
            recorded_cwd = config.get("cwd")
            if recorded_cwd not in (None, ""):
                add(Path(str(recorded_cwd)).expanduser() / weight_raw)

        output_raw_value = config.get("output_dir")
        if output_raw_value not in (None, ""):
            output_raw = Path(str(output_raw_value)).expanduser()
            try:
                common = Path(os.path.commonpath([str(output_raw), str(weight_raw)]))
                output_tail = output_raw.relative_to(common)
                weight_tail = weight_raw.relative_to(common)
                tail_parts = output_tail.parts
                if (not tail_parts or
                        tuple(run_dir.parts[-len(tail_parts):]) == tuple(tail_parts)):
                    actual_common = run_dir
                    for _part in tail_parts:
                        actual_common = actual_common.parent
                    add(actual_common / weight_tail)
            except (ValueError, OSError):
                pass

        # Legacy fallback. It is intentionally tried after CWD/layout remap.
        add(run_dir / weight_raw)

    if run_dir.parent.name in {"pi-pinn", "pinn"}:
        launcher_root = run_dir.parent.parent
        add(launcher_root / "weights" / run_dir.parent.name / run_dir.name)
    add(run_dir / "weights")
    add(run_dir)

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "iterates").is_dir():
            return candidate
    raise FileNotFoundError(
        "could not resolve a weight directory containing iterates/; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _load_outer_index(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing outer_history.csv: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        outer = int(row["outer_iter"])
        if outer in out:
            raise ValueError(f"duplicate outer_iter={outer} in {path}")
        out[outer] = row
    return out


def discover_checkpoints(weight_dir: Path, explicit: Optional[Sequence[int]] = None) -> List[Tuple[int, Path]]:
    files = sorted((weight_dir / "iterates").glob("value_net_iter*.pt"))
    found: Dict[int, Path] = {}
    for path in files:
        match = CHECKPOINT_RE.search(path.name)
        if match:
            outer = int(match.group(1))
            if outer in found:
                raise ValueError(f"duplicate checkpoint outer={outer}")
            found[outer] = path.resolve()
    if not found:
        raise FileNotFoundError(f"no iterate checkpoints under {weight_dir / 'iterates'}")
    if explicit:
        missing = sorted(set(int(value) for value in explicit) - set(found))
        if missing:
            raise ValueError(f"requested checkpoints are missing: {missing}")
        found = {key: found[key] for key in sorted(set(int(value) for value in explicit))}
    return sorted(found.items())


@dataclass
class RunSpec:
    run_dir: Path
    config_path: Path
    market_path: Path
    weight_dir: Path
    config: Mapping[str, Any]
    args: Mapping[str, Any]
    problem: LiuProblem
    closed_form: AffineClosedForm
    checkpoints: List[Tuple[int, Path]]
    all_checkpoints: List[Tuple[int, Path]]
    outer_index: Dict[int, Dict[str, str]]
    seed: int
    market_seed: int
    eval_margin: float
    eval_x_margin: float
    eval_w_min_override: Optional[float]
    eval_w_max_override: Optional[float]
    eval_w_bounds: Tuple[float, float]
    eval_x_bounds: Tuple[float, float]
    evaluation_window: Mapping[str, Any]
    market_hash: str
    group: str
    checkpoint_selection: str
    terminal_state_hash: str
    training_protocol_hash: str
    training_protocol_args: Mapping[str, Any]


def load_run(run_dir: Path, *, weight_dir_override: Optional[Path] = None,
             checkpoint_subset: Optional[Sequence[int]] = None,
             eval_margin_override: Optional[float] = None,
             eval_w_min_override: Optional[float] = None,
             eval_w_max_override: Optional[float] = None,
             eval_x_margin_override: Optional[float] = None,
             allow_sparse_subset: bool = False) -> RunSpec:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    market_path = run_dir / "market_params.npz"
    status_path = run_dir / "status.json"
    if not config_path.is_file() or not market_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(
            "run must contain config.json, market_params.npz, and status.json"
        )
    terminal_markers = [name for name in ("_SUCCESS", "_STOPPED_EARLY", "_FAILED")
                        if (run_dir / name).is_file()]
    if terminal_markers != ["_SUCCESS"]:
        raise ValueError(
            "exact-map input must have the unique terminal marker _SUCCESS; "
            f"found {terminal_markers}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("config.json must contain a JSON object")
    args = config.get("args", config)
    if not isinstance(args, Mapping):
        raise TypeError("config.json args must contain a JSON object")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not isinstance(status, Mapping):
        raise TypeError("status.json must contain a JSON object")
    if str(status.get("status", "")) != "success":
        raise ValueError(f"run status is not success: {status.get('status')!r}")
    if str(_pick(args, ("model_type",), default="")).lower() != "pipinn":
        raise ValueError("Liu exact-map input must be a PI-PINN run")
    if int(_pick(args, ("m_states",), required=True)) != 1:
        raise ValueError("the independent Liu FD evaluator currently supports M=1 only")
    if str(_pick(args, ("risk_premium_mode",), default="affine")).lower() != "affine":
        raise ValueError("exact-map ratio requires the affine benchmark reference")
    if float(_pick(args, ("nonaffine_eps",), default=0.0)) != 0.0:
        raise ValueError("exact-map ratio is unavailable for non-affine eps>0")

    with np.load(market_path, allow_pickle=False) as payload:
        market = {key: np.asarray(payload[key]) for key in payload.files}
    required = ("K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0", "lam0",
                "X_min", "X_max", "gamma", "r", "tau_max", "W_min", "W_max",
                "seed", "market_seed")
    missing = [key for key in required if key not in market]
    if missing:
        raise KeyError(f"market_params.npz is missing {missing}")
    try:
        validate_market_snapshot(market)
    except ValueError as exc:
        raise ValueError(f"{market_path}: invalid market snapshot: {exc}") from exc
    K = float(np.asarray(market["K"]).reshape(1, 1)[0, 0])
    xbar = float(np.asarray(market["xbar"]).reshape(-1)[0])
    sigma_x = float(np.asarray(market["SigmaX"]).reshape(1, 1)[0, 0])
    rho = np.asarray(market["rho"], dtype=np.float64).reshape(-1, 1)
    Lam = np.asarray(market["Lam"], dtype=np.float64).reshape(-1, 1)
    Gamma = np.asarray(market["Gamma"], dtype=np.float64).reshape(-1, 1)
    Q = float(np.asarray(market["Q"]).reshape(1, 1)[0, 0])
    k0 = float(np.asarray(market["k0"]).reshape(-1)[0])
    lam0 = np.asarray(market["lam0"], dtype=np.float64).reshape(-1)
    if not np.allclose(Q, sigma_x * sigma_x, rtol=1e-10, atol=1e-12):
        raise ValueError("saved Q is inconsistent with SigmaX")
    if not np.allclose(Gamma, rho * sigma_x, rtol=1e-10, atol=1e-12):
        raise ValueError("saved Gamma is inconsistent with rho and SigmaX")
    if not math.isclose(k0, K * xbar, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("saved k0 is inconsistent with K*xbar")
    if (Lam.shape[0] != lam0.size or Gamma.shape[0] != lam0.size
            or rho.shape[0] != lam0.size):
        raise ValueError("market asset dimensions are inconsistent")

    w_min = float(np.asarray(market["W_min"]).reshape(-1)[0])
    w_max = float(np.asarray(market["W_max"]).reshape(-1)[0])
    x_min = float(np.asarray(market["X_min"]).reshape(-1)[0])
    x_max = float(np.asarray(market["X_max"]).reshape(-1)[0])
    if not (0.0 < w_min < w_max and x_min < x_max):
        raise ValueError("saved wealth/factor bounds are invalid")
    seed = int(_pick(args, ("seed",), required=True))
    market_seed = int(_pick(args, ("market_seed",), default=seed))
    snapshot_seed = int(np.asarray(market["seed"]).reshape(-1)[0])
    snapshot_market_seed = int(np.asarray(market["market_seed"]).reshape(-1)[0])
    if snapshot_seed != seed:
        raise ValueError("config and snapshot training seeds disagree")
    if snapshot_market_seed != market_seed:
        raise ValueError("config and snapshot market seeds disagree")
    configured_n_assets = int(_pick(args, ("n_assets",), required=True))
    if configured_n_assets != lam0.size:
        raise ValueError(
            f"config n_assets={configured_n_assets} but snapshot has {lam0.size} assets"
        )

    def require_config_snapshot_scalar(arg_name: str, snapshot_name: str,
                                       snapshot_value: float) -> None:
        configured = float(_pick(args, (arg_name,), required=True))
        if not math.isclose(configured, snapshot_value, rel_tol=1e-12, abs_tol=1e-13):
            raise ValueError(
                f"config {arg_name}={configured!r} disagrees with snapshot "
                f"{snapshot_name}={snapshot_value!r}"
            )

    require_config_snapshot_scalar(
        "gamma", "gamma", float(np.asarray(market["gamma"]).reshape(-1)[0])
    )
    require_config_snapshot_scalar("r", "r", float(np.asarray(market["r"]).reshape(-1)[0]))
    require_config_snapshot_scalar(
        "tau_max", "tau_max", float(np.asarray(market["tau_max"]).reshape(-1)[0])
    )
    require_config_snapshot_scalar("w_min", "W_min", w_min)
    require_config_snapshot_scalar("w_max", "W_max", w_max)
    problem = LiuProblem(
        horizon=float(np.asarray(market["tau_max"]).reshape(-1)[0]),
        y_min=math.log(w_min), y_max=math.log(w_max),
        x_min=x_min, x_max=x_max,
        gamma=float(np.asarray(market["gamma"]).reshape(-1)[0]),
        risk_free=float(np.asarray(market["r"]).reshape(-1)[0]),
        K=K, k0=k0, Q=Q, Gamma=Gamma[:, 0], lam0=lam0, Lam=Lam[:, 0],
    )
    closed_form = solve_affine_closed_form(problem)
    stored_cf = run_dir / "closed_form_ode.npz"
    if not stored_cf.is_file():
        raise FileNotFoundError("run is missing closed_form_ode.npz")
    with np.load(stored_cf, allow_pickle=False) as payload:
        missing_cf = [name for name in ("t", "y", "success") if name not in payload.files]
        if missing_cf:
            raise ValueError(f"closed_form_ode.npz is missing {missing_cf}")
        if not bool(np.asarray(payload["success"]).reshape(-1)[0]):
            raise ValueError("saved closed-form ODE solve was unsuccessful")
        stored_t = np.asarray(payload["t"], dtype=np.float64)
        stored_y = np.asarray(payload["y"], dtype=np.float64)
        if (stored_t.ndim != 1 or stored_t.size < 2
                or stored_y.shape != (3, stored_t.size)
                or not np.all(np.isfinite(stored_t)) or not np.all(np.isfinite(stored_y))
                or np.any(np.diff(stored_t) <= 0.0)):
            raise ValueError("closed_form_ode.npz has malformed t/y arrays")
        if (not math.isclose(float(stored_t[0]), 0.0, abs_tol=1e-13)
                or not math.isclose(float(stored_t[-1]), problem.horizon,
                                    rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError("closed_form_ode.npz does not cover the configured horizon")
        probe = np.linspace(0.0, problem.horizon, 33)
        stored = np.stack([np.interp(probe, stored_t, stored_y[i]) for i in range(3)])
        recomputed = np.stack(closed_form.coefficients(probe))
        if not np.allclose(stored, recomputed, rtol=2e-8, atol=2e-10):
            raise ValueError("stored closed_form_ode.npz disagrees with independent Riccati solve")

    weight_dir = _resolve_weight_dir(config, run_dir, weight_dir_override)
    all_checkpoints = discover_checkpoints(weight_dir)
    checkpoints = (discover_checkpoints(weight_dir, checkpoint_subset)
                   if checkpoint_subset else all_checkpoints)
    expected_final = int(_pick(args, ("outer_iters",), default=all_checkpoints[-1][0]))
    actual = [outer for outer, _path in all_checkpoints]
    expected = list(range(1, expected_final + 1))
    if actual != expected:
        raise ValueError(f"Liu exact/E4 audit requires all outer checkpoints; got {actual}, expected {expected}")
    selected = [outer for outer, _path in checkpoints]
    if (checkpoint_subset and not allow_sparse_subset
            and selected != list(range(1, selected[-1] + 1))):
        raise ValueError(
            "a checkpoint subset must be the contiguous prefix 1..k because E4 for "
            "checkpoint k reuses the exact-map solve from checkpoint k-1"
        )
    final_path = weight_dir / "value_net_final.pt"
    last_path = weight_dir / "value_net_last.pt"
    missing_official = [str(path) for path in (final_path, last_path) if not path.is_file()]
    if missing_official:
        raise FileNotFoundError(f"missing official terminal checkpoint(s): {missing_official}")
    terminal_hashes = {
        "final": canonical_tensor_state_hash(final_path),
        "last": canonical_tensor_state_hash(last_path),
        "last_iterate": canonical_tensor_state_hash(all_checkpoints[-1][1]),
    }
    if len(set(terminal_hashes.values())) != 1:
        raise ValueError(
            "value_net_final.pt, value_net_last.pt, and the final iterate do not "
            f"contain the same tensor state: {terminal_hashes}"
        )
    outer_index = _load_outer_index(run_dir / "outer_history.csv")
    if sorted(outer_index) != expected:
        raise ValueError("outer_history schedule does not match the complete checkpoint schedule")
    for outer in expected:
        row = outer_index[outer]
        if int(row.get("frozen_policy_iter", outer - 1)) != outer - 1:
            raise ValueError(f"outer={outer} has inconsistent frozen_policy_iter")
        if int(row.get("improved_policy_iter", outer)) != outer:
            raise ValueError(f"outer={outer} has inconsistent improved_policy_iter")

    margin = (float(eval_margin_override) if eval_margin_override is not None
              else _first_margin(_pick(args, ("eval_margin",), default="0.0")))
    evaluation_window = _resolve_evaluation_window(
        saved_w_min=w_min,
        saved_w_max=w_max,
        saved_x_min=x_min,
        saved_x_max=x_max,
        eval_margin=margin,
        eval_w_min_override=eval_w_min_override,
        eval_w_max_override=eval_w_max_override,
        eval_x_margin_override=eval_x_margin_override,
    )
    ev_w = (
        float(evaluation_window["ev_w_min"]),
        float(evaluation_window["ev_w_max"]),
    )
    ev_x = (
        float(evaluation_window["ev_x_min"]),
        float(evaluation_window["ev_x_max"]),
    )
    # ``seed`` records the network-training seed and is expected to differ
    # across repetitions.  Every actual market array (including market_seed)
    # participates in the cross-seed market identity.
    market_hash = canonical_array_hash({
        key: market[key] for key in sorted(market) if key != "seed"
    })
    training_args = _training_protocol_args(args)
    training_protocol_hash = stable_hash({"args": training_args})
    group_payload = {
        "problem": "liu", "n_assets": problem.n_assets, "m_states": 1,
        "market_sha256": market_hash, "gamma": problem.gamma,
        "horizon": problem.horizon,
        "evaluation_window": evaluation_window,
        "theta_clip_abs": _pick(args, ("theta_clip_abs",), default=None),
        "theta_init_method": _pick(args, ("theta_init_method",), default="myopic"),
        "theta_init_scale": float(_pick(args, ("theta_init_scale",), default=1.0)),
        "training_protocol_hash": training_protocol_hash,
        "vww_guard": VWW_GUARD,
    }
    return RunSpec(
        run_dir=run_dir, config_path=config_path, market_path=market_path,
        weight_dir=weight_dir, config=config, args=args, problem=problem,
        closed_form=closed_form, checkpoints=checkpoints,
        all_checkpoints=all_checkpoints, outer_index=outer_index, seed=seed,
        market_seed=market_seed, eval_margin=margin,
        eval_x_margin=float(evaluation_window["eval_x_margin"]),
        eval_w_min_override=evaluation_window["eval_w_min_override"],
        eval_w_max_override=evaluation_window["eval_w_max_override"],
        eval_w_bounds=ev_w, eval_x_bounds=ev_x,
        evaluation_window=evaluation_window, market_hash=market_hash,
        group=stable_hash(group_payload)[:12],
        checkpoint_selection=("explicit_subset" if checkpoint_subset else "all"),
        terminal_state_hash=terminal_hashes["final"],
        training_protocol_hash=training_protocol_hash,
        training_protocol_args=training_args,
    )


def _state_dict(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a state_dict: {path}")
    for key in ("state_dict", "model_state_dict", "value_net_state_dict", "model"):
        if key in payload and isinstance(payload[key], Mapping):
            payload = payload[key]
            break
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key)
        for prefix in ("module.", "value_net.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _natural_key(name: str) -> Tuple[Any, ...]:
    return tuple(int(item) if item.isdigit() else item for item in re.split(r"(\d+)", name))


def _policy_extension_coordinates(
    problem: LiuProblem,
    y: Array,
    x: Array,
    mode: str,
) -> Tuple[Array, Array, Dict[str, float]]:
    """Apply the declared finite-domain extension to policy query points.

    ``boundary-projection`` holds the normalized feedback constant outside the
    saved nominal collocation rectangle.  ``neural-extrapolation`` retains the
    raw network extrapolation only as an explicit sensitivity mode.
    """

    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized not in POLICY_EXTENSIONS:
        raise ValueError(
            f"unsupported policy extension {mode!r}; choose from {POLICY_EXTENSIONS}"
        )
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.shape != xx.shape:
        raise ValueError("policy y/x query arrays must have identical shapes")
    outside_y = (yy < problem.y_min) | (yy > problem.y_max)
    outside_x = (xx < problem.x_min) | (xx > problem.x_max)
    outside_any = outside_y | outside_x
    if normalized == "boundary-projection":
        eval_y = np.clip(yy, problem.y_min, problem.y_max)
        eval_x = np.clip(xx, problem.x_min, problem.x_max)
    else:
        eval_y = yy
        eval_x = xx
    diagnostics = {
        "points": float(yy.size),
        "outside_collocation_count": float(np.count_nonzero(outside_any)),
        "outside_collocation_y_count": float(np.count_nonzero(outside_y)),
        "outside_collocation_x_count": float(np.count_nonzero(outside_x)),
    }
    return eval_y, eval_x, diagnostics


class TorchCheckpointEvaluator:
    """Import-safe adapter for the fixed ``(w,x,tau)->V`` Liu MLP."""

    def __init__(self, checkpoint: Path, run: RunSpec, device: str) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for checkpoint evaluation") from exc
        self.torch = torch
        self.run = run
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        state = _state_dict(torch, checkpoint)
        weights = sorted(
            [(key, value) for key, value in state.items()
             if key.endswith(".weight") and getattr(value, "ndim", 0) == 2],
            key=lambda item: _natural_key(item[0]),
        )
        if len(weights) < 2:
            raise ValueError("could not infer Liu MLP layers")
        self.dtype = weights[0][1].dtype
        layers = []
        for key, weight in weights:
            bias_key = key[:-6] + "bias"
            if bias_key not in state:
                raise KeyError(f"missing bias for {key}")
            layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0])).to(dtype=weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(weight)
                layer.bias.copy_(state[bias_key])
            layers.append(layer)
        if layers[0].in_features != 3 or layers[-1].out_features != 1:
            raise ValueError("expected a three-input scalar-output Liu network")
        hidden = int(_pick(run.args, ("value_hidden",), default=layers[0].out_features))
        depth = int(_pick(run.args, ("value_depth",), default=len(layers) - 1))
        if len(layers) - 1 != depth or any(layer.out_features != hidden for layer in layers[:-1]):
            raise ValueError("checkpoint architecture disagrees with config")

        class MLP(nn.Module):
            def __init__(self, modules: Sequence[Any]) -> None:
                super().__init__()
                self.layers = nn.ModuleList(modules)

            def forward(self, inputs: Any) -> Any:
                value = inputs
                for layer in self.layers[:-1]:
                    value = torch.tanh(layer(value))
                return self.layers[-1](value)

        self.model = MLP(layers).to(device=self.device, dtype=self.dtype).eval()
        clip = _pick(run.args, ("theta_clip_abs",), default=None)
        self.theta_clip_abs = None if clip in (None, "", "none", "null") else float(clip)

    def bundle_at_points(self, tau: Array, y: Array, x: Array,
                         chunk: int = 32768) -> Tuple[Array, Array, Array, Array]:
        torch = self.torch
        t = np.asarray(tau, dtype=np.float64).reshape(-1)
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        if not (t.shape == yy.shape == xx.shape):
            raise ValueError("tau, y, x point arrays must have equal shape")
        outputs: List[List[Array]] = [[], [], [], []]
        for start in range(0, t.size, int(chunk)):
            stop = min(start + int(chunk), t.size)
            w_t = torch.as_tensor(np.exp(yy[start:stop, None]), device=self.device,
                                  dtype=self.dtype).requires_grad_(True)
            x_t = torch.as_tensor(xx[start:stop, None], device=self.device,
                                  dtype=self.dtype).requires_grad_(True)
            tau_t = torch.as_tensor(t[start:stop, None], device=self.device, dtype=self.dtype)
            value = self.model(torch.cat([w_t, x_t, tau_t], dim=1))
            value_w = torch.autograd.grad(value, w_t, torch.ones_like(value),
                                          create_graph=True, retain_graph=True)[0]
            value_ww = torch.autograd.grad(value_w, w_t, torch.ones_like(value_w),
                                           create_graph=False, retain_graph=True)[0]
            value_wx = torch.autograd.grad(value_w, x_t, torch.ones_like(value_w),
                                           create_graph=False)[0]
            for target, tensor in zip(outputs, (value, value_w, value_ww, value_wx)):
                target.append(tensor.detach().cpu().numpy().reshape(-1))
        return tuple(np.concatenate(items) for items in outputs)  # type: ignore[return-value]

    def bundle_on_tensor_grid(self, tau: Array, y: Array, x: Array) -> Tuple[Array, Array, Array, Array]:
        tt, yy, xx = np.meshgrid(np.asarray(tau), np.asarray(y), np.asarray(x), indexing="ij")
        shape = tt.shape
        return tuple(item.reshape(shape) for item in self.bundle_at_points(tt.ravel(), yy.ravel(), xx.ravel()))  # type: ignore[return-value]

    def _policy_arrays(self, tau: Array, y: Array, x: Array,
                       chunk: int = 32768,
                       policy_extension: str = "boundary-projection",
                       ) -> Tuple[Array, Dict[str, float]]:
        torch = self.torch
        t = np.asarray(tau, dtype=np.float64).reshape(-1)
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        if not (t.shape == yy.shape == xx.shape):
            raise ValueError("tau, y, x policy arrays must have equal shape")
        eval_y, eval_x, extension_diag = _policy_extension_coordinates(
            self.run.problem, yy, xx, policy_extension
        )
        result: List[Array] = []
        counts = {"guard_count": 0.0, "positive_curvature_count": 0.0,
                  "theta_any_clip_count": 0.0, "theta_component_clip_count": 0.0,
                  "outside_collocation_count": extension_diag["outside_collocation_count"],
                  "outside_collocation_y_count": extension_diag["outside_collocation_y_count"],
                  "outside_collocation_x_count": extension_diag["outside_collocation_x_count"]}
        for start in range(0, t.size, int(chunk)):
            stop = min(start + int(chunk), t.size)
            w_t = torch.as_tensor(np.exp(eval_y[start:stop, None]), device=self.device,
                                  dtype=self.dtype).requires_grad_(True)
            x_t = torch.as_tensor(eval_x[start:stop, None], device=self.device,
                                  dtype=self.dtype).requires_grad_(True)
            tau_t = torch.as_tensor(t[start:stop, None], device=self.device, dtype=self.dtype)
            value = self.model(torch.cat([w_t, x_t, tau_t], dim=1))
            value_w = torch.autograd.grad(value, w_t, torch.ones_like(value),
                                          create_graph=True, retain_graph=True)[0]
            value_ww = torch.autograd.grad(value_w, w_t, torch.ones_like(value_w),
                                           create_graph=False, retain_graph=True)[0]
            value_wx = torch.autograd.grad(value_w, x_t, torch.ones_like(value_w),
                                           create_graph=False)[0]
            lam0 = torch.as_tensor(self.run.problem.lam0, device=self.device, dtype=self.dtype)
            Lam = torch.as_tensor(self.run.problem.Lam, device=self.device, dtype=self.dtype)
            Gamma = torch.as_tensor(self.run.problem.Gamma, device=self.device, dtype=self.dtype)
            lam = lam0[None, :] + x_t * Lam[None, :]
            numerator = lam * value_w + value_wx * Gamma[None, :]
            guard = value_ww > -VWW_GUARD
            safe = torch.clamp(value_ww, max=-VWW_GUARD)
            raw_theta = -numerator / safe
            theta = raw_theta
            any_clip = torch.zeros(raw_theta.shape[0], device=self.device, dtype=torch.bool)
            component_clip = torch.zeros_like(raw_theta, dtype=torch.bool)
            if self.theta_clip_abs is not None:
                c = float(self.theta_clip_abs)
                component_clip = torch.abs(raw_theta) > c
                any_clip = torch.any(component_clip, dim=1)
                theta = torch.clamp(raw_theta, -c, c)
            result.append((theta / w_t).detach().cpu().numpy())
            counts["guard_count"] += float(guard.sum().item())
            counts["positive_curvature_count"] += float((value_ww >= 0.0).sum().item())
            counts["theta_any_clip_count"] += float(any_clip.sum().item())
            # Point-denominator scaling gives the componentwise rate after division by points.
            counts["theta_component_clip_count"] += float(component_clip.to(torch.float64).mean(dim=1).sum().item())
        vartheta = np.concatenate(result, axis=0)
        row_l2 = np.linalg.norm(vartheta, axis=1)
        diag = {"points": float(t.size), **counts,
                "vartheta_l2_min": float(np.min(row_l2)),
                "vartheta_l2_max": float(np.max(row_l2)),
                "vartheta_component_min": float(np.min(vartheta)),
                "vartheta_component_max": float(np.max(vartheta))}
        return vartheta, diag

    def precompute_policy(self, tau: Array, y: Array, x: Array,
                          chunk: int = 32768,
                          policy_extension: str = "boundary-projection",
                          ) -> Tuple[Any, str, Dict[str, float]]:
        tau_grid = np.asarray(tau, dtype=np.float64).reshape(-1)
        y_grid = np.asarray(y, dtype=np.float64).reshape(-1)
        x_grid = np.asarray(x, dtype=np.float64).reshape(-1)
        tt, yy, xx = np.meshgrid(tau_grid, y_grid, x_grid, indexing="ij")
        vartheta, diag = self._policy_arrays(
            tt.ravel(), yy.ravel(), xx.ravel(), chunk=chunk,
            policy_extension=policy_extension,
        )
        values = vartheta.reshape(tau_grid.size, y_grid.size, x_grid.size, -1)
        digest = hashlib.sha256()
        digest.update(str(policy_extension).encode("utf-8") + b"\0")
        for item in (tau_grid, y_grid, x_grid, values):
            digest.update(np.asarray(item, dtype="<f8").tobytes())

        def policy(time_value: float, query_y: Array, query_x: Array) -> Tuple[Array, Mapping[str, float]]:
            index = int(np.argmin(np.abs(tau_grid - float(time_value))))
            if not math.isclose(float(tau_grid[index]), float(time_value), rel_tol=0.0, abs_tol=1e-11):
                raise KeyError(f"tau={time_value} is not a precomputed policy node")
            if query_y.shape != (y_grid.size, x_grid.size) or query_x.shape != query_y.shape:
                raise ValueError("FD solver requested a different spatial policy grid")
            if not (np.allclose(query_y[:, 0], y_grid, rtol=0.0, atol=1e-12)
                    and np.allclose(query_x[0], x_grid, rtol=0.0, atol=1e-12)):
                raise ValueError("FD policy grid values do not match precomputation")
            n_space = float(y_grid.size * x_grid.size)
            per_time = {"points": n_space}
            # Counts are recomputed cheaply from the stored aggregate only for reporting;
            # solve-level fractions remain exact over the same tensor grid.
            for key in ("guard_count", "positive_curvature_count", "theta_any_clip_count",
                        "theta_component_clip_count", "outside_collocation_count",
                        "outside_collocation_y_count", "outside_collocation_x_count"):
                per_time[key] = float(diag[key]) / tau_grid.size
            selected = values[index]
            row_l2 = np.linalg.norm(selected, axis=-1)
            per_time.update({
                "vartheta_l2_min": float(np.min(row_l2)),
                "vartheta_l2_max": float(np.max(row_l2)),
                "vartheta_component_min": float(np.min(selected)),
                "vartheta_component_max": float(np.max(selected)),
            })
            return values[index], per_time

        return policy, digest.hexdigest(), diag

    def policy_diagnostics(
        self,
        tau: Array,
        y: Array,
        x: Array,
        policy_extension: str = "boundary-projection",
    ) -> Dict[str, float]:
        _policy, _digest, raw = self.precompute_policy(
            tau, y, x, policy_extension=policy_extension
        )
        points = max(float(raw["points"]), 1.0)
        return {
            "guard_frac": float(raw["guard_count"]) / points,
            "positive_curvature_frac": float(raw["positive_curvature_count"]) / points,
            "theta_any_clip_frac": float(raw["theta_any_clip_count"]) / points,
            "theta_component_clip_frac": float(raw["theta_component_clip_count"]) / points,
            "outside_collocation_frac": (
                float(raw["outside_collocation_count"]) / points
            ),
            "outside_collocation_y_frac": (
                float(raw["outside_collocation_y_count"]) / points
            ),
            "outside_collocation_x_frac": (
                float(raw["outside_collocation_x_count"]) / points
            ),
        }


def _initial_policy(
    run: RunSpec,
    tau_grid: Array,
    y_grid: Array,
    x_grid: Array,
    policy_extension: str = "boundary-projection",
) -> Tuple[Any, str]:
    method = str(_pick(run.args, ("theta_init_method",), default="myopic")).lower()
    scale = float(_pick(run.args, ("theta_init_scale",), default=1.0))
    clip_raw = _pick(run.args, ("theta_clip_abs",), default=None)
    clip = None if clip_raw in (None, "", "none", "null") else float(clip_raw)
    tt, yy, xx = np.meshgrid(tau_grid, y_grid, x_grid, indexing="ij")
    eval_yy, eval_xx, extension_diag = _policy_extension_coordinates(
        run.problem, yy, xx, policy_extension
    )
    if method == "myopic":
        values = scale * run.problem.risk_premium(eval_xx) / run.problem.gamma
    elif method == "zero":
        values = np.zeros((*eval_xx.shape, run.problem.n_assets), dtype=np.float64)
    elif method == "closed_form":
        values = scale * run.closed_form.optimal_vartheta(tt, eval_xx)
    else:
        raise ValueError(f"unsupported theta_init_method={method!r}")
    if clip is not None:
        wealth = np.exp(eval_yy)[..., None]
        values = np.clip(wealth * values, -clip, clip) / wealth
    digest_builder = hashlib.sha256()
    digest_builder.update(str(policy_extension).encode("utf-8") + b"\0")
    for item in (tau_grid, y_grid, x_grid, values):
        digest_builder.update(np.asarray(item, dtype="<f8").tobytes())
    digest = digest_builder.hexdigest()

    def policy(time_value: float, query_y: Array, query_x: Array) -> Tuple[Array, Mapping[str, float]]:
        index = int(np.argmin(np.abs(tau_grid - float(time_value))))
        if not math.isclose(float(tau_grid[index]), float(time_value), rel_tol=0.0, abs_tol=1e-11):
            raise KeyError("initial policy requested at an unregistered tau node")
        n_time = max(float(tau_grid.size), 1.0)
        return values[index], {
            "points": float(query_y.size),
            "outside_collocation_count": (
                float(extension_diag["outside_collocation_count"]) / n_time
            ),
            "outside_collocation_y_count": (
                float(extension_diag["outside_collocation_y_count"]) / n_time
            ),
            "outside_collocation_x_count": (
                float(extension_diag["outside_collocation_x_count"]) / n_time
            ),
        }

    return policy, digest


def _metric_to_row(prefix: str, metric: Mapping[str, float]) -> Dict[str, float]:
    rename = {
        "value_sup": "value",
        "vw_sup": "vw",
        "vww_sup": "vww",
        "vwx_sup": "vwx",
        "bundle_sup": "bundle",
        "x_norm": "X",
    }
    missing = sorted(set(rename) - set(metric))
    extra = sorted(set(metric) - set(rename))
    if missing or extra:
        raise ValueError(
            f"unexpected X-norm component schema: missing={missing}, extra={extra}"
        )
    return {
        f"{prefix}_{rename[key]}": float(metric[key])
        for key in rename
    }


def _policy_diag_from_fd(fd: Mapping[str, float]) -> Dict[str, float]:
    return {
        "guard": float(fd.get("policy_guard_count", 0.0)),
        "positive": float(fd.get("policy_positive_curvature_count", 0.0)),
        "any_clip": float(fd.get("policy_theta_any_clip_count", 0.0)),
        "component_clip": float(fd.get("policy_theta_component_clip_count", 0.0)),
        "l2_min": float(fd.get("policy_vartheta_l2_min", float("nan"))),
        "l2_max": float(fd.get("policy_vartheta_l2_max", float("nan"))),
        "component_min": float(fd.get("policy_vartheta_component_min", float("nan"))),
        "component_max": float(fd.get("policy_vartheta_component_max", float("nan"))),
        "outside": float(fd.get("policy_outside_collocation_count", 0.0)),
        "outside_y": float(fd.get("policy_outside_collocation_y_count", 0.0)),
        "outside_x": float(fd.get("policy_outside_collocation_x_count", 0.0)),
    }


def _map_variant(fd_policy: Mapping[str, float], ev: Mapping[str, float]) -> Tuple[str, int]:
    modified = any(float(fd_policy.get(key, 0.0)) > 0.0 or float(ev.get(key, 0.0)) > 0.0
                   for key in ("guard", "any_clip", "component_clip"))
    nonconcave = float(fd_policy.get("positive", 0.0)) > 0.0 or float(ev.get("positive", 0.0)) > 0.0
    if modified:
        return "sampled_guarded_clipped", 0
    if nonconcave:
        return "sampled_nonconcave_source_policy", 0
    return "locally_unmodified_on_sampled_xfd", 1


def _map_definition(policy_extension: str) -> Tuple[str, str]:
    normalized = str(policy_extension).strip().lower().replace("_", "-")
    if normalized == "boundary-projection":
        return (
            "finite_domain_boundary_projected_policy_extension",
            "not_a_whole_space_map",
        )
    if normalized == "neural-extrapolation":
        return (
            "finite_domain_raw_neural_extrapolation",
            "not_verified_by_finite_domain",
        )
    raise ValueError(f"unsupported policy extension {policy_extension!r}")


def _normalize_boundaries(boundaries: Sequence[str]) -> List[str]:
    normalized = [
        str(value).strip().replace("_", "-").lower()
        for value in boundaries
    ]
    if not normalized or any(value not in BOUNDARY_VARIANTS for value in normalized):
        raise ValueError(
            "unsupported boundary list; choose from "
            + ", ".join(BOUNDARY_VARIANTS)
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("boundary variants must be unique")
    return normalized


def _verification_set(
    checkpoints: Sequence[Tuple[int, Path]],
    spec: str,
    *,
    include_alpha0: bool = True,
) -> set[int]:
    outers = [outer for outer, _path in checkpoints]
    text = str(spec).strip().lower()
    if text == "all":
        return set(outers) | ({0} if include_alpha0 else set())
    if text in {"none", ""}:
        return set()
    out: set[int] = set()
    for token in re.split(r"[\s,]+", text):
        if token == "first": out.add(outers[0])
        elif token == "middle": out.add(outers[len(outers) // 2])
        elif token == "last": out.add(outers[-1])
        elif token: out.add(int(token))
    unknown = out - set(outers)
    if unknown:
        raise ValueError(f"verification checkpoints are unavailable: {sorted(unknown)}")
    return out | ({0} if out and include_alpha0 else set())


def _variant_schedule(
    outer: int,
    verification: set[int],
    *,
    finest: int,
    largest_wealth_domain: float,
    largest_factor_domain: float,
    primary_boundary: str,
    grid_factors: Sequence[int],
    domain_pairs: Sequence[Tuple[float, float]],
    boundaries: Sequence[str],
) -> List[Tuple[int, float, float, str]]:
    primary = (
        int(finest),
        float(largest_wealth_domain),
        float(largest_factor_domain),
        str(primary_boundary),
    )
    if int(outer) not in verification:
        return [primary]
    numerical_variants = [
        (int(factor), float(wealth_domain), float(factor_domain), str(boundary))
        for factor in sorted(set(grid_factors))
        for wealth_domain, factor_domain in domain_pairs
        for boundary in (primary_boundary,)
    ]
    boundary_variants = [
        (
            int(finest),
            float(largest_wealth_domain),
            float(largest_factor_domain),
            str(boundary),
        )
        for boundary in boundaries
        if str(boundary) != str(primary_boundary)
    ]
    return numerical_variants + boundary_variants


def _e4_source_outer(target_outer: int) -> int:
    target = int(target_outer)
    if target < 1:
        raise ValueError("E4 target outer iteration must be positive")
    return target - 1


def _assess(
    rows: List[Dict[str, Any]],
    *,
    key_name: str,
    value_name: str,
    finest: int,
    largest_wealth_domain: float,
    largest_factor_domain: float,
    primary_boundary: str,
    grid_factors: Sequence[int],
    domain_mode: str,
    domain_pairs: Sequence[Tuple[float, float]],
    boundaries: Sequence[str],
    envelope_name: str,
    abs_tolerance: float,
    rel_tolerance: float,
    refinement_rule: str = "cartesian",
) -> None:
    """Assess grid/domain refinement and report boundary-BVP sensitivity.

    Boundary replacement is deliberately excluded from ``refinement_status``:
    it changes the finite-domain boundary-value problem rather than refining
    the primary one.
    """
    if domain_mode not in {"coupled", "split"}:
        raise ValueError(f"unsupported domain mode {domain_mode!r}")
    refinement_rule = str(refinement_rule).strip().lower()
    if refinement_rule not in {"cartesian", "merton-axis"}:
        raise ValueError(
            "refinement_rule must be cartesian or merton-axis"
        )
    normalized_pairs = sorted({
        (float(wealth_domain), float(factor_domain))
        for wealth_domain, factor_domain in domain_pairs
    })
    primary_key = (
        int(finest),
        float(largest_wealth_domain),
        float(largest_factor_domain),
        str(primary_boundary),
    )
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row[key_name]), []).append(row)
    for _key, group in grouped.items():
        lookup = {
            (
                int(row["grid_factor"]),
                float(row["wealth_domain_factor"]),
                float(row["factor_domain_factor"]),
                str(row["boundary"]),
            ): row
            for row in group
        }
        primary = lookup.get(primary_key)
        if primary is None:
            continue
        value = float(primary[value_name])
        if not math.isfinite(value):
            primary.update(
                grid_abs_change=float("nan"), grid_rel_change=float("nan"),
                domain_abs_change=float("nan"), domain_rel_change=float("nan"),
                wealth_domain_abs_change=float("nan"),
                wealth_domain_rel_change=float("nan"),
                factor_domain_abs_change=float("nan"),
                factor_domain_rel_change=float("nan"),
                refinement_tolerance=float("nan"),
                numerical_abs_change=float("nan"),
                numerical_tolerance_ratio=float("nan"),
                boundary_abs_change=float("nan"),
                boundary_rel_change=float("nan"),
                boundary_tolerance_ratio=float("nan"),
                boundary_sensitivity_status="undefined_value",
                **{envelope_name: float("nan")},
                refinement_status="undefined_value",
            )
            if value_name == "rho_exact":
                primary["contraction_status"] = "undefined_denominator"
            continue
        if not int(primary.get("is_verification", 0)):
            primary.update(
                grid_abs_change="", grid_rel_change="",
                domain_abs_change="", domain_rel_change="",
                wealth_domain_abs_change="", wealth_domain_rel_change="",
                factor_domain_abs_change="", factor_domain_rel_change="",
                refinement_tolerance="", numerical_abs_change="",
                numerical_tolerance_ratio="", boundary_abs_change="",
                boundary_rel_change="", boundary_tolerance_ratio="",
                boundary_sensitivity_status="not_checked",
                **{envelope_name: ""},
                refinement_status="not_checked",
            )
            if value_name == "rho_exact":
                primary["contraction_status"] = (
                    "observed_below_one_without_sensitivity_pass" if value < 1.0
                    else "primary_not_below_one"
                )
            continue
        numerical_comparisons = {
            "grid": [
                lookup.get((
                    factor, largest_wealth_domain, largest_factor_domain,
                    primary_boundary,
                ))
                for factor in sorted(set(grid_factors)) if factor < finest
            ],
        }
        boundary_candidates = [
            lookup.get((
                finest, largest_wealth_domain, largest_factor_domain, boundary,
            ))
            for boundary in boundaries if boundary != primary_boundary
        ]
        if domain_mode == "coupled":
            numerical_comparisons["domain"] = [
                lookup.get((finest, wealth_domain, factor_domain, primary_boundary))
                for wealth_domain, factor_domain in normalized_pairs
                if (wealth_domain, factor_domain)
                != (largest_wealth_domain, largest_factor_domain)
            ]
        else:
            previous_wealth = max(
                (
                    wealth_domain
                    for wealth_domain, _factor_domain in normalized_pairs
                    if wealth_domain < largest_wealth_domain
                ),
                default=None,
            )
            previous_factor = max(
                (
                    factor_domain
                    for _wealth_domain, factor_domain in normalized_pairs
                    if factor_domain < largest_factor_domain
                ),
                default=None,
            )
            numerical_comparisons["wealth_domain"] = [
                lookup.get((
                    finest, wealth_domain, largest_factor_domain, primary_boundary,
                ))
                for wealth_domain, factor_domain in normalized_pairs
                if (factor_domain == largest_factor_domain
                    and (
                        wealth_domain < largest_wealth_domain
                        if refinement_rule == "merton-axis"
                        else wealth_domain == previous_wealth
                    ))
            ]
            numerical_comparisons["factor_domain"] = [
                lookup.get((
                    finest, largest_wealth_domain, factor_domain, primary_boundary,
                ))
                for wealth_domain, factor_domain in normalized_pairs
                if (wealth_domain == largest_wealth_domain
                    and (
                        factor_domain < largest_factor_domain
                        if refinement_rule == "merton-axis"
                        else factor_domain == previous_factor
                    ))
            ]
        tolerance = float(abs_tolerance) + float(rel_tolerance) * abs(value)
        changes: Dict[str, float] = {}
        axis_complete = True
        axis_passed = True
        axis_deltas: List[float] = []
        for prefix, candidates in numerical_comparisons.items():
            if not candidates or any(
                candidate is None or not math.isfinite(float(candidate[value_name]))
                for candidate in candidates
            ):
                axis_complete = False
                changes[prefix] = float("nan")
                continue
            deltas = [abs(value - float(candidate[value_name]))
                      for candidate in candidates if candidate is not None]
            changes[prefix] = max(deltas)
            axis_deltas.append(changes[prefix])
            axis_passed = axis_passed and all(
                delta <= tolerance for delta in deltas
            )
        numerical_expected_keys = {
            (
                int(factor), float(wealth_domain), float(factor_domain),
                str(primary_boundary),
            )
            for factor in sorted(set(grid_factors))
            for wealth_domain, factor_domain in normalized_pairs
        }
        full_candidates = [lookup.get(key) for key in numerical_expected_keys]
        cartesian_complete = all(
            candidate is not None
            and math.isfinite(float(candidate[value_name]))
            for candidate in full_candidates
        )
        full_deltas = (
            [
                abs(value - float(candidate[value_name]))
                for candidate in full_candidates
                if candidate is not None
            ]
            if cartesian_complete else []
        )
        cartesian_passed = (
            cartesian_complete
            and all(delta <= tolerance for delta in full_deltas)
        )
        scale = max(abs(value), 1e-300)
        primary["grid_abs_change"] = changes["grid"]
        primary["grid_rel_change"] = changes["grid"] / scale
        if domain_mode == "coupled":
            domain_change = changes["domain"]
            wealth_change: Any = ""
            factor_change: Any = ""
        else:
            wealth_change = changes["wealth_domain"]
            factor_change = changes["factor_domain"]
            domain_change = max(wealth_change, factor_change)
        primary["domain_abs_change"] = domain_change
        primary["domain_rel_change"] = domain_change / scale
        primary["wealth_domain_abs_change"] = wealth_change
        primary["wealth_domain_rel_change"] = (
            "" if wealth_change == "" else wealth_change / scale
        )
        primary["factor_domain_abs_change"] = factor_change
        primary["factor_domain_rel_change"] = (
            "" if factor_change == "" else factor_change / scale
        )
        if not boundary_candidates:
            boundary_change: Any = ""
            boundary_rel_change: Any = ""
            boundary_tolerance_ratio: Any = ""
            boundary_status = "not_applicable"
        elif any(
            candidate is None
            or not math.isfinite(float(candidate[value_name]))
            for candidate in boundary_candidates
        ):
            boundary_change = float("nan")
            boundary_rel_change = float("nan")
            boundary_tolerance_ratio = float("nan")
            boundary_status = "incomplete"
        else:
            boundary_deltas = [
                abs(value - float(candidate[value_name]))
                for candidate in boundary_candidates
                if candidate is not None
            ]
            boundary_change = max(boundary_deltas, default=0.0)
            boundary_rel_change = boundary_change / scale
            boundary_tolerance_ratio = (
                boundary_change / tolerance
                if tolerance > 0.0 else float("inf")
            )
            boundary_status = "reported"
        primary["boundary_abs_change"] = boundary_change
        primary["boundary_rel_change"] = boundary_rel_change
        primary["boundary_tolerance_ratio"] = boundary_tolerance_ratio
        primary["boundary_sensitivity_status"] = boundary_status
        if refinement_rule == "cartesian":
            complete = cartesian_complete and axis_complete
            passed = cartesian_passed
            envelope_delta = max(full_deltas, default=0.0)
            envelope = (
                value + envelope_delta
                if value_name == "rho_exact"
                else envelope_delta
            )
        else:
            complete = axis_complete
            passed = axis_passed
            envelope = value + sum(axis_deltas)
            envelope_delta = max(axis_deltas, default=0.0)
        primary["refinement_tolerance"] = tolerance
        primary["numerical_abs_change"] = envelope_delta
        primary["numerical_tolerance_ratio"] = (
            envelope_delta / tolerance
            if tolerance > 0.0 else float("inf")
        )
        if complete:
            # Both modes assess only grid/domain sensitivity for the primary
            # boundary. Boundary replacement solves a different finite-domain
            # BVP and is therefore reported above, never used as a refinement
            # pass/fail gate.
            primary[envelope_name] = envelope
            primary["refinement_status"] = "pass" if passed else "fail"
        else:
            primary[envelope_name] = float("nan")
            primary["refinement_status"] = "incomplete"
        if value_name == "rho_exact":
            prefix = ("" if int(primary.get("local_map_unmodified_on_xfd", 0))
                      else "sampled_modified_map_")
            if primary["refinement_status"] == "pass" and float(primary[envelope_name]) < 1.0:
                primary["contraction_status"] = prefix + "sensitivity_stable_below_one"
            elif primary["refinement_status"] == "pass":
                primary["contraction_status"] = prefix + "sensitivity_envelope_crosses_one"
            elif value < 1.0:
                primary["contraction_status"] = prefix + "observed_below_one_without_sensitivity_pass"
            else:
                primary["contraction_status"] = prefix + "primary_not_below_one"
    # Copy primary assessment fields into all variants only when they are the primary row.
    for row in rows:
        row.setdefault("grid_abs_change", "")
        row.setdefault("grid_rel_change", "")
        row.setdefault("domain_abs_change", "")
        row.setdefault("domain_rel_change", "")
        row.setdefault("wealth_domain_abs_change", "")
        row.setdefault("wealth_domain_rel_change", "")
        row.setdefault("factor_domain_abs_change", "")
        row.setdefault("factor_domain_rel_change", "")
        row.setdefault("refinement_tolerance", "")
        row.setdefault("numerical_abs_change", "")
        row.setdefault("numerical_tolerance_ratio", "")
        row.setdefault("boundary_abs_change", "")
        row.setdefault("boundary_rel_change", "")
        row.setdefault("boundary_tolerance_ratio", "")
        row.setdefault("boundary_sensitivity_status", "variant")
        row.setdefault(envelope_name, "")
        row.setdefault("refinement_status", "variant")
        if value_name == "rho_exact":
            row.setdefault("contraction_status", "variant")


def _build_fd_grid(
    problem: LiuProblem,
    *,
    base_ny: int,
    base_nx: int,
    base_nt: int,
    grid_factor: int,
    wealth_domain_factor: float,
    factor_domain_factor: float,
    wealth_y_bounds: Optional[Tuple[float, float]] = None,
) -> FDGrid:
    """Build a grid while preserving each coordinate's nominal spacing."""

    if wealth_y_bounds is None:
        y_center = 0.5 * (problem.y_min + problem.y_max)
        y_half = (
            0.5 * (problem.y_max - problem.y_min) * wealth_domain_factor
        )
        fd_y_min = y_center - y_half
        fd_y_max = y_center + y_half
        width_factor = float(wealth_domain_factor)
    else:
        fd_y_min, fd_y_max = (float(value) for value in wealth_y_bounds)
        if (
            not math.isfinite(fd_y_min)
            or not math.isfinite(fd_y_max)
            or not fd_y_min < problem.y_min
            or not fd_y_max > problem.y_max
        ):
            raise ValueError(
                "explicit FD wealth bounds must be finite and strictly contain "
                "the saved training log-wealth interval"
            )
        width_factor = (
            (fd_y_max - fd_y_min) / (problem.y_max - problem.y_min)
        )
        if not math.isclose(
            width_factor,
            float(wealth_domain_factor),
            rel_tol=1e-12,
            abs_tol=1e-13,
        ):
            raise ValueError(
                "explicit FD wealth bounds disagree with their log-width factor"
            )
    x_center = 0.5 * (problem.x_min + problem.x_max)
    x_half = (
        0.5 * (problem.x_max - problem.x_min) * factor_domain_factor
    )
    ny = max(
        7,
        int(round(width_factor * (base_ny - 1))) * grid_factor + 1,
    )
    nx = max(
        7,
        int(round(factor_domain_factor * (base_nx - 1))) * grid_factor + 1,
    )
    nt = int(base_nt) * grid_factor
    return FDGrid(
        fd_y_min, fd_y_max,
        x_center - x_half, x_center + x_half,
        ny, nx, nt,
    )


class _SingleKeyCache:
    """Retain only the current spatial-grid policy, reused across boundaries."""

    def __init__(self) -> None:
        self._key: Any = None
        self._value: Any = None
        self._defined = False

    @property
    def retained_key(self) -> Any:
        return self._key if self._defined else None

    def get_or_create(self, key: Any, factory: Callable[[], Any]) -> Any:
        if not self._defined or key != self._key:
            # Release the previous potentially multi-gigabyte closure before
            # allocating the next grid's precomputation.
            self._value = None
            self._key = None
            self._defined = False
            self._value = factory()
            self._key = key
            self._defined = True
        return self._value


class _PreviousMapBuffer:
    """One-outer streaming buffer for the shifted E4 comparison."""

    def __init__(self) -> None:
        self.outer: Optional[int] = None
        self.variants: Optional[Mapping[Any, Any]] = None

    def retain(self, outer: int, variants: Mapping[Any, Any]) -> None:
        self.outer = int(outer)
        self.variants = variants

    def consume(self, expected_outer: int) -> Mapping[Any, Any]:
        if self.variants is None or self.outer != int(expected_outer):
            raise RuntimeError(
                "streamed E4 source does not match the immediately preceding "
                f"exact-map iterate: expected {expected_outer}, retained {self.outer}"
            )
        variants = self.variants
        self.variants = None
        self.outer = None
        return variants


def required_e4_refinement_iterations(
    e4_rows: Sequence[Mapping[str, Any]],
    *,
    min_paper_checkpoint: int = 0,
) -> List[int]:
    """Return the Merton-style minimum E4 evidence set.

    Liu target 1 is the initial analytic-alpha0 defect.  It is required when
    eligible, followed by the first and last adjacent targets (targets above
    1) and the target with the largest primary ``e_approx_X``.  This is the
    shifted-index equivalent of Merton's initial + first/last adjacent +
    worst rule. Exact ties for the worst error are resolved in favor of the
    earlier target. A nonzero ``min_paper_checkpoint`` is an explicit
    initialization-transient floor; the default zero excludes nothing.
    """

    if isinstance(min_paper_checkpoint, bool) or not isinstance(
        min_paper_checkpoint, (int, np.integer)
    ):
        raise ValueError("min_paper_checkpoint must be a nonnegative integer")
    floor = int(min_paper_checkpoint)
    if floor < 0:
        raise ValueError("min_paper_checkpoint must be a nonnegative integer")
    by_iter: Dict[int, float] = {}
    seen: set[int] = set()
    for row in e4_rows:
        raw_target = row["target_outer_iter"]
        if isinstance(raw_target, bool):
            raise ValueError(
                f"invalid E4 target_outer_iter={raw_target!r}"
            )
        try:
            target = int(raw_target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid E4 target_outer_iter={raw_target!r}"
            ) from exc
        if isinstance(raw_target, (float, np.floating)) and (
            not math.isfinite(float(raw_target))
            or float(raw_target) != float(target)
        ):
            raise ValueError(
                f"invalid E4 target_outer_iter={raw_target!r}"
            )
        if target < 1:
            raise ValueError(
                f"invalid E4 target_outer_iter={target}; expected a positive integer"
            )
        value = float(row["e_approx_X"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"invalid e_approx_X={value!r} for target_outer_iter={target}"
            )
        if target in seen:
            raise ValueError(
                f"duplicate primary E4 target_outer_iter={target}"
            )
        seen.add(target)
        if target >= floor:
            by_iter[target] = value
    if not by_iter:
        return []
    schedule = sorted(by_iter)
    adjacent = [target for target in schedule if target > 1]
    required = {1} if 1 in by_iter else set()
    if adjacent:
        required.update((adjacent[0], adjacent[-1]))
    worst = max(by_iter, key=lambda value: (by_iter[value], -value))
    required.add(worst)
    return sorted(required)


def summarize_e4_refinement(
    e4_rows: Sequence[Mapping[str, Any]],
    *,
    min_paper_checkpoint: int = 0,
) -> Dict[str, Any]:
    """Summarize the Merton-style audited E4 refinement evidence."""

    required = required_e4_refinement_iterations(
        e4_rows, min_paper_checkpoint=min_paper_checkpoint
    )
    by_iter = {
        int(row["target_outer_iter"]): str(row.get("refinement_status", ""))
        for row in e4_rows
    }
    statuses = {
        str(target): by_iter.get(target, "missing") for target in required
    }
    if required and all(value == "pass" for value in statuses.values()):
        evidence_status = "pass"
    elif any(value == "fail" for value in statuses.values()):
        evidence_status = "fail"
    else:
        evidence_status = "incomplete"
    return {
        "required_iterations": required,
        "required_statuses": statuses,
        "evidence_status": evidence_status,
    }


def _paper_aggregation_eligible(
    *,
    skip_e4: bool,
    checkpoint_selection: str,
    primary_rows: Sequence[Mapping[str, Any]],
    primary_e4: Sequence[Mapping[str, Any]],
    min_paper_checkpoint: int = 0,
) -> bool:
    """Return the combined exact-map/E4 paper-evidence eligibility flag."""

    e4_summary = summarize_e4_refinement(
        primary_e4, min_paper_checkpoint=min_paper_checkpoint
    )
    return bool(
        not skip_e4
        and checkpoint_selection == "all"
        and primary_rows
        and primary_e4
        and all(int(row["denominator_defined"]) == 1 for row in primary_rows)
        and all(
            str(row.get("refinement_status", "")) == "pass"
            for row in primary_rows
        )
        and e4_summary["evidence_status"] == "pass"
    )


def evaluate_run(run: RunSpec, output: Path, *, device: str,
                 base_ny: int, base_nx: int, base_nt: int,
                 eval_ny: int, eval_nx: int, grid_factors: Sequence[int],
                 domain_mode: str,
                 wealth_domain_factors: Sequence[float],
                 factor_domain_factors: Sequence[float],
                 domain_pairs: Sequence[Tuple[float, float]],
                 boundaries: Sequence[str],
                 verify_checkpoints: str, drift_scheme: str, peclet_limit: float,
                 theta_method: float, startup_be_steps: int,
                 denominator_tolerance: float, refinement_abs_tolerance: float,
                 refinement_rel_tolerance: float,
                 ellipticity_tolerance: float,
                 linear_residual_tolerance: float = 1.0e-8,
                 boundary_condition_limit: float = 1.0e12,
                 policy_extension: str = "boundary-projection",
                 skip_e4: bool = False,
                 wealth_domain_parameterization: str = (
                     "symmetric_log_half_width_factor"
                 ),
                 wealth_domain_bounds: Optional[
                     Sequence[Mapping[str, float]]
                 ] = None,
                 requested_fd_w_mins: Optional[Sequence[float]] = None,
                 requested_fd_w_maxs: Optional[Sequence[float]] = None,
                 refinement_rule: str = "cartesian",
                 min_paper_checkpoint: int = 0,
                 overwrite: bool = False,
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output = output.expanduser().resolve()
    _prepare_output(output, overwrite)
    for name in ("_SUCCESS_EXACT_MAP", "_FAILED_EXACT_MAP"):
        path = output / name
        if path.exists(): path.unlink()
    if min(base_ny, base_nx) < 7 or base_nt < 2 or min(eval_ny, eval_nx) < 3:
        raise ValueError("FD/evaluation grids are too small")
    if not grid_factors or any(int(value) < 1 for value in grid_factors):
        raise ValueError("grid factors must be positive")
    normalized_grid_factors = [int(value) for value in grid_factors]
    if len(normalized_grid_factors) != len(set(normalized_grid_factors)):
        raise ValueError("grid factors must be unique")
    grid_factors = sorted(normalized_grid_factors)
    if isinstance(min_paper_checkpoint, bool) or not isinstance(
        min_paper_checkpoint, (int, np.integer)
    ):
        raise ValueError("min_paper_checkpoint must be a nonnegative integer")
    min_paper_checkpoint = int(min_paper_checkpoint)
    if min_paper_checkpoint < 0:
        raise ValueError("min_paper_checkpoint must be a nonnegative integer")
    refinement_rule = str(refinement_rule).strip().lower()
    if refinement_rule not in {"cartesian", "merton-axis"}:
        raise ValueError(
            "refinement_rule must be cartesian or merton-axis"
        )
    checkpoint_schedule = [int(outer) for outer, _path in run.checkpoints]
    paper_checkpoint_schedule = [
        outer for outer in checkpoint_schedule
        if outer >= min_paper_checkpoint
    ]
    if not skip_e4 and not paper_checkpoint_schedule:
        raise ValueError(
            "min_paper_checkpoint excludes every available E4 target"
        )
    excluded_initial_checkpoints = [
        outer for outer in checkpoint_schedule
        if outer < min_paper_checkpoint
    ]
    domain_mode = str(domain_mode).strip().lower()
    wealth_domain_factors = sorted(set(float(value) for value in wealth_domain_factors))
    factor_domain_factors = sorted(set(float(value) for value in factor_domain_factors))
    domain_pairs = sorted(set(
        (float(wealth_domain), float(factor_domain))
        for wealth_domain, factor_domain in domain_pairs
    ))
    if domain_mode not in {"coupled", "split"}:
        raise ValueError("domain mode must be coupled or split")
    if (not wealth_domain_factors or not factor_domain_factors
            or any(not math.isfinite(value) or value <= 1.0
                   for value in wealth_domain_factors + factor_domain_factors)):
        raise ValueError("wealth/factor domain factors must be finite and strictly larger than one")
    expected_pairs = (
        [(value, value) for value in wealth_domain_factors]
        if domain_mode == "coupled"
        else [
            (wealth_domain, factor_domain)
            for wealth_domain in wealth_domain_factors
            for factor_domain in factor_domain_factors
        ]
    )
    if domain_mode == "coupled" and wealth_domain_factors != factor_domain_factors:
        raise ValueError("coupled domain mode requires identical wealth/factor factor lists")
    if domain_pairs != sorted(set(expected_pairs)):
        raise ValueError(
            f"{domain_mode} domain pairs disagree with the declared factor lists"
        )
    wealth_domain_parameterization = (
        str(wealth_domain_parameterization).strip().lower()
    )
    if wealth_domain_parameterization not in {
        "symmetric_log_half_width_factor",
        "explicit_absolute_bounds",
    }:
        raise ValueError(
            "wealth domain parameterization must be "
            "symmetric_log_half_width_factor or explicit_absolute_bounds"
        )
    requested_fd_w_mins = [
        float(value) for value in (requested_fd_w_mins or [])
    ]
    requested_fd_w_maxs = [
        float(value) for value in (requested_fd_w_maxs or [])
    ]
    if wealth_domain_parameterization == "explicit_absolute_bounds":
        if domain_mode != "split":
            raise ValueError("explicit absolute wealth bounds require split domain mode")
        raw_bounds = list(wealth_domain_bounds or [])
        if (
            len(raw_bounds) != len(wealth_domain_factors)
            or len(requested_fd_w_mins) != len(raw_bounds)
            or len(requested_fd_w_maxs) != len(raw_bounds)
        ):
            raise ValueError(
                "explicit FD wealth bounds disagree with the declared wealth "
                "domain schedule"
            )
        canonical_wealth_bounds: List[Dict[str, float]] = []
        for expected_factor, raw, requested_min, requested_max in zip(
            wealth_domain_factors,
            raw_bounds,
            requested_fd_w_mins,
            requested_fd_w_maxs,
        ):
            if not isinstance(raw, Mapping):
                raise ValueError("each explicit FD wealth bound must be an object")
            record = {
                field: float(raw[field])
                for field in (
                    "wealth_domain_factor",
                    "fd_y_min",
                    "fd_y_max",
                    "fd_w_min",
                    "fd_w_max",
                )
            }
            if not math.isclose(
                record["wealth_domain_factor"],
                expected_factor,
                rel_tol=1e-12,
                abs_tol=1e-13,
            ):
                raise ValueError(
                    "explicit FD wealth bound has the wrong log-width factor"
                )
            if (
                not math.isclose(
                    record["fd_w_min"], requested_min,
                    rel_tol=1e-12, abs_tol=1e-13,
                )
                or not math.isclose(
                    record["fd_w_max"], requested_max,
                    rel_tol=1e-12, abs_tol=1e-13,
                )
                or not math.isclose(
                    record["fd_y_min"], math.log(record["fd_w_min"]),
                    rel_tol=1e-12, abs_tol=1e-13,
                )
                or not math.isclose(
                    record["fd_y_max"], math.log(record["fd_w_max"]),
                    rel_tol=1e-12, abs_tol=1e-13,
                )
                or not record["fd_y_min"] < run.problem.y_min
                or not record["fd_y_max"] > run.problem.y_max
            ):
                raise ValueError(
                    "explicit FD wealth bounds are inconsistent or do not "
                    "strictly contain the saved training domain"
                )
            canonical_wealth_bounds.append(record)
    else:
        if wealth_domain_bounds or requested_fd_w_mins or requested_fd_w_maxs:
            raise ValueError(
                "factor-based wealth domains cannot carry explicit FD endpoints"
            )
        y_center = 0.5 * (run.problem.y_min + run.problem.y_max)
        y_half_base = 0.5 * (run.problem.y_max - run.problem.y_min)
        canonical_wealth_bounds = []
        for value in wealth_domain_factors:
            fd_y_min = y_center - y_half_base * value
            fd_y_max = y_center + y_half_base * value
            canonical_wealth_bounds.append({
                "wealth_domain_factor": value,
                "fd_y_min": fd_y_min,
                "fd_y_max": fd_y_max,
                "fd_w_min": math.exp(fd_y_min),
                "fd_w_max": math.exp(fd_y_max),
            })
    wealth_bounds_by_factor = {
        float(item["wealth_domain_factor"]): (
            float(item["fd_y_min"]),
            float(item["fd_y_max"]),
        )
        for item in canonical_wealth_bounds
    }
    if set(wealth_bounds_by_factor) != set(wealth_domain_factors):
        raise ValueError("FD wealth-bound schedule has duplicate or missing factors")
    if (
        not math.isfinite(refinement_abs_tolerance)
        or not math.isfinite(refinement_rel_tolerance)
        or refinement_abs_tolerance < 0.0
        or refinement_rel_tolerance < 0.0
    ):
        raise ValueError("refinement tolerances must be finite and nonnegative")
    if (
        not math.isfinite(denominator_tolerance)
        or denominator_tolerance < 0.0
    ):
        raise ValueError(
            "denominator tolerance must be finite and nonnegative"
        )
    if (
        not math.isfinite(ellipticity_tolerance)
        or ellipticity_tolerance < 0.0
    ):
        raise ValueError(
            "ellipticity tolerance must be finite and nonnegative"
        )
    if not math.isfinite(linear_residual_tolerance) or linear_residual_tolerance <= 0.0:
        raise ValueError("linear residual tolerance must be finite and positive")
    if not math.isfinite(boundary_condition_limit) or boundary_condition_limit < 1.0:
        raise ValueError("boundary condition limit must be finite and at least one")
    policy_extension = str(policy_extension).strip().lower().replace("_", "-")
    if policy_extension not in POLICY_EXTENSIONS:
        raise ValueError(f"unsupported policy extension {policy_extension!r}")
    if skip_e4 and run.checkpoint_selection == "all":
        # Supported, but make the exploratory nature explicit in provenance.
        analysis_mode = "exact_map_only_pilot"
    elif skip_e4:
        analysis_mode = "sparse_exact_map_only_pilot"
    elif run.checkpoint_selection != "all":
        analysis_mode = "contiguous_prefix_exact_map_and_e4_pilot"
    else:
        analysis_mode = "exact_map_and_e4"
    map_definition, whole_space_claim = _map_definition(policy_extension)
    boundaries = _normalize_boundaries(boundaries)

    implementation_hashes = {
        "driver": sha256_file(Path(__file__).resolve()),
        "core": sha256_file(Path(__file__).with_name("liu_exact_map_core.py").resolve()),
    }
    protocol_hash = stable_hash({
        "group": run.group, "base_ny": base_ny, "base_nx": base_nx,
        "base_nt": base_nt, "eval_ny": eval_ny, "eval_nx": eval_nx,
        "grid_factors": list(grid_factors),
        "domain_mode": domain_mode,
        "wealth_domain_factors": wealth_domain_factors,
        "factor_domain_factors": factor_domain_factors,
        "wealth_domain_parameterization": wealth_domain_parameterization,
        "wealth_domain_bounds": canonical_wealth_bounds,
        "requested_fd_w_mins": requested_fd_w_mins,
        "requested_fd_w_maxs": requested_fd_w_maxs,
        "domain_pairs": [
            {
                "wealth_domain_factor": wealth_domain,
                "factor_domain_factor": factor_domain,
            }
            for wealth_domain, factor_domain in domain_pairs
        ],
        "boundaries": boundaries, "verify": verify_checkpoints,
        "selected_checkpoints": [outer for outer, _path in run.checkpoints],
        "refinement_rule": refinement_rule,
        "refinement_scope": REFINEMENT_SCOPE,
        "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
        "min_paper_checkpoint": min_paper_checkpoint,
        "paper_checkpoint_schedule": paper_checkpoint_schedule,
        "analysis_mode": analysis_mode, "skip_e4": bool(skip_e4),
        "policy_extension": policy_extension,
        "evaluation_window": dict(run.evaluation_window),
        "drift_scheme": drift_scheme, "peclet_limit": peclet_limit,
        "theta_method": theta_method, "startup_be_steps": startup_be_steps,
        "denominator_tolerance": denominator_tolerance,
        "refinement_abs_tolerance": refinement_abs_tolerance,
        "refinement_rel_tolerance": refinement_rel_tolerance,
        "ellipticity_tolerance": ellipticity_tolerance,
        "linear_residual_tolerance": linear_residual_tolerance,
        "boundary_condition_limit": boundary_condition_limit,
        "implementation_hashes": implementation_hashes,
        "checkpoint_selection": run.checkpoint_selection,
    })[:16]
    finest = max(int(value) for value in grid_factors)
    largest_wealth_domain = max(wealth_domain_factors)
    largest_factor_domain = max(factor_domain_factors)
    primary_boundary = boundaries[0]
    primary_domain_pair = (largest_wealth_domain, largest_factor_domain)
    if primary_domain_pair not in domain_pairs:
        raise ValueError(
            "the primary pair (maximum wealth factor, maximum factor factor) "
            "is not in the domain schedule"
        )
    domain_design = {
        "mode": domain_mode,
        "legacy_shared_shorthand": bool(domain_mode == "coupled"),
        "wealth_domain_factors": wealth_domain_factors,
        "factor_domain_factors": factor_domain_factors,
        "domain_pairs": [
            {
                "wealth_domain_factor": wealth_domain,
                "factor_domain_factor": factor_domain,
            }
            for wealth_domain, factor_domain in domain_pairs
        ],
        "primary_wealth_domain_factor": largest_wealth_domain,
        "primary_factor_domain_factor": largest_factor_domain,
        "wealth_domain_parameterization": wealth_domain_parameterization,
        "wealth_domain_bounds": canonical_wealth_bounds,
        "requested_fd_w_mins": requested_fd_w_mins,
        "requested_fd_w_maxs": requested_fd_w_maxs,
        "wealth_grid_size_rule": (
            "ny=(round(log_width_ratio*(base_ny-1))*grid_factor)+1"
        ),
        "definition": (
            "explicit absolute wealth bounds with log-width-ratio labels; "
            "factor factors independently expand the saved state-factor "
            "half-width"
            if wealth_domain_parameterization == "explicit_absolute_bounds"
            else "wealth factors expand the saved log-wealth half-width; "
            "factor factors independently expand the saved state-factor "
            "half-width"
        ),
    }
    verification = _verification_set(
        run.checkpoints, verify_checkpoints, include_alpha0=not skip_e4
    )

    ev_w_min, ev_w_max = run.eval_w_bounds
    ev_x_min, ev_x_max = run.eval_x_bounds
    ev_y = np.linspace(math.log(ev_w_min), math.log(ev_w_max), int(eval_ny))
    ev_x = np.linspace(ev_x_min, ev_x_max, int(eval_nx))
    ev_tau = np.linspace(0.0, run.problem.horizon, int(base_nt) + 1)[1:]
    tt, yy, xx = np.meshgrid(ev_tau, ev_y, ev_x, indexing="ij")
    reference = run.closed_form.wealth_bundle(tt, yy, xx)

    checkpoint_hashes: Dict[int, str] = {}
    rows: List[Dict[str, Any]] = []
    e4_rows: List[Dict[str, Any]] = []
    previous_map_buffer = _PreviousMapBuffer()

    def grid_spec(
        factor: int,
        wealth_domain_factor: float,
        factor_domain_factor: float,
    ) -> Tuple[FDGrid, Array, Array, Array]:
        grid = _build_fd_grid(
            run.problem,
            base_ny=base_ny,
            base_nx=base_nx,
            base_nt=base_nt,
            grid_factor=factor,
            wealth_domain_factor=wealth_domain_factor,
            factor_domain_factor=factor_domain_factor,
            wealth_y_bounds=wealth_bounds_by_factor[wealth_domain_factor],
        )
        y_grid = np.linspace(grid.y_min, grid.y_max, grid.ny)
        x_grid = np.linspace(grid.x_min, grid.x_max, grid.nx)
        tau_mid = (np.arange(grid.nt, dtype=np.float64) + 0.5) * run.problem.horizon / grid.nt
        return grid, tau_mid, y_grid, x_grid

    def append_e4_row(
        *,
        target_outer: int,
        checkpoint: Path,
        input_bundle: Tuple[Array, ...],
        variant: Tuple[int, float, float, str],
        fd_bundle: Tuple[Array, ...],
        source_policy_hash: str,
        source_diag: Mapping[str, Any],
    ) -> None:
        factor, wealth_domain_factor, factor_domain_factor, boundary = variant
        source_outer = _e4_source_outer(target_outer)
        grid, _tau_mid, _y_grid, _x_grid = grid_spec(
            factor, wealth_domain_factor, factor_domain_factor
        )
        metric = x_norm_components(*input_bundle, fd_bundle)
        primary = variant == (
            finest, largest_wealth_domain, largest_factor_domain,
            primary_boundary,
        )
        e4_rows.append({
            "problem": "liu", "group": run.group, "protocol_hash": protocol_hash,
            "model_type": "pipinn", "n_assets": run.problem.n_assets, "m_states": 1,
            "seed": run.seed, "market_seed": run.market_seed,
            "target_outer_iter": target_outer, "frozen_policy_iter": source_outer,
            "policy_source_outer_iter": source_outer,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hashes[target_outer],
            "market_sha256": run.market_hash,
            "domain_mode": domain_mode,
            "domain_factor": (
                wealth_domain_factor if domain_mode == "coupled" else ""
            ),
            "wealth_domain_factor": wealth_domain_factor,
            "factor_domain_factor": factor_domain_factor,
            "source_policy_hash": source_policy_hash,
            "fd_reference_source": (
                "analytic_alpha0_fd_solve" if source_outer == 0
                else f"reused_exact_map_source_outer_{source_outer}"
            ),
            "boundary": boundary, "grid_factor": factor,
            "ny": grid.ny, "nx": grid.nx, "nt": grid.nt,
            "fd_y_min": grid.y_min, "fd_y_max": grid.y_max,
            "fd_w_min": math.exp(grid.y_min), "fd_w_max": math.exp(grid.y_max),
            "fd_x_min": grid.x_min, "fd_x_max": grid.x_max,
            "is_primary": int(primary),
            "is_verification": int(source_outer in verification),
            "eval_margin": run.eval_margin,
            "eval_x_margin": run.eval_x_margin,
            "eval_w_min_override": run.eval_w_min_override,
            "eval_w_max_override": run.eval_w_max_override,
            "ev_w_min": ev_w_min,
            "ev_w_max": ev_w_max,
            "ev_x_min": ev_x_min,
            "ev_x_max": ev_x_max,
            "analysis_mode": analysis_mode,
            "refinement_rule": refinement_rule,
            "refinement_scope": REFINEMENT_SCOPE,
            "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
            "min_paper_checkpoint": min_paper_checkpoint,
            "policy_extension": policy_extension,
            "map_definition": map_definition,
            **_metric_to_row("e_approx", metric),
            "support_status": "defined",
            "source_min_log_joint_eig": source_diag["min_log_joint_eig"],
            "source_max_log_joint_eig": source_diag["max_log_joint_eig"],
            "source_min_original_joint_eig": source_diag[
                "min_original_joint_eig"
            ],
            "source_max_original_joint_eig": source_diag[
                "max_original_joint_eig"
            ],
            "source_nonpositive_log_eig_fraction": source_diag[
                "nonpositive_log_eig_fraction"
            ],
            "source_outside_collocation_fraction_fd": source_diag.get(
                "outside_collocation_fraction_fd",
                source_diag.get("policy_outside_collocation_count", 0.0),
            ),
        })

    for position, (outer, checkpoint) in enumerate(run.checkpoints, start=1):
        print(f"[liu exact-map] seed={run.seed} checkpoint={position}/{len(run.checkpoints)} outer={outer}")
        evaluator = TorchCheckpointEvaluator(checkpoint, run, device)
        checkpoint_hashes[outer] = sha256_file(checkpoint)
        input_bundle = evaluator.bundle_on_tensor_grid(ev_tau, ev_y, ev_x)
        input_metric = x_norm_components(*input_bundle, reference)
        if not skip_e4:
            source_outer = _e4_source_outer(outer)
            if source_outer == 0:
                initial_policy_cache = _SingleKeyCache()
                for (
                    factor, wealth_domain_factor, factor_domain_factor, boundary
                ) in _variant_schedule(
                    0, verification, finest=finest,
                    largest_wealth_domain=largest_wealth_domain,
                    largest_factor_domain=largest_factor_domain,
                    primary_boundary=primary_boundary,
                    grid_factors=grid_factors, domain_pairs=domain_pairs,
                    boundaries=boundaries,
                ):
                    grid, tau_mid, y_grid, x_grid = grid_spec(
                        factor, wealth_domain_factor, factor_domain_factor
                    )
                    cache_key = (
                        factor, wealth_domain_factor, factor_domain_factor
                    )
                    if initial_policy_cache.retained_key != cache_key:
                        init_policy = None
                        init_hash = None
                    init_policy, init_hash = initial_policy_cache.get_or_create(
                        cache_key,
                        lambda: _initial_policy(
                            run, tau_mid, y_grid, x_grid,
                            policy_extension=policy_extension,
                        ),
                    )
                    solution = solve_frozen_policy(
                        run.problem, init_policy, grid,
                        theta_method=theta_method,
                        startup_be_steps=startup_be_steps,
                        drift_scheme=drift_scheme,
                        peclet_limit=peclet_limit, boundary=boundary,
                        exact_boundary_value=(
                            lambda t, y, x: run.closed_form.value(t, y, x)
                        ),
                        ellipticity_tolerance=ellipticity_tolerance,
                        linear_residual_tolerance=linear_residual_tolerance,
                        boundary_condition_limit=boundary_condition_limit,
                    )
                    initial_fd_bundle = evaluate_fd_wealth_bundle(
                        solution, ev_tau, ev_y, ev_x
                    )
                    initial_source_diag = solution.diagnostics.as_dict()
                    solution = None
                    append_e4_row(
                        target_outer=outer, checkpoint=checkpoint,
                        input_bundle=input_bundle,
                        variant=(
                            factor, wealth_domain_factor,
                            factor_domain_factor, boundary,
                        ),
                        fd_bundle=initial_fd_bundle,
                        source_policy_hash=init_hash,
                        source_diag=initial_source_diag,
                    )
                    initial_fd_bundle = None
                    initial_source_diag = None
                init_policy = None
                init_hash = None
                initial_policy_cache = None
            else:
                source_variants = previous_map_buffer.consume(source_outer)
                for variant, (fd_bundle, source_row) in source_variants.items():
                    append_e4_row(
                        target_outer=outer, checkpoint=checkpoint,
                        input_bundle=input_bundle, variant=variant,
                        fd_bundle=fd_bundle,
                        source_policy_hash=str(source_row["policy_hash"]),
                        source_diag=source_row,
                    )
                del fd_bundle, source_row, source_variants
            # The source map has now been reduced to E4 scalar rows. Drop all
            # previous evaluation bundles before constructing the current map.
        ev_policy = evaluator.policy_diagnostics(
            ev_tau, ev_y, ev_x, policy_extension=policy_extension
        )
        ev_variant_diag = {"guard": ev_policy["guard_frac"],
                           "positive": ev_policy["positive_curvature_frac"],
                           "any_clip": ev_policy["theta_any_clip_frac"],
                           "component_clip": ev_policy["theta_component_clip_frac"]}
        current_map_variants: Dict[
            Tuple[int, float, float, str],
            Tuple[Tuple[Array, ...], Dict[str, Any]],
        ] = {}
        policy_cache = _SingleKeyCache()
        for factor, wealth_domain_factor, factor_domain_factor, boundary in _variant_schedule(
            outer, verification, finest=finest,
            largest_wealth_domain=largest_wealth_domain,
            largest_factor_domain=largest_factor_domain,
            primary_boundary=primary_boundary, grid_factors=grid_factors,
            domain_pairs=domain_pairs, boundaries=boundaries,
        ):
            grid, tau_mid, y_grid, x_grid = grid_spec(
                factor, wealth_domain_factor, factor_domain_factor
            )
            cache_key = (factor, wealth_domain_factor, factor_domain_factor)
            if policy_cache.retained_key != cache_key:
                policy = None
                policy_hash = None
            policy, policy_hash = policy_cache.get_or_create(
                cache_key,
                lambda: evaluator.precompute_policy(
                    tau_mid, y_grid, x_grid,
                    policy_extension=policy_extension,
                )[:2],
            )
            solution = solve_frozen_policy(
                run.problem, policy, grid, theta_method=theta_method,
                startup_be_steps=startup_be_steps, drift_scheme=drift_scheme,
                peclet_limit=peclet_limit, boundary=boundary,
                exact_boundary_value=(lambda t, y, x: run.closed_form.value(t, y, x)),
                ellipticity_tolerance=ellipticity_tolerance,
                linear_residual_tolerance=linear_residual_tolerance,
                boundary_condition_limit=boundary_condition_limit,
            )
            map_bundle = evaluate_fd_wealth_bundle(solution, ev_tau, ev_y, ev_x)
            map_metric = x_norm_components(*map_bundle, reference)
            fd_diag = solution.diagnostics.as_dict()
            solution = None
            fd_policy = _policy_diag_from_fd(fd_diag)
            map_variant, unmodified = _map_variant(fd_policy, ev_variant_diag)
            denominator = float(input_metric["x_norm"])
            denominator_defined = math.isfinite(denominator) and denominator > denominator_tolerance
            rho = float(map_metric["x_norm"] / denominator) if denominator_defined else float("nan")
            primary = (
                factor, wealth_domain_factor, factor_domain_factor, boundary
            ) == (
                finest, largest_wealth_domain, largest_factor_domain,
                primary_boundary,
            )
            row: Dict[str, Any] = {
                "problem": "liu", "group": run.group, "protocol_hash": protocol_hash,
                "model_type": "pipinn", "n_assets": run.problem.n_assets, "m_states": 1,
                "seed": run.seed, "market_seed": run.market_seed,
                "source_outer_iter": outer,
                "frozen_policy_iter": int(run.outer_index[outer].get("frozen_policy_iter", outer - 1)),
                "greedy_policy_iter": int(run.outer_index[outer].get("improved_policy_iter", outer)),
                "target_value_outer_iter": outer + 1,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hashes[outer],
                "market_sha256": run.market_hash, "network_dtype": str(evaluator.dtype).replace("torch.", ""),
                "eval_margin": run.eval_margin,
                "eval_x_margin": run.eval_x_margin,
                "eval_w_min_override": run.eval_w_min_override,
                "eval_w_max_override": run.eval_w_max_override,
                "ev_w_min": ev_w_min, "ev_w_max": ev_w_max,
                "ev_x_min": ev_x_min, "ev_x_max": ev_x_max,
                "domain_mode": domain_mode,
                "domain_factor": (
                    wealth_domain_factor if domain_mode == "coupled" else ""
                ),
                "wealth_domain_factor": wealth_domain_factor,
                "factor_domain_factor": factor_domain_factor,
                "fd_y_min": grid.y_min, "fd_y_max": grid.y_max,
                "fd_w_min": math.exp(grid.y_min), "fd_w_max": math.exp(grid.y_max),
                "fd_x_min": grid.x_min, "fd_x_max": grid.x_max,
                "boundary": boundary, "drift_scheme": drift_scheme, "grid_factor": factor,
                "ny": grid.ny, "nx": grid.nx, "nt": grid.nt,
                "dy": (grid.y_max-grid.y_min)/(grid.ny-1),
                "dx": (grid.x_max-grid.x_min)/(grid.nx-1), "dt": run.problem.horizon/grid.nt,
                "is_primary": int(primary), "is_verification": int(outer in verification),
                **_metric_to_row("e_input", input_metric), **_metric_to_row("e_map", map_metric),
                "rho_exact": rho, "denominator_defined": int(denominator_defined),
                "support_status": ("defined" if denominator_defined else "undefined_denominator"),
                "checkpoint_selection": run.checkpoint_selection,
                "analysis_mode": analysis_mode,
                "refinement_rule": refinement_rule,
                "refinement_scope": REFINEMENT_SCOPE,
                "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
                "min_paper_checkpoint": min_paper_checkpoint,
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "map_variant": map_variant,
                "local_map_unmodified_on_xfd": unmodified,
                "local_greedy_unmodified_on_policy_support": unmodified,
                "whole_space_map_claim": whole_space_claim,
                "outside_collocation_fraction_fd": fd_policy["outside"],
                "outside_collocation_y_fraction_fd": fd_policy["outside_y"],
                "outside_collocation_x_fraction_fd": fd_policy["outside_x"],
                "guard_frac_fd": fd_policy["guard"],
                "positive_curvature_frac_fd": fd_policy["positive"],
                "theta_any_clip_frac_fd": fd_policy["any_clip"],
                "theta_component_clip_frac_fd": fd_policy["component_clip"],
                "guard_frac_ev": ev_policy["guard_frac"],
                "positive_curvature_frac_ev": ev_policy["positive_curvature_frac"],
                "theta_any_clip_frac_ev": ev_policy["theta_any_clip_frac"],
                "theta_component_clip_frac_ev": ev_policy["theta_component_clip_frac"],
                "vartheta_l2_min_fd": fd_policy["l2_min"],
                "vartheta_l2_max_fd": fd_policy["l2_max"],
                "vartheta_component_min_fd": fd_policy["component_min"],
                "vartheta_component_max_fd": fd_policy["component_max"],
                "min_log_joint_eig": fd_diag["min_log_joint_eig"],
                "max_log_joint_eig": fd_diag["max_log_joint_eig"],
                "min_original_joint_eig": fd_diag["min_original_joint_eig"],
                "max_original_joint_eig": fd_diag["max_original_joint_eig"],
                "nonpositive_log_eig_fraction": fd_diag["nonpositive_log_eig_fraction"],
                "max_peclet_y": fd_diag["max_peclet_y"], "max_peclet_x": fd_diag["max_peclet_x"],
                "upwind_y_fraction": fd_diag["upwind_y_fraction"],
                "upwind_x_fraction": fd_diag["upwind_x_fraction"],
                "max_linear_residual": fd_diag["max_linear_residual"],
                "linear_residual_tolerance": linear_residual_tolerance,
                "boundary_elimination_size": fd_diag["boundary_elimination_size"],
                "boundary_elimination_rank": fd_diag["boundary_elimination_rank"],
                "boundary_elimination_cond_inf": fd_diag["boundary_elimination_cond_inf"],
                "min_linear_system_lu_pivot_ratio": fd_diag[
                    "min_linear_system_lu_pivot_ratio"
                ],
                "policy_hash": policy_hash,
            }
            rows.append(row)
            if not skip_e4 and position < len(run.checkpoints):
                current_map_variants[(
                    factor, wealth_domain_factor, factor_domain_factor, boundary
                )] = (map_bundle, row)
        if not skip_e4 and position < len(run.checkpoints):
            previous_map_buffer.retain(outer, current_map_variants)
        policy = None
        policy_hash = None
        policy_cache = None
        solution = None
        map_bundle = None

    _assess(
        rows, key_name="source_outer_iter", value_name="rho_exact",
        finest=finest,
        largest_wealth_domain=largest_wealth_domain,
        largest_factor_domain=largest_factor_domain,
        primary_boundary=primary_boundary, grid_factors=grid_factors,
        domain_mode=domain_mode, domain_pairs=domain_pairs,
        boundaries=boundaries, envelope_name="rho_sensitivity_envelope",
        abs_tolerance=refinement_abs_tolerance,
        rel_tolerance=refinement_rel_tolerance,
        refinement_rule=refinement_rule,
    )

    if not skip_e4:
        _assess(
            e4_rows, key_name="target_outer_iter", value_name="e_approx_X",
            finest=finest,
            largest_wealth_domain=largest_wealth_domain,
            largest_factor_domain=largest_factor_domain,
            primary_boundary=primary_boundary, grid_factors=grid_factors,
            domain_mode=domain_mode, domain_pairs=domain_pairs,
            boundaries=boundaries,
            envelope_name="approx_sensitivity_envelope",
            abs_tolerance=refinement_abs_tolerance,
            rel_tolerance=refinement_rel_tolerance,
            refinement_rule=refinement_rule,
        )

    primary_rows = [row for row in rows if int(row["is_primary"]) == 1]
    primary_e4 = [row for row in e4_rows if int(row["is_primary"]) == 1]
    e4_refinement_summary = summarize_e4_refinement(
        primary_e4, min_paper_checkpoint=min_paper_checkpoint
    )
    write_csv(output / "exact_map_refinement.csv", rows, RATIO_FIELDS)
    write_csv(output / "exact_map_ratios.csv", primary_rows, RATIO_FIELDS)
    write_csv(output / "e4_approximation_refinement.csv", e4_rows, E4_FIELDS)
    write_csv(output / "e4_approximation_errors.csv", primary_e4, E4_FIELDS)
    config_payload = {
        "run_dir": str(run.run_dir), "config_path": str(run.config_path),
        "config_sha256": sha256_file(run.config_path), "market_path": str(run.market_path),
        "market_sha256": run.market_hash, "market_file_sha256": sha256_file(run.market_path),
        "weight_dir": str(run.weight_dir), "terminal_state_hash": run.terminal_state_hash,
        "training_protocol_hash": run.training_protocol_hash,
        "training_protocol_args": run.training_protocol_args,
        "group": run.group, "protocol_hash": protocol_hash,
        "implementation_hashes": implementation_hashes,
        "checkpoint_selection": run.checkpoint_selection,
        "analysis_mode": analysis_mode,
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "whole_space_map_claim": whole_space_claim,
        "evaluation_window": dict(run.evaluation_window),
        "domain_design": domain_design,
        "collocation_bounds": {
            "y_min": run.problem.y_min, "y_max": run.problem.y_max,
            "w_min": math.exp(run.problem.y_min), "w_max": math.exp(run.problem.y_max),
            "x_min": run.problem.x_min, "x_max": run.problem.x_max,
            "interpretation": "saved nominal training bounds",
        },
        "checkpoint_schedule": checkpoint_schedule,
        "refinement_rule": refinement_rule,
        "refinement_scope": REFINEMENT_SCOPE,
        "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
        "primary_boundary": primary_boundary,
        "comparison_boundaries": [
            boundary
            for boundary in boundaries
            if boundary != primary_boundary
        ],
        "min_paper_checkpoint": min_paper_checkpoint,
        "excluded_initial_checkpoints": excluded_initial_checkpoints,
        "paper_checkpoint_schedule": paper_checkpoint_schedule,
        "e4_refinement_rule": {
            "required_set": "initial_first_last_worst_e_approx_X",
            "worst_tie_break": "lowest_target_outer_iter",
            "variant_pass_fail": refinement_rule,
            "interaction_failures": (
                "included" if refinement_rule == "cartesian" else "excluded"
            ),
            "sensitivity_axes": REFINEMENT_SCOPE,
            "boundary_replacement": BOUNDARY_SENSITIVITY_ROLE,
            "sensitivity_envelope": (
                "max_primary_boundary_cartesian_grid_domain_change"
                if refinement_rule == "cartesian"
                else "primary_plus_sum_grid_domain_axis_max_abs_changes"
            ),
        },
        "checkpoint_file_hashes": {str(outer): checkpoint_hashes[outer]
                                   for outer, _path in run.checkpoints},
        "grid": {"base_ny": base_ny, "base_nx": base_nx, "base_nt": base_nt,
                 "eval_ny": eval_ny, "eval_nx": eval_nx,
                 "grid_factors": list(grid_factors),
                 "domain_mode": domain_mode,
                 "domain_factors": (
                     wealth_domain_factors if domain_mode == "coupled" else []
                 ),
                 "wealth_domain_factors": wealth_domain_factors,
                 "factor_domain_factors": factor_domain_factors,
                 "wealth_domain_parameterization": (
                     wealth_domain_parameterization
                 ),
                 "wealth_domain_bounds": canonical_wealth_bounds,
                 "requested_fd_w_mins": requested_fd_w_mins,
                 "requested_fd_w_maxs": requested_fd_w_maxs,
                 "wealth_grid_size_rule": domain_design[
                     "wealth_grid_size_rule"
                 ],
                 "domain_pairs": domain_design["domain_pairs"],
                 "boundaries": boundaries, "verify_checkpoints": verify_checkpoints,
                 "drift_scheme": drift_scheme, "peclet_limit": peclet_limit,
                 "theta_method": theta_method, "startup_be_steps": startup_be_steps,
                 "linear_residual_tolerance": linear_residual_tolerance,
                 "boundary_condition_limit": boundary_condition_limit},
        "refinement_abs_tolerance": refinement_abs_tolerance,
        "refinement_rel_tolerance": refinement_rel_tolerance,
        "denominator_tolerance": denominator_tolerance,
        "ellipticity_tolerance": ellipticity_tolerance,
        "e4_status": ("skipped_by_exact_map_pilot" if skip_e4 else "computed"),
        "norm": "sup|V| + sup sqrt(Vw^2+Vww^2+Vwx^2)",
        "indexing": "checkpoint k ~= E(alpha[k-1]); map(checkpoint k)=E(alpha[k])",
    }
    atomic_json(output / "exact_map_config.json", config_payload)
    artifact_sha256 = {
        name: sha256_file(output / name)
        for name in (
            "exact_map_refinement.csv",
            "exact_map_ratios.csv",
            "e4_approximation_refinement.csv",
            "e4_approximation_errors.csv",
            "exact_map_config.json",
        )
    }
    status_payload = {
        "status": "success", "analysis_mode": analysis_mode,
        "paper_aggregation_eligible": _paper_aggregation_eligible(
            skip_e4=skip_e4,
            checkpoint_selection=run.checkpoint_selection,
            primary_rows=primary_rows,
            primary_e4=primary_e4,
            min_paper_checkpoint=min_paper_checkpoint,
        ),
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "evaluation_window": dict(run.evaluation_window),
        "domain_design": domain_design,
        "n_exact_rows": len(primary_rows),
        "n_refinement_rows": len(rows), "n_e4_rows": len(primary_e4),
        "n_e4_refinement_rows": len(e4_rows),
        "refinement_rule": refinement_rule,
        "refinement_scope": REFINEMENT_SCOPE,
        "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
        "primary_boundary": primary_boundary,
        "comparison_boundaries": [
            boundary
            for boundary in boundaries
            if boundary != primary_boundary
        ],
        "boundary_sensitivity_available": bool(len(boundaries) > 1),
        "exact_boundary_sensitivity_incomplete_outers": [
            int(row["source_outer_iter"])
            for row in primary_rows
            if str(row.get("boundary_sensitivity_status", ""))
            == "incomplete"
        ],
        "e4_boundary_sensitivity_incomplete_targets": [
            int(row["target_outer_iter"])
            for row in primary_e4
            if str(row.get("boundary_sensitivity_status", ""))
            == "incomplete"
        ],
        "min_paper_checkpoint": min_paper_checkpoint,
        "excluded_initial_checkpoints": excluded_initial_checkpoints,
        "paper_checkpoint_schedule": paper_checkpoint_schedule,
        "e4_refinement_required_iterations": (
            e4_refinement_summary["required_iterations"]
        ),
        "e4_refinement_required_statuses": (
            e4_refinement_summary["required_statuses"]
        ),
        "e4_refinement_evidence_status": (
            e4_refinement_summary["evidence_status"]
        ),
        "n_e4_refinement_pass": sum(
            str(row.get("refinement_status", "")) == "pass"
            for row in primary_e4
        ),
        "e4_status": ("skipped_by_exact_map_pilot" if skip_e4 else "computed"),
        "all_denominators_defined": all(int(row["denominator_defined"]) for row in primary_rows),
        "undefined_denominator_outers": [int(row["source_outer_iter"]) for row in primary_rows
                                         if not int(row["denominator_defined"])],
        "all_refinement_pass": all(row["refinement_status"] == "pass"
                                   for row in primary_rows + primary_e4),
        "exact_refinement_failures": [int(row["source_outer_iter"]) for row in primary_rows
                                      if row["refinement_status"] != "pass"],
        "e4_refinement_failures": [int(row["target_outer_iter"]) for row in primary_e4
                                   if row["refinement_status"] != "pass"],
        "all_e4_source_policies_elliptic": (
            None if skip_e4 else all(
                float(row["source_min_log_joint_eig"]) > ellipticity_tolerance
                and float(row["source_nonpositive_log_eig_fraction"]) == 0.0
                for row in primary_e4
            )
        ),
        "nonelliptic_e4_targets": [
            int(row["target_outer_iter"])
            for row in primary_e4
            if (float(row["source_min_log_joint_eig"]) <= ellipticity_tolerance
                or float(row["source_nonpositive_log_eig_fraction"]) != 0.0)
        ],
        "artifact_sha256": artifact_sha256,
    }
    atomic_json(output / "exact_map_status.json", status_payload)
    (output / "_SUCCESS_EXACT_MAP").touch()
    return primary_rows, primary_e4


def _parse_ints(text: str) -> List[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _resolve_domain_design(
    *,
    legacy: Optional[str],
    wealth: Optional[str],
    factor: Optional[str],
    fd_w_min: Optional[str] = None,
    fd_w_max: Optional[str] = None,
    problem: Optional[LiuProblem] = None,
) -> Dict[str, Any]:
    """Resolve factor-based or explicit absolute FD-domain schedules."""

    has_legacy = legacy is not None and str(legacy).strip() != ""
    has_wealth = wealth is not None and str(wealth).strip() != ""
    has_factor = factor is not None and str(factor).strip() != ""
    has_fd_w_min = fd_w_min is not None and str(fd_w_min).strip() != ""
    has_fd_w_max = fd_w_max is not None and str(fd_w_max).strip() != ""
    if has_fd_w_min != has_fd_w_max:
        raise ValueError("--fd-w-min and --fd-w-max must be provided together")
    has_absolute_wealth = has_fd_w_min and has_fd_w_max
    if has_absolute_wealth and (has_legacy or has_wealth):
        raise ValueError(
            "--fd-w-min/--fd-w-max cannot be combined with "
            "--domain-factors or --wealth-domain-factors"
        )
    if has_absolute_wealth and not has_factor:
        raise ValueError(
            "--factor-domain-factors must be provided with "
            "--fd-w-min/--fd-w-max"
        )
    if has_legacy and (has_wealth or has_factor):
        raise ValueError(
            "--domain-factors cannot be combined with "
            "--wealth-domain-factors/--factor-domain-factors"
        )
    if not has_absolute_wealth and has_wealth != has_factor:
        raise ValueError(
            "--wealth-domain-factors and --factor-domain-factors must be "
            "provided together"
        )

    if has_absolute_wealth:
        if problem is None:
            raise ValueError(
                "the saved training wealth bounds are required to resolve "
                "absolute FD wealth domains"
            )
        fd_w_mins = _parse_floats(str(fd_w_min))
        fd_w_maxs = _parse_floats(str(fd_w_max))
        if not fd_w_mins or len(fd_w_mins) != len(fd_w_maxs):
            raise ValueError(
                "--fd-w-min and --fd-w-max must contain equally many "
                "comma-separated values"
            )
        saved_w_min = math.exp(problem.y_min)
        saved_w_max = math.exp(problem.y_max)
        wealth_bounds: List[Dict[str, float]] = []
        previous_min = float("inf")
        previous_max = -float("inf")
        previous_factor = 1.0
        for position, (requested_min, requested_max) in enumerate(
            zip(fd_w_mins, fd_w_maxs),
            start=1,
        ):
            valid_positive = (
                math.isfinite(requested_min)
                and math.isfinite(requested_max)
                and requested_min > 0.0
                and requested_max > 0.0
            )
            fd_y_min = math.log(requested_min) if valid_positive else float("nan")
            fd_y_max = math.log(requested_max) if valid_positive else float("nan")
            strictly_below_saved = (
                fd_y_min < problem.y_min
                and not math.isclose(
                    fd_y_min, problem.y_min, rel_tol=1e-12, abs_tol=1e-13
                )
            )
            strictly_above_saved = (
                fd_y_max > problem.y_max
                and not math.isclose(
                    fd_y_max, problem.y_max, rel_tol=1e-12, abs_tol=1e-13
                )
            )
            if (
                not valid_positive
                or not strictly_below_saved
                or not strictly_above_saved
            ):
                raise ValueError(
                    "each explicit FD wealth interval must be finite, positive, "
                    "and strictly contain the saved training wealth interval "
                    f"[{saved_w_min:.17g}, {saved_w_max:.17g}]; invalid pair "
                    f"{position}=({requested_min!r}, {requested_max!r})"
                )
            if position > 1 and not (
                requested_min < previous_min and requested_max > previous_max
            ):
                raise ValueError(
                    "explicit FD wealth intervals must be listed from the "
                    "narrowest to the widest and form a strictly nested chain"
                )
            width_factor = (
                (fd_y_max - fd_y_min) / (problem.y_max - problem.y_min)
            )
            if not math.isfinite(width_factor) or not width_factor > previous_factor:
                raise ValueError(
                    "explicit FD wealth intervals must have strictly increasing "
                    "finite log-width factors larger than one"
                )
            wealth_bounds.append({
                "wealth_domain_factor": width_factor,
                "fd_y_min": fd_y_min,
                "fd_y_max": fd_y_max,
                "fd_w_min": requested_min,
                "fd_w_max": requested_max,
            })
            previous_min = requested_min
            previous_max = requested_max
            previous_factor = width_factor
        mode = "split"
        wealth_factors = [
            float(item["wealth_domain_factor"]) for item in wealth_bounds
        ]
        factor_factors = sorted(set(_parse_floats(str(factor))))
        pairs = [
            (wealth_factor, factor_factor)
            for wealth_factor in wealth_factors
            for factor_factor in factor_factors
        ]
        parameterization = "explicit_absolute_bounds"
    elif has_wealth:
        mode = "split"
        wealth_factors = sorted(set(_parse_floats(str(wealth))))
        factor_factors = sorted(set(_parse_floats(str(factor))))
        pairs = [
            (wealth_factor, factor_factor)
            for wealth_factor in wealth_factors
            for factor_factor in factor_factors
        ]
        wealth_bounds = []
        fd_w_mins = []
        fd_w_maxs = []
        parameterization = "symmetric_log_half_width_factor"
    else:
        mode = "coupled"
        shared = sorted(set(_parse_floats(
            str(legacy) if has_legacy else "1.5,2.0"
        )))
        wealth_factors = list(shared)
        factor_factors = list(shared)
        pairs = [(value, value) for value in shared]
        wealth_bounds = []
        fd_w_mins = []
        fd_w_maxs = []
        parameterization = "symmetric_log_half_width_factor"

    if (not wealth_factors or not factor_factors
            or any(not math.isfinite(value) or value <= 1.0
                   for value in wealth_factors + factor_factors)):
        raise ValueError(
            "all wealth/factor domain factors must be finite and strictly "
            "larger than one"
        )
    return {
        "mode": mode,
        "wealth_domain_factors": wealth_factors,
        "factor_domain_factors": factor_factors,
        "domain_pairs": pairs,
        "wealth_domain_parameterization": parameterization,
        "wealth_domain_bounds": wealth_bounds,
        "requested_fd_w_mins": fd_w_mins,
        "requested_fd_w_maxs": fd_w_maxs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Liu M=1 exact PI-map / E4 FD evaluator")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoints", default="",
        help=(
            "Exploratory comma-separated subset. Arbitrary sparse subsets require "
            "--skip-e4; paper mode uses all checkpoints."
        ),
    )
    parser.add_argument(
        "--skip-e4", action="store_true",
        help=(
            "Exact-map-only pilot mode. This permits sparse --checkpoints and "
            "does not produce paper-eligible E4 results."
        ),
    )
    parser.add_argument(
        "--min-paper-checkpoint",
        type=int,
        default=0,
        help=(
            "Exclude E4 targets below this checkpoint from paper evidence and "
            "paper E4 aggregation. Zero (default) excludes nothing; for example, "
            "use 2 only when checkpoint 1 is a documented initialization "
            "transient. Exact-map refinement remains strict at every checkpoint."
        ),
    )
    parser.add_argument(
        "--eval-margin", type=float, default=None,
        help=(
            "Legacy symmetric margin for both wealth and factor evaluation "
            "intervals. The saved first margin is used when omitted."
        ),
    )
    parser.add_argument(
        "--eval-w-min", type=float, default=None,
        help=(
            "Replace only the wealth lower endpoint after --eval-margin is "
            "applied. This changes X_ev and its norm, not the FD rectangle."
        ),
    )
    parser.add_argument(
        "--eval-w-max", type=float, default=None,
        help=(
            "Replace only the wealth upper endpoint after --eval-margin is "
            "applied. This changes X_ev and its norm, not the FD rectangle."
        ),
    )
    parser.add_argument(
        "--eval-x-margin", type=float, default=None,
        help=(
            "Optional symmetric factor-only margin. When omitted, the factor "
            "window continues to use --eval-margin."
        ),
    )
    parser.add_argument("--base-ny", type=int, default=41)
    parser.add_argument("--base-nx", type=int, default=41)
    parser.add_argument("--base-nt", type=int, default=80)
    parser.add_argument("--eval-ny", type=int, default=41)
    parser.add_argument("--eval-nx", type=int, default=41)
    parser.add_argument("--grid-factors", default="1,2")
    parser.add_argument(
        "--domain-factors", default=None,
        help=(
            "Legacy coupled shorthand: each D expands wealth and factor "
            "together, scheduling only (D,D). Cannot be combined with the "
            "split domain options. If no domain option is supplied, defaults "
            "to coupled 1.5,2.0."
        ),
    )
    parser.add_argument(
        "--wealth-domain-factors", default=None,
        help=(
            "Comma-separated expansion factors for the saved log-wealth "
            "half-width. Must be supplied together with "
            "--factor-domain-factors; their Cartesian product is evaluated."
        ),
    )
    parser.add_argument(
        "--factor-domain-factors", default=None,
        help=(
            "Comma-separated expansion factors for the saved state-factor "
            "half-width. Must be supplied together with "
            "--wealth-domain-factors, or together with explicit "
            "--fd-w-min/--fd-w-max bounds; their Cartesian product is evaluated."
        ),
    )
    parser.add_argument(
        "--fd-w-min", "--fd-w-mins", dest="fd_w_min", default=None,
        help=(
            "One value, or a narrowest-to-widest comma-separated schedule, "
            "of absolute FD wealth lower endpoints. Must be paired positionally "
            "with --fd-w-max and cannot be combined with wealth/domain factors."
        ),
    )
    parser.add_argument(
        "--fd-w-max", "--fd-w-maxs", dest="fd_w_max", default=None,
        help=(
            "One value, or a narrowest-to-widest comma-separated schedule, "
            "of absolute FD wealth upper endpoints. Every pair must strictly "
            "contain the saved training wealth interval."
        ),
    )
    parser.add_argument(
        "--boundaries", default="linearity,exact-dirichlet",
        help=(
            "Comma-separated boundary variants in primary-first order: "
            "linearity, crra-robin, exact-dirichlet. The first boundary is "
            "the numerical-refinement primary; replacements are reported as "
            "separate BVP sensitivity and never enter refinement pass/fail."
        ),
    )
    parser.add_argument("--verify-checkpoints", default="all")
    parser.add_argument("--drift-scheme", choices=("central", "adaptive", "monotone"), default="adaptive")
    parser.add_argument("--peclet-limit", type=float, default=1.0)
    parser.add_argument("--theta-method", type=float, default=0.5)
    parser.add_argument("--startup-be-steps", type=int, default=2)
    parser.add_argument(
        "--linear-residual-tolerance", type=float, default=1e-8,
        help="Hard upper bound for every normalized sparse linear-solve residual.",
    )
    parser.add_argument(
        "--boundary-condition-limit", type=float, default=1e12,
        help="Hard upper bound for the boundary-elimination infinity-norm condition number.",
    )
    parser.add_argument(
        "--policy-extension", choices=POLICY_EXTENSIONS,
        default="boundary-projection",
        help=(
            "Define the frozen policy outside saved nominal collocation bounds. "
            "boundary-projection is the finite-domain paper diagnostic; "
            "neural-extrapolation is sensitivity-only."
        ),
    )
    parser.add_argument("--denominator-tolerance", type=float, default=1e-12)
    parser.add_argument("--refinement-abs-tolerance", type=float, default=1e-2)
    parser.add_argument("--refinement-rel-tolerance", type=float, default=2e-2)
    parser.add_argument(
        "--refinement-rule",
        choices=("cartesian", "merton-axis"),
        default="cartesian",
        help=(
            "Grid/domain refinement pass/fail rule within the primary "
            "boundary. cartesian (default) includes every grid/domain "
            "interaction; merton-axis tests only one-at-a-time numerical "
            "axes. Boundary replacement is always reported separately."
        ),
    )
    parser.add_argument("--ellipticity-tolerance", type=float, default=0.0,
                        help="Require every sampled joint log-coordinate diffusion eigenvalue to be strictly larger than this threshold (default: 0)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_paper_checkpoint < 0:
        raise ValueError("--min-paper-checkpoint must be nonnegative")
    subset = _parse_ints(args.checkpoints) if args.checkpoints.strip() else None
    requested_run_dir = args.run_dir.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output is not None
              else requested_run_dir / "liu_exact_map_fd")
    # All long-running work is staged next to the requested destination. An
    # invalid or interrupted --overwrite therefore cannot erase a prior
    # successful audit. Unrelated files in the destination remain untouched.
    had_managed_output = _check_output(output, args.overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".liu-exact-map-stage-", dir=str(output.parent)
    ) as stage_text:
        stage = Path(stage_text)
        try:
            run = load_run(
                requested_run_dir,
                weight_dir_override=args.weight_dir,
                checkpoint_subset=subset,
                eval_margin_override=args.eval_margin,
                eval_w_min_override=args.eval_w_min,
                eval_w_max_override=args.eval_w_max,
                eval_x_margin_override=args.eval_x_margin,
                allow_sparse_subset=bool(args.skip_e4),
            )
            domain_design = _resolve_domain_design(
                legacy=args.domain_factors,
                wealth=args.wealth_domain_factors,
                factor=args.factor_domain_factors,
                fd_w_min=args.fd_w_min,
                fd_w_max=args.fd_w_max,
                problem=getattr(run, "problem", None),
            )
            evaluate_run(
                run, stage, device=args.device, base_ny=args.base_ny,
                base_nx=args.base_nx, base_nt=args.base_nt,
                eval_ny=args.eval_ny, eval_nx=args.eval_nx,
                grid_factors=_parse_ints(args.grid_factors),
                domain_mode=domain_design["mode"],
                wealth_domain_factors=domain_design["wealth_domain_factors"],
                factor_domain_factors=domain_design["factor_domain_factors"],
                domain_pairs=domain_design["domain_pairs"],
                boundaries=[item.strip() for item in args.boundaries.split(",") if item.strip()],
                verify_checkpoints=args.verify_checkpoints,
                drift_scheme=args.drift_scheme, peclet_limit=args.peclet_limit,
                theta_method=args.theta_method, startup_be_steps=args.startup_be_steps,
                denominator_tolerance=args.denominator_tolerance,
                refinement_abs_tolerance=args.refinement_abs_tolerance,
                refinement_rel_tolerance=args.refinement_rel_tolerance,
                ellipticity_tolerance=args.ellipticity_tolerance,
                linear_residual_tolerance=args.linear_residual_tolerance,
                boundary_condition_limit=args.boundary_condition_limit,
                policy_extension=args.policy_extension,
                skip_e4=bool(args.skip_e4),
                wealth_domain_parameterization=domain_design[
                    "wealth_domain_parameterization"
                ],
                wealth_domain_bounds=domain_design["wealth_domain_bounds"],
                requested_fd_w_mins=domain_design["requested_fd_w_mins"],
                requested_fd_w_maxs=domain_design["requested_fd_w_maxs"],
                refinement_rule=args.refinement_rule,
                min_paper_checkpoint=args.min_paper_checkpoint,
                overwrite=True,
            )
        except Exception as exc:
            if not had_managed_output:
                _prepare_output(output, overwrite=True)
                atomic_json(
                    output / "exact_map_status.json",
                    {"status": "failed", "error": repr(exc)},
                )
                (output / "_FAILED_EXACT_MAP").touch()
            raise
        _commit_staged_output(stage, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
