#!/usr/bin/env python3
"""Post-hoc Liu E9 evaluation on nested full-dimensional windows.

This program is intentionally independent of training.  It discovers only
completed affine Liu runs, loads only the official ``value_net_final.pt``,
and evaluates corresponding deterministic points on margins
``0.05,0.10,0.20,0.30`` by default.  The source run directories are opened
read-only; all new artifacts are written to a separate derived directory.

For each run and margin it reports errors in

* value ``V``;
* the reduced derivative bundle ``(V_w, V_ww, V_wx)``;
* wealth-normalized control ``vartheta = theta / w``;
* the fraction of points at which the concavity guard is active.

The per-seed values are aggregated with the sample standard deviation and a
two-sided Student-t 95% confidence interval.  When an existing full-
dimensional ``metrics.csv`` used the same design, ``RelL2_V`` and the legacy
full-dimensional name ``RelL2_theta`` (whose inputs are theta/w) are
cross-checked.  This is distinct from the legacy per-outer raw-theta diagnostic
``diag_RelL2_theta``.

PyTorch is imported lazily.  Run discovery, provenance checks, analytic
references, metrics, and aggregation can therefore be unit-tested without a
PyTorch installation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from aggregate_seeds import (
        canonical_market_hash,
        find_runs,
        group_key,
        load_config_args,
        load_config_args_raw,
        parse_seed_spec,
        run_status,
        run_updated_at,
        t_crit_95,
    )
    from joint_market_setup_dirichlet import validate_market_snapshot
except ModuleNotFoundError:  # package-style import from the repository root
    from .aggregate_seeds import (
        canonical_market_hash,
        find_runs,
        group_key,
        load_config_args,
        load_config_args_raw,
        parse_seed_spec,
        run_status,
        run_updated_at,
        t_crit_95,
    )
    from .joint_market_setup_dirichlet import validate_market_snapshot


VWW_GUARD = 1.0e-8
DEFAULT_MARGINS = (0.05, 0.10, 0.20, 0.30)
PER_RUN_FILE = "e9_margin_bundle_per_run.csv"
SUMMARY_FILE = "e9_margin_bundle_summary.csv"
CROSSCHECK_FILE = "e9_margin_bundle_crosscheck.csv"
PROVENANCE_FILE = "e9_margin_bundle_provenance.json"
SUCCESS_MARKER = "_SUCCESS_E9_MARGIN_BUNDLE"
OUTPUT_FILES = {
    PER_RUN_FILE,
    SUMMARY_FILE,
    CROSSCHECK_FILE,
    PROVENANCE_FILE,
    SUCCESS_MARKER,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_array_hash(named_arrays: Iterable[Tuple[str, np.ndarray]]) -> str:
    """Machine-independent hash of named numeric arrays."""
    digest = hashlib.sha256()
    for name, raw in named_arrays:
        array = np.asarray(raw)
        if array.dtype.hasobject:
            raise ValueError(f"object dtype is not canonical: {name}")
        if array.dtype.kind in "fciub":
            dtype = array.dtype
            if dtype.byteorder == ">" or (
                dtype.byteorder == "=" and not np.little_endian
            ):
                array = array.byteswap().view(dtype.newbyteorder("<"))
            else:
                array = array.astype(dtype.newbyteorder("<"), copy=False)
        array = np.ascontiguousarray(array)
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _scalar(values: Mapping[str, np.ndarray], key: str) -> float:
    if key not in values:
        raise ValueError(f"market snapshot is missing {key!r}")
    array = np.asarray(values[key])
    if array.size != 1:
        raise ValueError(f"market field {key!r} must be scalar, got {array.shape}")
    value = float(array.reshape(-1)[0])
    if not math.isfinite(value):
        raise ValueError(f"market field {key!r} is not finite")
    return value


def _integer_scalar(values: Mapping[str, np.ndarray], key: str) -> int:
    value = _scalar(values, key)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"market field {key!r} must be integer-valued")
    return rounded


def parse_int_list(text: str) -> List[int]:
    if not str(text or "").strip():
        return []
    return parse_seed_spec(text)


def parse_margins(text: str) -> List[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--margins must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate margins are not allowed: {text!r}")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(f"evaluation margin must be in [0,1), got {value}")
    return values


def normalize_model(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "pi-pinn":
        text = "pipinn"
    if text not in {"pinn", "pipinn"}:
        raise ValueError(f"unknown Liu model type {value!r}")
    return text


def parse_models(text: str) -> Optional[List[str]]:
    raw = str(text or "auto").strip().lower()
    if raw == "auto":
        return None
    if raw == "both":
        return ["pinn", "pipinn"]
    result: List[str] = []
    for token in re.split(r"[\s,]+", raw):
        if token:
            model = normalize_model(token)
            if model not in result:
                result.append(model)
    if not result:
        raise ValueError("--models selected no methods")
    return result


def shrink_bounds(
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    margin: float,
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Use the trainers' half-width convention, retaining 1-margin length."""
    removed = 0.5 * float(margin) * (upper - lower)
    return lower + removed, upper - removed


def map_base_design(
    unit_points: np.ndarray,
    *,
    margin: float,
    w_min: float,
    w_max: float,
    x_min: np.ndarray,
    x_max: np.ndarray,
    tau_max: float,
    tau_epsilon: float = 1.0e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one unit-cube design into a nested full-dimensional window."""
    unit = np.asarray(unit_points, dtype=np.float64)
    x_min = np.asarray(x_min, dtype=np.float64).reshape(-1)
    x_max = np.asarray(x_max, dtype=np.float64).reshape(-1)
    if unit.ndim != 2 or unit.shape[1] != x_min.size + 2:
        raise ValueError(
            f"unit design shape must be (P,{x_min.size + 2}), got {unit.shape}"
        )
    if unit.shape[0] < 1 or not np.all(np.isfinite(unit)):
        raise ValueError("unit design must be nonempty and finite")
    if np.any(unit < 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit design must lie in [0,1)")
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0,1), got {margin}")
    if not 0.0 <= tau_epsilon < tau_max:
        raise ValueError("require 0 <= tau_epsilon < tau_max")
    w_lo, w_hi = shrink_bounds(float(w_min), float(w_max), margin)
    x_lo, x_hi = shrink_bounds(x_min, x_max, margin)
    tau = tau_epsilon + unit[:, 0] * (tau_max - tau_epsilon)
    wealth = float(w_lo) + unit[:, 1] * (float(w_hi) - float(w_lo))
    state = np.asarray(x_lo)[None, :] + unit[:, 2:] * (
        np.asarray(x_hi) - np.asarray(x_lo)
    )[None, :]
    return tau, wealth, state


@dataclass(frozen=True)
class ClosedFormData:
    t: np.ndarray
    y: np.ndarray
    m_states: int

    def coefficients(self, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_array = np.asarray(tau, dtype=np.float64).reshape(-1)
        interpolated = np.column_stack(
            [np.interp(tau_array, self.t, row) for row in self.y]
        )
        m = self.m_states
        a = interpolated[:, 0]
        b = interpolated[:, 1:1 + m]
        C = interpolated[:, 1 + m:].reshape(-1, m, m)
        C = 0.5 * (C + np.swapaxes(C, 1, 2))
        return a, b, C


@dataclass(frozen=True)
class ReferenceValues:
    value: np.ndarray
    value_w: np.ndarray
    value_ww: np.ndarray
    value_wx: np.ndarray
    vartheta: np.ndarray


def analytic_reference(
    tau: np.ndarray,
    wealth: np.ndarray,
    state: np.ndarray,
    *,
    closed_form: ClosedFormData,
    gamma: float,
    r: float,
    lam0: np.ndarray,
    Lam: np.ndarray,
    Gamma: np.ndarray,
) -> ReferenceValues:
    """Vectorized affine Liu closed form and its reduced derivative bundle."""
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    wealth = np.asarray(wealth, dtype=np.float64).reshape(-1)
    state = np.asarray(state, dtype=np.float64)
    lam0 = np.asarray(lam0, dtype=np.float64).reshape(-1)
    Lam = np.asarray(Lam, dtype=np.float64)
    Gamma = np.asarray(Gamma, dtype=np.float64)
    p = tau.size
    m = closed_form.m_states
    n = lam0.size
    if wealth.shape != (p,) or state.shape != (p, m):
        raise ValueError("tau, wealth, and state point counts/dimensions disagree")
    if Lam.shape != (n, m) or Gamma.shape != (n, m):
        raise ValueError("Lam/Gamma dimensions disagree with lam0 and closed form")
    if np.any(wealth <= 0.0) or abs(1.0 - gamma) < 1.0e-14:
        raise ValueError("analytic CRRA reference requires positive wealth and gamma != 1")

    a, b, C = closed_form.coefficients(tau)
    grad_log_phi = b + np.einsum("pij,pj->pi", C, state)
    log_phi = a + np.einsum("pi,pi->p", b, state)
    log_phi += 0.5 * np.einsum("pi,pij,pj->p", state, C, state)
    phi = np.exp(log_phi)
    discount = np.exp((1.0 - gamma) * float(r) * tau)
    value = discount * np.power(wealth, 1.0 - gamma) * phi / (1.0 - gamma)
    value_w = discount * np.power(wealth, -gamma) * phi
    value_ww = -gamma * discount * np.power(wealth, -gamma - 1.0) * phi
    value_wx = value_w[:, None] * grad_log_phi
    lam_x = lam0[None, :] + state @ Lam.T
    vartheta = (lam_x + grad_log_phi @ Gamma.T) / gamma
    arrays = (value, value_w, value_ww, value_wx, vartheta)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("affine closed-form reference became nonfinite")
    return ReferenceValues(*arrays)


def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape or estimate.size == 0:
        raise ValueError("relative-L2 arrays must have equal nonempty shapes")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(reference)):
        raise ValueError("relative-L2 arrays must be finite")
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("relative-L2 reference norm is zero")
    return float(np.linalg.norm((estimate - reference).reshape(-1)) / denominator)


def pointwise_sup(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape or estimate.size == 0:
        raise ValueError("sup-error arrays must have equal nonempty shapes")
    difference = estimate - reference
    if difference.ndim == 1:
        return float(np.max(np.abs(difference)))
    return float(np.max(np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1)))


def compute_error_metrics(
    *,
    value: np.ndarray,
    value_w: np.ndarray,
    value_ww: np.ndarray,
    value_wx: np.ndarray,
    vartheta: np.ndarray,
    reference: ReferenceValues,
    guard_threshold: float = VWW_GUARD,
) -> Dict[str, float]:
    bundle = np.column_stack((
        np.asarray(value_w).reshape(-1),
        np.asarray(value_ww).reshape(-1),
        np.asarray(value_wx),
    ))
    reference_bundle = np.column_stack((
        reference.value_w.reshape(-1),
        reference.value_ww.reshape(-1),
        reference.value_wx,
    ))
    value_ww_array = np.asarray(value_ww, dtype=np.float64).reshape(-1)
    metrics = {
        "RelL2_V": relative_l2(value, reference.value),
        "Sup_V": pointwise_sup(value, reference.value),
        "RelL2_bundle": relative_l2(bundle, reference_bundle),
        "Sup_bundle": pointwise_sup(bundle, reference_bundle),
        "Sup_Vw": pointwise_sup(np.asarray(value_w).reshape(-1), reference.value_w),
        "Sup_Vww": pointwise_sup(value_ww_array, reference.value_ww),
        "Sup_Vwx": pointwise_sup(value_wx, reference.value_wx),
        "RelL2_vartheta": relative_l2(vartheta, reference.vartheta),
        "Sup_vartheta": pointwise_sup(vartheta, reference.vartheta),
        "guard_frac": float(np.mean(value_ww_array > -float(guard_threshold))),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("a margin-bundle metric is nonfinite")
    return metrics


def mean_std_t_ci(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("seed summary requires nonempty finite values")
    mean = float(np.mean(array))
    if array.size == 1:
        return {
            "mean": mean,
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(int(array.size))
    half_width = float(t_crit_95(int(array.size) - 1)) * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_lo": mean - half_width,
        "ci95_hi": mean + half_width,
    }


@dataclass(frozen=True)
class MarketData:
    K: np.ndarray
    xbar: np.ndarray
    SigmaX: np.ndarray
    rho: np.ndarray
    Lam: np.ndarray
    Q: np.ndarray
    Gamma: np.ndarray
    k0: np.ndarray
    lam0: np.ndarray
    X_min: np.ndarray
    X_max: np.ndarray
    eta: np.ndarray
    gamma: float
    r: float
    tau_max: float
    W_min: float
    W_max: float
    training_seed: int
    market_seed: int
    min_eig_Q: float
    min_eig_joint: float

    @property
    def n_assets(self) -> int:
        return int(self.lam0.size)

    @property
    def m_states(self) -> int:
        return int(self.xbar.size)


def load_market(path: Path) -> MarketData:
    with np.load(path, allow_pickle=False) as source:
        values = {key: np.asarray(source[key]).copy() for key in source.files}
    required_arrays = (
        "K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0",
        "lam0", "X_min", "X_max", "eta",
    )
    missing = [key for key in required_arrays if key not in values]
    if missing:
        raise ValueError(f"{path}: missing market arrays {missing}")
    arrays = {
        key: np.asarray(values[key], dtype=np.float64)
        for key in required_arrays
    }
    xbar = arrays["xbar"].reshape(-1)
    lam0 = arrays["lam0"].reshape(-1)
    eta = arrays["eta"].reshape(-1)
    m = int(xbar.size)
    n = int(lam0.size)
    shapes = {
        "K": (m, m), "SigmaX": (m, m), "rho": (n, m),
        "Lam": (n, m), "Q": (m, m), "Gamma": (n, m),
        "k0": (m,), "X_min": (m,), "X_max": (m,), "eta": (m,),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} has shape {arrays[key].shape}, expected {shape}")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError(f"{path}: market snapshot contains NaN or infinity")
    if np.any(eta <= 0.0) or np.any(arrays["X_max"] <= arrays["X_min"]):
        raise ValueError(f"{path}: invalid state scale or state bounds")
    if not np.allclose(arrays["Q"], arrays["SigmaX"] @ arrays["SigmaX"].T,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: Q != SigmaX @ SigmaX.T")
    if not np.allclose(arrays["Gamma"], arrays["rho"] @ arrays["SigmaX"].T,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: Gamma != rho @ SigmaX.T")
    if not np.allclose(arrays["k0"], arrays["K"] @ xbar,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: k0 != K @ xbar")
    q_symmetric = 0.5 * (arrays["Q"] + arrays["Q"].T)
    min_eig_q = float(np.linalg.eigvalsh(q_symmetric)[0])
    try:
        rho_diagnostics = validate_market_snapshot(values)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid market snapshot: {exc}") from exc
    min_eig_joint = float(rho_diagnostics["min_eig"])
    if min_eig_q <= 0.0:
        raise ValueError(
            f"{path}: non-elliptic snapshot (min eig Q={min_eig_q:.3e}, "
            f"joint={min_eig_joint:.3e})"
        )
    gamma = _scalar(values, "gamma")
    tau_max = _scalar(values, "tau_max")
    w_min = _scalar(values, "W_min")
    w_max = _scalar(values, "W_max")
    if gamma <= 0.0 or abs(gamma - 1.0) < 1.0e-12:
        raise ValueError(f"{path}: affine CRRA reference requires gamma>0, gamma!=1")
    if tau_max <= 0.0 or w_min <= 0.0 or w_max <= w_min:
        raise ValueError(f"{path}: invalid time/wealth domain")
    return MarketData(
        K=arrays["K"], xbar=xbar, SigmaX=arrays["SigmaX"], rho=arrays["rho"],
        Lam=arrays["Lam"], Q=arrays["Q"], Gamma=arrays["Gamma"],
        k0=arrays["k0"], lam0=lam0, X_min=arrays["X_min"],
        X_max=arrays["X_max"], eta=eta, gamma=gamma,
        r=_scalar(values, "r"), tau_max=tau_max, W_min=w_min, W_max=w_max,
        training_seed=_integer_scalar(values, "seed"),
        market_seed=_integer_scalar(values, "market_seed"),
        min_eig_Q=min_eig_q, min_eig_joint=min_eig_joint,
    )


def load_closed_form(path: Path, m_states: int, tau_max: float) -> ClosedFormData:
    with np.load(path, allow_pickle=False) as source:
        if "success" not in source.files:
            raise ValueError(f"{path}: missing ODE success flag")
        if not bool(np.asarray(source["success"]).reshape(-1)[0]):
            raise ValueError(f"{path}: closed-form ODE solve was unsuccessful")
        t = np.asarray(source["t"], dtype=np.float64).copy()
        y = np.asarray(source["y"], dtype=np.float64).copy()
    rows = 1 + m_states + m_states * m_states
    if t.ndim != 1 or y.shape != (rows, t.size):
        raise ValueError(f"{path}: expected t=(L,), y=({rows},L); got {t.shape}, {y.shape}")
    if (t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))
            or np.any(np.diff(t) <= 0.0)):
        raise ValueError(f"{path}: invalid closed-form interpolation grid")
    if not math.isclose(float(t[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{path}: closed-form grid must start at tau=0")
    if float(np.max(np.abs(y[:, 0]))) > 1.0e-10:
        raise ValueError(f"{path}: terminal coefficients at tau=0 are not zero")
    if not math.isclose(float(t[-1]), float(tau_max), rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"{path}: ODE horizon {t[-1]} != market horizon {tau_max}")
    return ClosedFormData(t=t, y=y, m_states=m_states)


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    model_type: str
    n_assets: int
    m_states: int
    seed: int
    group: str
    updated_at: str
    config_doc: Dict[str, Any]
    config_args: Dict[str, Any]
    effective_eval_args: Dict[str, Any]
    checkpoint: Optional[Path] = None
    checkpoint_sha256: str = ""
    config_sha256: str = ""
    market_file_sha256: str = ""
    market_hash: str = ""
    closed_form_file_sha256: str = ""
    closed_form_hash: str = ""


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _as_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    return result


def _check_status(run_dir: Path) -> None:
    if run_status(str(run_dir)) != "success":
        raise ValueError(f"{run_dir}: E9 accepts only an unambiguous _SUCCESS run")
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise ValueError(f"{run_dir}: missing training status.json")
    status = _read_json(status_path)
    if str(status.get("status", "")) != "success":
        raise ValueError(f"{status_path}: status is not success")


def _check_affine_paper_config(record: RunRecord) -> None:
    args = record.config_args
    if bool(args.get("eval_only", False)):
        raise ValueError(f"{record.run_dir}: config.json is not a training provenance record")
    if bool(args.get("timing_mode", False)):
        raise ValueError(f"{record.run_dir}: timing-mode runs are not paper final models")
    if record.model_type == "pipinn":
        mode = str(args.get("risk_premium_mode", "affine")).strip().lower()
        try:
            epsilon = float(args.get("nonaffine_eps", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{record.run_dir}: invalid nonaffine_eps") from exc
        if mode != "affine" or epsilon != 0.0:
            raise ValueError(
                f"{record.run_dir}: E9 analytic reference requires the affine main model; "
                f"got risk_premium_mode={mode!r}, nonaffine_eps={epsilon}"
            )
    hidden = _as_int(args.get("value_hidden", 256), "value_hidden")
    depth = _as_int(args.get("value_depth", 3), "value_depth")
    if hidden < 1 or depth < 1:
        raise ValueError(f"{record.run_dir}: invalid value-network architecture")


def is_affine_paper_config(model_type: str, args: Mapping[str, Any]) -> bool:
    """Cheap discovery filter; semantic checks still run after selection."""
    if bool(args.get("eval_only", False)) or bool(args.get("timing_mode", False)):
        return False
    if model_type == "pinn":
        return True
    mode = str(args.get("risk_premium_mode", "affine")).strip().lower()
    try:
        epsilon = float(args.get("nonaffine_eps", 0.0))
    except (TypeError, ValueError):
        return False
    return mode == "affine" and epsilon == 0.0


def resolve_final_checkpoint(record: RunRecord, out_root: Path) -> Path:
    raw_weight_dir = record.config_doc.get("weight_dir")
    candidates: List[Path] = []
    if raw_weight_dir:
        raw = Path(str(raw_weight_dir)).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            cwd = record.config_doc.get("cwd")
            if cwd:
                candidates.append(Path(str(cwd)) / raw)
            candidates.extend((record.run_dir / raw, out_root / raw, raw))
    method = "pinn" if record.model_type == "pinn" else "pi-pinn"
    candidates.extend((
        record.run_dir,
        out_root / "weights" / method / record.run_dir.name,
        out_root.parent / "weights" / method / record.run_dir.name,
    ))
    checked: List[str] = []
    seen: set[str] = set()
    for directory in candidates:
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        path = directory / "value_net_final.pt"
        checked.append(str(path))
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"{record.run_dir}: official value_net_final.pt not found; no last/best fallback "
        "is allowed in paper mode. Checked:\n  " + "\n  ".join(checked)
    )


def discover_runs(
    out_root: Path,
    *,
    models: Optional[Sequence[str]],
    n_assets: Sequence[int],
    m_states: Sequence[int],
    seeds: Sequence[int],
    run_name_regex: str = "",
    strict_seed_set: bool = False,
    min_seeds: int = 1,
) -> Dict[Tuple[str, int, int, str], List[RunRecord]]:
    """Discover one newest affine training attempt per configuration/seed.

    Deduplication deliberately happens *before* status filtering.  Otherwise an
    older successful directory can be resurrected after a newer rerun of the
    same configuration/seed failed.  A selected configuration is eligible only
    when every requested seed's newest attempt is successful.
    """
    wanted_models = set(models) if models is not None else None
    wanted_n = set(int(value) for value in n_assets)
    wanted_m = set(int(value) for value in m_states)
    wanted_seeds = set(int(value) for value in seeds)
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int, int, str, int], RunRecord] = {}

    for directory_text in find_runs(str(out_root)):
        run_dir = Path(directory_text).resolve()
        raw_args = load_config_args_raw(str(run_dir))
        if raw_args is None:
            continue
        try:
            model = normalize_model(raw_args.get("model_type"))
            n = _as_int(raw_args.get("n_assets"), "n_assets")
            m = _as_int(raw_args.get("m_states"), "m_states")
            seed = _as_int(raw_args.get("seed"), "seed")
        except ValueError:
            continue
        if wanted_models is not None and model not in wanted_models:
            continue
        if wanted_n and n not in wanted_n:
            continue
        if wanted_m and m not in wanted_m:
            continue
        if not is_affine_paper_config(model, raw_args):
            continue
        try:
            relative = str(run_dir.relative_to(out_root.resolve()))
        except ValueError:
            relative = str(run_dir)
        if pattern and not pattern.search(relative):
            continue
        config_doc = _read_json(run_dir / "config.json")
        effective = load_config_args(str(run_dir)) or dict(raw_args)
        group, _ = group_key(dict(raw_args))
        record = RunRecord(
            run_dir=run_dir, model_type=model, n_assets=n, m_states=m,
            seed=seed, group=group, updated_at=run_updated_at(str(run_dir)),
            config_doc=config_doc, config_args=dict(raw_args),
            effective_eval_args=dict(effective),
        )
        key = (model, n, m, group, seed)
        previous = newest.get(key)
        if previous is None or (record.updated_at, str(record.run_dir)) >= (
            previous.updated_at, str(previous.run_dir)
        ):
            newest[key] = record

    by_cell_group: Dict[Tuple[str, int, int], Dict[str, List[RunRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in newest.values():
        by_cell_group[(record.model_type, record.n_assets, record.m_states)][record.group].append(record)
    if not by_cell_group:
        raise ValueError("no Liu training attempts match the requested filters")

    selected: Dict[Tuple[str, int, int, str], List[RunRecord]] = {}
    errors: List[str] = []
    for cell, groups in sorted(by_cell_group.items()):
        eligible_groups: Dict[str, List[RunRecord]] = {}
        blocked_groups: Dict[str, List[str]] = {}
        for group, records in groups.items():
            available = {record.seed: record for record in records}
            if wanted_seeds and not wanted_seeds.issubset(available):
                continue
            chosen_seeds = sorted(wanted_seeds if wanted_seeds else available)
            chosen = [available[seed] for seed in chosen_seeds]
            non_success = [
                f"seed={record.seed} status={run_status(str(record.run_dir))} "
                f"run={record.run_dir}"
                for record in chosen
                if run_status(str(record.run_dir)) != "success"
            ]
            if non_success:
                blocked_groups[group] = non_success
                continue
            if len(chosen) >= min_seeds:
                eligible_groups[group] = chosen
        if len(eligible_groups) != 1:
            blocked_text = ""
            if blocked_groups:
                details = [
                    f"group={group}: " + "; ".join(items)
                    for group, items in sorted(blocked_groups.items())
                ]
                blocked_text = (
                    "; newest attempt(s) are not successful (older successes "
                    "are intentionally not reused): " + " | ".join(details)
                )
            errors.append(
                f"model={cell[0]}, N={cell[1]}, M={cell[2]}: expected one matching "
                f"training configuration, found {sorted(eligible_groups)}; narrow with "
                "--run-name-regex or specify --seeds" + blocked_text
            )
            continue
        group, records = next(iter(eligible_groups.items()))
        available_all = {
            record.seed for record in groups[group]
            if run_status(str(record.run_dir)) == "success"
        }
        if strict_seed_set and wanted_seeds and available_all != wanted_seeds:
            errors.append(
                f"model={cell[0]}, N={cell[1]}, M={cell[2]}: successful seeds "
                f"{sorted(available_all)} != requested exact set {sorted(wanted_seeds)}"
            )
            continue
        selected[(cell[0], cell[1], cell[2], group)] = sorted(records, key=lambda item: item.seed)
    if errors:
        raise ValueError("E9 run selection failed:\n  - " + "\n  - ".join(errors))
    return selected


def enforce_strict_crosschecks(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require every requested legacy-metric comparison to be available and pass."""
    failures = [row for row in rows if str(row.get("status", "")) != "pass"]
    if not failures:
        return
    counts: Dict[str, int] = defaultdict(int)
    for row in failures:
        counts[str(row.get("status", "unknown"))] += 1
    preview = "; ".join(
        f"{row.get('model_type')}/M{row.get('m_states')}/"
        f"seed{row.get('training_seed')}/m{row.get('eval_margin')}/"
        f"{row.get('legacy_metric')}={row.get('status', 'unknown')}"
        for row in failures[:8]
    )
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    raise ValueError(
        f"{len(failures)} existing metric cross-check(s) did not pass "
        f"({summary}): {preview}"
    )


def _check_config_value(
    record: RunRecord,
    key: str,
    expected: float,
    *,
    atol: float = 1.0e-12,
) -> None:
    if key not in record.config_args or record.config_args[key] is None:
        raise ValueError(f"{record.run_dir}: config is missing {key!r}")
    try:
        actual = float(record.config_args[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.run_dir}: config {key!r} is not numeric") from exc
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=atol):
        raise ValueError(
            f"{record.run_dir}: config {key}={actual} disagrees with snapshot {expected}"
        )


def validate_run_provenance(
    selected: Mapping[Tuple[str, int, int, str], Sequence[RunRecord]],
    out_root: Path,
) -> Tuple[
    Dict[Tuple[str, int, int, str], List[RunRecord]],
    Dict[Path, MarketData],
    Dict[Path, ClosedFormData],
]:
    """Validate every config/market/closed-form/final-checkpoint relationship."""
    updated: Dict[Tuple[str, int, int, str], List[RunRecord]] = {}
    markets: Dict[Path, MarketData] = {}
    closed_forms: Dict[Path, ClosedFormData] = {}
    economic_hashes: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    closed_hashes: Dict[Tuple[int, int], set[str]] = defaultdict(set)

    for cell, records in selected.items():
        group_records: List[RunRecord] = []
        for record in records:
            _check_status(record.run_dir)
            _check_affine_paper_config(record)
            market_path = record.run_dir / "market_params.npz"
            closed_path = record.run_dir / "closed_form_ode.npz"
            if not market_path.is_file() or not closed_path.is_file():
                raise FileNotFoundError(
                    f"{record.run_dir}: missing market_params.npz or closed_form_ode.npz"
                )
            market = load_market(market_path)
            if (market.n_assets, market.m_states, market.training_seed) != (
                record.n_assets, record.m_states, record.seed
            ):
                raise ValueError(
                    f"{market_path}: snapshot (N={market.n_assets},M={market.m_states},"
                    f"seed={market.training_seed}) disagrees with config "
                    f"(N={record.n_assets},M={record.m_states},seed={record.seed})"
                )
            for key, expected in (
                ("gamma", market.gamma), ("r", market.r),
                ("tau_max", market.tau_max), ("w_min", market.W_min),
                ("w_max", market.W_max), ("market_seed", market.market_seed),
            ):
                _check_config_value(record, key, float(expected), atol=1.0e-9 if key == "market_seed" else 1.0e-12)
            scale = float(record.config_args.get("x_range_scale", float("nan")))
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{record.run_dir}: invalid x_range_scale")
            if (not np.allclose(market.X_min, market.xbar - scale * market.eta,
                                rtol=1.0e-10, atol=1.0e-12)
                    or not np.allclose(market.X_max, market.xbar + scale * market.eta,
                                       rtol=1.0e-10, atol=1.0e-12)):
                raise ValueError(f"{market_path}: state bounds disagree with xbar/eta/x_range_scale")

            closed_form = load_closed_form(closed_path, record.m_states, market.tau_max)
            checkpoint = resolve_final_checkpoint(record, out_root)
            market_hash = canonical_market_hash(str(market_path))
            closed_hash = canonical_array_hash((("t", closed_form.t), ("y", closed_form.y)))
            economic_hashes[(record.n_assets, record.m_states)].add(market_hash)
            closed_hashes[(record.n_assets, record.m_states)].add(closed_hash)
            updated_record = replace(
                record,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                config_sha256=sha256_file(record.run_dir / "config.json"),
                market_file_sha256=sha256_file(market_path),
                market_hash=market_hash,
                closed_form_file_sha256=sha256_file(closed_path),
                closed_form_hash=closed_hash,
            )
            markets[record.run_dir] = market
            closed_forms[record.run_dir] = closed_form
            group_records.append(updated_record)
        updated[cell] = group_records

    for dimensions, hashes in sorted(economic_hashes.items()):
        if len(hashes) != 1:
            raise ValueError(
                f"N={dimensions[0]}, M={dimensions[1]}: market snapshots differ across "
                f"selected methods/seeds: {sorted(hashes)}"
            )
    for dimensions, hashes in sorted(closed_hashes.items()):
        if len(hashes) != 1:
            raise ValueError(
                f"N={dimensions[0]}, M={dimensions[1]}: closed-form coefficient grids "
                f"differ across selected methods/seeds: {sorted(hashes)}"
            )
    return updated, markets, closed_forms


def import_torch(device_text: str, torch_num_threads: int) -> Tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to evaluate value_net_final.pt; provenance and pure "
            "helper tests do not require it"
        ) from exc
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    requested = str(device_text or "auto").strip().lower()
    if requested in {"", "auto"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device_text}, but CUDA is unavailable")
        device = torch.device(device_text)
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index < 0 or index >= int(torch.cuda.device_count()):
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        device = torch.device(f"cuda:{index}")
        torch.cuda.get_device_properties(device)
    return torch, nn, device


def build_value_network(torch: Any, nn: Any, m_states: int, hidden: int, depth: int) -> Any:
    class ValueNetND(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_dim = m_states + 2
            layers: List[Any] = []
            for _ in range(depth):
                layers.extend((nn.Linear(in_dim, hidden), nn.Tanh()))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, wealth: Any, state: Any, tau: Any) -> Any:
            return self.net(torch.cat((wealth, state, tau), dim=1))

    return ValueNetND()


def load_final_model(record: RunRecord, torch: Any, nn: Any, device: Any) -> Any:
    if record.checkpoint is None or record.checkpoint.name != "value_net_final.pt":
        raise ValueError("paper-mode loader accepts only value_net_final.pt")
    hidden = int(record.config_args.get("value_hidden", 256))
    depth = int(record.config_args.get("value_depth", 3))
    model = build_value_network(torch, nn, record.m_states, hidden, depth).to(device)
    try:
        state = torch.load(record.checkpoint, map_location=device, weights_only=True)
    except TypeError:  # supported legacy torch versions
        state = torch.load(record.checkpoint, map_location=device)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{record.checkpoint}: checkpoint is not a state dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def torch_base_design(torch: Any, n_points: int, m_states: int, base_seed: int) -> np.ndarray:
    """Reproduce the trainers' CPU torch.Generator evaluation design exactly."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(base_seed))
    return torch.rand(int(n_points), 2 + int(m_states), generator=generator).numpy().astype(np.float64)


def evaluate_model_bundle(
    model: Any,
    *,
    tau: np.ndarray,
    wealth: np.ndarray,
    state: np.ndarray,
    market: MarketData,
    torch: Any,
    device: Any,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    p = int(np.asarray(tau).size)
    outputs: Dict[str, List[np.ndarray]] = {
        "value": [], "value_w": [], "value_ww": [], "value_wx": [],
    }
    for start in range(0, p, chunk_size):
        stop = min(start + chunk_size, p)
        w_tensor = torch.tensor(
            wealth[start:stop, None], dtype=torch.float32, device=device,
            requires_grad=True,
        )
        x_tensor = torch.tensor(
            state[start:stop], dtype=torch.float32, device=device,
            requires_grad=True,
        )
        tau_tensor = torch.tensor(
            tau[start:stop, None], dtype=torch.float32, device=device,
        )
        value = model(w_tensor, x_tensor, tau_tensor)
        value_w = torch.autograd.grad(
            value, w_tensor, grad_outputs=torch.ones_like(value),
            create_graph=True, retain_graph=True,
        )[0]
        value_ww = torch.autograd.grad(
            value_w, w_tensor, grad_outputs=torch.ones_like(value_w),
            create_graph=False, retain_graph=True,
        )[0]
        value_wx = torch.autograd.grad(
            value_w, x_tensor, grad_outputs=torch.ones_like(value_w),
            create_graph=False, retain_graph=False,
        )[0]
        for name, tensor in (
            ("value", value), ("value_w", value_w),
            ("value_ww", value_ww), ("value_wx", value_wx),
        ):
            outputs[name].append(tensor.detach().cpu().numpy())

    value = np.concatenate(outputs["value"], axis=0).reshape(-1).astype(np.float64)
    value_w = np.concatenate(outputs["value_w"], axis=0).reshape(-1).astype(np.float64)
    value_ww = np.concatenate(outputs["value_ww"], axis=0).reshape(-1).astype(np.float64)
    value_wx = np.concatenate(outputs["value_wx"], axis=0).astype(np.float64)
    lam_x = market.lam0[None, :] + np.asarray(state, dtype=np.float64) @ market.Lam.T
    numerator = lam_x * value_w[:, None] + value_wx @ market.Gamma.T
    theta = -numerator / np.minimum(value_ww, -VWW_GUARD)[:, None]
    vartheta = theta / np.asarray(wealth, dtype=np.float64)[:, None]
    result = {
        "value": value, "value_w": value_w, "value_ww": value_ww,
        "value_wx": value_wx, "vartheta": vartheta,
    }
    if any(not np.all(np.isfinite(array)) for array in result.values()):
        raise FloatingPointError("network value/derivative/control evaluation became nonfinite")
    return result


def read_existing_metrics(run_dir: Path) -> Dict[Tuple[float, str], float]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    result: Dict[Tuple[float, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if str(row.get("scope", "")) != "fulldim":
                    continue
                margin = float(row["eval_margin"])
                metric = str(row["metric"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(margin) and math.isfinite(value):
                result[(margin, metric)] = value
    return result


def _matching_margin_key(
    values: Mapping[Tuple[float, str], float], margin: float, metric: str
) -> Optional[Tuple[float, str]]:
    for key in values:
        if key[1] == metric and math.isclose(key[0], margin, rel_tol=0.0, abs_tol=1.0e-12):
            return key
    return None


def evaluate_selected_runs(
    selected: Mapping[Tuple[str, int, int, str], Sequence[RunRecord]],
    markets: Mapping[Path, MarketData],
    closed_forms: Mapping[Path, ClosedFormData],
    *,
    margins: Sequence[float],
    n_points: int,
    base_seed: int,
    tau_epsilon: float,
    chunk_size: int,
    torch: Any,
    nn: Any,
    device: Any,
    crosscheck_rtol: float,
    crosscheck_atol: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, str]]:
    per_run_rows: List[Dict[str, Any]] = []
    crosscheck_rows: List[Dict[str, Any]] = []
    base_designs: Dict[int, np.ndarray] = {}
    design_hashes: Dict[int, str] = {}
    for records in selected.values():
        m = records[0].m_states
        if m not in base_designs:
            design = torch_base_design(torch, n_points, m, base_seed)
            base_designs[m] = design
            design_hashes[m] = canonical_array_hash((("unit_points", design),))

    for cell, records in sorted(selected.items()):
        for record in records:
            market = markets[record.run_dir]
            closed_form = closed_forms[record.run_dir]
            model = load_final_model(record, torch, nn, device)
            existing = read_existing_metrics(record.run_dir)
            effective_points = int(record.effective_eval_args.get("test_points", 0) or 0)
            comparable_design = effective_points == n_points and base_seed == 727 and math.isclose(
                tau_epsilon, 1.0e-3, rel_tol=0.0, abs_tol=1.0e-15
            )
            for margin in margins:
                tau, wealth, state = map_base_design(
                    base_designs[record.m_states], margin=float(margin),
                    w_min=market.W_min, w_max=market.W_max,
                    x_min=market.X_min, x_max=market.X_max,
                    tau_max=market.tau_max, tau_epsilon=tau_epsilon,
                )
                reference = analytic_reference(
                    tau, wealth, state, closed_form=closed_form,
                    gamma=market.gamma, r=market.r, lam0=market.lam0,
                    Lam=market.Lam, Gamma=market.Gamma,
                )
                estimate = evaluate_model_bundle(
                    model, tau=tau, wealth=wealth, state=state, market=market,
                    torch=torch, device=device, chunk_size=chunk_size,
                )
                metrics = compute_error_metrics(reference=reference, **estimate)
                common = {
                    "group": record.group,
                    "model_type": record.model_type,
                    "n_assets": record.n_assets,
                    "m_states": record.m_states,
                    "training_seed": record.seed,
                    "eval_margin": float(margin),
                    "n_points": n_points,
                    "base_seed": base_seed,
                    "run_dir": str(record.run_dir),
                    "checkpoint": str(record.checkpoint),
                    "checkpoint_sha256": record.checkpoint_sha256,
                    "market_hash": record.market_hash,
                    "closed_form_hash": record.closed_form_hash,
                }
                for metric, value in metrics.items():
                    per_run_rows.append({**common, "metric": metric, "value": value})

                new_for_legacy = {
                    "RelL2_V": metrics["RelL2_V"],
                    "RelL2_theta": metrics["RelL2_vartheta"],
                }
                for legacy_metric, new_value in new_for_legacy.items():
                    key = _matching_margin_key(existing, float(margin), legacy_metric)
                    old_value = existing[key] if key is not None else float("nan")
                    if key is None:
                        status = "not_available"
                    elif not comparable_design:
                        status = "not_comparable_design"
                    else:
                        close = math.isclose(
                            old_value, new_value,
                            rel_tol=crosscheck_rtol, abs_tol=crosscheck_atol,
                        )
                        status = "pass" if close else "mismatch"
                    crosscheck_rows.append({
                        **{key: value for key, value in common.items()
                           if key not in {"checkpoint", "checkpoint_sha256"}},
                        "legacy_metric": legacy_metric,
                        "new_metric": "RelL2_vartheta" if legacy_metric == "RelL2_theta" else "RelL2_V",
                        "existing_value": old_value,
                        "new_value": new_value,
                        "abs_difference": abs(old_value - new_value) if math.isfinite(old_value) else float("nan"),
                        "effective_existing_test_points": effective_points,
                        "status": status,
                    })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return per_run_rows, crosscheck_rows, design_hashes


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["group"], row["model_type"], int(row["n_assets"]),
            int(row["m_states"]), float(row["eval_margin"]), row["metric"],
        )
        groups[key].append(row)
    output: List[Dict[str, Any]] = []
    for key, metric_rows in sorted(groups.items()):
        ordered = sorted(metric_rows, key=lambda row: int(row["training_seed"]))
        values = [float(row["value"]) for row in ordered]
        summary = mean_std_t_ci(values)
        output.append({
            "group": key[0], "model_type": key[1], "n_assets": key[2],
            "m_states": key[3], "eval_margin": key[4], "metric": key[5],
            "n": len(values), "seeds": ",".join(str(row["training_seed"]) for row in ordered),
            **summary,
        })
    return output


def _csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.17g}"
    return value


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing_owned = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing_owned and not overwrite:
        raise FileExistsError(
            f"derived E9 output already exists in {output}: {sorted(existing_owned)}; "
            "pass --overwrite to replace this evaluator's files"
        )


def commit_staged_output(stage: Path, output: Path) -> None:
    """Install a complete E9 artifact set, rolling back on commit failure."""

    output.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=".e9_margin_backup_", dir=str(output.parent)))
    moved_old: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        for name in sorted(OUTPUT_FILES):
            original = output / name
            if original.exists() or original.is_symlink():
                saved = backup / name
                os.replace(original, saved)
                moved_old.append((saved, original))
        names = [
            name for name in sorted(OUTPUT_FILES)
            if name != SUCCESS_MARKER
            and ((stage / name).is_file() or (stage / name).is_symlink())
        ]
        if (stage / SUCCESS_MARKER).is_file() or (stage / SUCCESS_MARKER).is_symlink():
            names.append(SUCCESS_MARKER)
        missing = sorted(OUTPUT_FILES - set(names))
        if missing:
            raise RuntimeError(f"staged E9 artifact set is incomplete: {missing}")
        for name in names:
            destination = output / name
            os.replace(stage / name, destination)
            installed.append(destination)
    except Exception:
        for path in reversed(installed):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        for saved, original in reversed(moved_old):
            if saved.exists() or saved.is_symlink():
                os.replace(saved, original)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def ensure_separate_output(output: Path, records: Iterable[RunRecord]) -> None:
    resolved_output = output.resolve()
    for record in records:
        run_dir = record.run_dir.resolve()
        if resolved_output == run_dir or run_dir in resolved_output.parents:
            raise ValueError(
                f"derived output {resolved_output} is inside source run {run_dir}; "
                "choose a separate directory"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate affine Liu final checkpoints on E9 nested windows."
    )
    parser.add_argument("--out-root", required=True, help="Root containing completed Liu runs")
    parser.add_argument(
        "--output", default="",
        help="Derived output directory (default: OUT_ROOT/derived/e9_margin_bundle)",
    )
    parser.add_argument("--models", default="auto", help="auto, both, pinn, pipinn, or comma list")
    parser.add_argument("--n-assets", default="", help="Optional comma/range filter")
    parser.add_argument("--m-states", default="", help="Optional comma/range filter")
    parser.add_argument(
        "--seeds", default="",
        help="Arbitrary comma/range seed subset; default uses every discovered seed",
    )
    parser.add_argument(
        "--expected-seeds", default="",
        help="Exact comma/range seed set required in every selected configuration",
    )
    parser.add_argument(
        "--strict-seed-set", action="store_true",
        help="Also fail if a selected configuration contains successful seeds outside --seeds",
    )
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument(
        "--margins", default=",".join(f"{value:.2f}" for value in DEFAULT_MARGINS),
        help="Nested half-width margins (default: 0.05,0.10,0.20,0.30)",
    )
    parser.add_argument(
        "--n-points", type=int, default=100000,
        help="Deterministic points per margin (paper default: 100000)",
    )
    parser.add_argument("--base-seed", type=int, default=727)
    parser.add_argument("--tau-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--crosscheck-rtol", type=float, default=5.0e-5)
    parser.add_argument("--crosscheck-atol", type=float, default=5.0e-7)
    parser.add_argument(
        "--strict-crosscheck", action="store_true",
        help=(
            "Require every requested metrics.csv value/policy comparison to exist, "
            "use the same design, and agree numerically"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_seeds < 1 or args.n_points < 1 or args.chunk_size < 1:
        raise SystemExit("--min-seeds, --n-points, and --chunk-size must be positive")
    if args.crosscheck_rtol < 0.0 or args.crosscheck_atol < 0.0:
        raise SystemExit("cross-check tolerances must be nonnegative")
    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out-root is not a directory: {out_root}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else out_root / "derived" / "e9_margin_bundle"
    )
    stage: Optional[Path] = None
    try:
        models = parse_models(args.models)
        n_assets = parse_int_list(args.n_assets)
        m_states = parse_int_list(args.m_states)
        if args.seeds and args.expected_seeds:
            selected_seed_spec = parse_int_list(args.seeds)
            expected_seed_spec = parse_int_list(args.expected_seeds)
            if selected_seed_spec != expected_seed_spec:
                raise ValueError(
                    "--seeds and --expected-seeds cannot select different sets"
                )
            seeds = selected_seed_spec
        else:
            seeds = parse_int_list(args.expected_seeds or args.seeds)
        strict_seed_set = bool(args.strict_seed_set or args.expected_seeds)
        margins = parse_margins(args.margins)
        selected = discover_runs(
            out_root, models=models, n_assets=n_assets, m_states=m_states,
            seeds=seeds, run_name_regex=args.run_name_regex,
            strict_seed_set=strict_seed_set, min_seeds=args.min_seeds,
        )
        selected, markets, closed_forms = validate_run_provenance(selected, out_root)
        all_records = [record for records in selected.values() for record in records]
        ensure_separate_output(output, all_records)
        torch, nn, device = import_torch(args.device, args.torch_num_threads)
        rows, crosschecks, design_hashes = evaluate_selected_runs(
            selected, markets, closed_forms, margins=margins,
            n_points=args.n_points, base_seed=args.base_seed,
            tau_epsilon=args.tau_epsilon, chunk_size=args.chunk_size,
            torch=torch, nn=nn, device=device,
            crosscheck_rtol=args.crosscheck_rtol,
            crosscheck_atol=args.crosscheck_atol,
        )
        if args.strict_crosscheck:
            enforce_strict_crosschecks(crosschecks)
        summaries = aggregate_rows(rows)

        # Do not touch the derived directory until discovery, all provenance
        # checks, checkpoint loads, evaluations, and optional strict
        # cross-checks have succeeded.
        prepare_output(output, args.overwrite)
        stage = Path(tempfile.mkdtemp(
            prefix=".e9_margin_stage_", dir=str(output.parent)
        ))
        per_run_fields = (
            "group", "model_type", "n_assets", "m_states", "training_seed",
            "eval_margin", "n_points", "base_seed", "metric", "value",
            "run_dir", "checkpoint", "checkpoint_sha256", "market_hash",
            "closed_form_hash",
        )
        summary_fields = (
            "group", "model_type", "n_assets", "m_states", "eval_margin",
            "metric", "n", "seeds", "mean", "std", "sem", "ci95_lo", "ci95_hi",
        )
        crosscheck_fields = (
            "group", "model_type", "n_assets", "m_states", "training_seed",
            "eval_margin", "n_points", "base_seed", "legacy_metric", "new_metric",
            "existing_value", "new_value", "abs_difference",
            "effective_existing_test_points", "status", "run_dir", "market_hash",
            "closed_form_hash",
        )
        write_csv_atomic(stage / PER_RUN_FILE, rows, per_run_fields)
        write_csv_atomic(stage / SUMMARY_FILE, summaries, summary_fields)
        write_csv_atomic(stage / CROSSCHECK_FILE, crosschecks, crosscheck_fields)
        provenance_runs = []
        for record in all_records:
            market = markets[record.run_dir]
            provenance_runs.append({
                "model_type": record.model_type,
                "n_assets": record.n_assets,
                "m_states": record.m_states,
                "training_seed": record.seed,
                "group": record.group,
                "run_dir": str(record.run_dir),
                "config_sha256": record.config_sha256,
                "market_params_file_sha256": record.market_file_sha256,
                "canonical_market_hash": record.market_hash,
                "closed_form_file_sha256": record.closed_form_file_sha256,
                "canonical_closed_form_hash": record.closed_form_hash,
                "checkpoint": str(record.checkpoint),
                "checkpoint_sha256": record.checkpoint_sha256,
                "min_eig_Q": market.min_eig_Q,
                "min_eig_joint_innovation": market.min_eig_joint,
            })
        write_json_atomic(stage / PROVENANCE_FILE, {
            "schema_version": 1,
            "created_at": utc_now(),
            "host": socket.gethostname(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_out_root": str(out_root),
            "output": str(output),
            "paper_mode": {
                "completed_success_only": True,
                "affine_only": True,
                "official_checkpoint_only": "value_net_final.pt",
                "source_runs_mutated": False,
            },
            "evaluation": {
                "margins": margins,
                "n_points": args.n_points,
                "base_seed": args.base_seed,
                "tau_epsilon": args.tau_epsilon,
                "chunk_size": args.chunk_size,
                "device": str(device),
                "vww_guard": VWW_GUARD,
                "base_design_hash_by_m": {str(key): value for key, value in design_hashes.items()},
            },
            "seed_selection": {
                "requested": seeds,
                "strict_set": strict_seed_set,
                "selection_mode": "expected_exact" if args.expected_seeds else "subset_or_all",
                "min_seeds": args.min_seeds,
            },
            "crosscheck": {
                "rtol": args.crosscheck_rtol,
                "atol": args.crosscheck_atol,
                "strict": bool(args.strict_crosscheck),
                "counts": {
                    status: sum(row["status"] == status for row in crosschecks)
                    for status in ("pass", "mismatch", "not_available", "not_comparable_design")
                },
            },
            "runs": provenance_runs,
            "artifacts": [PER_RUN_FILE, SUMMARY_FILE, CROSSCHECK_FILE],
        })
        (stage / SUCCESS_MARKER).touch()
        commit_staged_output(stage, output)
    except (ValueError, OSError, FileExistsError, RuntimeError, FloatingPointError) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

    print(
        f"[done] evaluated {len(all_records)} final checkpoint(s), "
        f"{len(margins)} margin(s); wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
