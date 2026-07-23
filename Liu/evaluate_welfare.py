#!/usr/bin/env python3
"""Evaluate certainty-equivalent wealth and welfare loss after training.

The evaluator is deliberately separate from both training programs.  For
each successful training seed it loads the official ``value_net_final.pt``
checkpoint, simulates its final greedy policy, and only then aggregates CE
and WL across training seeds.  The closed-form policy is simulated with the
same Euler scheme and common random numbers, so its Monte Carlo CE is the WL
denominator.  The analytic continuous-time CE is reported only as a
validation diagnostic.

Paper defaults are fixed here:

* w0 = 0.5 and x0 = xbar;
* 100,000 paths and 1,000 log-wealth Euler steps;
* projected network-policy extension for the main result;
* seed set 1,...,10 and dimensions M in {1,3,5};
* one canonical market snapshot per M across both methods and all seeds.

PyTorch is imported only when simulation starts.  Consequently ``--help``,
run discovery, and the pure NumPy statistical helpers work on a lightweight
post-processing machine without PyTorch installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    canonical_market_hash,
    find_runs,
    group_key,
    load_config_args_raw,
    parse_seed_spec,
    run_status,
    run_updated_at,
    t_crit_95,
)
from joint_market_setup_dirichlet import validate_market_snapshot


VWW_GUARD = 1.0e-8
RESUME_SCHEMA_VERSION = 1
WELFARE_FIELDS = (
    "model_type",
    "training_seed",
    "M",
    "policy",
    "extension",
    "mc_seed",
    "n_paths",
    "n_steps",
    "dt",
    "expected_utility",
    "se_expected_utility",
    "ce",
    "se_ce",
    "wl",
    "se_wl",
    "state_exit_step_frac",
    "state_exit_path_frac",
    "wealth_exit_step_frac",
    "wealth_exit_path_frac",
    "projection_step_frac",
    "vww_guard_frac",
    "max_policy_norm",
)


class ResumeSignatureError(RuntimeError):
    """Raised before output mutation when existing results are incompatible."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scalar(values: Mapping[str, np.ndarray], key: str) -> float:
    if key not in values:
        raise ValueError(f"market snapshot is missing {key!r}")
    arr = np.asarray(values[key])
    if arr.size != 1:
        raise ValueError(f"market field {key!r} must be scalar, got shape {arr.shape}")
    result = float(arr.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"market field {key!r} is not finite")
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Pure NumPy CE/WL statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilityStats:
    expected_utility: float
    se_expected_utility: float
    ce: float
    se_ce: float
    log_mean_power: float


@dataclass(frozen=True)
class WelfareStats:
    wl: float
    se_wl: float
    ce_ratio: float


def _exp_or_boundary(log_value: float) -> float:
    """Exponentiate a log value without emitting an overflow warning."""
    if math.isnan(log_value):
        return float("nan")
    if log_value > math.log(np.finfo(float).max):
        return float("inf")
    # Returning zero below the subnormal range is the correct floating-point
    # limit and avoids a noisy underflow warning.
    if log_value < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return math.exp(log_value)


def _scaled_power(log_wealth: np.ndarray, q: float) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(log_wealth, dtype=np.float64).reshape(-1)
    if y.size < 2:
        raise ValueError("at least two Monte Carlo paths are required")
    if not np.all(np.isfinite(y)):
        raise ValueError("terminal log wealth contains NaN or infinity")
    if not math.isfinite(q) or abs(q) < 1.0e-12:
        raise ValueError("CRRA gamma=1 (log utility) is not supported by this CE formula")
    z = q * y
    shift = float(np.max(z))
    scaled = np.exp(z - shift)
    mean_scaled = float(np.mean(scaled))
    if not math.isfinite(mean_scaled) or mean_scaled <= 0.0:
        raise ValueError("failed to form a positive mean CRRA power")
    log_mean = shift + math.log(mean_scaled)
    return scaled, mean_scaled, log_mean


def utility_statistics(log_wealth: np.ndarray, gamma: float) -> UtilityStats:
    """Compute expected CRRA utility, CE, and delta-method MC SE.

    The power terms are shifted before exponentiation.  CE is therefore
    ``U^{-1}(mean U(W_T))`` rather than mean terminal wealth or mean pathwise
    CE.  ``se_ce`` uses the pathwise influence value specified in the paper
    protocol.
    """
    q = 1.0 - float(gamma)
    scaled, mean_scaled, log_mean = _scaled_power(log_wealth, q)
    n_paths = int(scaled.size)

    log_abs_expected_utility = log_mean - math.log(abs(q))
    expected_utility = math.copysign(_exp_or_boundary(log_abs_expected_utility), q)

    sd_scaled = float(np.std(scaled, ddof=1))
    if sd_scaled == 0.0:
        se_expected_utility = 0.0
    else:
        shift = log_mean - math.log(mean_scaled)
        log_se_utility = (
            shift + math.log(sd_scaled) - math.log(abs(q))
            - 0.5 * math.log(n_paths)
        )
        se_expected_utility = _exp_or_boundary(log_se_utility)

    log_ce = log_mean / q
    ce = _exp_or_boundary(log_ce)
    normalized_power = scaled / mean_scaled
    ce_influence = (ce / q) * (normalized_power - 1.0)
    se_ce = float(np.std(ce_influence, ddof=1) / math.sqrt(n_paths))
    return UtilityStats(
        expected_utility=expected_utility,
        se_expected_utility=se_expected_utility,
        ce=ce,
        se_ce=se_ce,
        log_mean_power=log_mean,
    )


def paired_welfare_statistics(
    learned_log_wealth: np.ndarray,
    optimal_log_wealth: np.ndarray,
    gamma: float,
) -> WelfareStats:
    """Compute WL and its paired delta-method MC SE under common paths."""
    learned = np.asarray(learned_log_wealth, dtype=np.float64).reshape(-1)
    optimal = np.asarray(optimal_log_wealth, dtype=np.float64).reshape(-1)
    if learned.shape != optimal.shape:
        raise ValueError(
            f"paired terminal samples must have equal shape, got {learned.shape} and {optimal.shape}"
        )
    q = 1.0 - float(gamma)
    power_a, mean_a, log_mean_a = _scaled_power(learned, q)
    power_o, mean_o, log_mean_o = _scaled_power(optimal, q)

    log_ratio = (log_mean_a - log_mean_o) / q
    ratio = _exp_or_boundary(log_ratio)
    wl = 1.0 - ratio  # Deliberately do not clip small negative MC outcomes.
    influence_log_ratio = (
        (power_a / mean_a - 1.0) - (power_o / mean_o - 1.0)
    ) / q
    influence_wl = -ratio * influence_log_ratio
    se_wl = float(np.std(influence_wl, ddof=1) / math.sqrt(learned.size))
    return WelfareStats(wl=wl, se_wl=se_wl, ce_ratio=ratio)


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("seed summary requires nonempty finite values")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, float("nan"), float("nan"), float("nan")
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(int(arr.size))
    half_width = t_crit_95(int(arr.size) - 1) * sem
    return mean, std, sem, mean - half_width, mean + half_width


# ---------------------------------------------------------------------------
# Strict run discovery and provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    model_type: str
    n_assets: int
    m_states: int
    seed: int
    group: str
    updated_at: str
    status: str
    config_args: Dict[str, Any]
    config_doc: Dict[str, Any]
    market_hash: str = ""


def normalize_models(text: str) -> List[str]:
    raw = str(text or "both").strip().lower()
    if raw == "both":
        return ["pinn", "pipinn"]
    result: List[str] = []
    aliases = {"pinn": "pinn", "pipinn": "pipinn", "pi-pinn": "pipinn"}
    for token in re.split(r"[\s,]+", raw):
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"unknown model in --models: {token!r}")
        model = aliases[token]
        if model not in result:
            result.append(model)
    if not result:
        raise ValueError("--models selected no methods")
    return result


def _read_config_doc(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{run_dir / 'config.json'} is not a JSON object")
    return value


def discover_paper_runs(
    out_root: Path,
    models: Sequence[str],
    m_states: Sequence[int],
    expected_seeds: Sequence[int],
    run_name_regex: str = "",
    allow_incomplete: bool = False,
    expected_n_assets: Optional[int] = None,
) -> Dict[Tuple[str, int], List[RunRecord]]:
    """Select exactly one configuration and the requested seeds per method/M.

    Reruns are deduplicated by taking the newest record for a given
    (configuration, seed) *before* checking status.  Thus an old ``_SUCCESS``
    cannot mask a newer failed attempt in another directory.
    """
    wanted_models = set(models)
    wanted_m = set(int(x) for x in m_states)
    expected = set(int(x) for x in expected_seeds)
    pattern = re.compile(run_name_regex) if run_name_regex else None
    wrong_asset_dimensions: List[Tuple[str, int, str]] = []
    nonaffine_runs: List[Tuple[str, int, str]] = []

    newest: Dict[Tuple[str, int], RunRecord] = {}
    for run_dir_text in find_runs(str(out_root)):
        run_dir = Path(run_dir_text)
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None:
            continue
        model = str(cfg.get("model_type", "")).lower()
        if model == "pi-pinn":
            model = "pipinn"
        m = _as_int(cfg.get("m_states"))
        n = _as_int(cfg.get("n_assets"))
        seed = _as_int(cfg.get("seed"))
        if model not in wanted_models or m not in wanted_m or n is None or seed is None:
            continue
        try:
            relative_name = str(run_dir.relative_to(out_root))
        except ValueError:
            relative_name = str(run_dir)
        if pattern and not pattern.search(relative_name):
            continue
        if expected_n_assets is not None and n != int(expected_n_assets):
            wrong_asset_dimensions.append((
                model,
                int(m),
                f"{relative_name}: N={n}, expected N={int(expected_n_assets)}",
            ))
            continue
        try:
            nonaffine_eps = float(cfg.get("nonaffine_eps", 0.0) or 0.0)
        except (TypeError, ValueError):
            nonaffine_runs.append((
                model,
                int(m),
                f"{relative_name}: invalid nonaffine_eps={cfg.get('nonaffine_eps')!r}",
            ))
            continue
        risk_premium_mode = str(cfg.get("risk_premium_mode", "affine")).strip().lower()
        if risk_premium_mode not in {"affine", "tanh"}:
            nonaffine_runs.append((
                model,
                int(m),
                f"{relative_name}: unsupported risk_premium_mode={risk_premium_mode!r}",
            ))
            continue
        # Match the training/reference gate exactly: any nonzero epsilon,
        # however small, is a genuinely non-affine experiment.  A numerical
        # tolerance here could silently apply an affine CE denominator to the
        # wrong model.
        if not math.isfinite(nonaffine_eps) or nonaffine_eps != 0.0:
            nonaffine_runs.append((
                model,
                int(m),
                f"{relative_name}: nonaffine_eps={nonaffine_eps:g}; "
                "the affine closed-form welfare denominator is unavailable",
            ))
            continue

        group, _canonical = group_key(cfg)
        config_doc = _read_config_doc(run_dir)
        record = RunRecord(
            run_dir=run_dir,
            model_type=model,
            n_assets=n,
            m_states=m,
            seed=seed,
            group=group,
            updated_at=run_updated_at(str(run_dir)),
            status=run_status(str(run_dir)),
            config_args=dict(cfg),
            config_doc=config_doc,
        )
        key = (group, seed)
        previous = newest.get(key)
        if previous is None or (record.updated_at, str(record.run_dir)) >= (
            previous.updated_at,
            str(previous.run_dir),
        ):
            newest[key] = record

    groups_by_cell: Dict[Tuple[str, int], Dict[str, List[RunRecord]]] = {}
    for record in newest.values():
        cell = (record.model_type, record.m_states)
        groups_by_cell.setdefault(cell, {}).setdefault(record.group, []).append(record)

    selected: Dict[Tuple[str, int], List[RunRecord]] = {}
    errors: List[str] = []
    for model in models:
        for m in m_states:
            cell = (model, int(m))
            # Failed/unknown attempts do not create a paper configuration and
            # do not contribute seed IDs.  We still deduplicate newest-first
            # above, then retain only groups with at least one newest SUCCESS.
            # This makes an extra failed seed harmless while an extra
            # successful seed remains a fatal exact-set violation.
            groups = {
                group: records
                for group, records in groups_by_cell.get(cell, {}).items()
                if any(record.status == "success" for record in records)
            }
            if len(groups) != 1:
                errors.append(
                    f"model={model}, M={m}: expected exactly one training configuration, "
                    f"found successful groups={sorted(groups)}; narrow with --run-name-regex"
                )
                continue
            group, records = next(iter(groups.items()))
            successful_by_seed = {
                record.seed: record for record in records if record.status == "success"
            }
            successful = set(successful_by_seed)
            if expected and successful != expected and not allow_incomplete:
                latest_non_success = {
                    record.seed: record.status
                    for record in records
                    if record.status != "success" and record.seed in expected
                }
                errors.append(
                    f"model={model}, M={m}, group={group}: successful seeds={sorted(successful)}, "
                    f"expected={sorted(expected)}, missing={sorted(expected - successful)}, "
                    f"extra successful={sorted(successful - expected)}, "
                    f"latest non-success requested={latest_non_success}"
                )
                continue
            seeds = sorted(expected & successful if expected else successful)
            if allow_incomplete and expected and successful != expected:
                print(
                    f"[warn] exploratory incomplete cell model={model}, M={m}: "
                    f"using seeds={seeds}; successful={sorted(successful)}"
                )
            if not seeds:
                errors.append(f"model={model}, M={m}: no requested seeds are successful")
                continue
            cell_records = [successful_by_seed[seed] for seed in seeds]
            selected[cell] = sorted(cell_records, key=lambda record: record.seed)

    missing_cells = {
        (str(model), int(m)) for model in models for m in m_states
    } - set(selected)
    relevant_wrong_n = [detail for model, m, detail in wrong_asset_dimensions
                        if (model, m) in missing_cells]
    relevant_nonaffine = [detail for model, m, detail in nonaffine_runs
                          if (model, m) in missing_cells]
    if relevant_wrong_n:
        errors.append(
            "asset-dimension mismatch:\n    " + "\n    ".join(sorted(relevant_wrong_n))
        )
    if relevant_nonaffine:
        errors.append(
            "non-affine runs are not valid for this closed-form CE/WL evaluator:\n    "
            + "\n    ".join(sorted(relevant_nonaffine))
        )
    if errors:
        raise ValueError("paper-run validation failed:\n  - " + "\n  - ".join(errors))

    # Market identity is a cross-method condition, so validate only after all
    # requested cells have passed exact seed/status checks.
    market_hashes_by_m: Dict[int, set[str]] = {}
    updated_selected: Dict[Tuple[str, int], List[RunRecord]] = {}
    for cell, records in selected.items():
        with_hash: List[RunRecord] = []
        for record in records:
            market_path = record.run_dir / "market_params.npz"
            try:
                digest = canonical_market_hash(str(market_path))
            except Exception as exc:
                raise ValueError(f"invalid market snapshot {market_path}: {exc}") from exc
            with_hash.append(replace(record, market_hash=digest))
            market_hashes_by_m.setdefault(record.m_states, set()).add(digest)
        updated_selected[cell] = with_hash
    for m, hashes in sorted(market_hashes_by_m.items()):
        if len(hashes) != 1:
            raise ValueError(
                f"M={m}: market snapshot differs across methods/seeds; hashes={sorted(hashes)}"
            )
    return updated_selected


def resolve_checkpoint(
    record: RunRecord,
    out_root: Path,
    allow_fallback: bool,
) -> Path:
    raw_weight_dir = record.config_doc.get("weight_dir") or record.config_args.get("weight_root")
    directory_candidates: List[Path] = []
    if raw_weight_dir:
        raw = Path(str(raw_weight_dir)).expanduser()
        if raw.is_absolute():
            directory_candidates.append(raw)
        else:
            recorded_cwd = record.config_doc.get("cwd")
            if recorded_cwd:
                directory_candidates.append(Path(str(recorded_cwd)) / raw)
            directory_candidates.extend([record.run_dir / raw, out_root / raw, raw])
    method_dir = "pinn" if record.model_type == "pinn" else "pi-pinn"
    directory_candidates.append(out_root / "weights" / method_dir / record.run_dir.name)

    unique_dirs: List[Path] = []
    seen: set[str] = set()
    for directory in directory_candidates:
        key = str(directory)
        if key not in seen:
            unique_dirs.append(directory)
            seen.add(key)

    names = ["value_net_final.pt"]
    if allow_fallback:
        names.extend(["value_net_last.pt", "value_net_best.pt"])
    checked: List[str] = []
    for name in names:
        for directory in unique_dirs:
            path = directory / name
            checked.append(str(path))
            if path.is_file():
                if name != "value_net_final.pt":
                    print(f"[warn] legacy checkpoint fallback for {record.run_dir}: {path}")
                return path
    suffix = " (fallback disabled)" if not allow_fallback else ""
    raise FileNotFoundError(
        f"official final checkpoint not found for {record.run_dir}{suffix}; checked:\n  "
        + "\n  ".join(checked)
    )


# ---------------------------------------------------------------------------
# Market and closed-form objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    K: np.ndarray
    xbar: np.ndarray
    SigmaX: np.ndarray
    rho: np.ndarray
    Lam: np.ndarray
    Gamma: np.ndarray
    lam0: np.ndarray
    X_min: np.ndarray
    X_max: np.ndarray
    gamma: float
    r: float
    horizon: float
    W_min: float
    W_max: float
    market_seed: int
    joint_cholesky: np.ndarray

    @property
    def n_assets(self) -> int:
        return int(self.lam0.size)

    @property
    def m_states(self) -> int:
        return int(self.xbar.size)


@dataclass(frozen=True)
class ClosedFormData:
    t: np.ndarray
    y: np.ndarray
    m_states: int

    def coefficients(self, tau: float) -> Tuple[float, np.ndarray, np.ndarray]:
        state = np.asarray(
            [np.interp(float(tau), self.t, row) for row in self.y], dtype=np.float64
        )
        m = self.m_states
        a = float(state[0])
        b = state[1:1 + m]
        C = state[1 + m:].reshape(m, m)
        C = 0.5 * (C + C.T)
        return a, b, C


def load_market(path: Path) -> MarketData:
    with np.load(path, allow_pickle=False) as source:
        values = {key: np.asarray(source[key]).copy() for key in source.files}
    arrays = {
        key: np.asarray(values[key], dtype=np.float64)
        for key in (
            "K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0",
            "lam0", "X_min", "X_max",
        )
    }
    xbar = arrays["xbar"].reshape(-1)
    lam0 = arrays["lam0"].reshape(-1)
    m = int(xbar.size)
    n = int(lam0.size)
    expected_shapes = {
        "K": (m, m),
        "SigmaX": (m, m),
        "rho": (n, m),
        "Lam": (n, m),
        "Gamma": (n, m),
        "Q": (m, m),
        "k0": (m,),
        "X_min": (m,),
        "X_max": (m,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} shape={arrays[key].shape}, expected={shape}")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError(f"{path}: market snapshot contains a nonfinite value")
    if not np.allclose(arrays["Gamma"], arrays["rho"] @ arrays["SigmaX"].T,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: Gamma != rho @ SigmaX.T")
    if not np.allclose(arrays["Q"], arrays["SigmaX"] @ arrays["SigmaX"].T,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: Q != SigmaX @ SigmaX.T")
    if not np.allclose(arrays["k0"], arrays["K"] @ xbar,
                       rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{path}: k0 != K @ xbar")

    gamma = _scalar(values, "gamma")
    horizon = _scalar(values, "tau_max")
    w_min = _scalar(values, "W_min")
    w_max = _scalar(values, "W_max")
    if abs(1.0 - gamma) < 1.0e-12:
        raise ValueError("gamma=1 requires a separate log-utility CE implementation")
    if horizon <= 0.0 or w_min <= 0.0 or w_max <= w_min:
        raise ValueError(f"{path}: invalid horizon or wealth bounds")
    if np.any(arrays["X_max"] <= arrays["X_min"]):
        raise ValueError(f"{path}: invalid state bounds")

    try:
        validate_market_snapshot(values)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid market snapshot: {exc}") from exc

    joint = np.block([
        [np.eye(n), arrays["rho"]],
        [arrays["rho"].T, np.eye(m)],
    ])
    joint = 0.5 * (joint + joint.T)
    min_eigenvalue = float(np.linalg.eigvalsh(joint)[0])
    if min_eigenvalue <= 0.0:
        raise ValueError(
            f"{path}: joint innovation covariance is not positive definite "
            f"(min eigenvalue={min_eigenvalue:.3e}); no silent jitter is applied"
        )
    joint_cholesky = np.linalg.cholesky(joint)

    market_seed_float = _scalar(values, "market_seed")
    market_seed = int(round(market_seed_float))
    if not math.isclose(market_seed_float, market_seed, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{path}: market_seed must be integer-valued")

    return MarketData(
        K=arrays["K"],
        xbar=xbar,
        SigmaX=arrays["SigmaX"],
        rho=arrays["rho"],
        Lam=arrays["Lam"],
        Gamma=arrays["Gamma"],
        lam0=lam0,
        X_min=arrays["X_min"],
        X_max=arrays["X_max"],
        gamma=gamma,
        r=_scalar(values, "r"),
        horizon=horizon,
        W_min=w_min,
        W_max=w_max,
        market_seed=market_seed,
        joint_cholesky=joint_cholesky,
    )


def load_closed_form(path: Path, m_states: int) -> ClosedFormData:
    with np.load(path, allow_pickle=False) as source:
        if "success" in source.files and not bool(np.asarray(source["success"]).reshape(-1)[0]):
            raise ValueError(f"closed-form ODE solve was not successful: {path}")
        t = np.asarray(source["t"], dtype=np.float64).copy()
        y = np.asarray(source["y"], dtype=np.float64).copy()
    expected_rows = 1 + m_states + m_states * m_states
    if t.ndim != 1 or y.shape != (expected_rows, t.size):
        raise ValueError(
            f"{path}: expected t=(L,), y=({expected_rows},L), got t={t.shape}, y={y.shape}"
        )
    if (t.size < 2 or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0.0) or not np.all(np.isfinite(y))):
        raise ValueError(f"{path}: invalid closed-form interpolation grid")
    if not math.isclose(float(t[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{path}: closed-form tau grid must begin at zero")
    if float(np.max(np.abs(y[:, 0]))) > 1.0e-10:
        raise ValueError(f"{path}: closed-form terminal coefficients at tau=0 are not zero")
    return ClosedFormData(t=t, y=y, m_states=m_states)


def canonical_closed_form_hash(closed_form: ClosedFormData) -> str:
    """Hash the complete interpolation grid used by the optimal policy.

    Equality of the scalar CE at ``xbar`` is not enough: different ``b`` or
    ``C`` coefficients can agree at that single state while inducing a
    different pathwise optimal policy.  Both arrays are already normalized
    to finite float64 by ``load_closed_form``; include names and shapes to
    make the identity unambiguous.
    """
    digest = hashlib.sha256()
    digest.update(f"m_states={closed_form.m_states}\0".encode("ascii"))
    for name, values in (("t", closed_form.t), ("y", closed_form.y)):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def exact_optimal_statistics(
    market: MarketData,
    closed_form: ClosedFormData,
    w0: float,
) -> Tuple[float, float]:
    q = 1.0 - market.gamma
    a, b, C = closed_form.coefficients(market.horizon)
    log_phi = float(a + b @ market.xbar + 0.5 * market.xbar @ C @ market.xbar)
    log_mean_power = q * (market.r * market.horizon + math.log(w0)) + log_phi
    log_ce = log_mean_power / q
    expected_utility = math.copysign(
        _exp_or_boundary(log_mean_power - math.log(abs(q))), q
    )
    return expected_utility, _exp_or_boundary(log_ce)


@dataclass(frozen=True)
class DimensionContext:
    representative: RunRecord
    market: MarketData
    closed_form: ClosedFormData


def _check_config_scalar(
    record: RunRecord,
    key: str,
    expected: float,
    *,
    atol: float = 1.0e-12,
) -> None:
    if key not in record.config_args or record.config_args[key] is None:
        return
    try:
        actual = float(record.config_args[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.run_dir}: config {key!r} is not numeric") from exc
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=atol):
        raise ValueError(
            f"{record.run_dir}: config {key}={actual} disagrees with market snapshot {expected}"
        )


def validate_numpy_inputs(
    selected: Mapping[Tuple[str, int], Sequence[RunRecord]],
    models: Sequence[str],
    m_states: Sequence[int],
    w0: float,
) -> Dict[int, DimensionContext]:
    """Semantically validate every market/ODE pair without importing torch."""
    representative_by_m: Dict[int, RunRecord] = {}
    for m in m_states:
        for model in models:
            records = selected.get((model, int(m)), ())
            if records:
                representative_by_m[int(m)] = records[0]
                break
        if int(m) not in representative_by_m:
            raise ValueError(f"M={m}: no representative run")

    contexts: Dict[int, DimensionContext] = {}
    exact_ce_by_m: Dict[int, float] = {}
    closed_form_hash_by_m: Dict[int, str] = {}
    for cell in sorted(selected):
        for record in selected[cell]:
            market_path = record.run_dir / "market_params.npz"
            closed_form_path = record.run_dir / "closed_form_ode.npz"
            market = load_market(market_path)
            if market.m_states != record.m_states or market.n_assets != record.n_assets:
                raise ValueError(
                    f"{market_path}: dimensions (N={market.n_assets},M={market.m_states}) "
                    f"disagree with config (N={record.n_assets},M={record.m_states})"
                )
            _check_config_scalar(record, "gamma", market.gamma)
            _check_config_scalar(record, "r", market.r)
            _check_config_scalar(record, "tau_max", market.horizon)
            _check_config_scalar(record, "w_min", market.W_min)
            _check_config_scalar(record, "w_max", market.W_max)
            if record.config_args.get("market_seed") is not None:
                _check_config_scalar(record, "market_seed", float(market.market_seed), atol=1.0e-9)

            closed_form = load_closed_form(closed_form_path, record.m_states)
            if market.horizon > float(closed_form.t[-1]) + 1.0e-12:
                raise ValueError(
                    f"{closed_form_path}: ODE grid ends at {closed_form.t[-1]}, "
                    f"before horizon {market.horizon}"
                )
            exact_utility, exact_ce = exact_optimal_statistics(market, closed_form, w0)
            if not math.isfinite(exact_utility) or not math.isfinite(exact_ce) or exact_ce <= 0.0:
                raise ValueError(f"{closed_form_path}: invalid exact CRRA utility/CE")
            prior_ce = exact_ce_by_m.setdefault(record.m_states, exact_ce)
            if not math.isclose(exact_ce, prior_ce, rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise ValueError(
                    f"M={record.m_states}: closed-form exact CE differs across selected runs "
                    f"({prior_ce} vs {exact_ce})"
                )
            closed_form_hash = canonical_closed_form_hash(closed_form)
            prior_hash = closed_form_hash_by_m.setdefault(record.m_states, closed_form_hash)
            if closed_form_hash != prior_hash:
                raise ValueError(
                    f"M={record.m_states}: closed-form coefficient grids differ across "
                    f"selected runs ({prior_hash} vs {closed_form_hash}); refusing to "
                    "choose one pathwise optimal-policy denominator"
                )
            if record == representative_by_m[record.m_states]:
                contexts[record.m_states] = DimensionContext(record, market, closed_form)

    missing = set(int(m) for m in m_states) - set(contexts)
    if missing:
        raise ValueError(f"missing semantically validated dimension contexts: {sorted(missing)}")
    return contexts


def build_resume_signature(
    *,
    args: argparse.Namespace,
    selected: Mapping[Tuple[str, int], Sequence[RunRecord]],
    checkpoints: Mapping[Tuple[str, int, int], Path],
    models: Sequence[str],
    m_states: Sequence[int],
    expected_seeds: Sequence[int],
    extensions: Sequence[str],
) -> Tuple[str, Dict[str, Any], Dict[Tuple[str, int, int], Dict[str, Any]]]:
    checkpoint_meta: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    run_payload: List[Dict[str, Any]] = []
    for cell in sorted(selected):
        for record in sorted(selected[cell], key=lambda item: item.seed):
            key = (record.model_type, record.m_states, record.seed)
            checkpoint = checkpoints[key]
            checkpoint_stat = checkpoint.stat()
            closed_form_path = record.run_dir / "closed_form_ode.npz"
            metadata = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "size": int(checkpoint_stat.st_size),
                "mtime_ns": int(checkpoint_stat.st_mtime_ns),
            }
            checkpoint_meta[key] = metadata
            run_payload.append({
                "model_type": record.model_type,
                "M": record.m_states,
                "training_seed": record.seed,
                "group": record.group,
                "run_dir": str(record.run_dir),
                "market_hash": record.market_hash,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": metadata["sha256"],
                "checkpoint_size": metadata["size"],
                "closed_form_path": str(closed_form_path),
                "closed_form_sha256": sha256_file(closed_form_path),
            })

    payload: Dict[str, Any] = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "protocol": {
            "models": list(models),
            "m_states": [int(m) for m in m_states],
            "expected_seeds": [int(seed) for seed in expected_seeds],
            "selected_seeds": {
                f"{model}:M{m}": [record.seed for record in selected[(model, int(m))]]
                for model in models for m in m_states
            },
            "allow_incomplete": bool(args.allow_incomplete),
            "run_name_regex": str(args.run_name_regex),
            "n_paths": int(args.n_paths),
            "n_steps": int(args.n_steps),
            "w0": float(args.w0),
            "x0": "xbar",
            "mc_seed": int(args.mc_seed),
            # The RNG is consumed path-batch first, so path_batch is part of
            # the actual common-random-number protocol, not merely tuning.
            "path_batch": int(args.path_batch),
            "policy_chunk": int(args.policy_chunk),
            "extensions": list(extensions),
            "checkpoint_fallback_allowed": bool(args.allow_checkpoint_fallback),
            # The requested spelling (``auto`` versus ``cuda:0``) is not
            # part of the scientific protocol.  Once learned-policy work
            # actually starts, the effective device and library versions are
            # recorded and checked separately before a partial resume.
            "torch_num_threads": int(args.torch_num_threads),
            "network_dtype": "float32",
            "simulation_dtype": "float64",
            "vww_guard": VWW_GUARD,
            "wealth_scheme": "log-wealth Euler",
            "joint_covariance": "[[I_N,rho],[rho.T,I_M]]",
        },
        "runs": run_payload,
    }
    return canonical_json_hash(payload), payload, checkpoint_meta


def verify_checkpoint_unchanged(path: Path, metadata: Mapping[str, Any]) -> None:
    stat = path.stat()
    if int(stat.st_size) != int(metadata["size"]) or int(stat.st_mtime_ns) != int(metadata["mtime_ns"]):
        raise RuntimeError(
            f"checkpoint changed after resume signature was computed: {path}; restart evaluation"
        )


# ---------------------------------------------------------------------------
# Simulation and lazy PyTorch policy evaluation
# ---------------------------------------------------------------------------


@dataclass
class SimulationDiagnostics:
    state_exit_step_frac: float
    state_exit_path_frac: float
    wealth_exit_step_frac: float
    wealth_exit_path_frac: float
    projection_step_frac: float
    vww_guard_frac: float
    max_policy_norm: float


@dataclass
class SimulationResult:
    terminal_log_wealth: np.ndarray
    terminal_state: np.ndarray
    diagnostics: SimulationDiagnostics


@dataclass
class _DiagnosticCounts:
    state_exit_steps: int = 0
    state_exit_paths: int = 0
    wealth_exit_steps: int = 0
    wealth_exit_paths: int = 0
    projection_steps: int = 0
    guard_points: int = 0
    max_policy_norm: float = 0.0


def _finalize_diagnostics(
    counts: _DiagnosticCounts,
    n_paths: int,
    n_steps: int,
) -> SimulationDiagnostics:
    observations = float(n_paths * n_steps)
    return SimulationDiagnostics(
        state_exit_step_frac=counts.state_exit_steps / observations,
        state_exit_path_frac=counts.state_exit_paths / float(n_paths),
        wealth_exit_step_frac=counts.wealth_exit_steps / observations,
        wealth_exit_path_frac=counts.wealth_exit_paths / float(n_paths),
        projection_step_frac=counts.projection_steps / observations,
        vww_guard_frac=counts.guard_points / observations,
        max_policy_norm=counts.max_policy_norm,
    )


def _outside_state(x: np.ndarray, market: MarketData) -> np.ndarray:
    return np.any((x < market.X_min) | (x > market.X_max), axis=1)


def _precompute_closed_form_coefficients(
    closed_form: ClosedFormData,
    horizon: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dt = horizon / n_steps
    taus = horizon - np.arange(n_steps, dtype=np.float64) * dt
    b_values = np.empty((n_steps, closed_form.m_states), dtype=np.float64)
    C_values = np.empty((n_steps, closed_form.m_states, closed_form.m_states), dtype=np.float64)
    for index, tau in enumerate(taus):
        _a, b_values[index], C_values[index] = closed_form.coefficients(float(tau))
    return b_values, C_values


def simulate_optimal_policy(
    market: MarketData,
    closed_form: ClosedFormData,
    *,
    n_paths: int,
    n_steps: int,
    path_batch: int,
    w0: float,
    mc_seed: int,
) -> SimulationResult:
    dt = market.horizon / n_steps
    sqrt_dt = math.sqrt(dt)
    b_values, C_values = _precompute_closed_form_coefficients(
        closed_form, market.horizon, n_steps
    )
    terminal_y = np.empty(n_paths, dtype=np.float64)
    terminal_x = np.empty((n_paths, market.m_states), dtype=np.float64)
    counts = _DiagnosticCounts()
    rng = np.random.default_rng(mc_seed)
    log_w_min, log_w_max = math.log(market.W_min), math.log(market.W_max)

    for start in range(0, n_paths, path_batch):
        stop = min(start + path_batch, n_paths)
        batch = stop - start
        x = np.broadcast_to(market.xbar, (batch, market.m_states)).copy()
        y = np.full(batch, math.log(w0), dtype=np.float64)
        state_ever = np.zeros(batch, dtype=bool)
        wealth_ever = np.zeros(batch, dtype=bool)

        for step in range(n_steps):
            state_exit = _outside_state(x, market)
            wealth_exit = (y < log_w_min) | (y > log_w_max)
            counts.state_exit_steps += int(np.count_nonzero(state_exit))
            counts.wealth_exit_steps += int(np.count_nonzero(wealth_exit))
            state_ever |= state_exit
            wealth_ever |= wealth_exit

            lam_x = market.lam0 + x @ market.Lam.T
            grad_log_phi = b_values[step] + x @ C_values[step].T
            policy = (lam_x + grad_log_phi @ market.Gamma.T) / market.gamma
            if not np.all(np.isfinite(policy)):
                raise FloatingPointError(f"closed-form policy became nonfinite at Euler step {step}")
            counts.max_policy_norm = max(
                counts.max_policy_norm,
                float(np.max(np.linalg.norm(policy, axis=1))),
            )

            standard = rng.standard_normal((batch, market.n_assets + market.m_states))
            innovation = standard @ market.joint_cholesky.T
            xi_r = innovation[:, :market.n_assets]
            xi_x = innovation[:, market.n_assets:]
            y += (
                market.r + np.sum(policy * lam_x, axis=1)
                - 0.5 * np.sum(policy * policy, axis=1)
            ) * dt + np.sum(policy * xi_r, axis=1) * sqrt_dt
            x += (market.xbar - x) @ market.K.T * dt + xi_x @ market.SigmaX.T * sqrt_dt

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
            raise FloatingPointError("closed-form Euler simulation produced a nonfinite state")
        # Step fractions refer to the n_steps policy-evaluation times.  Path
        # fractions mean "ever outside on [0,T]", so include the terminal
        # state even though no policy is evaluated there.
        state_ever |= _outside_state(x, market)
        wealth_ever |= (y < log_w_min) | (y > log_w_max)
        counts.state_exit_paths += int(np.count_nonzero(state_ever))
        counts.wealth_exit_paths += int(np.count_nonzero(wealth_ever))
        terminal_y[start:stop] = y
        terminal_x[start:stop] = x

    return SimulationResult(
        terminal_log_wealth=terminal_y,
        terminal_state=terminal_x,
        diagnostics=_finalize_diagnostics(counts, n_paths, n_steps),
    )


def import_torch(device_spec: str, torch_num_threads: int) -> Tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for learned-policy simulation. Install/use the "
            "training environment; run discovery and NumPy statistics do not require it."
        ) from exc
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    spec = str(device_spec or "auto").strip().lower()
    if spec in {"", "auto"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if spec.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device={device_spec}, but CUDA is unavailable")
        device = torch.device(device_spec)
    if device.type == "cuda":
        # Canonicalize the default CUDA device so ``auto`` and an explicit
        # ``cuda:0`` compare equal during a later partial resume, and prove
        # the selected logical index is usable before any output is mutated.
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        count = int(torch.cuda.device_count())
        if index < 0 or index >= count:
            raise RuntimeError(
                f"requested CUDA index {index}, but only {count} logical device(s) are visible"
            )
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
                layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, w: Any, x: Any, tau: Any) -> Any:
            return self.net(torch.cat([w, x, tau], dim=1))

    return ValueNetND()


def load_value_network(
    record: RunRecord,
    checkpoint: Path,
    torch: Any,
    nn: Any,
    device: Any,
) -> Any:
    hidden = int(record.config_args.get("value_hidden", 256))
    depth = int(record.config_args.get("value_depth", 3))
    model = build_value_network(torch, nn, record.m_states, hidden, depth).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # older supported PyTorch releases
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain a state dict: {checkpoint}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_network_policy(
    model: Any,
    market: MarketData,
    *,
    x: np.ndarray,
    log_wealth: np.ndarray,
    tau: float,
    extension: str,
    policy_chunk: int,
    torch: Any,
    device: Any,
    market_tensors: Tuple[Any, Any, Any],
) -> Tuple[np.ndarray, int]:
    if extension not in {"projected", "raw"}:
        raise ValueError(f"unknown network-policy extension: {extension}")
    n_points = int(x.shape[0])
    result = np.empty((n_points, market.n_assets), dtype=np.float64)
    guard_count = 0
    log_float32_max = math.log(float(np.finfo(np.float32).max))

    for start in range(0, n_points, policy_chunk):
        stop = min(start + policy_chunk, n_points)
        x_raw = x[start:stop]
        y_raw = log_wealth[start:stop]
        if extension == "projected":
            x_eval = np.clip(x_raw, market.X_min, market.X_max)
            w_eval = np.exp(np.clip(y_raw, math.log(market.W_min), math.log(market.W_max)))
        else:
            if np.any(y_raw > log_float32_max):
                raise FloatingPointError(
                    "raw policy wealth exceeds float32 network range; this is an "
                    "unprojected-policy instability and is not silently clipped"
                )
            w_eval = np.exp(y_raw)
            if not np.all(np.isfinite(w_eval)):
                raise FloatingPointError("raw policy wealth is nonfinite")
            x_eval = x_raw

        w_t = torch.as_tensor(w_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        x_t = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
        tau_t = torch.full((stop - start, 1), float(tau), dtype=torch.float32, device=device)
        w_t.requires_grad_(True)
        x_t.requires_grad_(True)

        value = model(w_t, x_t, tau_t)
        value_w = torch.autograd.grad(
            value, w_t, grad_outputs=torch.ones_like(value), create_graph=True, retain_graph=True
        )[0]
        value_wx = torch.autograd.grad(
            value_w, x_t, grad_outputs=torch.ones_like(value_w),
            create_graph=False, retain_graph=True
        )[0]
        value_ww = torch.autograd.grad(
            value_w, w_t, grad_outputs=torch.ones_like(value_w),
            create_graph=False, retain_graph=False
        )[0]
        value_ww_safe = torch.clamp(value_ww, max=-VWW_GUARD)
        guard_count += int(torch.count_nonzero(value_ww > -VWW_GUARD).item())

        lam0_t, Lam_t, Gamma_t = market_tensors
        lam_eval = lam0_t.unsqueeze(0) + x_t @ Lam_t.T
        numerator = lam_eval * value_w + value_wx @ Gamma_t.T
        # For projected extension w_t is the projected wealth by construction.
        policy = -numerator / (w_t * value_ww_safe)
        result[start:stop] = policy.detach().cpu().numpy().astype(np.float64, copy=False)
        del value, value_w, value_wx, value_ww, value_ww_safe, policy

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("network greedy policy became nonfinite")
    return result, guard_count


def simulate_network_policy(
    model: Any,
    market: MarketData,
    *,
    extension: str,
    n_paths: int,
    n_steps: int,
    path_batch: int,
    policy_chunk: int,
    w0: float,
    mc_seed: int,
    torch: Any,
    device: Any,
) -> SimulationResult:
    dt = market.horizon / n_steps
    sqrt_dt = math.sqrt(dt)
    terminal_y = np.empty(n_paths, dtype=np.float64)
    terminal_x = np.empty((n_paths, market.m_states), dtype=np.float64)
    counts = _DiagnosticCounts()
    rng = np.random.default_rng(mc_seed)
    log_w_min, log_w_max = math.log(market.W_min), math.log(market.W_max)
    market_tensors = (
        torch.as_tensor(market.lam0, dtype=torch.float32, device=device),
        torch.as_tensor(market.Lam, dtype=torch.float32, device=device),
        torch.as_tensor(market.Gamma, dtype=torch.float32, device=device),
    )

    for start in range(0, n_paths, path_batch):
        stop = min(start + path_batch, n_paths)
        batch = stop - start
        x = np.broadcast_to(market.xbar, (batch, market.m_states)).copy()
        y = np.full(batch, math.log(w0), dtype=np.float64)
        state_ever = np.zeros(batch, dtype=bool)
        wealth_ever = np.zeros(batch, dtype=bool)

        for step in range(n_steps):
            state_exit = _outside_state(x, market)
            wealth_exit = (y < log_w_min) | (y > log_w_max)
            counts.state_exit_steps += int(np.count_nonzero(state_exit))
            counts.wealth_exit_steps += int(np.count_nonzero(wealth_exit))
            state_ever |= state_exit
            wealth_ever |= wealth_exit
            if extension == "projected":
                counts.projection_steps += int(np.count_nonzero(state_exit | wealth_exit))

            tau = market.horizon - step * dt
            policy, guarded = evaluate_network_policy(
                model,
                market,
                x=x,
                log_wealth=y,
                tau=tau,
                extension=extension,
                policy_chunk=policy_chunk,
                torch=torch,
                device=device,
                market_tensors=market_tensors,
            )
            counts.guard_points += guarded
            counts.max_policy_norm = max(
                counts.max_policy_norm,
                float(np.max(np.linalg.norm(policy, axis=1))),
            )

            # The extension changes how the control is obtained.  Wealth still
            # evolves under the actual, unprojected market state X_k.
            lam_actual = market.lam0 + x @ market.Lam.T
            standard = rng.standard_normal((batch, market.n_assets + market.m_states))
            innovation = standard @ market.joint_cholesky.T
            xi_r = innovation[:, :market.n_assets]
            xi_x = innovation[:, market.n_assets:]
            y += (
                market.r + np.sum(policy * lam_actual, axis=1)
                - 0.5 * np.sum(policy * policy, axis=1)
            ) * dt + np.sum(policy * xi_r, axis=1) * sqrt_dt
            x += (market.xbar - x) @ market.K.T * dt + xi_x @ market.SigmaX.T * sqrt_dt

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
            raise FloatingPointError(
                f"{extension} network-policy Euler simulation produced a nonfinite state"
            )
        state_ever |= _outside_state(x, market)
        wealth_ever |= (y < log_w_min) | (y > log_w_max)
        counts.state_exit_paths += int(np.count_nonzero(state_ever))
        counts.wealth_exit_paths += int(np.count_nonzero(wealth_ever))
        terminal_y[start:stop] = y
        terminal_x[start:stop] = x

    return SimulationResult(
        terminal_log_wealth=terminal_y,
        terminal_state=terminal_x,
        diagnostics=_finalize_diagnostics(counts, n_paths, n_steps),
    )


_DIAGNOSTIC_FIELDS = tuple(SimulationDiagnostics.__dataclass_fields__)


def save_optimal_cache(
    path: Path,
    result: SimulationResult,
    *,
    resume_signature: str,
    m_states: int,
    n_paths: int,
    n_steps: int,
    mc_seed: int,
) -> None:
    """Atomically cache paired optimal terminal samples and diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    values: Dict[str, Any] = {
        "cache_schema_version": np.asarray([1], dtype=np.int64),
        "resume_signature": np.asarray([resume_signature]),
        "M": np.asarray([m_states], dtype=np.int64),
        "n_paths": np.asarray([n_paths], dtype=np.int64),
        "n_steps": np.asarray([n_steps], dtype=np.int64),
        "mc_seed": np.asarray([mc_seed], dtype=np.int64),
        "terminal_log_wealth": np.asarray(result.terminal_log_wealth, dtype=np.float64),
        "terminal_state": np.asarray(result.terminal_state, dtype=np.float64),
    }
    for field in _DIAGNOSTIC_FIELDS:
        values[f"diag_{field}"] = np.asarray(
            [getattr(result.diagnostics, field)], dtype=np.float64
        )
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def load_optimal_cache(
    path: Path,
    *,
    resume_signature: str,
    m_states: int,
    n_paths: int,
    n_steps: int,
    mc_seed: int,
) -> Optional[SimulationResult]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as source:
            def scalar(key: str) -> Any:
                return np.asarray(source[key]).reshape(-1)[0].item()

            if int(scalar("cache_schema_version")) != 1:
                raise ValueError("unknown cache schema")
            if str(scalar("resume_signature")) != resume_signature:
                raise ValueError("resume signature mismatch")
            if (
                int(scalar("M")) != m_states
                or int(scalar("n_paths")) != n_paths
                or int(scalar("n_steps")) != n_steps
                or int(scalar("mc_seed")) != mc_seed
            ):
                raise ValueError("cached protocol dimensions mismatch")
            terminal_y = np.asarray(source["terminal_log_wealth"], dtype=np.float64).copy()
            terminal_x = np.asarray(source["terminal_state"], dtype=np.float64).copy()
            diagnostic_values = {
                field: float(scalar(f"diag_{field}")) for field in _DIAGNOSTIC_FIELDS
            }
        if terminal_y.shape != (n_paths,) or terminal_x.shape != (n_paths, m_states):
            raise ValueError(
                f"cached terminal shapes are {terminal_y.shape}/{terminal_x.shape}, "
                f"expected {(n_paths,)}/{(n_paths, m_states)}"
            )
        if not np.all(np.isfinite(terminal_y)) or not np.all(np.isfinite(terminal_x)):
            raise ValueError("cached terminal samples are nonfinite")
        if any(not math.isfinite(value) for value in diagnostic_values.values()):
            raise ValueError("cached diagnostics are nonfinite")
        for field in _DIAGNOSTIC_FIELDS:
            if field.endswith("_frac") and not (0.0 <= diagnostic_values[field] <= 1.0):
                raise ValueError(f"cached fraction {field} is outside [0,1]")
        if diagnostic_values["max_policy_norm"] < 0.0:
            raise ValueError("cached max_policy_norm is negative")
        return SimulationResult(
            terminal_log_wealth=terminal_y,
            terminal_state=terminal_x,
            diagnostics=SimulationDiagnostics(**diagnostic_values),
        )
    except Exception as exc:
        print(f"[warn] ignoring invalid optimal cache {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def welfare_row(
    *,
    model_type: str,
    training_seed: Any,
    m_states: int,
    policy: str,
    extension: str,
    mc_seed: int,
    n_paths: int,
    n_steps: int,
    dt: float,
    utility: UtilityStats,
    welfare: WelfareStats,
    diagnostics: SimulationDiagnostics,
) -> Dict[str, Any]:
    return {
        "model_type": model_type,
        "training_seed": training_seed,
        "M": m_states,
        "policy": policy,
        "extension": extension,
        "mc_seed": mc_seed,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "dt": dt,
        "expected_utility": utility.expected_utility,
        "se_expected_utility": utility.se_expected_utility,
        "ce": utility.ce,
        "se_ce": utility.se_ce,
        "wl": welfare.wl,
        "se_wl": welfare.se_wl,
        **asdict(diagnostics),
    }


WelfareKey = Tuple[str, int, Optional[int], str, str]


def welfare_row_key(row: Mapping[str, Any]) -> Optional[WelfareKey]:
    try:
        model = str(row["model_type"])
        m_states = int(float(row["M"]))
        policy = str(row["policy"])
        extension = str(row["extension"])
        seed_text = str(row.get("training_seed", "")).strip()
        seed = None if seed_text == "" else int(float(seed_text))
    except (KeyError, TypeError, ValueError):
        return None
    return model, m_states, seed, policy, extension


def expected_welfare_keys(
    selected: Mapping[Tuple[str, int], Sequence[RunRecord]],
    m_states: Sequence[int],
    extensions: Sequence[str],
) -> set[WelfareKey]:
    keys: set[WelfareKey] = {
        ("closed_form", int(m), None, "optimal", "optimal") for m in m_states
    }
    for records in selected.values():
        for record in records:
            for extension in extensions:
                keys.add((
                    record.model_type,
                    record.m_states,
                    record.seed,
                    "greedy_final",
                    extension,
                ))
    return keys


def _completed_row_is_valid(
    row: Mapping[str, Any],
    key: WelfareKey,
    *,
    args: argparse.Namespace,
    contexts: Mapping[int, DimensionContext],
) -> bool:
    if any(field not in row for field in WELFARE_FIELDS):
        return False
    numeric_fields = [
        field for field in WELFARE_FIELDS
        if field not in {"model_type", "training_seed", "policy", "extension"}
    ]
    try:
        numbers = {field: float(row[field]) for field in numeric_fields}
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in numbers.values()):
        return False
    m = key[1]
    expected_dt = contexts[m].market.horizon / int(args.n_steps)
    fractions_valid = all(
        0.0 <= numbers[field] <= 1.0
        for field in (
            "state_exit_step_frac", "state_exit_path_frac",
            "wealth_exit_step_frac", "wealth_exit_path_frac",
            "projection_step_frac", "vww_guard_frac",
        )
    )
    return (
        math.isclose(numbers["M"], m, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(numbers["mc_seed"], int(args.mc_seed), rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(numbers["n_paths"], int(args.n_paths), rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(numbers["n_steps"], int(args.n_steps), rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(numbers["dt"], expected_dt, rel_tol=1.0e-12, abs_tol=1.0e-15)
        and numbers["ce"] > 0.0
        and numbers["se_expected_utility"] >= 0.0
        and numbers["se_ce"] >= 0.0
        and numbers["se_wl"] >= 0.0
        and fractions_valid
        and numbers["max_policy_norm"] >= 0.0
    )


def load_completed_rows(
    path: Path,
    *,
    expected_keys: set[WelfareKey],
    args: argparse.Namespace,
    contexts: Mapping[int, DimensionContext],
) -> Dict[WelfareKey, Dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: Dict[WelfareKey, Dict[str, Any]] = {}
    ignored = 0
    duplicates = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = welfare_row_key(row)
            if (
                key is None
                or key not in expected_keys
                or not _completed_row_is_valid(row, key, args=args, contexts=contexts)
            ):
                ignored += 1
                continue
            if key in completed:
                duplicates += 1
            # Keep the last complete row, matching incremental append/rewrite
            # semantics from older interrupted evaluator versions.
            completed[key] = dict(row)
    if ignored:
        print(f"[resume] ignored {ignored} incomplete/unexpected welfare_metrics row(s)")
    if duplicates:
        print(f"[resume] collapsed {duplicates} duplicate welfare_metrics row(s)")
    return completed


def sorted_welfare_rows(rows: Mapping[WelfareKey, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Tuple[WelfareKey, Mapping[str, Any]]) -> Tuple[Any, ...]:
        key = item[0]
        model_order = {"closed_form": 0, "pinn": 1, "pipinn": 2}
        extension_order = {"optimal": 0, "projected": 1, "raw": 2}
        return (
            key[1],
            model_order.get(key[0], 99),
            -1 if key[2] is None else key[2],
            extension_order.get(key[4], 99),
        )

    return [dict(row) for _key, row in sorted(rows.items(), key=sort_key)]


def build_seed_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row["model_type"] == "closed_form":
            continue
        key = (
            str(row["model_type"]), int(row["M"]), str(row["policy"]), str(row["extension"])
        )
        grouped.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in ordered]
        for metric in ("expected_utility", "ce", "wl"):
            mean, std, sem, ci_low, ci_high = mean_std_ci(
                [float(row[metric]) for row in ordered]
            )
            output.append({
                "model_type": key[0],
                "M": key[1],
                "policy": key[2],
                "extension": key[3],
                "metric": metric,
                "n_seeds": len(seeds),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "seeds": ";".join(str(seed) for seed in seeds),
            })
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training CRRA certainty-equivalent and welfare-loss evaluation."
    )
    parser.add_argument("--out-root", required=True, help="OUT_ROOT used by tune_pipinn.sh")
    parser.add_argument(
        "--output", default=None, help="Output directory (default: <out-root>/welfare_summary)"
    )
    parser.add_argument("--models", default="both", help="both, pinn, pipinn, or a comma list")
    parser.add_argument(
        "--n-assets", type=int, default=30,
        help="Exact risky-asset dimension required in every selected run (default: 30)",
    )
    parser.add_argument("--m-states", default="1,3,5", help="Paper state dimensions")
    parser.add_argument(
        "--expected-seeds", default="",
        help=(
            "Optional exact successful seed set (comma/space/range syntax). "
            "Specify it explicitly for paper aggregation; no fixed seed numbering is assumed."
        ),
    )
    parser.add_argument(
        "--run-name-regex", default="", help="Regex used to narrow run paths if OUT_ROOT has ablations"
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Exploratory only: permit missing expected seeds (paper output should not use this)",
    )
    parser.add_argument(
        "--allow-checkpoint-fallback", action="store_true",
        help="Exploratory legacy fallback to final->last->best; default requires official final",
    )
    parser.add_argument("--n-paths", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1_000)
    parser.add_argument("--w0", type=float, default=0.5)
    parser.add_argument("--mc-seed", type=int, default=2718)
    parser.add_argument("--path-batch", type=int, default=4096)
    parser.add_argument("--policy-chunk", type=int, default=4096)
    parser.add_argument("--include-raw", action="store_true", help="Also run raw extension sensitivity")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start from scratch and overwrite matching/mismatched partial outputs (default resumes safely)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate run/seed/market/closed-form/checkpoint provenance without importing PyTorch",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else out_root / "welfare_summary"
    if args.n_paths < 2 or args.n_steps <= 0 or args.path_batch <= 0 or args.policy_chunk <= 0:
        raise ValueError("n_paths>=2 and positive n_steps/path_batch/policy_chunk are required")
    if not math.isfinite(args.w0) or args.w0 <= 0.0:
        raise ValueError("--w0 must be positive and finite")
    models = normalize_models(args.models)
    m_states = parse_seed_spec(args.m_states)
    expected_seeds = parse_seed_spec(args.expected_seeds)
    if not m_states:
        raise ValueError("--m-states cannot be empty")
    extensions = ["projected"] + (["raw"] if args.include_raw else [])

    selected = discover_paper_runs(
        out_root,
        models,
        m_states,
        expected_seeds,
        args.run_name_regex,
        args.allow_incomplete,
        args.n_assets,
    )
    checkpoints: Dict[Tuple[str, int, int], Path] = {}
    for (model, m), records in selected.items():
        for record in records:
            checkpoints[(model, m, record.seed)] = resolve_checkpoint(
                record, out_root, args.allow_checkpoint_fallback
            )

    # This preflight deliberately loads every market and closed-form file,
    # not only the representative used for simulation.  It remains pure
    # NumPy, so --validate-only never imports torch.
    contexts = validate_numpy_inputs(selected, models, m_states, args.w0)
    resume_signature, signature_payload, checkpoint_meta = build_resume_signature(
        args=args,
        selected=selected,
        checkpoints=checkpoints,
        models=models,
        m_states=m_states,
        expected_seeds=expected_seeds,
        extensions=extensions,
    )

    config_path = output / "welfare_config.json"
    metrics_path = output / "welfare_metrics.csv"
    seed_summary_path = output / "welfare_seed_summary.csv"
    validation_path = output / "welfare_validation.csv"
    output_exists = output.exists()
    existing_artifacts: List[Path] = []
    if output_exists:
        existing_artifacts = [
            path for path in (config_path, metrics_path, seed_summary_path, validation_path)
            if path.exists()
        ]
        existing_artifacts.extend(output.glob("optimal_paths_M*.npz"))

    prior_config: Optional[Dict[str, Any]] = None
    if not args.no_resume and existing_artifacts:
        if not config_path.is_file():
            raise ResumeSignatureError(
                f"existing welfare artifacts under {output} have no resume signature; "
                "use --no-resume to start from scratch"
            )
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                prior_value = json.load(handle)
            if not isinstance(prior_value, dict):
                raise ValueError("config is not a JSON object")
            prior_config = prior_value
        except Exception as exc:
            raise ResumeSignatureError(
                f"cannot verify existing resume provenance {config_path}: {exc}; "
                "use --no-resume to start from scratch"
            ) from exc
        prior_signature = str(prior_config.get("resume_signature", ""))
        if prior_signature != resume_signature:
            raise ResumeSignatureError(
                "existing welfare outputs have a different or missing resume signature "
                f"(existing={prior_signature or '<missing>'}, current={resume_signature}); "
                "use --no-resume to discard partial results intentionally"
            )
        prior_numpy = str(prior_config.get("runtime_numpy_version", ""))
        if prior_numpy and prior_numpy != np.__version__ and not args.validate_only:
            raise ResumeSignatureError(
                f"resume NumPy version changed ({prior_numpy} -> {np.__version__}); "
                "use --no-resume to avoid mixing Monte Carlo streams/numerics"
            )

    expected_keys = expected_welfare_keys(selected, m_states, extensions)
    row_map: Dict[WelfareKey, Dict[str, Any]] = {}
    if prior_config is not None and not args.no_resume:
        row_map = load_completed_rows(
            metrics_path,
            expected_keys=expected_keys,
            args=args,
            contexts=contexts,
        )
        print(
            f"[resume] signature matched; recovered {len(row_map)}/{len(expected_keys)} "
            "completed welfare row(s)"
        )

    # Resolve and compare the effective learned-policy runtime before
    # creating/rewriting ANY output.  A partial resume with incompatible
    # device or PyTorch numerics must be a non-mutating failure.  Completed
    # rows and --validate-only remain usable without importing torch.
    torch = nn = device = None
    pending_learned = any(
        key[0] != "closed_form" for key in (expected_keys - set(row_map))
    )
    if not args.validate_only and pending_learned:
        torch, nn, device = import_torch(args.device, args.torch_num_threads)
        if prior_config is not None and not args.no_resume:
            prior_device = str(prior_config.get("runtime_device", ""))
            prior_torch = str(prior_config.get("torch_version", ""))
            if prior_device and prior_device != str(device):
                raise ResumeSignatureError(
                    f"resume runtime device changed ({prior_device} -> {device}); "
                    "use --no-resume to avoid mixing learned-policy numerics"
                )
            if prior_torch and prior_torch != str(torch.__version__):
                raise ResumeSignatureError(
                    f"resume PyTorch version changed ({prior_torch} -> {torch.__version__}); "
                    "use --no-resume to avoid mixing learned-policy numerics"
                )

    summary_fields = (
        "model_type", "M", "policy", "extension", "metric", "n_seeds",
        "mean", "std", "sem", "ci95_low", "ci95_high", "seeds",
    )
    validation_fields = (
        "M", "market_hash", "mc_seed", "n_paths", "n_steps", "dt",
        "expected_utility_mc", "se_expected_utility_mc", "expected_utility_exact",
        "ce_mc", "se_ce_mc", "ce_exact", "ce_abs_error", "ce_relative_error",
    )

    provenance = {
        "created_at": (
            prior_config.get("created_at", utc_now()) if prior_config is not None else utc_now()
        ),
        "status": "validated" if args.validate_only else "running",
        "arguments": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "resume_signature": resume_signature,
        "resume_signature_payload": signature_payload,
        "resume_enabled": not bool(args.no_resume),
        "resumed_completed_rows": len(row_map),
        "fixed_protocol": {
            "initial_state": "x0=xbar",
            "wealth_scheme": "log-wealth Euler",
            "joint_innovation_covariance": "[[I_N,rho],[rho.T,I_M]]",
            "network_policy": "final greedy",
            "main_extension": "projected",
            "vww_guard": VWW_GUARD,
            "welfare_loss": "1-CE_learned/CE_optimal_Euler_CRN",
            "training_seed_aggregation": "seed metric first, then mean/sample SD/Student-t CI",
        },
        "runs": [
            {
                "model_type": record.model_type,
                "M": record.m_states,
                "training_seed": record.seed,
                "group": record.group,
                "run_dir": str(record.run_dir),
                "market_hash": record.market_hash,
                "checkpoint": str(checkpoints[(record.model_type, record.m_states, record.seed)]),
                "checkpoint_sha256": checkpoint_meta[
                    (record.model_type, record.m_states, record.seed)
                ]["sha256"],
            }
            for records in selected.values()
            for record in records
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    # --no-resume explicitly invalidates old scalar rows.  Empty them before
    # installing the new signature: even a process kill between atomic writes
    # can then lose work (as requested) but can never bless stale rows with a
    # new checkpoint/protocol signature.
    if args.no_resume:
        row_map = {}
        write_csv(metrics_path, [], WELFARE_FIELDS)
        write_csv(seed_summary_path, [], summary_fields)
        write_csv(validation_path, [], validation_fields)
    if prior_config is not None:
        provenance["resumed_at"] = utc_now()
        provenance["resumed_from_status"] = prior_config.get("status", "unknown")
        for runtime_key in ("runtime_device", "torch_version", "runtime_numpy_version"):
            if prior_config.get(runtime_key):
                provenance[runtime_key] = prior_config[runtime_key]
    if not args.validate_only:
        provenance["runtime_numpy_version"] = np.__version__
        if device is not None:
            provenance["runtime_device"] = str(device)
            provenance["torch_version"] = str(torch.__version__)
    write_json(output / "welfare_config.json", provenance)
    args._provenance_written = True

    if args.validate_only:
        if prior_config is not None and prior_config.get("status") == "success":
            provenance["status"] = "success"
        provenance["last_validated_at"] = utc_now()
        provenance["semantic_validation"] = {
            "markets": "passed",
            "closed_form_odes": "passed",
            "checkpoints_hashed": len(checkpoint_meta),
        }
        write_json(config_path, provenance)
        print(f"[welfare] validation passed; wrote {output / 'welfare_config.json'}")
        return

    print(f"[welfare] paths={args.n_paths}, steps={args.n_steps}")
    if device is not None:
        print(f"[welfare] device={device}")
    validation_rows: List[Dict[str, Any]] = []

    def checkpoint_scalar_outputs() -> None:
        rows_now = sorted_welfare_rows(row_map)
        write_csv(metrics_path, rows_now, WELFARE_FIELDS)
        write_csv(seed_summary_path, build_seed_summary(rows_now), summary_fields)

    # Immediately normalize an older partial file: remove unexpected,
    # incomplete, and duplicate rows before continuing.
    checkpoint_scalar_outputs()
    write_csv(validation_path, [], validation_fields)

    for m in m_states:
        context = contexts[int(m)]
        representative = context.representative
        market = context.market
        closed_form = context.closed_form
        if not (market.W_min <= args.w0 <= market.W_max):
            print(
                f"[warn] w0={args.w0:g} lies outside training wealth bounds "
                f"[{market.W_min:g},{market.W_max:g}] for M={m}"
            )

        cache_path = output / f"optimal_paths_M{m}.npz"
        optimal = None if args.no_resume else load_optimal_cache(
            cache_path,
            resume_signature=resume_signature,
            m_states=int(m),
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            mc_seed=args.mc_seed,
        )
        if optimal is None:
            print(f"[welfare] M={m}: simulating closed-form optimal policy")
            optimal = simulate_optimal_policy(
                market,
                closed_form,
                n_paths=args.n_paths,
                n_steps=args.n_steps,
                path_batch=args.path_batch,
                w0=args.w0,
                mc_seed=args.mc_seed,
            )
            save_optimal_cache(
                cache_path,
                optimal,
                resume_signature=resume_signature,
                m_states=int(m),
                n_paths=args.n_paths,
                n_steps=args.n_steps,
                mc_seed=args.mc_seed,
            )
            print(f"[welfare] M={m}: cached optimal paired paths at {cache_path}")
        else:
            print(f"[resume] M={m}: loaded cached optimal paired paths")

        optimal_utility = utility_statistics(optimal.terminal_log_wealth, market.gamma)
        optimal_welfare = paired_welfare_statistics(
            optimal.terminal_log_wealth, optimal.terminal_log_wealth, market.gamma
        )
        dt = market.horizon / args.n_steps
        optimal_key: WelfareKey = ("closed_form", int(m), None, "optimal", "optimal")
        row_map[optimal_key] = welfare_row(
            model_type="closed_form",
            training_seed="",
            m_states=m,
            policy="optimal",
            extension="optimal",
            mc_seed=args.mc_seed,
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            dt=dt,
            utility=optimal_utility,
            welfare=optimal_welfare,
            diagnostics=optimal.diagnostics,
        )
        exact_utility, exact_ce = exact_optimal_statistics(market, closed_form, args.w0)
        validation_rows.append({
            "M": m,
            "market_hash": representative.market_hash,
            "mc_seed": args.mc_seed,
            "n_paths": args.n_paths,
            "n_steps": args.n_steps,
            "dt": dt,
            "expected_utility_mc": optimal_utility.expected_utility,
            "se_expected_utility_mc": optimal_utility.se_expected_utility,
            "expected_utility_exact": exact_utility,
            "ce_mc": optimal_utility.ce,
            "se_ce_mc": optimal_utility.se_ce,
            "ce_exact": exact_ce,
            "ce_abs_error": abs(optimal_utility.ce - exact_ce),
            "ce_relative_error": abs(optimal_utility.ce - exact_ce) / exact_ce,
        })
        checkpoint_scalar_outputs()
        write_csv(validation_path, validation_rows, validation_fields)

        for model in models:
            for record in selected[(model, m)]:
                pending_extensions = [
                    extension for extension in extensions
                    if (
                        model, int(m), record.seed, "greedy_final", extension
                    ) not in row_map
                ]
                if not pending_extensions:
                    print(
                        f"[resume] M={m}, model={model}, seed={record.seed}: "
                        "all requested extensions already complete"
                    )
                    continue
                checkpoint = checkpoints[(model, m, record.seed)]
                verify_checkpoint_unchanged(
                    checkpoint, checkpoint_meta[(model, m, record.seed)]
                )
                if torch is None or nn is None or device is None:
                    raise AssertionError(
                        "learned-policy runtime was not resolved by the non-mutating preflight"
                    )
                print(
                    f"[welfare] M={m}, model={model}, seed={record.seed}: "
                    f"loading {checkpoint.name}"
                )
                network = load_value_network(record, checkpoint, torch, nn, device)
                for extension in pending_extensions:
                    print(
                        f"[welfare] M={m}, model={model}, seed={record.seed}, "
                        f"extension={extension}"
                    )
                    learned = simulate_network_policy(
                        network,
                        market,
                        extension=extension,
                        n_paths=args.n_paths,
                        n_steps=args.n_steps,
                        path_batch=args.path_batch,
                        policy_chunk=args.policy_chunk,
                        w0=args.w0,
                        mc_seed=args.mc_seed,
                        torch=torch,
                        device=device,
                    )
                    # Resetting the same generator and using the same batch
                    # layout must reproduce the state path exactly.  This
                    # assertion guards the paired WL calculation against a
                    # future accidental loss of common random numbers.
                    if not np.array_equal(learned.terminal_state, optimal.terminal_state):
                        raise RuntimeError(
                            "common-random-number check failed: learned and optimal state paths differ"
                        )
                    learned_utility = utility_statistics(
                        learned.terminal_log_wealth, market.gamma
                    )
                    learned_welfare = paired_welfare_statistics(
                        learned.terminal_log_wealth,
                        optimal.terminal_log_wealth,
                        market.gamma,
                    )
                    learned_key: WelfareKey = (
                        model, int(m), record.seed, "greedy_final", extension
                    )
                    row_map[learned_key] = welfare_row(
                        model_type=model,
                        training_seed=record.seed,
                        m_states=m,
                        policy="greedy_final",
                        extension=extension,
                        mc_seed=args.mc_seed,
                        n_paths=args.n_paths,
                        n_steps=args.n_steps,
                        dt=dt,
                        utility=learned_utility,
                        welfare=learned_welfare,
                        diagnostics=learned.diagnostics,
                    )
                    checkpoint_scalar_outputs()
                del network
                if device is not None and str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

    missing_keys = expected_keys - set(row_map)
    if missing_keys:
        raise RuntimeError(f"welfare evaluation ended with missing rows: {sorted(missing_keys)}")
    checkpoint_scalar_outputs()
    write_csv(validation_path, validation_rows, validation_fields)
    provenance["status"] = "success"
    provenance["completed_at"] = utc_now()
    provenance["completed_rows"] = len(row_map)
    provenance["outputs"] = {
        "welfare_metrics": str(metrics_path),
        "welfare_seed_summary": str(seed_summary_path),
        "welfare_validation": str(validation_path),
        "optimal_path_caches": [str(output / f"optimal_paths_M{m}.npz") for m in m_states],
    }
    write_json(output / "welfare_config.json", provenance)
    print(f"[welfare] complete: {output}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args._provenance_written = False
    try:
        run(args)
    except BaseException as exc:
        # A long Monte Carlo job may be interrupted well after partial CSVs
        # were atomically checkpointed.  Never leave their provenance marked
        # as "running" indefinitely.
        out_root = Path(args.out_root).expanduser().resolve()
        output = (
            Path(args.output).expanduser().resolve()
            if args.output else out_root / "welfare_summary"
        )
        config_path = output / "welfare_config.json"
        if getattr(args, "_provenance_written", False) and config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    provenance = json.load(handle)
                if isinstance(provenance, dict):
                    provenance["status"] = "failed"
                    provenance["failed_at"] = utc_now()
                    provenance["error_type"] = type(exc).__name__
                    provenance["error"] = str(exc)
                    write_json(config_path, provenance)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
