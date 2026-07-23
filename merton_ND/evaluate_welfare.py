#!/usr/bin/env python3
"""Post-training total-lifetime welfare evaluation for the Merton runs.

The Merton objective contains both intermediate consumption and terminal
bequest.  This evaluator therefore simulates the pathwise discounted payoff

    int_0^T exp(-rho*t) U(c_t) dt
        + exp(-rho*T) epsilon U(W_T),

using the final greedy policy of each official ``value_net_final.pt``.  The
closed-form optimal policy is simulated with the same log-Euler scheme and
the same Brownian draws.  Its Monte Carlo objective, rather than the analytic
continuous-time value, is the denominator of the reported CE/WL comparison.

PyTorch is imported lazily.  Run discovery, provenance validation, analytic
checks, and all statistical helpers remain usable with ``--validate-only`` on
a NumPy-only post-processing machine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    canonical_market_hash,
    find_runs,
    group_key,
    load_config_args_raw,
    parse_int_spec,
    parse_seed_spec,
    run_status,
    run_updated_at,
    t_crit_95,
)


RESUME_SCHEMA_VERSION = 2
POLICY_CONTRACT_VERSION = "merton-final-greedy-logw-v2-network-provenance"


class ResumeSignatureError(RuntimeError):
    """Raised before output mutation when partial results are incompatible."""


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
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scalar(values: Mapping[str, np.ndarray], key: str) -> float:
    if key not in values:
        raise ValueError(f"market snapshot is missing {key!r}")
    array = np.asarray(values[key])
    if array.size != 1:
        raise ValueError(f"market field {key!r} must be scalar, got {array.shape}")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"market field {key!r} is nonfinite")
    return result


# ---------------------------------------------------------------------------
# Pure NumPy welfare statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TotalUtilityStats:
    expected_total_utility: float
    se_expected_total_utility: float


@dataclass(frozen=True)
class PairedWelfareStats:
    utility_gap: float
    se_utility_gap: float
    q: float
    se_q: float
    ce0: float
    se_ce0: float
    wl: float
    se_wl: float


def total_utility_statistics(pathwise_total_utility: np.ndarray) -> TotalUtilityStats:
    values = np.asarray(pathwise_total_utility, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("at least two Monte Carlo paths are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("pathwise total utility contains NaN or infinity")
    return TotalUtilityStats(
        expected_total_utility=float(np.mean(values)),
        se_expected_total_utility=float(np.std(values, ddof=1) / math.sqrt(values.size)),
    )


def paired_welfare_statistics(
    learned_pathwise_utility: np.ndarray,
    optimal_pathwise_utility: np.ndarray,
    gamma: float,
    w0: float,
) -> PairedWelfareStats:
    """Return initial-wealth-equivalent CE/WL and paired delta-method SEs.

    For CRRA homogeneity, ``q=(J/J*)**(1/(1-gamma))`` and ``CE0=w0*q``.
    The paired influence function retains the covariance induced by common
    Brownian draws.  No clipping is applied to a small negative Monte Carlo
    welfare loss.
    """
    learned = np.asarray(learned_pathwise_utility, dtype=np.float64).reshape(-1)
    optimal = np.asarray(optimal_pathwise_utility, dtype=np.float64).reshape(-1)
    if learned.shape != optimal.shape or learned.size < 2:
        raise ValueError("paired learned/optimal utility samples need equal size >= 2")
    if not np.all(np.isfinite(learned)) or not np.all(np.isfinite(optimal)):
        raise ValueError("paired utility samples contain NaN or infinity")
    exponent = 1.0 / (1.0 - float(gamma))
    if not math.isfinite(exponent):
        raise ValueError("gamma=1 log utility requires a different welfare formula")
    mean_l = float(np.mean(learned))
    mean_o = float(np.mean(optimal))
    if mean_l == 0.0 or mean_o == 0.0 or mean_l * mean_o <= 0.0:
        raise ValueError(
            "learned and optimal mean utilities must be finite, nonzero, and have equal sign"
        )
    ratio = mean_l / mean_o
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("J/J* is not a finite positive CRRA ratio")
    q = float(math.exp(exponent * math.log(ratio)))
    influence_log_q = exponent * (
        (learned - mean_l) / mean_l - (optimal - mean_o) / mean_o
    )
    influence_q = q * influence_log_q
    se_q = float(np.std(influence_q, ddof=1) / math.sqrt(learned.size))
    ce0 = float(w0) * q
    se_ce0 = float(w0) * se_q
    return PairedWelfareStats(
        utility_gap=mean_l - mean_o,
        se_utility_gap=float(np.std(learned - optimal, ddof=1) / math.sqrt(learned.size)),
        q=q,
        se_q=se_q,
        ce0=ce0,
        se_ce0=se_ce0,
        wl=1.0 - q,
        se_wl=se_q,
    )


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("seed summary requires nonempty finite values")
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, 0.0, float("nan"), float("nan"), float("nan")
    std = float(np.std(array, ddof=1))
    sem = std / math.sqrt(array.size)
    half_width = t_crit_95(int(array.size) - 1) * sem
    return mean, std, sem, mean - half_width, mean + half_width


# ---------------------------------------------------------------------------
# Run discovery, market data, and checkpoint provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    model_type: str
    n_assets: int
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
    aliases = {"pinn": "pinn", "pipinn": "pipinn", "pi-pinn": "pipinn"}
    result: List[str] = []
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
    n_assets: Sequence[int],
    expected_seeds: Sequence[int],
    min_seeds: int = 1,
    outer_iters: Optional[int] = None,
    run_name_regex: str = "",
    allow_incomplete: bool = False,
) -> Dict[Tuple[str, int], List[RunRecord]]:
    """Select one successful configuration and the requested seeds per cell.

    A nonempty ``expected_seeds`` is an exact-set paper contract unless
    ``allow_incomplete`` is enabled.  ``min_seeds`` is independent of that
    contract and always applies to the selected successful runs, including
    exploratory discovery when ``expected_seeds`` is empty.
    """
    if (
        isinstance(min_seeds, bool)
        or int(min_seeds) != min_seeds
        or int(min_seeds) < 1
    ):
        raise ValueError("min_seeds must be a positive integer")
    min_seeds = int(min_seeds)
    wanted_models = set(models)
    wanted_n = set(int(n) for n in n_assets)
    expected = set(int(seed) for seed in expected_seeds)
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int], RunRecord] = {}
    for run_text in find_runs(str(out_root)):
        run_dir = Path(run_text)
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None:
            continue
        model = str(cfg.get("model_type", "")).lower()
        if model == "pi-pinn":
            model = "pipinn"
        n = _as_int(cfg.get("n_assets"))
        seed = _as_int(cfg.get("seed"))
        if model not in wanted_models or n not in wanted_n or seed is None:
            continue
        if _as_int(cfg.get("m_states", 1)) != 1:
            continue
        if outer_iters is not None and _as_int(cfg.get("outer_iters")) != outer_iters:
            continue
        try:
            relative_name = str(run_dir.relative_to(out_root))
        except ValueError:
            relative_name = str(run_dir)
        if pattern and not pattern.search(relative_name):
            continue
        group, _canonical = group_key(cfg)
        record = RunRecord(
            run_dir=run_dir,
            model_type=model,
            n_assets=int(n),
            seed=int(seed),
            group=group,
            updated_at=run_updated_at(str(run_dir)),
            status=run_status(str(run_dir)),
            config_args=dict(cfg),
            config_doc=_read_config_doc(run_dir),
        )
        key = (group, record.seed)
        old = newest.get(key)
        if old is None or (record.updated_at, str(record.run_dir)) >= (
            old.updated_at, str(old.run_dir)
        ):
            newest[key] = record

    by_cell: Dict[Tuple[str, int], Dict[str, List[RunRecord]]] = {}
    for record in newest.values():
        by_cell.setdefault((record.model_type, record.n_assets), {}).setdefault(
            record.group, []
        ).append(record)

    selected: Dict[Tuple[str, int], List[RunRecord]] = {}
    errors: List[str] = []
    for model in models:
        for n in n_assets:
            cell = (model, int(n))
            groups = {
                group: records for group, records in by_cell.get(cell, {}).items()
                if any(record.status == "success" for record in records)
            }
            if len(groups) != 1:
                errors.append(
                    f"model={model}, N={n}: expected exactly one successful training "
                    f"configuration, found groups={sorted(groups)}; narrow --run-name-regex"
                )
                continue
            group, records = next(iter(groups.items()))
            successful = {r.seed: r for r in records if r.status == "success"}
            found = set(successful)
            if expected and found != expected and not allow_incomplete:
                errors.append(
                    f"model={model}, N={n}, group={group}: successful seeds={sorted(found)}, "
                    f"expected={sorted(expected)}, missing={sorted(expected-found)}, "
                    f"extra={sorted(found-expected)}"
                )
                continue
            seeds = sorted(expected & found if expected else found)
            if not seeds:
                errors.append(f"model={model}, N={n}: no requested successful seeds")
                continue
            if len(seeds) < min_seeds:
                errors.append(
                    f"model={model}, N={n}, group={group}: "
                    f"{len(seeds)} selected successful seeds < min_seeds={min_seeds}; "
                    f"seeds={seeds}"
                )
                continue
            if allow_incomplete and expected and found != expected:
                print(f"[warn] exploratory incomplete model={model}, N={n}: seeds={seeds}")
            selected[cell] = [successful[seed] for seed in seeds]
    if errors:
        raise ValueError("paper-run validation failed:\n  - " + "\n  - ".join(errors))

    hashes_by_n: Dict[int, set[str]] = {}
    with_hash: Dict[Tuple[str, int], List[RunRecord]] = {}
    for cell, records in selected.items():
        updated: List[RunRecord] = []
        for record in records:
            market_path = record.run_dir / "market_params.npz"
            digest = canonical_market_hash(str(market_path))
            updated_record = RunRecord(**{**record.__dict__, "market_hash": digest})
            updated.append(updated_record)
            hashes_by_n.setdefault(record.n_assets, set()).add(digest)
        with_hash[cell] = updated
    for n, hashes in hashes_by_n.items():
        if len(hashes) != 1:
            raise ValueError(
                f"N={n}: market snapshot differs across methods/seeds; hashes={sorted(hashes)}"
            )
    return with_hash


def resolve_checkpoint(record: RunRecord, out_root: Path, allow_fallback: bool) -> Path:
    raw_weight_dir = record.config_doc.get("weight_dir") or record.config_args.get("weight_root")
    candidates: List[Path] = []
    if raw_weight_dir:
        raw = Path(str(raw_weight_dir)).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            if record.config_doc.get("cwd"):
                candidates.append(Path(str(record.config_doc["cwd"])) / raw)
            candidates.extend([record.run_dir / raw, out_root / raw, raw])
    candidates.append(out_root / "weights" / record.model_type / record.run_dir.name)
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))
    names = ["value_net_final.pt"]
    if allow_fallback:
        names.extend(["value_net_last.pt", "value_net_best_diag.pt", "value_net_best.pt"])
    checked: List[str] = []
    for name in names:
        for directory in unique:
            path = directory / name
            checked.append(str(path))
            if path.is_file():
                if name != "value_net_final.pt":
                    print(f"[warn] legacy checkpoint fallback for {record.run_dir}: {path}")
                else:
                    status_path = record.run_dir / "status.json"
                    if status_path.is_file():
                        with status_path.open("r", encoding="utf-8") as handle:
                            status_doc = json.load(handle)
                        recorded_hash = str(
                            status_doc.get("final_checkpoint_file_sha256", "")
                        ).strip()
                        if recorded_hash and sha256_file(path) != recorded_hash:
                            raise ValueError(
                                f"{path}: file hash disagrees with status.json official final hash"
                            )
                return path.resolve()
    raise FileNotFoundError(
        f"official final checkpoint not found for {record.run_dir}; checked:\n  "
        + "\n  ".join(checked)
    )


@dataclass(frozen=True)
class MarketData:
    mu_excess: np.ndarray
    sigma: np.ndarray
    chol: np.ndarray
    sigma_inv_mu: np.ndarray
    pi_star: np.ndarray
    theta: float
    nu: float
    gamma: float
    risk_free: float
    discount: float
    bequest: float
    horizon: float
    w_min: float
    w_max: float
    n_assets: int
    market_seed: int


def load_market(path: Path) -> MarketData:
    with np.load(path, allow_pickle=False) as source:
        values = {key: np.asarray(source[key]).copy() for key in source.files}
    mu = np.asarray(values["mu_excess"], dtype=np.float64).reshape(-1)
    n = int(round(_scalar(values, "n_assets")))
    sigma = np.asarray(values["Sigma_safe"], dtype=np.float64)
    chol_key = "chol" if "chol" in values else "L"
    chol = np.asarray(values[chol_key], dtype=np.float64)
    if mu.shape != (n,) or sigma.shape != (n, n) or chol.shape != (n, n):
        raise ValueError(f"{path}: inconsistent market dimensions for N={n}")
    if not all(np.all(np.isfinite(x)) for x in (mu, sigma, chol)):
        raise ValueError(f"{path}: market arrays contain NaN or infinity")
    if not np.allclose(sigma, sigma.T, rtol=1e-11, atol=1e-12):
        raise ValueError(f"{path}: Sigma_safe is not symmetric")
    if float(np.linalg.eigvalsh(sigma)[0]) <= 0.0:
        raise ValueError(f"{path}: Sigma_safe is not positive definite")
    if not np.allclose(chol @ chol.T, sigma, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{path}: saved Cholesky factor is inconsistent with Sigma_safe")
    solved = np.linalg.solve(sigma, mu)
    if "Sigma_inv_mu" in values:
        recorded = np.asarray(values["Sigma_inv_mu"], dtype=np.float64).reshape(-1)
        if not np.allclose(recorded, solved, rtol=1e-8, atol=1e-10):
            raise ValueError(f"{path}: Sigma_inv_mu is inconsistent")
    pi_star = np.asarray(values["pi_star"], dtype=np.float64).reshape(-1)
    gamma = _scalar(values, "gamma")
    if pi_star.shape != (n,) or not np.allclose(
        pi_star, solved / gamma, rtol=1e-8, atol=1e-10
    ):
        raise ValueError(f"{path}: pi_star is inconsistent with Sigma^-1 mu / gamma")
    theta = _scalar(values, "Theta")
    if not math.isclose(theta, float(mu @ solved), rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{path}: Theta is inconsistent")
    risk_free = _scalar(values, "r")
    discount = _scalar(values, "rho_discount")
    nu = discount / gamma - (1.0 - gamma) * (
        theta / (2.0 * gamma * gamma) + risk_free / gamma
    )
    recorded_nu = _scalar(values, "nu")
    if not math.isclose(recorded_nu, nu, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{path}: nu is inconsistent")
    horizon = _scalar(values, "T")
    bequest = _scalar(values, "epsilon")
    w_min = _scalar(values, "w_min")
    w_max = _scalar(values, "w_max")
    if (gamma <= 0.0 or math.isclose(gamma, 1.0) or horizon <= 0.0
            or bequest <= 0.0 or not math.isfinite(nu)):
        raise ValueError(f"{path}: evaluator requires gamma>0, gamma!=1, T>0")
    if not 0.0 < w_min < w_max:
        raise ValueError(f"{path}: invalid training wealth interval")
    market_seed_float = _scalar(values, "market_seed")
    market_seed = int(round(market_seed_float))
    if not math.isclose(market_seed_float, market_seed, abs_tol=1e-9):
        raise ValueError(f"{path}: market_seed must be integer valued")
    return MarketData(
        mu_excess=mu,
        sigma=sigma,
        chol=chol,
        sigma_inv_mu=solved,
        pi_star=pi_star,
        theta=theta,
        nu=nu,
        gamma=gamma,
        risk_free=risk_free,
        discount=discount,
        bequest=bequest,
        horizon=horizon,
        w_min=w_min,
        w_max=w_max,
        n_assets=n,
        market_seed=market_seed,
    )


def optimal_consumption_ratio(market: MarketData, t: Any) -> np.ndarray:
    """Closed-form c/W for terminal bequest ``epsilon*U(W_T)``."""
    time = np.asarray(t, dtype=np.float64)
    terminal_scale = market.bequest ** (1.0 / market.gamma)
    if abs(market.nu) < 1.0e-12:
        return 1.0 / (market.horizon - time + terminal_scale)
    denominator = 1.0 + (
        market.nu * terminal_scale - 1.0
    ) * np.exp(-market.nu * (market.horizon - time))
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise ValueError("closed-form consumption denominator is nonpositive/nonfinite")
    return market.nu / denominator


def analytic_optimal_value(market: MarketData, w0: float) -> float:
    terminal_scale = market.bequest ** (1.0 / market.gamma)
    if abs(market.nu) < 1.0e-12:
        scale = (market.horizon + terminal_scale) ** market.gamma
        return scale * float(w0) ** (1.0 - market.gamma) / (1.0 - market.gamma)
    denominator = 1.0 + (market.nu * terminal_scale - 1.0) * math.exp(
        -market.nu * market.horizon
    )
    scale = (denominator / market.nu) ** market.gamma
    return scale * float(w0) ** (1.0 - market.gamma) / (1.0 - market.gamma)


def _check_config_scalar(record: RunRecord, key: str, expected: float) -> None:
    value = record.config_args.get(key)
    if value is None:
        return
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record.run_dir}: config {key!r} is not numeric") from exc
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{record.run_dir}: config {key}={actual} disagrees with market {expected}"
        )


def validate_numpy_inputs(
    selected: Mapping[Tuple[str, int], Sequence[RunRecord]], w0: float
) -> Dict[int, MarketData]:
    contexts: Dict[int, MarketData] = {}
    for cell in sorted(selected):
        for record in selected[cell]:
            market = load_market(record.run_dir / "market_params.npz")
            if market.n_assets != record.n_assets:
                raise ValueError(f"{record.run_dir}: market/config N mismatch")
            for key, expected in (
                ("gamma", market.gamma), ("r", market.risk_free),
                ("rho_discount", market.discount),
                ("epsilon_bequest", market.bequest), ("tau_max", market.horizon),
                ("w_min", market.w_min), ("w_max", market.w_max),
                ("market_seed", float(market.market_seed)),
            ):
                _check_config_scalar(record, key, expected)
            exact = analytic_optimal_value(market, w0)
            if not math.isfinite(exact):
                raise ValueError(f"{record.run_dir}: analytic optimal value is nonfinite")
            previous = contexts.setdefault(record.n_assets, market)
            if not math.isclose(
                analytic_optimal_value(previous, w0), exact, rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(f"N={record.n_assets}: analytic value differs across runs")
    return contexts


@dataclass(frozen=True)
class PolicyContract:
    bounds_mode: str
    vw_guard: float
    numerator_guard: float
    denominator_guard: float
    kappa_min: Optional[float]
    kappa_max: Optional[float]
    consumption_min: Optional[float]
    consumption_max: Optional[float]
    portfolio_min: Optional[float]
    portfolio_max: Optional[float]


def _optional_number(value: Any, label: str) -> Optional[float]:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite or null")
    return result


def policy_contract(record: RunRecord) -> PolicyContract:
    cfg = record.config_doc
    mode = str(cfg.get("policy_guard_mode", "")).strip().lower()
    version = str(cfg.get("policy_guard_version", ""))
    # PI-PINN records the complete map at top level.  Direct PINN records its
    # evaluation-only greedy map in args because the smooth HJB guard is a
    # separate training-residual continuation and is not used for controls.
    direct_eval_guard = str(
        record.config_args.get("evaluation_policy_guard_mode", "")
    ).strip().lower()
    current_pipinn = mode == "trainer-one-sided" and version == "merton-logw-v1"
    current_pinn = (
        record.model_type == "pinn" and not mode
        and direct_eval_guard == "one-sided-hard-clamp"
    )
    if not (current_pipinn or current_pinn):
        raise ValueError(
            f"{record.run_dir}: unsupported/missing greedy guard contract "
            f"mode={mode!r}, version={version!r}"
        )
    bounds_mode = str(cfg.get("policy_bounds_mode", "")).strip().lower()
    if bounds_mode not in {"stabilized", "none"}:
        raise ValueError(f"{record.run_dir}: invalid policy_bounds_mode={bounds_mode!r}")
    vw_guard = float(cfg.get("vw_guard", 1e-8))
    numerator = float(cfg.get("policy_numerator_guard_eps", 1e-8))
    denominator = float(cfg.get("policy_denominator_guard_eps", cfg.get("denominator_guard", 1e-8)))
    if vw_guard <= 0.0 or numerator <= 0.0 or denominator <= 0.0:
        raise ValueError(f"{record.run_dir}: policy guard epsilons must be positive")
    if bounds_mode == "none":
        bound_values = [
            cfg.get(key) for key in (
                "policy_kappa_min", "policy_kappa_max", "policy_c_min", "policy_c_max",
                "policy_pi_min", "policy_pi_max",
            )
        ]
        if any(_optional_number(value, "unbounded policy metadata") is not None for value in bound_values):
            raise ValueError(
                f"{record.run_dir}: policy_bounds_mode=none but resolved finite bounds remain"
            )
        return PolicyContract(
            bounds_mode, vw_guard, numerator, denominator,
            None, None, None, None, None, None,
        )
    resolved = {
        key: _optional_number(cfg.get(key), key) for key in (
            "policy_kappa_min", "policy_kappa_max", "policy_c_min", "policy_c_max",
            "policy_pi_min", "policy_pi_max",
        )
    }
    consumption_keys = (
        "policy_kappa_min", "policy_kappa_max", "policy_c_min", "policy_c_max"
    )
    if any(resolved[key] is None for key in consumption_keys):
        raise ValueError(
            f"{record.run_dir}: stabilized policy is missing resolved consumption bounds"
        )
    if (resolved["policy_pi_min"] is None) != (resolved["policy_pi_max"] is None):
        raise ValueError(
            f"{record.run_dir}: portfolio lower/upper bounds must both be finite or both null"
        )
    for lower_key, upper_key in (
        ("policy_kappa_min", "policy_kappa_max"),
        ("policy_c_min", "policy_c_max"),
        ("policy_pi_min", "policy_pi_max"),
    ):
        lower = resolved[lower_key]
        upper = resolved[upper_key]
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"{record.run_dir}: resolved bounds require "
                f"{lower_key}<={upper_key}"
            )
    if (resolved["policy_kappa_min"] is not None
            and resolved["policy_kappa_min"] < 0.0):
        raise ValueError(f"{record.run_dir}: kappa lower bound must be nonnegative")
    if (resolved["policy_c_min"] is not None
            and resolved["policy_c_min"] <= 0.0):
        raise ValueError(f"{record.run_dir}: consumption lower bound must be positive")
    return PolicyContract(
        bounds_mode, vw_guard, numerator, denominator,
        resolved["policy_kappa_min"], resolved["policy_kappa_max"],
        resolved["policy_c_min"], resolved["policy_c_max"],
        resolved["policy_pi_min"], resolved["policy_pi_max"],
    )


def network_contract(record: RunRecord) -> Dict[str, Any]:
    """Validate the exact log-wealth MLP interpretation before loading weights."""
    cfg = record.config_doc
    expected = {
        "network_time_coordinate": "t",
        "network_input_order": "t,y",
        "network_input_transform": "identity",
        "network_activation": "tanh",
        "network_dtype": "float32",
    }
    resolved: Dict[str, Any] = {}
    for key, wanted in expected.items():
        raw = cfg.get(key, record.config_args.get(key))
        if raw is None and key == "network_activation":
            raw = cfg.get("activation", record.config_args.get("activation"))
        value = str(raw).strip().lower() if raw is not None else ""
        if value != wanted:
            raise ValueError(
                f"{record.run_dir}: unsupported/missing {key}={raw!r}; "
                f"expected {wanted!r}"
            )
        resolved[key] = value
    marker = str(cfg.get("trainer_source_marker", "")).strip()
    allowed_markers = {
        "pipinn": "merton-pipinn-logw-trainer-one-sided-selection-v2",
        "pinn": "merton-direct-pinn-logw-heldout-scheduler-v1",
    }
    if marker != allowed_markers[record.model_type]:
        raise ValueError(
            f"{record.run_dir}: unsupported trainer_source_marker={marker!r}"
        )
    source_hash = str(cfg.get("trainer_source_sha256", "")).strip().lower()
    if len(source_hash) != 64 or any(ch not in "0123456789abcdef" for ch in source_hash):
        raise ValueError(
            f"{record.run_dir}: missing/invalid trainer_source_sha256 provenance"
        )
    hidden = int(record.config_args.get("value_hidden", 256))
    depth = int(record.config_args.get("value_depth", 3))
    if hidden <= 0 or depth <= 0:
        raise ValueError(f"{record.run_dir}: invalid value-network architecture")
    resolved.update({
        "trainer_source_marker": marker,
        "trainer_source_sha256": source_hash,
        "value_hidden": hidden,
        "value_depth": depth,
    })
    return resolved


# ---------------------------------------------------------------------------
# Log-Euler simulation
# ---------------------------------------------------------------------------


@dataclass
class SimulationDiagnostics:
    wealth_outside_path_time_frac: float = 0.0
    wealth_outside_path_frac: float = 0.0
    wealth_below_path_time_frac: float = 0.0
    wealth_above_path_time_frac: float = 0.0
    vw_guard_frac: float = 0.0
    numerator_guard_frac: float = 0.0
    denominator_guard_frac: float = 0.0
    kappa_low_bound_frac: float = 0.0
    kappa_high_bound_frac: float = 0.0
    consumption_low_bound_frac: float = 0.0
    consumption_high_bound_frac: float = 0.0
    portfolio_any_bound_frac: float = 0.0
    portfolio_component_bound_frac: float = 0.0
    max_abs_portfolio: float = 0.0
    max_consumption_wealth_ratio: float = 0.0
    min_log_wealth: float = 0.0
    max_log_wealth: float = 0.0
    finite_check_passed: int = 1


@dataclass
class PolicyEvaluation:
    consumption: np.ndarray
    portfolio: np.ndarray
    masks: Dict[str, np.ndarray]


@dataclass
class SimulationResult:
    pathwise_total_utility: np.ndarray
    terminal_log_wealth: np.ndarray
    diagnostics: SimulationDiagnostics


def crra_utility_from_log(log_amount: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(log_amount, dtype=np.float64)
    exponent = 1.0 - float(gamma)
    if math.isclose(exponent, 0.0):
        raise ValueError("gamma=1 log utility is not supported")
    result = np.exp(exponent * values) / exponent
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("CRRA utility overflow/underflow produced a nonfinite value")
    return result


_MASK_FIELDS = (
    "vw_guard", "numerator_guard", "denominator_guard", "kappa_low_bound",
    "kappa_high_bound", "consumption_low_bound", "consumption_high_bound",
    "portfolio_any_bound", "portfolio_component_bound",
)


def _batch_seed(mc_seed: int, n_assets: int, start: int) -> np.random.SeedSequence:
    # A fresh generator for each fixed path batch makes every policy consume
    # exactly the same Brownian array regardless of its numerical trajectory.
    return np.random.SeedSequence([int(mc_seed), int(n_assets), int(start)])


def simulate_policy(
    market: MarketData,
    *,
    policy: Callable[[float, np.ndarray], PolicyEvaluation],
    n_paths: int,
    n_steps: int,
    w0: float,
    mc_seed: int,
    path_batch: int,
) -> SimulationResult:
    dt = market.horizon / int(n_steps)
    sqrt_dt = math.sqrt(dt)
    log_min, log_max = math.log(market.w_min), math.log(market.w_max)
    payoff = np.empty(n_paths, dtype=np.float64)
    terminal_y = np.empty(n_paths, dtype=np.float64)
    total_eval = 0
    below_count = above_count = outside_path_count = 0
    mask_counts = {name: 0 for name in _MASK_FIELDS}
    portfolio_component_denominator = 0
    max_abs_pi = 0.0
    max_kappa = 0.0
    global_min_y = math.inf
    global_max_y = -math.inf

    for start in range(0, n_paths, path_batch):
        stop = min(start + path_batch, n_paths)
        batch = stop - start
        rng = np.random.default_rng(_batch_seed(mc_seed, market.n_assets, start))
        y = np.full(batch, math.log(w0), dtype=np.float64)
        accumulated = np.zeros(batch, dtype=np.float64)
        outside_ever = np.zeros(batch, dtype=bool)
        for step in range(n_steps):
            if not np.all(np.isfinite(y)):
                raise FloatingPointError(f"nonfinite log wealth before step {step}")
            below = y < log_min
            above = y > log_max
            outside_ever |= below | above
            below_count += int(np.count_nonzero(below))
            above_count += int(np.count_nonzero(above))
            total_eval += batch
            global_min_y = min(global_min_y, float(np.min(y)))
            global_max_y = max(global_max_y, float(np.max(y)))

            t = step * dt
            evaluated = policy(t, y)
            c = np.asarray(evaluated.consumption, dtype=np.float64).reshape(batch)
            pi = np.asarray(evaluated.portfolio, dtype=np.float64).reshape(
                batch, market.n_assets
            )
            if not np.all(np.isfinite(c)) or not np.all(np.isfinite(pi)):
                raise FloatingPointError(f"policy produced NaN or infinity at step {step}")
            if np.any(c <= 0.0):
                raise FloatingPointError(f"policy produced nonpositive consumption at step {step}")
            wealth = np.exp(y)
            if not np.all(np.isfinite(wealth)) or np.any(wealth <= 0.0):
                raise FloatingPointError(f"wealth exponentiation failed at step {step}")
            kappa = c / wealth
            if not np.all(np.isfinite(kappa)):
                raise FloatingPointError(f"nonfinite c/W at step {step}")
            max_abs_pi = max(max_abs_pi, float(np.max(np.abs(pi))))
            max_kappa = max(max_kappa, float(np.max(kappa)))
            for name in _MASK_FIELDS:
                mask = np.asarray(evaluated.masks.get(name, False), dtype=bool)
                mask_counts[name] += int(np.count_nonzero(mask))
            portfolio_component_denominator += batch * market.n_assets

            accumulated += (
                math.exp(-market.discount * t)
                * crra_utility_from_log(np.log(c), market.gamma)
                * dt
            )
            variance = np.einsum("bi,ij,bj->b", pi, market.sigma, pi)
            if np.any(variance < -1e-10) or not np.all(np.isfinite(variance)):
                raise FloatingPointError(f"invalid portfolio variance at step {step}")
            variance = np.maximum(variance, 0.0)
            drift = (
                market.risk_free + pi @ market.mu_excess - kappa - 0.5 * variance
            )
            normals = rng.standard_normal((batch, market.n_assets), dtype=np.float64)
            asset_shock = normals @ market.chol.T
            y = y + drift * dt + np.einsum("bi,bi->b", pi, asset_shock) * sqrt_dt

        if not np.all(np.isfinite(y)):
            raise FloatingPointError("nonfinite terminal log wealth")
        terminal = (
            math.exp(-market.discount * market.horizon)
            * market.bequest
            * crra_utility_from_log(y, market.gamma)
        )
        accumulated += terminal
        if not np.all(np.isfinite(accumulated)):
            raise FloatingPointError("pathwise total utility is nonfinite")
        payoff[start:stop] = accumulated
        terminal_y[start:stop] = y
        outside_ever |= (y < log_min) | (y > log_max)
        outside_path_count += int(np.count_nonzero(outside_ever))
        global_min_y = min(global_min_y, float(np.min(y)))
        global_max_y = max(global_max_y, float(np.max(y)))

    masks_denominator = float(total_eval)
    diagnostics = SimulationDiagnostics(
        wealth_outside_path_time_frac=(below_count + above_count) / masks_denominator,
        wealth_outside_path_frac=outside_path_count / float(n_paths),
        wealth_below_path_time_frac=below_count / masks_denominator,
        wealth_above_path_time_frac=above_count / masks_denominator,
        vw_guard_frac=mask_counts["vw_guard"] / masks_denominator,
        numerator_guard_frac=mask_counts["numerator_guard"] / masks_denominator,
        denominator_guard_frac=mask_counts["denominator_guard"] / masks_denominator,
        kappa_low_bound_frac=mask_counts["kappa_low_bound"] / masks_denominator,
        kappa_high_bound_frac=mask_counts["kappa_high_bound"] / masks_denominator,
        consumption_low_bound_frac=mask_counts["consumption_low_bound"] / masks_denominator,
        consumption_high_bound_frac=mask_counts["consumption_high_bound"] / masks_denominator,
        portfolio_any_bound_frac=mask_counts["portfolio_any_bound"] / masks_denominator,
        portfolio_component_bound_frac=(
            mask_counts["portfolio_component_bound"] / float(portfolio_component_denominator)
        ),
        max_abs_portfolio=max_abs_pi,
        max_consumption_wealth_ratio=max_kappa,
        min_log_wealth=global_min_y,
        max_log_wealth=global_max_y,
        finite_check_passed=1,
    )
    return SimulationResult(payoff, terminal_y, diagnostics)


def optimal_policy_callable(market: MarketData) -> Callable[[float, np.ndarray], PolicyEvaluation]:
    def policy(t: float, y: np.ndarray) -> PolicyEvaluation:
        wealth = np.exp(y)
        kappa = float(optimal_consumption_ratio(market, t))
        consumption = kappa * wealth
        portfolio = np.broadcast_to(market.pi_star, (y.size, market.n_assets)).copy()
        return PolicyEvaluation(consumption, portfolio, {})
    return policy


# ---------------------------------------------------------------------------
# Lazy PyTorch final-greedy policy
# ---------------------------------------------------------------------------


def import_torch(device_spec: str, torch_num_threads: int) -> Tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required unless --validate-only is used") from exc
    torch.set_num_threads(max(1, int(torch_num_threads)))
    requested = str(device_spec).lower()
    if requested == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return torch, nn, device


def build_value_network(torch: Any, nn: Any, hidden: int, depth: int) -> Any:
    layers: List[Any] = []
    in_dim = 2
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))

    class ValueNetLogW(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, t: Any, y: Any) -> Any:
            return self.net(torch.cat([t, y], dim=1))

    return ValueNetLogW()


def load_value_network(
    torch: Any, nn: Any, record: RunRecord, checkpoint: Path, device: Any
) -> Any:
    hidden = int(record.config_args.get("value_hidden", 256))
    depth = int(record.config_args.get("value_depth", 3))
    network = build_value_network(torch, nn, hidden, depth).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"{checkpoint}: checkpoint is not a state dictionary")
    network.load_state_dict(state, strict=True)
    network.eval()
    return network


def network_policy_callable(
    *,
    torch: Any,
    network: Any,
    device: Any,
    market: MarketData,
    contract: PolicyContract,
    policy_chunk: int,
) -> Callable[[float, np.ndarray], PolicyEvaluation]:
    def evaluate(t_value: float, y_values: np.ndarray) -> PolicyEvaluation:
        y_np = np.asarray(y_values, dtype=np.float64).reshape(-1)
        c_parts: List[np.ndarray] = []
        pi_parts: List[np.ndarray] = []
        mask_parts: Dict[str, List[np.ndarray]] = {name: [] for name in _MASK_FIELDS}
        sigma_inv_mu = torch.as_tensor(
            market.sigma_inv_mu, dtype=torch.float32, device=device
        ).reshape(1, -1)
        for start in range(0, y_np.size, policy_chunk):
            chunk = y_np[start:start + policy_chunk]
            y = torch.as_tensor(chunk, dtype=torch.float32, device=device).reshape(-1, 1)
            y.requires_grad_(True)
            t = torch.full_like(y, float(t_value))
            with torch.enable_grad():
                value = network(t, y)
                value_y = torch.autograd.grad(
                    value, y, grad_outputs=torch.ones_like(value),
                    create_graph=True, retain_graph=True,
                )[0]
                value_yy = torch.autograd.grad(
                    value_y, y, grad_outputs=torch.ones_like(value_y),
                    create_graph=False, retain_graph=False,
                )[0]
                wealth = torch.exp(y)
                value_w = value_y / wealth
                vw_guard = value_w < contract.vw_guard
                value_w_safe = torch.clamp(value_w, min=contract.vw_guard)
                c_raw = value_w_safe.pow(-1.0 / market.gamma)
                kappa_raw = c_raw / wealth
                kappa = kappa_raw
                kappa_low = torch.zeros_like(kappa, dtype=torch.bool)
                kappa_high = torch.zeros_like(kappa, dtype=torch.bool)
                if contract.kappa_min is not None:
                    kappa_low = kappa < contract.kappa_min
                    kappa = torch.clamp(kappa, min=contract.kappa_min)
                if contract.kappa_max is not None:
                    kappa_high = kappa > contract.kappa_max
                    kappa = torch.clamp(kappa, max=contract.kappa_max)
                consumption = kappa * wealth
                c_low = torch.zeros_like(consumption, dtype=torch.bool)
                c_high = torch.zeros_like(consumption, dtype=torch.bool)
                if contract.consumption_min is not None:
                    c_low = consumption < contract.consumption_min
                    consumption = torch.clamp(consumption, min=contract.consumption_min)
                if contract.consumption_max is not None:
                    c_high = consumption > contract.consumption_max
                    consumption = torch.clamp(consumption, max=contract.consumption_max)

                numerator_guard = value_y < contract.numerator_guard
                positive_denom = value_y - value_yy
                denominator_guard = positive_denom < contract.denominator_guard
                scalar = torch.clamp(value_y, min=contract.numerator_guard) / torch.clamp(
                    positive_denom, min=contract.denominator_guard
                )
                pi_raw = scalar * sigma_inv_mu
                portfolio = pi_raw
                component_clip = torch.zeros_like(portfolio, dtype=torch.bool)
                if contract.portfolio_min is not None:
                    component_clip |= portfolio < contract.portfolio_min
                    portfolio = torch.clamp(portfolio, min=contract.portfolio_min)
                if contract.portfolio_max is not None:
                    component_clip |= portfolio > contract.portfolio_max
                    portfolio = torch.clamp(portfolio, max=contract.portfolio_max)
                any_clip = torch.any(component_clip, dim=1, keepdim=True)

            def cpu(array: Any) -> np.ndarray:
                return array.detach().cpu().numpy()

            c_parts.append(cpu(consumption).reshape(-1))
            pi_parts.append(cpu(portfolio))
            masks = {
                "vw_guard": vw_guard,
                "numerator_guard": numerator_guard,
                "denominator_guard": denominator_guard,
                "kappa_low_bound": kappa_low,
                "kappa_high_bound": kappa_high,
                "consumption_low_bound": c_low,
                "consumption_high_bound": c_high,
                "portfolio_any_bound": any_clip,
                "portfolio_component_bound": component_clip,
            }
            for name, tensor in masks.items():
                mask_parts[name].append(cpu(tensor).astype(bool, copy=False))
        return PolicyEvaluation(
            np.concatenate(c_parts),
            np.concatenate(pi_parts, axis=0),
            {name: np.concatenate(parts, axis=0) for name, parts in mask_parts.items()},
        )
    return evaluate


# ---------------------------------------------------------------------------
# Output and resume
# ---------------------------------------------------------------------------


DIAGNOSTIC_FIELDS = tuple(SimulationDiagnostics.__dataclass_fields__)
WELFARE_FIELDS = (
    "model_type", "training_seed", "N", "policy", "mc_seed", "n_paths",
    "n_steps", "dt", "w0", "expected_total_utility",
    "se_expected_total_utility", "optimal_mc_total_utility",
    "utility_gap", "se_utility_gap", "q", "se_q", "ce0", "se_ce0", "wl", "se_wl",
    *DIAGNOSTIC_FIELDS,
)


def welfare_row(
    *, model_type: str, training_seed: Any, n_assets: int, policy: str,
    mc_seed: int, n_paths: int, n_steps: int, dt: float, w0: float,
    utility: TotalUtilityStats, optimal_utility: float,
    welfare: PairedWelfareStats, diagnostics: SimulationDiagnostics,
) -> Dict[str, Any]:
    return {
        "model_type": model_type, "training_seed": training_seed, "N": n_assets,
        "policy": policy, "mc_seed": mc_seed, "n_paths": n_paths,
        "n_steps": n_steps, "dt": dt, "w0": w0,
        "expected_total_utility": utility.expected_total_utility,
        "se_expected_total_utility": utility.se_expected_total_utility,
        "optimal_mc_total_utility": optimal_utility,
        **asdict(welfare), **asdict(diagnostics),
    }


WelfareKey = Tuple[str, int, Optional[int]]


def row_key(row: Mapping[str, Any]) -> Optional[WelfareKey]:
    try:
        model = str(row["model_type"])
        n = int(float(row["N"]))
        seed_text = str(row.get("training_seed", "")).strip()
        seed = None if not seed_text else int(float(seed_text))
        return model, n, seed
    except (KeyError, TypeError, ValueError):
        return None


def build_seed_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row["model_type"] == "closed_form":
            continue
        groups.setdefault((str(row["model_type"]), int(row["N"])), []).append(row)
    output: List[Dict[str, Any]] = []
    for (model, n), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in ordered]
        for metric in ("expected_total_utility", "q", "ce0", "wl"):
            mean, std, sem, low, high = mean_std_ci([float(row[metric]) for row in ordered])
            output.append({
                "model_type": model, "N": n, "metric": metric, "n_seeds": len(seeds),
                "mean": mean, "std": std, "sem": sem,
                "ci95_low": low, "ci95_high": high,
                "seeds": ";".join(str(seed) for seed in seeds),
            })
    return output


def _valid_completed_row(row: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if any(field not in row for field in WELFARE_FIELDS):
        return False
    text_fields = {"model_type", "training_seed", "policy"}
    try:
        numbers = {field: float(row[field]) for field in WELFARE_FIELDS if field not in text_fields}
    except (TypeError, ValueError):
        return False
    fractions = [field for field in DIAGNOSTIC_FIELDS if field.endswith("_frac")]
    return (
        all(math.isfinite(value) for value in numbers.values())
        and int(numbers["n_paths"]) == int(args.n_paths)
        and int(numbers["n_steps"]) == int(args.n_steps)
        and int(numbers["mc_seed"]) == int(args.mc_seed)
        and math.isclose(numbers["w0"], float(args.w0), rel_tol=0.0, abs_tol=1e-15)
        and numbers["q"] > 0.0 and numbers["ce0"] > 0.0
        and all(0.0 <= numbers[field] <= 1.0 for field in fractions)
        and int(numbers["finite_check_passed"]) == 1
    )


def load_completed_rows(
    path: Path, expected: set[WelfareKey], args: argparse.Namespace
) -> Dict[WelfareKey, Dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: Dict[WelfareKey, Dict[str, Any]] = {}
    ignored = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row_key(row)
            if key is None or key not in expected or not _valid_completed_row(row, args):
                ignored += 1
                continue
            rows[key] = dict(row)
    if ignored:
        print(f"[resume] ignored {ignored} invalid or unexpected metrics row(s)")
    return rows


def sorted_rows(row_map: Mapping[WelfareKey, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    order = {"closed_form": 0, "pinn": 1, "pipinn": 2}
    return [
        dict(row) for key, row in sorted(
            row_map.items(), key=lambda item: (
                item[0][1], order.get(item[0][0], 99),
                -1 if item[0][2] is None else item[0][2],
            )
        )
    ]


def save_optimal_cache(path: Path, result: SimulationResult, signature: str, n: int) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        schema=np.asarray([1]), signature=np.asarray([signature]), N=np.asarray([n]),
        pathwise_total_utility=result.pathwise_total_utility,
        terminal_log_wealth=result.terminal_log_wealth,
        **{f"diag_{key}": np.asarray([value]) for key, value in asdict(result.diagnostics).items()},
    )
    os.replace(temporary, path)


def load_optimal_cache(path: Path, signature: str, n_paths: int) -> Optional[SimulationResult]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as source:
            if int(source["schema"][0]) != 1 or str(source["signature"][0]) != signature:
                return None
            payoff = np.asarray(source["pathwise_total_utility"], dtype=np.float64).copy()
            terminal = np.asarray(source["terminal_log_wealth"], dtype=np.float64).copy()
            diagnostics = SimulationDiagnostics(**{
                key: float(source[f"diag_{key}"][0])
                if key != "finite_check_passed" else int(source[f"diag_{key}"][0])
                for key in DIAGNOSTIC_FIELDS
            })
        if payoff.shape != (n_paths,) or terminal.shape != (n_paths,):
            return None
        if not np.all(np.isfinite(payoff)) or not np.all(np.isfinite(terminal)):
            return None
        return SimulationResult(payoff, terminal, diagnostics)
    except Exception as exc:
        print(f"[warn] ignoring invalid optimal cache {path}: {exc}")
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Merton consumption-plus-bequest CE0 and welfare loss."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--models", default="both")
    parser.add_argument("--n-assets", default="10,50")
    parser.add_argument(
        "--expected-seeds",
        default="",
        help=(
            "Optional exact training-seed set. Leave empty to use all successful "
            "seeds in the selected configuration; pass the paper seed list for "
            "strict paper validation."
        ),
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help=(
            "Minimum selected successful seeds required per method/dimension "
            "(default: 1). This remains enforced with --allow-incomplete."
        ),
    )
    parser.add_argument(
        "--outer-iters", type=int, default=None,
        help="Select one training budget when the sweep root contains multiple budgets.",
    )
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-checkpoint-fallback", action="store_true")
    parser.add_argument("--n-paths", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=1_000)
    parser.add_argument("--w0", type=float, default=0.5)
    parser.add_argument("--mc-seed", type=int, default=2718)
    parser.add_argument("--path-batch", type=int, default=4096)
    parser.add_argument("--policy-chunk", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.n_paths < 2 or args.n_steps <= 0 or args.path_batch <= 0 or args.policy_chunk <= 0:
        raise ValueError("n_paths>=2 and positive n_steps/path_batch/policy_chunk are required")
    if args.mc_seed < 0:
        raise ValueError("--mc-seed must be nonnegative")
    if not math.isfinite(args.w0) or args.w0 <= 0.0:
        raise ValueError("--w0 must be positive and finite")
    out_root = Path(args.out_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else out_root / "welfare_summary"
    )
    models = normalize_models(args.models)
    n_assets = parse_int_spec(args.n_assets, label="--n-assets")
    expected_seeds = parse_seed_spec(args.expected_seeds)
    if not n_assets:
        raise ValueError("--n-assets cannot be empty")
    if any(n <= 0 for n in n_assets):
        raise ValueError("--n-assets values must be positive")
    if args.min_seeds < 1:
        raise ValueError("--min-seeds must be at least 1")
    if args.outer_iters is not None and args.outer_iters <= 0:
        raise ValueError("--outer-iters must be positive")
    selected = discover_paper_runs(
        out_root, models, n_assets, expected_seeds,
        min_seeds=args.min_seeds,
        outer_iters=args.outer_iters,
        run_name_regex=args.run_name_regex,
        allow_incomplete=args.allow_incomplete,
    )
    checkpoints: Dict[Tuple[str, int, int], Path] = {}
    contracts: Dict[Tuple[str, int, int], PolicyContract] = {}
    network_contracts: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    run_payload: List[Dict[str, Any]] = []
    for cell in sorted(selected):
        for record in selected[cell]:
            key = (record.model_type, record.n_assets, record.seed)
            checkpoint = resolve_checkpoint(record, out_root, args.allow_checkpoint_fallback)
            checkpoints[key] = checkpoint
            contracts[key] = policy_contract(record)
            network_contracts[key] = network_contract(record)
            run_payload.append({
                "model_type": record.model_type, "N": record.n_assets,
                "training_seed": record.seed, "run_dir": str(record.run_dir),
                "group": record.group, "market_hash": record.market_hash,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "policy_contract": asdict(contracts[key]),
                "network_contract": network_contracts[key],
            })
    markets = validate_numpy_inputs(selected, args.w0)
    signature_payload = {
        "schema": RESUME_SCHEMA_VERSION,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "protocol": {
            "models": models, "n_assets": n_assets, "expected_seeds": expected_seeds,
            "min_seeds": int(args.min_seeds),
            "selected_seeds": {
                f"{model}:N{n}": [record.seed for record in selected[(model, n)]]
                for model in models for n in n_assets
            },
            "n_paths": args.n_paths, "n_steps": args.n_steps, "w0": args.w0,
            "mc_seed": args.mc_seed, "path_batch": args.path_batch,
            "policy_chunk": args.policy_chunk,
            "outer_iters": args.outer_iters,
            "run_name_regex": args.run_name_regex,
            "allow_incomplete": bool(args.allow_incomplete),
            "allow_checkpoint_fallback": bool(args.allow_checkpoint_fallback),
            "torch_num_threads": int(args.torch_num_threads),
            "network_dtype": "float32",
            "simulation_dtype": "float64",
            "wealth_scheme": "log-Euler; left-Riemann running utility",
            "objective": "E[int exp(-rho*t)U(c)dt + exp(-rho*T)epsilon*U(W_T)]",
            "optimal_denominator": "optimal policy Monte Carlo under identical discretization/CRN",
            "wealth_domain_behavior": "raw network extrapolation; never clipped to training domain",
        },
        "runs": run_payload,
    }
    signature = canonical_json_hash(signature_payload)
    config_path = output / "welfare_config.json"
    metrics_path = output / "welfare_metrics.csv"
    summary_path = output / "welfare_seed_summary.csv"
    validation_path = output / "welfare_validation.csv"

    prior: Optional[Dict[str, Any]] = None
    if not args.no_resume and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if str(prior.get("resume_signature", "")) != signature:
            raise ResumeSignatureError(
                "existing welfare output has a different protocol/checkpoint signature; "
                "use --no-resume to replace it intentionally"
            )
    elif not args.no_resume and output.exists() and any(output.iterdir()):
        raise ResumeSignatureError(
            f"existing artifacts under {output} have no compatible welfare_config.json; "
            "use --no-resume"
        )

    expected_keys: set[WelfareKey] = {
        ("closed_form", n, None) for n in n_assets
    }
    for records in selected.values():
        expected_keys.update((r.model_type, r.n_assets, r.seed) for r in records)
    row_map = {} if args.no_resume else load_completed_rows(metrics_path, expected_keys, args)

    # Resolve the learned-policy runtime before mutating any output. A partial
    # resume must not silently mix CPU/GPU autograd or library versions.
    pending_learned = any(
        key[0] != "closed_form" for key in (expected_keys - set(row_map))
    )
    torch = nn = device = None
    if not args.validate_only and pending_learned:
        torch, nn, device = import_torch(args.device, args.torch_num_threads)
        if prior is not None and not args.no_resume:
            prior_numpy = str(prior.get("runtime_numpy_version", ""))
            prior_torch = str(prior.get("runtime_torch_version", ""))
            prior_device = str(prior.get("runtime_device", ""))
            if prior_numpy and prior_numpy != np.__version__:
                raise ResumeSignatureError(
                    f"resume NumPy version changed ({prior_numpy} -> {np.__version__}); "
                    "use --no-resume"
                )
            if prior_torch and prior_torch != str(torch.__version__):
                raise ResumeSignatureError(
                    f"resume PyTorch version changed ({prior_torch} -> {torch.__version__}); "
                    "use --no-resume"
                )
            if prior_device and prior_device != str(device):
                raise ResumeSignatureError(
                    f"resume device changed ({prior_device} -> {device}); use --no-resume"
                )

    provenance = {
        "created_at": prior.get("created_at", utc_now()) if prior else utc_now(),
        "updated_at": utc_now(),
        "status": (
            str(prior.get("status", "validated"))
            if args.validate_only and prior is not None else
            ("validated" if args.validate_only else "running")
        ),
        "resume_signature": signature,
        "resume_signature_payload": signature_payload,
        "arguments": vars(args),
        "runtime_numpy_version": np.__version__,
        "fixed_protocol": {
            "initial_wealth": args.w0,
            "running_quadrature": "discounted left Riemann sum",
            "terminal_bequest": "exp(-rho*T)*epsilon*U(W_T)",
            "ce0": "w0*(J/J_optimal_MC)^(1/(1-gamma))",
            "wl": "1-CE0/w0",
            "common_random_numbers": "same per-path-batch asset Brownian draws",
            "official_checkpoint": "value_net_final.pt",
            "network_domain_extension": "unprojected/raw; exits diagnosed",
            "analytic_value_audit": (
                "MC-minus-analytic and z-score are discretization/MC diagnostics; "
                "the z-score is not a hypothesis test because deterministic "
                "time-discretization bias is present"
            ),
            "path_time_exit_denominator": (
                "pre-step policy-evaluation states only; path-level exit also "
                "includes the terminal state"
            ),
        },
        "runs": run_payload,
    }
    if torch is not None:
        provenance["runtime_torch_version"] = str(torch.__version__)
        provenance["runtime_device"] = str(device)
    elif prior is not None:
        for key in ("runtime_torch_version", "runtime_device"):
            if key in prior:
                provenance[key] = prior[key]
    output.mkdir(parents=True, exist_ok=True)
    if args.no_resume:
        row_map = {}
        write_csv(metrics_path, [], WELFARE_FIELDS)
    write_json(config_path, provenance)
    write_csv(metrics_path, sorted_rows(row_map), WELFARE_FIELDS)

    validation_rows: List[Dict[str, Any]] = []
    for n in n_assets:
        market = markets[n]
        validation_rows.append({
            "N": n, "market_hash": selected[(models[0], n)][0].market_hash,
            "analytic_optimal_value": analytic_optimal_value(market, args.w0),
            "optimal_mc_total_utility": "", "optimal_mc_se": "",
            "mc_minus_analytic": "", "mc_z_score": "",
            "gamma": market.gamma, "rho_discount": market.discount,
            "epsilon_bequest": market.bequest, "T": market.horizon,
        })
    validation_fields = (
        "N", "market_hash", "analytic_optimal_value", "optimal_mc_total_utility",
        "optimal_mc_se", "mc_minus_analytic", "mc_z_score", "gamma",
        "rho_discount", "epsilon_bequest", "T",
    )
    write_csv(validation_path, validation_rows, validation_fields)
    summary_fields = (
        "model_type", "N", "metric", "n_seeds", "mean", "std", "sem",
        "ci95_low", "ci95_high", "seeds",
    )
    write_csv(summary_path, build_seed_summary(sorted_rows(row_map)), summary_fields)
    if args.validate_only:
        provenance["updated_at"] = utc_now()
        write_json(config_path, provenance)
        print(f"[validated] {len(run_payload)} official checkpoints; output={output}")
        return

    optimal_by_n: Dict[int, SimulationResult] = {}
    for n in n_assets:
        market = markets[n]
        cache_path = output / f"optimal_paths_N{n}.npz"
        optimal = None if args.no_resume else load_optimal_cache(
            cache_path, signature, args.n_paths
        )
        if optimal is None:
            print(f"[simulate] N={n} optimal policy")
            optimal = simulate_policy(
                market, policy=optimal_policy_callable(market), n_paths=args.n_paths,
                n_steps=args.n_steps, w0=args.w0, mc_seed=args.mc_seed,
                path_batch=args.path_batch,
            )
            save_optimal_cache(cache_path, optimal, signature, n)
        else:
            print(f"[resume] N={n} optimal pathwise utility cache")
        optimal_by_n[n] = optimal
        optimal_stats = total_utility_statistics(optimal.pathwise_total_utility)
        optimal_welfare = paired_welfare_statistics(
            optimal.pathwise_total_utility, optimal.pathwise_total_utility,
            market.gamma, args.w0,
        )
        row_map[("closed_form", n, None)] = welfare_row(
            model_type="closed_form", training_seed="", n_assets=n,
            policy="optimal", mc_seed=args.mc_seed, n_paths=args.n_paths,
            n_steps=args.n_steps, dt=market.horizon / args.n_steps, w0=args.w0,
            utility=optimal_stats,
            optimal_utility=optimal_stats.expected_total_utility,
            welfare=optimal_welfare, diagnostics=optimal.diagnostics,
        )
        exact = analytic_optimal_value(market, args.w0)
        difference = optimal_stats.expected_total_utility - exact
        validation_rows[n_assets.index(n)].update({
            "optimal_mc_total_utility": optimal_stats.expected_total_utility,
            "optimal_mc_se": optimal_stats.se_expected_total_utility,
            "mc_minus_analytic": difference,
            "mc_z_score": (
                difference / optimal_stats.se_expected_total_utility
                if optimal_stats.se_expected_total_utility > 0.0 else float("nan")
            ),
        })
        write_csv(metrics_path, sorted_rows(row_map), WELFARE_FIELDS)
        write_csv(validation_path, validation_rows, validation_fields)

    for model in models:
        for n in n_assets:
            market = markets[n]
            optimal = optimal_by_n[n]
            optimal_mean = total_utility_statistics(
                optimal.pathwise_total_utility
            ).expected_total_utility
            for record in selected[(model, n)]:
                key = (model, n, record.seed)
                if key in row_map:
                    print(f"[resume] model={model}, N={n}, seed={record.seed}")
                    continue
                if torch is None or nn is None or device is None:
                    raise RuntimeError("internal error: pending learned policy has no PyTorch runtime")
                checkpoint = checkpoints[key]
                # Detect a file replacement after the resume signature/hash preflight.
                expected_hash = next(
                    item["checkpoint_sha256"] for item in run_payload
                    if item["model_type"] == model and item["N"] == n
                    and item["training_seed"] == record.seed
                )
                if sha256_file(checkpoint) != expected_hash:
                    raise RuntimeError(f"checkpoint changed during evaluation: {checkpoint}")
                print(f"[simulate] model={model}, N={n}, seed={record.seed}")
                network = load_value_network(torch, nn, record, checkpoint, device)
                learned = simulate_policy(
                    market,
                    policy=network_policy_callable(
                        torch=torch, network=network, device=device, market=market,
                        contract=contracts[key], policy_chunk=args.policy_chunk,
                    ),
                    n_paths=args.n_paths, n_steps=args.n_steps, w0=args.w0,
                    mc_seed=args.mc_seed, path_batch=args.path_batch,
                )
                utility = total_utility_statistics(learned.pathwise_total_utility)
                welfare = paired_welfare_statistics(
                    learned.pathwise_total_utility, optimal.pathwise_total_utility,
                    market.gamma, args.w0,
                )
                row_map[key] = welfare_row(
                    model_type=model, training_seed=record.seed, n_assets=n,
                    policy="greedy_final", mc_seed=args.mc_seed,
                    n_paths=args.n_paths, n_steps=args.n_steps,
                    dt=market.horizon / args.n_steps, w0=args.w0,
                    utility=utility, optimal_utility=optimal_mean,
                    welfare=welfare, diagnostics=learned.diagnostics,
                )
                write_csv(metrics_path, sorted_rows(row_map), WELFARE_FIELDS)
                write_csv(
                    summary_path, build_seed_summary(sorted_rows(row_map)), summary_fields
                )
                del network
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    missing = expected_keys - set(row_map)
    if missing:
        raise RuntimeError(f"evaluation ended with missing welfare rows: {sorted(missing)}")
    provenance["status"] = "success"
    provenance["updated_at"] = utc_now()
    provenance["completed_rows"] = len(row_map)
    write_json(config_path, provenance)
    write_csv(metrics_path, sorted_rows(row_map), WELFARE_FIELDS)
    write_csv(summary_path, build_seed_summary(sorted_rows(row_map)), summary_fields)
    print(f"[done] Merton total-lifetime welfare evaluation: {output}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        # Do not overwrite a prior valid signature on a preflight failure.
        print(f"[error] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
