"""Residual-tolerance aggregation -> error-floor scaling.

Two deliberately distinct protocols are supported:

* Paper E6: each target branch starts from one seed-level common warm-up.
* Independent diagnostic: a standard PI-PINN run starts from the ordinary
  analytic/myopic initialization separately for every target.

The independent protocol is opt-in via ``--standard-independent`` and is
never pooled with common-warm-start branches.  In either protocol the
aggregator collects the requested target and official ACHIEVED residual level
from the post-restore fixed-Q_res column in ``outer_history.csv``.  For paper
E6 the separate common warm-up is excluded structurally; for an independent
standard run all outer iterations are included.  It also records
target-reached status, optimizer steps, the final ``e_Xev`` trajectory
diagnostic, and full-dimensional value/normalized-control errors.  Then, per
configuration group (everything shared except training seed and pres_target):

  per_target.csv   n_runs, n_target_reached, achieved p_res/error/steps
                   mean, sample SD, SEM, and Student-t 95% CI per target
  points.csv       one row per run (x = achieved p_res, y = error)
  fit.csv          log10(error) ~ log10(achieved p_res), including the
                   primary seed-cluster-robust slope interval
  e6_error_floor_*.{png,pdf,svg,eps}
                   one log-log figure per metric, with seed points,
                   target mean +/- sample SD, the pooled fitted slope,
                   and a slope-one reference

Per the protocol: runs that did NOT reach their target enter the fit with
their ACHIEVED residual (never the nominal target); reached runs also use
the achieved value, which is <= target by construction, so the x-axis is
uniformly "achieved p_res".

Standalone: stdlib + numpy (+ scipy if available for exact t quantiles).

Usage:
    python3 aggregate_e6.py --out-root <OUT_ROOT>
    python3 aggregate_e6.py --out-root <OUT_ROOT> --metrics RelL2_V,RelL2_theta
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
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from aggregate_seeds import (  # reuse shared helpers
    GROUP_IGNORE_KEYS, canonical_market_hash, find_runs, load_config_args,
    load_config_args_raw, parse_seed_spec, run_status, run_updated_at,
    t_crit_95, fmt,
)

# E6 groups additionally collapse over the tolerance itself.
E6_IGNORE_KEYS = set(GROUP_IGNORE_KEYS) | {"pres_target"}

E6_WARMUP_ROLE = "warmup"
E6_TARGET_BRANCH_ROLE = "target_branch"
E6_STANDARD_ROLE = "standard"
E6_TARGET_PHASE = "target"
E6_STANDARD_PHASE = "standard"
E6_WARM_START_PROTOCOL = "liu-e6-common-warm-start-v1"
E6_TARGET_RESIDUAL_SEMANTICS = (
    "max_target_phase_outer_post_restore_fixed_qres_excluding_warmup"
)
E6_TARGET_REACHED_SEMANTICS = (
    "all_target_phase_outer_post_restore_fixed_qres_at_or_below_target_"
    "excluding_warmup"
)
E6_STANDARD_RESIDUAL_SEMANTICS = "max_outer_post_restore_fixed_qres"
E6_STANDARD_TARGET_REACHED_SEMANTICS = (
    "all_outer_post_restore_fixed_qres_at_or_below_target"
)
E6_WARM_START_STATUS_FIELDS = (
    "e6_warm_start_protocol",
    "e6_warm_start_source",
    "e6_warm_start_id",
    "e6_warm_start_model_sha256",
    "e6_warm_start_optimizer_sha256",
    "e6_warm_start_rng_sha256",
    "e6_warm_start_bundle_sha256",
    "e6_warm_start_loaded_bundle_sha256",
    "e6_warmup_target",
    "e6_warmup_post_restore_pres",
    "e6_warmup_optimizer_steps",
    "e6_target_phase_outer_count",
    "e6_target_phase_start_algorithm_iter",
    "first_target_policy_source",
)
E6_COMMON_WARM_START_FIELDS = (
    "e6_warm_start_protocol",
    "e6_warm_start_source",
    "e6_warm_start_id",
    "e6_warm_start_model_sha256",
    "e6_warm_start_optimizer_sha256",
    "e6_warm_start_rng_sha256",
    "e6_warm_start_bundle_sha256",
    "e6_warmup_target",
    "e6_warmup_post_restore_pres",
    "e6_warmup_optimizer_steps",
    "e6_target_phase_start_algorithm_iter",
    "first_target_policy_source",
)
# These values identify a realized, seed-specific bundle or its filesystem
# location.  They must not split one residual-target panel into separate
# training groups.  The role, protocol, branch budget, and policy-source
# contract remain in the group key.
E6_IGNORE_KEYS.update({
    "e6_warm_start",
    "e6_warmup_bundle",
    "e6_warm_start_source",
    "e6_warm_start_id",
    "e6_warm_start_model_sha256",
    "e6_warm_start_optimizer_sha256",
    "e6_warm_start_rng_sha256",
    "e6_warm_start_bundle_sha256",
    "e6_warm_start_loaded_bundle_sha256",
    "e6_warmup_post_restore_pres",
    "e6_warmup_optimizer_steps",
})


def e6_group_key(args: Dict[str, Any]) -> str:
    core = {k: args[k] for k in sorted(args) if k not in E6_IGNORE_KEYS}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def read_status(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "status.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def residual_history_summary(
    run_dir: str,
    *,
    allow_legacy: bool = False,
    e6_role: str = "",
) -> Dict[str, Any]:
    """Read the residual trajectory with explicit checkpoint-state semantics.

    A paper ``target_branch`` must contain only target-phase rows.  An
    independent ``standard`` run must contain only standard-phase rows.  Both
    must use ``val_pres_post_restore``: the held-out residual of the exact
    restored checkpoint subsequently used for evaluation.  This makes the
    protocol boundary structural instead of an analyst-selected outer cutoff.
    """
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return {
            "achieved": None,
            "semantics": "missing",
            "target_row_count": 0,
            "residual_row_count": 0,
            "outer_iters": [],
            "algorithm_outer_iters": [],
        }
    role = str(e6_role).strip()
    target_branch = role == E6_TARGET_BRANCH_ROLE
    standard_run = role == E6_STANDARD_ROLE
    strict_current_role = target_branch or standard_run
    expected_phase = E6_TARGET_PHASE if target_branch else E6_STANDARD_PHASE
    role_label = E6_TARGET_BRANCH_ROLE if target_branch else E6_STANDARD_ROLE
    best: Optional[float] = None
    residual_row_count = 0
    outer_iters: List[int] = []
    algorithm_outer_iters: List[int] = []
    reset_modes: List[Optional[int]] = []
    outer_start_lrs: List[Optional[float]] = []
    next_outer_lrs: List[Optional[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        if "val_pres_post_restore" in fields:
            field = "val_pres_post_restore"
            semantics = "max_outer_post_restore_fixed_qres"
        elif allow_legacy and "val_pres" in fields:
            field = "val_pres"
            semantics = "legacy_val_pres_explicitly_allowed"
        else:
            return {
                "achieved": None,
                "semantics": "missing_post_restore_residual",
                "target_row_count": 0,
                "residual_row_count": 0,
                "outer_iters": [],
                "algorithm_outer_iters": [],
            }
        if strict_current_role:
            missing = sorted(
                {"outer_iter", "algorithm_outer_iter", "e6_phase"} - fields
            )
            if missing:
                raise ValueError(
                    f"{path}: {role_label} residual history is missing "
                    f"required columns {missing}"
                )
            if field != "val_pres_post_restore":
                raise ValueError(
                    f"{path}: {role_label} requires val_pres_post_restore"
                )
            semantics = (
                E6_TARGET_RESIDUAL_SEMANTICS
                if target_branch
                else E6_STANDARD_RESIDUAL_SEMANTICS
            )
        for row in reader:
            if strict_current_role:
                phase = str(row.get("e6_phase", "")).strip()
                if phase != expected_phase:
                    raise ValueError(
                        f"{path}: {role_label} contains e6_phase={phase!r}; "
                        f"all rows must be {expected_phase!r}"
                    )
                try:
                    outer = int(float(str(row.get("outer_iter", "")).strip()))
                    algorithm_outer = int(
                        float(str(row.get("algorithm_outer_iter", "")).strip())
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}: invalid {expected_phase}-phase outer index"
                    ) from exc
                outer_iters.append(outer)
                algorithm_outer_iters.append(algorithm_outer)
                residual_row_count += 1
                try:
                    raw_reset = float(
                        str(row.get("e6_reset_lr_each_outer", "")).strip()
                    )
                    reset_modes.append(
                        int(raw_reset) if raw_reset in {0.0, 1.0} else None
                    )
                except (TypeError, ValueError):
                    reset_modes.append(None)
                for column, destination in (
                    ("outer_start_lr", outer_start_lrs),
                    ("lr_carried_next", next_outer_lrs),
                ):
                    try:
                        value = float(str(row.get(column, "")).strip())
                        destination.append(
                            value if math.isfinite(value) and value > 0.0
                            else None
                        )
                    except (TypeError, ValueError):
                        destination.append(None)
            raw = str(row.get(field, "")).strip()
            try:
                v = float(raw)
            except (TypeError, ValueError):
                if strict_current_role:
                    raise ValueError(
                        f"{path}: every {expected_phase} row requires a positive finite "
                        f"{field}; got {raw!r}"
                    )
                continue
            if not math.isfinite(v) or v <= 0.0:
                if strict_current_role:
                    raise ValueError(
                        f"{path}: every {expected_phase} row requires a positive finite "
                        f"{field}; got {raw!r}"
                    )
                continue
            best = v if best is None else max(best, v)
    if strict_current_role:
        if residual_row_count == 0:
            raise ValueError(
                f"{path}: {role_label} has no {expected_phase}-phase rows"
            )
        if len(set(outer_iters)) != len(outer_iters):
            raise ValueError(
                f"{path}: duplicate outer_iter in {expected_phase} phase"
            )
        if len(set(algorithm_outer_iters)) != len(algorithm_outer_iters):
            raise ValueError(
                f"{path}: duplicate algorithm_outer_iter in {expected_phase} phase"
            )
        ordered_outer = sorted(outer_iters)
        if ordered_outer != list(range(1, len(ordered_outer) + 1)):
            raise ValueError(
                f"{path}: {expected_phase}-phase outer_iter must be contiguous from 1; "
                f"found {ordered_outer}"
            )
        ordered_algorithm = sorted(algorithm_outer_iters)
        expected_algorithm = list(
            range(ordered_algorithm[0], ordered_algorithm[0] + len(ordered_algorithm))
        )
        if ordered_algorithm != expected_algorithm:
            raise ValueError(
                f"{path}: {expected_phase}-phase algorithm_outer_iter must be contiguous; "
                f"found {ordered_algorithm}"
            )
        if standard_run and ordered_algorithm[0] != 1:
            raise ValueError(
                f"{path}: standard-phase algorithm_outer_iter must start at 1; "
                f"found {ordered_algorithm}"
            )
    return {
        "achieved": best,
        "semantics": semantics,
        "target_row_count": residual_row_count if target_branch else 0,
        "residual_row_count": residual_row_count,
        "outer_iters": sorted(outer_iters),
        "algorithm_outer_iters": sorted(algorithm_outer_iters),
        "e6_reset_lr_each_outer": reset_modes,
        "outer_start_lrs": outer_start_lrs,
        "lr_carried_next": next_outer_lrs,
    }


def achieved_pres_from_outer_history(
    run_dir: str,
    *,
    allow_legacy: bool = False,
    e6_role: str = "",
) -> Tuple[Optional[float], str]:
    summary = residual_history_summary(
        run_dir, allow_legacy=allow_legacy, e6_role=e6_role
    )
    return summary["achieved"], str(summary["semantics"])


def target_branch_warm_start_provenance(
    run_dir: str,
    *,
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    history: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and return the auditable seed-level warm-start contract."""
    errors: List[str] = []
    provenance = {key: status.get(key) for key in E6_WARM_START_STATUS_FIELDS}
    for key in E6_WARM_START_STATUS_FIELDS:
        value = provenance.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"{run_dir}: target_branch status missing {key}")

    if str(provenance.get("e6_warm_start_protocol") or "").strip() != (
        E6_WARM_START_PROTOCOL
    ):
        errors.append(
            f"{run_dir}: e6_warm_start_protocol must be "
            f"{E6_WARM_START_PROTOCOL!r}"
        )

    hash_fields = (
        "e6_warm_start_model_sha256",
        "e6_warm_start_optimizer_sha256",
        "e6_warm_start_rng_sha256",
        "e6_warm_start_bundle_sha256",
        "e6_warm_start_loaded_bundle_sha256",
    )
    for key in hash_fields:
        value = str(provenance.get(key) or "").strip().lower()
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            errors.append(
                f"{run_dir}: {key} must be a 64-character lowercase SHA256"
            )
        provenance[key] = value
    source_bundle = str(provenance.get("e6_warm_start_bundle_sha256") or "")
    loaded_bundle = str(
        provenance.get("e6_warm_start_loaded_bundle_sha256") or ""
    )
    if source_bundle and loaded_bundle and source_bundle != loaded_bundle:
        errors.append(
            f"{run_dir}: loaded warm-start bundle hash does not match source "
            f"({loaded_bundle} != {source_bundle})"
        )

    def finite_float(key: str) -> Optional[float]:
        try:
            value = float(provenance.get(key))
        except (TypeError, ValueError):
            errors.append(f"{run_dir}: {key} must be finite")
            return None
        if not math.isfinite(value):
            errors.append(f"{run_dir}: {key} must be finite")
            return None
        provenance[key] = value
        return value

    warmup_target = finite_float("e6_warmup_target")
    warmup_pres = finite_float("e6_warmup_post_restore_pres")
    if warmup_target is not None and not math.isclose(
        warmup_target, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append(
            f"{run_dir}: e6_warmup_target must equal 1.0, got {warmup_target:g}"
        )
    if warmup_pres is not None:
        if warmup_pres <= 0.0:
            errors.append(
                f"{run_dir}: e6_warmup_post_restore_pres must be positive"
            )
        if (
            warmup_target is not None
            and warmup_pres > warmup_target * (1.0 + 1e-9)
        ):
            errors.append(
                f"{run_dir}: warm-up checkpoint did not achieve its target "
                f"({warmup_pres:.6g} > {warmup_target:.6g})"
            )

    for key in (
        "e6_warmup_optimizer_steps",
        "e6_target_phase_outer_count",
        "e6_target_phase_start_algorithm_iter",
    ):
        raw = provenance.get(key)
        try:
            number = float(raw)
            value = int(number)
        except (TypeError, ValueError):
            errors.append(f"{run_dir}: {key} must be an integer")
            continue
        if not math.isfinite(number) or number != value:
            errors.append(f"{run_dir}: {key} must be an integer")
            continue
        provenance[key] = value
        if key == "e6_warmup_optimizer_steps" and value < 0:
            errors.append(f"{run_dir}: {key} must be nonnegative")
        if key != "e6_warmup_optimizer_steps" and value < 1:
            errors.append(f"{run_dir}: {key} must be positive")

    if str(provenance.get("first_target_policy_source") or "").strip() != (
        "warm_start_value_net"
    ):
        errors.append(
            f"{run_dir}: first_target_policy_source must be "
            "'warm_start_value_net'"
        )

    target_count = provenance.get("e6_target_phase_outer_count")
    history_count = int(history.get("target_row_count", 0))
    if isinstance(target_count, int) and target_count != history_count:
        errors.append(
            f"{run_dir}: e6_target_phase_outer_count={target_count} does not "
            f"match target history rows={history_count}"
        )
    try:
        configured_count = int(config.get("outer_iters"))
    except (TypeError, ValueError):
        configured_count = None
    if (
        isinstance(target_count, int)
        and configured_count is not None
        and target_count != configured_count
    ):
        errors.append(
            f"{run_dir}: e6_target_phase_outer_count={target_count} does not "
            f"match config outer_iters={configured_count}"
        )
    algorithm_iters = list(history.get("algorithm_outer_iters", ()))
    start = provenance.get("e6_target_phase_start_algorithm_iter")
    if algorithm_iters and isinstance(start, int) and start != algorithm_iters[0]:
        errors.append(
            f"{run_dir}: e6_target_phase_start_algorithm_iter={start} does not "
            f"match first history algorithm_outer_iter={algorithm_iters[0]}"
        )
    if isinstance(start, int) and start != 2:
        errors.append(
            f"{run_dir}: one-outer common warm-up requires target phase to "
            f"start at algorithm_outer_iter=2, got {start}"
        )
    return provenance, errors


def total_steps_from_outer_history(run_dir: str) -> Optional[float]:
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None
    tot = 0.0
    seen = False
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("inner_epochs_used", "")).strip()
            if not raw:
                continue
            try:
                tot += float(raw)
                seen = True
            except ValueError:
                continue
    return tot if seen else None


def pick_metric_value(run_dir: str, metric: str) -> Optional[Tuple[float, float]]:
    """Return (value, eval_margin_used) for the requested metric.

    FULL-DIMENSIONAL rows only, at the run's PRIMARY margin (first listed in
    config; smallest recorded margin as a fallback). Legacy slice rows are
    ignored.
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("metric", "")) != metric:
                continue
            if str(row.get("scope", "") or "slice") != "fulldim":
                continue
            try:
                val = float(row["value"])
            except (KeyError, ValueError):
                continue
            mraw = str(row.get("eval_margin", "")).strip()
            margin = float(mraw) if mraw else 0.0
            rows.append((margin, val))
    if not rows:
        return None

    cfg = load_config_args(run_dir) or {}
    primary = None
    try:
        margins = [float(x) for x in str(cfg.get("eval_margin", "")).split(",") if str(x).strip()]
        primary = margins[0] if margins else None
    except ValueError:
        primary = None

    if primary is not None:
        exact = [r for r in rows if abs(r[0] - primary) < 1e-12]
        if exact:
            m, v = exact[-1]
            return (v, m)
    m, v = sorted(rows, key=lambda z: z[0])[0]
    return (v, m)


def _primary_margin_from_raw_config(run_dir: str) -> Optional[float]:
    cfg = load_config_args_raw(run_dir) or {}
    try:
        values = [
            float(part)
            for part in str(cfg.get("eval_margin", "")).split(",")
            if str(part).strip()
        ]
    except (TypeError, ValueError):
        return None
    if not values or not math.isfinite(values[0]):
        return None
    return float(values[0])


def pick_outer_metric_value(
    run_dir: str, metric: str
) -> Optional[Tuple[float, float]]:
    """Return the final finite target-phase trajectory diagnostic.

    ``e_Xev`` is recorded during training rather than in ``metrics.csv``.
    Its evaluation margin therefore comes from the raw training config, never
    from a later eval-only overlay.
    """
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None
    rows: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if metric not in set(reader.fieldnames or ()):
            return None
        for row in reader:
            try:
                outer = int(float(str(row.get("outer_iter", "")).strip()))
            except (TypeError, ValueError):
                continue
            if outer in rows:
                raise ValueError(
                    f"{path}: duplicate outer_iter={outer}; final {metric} is ambiguous"
                )
            rows[outer] = str(row.get(metric, ""))
    if not rows:
        return None
    status = read_status(run_dir)
    declared = status.get("final_outer_iter")
    try:
        final_outer = (
            int(declared) if declared not in (None, "") else max(rows)
        )
    except (TypeError, ValueError):
        return None
    if final_outer != max(rows) or final_outer not in rows:
        raise ValueError(
            f"{path}: final_outer_iter={final_outer} does not match "
            f"outer-history maximum={max(rows)}"
        )
    try:
        value = float(rows[final_outer])
    except (TypeError, ValueError):
        return None
    margin = _primary_margin_from_raw_config(run_dir)
    if not math.isfinite(value) or value <= 0.0 or margin is None:
        return None
    return value, margin


def pick_error_value(run_dir: str, metric: str) -> Optional[Tuple[float, float]]:
    if metric == "e_Xev":
        return pick_outer_metric_value(run_dir, metric)
    return pick_metric_value(run_dir, metric)


def mean_std(vals: List[float]) -> Tuple[float, float, int]:
    a = np.asarray(vals, dtype=float)
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if n > 1 else float("nan"), n


def parse_float_spec(text: str) -> List[float]:
    """Parse a comma/space separated exact tolerance set."""

    values: List[float] = []
    for token in re.split(r"[\s,]+", str(text or "").strip()):
        if not token:
            continue
        value = float(token)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"tolerances must be positive finite values, got {token!r}")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate tolerances in {text!r}")
    return sorted(values)


def _float_key(value: float) -> str:
    return f"{float(value):.17g}"


def cluster_robust_slope_se(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    """Seed-cluster-robust (CR1) standard error for the log-log OLS slope.

    Sandwich estimator with the small-G correction G/(G-1) * (n-1)/(n-k);
    CI uses t with G-1 degrees of freedom.
    """
    lx, ly = np.log10(x), np.log10(y)
    n = int(lx.size)
    labels = np.unique(clusters)
    G = int(labels.size)
    out = {"G": G, "se": float("nan"), "ci_lo": float("nan"),
           "ci_hi": float("nan"), "t_crit": float("nan")}
    if n < 2 or G < 2:
        return out
    X = np.vstack([lx, np.ones(n)]).T
    # Degenerate design (all achieved residuals identical -- e.g. every run
    # hit the same floor): the slope is NOT identified. Report NaN instead of
    # crashing or faking a number via pseudoinverse.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return out
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ ly)
    u = ly - X @ beta
    meat = np.zeros((2, 2))
    for gname in labels:
        m_ = clusters == gname
        s_g = X[m_].T @ u[m_]
        meat += np.outer(s_g, s_g)
    k = 2
    corr = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    V = corr * XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 0.0)))
    tc = t_crit_95(G - 1)
    slope = float(beta[0])
    out.update({"se": se, "ci_lo": slope - tc * se, "ci_hi": slope + tc * se, "t_crit": tc})
    return out


def ols_loglog(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS of log10(y) on log10(x): slope, intercept, stderr(slope), 95% CI, R^2."""
    lx, ly = np.log10(x), np.log10(y)
    n = int(lx.size)
    out: Dict[str, float] = {"n": n}
    if n < 2:
        for k in ["slope", "intercept", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "r2"]:
            out[k] = float("nan")
        return out
    X = np.vstack([lx, np.ones(n)]).T
    if np.linalg.matrix_rank(X) < X.shape[1]:
        # Constant x: slope unidentified.
        for k in ["slope", "intercept", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "r2"]:
            out[k] = float("nan")
        return out
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    resid = ly - X @ beta
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(float(np.sum((ly - ly.mean()) ** 2)), 1e-300)
    if n > 2:
        s2 = float(np.sum(resid ** 2)) / (n - 2)
        sxx = float(np.sum((lx - lx.mean()) ** 2))
        se = math.sqrt(s2 / max(sxx, 1e-300))
        tc = t_crit_95(n - 2)
        out.update({"slope_se": se, "ci95_lo": slope - tc * se,
                    "ci95_hi": slope + tc * se, "t_crit": tc})
    else:
        out.update({"slope_se": float("nan"), "ci95_lo": float("nan"),
                    "ci95_hi": float("nan"), "t_crit": float("nan")})
    out.update({"slope": slope, "intercept": intercept, "r2": r2})
    return out


def _summary_stats(values: List[float]) -> Dict[str, float]:
    mean, std, n = mean_std(values)
    if n > 1 and math.isfinite(std):
        sem = std / math.sqrt(n)
        half = t_crit_95(n - 1) * sem
        lo, hi = mean - half, mean + half
    else:
        sem = lo = hi = float("nan")
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "ci95_lo": lo, "ci95_hi": hi,
    }


def _parse_formats(text: str) -> List[str]:
    formats = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    allowed = {"png", "pdf", "svg", "eps"}
    if not formats or len(formats) != len(set(formats)) or any(x not in allowed for x in formats):
        raise ValueError(
            f"--formats must be a unique comma-separated subset of {sorted(allowed)}"
        )
    return formats


def _owned_e6_paths(output: Path) -> List[Path]:
    fixed = [
        output / "points.csv", output / "per_target.csv", output / "fit.csv",
        output / "e6_metadata.json", output / "_SUCCESS_E6",
    ]
    return fixed + sorted(output.glob("e6_error_floor_*.*")) if output.is_dir() else fixed


def _prepare_e6_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output exists and is not a directory: {output}")
    existing = [path for path in _owned_e6_paths(output) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "E6 output already contains managed artifacts; pass --overwrite to replace them: "
            + ", ".join(path.name for path in existing)
        )


def _plot_e6_cells(
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    output: Path,
    args: argparse.Namespace,
    formats: List[str],
) -> List[str]:
    if args.skip_plot:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required unless --skip-plot is used") from exc

    plt.rcParams.update({"font.size": args.font_size})
    metric_labels = {
        "e_Xev": r"Final $e_{X_{\mathrm{ev}}}$",
        "RelL2_V": r"Final relative-$L^2$ value error",
        # The historical metrics.csv name stores theta/w in current Liu runs.
        "RelL2_theta": r"Final relative-$L^2$ normalized-control error",
        "RelL2_vartheta": r"Final relative-$L^2$ normalized-control error",
    }
    written: List[str] = []
    for (group, metric, margin), rows in sorted(points.items()):
        by_target: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        valid_rows = [
            row for row in rows
            if math.isfinite(float(row["achieved_pres"]))
            and math.isfinite(float(row["error"]))
            and float(row["achieved_pres"]) > 0.0
            and float(row["error"]) > 0.0
        ]
        for row in valid_rows:
            by_target[float(row["pres_target"])].append(row)
        x_mean: List[float] = []
        y_mean: List[float] = []
        y_lo: List[float] = []
        y_hi: List[float] = []
        for target in sorted(by_target):
            cell = by_target[target]
            xs = np.asarray([float(row["achieved_pres"]) for row in cell], dtype=float)
            ys = np.asarray([float(row["error"]) for row in cell], dtype=float)
            xm = float(xs.mean())
            ym = float(ys.mean())
            sd = float(ys.std(ddof=1)) if ys.size > 1 else float("nan")
            x_mean.append(xm)
            y_mean.append(ym)
            if math.isfinite(sd):
                y_lo.append(max(ym - sd, np.finfo(float).tiny))
                y_hi.append(ym + sd)
            else:
                y_lo.append(ym)
                y_hi.append(ym)

        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        reached = [
            row for row in valid_rows if int(row.get("target_reached", 0)) == 1
        ]
        unreached = [
            row for row in valid_rows if int(row.get("target_reached", 0)) == 0
        ]
        if reached:
            ax.scatter(
                [float(row["achieved_pres"]) for row in reached],
                [float(row["error"]) for row in reached],
                s=22, color="#4C78A8", alpha=0.38, linewidths=0,
                marker="o", label="seed (target reached)", zorder=1,
            )
        if unreached:
            ax.scatter(
                [float(row["achieved_pres"]) for row in unreached],
                [float(row["error"]) for row in unreached],
                s=30, color="#E45756", alpha=0.72, linewidths=1.1,
                marker="x", label="seed (target not reached)", zorder=1,
            )
        order = np.argsort(np.asarray(x_mean, dtype=float))
        xx = np.asarray(x_mean, dtype=float)[order]
        yy = np.asarray(y_mean, dtype=float)[order]
        lo = np.asarray(y_lo, dtype=float)[order]
        hi = np.asarray(y_hi, dtype=float)[order]
        ax.plot(
            xx, yy, color="#1F4E79", marker="o", linewidth=1.7,
            markersize=5, label=r"target mean $\pm$ sample SD", zorder=3,
        )
        ax.fill_between(
            xx, lo, hi, color="#4C78A8", alpha=0.18, linewidth=0, zorder=2,
        )

        pooled_x = np.asarray(
            [float(row["achieved_pres"]) for row in valid_rows], dtype=float
        )
        pooled_y = np.asarray(
            [float(row["error"]) for row in valid_rows], dtype=float
        )
        fit = ols_loglog(pooled_x, pooled_y)
        if (
            pooled_x.size >= 2
            and not math.isclose(
                float(pooled_x.min()), float(pooled_x.max()),
                rel_tol=1e-12, abs_tol=0.0,
            )
            and math.isfinite(fit.get("slope", float("nan")))
            and math.isfinite(fit.get("intercept", float("nan")))
        ):
            fit_x = np.geomspace(
                float(pooled_x.min()), float(pooled_x.max()), 160
            )
            fit_y = 10.0 ** (
                float(fit["intercept"])
                + float(fit["slope"]) * np.log10(fit_x)
            )
            ax.plot(
                fit_x, fit_y, "--", color="#333333", linewidth=1.35,
                label=rf"pooled fit: slope={fit['slope']:.2f}", zorder=4,
            )

        # A unit-slope guide needs a vertical placement.  Center it on the
        # target means via the geometric mean of y_bar/x_bar; only its slope,
        # not its intercept, is diagnostic.
        if xx.size:
            ref_x_lo = float(xx.min())
            ref_x_hi = float(xx.max())
            if math.isclose(ref_x_lo, ref_x_hi, rel_tol=1e-12, abs_tol=0.0):
                ref_x_lo *= 0.8
                ref_x_hi *= 1.25
            reference_x = np.geomspace(ref_x_lo, ref_x_hi, 160)
            reference_anchor = float(np.exp(np.mean(np.log(yy / xx))))
            ax.plot(
                reference_x, reference_anchor * reference_x, ":",
                color="#777777", linewidth=1.35,
                label="slope-1 reference", zorder=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\mathrm{res}}$")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.grid(True, which="both", alpha=0.24, linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels, fontsize=max(6.0, 0.75 * args.font_size),
                loc="best",
            )
        fig.tight_layout()
        margin_token = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{margin:g}")
        metric_token = re.sub(r"[^0-9A-Za-z_.-]+", "_", metric)
        stem = f"e6_error_floor_{group}_{metric_token}_margin_{margin_token}"
        for extension in formats:
            path = output / f"{stem}.{extension}"
            save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
            if extension == "png":
                save_kwargs["dpi"] = args.dpi
            fig.savefig(path, format=extension, **save_kwargs)
            written.append(path.name)
        plt.close(fig)
    return written


def _write_e6_outputs(
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    stage: Path,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    formats: List[str],
) -> None:
    stage.mkdir(parents=True, exist_ok=True)

    pts_path = stage / "points.csv"
    pts_fields = [
        "group", "metric", "run_dir", "seed", "pres_target", "achieved_pres",
        "e6_role", "e6_reset_lr_each_outer", "residual_semantics",
        "target_reached",
        "total_inner_steps", "error", "eval_margin", "market_hash",
        *E6_WARM_START_STATUS_FIELDS,
    ]
    with pts_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pts_fields)
        writer.writeheader()
        for (group, metric, _margin), rows in sorted(points.items()):
            for row in sorted(rows, key=lambda item: (item["pres_target"], item["seed"])):
                writer.writerow({"group": group, "metric": metric, **row})

    per_path = stage / "per_target.csv"
    quantities = ("achieved_pres", "error", "total_inner_steps")
    per_fields = [
        "group", "metric", "eval_margin", "pres_target", "n_runs",
        "n_target_reached", "seeds", "e6_role", "residual_semantics",
        "e6_reset_lr_each_outer", "warm_start_protocol", "warmup_target",
    ] + [f"{name}_{field}" for name in quantities
         for field in ("mean", "std", "sem", "ci95_lo", "ci95_hi")]
    with per_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_fields)
        writer.writeheader()
        for (group, metric, margin), rows in sorted(points.items()):
            by_target: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_target[float(row["pres_target"])].append(row)
            for target in sorted(by_target):
                cell = by_target[target]
                output_row: Dict[str, Any] = {
                    "group": group, "metric": metric, "eval_margin": margin,
                    "pres_target": target, "n_runs": len(cell),
                    "n_target_reached": sum(int(row["target_reached"]) for row in cell),
                    "seeds": ";".join(str(seed) for seed in sorted(row["seed"] for row in cell)),
                    "e6_role": cell[0].get("e6_role", ""),
                    "e6_reset_lr_each_outer": cell[0].get(
                        "e6_reset_lr_each_outer", ""
                    ),
                    "residual_semantics": cell[0].get("residual_semantics", ""),
                    "warm_start_protocol": cell[0].get(
                        "e6_warm_start_protocol", ""
                    ),
                    "warmup_target": cell[0].get("e6_warmup_target", ""),
                }
                for name in quantities:
                    values = [float(row[name]) for row in cell if row[name] != ""]
                    stats = _summary_stats(values)
                    for field in ("mean", "std", "sem", "ci95_lo", "ci95_hi"):
                        output_row[f"{name}_{field}"] = fmt(stats[field])
                writer.writerow(output_row)

    fit_path = stage / "fit.csv"
    fit_fields = [
        "group", "metric", "eval_margin", "se_type", "n_points", "n_clusters",
        "slope", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "intercept", "r2",
        "e6_role", "e6_reset_lr_each_outer", "residual_semantics",
        "warm_start_protocol", "warmup_target",
    ]
    with fit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fit_fields)
        writer.writeheader()
        for (group, metric, margin), rows in sorted(points.items()):
            x = np.asarray([row["achieved_pres"] for row in rows], dtype=float)
            y = np.asarray([row["error"] for row in rows], dtype=float)
            seeds = np.asarray([str(row["seed"]) for row in rows])
            tolerances = np.asarray([row["pres_target"] for row in rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y, seeds, tolerances = x[ok], y[ok], seeds[ok], tolerances[ok]
            result = ols_loglog(x, y)
            clustered = cluster_robust_slope_se(x, y, seeds)
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "cluster_seed", "n_points": result["n"],
                "n_clusters": clustered["G"], "slope": fmt(result["slope"]),
                "slope_se": fmt(clustered["se"]), "ci95_lo": fmt(clustered["ci_lo"]),
                "ci95_hi": fmt(clustered["ci_hi"]),
                "t_crit": fmt(clustered["t_crit"]),
                "intercept": fmt(result["intercept"]), "r2": fmt(result["r2"], 4),
                "e6_role": rows[0].get("e6_role", ""),
                "e6_reset_lr_each_outer": rows[0].get(
                    "e6_reset_lr_each_outer", ""
                ),
                "residual_semantics": rows[0].get("residual_semantics", ""),
                "warm_start_protocol": rows[0].get(
                    "e6_warm_start_protocol", ""
                ),
                "warmup_target": rows[0].get("e6_warmup_target", ""),
            })
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "ols_iid", "n_points": result["n"], "n_clusters": "",
                "slope": fmt(result["slope"]), "slope_se": fmt(result["slope_se"]),
                "ci95_lo": fmt(result["ci95_lo"]), "ci95_hi": fmt(result["ci95_hi"]),
                "t_crit": fmt(result["t_crit"]), "intercept": fmt(result["intercept"]),
                "r2": fmt(result["r2"], 4),
                "e6_role": rows[0].get("e6_role", ""),
                "e6_reset_lr_each_outer": rows[0].get(
                    "e6_reset_lr_each_outer", ""
                ),
                "residual_semantics": rows[0].get("residual_semantics", ""),
                "warm_start_protocol": rows[0].get(
                    "e6_warm_start_protocol", ""
                ),
                "warmup_target": rows[0].get("e6_warmup_target", ""),
            })
            target_x: List[float] = []
            target_y: List[float] = []
            for target in sorted(set(tolerances.tolist())):
                mask = tolerances == target
                target_x.append(float(np.exp(np.mean(np.log(x[mask])))))
                target_y.append(float(np.exp(np.mean(np.log(y[mask])))))
            target_result = ols_loglog(np.asarray(target_x), np.asarray(target_y))
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "tolerance_mean", "n_points": target_result["n"],
                "n_clusters": "", "slope": fmt(target_result["slope"]),
                "slope_se": fmt(target_result["slope_se"]),
                "ci95_lo": fmt(target_result["ci95_lo"]),
                "ci95_hi": fmt(target_result["ci95_hi"]),
                "t_crit": fmt(target_result["t_crit"]),
                "intercept": fmt(target_result["intercept"]),
                "r2": fmt(target_result["r2"], 4),
                "e6_role": rows[0].get("e6_role", ""),
                "e6_reset_lr_each_outer": rows[0].get(
                    "e6_reset_lr_each_outer", ""
                ),
                "residual_semantics": rows[0].get("residual_semantics", ""),
                "warm_start_protocol": rows[0].get(
                    "e6_warm_start_protocol", ""
                ),
                "warmup_target": rows[0].get("e6_warmup_target", ""),
            })

    metadata["figure_files"] = _plot_e6_cells(points, stage, args, formats)
    (stage / "e6_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "_SUCCESS_E6").write_text("success\n", encoding="utf-8")


def _commit_e6_outputs(stage: Path, output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in _owned_e6_paths(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("managed E6 output appeared during staged generation")
    backup = Path(tempfile.mkdtemp(prefix=".e6_backup_", dir=str(output.parent)))
    moved_old: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    marker = stage / "_SUCCESS_E6"
    try:
        for path in existing:
            saved = backup / path.name
            os.replace(path, saved)
            moved_old.append((saved, path))
        ordered = [path for path in sorted(stage.iterdir()) if path != marker]
        ordered.append(marker)
        for path in ordered:
            destination = output / path.name
            os.replace(path, destination)
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


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Aggregate the E6 residual-tolerance sweep.")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--output", type=str, default=None, help="Default: <out-root>/e6_summary")
    ap.add_argument("--model-type", type=str, default="pipinn", help="Which model to aggregate (E6 is PI-PINN).")
    ap.add_argument(
        "--metrics",
        type=str,
        default="e_Xev,RelL2_V,RelL2_theta",
        help=(
            "Comma-separated errors to fit against achieved p_res. In current "
            "Liu metrics.csv, RelL2_theta is the normalized control theta/w."
        ),
    )
    ap.add_argument("--include-stopped", action="store_true")
    ap.add_argument("--expected-seeds", default="",
                    help="Exact seed set required at every tolerance, e.g. 1-10 or 1,3,7.")
    ap.add_argument(
        "--expected-tolerances",
        "--expected-targets",
        dest="expected_tolerances",
        default="",
        help=(
            "Exact positive pres_target set, e.g. 1,0.5,0.1,0.05,0.01. "
            "--expected-targets is a synonymous paper-facing spelling."
        ),
    )
    ap.add_argument("--min-runs-per-tolerance", type=int, default=1)
    ap.add_argument("--expected-n-assets", type=int, default=None)
    ap.add_argument("--expected-m-states", type=int, default=None)
    ap.add_argument(
        "--allow-legacy-residual-semantics",
        action="store_true",
        help=(
            "Diagnostic only: permit legacy outer_history val_pres when "
            "val_pres_post_restore is absent."
        ),
    )
    ap.add_argument(
        "--require-common-warm-start",
        action="store_true",
        help=(
            "Paper mode: require every selected run to be an auditable "
            "target_branch from one seed-level common warm-start bundle."
        ),
    )
    ap.add_argument(
        "--independent-standard-runs",
        "--standard-independent",
        dest="independent_standard_runs",
        action="store_true",
        help=(
            "Diagnostic mode: aggregate only e6_role=standard runs that were "
            "trained independently from the ordinary initialization at every "
            "p_res target. This is not the common-warm-start paper E6 design."
        ),
    )
    ap.add_argument(
        "--expected-e6-reset-lr-each-outer",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Require every selected target branch to use this outer-start LR "
            "protocol. Use 1 for the current paper E6 design."
        ),
    )
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--fig-width", type=float, default=6.4)
    ap.add_argument("--fig-height", type=float, default=4.2)
    ap.add_argument("--font-size", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.min_runs_per_tolerance < 1:
        raise ValueError("--min-runs-per-tolerance must be positive")
    if args.independent_standard_runs and args.require_common_warm_start:
        raise ValueError(
            "--independent-standard-runs and --require-common-warm-start "
            "select mutually exclusive protocols"
        )
    if (
        args.independent_standard_runs
        and args.expected_e6_reset_lr_each_outer is not None
    ):
        raise ValueError(
            "--expected-e6-reset-lr-each-outer is target-branch-only; "
            "independent standard runs are required to record reset mode 0"
        )
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    expected_tolerances = parse_float_spec(args.expected_tolerances)
    expected_tolerance_keys = {_float_key(value) for value in expected_tolerances}
    formats = _parse_formats(args.formats)
    if args.fig_width <= 0 or args.fig_height <= 0 or args.font_size <= 0 or args.dpi <= 0:
        raise ValueError("figure dimensions, font size, and dpi must be positive")

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "e6_summary")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics must contain at least one metric")
    accepted = {"success"} | ({"stopped_early"} if args.include_stopped else set())
    validation_errors: List[str] = []

    # Select the newest attempt before inspecting its status. This prevents a
    # failed rerun from reviving an older success for the same tolerance/seed.
    newest: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for run_dir in find_runs(out_root):
        cfg = load_config_args_raw(run_dir)
        if cfg is None or str(cfg.get("model_type", "")) != args.model_type:
            continue
        try:
            n_assets = int(cfg.get("n_assets", -1))
            m_states = int(cfg.get("m_states", -1))
        except (TypeError, ValueError):
            continue
        if args.expected_n_assets is not None and n_assets != args.expected_n_assets:
            continue
        if args.expected_m_states is not None and m_states != args.expected_m_states:
            continue
        e6_role = str(cfg.get("e6_role", "")).strip()
        # The warm-up artifact may also have target=1, but it is an input to
        # every target branch rather than a target cell of its own.
        if e6_role == E6_WARMUP_ROLE:
            continue
        if args.independent_standard_runs:
            if e6_role != E6_STANDARD_ROLE:
                validation_errors.append(
                    f"{run_dir}: --independent-standard-runs requires "
                    f"e6_role={E6_STANDARD_ROLE!r}; got {e6_role!r}"
                )
                continue
        elif e6_role not in ("", E6_TARGET_BRANCH_ROLE):
            validation_errors.append(
                f"{run_dir}: unsupported e6_role={e6_role!r}; use "
                "--independent-standard-runs for independent standard sweeps"
            )
            continue
        if args.require_common_warm_start and e6_role != E6_TARGET_BRANCH_ROLE:
            validation_errors.append(
                f"{run_dir}: --require-common-warm-start rejects unlabeled/"
                f"legacy E6 run (e6_role={e6_role!r})"
            )
            continue
        reset_present = "e6_reset_lr_each_outer" in cfg
        reset_raw = cfg.get("e6_reset_lr_each_outer", None)
        reset_valid = (
            isinstance(reset_raw, bool)
            or (
                isinstance(reset_raw, (int, float))
                and not isinstance(reset_raw, bool)
                and math.isfinite(float(reset_raw))
                and float(reset_raw) in {0.0, 1.0}
            )
        )
        if reset_present and not reset_valid:
            validation_errors.append(
                f"{run_dir}: e6_reset_lr_each_outer must be 0/1, "
                f"got {reset_raw!r}"
            )
            continue
        if (
            e6_role == E6_STANDARD_ROLE
            and (not reset_present or int(bool(reset_raw)) != 0)
        ):
            validation_errors.append(
                f"{run_dir}: independent standard runs require explicit "
                "e6_reset_lr_each_outer=0"
            )
            continue
        if (
            e6_role == E6_TARGET_BRANCH_ROLE
            and not reset_present
            and (
                args.require_common_warm_start
                or args.expected_e6_reset_lr_each_outer is not None
            )
        ):
            validation_errors.append(
                f"{run_dir}: current E6 protocol requires explicit "
                "e6_reset_lr_each_outer in config.json"
            )
            continue
        reset_mode: Any = int(bool(reset_raw)) if reset_present else ""
        if (
            args.expected_e6_reset_lr_each_outer is not None
            and reset_mode != int(args.expected_e6_reset_lr_each_outer)
        ):
            validation_errors.append(
                f"{run_dir}: e6_reset_lr_each_outer={reset_mode!r} != "
                f"expected={args.expected_e6_reset_lr_each_outer}"
            )
            continue
        target_raw = cfg.get("pres_target", None)
        try:
            target = float(target_raw) if target_raw not in (None, "") else None
            seed = int(cfg.get("seed"))
        except (TypeError, ValueError):
            continue
        if target is None or not math.isfinite(target) or target <= 0.0:
            continue
        group = e6_group_key(cfg)
        record = {
            "group": group,
            "config": cfg,
            "run_dir_abs": os.path.abspath(run_dir),
            "run_dir": os.path.relpath(run_dir, out_root),
            "seed": seed,
            "pres_target": target,
            "e6_role": e6_role,
            "e6_reset_lr_each_outer": reset_mode,
            "status": run_status(run_dir),
            "updated_at": run_updated_at(run_dir),
        }
        key = (group, args.model_type, _float_key(target), seed)
        previous = newest.get(key)
        if previous is None or (record["updated_at"], record["run_dir_abs"]) >= (
            previous["updated_at"], previous["run_dir_abs"]
        ):
            newest[key] = record

    if not newest:
        if validation_errors:
            raise ValueError(
                "E6 protocol validation failed:\n  - "
                + "\n  - ".join(validation_errors)
            )
        raise ValueError("no E6 runs with a positive pres_target match the requested filters")

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in newest.values():
        by_group[record["group"]].append(record)

    selected_records: List[Dict[str, Any]] = []
    for group, records in sorted(by_group.items()):
        observed_tolerance_keys = {
            _float_key(record["pres_target"])
            for record in records if record["status"] in accepted
        }
        if expected_tolerance_keys and observed_tolerance_keys != expected_tolerance_keys:
            validation_errors.append(
                f"group={group}: tolerances={sorted(observed_tolerance_keys)} != "
                f"expected={sorted(expected_tolerance_keys)}"
            )
        tolerance_keys = expected_tolerance_keys or observed_tolerance_keys
        group_selected: List[Dict[str, Any]] = []
        for tolerance_key in sorted(tolerance_keys, key=float):
            attempts = [
                record for record in records
                if _float_key(record["pres_target"]) == tolerance_key
            ]
            accepted_rows = [record for record in attempts if record["status"] in accepted]
            accepted_seed_set = {record["seed"] for record in accepted_rows}
            if expected_seeds and accepted_seed_set != expected_seeds:
                statuses = {record["seed"]: record["status"] for record in attempts}
                validation_errors.append(
                    f"group={group}, tolerance={tolerance_key}: accepted seeds="
                    f"{sorted(accepted_seed_set)} != expected={sorted(expected_seeds)}; "
                    f"newest statuses={statuses}"
                )
            if len(accepted_rows) < args.min_runs_per_tolerance:
                validation_errors.append(
                    f"group={group}, tolerance={tolerance_key}: {len(accepted_rows)} accepted "
                    f"runs < min-runs-per-tolerance={args.min_runs_per_tolerance}"
                )
            group_selected.extend(accepted_rows)

        market_hashes: set[str] = set()
        for record in group_selected:
            market_path = os.path.join(record["run_dir_abs"], "market_params.npz")
            try:
                record["market_hash"] = canonical_market_hash(market_path)
                market_hashes.add(record["market_hash"])
            except Exception as exc:
                validation_errors.append(f"{record['run_dir_abs']}: invalid market snapshot: {exc}")
        if len(market_hashes) != 1:
            validation_errors.append(
                f"group={group}: expected one canonical market snapshot, found "
                f"{sorted(market_hashes)}"
            )
        selected_records.extend(group_selected)

    if validation_errors:
        raise ValueError("E6 support validation failed:\n  - " + "\n  - ".join(validation_errors))

    # points[(group, metric, primary margin)] -> one row per accepted run.
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    panel_records: List[Dict[str, Any]] = []
    for record in selected_records:
        run_dir = record["run_dir_abs"]
        target = float(record["pres_target"])
        e6_role = str(record.get("e6_role", ""))
        status = read_status(run_dir)
        if str(status.get("status", "")) != record["status"]:
            raise ValueError(
                f"{run_dir}: terminal marker/status.json disagreement: "
                f"{record['status']!r} vs {status.get('status')!r}"
            )
        try:
            history = residual_history_summary(
                run_dir,
                allow_legacy=bool(args.allow_legacy_residual_semantics),
                e6_role=e6_role,
            )
        except ValueError as exc:
            validation_errors.append(str(exc))
            continue
        achieved = history["achieved"]
        residual_semantics = str(history["semantics"])
        provenance: Dict[str, Any] = {
            key: "" for key in E6_WARM_START_STATUS_FIELDS
        }
        if e6_role == E6_TARGET_BRANCH_ROLE:
            provenance, provenance_errors = target_branch_warm_start_provenance(
                run_dir, config=record["config"], status=status, history=history
            )
            validation_errors.extend(provenance_errors)
            if (
                args.require_common_warm_start
                or args.expected_e6_reset_lr_each_outer is not None
            ):
                history_modes = history.get(
                    "e6_reset_lr_each_outer", []
                )
                expected_mode = int(record["e6_reset_lr_each_outer"])
                if (
                    len(history_modes) != int(history["target_row_count"])
                    or any(mode != expected_mode for mode in history_modes)
                ):
                    validation_errors.append(
                        f"{run_dir}: every target outer row must record "
                        f"e6_reset_lr_each_outer={expected_mode}"
                    )
                if expected_mode == 1:
                    try:
                        expected_lr = float(
                            record["config"]["carry_lr_max"]
                        )
                    except (KeyError, TypeError, ValueError):
                        expected_lr = float("nan")
                    if not math.isfinite(expected_lr) or expected_lr <= 0.0:
                        validation_errors.append(
                            f"{run_dir}: reset mode requires positive finite "
                            "carry_lr_max in config"
                        )
                    else:
                        for field, values in (
                            ("outer_start_lr", history.get("outer_start_lrs", [])),
                            ("lr_carried_next", history.get("lr_carried_next", [])),
                        ):
                            if (
                                len(values) != int(history["target_row_count"])
                                or any(
                                    value is None
                                    or not math.isclose(
                                        float(value), expected_lr,
                                        rel_tol=1e-12, abs_tol=0.0,
                                    )
                                    for value in values
                                )
                            ):
                                validation_errors.append(
                                    f"{run_dir}: every target outer {field} "
                                    f"must equal carry_lr_max={expected_lr:.12g}"
                                )
                        status_lr = status.get(
                            "e6_target_outer_start_lr", None
                        )
                        if (
                            not isinstance(status_lr, (int, float))
                            or not math.isfinite(float(status_lr))
                            or not math.isclose(
                                float(status_lr), expected_lr,
                                rel_tol=1e-12, abs_tol=0.0,
                            )
                        ):
                            validation_errors.append(
                                f"{run_dir}: status e6_target_outer_start_lr "
                                f"must equal carry_lr_max={expected_lr:.12g}"
                            )
            status_semantics = str(status.get("pres_max_semantics", ""))
            if status_semantics != E6_TARGET_RESIDUAL_SEMANTICS:
                validation_errors.append(
                    f"{run_dir}: target_branch pres_max_semantics="
                    f"{status_semantics!r}, expected "
                    f"{E6_TARGET_RESIDUAL_SEMANTICS!r}"
                )
            status_achieved = status.get("pres_max")
            if not isinstance(status_achieved, (int, float)) or not math.isfinite(
                float(status_achieved)
                if isinstance(status_achieved, (int, float))
                else float("nan")
            ):
                validation_errors.append(
                    f"{run_dir}: target_branch status pres_max must be finite"
                )
            elif achieved is not None and not math.isclose(
                float(status_achieved),
                float(achieved),
                rel_tol=1e-10,
                abs_tol=0.0,
            ):
                validation_errors.append(
                    f"{run_dir}: status pres_max={float(status_achieved):.12g} "
                    f"does not match recomputed target-history max="
                    f"{float(achieved):.12g}"
                )
        elif e6_role == E6_STANDARD_ROLE:
            if str(status.get("e6_role", "")).strip() != E6_STANDARD_ROLE:
                validation_errors.append(
                    f"{run_dir}: status e6_role must equal "
                    f"{E6_STANDARD_ROLE!r}"
                )
            status_reset = status.get("e6_reset_lr_each_outer", None)
            status_reset_valid = (
                isinstance(status_reset, bool)
                or (
                    isinstance(status_reset, (int, float))
                    and not isinstance(status_reset, bool)
                    and math.isfinite(float(status_reset))
                    and float(status_reset) in {0.0, 1.0}
                )
            )
            if not status_reset_valid or int(bool(status_reset)) != 0:
                validation_errors.append(
                    f"{run_dir}: independent standard status must record "
                    "e6_reset_lr_each_outer=0"
                )
            history_modes = history.get("e6_reset_lr_each_outer", [])
            if (
                len(history_modes) != int(history["residual_row_count"])
                or any(mode != 0 for mode in history_modes)
            ):
                validation_errors.append(
                    f"{run_dir}: every standard outer row must record "
                    "e6_reset_lr_each_outer=0"
                )
            status_semantics = str(status.get("pres_max_semantics", ""))
            if status_semantics != E6_STANDARD_RESIDUAL_SEMANTICS:
                validation_errors.append(
                    f"{run_dir}: standard pres_max_semantics="
                    f"{status_semantics!r}, expected "
                    f"{E6_STANDARD_RESIDUAL_SEMANTICS!r}"
                )
            status_achieved = status.get("pres_max")
            if not isinstance(status_achieved, (int, float)) or not math.isfinite(
                float(status_achieved)
                if isinstance(status_achieved, (int, float))
                else float("nan")
            ):
                validation_errors.append(
                    f"{run_dir}: standard status pres_max must be finite"
                )
            elif achieved is not None and not math.isclose(
                float(status_achieved),
                float(achieved),
                rel_tol=1e-10,
                abs_tol=0.0,
            ):
                validation_errors.append(
                    f"{run_dir}: status pres_max={float(status_achieved):.12g} "
                    f"does not match recomputed standard-history max="
                    f"{float(achieved):.12g}"
                )
        if achieved is None or not math.isfinite(float(achieved)) or not (
            float(achieved) > 0
        ):
            validation_errors.append(
                f"{run_dir}: missing positive official post-restore p_res "
                f"(history semantics={residual_semantics!r})"
            )
            continue
        steps = status.get("total_inner_steps", None)
        if not isinstance(steps, (int, float)):
            steps = total_steps_from_outer_history(run_dir)
        reached = bool(achieved <= target * (1.0 + 1e-9))
        if e6_role == E6_TARGET_BRANCH_ROLE:
            reached_semantics = str(status.get("target_reached_semantics", ""))
            if reached_semantics != E6_TARGET_REACHED_SEMANTICS:
                validation_errors.append(
                    f"{run_dir}: target_branch target_reached_semantics="
                    f"{reached_semantics!r}, expected "
                    f"{E6_TARGET_REACHED_SEMANTICS!r}"
                )
            raw_reached = status.get("target_reached")
            valid_reached = (
                isinstance(raw_reached, bool)
                or (
                    isinstance(raw_reached, (int, float))
                    and not isinstance(raw_reached, bool)
                    and math.isfinite(float(raw_reached))
                    and float(raw_reached) in {0.0, 1.0}
                )
            )
            if not valid_reached:
                validation_errors.append(
                    f"{run_dir}: target_branch status target_reached must be "
                    f"boolean or numeric 0/1, got {raw_reached!r}"
                )
            elif bool(raw_reached) != reached:
                validation_errors.append(
                    f"{run_dir}: status target_reached={bool(raw_reached)} "
                    f"conflicts with achieved p_res={float(achieved):.6g} and "
                    f"target={target:.6g}"
                )
            if (
                record["e6_reset_lr_each_outer"] != ""
                or args.require_common_warm_start
                or args.expected_e6_reset_lr_each_outer is not None
            ):
                status_reset = status.get("e6_reset_lr_each_outer", None)
                status_reset_valid = (
                    isinstance(status_reset, bool)
                    or (
                        isinstance(status_reset, (int, float))
                        and not isinstance(status_reset, bool)
                        and math.isfinite(float(status_reset))
                        and float(status_reset) in {0.0, 1.0}
                    )
                )
                if not status_reset_valid:
                    validation_errors.append(
                        f"{run_dir}: target_branch status must record "
                        "e6_reset_lr_each_outer as 0/1"
                    )
                elif int(bool(status_reset)) != int(
                    record["e6_reset_lr_each_outer"]
                ):
                    validation_errors.append(
                        f"{run_dir}: config/status e6_reset_lr_each_outer "
                        "disagreement"
                    )
        elif e6_role == E6_STANDARD_ROLE:
            reached_semantics = str(status.get("target_reached_semantics", ""))
            if reached_semantics != E6_STANDARD_TARGET_REACHED_SEMANTICS:
                validation_errors.append(
                    f"{run_dir}: standard target_reached_semantics="
                    f"{reached_semantics!r}, expected "
                    f"{E6_STANDARD_TARGET_REACHED_SEMANTICS!r}"
                )
            raw_reached = status.get("target_reached")
            valid_reached = (
                isinstance(raw_reached, bool)
                or (
                    isinstance(raw_reached, (int, float))
                    and not isinstance(raw_reached, bool)
                    and math.isfinite(float(raw_reached))
                    and float(raw_reached) in {0.0, 1.0}
                )
            )
            if not valid_reached:
                validation_errors.append(
                    f"{run_dir}: standard status target_reached must be "
                    f"boolean or numeric 0/1, got {raw_reached!r}"
                )
            elif bool(raw_reached) != reached:
                validation_errors.append(
                    f"{run_dir}: standard status target_reached="
                    f"{bool(raw_reached)} conflicts with achieved "
                    f"p_res={float(achieved):.6g} and target={target:.6g}"
                )

        panel_records.append({
            "group": record["group"],
            "seed": record["seed"],
            "pres_target": target,
            "e6_role": e6_role,
            "e6_reset_lr_each_outer": record[
                "e6_reset_lr_each_outer"
            ],
            **provenance,
        })
        for metric in metrics:
            picked = pick_error_value(run_dir, metric)
            if picked is None:
                location = (
                    "final target-phase outer_history diagnostic"
                    if metric == "e_Xev"
                    else "required fulldim metric"
                )
                validation_errors.append(
                    f"{run_dir}: {location} {metric!r} is missing"
                )
                continue
            val, margin_used = picked
            if not math.isfinite(float(val)) or float(val) <= 0.0:
                validation_errors.append(
                    f"{run_dir}: required metric {metric!r} must be positive "
                    f"and finite"
                )
                continue
            points[(record["group"], metric, margin_used)].append({
                "run_dir": record["run_dir"],
                "seed": record["seed"],
                "pres_target": target,
                "achieved_pres": float(achieved),
                "e6_role": e6_role,
                "e6_reset_lr_each_outer": record[
                    "e6_reset_lr_each_outer"
                ],
                "residual_semantics": residual_semantics,
                "target_reached": int(reached),
                "total_inner_steps": steps if steps is not None else "",
                "error": float(val),
                "eval_margin": margin_used,
                "market_hash": record["market_hash"],
                **provenance,
            })

    # A training seed may have its own warm-up, but every residual target of
    # that seed must branch from exactly the same model+Adam+RNG state.
    common_by_group_seed: Dict[Tuple[str, int], List[Dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in panel_records:
        if row.get("e6_role") == E6_TARGET_BRANCH_ROLE:
            common_by_group_seed[(str(row["group"]), int(row["seed"]))].append(row)
    for (group, seed), rows in sorted(common_by_group_seed.items()):
        for field in E6_COMMON_WARM_START_FIELDS:
            values = {
                json.dumps(row.get(field), sort_keys=True, default=str)
                for row in rows
            }
            if len(values) != 1:
                by_target = {
                    f"{float(row['pres_target']):g}": row.get(field)
                    for row in rows
                }
                validation_errors.append(
                    f"group={group} seed={seed}: common warm-start field "
                    f"{field} differs across targets: {by_target}"
                )

    if validation_errors:
        raise ValueError(
            "E6 protocol validation failed:\n  - "
            + "\n  - ".join(validation_errors)
        )

    if not points:
        raise ValueError("no complete E6 metric rows were selected")

    # ``eval_margin`` is deliberately excluded from the training group hash,
    # but a paper-facing residual sweep cannot pool runs whose official metric
    # rows use different primary windows.  Validate support again after the
    # metric/margin split so a nominally balanced seed cell cannot fragment
    # into several one-seed summaries.
    support_errors: List[str] = []
    margins_by_group: Dict[str, set[float]] = defaultdict(set)
    for (group, _metric, margin) in points:
        margins_by_group[group].add(float(margin))
    for group, margins in sorted(margins_by_group.items()):
        if len(margins) != 1:
            support_errors.append(
                f"group={group}: mixed primary eval margins {sorted(margins)}"
            )
    for (group, metric, margin), rows in sorted(points.items()):
        by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_target[_float_key(float(row["pres_target"]))].append(row)
        for tolerance_key, cell in sorted(by_target.items(), key=lambda item: float(item[0])):
            seeds = {int(row["seed"]) for row in cell}
            if expected_seeds and seeds != expected_seeds:
                support_errors.append(
                    f"group={group}, metric={metric}, margin={margin}, "
                    f"tolerance={tolerance_key}: seeds={sorted(seeds)} != "
                    f"expected={sorted(expected_seeds)}"
                )
            if len(cell) < args.min_runs_per_tolerance:
                support_errors.append(
                    f"group={group}, metric={metric}, margin={margin}, "
                    f"tolerance={tolerance_key}: {len(cell)} metric rows < "
                    f"min-runs-per-tolerance={args.min_runs_per_tolerance}"
                )
    if support_errors:
        raise ValueError(
            "E6 metric-window support validation failed:\n  - "
            + "\n  - ".join(support_errors)
        )

    output = Path(summary_dir).resolve()
    _prepare_e6_output(output, args.overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregation_protocol = (
        "standard-independent"
        if args.independent_standard_runs
        else (
            "common-warm-target-branch"
            if args.require_common_warm_start
            else "target-branch-or-legacy"
        )
    )
    residual_definition = (
        "standard-independent: max over all standard outer rows of fixed-Q_res "
        "p_res measured on the official post-restore checkpoint; outer 1 is "
        "included and every target is trained from its own ordinary initialization"
        if args.independent_standard_runs
        else (
            "target_branch: max over target-phase outer rows of fixed-Q_res "
            "p_res measured on the official post-restore checkpoint, excluding "
            "the separate common warm-up"
        )
    )
    metadata: Dict[str, Any] = {
        "schema_version": 7,
        "out_root": out_root,
        "model_type": args.model_type,
        "aggregation_protocol": aggregation_protocol,
        "independent_standard_runs": bool(args.independent_standard_runs),
        "metrics": metrics,
        "expected_seeds": sorted(expected_seeds),
        "expected_tolerances": expected_tolerances,
        "min_runs_per_tolerance": args.min_runs_per_tolerance,
        "expected_n_assets": args.expected_n_assets,
        "expected_m_states": args.expected_m_states,
        "expected_e6_reset_lr_each_outer":
            args.expected_e6_reset_lr_each_outer,
        "n_groups": len({key[0] for key in points}),
        "n_metric_cells": len(points),
        "n_point_rows": sum(len(rows) for rows in points.values()),
        "market_hashes": sorted({row["market_hash"] for rows in points.values() for row in rows}),
        "uncertainty": "sample SD and Student-t 95% CI across training seeds within tolerance",
        "fit_x": "achieved p_res (never nominal target)",
        "figure_semantics": {
            "x_label": "p_res",
            "x_values": "official achieved post-restore p_res",
            "target_curve": "target-wise arithmetic mean with sample-SD band in error",
            "fitted_line": (
                "pooled log-log OLS fit across seed-level points; dashed line "
                "and slope shown in the legend"
            ),
            "slope_one_reference": (
                "dotted unit-slope guide anchored at the geometric mean of "
                "target-level error_mean/achieved_pres_mean"
            ),
        },
        "residual_definition": residual_definition,
        "target_branch_residual_semantics": E6_TARGET_RESIDUAL_SEMANTICS,
        "target_branch_target_reached_semantics":
            E6_TARGET_REACHED_SEMANTICS,
        "standard_independent_residual_semantics":
            E6_STANDARD_RESIDUAL_SEMANTICS,
        "standard_independent_target_reached_semantics":
            E6_STANDARD_TARGET_REACHED_SEMANTICS,
        "common_warm_start_required": bool(args.require_common_warm_start),
        "e6_reset_lr_each_outer_by_group": {
            group: sorted(
                {
                    int(row["e6_reset_lr_each_outer"])
                    for (point_group, _metric, _margin), rows in points.items()
                    if point_group == group
                    for row in rows
                    if row["e6_reset_lr_each_outer"] != ""
                }
            )
            for group in sorted({key[0] for key in points})
        },
        "outer_lr_protocol": (
            "independent standard runs use the ordinary configured PI-PINN "
            "schedule and require e6_reset_lr_each_outer=0"
            if args.independent_standard_runs
            else (
                "e6_reset_lr_each_outer is target-branch-only, excluded from "
                "warm-start compatibility, retained in aggregate group identity; "
                "mode 1 preserves model/Adam moments and recreates the scheduler "
                "after resetting every target outer-start LR to carry_lr_max"
            )
        ),
        "legacy_residual_semantics_allowed": bool(
            args.allow_legacy_residual_semantics
        ),
        "warm_start_contract": (
            "not applicable: each tolerance is an independent standard run "
            "from the ordinary initialization; results must not be described "
            "as common-warm-start E6"
            if args.independent_standard_runs
            else (
                "warmup-role artifacts are excluded before deduplication; all "
                "target branches of one (group,seed) share protocol/source/id, "
                "model, optimizer, RNG and bundle hashes plus warm-up outcome; "
                "the loaded bundle hash equals the source bundle hash"
            )
        ),
        "metric_semantics": {
            "e_Xev": (
                "final standard-run training diagnostic on the raw training "
                "primary evaluation margin"
                if args.independent_standard_runs
                else (
                    "final target-phase training diagnostic on the raw training "
                    "primary evaluation margin"
                )
            ),
            "RelL2_V": "full-dimensional relative-L2 value error",
            "RelL2_theta": (
                "legacy metrics.csv name for the full-dimensional "
                "wealth-normalized control theta/w relative-L2 error"
            ),
        },
    }
    stage = Path(tempfile.mkdtemp(prefix=".e6_stage_", dir=str(output.parent)))
    try:
        _write_e6_outputs(points, stage, args, metadata, formats)
        _commit_e6_outputs(stage, output, args.overwrite)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(f"[e6] wrote validated artifacts to: {output}")
    return metadata


if __name__ == "__main__":
    main()
