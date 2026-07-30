#!/usr/bin/env python3
"""Compute the Merton exact PI-map ratio with an independent FD evaluator.

For every saved PI-PINN iterate ``v_n`` this program computes the greedy
feedback ``alpha_(n+1)=G(v_n)``, freezes that feedback, solves its *linear*
policy-evaluation PDE by finite differences, and reports

    ||E(G(v_n)) - V*||_Xev / ||v_n - V*||_Xev.

It never substitutes the next neural iterate in the numerator.  Exact-map
outputs are kept separate from the empirical ``e_(n+1)/e_n`` files because
the two diagnostics have different meanings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from merton_exact_map_core import (
    FDGrid,
    MertonProblem,
    constant_proportional_closed_form,
    constant_proportional_policy,
    crra_closed_form,
    evaluate_fd_bundle,
    solve_frozen_policy,
    log_to_wealth_derivatives,
    x_norm_components,
)


CHECKPOINT_RE = re.compile(r"value_net_iter(\d+)\.pt$")
POLICY_EXTENSIONS = ("boundary-projection", "neural-extrapolation")
SUPPORTED_PLOT_FORMATS = ("png", "pdf", "svg", "eps")
E6_WARMUP_BUNDLE_SCHEMA_VERSION = 1
E6_WARMUP_BUNDLE_KIND = "merton-pipinn-e6-common-warmup-v1"
E6_WARM_START_PROTOCOL = "merton-e6-common-warm-start-v1"
EXACT_PROTOCOL_COMPATIBILITY_VERSION = (
    "merton-exact-map-fd-seed-neutral-v1"
)
REFINEMENT_SEMANTICS_VERSION = (
    "merton-fd-grid-domain-gate-boundary-sensitivity-v2"
)
BOUNDARY_SEMANTICS = {
    "robin": (
        "primary homogeneous CRRA Robin closure u_y=(1-gamma)u; "
        "does not inject the optimal value amplitude"
    ),
    "exact-dirichlet": (
        "optimal-reference Dirichlet sensitivity audit; compatibility name "
        "that injects closed-form V* and is not an exact boundary oracle for "
        "a nonoptimal frozen policy"
    ),
}
RATIO_FIELDS = [
    "problem", "group", "protocol_hash", "model_type", "e6_role",
    "n_assets", "seed", "market_seed",
    "horizon", "gamma", "discount", "bequest", "risk_free", "network_dtype",
    "checkpoint_outer_iter", "source_iter", "target_policy_iter", "checkpoint",
    "checkpoint_sha256", "checkpoint_state_sha256",
    "market_sha256", "eval_margin", "eval_window_mode",
    "eval_w_min_requested", "eval_w_min_symmetric",
    "ev_tau_min", "ev_tau_max", "ev_y_min", "ev_y_max",
    "ev_w_min", "ev_w_max",
    "fd_margin", "fd_y_min", "fd_y_max", "boundary", "boundary_semantics",
    "boundary_role", "is_boundary_representative",
    "drift_scheme",
    "grid_factor", "ny", "nt", "dy", "dt", "is_primary", "is_verification",
    "analysis_mode", "policy_extension", "map_definition",
    "outside_collocation_count_fd", "outside_collocation_fraction_fd",
    "e_input_value", "e_input_vw", "e_input_vww", "e_input_vy", "e_input_vyy",
    "e_input_deriv", "e_input_X", "e_map_value", "e_map_vw", "e_map_vww",
    "e_map_vy", "e_map_vyy", "e_map_deriv", "e_map_X",
    "rho_exact", "denominator_defined", "map_variant", "local_map_unmodified_on_xfd",
    "whole_space_map_claim", "checkpoint_selection",
    "vw_guard_frac_fd", "pi_numerator_guard_frac_fd", "denom_guard_frac_fd",
    "positive_curvature_frac_fd",
    "kappa_clip_frac_fd", "consumption_clip_frac_fd", "portfolio_any_clip_frac_fd",
    "portfolio_component_clip_frac_fd", "vw_guard_frac_ev",
    "pi_numerator_guard_frac_ev", "denom_guard_frac_ev",
    "positive_curvature_frac_ev", "kappa_clip_frac_ev", "consumption_clip_frac_ev",
    "portfolio_any_clip_frac_ev", "portfolio_component_clip_frac_ev",
    "min_diffusion", "max_diffusion", "min_diffusion_variance",
    "max_diffusion_variance", "min_diffusion_variance_ev",
    "max_diffusion_variance_ev", "max_peclet", "upwind_fraction",
    "max_linear_residual", "linear_residual_tolerance",
    "boundary_elimination_size", "boundary_elimination_rank",
    "boundary_elimination_cond_inf", "min_linear_system_lu_pivot_ratio",
    "policy_hash", "grid_abs_change", "grid_rel_change",
    "domain_abs_change", "domain_rel_change", "boundary_abs_change",
    "boundary_rel_change", "boundary_sensitivity_status",
    "rho_grid_domain_sensitivity_envelope", "rho_sensitivity_envelope",
    "grid_domain_refinement_status", "refinement_semantics_version",
    "refinement_status", "contraction_status",
]

# E4 evaluates the neural policy-evaluation error at iteration ``n`` by
# comparing the *next* neural checkpoint with the FD value generated from the
# current checkpoint's greedy policy.  Keeping this in a separate table avoids
# conflating the exact-map contraction ratio with the approximation hypothesis
# ``delta_n = v_tilde_n - v^{alpha_n}``.
DEFECT_FIELDS = [
    "problem", "group", "protocol_hash", "model_type", "e6_role",
    "n_assets", "seed",
    "market_seed", "eval_margin", "eval_window_mode",
    "eval_w_min_requested", "eval_w_min_symmetric",
    "ev_tau_min", "ev_tau_max", "ev_y_min",
    "ev_y_max", "ev_w_min", "ev_w_max", "defect_iter", "defect_kind",
    "checkpoint_outer_iter", "source_iter",
    "target_policy_iter", "next_checkpoint_outer_iter", "next_neural_iter",
    "checkpoint_state_sha256", "next_checkpoint_state_sha256",
    "frozen_policy_sha256",
    "fd_margin", "boundary", "boundary_role",
    "grid_factor", "ny", "nt", "is_verification",
    "analysis_mode", "policy_extension", "map_definition",
    "outside_collocation_count_fd", "outside_collocation_fraction_fd",
    "delta_value_sup", "delta_vw_sup", "delta_vww_sup", "delta_vy_sup",
    "delta_vyy_sup", "delta_bundle_sup", "delta_X",
    "defect_grid_abs_change", "defect_grid_rel_change",
    "defect_domain_abs_change", "defect_domain_rel_change",
    "defect_boundary_abs_change", "defect_boundary_rel_change",
    "boundary_sensitivity_status",
    "defect_grid_domain_sensitivity_envelope", "defect_sensitivity_envelope",
    "p_res_post_restore", "p_res_source", "residual_semantics",
    "evaluated_bundle_path", "evaluated_bundle_sha256",
    "grid_domain_refinement_status", "refinement_semantics_version",
    "refinement_status", "map_variant", "local_map_unmodified_on_xfd",
    "whole_space_map_claim",
]

DEFECT_REFINEMENT_FIELDS = [
    "problem", "group", "protocol_hash", "model_type", "e6_role",
    "n_assets", "seed",
    "market_seed", "eval_margin", "eval_window_mode",
    "eval_w_min_requested", "eval_w_min_symmetric",
    "ev_tau_min", "ev_tau_max", "ev_y_min",
    "ev_y_max", "ev_w_min", "ev_w_max", "defect_iter", "defect_kind",
    "target_policy_iter",
    "next_checkpoint_outer_iter", "frozen_policy_sha256",
    "fd_margin", "boundary", "boundary_role", "is_boundary_representative",
    "grid_factor", "ny", "nt", "is_primary",
    "is_verification", "analysis_mode", "policy_extension", "map_definition",
    "outside_collocation_count_fd", "outside_collocation_fraction_fd",
    "delta_value_sup", "delta_vw_sup", "delta_vww_sup",
    "delta_vy_sup", "delta_vyy_sup", "delta_bundle_sup", "delta_X",
    "defect_grid_abs_change", "defect_grid_rel_change",
    "defect_domain_abs_change", "defect_domain_rel_change",
    "defect_boundary_abs_change", "defect_boundary_rel_change",
    "boundary_sensitivity_status",
    "defect_grid_domain_sensitivity_envelope", "defect_sensitivity_envelope",
    "grid_domain_refinement_status", "refinement_semantics_version",
    "refinement_status", "map_variant", "local_map_unmodified_on_xfd",
    "whole_space_map_claim",
]

ACTIVATION_FIELDS = [
    "vw_guard_frac_fd", "pi_numerator_guard_frac_fd", "denom_guard_frac_fd",
    "positive_curvature_frac_fd",
    "kappa_clip_frac_fd", "consumption_clip_frac_fd", "portfolio_any_clip_frac_fd",
    "portfolio_component_clip_frac_fd", "vw_guard_frac_ev",
    "pi_numerator_guard_frac_ev", "denom_guard_frac_ev",
    "positive_curvature_frac_ev", "kappa_clip_frac_ev", "consumption_clip_frac_ev",
    "portfolio_any_clip_frac_ev", "portfolio_component_clip_frac_ev",
]

ELLIPTICITY_FIELDS = [
    "min_diffusion_variance", "max_diffusion_variance",
    "min_diffusion_variance_ev", "max_diffusion_variance_ev",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash numerical content rather than NPZ zip metadata/timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_indexing_contract(
    value: Any,
) -> Dict[str, Any]:
    """Return only the seed-neutral checkpoint-indexing semantics.

    ``exact_map_config.json`` also records per-checkpoint lookup maps for
    auditability.  Those maps are derived from the saved schedule rather than
    from the numerical FD protocol, so the compatibility fingerprint keeps
    only the indexing contract used by the exact-map solver.
    """
    payload = value if isinstance(value, Mapping) else {}
    keys = (
        "checkpoint_outer_index_base",
        "source_iter_offset_from_checkpoint_outer",
        "target_policy_iter_offset_from_checkpoint_outer",
        "algorithm_outer_index_offset",
        "warmup_excluded_from_outer_history",
        "source",
    )
    return {
        key: payload.get(key)
        for key in keys
        if key in payload
    }


def exact_protocol_compatibility_payload(
    exact_cfg: Mapping[str, Any],
    *,
    training_group: str,
) -> Dict[str, Any]:
    """Build a seed-neutral FD/evaluation protocol payload.

    The warm-start bundle and model-state hashes remain in
    ``warm_start_predecessor`` and are still checked when a run is loaded.
    They identify a seed-specific input artifact, however, not a numerical FD
    protocol.  Consequently they must not split otherwise identical
    cross-seed exact-map aggregations.

    New outputs store the producer payload directly.  For outputs created
    before this contract existed, the same payload is reconstructed from
    ``exact_map_config.json`` so aggregation does not require rerunning the FD
    solves.
    """
    stored = exact_cfg.get("protocol_compatibility_payload")
    if isinstance(stored, Mapping):
        payload = dict(stored)
        payload["training_group"] = str(training_group)
        # Refuse to propagate the two legacy seed-specific fields even if a
        # hand-edited config inserted them into the stored payload.
        payload.pop("warm_start_bundle_file_hash", None)
        payload.pop("warm_start_model_state_hash", None)
        payload["compatibility_version"] = (
            EXACT_PROTOCOL_COMPATIBILITY_VERSION
        )
        return payload

    grid_value = exact_cfg.get("grid")
    grid = grid_value if isinstance(grid_value, Mapping) else {}
    bounds_value = exact_cfg.get("evaluation_bounds")
    bounds = bounds_value if isinstance(bounds_value, Mapping) else {}
    problem_value = exact_cfg.get("problem")
    problem = problem_value if isinstance(problem_value, Mapping) else {}
    implementation_value = exact_cfg.get("implementation_hashes")
    implementation_hashes = (
        dict(implementation_value)
        if isinstance(implementation_value, Mapping)
        else implementation_value
    )
    analysis_mode = str(exact_cfg.get("analysis_mode", ""))
    skip_e4 = analysis_mode in {
        "exact_map_only_pilot",
        "sparse_exact_map_only_pilot",
    }
    eval_w_min = bounds.get(
        "w_min", grid.get("evaluation_w_min")
    )
    eval_w_max = bounds.get(
        "w_max", grid.get("evaluation_w_max")
    )
    eval_y_min = bounds.get(
        "y_min", grid.get("evaluation_y_min")
    )
    eval_y_max = bounds.get(
        "y_max", grid.get("evaluation_y_max")
    )
    eval_tau_max = grid.get(
        "evaluation_tau_max", problem.get("horizon")
    )
    return {
        "compatibility_version": EXACT_PROTOCOL_COMPATIBILITY_VERSION,
        "training_group": str(training_group),
        "base_ny": grid.get("base_ny"),
        "base_nt": grid.get("base_nt"),
        "eval_ny": grid.get("eval_ny"),
        "evaluation_calendar_time_domain": grid.get(
            "evaluation_time_calendar_domain", "[0,T)"
        ),
        "primary_evaluation_window": {
            "eval_margin": exact_cfg.get("eval_margin"),
            "eval_window_mode": exact_cfg.get(
                "eval_window_mode", "symmetric-margin"
            ),
            "eval_w_min_requested": exact_cfg.get(
                "eval_w_min_requested"
            ),
            "eval_w_min_symmetric": exact_cfg.get(
                "eval_w_min_symmetric", eval_w_min
            ),
            "ev_tau_min": grid.get("evaluation_tau_min"),
            "ev_tau_max": eval_tau_max,
            "ev_y_min": eval_y_min,
            "ev_y_max": eval_y_max,
            "ev_w_min": eval_w_min,
            "ev_w_max": eval_w_max,
        },
        "grid_factors": list(grid.get("grid_factors", [])),
        "fd_margins": list(grid.get("fd_margins", [])),
        "boundaries": list(grid.get("boundaries", [])),
        "verify_checkpoints": grid.get("verify_checkpoints"),
        "analysis_mode": analysis_mode,
        "skip_e4": skip_e4,
        "e6_role": exact_cfg.get("e6_role", "standard"),
        "checkpoint_indexing": _checkpoint_indexing_contract(
            exact_cfg.get("checkpoint_indexing")
        ),
        "initial_defect_mode": exact_cfg.get("initial_defect_mode"),
        "policy_extension": exact_cfg.get("policy_extension"),
        "defect_refinement_selector": (
            "first_defect_plus_verify_checkpoints; paper evidence requires "
            "first+last+worst defects"
        ),
        "drift_scheme": grid.get("drift_scheme"),
        "peclet_limit": grid.get("peclet_limit"),
        "theta_method": grid.get("theta_method"),
        "rannacher_steps": grid.get(
            "initial_full_dt_backward_euler_steps"
        ),
        "denominator_tolerance": exact_cfg.get(
            "denominator_tolerance"
        ),
        "linear_residual_tolerance": exact_cfg.get(
            "linear_residual_tolerance",
            grid.get("linear_residual_tolerance"),
        ),
        "boundary_condition_limit": exact_cfg.get(
            "boundary_condition_limit",
            grid.get("boundary_condition_limit"),
        ),
        "solver_ellipticity_tolerance": exact_cfg.get(
            "solver_ellipticity_tolerance",
            grid.get("solver_ellipticity_tolerance"),
        ),
        "refinement_abs_tolerance": exact_cfg.get(
            "refinement_abs_tolerance"
        ),
        "refinement_rel_tolerance": exact_cfg.get(
            "refinement_rel_tolerance"
        ),
        "implementation_hashes": implementation_hashes,
        "checkpoint_selection": exact_cfg.get("checkpoint_selection"),
    }


def exact_protocol_compatibility_hash(
    exact_cfg: Mapping[str, Any],
    *,
    training_group: str,
) -> str:
    return stable_hash(
        exact_protocol_compatibility_payload(
            exact_cfg, training_group=training_group
        )
    )[:16]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically persist an evaluated numerical bundle without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    required = set(fields)
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise KeyError(
                f"{path.name} row {index} is missing schema fields: {missing}"
            )
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        # Silently dropping an unexpected metric key previously made it
        # possible to publish blank component columns.  Artifact schemas are
        # contracts: a producer-side spelling error must fail immediately.
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


METRIC_COMPONENT_MAP = {
    "value_sup": "value",
    "vw_sup": "vw",
    "vww_sup": "vww",
    "vy_sup": "vy",
    "vyy_sup": "vyy",
    "derivative_sup": "deriv",
    "x_norm": "X",
}


def metric_to_columns(prefix: str, metric: Mapping[str, Any]) -> Dict[str, float]:
    """Map a complete X-norm diagnostic to an explicit CSV component schema."""
    missing = [key for key in METRIC_COMPONENT_MAP if key not in metric]
    if missing:
        raise KeyError(
            f"{prefix} metric is missing required components: {', '.join(missing)}"
        )
    out: Dict[str, float] = {}
    for source, suffix in METRIC_COMPONENT_MAP.items():
        value = float(metric[source])
        if not math.isfinite(value):
            raise ValueError(f"{prefix}_{suffix} must be finite, got {value!r}")
        out[f"{prefix}_{suffix}"] = value
    return out


def bundle_storage_arrays(
    prefix: str,
    bundle: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return unambiguous log- and wealth-coordinate bundle arrays for NPZ."""
    value, value_y, value_yy = (
        np.asarray(bundle[0]),
        np.asarray(bundle[1]),
        np.asarray(bundle[2]),
    )
    value_w, value_ww = log_to_wealth_derivatives(value_y, value_yy, y)
    return {
        f"{prefix}_value": value,
        f"{prefix}_vy": value_y,
        f"{prefix}_vyy": value_yy,
        f"{prefix}_vw": value_w,
        f"{prefix}_vww": value_ww,
    }


def _quarantine_previous_outputs(output: Path) -> None:
    """Move old canonical artifacts aside before a new attempt.

    A failed rerun must never leave a stale CSV/NPZ under its canonical name.
    Moving, rather than deleting, preserves prior evidence for manual audit.
    """
    names = (
        "exact_map_refinement.csv",
        "exact_map_ratios.csv",
        "exact_map_defects.csv",
        "exact_map_defect_refinement.csv",
        "exact_map_config.json",
        "exact_map_status.json",
        "evaluated_bundles",
        "_SUCCESS_EXACT_MAP",
        "_FAILED_EXACT_MAP",
        "_STALE_EXACT_MAP",
        "_INCOMPLETE_EXACT_MAP",
    )
    existing = [output / name for name in names if (output / name).exists()]
    if not existing:
        return
    stale = output / "stale_outputs" / str(time.time_ns())
    stale.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, stale / path.name)


def _policy_extension_y(
    problem: MertonProblem,
    y: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized not in POLICY_EXTENSIONS:
        raise ValueError(
            f"unsupported policy extension {mode!r}; choose from {POLICY_EXTENSIONS}"
        )
    actual = np.asarray(y, dtype=np.float64)
    outside = (actual < problem.y_min) | (actual > problem.y_max)
    if normalized == "boundary-projection":
        evaluated = np.clip(actual, problem.y_min, problem.y_max)
    else:
        evaluated = actual
    return evaluated, outside


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


def load_outer_residuals(run_dir: Path) -> Tuple[Dict[int, float], str]:
    """Load the official per-outer residual attached to neural checkpoints.

    New trainers record ``val_pres_post_restore`` for the official restored
    model.  A legacy ``val_pres`` fallback is retained only so an old exact-map
    run can still be diagnosed; E4 paper aggregation rejects that legacy
    semantics instead of silently mixing pre- and post-restore states.
    """
    path = run_dir / "outer_history.csv"
    if not path.is_file():
        return {}, "missing"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "val_pres_post_restore" in fields:
            field = "val_pres_post_restore"
            semantics = "official_post_restore"
        elif "val_pres" in fields:
            field = "val_pres"
            semantics = "legacy_val_pres"
        else:
            return {}, "missing"
        values: Dict[int, float] = {}
        for row in reader:
            try:
                outer = int(float(str(row.get("outer_iter", ""))))
                value = float(str(row.get(field, "")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                values[outer] = value
    return values, semantics


def parse_float_list(text: str) -> List[float]:
    values = [float(item) for item in re.split(r"[\s,]+", str(text).strip()) if item]
    if not values:
        raise ValueError("expected a nonempty numeric list")
    return values


def parse_plot_formats(text: str) -> List[str]:
    formats = [
        item.lower().lstrip(".")
        for item in re.split(r"[\s,]+", str(text).strip())
        if item
    ]
    if not formats:
        raise ValueError("--formats must contain at least one format")
    if len(set(formats)) != len(formats):
        raise ValueError(f"duplicate formats in --formats={text!r}")
    invalid = sorted(set(formats) - set(SUPPORTED_PLOT_FORMATS))
    if invalid:
        raise ValueError(
            f"unsupported figure formats {invalid}; choose from "
            f"{list(SUPPORTED_PLOT_FORMATS)}"
        )
    return formats


def resolve_plot_formats(formats: str, legacy_format: str) -> List[str]:
    """Resolve the multi-format CLI while preserving the old --format flag."""
    if str(formats).strip() and str(legacy_format).strip():
        raise ValueError("--formats and legacy --format cannot be used together")
    if str(formats).strip():
        return parse_plot_formats(formats)
    if str(legacy_format).strip():
        return parse_plot_formats(legacy_format)
    return ["png"]


def eval_w_min_output_name(value: float) -> str:
    """Return a stable, human-readable directory name for a wealth cutoff."""
    text = repr(float(value)).lower()
    slug = (
        text.replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
    return f"eval_w_min_{slug}"


def parse_int_list(text: str) -> List[int]:
    values = [int(item) for item in re.split(r"[\s,]+", str(text).strip()) if item]
    if not values:
        raise ValueError("expected a nonempty integer list")
    return values


def parse_seed_spec(text: str) -> List[int]:
    out: set[int] = set()
    for token in re.split(r"[\s,]+", str(text).strip()):
        if not token:
            continue
        if "-" in token:
            lo_text, hi_text = token.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"invalid seed range: {token}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return sorted(out)


def first_margin(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = parse_float_list(str(value if value is not None else "0.10"))
    margin = float(values[0])
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"eval margin must lie in [0,1), got {margin}")
    return margin


def shrink_bounds(lo: float, hi: float, margin: float) -> Tuple[float, float]:
    if not math.isfinite(float(margin)) or not float(margin) < 1.0:
        raise ValueError("margin must be finite and smaller than one")
    center = 0.5 * (float(lo) + float(hi))
    half = 0.5 * (float(hi) - float(lo)) * (1.0 - float(margin))
    return center - half, center + half


def _merged_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(raw)
    for key in ("args", "config", "resolved"):
        nested = raw.get(key)
        if isinstance(nested, Mapping):
            merged.update(nested)
    return merged


def _pick(cfg: Mapping[str, Any], names: Sequence[str], *, default: Any = None, required: bool = False) -> Any:
    for name in names:
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
    if required:
        raise KeyError(f"missing required config field; tried {', '.join(names)}")
    return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    return float(value)


def _canonical_guard_mode(value: Any) -> str:
    mode = str(value).strip().replace("_", "-").lower()
    if mode in {"trainer-one-sided", "current-one-sided", "log-one-sided"}:
        return "one-sided"
    return mode


def _snapshot_scalar(market: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in market:
            value = np.asarray(market[name]).reshape(-1)
            if value.size != 1:
                raise ValueError(f"market snapshot field {name} must be scalar")
            return value[0].item()
    return None


def _resolved_value(
    cfg: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: Any,
    provenance: Dict[str, str],
    label: str,
) -> Any:
    for name in names:
        if name in cfg and cfg[name] not in (None, ""):
            provenance[label] = f"config:{name}"
            return cfg[name]
    provenance[label] = "trainer-contract-default"
    return default


@dataclass(frozen=True)
class PolicySpec:
    guard_mode: str
    vw_guard: float
    numerator_guard: float
    denominator_guard: float
    kappa_min: Optional[float]
    kappa_max: Optional[float]
    consumption_min: Optional[float]
    consumption_max: Optional[float]
    portfolio_min: Optional[float]
    portfolio_max: Optional[float]


@dataclass(frozen=True)
class NetworkSpec:
    time_coordinate: str
    input_order: str
    input_transform: str
    activation: str
    dtype: str


@dataclass
class RunSpec:
    run_dir: Path
    config_path: Path
    market_path: Path
    weight_dir: Path
    checkpoints: List[Tuple[int, Path]]
    config: Dict[str, Any]
    problem: MertonProblem
    sigma_inv_mu: np.ndarray
    policy: PolicySpec
    network: NetworkSpec
    seed: int
    market_seed: int
    model_type: str
    eval_margin: float
    eval_window_mode: str
    eval_w_min_requested: Optional[float]
    eval_w_min_symmetric: float
    eval_y_bounds: Tuple[float, float]
    eval_w_bounds: Tuple[float, float]
    market_hash: str
    group: str
    checkpoint_selection: str
    metadata_provenance: Dict[str, str]
    training_protocol: Dict[str, Any]
    checkpoint_schedule: List[int]
    final_checkpoint: Optional[Path]
    final_checkpoint_state_hash: str
    checkpoint_manifest_path: Optional[Path]
    checkpoint_manifest_hash: str
    e6_role: str
    checkpoint_source_iters: Dict[int, int]
    checkpoint_target_policy_iters: Dict[int, int]
    checkpoint_indexing_provenance: Dict[str, Any]
    initial_defect_mode: str
    warm_start_bundle_path: Optional[Path]
    warm_start_bundle_file_hash: str
    warm_start_model_state_hash: str
    warm_start_provenance: Dict[str, Any]


def _configured_path_candidates(
    value: Any,
    run_dir: Path,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Path]:
    """Resolve recorded paths without assuming they were run-dir relative.

    Older launchers recorded ``weight_dir`` relative to the shell working
    directory while ``config.json`` lives several levels below it.  Joining
    that path to ``run_dir`` duplicates the output prefix.  The recorder also
    stores its launch ``cwd``, so prefer that provenance, then retain the
    historical run-directory interpretation and the invocation directory as
    compatibility fallbacks.
    """
    if value in (None, ""):
        return []
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]

    roots: List[Path] = []
    if cfg is not None:
        recorded_cwd = _pick(
            cfg,
            ("cwd", "working_directory", "launch_cwd"),
            default=None,
        )
        if recorded_cwd not in (None, ""):
            roots.append(Path(str(recorded_cwd)).expanduser().resolve())
    roots.extend((run_dir.resolve(), Path.cwd().resolve()))

    candidates: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidate = (root / raw).resolve()
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def _resolve_path(
    value: Any,
    run_dir: Path,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[Path]:
    candidates = _configured_path_candidates(value, run_dir, cfg)
    return candidates[0] if candidates else None


def discover_checkpoints(weight_dir: Path, explicit: Sequence[Path] = ()) -> List[Tuple[int, Path]]:
    files = [Path(path).expanduser().resolve() for path in explicit]
    if not files:
        files = sorted((weight_dir / "iterates").glob("value_net_iter*.pt"))
    checkpoints: Dict[int, Path] = {}
    for path in files:
        match = CHECKPOINT_RE.search(path.name)
        if not match:
            raise ValueError(f"checkpoint must be named value_net_iterNNNN.pt: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        outer = int(match.group(1))
        if outer in checkpoints and checkpoints[outer] != path:
            raise ValueError(f"duplicate checkpoint for outer iteration {outer}")
        checkpoints[outer] = path
    if not checkpoints:
        raise FileNotFoundError(
            f"no iterate checkpoints under {weight_dir / 'iterates'}; "
            "a final/best checkpoint alone cannot reconstruct an exact-map trajectory"
        )
    return sorted(checkpoints.items())


def load_run_spec(
    run_dir: Path,
    *,
    explicit_checkpoints: Sequence[Path] = (),
    weight_dir_override: Optional[Path] = None,
    policy_mode_override: str = "",
    time_coordinate_override: str = "",
    eval_margin_override: Optional[float] = None,
    eval_w_min_override: Optional[float] = None,
    network_dtype: str = "training",
) -> RunSpec:
    run_dir = run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"structured exact-map run requires {config_path}")
    failed_markers = [
        name
        for name in ("_FAILED", "_FAILED_TRAINING", "_FAILED_EVAL", "_STOPPED_EARLY")
        if (run_dir / name).is_file()
    ]
    if failed_markers:
        raise ValueError(f"run has failure/early-stop markers {failed_markers}: {run_dir}")
    success_marked = any((run_dir / name).is_file() for name in ("_SUCCESS", "_SUCCESS_TRAINING"))
    status_complete = False
    status_payload: Dict[str, Any] = {}
    status_path = run_dir / "status.json"
    if status_path.is_file():
        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            status_value = str(status_payload.get("status", ""))
            status_complete = status_value in {"success", "training_complete", "complete", "completed"}
        except Exception:
            status_complete = False
    if not success_marked or (status_path.is_file() and not status_complete):
        raise ValueError(
            f"run is not marked complete: {run_dir}; exact-map evaluation refuses partial checkpoints"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    cfg = _merged_config(raw_config)
    metadata_provenance: Dict[str, str] = {}

    market_path_value = _pick(cfg, ("market_params_path", "market_path"), default=None)
    market_candidates = _configured_path_candidates(
        market_path_value, run_dir, cfg
    )
    market_candidates.append((run_dir / "market_params.npz").resolve())
    market_path = next(
        (path for path in market_candidates if path.is_file()),
        market_candidates[0],
    )
    if not market_path.is_file():
        raise FileNotFoundError(
            f"structured exact-map run requires a frozen market snapshot: {market_path}"
        )
    market = np.load(market_path, allow_pickle=False)
    mu = None
    sigma = None
    for name in ("mu_excess", "mu_excess_np", "mu_R"):
        if name in market:
            mu = np.asarray(market[name], dtype=np.float64).reshape(-1)
            break
    for name in ("Sigma_safe", "Sigma", "sigma", "Sigma_R"):
        if name in market:
            sigma = np.asarray(market[name], dtype=np.float64)
            break
    if mu is None or sigma is None:
        raise KeyError("market_params.npz must contain mu_excess and Sigma_safe/Sigma")
    sigma_inv_mu_snapshot = (
        np.asarray(market["Sigma_inv_mu"], dtype=np.float64).reshape(-1)
        if "Sigma_inv_mu" in market
        else np.linalg.solve(sigma, mu)
    )
    if sigma_inv_mu_snapshot.shape != mu.shape:
        raise ValueError("Sigma_inv_mu in market snapshot has an incompatible shape")
    sigma_inv_mu_recomputed = np.linalg.solve(sigma, mu)
    if not np.allclose(
        sigma_inv_mu_snapshot,
        sigma_inv_mu_recomputed,
        rtol=5e-10,
        atol=5e-12,
    ):
        raise ValueError(
            "Sigma_inv_mu in market_params.npz is inconsistent with "
            "Sigma_safe and mu_excess"
        )
    configured_n_assets = _pick(cfg, ("n_assets", "N_ASSETS"), default=None)
    snapshot_n_assets = _snapshot_scalar(market, ("n_assets", "N_ASSETS"))
    for source, value in (
        ("config", configured_n_assets), ("market snapshot", snapshot_n_assets)
    ):
        if value is not None and int(value) != int(mu.size):
            raise ValueError(
                f"{source} n_assets={int(value)} does not match market dimension={mu.size}"
            )
    metadata_provenance["n_assets"] = (
        "config:n_assets" if configured_n_assets is not None
        else ("market:n_assets" if snapshot_n_assets is not None else "market-array-shape")
    )

    def scalar_with_snapshot(
        label: str,
        config_names: Sequence[str],
        snapshot_names: Sequence[str],
        *,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        for name in config_names:
            if name in cfg and cfg[name] not in (None, ""):
                value = cfg[name]
                metadata_provenance[label] = f"config:{name}"
                snapshot_value = _snapshot_scalar(market, snapshot_names)
                if snapshot_value is not None and not math.isclose(
                    float(value), float(snapshot_value), rel_tol=5e-10, abs_tol=5e-12
                ):
                    raise ValueError(
                        f"config {name}={value} does not match market snapshot "
                        f"{snapshot_names[0]}={snapshot_value}"
                    )
                return value
        snapshot_value = _snapshot_scalar(market, snapshot_names)
        if snapshot_value is not None:
            metadata_provenance[label] = f"market:{next(name for name in snapshot_names if name in market)}"
            return snapshot_value
        if required and default is None:
            raise KeyError(
                f"missing required {label}; tried config fields {', '.join(config_names)} "
                f"and market fields {', '.join(snapshot_names)}"
            )
        metadata_provenance[label] = "trainer-contract-default"
        return default

    horizon = float(scalar_with_snapshot(
        "horizon", ("tau_max", "T_FINAL", "t_max", "horizon"), ("T", "horizon"),
        required=True))
    w_min = float(scalar_with_snapshot(
        "w_min", ("w_min", "W_min", "x_min"), ("w_min", "x_min"), required=True))
    w_max = float(scalar_with_snapshot(
        "w_max", ("w_max", "W_max", "x_max"), ("w_max", "x_max"), required=True))
    if not 0.0 < w_min < w_max:
        raise ValueError("wealth bounds must satisfy 0 < w_min < w_max")
    gamma = float(scalar_with_snapshot(
        "gamma", ("gamma", "gamma_risk"), ("gamma",), required=True))
    discount = float(scalar_with_snapshot(
        "discount", ("rho_discount", "discount", "rho"),
        ("rho_discount", "discount"), required=True))
    bequest = float(scalar_with_snapshot(
        "bequest", ("epsilon_bequest", "epsilon", "bequest", "bequest_weight"),
        ("epsilon", "epsilon_bequest", "bequest"), default=1.0))
    risk_free = float(scalar_with_snapshot(
        "risk_free", ("r", "r_rate", "risk_free"), ("r", "risk_free"), required=True))
    problem = MertonProblem(
        horizon=horizon,
        y_min=float(np.log(w_min)),
        y_max=float(np.log(w_max)),
        gamma=gamma,
        discount=discount,
        bequest=bequest,
        risk_free=risk_free,
        mu_excess=mu,
        sigma=sigma,
    )

    if policy_mode_override:
        mode = _canonical_guard_mode(policy_mode_override)
        metadata_provenance["policy_guard_mode"] = "cli-override"
    else:
        configured_mode = _pick(
            cfg, ("policy_guard_mode", "guard_mode", "denominator_guard_mode"), default=""
        )
        if not configured_mode:
            raise KeyError(
                "config.json does not identify the policy guard; pass --policy-mode "
                "only after auditing the trainer that produced this legacy checkpoint"
            )
        mode = _canonical_guard_mode(configured_mode)
        metadata_provenance["policy_guard_mode"] = "config:policy_guard_mode"
        guard_version = str(_pick(cfg, ("policy_guard_version",), default=""))
        if mode == "one-sided" and guard_version not in {"", "merton-logw-v1"}:
            raise ValueError(
                f"unsupported one-sided policy_guard_version={guard_version!r}"
            )

    bounds_mode = str(_pick(
        cfg, ("policy_bounds_mode",), default="stabilized"
    )).strip().lower()
    if bounds_mode not in {"stabilized", "none"}:
        raise ValueError(
            f"unsupported policy_bounds_mode={bounds_mode!r}; expected stabilized or none"
        )
    metadata_provenance["policy_bounds_mode"] = (
        "config:policy_bounds_mode" if "policy_bounds_mode" in cfg
        else "trainer-contract-default"
    )

    utility_cap = float(_resolved_value(
        cfg, ("utility_cap", "M_utility_cap"), default=1e3,
        provenance=metadata_provenance, label="utility_cap"))
    if gamma <= 1.0 or utility_cap <= 0.0:
        raise ValueError("trainer policy map requires gamma > 1 and utility_cap > 0")
    consumption_floor = ((gamma - 1.0) * utility_cap) ** (-1.0 / (gamma - 1.0))

    if bounds_mode == "none":
        kappa_min = kappa_max = None
        consumption_min = consumption_max = None
        portfolio_min = portfolio_max = None
        for label in (
            "kappa_min", "kappa_max", "consumption_min", "consumption_max",
            "pi_clip_abs",
        ):
            metadata_provenance[label] = "config:policy_bounds_mode=none"
    else:
        derived_kappa_min = consumption_floor / w_min
        # Prefer resolved trainer metadata over raw CLI inputs.
        kappa_min = _optional_float(_resolved_value(
            cfg, ("policy_kappa_min", "kappa_min_bound", "kappa_min"),
            default=derived_kappa_min, provenance=metadata_provenance, label="kappa_min"))
        # ``kappa_max`` is a synthetic-market conditioning argument in the
        # trainer and must not be confused with the policy bound.
        kappa_max = _optional_float(_resolved_value(
            cfg, ("policy_kappa_max", "kappa_max_bound", "consumption_kappa_max"),
            default=3.0, provenance=metadata_provenance, label="kappa_max"))
        consumption_min = _optional_float(_resolved_value(
            cfg, ("policy_c_min", "c_min_bound", "consumption_min"),
            default=consumption_floor, provenance=metadata_provenance, label="consumption_min"))
        consumption_max = _optional_float(_resolved_value(
            cfg, ("policy_c_max", "c_max_bound", "consumption_max"),
            default=w_max, provenance=metadata_provenance, label="consumption_max"))

        clip_abs_present = "pi_clip_abs" in cfg
        clip_abs_value = cfg.get("pi_clip_abs")
        if clip_abs_present:
            pi_clip_abs = _optional_float(clip_abs_value)
            metadata_provenance["pi_clip_abs"] = "config:pi_clip_abs"
            if pi_clip_abs is not None and (
                not math.isfinite(pi_clip_abs) or pi_clip_abs <= 0.0
            ):
                raise ValueError("pi_clip_abs must be positive or none")
            portfolio_min = None if pi_clip_abs is None else -float(pi_clip_abs)
            portfolio_max = None if pi_clip_abs is None else float(pi_clip_abs)
        else:
            legacy_min = _optional_float(_pick(
                cfg, ("policy_pi_min", "pi_min_bound", "pi_min", "portfolio_min"), default=None))
            legacy_max = _optional_float(_pick(
                cfg, ("policy_pi_max", "pi_max_bound", "pi_max", "portfolio_max"), default=None))
            if legacy_min is not None or legacy_max is not None:
                if legacy_min is None or legacy_max is None or not math.isclose(
                    abs(legacy_min), abs(legacy_max), rel_tol=0.0, abs_tol=1e-12
                ) or not legacy_min < 0.0 < legacy_max:
                    raise ValueError("legacy portfolio bounds must be finite and symmetric")
                portfolio_min, portfolio_max = legacy_min, legacy_max
                metadata_provenance["pi_clip_abs"] = "legacy-symmetric-pi-bounds"
            else:
                portfolio_min, portfolio_max = -2.0, 2.0
                metadata_provenance["pi_clip_abs"] = "trainer-contract-default"

    policy = PolicySpec(
        guard_mode=mode,
        vw_guard=float(_resolved_value(
            cfg, ("vw_guard", "marginal_value_guard"), default=1e-8,
            provenance=metadata_provenance, label="vw_guard")),
        numerator_guard=float(_resolved_value(
            cfg,
            ("policy_numerator_guard_eps", "numerator_guard", "policy_guard_eps"),
            default=1e-8,
            provenance=metadata_provenance,
            label="numerator_guard",
        )),
        denominator_guard=float(_resolved_value(
            cfg,
            (
                "policy_denominator_guard_eps", "denominator_guard",
                "vww_guard", "policy_guard_eps",
            ),
            default=1e-8,
            provenance=metadata_provenance, label="denominator_guard")),
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        consumption_min=consumption_min,
        consumption_max=consumption_max,
        portfolio_min=portfolio_min,
        portfolio_max=portfolio_max,
    )
    if policy.vw_guard <= 0.0:
        raise ValueError("vw_guard must be positive")
    if policy.numerator_guard <= 0.0:
        raise ValueError("policy numerator guard epsilon must be positive")
    if policy.denominator_guard <= 0.0:
        raise ValueError("policy denominator guard epsilon must be positive")
    if time_coordinate_override:
        time_coordinate = str(time_coordinate_override).lower()
        metadata_provenance["network_time_coordinate"] = "cli-override"
    else:
        time_coordinate = str(_resolved_value(
            cfg, ("network_time_coordinate", "time_coordinate"), default="t",
            provenance=metadata_provenance, label="network_time_coordinate")).lower()
    if time_coordinate not in {"t", "tau"}:
        raise ValueError("network_time_coordinate must be 't' or 'tau'")
    input_transform = str(_resolved_value(
        cfg, ("network_input_transform", "input_transform"), default="none",
        provenance=metadata_provenance, label="network_input_transform")).lower()
    if input_transform not in {"none", "identity"}:
        raise ValueError(
            f"unsupported network input transform {input_transform!r}; "
            "the exact evaluator must reconstruct the trainer's forward map exactly"
        )
    if network_dtype == "training":
        resolved_dtype = str(_resolved_value(
            cfg, ("network_dtype", "training_dtype", "dtype"), default="float32",
            provenance=metadata_provenance, label="network_dtype")).lower()
    else:
        resolved_dtype = str(network_dtype).lower()
        metadata_provenance["network_dtype"] = "cli-override"
    if resolved_dtype not in {"float32", "float64"}:
        raise ValueError(f"unsupported network dtype: {resolved_dtype}")
    input_order = str(_resolved_value(
        cfg, ("network_input_order", "input_order"), default=f"{time_coordinate},y",
        provenance=metadata_provenance, label="network_input_order"))
    input_tokens = [token.strip().lower() for token in input_order.split(",")]
    if len(input_tokens) != 2 or set(input_tokens) != {time_coordinate, "y"}:
        raise ValueError(
            f"network_input_order={input_order!r} is inconsistent with "
            f"network_time_coordinate={time_coordinate!r}"
        )
    network = NetworkSpec(
        time_coordinate=time_coordinate,
        input_order=input_order,
        input_transform=input_transform,
        activation=str(_resolved_value(
            cfg, ("activation", "value_activation"), default="tanh",
            provenance=metadata_provenance, label="activation")),
        dtype=resolved_dtype,
    )
    margin = first_margin(
        eval_margin_override
        if eval_margin_override is not None
        else _pick(cfg, ("eval_margin",), default="0.10")
    )
    margin_coordinate = str(
        _pick(cfg, ("eval_margin_coordinate",), default="y")
    ).lower()
    if margin_coordinate not in {"y", "log-w", "log_wealth", "log-wealth"}:
        raise ValueError(
            f"Merton exact-map evaluation requires a log-wealth eval margin, got {margin_coordinate!r}"
        )
    # Merton is trained and diagnosed in y=log(w); the primary Q_ev margin
    # therefore shrinks only the log-wealth axis. The time axis is not margin-
    # contracted; evaluate_run applies the trainer's [0,T) endpoint convention.
    symmetric_eval_y_bounds = shrink_bounds(
        problem.y_min, problem.y_max, margin
    )
    symmetric_eval_w_bounds = tuple(
        float(math.exp(value)) for value in symmetric_eval_y_bounds
    )
    if eval_w_min_override is None:
        eval_window_mode = "symmetric-margin"
        eval_w_min_requested = None
        eval_w_bounds = symmetric_eval_w_bounds
    else:
        candidate = float(eval_w_min_override)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError(
                "evaluation w_min must be finite and strictly positive"
            )
        symmetric_lower, symmetric_upper = symmetric_eval_w_bounds
        training_lower = float(math.exp(problem.y_min))
        lower_tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, abs(training_lower)
        )
        if candidate < training_lower - lower_tolerance:
            raise ValueError(
                "evaluation w_min must remain inside the saved nominal "
                f"training interval: got {candidate:.17g}, but training "
                f"w_min is {training_lower:.17g}"
            )
        if candidate >= symmetric_upper:
            raise ValueError(
                "evaluation w_min must lie strictly below the symmetric "
                f"eval-margin upper bound {symmetric_upper:.17g}"
            )
        # Preserve the raw request for provenance but clamp a roundoff-only
        # undershoot to the exact saved training endpoint.
        eval_window_mode = "lower-wealth-override"
        eval_w_min_requested = candidate
        eval_w_bounds = (max(candidate, training_lower), symmetric_upper)
    eval_y_bounds = (
        float(math.log(eval_w_bounds[0])),
        symmetric_eval_y_bounds[1],
    )

    if weight_dir_override is not None:
        weight_dir = weight_dir_override.expanduser().resolve()
    else:
        configured_candidates = _configured_path_candidates(
            _pick(cfg, ("weight_dir", "weight_root", "weights_dir"), default=None),
            run_dir,
            cfg,
        )
        candidates = configured_candidates + [run_dir / "weights", run_dir]
        weight_dir = next(
            (path.resolve() for path in candidates if path is not None and (path / "iterates").is_dir()),
            (configured_candidates[0] if configured_candidates else run_dir).resolve(),
        )
    # Validate the completed run against its full on-disk schedule even when
    # the caller selects a smaller exploratory checkpoint subset.  Otherwise
    # --checkpoint could bypass final-hash/E3-b provenance checks.
    available_checkpoints = discover_checkpoints(weight_dir)
    checkpoints = (
        discover_checkpoints(weight_dir, explicit_checkpoints)
        if explicit_checkpoints
        else available_checkpoints
    )
    checkpoint_selection = "explicit_subset" if explicit_checkpoints else "all"
    config_final = int(_pick(cfg, ("final_outer_iter", "outer_iters"), default=0) or 0)
    status_final = int(
        status_payload.get("final_outer_iter", status_payload.get("outer_iters", 0)) or 0
    )
    if status_final > 0 and config_final > 0 and status_final != config_final:
        raise ValueError(
            f"status final_outer_iter={status_final} does not match "
            f"config outer_iters={config_final}"
        )
    expected_final = status_final or config_final
    actual_outers = [outer for outer, _path in available_checkpoints]
    recorded_e6_roles: Dict[str, str] = {}
    for source, value in (
        ("config", _pick(cfg, ("e6_role",), default=None)),
        ("status", status_payload.get("e6_role")),
    ):
        if value not in (None, ""):
            recorded_e6_roles[source] = str(value).strip().lower()
    if len(set(recorded_e6_roles.values())) > 1:
        raise ValueError(
            "inconsistent e6_role metadata: "
            + ", ".join(
                f"{source}={value!r}"
                for source, value in sorted(recorded_e6_roles.items())
            )
        )
    e6_role = next(iter(recorded_e6_roles.values()), "standard")
    if e6_role not in {"standard", "warmup", "target_branch"}:
        raise ValueError(
            f"unsupported e6_role={e6_role!r}; expected standard, warmup, "
            "or target_branch"
        )
    checkpoint_source_iters: Dict[int, int] = {
        outer: outer - 1 for outer in actual_outers
    }
    checkpoint_target_policy_iters: Dict[int, int] = {
        outer: outer for outer in actual_outers
    }
    checkpoint_indexing_provenance: Dict[str, Any] = {
        "checkpoint_outer_index_base": 1,
        "source_iter_offset_from_checkpoint_outer": -1,
        "target_policy_iter_offset_from_checkpoint_outer": 0,
        "source": "legacy-standard-default",
    }
    warm_start_provenance: Dict[str, Any] = {}
    checkpoint_manifest_path = weight_dir / "checkpoint_manifest.json"
    checkpoint_manifest_hash = ""
    manifest_payload: Dict[str, Any] = {}
    manifest_entries: Dict[int, Mapping[str, Any]] = {}
    if checkpoint_manifest_path.is_file():
        manifest_payload = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        checkpoint_manifest_hash = sha256_file(checkpoint_manifest_path)
        schema_version = int(manifest_payload.get("schema_version", 0) or 0)
        if schema_version < 1:
            raise ValueError(
                f"unsupported checkpoint manifest schema_version={schema_version}"
            )
        manifest_status = str(manifest_payload.get("status", "")).lower()
        if manifest_status not in {
            "complete", "completed", "success", "training-complete", "training_complete"
        }:
            raise ValueError(
                f"checkpoint manifest is not complete: status={manifest_status!r}"
            )
        manifest_e6_role_value = manifest_payload.get("e6_role")
        if manifest_e6_role_value not in (None, ""):
            manifest_e6_role = str(manifest_e6_role_value).strip().lower()
            recorded_e6_roles["manifest"] = manifest_e6_role
            if len(set(recorded_e6_roles.values())) > 1:
                raise ValueError(
                    "inconsistent e6_role metadata: "
                    + ", ".join(
                        f"{source}={value!r}"
                        for source, value in sorted(recorded_e6_roles.items())
                    )
                )
            e6_role = manifest_e6_role
        if e6_role not in {"standard", "warmup", "target_branch"}:
            raise ValueError(
                f"unsupported manifest e6_role={e6_role!r}"
            )
        is_target_branch = e6_role == "target_branch"
        expected_algorithm_offset = 1 if is_target_branch else 0
        if int(
            manifest_payload.get(
                "algorithm_outer_index_offset", expected_algorithm_offset
            )
        ) != expected_algorithm_offset:
            raise ValueError(
                "manifest algorithm_outer_index_offset does not match "
                f"e6_role={e6_role!r}"
            )
        warmup_excluded = bool(
            manifest_payload.get(
                "warmup_excluded_from_outer_history", is_target_branch
            )
        )
        if warmup_excluded != is_target_branch:
            raise ValueError(
                "manifest warmup_excluded_from_outer_history does not match "
                f"e6_role={e6_role!r}"
            )
        raw_indexing = manifest_payload.get("indexing")
        if raw_indexing is None:
            if is_target_branch:
                raise ValueError(
                    "E6 target_branch checkpoint manifest is missing indexing "
                    "provenance"
                )
            raw_indexing = {}
        if not isinstance(raw_indexing, Mapping):
            raise ValueError("manifest indexing must be an object")
        expected_source_offset = 0 if is_target_branch else -1
        expected_target_offset = 1 if is_target_branch else 0
        source_offset = int(
            raw_indexing.get(
                "source_iter_offset_from_checkpoint_outer",
                expected_source_offset,
            )
        )
        target_offset = int(
            raw_indexing.get(
                "target_policy_iter_offset_from_checkpoint_outer",
                expected_target_offset,
            )
        )
        if source_offset != expected_source_offset:
            raise ValueError(
                "manifest source_iter_offset_from_checkpoint_outer="
                f"{source_offset} does not match e6_role={e6_role!r}"
            )
        if target_offset != expected_target_offset:
            raise ValueError(
                "manifest target_policy_iter_offset_from_checkpoint_outer="
                f"{target_offset} does not match e6_role={e6_role!r}"
            )
        checkpoint_indexing_provenance = dict(raw_indexing)
        checkpoint_indexing_provenance.update({
            "checkpoint_outer_index_base": int(
                raw_indexing.get("checkpoint_outer_index_base", 1)
            ),
            "source_iter_offset_from_checkpoint_outer": source_offset,
            "target_policy_iter_offset_from_checkpoint_outer": target_offset,
            "algorithm_outer_index_offset": expected_algorithm_offset,
            "warmup_excluded_from_outer_history": warmup_excluded,
            "source": "checkpoint_manifest",
        })
        if checkpoint_indexing_provenance["checkpoint_outer_index_base"] != 1:
            raise ValueError(
                "manifest checkpoint_outer_index_base must equal one"
            )
        raw_warm_start_provenance = manifest_payload.get(
            "warmup_provenance", {}
        )
        if not isinstance(raw_warm_start_provenance, Mapping):
            raise ValueError("manifest warmup_provenance must be an object")
        warm_start_provenance = dict(raw_warm_start_provenance)
        if is_target_branch and not warm_start_provenance:
            raise ValueError(
                "E6 target_branch manifest is missing warm-up provenance"
            )
        declared_manifest_hash = str(
            status_payload.get("checkpoint_manifest_sha256", "")
        )
        if declared_manifest_hash and declared_manifest_hash != checkpoint_manifest_hash:
            raise ValueError(
                "checkpoint_manifest.json does not match status.json provenance hash"
            )
        for provenance_key in (
            "trainer_protocol", "trainer_protocol_version",
            "trainer_source_marker", "trainer_source_sha256",
            "inner_selection_restore_contract", "checkpoint_timing_contract",
            "policy_guard_mode", "policy_guard_version", "policy_bounds_mode",
            "policy_numerator_guard_eps", "policy_denominator_guard_eps",
        ):
            manifest_value = manifest_payload.get(provenance_key)
            config_value = cfg.get(provenance_key)
            if (
                manifest_value not in (None, "")
                and config_value not in (None, "")
                and manifest_value != config_value
            ):
                raise ValueError(
                    f"checkpoint manifest {provenance_key} does not match config.json"
                )
        manifest_bounds = manifest_payload.get("resolved_policy_bounds")
        if manifest_bounds is not None:
            if not isinstance(manifest_bounds, Mapping):
                raise ValueError("manifest resolved_policy_bounds must be an object")
            expected_bounds = {
                "portfolio_min": policy.portfolio_min,
                "portfolio_max": policy.portfolio_max,
                "kappa_min": policy.kappa_min,
                "kappa_max": policy.kappa_max,
                "consumption_min": policy.consumption_min,
                "consumption_max": policy.consumption_max,
            }
            if dict(manifest_bounds) != expected_bounds:
                raise ValueError(
                    "manifest resolved_policy_bounds does not match config.json"
                )
        resolved_manifest_protocol = manifest_payload.get("resolved_training_protocol")
        if resolved_manifest_protocol is not None:
            if not isinstance(resolved_manifest_protocol, Mapping):
                raise ValueError("manifest resolved_training_protocol must be an object")
            expected_inner_restore = bool(int(
                _pick(cfg, ("inner_best_restore",), default=1)
            ))
            if bool(resolved_manifest_protocol.get("inner_best_restore")) != expected_inner_restore:
                raise ValueError(
                    "manifest inner_best_restore does not match config.json"
                )
            expected_restore_label = (
                "model-plus-optimizer" if expected_inner_restore
                else "final-inner-iterate"
            )
            if resolved_manifest_protocol.get("inner_restore") != expected_restore_label:
                raise ValueError(
                    "manifest inner_restore does not match the resolved training protocol"
                )
            if resolved_manifest_protocol.get("checkpoint_timing") != (
                "post-policy-evaluation-after-optional-heldout-restore"
            ):
                raise ValueError("manifest checkpoint_timing is unsupported")
        raw_entries = manifest_payload.get("checkpoints")
        if not isinstance(raw_entries, list):
            raise ValueError("checkpoint manifest must contain a checkpoints list")
        for entry in raw_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("checkpoint manifest entries must be objects")
            outer = int(entry.get("checkpoint_outer_iter", 0) or 0)
            if outer < 1 or outer in manifest_entries:
                raise ValueError(
                    f"invalid/duplicate checkpoint_outer_iter={outer} in manifest"
                )
            expected_source_iter = outer + source_offset
            expected_target_policy_iter = outer + target_offset
            if int(
                entry.get("source_iter", expected_source_iter)
            ) != expected_source_iter:
                raise ValueError(
                    f"manifest outer={outer} must use paper "
                    f"source_iter={expected_source_iter}"
                )
            if int(
                entry.get(
                    "target_policy_iter", expected_target_policy_iter
                )
            ) != expected_target_policy_iter:
                raise ValueError(
                    f"manifest outer={outer} must use "
                    f"target_policy_iter={expected_target_policy_iter}"
                )
            checkpoint_source_iters[outer] = expected_source_iter
            checkpoint_target_policy_iters[outer] = (
                expected_target_policy_iter
            )
            manifest_entries[outer] = entry
        if sorted(manifest_entries) != actual_outers:
            raise ValueError(
                f"checkpoint manifest schedule={sorted(manifest_entries)}, "
                f"on-disk schedule={actual_outers}"
            )
        requested = int(manifest_payload.get("requested_outer_iters", 0) or 0)
        if expected_final and requested and requested != expected_final:
            raise ValueError(
                f"manifest requested_outer_iters={requested} does not match final outer {expected_final}"
            )
        completed = int(manifest_payload.get("completed_outer_iters", 0) or 0)
        if completed and expected_final and completed != expected_final:
            raise ValueError(
                f"manifest completed_outer_iters={completed} does not match final outer {expected_final}"
            )
        manifest_checkpoint_policy = manifest_payload.get("checkpoint_policy")
        if isinstance(manifest_checkpoint_policy, Mapping):
            manifest_e3b = bool(manifest_checkpoint_policy.get("e3b_checkpoints", False))
            config_e3b = bool(_pick(cfg, ("e3b_checkpoints",), default=False))
            if manifest_e3b != config_e3b:
                raise ValueError(
                    "manifest e3b_checkpoints does not match config.json"
                )
        for outer, path in available_checkpoints:
            entry = manifest_entries[outer]
            entry_path = Path(str(entry.get("path", "")))
            resolved_entry_path = (
                entry_path.resolve() if entry_path.is_absolute()
                else (weight_dir / entry_path).resolve()
            )
            if resolved_entry_path != path.resolve():
                raise ValueError(
                    f"manifest path for outer={outer} does not match discovered checkpoint"
                )
            file_hash = str(entry.get("file_sha256", ""))
            if file_hash and sha256_file(path) != file_hash:
                raise ValueError(f"checkpoint outer={outer} file_sha256 mismatch")
            state_hash = str(entry.get("state_sha256", ""))
            if not state_hash:
                raise ValueError(f"checkpoint outer={outer} is missing state_sha256")
            if canonical_checkpoint_state_hash(path) != state_hash:
                raise ValueError(f"checkpoint outer={outer} state_sha256 mismatch")
    elif status_payload.get("checkpoint_manifest_sha256"):
        raise FileNotFoundError(
            "status.json declares checkpoint_manifest_sha256 but checkpoint_manifest.json is missing"
        )
    elif e6_role == "target_branch":
        raise FileNotFoundError(
            "E6 target_branch exact-map evaluation requires "
            "weights/checkpoint_manifest.json with explicit indexing provenance"
        )

    final_checkpoint: Optional[Path] = None
    final_checkpoint_state_hash = ""
    if expected_final > 0:
        if actual_outers[-1] != expected_final:
            raise ValueError(
                f"iterate schedule ends at {actual_outers[-1]}, but completed run declares final outer {expected_final}"
            )
        final_checkpoint = weight_dir / "value_net_final.pt"
        status_final_candidates = _configured_path_candidates(
            status_payload.get("final_weight_path"), run_dir, cfg
        )
        if not final_checkpoint.is_file():
            final_checkpoint = next(
                (path for path in status_final_candidates if path.is_file()),
                status_final_candidates[0] if status_final_candidates else final_checkpoint,
            )
        if not final_checkpoint.is_file():
            raise FileNotFoundError(f"completed run is missing official final checkpoint: {final_checkpoint}")
        declared_final_hash = str(status_payload.get("final_checkpoint_sha256", ""))
        if declared_final_hash:
            if sha256_file(final_checkpoint) != declared_final_hash:
                raise ValueError("value_net_final.pt does not match status.json provenance hash")
        final_checkpoint_state_hash = canonical_checkpoint_state_hash(final_checkpoint)
        iterate_final_state_hash = canonical_checkpoint_state_hash(available_checkpoints[-1][1])
        if iterate_final_state_hash != final_checkpoint_state_hash:
            raise ValueError(
                "official final checkpoint and final iterate snapshot have different canonical states"
            )
        declared_state_hash = str(
            status_payload.get("final_checkpoint_state_sha256", "")
        )
        if declared_state_hash and declared_state_hash != final_checkpoint_state_hash:
            raise ValueError(
                "value_net_final.pt does not match status.json canonical-state provenance hash"
            )
        declared_iterate_state_hash = str(
            status_payload.get("final_iterate_state_sha256", "")
        )
        if declared_iterate_state_hash and declared_iterate_state_hash != iterate_final_state_hash:
            raise ValueError(
                "final iterate does not match status.json canonical-state provenance hash"
            )
        declared_iterate_file_hash = str(
            status_payload.get("final_iterate_file_sha256", "")
        )
        if (
            declared_iterate_file_hash
            and sha256_file(available_checkpoints[-1][1]) != declared_iterate_file_hash
        ):
            raise ValueError(
                "final iterate does not match status.json file provenance hash"
            )
        declared_final_file_hash = str(
            status_payload.get("final_checkpoint_file_sha256", "")
        )
        if declared_final_file_hash and sha256_file(final_checkpoint) != declared_final_file_hash:
            raise ValueError(
                "value_net_final.pt does not match status.json file provenance hash"
            )
        declared_last_state_hash = str(
            status_payload.get("last_checkpoint_state_sha256", "")
        )
        declared_last_file_hash = str(
            status_payload.get("last_checkpoint_file_sha256", "")
        )
        if declared_last_state_hash or declared_last_file_hash:
            last_checkpoint = weight_dir / "value_net_last.pt"
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    "status.json declares last-checkpoint provenance but "
                    f"{last_checkpoint} is missing"
                )
            if (
                declared_last_state_hash
                and canonical_checkpoint_state_hash(last_checkpoint)
                != declared_last_state_hash
            ):
                raise ValueError(
                    "value_net_last.pt does not match status.json canonical-state provenance hash"
                )
            if declared_last_state_hash and declared_last_state_hash != final_checkpoint_state_hash:
                raise ValueError(
                    "value_net_last.pt and value_net_final.pt have different declared states"
                )
            if declared_last_file_hash and sha256_file(last_checkpoint) != declared_last_file_hash:
                raise ValueError(
                    "value_net_last.pt does not match status.json file provenance hash"
                )
        if manifest_payload:
            official = manifest_payload.get("official_final")
            if not isinstance(official, Mapping):
                raise ValueError("checkpoint manifest is missing official_final")
            if int(official.get("outer_iter", 0) or 0) != expected_final:
                raise ValueError("manifest official_final outer_iter mismatch")
            if str(official.get("state_sha256", "")) != final_checkpoint_state_hash:
                raise ValueError("manifest official_final state_sha256 mismatch")
            artifacts = official.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError("manifest official_final must contain artifacts")
            for label, artifact in artifacts.items():
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"manifest official_final artifact {label} is invalid")
                artifact_path = Path(str(artifact.get("path", "")))
                resolved = (
                    artifact_path.resolve() if artifact_path.is_absolute()
                    else (weight_dir / artifact_path).resolve()
                )
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                artifact_file_hash = str(artifact.get("file_sha256", ""))
                if artifact_file_hash and sha256_file(resolved) != artifact_file_hash:
                    raise ValueError(
                        f"manifest official_final artifact {label} file_sha256 mismatch"
                    )
                artifact_state_hash = str(artifact.get("state_sha256", ""))
                if artifact_state_hash != final_checkpoint_state_hash:
                    raise ValueError(
                        f"manifest official_final artifact {label} state_sha256 mismatch"
                    )
                if canonical_checkpoint_state_hash(resolved) != artifact_state_hash:
                    raise ValueError(
                        f"manifest official_final artifact {label} canonical state mismatch"
                    )
        if bool(_pick(cfg, ("e3b_checkpoints",), default=False)):
            expected_schedule = {
                *range(1, min(10, expected_final) + 1),
                *range(10, expected_final + 1, 10),
                expected_final,
            }
            if set(actual_outers) != expected_schedule:
                raise ValueError(
                    f"E3-b checkpoint schedule={actual_outers}, expected={sorted(expected_schedule)}"
                )
        else:
            every = int(_pick(cfg, ("save_iterate_every",), default=0) or 0)
            if every > 0:
                expected_schedule = {*range(every, expected_final + 1, every), expected_final}
                if set(actual_outers) != expected_schedule:
                    raise ValueError(
                        f"checkpoint schedule={actual_outers}, expected={sorted(expected_schedule)}"
                    )
    snapshot_seed = _snapshot_scalar(market, ("seed", "train_seed"))
    snapshot_market_seed = _snapshot_scalar(market, ("market_seed",))
    configured_seed = _pick(cfg, ("seed", "train_seed"), default=None)
    configured_market_seed = _pick(cfg, ("market_seed",), default=None)
    seed = int(configured_seed if configured_seed is not None else (
        snapshot_seed if snapshot_seed is not None else 12
    ))
    market_seed = int(configured_market_seed if configured_market_seed is not None else (
        snapshot_market_seed if snapshot_market_seed is not None else 12
    ))
    metadata_provenance["seed"] = (
        "config" if configured_seed is not None else
        ("market:seed" if snapshot_seed is not None else "trainer-contract-default")
    )
    metadata_provenance["market_seed"] = (
        "config" if configured_market_seed is not None else
        ("market:market_seed" if snapshot_market_seed is not None else "trainer-contract-default")
    )
    if snapshot_seed is not None and int(snapshot_seed) != seed:
        raise ValueError(
            f"config seed={seed} does not match market_params.npz seed={int(snapshot_seed)}"
        )
    if "market_seed" in market:
        if int(snapshot_market_seed) != market_seed:
            raise ValueError(
                f"config market_seed={market_seed} does not match "
                f"market_params.npz market_seed={int(snapshot_market_seed)}"
            )
    model_type = str(_pick(cfg, ("model_type",), default="pipinn")).lower()
    if model_type not in {"pipinn", "pi-pinn", "pinn-pi"}:
        raise ValueError(f"exact PI-map input must be a PI-PINN run, got model_type={model_type}")
    initial_defect_mode = (
        "warm-start-value"
        if e6_role == "target_branch"
        else "analytic-initial-policy"
    )
    warm_start_bundle_path: Optional[Path] = None
    warm_start_bundle_file_hash = ""
    warm_start_model_state_hash = ""
    if e6_role == "target_branch":
        # The target run does not copy v~_0 into its own iterate directory.
        # Merge redundant recorder fields only as fallbacks; the manifest
        # remains the primary indexing/provenance authority.
        warm_keys = {
            key
            for payload in (cfg, status_payload, warm_start_provenance)
            for key in payload
            if str(key).startswith("e6_warm")
            or key in {
                "first_target_policy_source",
                "e6_target_phase_start_algorithm_iter",
            }
        }
        merged_warm_provenance: Dict[str, Any] = {}
        for payload in (cfg, status_payload, warm_start_provenance):
            for key in warm_keys:
                value = payload.get(key)
                if value not in (None, ""):
                    merged_warm_provenance[key] = value
        warm_start_provenance = merged_warm_provenance

        first_policy_source = str(
            warm_start_provenance.get(
                "first_target_policy_source", ""
            )
        )
        if first_policy_source != "warm_start_value_net":
            raise ValueError(
                "E6 target_branch warm-up provenance must declare "
                "first_target_policy_source='warm_start_value_net'"
            )
        warmup_outer_count = int(
            warm_start_provenance.get("e6_warmup_outer_count", 0) or 0
        )
        if warmup_outer_count != 1:
            raise ValueError(
                "E6 target_branch exact-map indexing currently requires "
                "exactly one common warm-up outer"
            )
        target_start = int(
            warm_start_provenance.get(
                "e6_target_phase_start_algorithm_iter", 0
            )
            or 0
        )
        if target_start != 2:
            raise ValueError(
                "E6 target_branch must start its target phase at algorithm "
                "outer iteration 2"
            )

        path_values: List[Any] = []
        for key in (
            "e6_warmup_bundle_path",
            "e6_warm_start_source",
            "e6_warm_start",
        ):
            for payload in (
                warm_start_provenance,
                status_payload,
                cfg,
            ):
                value = payload.get(key)
                if value not in (None, "") and value not in path_values:
                    path_values.append(value)
        warm_candidates: List[Path] = []
        seen_warm_candidates: set[str] = set()
        for value in path_values:
            for candidate in _configured_path_candidates(
                value, run_dir, cfg
            ):
                key = str(candidate)
                if key not in seen_warm_candidates:
                    warm_candidates.append(candidate)
                    seen_warm_candidates.add(key)
        # A moved output tree invalidates recorded absolute paths.  The E6
        # launcher uses this deterministic sweep-relative location.
        if len(run_dir.parents) >= 2:
            sweep_root = run_dir.parent.parent
            fallback = (
                sweep_root
                / "e6_warm_starts"
                / f"n_assets{problem.n_assets}"
                / f"seed{seed}"
                / "e6_warm_start.pt"
            ).resolve()
            if str(fallback) not in seen_warm_candidates:
                warm_candidates.append(fallback)
        warm_start_bundle_path = next(
            (path for path in warm_candidates if path.is_file()),
            None,
        )
        if warm_start_bundle_path is None:
            attempted = ", ".join(str(path) for path in warm_candidates)
            raise FileNotFoundError(
                "E6 target_branch warm-start bundle is missing; tried "
                f"{attempted or '<no recorded path>'}"
            )

        declared_bundle_hashes = {
            str(value)
            for key in (
                "e6_warmup_bundle_file_sha256",
                "e6_warm_start_bundle_sha256",
                "e6_warm_start_loaded_bundle_sha256",
            )
            for value in [warm_start_provenance.get(key)]
            if value not in (None, "")
        }
        if not declared_bundle_hashes:
            raise ValueError(
                "E6 target_branch warm-up provenance is missing the bundle "
                "file hash"
            )
        if len(declared_bundle_hashes) != 1:
            raise ValueError(
                "E6 target_branch warm-up bundle hashes disagree"
            )
        warm_start_bundle_file_hash = sha256_file(
            warm_start_bundle_path
        )
        if warm_start_bundle_file_hash not in declared_bundle_hashes:
            raise ValueError(
                "E6 target_branch warm-start bundle hash mismatch"
            )

        declared_state_hashes = {
            str(value)
            for key in (
                "e6_warmup_model_state_sha256",
                "e6_warm_start_model_sha256",
            )
            for value in [warm_start_provenance.get(key)]
            if value not in (None, "")
        }
        if not declared_state_hashes:
            raise ValueError(
                "E6 target_branch warm-up provenance is missing the model "
                "state hash"
            )
        if len(declared_state_hashes) != 1:
            raise ValueError(
                "E6 target_branch warm-up model-state hashes disagree"
            )
        warm_start_model_state_hash = (
            _load_and_validate_e6_warmup_bundle(
                warm_start_bundle_path,
                expected_seed=seed,
                expected_market_seed=market_seed,
                expected_n_assets=problem.n_assets,
            )
        )
        if warm_start_model_state_hash not in declared_state_hashes:
            raise ValueError(
                "E6 target_branch warm-start model-state hash mismatch"
            )

    market_hash = canonical_array_hash({
        "mu_excess": mu,
        "Sigma": sigma,
        "Sigma_inv_mu": sigma_inv_mu_snapshot,
    })
    training_defaults: Dict[str, Any] = {
        "value_hidden": 256, "value_depth": 3,
        "outer_iters": 500, "eval_epochs": 200, "batch_size": 3000,
        "terminal_frac": 0.5, "lr": 5e-4,
        "scheduler_patience": 10, "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-8, "lr_schedule": "carry_plateau",
        "adam_reset": "keep", "carry_lr_min": 1e-5, "carry_lr_max": 5e-4,
        "w_terminal": 10.0, "w_shape": 1.0, "w_eta": 3.0,
        "eta_focus_w": None, "eta_clip": 10.0,
        "pi_init_method": "myopic", "pi_init_scale": 1.0,
        "c_init_method": "proportional",
        "inner_best_restore": 1, "sel_points": 10000,
        "sel_terminal_points": 2000, "sel_every": 50, "sel_patience": 6,
        "pe_resample_every": 0, "pres_target": None,
        "val_points": 100000, "val_terminal_points": 10000, "val_every": 1,
        "pde_stop_threshold": None, "pde_stop_start_outer": 0,
        "pde_stop_patience": 1, "timing_mode": False,
    }

    def normalize_protocol_value(value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {"none", "null", "nan", ""}:
                return None
            return text
        if isinstance(value, np.generic):
            return value.item()
        return value

    training_protocol = {
        key: normalize_protocol_value(_pick(cfg, (key,), default=default))
        for key, default in training_defaults.items()
    }
    # Provenance fields affect whether two checkpoint trajectories implement
    # the same training/map contract, so they belong in the grouping key as
    # well as in the human-readable protocol record.
    for key in (
        "trainer_protocol", "trainer_protocol_version", "trainer_source",
        "trainer_source_marker", "trainer_source_sha256", "policy_guard_version",
        "inner_selection_restore_contract", "checkpoint_timing_contract",
    ):
        training_protocol[key] = normalize_protocol_value(
            _pick(cfg, (key,), default=None)
        )
    # These derived values are part of training semantics even when legacy
    # configs omitted their argparse spelling.
    training_protocol["utility_cap"] = utility_cap
    training_protocol["policy"] = asdict(policy)
    training_protocol["e6_role"] = e6_role
    training_protocol["checkpoint_indexing"] = dict(
        checkpoint_indexing_provenance
    )

    group_payload = {
        "problem": "merton",
        "n_assets": problem.n_assets,
        "market_sha256": market_hash,
        "horizon": problem.horizon,
        "gamma": problem.gamma,
        "discount": problem.discount,
        "bequest": problem.bequest,
        "risk_free": problem.risk_free,
        "w_bounds": [w_min, w_max],
        "eval_margin": margin,
        "policy": asdict(policy),
        "network": asdict(network),
        "training": training_protocol,
    }
    if eval_w_min_override is not None:
        group_payload["evaluation_window_override"] = {
            "eval_w_min_requested": eval_w_min_requested,
            "eval_w_min_symmetric": symmetric_eval_w_bounds[0],
            "eval_y_bounds": list(eval_y_bounds),
            "eval_w_bounds": list(eval_w_bounds),
        }
    return RunSpec(
        run_dir=run_dir,
        config_path=config_path,
        market_path=market_path,
        weight_dir=weight_dir,
        checkpoints=checkpoints,
        config=cfg,
        problem=problem,
        sigma_inv_mu=sigma_inv_mu_snapshot,
        policy=policy,
        network=network,
        seed=seed,
        market_seed=market_seed,
        model_type="pipinn",
        eval_margin=margin,
        eval_window_mode=eval_window_mode,
        eval_w_min_requested=eval_w_min_requested,
        eval_w_min_symmetric=symmetric_eval_w_bounds[0],
        eval_y_bounds=eval_y_bounds,
        eval_w_bounds=eval_w_bounds,
        market_hash=market_hash,
        group=stable_hash(group_payload)[:12],
        checkpoint_selection=checkpoint_selection,
        metadata_provenance=metadata_provenance,
        training_protocol=training_protocol,
        checkpoint_schedule=actual_outers,
        final_checkpoint=final_checkpoint,
        final_checkpoint_state_hash=final_checkpoint_state_hash,
        checkpoint_manifest_path=(
            checkpoint_manifest_path if checkpoint_manifest_path.is_file() else None
        ),
        checkpoint_manifest_hash=checkpoint_manifest_hash,
        e6_role=e6_role,
        checkpoint_source_iters=checkpoint_source_iters,
        checkpoint_target_policy_iters=checkpoint_target_policy_iters,
        checkpoint_indexing_provenance=checkpoint_indexing_provenance,
        initial_defect_mode=initial_defect_mode,
        warm_start_bundle_path=warm_start_bundle_path,
        warm_start_bundle_file_hash=warm_start_bundle_file_hash,
        warm_start_model_state_hash=warm_start_model_state_hash,
        warm_start_provenance=warm_start_provenance,
    )


def _state_dict_from_checkpoint(
    torch: Any,
    path: Path,
    *,
    allow_e6_warmup_bundle: bool = False,
) -> Mapping[str, Any]:
    if allow_e6_warmup_bundle:
        # The trusted trainer-produced E6 bundle contains NumPy/CUDA RNG
        # provenance in addition to tensors.  PyTorch's restricted
        # ``weights_only`` unpickler cannot deserialize that metadata, so use
        # the full loader only after load_run_spec has verified the recorded
        # bundle file hash.
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
    else:
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping):
        for key in (
            "state_dict",
            "model_state_dict",
            "value_net_state_dict",
            "model_state",
            "model",
        ):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint does not contain a state_dict: {path}")
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key)
        for prefix in ("module.", "value_net.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _load_e6_warmup_bundle_payload(path: Path) -> Any:
    """Load a hash-verified trainer E6 bundle, including non-tensor metadata.

    This helper is used only after ``load_run_spec`` has matched the complete
    file SHA-256 against the trainer provenance. Ordinary model checkpoints
    continue to use PyTorch's restricted ``weights_only=True`` path.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production evaluator requires Torch
        raise RuntimeError(
            "PyTorch is required to inspect E6 warm-start bundle provenance"
        ) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _validate_e6_warmup_bundle_payload(
    payload: Any,
    *,
    expected_seed: int,
    expected_market_seed: int,
    expected_n_assets: int,
) -> str:
    """Validate the trainer's common-warm-start artifact contract.

    Returns the canonical hash of the bundled value-network state only after
    both metadata and the bundle's internal state hash have been verified.
    Optimizer/RNG state is intentionally neither loaded into live objects nor
    used by the FD evaluator.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("E6 warm-start bundle must be a mapping")

    def required_int(name: str) -> int:
        value = payload.get(name)
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"E6 warm-start bundle has invalid {name}={value!r}"
            ) from exc

    if required_int("schema_version") != E6_WARMUP_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported E6 warm-start bundle schema_version="
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("kind") != E6_WARMUP_BUNDLE_KIND:
        raise ValueError(
            "unexpected E6 warm-start bundle kind="
            f"{payload.get('kind')!r}"
        )
    if payload.get("warm_start_protocol") != E6_WARM_START_PROTOCOL:
        raise ValueError(
            "unexpected E6 warm-start protocol="
            f"{payload.get('warm_start_protocol')!r}"
        )
    for name, expected in (
        ("seed", int(expected_seed)),
        ("market_seed", int(expected_market_seed)),
        ("n_assets", int(expected_n_assets)),
    ):
        actual = required_int(name)
        if actual != expected:
            raise ValueError(
                f"E6 warm-start bundle {name}={actual} does not match "
                f"target run {name}={expected}"
            )
    if required_int("outer_count") != 1:
        raise ValueError(
            "E6 warm-start bundle must end after exactly one outer solve"
        )

    try:
        warmup_target = float(payload.get("warmup_pres_target"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "E6 warm-start bundle has invalid warmup_pres_target"
        ) from exc
    if (
        not math.isfinite(warmup_target)
        or not math.isclose(
            warmup_target, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise ValueError(
            "E6 warm-start bundle must use warmup_pres_target=1.0, got "
            f"{warmup_target!r}"
        )
    try:
        achieved = float(payload.get("warmup_achieved_pres"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "E6 warm-start bundle has invalid warmup_achieved_pres"
        ) from exc
    if (
        not math.isfinite(achieved)
        or achieved <= 0.0
        or achieved > warmup_target * (1.0 + 1.0e-9)
    ):
        raise ValueError(
            "E6 warm-start bundle is not admissible: "
            f"warmup_achieved_pres={achieved!r} must lie in (0,1]"
        )

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("E6 warm-start bundle is missing model_state mapping")
    if not any(hasattr(value, "detach") for value in model_state.values()):
        raise ValueError(
            "E6 warm-start bundle model_state has no tensor values"
        )
    internal_state_hash = str(payload.get("model_state_sha256", ""))
    if not internal_state_hash:
        raise ValueError(
            "E6 warm-start bundle is missing model_state_sha256"
        )
    actual_state_hash = canonical_state_dict_hash(model_state)
    if actual_state_hash != internal_state_hash:
        raise ValueError(
            "E6 warm-start bundle internal model_state_sha256 mismatch"
        )
    return actual_state_hash


def _load_and_validate_e6_warmup_bundle(
    path: Path,
    *,
    expected_seed: int,
    expected_market_seed: int,
    expected_n_assets: int,
) -> str:
    return _validate_e6_warmup_bundle_payload(
        _load_e6_warmup_bundle_payload(path),
        expected_seed=expected_seed,
        expected_market_seed=expected_market_seed,
        expected_n_assets=expected_n_assets,
    )


def canonical_state_dict_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensor content independent of ``torch.save`` container bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def canonical_checkpoint_state_hash(
    path: Path,
    *,
    allow_e6_warmup_bundle: bool = False,
) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production evaluator requires Torch
        raise RuntimeError(
            "PyTorch is required to verify canonical checkpoint-state provenance"
        ) from exc
    return canonical_state_dict_hash(
        _state_dict_from_checkpoint(
            torch,
            path,
            allow_e6_warmup_bundle=allow_e6_warmup_bundle,
        )
    )


def _key_order(name: str) -> Tuple[Any, ...]:
    parts: List[Any] = []
    for token in re.split(r"(\d+)", name):
        parts.append(int(token) if token.isdigit() else token)
    return tuple(parts)


class TorchCheckpointEvaluator:
    """Import-safe adapter around a raw MLP state_dict and the shared G map."""

    def __init__(self, checkpoint: Path, run: RunSpec, device: str) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - depends on training env
            raise RuntimeError(
                "PyTorch is required for checkpoint evaluation. Run this command in the Merton training environment."
            ) from exc
        try:
            from merton_policy import (
                consumption_from_log_derivative,
                portfolio_from_log_derivatives,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("cannot import the shared merton_policy helpers") from exc

        self.torch = torch
        self.run = run
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.dtype = torch.float64 if run.network.dtype == "float64" else torch.float32
        self._consumption_helper = consumption_from_log_derivative
        self._portfolio_helper = portfolio_from_log_derivatives
        is_warm_bundle = (
            run.warm_start_bundle_path is not None
            and checkpoint.resolve() == run.warm_start_bundle_path.resolve()
        )
        state = _state_dict_from_checkpoint(
            torch,
            checkpoint,
            allow_e6_warmup_bundle=is_warm_bundle,
        )
        weight_items = sorted(
            [
                (key, tensor)
                for key, tensor in state.items()
                if str(key).endswith(".weight") and getattr(tensor, "ndim", 0) == 2
            ],
            key=lambda item: _key_order(item[0]),
        )
        if len(weight_items) < 2:
            raise ValueError(f"could not infer an MLP from checkpoint keys: {checkpoint}")
        expected_weight_dtype = (
            torch.float64 if run.network.dtype == "float64" else torch.float32
        )
        mismatched_dtypes = {
            str(weight.dtype) for _key, weight in weight_items
            if weight.dtype != expected_weight_dtype
        }
        if mismatched_dtypes and run.metadata_provenance.get("network_dtype") != "cli-override":
            raise ValueError(
                f"checkpoint weight dtype(s)={sorted(mismatched_dtypes)} do not match "
                f"recorded network_dtype={run.network.dtype}"
            )
        layers = []
        for key, weight in weight_items:
            bias_key = key[:-len("weight")] + "bias"
            if bias_key not in state:
                raise KeyError(f"missing bias for {key}")
            # Construct in the checkpoint dtype first.  Creating a default
            # float32 layer and only then casting would silently round a
            # genuinely float64 training checkpoint before evaluation.
            linear = nn.Linear(
                int(weight.shape[1]), int(weight.shape[0]), bias=True
            ).to(dtype=weight.dtype)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(state[bias_key])
            layers.append(linear)
        if int(layers[0].in_features) != 2 or int(layers[-1].out_features) != 1:
            raise ValueError(
                f"expected a two-input, scalar-output Merton network; got "
                f"{layers[0].in_features}->{layers[-1].out_features}"
            )
        recorded_depth = _pick(run.config, ("value_depth",), default=None)
        recorded_hidden = _pick(run.config, ("value_hidden",), default=None)
        if recorded_depth is not None and len(layers) - 1 != int(recorded_depth):
            raise ValueError(
                f"checkpoint depth={len(layers) - 1} does not match "
                f"config value_depth={recorded_depth}"
            )
        hidden_widths = [int(layer.out_features) for layer in layers[:-1]]
        if recorded_hidden is not None and any(
            width != int(recorded_hidden) for width in hidden_widths
        ):
            raise ValueError(
                f"checkpoint hidden widths={hidden_widths} do not match "
                f"config value_hidden={recorded_hidden}"
            )

        activation_name = run.network.activation.lower()
        activation_table = {
            "tanh": torch.tanh,
            "silu": torch.nn.functional.silu,
            "gelu": torch.nn.functional.gelu,
            "softplus": torch.nn.functional.softplus,
            "relu": torch.nn.functional.relu,
        }
        if activation_name not in activation_table:
            raise ValueError(f"unsupported inferred-network activation: {activation_name}")
        activation = activation_table[activation_name]

        class InferredMLP(nn.Module):
            def __init__(self, modules: Sequence[Any]) -> None:
                super().__init__()
                self.layers = nn.ModuleList(modules)

            def forward(self, inputs: Any) -> Any:
                value = inputs
                for layer in self.layers[:-1]:
                    value = activation(layer(value))
                return self.layers[-1](value)

        self.model = InferredMLP(layers).to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.sigma_inv_mu = torch.as_tensor(
            run.sigma_inv_mu, device=self.device, dtype=self.dtype
        )
        self.policy_hash = hashlib.sha256()

    def _inputs(self, tau: Any, y: Any) -> Any:
        torch = self.torch
        if self.run.network.time_coordinate == "tau":
            time_value = tau
        else:
            time_value = self.run.problem.horizon - tau
        tokens = [token.strip().lower() for token in self.run.network.input_order.split(",")]
        if len(tokens) != 2 or set(tokens) != {self.run.network.time_coordinate, "y"}:
            raise ValueError(f"unsupported network_input_order={self.run.network.input_order!r}")
        columns = []
        for token in tokens:
            columns.append(y if token == "y" else time_value)
        return torch.cat(columns, dim=1)

    def bundle_at_points(self, tau_values: np.ndarray, y_values: np.ndarray, chunk: int = 65536) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        tau_flat = np.asarray(tau_values, dtype=np.float64).reshape(-1)
        y_flat = np.asarray(y_values, dtype=np.float64).reshape(-1)
        if tau_flat.shape != y_flat.shape:
            raise ValueError("tau and y point arrays must have identical shapes")
        out_v: List[np.ndarray] = []
        out_y: List[np.ndarray] = []
        out_yy: List[np.ndarray] = []
        for start in range(0, tau_flat.size, int(chunk)):
            stop = min(start + int(chunk), tau_flat.size)
            tau_t = torch.as_tensor(
                tau_flat[start:stop, None], device=self.device, dtype=self.dtype
            )
            y_t = torch.as_tensor(
                y_flat[start:stop, None], device=self.device, dtype=self.dtype
            ).requires_grad_(True)
            value = self.model(self._inputs(tau_t, y_t))
            value_y = torch.autograd.grad(
                value, y_t, grad_outputs=torch.ones_like(value), create_graph=True, retain_graph=True
            )[0]
            value_yy = torch.autograd.grad(
                value_y, y_t, grad_outputs=torch.ones_like(value_y), create_graph=False
            )[0]
            out_v.append(value.detach().cpu().numpy().reshape(-1))
            out_y.append(value_y.detach().cpu().numpy().reshape(-1))
            out_yy.append(value_yy.detach().cpu().numpy().reshape(-1))
        return np.concatenate(out_v), np.concatenate(out_y), np.concatenate(out_yy)

    def bundle_on_tensor_grid(self, tau: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tt, yy = np.meshgrid(
            np.asarray(tau, dtype=np.float64), np.asarray(y, dtype=np.float64), indexing="ij"
        )
        shape = tt.shape
        bundle = self.bundle_at_points(tt.ravel(), yy.ravel())
        return tuple(item.reshape(shape) for item in bundle)  # type: ignore[return-value]

    @staticmethod
    def _mask_count(mask: Any) -> float:
        return float(mask.detach().sum().item())

    def policy(
        self,
        tau_value: float,
        y_values: np.ndarray,
        *,
        policy_extension: str = "boundary-projection",
    ) -> Tuple[np.ndarray, np.ndarray, Mapping[str, float]]:
        torch = self.torch
        y_np = np.asarray(y_values, dtype=np.float64).reshape(-1)
        eval_y_np, outside = _policy_extension_y(
            self.run.problem, y_np, policy_extension
        )
        tau_np = np.full_like(y_np, float(tau_value))
        tau_t = torch.as_tensor(tau_np[:, None], device=self.device, dtype=self.dtype)
        y_t = torch.as_tensor(
            eval_y_np[:, None], device=self.device, dtype=self.dtype
        ).requires_grad_(True)
        value = self.model(self._inputs(tau_t, y_t))
        value_y = torch.autograd.grad(
            value, y_t, grad_outputs=torch.ones_like(value), create_graph=True, retain_graph=True
        )[0]
        value_yy = torch.autograd.grad(
            value_y, y_t, grad_outputs=torch.ones_like(value_y), create_graph=False, retain_graph=True
        )[0]
        spec = self.run.policy
        consumption, c_diag = self._consumption_helper(
            value_y,
            y_t,
            gamma=self.run.problem.gamma,
            vw_guard=spec.vw_guard,
            kappa_min=spec.kappa_min,
            kappa_max=spec.kappa_max,
            consumption_min=spec.consumption_min,
            consumption_max=spec.consumption_max,
        )
        portfolio, p_diag = self._portfolio_helper(
            value_y,
            value_yy,
            y_t,
            self.sigma_inv_mu,
            guard_mode=spec.guard_mode,
            numerator_guard=spec.numerator_guard,
            denominator_guard=spec.denominator_guard,
            portfolio_min=spec.portfolio_min,
            portfolio_max=spec.portfolio_max,
        )
        extension_level_clip = torch.zeros_like(consumption, dtype=torch.bool)
        if str(policy_extension).strip().lower().replace("_", "-") == "boundary-projection":
            # Evaluate G only at the projected network coordinate, hold the
            # consumption/wealth ratio constant, and transfer it back to the
            # actual FD wealth.  Reapply configured level bounds so the
            # stabilized trainer policy contract remains true globally.
            actual_y_t = torch.as_tensor(
                y_np[:, None], device=self.device, dtype=self.dtype
            )
            kappa_boundary = consumption / torch.exp(y_t.detach())
            consumption_extended = kappa_boundary * torch.exp(actual_y_t)
            consumption = consumption_extended
            if spec.consumption_min is not None:
                extension_level_clip |= consumption < float(spec.consumption_min)
                consumption = torch.clamp(
                    consumption, min=float(spec.consumption_min)
                )
            if spec.consumption_max is not None:
                extension_level_clip |= consumption > float(spec.consumption_max)
                consumption = torch.clamp(
                    consumption, max=float(spec.consumption_max)
                )
        c_np = consumption.detach().cpu().numpy().reshape(-1)
        p_np = portfolio.detach().cpu().numpy()
        self.policy_hash.update(str(policy_extension).encode("utf-8") + b"\0")
        self.policy_hash.update(np.asarray([tau_value], dtype="<f8").tobytes())
        self.policy_hash.update(np.asarray(y_np, dtype="<f8").tobytes())
        self.policy_hash.update(np.asarray(c_np, dtype="<f8").tobytes())
        self.policy_hash.update(np.asarray(p_np, dtype="<f8").tobytes())
        n = float(y_np.size)
        component_n = float(p_np.size)
        diag = {
            "points": n,
            "vw_guard_count": self._mask_count(c_diag["vw_guard"]),
            "numerator_guard_count": self._mask_count(p_diag["numerator_guard"]),
            "denominator_guard_count": self._mask_count(p_diag["denominator_guard"]),
            "positive_curvature_count": self._mask_count(p_diag["positive_curvature"]),
            "kappa_clip_count": self._mask_count(c_diag["kappa_low_clip"] | c_diag["kappa_high_clip"]),
            "consumption_clip_count": self._mask_count(
                c_diag["consumption_low_clip"]
                | c_diag["consumption_high_clip"]
                | extension_level_clip
            ),
            "portfolio_any_clip_count": self._mask_count(p_diag["portfolio_any_clip"]),
            # Store the component rate on a point-denominator scale so the FD
            # accumulator's final division by points produces the true rate.
            "portfolio_component_clip_count": (
                self._mask_count(
                    p_diag["portfolio_low_clip_components"] | p_diag["portfolio_high_clip_components"]
                )
                * n / component_n
            ),
            "outside_collocation_count": float(np.count_nonzero(outside)),
        }
        variance = np.einsum(
            "bi,ij,bj->b", p_np, self.run.problem.sigma, p_np, optimize=True
        )
        diag["min_diffusion_variance"] = float(np.min(variance))
        diag["max_diffusion_variance"] = float(np.max(variance))
        return c_np, p_np, diag

    def precompute_policy(
        self,
        tau_values: np.ndarray,
        y_values: np.ndarray,
        *,
        chunk: int = 65536,
        policy_extension: str = "boundary-projection",
    ) -> Tuple[Any, str, Dict[str, float]]:
        """Batch all frozen feedback coefficients for one FD tensor grid.

        The returned callable only performs row lookup.  This changes neither
        G nor the FD scheme; it replaces hundreds of small second-derivative
        GPU launches by a few chunked launches and hashes the frozen arrays.
        """
        torch = self.torch
        tau_grid = np.asarray(tau_values, dtype=np.float64).reshape(-1)
        y_grid = np.asarray(y_values, dtype=np.float64).reshape(-1)
        tt, yy = np.meshgrid(tau_grid, y_grid, indexing="ij")
        flat_tau, flat_y = tt.ravel(), yy.ravel()
        eval_flat_y, outside_flat = _policy_extension_y(
            self.run.problem, flat_y, policy_extension
        )
        _value, value_y, value_yy = self.bundle_at_points(
            flat_tau, eval_flat_y, chunk=chunk
        )
        n_times, n_y = tt.shape
        storage_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        consumption = np.empty(flat_y.size, dtype=storage_dtype)
        portfolio = np.empty((flat_y.size, self.run.problem.n_assets), dtype=storage_dtype)
        count_keys = (
            "vw_guard_count", "numerator_guard_count", "denominator_guard_count",
            "positive_curvature_count",
            "kappa_clip_count", "consumption_clip_count", "portfolio_any_clip_count",
            "portfolio_component_clip_count",
            "outside_collocation_count",
        )
        counts = {key: np.zeros(n_times, dtype=np.float64) for key in count_keys}
        spec = self.run.policy
        for start in range(0, flat_y.size, int(chunk)):
            stop = min(start + int(chunk), flat_y.size)
            rows = np.arange(start, stop, dtype=np.int64) // n_y
            y_t = torch.as_tensor(
                eval_flat_y[start:stop, None],
                device=self.device,
                dtype=self.dtype,
            )
            vy_t = torch.as_tensor(value_y[start:stop, None], device=self.device, dtype=self.dtype)
            vyy_t = torch.as_tensor(value_yy[start:stop, None], device=self.device, dtype=self.dtype)
            c_t, c_diag = self._consumption_helper(
                vy_t,
                y_t,
                gamma=self.run.problem.gamma,
                vw_guard=spec.vw_guard,
                kappa_min=spec.kappa_min,
                kappa_max=spec.kappa_max,
                consumption_min=spec.consumption_min,
                consumption_max=spec.consumption_max,
            )
            p_t, p_diag = self._portfolio_helper(
                vy_t,
                vyy_t,
                y_t,
                self.sigma_inv_mu,
                guard_mode=spec.guard_mode,
                numerator_guard=spec.numerator_guard,
                denominator_guard=spec.denominator_guard,
                portfolio_min=spec.portfolio_min,
                portfolio_max=spec.portfolio_max,
            )
            extension_level_clip = torch.zeros_like(c_t, dtype=torch.bool)
            if str(policy_extension).strip().lower().replace("_", "-") == "boundary-projection":
                actual_y_t = torch.as_tensor(
                    flat_y[start:stop, None],
                    device=self.device,
                    dtype=self.dtype,
                )
                kappa_boundary = c_t / torch.exp(y_t)
                c_extended = kappa_boundary * torch.exp(actual_y_t)
                c_t = c_extended
                if spec.consumption_min is not None:
                    extension_level_clip |= c_t < float(spec.consumption_min)
                    c_t = torch.clamp(c_t, min=float(spec.consumption_min))
                if spec.consumption_max is not None:
                    extension_level_clip |= c_t > float(spec.consumption_max)
                    c_t = torch.clamp(c_t, max=float(spec.consumption_max))
            consumption[start:stop] = c_t.detach().cpu().numpy().reshape(-1)
            portfolio[start:stop] = p_t.detach().cpu().numpy()

            point_masks = {
                "vw_guard_count": c_diag["vw_guard"].reshape(-1),
                "numerator_guard_count": p_diag["numerator_guard"].reshape(-1),
                "denominator_guard_count": p_diag["denominator_guard"].reshape(-1),
                "positive_curvature_count": p_diag["positive_curvature"].reshape(-1),
                "kappa_clip_count": (c_diag["kappa_low_clip"] | c_diag["kappa_high_clip"]).reshape(-1),
                "consumption_clip_count": (
                    c_diag["consumption_low_clip"]
                    | c_diag["consumption_high_clip"]
                    | extension_level_clip
                ).reshape(-1),
                "portfolio_any_clip_count": p_diag["portfolio_any_clip"].reshape(-1),
                "outside_collocation_count": torch.as_tensor(
                    outside_flat[start:stop],
                    device=self.device,
                    dtype=torch.bool,
                ),
            }
            for key, mask in point_masks.items():
                weights = mask.detach().cpu().numpy().astype(np.float64, copy=False)
                counts[key] += np.bincount(rows, weights=weights, minlength=n_times)
            component_rate = (
                p_diag["portfolio_low_clip_components"] | p_diag["portfolio_high_clip_components"]
            ).to(dtype=torch.float64).mean(dim=1).detach().cpu().numpy()
            counts["portfolio_component_clip_count"] += np.bincount(
                rows, weights=component_rate, minlength=n_times
            )

        consumption_grid = consumption.reshape(n_times, n_y)
        portfolio_grid = portfolio.reshape(n_times, n_y, self.run.problem.n_assets)
        digest = hashlib.sha256()
        digest.update(str(policy_extension).encode("utf-8") + b"\0")
        digest.update(np.asarray(tau_grid, dtype="<f8").tobytes())
        digest.update(np.asarray(y_grid, dtype="<f8").tobytes())
        digest.update(np.asarray(consumption_grid, dtype="<f8").tobytes())
        digest.update(np.asarray(portfolio_grid, dtype="<f8").tobytes())

        # Midpoints and endpoint diagnostic grids are both strictly sorted.
        # Use nearest-index lookup with a tight equality check to avoid a
        # floating-key mismatch from independently constructed linspaces.
        def frozen_policy(tau_value: float, query_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Mapping[str, float]]:
            index = int(np.searchsorted(tau_grid, float(tau_value)))
            candidates = [idx for idx in (index - 1, index) if 0 <= idx < n_times]
            if not candidates:
                raise KeyError(f"tau={tau_value} is outside the precomputed policy grid")
            selected = min(candidates, key=lambda idx: abs(float(tau_grid[idx]) - float(tau_value)))
            if not math.isclose(float(tau_grid[selected]), float(tau_value), rel_tol=0.0, abs_tol=1e-11):
                raise KeyError(f"tau={tau_value} is not a precomputed frozen-policy node")
            query = np.asarray(query_y, dtype=np.float64).reshape(-1)
            if query.shape != y_grid.shape or not np.allclose(query, y_grid, rtol=0.0, atol=1e-12):
                raise ValueError("FD solver requested a y grid different from the frozen-policy grid")
            diag = {"points": float(n_y)}
            diag.update({key: float(values[selected]) for key, values in counts.items()})
            selected_portfolio = np.asarray(portfolio_grid[selected], dtype=np.float64)
            selected_variance = np.einsum(
                "bi,ij,bj->b", selected_portfolio, self.run.problem.sigma,
                selected_portfolio, optimize=True,
            )
            diag["min_diffusion_variance"] = float(np.min(selected_variance))
            diag["max_diffusion_variance"] = float(np.max(selected_variance))
            return consumption_grid[selected], portfolio_grid[selected], diag

        total_points = float(n_times * n_y)
        aggregate_diag = {
            key: float(values.sum()) / total_points for key, values in counts.items()
        }
        variance = np.einsum(
            "tbi,ij,tbj->tb",
            portfolio_grid,
            self.run.problem.sigma,
            portfolio_grid,
            optimize=True,
        )
        aggregate_diag["min_diffusion_variance"] = float(np.min(variance))
        aggregate_diag["max_diffusion_variance"] = float(np.max(variance))
        return frozen_policy, digest.hexdigest(), aggregate_diag

    def policy_diagnostics_on_grid(
        self,
        tau: np.ndarray,
        y: np.ndarray,
        *,
        policy_extension: str = "boundary-projection",
    ) -> Dict[str, float]:
        _policy, _digest, diagnostics = self.precompute_policy(
            tau, y, policy_extension=policy_extension
        )
        return diagnostics


def _fraction(diag: Mapping[str, float], key: str) -> float:
    return float(diag.get(f"policy_{key}", diag.get(key, 0.0)))


def _clip_summary(diag: Mapping[str, float]) -> Dict[str, float]:
    return {
        "vw": _fraction(diag, "vw_guard_count"),
        "numerator": _fraction(diag, "numerator_guard_count"),
        "denom": _fraction(diag, "denominator_guard_count"),
        "positive": _fraction(diag, "positive_curvature_count"),
        "kappa": _fraction(diag, "kappa_clip_count"),
        "consumption": _fraction(diag, "consumption_clip_count"),
        "portfolio_any": _fraction(diag, "portfolio_any_clip_count"),
        "portfolio_component": _fraction(diag, "portfolio_component_clip_count"),
    }


def _outside_summary(diag: Mapping[str, float]) -> Tuple[int, float]:
    fraction = float(diag.get("policy_outside_collocation_count", 0.0))
    points = float(diag.get("policy_points", 0.0))
    count = int(round(fraction * points))
    if fraction < 0.0 or fraction > 1.0 + 1e-12 or count < 0:
        raise ValueError(
            "invalid outside-collocation diagnostic "
            f"(count={count}, fraction={fraction}, points={points})"
        )
    return count, min(1.0, fraction)


def _map_variant(fd_diag: Mapping[str, float], ev_diag: Mapping[str, float]) -> Tuple[str, int]:
    fd = _clip_summary(fd_diag)
    ev = _clip_summary(ev_diag)
    nonconcave = fd["positive"] > 0.0 or ev["positive"] > 0.0
    active_keys = (
        "vw", "numerator", "denom", "kappa", "consumption",
        "portfolio_any", "portfolio_component",
    )
    modified = any(fd[key] > 0.0 or ev[key] > 0.0 for key in active_keys)
    if modified:
        return "sampled_guarded_clipped", 0
    if nonconcave:
        return "sampled_nonconcave_source_policy", 0
    return "locally_unmodified_on_sampled_xfd", 1


def build_initial_policy(
    run: RunSpec,
    *,
    policy_extension: str = "boundary-projection",
) -> Tuple[Any, str]:
    """Reconstruct the trainer's deterministic outer-zero policy exactly.

    Random initialization is intentionally rejected: the historical trainer
    sampled it independently at each call, so it is not a reproducible
    state-feedback function on the FD grid.  Paper runs use the deterministic
    myopic/proportional contract.
    """
    pi_method = str(run.training_protocol.get("pi_init_method", "myopic"))
    c_method = str(run.training_protocol.get("c_init_method", "proportional"))
    pi_scale = float(run.training_protocol.get("pi_init_scale", 1.0))
    if pi_method not in {"myopic", "zero"}:
        raise ValueError(
            "E4 delta_0 requires deterministic pi_init_method=myopic or zero; "
            f"got {pi_method!r}"
        )
    if c_method not in {"proportional", "zero"}:
        raise ValueError(
            "E4 delta_0 requires deterministic c_init_method=proportional or zero; "
            f"got {c_method!r}"
        )
    if not math.isfinite(pi_scale) or pi_scale <= 0.0:
        raise ValueError("pi_init_scale must be positive and finite")

    storage_dtype = (
        np.float32 if str(run.network.dtype).lower() == "float32" else np.float64
    )
    if pi_method == "myopic":
        pi_star_storage = np.asarray(
            run.problem.sigma_inv_mu / run.problem.gamma,
            dtype=storage_dtype,
        )
        portfolio_template = np.asarray(
            storage_dtype(pi_scale) * pi_star_storage, dtype=storage_dtype
        )
    else:
        portfolio_template = np.zeros(run.problem.n_assets, dtype=storage_dtype)
    policy_hash = stable_hash({
        "kind": "trainer-initial-policy",
        "policy_extension": policy_extension,
        "pi_init_method": pi_method,
        "pi_init_scale": pi_scale,
        "c_init_method": c_method,
        "discount": run.problem.discount,
        "portfolio_template": portfolio_template.tolist(),
        "policy_bounds": asdict(run.policy),
    })

    def policy(
        _tau: float,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Mapping[str, float]]:
        actual_y = np.asarray(y, dtype=storage_dtype).reshape(-1)
        evaluated_y, outside = _policy_extension_y(
            run.problem, actual_y, policy_extension
        )
        evaluated_y = np.asarray(evaluated_y, dtype=storage_dtype)
        boundary_wealth = np.exp(evaluated_y)
        actual_wealth = np.exp(actual_y)
        n = int(actual_y.size)
        portfolio_raw = np.broadcast_to(
            portfolio_template.reshape(1, -1),
            (n, run.problem.n_assets),
        ).copy()
        portfolio = portfolio_raw.copy()
        if run.policy.portfolio_min is not None:
            portfolio = np.maximum(portfolio, float(run.policy.portfolio_min))
        if run.policy.portfolio_max is not None:
            portfolio = np.minimum(portfolio, float(run.policy.portfolio_max))
        portfolio_component_clipped = portfolio != portfolio_raw

        if c_method == "proportional":
            consumption_raw = (
                storage_dtype(run.problem.discount) * boundary_wealth
            )
        else:
            floor = (
                1e-8
                if run.policy.consumption_min is None
                else float(run.policy.consumption_min)
            )
            consumption_raw = np.full(n, floor, dtype=storage_dtype)
        kappa_raw = consumption_raw / boundary_wealth
        kappa = kappa_raw.copy()
        if run.policy.kappa_min is not None:
            kappa = np.maximum(kappa, float(run.policy.kappa_min))
        if run.policy.kappa_max is not None:
            kappa = np.minimum(kappa, float(run.policy.kappa_max))
        boundary_consumption_raw = kappa * boundary_wealth
        boundary_consumption = boundary_consumption_raw.copy()
        if run.policy.consumption_min is not None:
            boundary_consumption = np.maximum(
                boundary_consumption, float(run.policy.consumption_min)
            )
        if run.policy.consumption_max is not None:
            boundary_consumption = np.minimum(
                boundary_consumption, float(run.policy.consumption_max)
            )
        boundary_level_clipped = (
            boundary_consumption != boundary_consumption_raw
        )
        if (
            str(policy_extension).strip().lower().replace("_", "-")
            == "boundary-projection"
        ):
            consumption_extended = (
                boundary_consumption / boundary_wealth * actual_wealth
            )
        else:
            consumption_extended = boundary_consumption
        consumption = consumption_extended.copy()
        if run.policy.consumption_min is not None:
            consumption = np.maximum(
                consumption, float(run.policy.consumption_min)
            )
        if run.policy.consumption_max is not None:
            consumption = np.minimum(
                consumption, float(run.policy.consumption_max)
            )
        extension_level_clipped = consumption != consumption_extended

        kappa_clipped = kappa != kappa_raw
        consumption_clipped = boundary_level_clipped | extension_level_clipped
        portfolio_any_clipped = np.any(portfolio_component_clipped, axis=1)
        return consumption, portfolio, {
            "points": float(n),
            "vw_guard_count": 0.0,
            "numerator_guard_count": 0.0,
            "denominator_guard_count": 0.0,
            "positive_curvature_count": 0.0,
            "kappa_clip_count": float(np.count_nonzero(kappa_clipped)),
            "consumption_clip_count": float(
                np.count_nonzero(consumption_clipped)
            ),
            "portfolio_any_clip_count": float(
                np.count_nonzero(portfolio_any_clipped)
            ),
            # FDDiagnostics divides this by the number of points, yielding
            # the fraction of clipped asset components.
            "portfolio_component_clip_count": float(
                np.count_nonzero(portfolio_component_clipped)
            ) / float(run.problem.n_assets),
            "outside_collocation_count": float(np.count_nonzero(outside)),
        }

    return policy, policy_hash


def policy_diagnostics_on_tensor_grid(
    policy: Any,
    problem: MertonProblem,
    tau: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    points = 0.0
    min_variance = float("inf")
    max_variance = 0.0
    for tau_value in np.asarray(tau, dtype=np.float64).reshape(-1):
        _c, portfolio, diagnostics = policy(float(tau_value), y)
        row_points = float(diagnostics.get("points", 0.0))
        points += row_points
        for key, value in diagnostics.items():
            if key != "points":
                sums[key] = sums.get(key, 0.0) + float(value)
        variance = np.einsum(
            "bi,ij,bj->b", portfolio, problem.sigma, portfolio, optimize=True
        )
        min_variance = min(min_variance, float(np.min(variance)))
        max_variance = max(max_variance, float(np.max(variance)))
    denominator = points if points > 0.0 else 1.0
    result = {
        f"policy_{key}": value / denominator for key, value in sums.items()
    }
    result.update({
        "policy_points": points,
        "min_diffusion_variance": min_variance,
        "max_diffusion_variance": max_variance,
    })
    return result


def select_verification_iterations(checkpoints: Sequence[Tuple[int, Path]], spec: str) -> set[int]:
    outers = [outer for outer, _path in checkpoints]
    text = str(spec).strip().lower()
    if text == "all":
        return set(outers)
    if text in {"none", ""}:
        return set()
    selected: set[int] = set()
    for token in re.split(r"[\s,]+", text):
        if token == "first":
            selected.add(outers[0])
        elif token == "middle":
            selected.add(outers[len(outers) // 2])
        elif token == "last":
            selected.add(outers[-1])
        elif token:
            value = int(token)
            if value not in outers:
                raise ValueError(f"verification checkpoint {value} is not available")
            selected.add(value)
    return selected


def _variant_key(row: Mapping[str, Any]) -> Tuple[int, float, str]:
    return int(row["grid_factor"]), float(row["fd_margin"]), str(row["boundary"])


def assess_refinement(
    rows: List[Dict[str, Any]],
    *,
    grid_factors: Sequence[int],
    fd_margins: Sequence[float],
    boundaries: Sequence[str],
    abs_tolerance: float,
    rel_tolerance: float,
) -> None:
    """Assess grid/domain convergence within each boundary closure.

    Replacing the boundary closure changes the finite-domain BVP, so that
    comparison is reported as a sensitivity diagnostic and is deliberately
    excluded from the binary ``refinement_status`` gate.  For backward
    compatibility, ``refinement_status`` and ``rho_sensitivity_envelope`` on
    the unique primary row remain the canonical fields; their v2 semantics
    are explicitly grid/domain-only.
    """
    by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_iter.setdefault(int(row["source_iter"]), []).append(row)
    finest = max(grid_factors)
    largest_domain_margin = min(fd_margins)
    primary_boundary = boundaries[0]
    coarse_factors = sorted({value for value in grid_factors if value < finest})
    smaller_domain_margins = sorted(
        {value for value in fd_margins if value > largest_domain_margin}
    )
    audit_boundaries = [value for value in boundaries if value != primary_boundary]

    for outer_rows in by_iter.values():
        lookup = {_variant_key(row): row for row in outer_rows}
        primary = lookup[(finest, largest_domain_margin, primary_boundary)]
        for row in outer_rows:
            boundary = str(row["boundary"])
            is_boundary_representative = (
                int(row["grid_factor"]) == finest
                and float(row["fd_margin"]) == largest_domain_margin
            )
            row["boundary_role"] = (
                "primary" if boundary == primary_boundary else "sensitivity"
            )
            row["is_boundary_representative"] = int(
                is_boundary_representative
            )
            row["refinement_semantics_version"] = (
                REFINEMENT_SEMANTICS_VERSION
            )
            row.setdefault("grid_abs_change", "")
            row.setdefault("grid_rel_change", "")
            row.setdefault("domain_abs_change", "")
            row.setdefault("domain_rel_change", "")
            row.setdefault("boundary_abs_change", "")
            row.setdefault("boundary_rel_change", "")
            row.setdefault("boundary_sensitivity_status", "variant")
            row.setdefault("rho_grid_domain_sensitivity_envelope", "")
            row.setdefault("rho_sensitivity_envelope", "")
            row.setdefault("grid_domain_refinement_status", "variant")
            row.setdefault("refinement_status", "variant")
            row.setdefault("contraction_status", "variant")

        representatives: Dict[str, Dict[str, Any]] = {}
        for boundary in boundaries:
            representative = lookup.get(
                (finest, largest_domain_margin, boundary)
            )
            if representative is None:
                continue
            representatives[boundary] = representative
            rho = float(representative["rho_exact"])
            if not math.isfinite(rho):
                representative.update({
                    "grid_abs_change": float("nan"),
                    "grid_rel_change": float("nan"),
                    "domain_abs_change": float("nan"),
                    "domain_rel_change": float("nan"),
                    "rho_grid_domain_sensitivity_envelope": float("nan"),
                    "grid_domain_refinement_status": (
                        "undefined_denominator"
                    ),
                })
                if representative is primary:
                    representative["rho_sensitivity_envelope"] = float("nan")
                    representative["refinement_status"] = (
                        "undefined_denominator"
                    )
                    representative["contraction_status"] = "undefined"
                continue

            comparisons: Dict[str, List[Optional[Dict[str, Any]]]] = {
                "grid": [
                    lookup.get((factor, largest_domain_margin, boundary))
                    for factor in coarse_factors
                ],
                "domain": [
                    lookup.get((finest, margin, boundary))
                    for margin in smaller_domain_margins
                ],
            }
            checked = True
            passed = True
            axis_deltas: List[float] = []
            changes: Dict[str, float] = {}
            tolerance = abs_tolerance + rel_tolerance * abs(rho)
            for name, axis_rows in comparisons.items():
                if not axis_rows or any(
                    comparison is None
                    or not math.isfinite(float(comparison["rho_exact"]))
                    for comparison in axis_rows
                ):
                    checked = False
                    changes[name] = float("nan")
                    continue
                axis_changes = [
                    abs(rho - float(comparison["rho_exact"]))
                    for comparison in axis_rows
                    if comparison is not None
                ]
                change = max(axis_changes)
                changes[name] = change
                axis_deltas.append(change)
                passed = passed and all(
                    value <= tolerance for value in axis_changes
                )

            grid_change = changes.get("grid", float("nan"))
            domain_change = changes.get("domain", float("nan"))
            if not bool(int(representative["is_verification"])):
                status = "not_checked"
            elif largest_domain_margin >= 0.0:
                status = "fd_domain_not_enlarged_beyond_training"
            elif not checked:
                status = "incomplete"
            else:
                status = "pass" if passed else "fail"
            envelope = (
                rho + sum(axis_deltas) if checked else float("nan")
            )
            representative.update({
                "grid_abs_change": grid_change,
                "grid_rel_change": (
                    grid_change
                    / max(abs(rho), np.finfo(float).tiny)
                ),
                "domain_abs_change": domain_change,
                "domain_rel_change": (
                    domain_change
                    / max(abs(rho), np.finfo(float).tiny)
                ),
                "rho_grid_domain_sensitivity_envelope": envelope,
                "grid_domain_refinement_status": status,
            })
            if representative is primary:
                # Backward-compatible aliases.  Boundary replacement is not
                # included in either the envelope or the pass/fail status.
                representative["rho_sensitivity_envelope"] = envelope
                representative["refinement_status"] = status

        primary_rho = float(primary["rho_exact"])
        if not audit_boundaries:
            primary["boundary_abs_change"] = float("nan")
            primary["boundary_rel_change"] = float("nan")
            primary["boundary_sensitivity_status"] = "not_requested"
        elif not bool(int(primary["is_verification"])):
            primary["boundary_abs_change"] = float("nan")
            primary["boundary_rel_change"] = float("nan")
            primary["boundary_sensitivity_status"] = "not_checked"
        else:
            audit_representatives = [
                representatives.get(boundary)
                for boundary in audit_boundaries
            ]
            boundary_complete = (
                math.isfinite(primary_rho)
                and all(
                    row is not None
                    and math.isfinite(float(row["rho_exact"]))
                    for row in audit_representatives
                )
            )
            if boundary_complete:
                boundary_deltas: List[float] = []
                for boundary, audit in zip(
                    audit_boundaries, audit_representatives
                ):
                    if audit is None:  # pragma: no cover - guarded above
                        continue
                    change = abs(
                        float(audit["rho_exact"]) - primary_rho
                    )
                    boundary_deltas.append(change)
                    audit["boundary_abs_change"] = change
                    audit["boundary_rel_change"] = (
                        change
                        / max(abs(primary_rho), np.finfo(float).tiny)
                    )
                    audit["boundary_sensitivity_status"] = (
                        f"reported_vs_{primary_boundary}"
                    )
                primary_change = max(boundary_deltas)
                primary["boundary_abs_change"] = primary_change
                primary["boundary_rel_change"] = (
                    primary_change
                    / max(abs(primary_rho), np.finfo(float).tiny)
                )
                primary["boundary_sensitivity_status"] = "reported"
            else:
                primary["boundary_abs_change"] = float("nan")
                primary["boundary_rel_change"] = float("nan")
                primary["boundary_sensitivity_status"] = "incomplete"

        map_unmodified = bool(int(primary["local_map_unmodified_on_xfd"]))
        status_prefix = "" if map_unmodified else "sampled_modified_map_"
        if not math.isfinite(primary_rho):
            primary["contraction_status"] = "undefined"
        elif primary["refinement_status"] == "pass":
            primary["contraction_status"] = (
                status_prefix + "sensitivity_stable_below_one"
                if float(primary["rho_sensitivity_envelope"]) < 1.0
                else status_prefix + "sensitivity_envelope_crosses_one"
            )
        elif primary_rho < 1.0:
            primary["contraction_status"] = (
                status_prefix + "observed_below_one_without_full_sensitivity_pass"
            )
        else:
            primary["contraction_status"] = status_prefix + "not_contractive"


def assess_defect_refinement(
    rows: List[Dict[str, Any]],
    *,
    grid_factors: Sequence[int],
    fd_margins: Sequence[float],
    boundaries: Sequence[str],
    abs_tolerance: float,
    rel_tolerance: float,
) -> None:
    """Attach boundary-specific grid/domain audits to every E4 defect.

    The next neural bundle is fixed while the frozen-policy FD value is
    recomputed over the same grid/domain/boundary variants used by the exact
    map audit.  Hence every reported change is a change in ``delta_X`` itself,
    rather than a proxy copied from the contraction-ratio refinement table.
    Boundary replacement changes the BVP and is therefore reported separately
    rather than entering the primary grid/domain refinement gate.
    """
    by_defect: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_defect.setdefault(int(row["defect_iter"]), []).append(row)
    finest = max(grid_factors)
    largest_domain_margin = min(fd_margins)
    primary_boundary = boundaries[0]
    coarse_factors = sorted({value for value in grid_factors if value < finest})
    smaller_domain_margins = sorted(
        {value for value in fd_margins if value > largest_domain_margin}
    )
    audit_boundaries = [
        value for value in boundaries if value != primary_boundary
    ]

    for defect_rows in by_defect.values():
        lookup = {_variant_key(row): row for row in defect_rows}
        primary = lookup[(finest, largest_domain_margin, primary_boundary)]
        for row in defect_rows:
            boundary = str(row["boundary"])
            is_boundary_representative = (
                int(row["grid_factor"]) == finest
                and float(row["fd_margin"]) == largest_domain_margin
            )
            row["boundary_role"] = (
                "primary" if boundary == primary_boundary else "sensitivity"
            )
            row["is_boundary_representative"] = int(
                is_boundary_representative
            )
            row["refinement_semantics_version"] = (
                REFINEMENT_SEMANTICS_VERSION
            )
            row.setdefault("defect_grid_abs_change", "")
            row.setdefault("defect_grid_rel_change", "")
            row.setdefault("defect_domain_abs_change", "")
            row.setdefault("defect_domain_rel_change", "")
            row.setdefault("defect_boundary_abs_change", "")
            row.setdefault("defect_boundary_rel_change", "")
            row.setdefault("boundary_sensitivity_status", "variant")
            row.setdefault(
                "defect_grid_domain_sensitivity_envelope", ""
            )
            row.setdefault("defect_sensitivity_envelope", "")
            row.setdefault("grid_domain_refinement_status", "variant")
            row.setdefault("refinement_status", "variant")

        representatives: Dict[str, Dict[str, Any]] = {}
        for boundary in boundaries:
            representative = lookup.get(
                (finest, largest_domain_margin, boundary)
            )
            if representative is None:
                continue
            representatives[boundary] = representative
            delta = float(representative["delta_X"])
            if not math.isfinite(delta):
                representative.update({
                    "defect_grid_abs_change": float("nan"),
                    "defect_grid_rel_change": float("nan"),
                    "defect_domain_abs_change": float("nan"),
                    "defect_domain_rel_change": float("nan"),
                    "defect_grid_domain_sensitivity_envelope": (
                        float("nan")
                    ),
                    "grid_domain_refinement_status": "undefined_defect",
                })
                if representative is primary:
                    representative["defect_sensitivity_envelope"] = (
                        float("nan")
                    )
                    representative["refinement_status"] = (
                        "undefined_defect"
                    )
                continue

            comparisons: Dict[str, List[Optional[Dict[str, Any]]]] = {
                "grid": [
                    lookup.get(
                        (factor, largest_domain_margin, boundary)
                    )
                    for factor in coarse_factors
                ],
                "domain": [
                    lookup.get((finest, margin, boundary))
                    for margin in smaller_domain_margins
                ],
            }
            checked = True
            passed = True
            axis_deltas: List[float] = []
            changes: Dict[str, float] = {}
            tolerance = abs_tolerance + rel_tolerance * abs(delta)
            for name, axis_rows in comparisons.items():
                if not axis_rows or any(
                    comparison is None
                    or not math.isfinite(float(comparison["delta_X"]))
                    for comparison in axis_rows
                ):
                    checked = False
                    changes[name] = float("nan")
                    continue
                axis_changes = [
                    abs(delta - float(comparison["delta_X"]))
                    for comparison in axis_rows
                    if comparison is not None
                ]
                change = max(axis_changes)
                changes[name] = change
                axis_deltas.append(change)
                passed = passed and all(
                    value <= tolerance for value in axis_changes
                )

            grid_change = changes.get("grid", float("nan"))
            domain_change = changes.get("domain", float("nan"))
            if not bool(int(representative["is_verification"])):
                status = "not_checked"
            elif largest_domain_margin >= 0.0:
                status = "fd_domain_not_enlarged_beyond_training"
            elif not checked:
                status = "incomplete"
            else:
                status = "pass" if passed else "fail"
            envelope = (
                delta + sum(axis_deltas) if checked else float("nan")
            )
            representative.update({
                "defect_grid_abs_change": grid_change,
                "defect_grid_rel_change": (
                    grid_change
                    / max(abs(delta), np.finfo(float).tiny)
                ),
                "defect_domain_abs_change": domain_change,
                "defect_domain_rel_change": (
                    domain_change
                    / max(abs(delta), np.finfo(float).tiny)
                ),
                "defect_grid_domain_sensitivity_envelope": envelope,
                "grid_domain_refinement_status": status,
            })
            if representative is primary:
                # Backward-compatible aliases with grid/domain-only v2
                # semantics.
                representative["defect_sensitivity_envelope"] = envelope
                representative["refinement_status"] = status

        primary_delta = float(primary["delta_X"])
        if not audit_boundaries:
            primary["defect_boundary_abs_change"] = float("nan")
            primary["defect_boundary_rel_change"] = float("nan")
            primary["boundary_sensitivity_status"] = "not_requested"
        elif not bool(int(primary["is_verification"])):
            primary["defect_boundary_abs_change"] = float("nan")
            primary["defect_boundary_rel_change"] = float("nan")
            primary["boundary_sensitivity_status"] = "not_checked"
        else:
            audit_representatives = [
                representatives.get(boundary)
                for boundary in audit_boundaries
            ]
            boundary_complete = (
                math.isfinite(primary_delta)
                and all(
                    row is not None
                    and math.isfinite(float(row["delta_X"]))
                    for row in audit_representatives
                )
            )
            if boundary_complete:
                boundary_deltas: List[float] = []
                for boundary, audit in zip(
                    audit_boundaries, audit_representatives
                ):
                    if audit is None:  # pragma: no cover - guarded above
                        continue
                    change = abs(
                        float(audit["delta_X"]) - primary_delta
                    )
                    boundary_deltas.append(change)
                    audit["defect_boundary_abs_change"] = change
                    audit["defect_boundary_rel_change"] = (
                        change
                        / max(abs(primary_delta), np.finfo(float).tiny)
                    )
                    audit["boundary_sensitivity_status"] = (
                        f"reported_vs_{primary_boundary}"
                    )
                primary_change = max(boundary_deltas)
                primary["defect_boundary_abs_change"] = primary_change
                primary["defect_boundary_rel_change"] = (
                    primary_change
                    / max(abs(primary_delta), np.finfo(float).tiny)
                )
                primary["boundary_sensitivity_status"] = "reported"
            else:
                primary["defect_boundary_abs_change"] = float("nan")
                primary["defect_boundary_rel_change"] = float("nan")
                primary["boundary_sensitivity_status"] = "incomplete"


def required_defect_refinement_iterations(
    defect_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Return the minimum E4 evidence set: first/last defects and the worst.

    Standard trajectories include delta_0, while E6 target branches begin at
    delta_1 after their common warm-up.  The same data-driven selector supports
    both without inventing an unavailable iteration.
    """
    if not defect_rows:
        return []
    by_iter: Dict[int, float] = {}
    for row in defect_rows:
        defect_iter = int(row["defect_iter"])
        delta = float(row["delta_X"])
        if not math.isfinite(delta) or delta < 0.0:
            raise ValueError(
                f"invalid delta_X={delta!r} for defect_iter={defect_iter}"
            )
        if defect_iter in by_iter:
            raise ValueError(f"duplicate primary defect_iter={defect_iter}")
        by_iter[defect_iter] = delta
    adjacent = sorted(value for value in by_iter if value > 0)
    required = {0} if 0 in by_iter else set()
    if adjacent:
        required.update((adjacent[0], adjacent[-1]))
    worst = max(by_iter, key=lambda value: (by_iter[value], -value))
    required.add(worst)
    return sorted(required)


def summarize_defect_refinement(
    defect_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize whether the minimum E4 FD evidence set is actually verified."""
    required = required_defect_refinement_iterations(defect_rows)
    by_iter = {
        int(row["defect_iter"]): str(row.get("refinement_status", ""))
        for row in defect_rows
    }
    statuses = {value: by_iter.get(value, "missing") for value in required}
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


def evaluate_run(
    run: RunSpec,
    output: Path,
    *,
    device: str,
    base_ny: int,
    base_nt: int,
    eval_ny: int,
    grid_factors: Sequence[int],
    fd_margins: Sequence[float],
    boundaries: Sequence[str],
    verify_checkpoints: str,
    drift_scheme: str,
    peclet_limit: float,
    theta_method: float,
    rannacher_steps: int,
    denominator_tolerance: float,
    refinement_abs_tolerance: float,
    refinement_rel_tolerance: float,
    policy_extension: str = "boundary-projection",
    linear_residual_tolerance: float = 1.0e-8,
    boundary_condition_limit: float = 1.0e12,
    solver_ellipticity_tolerance: float = 0.0,
    skip_e4: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _quarantine_previous_outputs(output)
    (output / "_INCOMPLETE_EXACT_MAP").touch()

    if any(value < 1 for value in grid_factors) or len(set(grid_factors)) != len(grid_factors):
        raise ValueError("grid factors must be unique positive integers")
    if any(not math.isfinite(value) or not value < run.eval_margin for value in fd_margins):
        raise ValueError(
            "every FD margin must be finite and smaller than the primary eval margin; "
            "negative margins enlarge X_FD beyond the training window"
        )
    if len(set(fd_margins)) != len(fd_margins):
        raise ValueError("fd margins must be unique")
    if (
        not math.isfinite(linear_residual_tolerance)
        or linear_residual_tolerance <= 0.0
    ):
        raise ValueError("linear residual tolerance must be finite and positive")
    if (
        not math.isfinite(boundary_condition_limit)
        or boundary_condition_limit < 1.0
    ):
        raise ValueError("boundary condition limit must be finite and at least one")
    if (
        not math.isfinite(solver_ellipticity_tolerance)
        or solver_ellipticity_tolerance < 0.0
    ):
        raise ValueError("solver ellipticity tolerance must be finite and nonnegative")
    policy_extension = str(policy_extension).strip().lower().replace("_", "-")
    if policy_extension not in POLICY_EXTENSIONS:
        raise ValueError(f"unsupported policy extension {policy_extension!r}")
    if not skip_e4 and run.checkpoint_selection != "all":
        selected = [int(outer) for outer, _path in run.checkpoints]
        if selected != list(range(1, selected[-1] + 1)):
            raise ValueError(
                "sparse checkpoint subsets require --skip-e4; E4 needs a "
                "contiguous prefix beginning at outer 1"
            )
    if skip_e4 and run.checkpoint_selection == "all":
        analysis_mode = "exact_map_only_pilot"
    elif skip_e4:
        analysis_mode = "sparse_exact_map_only_pilot"
    elif run.checkpoint_selection != "all":
        analysis_mode = "contiguous_prefix_exact_map_and_e4_pilot"
    else:
        analysis_mode = "exact_map_and_e4"
    map_definition, whole_space_claim = _map_definition(policy_extension)
    boundaries = [str(value).replace("_", "-").lower() for value in boundaries]
    if not boundaries or any(value not in {"robin", "exact-dirichlet"} for value in boundaries):
        raise ValueError("boundaries must contain robin and/or exact-dirichlet")
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("boundaries must be unique")
    if boundaries[0] != "robin":
        print(
            "[warning] the first boundary is the reported primary, but the "
            "paper protocol requires robin; exact-dirichlet injects optimal "
            "V* and is only a sensitivity audit",
            file=sys.stderr,
        )

    verification_outers = select_verification_iterations(
        run.checkpoints, verify_checkpoints
    )
    eval_y_min, eval_y_max = run.eval_y_bounds
    containment_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            abs(run.problem.y_min),
            abs(run.problem.y_max),
        )
    )
    for fd_margin in fd_margins:
        candidate_fd_y_min, candidate_fd_y_max = shrink_bounds(
            run.problem.y_min, run.problem.y_max, fd_margin
        )
        if (
            candidate_fd_y_min > eval_y_min + containment_tolerance
            or candidate_fd_y_max < eval_y_max - containment_tolerance
        ):
            raise ValueError(
                "every FD domain must contain the complete effective Q_ev "
                "window; "
                f"fd_margin={fd_margin:.17g} gives "
                f"[{candidate_fd_y_min:.17g}, {candidate_fd_y_max:.17g}], "
                f"but Q_ev is [{eval_y_min:.17g}, {eval_y_max:.17g}]. "
                "Use a smaller/negative --fd-margins value."
            )
    eval_tau_min = run.problem.horizon / int(base_nt)
    eval_tau_max = run.problem.horizon
    implementation_hashes = {
        "driver": sha256_file(Path(__file__).resolve()),
        "core": sha256_file(Path(__file__).with_name("merton_exact_map_core.py").resolve()),
        "policy": sha256_file(Path(__file__).with_name("merton_policy.py").resolve()),
    }
    protocol_compatibility_payload = {
        "compatibility_version": EXACT_PROTOCOL_COMPATIBILITY_VERSION,
        "training_group": run.group,
        "base_ny": base_ny,
        "base_nt": base_nt,
        "eval_ny": eval_ny,
        "evaluation_calendar_time_domain": "[0,T)",
        "primary_evaluation_window": {
            "eval_margin": run.eval_margin,
            "eval_window_mode": run.eval_window_mode,
            "eval_w_min_requested": run.eval_w_min_requested,
            "eval_w_min_symmetric": run.eval_w_min_symmetric,
            "ev_tau_min": eval_tau_min,
            "ev_tau_max": eval_tau_max,
            "ev_y_min": eval_y_min,
            "ev_y_max": eval_y_max,
            "ev_w_min": run.eval_w_bounds[0],
            "ev_w_max": run.eval_w_bounds[1],
        },
        "grid_factors": list(grid_factors),
        "fd_margins": list(fd_margins),
        "boundaries": list(boundaries),
        "verify_checkpoints": verify_checkpoints,
        "analysis_mode": analysis_mode,
        "skip_e4": bool(skip_e4),
        "e6_role": run.e6_role,
        "checkpoint_indexing": run.checkpoint_indexing_provenance,
        "initial_defect_mode": run.initial_defect_mode,
        "policy_extension": policy_extension,
        "defect_refinement_selector": (
            "first_defect_plus_verify_checkpoints; paper evidence requires "
            "first+last+worst defects"
        ),
        "drift_scheme": drift_scheme,
        "peclet_limit": peclet_limit,
        "theta_method": theta_method,
        "rannacher_steps": rannacher_steps,
        "denominator_tolerance": denominator_tolerance,
        "linear_residual_tolerance": linear_residual_tolerance,
        "boundary_condition_limit": boundary_condition_limit,
        "solver_ellipticity_tolerance": solver_ellipticity_tolerance,
        "refinement_abs_tolerance": refinement_abs_tolerance,
        "refinement_rel_tolerance": refinement_rel_tolerance,
        "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
        "refinement_gate_axes": ["grid", "domain"],
        "boundary_replacement_role": (
            "separate_sensitivity_not_refinement_gate"
        ),
        "implementation_hashes": implementation_hashes,
        "checkpoint_selection": run.checkpoint_selection,
    }
    protocol_hash = stable_hash(protocol_compatibility_payload)[:16]
    finest = max(grid_factors)
    largest_domain_margin = min(fd_margins)
    primary_boundary = boundaries[0]
    eval_y = np.linspace(eval_y_min, eval_y_max, int(eval_ny))
    # Q_ev follows the trainer's [0,T) calendar-time convention.  In
    # remaining time this is (0,T], so exclude tau=0 (the terminal face) and
    # keep tau=T (calendar time zero).  These nodes remain nested in every
    # factor-refined FD grid.
    tau_eval = np.linspace(0.0, run.problem.horizon, int(base_nt) + 1)[1:]
    tt_eval, yy_eval = np.meshgrid(tau_eval, eval_y, indexing="ij")
    closed_form = crra_closed_form(run.problem, tt_eval, yy_eval)
    training_y_width = run.problem.y_max - run.problem.y_min
    rows: List[Dict[str, Any]] = []
    defect_rows: List[Dict[str, Any]] = []
    defect_refinement_rows: List[Dict[str, Any]] = []
    residual_by_outer, residual_semantics = load_outer_residuals(run.run_dir)
    checkpoint_by_outer = dict(run.checkpoints)
    checkpoint_source_to_outer: Dict[int, int] = {}
    for checkpoint_outer, _checkpoint_path in run.checkpoints:
        source = int(run.checkpoint_source_iters[int(checkpoint_outer)])
        if source in checkpoint_source_to_outer:
            raise ValueError(
                f"duplicate paper source_iter={source} in checkpoint schedule"
            )
        checkpoint_source_to_outer[source] = int(checkpoint_outer)
    input_bundle_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Standard E4 starts at n=0: checkpoint outer 1 is v_tilde_0, while
    # alpha_0 is the configured analytic initialization (not G of a neural
    # checkpoint). E6 target branches use a different predecessor block below.
    # Delta_0 is always recomputed over the complete FD variant family so its
    # refinement status is based on the defect itself, not copied from an
    # unrelated contraction-ratio check.
    if (
        not skip_e4
        and run.initial_defect_mode == "analytic-initial-policy"
        and 1 in checkpoint_by_outer
    ):
        first_checkpoint = checkpoint_by_outer[1]
        first_evaluator = TorchCheckpointEvaluator(first_checkpoint, run, device)
        first_bundle = first_evaluator.bundle_on_tensor_grid(tau_eval, eval_y)
        input_bundle_cache[1] = first_bundle
        initial_policy, initial_policy_contract_hash = build_initial_policy(
            run, policy_extension=policy_extension
        )
        initial_ev_diag = policy_diagnostics_on_tensor_grid(
            initial_policy, run.problem, tau_eval, eval_y
        )
        initial_primary: Optional[Dict[str, Any]] = None
        for factor, fd_margin, boundary in [
            (factor, margin, boundary)
            for factor in sorted(grid_factors)
            for margin in sorted(fd_margins)
            for boundary in boundaries
        ]:
            initial_fd_y_min, initial_fd_y_max = shrink_bounds(
                run.problem.y_min, run.problem.y_max, fd_margin
            )
            initial_base_intervals = max(
                6,
                int(round(
                    (initial_fd_y_max - initial_fd_y_min)
                    / training_y_width
                    * (int(base_ny) - 1)
                )),
            )
            initial_ny = initial_base_intervals * int(factor) + 1
            initial_nt = int(base_nt) * int(factor)
            initial_grid = FDGrid(
                initial_fd_y_min,
                initial_fd_y_max,
                ny=initial_ny,
                nt=initial_nt,
            )
            initial_fd_y = np.linspace(
                initial_fd_y_min,
                initial_fd_y_max,
                initial_ny,
                dtype=np.float64,
            )
            initial_fd_dt = run.problem.horizon / initial_nt
            initial_fd_tau_mid = (
                np.arange(initial_nt, dtype=np.float64) + 0.5
            ) * initial_fd_dt
            initial_fd_c: List[np.ndarray] = []
            initial_fd_pi: List[np.ndarray] = []
            for tau_value in initial_fd_tau_mid:
                consumption_row, portfolio_row, _diag = initial_policy(
                    float(tau_value), initial_fd_y
                )
                initial_fd_c.append(np.asarray(consumption_row))
                initial_fd_pi.append(np.asarray(portfolio_row))
            initial_policy_hash = stable_hash({
                "contract_hash": initial_policy_contract_hash,
                "sampled_array_hash": canonical_array_hash({
                    "tau_mid": initial_fd_tau_mid,
                    "y": initial_fd_y,
                    "consumption": np.stack(initial_fd_c, axis=0),
                    "portfolio": np.stack(initial_fd_pi, axis=0),
                }),
            })
            initial_solution = solve_frozen_policy(
                run.problem,
                initial_policy,
                initial_grid,
                theta_method=theta_method,
                rannacher_steps=rannacher_steps,
                drift_scheme=drift_scheme,
                peclet_limit=peclet_limit,
                boundary=boundary,
                ellipticity_tolerance=solver_ellipticity_tolerance,
                linear_residual_tolerance=linear_residual_tolerance,
                boundary_condition_limit=boundary_condition_limit,
            )
            initial_map_bundle = evaluate_fd_bundle(
                initial_solution, tau_eval, eval_y
            )
            initial_delta = x_norm_components(
                *first_bundle, initial_map_bundle, yy_eval
            )
            initial_fd_diag = initial_solution.diagnostics.as_dict()
            (
                initial_outside_count,
                initial_outside_fraction,
            ) = _outside_summary(initial_fd_diag)
            initial_variant, initial_unmodified = _map_variant(
                initial_fd_diag, initial_ev_diag
            )
            is_primary_initial = (
                factor,
                fd_margin,
                boundary,
            ) == (finest, largest_domain_margin, primary_boundary)
            initial_refinement_row: Dict[str, Any] = {
                "problem": "merton",
                "group": run.group,
                "protocol_hash": protocol_hash,
                "model_type": "pipinn",
                "e6_role": run.e6_role,
                "n_assets": run.problem.n_assets,
                "seed": run.seed,
                "market_seed": run.market_seed,
                "eval_margin": run.eval_margin,
                "eval_window_mode": run.eval_window_mode,
                "eval_w_min_requested": run.eval_w_min_requested,
                "eval_w_min_symmetric": run.eval_w_min_symmetric,
                "ev_tau_min": float(tau_eval[0]),
                "ev_tau_max": float(tau_eval[-1]),
                "ev_y_min": float(eval_y[0]),
                "ev_y_max": float(eval_y[-1]),
                "ev_w_min": run.eval_w_bounds[0],
                "ev_w_max": run.eval_w_bounds[1],
                "defect_iter": 0,
                "defect_kind": "initial_policy_evaluation",
                "target_policy_iter": 0,
                "next_checkpoint_outer_iter": 1,
                "frozen_policy_sha256": initial_policy_hash,
                "fd_margin": fd_margin,
                "boundary": boundary,
                "grid_factor": factor,
                "ny": initial_ny,
                "nt": initial_nt,
                "is_primary": int(is_primary_initial),
                "is_verification": 1,
                "analysis_mode": analysis_mode,
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "outside_collocation_count_fd": initial_outside_count,
                "outside_collocation_fraction_fd": initial_outside_fraction,
                "delta_value_sup": initial_delta["value_sup"],
                "delta_vw_sup": initial_delta["vw_sup"],
                "delta_vww_sup": initial_delta["vww_sup"],
                "delta_vy_sup": initial_delta["vy_sup"],
                "delta_vyy_sup": initial_delta["vyy_sup"],
                "delta_bundle_sup": initial_delta["derivative_sup"],
                "delta_X": initial_delta["x_norm"],
                "map_variant": initial_variant,
                "local_map_unmodified_on_xfd": initial_unmodified,
                "whole_space_map_claim": whole_space_claim,
            }
            defect_refinement_rows.append(initial_refinement_row)
            if is_primary_initial:
                initial_primary = {
                    "map_bundle": initial_map_bundle,
                    "delta": initial_delta,
                    "variant": initial_variant,
                    "unmodified": initial_unmodified,
                    "policy_hash": initial_policy_hash,
                    "ny": initial_ny,
                    "nt": initial_nt,
                    "outside_count": initial_outside_count,
                    "outside_fraction": initial_outside_fraction,
                }
        if initial_primary is None:
            raise AssertionError("missing primary delta_0 FD variant")

        initial_c_rows: List[np.ndarray] = []
        initial_pi_rows: List[np.ndarray] = []
        for tau_value in tau_eval:
            consumption_row, portfolio_row, _diag = initial_policy(
                float(tau_value), eval_y
            )
            initial_c_rows.append(np.asarray(consumption_row))
            initial_pi_rows.append(np.asarray(portfolio_row))
        initial_bundle_path = (
            output / "evaluated_bundles" / "initial_policy_to_outer_0001.npz"
        )
        atomic_npz(
            initial_bundle_path,
            tau=tau_eval,
            y=eval_y,
            derivative_coordinate=np.asarray(
                "both_log_y_and_wealth_w", dtype="U32"
            ),
            initial_consumption=np.stack(initial_c_rows, axis=0),
            initial_portfolio=np.stack(initial_pi_rows, axis=0),
            **bundle_storage_arrays(
                "fd_map", initial_primary["map_bundle"], eval_y
            ),
            **bundle_storage_arrays("next_neural", first_bundle, eval_y),
            **bundle_storage_arrays("optimal", closed_form, eval_y),
        )
        defect_rows.append({
            "problem": "merton",
            "group": run.group,
            "protocol_hash": protocol_hash,
            "model_type": "pipinn",
            "e6_role": run.e6_role,
            "n_assets": run.problem.n_assets,
            "seed": run.seed,
            "market_seed": run.market_seed,
            "eval_margin": run.eval_margin,
            "eval_window_mode": run.eval_window_mode,
            "eval_w_min_requested": run.eval_w_min_requested,
            "eval_w_min_symmetric": run.eval_w_min_symmetric,
            "ev_tau_min": float(tau_eval[0]),
            "ev_tau_max": float(tau_eval[-1]),
            "ev_y_min": float(eval_y[0]),
            "ev_y_max": float(eval_y[-1]),
            "ev_w_min": run.eval_w_bounds[0],
            "ev_w_max": run.eval_w_bounds[1],
            "defect_iter": 0,
            "defect_kind": "initial_policy_evaluation",
            "checkpoint_outer_iter": 0,
            "source_iter": -1,
            "target_policy_iter": 0,
            "next_checkpoint_outer_iter": 1,
            "next_neural_iter": 0,
            "checkpoint_state_sha256": "",
            "next_checkpoint_state_sha256": canonical_checkpoint_state_hash(
                first_checkpoint
            ),
            "frozen_policy_sha256": initial_primary["policy_hash"],
            "fd_margin": largest_domain_margin,
            "boundary": primary_boundary,
            "grid_factor": finest,
            "ny": initial_primary["ny"],
            "nt": initial_primary["nt"],
            "is_verification": 1,
            "analysis_mode": analysis_mode,
            "policy_extension": policy_extension,
            "map_definition": map_definition,
            "outside_collocation_count_fd": (
                initial_primary.get("outside_count", 0)
            ),
            "outside_collocation_fraction_fd": (
                initial_primary.get("outside_fraction", 0.0)
            ),
            "delta_value_sup": initial_primary["delta"]["value_sup"],
            "delta_vw_sup": initial_primary["delta"]["vw_sup"],
            "delta_vww_sup": initial_primary["delta"]["vww_sup"],
            "delta_vy_sup": initial_primary["delta"]["vy_sup"],
            "delta_vyy_sup": initial_primary["delta"]["vyy_sup"],
            "delta_bundle_sup": initial_primary["delta"]["derivative_sup"],
            "delta_X": initial_primary["delta"]["x_norm"],
            "defect_grid_abs_change": "",
            "defect_grid_rel_change": "",
            "defect_domain_abs_change": "",
            "defect_domain_rel_change": "",
            "defect_boundary_abs_change": "",
            "defect_sensitivity_envelope": "",
            "p_res_post_restore": residual_by_outer.get(1, ""),
            "p_res_source": "",
            "residual_semantics": residual_semantics,
            "evaluated_bundle_path": str(
                initial_bundle_path.relative_to(output)
            ),
            "evaluated_bundle_sha256": sha256_file(initial_bundle_path),
            "refinement_status": "pending_defect_refinement",
            "map_variant": initial_primary["variant"],
            "local_map_unmodified_on_xfd": initial_primary["unmodified"],
            "whole_space_map_claim": whole_space_claim,
        })

    # E6 branches begin after one common warm-up policy evaluation.  Their
    # local checkpoint outer 1 is v~_1, while v~_0 lives only in the
    # hash-validated warm-start bundle.  Re-evaluate G(v~_0) independently by
    # FD and compare it with branch checkpoint 1 to obtain delta_1.  The
    # warm-up residual is intentionally excluded; branch local outer 1 supplies
    # the official post-restore p_res for this target-phase defect.
    first_e6_outer = checkpoint_source_to_outer.get(1)
    if (
        not skip_e4
        and run.initial_defect_mode == "warm-start-value"
        and first_e6_outer is not None
    ):
        if run.warm_start_bundle_path is None:
            raise AssertionError(
                "E6 warm-start predecessor mode has no warm-start bundle"
            )
        first_checkpoint = checkpoint_by_outer[first_e6_outer]
        predecessor_evaluator = TorchCheckpointEvaluator(
            run.warm_start_bundle_path, run, device
        )
        predecessor_bundle = predecessor_evaluator.bundle_on_tensor_grid(
            tau_eval, eval_y
        )
        first_evaluator = TorchCheckpointEvaluator(
            first_checkpoint, run, device
        )
        first_bundle = first_evaluator.bundle_on_tensor_grid(
            tau_eval, eval_y
        )
        input_bundle_cache[first_e6_outer] = first_bundle
        predecessor_ev_diag = predecessor_evaluator.policy_diagnostics_on_grid(
            tau_eval,
            eval_y,
            policy_extension=policy_extension,
        )
        predecessor_primary: Optional[Dict[str, Any]] = None
        cached_predecessor_policy_key: Optional[Tuple[int, float]] = None
        cached_predecessor_policy: Optional[Tuple[Any, str]] = None
        for factor, fd_margin, boundary in [
            (factor, margin, boundary)
            for factor in sorted(grid_factors)
            for margin in sorted(fd_margins)
            for boundary in boundaries
        ]:
            predecessor_fd_y_min, predecessor_fd_y_max = shrink_bounds(
                run.problem.y_min, run.problem.y_max, fd_margin
            )
            predecessor_base_intervals = max(
                6,
                int(round(
                    (predecessor_fd_y_max - predecessor_fd_y_min)
                    / training_y_width
                    * (int(base_ny) - 1)
                )),
            )
            predecessor_ny = (
                predecessor_base_intervals * int(factor) + 1
            )
            predecessor_nt = int(base_nt) * int(factor)
            predecessor_grid = FDGrid(
                predecessor_fd_y_min,
                predecessor_fd_y_max,
                ny=predecessor_ny,
                nt=predecessor_nt,
            )
            predecessor_policy_key = (int(factor), float(fd_margin))
            if (
                cached_predecessor_policy_key != predecessor_policy_key
                or cached_predecessor_policy is None
            ):
                predecessor_fd_y = np.linspace(
                    predecessor_fd_y_min,
                    predecessor_fd_y_max,
                    predecessor_ny,
                    dtype=np.float64,
                )
                predecessor_dt = run.problem.horizon / predecessor_nt
                predecessor_tau_mid = (
                    np.arange(predecessor_nt, dtype=np.float64) + 0.5
                ) * predecessor_dt
                (
                    predecessor_policy,
                    predecessor_policy_hash,
                    _precomputed_diag,
                ) = predecessor_evaluator.precompute_policy(
                    predecessor_tau_mid,
                    predecessor_fd_y,
                    policy_extension=policy_extension,
                )
                cached_predecessor_policy_key = predecessor_policy_key
                cached_predecessor_policy = (
                    predecessor_policy,
                    predecessor_policy_hash,
                )
            else:
                (
                    predecessor_policy,
                    predecessor_policy_hash,
                ) = cached_predecessor_policy
            predecessor_solution = solve_frozen_policy(
                run.problem,
                predecessor_policy,
                predecessor_grid,
                theta_method=theta_method,
                rannacher_steps=rannacher_steps,
                drift_scheme=drift_scheme,
                peclet_limit=peclet_limit,
                boundary=boundary,
                ellipticity_tolerance=solver_ellipticity_tolerance,
                linear_residual_tolerance=linear_residual_tolerance,
                boundary_condition_limit=boundary_condition_limit,
            )
            predecessor_map_bundle = evaluate_fd_bundle(
                predecessor_solution, tau_eval, eval_y
            )
            predecessor_delta = x_norm_components(
                *first_bundle,
                predecessor_map_bundle,
                yy_eval,
            )
            predecessor_fd_diag = predecessor_solution.diagnostics.as_dict()
            (
                predecessor_outside_count,
                predecessor_outside_fraction,
            ) = _outside_summary(predecessor_fd_diag)
            predecessor_variant, predecessor_unmodified = _map_variant(
                predecessor_fd_diag, predecessor_ev_diag
            )
            is_primary_predecessor = (
                factor,
                fd_margin,
                boundary,
            ) == (finest, largest_domain_margin, primary_boundary)
            defect_refinement_rows.append({
                "problem": "merton",
                "group": run.group,
                "protocol_hash": protocol_hash,
                "model_type": "pipinn",
                "e6_role": run.e6_role,
                "n_assets": run.problem.n_assets,
                "seed": run.seed,
                "market_seed": run.market_seed,
                "eval_margin": run.eval_margin,
                "eval_window_mode": run.eval_window_mode,
                "eval_w_min_requested": run.eval_w_min_requested,
                "eval_w_min_symmetric": run.eval_w_min_symmetric,
                "ev_tau_min": float(tau_eval[0]),
                "ev_tau_max": float(tau_eval[-1]),
                "ev_y_min": float(eval_y[0]),
                "ev_y_max": float(eval_y[-1]),
                "ev_w_min": run.eval_w_bounds[0],
                "ev_w_max": run.eval_w_bounds[1],
                "defect_iter": 1,
                "defect_kind": "e6_warm_start_policy_evaluation",
                "target_policy_iter": 1,
                "next_checkpoint_outer_iter": first_e6_outer,
                "frozen_policy_sha256": predecessor_policy_hash,
                "fd_margin": fd_margin,
                "boundary": boundary,
                "grid_factor": factor,
                "ny": predecessor_ny,
                "nt": predecessor_nt,
                "is_primary": int(is_primary_predecessor),
                "is_verification": 1,
                "analysis_mode": analysis_mode,
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "outside_collocation_count_fd": (
                    predecessor_outside_count
                ),
                "outside_collocation_fraction_fd": (
                    predecessor_outside_fraction
                ),
                "delta_value_sup": predecessor_delta["value_sup"],
                "delta_vw_sup": predecessor_delta["vw_sup"],
                "delta_vww_sup": predecessor_delta["vww_sup"],
                "delta_vy_sup": predecessor_delta["vy_sup"],
                "delta_vyy_sup": predecessor_delta["vyy_sup"],
                "delta_bundle_sup": predecessor_delta["derivative_sup"],
                "delta_X": predecessor_delta["x_norm"],
                "map_variant": predecessor_variant,
                "local_map_unmodified_on_xfd": predecessor_unmodified,
                "whole_space_map_claim": whole_space_claim,
            })
            if is_primary_predecessor:
                predecessor_primary = {
                    "map_bundle": predecessor_map_bundle,
                    "delta": predecessor_delta,
                    "variant": predecessor_variant,
                    "unmodified": predecessor_unmodified,
                    "policy_hash": predecessor_policy_hash,
                    "ny": predecessor_ny,
                    "nt": predecessor_nt,
                    "outside_count": predecessor_outside_count,
                    "outside_fraction": predecessor_outside_fraction,
                }
        if predecessor_primary is None:
            raise AssertionError("missing primary E6 delta_1 FD variant")

        predecessor_bundle_path = (
            output
            / "evaluated_bundles"
            / f"warm_start_to_outer_{first_e6_outer:04d}.npz"
        )
        atomic_npz(
            predecessor_bundle_path,
            tau=tau_eval,
            y=eval_y,
            derivative_coordinate=np.asarray(
                "both_log_y_and_wealth_w", dtype="U32"
            ),
            **bundle_storage_arrays(
                "input", predecessor_bundle, eval_y
            ),
            **bundle_storage_arrays(
                "fd_map", predecessor_primary["map_bundle"], eval_y
            ),
            **bundle_storage_arrays(
                "next_neural", first_bundle, eval_y
            ),
            **bundle_storage_arrays("optimal", closed_form, eval_y),
        )
        defect_rows.append({
            "problem": "merton",
            "group": run.group,
            "protocol_hash": protocol_hash,
            "model_type": "pipinn",
            "e6_role": run.e6_role,
            "n_assets": run.problem.n_assets,
            "seed": run.seed,
            "market_seed": run.market_seed,
            "eval_margin": run.eval_margin,
            "eval_window_mode": run.eval_window_mode,
            "eval_w_min_requested": run.eval_w_min_requested,
            "eval_w_min_symmetric": run.eval_w_min_symmetric,
            "ev_tau_min": float(tau_eval[0]),
            "ev_tau_max": float(tau_eval[-1]),
            "ev_y_min": float(eval_y[0]),
            "ev_y_max": float(eval_y[-1]),
            "ev_w_min": run.eval_w_bounds[0],
            "ev_w_max": run.eval_w_bounds[1],
            "defect_iter": 1,
            "defect_kind": "e6_warm_start_policy_evaluation",
            "checkpoint_outer_iter": 0,
            "source_iter": 0,
            "target_policy_iter": 1,
            "next_checkpoint_outer_iter": first_e6_outer,
            "next_neural_iter": 1,
            "checkpoint_state_sha256": (
                run.warm_start_model_state_hash
            ),
            "next_checkpoint_state_sha256": (
                canonical_checkpoint_state_hash(first_checkpoint)
            ),
            "frozen_policy_sha256": predecessor_primary["policy_hash"],
            "fd_margin": largest_domain_margin,
            "boundary": primary_boundary,
            "grid_factor": finest,
            "ny": predecessor_primary["ny"],
            "nt": predecessor_primary["nt"],
            "is_verification": 1,
            "analysis_mode": analysis_mode,
            "policy_extension": policy_extension,
            "map_definition": map_definition,
            "outside_collocation_count_fd": predecessor_primary[
                "outside_count"
            ],
            "outside_collocation_fraction_fd": predecessor_primary[
                "outside_fraction"
            ],
            "delta_value_sup": predecessor_primary["delta"]["value_sup"],
            "delta_vw_sup": predecessor_primary["delta"]["vw_sup"],
            "delta_vww_sup": predecessor_primary["delta"]["vww_sup"],
            "delta_vy_sup": predecessor_primary["delta"]["vy_sup"],
            "delta_vyy_sup": predecessor_primary["delta"]["vyy_sup"],
            "delta_bundle_sup": predecessor_primary["delta"][
                "derivative_sup"
            ],
            "delta_X": predecessor_primary["delta"]["x_norm"],
            "defect_grid_abs_change": "",
            "defect_grid_rel_change": "",
            "defect_domain_abs_change": "",
            "defect_domain_rel_change": "",
            "defect_boundary_abs_change": "",
            "defect_sensitivity_envelope": "",
            "p_res_post_restore": residual_by_outer.get(
                first_e6_outer, ""
            ),
            "p_res_source": "",
            "residual_semantics": residual_semantics,
            "evaluated_bundle_path": str(
                predecessor_bundle_path.relative_to(output)
            ),
            "evaluated_bundle_sha256": sha256_file(
                predecessor_bundle_path
            ),
            "refinement_status": "pending_defect_refinement",
            "map_variant": predecessor_primary["variant"],
            "local_map_unmodified_on_xfd": predecessor_primary[
                "unmodified"
            ],
            "whole_space_map_claim": whole_space_claim,
        })

    for checkpoint_index, (outer, checkpoint) in enumerate(run.checkpoints, start=1):
        source_iter = int(run.checkpoint_source_iters[int(outer)])
        target_policy_iter = int(
            run.checkpoint_target_policy_iters[int(outer)]
        )
        print(
            f"[exact-map] seed={run.seed} checkpoint={checkpoint_index}/{len(run.checkpoints)} "
            f"outer={outer} (v_{source_iter} -> policy {target_policy_iter}): {checkpoint.name}"
        )
        evaluator = TorchCheckpointEvaluator(checkpoint, run, device)
        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_state_hash = canonical_checkpoint_state_hash(checkpoint)
        input_bundle = input_bundle_cache.pop(int(outer), None)
        if input_bundle is None:
            input_bundle = evaluator.bundle_on_tensor_grid(tau_eval, eval_y)
        input_metric = x_norm_components(*input_bundle, closed_form, yy_eval)
        ev_policy_diag = evaluator.policy_diagnostics_on_grid(
            tau_eval, eval_y, policy_extension=policy_extension
        )
        is_verification = outer in verification_outers
        cached_policy_key: Optional[Tuple[int, float]] = None
        cached_policy: Optional[Tuple[Any, str]] = None
        next_outer = checkpoint_source_to_outer.get(source_iter + 1)
        next_checkpoint = checkpoint_by_outer.get(next_outer)
        next_source_iter: Optional[int] = None
        next_bundle: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = None
        if not skip_e4 and next_outer is not None and next_checkpoint is not None:
            next_source_iter = int(
                run.checkpoint_source_iters[int(next_outer)]
            )
            if next_source_iter != target_policy_iter:
                raise ValueError(
                    "checkpoint indexing is not adjacent in paper algorithm "
                    f"space: source={source_iter}, policy={target_policy_iter}, "
                    f"next_source={next_source_iter}"
                )
            if next_outer not in input_bundle_cache:
                next_evaluator = TorchCheckpointEvaluator(
                    next_checkpoint, run, device
                )
                input_bundle_cache[next_outer] = (
                    next_evaluator.bundle_on_tensor_grid(tau_eval, eval_y)
                )
            next_bundle = input_bundle_cache[next_outer]

        primary_variant = (finest, largest_domain_margin, primary_boundary)
        if is_verification:
            variants = [
                (factor, margin, boundary)
                for factor in sorted(grid_factors)
                for margin in sorted(fd_margins)
                for boundary in boundaries
            ]
        else:
            variants = [primary_variant]

        for factor, fd_margin, boundary in variants:
            fd_y_min, fd_y_max = shrink_bounds(
                run.problem.y_min, run.problem.y_max, fd_margin
            )
            # Keep dy fixed when changing only the FD-domain size; otherwise
            # boundary sensitivity would be confounded with grid refinement.
            base_intervals_here = max(
                6,
                int(round((fd_y_max - fd_y_min) / training_y_width * (int(base_ny) - 1))),
            )
            ny = base_intervals_here * int(factor) + 1
            nt = int(base_nt) * int(factor)
            if nt % int(base_nt) != 0:
                raise AssertionError("refined tau grid must contain the fixed evaluation grid")
            fd_grid = FDGrid(fd_y_min, fd_y_max, ny=ny, nt=nt)
            policy_key = (int(factor), float(fd_margin))
            if cached_policy_key != policy_key or cached_policy is None:
                fd_y_grid = np.linspace(fd_y_min, fd_y_max, ny, dtype=np.float64)
                fd_dt = run.problem.horizon / nt
                tau_midpoints = (np.arange(nt, dtype=np.float64) + 0.5) * fd_dt
                frozen_policy, policy_hash, _precomputed_diag = evaluator.precompute_policy(
                    tau_midpoints,
                    fd_y_grid,
                    policy_extension=policy_extension,
                )
                cached_policy_key = policy_key
                cached_policy = (frozen_policy, policy_hash)
            else:
                frozen_policy, policy_hash = cached_policy
            solution = solve_frozen_policy(
                run.problem,
                frozen_policy,
                fd_grid,
                theta_method=theta_method,
                rannacher_steps=rannacher_steps,
                drift_scheme=drift_scheme,
                peclet_limit=peclet_limit,
                boundary=boundary,
                ellipticity_tolerance=solver_ellipticity_tolerance,
                linear_residual_tolerance=linear_residual_tolerance,
                boundary_condition_limit=boundary_condition_limit,
            )
            map_bundle = evaluate_fd_bundle(solution, tau_eval, eval_y)
            map_metric = x_norm_components(*map_bundle, closed_form, yy_eval)
            fd_diag = solution.diagnostics.as_dict()
            map_variant, unmodified = _map_variant(fd_diag, ev_policy_diag)
            denominator = float(input_metric["x_norm"])
            denominator_defined = math.isfinite(denominator) and denominator > denominator_tolerance
            rho_exact = float(map_metric["x_norm"] / denominator) if denominator_defined else float("nan")
            fd_clip = _clip_summary(fd_diag)
            ev_clip = _clip_summary(ev_policy_diag)
            outside_count, outside_fraction = _outside_summary(fd_diag)
            row: Dict[str, Any] = {
                "problem": "merton",
                "group": run.group,
                "protocol_hash": protocol_hash,
                "model_type": "pipinn",
                "e6_role": run.e6_role,
                "n_assets": run.problem.n_assets,
                "seed": run.seed,
                "market_seed": run.market_seed,
                "horizon": run.problem.horizon,
                "gamma": run.problem.gamma,
                "discount": run.problem.discount,
                "bequest": run.problem.bequest,
                "risk_free": run.problem.risk_free,
                "network_dtype": run.network.dtype,
                "checkpoint_outer_iter": outer,
                "source_iter": source_iter,
                "target_policy_iter": target_policy_iter,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_state_sha256": checkpoint_state_hash,
                "market_sha256": run.market_hash,
                "eval_margin": run.eval_margin,
                "eval_window_mode": run.eval_window_mode,
                "eval_w_min_requested": run.eval_w_min_requested,
                "eval_w_min_symmetric": run.eval_w_min_symmetric,
                "ev_tau_min": float(tau_eval[0]),
                "ev_tau_max": float(tau_eval[-1]),
                "ev_y_min": float(eval_y[0]),
                "ev_y_max": float(eval_y[-1]),
                "ev_w_min": run.eval_w_bounds[0],
                "ev_w_max": run.eval_w_bounds[1],
                "fd_margin": fd_margin,
                "fd_y_min": fd_y_min,
                "fd_y_max": fd_y_max,
                "boundary": boundary,
                "boundary_semantics": BOUNDARY_SEMANTICS[boundary],
                "drift_scheme": drift_scheme,
                "grid_factor": factor,
                "ny": ny,
                "nt": nt,
                "dy": (fd_y_max - fd_y_min) / (ny - 1),
                "dt": run.problem.horizon / nt,
                "is_primary": int((factor, fd_margin, boundary) == primary_variant),
                "is_verification": int(is_verification),
                "analysis_mode": analysis_mode,
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "outside_collocation_count_fd": outside_count,
                "outside_collocation_fraction_fd": outside_fraction,
                "rho_exact": rho_exact,
                "denominator_defined": int(denominator_defined),
                "map_variant": map_variant,
                "local_map_unmodified_on_xfd": unmodified,
                "whole_space_map_claim": whole_space_claim,
                "checkpoint_selection": run.checkpoint_selection,
                "vw_guard_frac_fd": fd_clip["vw"],
                "pi_numerator_guard_frac_fd": fd_clip["numerator"],
                "denom_guard_frac_fd": fd_clip["denom"],
                "positive_curvature_frac_fd": fd_clip["positive"],
                "kappa_clip_frac_fd": fd_clip["kappa"],
                "consumption_clip_frac_fd": fd_clip["consumption"],
                "portfolio_any_clip_frac_fd": fd_clip["portfolio_any"],
                "portfolio_component_clip_frac_fd": fd_clip["portfolio_component"],
                "vw_guard_frac_ev": ev_clip["vw"],
                "pi_numerator_guard_frac_ev": ev_clip["numerator"],
                "denom_guard_frac_ev": ev_clip["denom"],
                "positive_curvature_frac_ev": ev_clip["positive"],
                "kappa_clip_frac_ev": ev_clip["kappa"],
                "consumption_clip_frac_ev": ev_clip["consumption"],
                "portfolio_any_clip_frac_ev": ev_clip["portfolio_any"],
                "portfolio_component_clip_frac_ev": ev_clip["portfolio_component"],
                "min_diffusion": fd_diag["min_diffusion"],
                "max_diffusion": fd_diag["max_diffusion"],
                "min_diffusion_variance": fd_diag["min_diffusion_variance"],
                "max_diffusion_variance": fd_diag["max_diffusion_variance"],
                "min_diffusion_variance_ev": ev_policy_diag["min_diffusion_variance"],
                "max_diffusion_variance_ev": ev_policy_diag["max_diffusion_variance"],
                "max_peclet": fd_diag["max_peclet"],
                "upwind_fraction": fd_diag["upwind_fraction"],
                "max_linear_residual": fd_diag["max_linear_residual"],
                "linear_residual_tolerance": linear_residual_tolerance,
                "boundary_elimination_size": fd_diag[
                    "boundary_elimination_size"
                ],
                "boundary_elimination_rank": fd_diag[
                    "boundary_elimination_rank"
                ],
                "boundary_elimination_cond_inf": fd_diag[
                    "boundary_elimination_cond_inf"
                ],
                "min_linear_system_lu_pivot_ratio": fd_diag[
                    "min_linear_system_lu_pivot_ratio"
                ],
                "policy_hash": policy_hash,
            }
            row.update(metric_to_columns("e_input", input_metric))
            row.update(metric_to_columns("e_map", map_metric))
            rows.append(row)

            # E4: the source checkpoint produces alpha_K and the FD map
            # v^{alpha_K}; checkpoint K+1 is the neural approximation
            # v_tilde_K of that same frozen-policy equation.  For verified
            # checkpoints, delta_K is evaluated for every FD variant so its
            # own refinement status has numerical evidence. Unselected
            # checkpoints retain a clearly labelled primary-only row.
            if next_checkpoint is not None and next_bundle is not None:
                delta_metric = x_norm_components(
                    *next_bundle, map_bundle, yy_eval
                )
                defect_refinement_rows.append({
                    "problem": "merton",
                    "group": run.group,
                    "protocol_hash": protocol_hash,
                    "model_type": "pipinn",
                    "e6_role": run.e6_role,
                    "n_assets": run.problem.n_assets,
                    "seed": run.seed,
                    "market_seed": run.market_seed,
                    "eval_margin": run.eval_margin,
                    "eval_window_mode": run.eval_window_mode,
                    "eval_w_min_requested": run.eval_w_min_requested,
                    "eval_w_min_symmetric": run.eval_w_min_symmetric,
                    "ev_tau_min": float(tau_eval[0]),
                    "ev_tau_max": float(tau_eval[-1]),
                    "ev_y_min": float(eval_y[0]),
                    "ev_y_max": float(eval_y[-1]),
                    "ev_w_min": run.eval_w_bounds[0],
                    "ev_w_max": run.eval_w_bounds[1],
                    "defect_iter": target_policy_iter,
                    "defect_kind": "adjacent_policy_evaluation",
                    "target_policy_iter": target_policy_iter,
                    "next_checkpoint_outer_iter": next_outer,
                    "frozen_policy_sha256": policy_hash,
                    "fd_margin": fd_margin,
                    "boundary": boundary,
                    "grid_factor": factor,
                    "ny": ny,
                    "nt": nt,
                    "is_primary": int(row["is_primary"]),
                    "is_verification": int(is_verification),
                    "analysis_mode": analysis_mode,
                    "policy_extension": policy_extension,
                    "map_definition": map_definition,
                    "outside_collocation_count_fd": outside_count,
                    "outside_collocation_fraction_fd": outside_fraction,
                    "delta_value_sup": delta_metric["value_sup"],
                    "delta_vw_sup": delta_metric["vw_sup"],
                    "delta_vww_sup": delta_metric["vww_sup"],
                    "delta_vy_sup": delta_metric["vy_sup"],
                    "delta_vyy_sup": delta_metric["vyy_sup"],
                    "delta_bundle_sup": delta_metric["derivative_sup"],
                    "delta_X": delta_metric["x_norm"],
                    "map_variant": map_variant,
                    "local_map_unmodified_on_xfd": unmodified,
                    "whole_space_map_claim": whole_space_claim,
                })

                if int(row["is_primary"]) == 1:
                    bundle_path = (
                        output / "evaluated_bundles"
                        / f"outer_{int(outer):04d}_to_{next_outer:04d}.npz"
                    )
                    atomic_npz(
                        bundle_path,
                        tau=tau_eval,
                        y=eval_y,
                        derivative_coordinate=np.asarray(
                            "both_log_y_and_wealth_w", dtype="U32"
                        ),
                        **bundle_storage_arrays(
                            "input", input_bundle, eval_y
                        ),
                        **bundle_storage_arrays(
                            "fd_map", map_bundle, eval_y
                        ),
                        **bundle_storage_arrays(
                            "next_neural", next_bundle, eval_y
                        ),
                        **bundle_storage_arrays(
                            "optimal", closed_form, eval_y
                        ),
                    )
                    defect_rows.append({
                        "problem": "merton",
                        "group": run.group,
                        "protocol_hash": protocol_hash,
                        "model_type": "pipinn",
                        "e6_role": run.e6_role,
                        "n_assets": run.problem.n_assets,
                        "seed": run.seed,
                        "market_seed": run.market_seed,
                        "eval_margin": run.eval_margin,
                        "eval_window_mode": run.eval_window_mode,
                        "eval_w_min_requested": run.eval_w_min_requested,
                        "eval_w_min_symmetric": run.eval_w_min_symmetric,
                        "ev_tau_min": float(tau_eval[0]),
                        "ev_tau_max": float(tau_eval[-1]),
                        "ev_y_min": float(eval_y[0]),
                        "ev_y_max": float(eval_y[-1]),
                        "ev_w_min": run.eval_w_bounds[0],
                        "ev_w_max": run.eval_w_bounds[1],
                        "defect_iter": target_policy_iter,
                        "defect_kind": "adjacent_policy_evaluation",
                        "checkpoint_outer_iter": int(outer),
                        "source_iter": source_iter,
                        "target_policy_iter": target_policy_iter,
                        "next_checkpoint_outer_iter": next_outer,
                        "next_neural_iter": next_source_iter,
                        "checkpoint_state_sha256": checkpoint_state_hash,
                        "next_checkpoint_state_sha256": (
                            canonical_checkpoint_state_hash(next_checkpoint)
                        ),
                        "frozen_policy_sha256": policy_hash,
                        "fd_margin": fd_margin,
                        "boundary": boundary,
                        "grid_factor": factor,
                        "ny": ny,
                        "nt": nt,
                        "is_verification": int(is_verification),
                        "analysis_mode": analysis_mode,
                        "policy_extension": policy_extension,
                        "map_definition": map_definition,
                        "outside_collocation_count_fd": outside_count,
                        "outside_collocation_fraction_fd": outside_fraction,
                        "delta_value_sup": delta_metric["value_sup"],
                        "delta_vw_sup": delta_metric["vw_sup"],
                        "delta_vww_sup": delta_metric["vww_sup"],
                        "delta_vy_sup": delta_metric["vy_sup"],
                        "delta_vyy_sup": delta_metric["vyy_sup"],
                        "delta_bundle_sup": delta_metric["derivative_sup"],
                        "delta_X": delta_metric["x_norm"],
                        "defect_grid_abs_change": "",
                        "defect_grid_rel_change": "",
                        "defect_domain_abs_change": "",
                        "defect_domain_rel_change": "",
                        "defect_boundary_abs_change": "",
                        "defect_sensitivity_envelope": "",
                        "p_res_post_restore": residual_by_outer.get(
                            next_outer, ""
                        ),
                        "p_res_source": residual_by_outer.get(int(outer), ""),
                        "residual_semantics": residual_semantics,
                        "evaluated_bundle_path": str(
                            bundle_path.relative_to(output)
                        ),
                        "evaluated_bundle_sha256": sha256_file(bundle_path),
                        "refinement_status": "pending_defect_refinement",
                        "map_variant": map_variant,
                        "local_map_unmodified_on_xfd": unmodified,
                        "whole_space_map_claim": whole_space_claim,
                    })

    assess_refinement(
        rows,
        grid_factors=grid_factors,
        fd_margins=fd_margins,
        boundaries=boundaries,
        abs_tolerance=refinement_abs_tolerance,
        rel_tolerance=refinement_rel_tolerance,
    )
    assess_defect_refinement(
        defect_refinement_rows,
        grid_factors=grid_factors,
        fd_margins=fd_margins,
        boundaries=boundaries,
        abs_tolerance=refinement_abs_tolerance,
        rel_tolerance=refinement_rel_tolerance,
    )
    primary_defect_refinement = [
        row for row in defect_refinement_rows if int(row["is_primary"]) == 1
    ]
    if len(primary_defect_refinement) != len(defect_rows):
        raise AssertionError(
            "expected exactly one primary FD-refinement row per E4 defect"
        )
    refinement_by_defect = {
        int(row["defect_iter"]): row for row in primary_defect_refinement
    }
    evidence_fields = (
        "boundary_role",
        "defect_grid_abs_change",
        "defect_grid_rel_change",
        "defect_domain_abs_change",
        "defect_domain_rel_change",
        "defect_boundary_abs_change",
        "defect_boundary_rel_change",
        "boundary_sensitivity_status",
        "defect_grid_domain_sensitivity_envelope",
        "defect_sensitivity_envelope",
        "grid_domain_refinement_status",
        "refinement_semantics_version",
        "refinement_status",
        "is_verification",
    )
    for defect in defect_rows:
        evidence = refinement_by_defect[int(defect["defect_iter"])]
        for field in evidence_fields:
            defect[field] = evidence[field]

    primary_rows = [row for row in rows if int(row["is_primary"]) == 1]
    if len(primary_rows) != len(run.checkpoints):
        raise AssertionError("expected exactly one primary exact-map row per checkpoint")
    expected_defect_iterations = (
        []
        if skip_e4
        else (
            (
                [0]
                if (
                    run.initial_defect_mode
                    == "analytic-initial-policy"
                    and 0 in checkpoint_source_to_outer
                )
                else (
                    [1]
                    if (
                        run.initial_defect_mode == "warm-start-value"
                        and 1 in checkpoint_source_to_outer
                    )
                    else []
                )
            )
            + [
                int(run.checkpoint_target_policy_iters[int(outer)])
                for outer, _path in run.checkpoints
                if (
                    int(run.checkpoint_source_iters[int(outer)]) + 1
                    in checkpoint_source_to_outer
                )
            ]
        )
    )
    observed_defect_iterations = sorted(
        int(row["defect_iter"]) for row in defect_rows
    )
    if observed_defect_iterations != sorted(expected_defect_iterations):
        raise AssertionError(
            "E4 defect iteration coverage mismatch: "
            f"found {observed_defect_iterations}, "
            f"expected {sorted(expected_defect_iterations)}"
        )
    for defect in defect_rows:
        defect_iter = int(defect["defect_iter"])
        expected_next_outer = checkpoint_source_to_outer.get(defect_iter)
        if not (
            int(defect["target_policy_iter"]) == defect_iter
            and int(defect["next_neural_iter"]) == defect_iter
            and expected_next_outer is not None
            and int(defect["next_checkpoint_outer_iter"])
            == expected_next_outer
        ):
            raise AssertionError(f"mis-indexed E4 defect row: {defect}")
    defect_refinement_summary = summarize_defect_refinement(defect_rows)
    verified_primary_rows = [
        row for row in primary_rows if bool(int(row["is_verification"]))
    ]
    verified_primary_statuses = [
        str(row.get("grid_domain_refinement_status", ""))
        for row in verified_primary_rows
    ]
    if not verified_primary_statuses:
        grid_domain_refinement_evidence_status = "not_checked"
    elif all(status == "pass" for status in verified_primary_statuses):
        grid_domain_refinement_evidence_status = "pass"
    elif any(status == "fail" for status in verified_primary_statuses):
        grid_domain_refinement_evidence_status = "fail"
    else:
        grid_domain_refinement_evidence_status = "incomplete"
    defect_grid_domain_refinement_evidence_status = (
        defect_refinement_summary["evidence_status"]
    )
    write_csv(output / "exact_map_refinement.csv", rows, RATIO_FIELDS)
    write_csv(output / "exact_map_ratios.csv", primary_rows, RATIO_FIELDS)
    write_csv(output / "exact_map_defects.csv", defect_rows, DEFECT_FIELDS)
    write_csv(
        output / "exact_map_defect_refinement.csv",
        defect_refinement_rows,
        DEFECT_REFINEMENT_FIELDS,
    )
    protocol = {
        "run_dir": str(run.run_dir),
        "config_path": str(run.config_path),
        "config_sha256": sha256_file(run.config_path),
        "market_path": str(run.market_path),
        "market_sha256": run.market_hash,
        "market_file_sha256": sha256_file(run.market_path),
        "weight_dir": str(run.weight_dir),
        "problem": {
            "horizon": run.problem.horizon,
            "w_min": math.exp(run.problem.y_min),
            "w_max": math.exp(run.problem.y_max),
            "gamma": run.problem.gamma,
            "discount": run.problem.discount,
            "bequest": run.problem.bequest,
            "risk_free": run.problem.risk_free,
            "n_assets": run.problem.n_assets,
        },
        "policy": asdict(run.policy),
        "network": asdict(run.network),
        "metadata_provenance": run.metadata_provenance,
        "training_protocol": run.training_protocol,
        "e6_role": run.e6_role,
        "initial_defect_mode": run.initial_defect_mode,
        "checkpoint_indexing": {
            **run.checkpoint_indexing_provenance,
            "checkpoint_outer_to_source_iter": {
                str(outer): int(source)
                for outer, source in sorted(
                    run.checkpoint_source_iters.items()
                )
            },
            "checkpoint_outer_to_target_policy_iter": {
                str(outer): int(target)
                for outer, target in sorted(
                    run.checkpoint_target_policy_iters.items()
                )
            },
        },
        "warm_start_predecessor": (
            {
                "path": str(run.warm_start_bundle_path),
                "file_sha256": run.warm_start_bundle_file_hash,
                "model_state_sha256": run.warm_start_model_state_hash,
                "source_iter": 0,
                "target_policy_iter": 1,
                "warmup_excluded_from_outer_history": True,
            }
            if run.warm_start_bundle_path is not None
            else None
        ),
        "eval_margin": run.eval_margin,
        "eval_window_mode": run.eval_window_mode,
        "eval_w_min_requested": run.eval_w_min_requested,
        "eval_w_min_symmetric": run.eval_w_min_symmetric,
        "evaluation_bounds": {
            "y_min": run.eval_y_bounds[0],
            "y_max": run.eval_y_bounds[1],
            "w_min": run.eval_w_bounds[0],
            "w_max": run.eval_w_bounds[1],
            "upper_bound_source": "symmetric_eval_margin",
            "lower_bound_source": (
                "explicit_eval_w_min"
                if run.eval_window_mode == "lower-wealth-override"
                else "symmetric_eval_margin"
            ),
        },
        "analysis_mode": analysis_mode,
        "paper_aggregation_eligible": bool(
            not skip_e4 and run.checkpoint_selection == "all"
        ),
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "whole_space_map_claim": whole_space_claim,
        "collocation_bounds": {
            "y_min": run.problem.y_min,
            "y_max": run.problem.y_max,
            "w_min": math.exp(run.problem.y_min),
            "w_max": math.exp(run.problem.y_max),
            "interpretation": "saved nominal training bounds",
        },
        "policy_extension_semantics": {
            "network_evaluation": (
                "clip y to saved nominal bounds"
                if policy_extension == "boundary-projection"
                else "evaluate the neural policy at the raw FD y coordinate"
            ),
            "portfolio": "pi(t,y)=pi_network(t,projected_y)",
            "consumption": (
                "kappa_boundary=c_network(t,projected_y)/exp(projected_y); "
                "c_extended=kappa_boundary*exp(actual_y), followed by the "
                "configured consumption-level bounds when present"
            ),
        },
        "outside_collocation_summary": {
            "max_count_fd": max(
                int(row["outside_collocation_count_fd"])
                for row in primary_rows
            ),
            "max_fraction_fd": max(
                float(row["outside_collocation_fraction_fd"])
                for row in primary_rows
            ),
            "counting_domain": (
                "FD tensor grid policy queries across all time midpoints"
            ),
        },
        "grid": {
            "base_ny": base_ny,
            "base_nt": base_nt,
            "eval_ny": eval_ny,
            "grid_factors": list(grid_factors),
            "fd_margins": list(fd_margins),
            "boundaries": list(boundaries),
            "boundary_semantics": {
                name: BOUNDARY_SEMANTICS[name] for name in boundaries
            },
            "primary_boundary_requirement": (
                "Use robin as the paper primary. exact-dirichlet is only an "
                "optimal-reference sensitivity audit."
            ),
            "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
            "refinement_gate_axes": ["grid", "domain"],
            "boundary_replacement_role": (
                "separate_sensitivity_not_refinement_gate"
            ),
            "boundary_specific_refinement": (
                "grid/domain changes and grid_domain_refinement_status are "
                "computed independently within each boundary closure on its "
                "finest-grid/largest-domain representative"
            ),
            "legacy_field_semantics": {
                "refinement_status": (
                    "primary-boundary grid/domain-only gate"
                ),
                "rho_sensitivity_envelope": (
                    "primary rho plus grid and domain changes only"
                ),
                "boundary_abs_change": (
                    "diagnostic boundary-replacement sensitivity; never a "
                    "refinement gate"
                ),
            },
            "verify_checkpoints": verify_checkpoints,
            "drift_scheme": drift_scheme,
            "peclet_limit": peclet_limit,
            "theta_method": theta_method,
            "initial_full_dt_backward_euler_steps": rannacher_steps,
            "linear_residual_tolerance": linear_residual_tolerance,
            "boundary_condition_limit": boundary_condition_limit,
            "solver_ellipticity_tolerance": solver_ellipticity_tolerance,
            "evaluation_time_coordinate": "tau=T-t",
            "evaluation_time_calendar_domain": "[0,T)",
            "evaluation_terminal_face_included": False,
            "evaluation_tau_min": float(tau_eval[0]),
            "evaluation_tau_max": float(tau_eval[-1]),
            "evaluation_y_min": float(eval_y[0]),
            "evaluation_y_max": float(eval_y[-1]),
            "evaluation_w_min": run.eval_w_bounds[0],
            "evaluation_w_max": run.eval_w_bounds[1],
        },
        "norm": "sup|V-V*| + sup sqrt((V_w-V*_w)^2 + (V_ww-V*_ww)^2)",
        "e4_defect": {
            "definition": (
                (
                    "target-branch delta_1=||v_tilde_1-"
                    "E(G(v_tilde_0))||_Xev from the hash-validated common "
                    "warm-start predecessor; subsequent delta_n use adjacent "
                    "branch checkpoints"
                )
                if run.initial_defect_mode == "warm-start-value"
                else (
                    "delta_0=||v_tilde_0-v^{alpha_0}||_Xev from the "
                    "configured analytic initial policy; for n>=1, delta_n="
                    "||v_tilde_n-E(G(v_tilde_(n-1)))||_Xev from adjacent "
                    "checkpoints"
                )
            ),
            "residual_semantics": residual_semantics,
            "n_defect_iterations": len(defect_rows),
            "status": (
                "skipped_by_exact_map_pilot" if skip_e4 else "computed"
            ),
            "defect_iterations": observed_defect_iterations,
            "defect_coverage": (
                "complete_for_saved_schedule"
                if observed_defect_iterations
                == sorted(expected_defect_iterations) else "incomplete"
            ),
            "refinement_selector": (
                "first_defect_plus_source_outer_in_verify_checkpoints"
            ),
            "refinement_required_iterations": (
                defect_refinement_summary["required_iterations"]
            ),
            "refinement_required_statuses": (
                defect_refinement_summary["required_statuses"]
            ),
            "refinement_evidence_status": (
                defect_refinement_summary["evidence_status"]
            ),
            "grid_domain_refinement_evidence_status": (
                defect_grid_domain_refinement_evidence_status
            ),
            "refinement_table": "exact_map_defect_refinement.csv",
            "evaluated_bundle_storage": (
                "evaluated_bundles/*.npz; tau/y plus initial controls or neural "
                "input, FD-map, next-neural, and optimal arrays. Every value "
                "bundle stores value plus both (V_y,V_yy) and correctly "
                "converted (V_w,V_ww); derivative_coordinate documents this. "
                "SHA-256 is recorded in exact_map_defects.csv"
            ),
        },
        "protocol_hash": protocol_hash,
        "protocol_compatibility_version": (
            EXACT_PROTOCOL_COMPATIBILITY_VERSION
        ),
        "protocol_compatibility_payload": (
            protocol_compatibility_payload
        ),
        "implementation_hashes": implementation_hashes,
        "refinement_abs_tolerance": refinement_abs_tolerance,
        "refinement_rel_tolerance": refinement_rel_tolerance,
        "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
        "refinement_gate_axes": ["grid", "domain"],
        "boundary_replacement_role": (
            "separate_sensitivity_not_refinement_gate"
        ),
        "grid_domain_refinement_evidence_status": (
            grid_domain_refinement_evidence_status
        ),
        "defect_grid_domain_refinement_evidence_status": (
            defect_grid_domain_refinement_evidence_status
        ),
        "denominator_tolerance": denominator_tolerance,
        "linear_residual_tolerance": linear_residual_tolerance,
        "boundary_condition_limit": boundary_condition_limit,
        "solver_ellipticity_tolerance": solver_ellipticity_tolerance,
        "checkpoint_hashes": {
            str(outer): sha256_file(path) for outer, path in run.checkpoints
        },
        "checkpoint_state_hashes": {
            str(outer): canonical_checkpoint_state_hash(path)
            for outer, path in run.checkpoints
        },
        "checkpoint_schedule_outer": run.checkpoint_schedule,
        "checkpoint_schedule_source_n": [
            int(run.checkpoint_source_iters[outer])
            for outer in run.checkpoint_schedule
        ],
        "checkpoint_manifest": (
            {
                "path": str(run.checkpoint_manifest_path),
                "file_sha256": run.checkpoint_manifest_hash,
            }
            if run.checkpoint_manifest_path is not None else None
        ),
        "official_final": (
            {
                "path": str(run.final_checkpoint),
                "state_sha256": run.final_checkpoint_state_hash,
                "file_sha256": sha256_file(run.final_checkpoint),
            }
            if run.final_checkpoint is not None else None
        ),
        "checkpoint_selection": run.checkpoint_selection,
    }
    atomic_json(output / "exact_map_config.json", protocol)
    status = {
        "status": "success",
        "analysis_mode": analysis_mode,
        "e6_role": run.e6_role,
        "initial_defect_mode": run.initial_defect_mode,
        "checkpoint_indexing": run.checkpoint_indexing_provenance,
        "eval_margin": run.eval_margin,
        "eval_window_mode": run.eval_window_mode,
        "eval_w_min_requested": run.eval_w_min_requested,
        "eval_w_min_symmetric": run.eval_w_min_symmetric,
        "ev_y_min": run.eval_y_bounds[0],
        "ev_y_max": run.eval_y_bounds[1],
        "ev_w_min": run.eval_w_bounds[0],
        "ev_w_max": run.eval_w_bounds[1],
        "protocol_hash": protocol_hash,
        "protocol_compatibility_version": (
            EXACT_PROTOCOL_COMPATIBILITY_VERSION
        ),
        "refinement_semantics_version": REFINEMENT_SEMANTICS_VERSION,
        "refinement_gate_axes": ["grid", "domain"],
        "boundary_replacement_role": (
            "separate_sensitivity_not_refinement_gate"
        ),
        "paper_aggregation_eligible": bool(
            not skip_e4 and run.checkpoint_selection == "all"
        ),
        "policy_extension": policy_extension,
        "map_definition": map_definition,
        "whole_space_map_claim": whole_space_claim,
        "e4_status": (
            "skipped_by_exact_map_pilot" if skip_e4 else "computed"
        ),
        "n_checkpoints": len(run.checkpoints),
        "n_primary_rows": len(primary_rows),
        "n_refinement_rows": len(rows),
        "n_defect_rows": len(defect_rows),
        "n_expected_defect_iterations": len(expected_defect_iterations),
        "defect_iterations": observed_defect_iterations,
        "max_delta_X": (
            max(float(row["delta_X"]) for row in defect_rows)
            if defect_rows else None
        ),
        "defect_refinement_required_iterations": (
            defect_refinement_summary["required_iterations"]
        ),
        "defect_refinement_required_statuses": (
            defect_refinement_summary["required_statuses"]
        ),
        "defect_refinement_evidence_status": (
            defect_refinement_summary["evidence_status"]
        ),
        "grid_domain_refinement_evidence_status": (
            grid_domain_refinement_evidence_status
        ),
        "defect_grid_domain_refinement_evidence_status": (
            defect_grid_domain_refinement_evidence_status
        ),
        "n_defect_refinement_rows": len(defect_refinement_rows),
        "n_defect_refinement_pass": sum(
            str(row.get("refinement_status", "")) == "pass"
            for row in primary_defect_refinement
        ),
        "n_evaluated_bundle_files": len(defect_rows),
        "defect_residual_semantics": residual_semantics,
        "n_modified_defect_policies": sum(
            not int(row["local_map_unmodified_on_xfd"]) for row in defect_rows
        ),
        "n_refinement_pass": sum(row.get("refinement_status") == "pass" for row in primary_rows),
        "n_sampled_modified_map": sum(
            not int(row["local_map_unmodified_on_xfd"]) for row in primary_rows
        ),
        "map_variants": sorted({str(row["map_variant"]) for row in primary_rows}),
        "max_outside_collocation_count_fd": max(
            int(row["outside_collocation_count_fd"]) for row in primary_rows
        ),
        "max_outside_collocation_fraction_fd": max(
            float(row["outside_collocation_fraction_fd"])
            for row in primary_rows
        ),
        "max_linear_residual": max(
            float(row["max_linear_residual"]) for row in primary_rows
        ),
        "linear_residual_tolerance": linear_residual_tolerance,
        "all_linear_residuals_within_tolerance": all(
            float(row["max_linear_residual"])
            <= float(linear_residual_tolerance)
            for row in primary_rows
        ),
        "max_boundary_elimination_cond_inf": max(
            float(row["boundary_elimination_cond_inf"])
            for row in primary_rows
        ),
        "boundary_condition_limit": boundary_condition_limit,
        "min_linear_system_lu_pivot_ratio": min(
            float(row["min_linear_system_lu_pivot_ratio"])
            for row in primary_rows
        ),
        "solver_ellipticity_tolerance": solver_ellipticity_tolerance,
        "max_activation_fractions": {
            field: max(float(row[field]) for row in primary_rows)
            for field in ACTIVATION_FIELDS
        },
        "all_primary_ratios_finite": all(math.isfinite(float(row["rho_exact"])) for row in primary_rows),
    }
    atomic_json(output / "exact_map_status.json", status)
    incomplete_marker = output / "_INCOMPLETE_EXACT_MAP"
    if incomplete_marker.exists():
        incomplete_marker.unlink()
    (output / "_SUCCESS_EXACT_MAP").touch()
    return primary_rows, rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(arr.std(ddof=1))
    half = t_critical_95(int(arr.size) - 1) * std / math.sqrt(int(arr.size))
    return mean, std, mean - half, mean + half


def validate_plot_options(
    *,
    dpi: int,
    fig_width: float,
    fig_height: float,
    font_size: float,
    line_width: float,
    line_alpha: float,
    marker_size: float,
    band_alpha: float,
    grid_alpha: float,
    floor_alpha: float,
    bbox_inches: str,
) -> None:
    for name, value in (
        ("dpi", dpi),
        ("fig_width", fig_width),
        ("fig_height", fig_height),
        ("font_size", font_size),
        ("line_width", line_width),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(marker_size)) or float(marker_size) < 0.0:
        raise ValueError("marker_size must be nonnegative and finite")
    for name, value, positive in (
        ("line_alpha", line_alpha, True),
        ("band_alpha", band_alpha, False),
        ("grid_alpha", grid_alpha, False),
        ("floor_alpha", floor_alpha, False),
    ):
        lower_ok = float(value) > 0.0 if positive else float(value) >= 0.0
        if not math.isfinite(float(value)) or not lower_ok or float(value) > 1.0:
            interval = "(0,1]" if positive else "[0,1]"
            raise ValueError(f"{name} must be finite and lie in {interval}")
    if bbox_inches not in ("tight", "standard"):
        raise ValueError("bbox_inches must be 'tight' or 'standard'")


def aggregate_exact_map(
    result_dirs: Sequence[Path],
    output: Path,
    *,
    expected_seeds: Sequence[int],
    floor_multiple: float,
    allow_incomplete: bool,
    allow_unverified: bool,
    require_locally_unmodified_map: bool,
    make_plot: bool,
    plot_format: Optional[str] = None,
    dpi: int = 300,
    plot_formats: Optional[Sequence[str]] = None,
    fig_width: float = 6.0,
    fig_height: float = 4.0,
    font_size: float = 12.0,
    font_family: str = "",
    line_width: float = 1.8,
    line_alpha: float = 1.0,
    marker_size: float = 4.0,
    band_alpha: float = 0.18,
    grid_alpha: float = 0.3,
    floor_alpha: float = 0.8,
    bbox_inches: str = "tight",
    min_seeds: int = 2,
    ellipticity_tolerance: float = 0.0,
    allow_degenerate_diffusion: bool = False,
    allow_neural_extrapolation: bool = False,
) -> None:
    if plot_formats is None:
        resolved_plot_formats = parse_plot_formats(plot_format or "png")
    elif isinstance(plot_formats, str):
        resolved_plot_formats = parse_plot_formats(plot_formats)
    else:
        resolved_plot_formats = parse_plot_formats(
            ",".join(str(value) for value in plot_formats)
        )
    validate_plot_options(
        dpi=dpi,
        fig_width=fig_width,
        fig_height=fig_height,
        font_size=font_size,
        line_width=line_width,
        line_alpha=line_alpha,
        marker_size=marker_size,
        band_alpha=band_alpha,
        grid_alpha=grid_alpha,
        floor_alpha=floor_alpha,
        bbox_inches=bbox_inches,
    )

    if isinstance(min_seeds, bool) or int(min_seeds) != min_seeds or int(min_seeds) < 1:
        raise ValueError("min_seeds must be a positive integer")
    min_seeds = int(min_seeds)
    if not math.isfinite(float(floor_multiple)) or float(floor_multiple) < 0.0:
        raise ValueError("floor_multiple must be finite and nonnegative")
    if (
        not math.isfinite(float(ellipticity_tolerance))
        or float(ellipticity_tolerance) < 0.0
    ):
        raise ValueError("ellipticity_tolerance must be finite and nonnegative")
    records: List[Dict[str, Any]] = []
    for result_dir in result_dirs:
        result_dir = result_dir.expanduser().resolve()
        if not (result_dir / "_SUCCESS_EXACT_MAP").is_file():
            raise ValueError(f"exact-map output is not marked successful: {result_dir}")
        exact_cfg: Optional[Dict[str, Any]] = None
        exact_cfg_path = result_dir / "exact_map_config.json"
        if exact_cfg_path.is_file():
            loaded_cfg = json.loads(
                exact_cfg_path.read_text(encoding="utf-8")
            )
            if not isinstance(loaded_cfg, Mapping):
                raise ValueError(
                    f"{exact_cfg_path}: exact-map config must be an object"
                )
            exact_cfg = dict(loaded_cfg)
        for row in read_csv(result_dir / "exact_map_ratios.csv"):
            required_components = [
                f"{prefix}_{suffix}"
                for prefix in ("e_input", "e_map")
                for suffix in ("value", "vw", "vww", "vy", "vyy", "deriv", "X")
            ]
            missing_components = [
                key for key in required_components
                if key not in row or str(row[key]).strip() == ""
            ]
            if missing_components:
                raise ValueError(
                    f"{result_dir}: exact-map component columns are missing/blank: "
                    f"{missing_components}"
                )
            parsed: Dict[str, Any] = dict(row)
            # Compatibility fallback for exact-map outputs generated before
            # the explicit lower-wealth sweep existed. Their effective
            # endpoints are already unambiguously stored in log wealth.
            parsed.setdefault("eval_window_mode", "symmetric-margin")
            parsed.setdefault(
                "ev_w_min", math.exp(float(parsed["ev_y_min"]))
            )
            parsed.setdefault(
                "ev_w_max", math.exp(float(parsed["ev_y_max"]))
            )
            parsed.setdefault("eval_w_min_requested", "")
            parsed.setdefault(
                "eval_w_min_symmetric", parsed["ev_w_min"]
            )
            requested_text = str(
                parsed.get("eval_w_min_requested", "")
            ).strip()
            parsed["eval_w_min_requested"] = (
                None if not requested_text else float(requested_text)
            )
            for key in (
                "n_assets", "seed", "market_seed", "source_iter", "target_policy_iter",
                "local_map_unmodified_on_xfd", "is_verification",
                "outside_collocation_count_fd",
            ):
                parsed[key] = int(float(parsed[key]))
            for key in (
                "horizon", "gamma", "discount", "bequest", "risk_free",
                "eval_margin", "eval_w_min_symmetric",
                "ev_y_min", "ev_y_max",
                "ev_w_min", "ev_w_max",
                "fd_margin", *required_components, "rho_exact",
                "rho_sensitivity_envelope", *ACTIVATION_FIELDS, *ELLIPTICITY_FIELDS,
                "outside_collocation_fraction_fd", "max_linear_residual",
                "linear_residual_tolerance", "boundary_elimination_cond_inf",
                "min_linear_system_lu_pivot_ratio",
            ):
                parsed[key] = float(parsed[key])
            for key in required_components:
                if not math.isfinite(float(parsed[key])):
                    raise ValueError(
                        f"{result_dir}: exact-map component {key} must be finite"
                    )
            if not (
                math.isclose(
                    math.log(float(parsed["ev_w_min"])),
                    float(parsed["ev_y_min"]),
                    rel_tol=0.0,
                    abs_tol=5e-12,
                )
                and math.isclose(
                    math.log(float(parsed["ev_w_max"])),
                    float(parsed["ev_y_max"]),
                    rel_tol=0.0,
                    abs_tol=5e-12,
                )
            ):
                raise ValueError(
                    f"{result_dir}: recorded wealth/log-wealth evaluation "
                    "endpoints are inconsistent"
                )
            if not (
                float(parsed["ev_w_min"])
                < float(parsed["ev_w_max"])
            ):
                raise ValueError(
                    f"{result_dir}: evaluation wealth bounds are invalid"
                )
            if (
                not math.isfinite(float(parsed["max_linear_residual"]))
                or float(parsed["max_linear_residual"])
                > float(parsed["linear_residual_tolerance"])
            ):
                raise ValueError(
                    f"{result_dir}: linear residual exceeds its recorded hard "
                    "tolerance"
                )
            parsed["result_dir"] = str(result_dir)
            source_protocol_hash = str(parsed["protocol_hash"])
            parsed["source_protocol_hash"] = source_protocol_hash
            if exact_cfg is not None:
                declared_protocol_hash = str(
                    exact_cfg.get("protocol_hash", "")
                )
                if (
                    declared_protocol_hash
                    and declared_protocol_hash != source_protocol_hash
                ):
                    raise ValueError(
                        f"{result_dir}: exact-map row/config protocol hash "
                        "mismatch"
                    )
                parsed["protocol_compatibility_hash"] = (
                    exact_protocol_compatibility_hash(
                        exact_cfg,
                        training_group=str(parsed["group"]),
                    )
                )
            else:
                # Unit-test fixtures and very early pilot outputs may not
                # carry exact_map_config.json.  Their original producer hash
                # is the only available protocol evidence.
                parsed["protocol_compatibility_hash"] = (
                    source_protocol_hash
                )
            records.append(parsed)
    if not records:
        raise ValueError("no exact-map primary rows were found")

    groups: Dict[
        Tuple[str, str, float, float], List[Dict[str, Any]]
    ] = {}
    for row in records:
        window_key = (
            str(row["group"]),
            str(row["eval_window_mode"]),
            float(row["ev_w_min"]),
            float(row["ev_w_max"]),
        )
        groups.setdefault(window_key, []).append(row)
    expected = set(int(seed) for seed in expected_seeds)
    per_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    floor_summary_rows: List[Dict[str, Any]] = []
    worst_rows: List[Dict[str, Any]] = []

    for window_key, group_rows in sorted(groups.items()):
        group = window_key[0]
        seeds = sorted({int(row["seed"]) for row in group_rows})
        if expected and set(seeds) != expected and not allow_incomplete:
            raise ValueError(f"group={group}: exact-map seeds={seeds}, expected={sorted(expected)}")
        selected = sorted(set(seeds) & expected) if expected else seeds
        if len(selected) < min_seeds:
            raise ValueError(
                f"group={group}: found {len(selected)} selected seed(s), "
                f"but min_seeds={min_seeds}"
            )
        markets = {str(row["market_sha256"]) for row in group_rows if int(row["seed"]) in selected}
        if len(markets) != 1:
            raise ValueError(f"group={group}: exact-map runs use different market snapshots")
        market_seeds = {
            int(row["market_seed"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        if len(market_seeds) != 1:
            raise ValueError(
                f"group={group}: training seeds do not share one fixed market_seed"
            )
        fixed_market_seed = next(iter(market_seeds))
        protocols = {
            str(row["protocol_compatibility_hash"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        if len(protocols) != 1:
            raise ValueError(f"group={group}: exact-map runs use different FD/evaluation protocols")
        protocol_hash = next(iter(protocols))
        analysis_modes = {
            str(row["analysis_mode"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        if analysis_modes != {"exact_map_and_e4"} and not allow_incomplete:
            raise ValueError(
                f"group={group}: paper aggregation requires "
                f"analysis_mode=exact_map_and_e4; found {sorted(analysis_modes)}"
            )
        extensions = {
            str(row["policy_extension"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        definitions = {
            str(row["map_definition"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        if len(extensions) != 1 or len(definitions) != 1:
            raise ValueError(
                f"group={group}: mixed policy extension/map definitions"
            )
        policy_extension = next(iter(extensions))
        map_definition = next(iter(definitions))
        if (
            policy_extension != "boundary-projection"
            or map_definition
            != "finite_domain_boundary_projected_policy_extension"
        ) and not allow_neural_extrapolation:
            raise ValueError(
                f"group={group}: paper aggregation requires the finite-domain "
                "boundary-projected policy extension; pass "
                "--allow-neural-extrapolation only for a documented sensitivity"
            )
        selections = {
            str(row["checkpoint_selection"])
            for row in group_rows
            if int(row["seed"]) in selected
        }
        if selections != {"all"} and not allow_incomplete:
            raise ValueError(
                f"group={group}: paper aggregation requires the complete checkpoint schedule; "
                f"found selections={sorted(selections)}"
            )
        by_seed: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for seed in selected:
            seed_rows = [row for row in group_rows if int(row["seed"]) == seed]
            indexed = {int(row["source_iter"]): row for row in seed_rows}
            if len(indexed) != len(seed_rows):
                raise ValueError(f"group={group}, seed={seed}: duplicate source_iter rows")
            by_seed[seed] = indexed
        supports = {seed: set(rows) for seed, rows in by_seed.items()}
        first_support = supports[selected[0]]
        if any(support != first_support for support in supports.values()):
            raise ValueError(f"group={group}: checkpoint schedules differ across seeds")
        outers = sorted(first_support)
        floors: Dict[int, float] = {}
        regular: Dict[int, set[int]] = {}
        for seed in selected:
            tail_len = max(1, int(math.ceil(0.1 * len(outers))))
            floor = float(np.median([by_seed[seed][outer]["e_input_X"] for outer in outers[-tail_len:]]))
            if not math.isfinite(floor) or floor < 0.0:
                raise ValueError(f"group={group}, seed={seed}: invalid late input-error scale {floor}")
            floors[seed] = floor
            if float(floor_multiple) == 0.0:
                # Paper default: retain every finite exact-map ratio.  The
                # late neural error is reported as a descriptive scale only;
                # it is not an FD discretization floor and must not hide late
                # non-contraction.
                regular[seed] = {
                    outer for outer in outers
                    if math.isfinite(float(by_seed[seed][outer]["rho_exact"]))
                }
            else:
                # Optional exploratory compatibility filter.  Its basis is
                # explicitly the late input-error scale, not a numerical FD
                # error estimate.
                regular[seed] = {
                    outer
                    for outer in outers
                    if math.isfinite(float(by_seed[seed][outer]["rho_exact"]))
                    and float(by_seed[seed][outer]["e_input_X"])
                    > float(floor_multiple) * floor
                }
            if not regular[seed]:
                raise ValueError(
                    f"group={group}, seed={seed}: empty regular support at floor multiple {floor_multiple:g}"
                )
        common = set.intersection(*(regular[seed] for seed in selected))
        if not common:
            raise ValueError(f"group={group}: common regular exact-map support is empty")

        # With the paper default floor_multiple=0, every finite ratio is in the
        # regular region. A positive value is an explicitly exploratory late
        # neural-error filter, not a numerical FD-floor claim.
        for seed in selected:
            relevant_rows = [by_seed[seed][outer] for outer in sorted(regular[seed])]
            nonfinite = [
                int(row["source_iter"])
                for row in relevant_rows
                if not math.isfinite(float(row["rho_exact"]))
            ]
            if nonfinite:
                raise ValueError(
                    f"group={group}, seed={seed}: non-finite exact-map ratios in regular region={nonfinite}"
                )
            degenerate = [
                int(row["source_iter"])
                for row in relevant_rows
                if (
                    not math.isfinite(float(row["min_diffusion_variance"]))
                    or not math.isfinite(float(row["min_diffusion_variance_ev"]))
                    or min(
                        float(row["min_diffusion_variance"]),
                        float(row["min_diffusion_variance_ev"]),
                    ) <= float(ellipticity_tolerance)
                )
            ]
            if degenerate and not allow_degenerate_diffusion:
                raise ValueError(
                    f"group={group}, seed={seed}: frozen diffusion variance is not "
                    f"above ellipticity_tolerance={ellipticity_tolerance:g} at "
                    f"regular checkpoints={degenerate}"
                )
            if not allow_unverified:
                bad = [
                    int(row["source_iter"])
                    for row in relevant_rows
                    if str(row["refinement_status"]) != "pass"
                ]
                if bad:
                    raise ValueError(
                        f"group={group}, seed={seed}: regular checkpoints without passed "
                        f"h/domain/boundary sensitivity audit={bad}; "
                        f"rerun with --verify-checkpoints all"
                    )
            if require_locally_unmodified_map:
                modified = [
                    int(row["source_iter"])
                    for row in relevant_rows
                    if not int(row["local_map_unmodified_on_xfd"])
                ]
                if modified:
                    raise ValueError(
                        f"group={group}, seed={seed}: locally unmodified G was required, but "
                        f"guard/clip/nonconcavity activates at regular checkpoints={modified}"
                    )

        for seed in selected:
            for outer in outers:
                source = by_seed[seed][outer]
                per_seed_row = {
                    "group": group,
                    "protocol_hash": protocol_hash,
                    "problem": "merton",
                    "n_assets": source["n_assets"],
                    "horizon": source["horizon"],
                    "gamma": source["gamma"],
                    "eval_margin": source["eval_margin"],
                    "eval_window_mode": source["eval_window_mode"],
                    "eval_w_min_requested": source[
                        "eval_w_min_requested"
                    ],
                    "eval_w_min_symmetric": source[
                        "eval_w_min_symmetric"
                    ],
                    "ev_y_min": source["ev_y_min"],
                    "ev_y_max": source["ev_y_max"],
                    "ev_w_min": source["ev_w_min"],
                    "ev_w_max": source["ev_w_max"],
                    "network_dtype": source["network_dtype"],
                    "seed": seed,
                    "market_seed": fixed_market_seed,
                    "checkpoint_outer_iter": source.get(
                        "checkpoint_outer_iter", int(outer) + 1
                    ),
                    "source_iter": outer,
                    "target_policy_iter": outer + 1,
                    "checkpoint_state_sha256": source.get(
                        "checkpoint_state_sha256", ""
                    ),
                    "e_input_X": source["e_input_X"],
                    "e_map_X": source["e_map_X"],
                    "rho_exact": source["rho_exact"],
                    "rho_sensitivity_envelope": source["rho_sensitivity_envelope"],
                    "floor": floors[seed],
                    "floor_basis": "late_input_error_scale_not_fd_floor",
                    "floor_multiple": floor_multiple,
                    "regular": int(outer in regular[seed]),
                    "common_regular": int(outer in common),
                    "refinement_status": source["refinement_status"],
                    "map_variant": source["map_variant"],
                    "local_map_unmodified_on_xfd": source["local_map_unmodified_on_xfd"],
                    "whole_space_map_claim": source["whole_space_map_claim"],
                    "analysis_mode": source["analysis_mode"],
                    "policy_extension": source["policy_extension"],
                    "map_definition": source["map_definition"],
                    "outside_collocation_count_fd": source[
                        "outside_collocation_count_fd"
                    ],
                    "outside_collocation_fraction_fd": source[
                        "outside_collocation_fraction_fd"
                    ],
                    "max_linear_residual": source["max_linear_residual"],
                    "linear_residual_tolerance": source[
                        "linear_residual_tolerance"
                    ],
                    "checkpoint_selection": source["checkpoint_selection"],
                    "contraction_status": source["contraction_status"],
                }
                per_seed_row.update({field: source[field] for field in ACTIVATION_FIELDS})
                per_seed_row.update({field: source[field] for field in ELLIPTICITY_FIELDS})
                per_seed_rows.append(per_seed_row)

        for outer in sorted(common):
            values = [float(by_seed[seed][outer]["rho_exact"]) for seed in selected]
            sensitivity_values = [
                float(by_seed[seed][outer]["rho_sensitivity_envelope"])
                for seed in selected
            ]
            mean, std, ci_low, ci_high = mean_std_ci(values)
            sensitivity_mean, sensitivity_std, _sensitivity_low, _sensitivity_high = mean_std_ci(
                sensitivity_values
            )
            statuses = [str(by_seed[seed][outer]["refinement_status"]) for seed in selected]
            unmodified = [
                int(by_seed[seed][outer]["local_map_unmodified_on_xfd"])
                for seed in selected
            ]
            activation_maxima = {
                f"{field}_max": max(
                    float(by_seed[seed][outer][field]) for seed in selected
                )
                for field in ACTIVATION_FIELDS
            }
            diffusion_variance_min = min(
                float(by_seed[seed][outer]["min_diffusion_variance"])
                for seed in selected
            )
            diffusion_variance_max = max(
                float(by_seed[seed][outer]["max_diffusion_variance"])
                for seed in selected
            )
            diffusion_variance_ev_min = min(
                float(by_seed[seed][outer]["min_diffusion_variance_ev"])
                for seed in selected
            )
            diffusion_variance_ev_max = max(
                float(by_seed[seed][outer]["max_diffusion_variance_ev"])
                for seed in selected
            )
            map_variants = sorted({
                str(by_seed[seed][outer]["map_variant"]) for seed in selected
            })
            contraction_statuses = sorted({
                str(by_seed[seed][outer]["contraction_status"]) for seed in selected
            })
            summary_row = {
                "group": group,
                "protocol_hash": protocol_hash,
                "problem": "merton",
                "n_assets": group_rows[0]["n_assets"],
                "horizon": group_rows[0]["horizon"],
                "gamma": group_rows[0]["gamma"],
                "eval_margin": group_rows[0]["eval_margin"],
                "eval_window_mode": group_rows[0]["eval_window_mode"],
                "eval_w_min_requested": group_rows[0][
                    "eval_w_min_requested"
                ],
                "eval_w_min_symmetric": group_rows[0][
                    "eval_w_min_symmetric"
                ],
                "ev_y_min": group_rows[0]["ev_y_min"],
                "ev_y_max": group_rows[0]["ev_y_max"],
                "ev_w_min": group_rows[0]["ev_w_min"],
                "ev_w_max": group_rows[0]["ev_w_max"],
                "network_dtype": group_rows[0]["network_dtype"],
                "market_seed": fixed_market_seed,
                "whole_space_map_claim": group_rows[0]["whole_space_map_claim"],
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "source_iter": outer,
                "target_policy_iter": outer + 1,
                "n_seeds": len(selected),
                "rho_exact_mean": mean,
                "rho_exact_std": std,
                "rho_exact_ci95_low": ci_low,
                "rho_exact_ci95_high": ci_high,
                "rho_sensitivity_envelope_mean": sensitivity_mean,
                "rho_sensitivity_envelope_std": sensitivity_std,
                "rho_sensitivity_envelope_max": max(sensitivity_values),
                "n_refinement_pass": statuses.count("pass"),
                "n_refinement_checked": sum(status not in {"not_checked", "variant"} for status in statuses),
                "n_locally_unmodified_map": sum(unmodified),
                "all_seed_locally_unmodified_map": int(sum(unmodified) == len(selected)),
                "all_seed_ratios_below_one": int(all(value < 1.0 for value in values)),
                "all_seed_sensitivity_envelope_below_one": int(
                    all(value < 1.0 for value in sensitivity_values)
                ),
                "diffusion_variance_min_across_seeds": diffusion_variance_min,
                "diffusion_variance_max_across_seeds": diffusion_variance_max,
                "diffusion_variance_ev_min_across_seeds": diffusion_variance_ev_min,
                "diffusion_variance_ev_max_across_seeds": diffusion_variance_ev_max,
                "ellipticity_tolerance": ellipticity_tolerance,
                "all_seed_diffusion_variance_above_tolerance": int(
                    min(diffusion_variance_min, diffusion_variance_ev_min)
                    > float(ellipticity_tolerance)
                ),
                "map_status": (
                    "all_locally_unmodified"
                    if sum(unmodified) == len(selected)
                    else "sampled_modification_active"
                ),
                "map_variants": "|".join(map_variants),
                "contraction_statuses": "|".join(contraction_statuses),
                "outside_collocation_fraction_fd_max": max(
                    float(by_seed[seed][outer][
                        "outside_collocation_fraction_fd"
                    ])
                    for seed in selected
                ),
                "max_linear_residual": max(
                    float(by_seed[seed][outer]["max_linear_residual"])
                    for seed in selected
                ),
                "linear_residual_tolerance": min(
                    float(by_seed[seed][outer]["linear_residual_tolerance"])
                    for seed in selected
                ),
            }
            summary_row.update(activation_maxima)
            summary_rows.append(summary_row)

        for outer in sorted(set(outers) - common):
            values = [
                float(by_seed[seed][outer]["rho_exact"])
                for seed in selected
                if math.isfinite(float(by_seed[seed][outer]["rho_exact"]))
            ]
            if not values:
                continue
            mean, std, _ci_low, _ci_high = mean_std_ci(values)
            floor_unmodified = [
                int(by_seed[seed][outer]["local_map_unmodified_on_xfd"])
                for seed in selected
            ]
            floor_activation_maxima = {
                f"{field}_max": max(
                    float(by_seed[seed][outer][field]) for seed in selected
                )
                for field in ACTIVATION_FIELDS
            }
            floor_row = {
                "group": group,
                "protocol_hash": protocol_hash,
                "problem": "merton",
                "n_assets": group_rows[0]["n_assets"],
                "horizon": group_rows[0]["horizon"],
                "gamma": group_rows[0]["gamma"],
                "eval_margin": group_rows[0]["eval_margin"],
                "eval_window_mode": group_rows[0]["eval_window_mode"],
                "eval_w_min_requested": group_rows[0][
                    "eval_w_min_requested"
                ],
                "eval_w_min_symmetric": group_rows[0][
                    "eval_w_min_symmetric"
                ],
                "ev_y_min": group_rows[0]["ev_y_min"],
                "ev_y_max": group_rows[0]["ev_y_max"],
                "ev_w_min": group_rows[0]["ev_w_min"],
                "ev_w_max": group_rows[0]["ev_w_max"],
                "network_dtype": group_rows[0]["network_dtype"],
                "market_seed": fixed_market_seed,
                "whole_space_map_claim": group_rows[0]["whole_space_map_claim"],
                "policy_extension": policy_extension,
                "map_definition": map_definition,
                "source_iter": outer,
                "target_policy_iter": outer + 1,
                "n_finite_seeds": len(values),
                "n_requested_seeds": len(selected),
                "rho_exact_mean": mean,
                "rho_exact_std": std,
                "floor_dominated": 1,
                "map_status": (
                    "all_locally_unmodified"
                    if sum(floor_unmodified) == len(selected)
                    else "sampled_modification_active"
                ),
                "map_variants": "|".join(sorted({
                    str(by_seed[seed][outer]["map_variant"]) for seed in selected
                })),
                "contraction_statuses": "|".join(sorted({
                    str(by_seed[seed][outer]["contraction_status"]) for seed in selected
                })),
            }
            floor_row.update(floor_activation_maxima)
            floor_summary_rows.append(floor_row)

        seed_maxima = [max(float(by_seed[seed][outer]["rho_exact"]) for outer in regular[seed]) for seed in selected]
        seed_sensitivity_maxima = [
            max(
                float(by_seed[seed][outer]["rho_sensitivity_envelope"])
                for outer in regular[seed]
            )
            for seed in selected
        ]
        mean_max, std_max, ci_low_max, ci_high_max = mean_std_ci(seed_maxima)
        common_means = [
            float(np.mean([float(by_seed[seed][outer]["rho_exact"]) for seed in selected]))
            for outer in common
        ]
        common_sensitivity_means = [
            float(np.mean([
                float(by_seed[seed][outer]["rho_sensitivity_envelope"])
                for seed in selected
            ]))
            for outer in common
        ]
        all_regular_locally_unmodified = all(
            int(by_seed[seed][outer]["local_map_unmodified_on_xfd"])
            for seed in selected
            for outer in regular[seed]
        )
        regular_source_rows = [
            by_seed[seed][outer]
            for seed in selected
            for outer in regular[seed]
        ]
        worst_activation_maxima = {
            f"{field}_max": max(float(row[field]) for row in regular_source_rows)
            for field in ACTIVATION_FIELDS
        }
        regular_diffusion_variance_min = min(
            min(
                float(row["min_diffusion_variance"]),
                float(row["min_diffusion_variance_ev"]),
            )
            for row in regular_source_rows
        )
        regular_diffusion_variance_max = max(
            max(
                float(row["max_diffusion_variance"]),
                float(row["max_diffusion_variance_ev"]),
            )
            for row in regular_source_rows
        )
        worst_row = {
            "group": group,
            "protocol_hash": protocol_hash,
            "problem": "merton",
            "n_assets": group_rows[0]["n_assets"],
            "horizon": group_rows[0]["horizon"],
            "gamma": group_rows[0]["gamma"],
            "eval_margin": group_rows[0]["eval_margin"],
            "eval_window_mode": group_rows[0]["eval_window_mode"],
            "eval_w_min_requested": group_rows[0][
                "eval_w_min_requested"
            ],
            "eval_w_min_symmetric": group_rows[0][
                "eval_w_min_symmetric"
            ],
            "ev_y_min": group_rows[0]["ev_y_min"],
            "ev_y_max": group_rows[0]["ev_y_max"],
            "ev_w_min": group_rows[0]["ev_w_min"],
            "ev_w_max": group_rows[0]["ev_w_max"],
            "network_dtype": group_rows[0]["network_dtype"],
            "market_seed": fixed_market_seed,
            "whole_space_map_claim": group_rows[0]["whole_space_map_claim"],
            "policy_extension": policy_extension,
            "map_definition": map_definition,
            "floor_multiple": floor_multiple,
            "n_seeds": len(selected),
            "n_common_iterations": len(common),
            "max_of_seed_mean_rho_exact": max(common_means),
            "mean_of_seed_max_rho_exact": mean_max,
            "std_of_seed_max_rho_exact": std_max,
            "ci95_low_of_seed_max": ci_low_max,
            "ci95_high_of_seed_max": ci_high_max,
            "max_of_seed_mean_rho_sensitivity_envelope": max(common_sensitivity_means),
            "max_seed_rho_sensitivity_envelope": max(seed_sensitivity_maxima),
            "regular_diffusion_variance_min": regular_diffusion_variance_min,
            "regular_diffusion_variance_max": regular_diffusion_variance_max,
            "ellipticity_tolerance": ellipticity_tolerance,
            "all_regular_diffusion_variance_above_tolerance": int(
                regular_diffusion_variance_min > float(ellipticity_tolerance)
            ),
            "all_regular_locally_unmodified_map": int(all_regular_locally_unmodified),
            "map_status": (
                "all_locally_unmodified"
                if all_regular_locally_unmodified
                else "sampled_modification_active"
            ),
            "map_variants": "|".join(sorted({
                str(row["map_variant"]) for row in regular_source_rows
            })),
            "contraction_statuses": "|".join(sorted({
                str(row["contraction_status"]) for row in regular_source_rows
            })),
            "outside_collocation_fraction_fd_max": max(
                float(row["outside_collocation_fraction_fd"])
                for row in regular_source_rows
            ),
            "max_linear_residual": max(
                float(row["max_linear_residual"])
                for row in regular_source_rows
            ),
            "linear_residual_tolerance": min(
                float(row["linear_residual_tolerance"])
                for row in regular_source_rows
            ),
        }
        worst_row.update(worst_activation_maxima)
        worst_rows.append(worst_row)

    output.mkdir(parents=True, exist_ok=True)
    activation_max_fields = [f"{field}_max" for field in ACTIVATION_FIELDS]
    evaluation_window_fields = [
        "eval_margin", "eval_window_mode", "eval_w_min_requested",
        "eval_w_min_symmetric", "ev_y_min", "ev_y_max", "ev_w_min",
        "ev_w_max",
    ]
    per_seed_fields = [
        "group", "protocol_hash", "problem", "n_assets", "horizon", "gamma",
        *evaluation_window_fields,
        "network_dtype", "seed", "market_seed", "checkpoint_outer_iter", "source_iter",
        "target_policy_iter", "checkpoint_state_sha256",
        "e_input_X", "e_map_X", "rho_exact", "rho_sensitivity_envelope",
        "floor", "floor_basis", "floor_multiple", "regular", "common_regular", "refinement_status",
        "map_variant", "local_map_unmodified_on_xfd", "whole_space_map_claim",
        "analysis_mode", "policy_extension", "map_definition",
        "outside_collocation_count_fd", "outside_collocation_fraction_fd",
        "max_linear_residual", "linear_residual_tolerance",
        "checkpoint_selection", "contraction_status", *ACTIVATION_FIELDS,
        *ELLIPTICITY_FIELDS,
    ]
    summary_fields = [
        "group", "protocol_hash", "problem", "n_assets", "horizon", "gamma",
        *evaluation_window_fields,
        "network_dtype", "market_seed", "whole_space_map_claim",
        "policy_extension", "map_definition", "source_iter",
        "target_policy_iter", "n_seeds",
        "rho_exact_mean", "rho_exact_std", "rho_exact_ci95_low", "rho_exact_ci95_high",
        "rho_sensitivity_envelope_mean", "rho_sensitivity_envelope_std",
        "rho_sensitivity_envelope_max",
        "n_refinement_pass", "n_refinement_checked", "n_locally_unmodified_map",
        "all_seed_locally_unmodified_map",
        "all_seed_ratios_below_one", "all_seed_sensitivity_envelope_below_one",
        "diffusion_variance_min_across_seeds", "diffusion_variance_max_across_seeds",
        "diffusion_variance_ev_min_across_seeds", "diffusion_variance_ev_max_across_seeds",
        "ellipticity_tolerance", "all_seed_diffusion_variance_above_tolerance",
        "outside_collocation_fraction_fd_max", "max_linear_residual",
        "linear_residual_tolerance",
        "map_status", "map_variants", "contraction_statuses", *activation_max_fields,
    ]
    worst_fields = [
        "group", "protocol_hash", "problem", "n_assets", "horizon", "gamma",
        *evaluation_window_fields,
        "network_dtype", "market_seed", "whole_space_map_claim",
        "policy_extension", "map_definition", "floor_multiple",
        "n_seeds", "n_common_iterations",
        "max_of_seed_mean_rho_exact", "mean_of_seed_max_rho_exact",
        "std_of_seed_max_rho_exact", "ci95_low_of_seed_max", "ci95_high_of_seed_max",
        "max_of_seed_mean_rho_sensitivity_envelope",
        "max_seed_rho_sensitivity_envelope",
        "regular_diffusion_variance_min", "regular_diffusion_variance_max",
        "ellipticity_tolerance", "all_regular_diffusion_variance_above_tolerance",
        "all_regular_locally_unmodified_map",
        "outside_collocation_fraction_fd_max", "max_linear_residual",
        "linear_residual_tolerance",
        "map_status", "map_variants", "contraction_statuses", *activation_max_fields,
    ]
    # Exact PI-map evidence is a separate FD experiment, not the empirical
    # relative-L2 Figure 2.  Keep the artifact names unambiguous.
    write_csv(output / "exact_map_ratios_by_seed.csv", per_seed_rows, per_seed_fields)
    write_csv(output / "exact_map_ratio_summary.csv", summary_rows, summary_fields)
    write_csv(output / "exact_map_floor_summary.csv", floor_summary_rows, [
        "group", "protocol_hash", "problem", "n_assets", "horizon", "gamma",
        *evaluation_window_fields,
        "network_dtype", "market_seed", "whole_space_map_claim",
        "policy_extension", "map_definition", "source_iter",
        "target_policy_iter", "n_finite_seeds",
        "n_requested_seeds", "rho_exact_mean", "rho_exact_std", "floor_dominated",
        "map_status", "map_variants", "contraction_statuses", *activation_max_fields,
    ])
    write_csv(output / "exact_map_worst_summary.csv", worst_rows, worst_fields)
    atomic_json(output / "exact_map_aggregate_config.json", {
        "result_dirs": [str(path.resolve()) for path in result_dirs],
        "expected_seeds": list(expected_seeds),
        "min_seeds": min_seeds,
        "floor_multiple": floor_multiple,
        "allow_incomplete": allow_incomplete,
        "allow_unverified": allow_unverified,
        "require_locally_unmodified_map": require_locally_unmodified_map,
        "ellipticity_tolerance": ellipticity_tolerance,
        "allow_degenerate_diffusion": allow_degenerate_diffusion,
        "allow_neural_extrapolation": allow_neural_extrapolation,
        "plot": {
            "enabled": bool(make_plot),
            "formats": list(resolved_plot_formats),
            "x_scale": "linear",
            "y_scale": "log",
            "x_label": "Iteration",
            "y_label": r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$",
            "series_label": r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$",
            "line_style": "solid",
            "aggregation_band": "pointwise seed mean plus/minus one sample SD",
            "floor_filter_used": bool(float(floor_multiple) > 0.0),
            "eligibility_definition": (
                "all finite exact-map ratios"
                if float(floor_multiple) == 0.0
                else (
                    "strict e_n > floor_multiple times the late "
                    "neural-input-error scale"
                )
            ),
            # Retained as an empty compatibility field. Eligibility is an
            # audit/filtering property, not the plotted series name.
            "eligible_label": "",
            "floor_dominated_label": (
                "Floor-dominated" if float(floor_multiple) > 0.0 else ""
            ),
            "contraction_threshold": 1.0,
            "legend_location": "inside upper right",
            "legend_bbox_to_anchor": [0.98, 0.92],
            "dpi": int(dpi),
            "fig_width": float(fig_width),
            "fig_height": float(fig_height),
            "font_size": float(font_size),
            "font_family": str(font_family),
            "line_width": float(line_width),
            "line_alpha": float(line_alpha),
            "marker_size": float(marker_size),
            "band_alpha": float(band_alpha),
            "grid_alpha": float(grid_alpha),
            "floor_alpha": float(floor_alpha),
            "bbox_inches": str(bbox_inches),
        },
        "evaluation_windows": sorted({
            (
                str(row["eval_window_mode"]),
                float(row["ev_w_min"]),
                float(row["ev_w_max"]),
            )
            for row in records
        }),
    })

    if make_plot:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb

        def blend_on_white(color: Any, alpha: float) -> Tuple[float, float, float]:
            rgb = np.asarray(to_rgb(color), dtype=float)
            return tuple(float(value) for value in alpha * rgb + (1.0 - alpha))

        groups = sorted({str(row["group"]) for row in summary_rows})
        rc_params: Dict[str, Any] = {"font.size": float(font_size)}
        if str(font_family).strip():
            rc_params["font.family"] = str(font_family).strip()
        save_bbox = "tight" if bbox_inches == "tight" else None

        for current_format in resolved_plot_formats:
            eps_compatible = current_format == "eps"
            with matplotlib.rc_context(rc_params):
                fig, ax = plt.subplots(
                    figsize=(float(fig_width), float(fig_height))
                )
                blue = "#0072B2"
                for group in groups:
                    plot_rows = sorted(
                        [
                            row for row in summary_rows
                            if str(row["group"]) == group
                        ],
                        key=lambda row: int(row["source_iter"]),
                    )
                    x = np.asarray(
                        [row["source_iter"] for row in plot_rows], dtype=float
                    )
                    mean = np.asarray(
                        [row["rho_exact_mean"] for row in plot_rows], dtype=float
                    )
                    std = np.asarray(
                        [row["rho_exact_std"] for row in plot_rows], dtype=float
                    )
                    (line,) = ax.plot(
                        x,
                        mean,
                        color=blue,
                        linestyle="-",
                        marker="o",
                        markersize=float(marker_size),
                        linewidth=float(line_width),
                        alpha=1.0 if eps_compatible else float(line_alpha),
                        label=(
                            r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$"
                            if group == groups[0]
                            else "_nolegend_"
                        ),
                    )
                    line_color = line.get_color()
                    if eps_compatible and line_alpha < 1.0:
                        line.set_color(blend_on_white(line_color, line_alpha))
                    lower = np.where(mean - std > 0.0, mean - std, np.nan)
                    ax.fill_between(
                        x,
                        lower,
                        mean + std,
                        color=(
                            blend_on_white(line_color, band_alpha)
                            if eps_compatible
                            else line_color
                        ),
                        alpha=1.0 if eps_compatible else float(band_alpha),
                    )
                    floor_rows = sorted(
                        [
                            row for row in floor_summary_rows
                            if str(row["group"]) == group
                        ],
                        key=lambda row: int(row["source_iter"]),
                    )
                    if floor_rows:
                        ax.scatter(
                            [row["source_iter"] for row in floor_rows],
                            [row["rho_exact_mean"] for row in floor_rows],
                            marker="x",
                            s=24,
                            color=(
                                blend_on_white("0.55", floor_alpha)
                                if eps_compatible
                                else "0.55"
                            ),
                            alpha=1.0 if eps_compatible else float(floor_alpha),
                            label=(
                                "Floor-dominated"
                                if group == groups[0]
                                else "_nolegend_"
                            ),
                        )
                ax.axhline(
                    1.0,
                    color="black",
                    linestyle="--",
                    linewidth=max(1.0, 0.8 * float(line_width)),
                    label="contraction threshold",
                )
                ax.set_yscale("log", nonpositive="clip")
                ax.set_xlabel("Iteration")
                ax.set_ylabel(r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$")
                if eps_compatible:
                    ax.grid(
                        True,
                        color=blend_on_white("black", grid_alpha),
                        alpha=1.0,
                        which="both",
                        linewidth=0.6,
                    )
                else:
                    ax.grid(
                        True,
                        alpha=float(grid_alpha),
                        which="both",
                        linewidth=0.6,
                    )
                ax.tick_params(axis="both", labelsize=0.9 * float(font_size))
                legend_font: Dict[str, Any] = {
                    "size": 0.8 * float(font_size)
                }
                if str(font_family).strip():
                    legend_font["family"] = str(font_family).strip()
                ax.legend(
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.92),
                    borderaxespad=0.0,
                    prop=legend_font,
                )
                fig.tight_layout()
                fig.savefig(
                    output / f"exact_map_contraction.{current_format}",
                    dpi=int(dpi),
                    bbox_inches=save_bbox,
                )
                plt.close(fig)


def discover_result_dirs(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    return sorted(
        path.parent
        for path in root.rglob("_SUCCESS_EXACT_MAP")
        if (
            "stale_outputs" not in path.relative_to(root).parts
            and (path.parent / "exact_map_ratios.csv").is_file()
            and (path.parent / "exact_map_config.json").is_file()
            and (path.parent / "exact_map_status.json").is_file()
        )
    )


def discover_run_dirs(root: Path, run_name_regex: str = "") -> List[Path]:
    pattern = re.compile(run_name_regex) if run_name_regex else None
    out: List[Path] = []
    for config_path in root.expanduser().resolve().rglob("config.json"):
        run_dir = config_path.parent
        if pattern and not pattern.search(run_dir.name):
            continue
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                cfg = _merged_config(json.load(handle))
            model_type = str(_pick(cfg, ("model_type",), default="")).lower()
            has_merton_fields = all(
                _pick(cfg, aliases, default=None) is not None
                for aliases in (
                    ("gamma", "gamma_risk"),
                    ("rho_discount", "discount", "rho"),
                    ("w_min", "W_min", "x_min"),
                    ("w_max", "W_max", "x_max"),
                )
            )
            if model_type in {"pipinn", "pi-pinn", "pinn-pi"} and has_merton_fields:
                out.append(run_dir)
        except Exception:
            continue
    return sorted(set(out))


def run_self_test() -> int:
    from merton_exact_map_core import analytic_optimal_policy

    problem = MertonProblem(
        horizon=1.0,
        y_min=float(np.log(0.05)),
        y_max=float(np.log(4.0)),
        gamma=2.0,
        discount=0.04,
        bequest=1.0,
        risk_free=0.03,
        mu_excess=np.asarray([0.08]),
        sigma=np.asarray([[0.04]]),
    )
    # Same open-terminal convention as Q_ev; the retained nodes are nested in
    # the 80- and 160-step self-test grids.
    tau = np.linspace(0.0, 1.0, 41)[1:]
    y = np.linspace(np.log(0.1), np.log(2.0), 61)
    tt, yy = np.meshgrid(tau, y, indexing="ij")
    def refinement_errors(
        test_problem: MertonProblem,
        policy: Any,
        reference: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> List[float]:
        errors: List[float] = []
        for ny in (81, 161):
            solution = solve_frozen_policy(
                test_problem,
                policy,
                FDGrid(test_problem.y_min, test_problem.y_max, ny=ny, nt=ny - 1),
                boundary="robin",
            )
            metric = x_norm_components(
                *evaluate_fd_bundle(solution, tau, y), reference, yy
            )
            errors.append(float(metric["x_norm"]))
        return errors

    optimal_errors = refinement_errors(
        problem,
        analytic_optimal_policy(problem),
        crra_closed_form(problem, tt, yy),
    )
    optimal_ratio = optimal_errors[1] / optimal_errors[0]
    print(
        "[self-test:optimal] "
        f"X-error coarse={optimal_errors[0]:.6e}, "
        f"fine={optimal_errors[1]:.6e}, ratio={optimal_ratio:.4f}"
    )
    if not (optimal_errors[1] < optimal_errors[0] and optimal_ratio < 0.4):
        raise RuntimeError("optimal-policy FD check did not show refinement convergence")

    # Independent manufactured solution: neither the frozen portfolio nor
    # consumption ratio satisfies the Merton first-order conditions, and the
    # terminal bequest coefficient is deliberately non-unit.  This catches
    # source/drift signs and remaining-time orientation that an optimal-only
    # check could accidentally share with the reference formula.
    nonoptimal_problem = MertonProblem(
        horizon=problem.horizon,
        y_min=problem.y_min,
        y_max=problem.y_max,
        gamma=problem.gamma,
        discount=problem.discount,
        bequest=2.25,
        risk_free=problem.risk_free,
        mu_excess=problem.mu_excess,
        sigma=problem.sigma,
    )
    portfolio = np.asarray([0.35])
    consumption_ratio = 0.12
    nonoptimal_errors = refinement_errors(
        nonoptimal_problem,
        constant_proportional_policy(
            nonoptimal_problem, portfolio, consumption_ratio
        ),
        constant_proportional_closed_form(
            nonoptimal_problem,
            tt,
            yy,
            portfolio,
            consumption_ratio,
        ),
    )
    nonoptimal_ratio = nonoptimal_errors[1] / nonoptimal_errors[0]
    print(
        "[self-test:nonoptimal-proportional] "
        f"X-error coarse={nonoptimal_errors[0]:.6e}, "
        f"fine={nonoptimal_errors[1]:.6e}, ratio={nonoptimal_ratio:.4f}"
    )
    if not (
        nonoptimal_errors[1] < nonoptimal_errors[0]
        and nonoptimal_ratio < 0.4
    ):
        raise RuntimeError(
            "nonoptimal proportional-policy FD check did not show refinement convergence"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent finite-difference evaluation of the Merton exact PI map."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", action="append", default=[], help="Structured Merton PI-PINN run; repeatable.")
    source.add_argument("--out-root", default="", help="Discover structured Merton PI-PINN runs recursively.")
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--weight-dir", default="", help="Override weight directory (single --run-dir only).")
    parser.add_argument("--checkpoint", action="append", default=[], help="Explicit value_net_iterNNNN.pt (single run only).")
    parser.add_argument(
        "--checkpoints",
        default="",
        help=(
            "Exploratory comma-separated outer indices, e.g. 1,5,10,15,20. "
            "Arbitrary sparse subsets require --skip-e4."
        ),
    )
    parser.add_argument("--output", default="", help="Single-run output; default <run-dir>/exact_map_fd.")
    parser.add_argument("--aggregate-output", default="", help="Default <out-root>/exact_map_paper.")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--result-dir", action="append", default=[], help="Existing exact_map_fd directory for --aggregate-only.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run PyTorch-free optimal and nonoptimal proportional-policy "
            "analytic FD refinement tests and exit."
        ),
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--network-dtype",
        choices=("training", "float32", "float64"),
        default="training",
        help="Use the recorded training dtype by default; float64 is an explicit sensitivity audit.",
    )
    parser.add_argument("--network-time-coordinate", choices=("t", "tau"), default="")
    parser.add_argument(
        "--policy-mode",
        choices=("one-sided", "legacy-signed", "log-concavity", "wealth-concavity", "none"),
        default="",
        help=(
            "Audited override; default derives the current trainer's one-sided "
            "numerator/curvature rule from its contract."
        ),
    )
    parser.add_argument("--eval-margin", type=float, default=None)
    parser.add_argument(
        "--eval-w-mins",
        default="",
        help=(
            "Optional absolute lower endpoint(s) for Q_ev in wealth "
            "coordinates, as a comma/space-separated list. Each value must "
            "remain within the saved training wealth interval and below the "
            "upper endpoint implied by --eval-margin. The upper endpoint, FD "
            "domains, checkpoints, and trained network are unchanged. Every "
            "explicit value is written to a separate "
            "eval_w_min_<value>/ result directory."
        ),
    )
    parser.add_argument(
        "--skip-e4",
        action="store_true",
        help=(
            "Exact-map-only pilot mode. It permits sparse --checkpoints and "
            "is never paper-aggregation eligible."
        ),
    )

    parser.add_argument("--base-ny", type=int, default=401)
    parser.add_argument("--base-nt", type=int, default=400)
    parser.add_argument("--eval-ny", type=int, default=201)
    parser.add_argument("--grid-factors", default="1,2", help="h and h/2 by default.")
    parser.add_argument(
        "--fd-margins",
        default="-1.0,-0.5",
        help=(
            "FD domains as half-width margin fractions of the training log-wealth interval. "
            "Negative values enlarge beyond training; the paper default uses 2x and 1.5x widths."
        ),
    )
    parser.add_argument(
        "--boundaries",
        default="robin,exact-dirichlet",
        help=(
            "Primary first. Use Robin as the paper primary; exact-dirichlet is "
            "a compatibility name for an optimal-reference Dirichlet "
            "sensitivity audit, not an exact neural-policy boundary oracle."
        ),
    )
    parser.add_argument(
        "--verify-checkpoints",
        default="all",
        help=(
            "Extra h/domain/boundary variants for exact ratios and adjacent "
            "E4 defects: all, none, first/middle/last, or explicit outer "
            "indices. The first available defect is always verified (delta_0 "
            "for standard runs, delta_1 for E6 target branches); paper E4 "
            "aggregation requires first/last/worst defect evidence."
        ),
    )
    parser.add_argument(
        "--drift-scheme",
        choices=("central", "adaptive", "monotone"),
        default="central",
    )
    parser.add_argument("--peclet-limit", type=float, default=1.0)
    parser.add_argument("--theta-method", type=float, default=0.5)
    parser.add_argument(
        "--rannacher-steps",
        type=int,
        default=2,
        help=(
            "Initial full-dt backward-Euler damping steps; this is not the "
            "classical two-half-step Rannacher construction."
        ),
    )
    parser.add_argument(
        "--linear-residual-tolerance",
        type=float,
        default=1e-8,
        help="Hard upper bound for every normalized tridiagonal solve residual.",
    )
    parser.add_argument(
        "--boundary-condition-limit",
        type=float,
        default=1e12,
        help="Hard upper bound for the lateral-boundary elimination condition number.",
    )
    parser.add_argument(
        "--solver-ellipticity-tolerance",
        type=float,
        default=0.0,
        help=(
            "Require every FD-sampled pi^T Sigma pi to be strictly above this "
            "solver-level threshold."
        ),
    )
    parser.add_argument(
        "--policy-extension",
        choices=POLICY_EXTENSIONS,
        default="boundary-projection",
        help=(
            "Define the frozen policy outside saved nominal collocation "
            "bounds. boundary-projection is the finite-domain primary; "
            "neural-extrapolation is sensitivity-only."
        ),
    )
    parser.add_argument("--denominator-tolerance", type=float, default=1e-12)
    parser.add_argument("--refinement-abs-tolerance", type=float, default=1e-2)
    parser.add_argument("--refinement-rel-tolerance", type=float, default=2e-2)

    parser.add_argument(
        "--expected-seeds",
        default="",
        help=(
            "Optional expected training-seed set. The empty default accepts the "
            "seeds discovered in the selected runs; pass the paper seed list "
            "explicitly when strict completeness validation is required."
        ),
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help=(
            "Minimum discovered/selected seed count required per exact-map "
            "aggregation group (default: 2). This is independent of the "
            "optional exact-set check from --expected-seeds."
        ),
    )
    parser.add_argument(
        "--floor-multiple", type=float, default=0.0,
        help=(
            "0 (paper default) retains every finite checkpoint. A positive value enables "
            "an exploratory cutoff relative to the late neural input-error scale; that "
            "scale is not an FD discretization floor."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help=(
            "Exploratory aggregation only: include checkpoints whose "
            "h/domain/boundary sensitivity audit did not pass."
        ),
    )
    parser.add_argument(
        "--require-locally-unmodified-map",
        action="store_true",
        help=(
            "Strict optional gate: fail aggregation if guard, clipping, or "
            "nonconcavity activates on a regular sampled enlarged domain."
        ),
    )
    parser.add_argument(
        "--ellipticity-tolerance", type=float, default=0.0,
        help=(
            "Require sampled pi^T Sigma pi to exceed this value on both the FD and "
            "evaluation domains during paper aggregation."
        ),
    )
    parser.add_argument(
        "--allow-degenerate-diffusion", action="store_true",
        help="Exploratory only: do not reject regular rows at/below the ellipticity tolerance.",
    )
    parser.add_argument(
        "--allow-neural-extrapolation",
        action="store_true",
        help=(
            "Exploratory aggregation only: permit raw neural-extrapolation "
            "finite-domain maps instead of the boundary-projected primary."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--formats",
        default="",
        help=(
            "Comma/space-separated figure formats, e.g. png,eps. "
            "Default: png."
        ),
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_PLOT_FORMATS,
        default="",
        help=(
            "Legacy single-format option. Use --formats for new commands; "
            "it cannot be combined with --formats."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=6.0)
    parser.add_argument("--fig-height", type=float, default=4.0)
    parser.add_argument("--font-size", type=float, default=12.0)
    parser.add_argument("--font-family", default="")
    parser.add_argument("--line-width", type=float, default=1.8)
    parser.add_argument("--line-alpha", type=float, default=1.0)
    parser.add_argument("--marker-size", type=float, default=4.0)
    parser.add_argument("--band-alpha", type=float, default=0.18)
    parser.add_argument("--grid-alpha", type=float, default=0.3)
    parser.add_argument("--floor-alpha", type=float, default=0.8)
    parser.add_argument(
        "--bbox-inches",
        choices=("tight", "standard"),
        default="tight",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()

    try:
        plot_formats = resolve_plot_formats(args.formats, args.format)
    except ValueError as exc:
        parser.error(str(exc))
    plot_kwargs = {
        "plot_formats": plot_formats,
        "dpi": args.dpi,
        "fig_width": args.fig_width,
        "fig_height": args.fig_height,
        "font_size": args.font_size,
        "font_family": args.font_family,
        "line_width": args.line_width,
        "line_alpha": args.line_alpha,
        "marker_size": args.marker_size,
        "band_alpha": args.band_alpha,
        "grid_alpha": args.grid_alpha,
        "floor_alpha": args.floor_alpha,
        "bbox_inches": args.bbox_inches,
    }
    try:
        validate_plot_options(
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            font_size=args.font_size,
            line_width=args.line_width,
            line_alpha=args.line_alpha,
            marker_size=args.marker_size,
            band_alpha=args.band_alpha,
            grid_alpha=args.grid_alpha,
            floor_alpha=args.floor_alpha,
            bbox_inches=args.bbox_inches,
        )
    except ValueError as exc:
        parser.error(str(exc))
    expected_seeds = parse_seed_spec(args.expected_seeds)
    if args.aggregate_only:
        if args.result_dir:
            result_dirs = [Path(value).expanduser().resolve() for value in args.result_dir]
        elif args.out_root:
            result_dirs = discover_result_dirs(Path(args.out_root))
        else:
            parser.error("--aggregate-only requires --result-dir or --out-root")
        if not result_dirs:
            raise SystemExit("no successful exact-map result directories were found")
        aggregate_output = (
            Path(args.aggregate_output).expanduser().resolve()
            if args.aggregate_output
            else (
                Path(args.out_root).expanduser().resolve() / "exact_map_paper"
                if args.out_root
                else Path.cwd() / "exact_map_paper"
            )
        )
        aggregate_exact_map(
            result_dirs,
            aggregate_output,
            expected_seeds=expected_seeds,
            floor_multiple=args.floor_multiple,
            allow_incomplete=args.allow_incomplete,
            allow_unverified=args.allow_unverified,
            require_locally_unmodified_map=args.require_locally_unmodified_map,
            make_plot=not args.no_plot,
            min_seeds=args.min_seeds,
            ellipticity_tolerance=args.ellipticity_tolerance,
            allow_degenerate_diffusion=args.allow_degenerate_diffusion,
            allow_neural_extrapolation=args.allow_neural_extrapolation,
            **plot_kwargs,
        )
        print(f"[done] exact-map aggregate: {aggregate_output}")
        return 0

    if args.run_dir:
        run_dirs = [Path(value).expanduser().resolve() for value in args.run_dir]
    elif args.out_root:
        run_dirs = discover_run_dirs(Path(args.out_root), args.run_name_regex)
    else:
        parser.error("provide --run-dir, --out-root, --aggregate-only, or --self-test")
    if not run_dirs:
        raise SystemExit("no structured Merton PI-PINN runs were found")
    if args.checkpoint and args.checkpoints:
        parser.error("--checkpoint and --checkpoints are mutually exclusive")
    if (
        args.weight_dir or args.checkpoint or args.checkpoints or args.output
    ) and len(run_dirs) != 1:
        parser.error(
            "--weight-dir, --checkpoint, --checkpoints, and --output require "
            "exactly one --run-dir"
        )

    grid_factors = parse_int_list(args.grid_factors)
    fd_margins = parse_float_list(args.fd_margins)
    boundaries = [item.strip() for item in args.boundaries.split(",") if item.strip()]
    if args.eval_w_mins.strip():
        eval_w_min_values = parse_float_list(args.eval_w_mins)
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in eval_w_min_values
        ):
            parser.error("--eval-w-mins values must be finite and positive")
        if len(set(eval_w_min_values)) != len(eval_w_min_values):
            parser.error("--eval-w-mins values must be unique")
        output_names = [
            eval_w_min_output_name(value) for value in eval_w_min_values
        ]
        if len(set(output_names)) != len(output_names):
            parser.error(
                "--eval-w-mins values map to non-unique output directory names"
            )
        eval_w_min_overrides: List[Optional[float]] = list(eval_w_min_values)
    else:
        eval_w_min_overrides = [None]
    result_dirs: List[Path] = []
    for run_dir in run_dirs:
        output_base = (
            Path(args.output).expanduser().resolve()
            if args.output
            else run_dir / "exact_map_fd"
        )
        for eval_w_min_override in eval_w_min_overrides:
            output = (
                output_base
                if eval_w_min_override is None
                else output_base / eval_w_min_output_name(
                    eval_w_min_override
                )
            )
            result_dirs.append(output)
            try:
                run = load_run_spec(
                    run_dir,
                    explicit_checkpoints=[
                        Path(value) for value in args.checkpoint
                    ],
                    weight_dir_override=(
                        Path(args.weight_dir) if args.weight_dir else None
                    ),
                    policy_mode_override=args.policy_mode,
                    time_coordinate_override=args.network_time_coordinate,
                    eval_margin_override=args.eval_margin,
                    eval_w_min_override=eval_w_min_override,
                    network_dtype=args.network_dtype,
                )
                if args.checkpoints:
                    requested = set(parse_int_list(args.checkpoints))
                    available = {
                        int(outer): path for outer, path in run.checkpoints
                    }
                    missing = sorted(requested - set(available))
                    if missing:
                        raise ValueError(
                            "requested checkpoint outer indices are "
                            f"unavailable: {missing}"
                        )
                    run.checkpoints = [
                        (outer, available[outer])
                        for outer in sorted(requested)
                    ]
                    run.checkpoint_selection = "explicit_subset"
                evaluate_run(
                    run,
                    output,
                    device=args.device,
                    base_ny=args.base_ny,
                    base_nt=args.base_nt,
                    eval_ny=args.eval_ny,
                    grid_factors=grid_factors,
                    fd_margins=fd_margins,
                    boundaries=boundaries,
                    verify_checkpoints=args.verify_checkpoints,
                    drift_scheme=args.drift_scheme,
                    peclet_limit=args.peclet_limit,
                    theta_method=args.theta_method,
                    rannacher_steps=args.rannacher_steps,
                    denominator_tolerance=args.denominator_tolerance,
                    refinement_abs_tolerance=args.refinement_abs_tolerance,
                    refinement_rel_tolerance=args.refinement_rel_tolerance,
                    policy_extension=args.policy_extension,
                    linear_residual_tolerance=(
                        args.linear_residual_tolerance
                    ),
                    boundary_condition_limit=args.boundary_condition_limit,
                    solver_ellipticity_tolerance=(
                        args.solver_ellipticity_tolerance
                    ),
                    skip_e4=args.skip_e4,
                )
                print(f"[done] exact-map run: {output}")
            except Exception as exc:
                output.mkdir(parents=True, exist_ok=True)
                # Preserve any prior or partially generated evidence under a
                # noncanonical audit path.  A failed attempt therefore
                # exposes no stale CSV/NPZ that could be mistaken for output.
                _quarantine_previous_outputs(output)
                marker = output / "_FAILED_EXACT_MAP"
                marker.write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
                atomic_json(output / "exact_map_status.json", {
                    "status": "failed",
                    "eval_w_min_requested": eval_w_min_override,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                raise

    should_auto_aggregate = len(run_dirs) > 1 or bool(args.out_root)
    if should_auto_aggregate and not args.skip_e4:
        aggregate_output = (
            Path(args.aggregate_output).expanduser().resolve()
            if args.aggregate_output
            else Path(args.out_root).expanduser().resolve() / "exact_map_paper"
        )
        aggregate_exact_map(
            result_dirs,
            aggregate_output,
            expected_seeds=expected_seeds,
            floor_multiple=args.floor_multiple,
            allow_incomplete=args.allow_incomplete,
            allow_unverified=args.allow_unverified,
            require_locally_unmodified_map=args.require_locally_unmodified_map,
            make_plot=not args.no_plot,
            min_seeds=args.min_seeds,
            ellipticity_tolerance=args.ellipticity_tolerance,
            allow_degenerate_diffusion=args.allow_degenerate_diffusion,
            allow_neural_extrapolation=args.allow_neural_extrapolation,
            **plot_kwargs,
        )
        print(f"[done] exact-map aggregate: {aggregate_output}")
    elif args.skip_e4 and should_auto_aggregate:
        print(
            "[pilot] --skip-e4 outputs were generated without paper aggregation"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
