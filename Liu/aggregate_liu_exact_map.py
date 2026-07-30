#!/usr/bin/env python3
"""Aggregate independent Liu M=1 exact-map and E4 finite-difference audits.

The aggregator is intentionally strict about experiment identity and support.
It never drops a seed/iteration with an undefined exact-map denominator and it
never combines different market snapshots, FD protocols, or checkpoint
schedules into a paper statistic.
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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import t as student_t

from liu_exact_map_fd import (
    BOUNDARY_SENSITIVITY_ROLE,
    REFINEMENT_SCOPE,
    summarize_e4_refinement,
)


EXACT_INPUT = "exact_map_ratios.csv"
E4_INPUT = "e4_approximation_errors.csv"
STATUS_INPUT = "exact_map_status.json"
CONFIG_INPUT = "exact_map_config.json"
HASHED_INPUTS = (
    "exact_map_refinement.csv",
    EXACT_INPUT,
    "e4_approximation_refinement.csv",
    E4_INPUT,
    CONFIG_INPUT,
)
RATIO_PLOT_FORMATS = ("png", "pdf", "svg")
RATIO_OUTPUTS = (
    "ratio_comparison_per_seed.csv",
    "ratio_comparison_summary.csv",
    "ratio_comparison_worst_per_seed.csv",
    "ratio_comparison_worst_summary.csv",
    "ratio_comparison_metadata.json",
)
RATIO_PLOT_STEMS = (
    "empirical_ratio",
    "exact_map_contraction",
    "ratio_comparison",
)
AGG_MANAGED_OUTPUTS = (
    "exact_map_per_seed.csv",
    "exact_map_summary.csv",
    "exact_map_worst_per_seed.csv",
    "exact_map_worst_summary.csv",
    "e4_per_seed.csv",
    "e4_summary.csv",
    *RATIO_OUTPUTS,
    *(
        f"{stem}.{suffix}"
        for stem in RATIO_PLOT_STEMS
        for suffix in RATIO_PLOT_FORMATS
    ),
    "exact_map_aggregate_status.json",
    "_SUCCESS_EXACT_MAP_AGG",
    "_FAILED_EXACT_MAP_AGG",
)

EXACT_METRICS = (
    "e_input_value", "e_input_vw", "e_input_vww", "e_input_vwx",
    "e_input_bundle", "e_input_X", "e_map_value", "e_map_vw",
    "e_map_vww", "e_map_vwx", "e_map_bundle", "e_map_X", "rho_exact",
    "rho_sensitivity_envelope", "guard_frac_fd", "positive_curvature_frac_fd",
    "theta_any_clip_frac_fd", "theta_component_clip_frac_fd",
    "guard_frac_ev", "positive_curvature_frac_ev", "theta_any_clip_frac_ev",
    "theta_component_clip_frac_ev", "min_log_joint_eig",
    "min_original_joint_eig", "nonpositive_log_eig_fraction",
    "max_peclet_y", "max_peclet_x",
    "upwind_y_fraction", "upwind_x_fraction", "max_linear_residual",
    "outside_collocation_fraction_fd", "outside_collocation_y_fraction_fd",
    "outside_collocation_x_fraction_fd", "boundary_elimination_cond_inf",
    "min_linear_system_lu_pivot_ratio",
    "grid_abs_change", "grid_rel_change",
    "domain_abs_change", "domain_rel_change",
    "wealth_domain_abs_change", "wealth_domain_rel_change",
    "factor_domain_abs_change", "factor_domain_rel_change",
    "refinement_tolerance", "numerical_abs_change",
    "numerical_tolerance_ratio",
    "boundary_abs_change", "boundary_rel_change",
    "boundary_tolerance_ratio",
)
E4_METRICS = (
    "e_approx_value", "e_approx_vw", "e_approx_vww", "e_approx_vwx",
    "e_approx_bundle", "e_approx_X", "approx_sensitivity_envelope",
    "source_min_log_joint_eig", "source_max_log_joint_eig",
    "source_min_original_joint_eig", "source_max_original_joint_eig",
    "source_nonpositive_log_eig_fraction",
    "source_outside_collocation_fraction_fd",
    "grid_abs_change", "grid_rel_change",
    "domain_abs_change", "domain_rel_change",
    "wealth_domain_abs_change", "wealth_domain_rel_change",
    "factor_domain_abs_change", "factor_domain_rel_change",
    "refinement_tolerance", "numerical_abs_change",
    "numerical_tolerance_ratio",
    "boundary_abs_change", "boundary_rel_change",
    "boundary_tolerance_ratio",
)
OPTIONAL_SUMMARY_METRICS = {
    "rho_sensitivity_envelope",
    "approx_sensitivity_envelope",
    # Coupled legacy mode intentionally leaves the two directional changes
    # blank because only diagonal Dw=Dx perturbations are identified.
    "wealth_domain_abs_change", "wealth_domain_rel_change",
    "factor_domain_abs_change", "factor_domain_rel_change",
    "boundary_abs_change", "boundary_rel_change",
    "boundary_tolerance_ratio",
    "refinement_tolerance", "numerical_abs_change",
    "numerical_tolerance_ratio",
}
EVALUATION_WINDOW_NUMERIC_FIELDS = (
    "eval_margin", "eval_x_margin",
    "ev_w_min", "ev_w_max", "ev_x_min", "ev_x_max",
    "saved_w_min", "saved_w_max", "saved_x_min", "saved_x_max",
)
EVALUATION_WINDOW_OPTIONAL_FIELDS = (
    "eval_w_min_override", "eval_w_max_override",
)
EVALUATION_WINDOW_ROW_FIELDS = (
    "eval_margin", "eval_x_margin",
    "eval_w_min_override", "eval_w_max_override",
    "ev_w_min", "ev_w_max", "ev_x_min", "ev_x_max",
)
DOMAIN_MODES = ("coupled", "split")
WEALTH_DOMAIN_PARAMETERIZATIONS = (
    "symmetric_log_half_width_factor",
    "explicit_absolute_bounds",
)
WEALTH_DOMAIN_PROVENANCE_FIELDS = (
    "wealth_domain_parameterization",
    "wealth_domain_bounds",
    "requested_fd_w_mins",
    "requested_fd_w_maxs",
    "wealth_grid_size_rule",
)
WEALTH_DOMAIN_BOUND_FIELDS = (
    "fd_y_min", "fd_y_max", "fd_w_min", "fd_w_max",
)
DOMAIN_ROW_FIELDS = (
    "domain_mode", "domain_factor",
    "wealth_domain_factor", "factor_domain_factor",
    *WEALTH_DOMAIN_BOUND_FIELDS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_artifact(candidates: Sequence[Path], expected_hash: str, label: str) -> Path:
    checked: List[str] = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if str(path) in checked:
            continue
        checked.append(str(path))
        if path.is_file() and sha256_file(path) == expected_hash:
            return path
    raise ValueError(f"cannot validate current {label} against hash {expected_hash}; tried {checked}")


def _canonical_evaluation_window(
    container: Mapping[str, Any],
    source: Path,
) -> Dict[str, Any]:
    raw = container.get("evaluation_window")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: missing evaluation_window provenance")
    result: Dict[str, Any] = {}
    for field in EVALUATION_WINDOW_NUMERIC_FIELDS:
        try:
            value = float(raw.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: invalid evaluation_window.{field}={raw.get(field)!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{source}: evaluation_window.{field} must be finite")
        result[field] = value
    for field in EVALUATION_WINDOW_OPTIONAL_FIELDS:
        value = raw.get(field)
        if value is None:
            result[field] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: invalid evaluation_window.{field}={value!r}"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"{source}: evaluation_window.{field} must be finite or null")
        result[field] = number
    definition = str(raw.get("definition", "")).strip()
    if not definition:
        raise ValueError(f"{source}: evaluation_window definition is missing")
    result["definition"] = definition
    fd_dependency = raw.get("fd_domain_depends_on_evaluation_window")
    if fd_dependency is not False:
        raise ValueError(
            f"{source}: FD domain must remain independent of the evaluation window"
        )
    result["fd_domain_depends_on_evaluation_window"] = False

    if not 0.0 <= result["eval_margin"] < 1.0:
        raise ValueError(f"{source}: eval_margin must lie in [0,1)")
    if not 0.0 <= result["eval_x_margin"] < 1.0:
        raise ValueError(f"{source}: eval_x_margin must lie in [0,1)")
    if not (
        0.0 < result["saved_w_min"] < result["saved_w_max"]
        and result["saved_x_min"] < result["saved_x_max"]
    ):
        raise ValueError(f"{source}: saved evaluation-domain bounds are invalid")

    base_delta = (
        0.5
        * result["eval_margin"]
        * (result["saved_w_max"] - result["saved_w_min"])
    )
    expected_w_min = (
        result["saved_w_min"] + base_delta
        if result["eval_w_min_override"] is None
        else result["eval_w_min_override"]
    )
    expected_w_max = (
        result["saved_w_max"] - base_delta
        if result["eval_w_max_override"] is None
        else result["eval_w_max_override"]
    )
    x_delta = (
        0.5
        * result["eval_x_margin"]
        * (result["saved_x_max"] - result["saved_x_min"])
    )
    expected = {
        "ev_w_min": expected_w_min,
        "ev_w_max": expected_w_max,
        "ev_x_min": result["saved_x_min"] + x_delta,
        "ev_x_max": result["saved_x_max"] - x_delta,
    }
    for field, target in expected.items():
        if not math.isclose(result[field], target, rel_tol=1e-12, abs_tol=1e-13):
            raise ValueError(
                f"{source}: evaluation_window.{field}={result[field]!r} "
                f"does not match its declared resolution {target!r}"
            )
    if not (
        result["saved_w_min"]
        <= result["ev_w_min"]
        < result["ev_w_max"]
        <= result["saved_w_max"]
    ):
        raise ValueError(f"{source}: effective wealth evaluation bounds are invalid")
    return result


def _validate_evaluation_window_rows(
    rows: Sequence[Mapping[str, str]],
    window: Mapping[str, Any],
    source: Path,
) -> None:
    for position, row in enumerate(rows, start=1):
        for field in EVALUATION_WINDOW_ROW_FIELDS:
            expected = window[field]
            raw = row.get(field, "")
            if expected is None:
                if str(raw).strip() != "":
                    raise ValueError(
                        f"{source}: row {position} {field} must be blank for a null override"
                    )
                continue
            observed = _number(row, field, source)
            if not math.isclose(
                observed, float(expected), rel_tol=1e-12, abs_tol=1e-13
            ):
                raise ValueError(
                    f"{source}: row {position} {field}={observed!r} "
                    f"does not match config value {expected!r}"
                )


def _finite_factor_list(raw: Any, field: str, source: Path) -> List[float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{source}: {field} must be a numeric list")
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid {field}={raw!r}") from exc
    if (
        not values
        or any(not math.isfinite(value) or value <= 1.0 for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError(
            f"{source}: {field} must be a sorted, unique, nonempty list of "
            "finite factors strictly larger than one"
        )
    return values


def _finite_numeric_list(raw: Any, field: str, source: Path) -> List[float]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{source}: {field} must be a numeric list")
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid {field}={raw!r}") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{source}: {field} must contain only finite values")
    return values


def _canonical_wealth_domain_provenance(
    raw: Mapping[str, Any],
    wealth_factors: Sequence[float],
    source: Path,
) -> Optional[Dict[str, Any]]:
    """Validate the optional resolved wealth-domain schema.

    Results written before the absolute-bound feature do not contain any of
    these keys and remain valid.  Once any key is present, however, the full
    parameterization and resolved-bound schedule are required.
    """

    present = any(field in raw for field in WEALTH_DOMAIN_PROVENANCE_FIELDS)
    if not present:
        return None

    parameterization = str(
        raw.get("wealth_domain_parameterization", "")
    ).strip().lower()
    if parameterization not in WEALTH_DOMAIN_PARAMETERIZATIONS:
        raise ValueError(
            f"{source}: invalid wealth_domain_parameterization="
            f"{parameterization!r}"
        )

    raw_bounds = raw.get("wealth_domain_bounds")
    if not isinstance(raw_bounds, Sequence) or isinstance(
        raw_bounds, (str, bytes)
    ):
        raise ValueError(f"{source}: wealth_domain_bounds must be a list")
    bounds: List[Dict[str, float]] = []
    for position, item in enumerate(raw_bounds, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{source}: wealth_domain_bounds[{position}] must be an object"
            )
        try:
            factor = float(item.get("wealth_domain_factor"))
            y_min = float(item.get("fd_y_min"))
            y_max = float(item.get("fd_y_max"))
            w_min = float(item.get("fd_w_min"))
            w_max = float(item.get("fd_w_max"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: invalid wealth_domain_bounds[{position}]={item!r}"
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (factor, y_min, y_max, w_min, w_max)
        ):
            raise ValueError(
                f"{source}: wealth_domain_bounds[{position}] must be finite"
            )
        if not (w_min > 0.0 and w_min < w_max and y_min < y_max):
            raise ValueError(
                f"{source}: wealth_domain_bounds[{position}] is not an "
                "ordered positive wealth interval"
            )
        if (
            not math.isclose(
                math.log(w_min), y_min, rel_tol=1e-12, abs_tol=1e-13
            )
            or not math.isclose(
                math.log(w_max), y_max, rel_tol=1e-12, abs_tol=1e-13
            )
        ):
            raise ValueError(
                f"{source}: wealth_domain_bounds[{position}] has inconsistent "
                "wealth/log-wealth endpoints"
            )
        bounds.append(
            {
                "wealth_domain_factor": factor,
                "fd_y_min": y_min,
                "fd_y_max": y_max,
                "fd_w_min": w_min,
                "fd_w_max": w_max,
            }
        )

    observed_factors = [item["wealth_domain_factor"] for item in bounds]
    expected_factors = [float(value) for value in wealth_factors]
    if observed_factors != expected_factors:
        raise ValueError(
            f"{source}: wealth_domain_bounds factors {observed_factors} do not "
            f"match wealth_domain_factors {expected_factors}"
        )

    result: Dict[str, Any] = {
        "wealth_domain_parameterization": parameterization,
        "wealth_domain_bounds": bounds,
    }
    grid_size_rule = str(raw.get("wealth_grid_size_rule", "")).strip()
    if not grid_size_rule:
        raise ValueError(f"{source}: missing wealth_grid_size_rule")
    result["wealth_grid_size_rule"] = grid_size_rule
    raw_mins = raw.get("requested_fd_w_mins")
    raw_maxs = raw.get("requested_fd_w_maxs")
    if parameterization == "explicit_absolute_bounds":
        if "requested_fd_w_mins" not in raw or "requested_fd_w_maxs" not in raw:
            raise ValueError(
                f"{source}: explicit_absolute_bounds requires "
                "requested_fd_w_mins and requested_fd_w_maxs"
            )
        mins = _finite_numeric_list(
            raw_mins, "requested_fd_w_mins", source
        )
        maxs = _finite_numeric_list(
            raw_maxs, "requested_fd_w_maxs", source
        )
        if len(mins) != len(bounds) or len(maxs) != len(bounds):
            raise ValueError(
                f"{source}: requested absolute wealth bounds must have one "
                "entry per wealth_domain_factor"
            )
        for position, (minimum, maximum, resolved) in enumerate(
            zip(mins, maxs, bounds), start=1
        ):
            if not (minimum > 0.0 and minimum < maximum):
                raise ValueError(
                    f"{source}: requested absolute wealth interval {position} "
                    "is invalid"
                )
            if (
                not math.isclose(
                    minimum, resolved["fd_w_min"],
                    rel_tol=1e-12, abs_tol=1e-13,
                )
                or not math.isclose(
                    maximum, resolved["fd_w_max"],
                    rel_tol=1e-12, abs_tol=1e-13,
                )
            ):
                raise ValueError(
                    f"{source}: requested absolute wealth interval {position} "
                    "does not match wealth_domain_bounds"
                )
        result["requested_fd_w_mins"] = mins
        result["requested_fd_w_maxs"] = maxs
    else:
        for field, value in (
            ("requested_fd_w_mins", raw_mins),
            ("requested_fd_w_maxs", raw_maxs),
        ):
            if value not in (None, []):
                raise ValueError(
                    f"{source}: {field} is only valid for "
                    "explicit_absolute_bounds"
                )
        result["requested_fd_w_mins"] = []
        result["requested_fd_w_maxs"] = []
    return result


def _canonical_domain_design(
    container: Mapping[str, Any],
    source: Path,
) -> Dict[str, Any]:
    """Canonicalize the coupled/split FD-domain protocol.

    The driver records the same top-level ``domain_design`` object in config
    and status. Keeping it explicit makes it impossible to pool results whose
    wealth and factor rectangles differ.
    """

    raw = container.get("domain_design")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: missing domain_design provenance")
    mode = str(raw.get("mode", "")).strip().lower()
    if mode not in DOMAIN_MODES:
        raise ValueError(f"{source}: invalid domain_design.mode={mode!r}")

    wealth = _finite_factor_list(
        raw.get("wealth_domain_factors"), "wealth_domain_factors", source
    )
    factor = _finite_factor_list(
        raw.get("factor_domain_factors"), "factor_domain_factors", source
    )
    raw_pairs = raw.get("domain_pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError(f"{source}: domain_pairs must be a list")
    pairs: List[Dict[str, float]] = []
    for position, item in enumerate(raw_pairs, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: domain_pairs[{position}] must be an object")
        try:
            dw = float(item.get("wealth_domain_factor"))
            dx = float(item.get("factor_domain_factor"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: invalid domain pair at position {position}: {item!r}"
            ) from exc
        if not (math.isfinite(dw) and math.isfinite(dx) and dw > 1.0 and dx > 1.0):
            raise ValueError(f"{source}: invalid domain pair at position {position}")
        pairs.append(
            {"wealth_domain_factor": dw, "factor_domain_factor": dx}
        )
    observed_pairs = [
        (item["wealth_domain_factor"], item["factor_domain_factor"])
        for item in pairs
    ]
    if observed_pairs != sorted(set(observed_pairs)):
        raise ValueError(f"{source}: domain_pairs must be sorted and unique")

    if mode == "coupled":
        expected_pairs = [(value, value) for value in wealth]
        if wealth != factor:
            raise ValueError(
                f"{source}: coupled mode requires matching wealth-domain and "
                "factor-domain lists"
            )
    else:
        expected_pairs = [(dw, dx) for dw in wealth for dx in factor]
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"{source}: domain_pairs do not match the declared {mode} design"
        )

    legacy_shared = raw.get("legacy_shared_shorthand")
    if not isinstance(legacy_shared, bool) or legacy_shared != (mode == "coupled"):
        raise ValueError(
            f"{source}: inconsistent domain_design.legacy_shared_shorthand"
        )
    definition = str(raw.get("definition", "")).strip()
    if not definition:
        raise ValueError(f"{source}: missing domain_design.definition")
    result: Dict[str, Any] = {
        "mode": mode,
        "legacy_shared_shorthand": legacy_shared,
        "wealth_domain_factors": wealth,
        "factor_domain_factors": factor,
        "domain_pairs": pairs,
        "primary_wealth_domain_factor": max(wealth),
        "primary_factor_domain_factor": max(factor),
        "definition": definition,
    }
    wealth_provenance = _canonical_wealth_domain_provenance(
        raw, wealth, source
    )
    if wealth_provenance is not None:
        result.update(wealth_provenance)
    for field in (
        "primary_wealth_domain_factor",
        "primary_factor_domain_factor",
    ):
        try:
            observed = float(raw.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: missing/invalid {field}") from exc
        if not math.isclose(
            observed, float(result[field]), rel_tol=1e-12, abs_tol=1e-13
        ):
            raise ValueError(f"{source}: inconsistent {field}={observed!r}")
    return result


def _validate_config_grid_domain_design(
    config: Mapping[str, Any],
    design: Mapping[str, Any],
    source: Path,
) -> Mapping[str, Any]:
    grid = config.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError(f"{source}: missing grid provenance")
    mode = str(design["mode"])
    if str(grid.get("domain_mode", "")).strip().lower() != mode:
        raise ValueError(f"{source}: grid/domain_design mode mismatch")
    for field in ("wealth_domain_factors", "factor_domain_factors"):
        values = _finite_factor_list(grid.get(field), field, source)
        if values != list(design[field]):
            raise ValueError(f"{source}: grid/domain_design {field} mismatch")
    raw_legacy = grid.get("domain_factors")
    if not isinstance(raw_legacy, Sequence) or isinstance(raw_legacy, (str, bytes)):
        raise ValueError(f"{source}: grid.domain_factors must be a list")
    try:
        legacy = [float(value) for value in raw_legacy]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid grid.domain_factors") from exc
    expected_legacy = (
        list(design["wealth_domain_factors"]) if mode == "coupled" else []
    )
    if legacy != expected_legacy:
        raise ValueError(f"{source}: grid/domain_design legacy domain_factors mismatch")
    grid_pairs = grid.get("domain_pairs")
    if grid_pairs != design["domain_pairs"]:
        raise ValueError(f"{source}: grid/domain_design domain_pairs mismatch")
    design_has_bounds = "wealth_domain_parameterization" in design
    grid_provenance = _canonical_wealth_domain_provenance(
        grid, design["wealth_domain_factors"], source
    )
    if design_has_bounds != (grid_provenance is not None):
        raise ValueError(
            f"{source}: grid/domain_design wealth-domain provenance mismatch"
        )
    if design_has_bounds:
        expected = {
            field: design[field]
            for field in WEALTH_DOMAIN_PROVENANCE_FIELDS
        }
        if grid_provenance != expected:
            raise ValueError(
                f"{source}: grid/domain_design resolved wealth bounds mismatch"
            )
    return grid


def _validate_resolved_wealth_geometry(
    design: Mapping[str, Any],
    evaluation_window: Mapping[str, Any],
    source: Path,
) -> None:
    bounds = list(design.get("wealth_domain_bounds", []))
    if not bounds:
        return
    saved_w_min = float(evaluation_window["saved_w_min"])
    saved_w_max = float(evaluation_window["saved_w_max"])
    saved_y_min = math.log(saved_w_min)
    saved_y_max = math.log(saved_w_max)
    saved_width = saved_y_max - saved_y_min
    saved_center = 0.5 * (saved_y_min + saved_y_max)
    parameterization = str(design["wealth_domain_parameterization"])
    previous_w_min = float("inf")
    previous_w_max = -float("inf")
    for position, item in enumerate(bounds, start=1):
        fd_y_min = float(item["fd_y_min"])
        fd_y_max = float(item["fd_y_max"])
        fd_w_min = float(item["fd_w_min"])
        fd_w_max = float(item["fd_w_max"])
        factor = float(item["wealth_domain_factor"])
        if (
            not fd_y_min < saved_y_min
            or math.isclose(
                fd_y_min, saved_y_min, rel_tol=1e-12, abs_tol=1e-13
            )
            or not fd_y_max > saved_y_max
            or math.isclose(
                fd_y_max, saved_y_max, rel_tol=1e-12, abs_tol=1e-13
            )
        ):
            raise ValueError(
                f"{source}: resolved FD wealth interval {position} does not "
                "strictly contain the saved training wealth interval"
            )
        observed_factor = (fd_y_max - fd_y_min) / saved_width
        if not math.isclose(
            observed_factor, factor, rel_tol=1e-12, abs_tol=1e-13
        ):
            raise ValueError(
                f"{source}: resolved FD wealth interval {position} has "
                "inconsistent log-width factor"
            )
        if parameterization == "symmetric_log_half_width_factor":
            if not math.isclose(
                0.5 * (fd_y_min + fd_y_max),
                saved_center,
                rel_tol=1e-12,
                abs_tol=1e-13,
            ):
                raise ValueError(
                    f"{source}: symmetric FD wealth interval {position} is "
                    "not centered on the saved log-wealth interval"
                )
        elif position > 1 and not (
            fd_w_min < previous_w_min and fd_w_max > previous_w_max
        ):
            raise ValueError(
                f"{source}: explicit FD wealth intervals must form a strictly "
                "nested narrowest-to-widest chain"
            )
        previous_w_min = fd_w_min
        previous_w_max = fd_w_max


def _validate_domain_rows(
    rows: Sequence[Mapping[str, str]],
    design: Mapping[str, Any],
    source: Path,
    *,
    primary_only: bool,
) -> None:
    allowed_pairs = {
        (
            float(item["wealth_domain_factor"]),
            float(item["factor_domain_factor"]),
        )
        for item in design["domain_pairs"]
    }
    primary_pair = (
        float(design["primary_wealth_domain_factor"]),
        float(design["primary_factor_domain_factor"]),
    )
    mode = str(design["mode"])
    bounds_by_factor = {
        float(item["wealth_domain_factor"]): item
        for item in design.get("wealth_domain_bounds", [])
    }
    for position, row in enumerate(rows, start=1):
        if str(row.get("domain_mode", "")).strip().lower() != mode:
            raise ValueError(f"{source}: row {position} domain_mode mismatch")
        pair = (
            _number(row, "wealth_domain_factor", source),
            _number(row, "factor_domain_factor", source),
        )
        if pair not in allowed_pairs:
            raise ValueError(
                f"{source}: row {position} uses undeclared domain pair {pair}"
            )
        raw_legacy = str(row.get("domain_factor", "")).strip()
        if mode == "split":
            if raw_legacy:
                raise ValueError(
                    f"{source}: row {position} domain_factor must be blank in split mode"
                )
        else:
            legacy = _number(row, "domain_factor", source)
            if not (
                math.isclose(legacy, pair[0], rel_tol=1e-12, abs_tol=1e-13)
                and math.isclose(legacy, pair[1], rel_tol=1e-12, abs_tol=1e-13)
            ):
                raise ValueError(
                    f"{source}: row {position} coupled domain factors disagree"
                )
        if primary_only and pair != primary_pair:
            raise ValueError(
                f"{source}: row {position} primary domain pair {pair} does not "
                f"match configured {primary_pair}"
            )
        if bounds_by_factor:
            expected_bounds = bounds_by_factor[pair[0]]
            for field in WEALTH_DOMAIN_BOUND_FIELDS:
                observed = _number(row, field, source)
                expected = float(expected_bounds[field])
                if not math.isclose(
                    observed, expected, rel_tol=1e-12, abs_tol=1e-13
                ):
                    raise ValueError(
                        f"{source}: row {position} {field}={observed!r} does "
                        f"not match configured wealth-domain bound {expected!r}"
                    )


def _validate_refinement_design(
    rows: Sequence[Mapping[str, str]],
    design: Mapping[str, Any],
    grid: Mapping[str, Any],
    source: Path,
    *,
    index_field: str,
    refinement_scope: Optional[str] = None,
) -> None:
    """Require the declared numerical and boundary-sensitivity design."""

    _validate_domain_rows(rows, design, source, primary_only=False)
    try:
        grid_factors = sorted(set(int(value) for value in grid.get("grid_factors", [])))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid grid_factors") from exc
    boundaries_raw = grid.get("boundaries")
    if (
        not grid_factors
        or any(value < 1 for value in grid_factors)
        or not isinstance(boundaries_raw, Sequence)
        or isinstance(boundaries_raw, (str, bytes))
    ):
        raise ValueError(f"{source}: invalid grid/boundary refinement design")
    boundaries = [str(value) for value in boundaries_raw]
    if not boundaries or len(boundaries) != len(set(boundaries)):
        raise ValueError(f"{source}: boundary variants must be nonempty and unique")
    if refinement_scope in (None, ""):
        # Legacy Liu artifacts used every boundary inside the full Cartesian
        # refinement product.
        verified_expected = {
            (
                factor,
                float(pair["wealth_domain_factor"]),
                float(pair["factor_domain_factor"]),
                boundary,
            )
            for factor in grid_factors
            for pair in design["domain_pairs"]
            for boundary in boundaries
        }
    elif refinement_scope == REFINEMENT_SCOPE:
        primary_boundary = boundaries[0]
        primary_pair = (
            float(design["primary_wealth_domain_factor"]),
            float(design["primary_factor_domain_factor"]),
        )
        finest = max(grid_factors)
        verified_expected = {
            (
                factor,
                float(pair["wealth_domain_factor"]),
                float(pair["factor_domain_factor"]),
                primary_boundary,
            )
            for factor in grid_factors
            for pair in design["domain_pairs"]
        }
        verified_expected.update(
            (
                finest,
                primary_pair[0],
                primary_pair[1],
                boundary,
            )
            for boundary in boundaries[1:]
        )
    else:
        raise ValueError(
            f"{source}: unsupported refinement_scope={refinement_scope!r}"
        )
    grouped: Dict[int, List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_integer(row, index_field, source), []).append(row)
    for index, group in grouped.items():
        verification_flags = {
            _integer(row, "is_verification", source) for row in group
        }
        if not verification_flags <= {0, 1} or len(verification_flags) != 1:
            raise ValueError(
                f"{source}: inconsistent is_verification flags at "
                f"{index_field}={index}: {sorted(verification_flags)}"
            )
        if refinement_scope == REFINEMENT_SCOPE:
            for row in group:
                if str(row.get("refinement_scope", "")) != REFINEMENT_SCOPE:
                    raise ValueError(
                        f"{source}: refinement_scope mismatch at "
                        f"{index_field}={index}"
                    )
                if str(row.get("boundary_sensitivity_role", "")) != (
                    BOUNDARY_SENSITIVITY_ROLE
                ):
                    raise ValueError(
                        f"{source}: boundary_sensitivity_role mismatch at "
                        f"{index_field}={index}"
                    )
        if verification_flags == {1}:
            expected = verified_expected
        else:
            expected = {
                (
                    max(grid_factors),
                    float(design["primary_wealth_domain_factor"]),
                    float(design["primary_factor_domain_factor"]),
                    boundaries[0],
                )
            }
        observed = {
            (
                _integer(row, "grid_factor", source),
                _number(row, "wealth_domain_factor", source),
                _number(row, "factor_domain_factor", source),
                str(row.get("boundary", "")),
            )
            for row in group
        }
        if len(observed) != len(group) or observed != expected:
            raise ValueError(
                f"{source}: incomplete/duplicate Cartesian refinement design at "
                f"{index_field}={index} (is_verification="
                f"{next(iter(verification_flags))})"
            )
        if refinement_scope == REFINEMENT_SCOPE:
            primary_key = (
                max(grid_factors),
                float(design["primary_wealth_domain_factor"]),
                float(design["primary_factor_domain_factor"]),
                boundaries[0],
            )
            for row in group:
                key = (
                    _integer(row, "grid_factor", source),
                    _number(row, "wealth_domain_factor", source),
                    _number(row, "factor_domain_factor", source),
                    str(row.get("boundary", "")),
                )
                expected_primary = int(key == primary_key)
                if _integer(row, "is_primary", source) != expected_primary:
                    raise ValueError(
                        f"{source}: is_primary mismatch at "
                        f"{index_field}={index}, variant={key}"
                    )


def _validate_provenance(
    directory: Path,
    config: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, str]],
    e4_rows: Sequence[Mapping[str, str]],
    *,
    allow_historical_driver: bool = False,
) -> None:
    if str(config.get("analysis_mode", "")) != "exact_map_and_e4":
        raise ValueError(
            f"{directory}: paper aggregation requires analysis_mode='exact_map_and_e4'; "
            f"got {config.get('analysis_mode')!r}"
        )
    policy_extension = str(config.get("policy_extension", ""))
    if policy_extension != "boundary-projection":
        raise ValueError(
            f"{directory}: paper aggregation requires the declared finite-domain "
            f"boundary-projection policy extension; got {policy_extension!r}"
        )
    if str(config.get("map_definition", "")) != (
        "finite_domain_boundary_projected_policy_extension"
    ):
        raise ValueError(f"{directory}: map-definition provenance is missing or inconsistent")
    refinement_rule = str(config.get("refinement_rule", ""))
    if refinement_rule not in {"cartesian", "merton-axis"}:
        raise ValueError(
            f"{directory}: unsupported refinement_rule={refinement_rule!r}"
        )
    refinement_scope = config.get("refinement_scope")
    if refinement_scope is not None:
        if str(refinement_scope) != REFINEMENT_SCOPE:
            raise ValueError(
                f"{directory}: unsupported refinement_scope="
                f"{refinement_scope!r}"
            )
        if str(config.get("boundary_sensitivity_role", "")) != (
            BOUNDARY_SENSITIVITY_ROLE
        ):
            raise ValueError(
                f"{directory}: boundary_sensitivity_role is missing or "
                "inconsistent"
            )
    raw_floor = config.get("min_paper_checkpoint")
    if isinstance(raw_floor, bool) or not isinstance(raw_floor, int) or raw_floor < 0:
        raise ValueError(
            f"{directory}: min_paper_checkpoint must be a nonnegative integer"
        )
    for rows, name in ((exact_rows, EXACT_INPUT), (e4_rows, E4_INPUT)):
        if _identity(rows, "refinement_rule", directory / name) != refinement_rule:
            raise ValueError(
                f"{directory / name}: refinement_rule does not match config"
            )
        if refinement_scope is not None:
            if _identity(
                rows, "refinement_scope", directory / name
            ) != REFINEMENT_SCOPE:
                raise ValueError(
                    f"{directory / name}: refinement_scope does not match "
                    "config"
                )
            if _identity(
                rows, "boundary_sensitivity_role", directory / name
            ) != BOUNDARY_SENSITIVITY_ROLE:
                raise ValueError(
                    f"{directory / name}: boundary_sensitivity_role does not "
                    "match config"
                )
        if any(
            _integer(row, "min_paper_checkpoint", directory / name) != raw_floor
            for row in rows
        ):
            raise ValueError(
                f"{directory / name}: min_paper_checkpoint does not match config"
            )
    evaluation_window = _canonical_evaluation_window(
        config, directory / CONFIG_INPUT
    )
    _validate_evaluation_window_rows(
        exact_rows, evaluation_window, directory / EXACT_INPUT
    )
    _validate_evaluation_window_rows(
        e4_rows, evaluation_window, directory / E4_INPUT
    )
    domain_design = _canonical_domain_design(
        config, directory / CONFIG_INPUT
    )
    _validate_resolved_wealth_geometry(
        domain_design, evaluation_window, directory / CONFIG_INPUT
    )
    _validate_config_grid_domain_design(
        config, domain_design, directory / CONFIG_INPUT
    )
    _validate_domain_rows(
        exact_rows, domain_design, directory / EXACT_INPUT, primary_only=True
    )
    _validate_domain_rows(
        e4_rows, domain_design, directory / E4_INPUT, primary_only=True
    )
    selection = str(config.get("checkpoint_selection", ""))
    if selection != "all":
        raise ValueError(
            f"{directory}: paper aggregation rejects checkpoint_selection={selection!r}; "
            "explicit checkpoint subsets are exploratory only"
        )
    raw_schedule = config.get("checkpoint_schedule")
    if (not isinstance(raw_schedule, Sequence)
            or isinstance(raw_schedule, (str, bytes))):
        raise ValueError(f"missing checkpoint_schedule: {directory / CONFIG_INPUT}")
    try:
        configured_schedule = [int(value) for value in raw_schedule]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint_schedule: {raw_schedule!r}") from exc
    if (not configured_schedule
            or configured_schedule != list(range(1, configured_schedule[-1] + 1))):
        raise ValueError(
            f"{directory}: checkpoint_schedule must be the contiguous full prefix 1..K; "
            f"got {configured_schedule}"
        )
    training_args = config.get("training_protocol_args")
    if not isinstance(training_args, Mapping):
        raise ValueError(f"missing training_protocol_args: {directory / CONFIG_INPUT}")
    raw_outer_iters = training_args.get("outer_iters")
    if isinstance(raw_outer_iters, bool):
        raise ValueError(
            f"{directory}: training_protocol_args.outer_iters must be a positive integer"
        )
    try:
        outer_iters = int(raw_outer_iters)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{directory}: invalid training_protocol_args.outer_iters={raw_outer_iters!r}"
        ) from exc
    expected_schedule = list(range(1, outer_iters + 1))
    if outer_iters <= 0 or configured_schedule != expected_schedule:
        raise ValueError(
            f"{directory}: checkpoint_schedule does not cover all training outer iterations: "
            f"schedule={configured_schedule}, outer_iters={outer_iters}"
        )

    implementation = config.get("implementation_hashes")
    if not isinstance(implementation, Mapping):
        raise ValueError(f"missing implementation hashes: {directory / CONFIG_INPUT}")
    current_driver = Path(__file__).with_name("liu_exact_map_fd.py").resolve()
    current_core = Path(__file__).with_name("liu_exact_map_core.py").resolve()
    for key, path in (("driver", current_driver), ("core", current_core)):
        expected = str(implementation.get(key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(
                f"{directory}: invalid recorded {key} implementation hash"
            )
        if key == "driver" and allow_historical_driver:
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"{directory}: current {key} source does not match derived protocol")

    run_dir = directory.parent
    source_config = _matching_artifact(
        [Path(str(config.get("config_path", ""))), run_dir / "config.json"],
        str(config.get("config_sha256", "")), "training config",
    )
    source_payload = read_json(source_config)
    _matching_artifact(
        [Path(str(config.get("market_path", ""))), run_dir / "market_params.npz"],
        str(config.get("market_file_sha256", "")), "market snapshot",
    )
    recorded_hashes = config.get("checkpoint_file_hashes")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError(f"missing checkpoint_file_hashes: {directory / CONFIG_INPUT}")
    try:
        recorded_outers = [int(value) for value in recorded_hashes]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint_file_hashes keys: {list(recorded_hashes)!r}") from exc
    if (len(recorded_outers) != len(set(recorded_outers))
            or sorted(recorded_outers) != configured_schedule):
        raise ValueError(
            f"{directory}: checkpoint hash key set {sorted(recorded_outers)} does not "
            f"match checkpoint_schedule {configured_schedule}"
        )

    exact_schedule = [
        _integer(row, "source_outer_iter", directory / EXACT_INPUT) for row in exact_rows
    ]
    e4_schedule = [
        _integer(row, "target_outer_iter", directory / E4_INPUT) for row in e4_rows
    ]
    if exact_schedule != configured_schedule or e4_schedule != configured_schedule:
        raise ValueError(
            f"{directory}: CSV/config checkpoint schedule mismatch: "
            f"config={configured_schedule}, exact={exact_schedule}, E4={e4_schedule}"
        )

    weight_candidates: List[Path] = []
    resolved_weight = config.get("weight_dir")
    if resolved_weight not in (None, ""):
        weight_candidates.append(Path(str(resolved_weight)))
    raw_weight = source_payload.get("weight_dir")
    recorded_cwd = source_payload.get("cwd")
    if raw_weight not in (None, ""):
        raw_path = Path(str(raw_weight))
        if raw_path.is_absolute():
            weight_candidates.append(raw_path)
        elif recorded_cwd not in (None, ""):
            weight_candidates.append(Path(str(recorded_cwd)) / raw_path)
    if run_dir.parent.name in {"pi-pinn", "pinn"}:
        weight_candidates.append(
            run_dir.parent.parent / "weights" / run_dir.parent.name / run_dir.name
        )

    exact_by_outer: Dict[int, Mapping[str, str]] = {}
    for row in exact_rows:
        outer = _integer(row, "source_outer_iter", directory / EXACT_INPUT)
        if _integer(row, "frozen_policy_iter", directory / EXACT_INPUT) != outer - 1:
            raise ValueError(f"{directory}: exact-map frozen-policy index mismatch at outer={outer}")
        if _integer(row, "greedy_policy_iter", directory / EXACT_INPUT) != outer:
            raise ValueError(f"{directory}: exact-map greedy-policy index mismatch at outer={outer}")
        if _integer(row, "target_value_outer_iter", directory / EXACT_INPUT) != outer + 1:
            raise ValueError(f"{directory}: exact-map target index mismatch at outer={outer}")
        if str(row.get("policy_extension", "")) != policy_extension:
            raise ValueError(f"{directory}: exact-map policy-extension mismatch at outer={outer}")
        expected = str(recorded_hashes.get(str(outer), ""))
        if not expected or str(row.get("checkpoint_sha256", "")) != expected:
            raise ValueError(f"{directory}: checkpoint provenance mismatch at outer={outer}")
        name = Path(str(row.get("checkpoint", ""))).name
        candidates = [Path(str(row.get("checkpoint", "")))]
        candidates.extend(weight / "iterates" / name for weight in weight_candidates)
        _matching_artifact(candidates, expected, f"checkpoint outer={outer}")
        exact_by_outer[outer] = row

    for row in e4_rows:
        target = _integer(row, "target_outer_iter", directory / E4_INPUT)
        source = target - 1
        if _integer(row, "frozen_policy_iter", directory / E4_INPUT) != source:
            raise ValueError(f"{directory}: E4 frozen-policy index mismatch at target={target}")
        if _integer(row, "policy_source_outer_iter", directory / E4_INPUT) != source:
            raise ValueError(f"{directory}: E4 policy-source index mismatch at target={target}")
        if str(row.get("policy_extension", "")) != policy_extension:
            raise ValueError(f"{directory}: E4 policy-extension mismatch at target={target}")
        if str(row.get("checkpoint_sha256", "")) != str(exact_by_outer[target]["checkpoint_sha256"]):
            raise ValueError(f"{directory}: E4 target checkpoint hash mismatch at target={target}")
        source_hash = str(row.get("source_policy_hash", ""))
        if not source_hash:
            raise ValueError(f"{directory}: missing E4 source policy hash at target={target}")
        if source == 0:
            if row.get("fd_reference_source") != "analytic_alpha0_fd_solve":
                raise ValueError(f"{directory}: E4 target=1 is not labelled as alpha0 FD solve")
        else:
            expected_policy = str(exact_by_outer[source].get("policy_hash", ""))
            if source_hash != expected_policy:
                raise ValueError(f"{directory}: shifted E4 source hash mismatch at target={target}")
            if row.get("fd_reference_source") != f"reused_exact_map_source_outer_{source}":
                raise ValueError(f"{directory}: shifted E4 source label mismatch at target={target}")


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


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


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


def discover_result_dirs(out_roots: Sequence[Path], explicit: Sequence[Path]) -> List[Path]:
    found: set[Path] = set()
    for value in explicit:
        directory = value.expanduser().resolve()
        if (directory / STATUS_INPUT).is_file():
            found.add(directory)
        elif (directory / "liu_exact_map_fd" / STATUS_INPUT).is_file():
            found.add((directory / "liu_exact_map_fd").resolve())
        else:
            raise FileNotFoundError(f"not a Liu exact-map result directory: {directory}")
    for root_value in out_roots:
        root = root_value.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if (root / STATUS_INPUT).is_file():
            found.add(root)
        for path in root.glob(f"**/{STATUS_INPUT}"):
            found.add(path.parent.resolve())
    if not found:
        raise FileNotFoundError("no Liu exact-map result directories were discovered")
    return sorted(found)


def _identity(rows: Sequence[Mapping[str, str]], field: str, source: Path) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1 or "" in values:
        raise ValueError(f"{source}: field {field!r} is missing or not constant: {values}")
    return next(iter(values))


def _integer(row: Mapping[str, str], field: str, source: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid integer {field!r}: {row.get(field)!r}") from exc


def _number(row: Mapping[str, Any], field: str, source: Path,
            *, allow_blank: bool = False) -> float:
    value = row.get(field, "")
    if allow_blank and str(value).strip() == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid numeric {field!r}: {value!r}") from exc


def _status_bool(status: Mapping[str, Any], field: str, source: Path) -> bool:
    value = status.get(field)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{source}: status field {field!r} must be boolean, got {value!r}")
    return bool(value)


def _status_int(status: Mapping[str, Any], field: str, source: Path) -> int:
    value = status.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{source}: status field {field!r} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: status field {field!r} must be an integer, got {value!r}"
        ) from exc


def _status_int_list(status: Mapping[str, Any], field: str, source: Path) -> List[int]:
    value = status.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{source}: status field {field!r} must be a list, got {value!r}")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid integer list in status field {field!r}") from exc


def _e4_paper_evidence(
    config: Mapping[str, Any],
    e4_rows: Sequence[Mapping[str, Any]],
    source: Path,
) -> Dict[str, Any]:
    """Reconstruct the Merton-style E4 evidence set from primary rows."""

    raw_floor = config.get("min_paper_checkpoint")
    if isinstance(raw_floor, bool) or not isinstance(raw_floor, int):
        raise ValueError(
            f"{source}: min_paper_checkpoint must be a nonnegative integer"
        )
    floor = int(raw_floor)
    if floor < 0:
        raise ValueError(
            f"{source}: min_paper_checkpoint must be a nonnegative integer"
        )
    full_schedule = [
        _integer(row, "target_outer_iter", source) for row in e4_rows
    ]
    if full_schedule != sorted(set(full_schedule)):
        raise ValueError(f"{source}: E4 target schedule is not sorted and unique")
    paper_schedule = [outer for outer in full_schedule if outer >= floor]
    excluded = [outer for outer in full_schedule if outer < floor]
    if not paper_schedule:
        raise ValueError(
            f"{source}: min_paper_checkpoint={floor} excludes every E4 target"
        )
    if config.get("paper_checkpoint_schedule") != paper_schedule:
        raise ValueError(
            f"{source}: config paper_checkpoint_schedule mismatch"
        )
    if config.get("excluded_initial_checkpoints") != excluded:
        raise ValueError(
            f"{source}: config excluded_initial_checkpoints mismatch"
        )
    rule = config.get("e4_refinement_rule")
    if not isinstance(rule, Mapping):
        raise ValueError(f"{source}: config is missing e4_refinement_rule")
    refinement_rule = str(config.get("refinement_rule", ""))
    if refinement_rule not in {"cartesian", "merton-axis"}:
        raise ValueError(
            f"{source}: unsupported refinement_rule={refinement_rule!r}"
        )
    refinement_scope = config.get("refinement_scope")
    expected_rule = {
        "required_set": "initial_first_last_worst_e_approx_X",
        "worst_tie_break": "lowest_target_outer_iter",
        "variant_pass_fail": refinement_rule,
        "interaction_failures": (
            "included" if refinement_rule == "cartesian" else "excluded"
        ),
        "sensitivity_envelope": (
            "historical_liu_max_cartesian_change"
            if refinement_rule == "cartesian"
            else "primary_plus_sum_axis_max_abs_changes"
        ),
    }
    if refinement_scope is not None:
        if str(refinement_scope) != REFINEMENT_SCOPE:
            raise ValueError(
                f"{source}: unsupported refinement_scope={refinement_scope!r}"
            )
        expected_rule.update(
            {
                "sensitivity_axes": REFINEMENT_SCOPE,
                "boundary_replacement": BOUNDARY_SENSITIVITY_ROLE,
                "sensitivity_envelope": (
                    "max_primary_boundary_cartesian_grid_domain_change"
                    if refinement_rule == "cartesian"
                    else "primary_plus_sum_grid_domain_axis_max_abs_changes"
                ),
            }
        )
    if dict(rule) != expected_rule:
        raise ValueError(f"{source}: unsupported e4_refinement_rule={rule!r}")
    summary = summarize_e4_refinement(
        e4_rows, min_paper_checkpoint=floor
    )
    return {
        "min_paper_checkpoint": floor,
        "refinement_rule": refinement_rule,
        "full_schedule": full_schedule,
        "paper_schedule": paper_schedule,
        "excluded_initial_checkpoints": excluded,
        **summary,
    }


def _validate_status_contract(
    directory: Path,
    status: Mapping[str, Any],
    config: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, str]],
    e4_rows: Sequence[Mapping[str, str]],
    exact_refinement_rows: Sequence[Mapping[str, str]],
    e4_refinement_rows: Sequence[Mapping[str, str]],
    *,
    allow_legacy_pre_merton: bool = False,
    require_exact_ellipticity: bool = True,
) -> None:
    """Require the success status to describe the primary CSVs exactly."""

    source = directory / STATUS_INPUT
    config_window = _canonical_evaluation_window(
        config, directory / CONFIG_INPUT
    )
    status_window = _canonical_evaluation_window(status, source)
    if status_window != config_window:
        raise ValueError(f"{source}: status/config evaluation_window mismatch")
    config_domain = _canonical_domain_design(
        config, directory / CONFIG_INPUT
    )
    grid = _validate_config_grid_domain_design(
        config, config_domain, directory / CONFIG_INPUT
    )
    refinement_scope = config.get("refinement_scope")
    status_domain = _canonical_domain_design(
        status, source
    )
    if status_domain != config_domain:
        raise ValueError(f"{source}: status/config domain design mismatch")
    if refinement_scope is not None:
        if str(status.get("refinement_scope", "")) != REFINEMENT_SCOPE:
            raise ValueError(f"{source}: status refinement_scope mismatch")
        if str(status.get("boundary_sensitivity_role", "")) != (
            BOUNDARY_SENSITIVITY_ROLE
        ):
            raise ValueError(
                f"{source}: status boundary_sensitivity_role mismatch"
            )
        boundaries = [str(value) for value in grid.get("boundaries", [])]
        if str(config.get("primary_boundary", "")) != boundaries[0]:
            raise ValueError(f"{source}: config primary_boundary mismatch")
        if list(config.get("comparison_boundaries", [])) != boundaries[1:]:
            raise ValueError(
                f"{source}: config comparison_boundaries mismatch"
            )
        if str(status.get("primary_boundary", "")) != boundaries[0]:
            raise ValueError(f"{source}: status primary_boundary mismatch")
        if list(status.get("comparison_boundaries", [])) != boundaries[1:]:
            raise ValueError(
                f"{source}: status comparison_boundaries mismatch"
            )
        if _status_bool(
            status, "boundary_sensitivity_available", source
        ) != (len(boundaries) > 1):
            raise ValueError(
                f"{source}: status boundary_sensitivity_available mismatch"
            )
    _validate_refinement_design(
        exact_refinement_rows,
        config_domain,
        grid,
        directory / "exact_map_refinement.csv",
        index_field="source_outer_iter",
        refinement_scope=(
            None if refinement_scope is None else str(refinement_scope)
        ),
    )
    _validate_refinement_design(
        e4_refinement_rows,
        config_domain,
        grid,
        directory / "e4_approximation_refinement.csv",
        index_field="target_outer_iter",
        refinement_scope=(
            None if refinement_scope is None else str(refinement_scope)
        ),
    )
    _validate_evaluation_window_rows(
        exact_refinement_rows,
        config_window,
        directory / "exact_map_refinement.csv",
    )
    _validate_evaluation_window_rows(
        e4_refinement_rows,
        config_window,
        directory / "e4_approximation_refinement.csv",
    )
    expected_counts = {
        "n_exact_rows": len(exact_rows),
        "n_refinement_rows": len(exact_refinement_rows),
        "n_e4_rows": len(e4_rows),
        "n_e4_refinement_rows": len(e4_refinement_rows),
    }
    for field, expected in expected_counts.items():
        observed = _status_int(status, field, source)
        if observed != expected:
            raise ValueError(
                f"{source}: status {field} mismatch: recorded={observed}, actual={expected}"
            )

    undefined: List[int] = []
    for row in exact_rows:
        outer = _integer(row, "source_outer_iter", directory / EXACT_INPUT)
        flag = _integer(row, "denominator_defined", directory / EXACT_INPUT)
        if flag not in (0, 1):
            raise ValueError(f"{directory / EXACT_INPUT}: denominator_defined must be 0 or 1")
        rho_finite = math.isfinite(
            _number(row, "rho_exact", directory / EXACT_INPUT, allow_blank=True)
        )
        if bool(flag) != rho_finite:
            raise ValueError(
                f"{directory / EXACT_INPUT}: denominator/rho consistency mismatch at outer={outer}"
            )
        if not flag:
            undefined.append(outer)
    if _status_int_list(status, "undefined_denominator_outers", source) != undefined:
        raise ValueError(
            f"{source}: status undefined_denominator_outers mismatch: expected {undefined}"
        )
    if _status_bool(status, "all_denominators_defined", source) != (not undefined):
        raise ValueError(f"{source}: status all_denominators_defined mismatch")

    exact_failures = [
        _integer(row, "source_outer_iter", directory / EXACT_INPUT)
        for row in exact_rows if row.get("refinement_status") != "pass"
    ]
    for row in exact_rows:
        if row.get("refinement_status") != "pass":
            continue
        outer = _integer(row, "source_outer_iter", directory / EXACT_INPUT)
        rho = _number(row, "rho_exact", directory / EXACT_INPUT)
        envelope = _number(
            row, "rho_sensitivity_envelope", directory / EXACT_INPUT
        )
        roundoff = 64.0 * np.finfo(float).eps * max(
            1.0, abs(rho), abs(envelope)
        )
        if envelope + roundoff < rho:
            raise ValueError(
                f"{directory / EXACT_INPUT}: sensitivity envelope is below the "
                f"primary exact ratio at outer={outer}: envelope={envelope}, rho={rho}"
            )
    e4_failures = [
        _integer(row, "target_outer_iter", directory / E4_INPUT)
        for row in e4_rows if row.get("refinement_status") != "pass"
    ]
    if _status_int_list(status, "exact_refinement_failures", source) != exact_failures:
        raise ValueError(f"{source}: status exact_refinement_failures mismatch")
    if _status_int_list(status, "e4_refinement_failures", source) != e4_failures:
        raise ValueError(f"{source}: status e4_refinement_failures mismatch")
    if _status_bool(status, "all_refinement_pass", source) != (
        not exact_failures and not e4_failures
    ):
        raise ValueError(f"{source}: status all_refinement_pass mismatch")
    if refinement_scope is not None:
        exact_boundary_incomplete = [
            _integer(
                row, "source_outer_iter", directory / EXACT_INPUT
            )
            for row in exact_rows
            if str(row.get("boundary_sensitivity_status", ""))
            == "incomplete"
        ]
        e4_boundary_incomplete = [
            _integer(row, "target_outer_iter", directory / E4_INPUT)
            for row in e4_rows
            if str(row.get("boundary_sensitivity_status", ""))
            == "incomplete"
        ]
        if _status_int_list(
            status,
            "exact_boundary_sensitivity_incomplete_outers",
            source,
        ) != exact_boundary_incomplete:
            raise ValueError(
                f"{source}: exact boundary-sensitivity status mismatch"
            )
        if _status_int_list(
            status,
            "e4_boundary_sensitivity_incomplete_targets",
            source,
        ) != e4_boundary_incomplete:
            raise ValueError(
                f"{source}: E4 boundary-sensitivity status mismatch"
            )

    e4_evidence = _e4_paper_evidence(
        config, e4_rows, directory / E4_INPUT
    )
    if not allow_legacy_pre_merton:
        if str(status.get("refinement_rule", "")) != str(
            e4_evidence["refinement_rule"]
        ):
            raise ValueError(f"{source}: status refinement_rule mismatch")
        if _status_int(status, "min_paper_checkpoint", source) != int(
            e4_evidence["min_paper_checkpoint"]
        ):
            raise ValueError(f"{source}: status min_paper_checkpoint mismatch")
        if _status_int_list(
            status, "excluded_initial_checkpoints", source
        ) != list(e4_evidence["excluded_initial_checkpoints"]):
            raise ValueError(
                f"{source}: status excluded_initial_checkpoints mismatch"
            )
        if _status_int_list(
            status, "paper_checkpoint_schedule", source
        ) != list(e4_evidence["paper_schedule"]):
            raise ValueError(
                f"{source}: status paper_checkpoint_schedule mismatch"
            )
        if _status_int_list(
            status, "e4_refinement_required_iterations", source
        ) != list(e4_evidence["required_iterations"]):
            raise ValueError(
                f"{source}: status e4_refinement_required_iterations mismatch"
            )
        recorded_statuses = status.get("e4_refinement_required_statuses")
        if not isinstance(recorded_statuses, Mapping) or {
            str(key): str(value) for key, value in recorded_statuses.items()
        } != dict(e4_evidence["required_statuses"]):
            raise ValueError(
                f"{source}: status e4_refinement_required_statuses mismatch"
            )
        if str(status.get("e4_refinement_evidence_status", "")) != str(
            e4_evidence["evidence_status"]
        ):
            raise ValueError(
                f"{source}: status e4_refinement_evidence_status mismatch"
            )
        if _status_int(status, "n_e4_refinement_pass", source) != sum(
            row.get("refinement_status") == "pass" for row in e4_rows
        ):
            raise ValueError(
                f"{source}: status n_e4_refinement_pass mismatch"
            )
        expected_paper_eligibility = bool(
            config.get("checkpoint_selection") == "all"
            and exact_rows
            and e4_rows
            and not undefined
            and not exact_failures
            and e4_evidence["evidence_status"] == "pass"
        )
    else:
        # The pre-Merton Liu status schema recorded the original
        # all-iteration/all-variant gate, but had no paper floor or required
        # evidence fields. Validate that historical contract before the E4
        # tolerance aggregator performs its in-memory semantic reassessment.
        expected_paper_eligibility = bool(
            config.get("checkpoint_selection") == "all"
            and exact_rows
            and e4_rows
            and not undefined
            and not exact_failures
            and not e4_failures
        )
    if _status_bool(status, "paper_aggregation_eligible", source) != (
        expected_paper_eligibility
    ):
        raise ValueError(f"{source}: status paper_aggregation_eligible mismatch")

    tolerance = _number(config, "ellipticity_tolerance", directory / CONFIG_INPUT)
    nonelliptic_e4 = [
        _integer(row, "target_outer_iter", directory / E4_INPUT)
        for row in e4_rows
        if (_number(row, "source_min_log_joint_eig", directory / E4_INPUT) <= tolerance
            or _number(row, "source_nonpositive_log_eig_fraction", directory / E4_INPUT) != 0.0)
    ]
    if _status_int_list(status, "nonelliptic_e4_targets", source) != nonelliptic_e4:
        raise ValueError(f"{source}: status nonelliptic_e4_targets mismatch")
    if _status_bool(status, "all_e4_source_policies_elliptic", source) != (not nonelliptic_e4):
        raise ValueError(f"{source}: status all_e4_source_policies_elliptic mismatch")
    if nonelliptic_e4:
        raise ValueError(
            f"{directory}: E4 source policy is not strictly elliptic at targets {nonelliptic_e4}"
        )
    nonelliptic_exact = [
        _integer(row, "source_outer_iter", directory / EXACT_INPUT)
        for row in exact_rows
        if (_number(row, "min_log_joint_eig", directory / EXACT_INPUT) <= tolerance
            or _number(row, "nonpositive_log_eig_fraction", directory / EXACT_INPUT) != 0.0)
    ]
    if nonelliptic_exact and require_exact_ellipticity:
        raise ValueError(
            f"{directory}: exact-map source policy is not strictly elliptic at outers "
            f"{nonelliptic_exact}"
        )


def _validate_artifact_hashes(directory: Path, status: Mapping[str, Any]) -> None:
    """Require all driver-managed inputs to match the successful-run manifest."""

    source = directory / STATUS_INPUT
    recorded = status.get("artifact_sha256")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{source}: missing artifact_sha256 manifest")
    expected_names = set(HASHED_INPUTS)
    recorded_names = {str(name) for name in recorded}
    if recorded_names != expected_names:
        raise ValueError(
            f"{source}: artifact_sha256 key set mismatch: "
            f"recorded={sorted(recorded_names)}, expected={sorted(expected_names)}"
        )
    for name in HASHED_INPUTS:
        path = directory / name
        expected = str(recorded.get(name, ""))
        if not path.is_file():
            raise ValueError(f"{source}: hashed artifact is missing: {path}")
        observed = sha256_file(path)
        if not expected or observed != expected:
            raise ValueError(
                f"{source}: artifact hash mismatch for {name}: "
                f"recorded={expected!r}, observed={observed}"
            )


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary statistics require a nonempty finite sample")
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else float("nan")
    sem = std / math.sqrt(count) if count > 1 else float("nan")
    half = (float(student_t.ppf(0.975, count - 1) * sem)
            if count > 1 else float("nan"))
    return {
        "n": count, "mean": mean, "std": std, "sem": sem,
        "ci95_low": mean - half, "ci95_high": mean + half,
        "min": float(np.min(array)), "max": float(np.max(array)),
    }


RATIO_METRICS = (
    "rho_empirical_X",
    "rho_exact",
    "rho_empirical_minus_exact",
)
RATIO_STAT_FIELDS = (
    "mean", "std", "sem", "ci95_low", "ci95_high", "min", "max",
)


def _ratio_comparison_rows(
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> List[Dict[str, Any]]:
    r"""Pair learned adjacent errors with the FD exact-map ratio at source k.

    The exact-map row at ``source_outer_iter=k`` uses the checkpoint
    :math:`\widetilde v_k` as its input and targets
    :math:`E(\alpha_k)`.  The empirical learned-step ratio at the same source
    index is therefore ``e_input_X(k+1) / e_input_X(k)``.  The final exact-map
    row has no learned ``k+1`` partner and remains in the exact-only outputs.
    """

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in exact_rows:
        grouped.setdefault(_integer(row, "seed", source), []).append(row)

    output: List[Dict[str, Any]] = []
    for seed in sorted(grouped):
        rows = sorted(
            grouped[seed],
            key=lambda row: _integer(row, "source_outer_iter", source),
        )
        for current, following in zip(rows, rows[1:]):
            current_outer = _integer(current, "source_outer_iter", source)
            following_outer = _integer(following, "source_outer_iter", source)
            target_outer = _integer(current, "target_value_outer_iter", source)
            if following_outer != current_outer + 1 or target_outer != following_outer:
                raise ValueError(
                    f"seed={seed}: empirical-ratio pairing is not contiguous at "
                    f"source outer={current_outer}; target={target_outer}, "
                    f"next source={following_outer}"
                )
            if (
                _integer(following, "frozen_policy_iter", source)
                != _integer(current, "greedy_policy_iter", source)
            ):
                raise ValueError(
                    f"seed={seed}: empirical/exact policy indices do not align "
                    f"between source outer={current_outer} and {following_outer}"
                )

            source_error = _number(
                current, "e_input_X", source, allow_blank=True
            )
            target_error = _number(
                following, "e_input_X", source, allow_blank=True
            )
            map_error = _number(
                current, "e_map_X", source, allow_blank=True
            )
            empirical_defined = bool(
                math.isfinite(source_error)
                and source_error > 0.0
                and math.isfinite(target_error)
                and target_error >= 0.0
            )
            exact_ratio = _number(
                current, "rho_exact", source, allow_blank=True
            )
            exact_defined = bool(
                _integer(current, "denominator_defined", source) == 1
                and math.isfinite(exact_ratio)
                and exact_ratio >= 0.0
            )
            empirical_ratio = (
                float(target_error / source_error)
                if empirical_defined else float("nan")
            )
            paired_defined = empirical_defined and exact_defined
            sensitivity = _number(
                current, "rho_sensitivity_envelope", source, allow_blank=True
            )
            refinement_status = str(current.get("refinement_status", ""))
            output.append({
                "seed": seed,
                "source_outer_iter": current_outer,
                "target_outer_iter": following_outer,
                "source_checkpoint_sha256": str(
                    current.get("checkpoint_sha256", "")
                ),
                "target_checkpoint_sha256": str(
                    following.get("checkpoint_sha256", "")
                ),
                "e_input_X_source": source_error,
                "e_input_X_target": target_error,
                "e_map_X_source": map_error,
                "rho_empirical_X": empirical_ratio,
                "rho_exact": exact_ratio,
                "rho_empirical_minus_exact": (
                    empirical_ratio - exact_ratio
                    if paired_defined else float("nan")
                ),
                "rho_sensitivity_envelope": sensitivity,
                "empirical_ratio_defined": int(empirical_defined),
                "exact_ratio_defined": int(exact_defined),
                "paired_ratio_defined": int(paired_defined),
                "denominator_defined": _integer(
                    current, "denominator_defined", source
                ),
                "refinement_status": refinement_status,
                "refinement_qualified": int(refinement_status == "pass"),
                "local_map_unmodified_on_xfd": _integer(
                    current, "local_map_unmodified_on_xfd", source
                ),
                "boundary_sensitivity_status": str(
                    current.get("boundary_sensitivity_status", "")
                ),
                "group": str(current.get("group", "")),
                "protocol_hash": str(current.get("protocol_hash", "")),
                "market_sha256": str(current.get("market_sha256", "")),
                "model_type": str(current.get("model_type", "pipinn")),
                "n_assets": str(current.get("n_assets", "")),
                "m_states": str(current.get("m_states", "")),
                "policy_extension": str(
                    current.get("policy_extension", "")
                ),
                "eval_margin": str(current.get("eval_margin", "")),
                "eval_x_margin": str(current.get("eval_x_margin", "")),
                "ev_w_min": str(current.get("ev_w_min", "")),
                "ev_w_max": str(current.get("ev_w_max", "")),
                "ev_x_min": str(current.get("ev_x_min", "")),
                "ev_x_max": str(current.get("ev_x_max", "")),
                "result_dir": str(current.get("result_dir", "")),
            })
    return output


def _blank_ratio_stats(prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{field}": "" for field in RATIO_STAT_FIELDS}


def _ratio_comparison_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> List[Dict[str, Any]]:
    """Summarize paired ratios without reducing the common seed sample."""

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            _integer(row, "source_outer_iter", source), []
        ).append(row)

    output: List[Dict[str, Any]] = []
    for outer in sorted(grouped):
        group = sorted(
            grouped[outer], key=lambda row: _integer(row, "seed", source)
        )
        seeds = [_integer(row, "seed", source) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"duplicate seed in ratio comparison at source outer={outer}: {seeds}"
            )
        targets = {
            _integer(row, "target_outer_iter", source) for row in group
        }
        if len(targets) != 1:
            raise ValueError(
                f"cross-seed target mismatch at source outer={outer}: {targets}"
            )
        paired_complete = all(
            _integer(row, "paired_ratio_defined", source) == 1 for row in group
        )
        refinement_pass = sum(
            str(row.get("refinement_status", "")) == "pass" for row in group
        )
        refinement_fail = sum(
            str(row.get("refinement_status", "")) == "fail" for row in group
        )
        locally_unmodified = sum(
            _integer(row, "local_map_unmodified_on_xfd", source) == 1
            for row in group
        )
        row_out: Dict[str, Any] = {
            "source_outer_iter": outer,
            "target_outer_iter": next(iter(targets)),
            "n_expected": len(group),
            "n_paired_defined": sum(
                _integer(row, "paired_ratio_defined", source) == 1
                for row in group
            ),
            "n_refinement_pass": refinement_pass,
            "n_refinement_fail": refinement_fail,
            "n_refinement_other": len(group) - refinement_pass - refinement_fail,
            "n_locally_unmodified": locally_unmodified,
            "summary_status": (
                "complete_refinement_qualified"
                if (
                    paired_complete
                    and refinement_pass == len(group)
                    and locally_unmodified == len(group)
                )
                else (
                    "complete_exploratory_partial_sensitivity"
                    if paired_complete
                    else "unavailable_incomplete_common_seed_sample"
                )
            ),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in RATIO_METRICS:
            if paired_complete:
                stats = _stats(
                    [_number(row, metric, source) for row in group]
                )
                row_out.update({
                    f"{metric}_{field}": stats[field]
                    for field in RATIO_STAT_FIELDS
                })
            else:
                row_out.update(_blank_ratio_stats(metric))

        envelope_values = [
            _number(
                row, "rho_sensitivity_envelope", source, allow_blank=True
            )
            for row in group
        ]
        if all(math.isfinite(value) for value in envelope_values):
            envelope_stats = _stats(envelope_values)
            row_out.update({
                f"rho_sensitivity_envelope_{field}": envelope_stats[field]
                for field in RATIO_STAT_FIELDS
            })
        else:
            row_out.update(_blank_ratio_stats("rho_sensitivity_envelope"))
        output.append(row_out)
    return output


def _exact_ratio_plot_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> List[Dict[str, Any]]:
    """Build full-schedule exact-ratio rows for exact-only plotting."""

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            _integer(row, "source_outer_iter", source), []
        ).append(row)
    output: List[Dict[str, Any]] = []
    for outer in sorted(grouped):
        group = sorted(
            grouped[outer], key=lambda row: _integer(row, "seed", source)
        )
        seeds = [_integer(row, "seed", source) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"duplicate seed in exact-ratio plot at source outer={outer}"
            )
        exact_values = [
            _number(row, "rho_exact", source, allow_blank=True)
            for row in group
        ]
        complete = all(
            _integer(row, "denominator_defined", source) == 1
            and math.isfinite(value)
            for row, value in zip(group, exact_values)
        )
        row_out: Dict[str, Any] = {
            "source_outer_iter": outer,
            "n_expected": len(group),
            "n_paired_defined": len(group) if complete else sum(
                _integer(row, "denominator_defined", source) == 1
                and math.isfinite(value)
                for row, value in zip(group, exact_values)
            ),
            "n_refinement_pass": sum(
                str(row.get("refinement_status", "")) == "pass"
                for row in group
            ),
            "n_locally_unmodified": sum(
                _integer(row, "local_map_unmodified_on_xfd", source) == 1
                for row in group
            ),
            "summary_status": (
                "complete_refinement_qualified"
                if (
                    complete
                    and all(
                        str(row.get("refinement_status", "")) == "pass"
                        for row in group
                    )
                    and all(
                        _integer(
                            row, "local_map_unmodified_on_xfd", source
                        ) == 1
                        for row in group
                    )
                )
                else (
                    "complete_exploratory_partial_sensitivity"
                    if complete
                    else "unavailable_incomplete_common_seed_sample"
                )
            ),
        }
        if complete:
            stats = _stats(exact_values)
            row_out.update({
                f"rho_exact_{field}": stats[field]
                for field in RATIO_STAT_FIELDS
            })
        else:
            row_out.update(_blank_ratio_stats("rho_exact"))
        envelope = [
            _number(
                row, "rho_sensitivity_envelope", source, allow_blank=True
            )
            for row in group
        ]
        if all(math.isfinite(value) for value in envelope):
            stats = _stats(envelope)
            row_out.update({
                f"rho_sensitivity_envelope_{field}": stats[field]
                for field in RATIO_STAT_FIELDS
            })
        else:
            row_out.update(_blank_ratio_stats("rho_sensitivity_envelope"))
        output.append(row_out)
    return output


def _annotate_ratio_floor_support(
    exact_rows: Sequence[Mapping[str, Any]],
    empirical_summary_rows: Sequence[Dict[str, Any]],
    exact_summary_rows: Sequence[Dict[str, Any]],
    *,
    floor_multiple: float,
    floor_value: Optional[float],
    source: Path,
) -> Dict[str, Any]:
    """Attach Merton-style common floor support to ratio plot summaries.

    The classification is display-only.  It does not remove raw exact-map or
    empirical rows and does not participate in refinement or paper-eligibility
    decisions.
    """

    multiple = float(floor_multiple)
    if not math.isfinite(multiple) or multiple < 0.0:
        raise ValueError("floor_multiple must be finite and nonnegative")
    if floor_value is not None:
        floor_value = float(floor_value)
        if not math.isfinite(floor_value) or floor_value < 0.0:
            raise ValueError("floor_value must be finite and nonnegative")

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in exact_rows:
        grouped.setdefault(_integer(row, "seed", source), []).append(row)
    if not grouped:
        raise ValueError("ratio floor classification requires exact-map rows")

    floors: Dict[int, float] = {}
    tail_counts: Dict[int, int] = {}
    exact_regular: Dict[Tuple[int, int], bool] = {}
    empirical_regular: Dict[Tuple[int, int], bool] = {}
    for seed, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: _integer(row, "source_outer_iter", source),
        )
        errors = np.asarray(
            [
                _number(row, "e_input_X", source, allow_blank=True)
                for row in ordered
            ],
            dtype=float,
        )
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size == 0:
            raise ValueError(f"seed={seed}: no finite e_input_X floor values")
        tail_count = max(1, int(math.ceil(0.10 * finite_errors.size)))
        tail_counts[seed] = tail_count
        floors[seed] = (
            float(floor_value)
            if floor_value is not None
            else float(np.median(finite_errors[-tail_count:]))
        )
        for row in ordered:
            outer = _integer(row, "source_outer_iter", source)
            error = _number(row, "e_input_X", source, allow_blank=True)
            exact_ratio = _number(
                row, "rho_exact", source, allow_blank=True
            )
            exact_defined = (
                _integer(row, "denominator_defined", source) == 1
                and math.isfinite(exact_ratio)
            )
            above_floor = (
                True
                if multiple == 0.0
                else (
                    math.isfinite(error)
                    and error > multiple * floors[seed]
                )
            )
            exact_regular[(seed, outer)] = exact_defined and above_floor

    empirical_by_seed_outer: Dict[Tuple[int, int], bool] = {}
    for row in exact_rows:
        # The empirical series uses adjacent exact-row input errors.  Its
        # source support therefore excludes the final exact-map checkpoint.
        seed = _integer(row, "seed", source)
        outer = _integer(row, "source_outer_iter", source)
        empirical_by_seed_outer[(seed, outer)] = False
    exact_by_seed = {
        seed: sorted(
            rows,
            key=lambda row: _integer(row, "source_outer_iter", source),
        )
        for seed, rows in grouped.items()
    }
    for seed, rows in exact_by_seed.items():
        for current, following in zip(rows, rows[1:]):
            outer = _integer(current, "source_outer_iter", source)
            next_outer = _integer(following, "source_outer_iter", source)
            current_error = _number(
                current, "e_input_X", source, allow_blank=True
            )
            next_error = _number(
                following, "e_input_X", source, allow_blank=True
            )
            defined = (
                next_outer == outer + 1
                and math.isfinite(current_error)
                and current_error > 0.0
                and math.isfinite(next_error)
                and next_error >= 0.0
            )
            above_floor = (
                True
                if multiple == 0.0
                else current_error > multiple * floors[seed]
            )
            empirical_regular[(seed, outer)] = defined and above_floor

    seeds = sorted(grouped)

    def annotate(
        summary_rows: Sequence[Dict[str, Any]],
        support: Mapping[Tuple[int, int], bool],
    ) -> Tuple[List[int], List[int]]:
        common: List[int] = []
        dominated: List[int] = []
        for row in summary_rows:
            outer = int(row["source_outer_iter"])
            ratio_iter = outer - 1
            is_common = all(
                bool(support.get((seed, outer), False)) for seed in seeds
            )
            row["ratio_iter"] = ratio_iter
            row["floor_multiple"] = multiple
            row["common_regular"] = int(is_common)
            (common if is_common else dominated).append(ratio_iter)
        return sorted(common), sorted(dominated)

    empirical_common, empirical_dominated = annotate(
        empirical_summary_rows, empirical_regular
    )
    exact_common, exact_dominated = annotate(
        exact_summary_rows, exact_regular
    )
    return {
        "definition": (
            "a ratio source is regular iff its e_input_X is strictly above "
            "floor_multiple times the seed late-input-error floor"
        ),
        "comparison": "strict_greater_than",
        "floor_multiple": multiple,
        "tail_fraction": 0.10,
        "floor_source": (
            "explicit_absolute_base_floor"
            if floor_value is not None
            else "median_last_ceil_10pct_e_input_X"
        ),
        "explicit_floor_value": floor_value,
        "floors_by_seed": {
            str(seed): floors[seed] for seed in seeds
        },
        "tail_counts_by_seed": {
            str(seed): tail_counts[seed] for seed in seeds
        },
        "empirical_common_ratio_iters": empirical_common,
        "empirical_floor_dominated_ratio_iters": empirical_dominated,
        "exact_common_ratio_iters": exact_common,
        "exact_floor_dominated_ratio_iters": exact_dominated,
        "raw_ratio_rows_retained": True,
        "display_only": True,
        "not_an_fd_discretization_floor": True,
        "positive_multiple_is_exploratory": multiple > 0.0,
        "refinement_and_claim_status_unchanged": True,
    }


RATIO_WORST_METRICS = (
    ("max_rho_empirical_X", "max_rho_empirical_X_outer"),
    ("max_rho_exact", "max_rho_exact_outer"),
    (
        "max_rho_sensitivity_envelope",
        "max_rho_sensitivity_envelope_outer",
    ),
)


def _ratio_comparison_worst_per_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: Path,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_integer(row, "seed", source), []).append(row)

    output: List[Dict[str, Any]] = []
    for seed in sorted(grouped):
        group = sorted(
            grouped[seed],
            key=lambda row: _integer(row, "source_outer_iter", source),
        )
        paired_complete = all(
            _integer(row, "paired_ratio_defined", source) == 1 for row in group
        )
        empirical_max = (
            _argmax_complete(
                group,
                metric="rho_empirical_X",
                index_field="source_outer_iter",
                source=source,
            )
            if paired_complete and group else None
        )
        exact_max = (
            _argmax_complete(
                group,
                metric="rho_exact",
                index_field="source_outer_iter",
                source=source,
            )
            if paired_complete and group else None
        )
        envelope_max = (
            _argmax_complete(
                group,
                metric="rho_sensitivity_envelope",
                index_field="source_outer_iter",
                source=source,
            )
            if group else None
        )
        row_out: Dict[str, Any] = {
            "seed": seed,
            "all_paired_ratios_defined": int(paired_complete),
            "all_source_refinement_pass": int(all(
                str(row.get("refinement_status", "")) == "pass"
                for row in group
            )),
            "all_locally_unmodified": int(all(
                _integer(row, "local_map_unmodified_on_xfd", source) == 1
                for row in group
            )),
            "n_pairs": len(group),
            "result_dir": str(group[0].get("result_dir", "")) if group else "",
        }
        for metric, result in (
            ("rho_empirical_X", empirical_max),
            ("rho_exact", exact_max),
            ("rho_sensitivity_envelope", envelope_max),
        ):
            row_out[f"max_{metric}"] = (
                result[0] if result is not None else ""
            )
            row_out[f"max_{metric}_outer"] = (
                result[1] if result is not None else ""
            )
        output.append(row_out)
    return output


def _ratio_comparison_worst_summary_rows(
    per_seed: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    n_expected = len(per_seed)
    for metric, outer_field in RATIO_WORST_METRICS:
        complete = [
            row for row in per_seed
            if str(row.get(metric, "")).strip() != ""
        ]
        row_out: Dict[str, Any] = {
            "metric": metric,
            "n_expected": n_expected,
            "n_complete": len(complete),
            "summary_status": (
                "complete_common_seed_sample"
                if n_expected > 0 and len(complete) == n_expected
                else "unavailable_incomplete_seed_sample"
            ),
            "n": "", "mean": "", "std": "", "sem": "",
            "ci95_low": "", "ci95_high": "", "min": "", "max": "",
            "global_max": "", "global_max_seed": "", "global_max_outer": "",
            "seeds": ",".join(str(row["seed"]) for row in per_seed),
        }
        if n_expected > 0 and len(complete) == n_expected:
            stats = _stats([float(row[metric]) for row in complete])
            worst = max(complete, key=lambda row: float(row[metric]))
            row_out.update(stats)
            row_out.update({
                "global_max": float(worst[metric]),
                "global_max_seed": int(worst["seed"]),
                "global_max_outer": int(worst[outer_field]),
            })
        output.append(row_out)
    return output


def _parse_plot_formats(text: str) -> List[str]:
    formats: List[str] = []
    for token in re.split(r"[\s,]+", str(text).strip().lower()):
        if not token:
            continue
        if token not in RATIO_PLOT_FORMATS:
            raise ValueError(
                f"unsupported plot format {token!r}; "
                f"choose from {RATIO_PLOT_FORMATS}"
            )
        if token not in formats:
            formats.append(token)
    if not formats:
        raise ValueError("--formats must contain at least one plot format")
    return formats


def _plot_ratio_comparison(
    summary_rows: Sequence[Mapping[str, Any]],
    exact_summary_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    formats: Sequence[str],
    fig_width: float,
    fig_height: float,
    font_size: float,
    font_family: str,
    line_width: float,
    band_alpha: float,
    floor_alpha: float,
    marker_size: float,
    grid_alpha: float,
    dpi: int,
    ratio_y_scale: str,
    plot_sensitivity_envelope: bool,
    ratio_series: str,
) -> List[str]:
    """Render a Merton-style learned-step / FD-map ratio comparison."""

    active_rows = (
        exact_summary_rows if ratio_series == "exact" else summary_rows
    )
    complete = [
        row for row in active_rows
        if str(row.get("summary_status", "")).startswith("complete_")
    ]
    if len(complete) != len(active_rows) or not complete:
        raise ValueError(
            "ratio plot requires a complete common-seed sample at every "
            "plotted source outer iteration"
        )
    ordered = sorted(
        complete, key=lambda row: int(row["source_outer_iter"])
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x = np.asarray([
        int(row.get("ratio_iter", int(row["source_outer_iter"]) - 1))
        for row in ordered
    ], dtype=float)
    common_regular = np.asarray([
        int(row.get("common_regular", 1)) == 1 for row in ordered
    ], dtype=bool)

    def values(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(
            [float(row[f"{prefix}_mean"]) for row in ordered],
            dtype=float,
        )
        std = np.asarray(
            [
                float(row[f"{prefix}_std"])
                if str(row[f"{prefix}_std"]).strip().lower()
                not in ("", "nan")
                else float("nan")
                for row in ordered
            ],
            dtype=float,
        )
        if ratio_y_scale == "log" and np.any(mean <= 0.0):
            raise ValueError(
                f"{prefix} means must be positive for "
                "--ratio-y-scale=log; use --ratio-y-scale=linear"
            )
        return mean, std

    def band(
        mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = mean - std
        if ratio_y_scale == "log":
            lower = np.where(lower > 0.0, lower, np.nan)
        else:
            lower = np.maximum(lower, 0.0)
        return lower, mean + std

    empirical_mean: Optional[np.ndarray] = None
    empirical_std: Optional[np.ndarray] = None
    empirical_lower: Optional[np.ndarray] = None
    empirical_upper: Optional[np.ndarray] = None
    if ratio_series in {"empirical", "both"}:
        empirical_mean, empirical_std = values("rho_empirical_X")
        empirical_lower, empirical_upper = band(
            empirical_mean, empirical_std
        )

    exact_mean: Optional[np.ndarray] = None
    exact_std: Optional[np.ndarray] = None
    exact_lower: Optional[np.ndarray] = None
    exact_upper: Optional[np.ndarray] = None
    if ratio_series in {"exact", "both"}:
        exact_mean, exact_std = values("rho_exact")
        exact_lower, exact_upper = band(exact_mean, exact_std)

    empirical_color = "#0072B2"
    exact_color = (
        "#0072B2" if ratio_series == "exact" else "#D55E00"
    )

    def contiguous_segments(mask: np.ndarray) -> List[np.ndarray]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        splits = np.where(
            np.diff(x[indices]).astype(int) != 1
        )[0] + 1
        return [
            segment
            for segment in np.split(indices, splits)
            if segment.size
        ]

    rc_params: Dict[str, Any] = {"font.size": float(font_size)}
    if str(font_family).strip():
        rc_params["font.family"] = str(font_family).strip()
    output_names: List[str] = []
    with matplotlib.rc_context(rc_params):
        fig, ax = plt.subplots(
            figsize=(float(fig_width), float(fig_height))
        )
        floor_label_used = False

        def plot_series(
            mean: np.ndarray,
            std: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            *,
            color: str,
            linestyle: str,
            label: str,
        ) -> None:
            nonlocal floor_label_used
            line_label_used = False
            for segment in contiguous_segments(common_regular):
                if np.any(np.isfinite(std[segment])):
                    ax.fill_between(
                        x[segment],
                        lower[segment],
                        upper[segment],
                        color=color,
                        alpha=float(band_alpha),
                        linewidth=0.0,
                        zorder=1,
                    )
                ax.plot(
                    x[segment],
                    mean[segment],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=float(marker_size),
                    linewidth=float(line_width),
                    label=label if not line_label_used else None,
                    zorder=3,
                )
                line_label_used = True
            dominated = ~common_regular
            if np.any(dominated):
                ax.scatter(
                    x[dominated],
                    mean[dominated],
                    color="#9E9E9E",
                    marker="x",
                    s=max(20.0, float(marker_size) ** 2),
                    linewidths=max(1.0, 0.75 * float(line_width)),
                    alpha=float(floor_alpha),
                    label=(
                        "Floor-dominated" if not floor_label_used else None
                    ),
                    zorder=4,
                )
                floor_label_used = True

        if empirical_mean is not None:
            plot_series(
                empirical_mean,
                empirical_std,
                empirical_lower,
                empirical_upper,
                color=empirical_color,
                linestyle="-",
                label=r"Empirical ratio $\widehat{\varrho}_n$",
            )
        if exact_mean is not None:
            plot_series(
                exact_mean,
                exact_std,
                exact_lower,
                exact_upper,
                color=exact_color,
                # A standalone exact-map figure follows the empirical
                # presentation with a solid point-connected mean curve.
                # Keep the dotted line only in the two-series overlay so the
                # empirical and exact curves remain distinguishable.
                linestyle="-" if ratio_series == "exact" else ":",
                label=r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$",
            )
        if plot_sensitivity_envelope:
            raw_envelope = [
                str(row.get("rho_sensitivity_envelope_max", "")).strip()
                for row in ordered
            ]
            if any(
                not value or value.lower() == "nan"
                for value in raw_envelope
            ):
                raise ValueError(
                    "--plot-sensitivity-envelope requires a finite envelope "
                    "for every plotted source outer iteration"
                )
            envelope = np.asarray(
                [float(value) for value in raw_envelope],
                dtype=float,
            )
            if ratio_y_scale == "log" and np.any(envelope <= 0.0):
                raise ValueError(
                    "sensitivity envelope must be positive for a log ratio plot"
                )
            envelope_mask = common_regular
            ax.plot(
                x[envelope_mask], envelope[envelope_mask],
                color=exact_color, linestyle="-.", linewidth=max(
                    1.0, 0.75 * float(line_width)
                ),
                label="FD sensitivity envelope (seed max)",
            )
        ax.axhline(
            1.0, color="black", linestyle="--",
            linewidth=max(1.0, 0.8 * float(line_width)),
            label="Contraction threshold",
        )
        if ratio_y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(r"Iteration")
        if ratio_series == "empirical":
            ax.set_ylabel(r"Empirical ratio $\widehat{\varrho}_n$")
        elif ratio_series == "exact":
            ax.set_ylabel(r"Exact-map ratio $\varrho_n^{\mathrm{FD}}$")
        else:
            ax.set_ylabel("Error ratio")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
        ax.tick_params(axis="both", labelsize=0.9 * float(font_size))
        ax.grid(
            True, which="both", alpha=float(grid_alpha), linewidth=0.6
        )
        ax.legend(
            frameon=False,
            bbox_to_anchor=(0.99, 0.95),
            loc="upper right" if ratio_series == "exact" else "upper right",
            prop={"size": 0.8 * float(font_size)},
        )
        fig.tight_layout()
        stem = {
            "empirical": "empirical_ratio",
            "exact": "exact_map_contraction",
            "both": "ratio_comparison",
        }[ratio_series]
        for suffix in formats:
            name = f"{stem}.{suffix}"
            fig.savefig(output / name, dpi=int(dpi), bbox_inches="tight")
            output_names.append(name)
        plt.close(fig)
    return output_names


def _summary_rows(rows: Sequence[Mapping[str, Any]], *, index_field: str,
                  metrics: Sequence[str], source: Path) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_integer(row, index_field, source), []).append(row)
    output: List[Dict[str, Any]] = []
    for index in sorted(grouped):
        group = grouped[index]
        seeds = sorted(_integer(row, "seed", source) for row in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed at {index_field}={index}: {seeds}")
        for metric in metrics:
            values = [_number(row, metric, source, allow_blank=True) for row in group]
            finite = [value for value in values if math.isfinite(value)]
            if not finite:
                # Sensitivity envelopes are intentionally blank for runs that
                # were explicitly generated without a refinement audit.
                if metric in OPTIONAL_SUMMARY_METRICS:
                    continue
                raise ValueError(f"no finite {metric} values at {index_field}={index}")
            if len(finite) != len(values):
                raise ValueError(
                    f"partly missing {metric} at {index_field}={index}; refusing to drop seeds"
                )
            stats = _stats(finite)
            output.append({
                index_field: index, "metric": metric, **stats,
                "seeds": ",".join(str(seed) for seed in seeds),
            })
    return output


WORST_METRICS = (
    ("max_rho_exact", "max_rho_exact_outer"),
    ("max_rho_sensitivity_envelope", "max_rho_sensitivity_envelope_outer"),
)


def _argmax_complete(rows: Sequence[Mapping[str, Any]], *, metric: str,
                     index_field: str, source: Path) -> Optional[tuple[float, int]]:
    """Return a deterministic argmax only for a complete finite schedule.

    Refusing a partly finite schedule is important here: a worst-case paper
    summary must not silently discard an undefined checkpoint.  Ties are
    resolved by the lowest outer iteration because ``rows`` are sorted first.
    """

    candidates: List[tuple[int, float]] = []
    for row in sorted(rows, key=lambda item: _integer(item, index_field, source)):
        value = _number(row, metric, source, allow_blank=True)
        if not math.isfinite(value):
            return None
        candidates.append((_integer(row, index_field, source), value))
    if not candidates:
        return None
    outer, value = max(candidates, key=lambda item: item[1])
    return float(value), int(outer)


def _worst_case_rows(exact_rows: Sequence[Mapping[str, Any]],
                     *, source: Path) -> List[Dict[str, Any]]:
    """Build one schedule-complete worst-case record per seed."""

    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in exact_rows:
        grouped.setdefault(_integer(row, "seed", source), []).append(row)
    output: List[Dict[str, Any]] = []
    for seed in sorted(grouped):
        rows = grouped[seed]
        denominator_complete = all(
            _integer(row, "denominator_defined", source) == 1
            and math.isfinite(_number(row, "rho_exact", source, allow_blank=True))
            for row in rows
        )
        sensitivity_complete = all(
            row.get("refinement_status") == "pass"
            and math.isfinite(
                _number(row, "rho_sensitivity_envelope", source, allow_blank=True)
            )
            for row in rows
        )
        locally_unmodified = all(
            _integer(row, "local_map_unmodified_on_xfd", source) == 1
            for row in rows
        )
        exact_max = (_argmax_complete(
            rows, metric="rho_exact", index_field="source_outer_iter", source=source,
        ) if denominator_complete else None)
        envelope_max = (_argmax_complete(
            rows, metric="rho_sensitivity_envelope",
            index_field="source_outer_iter", source=source,
        ) if sensitivity_complete else None)
        safe_below_one = bool(
            denominator_complete
            and sensitivity_complete
            and locally_unmodified
            and exact_max is not None
            and exact_max[0] < 1.0
            and envelope_max is not None
            and envelope_max[0] < 1.0
        )
        output.append({
            "seed": seed,
            "max_rho_exact": exact_max[0] if exact_max is not None else "",
            "max_rho_exact_outer": exact_max[1] if exact_max is not None else "",
            "max_rho_sensitivity_envelope": (
                envelope_max[0] if envelope_max is not None else ""
            ),
            "max_rho_sensitivity_envelope_outer": (
                envelope_max[1] if envelope_max is not None else ""
            ),
            "all_denominators_defined": int(denominator_complete),
            "all_exact_sensitivity_pass": int(sensitivity_complete),
            "all_locally_unmodified": int(locally_unmodified),
            "finite_domain_all_tested_ratios_below_one": int(safe_below_one),
            "n_outer": len(rows),
            "result_dir": str(rows[0].get("result_dir", "")),
        })
    return output


def _worst_summary_rows(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Summarize seedwise maxima without reducing the common seed sample."""

    output: List[Dict[str, Any]] = []
    n_expected = len(per_seed)
    for metric, outer_field in WORST_METRICS:
        complete = [row for row in per_seed if str(row.get(metric, "")).strip() != ""]
        row_out: Dict[str, Any] = {
            "metric": metric,
            "n_expected": n_expected,
            "n_complete": len(complete),
            "summary_status": (
                "complete_common_seed_sample"
                if len(complete) == n_expected and n_expected > 0
                else "unavailable_incomplete_seed_sample"
            ),
            "n": "", "mean": "", "std": "", "sem": "",
            "ci95_low": "", "ci95_high": "", "min": "", "max": "",
            "global_max": "", "global_max_seed": "", "global_max_outer": "",
            "seeds": ",".join(str(row["seed"]) for row in per_seed),
        }
        if len(complete) == n_expected and n_expected > 0:
            values = [float(row[metric]) for row in complete]
            stats = _stats(values)
            # Rows arrive in seed order.  ``max`` therefore resolves an exact
            # value tie by the lowest seed; each seed's tie was already
            # resolved by the lowest outer iteration.
            worst = max(complete, key=lambda item: float(item[metric]))
            row_out.update(stats)
            row_out.update({
                "global_max": float(worst[metric]),
                "global_max_seed": int(worst["seed"]),
                "global_max_outer": int(worst[outer_field]),
            })
        output.append(row_out)
    return output


def _prepare_output(output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in AGG_MANAGED_OUTPUTS if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"aggregate output already exists ({existing}); pass --overwrite to replace managed files"
        )
    for name in AGG_MANAGED_OUTPUTS:
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _check_output(output: Path, overwrite: bool) -> bool:
    existing = [name for name in AGG_MANAGED_OUTPUTS if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"aggregate output already exists ({existing}); pass --overwrite to replace managed files"
        )
    return bool(existing)


def _commit_staged_output(stage: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(
        prefix=".liu-exact-aggregate-backup-", dir=str(output.parent)
    ))
    moved_old: List[tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        for name in AGG_MANAGED_OUTPUTS:
            original = output / name
            if original.exists() or original.is_symlink():
                saved = backup / name
                os.replace(original, saved)
                moved_old.append((saved, original))
        names = [
            name for name in AGG_MANAGED_OUTPUTS
            if name != "_SUCCESS_EXACT_MAP_AGG"
            and ((stage / name).is_file() or (stage / name).is_symlink())
        ]
        if (stage / "_SUCCESS_EXACT_MAP_AGG").is_file():
            names.append("_SUCCESS_EXACT_MAP_AGG")
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


def aggregate(result_dirs: Sequence[Path], output: Path, *, expected_seeds: Sequence[int],
              min_seeds: int, allow_undefined_denominators: bool,
              allow_partial_sensitivity: bool, require_locally_unmodified: bool,
              overwrite: bool, plot_ratios: bool = False,
              plot_formats: Sequence[str] = ("png", "pdf"),
              fig_width: float = 6.5, fig_height: float = 4.2,
              font_size: float = 10.0, font_family: str = "",
              line_width: float = 2.0, band_alpha: float = 0.18,
              floor_alpha: float = 0.80,
              marker_size: float = 4.0, grid_alpha: float = 0.22,
              dpi: int = 300, ratio_y_scale: str = "log",
              plot_sensitivity_envelope: bool = False,
              ratio_series: str = "both",
              floor_multiple: float = 0.0,
              floor_value: Optional[float] = None,
              target_label: str = "") -> Mapping[str, Any]:
    had_managed_output = _check_output(output, overwrite)
    format_text = (
        plot_formats
        if isinstance(plot_formats, str)
        else ",".join(str(value) for value in plot_formats)
    )
    resolved_plot_formats = (
        _parse_plot_formats(str(format_text)) if plot_ratios else []
    )
    if ratio_y_scale not in {"linear", "log"}:
        raise ValueError("--ratio-y-scale must be one of: linear, log")
    if ratio_series not in {"empirical", "exact", "both"}:
        raise ValueError(
            "--ratio-series must be one of: empirical, exact, both"
        )
    for name, value in (
        ("fig_width", fig_width), ("fig_height", fig_height),
        ("font_size", font_size), ("line_width", line_width),
        ("marker_size", marker_size),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(float(band_alpha)) or not 0.0 <= float(band_alpha) <= 1.0:
        raise ValueError("band_alpha must lie in [0, 1]")
    if not math.isfinite(float(floor_alpha)) or not 0.0 <= float(floor_alpha) <= 1.0:
        raise ValueError("floor_alpha must lie in [0, 1]")
    if not math.isfinite(float(grid_alpha)) or not 0.0 <= float(grid_alpha) <= 1.0:
        raise ValueError("grid_alpha must lie in [0, 1]")
    if not math.isfinite(float(floor_multiple)) or float(floor_multiple) < 0.0:
        raise ValueError("floor_multiple must be finite and nonnegative")
    if floor_value is not None and (
        not math.isfinite(float(floor_value)) or float(floor_value) < 0.0
    ):
        raise ValueError("floor_value must be finite and nonnegative")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    exact_all: List[Dict[str, Any]] = []
    e4_all: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    seen_seed: Dict[int, Path] = {}

    try:
        for directory in result_dirs:
            status = read_json(directory / STATUS_INPUT)
            if (status.get("status") != "success"
                    or not (directory / "_SUCCESS_EXACT_MAP").is_file()
                    or (directory / "_FAILED_EXACT_MAP").exists()):
                raise ValueError(f"exact-map result is not successful: {directory}")
            _validate_artifact_hashes(directory, status)
            config = read_json(directory / CONFIG_INPUT)
            exact_rows = read_csv(directory / EXACT_INPUT)
            e4_rows = read_csv(directory / E4_INPUT)
            exact_refinement_rows = read_csv(directory / "exact_map_refinement.csv")
            e4_refinement_rows = read_csv(directory / "e4_approximation_refinement.csv")
            _validate_status_contract(
                directory,
                status,
                config,
                exact_rows,
                e4_rows,
                exact_refinement_rows,
                e4_refinement_rows,
            )
            _validate_provenance(directory, config, exact_rows, e4_rows)
            e4_evidence = _e4_paper_evidence(
                config, e4_rows, directory / E4_INPUT
            )
            seed = int(_identity(exact_rows, "seed", directory / EXACT_INPUT))
            if seed in seen_seed:
                raise ValueError(f"duplicate exact-map seed={seed}: {seen_seed[seed]} and {directory}")
            seen_seed[seed] = directory
            for rows, name in ((exact_rows, EXACT_INPUT), (e4_rows, E4_INPUT)):
                if any(_integer(row, "is_primary", directory / name) != 1 for row in rows):
                    raise ValueError(f"{directory / name} contains non-primary variants")
                if any(_integer(row, "seed", directory / name) != seed for row in rows):
                    raise ValueError(f"{directory / name} mixes seeds")
            exact_schedule = [_integer(row, "source_outer_iter", directory / EXACT_INPUT)
                              for row in exact_rows]
            e4_schedule = [_integer(row, "target_outer_iter", directory / E4_INPUT)
                           for row in e4_rows]
            if exact_schedule != sorted(set(exact_schedule)):
                raise ValueError(f"non-unique or unsorted exact-map schedule: {directory}")
            if e4_schedule != exact_schedule:
                raise ValueError(f"E4/exact-map schedules differ: {directory}")
            undefined = [row["source_outer_iter"] for row in exact_rows
                         if _integer(row, "denominator_defined", directory / EXACT_INPUT) != 1
                         or not math.isfinite(_number(row, "rho_exact", directory / EXACT_INPUT,
                                                     allow_blank=True))]
            if undefined and not allow_undefined_denominators:
                raise ValueError(
                    f"seed={seed} has undefined exact-map denominators at outer={undefined}; "
                    "these rows cannot be silently omitted"
                )
            if not allow_partial_sensitivity and (
                any(
                    row.get("refinement_status") != "pass"
                    for row in exact_rows
                )
                or e4_evidence["evidence_status"] != "pass"
            ):
                raise ValueError(
                    f"seed={seed} does not pass the exact-map audit and "
                    "Merton-style E4 initial/first/last/worst evidence audit"
                )
            if require_locally_unmodified and any(
                _integer(row, "local_map_unmodified_on_xfd", directory / EXACT_INPUT) != 1
                for row in exact_rows
            ):
                raise ValueError(f"seed={seed} activates a guard/clip on the sampled FD domain")

            group = _identity(exact_rows, "group", directory / EXACT_INPUT)
            protocol = _identity(exact_rows, "protocol_hash", directory / EXACT_INPUT)
            market = _identity(exact_rows, "market_sha256", directory / EXACT_INPUT)
            if _identity(e4_rows, "market_sha256", directory / E4_INPUT) != market:
                raise ValueError(f"E4 and exact-map market snapshot mismatch: {directory}")
            if _identity(e4_rows, "group", directory / E4_INPUT) != group:
                raise ValueError(f"E4 and exact-map group mismatch: {directory}")
            if _identity(e4_rows, "protocol_hash", directory / E4_INPUT) != protocol:
                raise ValueError(f"E4 and exact-map protocol mismatch: {directory}")
            if str(config.get("protocol_hash", "")) != protocol:
                raise ValueError(f"config/CSV protocol mismatch: {directory}")
            for row in exact_rows:
                row["result_dir"] = str(directory)
                exact_all.append(row)
            for row in e4_rows:
                row["result_dir"] = str(directory)
                if _integer(
                    row, "target_outer_iter", directory / E4_INPUT
                ) in set(e4_evidence["paper_schedule"]):
                    e4_all.append(row)
            records.append({
                "seed": seed, "directory": str(directory), "group": group,
                "protocol_hash": protocol, "market_sha256": market,
                "policy_extension": str(config.get("policy_extension", "")),
                "refinement_rule": e4_evidence["refinement_rule"],
                "refinement_scope": (
                    str(config.get("refinement_scope"))
                    if config.get("refinement_scope") is not None
                    else "legacy_cartesian_including_boundary"
                ),
                "boundary_sensitivity_role": (
                    str(config.get("boundary_sensitivity_role"))
                    if config.get("boundary_sensitivity_role") is not None
                    else "legacy_part_of_refinement_gate"
                ),
                "map_definition": str(config.get("map_definition", "")),
                "evaluation_window": _canonical_evaluation_window(
                    config, directory / CONFIG_INPUT
                ),
                "domain_design": _canonical_domain_design(
                    config, directory / CONFIG_INPUT
                ),
                "exact_schedule": exact_schedule, "e4_schedule": e4_schedule,
                "paper_e4_schedule": e4_evidence["paper_schedule"],
                "min_paper_checkpoint": e4_evidence[
                    "min_paper_checkpoint"
                ],
                "e4_refinement_required_iterations": e4_evidence[
                    "required_iterations"
                ],
                "e4_refinement_evidence_status": e4_evidence[
                    "evidence_status"
                ],
                "undefined_outers": undefined,
            })

        seeds = sorted(seen_seed)
        if len(seeds) < int(min_seeds):
            raise ValueError(f"found {len(seeds)} seeds, fewer than --min-seeds={min_seeds}: {seeds}")
        if expected_seeds and seeds != sorted(set(int(value) for value in expected_seeds)):
            raise ValueError(f"seed set mismatch: found {seeds}, expected {sorted(set(expected_seeds))}")
        for field in (
            "group", "protocol_hash", "market_sha256", "policy_extension",
            "refinement_rule", "refinement_scope",
            "boundary_sensitivity_role", "map_definition", "evaluation_window",
            "domain_design",
            "exact_schedule", "e4_schedule", "paper_e4_schedule",
            "min_paper_checkpoint",
        ):
            serialized = {json.dumps(record[field], sort_keys=True) for record in records}
            if len(serialized) != 1:
                raise ValueError(f"cross-seed {field} mismatch: {serialized}")

        # Undefined values remain visible in per-seed output when explicitly
        # permitted.  A paper summary cannot form a common-sample statistic,
        # so it is marked unavailable rather than computed on fewer seeds.
        has_undefined = any(record["undefined_outers"] for record in records)
        all_exact_sensitivity_pass = not any(
            row.get("refinement_status") != "pass"
            for row in exact_all
        )
        all_required_e4_evidence_pass = all(
            record["e4_refinement_evidence_status"] == "pass"
            for record in records
        )
        has_partial_sensitivity = not (
            all_exact_sensitivity_pass
            and all_required_e4_evidence_pass
        )
        exact_summary = [] if has_undefined else _summary_rows(
            exact_all, index_field="source_outer_iter", metrics=EXACT_METRICS,
            source=Path("combined exact-map rows"),
        )
        e4_summary = _summary_rows(
            e4_all, index_field="target_outer_iter", metrics=E4_METRICS,
            source=Path("combined E4 rows"),
        )
        ratio_rows = _ratio_comparison_rows(
            exact_all, source=Path("combined exact-map rows"),
        )
        for row in ratio_rows:
            row["target_label"] = str(target_label)
        ratio_summary = _ratio_comparison_summary_rows(
            ratio_rows, source=Path("combined ratio-comparison rows"),
        )
        exact_ratio_plot_summary = _exact_ratio_plot_summary_rows(
            exact_all, source=Path("combined exact-map rows"),
        )
        ratio_floor_evidence = _annotate_ratio_floor_support(
            exact_all,
            ratio_summary,
            exact_ratio_plot_summary,
            floor_multiple=floor_multiple,
            floor_value=floor_value,
            source=Path("combined exact-map rows"),
        )
        ratio_worst_per_seed = _ratio_comparison_worst_per_seed(
            ratio_rows, source=Path("combined ratio-comparison rows"),
        )
        ratio_worst_summary = _ratio_comparison_worst_summary_rows(
            ratio_worst_per_seed
        )
        worst_per_seed = _worst_case_rows(
            exact_all, source=Path("combined exact-map rows"),
        )
        worst_summary = _worst_summary_rows(worst_per_seed)
        exact_fields = list(exact_all[0].keys())
        e4_fields = list(e4_all[0].keys())
        summary_fields = [
            "source_outer_iter", "metric", "n", "mean", "std", "sem",
            "ci95_low", "ci95_high", "min", "max", "seeds",
        ]
        e4_summary_fields = ["target_outer_iter", *summary_fields[1:]]
        ratio_per_seed_fields = [
            "target_label", "seed",
            "source_outer_iter", "target_outer_iter",
            "source_checkpoint_sha256", "target_checkpoint_sha256",
            "e_input_X_source", "e_input_X_target", "e_map_X_source",
            "rho_empirical_X", "rho_exact",
            "rho_empirical_minus_exact", "rho_sensitivity_envelope",
            "empirical_ratio_defined", "exact_ratio_defined",
            "paired_ratio_defined", "denominator_defined",
            "refinement_status",
            "refinement_qualified", "local_map_unmodified_on_xfd",
            "boundary_sensitivity_status", "group", "protocol_hash",
            "market_sha256", "model_type", "n_assets", "m_states",
            "policy_extension", "eval_margin", "eval_x_margin",
            "ev_w_min", "ev_w_max", "ev_x_min", "ev_x_max",
            "result_dir",
        ]
        ratio_summary_fields = [
            "ratio_iter", "source_outer_iter", "target_outer_iter",
            "floor_multiple", "common_regular",
            "n_expected", "n_paired_defined",
            "n_refinement_pass", "n_refinement_fail",
            "n_refinement_other", "n_locally_unmodified",
            "summary_status",
            *[
                f"{metric}_{field}"
                for metric in RATIO_METRICS
                for field in RATIO_STAT_FIELDS
            ],
            *[
                f"rho_sensitivity_envelope_{field}"
                for field in RATIO_STAT_FIELDS
            ],
            "seeds",
        ]
        ratio_worst_per_seed_fields = [
            "seed",
            "max_rho_empirical_X", "max_rho_empirical_X_outer",
            "max_rho_exact", "max_rho_exact_outer",
            "max_rho_sensitivity_envelope",
            "max_rho_sensitivity_envelope_outer",
            "all_paired_ratios_defined",
            "all_source_refinement_pass",
            "all_locally_unmodified", "n_pairs", "result_dir",
        ]
        ratio_worst_summary_fields = [
            "metric", "n_expected", "n_complete", "summary_status",
            "n", "mean", "std", "sem", "ci95_low", "ci95_high",
            "min", "max", "global_max", "global_max_seed",
            "global_max_outer", "seeds",
        ]
        worst_per_seed_fields = [
            "seed", "max_rho_exact", "max_rho_exact_outer",
            "max_rho_sensitivity_envelope",
            "max_rho_sensitivity_envelope_outer",
            "all_denominators_defined", "all_exact_sensitivity_pass",
            "all_locally_unmodified",
            "finite_domain_all_tested_ratios_below_one",
            "n_outer", "result_dir",
        ]
        worst_summary_fields = [
            "metric", "n_expected", "n_complete", "summary_status",
            "n", "mean", "std", "sem", "ci95_low", "ci95_high",
            "min", "max", "global_max", "global_max_seed",
            "global_max_outer", "seeds",
        ]
        map_variants = sorted({str(row.get("map_variant", "")) for row in exact_all})
        if has_undefined:
            paper_status = "undefined_denominator_no_exact_summary"
        elif has_partial_sensitivity:
            paper_status = "exploratory_partial_sensitivity"
        else:
            paper_status = "complete"
        all_seed_outer_locally_unmodified = all(
            _integer(row, "local_map_unmodified_on_xfd", Path("combined exact-map rows")) == 1
            for row in exact_all
        )
        exact_worst = next(
            row for row in worst_summary if row["metric"] == "max_rho_exact"
        )
        envelope_worst = next(
            row for row in worst_summary
            if row["metric"] == "max_rho_sensitivity_envelope"
        )
        all_denominators_defined = not has_undefined
        envelope_complete = (
            envelope_worst["summary_status"] == "complete_common_seed_sample"
        )
        exact_complete = (
            exact_worst["summary_status"] == "complete_common_seed_sample"
        )
        global_exact = (
            float(exact_worst["global_max"]) if exact_complete else None
        )
        global_envelope = (
            float(envelope_worst["global_max"]) if envelope_complete else None
        )
        finite_domain_below_one = bool(
            all_denominators_defined
            and all_exact_sensitivity_pass
            and all_seed_outer_locally_unmodified
            and global_exact is not None
            and global_exact < 1.0
            and global_envelope is not None
            and global_envelope < 1.0
        )
        blockers: List[str] = []
        if not all_denominators_defined:
            blockers.append("undefined_denominator")
        if not all_exact_sensitivity_pass:
            blockers.append("exact_refinement_not_passed")
        if not all_seed_outer_locally_unmodified:
            blockers.append("guard_or_clip_modified_map_on_sampled_fd_domain")
        if not exact_complete:
            blockers.append("incomplete_exact_ratio_common_seed_sample")
        elif global_exact is not None and global_exact >= 1.0:
            blockers.append("global_exact_ratio_not_below_one")
        if not envelope_complete:
            blockers.append("incomplete_sensitivity_envelope_common_seed_sample")
        elif global_envelope is not None and global_envelope >= 1.0:
            blockers.append("global_sensitivity_envelope_not_below_one")
        claim_text = (
            "All tested primary finite-domain boundary-projected extension-map "
            "ratios and their sampled grid/wealth-domain/factor-domain "
            "sensitivity envelopes remained below one."
            if finite_domain_below_one else None
        )
        ratio_pair_complete = bool(ratio_rows) and all(
            _integer(
                row,
                "paired_ratio_defined",
                Path("combined ratio-comparison rows"),
            ) == 1
            for row in ratio_rows
        )
        ratio_source_exact_refinement_pass = bool(ratio_rows) and all(
            str(row.get("refinement_status", "")) == "pass"
            for row in ratio_rows
        )
        ratio_source_locally_unmodified = bool(ratio_rows) and all(
            _integer(
                row,
                "local_map_unmodified_on_xfd",
                Path("combined ratio-comparison rows"),
            ) == 1
            for row in ratio_rows
        )
        ratio_refinement_qualified = (
            ratio_source_exact_refinement_pass
            and ratio_source_locally_unmodified
        )
        if not ratio_rows:
            ratio_comparison_status = "unavailable_no_adjacent_checkpoint_pair"
        elif not ratio_pair_complete:
            ratio_comparison_status = (
                "unavailable_incomplete_common_seed_sample"
            )
        elif ratio_refinement_qualified:
            ratio_comparison_status = "complete_refinement_qualified"
        else:
            ratio_comparison_status = "exploratory_partial_sensitivity"
        ratio_support = sorted({
            _integer(
                row,
                "source_outer_iter",
                Path("combined ratio-comparison rows"),
            )
            for row in ratio_rows
        })
        ratio_worst_by_metric = {
            str(row["metric"]): {
                "global_max": row["global_max"],
                "seed": row["global_max_seed"],
                "outer": row["global_max_outer"],
                "summary_status": row["summary_status"],
            }
            for row in ratio_worst_summary
        }
        ratio_metadata: Dict[str, Any] = {
            "status": ratio_comparison_status,
            "definition": {
                "empirical": (
                    "rho_empirical_X(s,k) = "
                    "e_input_X(s,k+1) / e_input_X(s,k)"
                ),
                "fd_map": "rho_exact(s,k) from exact_map_ratios.csv",
                "gap": "rho_empirical_X(s,k) - rho_exact(s,k)",
            },
            "indexing": (
                "source_outer_iter=k; the learned adjacent pair is k -> k+1; "
                "the final exact-map checkpoint is exact-only"
            ),
            "norm_and_window": (
                "both ratios use e_input_X/e_map_X from the same validated "
                "X_ev norm and evaluation window"
            ),
            "statistics": (
                "ratios are formed within seed first, then summarized by "
                "arithmetic mean, sample SD, SEM, and Student-t 95% CI"
            ),
            "floor_classification": ratio_floor_evidence,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "target_label": str(target_label),
            "source_outer_support": ratio_support,
            "n_source_outers": len(ratio_support),
            "n_rows": len(ratio_rows),
            "all_paired_ratios_defined": ratio_pair_complete,
            "all_source_exact_refinement_pass": (
                ratio_source_exact_refinement_pass
            ),
            "all_source_locally_unmodified": (
                ratio_source_locally_unmodified
            ),
            "comparison_refinement_qualified": (
                ratio_refinement_qualified
            ),
            "all_required_e4_refinement_evidence_pass": (
                all_required_e4_evidence_pass
            ),
            "e4_refinement_rule": records[0]["refinement_rule"],
            "e4_required_iterations_by_seed": {
                str(record["seed"]): record[
                    "e4_refinement_required_iterations"
                ]
                for record in records
            },
            "refinement_scope": records[0]["refinement_scope"],
            "boundary_sensitivity_role": records[0][
                "boundary_sensitivity_role"
            ],
            "finite_domain_scope": (
                "boundary-projected finite-domain FD map; no whole-space claim"
            ),
            "figure2_distinction": (
                "this adjacent X_ev ratio diagnostic is not the relative-L2 "
                "Value/Policy convergence trajectory used by main Figure 2"
            ),
            "worst_case": ratio_worst_by_metric,
            "plot_requested": bool(plot_ratios),
            "plot_files": [],
            "plot": {
                "formats": resolved_plot_formats,
                "fig_width": float(fig_width),
                "fig_height": float(fig_height),
                "font_size": float(font_size),
                "font_family": str(font_family),
                "line_width": float(line_width),
                "band_alpha": float(band_alpha),
                "floor_alpha": float(floor_alpha),
                "marker_size": float(marker_size),
                "grid_alpha": float(grid_alpha),
                "dpi": int(dpi),
                "ratio_y_scale": ratio_y_scale,
                "ratio_series": ratio_series,
                "floor_multiple": float(floor_multiple),
                "floor_value": (
                    None if floor_value is None else float(floor_value)
                ),
                "plot_sensitivity_envelope": bool(
                    plot_sensitivity_envelope
                ),
            },
        }
        payload = {
            "status": "success", "paper_summary_status": paper_status,
            "n_seeds": len(seeds), "seeds": seeds,
            "group": records[0]["group"], "protocol_hash": records[0]["protocol_hash"],
            "market_sha256": records[0]["market_sha256"],
            "policy_extension": records[0]["policy_extension"],
            "refinement_rule": records[0]["refinement_rule"],
            "refinement_scope": records[0]["refinement_scope"],
            "boundary_sensitivity_role": records[0][
                "boundary_sensitivity_role"
            ],
            "map_definition": records[0]["map_definition"],
            "evaluation_window": records[0]["evaluation_window"],
            "domain_design": records[0]["domain_design"],
            "checkpoint_schedule": records[0]["exact_schedule"],
            "min_paper_checkpoint": records[0]["min_paper_checkpoint"],
            "paper_e4_checkpoint_schedule": records[0][
                "paper_e4_schedule"
            ],
            "e4_refinement_required_iterations": {
                str(record["seed"]): record[
                    "e4_refinement_required_iterations"
                ]
                for record in records
            },
            "all_exact_refinement_pass": all_exact_sensitivity_pass,
            "all_required_e4_refinement_evidence_pass": (
                all_required_e4_evidence_pass
            ),
            "result_dirs": [record["directory"] for record in records],
            "undefined_denominators": {
                str(record["seed"]): record["undefined_outers"]
                for record in records if record["undefined_outers"]
            },
            "map_variants": map_variants,
            # Keep the original key for consumers while making its
            # seed-by-outer scope explicit in the new key.
            "all_locally_unmodified": all_seed_outer_locally_unmodified,
            "all_seed_outer_locally_unmodified": all_seed_outer_locally_unmodified,
            "worst_case": {
                "max_rho_exact": {
                    "global_max": exact_worst["global_max"],
                    "seed": exact_worst["global_max_seed"],
                    "outer": exact_worst["global_max_outer"],
                    "summary_status": exact_worst["summary_status"],
                },
                "max_rho_sensitivity_envelope": {
                    "global_max": envelope_worst["global_max"],
                    "seed": envelope_worst["global_max_seed"],
                    "outer": envelope_worst["global_max_outer"],
                    "summary_status": envelope_worst["summary_status"],
                },
            },
            "finite_domain_all_tested_ratios_below_one": finite_domain_below_one,
            "finite_domain_ratio_claim_status": (
                "supported_on_tested_sampled_fd_audit"
                if finite_domain_below_one else "not_supported"
            ),
            "finite_domain_ratio_claim_blockers": blockers,
            "finite_domain_ratio_claim_text": claim_text,
            "ratio_comparison": ratio_metadata,
            "interpretation": (
                "finite-domain boundary-projected extension-map audit; "
                "no whole-space exact-map or contraction claim"
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".liu-exact-aggregate-stage-", dir=str(output.parent)
        ) as stage_text:
            stage = Path(stage_text)
            write_csv(stage / "exact_map_per_seed.csv", exact_all, exact_fields)
            write_csv(stage / "exact_map_summary.csv", exact_summary, summary_fields)
            write_csv(
                stage / "exact_map_worst_per_seed.csv",
                worst_per_seed, worst_per_seed_fields,
            )
            write_csv(
                stage / "exact_map_worst_summary.csv",
                worst_summary, worst_summary_fields,
            )
            write_csv(stage / "e4_per_seed.csv", e4_all, e4_fields)
            write_csv(stage / "e4_summary.csv", e4_summary, e4_summary_fields)
            write_csv(
                stage / "ratio_comparison_per_seed.csv",
                ratio_rows, ratio_per_seed_fields,
            )
            write_csv(
                stage / "ratio_comparison_summary.csv",
                ratio_summary, ratio_summary_fields,
            )
            write_csv(
                stage / "ratio_comparison_worst_per_seed.csv",
                ratio_worst_per_seed, ratio_worst_per_seed_fields,
            )
            write_csv(
                stage / "ratio_comparison_worst_summary.csv",
                ratio_worst_summary, ratio_worst_summary_fields,
            )
            if plot_ratios:
                ratio_metadata["plot_files"] = _plot_ratio_comparison(
                    ratio_summary,
                    exact_ratio_plot_summary,
                    stage,
                    formats=resolved_plot_formats,
                    fig_width=fig_width,
                    fig_height=fig_height,
                    font_size=font_size,
                    font_family=font_family,
                    line_width=line_width,
                    band_alpha=band_alpha,
                    floor_alpha=floor_alpha,
                    marker_size=marker_size,
                    grid_alpha=grid_alpha,
                    dpi=dpi,
                    ratio_y_scale=ratio_y_scale,
                    plot_sensitivity_envelope=plot_sensitivity_envelope,
                    ratio_series=ratio_series,
                )
            atomic_json(
                stage / "ratio_comparison_metadata.json",
                ratio_metadata,
            )
            atomic_json(stage / "exact_map_aggregate_status.json", payload)
            (stage / "_SUCCESS_EXACT_MAP_AGG").touch()
            _commit_staged_output(stage, output)
        return payload
    except Exception as exc:
        if not had_managed_output:
            _prepare_output(output, overwrite=True)
            atomic_json(output / "exact_map_aggregate_status.json", {
                "status": "failed", "error": repr(exc),
                "result_dirs": [str(path) for path in result_dirs],
            })
            (output / "_FAILED_EXACT_MAP_AGG").touch()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Liu M=1 exact-map / E4 FD results")
    parser.add_argument("--out-root", type=Path, action="append", default=[])
    parser.add_argument("--result-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--allow-undefined-denominators", action="store_true")
    parser.add_argument("--allow-partial-sensitivity", action="store_true")
    parser.add_argument("--require-locally-unmodified", action="store_true")
    parser.add_argument(
        "--plot-ratios",
        action="store_true",
        help=(
            "plot the matched learned-step empirical and finite-domain FD "
            "exact-map ratios"
        ),
    )
    parser.add_argument(
        "--plot-sensitivity-envelope",
        action="store_true",
        help=(
            "add the seedwise maximum FD numerical-sensitivity envelope as "
            "a separate line"
        ),
    )
    parser.add_argument(
        "--ratio-series",
        choices=("empirical", "exact", "both"),
        default="both",
        help=(
            "series shown by --plot-ratios; empirical here uses the matched "
            "custom-X_ev exact-audit errors"
        ),
    )
    parser.add_argument(
        "--target-label",
        default="",
        help="Optional nominal p_res label recorded in ratio outputs.",
    )
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--fig-width", type=float, default=6.5)
    parser.add_argument("--fig-height", type=float, default=4.2)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--font-family", default="")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--band-alpha", type=float, default=0.18)
    parser.add_argument("--floor-alpha", type=float, default=0.80)
    parser.add_argument("--marker-size", type=float, default=4.0)
    parser.add_argument("--grid-alpha", type=float, default=0.22)
    parser.add_argument(
        "--floor-multiple",
        type=float,
        default=0.0,
        help=(
            "Display-only Merton floor multiplier. The paper default 0 keeps "
            "all finite ratios; a positive value is exploratory."
        ),
    )
    parser.add_argument(
        "--floor-value",
        type=float,
        default=None,
        help=(
            "Optional absolute base floor shared across seeds; otherwise use "
            "the per-seed median of the last ceil(10%%) e_input_X values."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--ratio-y-scale", choices=("linear", "log"), default="log",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out_root and not args.result_dir:
        raise ValueError("provide at least one --out-root or --result-dir")
    if args.min_seeds < 1:
        raise ValueError("--min-seeds must be positive")
    result_dirs = discover_result_dirs(args.out_root, args.result_dir)
    aggregate(
        result_dirs, args.output.expanduser().resolve(),
        expected_seeds=parse_seed_spec(args.expected_seeds), min_seeds=args.min_seeds,
        allow_undefined_denominators=args.allow_undefined_denominators,
        allow_partial_sensitivity=args.allow_partial_sensitivity,
        require_locally_unmodified=args.require_locally_unmodified,
        overwrite=args.overwrite,
        plot_ratios=args.plot_ratios,
        plot_formats=_parse_plot_formats(args.formats),
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        font_size=args.font_size,
        font_family=args.font_family,
        line_width=args.line_width,
        band_alpha=args.band_alpha,
        floor_alpha=args.floor_alpha,
        marker_size=args.marker_size,
        grid_alpha=args.grid_alpha,
        dpi=args.dpi,
        ratio_y_scale=args.ratio_y_scale,
        plot_sensitivity_envelope=args.plot_sensitivity_envelope,
        ratio_series=args.ratio_series,
        floor_multiple=args.floor_multiple,
        floor_value=args.floor_value,
        target_label=args.target_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
