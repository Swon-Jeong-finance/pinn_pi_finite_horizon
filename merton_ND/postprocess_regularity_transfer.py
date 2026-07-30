#!/usr/bin/env python3
"""Aggregate the Merton E4 residual-to-approximation diagnostic.

For each residual-sweep run this script reads ``exact_map_defects.csv`` plus
its independent defect-level FD refinement table and forms

    p_hat_X = max_n ||v_tilde_n - v^{alpha_n}||_Xev.

For a standard PI-PINN run the maximum covers ``delta_0,...,delta_{K-1}``.
For an E6 common-warm-start target branch it covers the target-dependent
evaluations ``delta_1,...,delta_K``; the shared warm-up ``delta_0`` is
intentionally outside each branch.  Paper aggregation requires passing
grid/domain evidence for the first and last available defects and whichever
primary defect attains the run-wise maximum (and for ``delta_0`` whenever it
belongs to the run).  Replacing the primary homogeneous CRRA Robin closure
with optimal-reference Dirichlet values changes the finite-domain BVP, so its
trajectory, slope, and empirical upper envelope are reported separately as a
boundary-sensitivity audit rather than used as a refinement gate.

The x-axis is the maximum *official post-restore* held-out residual over the
same run.  Pre-restore target crossings are never accepted as the achieved
residual.  The script writes all per-run values, per-target seed summaries,
log-log fits, and the empirical through-origin upper envelope
``C_num = max(p_hat_X / p_res)``.
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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from aggregate_e6 import cluster_robust_slope_se, e6_group_key, ols_loglog
from aggregate_seeds import (
    canonical_market_hash,
    load_config_args_raw,
    parse_seed_spec,
    t_crit_95,
)
from merton_exact_map_core import log_to_wealth_derivatives


FORMATS = {"png", "pdf", "svg", "eps"}
EVALUATED_BUNDLE_FIELDS = ("value", "vy", "vyy", "vw", "vww")
EVALUATED_BUNDLE_COORDINATE = "both_log_y_and_wealth_w"
PRIMARY_BOUNDARY = "robin"
BOUNDARY_LABELS = {
    "robin": "homogeneous CRRA Robin (primary)",
    "exact-dirichlet": "optimal-reference Dirichlet audit",
}
GRID_DOMAIN_ROW_STATUS_FIELDS = (
    "grid_domain_refinement_status",
    "defect_grid_domain_refinement_status",
)
GRID_DOMAIN_EVIDENCE_STATUS_FIELDS = (
    "defect_grid_domain_refinement_evidence_status",
    "grid_domain_refinement_evidence_status",
    "defect_refinement_grid_domain_evidence_status",
)


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires nonempty finite values")
    mean = float(array.mean())
    if array.size == 1:
        return mean, 0.0, 0.0, float("nan"), float("nan")
    std = float(array.std(ddof=1))
    sem = std / math.sqrt(int(array.size))
    half = t_crit_95(int(array.size) - 1) * sem
    return mean, std, sem, mean - half, mean + half


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluated_bundle(path: Path, *, defect_iter: int) -> List[str]:
    """Validate the lossless E4 evaluated-bundle coordinate contract.

    Legacy archives that label log derivatives as wealth derivatives are
    intentionally rejected.  Every relevant bundle must contain both
    coordinate representations, and the stored wealth derivatives must agree
    with a fresh conversion from the stored ``y``, ``vy``, and ``vyy``.
    """
    expected_prefixes = {"fd_map", "next_neural", "optimal"}
    if int(defect_iter) > 0:
        expected_prefixes.add("input")

    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{path}: cannot open evaluated bundle with allow_pickle=False"
        ) from exc

    with archive_context as archive:
        names = set(archive.files)
        if "derivative_coordinate" not in names:
            raise ValueError(
                f"{path}: legacy/ambiguous evaluated bundle has no "
                "derivative_coordinate"
            )
        try:
            coordinate_array = np.asarray(archive["derivative_coordinate"])
        except ValueError as exc:
            raise ValueError(
                f"{path}: derivative_coordinate must be a pickle-free scalar "
                "string"
            ) from exc
        if (
            coordinate_array.size != 1
            or coordinate_array.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                f"{path}: derivative_coordinate must be a pickle-free scalar "
                "string"
            )
        coordinate_value = coordinate_array.reshape(-1)[0]
        if isinstance(coordinate_value, bytes):
            coordinate_value = coordinate_value.decode("utf-8")
        if str(coordinate_value) != EVALUATED_BUNDLE_COORDINATE:
            raise ValueError(
                f"{path}: derivative_coordinate={coordinate_value!r}, "
                f"expected {EVALUATED_BUNDLE_COORDINATE!r}; legacy ambiguous "
                "archives are not accepted"
            )

        if "y" not in names:
            raise ValueError(f"{path}: evaluated bundle is missing y")
        try:
            y_raw = np.asarray(archive["y"])
        except ValueError as exc:
            raise ValueError(
                f"{path}: y must be a pickle-free numerical array"
            ) from exc
        if (
            y_raw.ndim != 1
            or y_raw.size == 0
            or not np.issubdtype(y_raw.dtype, np.number)
            or np.issubdtype(y_raw.dtype, np.complexfloating)
        ):
            raise ValueError(f"{path}: y must be a nonempty real 1-D array")
        y = np.asarray(y_raw, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{path}: y contains non-finite values")

        observed_prefixes = set(expected_prefixes)
        for name in names:
            for field in EVALUATED_BUNDLE_FIELDS:
                suffix = f"_{field}"
                if name.endswith(suffix) and len(name) > len(suffix):
                    observed_prefixes.add(name[: -len(suffix)])

        for prefix in sorted(observed_prefixes):
            required = {
                f"{prefix}_{field}" for field in EVALUATED_BUNDLE_FIELDS
            }
            missing = sorted(required - names)
            if missing:
                raise ValueError(
                    f"{path}: evaluated bundle prefix {prefix!r} is missing "
                    f"required arrays {missing}"
                )
            arrays: Dict[str, np.ndarray] = {}
            for field in EVALUATED_BUNDLE_FIELDS:
                name = f"{prefix}_{field}"
                try:
                    raw = np.asarray(archive[name])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: {name} must be a pickle-free numerical array"
                    ) from exc
                if (
                    raw.size == 0
                    or not np.issubdtype(raw.dtype, np.number)
                    or np.issubdtype(raw.dtype, np.complexfloating)
                ):
                    raise ValueError(
                        f"{path}: {name} must be a nonempty real numerical array"
                    )
                value = np.asarray(raw, dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{path}: {name} contains non-finite values")
                arrays[field] = value

            shapes = {field: value.shape for field, value in arrays.items()}
            if len(set(shapes.values())) != 1:
                raise ValueError(
                    f"{path}: evaluated bundle prefix {prefix!r} has "
                    f"inconsistent shapes {shapes}"
                )
            shape = arrays["value"].shape
            if not shape or shape[-1] != y.size:
                raise ValueError(
                    f"{path}: evaluated bundle prefix {prefix!r} shape "
                    f"{shape} is incompatible with y size {y.size}"
                )
            try:
                recomputed_vw, recomputed_vww = log_to_wealth_derivatives(
                    arrays["vy"], arrays["vyy"], y
                )
            except (FloatingPointError, ValueError) as exc:
                raise ValueError(
                    f"{path}: cannot reconstruct wealth derivatives for "
                    f"prefix {prefix!r}"
                ) from exc
            if not (
                np.all(np.isfinite(recomputed_vw))
                and np.all(np.isfinite(recomputed_vww))
            ):
                raise ValueError(
                    f"{path}: reconstructed wealth derivatives are non-finite "
                    f"for prefix {prefix!r}"
                )
            for field, recomputed in (
                ("vw", recomputed_vw),
                ("vww", recomputed_vww),
            ):
                if not np.allclose(
                    arrays[field],
                    recomputed,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    max_abs = float(
                        np.max(np.abs(arrays[field] - recomputed))
                    )
                    raise ValueError(
                        f"{path}: {prefix}_{field} is inconsistent with "
                        f"stored y/vy/vyy (max_abs_difference={max_abs:.3e})"
                    )
    return sorted(observed_prefixes)


def exact_protocol_group(training_group: str, exact_cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """Separate E4 regressions whenever the numerical FD protocol differs."""
    stored_compatibility = exact_cfg.get(
        "protocol_compatibility_payload"
    )
    if isinstance(stored_compatibility, Mapping):
        payload = dict(stored_compatibility)
        payload["training_group"] = str(training_group)
        # Warm-start artifacts are seed-specific inputs whose hashes remain in
        # warm_start_predecessor for integrity checks.  They are not numerical
        # FD/evaluation settings and therefore must never split an E4 group.
        payload.pop("warm_start_bundle_file_hash", None)
        payload.pop("warm_start_model_state_hash", None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        protocol = hashlib.sha256(encoded).hexdigest()[:16]
        return f"{training_group}-{protocol[:8]}", protocol

    grid = exact_cfg.get("grid")
    grid = grid if isinstance(grid, Mapping) else {}
    problem = exact_cfg.get("problem")
    problem = problem if isinstance(problem, Mapping) else {}
    margin = _float(exact_cfg.get("eval_margin"))
    ev_y_min = _float(grid.get("evaluation_y_min"))
    ev_y_max = _float(grid.get("evaluation_y_max"))
    if not (math.isfinite(ev_y_min) and math.isfinite(ev_y_max)):
        w_min = _float(problem.get("w_min"))
        w_max = _float(problem.get("w_max"))
        if w_min > 0.0 and w_max > w_min and math.isfinite(margin):
            train_y_min = math.log(w_min)
            train_y_max = math.log(w_max)
            width = train_y_max - train_y_min
            ev_y_min = train_y_min + margin * width
            ev_y_max = train_y_max - margin * width
    payload = {
        # Reconstruct a compatibility fingerprint from numerical settings for
        # legacy outputs.  Do not include the opaque producer protocol_hash:
        # older versions included seed-specific E6 warm-start artifact hashes
        # in that value even though their FD/evaluation protocols were equal.
        "primary_evaluation_window": {
            "eval_margin": margin,
            "ev_tau_min": _float(grid.get("evaluation_tau_min")),
            "ev_tau_max": _float(grid.get("evaluation_tau_max")),
            "ev_y_min": ev_y_min,
            "ev_y_max": ev_y_max,
        },
        "grid": grid,
        "norm": exact_cfg.get("norm"),
        "policy": exact_cfg.get("policy"),
        "network": exact_cfg.get("network"),
        "implementation_hashes": exact_cfg.get("implementation_hashes"),
        "refinement_abs_tolerance": exact_cfg.get("refinement_abs_tolerance"),
        "refinement_rel_tolerance": exact_cfg.get("refinement_rel_tolerance"),
        "denominator_tolerance": exact_cfg.get("denominator_tolerance"),
        "checkpoint_selection": exact_cfg.get("checkpoint_selection"),
        # Standard and common-warm-start E6 branches use different local
        # checkpoint numbering.  Keep them in distinct E4 protocol groups
        # even if every numerical FD option happens to match.
        "e6_role": exact_cfg.get("e6_role", "standard"),
        "checkpoint_indexing": exact_cfg.get("checkpoint_indexing"),
        "initial_defect_mode": exact_cfg.get("initial_defect_mode"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    protocol = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{training_group}-{protocol[:8]}", protocol


def e4_checkpoint_index_contract(
    exact_cfg: Mapping[str, Any],
    completed_outer: int,
) -> Dict[str, Any]:
    """Validate and normalize standard versus E6 target-branch indexing.

    Training checkpoints are always named by their *local* outer index
    ``1,...,K``.  For a standard run checkpoint ``j`` stores
    ``v_tilde_{j-1}``; for an E6 target branch it stores ``v_tilde_j`` because
    the common warm-up bundle already supplied ``v_tilde_0``.
    """
    completed_outer = int(completed_outer)
    if completed_outer < 1:
        raise ValueError("E4 checkpoint indexing requires at least one outer")

    role = str(exact_cfg.get("e6_role", "standard")).strip().lower()
    if role not in {"standard", "target_branch"}:
        raise ValueError(f"unsupported exact-map e6_role={role!r}")

    checkpoint_outers = list(range(1, completed_outer + 1))
    if role == "target_branch":
        source_offset = 0
        target_offset = 1
        initial_defect_mode = str(
            exact_cfg.get("initial_defect_mode", "")
        ).strip()
        if not initial_defect_mode:
            raise ValueError(
                "E6 target-branch exact-map config must record "
                "initial_defect_mode"
            )
    else:
        source_offset = -1
        target_offset = 0
        initial_defect_mode = str(
            exact_cfg.get(
                "initial_defect_mode",
                "standard_analytic_initial_policy_delta0",
            )
        ).strip()

    expected_source = {
        outer: outer + source_offset for outer in checkpoint_outers
    }
    expected_target = {
        outer: outer + target_offset for outer in checkpoint_outers
    }

    declared_schedule = [
        int(value)
        for value in exact_cfg.get("checkpoint_schedule_outer", [])
    ]
    if declared_schedule != checkpoint_outers:
        raise ValueError(
            "E4 requires the complete local checkpoint schedule "
            f"{checkpoint_outers}; found {declared_schedule}"
        )

    declared_sources = [
        int(value)
        for value in exact_cfg.get("checkpoint_schedule_source_n", [])
    ]
    expected_source_sequence = [
        expected_source[outer] for outer in checkpoint_outers
    ]
    if declared_sources != expected_source_sequence:
        raise ValueError(
            f"{role} checkpoint source schedule={declared_sources}, "
            f"expected={expected_source_sequence}"
        )

    indexing = exact_cfg.get("checkpoint_indexing")
    if indexing is None:
        if role == "target_branch":
            raise ValueError(
                "E6 target-branch exact-map config is missing "
                "checkpoint_indexing"
            )
        # Backward compatibility for standard exact-map outputs created
        # before the explicit mapping object was added.
        indexing = {}
    if not isinstance(indexing, Mapping):
        raise ValueError("checkpoint_indexing must be an object")

    if indexing:
        declared_source_offset = int(
            indexing.get(
                "source_iter_offset_from_checkpoint_outer",
                source_offset,
            )
        )
        declared_target_offset = int(
            indexing.get(
                "target_policy_iter_offset_from_checkpoint_outer",
                target_offset,
            )
        )
        if (
            declared_source_offset != source_offset
            or declared_target_offset != target_offset
        ):
            raise ValueError(
                f"{role} checkpoint offsets "
                f"({declared_source_offset},{declared_target_offset}) do not "
                f"match ({source_offset},{target_offset})"
            )

        def _normalized_mapping(name: str) -> Dict[int, int]:
            raw = indexing.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"checkpoint_indexing.{name} must be an object"
                )
            try:
                return {
                    int(outer): int(iteration)
                    for outer, iteration in raw.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"checkpoint_indexing.{name} contains a non-integer key "
                    "or value"
                ) from exc

        declared_source_map = _normalized_mapping(
            "checkpoint_outer_to_source_iter"
        )
        declared_target_map = _normalized_mapping(
            "checkpoint_outer_to_target_policy_iter"
        )
        if declared_source_map != expected_source:
            raise ValueError(
                f"{role} checkpoint source mapping={declared_source_map}, "
                f"expected={expected_source}"
            )
        if declared_target_map != expected_target:
            raise ValueError(
                f"{role} checkpoint target-policy mapping="
                f"{declared_target_map}, expected={expected_target}"
            )

    expected_defect_iters = [
        expected_source[outer] for outer in checkpoint_outers
    ]
    checkpoint_outer_by_defect = {
        expected_source[outer]: outer for outer in checkpoint_outers
    }
    return {
        "e6_role": role,
        "initial_defect_mode": initial_defect_mode,
        "checkpoint_outers": checkpoint_outers,
        "source_iter_by_checkpoint_outer": expected_source,
        "target_policy_iter_by_checkpoint_outer": expected_target,
        "expected_defect_iters": expected_defect_iters,
        "checkpoint_outer_by_defect": checkpoint_outer_by_defect,
    }


def required_refinement_iterations(
    defect_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Minimum paper evidence: first/last available defects and the worst.

    Standard runs include ``delta_0`` in that available set; E6 target
    branches begin at ``delta_1`` because their common warm-up predecessor is
    outside the target-phase residual sweep.
    """
    by_iter: Dict[int, float] = {}
    for row in defect_rows:
        defect_iter = int(float(row["defect_iter"]))
        delta = _float(row.get("delta_X"))
        if not delta >= 0.0:
            raise ValueError(
                f"invalid delta_X={row.get('delta_X')!r} at defect_iter={defect_iter}"
            )
        if defect_iter in by_iter:
            raise ValueError(f"duplicate E4 defect_iter={defect_iter}")
        by_iter[defect_iter] = delta
    if not by_iter:
        return []
    required = {0} if 0 in by_iter else set()
    adjacent = sorted(value for value in by_iter if value > 0)
    if adjacent:
        required.update((adjacent[0], adjacent[-1]))
    required.add(max(by_iter, key=lambda value: (by_iter[value], -value)))
    return sorted(required)


def _first_declared_status(
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> Tuple[str, str]:
    for field in fields:
        value = str(row.get(field, "")).strip().lower()
        if value:
            return value, field
    return "", ""


def _defect_refinement_protocol(
    exact_cfg: Mapping[str, Any],
    *,
    result_dir: Path,
) -> Tuple[List[int], List[float], List[str], Tuple[int, float, str]]:
    grid = exact_cfg.get("grid")
    grid = grid if isinstance(grid, Mapping) else {}
    factor_sequence = [int(value) for value in grid.get("grid_factors", [])]
    margin_sequence = [float(value) for value in grid.get("fd_margins", [])]
    boundary_sequence = [
        str(value).strip().lower().replace("_", "-")
        for value in grid.get("boundaries", [])
    ]
    if not factor_sequence or not margin_sequence or not boundary_sequence:
        raise ValueError(f"{result_dir}: incomplete exact-map FD grid protocol")
    if (
        len(set(factor_sequence)) != len(factor_sequence)
        or len(set(margin_sequence)) != len(margin_sequence)
        or len(set(boundary_sequence)) != len(boundary_sequence)
    ):
        raise ValueError(f"{result_dir}: duplicate exact-map FD variants")
    if boundary_sequence[0] != PRIMARY_BOUNDARY:
        raise ValueError(
            f"{result_dir}: paper primary boundary must be "
            f"{PRIMARY_BOUNDARY!r}, found {boundary_sequence[0]!r}"
        )
    primary = (
        max(factor_sequence),
        min(margin_sequence),
        boundary_sequence[0],
    )
    return factor_sequence, margin_sequence, boundary_sequence, primary


def _read_defect_refinement_rows(
    result_dir: Path,
) -> List[Dict[str, str]]:
    path = result_dir / "exact_map_defect_refinement.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: E4 requires defect-level FD refinement evidence"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_fields = {
        "protocol_hash", "defect_iter", "grid_factor", "fd_margin",
        "boundary", "is_primary", "is_verification", "delta_X",
        "refinement_status",
    }
    if not rows:
        raise ValueError(f"{path}: empty defect-refinement table")
    missing = required_fields - set(rows[0])
    if missing:
        raise ValueError(
            f"{path}: missing defect-refinement fields {sorted(missing)}"
        )
    return rows


def assess_grid_domain_refinement_evidence(
    result_dir: Path,
    exact_cfg: Mapping[str, Any],
    defect_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Reconstruct E4 grid/domain eligibility from raw FD variant values.

    Historical producers stored a single ``refinement_status`` that combined
    grid, domain, and boundary replacement.  Boundary replacement changes the
    finite-domain BVP and is now a separate sensitivity audit.  Consequently
    this routine recomputes the paper gate from ``delta_X`` using only:

    * grid comparisons at the largest domain and primary Robin boundary; and
    * domain comparisons at the finest grid and primary Robin boundary.

    A producer-specific grid/domain status, when present, is authenticated
    against the raw reconstruction and takes precedence over the legacy
    combined status.
    """
    refinement_rows = _read_defect_refinement_rows(result_dir)
    factors, margins, boundaries, primary_variant = (
        _defect_refinement_protocol(exact_cfg, result_dir=result_dir)
    )
    finest, largest_domain_margin, primary_boundary = primary_variant
    declared_protocol = str(exact_cfg.get("protocol_hash", ""))
    if not declared_protocol:
        raise ValueError(f"{result_dir}: missing exact protocol_hash")
    if any(
        str(row.get("protocol_hash", "")) != declared_protocol
        for row in refinement_rows
    ):
        raise ValueError(
            f"{result_dir}: defect refinement protocol hash mismatch"
        )

    abs_tolerance = _float(exact_cfg.get("refinement_abs_tolerance"))
    rel_tolerance = _float(exact_cfg.get("refinement_rel_tolerance"))
    if not (
        math.isfinite(abs_tolerance)
        and abs_tolerance >= 0.0
        and math.isfinite(rel_tolerance)
        and rel_tolerance >= 0.0
    ):
        raise ValueError(
            f"{result_dir}: invalid grid/domain refinement tolerances "
            f"abs={exact_cfg.get('refinement_abs_tolerance')!r}, "
            f"rel={exact_cfg.get('refinement_rel_tolerance')!r}"
        )

    required = required_refinement_iterations(defect_rows)
    source_by_iter = {
        int(float(row["defect_iter"])): row for row in defect_rows
    }
    assessments: List[Dict[str, Any]] = []
    for defect_iter in required:
        candidates = [
            row for row in refinement_rows
            if int(float(row["defect_iter"])) == defect_iter
        ]
        lookup: Dict[Tuple[int, float, str], Mapping[str, Any]] = {}
        for row in candidates:
            key = (
                int(float(row["grid_factor"])),
                float(row["fd_margin"]),
                str(row["boundary"]).strip().lower().replace("_", "-"),
            )
            if key in lookup:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} has duplicate "
                    f"FD variant {key}"
                )
            lookup[key] = row

        relevant_variants = {
            (int(factor), largest_domain_margin, primary_boundary)
            for factor in factors
        } | {
            (finest, float(margin), primary_boundary)
            for margin in margins
        }
        missing_variants = sorted(relevant_variants - set(lookup))
        if missing_variants:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} lacks grid/domain "
                f"FD variants {missing_variants}"
            )
        relevant_rows = [lookup[key] for key in sorted(relevant_variants)]
        if any(
            int(float(row.get("is_verification", 0))) != 1
            for row in relevant_rows
        ):
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} grid/domain "
                "evidence contains non-verification variants"
            )

        primary = lookup[primary_variant]
        primaries = [
            row for row in candidates
            if int(float(row.get("is_primary", 0))) == 1
        ]
        if len(primaries) != 1 or primaries[0] is not primary:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} requires exactly "
                f"one primary FD variant {primary_variant}"
            )
        primary_delta = _float(primary.get("delta_X"))
        if not primary_delta >= 0.0:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} has invalid "
                f"primary delta_X={primary.get('delta_X')!r}"
            )
        if any(not _float(row.get("delta_X")) >= 0.0 for row in relevant_rows):
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} contains invalid "
                "grid/domain delta_X values"
            )

        tolerance = abs_tolerance + rel_tolerance * abs(primary_delta)
        grid_changes = [
            abs(
                primary_delta
                - _float(
                    lookup[
                        (int(factor), largest_domain_margin, primary_boundary)
                    ].get("delta_X")
                )
            )
            for factor in factors
            if int(factor) != finest
        ]
        domain_changes = [
            abs(
                primary_delta
                - _float(
                    lookup[
                        (finest, float(margin), primary_boundary)
                    ].get("delta_X")
                )
            )
            for margin in margins
            if float(margin) != largest_domain_margin
        ]
        grid_change = max(grid_changes, default=0.0)
        domain_change = max(domain_changes, default=0.0)
        failed_axes = []
        if not grid_changes:
            failed_axes.append("grid_not_checked")
        elif grid_change > tolerance:
            failed_axes.append("grid")
        if not domain_changes:
            failed_axes.append("domain_not_checked")
        elif domain_change > tolerance:
            failed_axes.append("domain")
        if largest_domain_margin >= 0.0:
            reconstructed_status = "fd_domain_not_enlarged_beyond_training"
        elif not grid_changes or not domain_changes:
            reconstructed_status = "incomplete"
        else:
            reconstructed_status = "pass" if not failed_axes else "fail"
        declared_status, declared_field = _first_declared_status(
            primary, GRID_DOMAIN_ROW_STATUS_FIELDS
        )
        if declared_status:
            if declared_status != reconstructed_status:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} declared "
                    f"{declared_field}={declared_status!r} disagrees with raw "
                    f"grid/domain reconstruction={reconstructed_status!r}"
                )
            eligibility_status = declared_status
            status_source = f"producer:{declared_field}"
        else:
            eligibility_status = reconstructed_status
            status_source = "legacy_raw_delta_X_reconstruction"

        source = source_by_iter[defect_iter]
        if not math.isclose(
            primary_delta,
            _float(source.get("delta_X")),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"{result_dir}: primary defect/refinement delta_X mismatch at "
                f"defect_iter={defect_iter}"
            )
        assessment = {
            "defect_iter": defect_iter,
            "primary_delta_X": primary_delta,
            "refinement_tolerance": tolerance,
            "grid_abs_change_recomputed": grid_change,
            "domain_abs_change_recomputed": domain_change,
            "failed_axes": ";".join(failed_axes),
            "grid_domain_refinement_status": eligibility_status,
            "grid_domain_status_source": status_source,
            "legacy_combined_refinement_status": str(
                primary.get("refinement_status", "")
            ).strip(),
            "source_legacy_combined_refinement_status": str(
                source.get("refinement_status", "")
            ).strip(),
            "primary_boundary": primary_boundary,
        }
        assessments.append(assessment)
        if eligibility_status != "pass":
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} grid/domain-only "
                f"refinement failed ({assessment['failed_axes']}); "
                f"grid={grid_change:.6e}, domain={domain_change:.6e}, "
                f"tolerance={tolerance:.6e}"
            )
    return required, assessments


def validate_defect_refinement_evidence(
    result_dir: Path,
    exact_cfg: Mapping[str, Any],
    defect_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Verify grid/domain-only E4 evidence at required defects.

    Boundary replacement is intentionally excluded: it is a distinct
    finite-domain BVP sensitivity audit, not a refinement axis.
    """
    required, _assessments = assess_grid_domain_refinement_evidence(
        result_dir, exact_cfg, defect_rows
    )
    return required


def read_defects(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no adjacent-checkpoint E4 defects in {path}")
    required = {
        "protocol_hash", "eval_margin", "ev_y_min", "ev_y_max",
        "delta_X", "defect_iter", "target_policy_iter",
        "next_checkpoint_outer_iter", "residual_semantics",
        "evaluated_bundle_path", "evaluated_bundle_sha256",
        "is_verification", "refinement_status",
        "defect_grid_abs_change", "defect_domain_abs_change",
        "defect_boundary_abs_change", "defect_sensitivity_envelope",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing E4 fields {sorted(missing)}")
    return rows


def official_residual(run_dir: Path) -> Tuple[float, int]:
    """Return max_n post-restore p_res and the number of finite outer rows."""
    path = run_dir / "outer_history.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    values: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "val_pres_post_restore" not in set(reader.fieldnames or ()):
            raise ValueError(
                f"{path}: E4 requires val_pres_post_restore from the official model"
            )
        for row in reader:
            value = _float(row.get("val_pres_post_restore"))
            if value > 0.0:
                values.append(value)
    if not values:
        raise ValueError(f"{path}: no positive post-restore residuals")
    return max(values), len(values)


def discover_exact_results(root: Path) -> Iterable[Path]:
    root = root.expanduser().resolve()
    for marker in sorted(root.rglob("_SUCCESS_EXACT_MAP")):
        # Exact-map reruns quarantine their previous canonical artifacts below
        # ``stale_outputs/<timestamp>``.  Those markers are audit evidence, not
        # current results, and must never be rediscovered as independent E4
        # observations.
        if "stale_outputs" in marker.relative_to(root).parts:
            continue
        result = marker.parent
        if (result / "exact_map_defects.csv").is_file() and (
            result / "exact_map_config.json"
        ).is_file():
            yield result


def _target_slug(value: Any) -> str:
    target = _float(value)
    if not math.isfinite(target):
        return "unknown"
    text = f"{target:.12g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def _copy_exact_map_evidence(
    result_dir: Path,
    destination: Path,
    *,
    seed: Any,
    target: Any,
) -> List[Dict[str, Any]]:
    """Copy the compact top-level exact-map evidence and return a manifest.

    Evaluated bundle arrays remain in their original result directory because
    they can be large.  Their paths and hashes are already authenticated by
    ``exact_map_defects.csv``.  The copied ``exact_map_*`` CSV/JSON/figure
    files are sufficient for one-place protocol and refinement triage.
    """
    destination.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for source in sorted(result_dir.glob("exact_map_*")):
        if not source.is_file():
            continue
        copied = destination / source.name
        shutil.copy2(source, copied)
        manifest.append({
            "seed": seed,
            "pres_target": target,
            "result_dir": str(result_dir),
            "source_path": str(source),
            "copied_path": str(copied),
            "sha256": sha256_file(copied),
            "size_bytes": copied.stat().st_size,
        })
    return manifest


def _refinement_audit_rows(
    result_dir: Path,
    exact_cfg: Mapping[str, Any],
    *,
    seed: Any,
    target: Any,
) -> List[Dict[str, Any]]:
    """Return one grid/domain diagnostic row for every primary defect."""
    defects_path = result_dir / "exact_map_defects.csv"
    refinement_path = result_dir / "exact_map_defect_refinement.csv"
    if not defects_path.is_file() or not refinement_path.is_file():
        return []
    try:
        defects = read_defects(defects_path)
        required = set(required_refinement_iterations(defects))
        rows = _read_defect_refinement_rows(result_dir)
        factors, margins, _boundaries, primary_variant = (
            _defect_refinement_protocol(exact_cfg, result_dir=result_dir)
        )
    except (OSError, ValueError, KeyError):
        return []
    finest, largest_domain_margin, primary_boundary = primary_variant
    abs_tolerance = _float(exact_cfg.get("refinement_abs_tolerance"))
    rel_tolerance = _float(exact_cfg.get("refinement_rel_tolerance"))
    source_status_by_iter = {
        int(float(row["defect_iter"])): str(
            row.get("refinement_status", "")
        ).strip()
        for row in defects
    }
    by_iter: Dict[int, Dict[Tuple[int, float, str], Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    duplicate_variants: Dict[int, set[Tuple[int, float, str]]] = defaultdict(
        set
    )
    for row in rows:
        try:
            defect_iter = int(float(row["defect_iter"]))
            key = (
                int(float(row["grid_factor"])),
                float(row["fd_margin"]),
                str(row["boundary"]).strip().lower().replace("_", "-"),
            )
        except (TypeError, ValueError, KeyError):
            continue
        if key in by_iter[defect_iter]:
            duplicate_variants[defect_iter].add(key)
        by_iter[defect_iter][key] = row

    audit: List[Dict[str, Any]] = []
    for row in rows:
        try:
            is_primary = int(float(row.get("is_primary", 0))) == 1
            defect_iter = int(float(row["defect_iter"]))
        except (TypeError, ValueError, KeyError):
            continue
        if not is_primary:
            continue
        delta = _float(row.get("delta_X"))
        tolerance = (
            abs_tolerance + rel_tolerance * abs(delta)
            if (
                math.isfinite(abs_tolerance)
                and math.isfinite(rel_tolerance)
                and math.isfinite(delta)
            )
            else float("nan")
        )
        lookup = by_iter.get(defect_iter, {})
        relevant_variants = {
            (int(factor), largest_domain_margin, primary_boundary)
            for factor in factors
        } | {
            (finest, float(margin), primary_boundary)
            for margin in margins
        }
        missing_variants = sorted(relevant_variants - set(lookup))
        changes = {"grid": float("nan"), "domain": float("nan")}
        if not missing_variants and math.isfinite(delta):
            grid_values = [
                abs(
                    delta
                    - _float(
                        lookup[
                            (
                                int(factor),
                                largest_domain_margin,
                                primary_boundary,
                            )
                        ].get("delta_X")
                    )
                )
                for factor in factors
                if int(factor) != finest
            ]
            domain_values = [
                abs(
                    delta
                    - _float(
                        lookup[
                            (finest, float(margin), primary_boundary)
                        ].get("delta_X")
                    )
                )
                for margin in margins
                if float(margin) != largest_domain_margin
            ]
            changes["grid"] = max(grid_values, default=0.0)
            changes["domain"] = max(domain_values, default=0.0)
        failed_axes = [
            name for name, change in changes.items()
            if (
                math.isfinite(change)
                and math.isfinite(tolerance)
                and change > tolerance
            )
        ]
        nonfinite_axes = [
            name for name, change in changes.items()
            if not math.isfinite(change)
        ]
        legacy_status = str(row.get("refinement_status", "")).strip()
        source_status = source_status_by_iter.get(defect_iter, "")
        declared_status, declared_field = _first_declared_status(
            row, GRID_DOMAIN_ROW_STATUS_FIELDS
        )
        reconstructed_status = (
            "pass"
            if not missing_variants and not nonfinite_axes and not failed_axes
            else "fail"
        )
        if declared_status:
            grid_domain_status = declared_status
            status_source = f"producer:{declared_field}"
        elif int(float(row.get("is_verification", 0))) != 1:
            grid_domain_status = "not_checked"
            status_source = "sparse_verification"
        else:
            grid_domain_status = reconstructed_status
            status_source = "legacy_raw_delta_X_reconstruction"
        failure_code = "pass"
        failure_reason = ""
        if not (
            math.isfinite(delta)
            and math.isfinite(tolerance)
            and tolerance >= 0.0
        ):
            failure_code = "invalid_tolerance_or_defect"
            failure_reason = (
                f"delta_X={row.get('delta_X', '')!r}, "
                f"tolerance={tolerance!r}"
            )
        elif duplicate_variants.get(defect_iter):
            failure_code = "duplicate_grid_domain_variants"
            failure_reason = "duplicate variants: " + repr(
                sorted(duplicate_variants[defect_iter])
            )
        elif missing_variants:
            failure_code = (
                "sparse_verification_not_checked"
                if int(float(row.get("is_verification", 0))) != 1
                else "missing_grid_domain_variants"
            )
            failure_reason = f"missing grid/domain variants: {missing_variants}"
        elif nonfinite_axes:
            failure_code = "nonfinite_grid_domain_evidence"
            failure_reason = "non-finite grid/domain change(s): " + ",".join(
                nonfinite_axes
            )
        elif failed_axes:
            failure_code = "grid_domain_tolerance_exceeded"
            failure_reason = "; ".join(
                f"{name}={changes[name]:.6e} > "
                f"tolerance={tolerance:.6e}"
                for name in failed_axes
            )
        elif declared_status and declared_status != reconstructed_status:
            failure_code = "grid_domain_status_mismatch"
            failure_reason = (
                f"declared {declared_field}={declared_status!r}, raw "
                f"reconstruction={reconstructed_status!r}"
            )
        elif grid_domain_status != "pass":
            failure_code = "grid_domain_status_not_pass"
            failure_reason = (
                f"grid_domain_refinement_status={grid_domain_status!r}"
            )
        audit.append({
            "seed": seed,
            "pres_target": target,
            "result_dir": str(result_dir),
            "defect_iter": defect_iter,
            "required_for_paper": int(defect_iter in required),
            "grid_factor": row.get("grid_factor", ""),
            "fd_margin": row.get("fd_margin", ""),
            "boundary": row.get("boundary", ""),
            "delta_X": row.get("delta_X", ""),
            "source_refinement_status": source_status,
            "refinement_status": legacy_status,
            "legacy_combined_refinement_status": legacy_status,
            "source_legacy_combined_refinement_status": source_status,
            "grid_domain_refinement_status": grid_domain_status,
            "grid_domain_status_source": status_source,
            "refinement_tolerance": tolerance,
            "defect_grid_abs_change": changes["grid"],
            "defect_domain_abs_change": changes["domain"],
            "defect_boundary_abs_change": row.get(
                "defect_boundary_abs_change", ""
            ),
            "defect_sensitivity_envelope": row.get(
                "defect_sensitivity_envelope", ""
            ),
            "failed_axes": ";".join(failed_axes),
            "nonfinite_axes": ";".join(nonfinite_axes),
            "missing_grid_domain_variants": repr(missing_variants),
            "failure_code": failure_code,
            "failure_reason": failure_reason,
        })
    return audit


def collect_runs_with_audit(
    root: Path,
    *,
    n_assets: int | None,
    run_name_regex: str,
    evidence_root: Path,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Collect every selected result without aborting at the first failure.

    The strict single-result collector remains the source of truth.  Running
    it on each result independently lets this diagnostic wrapper retain every
    integrity check while reporting all failures in one pass.
    """
    pattern = re.compile(run_name_regex) if run_name_regex else None
    passing: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    refinement_audits: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    for result_dir in discover_exact_results(root):
        exact_cfg = load_json(result_dir / "exact_map_config.json")
        run_dir = Path(
            str(exact_cfg.get("run_dir", result_dir.parent))
        ).expanduser().resolve()
        if pattern and not pattern.search(str(run_dir)):
            continue
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None:
            cfg = {}
        problem = exact_cfg.get("problem")
        problem = problem if isinstance(problem, Mapping) else {}
        assets_raw = cfg.get("n_assets", problem.get("n_assets", 0))
        try:
            assets = int(assets_raw)
        except (TypeError, ValueError):
            assets = 0
        if n_assets is not None and assets != int(n_assets):
            continue
        seed = cfg.get("seed", "")
        target = cfg.get("pres_target", "")
        result_tag = hashlib.sha256(
            str(result_dir).encode("utf-8")
        ).hexdigest()[:10]
        evidence_dir = (
            evidence_root
            / f"N{assets if assets > 0 else 'unknown'}"
            / f"seed_{seed if seed != '' else 'unknown'}"
            / f"target_{_target_slug(target)}"
            / f"result_{result_tag}"
        )
        try:
            manifest.extend(_copy_exact_map_evidence(
                result_dir,
                evidence_dir,
                seed=seed,
                target=target,
            ))
        except OSError as exc:
            # Evidence-copy failure is itself auditable and must not hide the
            # underlying numerical validation result.
            manifest.append({
                "seed": seed,
                "pres_target": target,
                "result_dir": str(result_dir),
                "source_path": "",
                "copied_path": str(evidence_dir),
                "sha256": "",
                "size_bytes": "",
                "copy_error": f"{type(exc).__name__}: {exc}",
            })
        refinement_audits.extend(_refinement_audit_rows(
            result_dir,
            exact_cfg,
            seed=seed,
            target=target,
        ))
        audit = {
            "n_assets": assets,
            "seed": seed,
            "pres_target": target,
            "run_dir": str(run_dir),
            "result_dir": str(result_dir),
            "evidence_dir": str(evidence_dir),
            "source_protocol_hash": str(
                exact_cfg.get("protocol_hash", "")
            ),
            "collection_status": "failed",
            "paper_eligible": 0,
            "error_type": "",
            "error_message": "",
            "failed_required_defects": "",
            "failed_axes": "",
        }
        try:
            selected = collect_runs(
                result_dir,
                n_assets=n_assets,
                run_name_regex=run_name_regex,
            )
            if len(selected) != 1:
                raise ValueError(
                    f"{result_dir}: expected one selected exact-map run, "
                    f"found {len(selected)}"
                )
            passing.extend(selected)
            audit["collection_status"] = "pass"
            audit["paper_eligible"] = 1
        except Exception as exc:  # Continue to report every selected run.
            audit["error_type"] = type(exc).__name__
            audit["error_message"] = str(exc)
            relevant = [
                row for row in refinement_audits
                if str(row["result_dir"]) == str(result_dir)
                and int(row["required_for_paper"]) == 1
                and str(row["failure_code"]) != "pass"
            ]
            audit["failed_required_defects"] = ";".join(
                str(row["defect_iter"]) for row in relevant
            )
            audit["failed_axes"] = ";".join(sorted({
                axis
                for row in relevant
                for axis in str(row["failed_axes"]).split(";")
                if axis
            }))
        audits.append(audit)

    # Preserve the strict collector's newest-run semantics if duplicate
    # canonical results happen to be present.
    newest: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    for row in passing:
        key = (
            str(row["group"]),
            float(row["pres_target"]),
            int(row["seed"]),
        )
        if key not in newest or str(row["updated_at"]) >= str(
            newest[key]["updated_at"]
        ):
            newest[key] = row
    return list(newest.values()), audits, refinement_audits, manifest


def collect_runs(
    root: Path,
    *,
    n_assets: int | None,
    run_name_regex: str,
) -> List[Dict[str, Any]]:
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, float, int], Dict[str, Any]] = {}
    for result_dir in discover_exact_results(root):
        exact_cfg = load_json(result_dir / "exact_map_config.json")
        run_dir = Path(str(exact_cfg.get("run_dir", result_dir.parent))).expanduser().resolve()
        if pattern and not pattern.search(str(run_dir)):
            continue
        # E4 is tied to the training-time checkpoint trajectory and the
        # producer's recorded Q_ev window.  A later eval-only overlay must not
        # change its training group or relabel its provenance.
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None or str(cfg.get("model_type", "")) != "pipinn":
            continue
        training_e6_role = str(cfg.get("e6_role", "standard")).strip().lower()
        declared_e6_role = exact_cfg.get("e6_role")
        if declared_e6_role is None:
            if training_e6_role == "target_branch":
                raise ValueError(
                    f"{result_dir}: E6 target-branch result is missing "
                    "exact-map e6_role provenance"
                )
            declared_e6_role = "standard"
        exact_e6_role = str(declared_e6_role).strip().lower()
        if exact_e6_role != training_e6_role:
            raise ValueError(
                f"{result_dir}: exact-map e6_role={exact_e6_role!r} does not "
                f"match training e6_role={training_e6_role!r}"
            )
        target = _float(cfg.get("pres_target"))
        if not target > 0.0:
            continue
        assets = int(cfg.get("n_assets", 0))
        if n_assets is not None and assets != int(n_assets):
            continue
        seed = int(cfg.get("seed"))
        defect_rows = read_defects(result_dir / "exact_map_defects.csv")
        semantics = {str(row.get("residual_semantics", "")) for row in defect_rows}
        if semantics != {"official_post_restore"}:
            raise ValueError(
                f"{result_dir}: E4 requires official_post_restore defect metadata; "
                f"found {sorted(semantics)}"
            )
        defects = [_float(row.get("delta_X")) for row in defect_rows]
        if not defects or any(not value >= 0.0 for value in defects):
            raise ValueError(f"{result_dir}: nonfinite/negative delta_X")
        declared_protocol = str(exact_cfg.get("protocol_hash", ""))
        row_protocols = {
            str(row.get("protocol_hash", "")) for row in defect_rows
        }
        if not declared_protocol or row_protocols != {declared_protocol}:
            raise ValueError(
                f"{result_dir}: E4 defect/config protocol hash mismatch"
            )
        eval_windows = {
            (
                _float(row.get("eval_margin")),
                _float(row.get("ev_y_min")),
                _float(row.get("ev_y_max")),
            )
            for row in defect_rows
        }
        if len(eval_windows) != 1 or any(
            not all(math.isfinite(value) for value in window)
            for window in eval_windows
        ):
            raise ValueError(
                f"{result_dir}: E4 defects mix or omit primary eval windows"
            )
        eval_window = next(iter(eval_windows))
        (
            required_refinement,
            grid_domain_assessments,
        ) = assess_grid_domain_refinement_evidence(
            result_dir, exact_cfg, defect_rows
        )
        exact_status = load_json(result_dir / "exact_map_status.json")
        declared_grid_domain_evidence, declared_grid_domain_field = (
            _first_declared_status(
                exact_status, GRID_DOMAIN_EVIDENCE_STATUS_FIELDS
            )
        )
        if (
            declared_grid_domain_evidence
            and declared_grid_domain_evidence != "pass"
        ):
            raise ValueError(
                f"{result_dir}: exact-map status declares "
                f"{declared_grid_domain_field}="
                f"{declared_grid_domain_evidence!r}, expected 'pass'"
            )
        recorded_required = [
            int(value)
            for value in exact_status.get(
                "defect_refinement_required_iterations", []
            )
        ]
        if recorded_required != required_refinement:
            raise ValueError(
                f"{result_dir}: exact-map status refinement iterations "
                f"{recorded_required} do not match {required_refinement}"
            )
        status = load_json(run_dir / "status.json")
        completed_outer = int(
            status.get(
                "completed_outer_iters",
                status.get("final_outer_iter", 0),
            )
        )
        if completed_outer < 2:
            raise ValueError(
                f"{result_dir}: E4 requires at least two completed outers"
            )
        try:
            index_contract = e4_checkpoint_index_contract(
                exact_cfg, completed_outer
            )
        except ValueError as exc:
            raise ValueError(
                f"{result_dir}: invalid E4 checkpoint provenance: {exc}. "
                "Train with e3b_checkpoints=false and save_iterate_every=1."
            ) from exc
        if len(defect_rows) != completed_outer:
            raise ValueError(
                f"{result_dir}: expected {completed_outer} complete E4 "
                f"defects, found {len(defect_rows)}"
            )
        expected_defect_iters = list(
            index_contract["expected_defect_iters"]
        )
        observed_defect_iters = sorted(
            int(float(row["defect_iter"])) for row in defect_rows
        )
        if observed_defect_iters != expected_defect_iters:
            raise ValueError(
                f"{result_dir}: E4 defect iterations={observed_defect_iters}, "
                f"expected={expected_defect_iters}"
            )
        for defect in defect_rows:
            relative = Path(str(defect["evaluated_bundle_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"{result_dir}: unsafe evaluated bundle path {relative}"
                )
            bundle = result_dir / relative
            if not bundle.is_file():
                raise FileNotFoundError(bundle)
            declared = str(defect["evaluated_bundle_sha256"])
            if not declared or sha256_file(bundle) != declared:
                raise ValueError(f"{bundle}: evaluated bundle hash mismatch")
            validate_evaluated_bundle(
                bundle,
                defect_iter=int(float(defect["defect_iter"])),
            )
        p_res, n_outer = official_residual(run_dir)
        if n_outer != completed_outer:
            raise ValueError(
                f"{run_dir}: found {n_outer} official residual rows for "
                f"{completed_outer} completed outers"
            )
        residual_by_outer: Dict[int, float] = {}
        with (run_dir / "outer_history.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                outer = int(float(row["outer_iter"]))
                if outer in residual_by_outer:
                    raise ValueError(
                        f"{run_dir}: duplicate outer={outer} residual row"
                    )
                residual_by_outer[outer] = float(
                    row["val_pres_post_restore"]
                )
        if sorted(residual_by_outer) != list(range(1, completed_outer + 1)):
            raise ValueError(
                f"{run_dir}: post-restore residual outer coverage is "
                f"{sorted(residual_by_outer)}, expected 1..{completed_outer}"
            )
        for defect in defect_rows:
            defect_iter = int(float(defect["defect_iter"]))
            target_outer = int(float(defect["next_checkpoint_outer_iter"]))
            expected_target_outer = int(
                index_contract["checkpoint_outer_by_defect"][defect_iter]
            )
            if target_outer != expected_target_outer:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} is attached to "
                    f"outer={target_outer}, expected {expected_target_outer} "
                    f"for e6_role={exact_e6_role}"
                )
            target_policy_iter = int(float(defect["target_policy_iter"]))
            if target_policy_iter != defect_iter:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} records "
                    f"target_policy_iter={target_policy_iter}, expected "
                    f"{defect_iter}"
                )
            recorded = _float(defect.get("p_res_post_restore"))
            expected_residual = residual_by_outer[target_outer]
            if not math.isclose(
                recorded, expected_residual, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} p_res={recorded} "
                    f"does not match outer_history={expected_residual}"
                )
        training_group = e6_group_key(cfg)
        group, exact_protocol = exact_protocol_group(training_group, exact_cfg)
        updated = str(status.get("updated_at", ""))
        market_hash = canonical_market_hash(str(run_dir / "market_params.npz"))
        defect_iters = sorted(
            int(float(row["defect_iter"])) for row in defect_rows
        )
        defect_refinement = sorted({
            str(row.get("refinement_status", "")) for row in defect_rows
        })
        record = {
            "group": group,
            "training_group": training_group,
            "exact_protocol_hash": exact_protocol,
            "source_protocol_hash": declared_protocol,
            "run_dir": str(run_dir),
            "result_dir": str(result_dir),
            "n_assets": assets,
            "seed": seed,
            "market_seed": int(cfg.get("market_seed", -1)),
            "market_hash": market_hash,
            "pres_target": target,
            "achieved_pres_post_restore": p_res,
            "p_hat_X": max(defects),
            "C_num_run": max(defects) / p_res,
            "n_outer_residuals": n_outer,
            "n_adjacent_defects": len(defect_rows),
            "completed_outer_iters": completed_outer,
            "e6_role": exact_e6_role,
            "initial_defect_mode": index_contract["initial_defect_mode"],
            "checkpoint_index_semantics": (
                "local_outer_j_stores_v_j"
                if exact_e6_role == "target_branch"
                else "local_outer_j_stores_v_j_minus_1"
            ),
            "defect_coverage": (
                "complete_target_branch_delta1_through_deltaK"
                if exact_e6_role == "target_branch"
                else "complete_all_outer_iterations_including_delta0"
            ),
            "defect_refinement_statuses": ";".join(defect_refinement),
            "defect_refinement_required_iterations": ";".join(
                str(value) for value in required_refinement
            ),
            "legacy_combined_refinement_evidence_status": str(
                exact_status.get(
                    "defect_refinement_evidence_status", ""
                )
            ),
            "grid_domain_refinement_evidence_status": "pass",
            "grid_domain_status_source": (
                f"producer:{declared_grid_domain_field}"
                if declared_grid_domain_evidence
                else "legacy_raw_delta_X_reconstruction"
            ),
            "grid_domain_required_statuses": ";".join(
                str(row["grid_domain_refinement_status"])
                for row in grid_domain_assessments
            ),
            "eval_margin": eval_window[0],
            "ev_y_min": eval_window[1],
            "ev_y_max": eval_window[2],
            "defect_iter_min": min(defect_iters),
            "defect_iter_max": max(defect_iters),
            "residual_semantics": "official_post_restore",
            "updated_at": updated,
        }
        key = (group, target, seed)
        if key not in newest or updated >= str(newest[key]["updated_at"]):
            newest[key] = record
    return list(newest.values())


def validate_panel(
    rows: Sequence[Mapping[str, Any]], expected_seeds: set[int], min_seeds: int
) -> None:
    if not rows:
        raise ValueError("no successful E4 residual-sweep exact-map outputs found")
    panels: Dict[Tuple[str, float], set[int]] = defaultdict(set)
    markets: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        panels[(str(row["group"]), float(row["pres_target"]))].add(int(row["seed"]))
        markets[str(row["group"])].add(str(row["market_hash"]))
    for (group, target), seeds in sorted(panels.items()):
        if expected_seeds and seeds != expected_seeds:
            raise ValueError(
                f"group={group}, target={target:g}: seeds={sorted(seeds)}, "
                f"expected={sorted(expected_seeds)}"
            )
        if len(seeds) < min_seeds:
            raise ValueError(
                f"group={group}, target={target:g}: {len(seeds)} seeds < {min_seeds}"
            )
    bad_markets = {group: hashes for group, hashes in markets.items() if len(hashes) != 1}
    if bad_markets:
        raise ValueError(f"E4 groups mix market snapshots: {bad_markets}")


def build_summaries(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_target: List[Dict[str, Any]] = []
    fits: List[Dict[str, Any]] = []
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["group"])].append(row)
    for group, group_rows in sorted(groups.items()):
        by_target: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_target[float(row["pres_target"])].append(row)
        for target, target_rows in sorted(by_target.items()):
            xm, xs, xsem, xlo, xhi = mean_std_ci(
                [float(row["achieved_pres_post_restore"]) for row in target_rows]
            )
            ym, ys, ysem, ylo, yhi = mean_std_ci(
                [float(row["p_hat_X"]) for row in target_rows]
            )
            per_target.append({
                "group": group,
                "pres_target": target,
                "n_seeds": len(target_rows),
                "seeds": ";".join(str(int(row["seed"])) for row in sorted(target_rows, key=lambda r: int(r["seed"]))),
                "achieved_pres_mean": xm,
                "achieved_pres_std": xs,
                "achieved_pres_sem": xsem,
                "achieved_pres_ci95_low": xlo,
                "achieved_pres_ci95_high": xhi,
                "p_hat_X_mean": ym,
                "p_hat_X_std": ys,
                "p_hat_X_sem": ysem,
                "p_hat_X_ci95_low": ylo,
                "p_hat_X_ci95_high": yhi,
                "C_num_target_max": max(float(row["C_num_run"]) for row in target_rows),
            })
        x = np.asarray([float(row["achieved_pres_post_restore"]) for row in group_rows])
        y = np.asarray([float(row["p_hat_X"]) for row in group_rows])
        clusters = np.asarray([str(row["seed"]) for row in group_rows])
        fit = ols_loglog(x, y)
        robust = cluster_robust_slope_se(x, y, clusters)
        fits.append({
            "group": group,
            "n_points": int(x.size),
            "n_seeds": int(len(set(clusters.tolist()))),
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r2": fit["r2"],
            "cluster_slope_se": robust["se"],
            "cluster_ci95_low": robust["ci_lo"],
            "cluster_ci95_high": robust["ci_hi"],
            "C_num_empirical_upper": float(np.max(y / x)),
            "fit_definition": "log10(p_hat_X) ~ intercept + slope*log10(p_res_post_restore)",
        })
    return per_target, fits


def collect_boundary_sensitivity(
    primary_rows: Sequence[Mapping[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Collect finest-grid/largest-domain trajectories for every boundary."""
    boundary_runs: List[Dict[str, Any]] = []
    boundary_pairs: List[Dict[str, Any]] = []
    boundary_audit: List[Dict[str, Any]] = []
    for source in primary_rows:
        result_dir = Path(str(source["result_dir"]))
        exact_cfg = load_json(result_dir / "exact_map_config.json")
        factors, margins, boundaries, primary_variant = (
            _defect_refinement_protocol(exact_cfg, result_dir=result_dir)
        )
        finest, largest_domain_margin, _primary_boundary = primary_variant
        abs_tolerance = _float(exact_cfg.get("refinement_abs_tolerance"))
        rel_tolerance = _float(exact_cfg.get("refinement_rel_tolerance"))
        defects = read_defects(result_dir / "exact_map_defects.csv")
        expected_iters = sorted(
            int(float(row["defect_iter"])) for row in defects
        )
        raw = _read_defect_refinement_rows(result_dir)
        trajectories: Dict[str, Dict[int, float]] = {}
        for boundary in boundaries:
            selected = [
                row for row in raw
                if int(float(row["grid_factor"])) == finest
                and float(row["fd_margin"]) == largest_domain_margin
                and str(row["boundary"]).strip().lower().replace("_", "-")
                == boundary
            ]
            values: Dict[int, float] = {}
            duplicates: List[int] = []
            invalid: List[int] = []
            for row in selected:
                iteration = int(float(row["defect_iter"]))
                value = _float(row.get("delta_X"))
                if iteration in values:
                    duplicates.append(iteration)
                elif not value >= 0.0:
                    invalid.append(iteration)
                else:
                    values[iteration] = value
            missing = sorted(set(expected_iters) - set(values))
            extra = sorted(set(values) - set(expected_iters))
            complete = not (missing or extra or duplicates or invalid)
            status = "pass" if complete else "incomplete"
            reason_parts = []
            if missing:
                reason_parts.append(f"missing={missing}")
            if extra:
                reason_parts.append(f"extra={extra}")
            if duplicates:
                reason_parts.append(f"duplicates={sorted(set(duplicates))}")
            if invalid:
                reason_parts.append(f"invalid={sorted(set(invalid))}")
            own_statuses: List[str] = []
            own_failed_iters: List[int] = []
            for iteration in expected_iters:
                primary_key = (
                    finest, largest_domain_margin, boundary
                )
                iter_rows = [
                    row for row in raw
                    if int(float(row["defect_iter"])) == iteration
                ]
                lookup = {
                    (
                        int(float(row["grid_factor"])),
                        float(row["fd_margin"]),
                        str(row["boundary"]).strip().lower().replace("_", "-"),
                    ): row
                    for row in iter_rows
                }
                relevant = {
                    (int(factor), largest_domain_margin, boundary)
                    for factor in factors
                } | {
                    (finest, float(margin), boundary)
                    for margin in margins
                }
                if primary_key not in lookup or relevant - set(lookup):
                    own_statuses.append("not_checked")
                    continue
                has_grid_comparison = any(
                    int(factor) != finest for factor in factors
                )
                has_domain_comparison = any(
                    float(margin) != largest_domain_margin
                    for margin in margins
                )
                if (
                    not has_grid_comparison
                    or not has_domain_comparison
                    or largest_domain_margin >= 0.0
                ):
                    own_statuses.append("not_checked")
                    continue
                base = _float(lookup[primary_key].get("delta_X"))
                comparison_values = [
                    _float(lookup[key].get("delta_X"))
                    for key in relevant
                ]
                if not base >= 0.0 or any(
                    not value >= 0.0 for value in comparison_values
                ):
                    own_statuses.append("invalid")
                    own_failed_iters.append(iteration)
                    continue
                tolerance = abs_tolerance + rel_tolerance * abs(base)
                changes = [
                    abs(base - value) for value in comparison_values
                ]
                own_status = (
                    "pass"
                    if all(change <= tolerance for change in changes)
                    else "fail"
                )
                own_statuses.append(own_status)
                if own_status == "fail":
                    own_failed_iters.append(iteration)
            if own_statuses and all(value == "pass" for value in own_statuses):
                own_grid_domain_status = "pass"
            elif any(value in {"fail", "invalid"} for value in own_statuses):
                own_grid_domain_status = "fail"
            else:
                own_grid_domain_status = "incomplete"
            own_counts = {
                value: own_statuses.count(value)
                for value in ("pass", "fail", "invalid", "not_checked")
            }
            if own_grid_domain_status != "pass":
                reason_parts.append(
                    "own grid/domain status="
                    f"{own_grid_domain_status}, failed_iters="
                    f"{sorted(set(own_failed_iters))}"
                )
            boundary_audit.append({
                "group": source["group"],
                "seed": source["seed"],
                "pres_target": source["pres_target"],
                "result_dir": str(result_dir),
                "boundary": boundary,
                "boundary_label": BOUNDARY_LABELS.get(boundary, boundary),
                "grid_factor": finest,
                "fd_margin": largest_domain_margin,
                "n_expected_iterations": len(expected_iters),
                "n_observed_iterations": len(values),
                "expected_iterations": ";".join(map(str, expected_iters)),
                "observed_iterations": ";".join(map(str, sorted(values))),
                "missing_iterations": ";".join(map(str, missing)),
                "full_trajectory_status": status,
                "own_grid_domain_status": own_grid_domain_status,
                "own_grid_domain_pass_count": own_counts["pass"],
                "own_grid_domain_fail_count": own_counts["fail"],
                "own_grid_domain_invalid_count": own_counts["invalid"],
                "own_grid_domain_not_checked_count": own_counts[
                    "not_checked"
                ],
                "own_grid_domain_failed_iterations": ";".join(
                    map(str, sorted(set(own_failed_iters)))
                ),
                "failure_reason": "; ".join(reason_parts),
            })
            if not complete:
                continue
            argmax_iter = max(values, key=lambda item: (values[item], -item))
            p_hat = values[argmax_iter]
            p_res = float(source["achieved_pres_post_restore"])
            boundary_runs.append({
                "group": source["group"],
                "training_group": source["training_group"],
                "exact_protocol_hash": source["exact_protocol_hash"],
                "source_protocol_hash": source["source_protocol_hash"],
                "run_dir": source["run_dir"],
                "result_dir": source["result_dir"],
                "n_assets": source["n_assets"],
                "seed": source["seed"],
                "market_seed": source["market_seed"],
                "market_hash": source["market_hash"],
                "pres_target": source["pres_target"],
                "achieved_pres_post_restore": p_res,
                "boundary": boundary,
                "boundary_label": BOUNDARY_LABELS.get(boundary, boundary),
                "boundary_role": (
                    "primary" if boundary == PRIMARY_BOUNDARY
                    else "sensitivity"
                ),
                "grid_factor": finest,
                "fd_margin": largest_domain_margin,
                "own_grid_domain_status": own_grid_domain_status,
                "paper_eligible": int(own_grid_domain_status == "pass"),
                "p_hat_X": p_hat,
                "C_num_run": p_hat / p_res,
                "argmax_defect_iter": argmax_iter,
                "n_defects": len(values),
                "defect_iter_min": min(values),
                "defect_iter_max": max(values),
            })
            if own_grid_domain_status == "pass":
                trajectories[boundary] = values

        if not {
            PRIMARY_BOUNDARY, "exact-dirichlet"
        }.issubset(trajectories):
            continue
        robin = trajectories[PRIMARY_BOUNDARY]
        direct = trajectories["exact-dirichlet"]
        selected_iters = {
            min(expected_iters),
            max(expected_iters),
            max(robin, key=lambda item: (robin[item], -item)),
            max(direct, key=lambda item: (direct[item], -item)),
        }
        robin_argmax = max(robin, key=lambda item: (robin[item], -item))
        direct_argmax = max(direct, key=lambda item: (direct[item], -item))
        tiny = np.finfo(float).tiny
        for iteration in sorted(selected_iters):
            robin_value = robin[iteration]
            direct_value = direct[iteration]
            roles = []
            if iteration == min(expected_iters):
                roles.append("first")
            if iteration == max(expected_iters):
                roles.append("last")
            if iteration == robin_argmax:
                roles.append("argmax_robin")
            if iteration == direct_argmax:
                roles.append("argmax_exact_dirichlet")
            boundary_pairs.append({
                "group": source["group"],
                "seed": source["seed"],
                "pres_target": source["pres_target"],
                "result_dir": source["result_dir"],
                "defect_iter": iteration,
                "selection_role": ";".join(roles),
                "robin_delta_X": robin_value,
                "exact_dirichlet_delta_X": direct_value,
                "abs_difference": abs(direct_value - robin_value),
                "signed_relative_difference_to_robin": (
                    (direct_value - robin_value)
                    / max(abs(robin_value), tiny)
                ),
                "absolute_relative_difference_to_robin": (
                    abs(direct_value - robin_value)
                    / max(abs(robin_value), tiny)
                ),
            })
    return boundary_runs, boundary_pairs, boundary_audit


def build_boundary_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_target: List[Dict[str, Any]] = []
    fits: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            int(row.get("paper_eligible", 0)) != 1
            or str(row.get("own_grid_domain_status")) != "pass"
        ):
            continue
        groups[(str(row["group"]), str(row["boundary"]))].append(row)
    for (group, boundary), group_rows in sorted(groups.items()):
        by_target: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_target[float(row["pres_target"])].append(row)
        for target, target_rows in sorted(by_target.items()):
            xm, xs, xsem, xlo, xhi = mean_std_ci(
                [float(row["achieved_pres_post_restore"]) for row in target_rows]
            )
            ym, ys, ysem, ylo, yhi = mean_std_ci(
                [float(row["p_hat_X"]) for row in target_rows]
            )
            per_target.append({
                "group": group,
                "boundary": boundary,
                "boundary_label": BOUNDARY_LABELS.get(boundary, boundary),
                "pres_target": target,
                "n_seeds": len(target_rows),
                "seeds": ";".join(
                    str(int(row["seed"]))
                    for row in sorted(
                        target_rows, key=lambda item: int(item["seed"])
                    )
                ),
                "achieved_pres_mean": xm,
                "achieved_pres_std": xs,
                "achieved_pres_sem": xsem,
                "achieved_pres_ci95_low": xlo,
                "achieved_pres_ci95_high": xhi,
                "p_hat_X_mean": ym,
                "p_hat_X_std": ys,
                "p_hat_X_sem": ysem,
                "p_hat_X_ci95_low": ylo,
                "p_hat_X_ci95_high": yhi,
                "C_num_target_max": max(
                    float(row["C_num_run"]) for row in target_rows
                ),
            })
        x = np.asarray([
            float(row["achieved_pres_post_restore"]) for row in group_rows
        ])
        y = np.asarray([float(row["p_hat_X"]) for row in group_rows])
        clusters = np.asarray([str(row["seed"]) for row in group_rows])
        fit = ols_loglog(x, y)
        robust = cluster_robust_slope_se(x, y, clusters)
        fits.append({
            "group": group,
            "boundary": boundary,
            "boundary_label": BOUNDARY_LABELS.get(boundary, boundary),
            "n_points": int(x.size),
            "n_seeds": int(len(set(clusters.tolist()))),
            "n_grid_domain_pass_runs": sum(
                str(row.get("own_grid_domain_status")) == "pass"
                for row in group_rows
            ),
            "n_grid_domain_fail_runs": sum(
                str(row.get("own_grid_domain_status")) == "fail"
                for row in group_rows
            ),
            "n_grid_domain_incomplete_runs": sum(
                str(row.get("own_grid_domain_status")) == "incomplete"
                for row in group_rows
            ),
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r2": fit["r2"],
            "cluster_slope_se": robust["se"],
            "cluster_ci95_low": robust["ci_lo"],
            "cluster_ci95_high": robust["ci_hi"],
            "C_num_empirical_upper": float(np.max(y / x)),
            "fit_definition": (
                "log10(p_hat_X) ~ intercept + "
                "slope*log10(p_res_post_restore)"
            ),
        })
    return per_target, fits


def summarize_boundary_pairs(
    pairs: Sequence[Mapping[str, Any]],
    boundary_runs: Sequence[Mapping[str, Any]],
    boundary_fits: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    groups = sorted({str(row["group"]) for row in boundary_runs})
    tiny = np.finfo(float).tiny
    for group in groups:
        group_pairs = [
            row for row in pairs if str(row["group"]) == group
        ]
        by_run: Dict[Tuple[int, float], Dict[str, Mapping[str, Any]]] = (
            defaultdict(dict)
        )
        for row in boundary_runs:
            if (
                str(row["group"]) == group
                and int(row.get("paper_eligible", 0)) == 1
                and str(row.get("own_grid_domain_status")) == "pass"
            ):
                by_run[
                    (int(row["seed"]), float(row["pres_target"]))
                ][str(row["boundary"])] = row
        run_diffs: List[Tuple[float, int, float]] = []
        for (seed, target), values in by_run.items():
            if {
                PRIMARY_BOUNDARY, "exact-dirichlet"
            }.issubset(values):
                robin = float(values[PRIMARY_BOUNDARY]["p_hat_X"])
                direct = float(values["exact-dirichlet"]["p_hat_X"])
                run_diffs.append((
                    abs(direct - robin) / max(abs(robin), tiny),
                    seed,
                    target,
                ))
        pair_diffs = [
            float(row["absolute_relative_difference_to_robin"])
            for row in group_pairs
        ]
        fit_by_boundary = {
            str(row["boundary"]): row
            for row in boundary_fits
            if str(row["group"]) == group
        }
        if not run_diffs or not pair_diffs or not {
            PRIMARY_BOUNDARY, "exact-dirichlet"
        }.issubset(fit_by_boundary):
            continue
        worst_pair = max(
            group_pairs,
            key=lambda row: float(
                row["absolute_relative_difference_to_robin"]
            ),
        )
        worst_run = max(run_diffs)
        robin_fit = fit_by_boundary[PRIMARY_BOUNDARY]
        direct_fit = fit_by_boundary["exact-dirichlet"]
        robin_slope = float(robin_fit["slope"])
        direct_slope = float(direct_fit["slope"])
        robin_cnum = float(robin_fit["C_num_empirical_upper"])
        direct_cnum = float(direct_fit["C_num_empirical_upper"])
        summaries.append({
            "group": group,
            "primary_boundary": PRIMARY_BOUNDARY,
            "audit_boundary": "exact-dirichlet",
            "audit_boundary_label": BOUNDARY_LABELS["exact-dirichlet"],
            "n_paired_runs": len(run_diffs),
            "n_selected_iteration_pairs": len(group_pairs),
            "selected_pair_rel_diff_median": float(np.median(pair_diffs)),
            "selected_pair_rel_diff_max": float(max(pair_diffs)),
            "selected_pair_argmax_seed": worst_pair["seed"],
            "selected_pair_argmax_target": worst_pair["pres_target"],
            "selected_pair_argmax_defect_iter": worst_pair["defect_iter"],
            "run_p_hat_rel_diff_median": float(np.median([
                value[0] for value in run_diffs
            ])),
            "run_p_hat_rel_diff_max": worst_run[0],
            "run_p_hat_argmax_seed": worst_run[1],
            "run_p_hat_argmax_target": worst_run[2],
            "robin_slope": robin_slope,
            "exact_dirichlet_slope": direct_slope,
            "slope_abs_difference": abs(direct_slope - robin_slope),
            "slope_relative_difference_to_robin": (
                abs(direct_slope - robin_slope)
                / max(abs(robin_slope), tiny)
            ),
            "robin_C_num_empirical_upper": robin_cnum,
            "exact_dirichlet_C_num_empirical_upper": direct_cnum,
            "C_num_abs_difference": abs(direct_cnum - robin_cnum),
            "C_num_relative_difference_to_robin": (
                abs(direct_cnum - robin_cnum)
                / max(abs(robin_cnum), tiny)
            ),
        })
    return summaries


def parse_formats(text: str) -> List[str]:
    values = [part.lower() for part in re.split(r"[\s,]+", str(text)) if part]
    if not values or len(values) != len(set(values)) or set(values) - FORMATS:
        raise ValueError(f"invalid --formats={text!r}")
    return values


def make_figure(
    per_target: Sequence[Mapping[str, Any]],
    fits: Sequence[Mapping[str, Any]],
    output: Path,
    formats: Sequence[str],
    dpi: int,
    *,
    fig_width: float = 4.8,
    fig_height: float = 3.4,
    font_size: float = 10.0,
    x_tick_count: int = 3,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plotted_x: List[float] = []
    rc = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        for color, fit_row in zip(colors, fits):
            group = str(fit_row["group"])
            rows = sorted(
                [row for row in per_target if str(row["group"]) == group],
                key=lambda row: float(row["achieved_pres_mean"]),
            )
            x = np.asarray(
                [float(row["achieved_pres_mean"]) for row in rows]
            )
            y = np.asarray([float(row["p_hat_X_mean"]) for row in rows])
            yerr = np.asarray([float(row["p_hat_X_std"]) for row in rows])
            plotted_x.extend(float(value) for value in x)
            # The solid curve is the target-wise seed mean with sample-SD
            # error bars.  Its meaning belongs in the paper caption, so the
            # plot intentionally carries no legend entry.
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                color=color,
                capsize=2.5,
            )
            # C_num is the empirical through-origin slope-one upper envelope:
            # C_num = max_run p_hat_X / p_res.  Keep the bound visible but
            # leave its definition and numerical value to the caption/table.
            c_num = float(fit_row["C_num_empirical_upper"])
            xx = np.geomspace(float(x.min()), float(x.max()), 100)
            ax.plot(
                xx,
                c_num * xx,
                linestyle="--",
                color=color,
                linewidth=1.0,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\mathrm{res}}$")
        ax.set_ylabel(r"$\widehat p_X$")
        ax.grid(True, which="both", alpha=0.25)

        unique_x = np.unique(np.asarray(plotted_x, dtype=np.float64))
        unique_x = unique_x[np.isfinite(unique_x) & (unique_x > 0.0)]
        if unique_x.size:
            count = min(int(x_tick_count), int(unique_x.size))
            indices = np.rint(
                np.linspace(0, unique_x.size - 1, count)
            ).astype(int)
            ticks = unique_x[np.unique(indices)]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _position: f"{value:.2g}")
            )
            # A log axis retains its own minor locator/formatter even after
            # replacing the major locator.  Suppress those numerical labels
            # so --x-tick-count controls the number actually shown.
            ax.xaxis.set_minor_formatter(NullFormatter())

        fig.tight_layout()
        for fmt in formats:
            fig.savefig(
                output / f"regularity_transfer.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Merton E4 regularity-transfer evidence."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig-width",
        type=float,
        default=4.8,
        help="figure width in inches (default: 4.8)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=3.4,
        help="figure height in inches (default: 3.4)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="axis-label and tick-label font size in points (default: 10)",
    )
    parser.add_argument(
        "--x-tick-count",
        type=int,
        default=3,
        help=(
            "maximum number of labeled x ticks, selected from the plotted "
            "target locations (default: 3)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--failure-mode",
        choices=("warn", "error"),
        default="warn",
        help=(
            "warn (default) writes a complete audit and returns success even "
            "when some exact-map runs fail paper checks; error writes the "
            "same audit first and then returns a nonzero status"
        ),
    )
    args = parser.parse_args(argv)
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2")
    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if not math.isfinite(args.fig_width) or args.fig_width <= 0.0:
        raise ValueError("--fig-width must be positive")
    if not math.isfinite(args.fig_height) or args.fig_height <= 0.0:
        raise ValueError("--fig-height must be positive")
    if not math.isfinite(args.font_size) or args.font_size <= 0.0:
        raise ValueError("--font-size must be positive")
    if args.x_tick_count < 1:
        raise ValueError("--x-tick-count must be positive")
    formats = parse_formats(args.formats)
    root = Path(args.out_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root / "regularity_transfer"
    )
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output is not empty: {output}; pass --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    evidence_root = output / "evidence"
    if args.overwrite and evidence_root.exists():
        shutil.rmtree(evidence_root)
    evidence_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for fmt in FORMATS:
            old_figure = output / f"regularity_transfer.{fmt}"
            if old_figure.is_file():
                old_figure.unlink()
        for marker_name in (
            "_SUCCESS_REGULARITY_TRANSFER",
            "_INCOMPLETE_REGULARITY_TRANSFER",
        ):
            marker = output / marker_name
            if marker.is_file():
                marker.unlink()

    rows, audits, refinement_audits, evidence_manifest = (
        collect_runs_with_audit(
            root,
            n_assets=args.n_assets,
            run_name_regex=args.run_name_regex,
            evidence_root=evidence_root,
        )
    )
    expected = set(parse_seed_spec(args.expected_seeds))
    audit_fields = [
        "n_assets", "seed", "pres_target", "run_dir", "result_dir",
        "evidence_dir", "source_protocol_hash", "collection_status",
        "paper_eligible", "error_type", "error_message",
        "failed_required_defects", "failed_axes",
    ]
    refinement_audit_fields = [
        "seed", "pres_target", "result_dir", "defect_iter",
        "required_for_paper", "grid_factor", "fd_margin", "boundary",
        "delta_X", "source_refinement_status", "refinement_status",
        "legacy_combined_refinement_status",
        "source_legacy_combined_refinement_status",
        "grid_domain_refinement_status", "grid_domain_status_source",
        "refinement_tolerance", "defect_grid_abs_change",
        "defect_domain_abs_change", "defect_boundary_abs_change",
        "defect_sensitivity_envelope", "failed_axes", "nonfinite_axes",
        "missing_grid_domain_variants",
        "failure_code", "failure_reason",
    ]
    manifest_fields = [
        "seed", "pres_target", "result_dir", "source_path",
        "copied_path", "sha256", "size_bytes", "copy_error",
    ]
    # These three files are deliberately written before panel validation.
    # A failed paper panel must still leave a complete cross-seed diagnosis.
    write_csv(
        output / "regularity_transfer_audit.csv",
        audits,
        audit_fields,
    )
    write_csv(
        output / "regularity_transfer_refinement_audit.csv",
        refinement_audits,
        refinement_audit_fields,
    )
    write_csv(
        output / "regularity_transfer_evidence_manifest.csv",
        evidence_manifest,
        manifest_fields,
    )

    warnings: List[str] = []
    for audit in audits:
        if str(audit.get("collection_status")) == "pass":
            continue
        warnings.append(
            "seed={seed} target={target}: {kind}: {message}; "
            "failed_required_defects={defects}; failed_axes={axes}".format(
                seed=audit.get("seed", "unknown"),
                target=audit.get("pres_target", "unknown"),
                kind=audit.get("error_type", "validation failure"),
                message=audit.get("error_message", ""),
                defects=audit.get("failed_required_defects", ""),
                axes=audit.get("failed_axes", ""),
            )
        )
    copy_failures = [
        row for row in evidence_manifest if row.get("copy_error")
    ]
    for row in copy_failures:
        warnings.append(
            f"seed={row.get('seed', 'unknown')} "
            f"target={row.get('pres_target', 'unknown')}: "
            f"evidence copy failed: {row.get('copy_error')}"
        )
    if expected and args.min_seeds > len(expected):
        warnings.append(
            f"--min-seeds={args.min_seeds} exceeds the "
            f"{len(expected)} explicitly expected seed(s) "
            f"{sorted(expected)}"
        )
    try:
        validate_panel(rows, expected, args.min_seeds)
    except (OSError, ValueError, KeyError) as exc:
        warnings.append(f"paper panel validation failed: {exc}")

    run_fields = [
        "group", "training_group", "exact_protocol_hash",
        "source_protocol_hash", "run_dir", "result_dir",
        "n_assets", "seed", "market_seed",
        "market_hash", "pres_target", "achieved_pres_post_restore",
        "p_hat_X", "C_num_run", "n_outer_residuals",
        "n_adjacent_defects", "completed_outer_iters", "e6_role",
        "initial_defect_mode", "checkpoint_index_semantics",
        "defect_coverage", "defect_refinement_statuses",
        "defect_refinement_required_iterations",
        "legacy_combined_refinement_evidence_status",
        "grid_domain_refinement_evidence_status",
        "grid_domain_status_source", "grid_domain_required_statuses",
        "eval_margin",
        "ev_y_min", "ev_y_max", "defect_iter_min", "defect_iter_max",
        "residual_semantics", "updated_at",
    ]
    write_csv(output / "regularity_transfer_runs.csv", rows, [
        *run_fields,
    ])
    per_target_fields = [
        "group", "pres_target", "n_seeds", "seeds", "achieved_pres_mean",
        "achieved_pres_std", "achieved_pres_sem", "achieved_pres_ci95_low",
        "achieved_pres_ci95_high", "p_hat_X_mean", "p_hat_X_std",
        "p_hat_X_sem", "p_hat_X_ci95_low", "p_hat_X_ci95_high",
        "C_num_target_max",
    ]
    fit_fields = [
        "group", "n_points", "n_seeds", "slope", "intercept", "r2",
        "cluster_slope_se", "cluster_ci95_low", "cluster_ci95_high",
        "C_num_empirical_upper", "fit_definition",
    ]
    boundary_runs: List[Dict[str, Any]] = []
    boundary_pairs: List[Dict[str, Any]] = []
    boundary_audit: List[Dict[str, Any]] = []
    boundary_per_target: List[Dict[str, Any]] = []
    boundary_fits: List[Dict[str, Any]] = []
    boundary_summary: List[Dict[str, Any]] = []
    boundary_warnings: List[str] = []
    try:
        boundary_runs, boundary_pairs, boundary_audit = (
            collect_boundary_sensitivity(rows)
        )
        boundary_per_target, boundary_fits = build_boundary_summaries(
            boundary_runs
        )
        boundary_summary = summarize_boundary_pairs(
            boundary_pairs, boundary_runs, boundary_fits
        )
        for audit in boundary_audit:
            if (
                str(audit["full_trajectory_status"]) == "pass"
                and str(audit["own_grid_domain_status"]) == "pass"
            ):
                continue
            boundary_warnings.append(
                "boundary sensitivity audit issue: "
                f"seed={audit['seed']} target={audit['pres_target']} "
                f"boundary={audit['boundary']} "
                f"trajectory={audit['full_trajectory_status']} "
                f"own_grid_domain={audit['own_grid_domain_status']} "
                f"{audit['failure_reason']}"
            )
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        boundary_warnings.append(
            "boundary sensitivity collection failed: "
            f"{type(exc).__name__}: {exc}"
        )

    boundary_run_fields = [
        "group", "training_group", "exact_protocol_hash",
        "source_protocol_hash", "run_dir", "result_dir", "n_assets",
        "seed", "market_seed", "market_hash", "pres_target",
        "achieved_pres_post_restore", "boundary", "boundary_label",
        "boundary_role", "grid_factor", "fd_margin", "p_hat_X",
        "own_grid_domain_status", "paper_eligible", "C_num_run",
        "argmax_defect_iter", "n_defects",
        "defect_iter_min", "defect_iter_max",
    ]
    boundary_audit_fields = [
        "group", "seed", "pres_target", "result_dir", "boundary",
        "boundary_label", "grid_factor", "fd_margin",
        "n_expected_iterations", "n_observed_iterations",
        "expected_iterations", "observed_iterations",
        "missing_iterations", "full_trajectory_status",
        "own_grid_domain_status", "own_grid_domain_pass_count",
        "own_grid_domain_fail_count", "own_grid_domain_invalid_count",
        "own_grid_domain_not_checked_count",
        "own_grid_domain_failed_iterations", "failure_reason",
    ]
    boundary_per_target_fields = [
        "group", "boundary", "boundary_label", "pres_target",
        "n_seeds", "seeds", "achieved_pres_mean", "achieved_pres_std",
        "achieved_pres_sem", "achieved_pres_ci95_low",
        "achieved_pres_ci95_high", "p_hat_X_mean", "p_hat_X_std",
        "p_hat_X_sem", "p_hat_X_ci95_low", "p_hat_X_ci95_high",
        "C_num_target_max",
    ]
    boundary_fit_fields = [
        "group", "boundary", "boundary_label", "n_points", "n_seeds",
        "n_grid_domain_pass_runs", "n_grid_domain_fail_runs",
        "n_grid_domain_incomplete_runs",
        "slope", "intercept", "r2", "cluster_slope_se",
        "cluster_ci95_low", "cluster_ci95_high",
        "C_num_empirical_upper", "fit_definition",
    ]
    boundary_pair_fields = [
        "group", "seed", "pres_target", "result_dir", "defect_iter",
        "selection_role", "robin_delta_X", "exact_dirichlet_delta_X",
        "abs_difference", "signed_relative_difference_to_robin",
        "absolute_relative_difference_to_robin",
    ]
    boundary_summary_fields = [
        "group", "primary_boundary", "audit_boundary",
        "audit_boundary_label", "n_paired_runs",
        "n_selected_iteration_pairs", "selected_pair_rel_diff_median",
        "selected_pair_rel_diff_max", "selected_pair_argmax_seed",
        "selected_pair_argmax_target", "selected_pair_argmax_defect_iter",
        "run_p_hat_rel_diff_median", "run_p_hat_rel_diff_max",
        "run_p_hat_argmax_seed", "run_p_hat_argmax_target",
        "robin_slope", "exact_dirichlet_slope", "slope_abs_difference",
        "slope_relative_difference_to_robin",
        "robin_C_num_empirical_upper",
        "exact_dirichlet_C_num_empirical_upper",
        "C_num_abs_difference", "C_num_relative_difference_to_robin",
    ]
    write_csv(
        output / "regularity_transfer_boundary_runs.csv",
        boundary_runs,
        boundary_run_fields,
    )
    write_csv(
        output / "regularity_transfer_boundary_audit.csv",
        boundary_audit,
        boundary_audit_fields,
    )
    write_csv(
        output / "regularity_transfer_boundary_per_target.csv",
        boundary_per_target,
        boundary_per_target_fields,
    )
    write_csv(
        output / "regularity_transfer_boundary_fit.csv",
        boundary_fits,
        boundary_fit_fields,
    )
    write_csv(
        output / "regularity_transfer_boundary_pairs.csv",
        boundary_pairs,
        boundary_pair_fields,
    )
    write_csv(
        output / "regularity_transfer_boundary_summary.csv",
        boundary_summary,
        boundary_summary_fields,
    )
    per_target: List[Dict[str, Any]] = []
    fits: List[Dict[str, Any]] = []
    paper_outputs_generated = False
    if not warnings:
        try:
            per_target, fits = build_summaries(rows)
            make_figure(
                per_target,
                fits,
                output,
                formats,
                args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                font_size=args.font_size,
                x_tick_count=args.x_tick_count,
            )
            paper_outputs_generated = True
        except (OSError, ValueError, KeyError, FloatingPointError) as exc:
            warnings.append(
                f"paper summary/figure generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            per_target = []
            fits = []
            for fmt in FORMATS:
                old_figure = output / f"regularity_transfer.{fmt}"
                if old_figure.is_file():
                    old_figure.unlink()
    write_csv(
        output / "regularity_transfer_per_target.csv",
        per_target,
        per_target_fields,
    )
    write_csv(
        output / "regularity_transfer_fit.csv",
        fits,
        fit_fields,
    )
    warnings.extend(boundary_warnings)

    warning_path = output / "regularity_transfer_warnings.txt"
    warning_text = "\n".join(warnings)
    warning_path.write_text(
        warning_text + ("\n" if warning_text else ""),
        encoding="utf-8",
    )
    status_payload = {
        "status": (
            "success"
            if paper_outputs_generated
            else "diagnostic_with_failures"
        ),
        "paper_eligible": paper_outputs_generated,
        "paper_outputs_generated": paper_outputs_generated,
        "failure_mode": args.failure_mode,
        "n_discovered_results": len(audits),
        "n_paper_eligible_runs": len(rows),
        "n_failed_results": sum(
            str(row.get("collection_status")) != "pass"
            for row in audits
        ),
        "n_refinement_audit_rows": len(refinement_audits),
        "n_boundary_run_rows": len(boundary_runs),
        "n_boundary_pair_rows": len(boundary_pairs),
        "n_boundary_incomplete_rows": sum(
            str(row.get("full_trajectory_status")) != "pass"
            for row in boundary_audit
        ),
        "boundary_sensitivity_complete": not bool(boundary_warnings),
        "n_evidence_files": sum(
            not bool(row.get("copy_error"))
            for row in evidence_manifest
        ),
        "warnings": warnings,
    }
    with (
        output / "regularity_transfer_status.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(status_payload, handle, indent=2, sort_keys=True)
    with (
        output / "regularity_transfer_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump({
            "arguments": vars(args),
            "residual_semantics": "max over official post-restore outer states",
            "p_hat_definition": (
                "max delta_X within each run: standard runs cover "
                "delta_0,...,delta_(K-1); E6 target branches cover the "
                "target-dependent delta_1,...,delta_K and exclude the shared "
                "common-warm-up delta_0"
            ),
            "C_num_definition": "max_run p_hat_X / p_res_post_restore",
            "fd_interpretation": (
                "The primary Figure S1 uses the finest-grid, largest-domain "
                "homogeneous CRRA Robin trajectory. Grid/domain eligibility "
                "is reconstructed from raw delta_X variants and excludes "
                "boundary replacement. The optimal-reference Dirichlet "
                "closure is reported separately as a finite-domain BVP "
                "sensitivity audit."
            ),
            "paper_policy": (
                "Only runs passing every strict exact-map, evaluated-bundle, "
                "and required defect-refinement check enter summaries or "
                "figures. Failed runs remain in the audit and evidence tree."
            ),
            "n_runs": len(rows),
            "n_discovered_results": len(audits),
            "n_failed_results": status_payload["n_failed_results"],
            "paper_outputs_generated": paper_outputs_generated,
            "warnings_file": warning_path.name,
            "audit_file": "regularity_transfer_audit.csv",
            "refinement_audit_file": (
                "regularity_transfer_refinement_audit.csv"
            ),
            "evidence_manifest_file": (
                "regularity_transfer_evidence_manifest.csv"
            ),
            "evidence_directory": "evidence",
            "boundary_runs_file": (
                "regularity_transfer_boundary_runs.csv"
            ),
            "boundary_per_target_file": (
                "regularity_transfer_boundary_per_target.csv"
            ),
            "boundary_fit_file": (
                "regularity_transfer_boundary_fit.csv"
            ),
            "boundary_pairs_file": (
                "regularity_transfer_boundary_pairs.csv"
            ),
            "boundary_summary_file": (
                "regularity_transfer_boundary_summary.csv"
            ),
            "boundary_audit_file": (
                "regularity_transfer_boundary_audit.csv"
            ),
        }, handle, indent=2, sort_keys=True)

    if paper_outputs_generated:
        (output / "_SUCCESS_REGULARITY_TRANSFER").write_text(
            "success\n", encoding="utf-8"
        )
    else:
        (output / "_INCOMPLETE_REGULARITY_TRANSFER").write_text(
            warning_text + ("\n" if warning_text else ""),
            encoding="utf-8",
        )
    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    print(
        "[done] Merton E4 regularity-transfer "
        f"{'paper outputs' if paper_outputs_generated else 'audit'}: {output}"
    )
    if warnings and args.failure_mode == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
