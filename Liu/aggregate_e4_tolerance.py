#!/usr/bin/env python3
"""Aggregate Liu E4 FD approximation audits across residual tolerances.

This is a post-processing bridge between already completed training runs and
already completed ``liu_exact_map_fd.py`` audits.  It never reruns the FD
solver.  The comparison unit is a training protocol that is identical after
removing *only* ``pres_target``; seed and filesystem locations have already
been removed by the exact-map driver.

For every seed/tolerance cell, maxima are formed over the requested outer
checkpoints first.  The reported mean, sample SD, and Student-t 95% interval
are then computed across those seedwise maxima.  Consequently no checkpoint
is treated as an independent replication.

Grid/domain refinement is assessed within the primary ``linearity`` BVP.
Alternative boundary closures are summarized as matched finite-domain BVP
sensitivity (slope, C_num, and relative changes) and never used as a binary
refinement gate. Legacy full-Cartesian artifacts are reassessed from their raw
refinement CSVs, so this semantic correction does not require rerunning FD.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_liu_exact_map import (
    CONFIG_INPUT,
    E4_INPUT,
    EXACT_INPUT,
    STATUS_INPUT,
    _identity,
    _integer,
    _matching_artifact,
    _number,
    _stats,
    _validate_artifact_hashes,
    _e4_paper_evidence,
    _validate_provenance,
    _validate_status_contract,
    discover_result_dirs,
    parse_seed_spec,
    read_csv,
    read_json,
)
from liu_exact_map_fd import (
    BOUNDARY_SENSITIVITY_ROLE,
    REFINEMENT_SCOPE,
    _assess,
)


PRIMARY_ERROR_METRICS = (
    "e_approx_value",
    "e_approx_bundle",
    "e_approx_X",
)
OPTIONAL_ERROR_METRICS = (
    "e_approx_control",
    "e_approx_theta",
    "e_approx_vartheta",
    "approx_sensitivity_envelope",
)
DIAGNOSTIC_SPECS = (
    # output name, source column, reduction
    ("min_source_min_log_joint_eig", "source_min_log_joint_eig", "min"),
    ("max_source_max_log_joint_eig", "source_max_log_joint_eig", "max"),
    (
        "max_source_nonpositive_log_eig_fraction",
        "source_nonpositive_log_eig_fraction",
        "max",
    ),
)
STATIC_OUTPUTS = (
    "e4_tolerance_per_seed.csv",
    "e4_tolerance_summary.csv",
    "e4_boundary_sensitivity_per_checkpoint.csv",
    "e4_boundary_sensitivity_per_seed.csv",
    "e4_boundary_sensitivity_summary.csv",
    "e4_tolerance_aggregate_status.json",
    "_SUCCESS_E4_TOLERANCE_AGG",
    "_FAILED_E4_TOLERANCE_AGG",
)
PLOT_STEM = "e4_tolerance_errors"
BOUNDARY_PLOT_STEM = "e4_boundary_sensitivity"
PLOT_FORMATS = ("png", "pdf", "svg", "eps")
PLOT_METRIC_SPECS = {
    "max_e_approx_X": {
        "aliases": ("x", "e_approx_x"),
        "y_label": r"$\widehat p_X$",
    },
    "max_e_approx_value": {
        "aliases": ("value", "e_approx_value"),
        "y_label": r"$\widehat p_{\mathrm{value}}$",
    },
    "max_e_approx_bundle": {
        "aliases": ("bundle", "e_approx_bundle"),
        "y_label": r"$\widehat p_{\mathrm{bundle}}$",
    },
    "max_e_approx_control": {
        "aliases": ("control", "e_approx_control"),
        "y_label": r"$\widehat p_{\mathrm{control}}$",
    },
    "max_e_approx_theta": {
        "aliases": ("theta", "e_approx_theta"),
        "y_label": r"$\widehat p_{\theta}$",
    },
    "max_e_approx_vartheta": {
        "aliases": ("vartheta", "e_approx_vartheta"),
        "y_label": r"$\widehat p_{\vartheta}$",
    },
    "max_approx_sensitivity_envelope": {
        "aliases": (
            "sensitivity",
            "sensitivity-envelope",
            "approx_sensitivity_envelope",
        ),
        "y_label": r"$\widehat p_{\mathrm{sens}}$",
    },
}


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_key(value: float) -> str:
    return float(value).hex()


def parse_float_spec(text: str) -> List[float]:
    values: Dict[str, float] = {}
    for token in re.split(r"[\s,]+", str(text).strip()):
        if not token:
            continue
        value = float(token)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"residual tolerances must be finite and positive: {token!r}")
        values[_float_key(value)] = value
    return sorted(values.values())


def _positive_target(value: Any, source: Path) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: invalid pres_target={value!r}"
        ) from exc
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(
            f"{source}: pres_target must be finite and positive, got {value!r}"
        )
    return target


def _path_selection_identity(directory: Path) -> Tuple[Optional[float], Optional[int]]:
    """Best-effort identity for a failed transactional launcher output."""

    target: Optional[float] = None
    seed: Optional[int] = None
    for path in (
        directory,
        directory.parent,
        directory.parent.parent,
    ):
        name = path.name
        seed_match = re.search(r"(?:^|_)seed(-?\d+)$", name)
        if seed is None and seed_match:
            seed = int(seed_match.group(1))
        target_match = re.match(
            r"^pres_(.+?)(?:_seed-?\d+)?$", name
        )
        if target is None and target_match:
            token = target_match.group(1).replace("p", ".")
            try:
                candidate = float(token)
            except ValueError:
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                target = candidate
    return target, seed


def _selection_target(directory: Path) -> float:
    config_path = directory / CONFIG_INPUT
    if config_path.is_file():
        config = read_json(config_path)
        training = config.get("training_protocol_args")
        if not isinstance(training, Mapping):
            raise ValueError(
                f"{config_path}: missing training_protocol_args needed by "
                "--select-target"
            )
        return _positive_target(training.get("pres_target"), config_path)
    target, _seed = _path_selection_identity(directory)
    if target is None:
        raise ValueError(
            f"{directory}: cannot determine pres_target needed by "
            "--select-target; failed outputs outside the standard "
            "pres_<target>/seed<seed> launcher layout must be supplied "
            "explicitly with --result-dir"
        )
    return target


def _selection_seed(directory: Path) -> int:
    config_path = directory / CONFIG_INPUT
    config: Mapping[str, Any] = {}
    if config_path.is_file():
        config = read_json(config_path)
        candidate_paths: List[Path] = []
        for raw in (
            config.get("config_path"),
            (
                Path(str(config.get("run_dir"))) / "config.json"
                if config.get("run_dir")
                else None
            ),
        ):
            if raw:
                candidate_paths.append(Path(str(raw)))
        for path in candidate_paths:
            if not path.is_file():
                continue
            source = read_json(path)
            args = source.get("args")
            raw_seed = (
                args.get("seed")
                if isinstance(args, Mapping)
                else source.get("seed")
            )
            if raw_seed is None or isinstance(raw_seed, bool):
                continue
            try:
                return int(raw_seed)
            except (TypeError, ValueError):
                continue

    e4_path = directory / E4_INPUT
    if e4_path.is_file():
        rows = read_csv(e4_path)
        return int(_identity(rows, "seed", e4_path))

    path_candidates = [directory, directory.parent]
    if config.get("run_dir"):
        path_candidates.append(Path(str(config["run_dir"])))
    for path in path_candidates:
        _target, seed = _path_selection_identity(path)
        if seed is not None:
            return seed
    raise ValueError(
        f"{directory}: cannot determine training seed needed by "
        "--select-seeds; failed outputs outside the standard "
        "pres_<target>/seed<seed> launcher layout must be supplied "
        "explicitly with --result-dir"
    )


def _select_result_dirs(
    result_dirs: Sequence[Path],
    *,
    select_targets: Sequence[float],
    select_seeds: Sequence[int],
) -> Tuple[List[Path], List[Path]]:
    """Filter discovered cells before expensive paper-grade validation."""

    target_keys = {_float_key(float(value)) for value in select_targets}
    seed_values = {int(value) for value in select_seeds}
    selected: List[Path] = []
    excluded: List[Path] = []
    for directory in result_dirs:
        if target_keys:
            target = _selection_target(directory)
            if _float_key(target) not in target_keys:
                excluded.append(directory)
                continue
        if seed_values:
            seed = _selection_seed(directory)
            if seed not in seed_values:
                excluded.append(directory)
                continue
        selected.append(directory)
    if not selected:
        raise ValueError(
            "selection excluded every discovered E4 result directory"
        )
    return selected, excluded


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _managed_names(output: Path) -> List[str]:
    names = list(STATIC_OUTPUTS)
    for stem in (PLOT_STEM, BOUNDARY_PLOT_STEM):
        names.extend(
            path.name
            for path in output.glob(f"{stem}.*")
            if path.suffix.lower().lstrip(".") in PLOT_FORMATS
        )
    return sorted(set(names))


def _check_output(output: Path, overwrite: bool) -> bool:
    existing = [name for name in _managed_names(output) if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"E4 tolerance output already contains {existing}; pass --overwrite"
        )
    return bool(existing)


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in _managed_names(output):
        path = output / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def _commit_stage(stage: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(
        prefix=".liu-e4-tolerance-backup-", dir=str(output.parent)
    ))
    moved_old: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        for name in _managed_names(output):
            original = output / name
            if original.exists() or original.is_symlink():
                saved = backup / name
                os.replace(original, saved)
                moved_old.append((saved, original))
        marker = stage / "_SUCCESS_E4_TOLERANCE_AGG"
        ordered = [path for path in sorted(stage.iterdir()) if path != marker]
        if marker.is_file() or marker.is_symlink():
            ordered.append(marker)
        for path in ordered:
            if not (path.is_file() or path.is_symlink()):
                continue
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


def _canonical_market_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as payload:
        # The network seed is intentionally excluded by the exact-map driver;
        # market_seed and every actual model parameter remain included.
        for name in sorted(key for key in payload.files if key != "seed"):
            value = np.ascontiguousarray(np.asarray(payload[name]))
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


_MERTON_CONFIG_FIELDS = (
    "refinement_rule",
    "min_paper_checkpoint",
    "excluded_initial_checkpoints",
    "paper_checkpoint_schedule",
    "e4_refinement_rule",
)
_BOUNDARY_CONFIG_FIELDS = (
    "refinement_scope",
    "boundary_sensitivity_role",
    "primary_boundary",
    "comparison_boundaries",
)
_MERTON_STATUS_FIELDS = (
    "refinement_rule",
    "min_paper_checkpoint",
    "excluded_initial_checkpoints",
    "paper_checkpoint_schedule",
    "e4_refinement_required_iterations",
    "e4_refinement_required_statuses",
    "e4_refinement_evidence_status",
    "n_e4_refinement_pass",
)
_BOUNDARY_STATUS_FIELDS = (
    "refinement_scope",
    "boundary_sensitivity_role",
    "primary_boundary",
    "comparison_boundaries",
    "boundary_sensitivity_available",
    "exact_boundary_sensitivity_incomplete_outers",
    "e4_boundary_sensitivity_incomplete_targets",
)


def _normalize_pre_merton_contract(
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, str]],
    e4_rows: Sequence[Mapping[str, str]],
    source: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    bool,
]:
    """Normalize the unambiguous Jul-24 Liu schema in memory.

    Raw files and their manifest are validated before this helper is called.
    The old driver had one fixed rule: full Cartesian variants, no paper
    checkpoint exclusion, and no Merton-style required-evidence fields.
    """

    missing_merton_config = {
        field for field in _MERTON_CONFIG_FIELDS if field not in config
    }
    missing_merton_status = {
        field for field in _MERTON_STATUS_FIELDS if field not in status
    }
    missing_boundary_config = {
        field for field in _BOUNDARY_CONFIG_FIELDS if field not in config
    }
    missing_boundary_status = {
        field for field in _BOUNDARY_STATUS_FIELDS if field not in status
    }
    merton_complete = not missing_merton_config and not missing_merton_status
    pre_merton = (
        missing_merton_config == set(_MERTON_CONFIG_FIELDS)
        and missing_merton_status == set(_MERTON_STATUS_FIELDS)
    )
    boundary_complete = (
        not missing_boundary_config and not missing_boundary_status
    )
    boundary_absent = (
        missing_boundary_config == set(_BOUNDARY_CONFIG_FIELDS)
        and missing_boundary_status == set(_BOUNDARY_STATUS_FIELDS)
    )
    if merton_complete and (boundary_complete or boundary_absent):
        return (
            dict(config),
            [dict(row) for row in exact_rows],
            [dict(row) for row in e4_rows],
            False,
        )
    if not pre_merton or not boundary_absent:
        raise ValueError(
            f"{source}: partial/unknown exact-map schema; missing config "
            f"Merton fields={sorted(missing_merton_config)}, missing status "
            f"Merton fields={sorted(missing_merton_status)}, missing config "
            f"boundary fields={sorted(missing_boundary_config)}, missing "
            f"status boundary fields={sorted(missing_boundary_status)}"
        )
    schedule_raw = config.get("checkpoint_schedule")
    if (
        not isinstance(schedule_raw, Sequence)
        or isinstance(schedule_raw, (str, bytes))
    ):
        raise ValueError(
            f"{source}: pre-Merton config is missing checkpoint_schedule"
        )
    try:
        schedule = [int(value) for value in schedule_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: invalid pre-Merton checkpoint_schedule"
        ) from exc
    if not schedule:
        raise ValueError(f"{source}: empty pre-Merton checkpoint_schedule")

    normalized = dict(config)
    normalized.update(
        {
            "refinement_rule": "cartesian",
            "min_paper_checkpoint": 0,
            "excluded_initial_checkpoints": [],
            "paper_checkpoint_schedule": schedule,
            "e4_refinement_rule": {
                "required_set": "initial_first_last_worst_e_approx_X",
                "worst_tie_break": "lowest_target_outer_iter",
                "variant_pass_fail": "cartesian",
                "interaction_failures": "included",
                "sensitivity_envelope": (
                    "historical_liu_max_cartesian_change"
                ),
            },
        }
    )

    def normalized_rows(
        rows: Sequence[Mapping[str, str]],
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["refinement_rule"] = "cartesian"
            item["min_paper_checkpoint"] = "0"
            output.append(item)
        return output

    return (
        normalized,
        normalized_rows(exact_rows),
        normalized_rows(e4_rows),
        True,
    )


def _canonical_domain_protocol(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    design = config.get("domain_design")
    if not isinstance(design, Mapping):
        raise ValueError("exact-map config is missing domain_design")
    fields = (
        "mode",
        "legacy_shared_shorthand",
        "wealth_domain_factors",
        "factor_domain_factors",
        "domain_pairs",
        "primary_wealth_domain_factor",
        "primary_factor_domain_factor",
    )
    missing = [field for field in fields if field not in design]
    if missing:
        raise ValueError(
            "exact-map domain_design is missing protocol fields: "
            f"{missing}"
        )
    parameterization = str(
        design.get(
            "wealth_domain_parameterization",
            "symmetric_log_half_width_factor",
        )
    )
    if parameterization not in {
        "symmetric_log_half_width_factor",
        "explicit_absolute_bounds",
    }:
        raise ValueError(
            "unsupported wealth_domain_parameterization="
            f"{parameterization!r}"
        )
    bounds = design.get("wealth_domain_bounds")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
        bounds = []
    normalized_bounds: List[Dict[str, float]] = []
    if bounds:
        try:
            normalized_bounds = [
                {
                    field: float(item[field])
                    for field in (
                        "wealth_domain_factor",
                        "fd_y_min",
                        "fd_y_max",
                        "fd_w_min",
                        "fd_w_max",
                    )
                }
                for item in bounds
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "invalid resolved wealth_domain_bounds"
            ) from exc
    elif parameterization == "symmetric_log_half_width_factor":
        collocation = config.get("collocation_bounds")
        if isinstance(collocation, Mapping):
            try:
                saved_y_min = float(collocation["y_min"])
                saved_y_max = float(collocation["y_max"])
                factors = [
                    float(value)
                    for value in design["wealth_domain_factors"]
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid saved collocation bounds for legacy wealth "
                    "domain reconstruction"
                ) from exc
            center = 0.5 * (saved_y_min + saved_y_max)
            half_width = 0.5 * (saved_y_max - saved_y_min)
            normalized_bounds = [
                {
                    "wealth_domain_factor": factor,
                    "fd_y_min": center - half_width * factor,
                    "fd_y_max": center + half_width * factor,
                    "fd_w_min": math.exp(
                        center - half_width * factor
                    ),
                    "fd_w_max": math.exp(
                        center + half_width * factor
                    ),
                }
                for factor in factors
            ]
    if parameterization == "explicit_absolute_bounds" and not normalized_bounds:
        raise ValueError(
            "explicit absolute wealth domains require resolved bounds"
        )
    return {
        **{field: design[field] for field in fields},
        "wealth_domain_parameterization": parameterization,
        "wealth_domain_bounds": normalized_bounds,
    }


def _canonical_grid_protocol(config: Mapping[str, Any]) -> Mapping[str, Any]:
    grid = config.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("exact-map config is missing grid")
    fields = (
        "base_ny",
        "base_nx",
        "base_nt",
        "eval_ny",
        "eval_nx",
        "grid_factors",
        "domain_mode",
        "boundaries",
        "verify_checkpoints",
        "drift_scheme",
        "peclet_limit",
        "theta_method",
        "startup_be_steps",
        "linear_residual_tolerance",
        "boundary_condition_limit",
    )
    missing = [field for field in fields if field not in grid]
    if missing:
        raise ValueError(
            f"exact-map grid is missing protocol fields: {missing}"
        )
    return {field: grid[field] for field in fields}


def _canonical_protocol(config: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    training = config.get("training_protocol_args")
    if not isinstance(training, Mapping):
        raise ValueError("exact-map config is missing training_protocol_args")
    # pres_target is the controlled E4 sweep variable.  Removing any other
    # training choice would allow a mathematically different cell to pool.
    canonical_training = {
        str(key): training[key]
        for key in sorted(training)
        if str(key) != "pres_target"
    }
    implementation = config.get("implementation_hashes")
    if not isinstance(implementation, Mapping):
        raise ValueError("exact-map config is missing implementation_hashes")
    relevant = {
        "training_protocol_args_without_pres_target": canonical_training,
        "numerical_core_sha256": implementation.get("core"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "checkpoint_schedule": config.get("checkpoint_schedule"),
        "refinement_rule": config.get("refinement_rule"),
        "min_paper_checkpoint": config.get("min_paper_checkpoint"),
        "paper_checkpoint_schedule": config.get("paper_checkpoint_schedule"),
        "normalized_e4_refinement_evidence": {
            "required_set": "initial_first_last_worst_e_approx_X",
            "worst_tie_break": "lowest_target_outer_iter",
            "variant_pass_fail": config.get("refinement_rule"),
            "interaction_failures": (
                "included"
                if config.get("refinement_rule") == "cartesian"
                else "excluded"
            ),
            "sensitivity_axes": REFINEMENT_SCOPE,
            "boundary_replacement": BOUNDARY_SENSITIVITY_ROLE,
        },
        "evaluation_window": config.get("evaluation_window"),
        "domain_design": _canonical_domain_protocol(config),
        "grid": _canonical_grid_protocol(config),
        "refinement_abs_tolerance": config.get("refinement_abs_tolerance"),
        "refinement_rel_tolerance": config.get("refinement_rel_tolerance"),
        "denominator_tolerance": config.get("denominator_tolerance"),
        "ellipticity_tolerance": config.get("ellipticity_tolerance"),
        "norm": config.get("norm"),
        "indexing": config.get("indexing"),
    }
    missing = [key for key, value in relevant.items() if value is None]
    if missing:
        raise ValueError(
            "exact-map config is missing mathematically relevant protocol fields: "
            f"{missing}"
        )
    return _stable_hash(relevant), relevant


def _achieved_pres(run_dir: Path, status: Mapping[str, Any]) -> float:
    raw = status.get("pres_max")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and float(raw) > 0:
        return float(raw)
    history = run_dir / "outer_history.csv"
    values: List[float] = []
    if history.is_file():
        with history.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    value = float(row.get("val_pres", ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    values.append(value)
    if not values:
        raise ValueError(f"{run_dir}: no positive achieved p_res in status/history")
    return max(values)


def _successful_training_status(
    source_config: Path,
) -> Tuple[Path, Mapping[str, Any], float]:
    run_dir = source_config.parent
    markers = [
        name
        for name in ("_SUCCESS", "_STOPPED_EARLY", "_FAILED")
        if (run_dir / name).is_file()
    ]
    if markers != ["_SUCCESS"]:
        raise ValueError(
            f"{run_dir}: E4 source must have unique current marker _SUCCESS; found {markers}"
        )
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise ValueError(f"{run_dir}: missing training status.json")
    status = read_json(status_path)
    if status.get("status") != "success":
        raise ValueError(
            f"{run_dir}: _SUCCESS/status.json disagreement: {status.get('status')!r}"
        )
    return run_dir, status, _achieved_pres(run_dir, status)


def _source_artifacts(
    directory: Path, config: Mapping[str, Any]
) -> Tuple[Path, Path]:
    source_config = _matching_artifact(
        [
            Path(str(config.get("config_path", ""))),
            Path(str(config.get("run_dir", ""))) / "config.json",
            directory.parent / "config.json",
        ],
        str(config.get("config_sha256", "")),
        "training config",
    )
    source_market = _matching_artifact(
        [
            Path(str(config.get("market_path", ""))),
            source_config.parent / "market_params.npz",
            directory.parent / "market_params.npz",
        ],
        str(config.get("market_file_sha256", "")),
        "market snapshot",
    )
    return source_config, source_market


def _attempt_time(directory: Path) -> int:
    return max(
        (directory / name).stat().st_mtime_ns
        for name in (STATUS_INPUT, CONFIG_INPUT)
        if (directory / name).exists()
    )


def _reassess_e4_numerical_refinement(
    config: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, str]],
    refinement_rows: Sequence[Mapping[str, str]],
    source: Path,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Reassess legacy rows after removing boundary replacement from the gate.

    The raw refinement CSV already contains every value required for this
    operation, so completed legacy FD runs do not need to be rerun.
    """

    grid = config.get("grid")
    design = config.get("domain_design")
    if not isinstance(grid, Mapping) or not isinstance(design, Mapping):
        raise ValueError(f"{source}: missing grid/domain refinement design")
    try:
        grid_factors = sorted(
            set(int(value) for value in grid.get("grid_factors", []))
        )
        domain_pairs = [
            (
                float(item["wealth_domain_factor"]),
                float(item["factor_domain_factor"]),
            )
            for item in design.get("domain_pairs", [])
        ]
        largest_wealth = float(
            design["primary_wealth_domain_factor"]
        )
        largest_factor = float(
            design["primary_factor_domain_factor"]
        )
        boundaries = [str(value) for value in grid.get("boundaries", [])]
        abs_tolerance = float(config["refinement_abs_tolerance"])
        rel_tolerance = float(config["refinement_rel_tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: invalid numerical refinement design"
        ) from exc
    if not grid_factors or not domain_pairs or not boundaries:
        raise ValueError(f"{source}: empty numerical refinement design")

    # Very small synthetic/legacy audits with no refinement direction cannot
    # be reclassified from their raw variants. Preserve their recorded
    # status; real paper audits contain at least two grid or domain levels.
    can_reassess = len(grid_factors) > 1 or len(set(domain_pairs)) > 1
    if not can_reassess:
        return [dict(row) for row in primary_rows], False

    reassessed_refinement = [dict(row) for row in refinement_rows]
    _assess(
        reassessed_refinement,
        key_name="target_outer_iter",
        value_name="e_approx_X",
        finest=max(grid_factors),
        largest_wealth_domain=largest_wealth,
        largest_factor_domain=largest_factor,
        primary_boundary=boundaries[0],
        grid_factors=grid_factors,
        domain_mode=str(design.get("mode", grid.get("domain_mode", ""))),
        domain_pairs=domain_pairs,
        boundaries=boundaries,
        envelope_name="approx_sensitivity_envelope",
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        refinement_rule=str(config.get("refinement_rule", "")),
    )
    assessed_by_target = {
        _integer(row, "target_outer_iter", source): row
        for row in reassessed_refinement
        if _integer(row, "is_primary", source) == 1
    }
    output: List[Dict[str, Any]] = []
    copied_fields = (
        "grid_abs_change",
        "grid_rel_change",
        "domain_abs_change",
        "domain_rel_change",
        "wealth_domain_abs_change",
        "wealth_domain_rel_change",
        "factor_domain_abs_change",
        "factor_domain_rel_change",
        "refinement_tolerance",
        "numerical_abs_change",
        "numerical_tolerance_ratio",
        "boundary_abs_change",
        "boundary_rel_change",
        "boundary_tolerance_ratio",
        "boundary_sensitivity_status",
        "approx_sensitivity_envelope",
        "refinement_status",
    )
    for raw in primary_rows:
        target = _integer(raw, "target_outer_iter", source)
        assessed = assessed_by_target.get(target)
        if assessed is None:
            raise ValueError(
                f"{source}: missing reassessed primary target {target}"
            )
        recorded_value = _number(raw, "e_approx_X", source)
        assessed_value = _number(assessed, "e_approx_X", source)
        if not math.isclose(
            recorded_value,
            assessed_value,
            rel_tol=1e-12,
            abs_tol=1e-13,
        ):
            raise ValueError(
                f"{source}: primary E4 value changed during reassessment at "
                f"target={target}"
            )
        row = dict(raw)
        row.update({field: assessed.get(field, "") for field in copied_fields})
        output.append(row)
    return output, True


def _validate_candidate(
    directory: Path,
    *,
    refinement_failure_mode: str = "error",
) -> Dict[str, Any]:
    status = read_json(directory / STATUS_INPUT)
    markers = [
        name
        for name in ("_SUCCESS_EXACT_MAP", "_FAILED_EXACT_MAP")
        if (directory / name).is_file()
    ]
    # A failed exact-map directory is never filtered out before attempt
    # selection.  Failing here prevents an older successful directory from
    # being silently revived when both are discoverable under the out-root.
    if status.get("status") != "success" or markers != ["_SUCCESS_EXACT_MAP"]:
        raise ValueError(
            f"exact-map attempt is not successful: {directory} "
            f"(status={status.get('status')!r}, markers={markers})"
        )
    _validate_artifact_hashes(directory, status)
    raw_config = read_json(directory / CONFIG_INPUT)
    raw_exact_rows = read_csv(directory / EXACT_INPUT)
    raw_e4_rows = read_csv(directory / E4_INPUT)
    exact_refinement = read_csv(directory / "exact_map_refinement.csv")
    e4_refinement = read_csv(directory / "e4_approximation_refinement.csv")
    (
        config,
        exact_rows,
        e4_rows,
        legacy_pre_merton,
    ) = _normalize_pre_merton_contract(
        raw_config,
        status,
        raw_exact_rows,
        raw_e4_rows,
        directory,
    )
    grid_config = config.get("grid")
    if (
        not isinstance(grid_config, Mapping)
        or str(grid_config.get("verify_checkpoints", "")) != "all"
    ):
        raise ValueError(
            f"{directory}: Figure S1 aggregation requires "
            "grid.verify_checkpoints='all'; subset verification is "
            "pilot-only"
        )
    _validate_status_contract(
        directory,
        status,
        config,
        exact_rows,
        e4_rows,
        exact_refinement,
        e4_refinement,
        allow_legacy_pre_merton=legacy_pre_merton,
        require_exact_ellipticity=False,
    )
    _validate_provenance(
        directory,
        config,
        exact_rows,
        e4_rows,
        allow_historical_driver=True,
    )
    if any(_integer(row, "is_primary", directory / E4_INPUT) != 1 for row in e4_rows):
        raise ValueError(f"{directory / E4_INPUT}: contains non-primary rows")
    recorded_e4_evidence = _e4_paper_evidence(
        config, e4_rows, directory / E4_INPUT
    )
    numerical_e4_rows, reassessed = _reassess_e4_numerical_refinement(
        config,
        e4_rows,
        e4_refinement,
        directory / "e4_approximation_refinement.csv",
    )
    e4_evidence = _e4_paper_evidence(
        config, numerical_e4_rows, directory / E4_INPUT
    )
    if (
        config.get("refinement_scope") is not None
        and reassessed
        and [
            str(row.get("refinement_status", ""))
            for row in numerical_e4_rows
        ]
        != [
            str(row.get("refinement_status", ""))
            for row in e4_rows
        ]
    ):
        raise ValueError(
            f"{directory}: recorded primary refinement statuses disagree "
            "with grid/domain-only reassessment"
        )
    if (
        e4_evidence["evidence_status"] != "pass"
        and refinement_failure_mode == "error"
    ):
        raise ValueError(
            f"{directory}: required grid/domain E4 refinement evidence "
            f"did not pass: {e4_evidence['required_statuses']}"
        )

    seed = int(_identity(e4_rows, "seed", directory / E4_INPUT))
    if any(_integer(row, "seed", directory / E4_INPUT) != seed for row in e4_rows):
        raise ValueError(f"{directory / E4_INPUT}: mixes training seeds")
    full_schedule = list(e4_evidence["full_schedule"])
    schedule = list(e4_evidence["paper_schedule"])
    paper_set = set(schedule)
    paper_e4_rows = [
        row for row in numerical_e4_rows
        if _integer(
            row, "target_outer_iter", directory / E4_INPUT
        ) in paper_set
    ]

    training = config.get("training_protocol_args")
    if not isinstance(training, Mapping):
        raise ValueError(f"{directory}: missing training_protocol_args")
    target = _positive_target(
        training.get("pres_target"), directory / CONFIG_INPUT
    )

    protocol_hash, protocol_payload = _canonical_protocol(config)
    implementation = config.get("implementation_hashes")
    if not isinstance(implementation, Mapping):
        raise ValueError(
            f"{directory}: missing implementation_hashes"
        )
    source_config, source_market = _source_artifacts(directory, config)
    run_dir, training_status, achieved = _successful_training_status(source_config)
    market_hash = _canonical_market_hash(source_market)
    row_market = _identity(e4_rows, "market_sha256", directory / E4_INPUT)
    exact_market = _identity(exact_rows, "market_sha256", directory / EXACT_INPUT)
    config_market = str(config.get("market_sha256", ""))
    if not config_market or len({market_hash, row_market, exact_market, config_market}) != 1:
        raise ValueError(
            f"{directory}: canonical market hash mismatch among file/config/CSVs"
        )
    return {
        "directory": directory,
        "attempt_time_ns": _attempt_time(directory),
        "seed": seed,
        "pres_target": target,
        "achieved_pres": achieved,
        "market_sha256": market_hash,
        "protocol_sha256": protocol_hash,
        "protocol_payload": protocol_payload,
        "full_schedule": full_schedule,
        "schedule": schedule,
        "e4_rows": paper_e4_rows,
        "e4_refinement_rows": e4_refinement,
        "grid": dict(config.get("grid", {})),
        "domain_design": dict(config.get("domain_design", {})),
        "refinement_abs_tolerance": float(
            config["refinement_abs_tolerance"]
        ),
        "refinement_rel_tolerance": float(
            config["refinement_rel_tolerance"]
        ),
        "all_e4_refinement_pass": all(
            row.get("refinement_status") == "pass"
            for row in numerical_e4_rows
        ),
        "e4_refinement_required_iterations": e4_evidence[
            "required_iterations"
        ],
        "e4_refinement_evidence_status": e4_evidence["evidence_status"],
        "recorded_e4_refinement_evidence_status": recorded_e4_evidence[
            "evidence_status"
        ],
        "boundary_separated_reassessment_applied": reassessed,
        "legacy_pre_merton_schema": legacy_pre_merton,
        "source_driver_sha256": str(implementation.get("driver", "")),
        "source_core_sha256": str(implementation.get("core", "")),
        "refinement_scope": (
            str(config.get("refinement_scope"))
            if config.get("refinement_scope") is not None
            else "legacy_cartesian_including_boundary"
        ),
        "min_paper_checkpoint": e4_evidence["min_paper_checkpoint"],
        "refinement_rule": e4_evidence["refinement_rule"],
        "run_dir": run_dir,
        "training_status_updated_at": training_status.get("updated_at", ""),
    }


def _arg_extreme(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    mode: str,
    source: Path,
) -> Tuple[float, int]:
    values: List[Tuple[float, int]] = []
    for row in rows:
        value = _number(row, field, source, allow_blank=True)
        if not math.isfinite(value):
            raise ValueError(f"{source}: missing/nonfinite {field}; refusing row deletion")
        values.append((value, _integer(row, "target_outer_iter", source)))
    if not values:
        raise ValueError(f"{source}: no requested E4 rows")
    if mode == "max":
        # Lowest outer deterministically wins an exact tie.
        value = max(item[0] for item in values)
    elif mode == "min":
        value = min(item[0] for item in values)
    else:  # pragma: no cover - internal programming error
        raise ValueError(mode)
    outer = min(item[1] for item in values if item[0] == value)
    return value, outer


def _available_optional_metrics(records: Sequence[Mapping[str, Any]]) -> List[str]:
    available: List[str] = []
    for metric in OPTIONAL_ERROR_METRICS:
        values = [
            _number(row, metric, Path(record["directory"]) / E4_INPUT, allow_blank=True)
            for record in records
            for row in record["e4_rows"]
        ]
        if values and all(math.isfinite(value) for value in values):
            available.append(metric)
        elif any(math.isfinite(value) for value in values):
            raise ValueError(f"optional E4 metric {metric} is only partly available")
    return available


def _per_seed_rows(
    records: Sequence[Mapping[str, Any]], checkpoints: Optional[Sequence[int]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    optional = _available_optional_metrics(records)
    error_metrics = [*PRIMARY_ERROR_METRICS, *optional]
    output: List[Dict[str, Any]] = []
    for record in records:
        schedule = list(record["schedule"])
        selected = schedule if checkpoints is None else list(checkpoints)
        missing = sorted(set(selected) - set(schedule))
        if missing:
            raise ValueError(
                f"{record['directory']}: requested checkpoints absent from E4 schedule: {missing}"
            )
        row_by_outer = {
            _integer(row, "target_outer_iter", Path(record["directory"]) / E4_INPUT): row
            for row in record["e4_rows"]
        }
        rows = [row_by_outer[outer] for outer in selected]
        required_rows = [
            row_by_outer[outer]
            for outer in record["e4_refinement_required_iterations"]
        ]
        item: Dict[str, Any] = {
            "protocol_sha256": record["protocol_sha256"],
            "market_sha256": record["market_sha256"],
            "seed": record["seed"],
            "pres_target": record["pres_target"],
            "achieved_pres": record["achieved_pres"],
            "checkpoint_schedule": ",".join(str(value) for value in schedule),
            "requested_checkpoints": ",".join(str(value) for value in selected),
            "n_checkpoints": len(selected),
            "all_e4_refinement_pass": int(
                bool(record["all_e4_refinement_pass"])
            ),
            "e4_refinement_evidence_pass": int(
                record["e4_refinement_evidence_status"] == "pass"
            ),
            "e4_refinement_evidence_status": record[
                "e4_refinement_evidence_status"
            ],
            "recorded_e4_refinement_evidence_status": record[
                "recorded_e4_refinement_evidence_status"
            ],
            "boundary_separated_reassessment_applied": int(
                bool(record["boundary_separated_reassessment_applied"])
            ),
            "e4_refinement_required_iterations": ",".join(
                str(value)
                for value in record["e4_refinement_required_iterations"]
            ),
            "min_paper_checkpoint": record["min_paper_checkpoint"],
            "refinement_rule": record["refinement_rule"],
            "all_source_policies_elliptic": 1,
            "result_dir": str(record["directory"]),
            "run_dir": str(record["run_dir"]),
        }
        source = Path(record["directory"]) / E4_INPUT
        for metric in error_metrics:
            value, outer = _arg_extreme(rows, metric, mode="max", source=source)
            name = f"max_{metric}"
            item[name] = value
            item[f"{name}_outer"] = outer
        for name, field, mode in DIAGNOSTIC_SPECS:
            value, outer = _arg_extreme(rows, field, mode=mode, source=source)
            item[name] = value
            item[f"{name}_outer"] = outer
        numerical_ratios = [
            _number(
                row,
                "numerical_tolerance_ratio",
                source,
                allow_blank=True,
            )
            for row in rows
        ]
        if numerical_ratios and all(
            math.isfinite(value) for value in numerical_ratios
        ):
            value = max(numerical_ratios)
            item["max_numerical_tolerance_ratio"] = value
            item["max_numerical_tolerance_ratio_outer"] = min(
                _integer(row, "target_outer_iter", source)
                for row, candidate in zip(rows, numerical_ratios)
                if candidate == value
            )
        else:
            item["max_numerical_tolerance_ratio"] = ""
            item["max_numerical_tolerance_ratio_outer"] = ""
        required_numerical_ratios = [
            _number(
                row,
                "numerical_tolerance_ratio",
                source,
                allow_blank=True,
            )
            for row in required_rows
        ]
        if required_numerical_ratios and all(
            math.isfinite(value) for value in required_numerical_ratios
        ):
            value = max(required_numerical_ratios)
            item["max_required_numerical_tolerance_ratio"] = value
            item["max_required_numerical_tolerance_ratio_outer"] = min(
                _integer(row, "target_outer_iter", source)
                for row, candidate in zip(
                    required_rows, required_numerical_ratios
                )
                if candidate == value
            )
        else:
            item["max_required_numerical_tolerance_ratio"] = ""
            item["max_required_numerical_tolerance_ratio_outer"] = ""
        output.append(item)
    output.sort(key=lambda row: (float(row["pres_target"]), int(row["seed"])))
    return output, error_metrics


def _summary_rows(
    per_seed: Sequence[Mapping[str, Any]], error_metrics: Sequence[str]
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[_float_key(float(row["pres_target"]))].append(row)
    summary_metrics = ["achieved_pres"]
    summary_metrics.extend(f"max_{metric}" for metric in error_metrics)
    summary_metrics.extend(name for name, _field, _mode in DIAGNOSTIC_SPECS)
    output: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: float.fromhex(item)):
        rows = sorted(grouped[key], key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in rows]
        for metric in summary_metrics:
            stats = _stats([float(row[metric]) for row in rows])
            output.append(
                {
                    "pres_target": float(rows[0]["pres_target"]),
                    "metric": metric,
                    **stats,
                    "seeds": ",".join(str(seed) for seed in seeds),
                }
            )
    return output


def _loglog_fit(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> Tuple[Any, Any, Any]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x))
        and math.isfinite(float(y))
        and float(x) > 0.0
        and float(y) > 0.0
    ]
    if len(pairs) < 2 or len({float(x).hex() for x, _y in pairs}) < 2:
        return "", "", ""
    log_x = np.log(np.asarray([x for x, _y in pairs], dtype=float))
    log_y = np.log(np.asarray([y for _x, y in pairs], dtype=float))
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fitted = intercept + slope * log_x
    residual = float(np.sum((log_y - fitted) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return float(slope), float(intercept), float(r_squared)


def _boundary_sensitivity_rows(
    records: Sequence[Mapping[str, Any]],
    checkpoints: Optional[Sequence[int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build direct boundary-BVP sensitivity evidence.

    Each boundary is evaluated at the same finest grid and largest domain.
    Boundary replacement never changes numerical refinement pass/fail.
    """

    per_checkpoint: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for record in records:
        grid = record["grid"]
        design = record["domain_design"]
        boundaries = [str(value) for value in grid.get("boundaries", [])]
        if not boundaries:
            raise ValueError(
                f"{record['directory']}: boundary list is empty"
            )
        primary_boundary = boundaries[0]
        if primary_boundary != "linearity":
            raise ValueError(
                f"{record['directory']}: Figure S1 requires linearity as "
                f"the primary boundary; got {primary_boundary!r}"
            )
        finest = max(int(value) for value in grid.get("grid_factors", []))
        primary_wealth = float(
            design["primary_wealth_domain_factor"]
        )
        primary_factor = float(
            design["primary_factor_domain_factor"]
        )
        schedule = list(record["schedule"])
        selected = schedule if checkpoints is None else list(checkpoints)
        source = Path(record["directory"]) / "e4_approximation_refinement.csv"
        lookup: Dict[Tuple[int, str], Mapping[str, Any]] = {}
        for row in record["e4_refinement_rows"]:
            if _integer(row, "grid_factor", source) != finest:
                continue
            if not math.isclose(
                _number(row, "wealth_domain_factor", source),
                primary_wealth,
                rel_tol=1e-12,
                abs_tol=1e-13,
            ):
                continue
            if not math.isclose(
                _number(row, "factor_domain_factor", source),
                primary_factor,
                rel_tol=1e-12,
                abs_tol=1e-13,
            ):
                continue
            key = (
                _integer(row, "target_outer_iter", source),
                str(row.get("boundary", "")),
            )
            if key in lookup:
                raise ValueError(
                    f"{source}: duplicate finest/largest boundary row {key}"
                )
            lookup[key] = row
        required = set(record["e4_refinement_required_iterations"])
        by_boundary: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for target in selected:
            primary = lookup.get((target, primary_boundary))
            if primary is None:
                raise ValueError(
                    f"{source}: missing primary boundary row at target={target}"
                )
            primary_x = _number(primary, "e_approx_X", source)
            former_tolerance = (
                float(record["refinement_abs_tolerance"])
                + float(record["refinement_rel_tolerance"])
                * abs(primary_x)
            )
            for boundary in boundaries:
                candidate = lookup.get((target, boundary))
                if candidate is None:
                    raise ValueError(
                        f"{source}: missing boundary={boundary!r} at "
                        f"target={target}"
                    )
                value = _number(candidate, "e_approx_X", source)
                value_component = _number(
                    candidate, "e_approx_value", source
                )
                bundle_component = _number(
                    candidate, "e_approx_bundle", source
                )
                abs_difference = abs(value - primary_x)
                relative_difference = abs_difference / max(
                    abs(primary_x), 1e-300
                )
                tolerance_ratio = (
                    abs_difference / former_tolerance
                    if former_tolerance > 0.0 else float("inf")
                )
                item = {
                    "protocol_sha256": record["protocol_sha256"],
                    "market_sha256": record["market_sha256"],
                    "pres_target": record["pres_target"],
                    "achieved_pres": record["achieved_pres"],
                    "seed": record["seed"],
                    "target_outer_iter": target,
                    "is_required_iteration": int(target in required),
                    "boundary": boundary,
                    "is_primary_boundary": int(
                        boundary == primary_boundary
                    ),
                    "e_approx_value": value_component,
                    "e_approx_bundle": bundle_component,
                    "e_approx_X": value,
                    "primary_e_approx_X": primary_x,
                    "abs_difference_from_primary": abs_difference,
                    "relative_difference_from_primary": relative_difference,
                    "former_refinement_tolerance": former_tolerance,
                    "former_boundary_tolerance_ratio": tolerance_ratio,
                    "result_dir": str(record["directory"]),
                }
                per_checkpoint.append(item)
                by_boundary[boundary].append(item)
        for boundary in boundaries:
            rows = by_boundary[boundary]
            maximum = max(float(row["e_approx_X"]) for row in rows)
            max_outer = min(
                int(row["target_outer_iter"])
                for row in rows
                if float(row["e_approx_X"]) == maximum
            )
            max_difference = max(
                float(row["relative_difference_from_primary"])
                for row in rows
            )
            difference_outer = min(
                int(row["target_outer_iter"])
                for row in rows
                if float(row["relative_difference_from_primary"])
                == max_difference
            )
            per_seed.append(
                {
                    "protocol_sha256": record["protocol_sha256"],
                    "market_sha256": record["market_sha256"],
                    "pres_target": record["pres_target"],
                    "achieved_pres": record["achieved_pres"],
                    "seed": record["seed"],
                    "boundary": boundary,
                    "is_primary_boundary": int(
                        boundary == primary_boundary
                    ),
                    "n_checkpoints": len(rows),
                    "max_e_approx_X": maximum,
                    "max_e_approx_X_outer": max_outer,
                    "C_num": maximum / float(record["achieved_pres"]),
                    "max_relative_difference_from_primary": max_difference,
                    "max_relative_difference_from_primary_outer": (
                        difference_outer
                    ),
                    "e4_refinement_evidence_status": record[
                        "e4_refinement_evidence_status"
                    ],
                    "result_dir": str(record["directory"]),
                }
            )

    per_checkpoint.sort(
        key=lambda row: (
            float(row["pres_target"]),
            int(row["seed"]),
            int(row["target_outer_iter"]),
            str(row["boundary"]),
        )
    )
    per_seed.sort(
        key=lambda row: (
            str(row["boundary"]),
            float(row["pres_target"]),
            int(row["seed"]),
        )
    )
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["boundary"])].append(row)
    primary_rows = grouped.get("linearity", [])
    primary_lookup = {
        (_float_key(float(row["pres_target"])), int(row["seed"])): row
        for row in primary_rows
    }
    if len(
        {_float_key(float(row["pres_target"])) for row in primary_rows}
    ) >= 2:
        primary_slope, _primary_intercept, _primary_r2 = _loglog_fit(
            [float(row["achieved_pres"]) for row in primary_rows],
            [float(row["max_e_approx_X"]) for row in primary_rows],
        )
    else:
        primary_slope, _primary_intercept, _primary_r2 = "", "", ""
    primary_c_num = (
        max(float(row["C_num"]) for row in primary_rows)
        if primary_rows else float("nan")
    )
    summary: List[Dict[str, Any]] = []
    for boundary in grouped:
        rows = grouped[boundary]
        if len(
            {_float_key(float(row["pres_target"])) for row in rows}
        ) >= 2:
            slope, intercept, r_squared = _loglog_fit(
                [float(row["achieved_pres"]) for row in rows],
                [float(row["max_e_approx_X"]) for row in rows],
            )
        else:
            slope, intercept, r_squared = "", "", ""
        worst_c_num = max(float(row["C_num"]) for row in rows)
        worst_c_row = min(
            (
                row
                for row in rows
                if float(row["C_num"]) == worst_c_num
            ),
            key=lambda row: (
                float(row["pres_target"]),
                int(row["seed"]),
            ),
        )
        matched_differences: List[Tuple[float, Mapping[str, Any]]] = []
        for row in rows:
            primary = primary_lookup[
                (
                    _float_key(float(row["pres_target"])),
                    int(row["seed"]),
                )
            ]
            difference = abs(
                float(row["max_e_approx_X"])
                - float(primary["max_e_approx_X"])
            ) / max(abs(float(primary["max_e_approx_X"])), 1e-300)
            matched_differences.append((difference, row))
        worst_difference = max(value for value, _row in matched_differences)
        worst_difference_row = min(
            (
                row
                for value, row in matched_differences
                if value == worst_difference
            ),
            key=lambda row: (
                float(row["pres_target"]),
                int(row["seed"]),
            ),
        )
        summary.append(
            {
                "boundary": boundary,
                "is_primary_boundary": int(boundary == "linearity"),
                "n_cells": len(rows),
                "n_tolerances": len(
                    {_float_key(float(row["pres_target"])) for row in rows}
                ),
                "n_seeds": len({int(row["seed"]) for row in rows}),
                "loglog_slope": slope,
                "loglog_intercept": intercept,
                "loglog_r_squared": r_squared,
                "slope_difference_from_primary": (
                    ""
                    if slope == "" or primary_slope == ""
                    else float(slope) - float(primary_slope)
                ),
                "C_num": worst_c_num,
                "C_num_relative_difference_from_primary": (
                    0.0
                    if boundary == "linearity"
                    else abs(worst_c_num - primary_c_num)
                    / max(abs(primary_c_num), 1e-300)
                ),
                "C_num_pres_target": worst_c_row["pres_target"],
                "C_num_seed": worst_c_row["seed"],
                "max_relative_difference_from_primary": worst_difference,
                "max_relative_difference_pres_target": (
                    worst_difference_row["pres_target"]
                ),
                "max_relative_difference_seed": worst_difference_row["seed"],
            }
        )
    summary.sort(key=lambda row: (not bool(row["is_primary_boundary"]), row["boundary"]))
    return per_checkpoint, per_seed, summary


def _parse_formats(text: str) -> List[str]:
    values: List[str] = []
    for token in re.split(r"[\s,]+", str(text).strip().lower()):
        if not token:
            continue
        if token not in PLOT_FORMATS:
            raise ValueError(f"unsupported plot format {token!r}; choose from {PLOT_FORMATS}")
        if token not in values:
            values.append(token)
    if not values:
        raise ValueError("--formats must contain at least one format")
    return values


def _parse_plot_metric(text: str) -> str:
    tokens = [
        token.strip()
        for token in re.split(r"[\s,]+", str(text))
        if token.strip()
    ]
    if len(tokens) != 1:
        raise ValueError(
            "the paper-style E4 plot requires exactly one --plot-metric"
        )
    requested = tokens[0].lower()
    for canonical, spec in PLOT_METRIC_SPECS.items():
        accepted = {
            canonical.lower(),
            *(str(value).lower() for value in spec["aliases"]),
        }
        if requested in accepted:
            return canonical
    raise ValueError(
        f"unsupported --plot-metric={tokens[0]!r}; choose one of "
        f"{sorted(PLOT_METRIC_SPECS)} or a documented short alias"
    )


def _parse_pair(text: str, option: str) -> Tuple[float, float]:
    values = [float(token.strip()) for token in str(text).split(",") if token.strip()]
    if len(values) != 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{option} requires two positive finite values")
    return values[0], values[1]


def _main_plot_series(
    summary: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> Dict[str, Any]:
    by_metric: Dict[str, Dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary:
        by_metric[str(row["metric"])][float(row["pres_target"])] = row
    x_rows = by_metric.get("achieved_pres", {})
    if not x_rows:
        raise ValueError("cannot plot without achieved_pres summaries")
    if metric not in by_metric:
        raise ValueError(f"requested plot metric is unavailable: {metric}")
    tolerances = sorted(set(x_rows) & set(by_metric[metric]))
    if not tolerances:
        raise ValueError(f"no target summaries are available for {metric}")
    x = np.asarray(
        [float(x_rows[value]["mean"]) for value in tolerances],
        dtype=float,
    )
    y = np.asarray(
        [float(by_metric[metric][value]["mean"]) for value in tolerances],
        dtype=float,
    )
    yerr = np.asarray(
        [float(by_metric[metric][value]["std"]) for value in tolerances],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(x <= 0.0)
        or np.any(y <= 0.0)
    ):
        raise ValueError(
            "paper-style E4 plotting requires positive finite target means"
        )
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]
    ordered_tolerances = [tolerances[int(index)] for index in order]

    ratios: List[float] = []
    for row in per_seed:
        try:
            achieved = float(row["achieved_pres"])
            value = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"per-seed plot data are missing {metric}"
            ) from exc
        if (
            not math.isfinite(achieved)
            or not math.isfinite(value)
            or achieved <= 0.0
            or value <= 0.0
        ):
            raise ValueError(
                "upper-envelope calculation requires positive finite "
                f"achieved_pres and {metric}"
            )
        ratios.append(value / achieved)
    if not ratios:
        raise ValueError("upper-envelope calculation has no per-seed rows")
    failed_tolerances = {
        _float_key(float(row["pres_target"]))
        for row in per_seed
        if int(row["e4_refinement_evidence_pass"]) != 1
    }
    return {
        "metric": metric,
        "tolerances": ordered_tolerances,
        "x": x,
        "y": y,
        "yerr": yerr,
        "C_num_empirical_upper": float(max(ratios)),
        "failed_positions": [
            position
            for position, tolerance in enumerate(ordered_tolerances)
            if _float_key(tolerance) in failed_tolerances
        ],
    }


def _selected_x_ticks(
    values: Sequence[float],
    count: int,
) -> np.ndarray:
    if count < 0:
        raise ValueError("x_tick_count must be nonnegative")
    unique = np.unique(np.asarray(values, dtype=float))
    unique = unique[np.isfinite(unique) & (unique > 0.0)]
    if count > 0 and count < unique.size:
        indices = np.rint(
            np.linspace(0, unique.size - 1, count)
        ).astype(int)
        return unique[np.unique(indices)]
    return unique


def _format_fixed_tick(value: float, decimal_places: int) -> str:
    text = f"{float(value):.{int(decimal_places)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _format_x_tick_labels(
    values: Sequence[float],
    nominal_targets: Sequence[float],
) -> List[str]:
    values_array = np.asarray(values, dtype=float)
    targets_array = np.asarray(nominal_targets, dtype=float)
    if values_array.shape != targets_array.shape:
        raise ValueError(
            "x tick values and nominal targets must have matching shapes"
        )
    if (
        np.any(~np.isfinite(values_array))
        or np.any(values_array <= 0.0)
        or np.any(~np.isfinite(targets_array))
        or np.any(targets_array <= 0.0)
    ):
        raise ValueError(
            "x tick values and nominal targets must be positive and finite"
        )

    decimal_places = [3 for _value in values_array]
    labels = [
        _format_fixed_tick(float(value), 3)
        for value in values_array
    ]
    for index, value in enumerate(values_array):
        while labels[index] == "0" and decimal_places[index] < 15:
            decimal_places[index] += 1
            labels[index] = _format_fixed_tick(
                float(value),
                decimal_places[index],
            )

    # Three decimal places are the paper-display rule. Add precision only for
    # a genuine collision so distinct positive ticks never share one label.
    for _ in range(16):
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            groups[label].append(index)
        collisions = [indices for indices in groups.values() if len(indices) > 1]
        if not collisions:
            return labels
        changed = False
        for indices in collisions:
            keeper = min(
                indices,
                key=lambda index: (
                    abs(float(labels[index]) - float(values_array[index]))
                    / float(values_array[index]),
                    index,
                ),
            )
            for index in indices:
                if index == keeper:
                    continue
                if decimal_places[index] < 15:
                    decimal_places[index] += 1
                    labels[index] = _format_fixed_tick(
                        float(values_array[index]),
                        decimal_places[index],
                    )
                    changed = True
                else:
                    labels[index] = f"{float(values_array[index]):.17g}"
        if not changed and len(set(labels)) != len(labels):
            raise ValueError("could not create unique x tick labels")
    raise ValueError("could not create unique x tick labels")


def _selected_x_tick_data(
    values: Sequence[float],
    nominal_targets: Sequence[float],
    count: int,
) -> Tuple[np.ndarray, List[str]]:
    values_array = np.asarray(values, dtype=float)
    targets = list(nominal_targets)
    if values_array.ndim != 1 or len(targets) != values_array.size:
        raise ValueError(
            "plot x values and nominal targets must be aligned vectors"
        )
    ticks = _selected_x_ticks(values_array, count)
    tick_targets: List[float] = []
    for tick in ticks:
        matches = np.flatnonzero(values_array == tick)
        if matches.size == 0:
            raise ValueError("selected x tick is missing its nominal target")
        tick_targets.append(float(targets[int(matches[0])]))
    return ticks, _format_x_tick_labels(ticks, tick_targets)


def _plot(
    summary: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    stage: Path,
    *,
    metrics: Sequence[str],
    formats: Sequence[str],
    figure_size: Tuple[float, float],
    dpi: int,
    font_size: float,
    x_tick_count: int = 0,
    mark_refinement_issues: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    if len(metrics) != 1:
        raise ValueError(
            "the paper-style E4 plot requires exactly one selected metric"
        )
    metric = str(metrics[0])
    if metric not in PLOT_METRIC_SPECS:
        raise ValueError(f"unsupported plot metric: {metric}")
    series = _main_plot_series(summary, per_seed, metric=metric)
    x = np.asarray(series["x"], dtype=float)
    y = np.asarray(series["y"], dtype=float)
    yerr = np.asarray(series["yerr"], dtype=float)
    color = "#0072B2"
    rc = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figure_size)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            color=color,
            linewidth=1.8,
            capsize=2.5,
        )
        c_num = float(series["C_num_empirical_upper"])
        xx = (
            np.geomspace(float(x.min()), float(x.max()), 100)
            if x.size > 1
            else np.asarray([float(x[0]), float(x[0])])
        )
        ax.plot(
            xx,
            c_num * xx,
            linestyle="--",
            color=color,
            linewidth=1.0,
        )
        if mark_refinement_issues and series["failed_positions"]:
            failed = list(series["failed_positions"])
            ax.scatter(
                x[failed],
                y[failed],
                marker="x",
                s=45,
                linewidths=1.5,
                color="crimson",
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\mathrm{res}}$")
        ax.set_ylabel(str(PLOT_METRIC_SPECS[metric]["y_label"]))
        ax.grid(True, which="both", alpha=0.25)

        ticks, tick_labels = _selected_x_tick_data(
            x,
            series["tolerances"],
            x_tick_count,
        )
        if ticks.size:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
            ax.xaxis.set_minor_formatter(NullFormatter())

        fig.tight_layout()
        for suffix in formats:
            fig.savefig(stage / f"{PLOT_STEM}.{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _plot_boundary_sensitivity(
    per_seed: Sequence[Mapping[str, Any]],
    boundary_summary: Sequence[Mapping[str, Any]],
    stage: Path,
    *,
    formats: Sequence[str],
    figure_size: Tuple[float, float],
    dpi: int,
    font_size: float,
) -> None:
    import matplotlib.pyplot as plt

    by_boundary: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in per_seed:
        by_boundary[str(row["boundary"])][
            _float_key(float(row["pres_target"]))
        ].append(row)
    slopes = {
        str(row["boundary"]): row.get("loglog_slope", "")
        for row in boundary_summary
    }
    styles = {
        "linearity": ("o", "-", "#1f77b4"),
        "exact-dirichlet": ("s", "--", "#d62728"),
        "crra-robin": ("^", "-.", "#2ca02c"),
    }
    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(figsize=figure_size)
        for boundary, cells in by_boundary.items():
            tolerance_keys = sorted(
                cells, key=lambda key: float.fromhex(key)
            )
            x = []
            xerr = []
            y = []
            yerr = []
            for key in tolerance_keys:
                rows = cells[key]
                achieved = np.asarray(
                    [float(row["achieved_pres"]) for row in rows],
                    dtype=float,
                )
                x.append(float(np.mean(achieved)))
                xerr.append(
                    float(np.std(achieved, ddof=1))
                    if achieved.size > 1 else 0.0
                )
                values = np.asarray(
                    [float(row["max_e_approx_X"]) for row in rows],
                    dtype=float,
                )
                y.append(float(np.mean(values)))
                yerr.append(
                    float(np.std(values, ddof=1))
                    if values.size > 1 else 0.0
                )
            marker, linestyle, color = styles.get(
                boundary, ("o", "-", None)
            )
            slope = slopes.get(boundary, "")
            label = boundary.replace("-", " ")
            if slope != "":
                label += rf" ($\hat\beta={float(slope):.3f}$)"
            ax.errorbar(
                np.asarray(x),
                np.asarray(y),
                xerr=np.asarray(xerr),
                yerr=np.asarray(yerr),
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=1.8,
                capsize=2.5,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Achieved $p_{\mathrm{res}}$")
        ax.set_ylabel(r"Seedwise worst $\widehat p_X$")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in formats:
            fig.savefig(
                stage / f"{BOUNDARY_PLOT_STEM}.{suffix}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)


def aggregate(
    result_dirs: Sequence[Path],
    output: Path,
    *,
    expected_seeds: Sequence[int],
    expected_tolerances: Sequence[float],
    min_runs_per_tolerance: int,
    checkpoints: Optional[Sequence[int]],
    make_plot: bool,
    plot_metrics: Sequence[str],
    formats: Sequence[str],
    figure_size: Tuple[float, float],
    dpi: int,
    font_size: float,
    overwrite: bool,
    select_targets: Sequence[float] = (),
    select_seeds: Sequence[int] = (),
    refinement_failure_mode: str = "error",
    x_tick_count: int = 0,
    mark_refinement_issues: bool = False,
) -> Mapping[str, Any]:
    if refinement_failure_mode not in {"error", "report"}:
        raise ValueError(
            "refinement_failure_mode must be 'error' or 'report'"
        )
    if x_tick_count < 0:
        raise ValueError("x_tick_count must be nonnegative")
    had_output = _check_output(output, overwrite)
    selected_result_dirs = list(result_dirs)
    excluded_result_dirs: List[Path] = []
    try:
        select_target_keys = {
            _float_key(float(value)) for value in select_targets
        }
        expected_target_keys = {
            _float_key(float(value)) for value in expected_tolerances
        }
        if (
            select_target_keys
            and expected_target_keys
            and select_target_keys != expected_target_keys
        ):
            raise ValueError(
                "--select-target and --expected-tolerances disagree: "
                f"selected={sorted(float(value) for value in select_targets)}, "
                f"expected={sorted(float(value) for value in expected_tolerances)}"
            )
        select_seed_set = {int(seed) for seed in select_seeds}
        expected_seed_set = {int(seed) for seed in expected_seeds}
        if (
            select_seed_set
            and expected_seed_set
            and select_seed_set != expected_seed_set
        ):
            raise ValueError(
                "--select-seeds and --expected-seeds disagree: "
                f"selected={sorted(select_seed_set)}, "
                f"expected={sorted(expected_seed_set)}"
            )
        effective_expected_tolerances = (
            list(expected_tolerances)
            if expected_tolerances
            else list(select_targets)
        )
        effective_expected_seeds = (
            list(expected_seeds)
            if expected_seeds
            else list(select_seeds)
        )
        selected_result_dirs, excluded_result_dirs = _select_result_dirs(
            result_dirs,
            select_targets=select_targets,
            select_seeds=select_seeds,
        )
        candidates = [
            _validate_candidate(
                directory,
                refinement_failure_mode=refinement_failure_mode,
            )
            for directory in selected_result_dirs
        ]

        # Deduplicate only after every selected discovered attempt has had its
        # status checked. Thus a selected newer failure cannot revive an older
        # successful cell. Among successful duplicates, newest wins.
        newest: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for record in candidates:
            key = (
                str(record["protocol_sha256"]),
                _float_key(float(record["pres_target"])),
                int(record["seed"]),
            )
            prior = newest.get(key)
            if prior is None or int(record["attempt_time_ns"]) > int(prior["attempt_time_ns"]):
                newest[key] = record
        records = list(newest.values())
        if not records:
            raise ValueError("no successful E4 exact-map results")
        if select_target_keys and select_seed_set:
            observed_cells = {
                (
                    _float_key(float(record["pres_target"])),
                    int(record["seed"]),
                )
                for record in records
            }
            expected_cells = {
                (target_key, seed)
                for target_key in select_target_keys
                for seed in select_seed_set
            }
            missing_cells = sorted(
                expected_cells - observed_cells,
                key=lambda item: (float.fromhex(item[0]), item[1]),
            )
            if missing_cells:
                rendered = [
                    {
                        "pres_target": float.fromhex(target_key),
                        "seed": seed,
                    }
                    for target_key, seed in missing_cells
                ]
                raise ValueError(
                    "selected target/seed Cartesian product is incomplete; "
                    f"missing={rendered}"
                )

        protocols = {str(record["protocol_sha256"]) for record in records}
        if len(protocols) != 1:
            raise ValueError(
                "cross-tolerance canonical protocol mismatch after excluding pres_target: "
                f"{sorted(protocols)}"
            )
        markets = {str(record["market_sha256"]) for record in records}
        if len(markets) != 1:
            raise ValueError(f"canonical market snapshot mismatch: {sorted(markets)}")
        schedules = {tuple(record["schedule"]) for record in records}
        if len(schedules) != 1:
            raise ValueError(f"cross-tolerance checkpoint schedule mismatch: {schedules}")

        by_tolerance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_tolerance[_float_key(float(record["pres_target"]))].append(record)
        observed_tolerances = sorted(
            (float.fromhex(key) for key in by_tolerance), key=float
        )
        if effective_expected_tolerances:
            observed_keys = {_float_key(value) for value in observed_tolerances}
            expected_keys = {
                _float_key(value)
                for value in effective_expected_tolerances
            }
            if observed_keys != expected_keys:
                raise ValueError(
                    "residual-tolerance set mismatch: "
                    f"found={observed_tolerances}, "
                    f"expected={sorted(effective_expected_tolerances)}"
                )

        common_seed_set: Optional[List[int]] = None
        for tolerance in observed_tolerances:
            cell = by_tolerance[_float_key(tolerance)]
            seeds = sorted(int(record["seed"]) for record in cell)
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"duplicate seed at pres_target={tolerance}: {seeds}")
            if len(seeds) < min_runs_per_tolerance:
                raise ValueError(
                    f"pres_target={tolerance:g} has {len(seeds)} runs, fewer than "
                    f"--min-runs-per-tolerance={min_runs_per_tolerance}"
                )
            if (
                effective_expected_seeds
                and seeds
                != sorted(
                    set(int(seed) for seed in effective_expected_seeds)
                )
            ):
                raise ValueError(
                    f"seed set mismatch at pres_target={tolerance:g}: "
                    f"found={seeds}, "
                    "expected="
                    f"{sorted(set(effective_expected_seeds))}"
                )
            if common_seed_set is None:
                common_seed_set = seeds
            elif seeds != common_seed_set:
                raise ValueError(
                    "cross-tolerance seed sets differ; refusing an unbalanced paper summary: "
                    f"first={common_seed_set}, at {tolerance:g}={seeds}"
                )

        records.sort(key=lambda row: (float(row["pres_target"]), int(row["seed"])))
        per_seed, error_metrics = _per_seed_rows(records, checkpoints)
        summary = _summary_rows(per_seed, error_metrics)
        main_plot_series: Optional[Dict[str, Any]] = None
        main_plot_ticks = np.asarray([], dtype=float)
        main_plot_tick_labels: List[str] = []
        if make_plot:
            if len(plot_metrics) != 1:
                raise ValueError(
                    "the paper-style E4 plot requires exactly one selected "
                    "metric"
                )
            main_plot_series = _main_plot_series(
                summary,
                per_seed,
                metric=str(plot_metrics[0]),
            )
            main_plot_ticks, main_plot_tick_labels = (
                _selected_x_tick_data(
                    main_plot_series["x"],
                    main_plot_series["tolerances"],
                    x_tick_count,
                )
            )
        (
            boundary_per_checkpoint,
            boundary_per_seed,
            boundary_summary,
        ) = _boundary_sensitivity_rows(records, checkpoints)
        per_fields = list(per_seed[0])
        summary_fields = [
            "pres_target",
            "metric",
            "n",
            "mean",
            "std",
            "sem",
            "ci95_low",
            "ci95_high",
            "min",
            "max",
            "seeds",
        ]
        per_seed_by_cell = {
            (
                _float_key(float(row["pres_target"])),
                int(row["seed"]),
            ): row
            for row in per_seed
        }
        payload: Dict[str, Any] = {
            "status": "success",
            "interpretation": (
                "seedwise worst E4 finite-domain FD approximation errors; "
                "no exact-map contraction claim"
            ),
            "n_tolerances": len(observed_tolerances),
            "tolerances": observed_tolerances,
            "n_seeds_per_tolerance": len(common_seed_set or []),
            "seeds": common_seed_set or [],
            "min_runs_per_tolerance": min_runs_per_tolerance,
            "protocol_sha256": records[0]["protocol_sha256"],
            "protocol": records[0]["protocol_payload"],
            "source_driver_sha256": sorted(
                {str(record["source_driver_sha256"]) for record in records}
            ),
            "source_core_sha256": sorted(
                {str(record["source_core_sha256"]) for record in records}
            ),
            "legacy_pre_merton_schema_result_dirs": [
                str(record["directory"])
                for record in records
                if bool(record["legacy_pre_merton_schema"])
            ],
            "market_sha256": records[0]["market_sha256"],
            "full_checkpoint_schedule": records[0]["full_schedule"],
            "checkpoint_schedule": records[0]["schedule"],
            "min_paper_checkpoint": records[0]["min_paper_checkpoint"],
            "refinement_rule": records[0]["refinement_rule"],
            "refinement_scope": REFINEMENT_SCOPE,
            "boundary_sensitivity_role": BOUNDARY_SENSITIVITY_ROLE,
            "primary_boundary": "linearity",
            "comparison_boundaries": [
                str(value)
                for value in records[0]["grid"].get("boundaries", [])[1:]
            ],
            "refinement_failure_mode": refinement_failure_mode,
            "all_required_numerical_refinement_evidence_pass": all(
                record["e4_refinement_evidence_status"] == "pass"
                for record in records
            ),
            "numerical_refinement_issue_cells": [
                {
                    "pres_target": record["pres_target"],
                    "seed": record["seed"],
                    "status": record[
                        "e4_refinement_evidence_status"
                    ],
                    "required_iterations": record[
                        "e4_refinement_required_iterations"
                    ],
                    "max_required_numerical_tolerance_ratio": per_seed_by_cell[
                        (
                            _float_key(float(record["pres_target"])),
                            int(record["seed"]),
                        )
                    ]["max_required_numerical_tolerance_ratio"],
                    "max_required_numerical_tolerance_ratio_outer": (
                        per_seed_by_cell[
                            (
                                _float_key(
                                    float(record["pres_target"])
                                ),
                                int(record["seed"]),
                            )
                        ][
                            "max_required_numerical_tolerance_ratio_outer"
                        ]
                    ),
                }
                for record in records
                if record["e4_refinement_evidence_status"] != "pass"
            ],
            "legacy_boundary_separated_reassessment_applied": any(
                bool(record["boundary_separated_reassessment_applied"])
                and record["refinement_scope"]
                == "legacy_cartesian_including_boundary"
                for record in records
            ),
            "e4_refinement_evidence_rule": (
                "initial_first_last_worst_e_approx_X"
            ),
            "e4_refinement_required_iterations": {
                f"{record['pres_target']:g}/seed{record['seed']}": record[
                    "e4_refinement_required_iterations"
                ]
                for record in records
            },
            "requested_checkpoints": (
                list(checkpoints) if checkpoints is not None else records[0]["schedule"]
            ),
            "error_metrics": error_metrics,
            "main_figure": (
                {
                    "plot_stem": PLOT_STEM,
                    "metric": main_plot_series["metric"],
                    "x_label": r"$p_{\mathrm{res}}$",
                    "y_label": PLOT_METRIC_SPECS[
                        str(main_plot_series["metric"])
                    ]["y_label"],
                    "C_num_empirical_upper": main_plot_series[
                        "C_num_empirical_upper"
                    ],
                    "upper_envelope_definition": (
                        "C_num=max_seed_target(metric/achieved_pres); "
                        "guide=C_num*p_res"
                    ),
                    "achieved_pres_mean_ticks": [
                        float(value) for value in main_plot_ticks
                    ],
                    "achieved_pres_mean_tick_labels": (
                        main_plot_tick_labels
                    ),
                    "x_tick_count": x_tick_count,
                    "refinement_issue_markers": bool(
                        mark_refinement_issues
                    ),
                    "legend": False,
                }
                if main_plot_series is not None
                else None
            ),
            "boundary_sensitivity_outputs": {
                "per_checkpoint": (
                    "e4_boundary_sensitivity_per_checkpoint.csv"
                ),
                "per_seed": "e4_boundary_sensitivity_per_seed.csv",
                "summary": "e4_boundary_sensitivity_summary.csv",
                "plot_stem": (
                    BOUNDARY_PLOT_STEM
                    if (
                        make_plot
                        and len(
                            {
                                str(row["boundary"])
                                for row in boundary_per_seed
                            }
                        )
                        > 1
                    )
                    else None
                ),
            },
            "result_dirs": [str(record["directory"]) for record in records],
            "selection": {
                "select_target": sorted(
                    float(value) for value in select_targets
                ),
                "select_seeds": sorted(
                    set(int(seed) for seed in select_seeds)
                ),
                "effective_expected_tolerances": sorted(
                    float(value)
                    for value in effective_expected_tolerances
                ),
                "effective_expected_seeds": sorted(
                    set(int(seed) for seed in effective_expected_seeds)
                ),
                "n_discovered_result_dirs": len(result_dirs),
                "n_selected_result_dirs": len(selected_result_dirs),
                "n_excluded_result_dirs": len(excluded_result_dirs),
                "excluded_result_dirs": [
                    str(path) for path in excluded_result_dirs
                ],
                "filter_applied_before_candidate_validation": bool(
                    select_targets or select_seeds
                ),
            },
            "newest_successful_attempt_selected_after_status_validation": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".liu-e4-tolerance-stage-", dir=str(output.parent)
        ) as stage_text:
            stage = Path(stage_text)
            _write_csv(stage / "e4_tolerance_per_seed.csv", per_seed, per_fields)
            _write_csv(stage / "e4_tolerance_summary.csv", summary, summary_fields)
            _write_csv(
                stage / "e4_boundary_sensitivity_per_checkpoint.csv",
                boundary_per_checkpoint,
                list(boundary_per_checkpoint[0]),
            )
            _write_csv(
                stage / "e4_boundary_sensitivity_per_seed.csv",
                boundary_per_seed,
                list(boundary_per_seed[0]),
            )
            _write_csv(
                stage / "e4_boundary_sensitivity_summary.csv",
                boundary_summary,
                list(boundary_summary[0]),
            )
            if make_plot:
                _plot(
                    summary,
                    per_seed,
                    stage,
                    metrics=plot_metrics,
                    formats=formats,
                    figure_size=figure_size,
                    dpi=dpi,
                    font_size=font_size,
                    x_tick_count=x_tick_count,
                    mark_refinement_issues=mark_refinement_issues,
                )
                if len(
                    {
                        str(row["boundary"])
                        for row in boundary_per_seed
                    }
                ) > 1:
                    _plot_boundary_sensitivity(
                        boundary_per_seed,
                        boundary_summary,
                        stage,
                        formats=formats,
                        figure_size=figure_size,
                        dpi=dpi,
                        font_size=font_size,
                    )
            _atomic_json(stage / "e4_tolerance_aggregate_status.json", payload)
            (stage / "_SUCCESS_E4_TOLERANCE_AGG").touch()
            _commit_stage(stage, output)
        return payload
    except Exception as exc:
        if not had_output:
            _prepare_output(output)
            _atomic_json(
                output / "e4_tolerance_aggregate_status.json",
                {
                    "status": "failed",
                    "error": repr(exc),
                    "result_dirs": [str(path) for path in result_dirs],
                    "selection": {
                        "select_target": sorted(
                            float(value) for value in select_targets
                        ),
                        "select_seeds": sorted(
                            set(int(seed) for seed in select_seeds)
                        ),
                        "selected_result_dirs": [
                            str(path) for path in selected_result_dirs
                        ],
                        "excluded_result_dirs": [
                            str(path) for path in excluded_result_dirs
                        ],
                    },
                },
            )
            (output / "_FAILED_E4_TOLERANCE_AGG").touch()
        raise


def _parse_checkpoints(text: str) -> Optional[List[int]]:
    if str(text).strip().lower() in ("", "all"):
        return None
    values = parse_seed_spec(text)
    if not values or any(value < 1 for value in values):
        raise ValueError("--checkpoints must be 'all' or positive outer indices")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate existing Liu E4 FD audits across residual tolerances"
    )
    parser.add_argument("--out-root", type=Path, action="append", default=[])
    parser.add_argument("--result-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--select-target",
        action="append",
        default=[],
        metavar="TARGET[,TARGET...]",
        help=(
            "include only these nominal pres_target cells before validation; "
            "accepts comma/space-separated values and may be repeated"
        ),
    )
    parser.add_argument(
        "--select-seeds",
        action="append",
        default=[],
        metavar="SEED[,SEED...]",
        help=(
            "include only these training seeds before validation; accepts the "
            "same comma/space/range syntax as --expected-seeds and may be "
            "repeated"
        ),
    )
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--expected-tolerances", default="")
    parser.add_argument("--min-runs-per-tolerance", type=int, default=1)
    parser.add_argument("--checkpoints", default="all")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-metric",
        default=None,
        help=(
            "single E4 line for the paper-style plot; default "
            "max_e_approx_X. Short aliases include X, value, and bundle"
        ),
    )
    parser.add_argument(
        "--plot-metrics",
        default=None,
        help=(
            "backward-compatible alias for --plot-metric; exactly one "
            "metric is now required"
        ),
    )
    parser.add_argument("--formats", default="png")
    parser.add_argument("--figure-size", default="4.8,3.4")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument(
        "--x-tick-count",
        type=int,
        default=0,
        help=(
            "number of achieved-p_res target means labeled on the x-axis; "
            "0 (default) labels every selected target mean"
        ),
    )
    parser.add_argument(
        "--mark-refinement-issues",
        action="store_true",
        help=(
            "optionally overlay red x markers for required numerical "
            "refinement issues; off by default for the paper figure"
        ),
    )
    parser.add_argument(
        "--refinement-failure-mode",
        choices=("error", "report"),
        default="error",
        help=(
            "error (default) stops on failed required grid/domain evidence; "
            "report still writes tables/plots and records affected tolerance "
            "points in CSV/status. Boundary replacement is always "
            "report-only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out_root and not args.result_dir:
        raise ValueError("provide at least one --out-root or --result-dir")
    if args.min_runs_per_tolerance < 1:
        raise ValueError("--min-runs-per-tolerance must be positive")
    if args.dpi < 1 or not math.isfinite(args.font_size) or args.font_size <= 0:
        raise ValueError("--dpi and --font-size must be positive")
    if args.x_tick_count < 0:
        raise ValueError("--x-tick-count must be nonnegative")
    if args.plot_metric is not None and args.plot_metrics is not None:
        raise ValueError(
            "use only one of --plot-metric or --plot-metrics"
        )
    raw_plot_metric = (
        args.plot_metric
        if args.plot_metric is not None
        else args.plot_metrics
        if args.plot_metrics is not None
        else "max_e_approx_X"
    )
    plot_metric = _parse_plot_metric(raw_plot_metric)
    result_dirs = discover_result_dirs(args.out_root, args.result_dir)
    select_targets = parse_float_spec(
        ",".join(str(value) for value in args.select_target)
    )
    aggregate(
        result_dirs,
        args.output.expanduser().resolve(),
        expected_seeds=parse_seed_spec(args.expected_seeds),
        expected_tolerances=parse_float_spec(args.expected_tolerances),
        min_runs_per_tolerance=args.min_runs_per_tolerance,
        checkpoints=_parse_checkpoints(args.checkpoints),
        make_plot=bool(args.plot),
        plot_metrics=[plot_metric],
        formats=_parse_formats(args.formats),
        figure_size=_parse_pair(args.figure_size, "--figure-size"),
        dpi=args.dpi,
        font_size=args.font_size,
        overwrite=bool(args.overwrite),
        select_targets=select_targets,
        select_seeds=parse_seed_spec(
            ",".join(str(value) for value in args.select_seeds)
        ),
        refinement_failure_mode=args.refinement_failure_mode,
        x_tick_count=args.x_tick_count,
        mark_refinement_issues=bool(args.mark_refinement_issues),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
