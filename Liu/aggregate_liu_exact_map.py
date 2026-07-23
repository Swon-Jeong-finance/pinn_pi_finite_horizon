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
AGG_MANAGED_OUTPUTS = (
    "exact_map_per_seed.csv",
    "exact_map_summary.csv",
    "exact_map_worst_per_seed.csv",
    "exact_map_worst_summary.csv",
    "e4_per_seed.csv",
    "e4_summary.csv",
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
)
E4_METRICS = (
    "e_approx_value", "e_approx_vw", "e_approx_vww", "e_approx_vwx",
    "e_approx_bundle", "e_approx_X", "approx_sensitivity_envelope",
    "source_min_log_joint_eig", "source_max_log_joint_eig",
    "source_min_original_joint_eig", "source_max_original_joint_eig",
    "source_nonpositive_log_eig_fraction",
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


def _validate_provenance(directory: Path, config: Mapping[str, Any],
                         exact_rows: Sequence[Mapping[str, str]],
                         e4_rows: Sequence[Mapping[str, str]]) -> None:
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
        if not expected or sha256_file(path) != expected:
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
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
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


def _validate_status_contract(directory: Path, status: Mapping[str, Any],
                              config: Mapping[str, Any],
                              exact_rows: Sequence[Mapping[str, str]],
                              e4_rows: Sequence[Mapping[str, str]],
                              exact_refinement_rows: Sequence[Mapping[str, str]],
                              e4_refinement_rows: Sequence[Mapping[str, str]]) -> None:
    """Require the success status to describe the primary CSVs exactly."""

    source = directory / STATUS_INPUT
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
    if nonelliptic_exact:
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
                if metric.endswith("sensitivity_envelope"):
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
              overwrite: bool) -> Mapping[str, Any]:
    had_managed_output = _check_output(output, overwrite)
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
            if not allow_partial_sensitivity and any(
                row.get("refinement_status") != "pass" for row in exact_rows + e4_rows
            ):
                raise ValueError(
                    f"seed={seed} does not pass the requested grid/domain/boundary sensitivity audit"
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
                e4_all.append(row)
            records.append({
                "seed": seed, "directory": str(directory), "group": group,
                "protocol_hash": protocol, "market_sha256": market,
                "exact_schedule": exact_schedule, "e4_schedule": e4_schedule,
                "undefined_outers": undefined,
            })

        seeds = sorted(seen_seed)
        if len(seeds) < int(min_seeds):
            raise ValueError(f"found {len(seeds)} seeds, fewer than --min-seeds={min_seeds}: {seeds}")
        if expected_seeds and seeds != sorted(set(int(value) for value in expected_seeds)):
            raise ValueError(f"seed set mismatch: found {seeds}, expected {sorted(set(expected_seeds))}")
        for field in ("group", "protocol_hash", "market_sha256", "exact_schedule", "e4_schedule"):
            serialized = {json.dumps(record[field], sort_keys=True) for record in records}
            if len(serialized) != 1:
                raise ValueError(f"cross-seed {field} mismatch: {serialized}")

        # Undefined values remain visible in per-seed output when explicitly
        # permitted.  A paper summary cannot form a common-sample statistic,
        # so it is marked unavailable rather than computed on fewer seeds.
        has_undefined = any(record["undefined_outers"] for record in records)
        has_partial_sensitivity = any(
            row.get("refinement_status") != "pass" for row in exact_all + e4_all
        )
        exact_summary = [] if has_undefined else _summary_rows(
            exact_all, index_field="source_outer_iter", metrics=EXACT_METRICS,
            source=Path("combined exact-map rows"),
        )
        e4_summary = _summary_rows(
            e4_all, index_field="target_outer_iter", metrics=E4_METRICS,
            source=Path("combined E4 rows"),
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
        all_sensitivity_pass = not has_partial_sensitivity
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
            and all_sensitivity_pass
            and all_seed_outer_locally_unmodified
            and global_exact is not None
            and global_exact < 1.0
            and global_envelope is not None
            and global_envelope < 1.0
        )
        blockers: List[str] = []
        if not all_denominators_defined:
            blockers.append("undefined_denominator")
        if not all_sensitivity_pass:
            blockers.append("sensitivity_not_passed_for_all_exact_and_e4_rows")
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
            "All tested primary exact-map ratios and their finite-domain sampled-map "
            "sensitivity envelopes remained below one."
            if finite_domain_below_one else None
        )
        payload = {
            "status": "success", "paper_summary_status": paper_status,
            "n_seeds": len(seeds), "seeds": seeds,
            "group": records[0]["group"], "protocol_hash": records[0]["protocol_hash"],
            "market_sha256": records[0]["market_sha256"],
            "checkpoint_schedule": records[0]["exact_schedule"],
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
            "interpretation": (
                "finite-domain sampled-map audit; no whole-space contraction claim"
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
