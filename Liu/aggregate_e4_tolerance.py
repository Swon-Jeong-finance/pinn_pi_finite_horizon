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
    _validate_provenance,
    _validate_status_contract,
    discover_result_dirs,
    parse_seed_spec,
    read_csv,
    read_json,
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
    "e4_tolerance_aggregate_status.json",
    "_SUCCESS_E4_TOLERANCE_AGG",
    "_FAILED_E4_TOLERANCE_AGG",
)
PLOT_STEM = "e4_tolerance_errors"
PLOT_FORMATS = ("png", "pdf", "svg", "eps")


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
    names.extend(
        path.name
        for path in output.glob(f"{PLOT_STEM}.*")
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
    relevant = {
        "training_protocol_args_without_pres_target": canonical_training,
        "implementation_hashes": config.get("implementation_hashes"),
        "checkpoint_selection": config.get("checkpoint_selection"),
        "checkpoint_schedule": config.get("checkpoint_schedule"),
        "grid": config.get("grid"),
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


def _validate_candidate(directory: Path) -> Dict[str, Any]:
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
    config = read_json(directory / CONFIG_INPUT)
    exact_rows = read_csv(directory / EXACT_INPUT)
    e4_rows = read_csv(directory / E4_INPUT)
    exact_refinement = read_csv(directory / "exact_map_refinement.csv")
    e4_refinement = read_csv(directory / "e4_approximation_refinement.csv")
    _validate_status_contract(
        directory,
        status,
        config,
        exact_rows,
        e4_rows,
        exact_refinement,
        e4_refinement,
    )
    _validate_provenance(directory, config, exact_rows, e4_rows)
    if any(_integer(row, "is_primary", directory / E4_INPUT) != 1 for row in e4_rows):
        raise ValueError(f"{directory / E4_INPUT}: contains non-primary rows")
    if any(row.get("refinement_status") != "pass" for row in e4_rows):
        raise ValueError(f"{directory}: E4 grid/domain/boundary refinement did not pass")

    seed = int(_identity(e4_rows, "seed", directory / E4_INPUT))
    if any(_integer(row, "seed", directory / E4_INPUT) != seed for row in e4_rows):
        raise ValueError(f"{directory / E4_INPUT}: mixes training seeds")
    schedule = [_integer(row, "target_outer_iter", directory / E4_INPUT) for row in e4_rows]
    if schedule != sorted(set(schedule)):
        raise ValueError(f"{directory}: E4 schedule is not sorted and unique")

    training = config.get("training_protocol_args")
    if not isinstance(training, Mapping):
        raise ValueError(f"{directory}: missing training_protocol_args")
    raw_target = training.get("pres_target")
    try:
        target = float(raw_target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{directory}: invalid pres_target={raw_target!r}") from exc
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"{directory}: pres_target must be finite and positive")

    protocol_hash, protocol_payload = _canonical_protocol(config)
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
        "schedule": schedule,
        "e4_rows": e4_rows,
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
        item: Dict[str, Any] = {
            "protocol_sha256": record["protocol_sha256"],
            "market_sha256": record["market_sha256"],
            "seed": record["seed"],
            "pres_target": record["pres_target"],
            "achieved_pres": record["achieved_pres"],
            "checkpoint_schedule": ",".join(str(value) for value in schedule),
            "requested_checkpoints": ",".join(str(value) for value in selected),
            "n_checkpoints": len(selected),
            "all_e4_refinement_pass": 1,
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


def _parse_pair(text: str, option: str) -> Tuple[float, float]:
    values = [float(token.strip()) for token in str(text).split(",") if token.strip()]
    if len(values) != 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{option} requires two positive finite values")
    return values[0], values[1]


def _plot(
    summary: Sequence[Mapping[str, Any]],
    stage: Path,
    *,
    metrics: Sequence[str],
    formats: Sequence[str],
    figure_size: Tuple[float, float],
    dpi: int,
    font_size: float,
) -> None:
    import matplotlib.pyplot as plt

    by_metric: Dict[str, Dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary:
        by_metric[str(row["metric"])][float(row["pres_target"])] = row
    x_rows = by_metric.get("achieved_pres", {})
    if not x_rows:
        raise ValueError("cannot plot without achieved_pres summaries")
    labels = {
        "max_e_approx_value": "Value",
        "max_e_approx_bundle": "Bundle",
        "max_e_approx_X": "Value + bundle",
        "max_e_approx_control": "Control",
        "max_e_approx_theta": "Control",
        "max_e_approx_vartheta": "Control",
        "max_approx_sensitivity_envelope": "Sensitivity envelope",
    }
    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(figsize=figure_size)
        for metric in metrics:
            if metric not in by_metric:
                raise ValueError(f"requested plot metric is unavailable: {metric}")
            tolerances = sorted(set(x_rows) & set(by_metric[metric]))
            x = np.asarray([float(x_rows[value]["mean"]) for value in tolerances])
            y = np.asarray([float(by_metric[metric][value]["mean"]) for value in tolerances])
            xerr = np.asarray([float(x_rows[value]["std"]) for value in tolerances])
            yerr = np.asarray([float(by_metric[metric][value]["std"]) for value in tolerances])
            xerr = np.where(np.isfinite(xerr), xerr, 0.0)
            yerr = np.where(np.isfinite(yerr), yerr, 0.0)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=2.5,
                label=labels.get(metric, metric),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Achieved $p_{\mathrm{res}}$")
        ax.set_ylabel("Seedwise worst E4 approximation error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in formats:
            fig.savefig(stage / f"{PLOT_STEM}.{suffix}", dpi=dpi, bbox_inches="tight")
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
) -> Mapping[str, Any]:
    had_output = _check_output(output, overwrite)
    try:
        candidates = [_validate_candidate(directory) for directory in result_dirs]

        # Deduplicate only after every discovered attempt has had its status
        # checked.  Thus a discoverable newer failure cannot revive an older
        # successful cell.  Among successful duplicates, newest wins.
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
        if expected_tolerances:
            observed_keys = {_float_key(value) for value in observed_tolerances}
            expected_keys = {_float_key(value) for value in expected_tolerances}
            if observed_keys != expected_keys:
                raise ValueError(
                    "residual-tolerance set mismatch: "
                    f"found={observed_tolerances}, expected={sorted(expected_tolerances)}"
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
            if expected_seeds and seeds != sorted(set(int(seed) for seed in expected_seeds)):
                raise ValueError(
                    f"seed set mismatch at pres_target={tolerance:g}: "
                    f"found={seeds}, expected={sorted(set(expected_seeds))}"
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
            "market_sha256": records[0]["market_sha256"],
            "checkpoint_schedule": records[0]["schedule"],
            "requested_checkpoints": (
                list(checkpoints) if checkpoints is not None else records[0]["schedule"]
            ),
            "error_metrics": error_metrics,
            "result_dirs": [str(record["directory"]) for record in records],
            "newest_successful_attempt_selected_after_status_validation": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".liu-e4-tolerance-stage-", dir=str(output.parent)
        ) as stage_text:
            stage = Path(stage_text)
            _write_csv(stage / "e4_tolerance_per_seed.csv", per_seed, per_fields)
            _write_csv(stage / "e4_tolerance_summary.csv", summary, summary_fields)
            if make_plot:
                _plot(
                    summary,
                    stage,
                    metrics=plot_metrics,
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
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--expected-tolerances", default="")
    parser.add_argument("--min-runs-per-tolerance", type=int, default=1)
    parser.add_argument("--checkpoints", default="all")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-metrics",
        default="max_e_approx_value,max_e_approx_bundle,max_e_approx_X",
    )
    parser.add_argument("--formats", default="png")
    parser.add_argument("--figure-size", default="6.4,4.5")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=11.0)
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
    plot_metrics = [
        token.strip() for token in args.plot_metrics.split(",") if token.strip()
    ]
    if args.plot and not plot_metrics:
        raise ValueError("--plot-metrics must be nonempty when --plot is used")
    result_dirs = discover_result_dirs(args.out_root, args.result_dir)
    aggregate(
        result_dirs,
        args.output.expanduser().resolve(),
        expected_seeds=parse_seed_spec(args.expected_seeds),
        expected_tolerances=parse_float_spec(args.expected_tolerances),
        min_runs_per_tolerance=args.min_runs_per_tolerance,
        checkpoints=_parse_checkpoints(args.checkpoints),
        make_plot=bool(args.plot),
        plot_metrics=plot_metrics,
        formats=_parse_formats(args.formats),
        figure_size=_parse_pair(args.figure_size, "--figure-size"),
        dpi=args.dpi,
        font_size=args.font_size,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
