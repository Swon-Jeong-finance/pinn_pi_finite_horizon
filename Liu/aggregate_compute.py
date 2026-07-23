"""Aggregate Liu E8 timing runs and create error-versus-computation figures.

The training scripts record CUDA-synchronized core wall time (excluding final
checkpoint serialization), end-to-end elapsed time, optimizer steps and
separate training/evaluation CUDA peaks in ``status.json``. This module turns those
run-level observations into reproducible setting/seed tables.  By default it
accepts only ``timing_mode=1`` runs, so ordinary paper runs (which include
diagnostic and checkpoint I/O) cannot be mixed into the core-timing table by
accident.

This is a read-only post-processor: source run directories and checkpoints are
never changed.  Derived files are written under ``<out-root>/compute_summary``
unless ``--output`` is supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    GROUP_IGNORE_KEYS,
    canonical_market_hash,
    find_runs,
    load_metrics_rows,
    parse_seed_spec,
    t_crit_95,
)


MEASURES = (
    "core_train_wall_sec",
    "total_optimizer_steps",
    "train_gpu_peak_mem_bytes",
    "eval_gpu_peak_mem_bytes",
    "RelL2_V",
    "RelL2_theta",
)

RUNTIME_GROUP_KEYS = (
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_compute_capability",
    "torch_version",
    "cuda_runtime_version",
    "cudnn_version",
    "python_version",
    "numpy_version",
    "platform",
)


class RunIntegrityError(ValueError):
    """An in-scope timing run is internally inconsistent or incomplete."""

DERIVED_FILENAMES = (
    "compute_runs.csv",
    "compute_summary_long.csv",
    "compute_table.csv",
    "compute_groups.json",
    "compute_metadata.json",
    "e8_error_vs_compute.png",
    "e8_error_vs_compute.pdf",
    "e8_error_vs_compute.svg",
    "e8_error_vs_compute.eps",
)


@dataclass(frozen=True)
class Observation:
    run_dir: str
    group_id: str
    model_type: str
    n_assets: int
    m_states: int
    seed: int
    market_hash: str
    gpu_name: str
    gpu_total_memory_bytes: int
    timing_mode: bool
    values: Mapping[str, float]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite_float(value: Any, *, name: str, required: bool = True) -> float:
    if value in (None, ""):
        if required:
            raise ValueError(f"missing {name}")
        return float("nan")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite {name}: {value!r}")
    return out


def _canonical_group(
    args: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Seed/device-directory independent training and hardware signature."""

    core = {
        key: args[key]
        for key in sorted(args)
        if key not in GROUP_IGNORE_KEYS
    }
    # A timing comparison is hardware-specific.  CUDA ordinal is intentionally
    # excluded; the physical GPU model and software stack are retained.
    environment = {key: runtime.get(key, "unknown") for key in RUNTIME_GROUP_KEYS}
    payload = {"training": core, "runtime": environment}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12], payload


def _validate_runtime_environment(
    runtime: Mapping[str, Any], *, allow_unknown: bool
) -> Dict[str, Any]:
    """Return a normalized, paper-auditable CUDA runtime signature."""

    normalized = {key: runtime.get(key) for key in RUNTIME_GROUP_KEYS}
    missing = [
        key for key, value in normalized.items()
        if value is None or str(value).strip().lower() in {"", "unknown", "none"}
    ]
    try:
        total_memory = int(normalized.get("gpu_total_memory_bytes"))
    except (TypeError, ValueError):
        total_memory = 0
    if total_memory <= 0 and "gpu_total_memory_bytes" not in missing:
        missing.append("gpu_total_memory_bytes")

    effective = str(runtime.get("effective_device", runtime.get("effective_cuda_device", "")))
    if not effective.startswith("cuda"):
        missing.append("effective_device(cuda)")
    if runtime.get("cuda_available") is not True:
        missing.append("cuda_available=true")

    if missing and not allow_unknown:
        raise RunIntegrityError(
            "incomplete CUDA runtime metadata: " + ", ".join(sorted(set(missing)))
        )
    for key in missing:
        if key in normalized:
            normalized[key] = "unknown"
    normalized["gpu_total_memory_bytes"] = total_memory if total_memory > 0 else "unknown"
    normalized["effective_device"] = effective or "unknown"
    normalized["cuda_available"] = bool(runtime.get("cuda_available", False))
    return normalized


def _metric_lookup(run_dir: str, margin: float) -> Dict[str, float]:
    selected: Dict[str, float] = {}
    for row in load_metrics_rows(run_dir):
        if not math.isclose(float(row["eval_margin"]), margin, rel_tol=0.0, abs_tol=5e-10):
            continue
        metric = str(row["metric"])
        if metric in {"RelL2_V", "RelL2_theta"}:
            if metric in selected:
                raise ValueError(
                    f"duplicate {metric} at eval_margin={margin} in {run_dir}"
                )
            selected[metric] = float(row["value"])
    return selected


def collect_observations(
    out_root: str,
    *,
    headline_margin: float,
    require_timing_mode: bool = True,
    expected_models: Sequence[str] = (),
    expected_m_states: Sequence[int] = (),
    expected_n_assets: Optional[int] = None,
    allow_unknown_runtime: bool = False,
) -> Tuple[List[Observation], Dict[str, Dict[str, Any]], List[str]]:
    """Discover valid timing runs; return observations, group payloads, skips."""

    models = set(expected_models)
    dims = set(int(value) for value in expected_m_states)
    observations: List[Observation] = []
    groups: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []

    for run_dir in find_runs(os.path.abspath(out_root)):
        in_scope = False
        try:
            config = _read_json(os.path.join(run_dir, "config.json"))
            status = _read_json(os.path.join(run_dir, "status.json"))
            args = config.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("config.args is not an object")
            model = str(args.get("model_type", ""))
            n_assets = int(args["n_assets"])
            m_states = int(args["m_states"])
            seed = int(args["seed"])
            if models and model not in models:
                continue
            if dims and m_states not in dims:
                continue
            if expected_n_assets is not None and n_assets != int(expected_n_assets):
                continue
            in_scope = True

            config_timing = args.get("timing_mode")
            status_timing = status.get("timing_mode")
            if config_timing is None or status_timing is None:
                raise RunIntegrityError("timing_mode missing from config or status")
            if bool(config_timing) != bool(status_timing):
                raise RunIntegrityError(
                    f"timing_mode disagreement: config={config_timing!r}, "
                    f"status={status_timing!r}"
                )
            timing_mode = bool(config_timing)
            if require_timing_mode and not timing_mode:
                skipped.append(f"{run_dir}: not a timing_mode run")
                continue

            completion = os.path.join(run_dir, "_SUCCESS")
            conflicts = [
                name for name in ("_FAILED", "_STOPPED_EARLY")
                if os.path.exists(os.path.join(run_dir, name))
            ]
            if not os.path.isfile(completion):
                raise RunIntegrityError("missing _SUCCESS marker")
            if str(status.get("status", "")) != "success":
                raise RunIntegrityError(
                    f"_SUCCESS/status.json disagreement: status={status.get('status')!r}"
                )
            if conflicts:
                raise RunIntegrityError(
                    f"conflicting completion markers alongside _SUCCESS: {conflicts}"
                )

            runtime = config.get("runtime_environment", {})
            if not isinstance(runtime, dict):
                raise RunIntegrityError("runtime_environment is not an object")
            runtime = _validate_runtime_environment(
                runtime, allow_unknown=bool(allow_unknown_runtime)
            )

            market_path = os.path.join(run_dir, "market_params.npz")
            if not os.path.isfile(market_path):
                raise ValueError("missing market_params.npz")
            market_hash = canonical_market_hash(market_path)
            metrics = _metric_lookup(run_dir, headline_margin)
            core_wall = status.get("core_train_wall_sec")
            if core_wall in (None, "") and not require_timing_mode:
                core_wall = status.get("train_wall_sec", status.get("elapsed_sec"))
            values = {
                "core_train_wall_sec": _finite_float(
                    core_wall,
                    name="core_train_wall_sec",
                ),
                "total_optimizer_steps": _finite_float(
                    status.get("total_optimizer_steps", status.get("total_inner_steps")),
                    name="total_optimizer_steps",
                ),
                "train_gpu_peak_mem_bytes": _finite_float(
                    status.get("train_gpu_peak_mem_bytes"),
                    name="train_gpu_peak_mem_bytes",
                ),
                "eval_gpu_peak_mem_bytes": _finite_float(
                    status.get("eval_gpu_peak_mem_bytes"),
                    name="eval_gpu_peak_mem_bytes",
                ),
                "RelL2_V": _finite_float(metrics.get("RelL2_V"), name="RelL2_V"),
                "RelL2_theta": _finite_float(
                    metrics.get("RelL2_theta"), name="RelL2_theta"
                ),
            }
            if values["core_train_wall_sec"] <= 0 or values["total_optimizer_steps"] <= 0:
                raise RunIntegrityError("wall time and optimizer steps must be positive")
            if values["train_gpu_peak_mem_bytes"] <= 0 or values["eval_gpu_peak_mem_bytes"] <= 0:
                raise RunIntegrityError("training/evaluation GPU peaks must be positive")

            group_id, group_payload = _canonical_group(args, runtime)
            groups.setdefault(group_id, group_payload)
            if groups[group_id] != group_payload:
                raise RunIntegrityError(f"group hash collision: {group_id}")
            observations.append(
                Observation(
                    run_dir=os.path.abspath(run_dir),
                    group_id=group_id,
                    model_type=model,
                    n_assets=n_assets,
                    m_states=m_states,
                    seed=seed,
                    market_hash=market_hash,
                    gpu_name=str(runtime["gpu_name"]),
                    gpu_total_memory_bytes=(
                        int(runtime["gpu_total_memory_bytes"])
                        if runtime["gpu_total_memory_bytes"] != "unknown" else 0
                    ),
                    timing_mode=timing_mode,
                    values=values,
                )
            )
        except RunIntegrityError as exc:
            raise ValueError(f"invalid in-scope timing run {run_dir}: {exc}") from exc
        except Exception as exc:
            if in_scope:
                raise ValueError(
                    f"invalid in-scope timing run {run_dir}: {type(exc).__name__}: {exc}"
                ) from exc
            skipped.append(f"{run_dir}: {type(exc).__name__}: {exc}")
    return observations, groups, skipped


def summarize(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    mean = float(np.mean(array))
    if array.size == 1:
        std = sem = ci_lo = ci_hi = float("nan")
    else:
        std = float(np.std(array, ddof=1))
        sem = std / math.sqrt(int(array.size))
        half = t_crit_95(int(array.size) - 1) * sem
        ci_lo, ci_hi = mean - half, mean + half
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


def validate_support(
    observations: Sequence[Observation],
    *,
    expected_seeds: Sequence[int],
    min_runs: int,
) -> None:
    by_group: Dict[str, List[Observation]] = defaultdict(list)
    for observation in observations:
        by_group[observation.group_id].append(observation)
    if not by_group:
        raise ValueError("no eligible timing runs found")
    errors: List[str] = []
    expected = set(int(seed) for seed in expected_seeds)
    for group_id, rows in sorted(by_group.items()):
        seeds = [row.seed for row in rows]
        if len(seeds) != len(set(seeds)):
            errors.append(f"{group_id}: duplicate timing seeds {sorted(seeds)}")
        if len(rows) < int(min_runs):
            errors.append(f"{group_id}: {len(rows)} runs < min-runs={min_runs}")
        if expected and set(seeds) != expected:
            errors.append(
                f"{group_id}: seeds={sorted(set(seeds))}, expected={sorted(expected)}"
            )
        markets = {row.market_hash for row in rows}
        if len(markets) != 1:
            errors.append(f"{group_id}: mixed market snapshots ({len(markets)})")
    if errors:
        raise ValueError("timing support validation failed:\n  - " + "\n  - ".join(errors))


def validate_expected_cells(
    observations: Sequence[Observation],
    *,
    models: Sequence[str],
    m_states: Sequence[int],
) -> None:
    """Require one unambiguous timing configuration per method/dimension."""

    expected = {(str(model), int(dim)) for model in models for dim in m_states}
    groups_by_cell: Dict[Tuple[str, int], set[str]] = defaultdict(set)
    markets_by_dim: Dict[int, set[str]] = defaultdict(set)
    for row in observations:
        groups_by_cell[(row.model_type, row.m_states)].add(row.group_id)
        markets_by_dim[row.m_states].add(row.market_hash)
    actual = set(groups_by_cell)
    errors: List[str] = []
    if actual != expected:
        errors.append(
            f"method/M cells={sorted(actual)}, expected={sorted(expected)}"
        )
    for cell, group_ids in sorted(groups_by_cell.items()):
        if len(group_ids) != 1:
            errors.append(f"{cell}: ambiguous timing configurations {sorted(group_ids)}")
    for dim, market_hashes in sorted(markets_by_dim.items()):
        if len(market_hashes) != 1:
            errors.append(
                f"M={dim}: methods/seeds use {len(market_hashes)} market snapshots"
            )
    if errors:
        raise ValueError("timing cell validation failed:\n  - " + "\n  - ".join(errors))


def _write_csv(path: str, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def build_output_rows(
    observations: Sequence[Observation],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    group_meta: Dict[str, Observation] = {}
    for observation in observations:
        group_meta.setdefault(observation.group_id, observation)
        row: Dict[str, Any] = {
            "group_id": observation.group_id,
            "model_type": observation.model_type,
            "n_assets": observation.n_assets,
            "m_states": observation.m_states,
            "seed": observation.seed,
            "gpu_name": observation.gpu_name,
            "gpu_total_memory_bytes": observation.gpu_total_memory_bytes,
            "market_hash": observation.market_hash,
            "run_dir": observation.run_dir,
        }
        row.update(observation.values)
        run_rows.append(row)
        for measure, value in observation.values.items():
            grouped[(observation.group_id, measure)].append(float(value))

    long_rows: List[Dict[str, Any]] = []
    by_group_summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    by_group_seeds: Dict[str, List[int]] = defaultdict(list)
    for observation in observations:
        by_group_seeds[observation.group_id].append(observation.seed)
    for (group_id, measure), values in sorted(grouped.items()):
        stats = summarize(values)
        meta = group_meta[group_id]
        long_rows.append({
            "group_id": group_id,
            "model_type": meta.model_type,
            "n_assets": meta.n_assets,
            "m_states": meta.m_states,
            "gpu_name": meta.gpu_name,
            "gpu_total_memory_bytes": meta.gpu_total_memory_bytes,
            "measure": measure,
            "seeds": ",".join(str(v) for v in sorted(set(by_group_seeds[group_id]))),
            **stats,
        })
        by_group_summary[group_id][measure] = stats

    table_rows: List[Dict[str, Any]] = []
    for group_id in sorted(by_group_summary):
        meta = group_meta[group_id]
        row: Dict[str, Any] = {
            "group_id": group_id,
            "model_type": meta.model_type,
            "n_assets": meta.n_assets,
            "m_states": meta.m_states,
            "gpu_name": meta.gpu_name,
            "gpu_total_memory_bytes": meta.gpu_total_memory_bytes,
            "seeds": ",".join(str(v) for v in sorted(set(by_group_seeds[group_id]))),
        }
        for measure in MEASURES:
            stats = by_group_summary[group_id].get(measure, summarize([]))
            for name in ("n", "mean", "std", "ci95_lo", "ci95_hi"):
                row[f"{measure}_{name}"] = stats[name]
        table_rows.append(row)
    return run_rows, long_rows, table_rows


def _check_output(path: str, overwrite: bool) -> None:
    existing = [name for name in DERIVED_FILENAMES if os.path.exists(os.path.join(path, name))]
    if existing and not overwrite:
        raise ValueError(
            f"derived output already exists ({existing}); pass --overwrite to replace it"
        )


def _commit_staged_output(stage: str, output: str, *, overwrite: bool) -> None:
    """Commit a fully generated output set while retaining unrelated files."""

    os.makedirs(output, exist_ok=True)
    generated = {
        name for name in os.listdir(stage)
        if os.path.isfile(os.path.join(stage, name))
    }
    for name in sorted(generated):
        os.replace(os.path.join(stage, name), os.path.join(output, name))
    if overwrite:
        for name in DERIVED_FILENAMES:
            target = os.path.join(output, name)
            if name not in generated and (os.path.isfile(target) or os.path.islink(target)):
                os.remove(target)


def _plot(
    table_rows: Sequence[Mapping[str, Any]],
    output_dir: str,
    *,
    formats: Sequence[str],
    fig_width: float,
    fig_height: float,
    font_size: float,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    plt.rcParams.update({"font.size": font_size})
    resources = (
        ("core_train_wall_sec", "Core training wall-clock (s)"),
        ("total_optimizer_steps", "Optimizer steps"),
        ("train_gpu_peak_mem_bytes", "Training peak GPU memory (GiB)"),
    )
    errors = (("RelL2_V", "Value"), ("RelL2_theta", "Policy"))
    colors = {"pinn": "#4c78a8", "pipinn": "#f58518"}
    markers = {"pinn": "o", "pipinn": "s"}
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), squeeze=False)
    for row_index, (error, ylabel) in enumerate(errors):
        for col_index, (resource, xlabel) in enumerate(resources):
            ax = axes[row_index][col_index]
            for row in sorted(
                table_rows,
                key=lambda item: (int(item["m_states"]), str(item["model_type"])),
            ):
                x = float(row.get(f"{resource}_mean", float("nan")))
                y = float(row.get(f"{error}_mean", float("nan")))
                x_std = float(row.get(f"{resource}_std", float("nan")))
                y_std = float(row.get(f"{error}_std", float("nan")))
                if resource.endswith("_mem_bytes"):
                    x /= 1024.0 ** 3
                    x_std /= 1024.0 ** 3
                if not (math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0):
                    continue
                model = str(row["model_type"])
                label = f"{model}, M={row['m_states']}"
                xerr = None
                if math.isfinite(x_std) and x_std >= 0:
                    xerr = np.asarray([[min(x_std, 0.999 * x)], [x_std]])
                yerr = None
                if math.isfinite(y_std) and y_std >= 0:
                    # The exact SD remains in the CSV.  On a logarithmic axis
                    # a nonpositive mean-SD endpoint is represented by a
                    # truncated lower whisker rather than a fabricated value.
                    yerr = np.asarray([[min(y_std, 0.999 * y)], [y_std]])
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    color=colors.get(model, "#666666"),
                    marker=markers.get(model, "o"),
                    markersize=5.5,
                    capsize=2.0,
                    linewidth=0.8,
                    linestyle="none",
                    label=label,
                    zorder=3,
                )
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
            if row_index == 1:
                ax.set_xlabel(xlabel)
            if col_index == 0:
                ax.set_ylabel(f"Relative $L^2$ {ylabel.lower()} error")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        unique: Dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        fig.legend(
            list(unique.values()),
            list(unique.keys()),
            loc="upper center",
            ncol=max(1, min(6, len(unique))),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91 if handles else 1.0))
    for fmt in formats:
        fig.savefig(
            os.path.join(output_dir, f"e8_error_vs_compute.{fmt}"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = os.path.abspath(args.output or os.path.join(args.out_root, "compute_summary"))
    _check_output(output, bool(args.overwrite))
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown_models = sorted(set(requested_models) - {"pinn", "pipinn"})
    if not requested_models or unknown_models:
        raise ValueError(
            f"--models must select pinn and/or pipinn; unknown={unknown_models}"
        )
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = sorted(set(formats) - {"png", "pdf", "svg", "eps"})
    if not formats or unsupported:
        raise ValueError(f"unsupported or empty figure formats: {unsupported}")
    requested_dims = parse_seed_spec(args.m_states)
    observations, groups, skipped = collect_observations(
        args.out_root,
        headline_margin=float(args.headline_margin),
        require_timing_mode=not bool(args.include_main_runs),
        expected_models=requested_models,
        expected_m_states=requested_dims,
        expected_n_assets=args.n_assets,
        allow_unknown_runtime=bool(getattr(args, "allow_unknown_runtime", False)),
    )
    expected_seeds = parse_seed_spec(args.expected_seeds)
    validate_support(
        observations,
        expected_seeds=expected_seeds,
        min_runs=int(args.min_runs),
    )
    validate_expected_cells(
        observations,
        models=requested_models,
        m_states=requested_dims,
    )
    run_rows, long_rows, table_rows = build_output_rows(observations)

    run_fields = [
        "group_id", "model_type", "n_assets", "m_states", "seed", "gpu_name",
        "gpu_total_memory_bytes", "market_hash", *MEASURES, "run_dir",
    ]
    long_fields = [
        "group_id", "model_type", "n_assets", "m_states", "gpu_name",
        "gpu_total_memory_bytes", "measure",
        "n", "mean", "std", "sem", "ci95_lo", "ci95_hi", "seeds",
    ]
    table_fields = [
        "group_id", "model_type", "n_assets", "m_states", "gpu_name",
        "gpu_total_memory_bytes", "seeds",
    ] + [f"{measure}_{name}" for measure in MEASURES for name in ("n", "mean", "std", "ci95_lo", "ci95_hi")]
    metadata = {
        "out_root": os.path.abspath(args.out_root),
        "headline_margin": float(args.headline_margin),
        "timing_mode_required": not bool(args.include_main_runs),
        "expected_seeds": expected_seeds,
        "n_observations": len(observations),
        "n_groups": len(groups),
        "skipped_runs": skipped,
        "runtime_metadata_required": not bool(getattr(args, "allow_unknown_runtime", False)),
        "wall_clock_measure": "core_train_wall_sec (CUDA-synchronized; final checkpoint I/O excluded)",
        "single_seed_note": (
            "sample SD and confidence intervals are undefined and encoded as NaN when n=1"
        ),
    }

    parent = os.path.dirname(output) or os.curdir
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compute-summary-stage-", dir=parent) as stage:
        _write_csv(os.path.join(stage, "compute_runs.csv"), run_rows, run_fields)
        _write_csv(os.path.join(stage, "compute_summary_long.csv"), long_rows, long_fields)
        _write_csv(os.path.join(stage, "compute_table.csv"), table_rows, table_fields)
        with open(os.path.join(stage, "compute_groups.json"), "w", encoding="utf-8") as stream:
            json.dump(groups, stream, indent=2, sort_keys=True)
        with open(os.path.join(stage, "compute_metadata.json"), "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        if not args.skip_figure:
            _plot(
                table_rows,
                stage,
                formats=formats,
                fig_width=float(args.fig_width),
                fig_height=float(args.fig_height),
                font_size=float(args.font_size),
                dpi=int(args.dpi),
            )
        _commit_staged_output(stage, output, overwrite=bool(args.overwrite))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--models", default="pinn,pipinn")
    parser.add_argument("--m-states", default="1,3,5")
    parser.add_argument("--n-assets", type=int, default=30)
    parser.add_argument("--headline-margin", type=float, default=0.10)
    parser.add_argument(
        "--include-main-runs",
        action="store_true",
        help="Exploratory only: allow ordinary runs whose wall time includes diagnostics.",
    )
    parser.add_argument(
        "--allow-unknown-runtime",
        action="store_true",
        help="Exploratory legacy mode only: accept incomplete hardware metadata.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--fig-width", type=float, default=10.5)
    parser.add_argument("--fig-height", type=float, default=6.2)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = run(args)
    print(
        f"[done] E8 compute summary: {metadata['n_observations']} runs, "
        f"{metadata['n_groups']} groups"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
