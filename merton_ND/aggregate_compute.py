#!/usr/bin/env python3
"""Aggregate paper E8 compute-cost runs for the Merton experiments.

Only clean training runs produced with ``--timing-mode`` are eligible.  The
script intentionally does not infer or backfill either costs or errors:

* costs come from the four named fields in the successful ``status.json``;
* errors come from exact metric names in ``metrics.csv`` at the run's recorded
  primary full-dimensional evaluation margin;
* an ``e_Xev`` request is accepted only when ``metrics.csv`` itself contains
  an official full-dimensional ``e_Xev`` row.  ``outer_history.csv`` is never
  used as a fallback.

The strict separation is important for E8: training diagnostics are disabled
in timing mode, while the final error remains a normal held-out evaluation of
the official final model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    canonical_market_hash,
    parse_int_spec,
    parse_seed_spec,
    t_crit_95,
)


SCHEMA_VERSION = 1
COMPUTE_FIELDS: Mapping[str, Tuple[str, str]] = {
    "train_wall_sec": ("Training wall-clock", "seconds"),
    "total_optimizer_steps": ("Total optimizer steps", "steps"),
    "train_gpu_peak_mem_bytes": ("Training peak GPU memory", "bytes"),
    "eval_gpu_peak_mem_bytes": ("Evaluation peak GPU memory", "bytes"),
}
DEFAULT_ERROR_METRICS = ("RelL2_V", "RelL2_D", "RelL2_pi", "RelL2_c")
SUPPORTED_ERROR_METRICS = {
    "e_Xev", "RelL2_V", "RelL2_pi", "RelL2_c", "RelL2_D",
}

# These knobs cannot change the timed optimizer path (or are deliberately
# required below).  Every other argument remains in the method-configuration
# hash, so an accidental scientific/configuration change splits the group and
# fails validation instead of being averaged away.  Print cadences are *not*
# ignored: console I/O contributes to the observed wall-clock time.
METHOD_CONFIG_IGNORE = {
    "seed", "run_tag", "model_type", "device", "output_root", "weight_root",
    "stop_flag_path", "timing_mode", "skip_figures", "skip_plots",
    "save_iterate_every", "e3b_checkpoints", "eval_only",
    "allow_legacy_best_eval",
}


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    relative_run_dir: str
    method: str
    seed: int
    n_assets: int
    m_states: int
    outer_iters: int
    setting_id: str
    method_config_id: str
    market_hash: str
    primary_margin: float
    updated_at: str
    compute: Mapping[str, float]
    errors: Mapping[str, float]


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"not a boolean value: {value!r}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is missing or non-numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}")
    if positive and result <= 0.0:
        raise ValueError(f"{label} must be positive, got {result!r}")
    return result


def _parse_primary_margin(args: Mapping[str, Any], status: Mapping[str, Any], run: Path) -> float:
    raw = str(args.get("eval_margin", "")).strip()
    values: List[float] = []
    for token in raw.split(","):
        if token.strip():
            values.append(_finite_number(token, label=f"{run}: eval_margin"))
    if not values:
        raise ValueError(f"{run}: config has no eval_margin")
    recorded = _finite_number(
        status.get("primary_margin"), label=f"{run}: status.primary_margin"
    )
    if not math.isclose(values[0], recorded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{run}: primary margin mismatch: config first={values[0]:g}, "
            f"status={recorded:g}"
        )
    recorded_margins = status.get("eval_margins")
    if not isinstance(recorded_margins, list) or not recorded_margins:
        raise ValueError(f"{run}: status.eval_margins is missing or empty")
    status_margins = [
        _finite_number(x, label=f"{run}: status.eval_margins")
        for x in recorded_margins
    ]
    if len(status_margins) != len(values) or any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(status_margins, values)
    ):
        raise ValueError(
            f"{run}: status.eval_margins does not match training config: "
            f"{status_margins} != {values}"
        )
    return recorded


def _load_official_errors(
    run: Path,
    *,
    primary_margin: float,
    requested: Sequence[str],
) -> Dict[str, float]:
    path = run / "metrics.csv"
    if not path.is_file():
        raise ValueError(f"{run}: metrics.csv is missing")
    matches: Dict[str, List[float]] = {name: [] for name in requested}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"scope", "eval_margin", "metric", "value"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            if str(row.get("scope", "")) != "fulldim":
                continue
            try:
                margin = float(row["eval_margin"])
            except (TypeError, ValueError):
                continue
            if not math.isclose(margin, primary_margin, rel_tol=0.0, abs_tol=1e-12):
                continue
            name = str(row.get("metric", ""))
            if name not in matches:
                continue
            value = _finite_number(row.get("value"), label=f"{path}: {name}")
            if value < 0.0:
                raise ValueError(f"{path}: official error {name} is negative ({value})")
            matches[name].append(value)
    out: Dict[str, float] = {}
    for name in requested:
        values = matches[name]
        if len(values) != 1:
            hint = (
                " No outer_history fallback is permitted for e_Xev."
                if name == "e_Xev" else ""
            )
            raise ValueError(
                f"{run}: expected exactly one fulldim {name} row at primary "
                f"margin={primary_margin:g}, found {len(values)}.{hint}"
            )
        out[name] = values[0]
    return out


def _method_config_hash(args: Mapping[str, Any]) -> str:
    core = {
        key: args[key]
        for key in sorted(args)
        if key not in METHOD_CONFIG_IGNORE
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _setting_id(
    *, n_assets: int, m_states: int, market_hash: str,
    outer_iters: int, primary_margin: float, test_points: int,
) -> str:
    protocol = {
        "n_assets": n_assets,
        "m_states": m_states,
        "outer_iters": outer_iters,
        "market_hash": market_hash,
        "primary_margin": primary_margin,
        "test_points": test_points,
    }
    digest = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"N{n_assets}_M{m_states}_K{outer_iters}_{digest}"


def find_run_dirs(root: Path) -> List[Path]:
    return sorted(
        path.parent for path in root.rglob("config.json")
        if "compute_summary" not in path.parts
    )


def load_run(
    run: Path,
    *,
    out_root: Path,
    error_metrics: Sequence[str],
    allow_eval_figures: bool,
) -> RunRecord:
    config = _read_json(run / "config.json")
    args = config.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{run}: config.json.args must be an object")
    if not _truth(args.get("timing_mode", False)):
        raise ValueError(f"{run}: E8 requires config timing_mode=true")
    if _truth(args.get("eval_only", False)):
        raise ValueError(f"{run}: eval-only runs are not E8 training runs")
    if _truth(args.get("skip_eval", False)):
        raise ValueError(f"{run}: skip_eval=true leaves no official final error")
    if not allow_eval_figures and not (
        _truth(args.get("skip_figures", False))
        or _truth(args.get("skip_plots", False))
    ):
        raise ValueError(
            f"{run}: timing evaluation must set skip_figures=true (or skip_plots=true) "
            "so plotting cannot contaminate eval_gpu_peak_mem_bytes; pass "
            "--allow-eval-figures only for a documented legacy analysis"
        )
    if (run / "config_eval.json").exists() or (run / "_SUCCESS_EVAL").exists():
        raise ValueError(
            f"{run}: eval-only artifacts are present; E8 requires metrics and "
            "evaluation peak from the same clean timing run"
        )

    markers = [name for name in ("_SUCCESS", "_STOPPED_EARLY", "_FAILED") if (run / name).exists()]
    if markers != ["_SUCCESS"]:
        raise ValueError(f"{run}: expected only _SUCCESS, found {markers}")
    status = _read_json(run / "status.json")
    if str(status.get("status", "")) != "success":
        raise ValueError(f"{run}: status.json does not report success")
    if status.get("skipped_eval") is True:
        raise ValueError(f"{run}: status reports skipped_eval=true")

    method = str(args.get("model_type", "")).strip()
    if method not in {"pinn", "pipinn"}:
        raise ValueError(f"{run}: unsupported model_type={method!r}")
    if str(status.get("model_type", "")) != method:
        raise ValueError(
            f"{run}: status/config model_type mismatch "
            f"({status.get('model_type')!r} != {method!r})"
        )
    run_tag = str(args.get("run_tag", ""))
    if str(status.get("run_tag", "")) != run_tag:
        raise ValueError(f"{run}: status/config run_tag mismatch")

    try:
        seed = int(args["seed"])
        n_assets = int(args["n_assets"])
        m_states = int(args.get("m_states", 1))
        test_points = int(args["test_points"])
        outer_iters = int(args["outer_iters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{run}: missing/invalid seed, n_assets, m_states, or test_points") from exc
    if n_assets <= 0 or m_states != 1 or test_points <= 0 or outer_iters <= 0:
        raise ValueError(
            f"{run}: expected n_assets>0, Merton m_states=1, test_points>0, "
            f"outer_iters>0; got N={n_assets}, M={m_states}, "
            f"test_points={test_points}, outer_iters={outer_iters}"
        )

    compute: Dict[str, float] = {}
    for field in COMPUTE_FIELDS:
        value = _finite_number(status.get(field), label=f"{run}: status.{field}", positive=True)
        if field == "total_optimizer_steps" and not value.is_integer():
            raise ValueError(f"{run}: total_optimizer_steps must be an integer, got {value}")
        compute[field] = value

    primary_margin = _parse_primary_margin(args, status, run)
    errors = _load_official_errors(
        run, primary_margin=primary_margin, requested=error_metrics
    )
    market_path = run / "market_params.npz"
    try:
        market_hash = canonical_market_hash(str(market_path))
    except Exception as exc:
        raise ValueError(f"{run}: invalid canonical market snapshot: {exc}") from exc
    setting_id = _setting_id(
        n_assets=n_assets,
        m_states=m_states,
        outer_iters=outer_iters,
        market_hash=market_hash,
        primary_margin=primary_margin,
        test_points=test_points,
    )
    return RunRecord(
        run_dir=run,
        relative_run_dir=os.path.relpath(run, out_root),
        method=method,
        seed=seed,
        n_assets=n_assets,
        m_states=m_states,
        outer_iters=outer_iters,
        setting_id=setting_id,
        method_config_id=_method_config_hash(args),
        market_hash=market_hash,
        primary_margin=primary_margin,
        updated_at=str(status.get("updated_at", "")),
        compute=compute,
        errors=errors,
    )


def _summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("summary received empty or non-finite values")
    mean = float(arr.mean())
    if arr.size == 1:
        return {
            "n": 1, "mean": mean, "std": float("nan"),
            "sem": float("nan"), "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
        }
    std = float(arr.std(ddof=1))
    sem = std / math.sqrt(arr.size)
    half = t_crit_95(int(arr.size - 1)) * sem
    return {
        "n": int(arr.size), "mean": mean, "std": std, "sem": sem,
        "ci95_lo": mean - half, "ci95_hi": mean + half,
    }


def _display(mean: float, std: float) -> str:
    if not math.isfinite(std):
        return f"{mean:.6e} +/- NA (n=1)"
    return f"{mean:.6e} +/- {std:.6e}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def validate_panel(
    records: Sequence[RunRecord],
    *,
    expected_seeds: Sequence[int],
    expected_n_assets: Sequence[int],
    expected_methods: Sequence[str],
    min_seeds: int,
    require_sample_sd: bool,
) -> None:
    errors: List[str] = []
    expected_seed_set = set(expected_seeds)
    observed_ns = {record.n_assets for record in records}
    observed_methods = {record.method for record in records}
    if expected_n_assets and observed_ns != set(expected_n_assets):
        errors.append(f"observed N={sorted(observed_ns)}, expected exactly={sorted(expected_n_assets)}")
    if expected_methods and observed_methods != set(expected_methods):
        errors.append(
            f"observed methods={sorted(observed_methods)}, expected exactly={sorted(expected_methods)}"
        )
    markets_by_n: Dict[int, set] = defaultdict(set)
    for record in records:
        markets_by_n[record.n_assets].add(record.market_hash)
    for n_assets, hashes in sorted(markets_by_n.items()):
        if len(hashes) != 1:
            errors.append(
                f"N={n_assets}: distinct canonical Merton markets found: {sorted(hashes)}"
            )
        # More than one training budget (for example K=20 and K=30) is a
        # legitimate collection of E8 settings.  Each setting is validated
        # independently below; only the economic market must remain common.

    grouped: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.setting_id, record.method)].append(record)
    for (setting, method), rows in sorted(grouped.items()):
        seeds = [row.seed for row in rows]
        if len(seeds) != len(set(seeds)):
            errors.append(f"setting={setting} method={method}: duplicate seeds {sorted(seeds)}")
        if expected_seed_set and set(seeds) != expected_seed_set:
            errors.append(
                f"setting={setting} method={method}: seeds={sorted(set(seeds))}, "
                f"expected={sorted(expected_seed_set)}"
            )
        if len(set(seeds)) < min_seeds:
            errors.append(
                f"setting={setting} method={method}: only {len(set(seeds))} seeds, "
                f"minimum={min_seeds}"
            )
        if require_sample_sd and len(set(seeds)) < 2:
            errors.append(
                f"setting={setting} method={method}: sample SD requires at least two seeds"
            )
        configs = {row.method_config_id for row in rows}
        if len(configs) != 1:
            errors.append(
                f"setting={setting} method={method}: multiple method configurations "
                f"across seeds: {sorted(configs)}"
            )

    for setting in sorted({record.setting_id for record in records}):
        methods = {record.method for record in records if record.setting_id == setting}
        required = set(expected_methods) if expected_methods else observed_methods
        if methods != required:
            errors.append(
                f"setting={setting}: methods={sorted(methods)}, required={sorted(required)}"
            )
    if errors:
        raise ValueError("E8 panel validation failed:\n- " + "\n- ".join(errors))


def build_outputs(
    records: Sequence[RunRecord], error_metrics: Sequence[str]
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
]:
    per_run: List[Dict[str, Any]] = []
    for record in sorted(records, key=lambda x: (x.n_assets, x.setting_id, x.method, x.seed)):
        row: Dict[str, Any] = {
            "run_dir": record.relative_run_dir,
            "updated_at": record.updated_at,
            "setting_id": record.setting_id,
            "method_config_id": record.method_config_id,
            "model_type": record.method,
            "n_assets": record.n_assets,
            "m_states": record.m_states,
            "outer_iters": record.outer_iters,
            "seed": record.seed,
            "market_hash": record.market_hash,
            "primary_margin": record.primary_margin,
        }
        row.update(record.compute)
        row["train_gpu_peak_mem_mib"] = record.compute["train_gpu_peak_mem_bytes"] / 2**20
        row["eval_gpu_peak_mem_mib"] = record.compute["eval_gpu_peak_mem_bytes"] / 2**20
        row.update(record.errors)
        per_run.append(row)

    grouped: Dict[Tuple[str, str, int, str], List[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.setting_id, record.method, record.n_assets, record.method_config_id)].append(record)

    compute_summary: List[Dict[str, Any]] = []
    error_summary: List[Dict[str, Any]] = []
    compute_table: List[Dict[str, Any]] = []
    for (setting, method, n_assets, method_config), rows in sorted(grouped.items()):
        seeds = ",".join(str(x) for x in sorted(record.seed for record in rows))
        wide: Dict[str, Any] = {
            "setting_id": setting,
            "method_config_id": method_config,
            "model_type": method,
            "n_assets": n_assets,
            "outer_iters": rows[0].outer_iters,
            "n_seeds": len(rows),
            "seeds": seeds,
        }
        for metric, (label, unit) in COMPUTE_FIELDS.items():
            stats = _summary([record.compute[metric] for record in rows])
            compute_summary.append({
                "setting_id": setting,
                "method_config_id": method_config,
                "model_type": method,
                "n_assets": n_assets,
                "outer_iters": rows[0].outer_iters,
                "metric": metric,
                "label": label,
                "unit": unit,
                "seeds": seeds,
                **stats,
                "mean_plus_minus_sample_sd": _display(stats["mean"], stats["std"]),
            })
            scale = 2**20 if metric.endswith("mem_bytes") else 1.0
            suffix = "_mib" if scale != 1.0 else ""
            wide[f"{metric}_mean{suffix}"] = stats["mean"] / scale
            wide[f"{metric}_sample_sd{suffix}"] = stats["std"] / scale
            wide[f"{metric}_mean_plus_minus_sample_sd{suffix}"] = _display(
                stats["mean"] / scale, stats["std"] / scale
            )
        for metric in error_metrics:
            stats = _summary([record.errors[metric] for record in rows])
            error_summary.append({
                "setting_id": setting,
                "method_config_id": method_config,
                "model_type": method,
                "n_assets": n_assets,
                "outer_iters": rows[0].outer_iters,
                "metric": metric,
                "source": "metrics.csv:scope=fulldim, exact recorded primary_margin",
                "unit": "relative_error" if metric.startswith("RelL2") else "Xev_norm",
                "seeds": seeds,
                **stats,
                "mean_plus_minus_sample_sd": _display(stats["mean"], stats["std"]),
            })
        compute_table.append(wide)

    points: List[Dict[str, Any]] = []
    for record in records:
        for error_name in error_metrics:
            for compute_name, (_, unit) in COMPUTE_FIELDS.items():
                compute_value = record.compute[compute_name]
                plot_value = compute_value / 2**20 if compute_name.endswith("mem_bytes") else compute_value
                points.append({
                    "setting_id": record.setting_id,
                    "method_config_id": record.method_config_id,
                    "model_type": record.method,
                    "n_assets": record.n_assets,
                    "outer_iters": record.outer_iters,
                    "seed": record.seed,
                    "error_metric": error_name,
                    "error_value": record.errors[error_name],
                    "error_source": "official final metrics.csv fulldim primary-margin row",
                    "compute_metric": compute_name,
                    "compute_value_raw": compute_value,
                    "compute_value_plot": plot_value,
                    "compute_unit_raw": unit,
                    "compute_unit_plot": "MiB" if compute_name.endswith("mem_bytes") else unit,
                })
    point_summary: List[Dict[str, Any]] = []
    for (setting, method, n_assets, method_config), rows in sorted(grouped.items()):
        for error_name in error_metrics:
            error_stats = _summary([row.errors[error_name] for row in rows])
            for compute_name, (_, unit) in COMPUTE_FIELDS.items():
                scale = 2**20 if compute_name.endswith("mem_bytes") else 1.0
                compute_stats = _summary([
                    row.compute[compute_name] / scale for row in rows
                ])
                point_summary.append({
                    "setting_id": setting,
                    "method_config_id": method_config,
                    "model_type": method,
                    "n_assets": n_assets,
                    "outer_iters": rows[0].outer_iters,
                    "n_seeds": len(rows),
                    "seeds": ",".join(str(row.seed) for row in sorted(rows, key=lambda x: x.seed)),
                    "error_metric": error_name,
                    "error_mean": error_stats["mean"],
                    "error_sample_sd": error_stats["std"],
                    "compute_metric": compute_name,
                    "compute_mean_plot_unit": compute_stats["mean"],
                    "compute_sample_sd_plot_unit": compute_stats["std"],
                    "compute_unit_plot": "MiB" if scale != 1.0 else unit,
                })
    return per_run, compute_summary, compute_table, error_summary, points, point_summary


def _parse_formats(text: str) -> List[str]:
    out: List[str] = []
    for token in text.split(","):
        fmt = token.strip().lower().lstrip(".")
        if not fmt:
            continue
        if not re.fullmatch(r"[a-z0-9]+", fmt):
            raise ValueError(f"invalid figure format: {fmt!r}")
        if fmt not in out:
            out.append(fmt)
    if not out:
        raise ValueError("at least one figure format is required")
    return out


def _plot_outputs(
    output: Path,
    records: Sequence[RunRecord],
    error_metrics: Sequence[str],
    *, formats: Sequence[str], dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    groups: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
    for record in records:
        groups[(record.setting_id, record.method)].append(record)
    group_keys = sorted(
        groups,
        key=lambda key: (
            groups[key][0].n_assets,
            groups[key][0].outer_iters,
            key[1],
        ),
    )
    labels = [
        f"N={groups[key][0].n_assets}, K={groups[key][0].outer_iters}\n{key[1]}"
        for key in group_keys
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for ax, (metric, (label, unit)) in zip(axes.ravel(), COMPUTE_FIELDS.items()):
        means, sds = [], []
        scale = 2**20 if metric.endswith("mem_bytes") else 1.0
        for key in group_keys:
            stats = _summary([row.compute[metric] / scale for row in groups[key]])
            means.append(stats["mean"])
            sds.append(0.0 if not math.isfinite(stats["std"]) else stats["std"])
        x = np.arange(len(group_keys))
        ax.bar(x, means, yerr=sds, capsize=3, color="#4C78A8", alpha=0.85)
        ax.set_xticks(x, labels, rotation=0)
        ax.set_ylabel("MiB" if scale != 1.0 else unit)
        ax.set_title(label + " (mean +/- sample SD)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output / f"e8_compute_costs.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    colors = {"pinn": "#4C78A8", "pipinn": "#E45756"}
    for error_name in error_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
        for ax, (compute_name, (compute_label, compute_unit)) in zip(
            axes.ravel(), COMPUTE_FIELDS.items()
        ):
            scale = 2**20 if compute_name.endswith("mem_bytes") else 1.0
            for key in group_keys:
                rows = groups[key]
                x = np.asarray([row.compute[compute_name] / scale for row in rows])
                y = np.asarray([row.errors[error_name] for row in rows])
                method = key[1]
                label = f"N={rows[0].n_assets}, K={rows[0].outer_iters}, {method}"
                ax.scatter(x, y, s=24, alpha=0.38, color=colors.get(method), label=label)
                sx = float(x.std(ddof=1)) if x.size > 1 else 0.0
                sy = float(y.std(ddof=1)) if y.size > 1 else 0.0
                ax.errorbar(
                    [float(x.mean())], [float(y.mean())], xerr=[[sx], [sx]],
                    yerr=[[sy], [sy]], fmt="o", ms=5, capsize=3,
                    color=colors.get(method), linewidth=1.2,
                )
            if all(row.compute[compute_name] > 0 for row in records):
                ax.set_xscale("log")
            if all(row.errors[error_name] > 0 for row in records):
                ax.set_yscale("log")
            ax.set_xlabel(f"{compute_label} ({'MiB' if scale != 1.0 else compute_unit})")
            ax.set_ylabel(error_name)
            ax.grid(True, which="both", alpha=0.25)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=max(1, len(by_label)))
        fig.suptitle(
            f"E8 final {error_name} versus compute cost\n"
            "points=seeds; large markers=method/setting mean; bars=sample SD",
            y=1.03,
        )
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", error_name)
        for fmt in formats:
            fig.savefig(
                output / f"e8_error_vs_compute_{safe_name}.{fmt}",
                dpi=dpi, bbox_inches="tight",
            )
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate strict timing-mode Merton E8 compute-cost runs."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default=None,
                        help="default: <out-root>/compute_summary")
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--expected-n-assets", default="")
    parser.add_argument("--expected-methods", default="pinn,pipinn")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument(
        "--require-sample-sd", action="store_true",
        help="fail unless every method/setting has at least two seeds",
    )
    parser.add_argument(
        "--error-metrics", default=",".join(DEFAULT_ERROR_METRICS),
        help=("exact metrics.csv names; no fallback "
              "(default: RelL2_V,RelL2_D,RelL2_pi,RelL2_c)"),
    )
    parser.add_argument(
        "--allow-eval-figures", action="store_true",
        help="allow legacy timing runs whose evaluation peak may include plotting",
    )
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    out_root = Path(ns.out_root).expanduser().resolve()
    output = Path(ns.output).expanduser().resolve() if ns.output else out_root / "compute_summary"
    if not out_root.is_dir():
        raise ValueError(f"out root is not a directory: {out_root}")
    if ns.min_seeds < 1:
        raise ValueError("--min-seeds must be positive")
    if ns.dpi < 36:
        raise ValueError("--dpi must be at least 36")
    error_metrics = [x.strip() for x in ns.error_metrics.split(",") if x.strip()]
    if not error_metrics or len(error_metrics) != len(set(error_metrics)):
        raise ValueError("--error-metrics must contain unique non-empty names")
    unsupported = set(error_metrics) - SUPPORTED_ERROR_METRICS
    if unsupported:
        raise ValueError(f"unsupported error metrics: {sorted(unsupported)}")
    expected_methods = [x.strip() for x in ns.expected_methods.split(",") if x.strip()]
    if set(expected_methods) - {"pinn", "pipinn"}:
        raise ValueError(f"unsupported expected methods: {expected_methods}")
    expected_seeds = parse_seed_spec(ns.expected_seeds)
    expected_ns = parse_int_spec(ns.expected_n_assets, label="--expected-n-assets")
    formats = _parse_formats(ns.formats)

    run_dirs = find_run_dirs(out_root)
    if not run_dirs:
        raise ValueError(f"no config.json runs found under {out_root}")
    records = [
        load_run(
            run,
            out_root=out_root,
            error_metrics=error_metrics,
            allow_eval_figures=bool(ns.allow_eval_figures),
        )
        for run in run_dirs
    ]
    validate_panel(
        records,
        expected_seeds=expected_seeds,
        expected_n_assets=expected_ns,
        expected_methods=expected_methods,
        min_seeds=int(ns.min_seeds),
        require_sample_sd=bool(ns.require_sample_sd),
    )

    if output.exists() and any(output.iterdir()) and not ns.overwrite:
        raise ValueError(f"output directory is non-empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    (
        per_run, compute_summary, compute_table, error_summary, points,
        point_summary,
    ) = build_outputs(records, error_metrics)
    per_run_fields = list(per_run[0])
    _write_csv(output / "e8_per_run.csv", per_run, per_run_fields)
    summary_fields = [
        "setting_id", "method_config_id", "model_type", "n_assets", "outer_iters", "metric",
        "label", "source", "unit", "n", "mean", "std", "sem", "ci95_lo",
        "ci95_hi", "mean_plus_minus_sample_sd", "seeds",
    ]
    _write_csv(output / "e8_compute_summary.csv", compute_summary, summary_fields)
    _write_csv(output / "e8_error_summary.csv", error_summary, summary_fields)
    _write_csv(output / "e8_compute_table.csv", compute_table, list(compute_table[0]))
    _write_csv(output / "e8_error_vs_compute.csv", points, list(points[0]))
    _write_csv(
        output / "e8_error_vs_compute_summary.csv",
        point_summary,
        list(point_summary[0]),
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "out_root": str(out_root),
        "n_runs": len(records),
        "settings": sorted({record.setting_id for record in records}),
        "methods": sorted({record.method for record in records}),
        "seeds": sorted({record.seed for record in records}),
        "error_metrics": error_metrics,
        "error_semantics": {
            name: (
                "exactly one official final metrics.csv row with scope=fulldim at "
                "status.primary_margin; no cross-margin, outer-history, or prior-finite fallback"
            ) for name in error_metrics
        },
        "compute_semantics": {
            name: {"label": label, "unit": unit, "source": f"status.json:{name}"}
            for name, (label, unit) in COMPUTE_FIELDS.items()
        },
        "eligibility": {
            "timing_mode": True,
            "training_status_and_marker": "status=success and sole _SUCCESS marker",
            "skip_eval": False,
            "eval_only_artifacts": False,
            "evaluation_plots_disabled": not bool(ns.allow_eval_figures),
            "canonical_market_required": True,
            "one_method_configuration_per_setting_across_seeds": True,
        },
        "sample_sd": (
            "ddof=1; recorded as NaN/NA when n=1. Use --require-sample-sd for a paper "
            "panel that must estimate seed variability."
        ),
        "memory_display_unit": "MiB = bytes / 2^20; raw byte fields are retained",
        "hardware_caveat": (
            "The current trainer records CUDA peak bytes but not accelerator model. "
            "Run the timing panel on homogeneous GPUs and report the GPU model externally."
        ),
    }
    _write_json(output / "e8_metadata.json", metadata)
    if not ns.no_plots:
        _plot_outputs(
            output, records, error_metrics, formats=formats, dpi=int(ns.dpi)
        )
    print(f"[E8] runs={len(records)} settings={len(metadata['settings'])}")
    print(f"[E8] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
