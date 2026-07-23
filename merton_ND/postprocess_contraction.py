#!/usr/bin/env python3
"""Create the Merton paper Figure-2 empirical convergence trajectories.

The paper figure uses diagnostics written at every PI-PINN outer iteration on
the fixed held-out diagnostic set.  It plots pointwise arithmetic seed means
with matching plus/minus one sample-standard-deviation bands on a logarithmic
axis.  Individual seed trajectories are hidden by default.

The value curve is ``diag_RelL2_V``.  The default Policy curve is constructed
*within each seed and outer iteration* as

    sqrt((diag_RelL2_pi**2 + diag_RelL2_c**2) / 2).

``--policy-curve`` can instead select the portfolio curve, the consumption
curve, or show both components separately.  Irrespective of that plotting
choice, the raw value, portfolio, and consumption trajectories and the
derived policy-RMS trajectory are retained in the output CSV and metadata.

This is an empirical relative-L2 convergence diagnostic.  It does not read
``e_Xev``, does not form a one-step ratio, and is not an exact PI-map result.
Exact-map ratios remain a separate FD experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    canonical_market_hash,
    find_runs,
    group_key,
    load_config_args_raw,
    parse_seed_spec,
    run_status,
    run_updated_at,
    t_crit_95,
)


VALUE_METRIC = "diag_RelL2_V"
PI_METRIC = "diag_RelL2_pi"
CONSUMPTION_METRIC = "diag_RelL2_c"
POLICY_RMS_METRIC = "diag_RelL2_policy_rms"
RAW_METRICS = (VALUE_METRIC, PI_METRIC, CONSUMPTION_METRIC)
EXPORTED_METRICS = (*RAW_METRICS, POLICY_RMS_METRIC)

POLICY_CURVE_METRICS = {
    "rms": (VALUE_METRIC, POLICY_RMS_METRIC),
    "pi": (VALUE_METRIC, PI_METRIC),
    "c": (VALUE_METRIC, CONSUMPTION_METRIC),
    "separate": (VALUE_METRIC, PI_METRIC, CONSUMPTION_METRIC),
}

METRIC_LEGEND_LABELS = {
    VALUE_METRIC: "Value",
    PI_METRIC: "Portfolio",
    CONSUMPTION_METRIC: "Consumption",
    POLICY_RMS_METRIC: "Policy",
}

METRIC_DEFINITIONS = {
    VALUE_METRIC: "fixed-Q_ev relative L2 error of the value function",
    PI_METRIC: "fixed-Q_ev relative L2 error of the portfolio control",
    CONSUMPTION_METRIC: "fixed-Q_ev relative L2 error of the consumption control",
    POLICY_RMS_METRIC: (
        "within-seed, within-outer sqrt((diag_RelL2_pi^2 + "
        "diag_RelL2_c^2)/2); seed aggregation is applied only afterwards"
    ),
}

SUPPORTED_FORMATS = {"png", "pdf", "svg", "eps"}
OUTPUT_BASENAME = "figure2_empirical_convergence"
CURRENT_OUTPUT_FILES = {
    "figure2_trajectories.csv",
    "figure2_pointwise_summary.csv",
    "figure2_endpoint_summary.csv",
    "figure2_seed_decay_fits.csv",
    "figure2_decay_summary.csv",
    "figure2_runs_used.csv",
    "figure2_metadata.json",
}
LEGACY_OUTPUT_FILES = {
    "figure2_ratios.csv",
    "figure2_summary.csv",
    "figure2_worst_summary.csv",
    "supplemental_diagnostic_summary.csv",
    "postprocess_config.json",
    "figure2_contraction.png",
    "figure2_contraction.pdf",
    "figure2_contraction.svg",
    "figure2_contraction.eps",
}


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def primary_eval_margin(cfg: Mapping[str, Any]) -> float:
    raw = cfg.get("eval_margin", "0.10")
    if isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
    else:
        values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        values = [0.0]
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in values):
        raise ValueError(f"invalid eval_margin={raw!r}")
    return values[0]


def convergence_group_key(cfg: Mapping[str, Any]) -> str:
    """Training group plus the primary held-out diagnostic window."""
    training_group, _canonical = group_key(dict(cfg))
    payload = json.dumps(
        {
            "training_group": training_group,
            "primary_eval_margin": primary_eval_margin(cfg),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def parse_window(text: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[-:]\s*(\d+)\s*", str(text))
    if not match:
        raise ValueError(f"invalid fit window {text!r}; use START-END")
    start, end = int(match.group(1)), int(match.group(2))
    if start < 1 or end < start or end - start + 1 < 3:
        raise ValueError(
            f"invalid fit window {text!r}; require at least three points and 1 <= START <= END"
        )
    return start, end


def parse_windows(primary: str, sensitivity: str) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    primary_window = parse_window(primary)
    windows = [primary_window]
    for token in re.split(r"[;,]+", str(sensitivity or "")):
        if not token.strip():
            continue
        window = parse_window(token)
        if window not in windows:
            windows.append(window)
    return primary_window, sorted(windows, key=lambda item: (item[0], item[1]))


def format_window(window: Tuple[int, int]) -> str:
    return f"{window[0]}-{window[1]}"


def parse_formats(text: str) -> List[str]:
    formats = [part.lower() for part in re.split(r"[\s,]+", str(text)) if part]
    if not formats:
        raise ValueError("--formats must contain at least one format")
    if len(set(formats)) != len(formats):
        raise ValueError(f"duplicate formats in --formats={text!r}")
    invalid = sorted(set(formats) - SUPPORTED_FORMATS)
    if invalid:
        raise ValueError(f"unsupported figure formats: {invalid}")
    return formats


def owned_output_names() -> set[str]:
    names = set(CURRENT_OUTPUT_FILES) | set(LEGACY_OUTPUT_FILES)
    names.update(f"{OUTPUT_BASENAME}.{fmt}" for fmt in SUPPORTED_FORMATS)
    return names


def prepare_output(output: Path, overwrite: bool) -> None:
    """Create output safely; overwrite only artifacts owned by this script."""
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        return
    entries = list(output.iterdir())
    if not entries:
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --overwrite to replace "
            "only known Figure-2 artifacts"
        )
    owned = owned_output_names()
    blocked = [entry.name for entry in entries if entry.name in owned and not entry.is_file()]
    if blocked:
        raise ValueError(
            "refusing --overwrite because reserved output paths are not regular files: "
            f"{blocked}"
        )
    for entry in entries:
        if entry.name in owned:
            entry.unlink()


def read_outer_history(path: Path) -> Dict[str, Dict[int, float]]:
    """Read the three raw diagnostics, rejecting duplicate or malformed rows."""
    series: Dict[str, Dict[int, float]] = {metric: {} for metric in RAW_METRICS}
    seen_outer: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty outer history: {path}")
        required = ["outer_iter", *RAW_METRICS]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for row_number, row in enumerate(reader, start=2):
            outer = _int(row.get("outer_iter"))
            if outer is None:
                raise ValueError(f"{path}: invalid outer_iter on CSV row {row_number}")
            if outer in seen_outer:
                raise ValueError(f"{path}: duplicate outer_iter={outer}")
            seen_outer.add(outer)
            for metric in RAW_METRICS:
                value = _float(row.get(metric))
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}: metric={metric} is nonfinite at outer_iter={outer}"
                    )
                if value <= 0.0:
                    raise ValueError(
                        f"{path}: metric={metric} is nonpositive at outer_iter={outer}; "
                        "a logarithmic plot requires positive errors"
                    )
                series[metric][outer] = value
    return series


def discover_groups(
    out_root: Path,
    n_assets: int | None,
    outer_iters: int | None,
    primary_margin: float,
    run_name_regex: str,
) -> Dict[str, Dict[str, Any]]:
    """Find the newest PI-PINN run per seed and configuration.

    Deduplication precedes success filtering, so an older successful run cannot
    hide a newer failed rerun with the same seed and configuration.
    """
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int], Tuple[str, Path, Dict[str, Any], str]] = {}
    group_meta: Dict[str, Dict[str, Any]] = {}

    for run_dir_text in find_runs(str(out_root)):
        run_dir = Path(run_dir_text)
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None or str(cfg.get("model_type", "")) != "pipinn":
            continue
        m_states = _int(cfg.get("m_states", 1))
        if m_states != 1:
            continue
        cfg_n_assets = _int(cfg.get("n_assets"))
        if n_assets is not None and cfg_n_assets != n_assets:
            continue
        cfg_outer_iters = _int(cfg.get("outer_iters"))
        if outer_iters is not None and cfg_outer_iters != outer_iters:
            continue
        if pattern and not pattern.search(str(run_dir)):
            continue
        margin = primary_eval_margin(cfg)
        if not math.isclose(margin, primary_margin, rel_tol=0.0, abs_tol=1e-12):
            continue
        seed = _int(cfg.get("seed"))
        if seed is None:
            continue

        group = convergence_group_key(cfg)
        updated = run_updated_at(str(run_dir))
        status = run_status(str(run_dir))
        key = (group, seed)
        if key not in newest or updated >= newest[key][0]:
            newest[key] = (updated, run_dir, cfg, status)
        group_meta[group] = {
            "group": group,
            "model_type": "pipinn",
            "n_assets": cfg_n_assets,
            "m_states": 1,
            "outer_iters": cfg_outer_iters,
            "primary_eval_margin": margin,
        }

    groups: Dict[str, Dict[str, Any]] = {}
    for (group, seed), (updated, run_dir, cfg, status) in newest.items():
        entry = groups.setdefault(
            group,
            {
                **group_meta[group],
                "runs": {},
                "configs": {},
                "latest": {},
                "market_hashes": {},
                "market_errors": {},
            },
        )
        entry["latest"][seed] = {
            "run_dir": run_dir,
            "status": status,
            "updated_at": updated,
        }
        if status != "success":
            continue
        history_path = run_dir / "outer_history.csv"
        if not history_path.is_file():
            entry["latest"][seed]["status"] = "missing_outer_history"
            continue
        entry["runs"][seed] = run_dir
        entry["configs"][seed] = cfg
        try:
            entry["market_hashes"][seed] = canonical_market_hash(
                str(run_dir / "market_params.npz")
            )
            entry["market_errors"][seed] = ""
        except Exception as exc:
            entry["market_hashes"][seed] = ""
            entry["market_errors"][seed] = str(exc)
    return {group: meta for group, meta in groups.items() if meta["runs"]}


def select_group(groups: Mapping[str, Dict[str, Any]], group_id: str) -> Dict[str, Any]:
    if group_id:
        if group_id not in groups:
            raise ValueError(
                f"--group-id={group_id!r} not found; available groups={sorted(groups)}"
            )
        return groups[group_id]
    if len(groups) != 1:
        details = {
            group: {
                "n_assets": meta["n_assets"],
                "successful_seeds": sorted(meta["runs"]),
            }
            for group, meta in groups.items()
        }
        raise ValueError(
            "expected exactly one eligible Merton PI-PINN configuration; narrow the "
            f"selection with --group-id or --run-name-regex. Candidates: {details}"
        )
    return next(iter(groups.values()))


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("summary values must be nonempty and finite")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0, 0.0, float("nan"), float("nan")
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(int(arr.size))
    half = t_crit_95(int(arr.size) - 1) * sem
    return mean, std, sem, mean - half, mean + half


def validate_and_load(
    meta: Dict[str, Any],
    expected_seeds: set[int],
    min_seeds: int,
) -> Tuple[List[int], Dict[int, Dict[str, Dict[int, float]]], List[Dict[str, Any]]]:
    available = set(meta["runs"])
    if expected_seeds and available != expected_seeds:
        latest_status = {
            seed: str(record["status"])
            for seed, record in sorted(meta["latest"].items())
        }
        raise ValueError(
            f"group={meta['group']}: successful seeds={sorted(available)}, expected "
            f"exactly={sorted(expected_seeds)}; latest statuses={latest_status}"
        )
    seeds = sorted(expected_seeds if expected_seeds else available)
    if len(seeds) < min_seeds:
        raise ValueError(
            f"group={meta['group']}: found {len(seeds)} successful seeds, "
            f"but --min-seeds={min_seeds}"
        )

    market_errors = [
        (seed, meta["market_errors"].get(seed, "missing market hash"))
        for seed in seeds
        if not meta["market_hashes"].get(seed)
    ]
    if market_errors:
        raise ValueError(f"invalid Merton market snapshots: {market_errors}")
    hashes = {meta["market_hashes"][seed] for seed in seeds}
    if len(hashes) != 1:
        raise ValueError(
            f"selected seeds have {len(hashes)} distinct canonical Merton markets"
        )

    histories: Dict[int, Dict[str, Dict[int, float]]] = {}
    run_rows: List[Dict[str, Any]] = []
    common_outer_iters: int | None = None
    for seed in seeds:
        run_dir: Path = meta["runs"][seed]
        cfg = meta["configs"][seed]
        if convergence_group_key(cfg) != meta["group"]:
            raise ValueError(f"{run_dir}: configuration changed during selection")
        if _int(cfg.get("m_states", 1)) != 1:
            raise ValueError(f"{run_dir}: Merton requires m_states=1")
        diag_every = _int(cfg.get("diag_every"))
        if diag_every != 1:
            raise ValueError(
                f"{run_dir}: diag_every={diag_every}; Figure 2 requires diag_every=1"
            )
        outer_iters = _int(cfg.get("outer_iters"))
        if outer_iters is None or outer_iters < 3:
            raise ValueError(f"{run_dir}: invalid outer_iters={outer_iters}")
        if common_outer_iters is None:
            common_outer_iters = outer_iters
        elif outer_iters != common_outer_iters:
            raise ValueError(
                f"group={meta['group']}: seed={seed} has outer_iters={outer_iters}, "
                f"expected {common_outer_iters}"
            )

        history = read_outer_history(run_dir / "outer_history.csv")
        expected_outers = list(range(1, outer_iters + 1))
        for metric in RAW_METRICS:
            actual = sorted(history[metric])
            if actual != expected_outers:
                missing = sorted(set(expected_outers) - set(actual))
                extra = sorted(set(actual) - set(expected_outers))
                raise ValueError(
                    f"{run_dir}: metric={metric} must cover outer 1..{outer_iters} "
                    f"exactly; missing={missing}, extra={extra}"
                )
        history[POLICY_RMS_METRIC] = {
            outer: math.sqrt(
                (history[PI_METRIC][outer] ** 2 + history[CONSUMPTION_METRIC][outer] ** 2)
                / 2.0
            )
            for outer in expected_outers
        }
        histories[seed] = history
        run_rows.append(
            {
                "group": meta["group"],
                "model_type": meta["model_type"],
                "n_assets": meta["n_assets"],
                "m_states": 1,
                "seed": seed,
                "run_dir": str(run_dir),
                "outer_iters": outer_iters,
                "diag_every": diag_every,
                "primary_eval_margin": primary_eval_margin(cfg),
                "pi_init_method": str(cfg.get("pi_init_method", "")),
                "pi_init_scale": cfg.get("pi_init_scale", ""),
                "policy_bounds_mode": str(cfg.get("policy_bounds_mode", "")),
                "market_hash": meta["market_hashes"][seed],
            }
        )
    return seeds, histories, run_rows


def fit_log_linear(series: Mapping[int, float], window: Tuple[int, int]) -> Dict[str, float | int]:
    outers = list(range(window[0], window[1] + 1))
    missing = [outer for outer in outers if outer not in series]
    if missing:
        raise ValueError(f"fit window {format_window(window)} is missing outer={missing}")
    values = np.asarray([series[outer] for outer in outers], dtype=float)
    x = np.asarray(outers, dtype=float)
    y = np.log(values)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = float("nan") if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot
    return {
        "intercept": float(intercept),
        "log_rho": float(slope),
        "rho": float(math.exp(float(slope))),
        "r_squared": r_squared,
        "n_points": len(outers),
        "start_value": float(values[0]),
        "end_value": float(values[-1]),
        "observed_reduction_factor": float(values[0] / values[-1]),
    }


def build_tables(
    meta: Mapping[str, Any],
    histories: Mapping[int, Mapping[str, Mapping[int, float]]],
    seeds: Sequence[int],
    plotted_metrics: Sequence[str],
    windows: Sequence[Tuple[int, int]],
    primary_window: Tuple[int, int],
    write_decay_fits: bool,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    trajectory_rows: List[Dict[str, Any]] = []
    pointwise_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    decay_rows: List[Dict[str, Any]] = []
    identity = {
        "group": meta["group"],
        "model_type": meta["model_type"],
        "n_assets": meta["n_assets"],
        "m_states": 1,
    }

    for metric in EXPORTED_METRICS:
        outer_grid = sorted(histories[seeds[0]][metric])
        is_plotted = int(metric in plotted_metrics)
        for seed in seeds:
            for outer in outer_grid:
                trajectory_rows.append(
                    {
                        **identity,
                        "seed": seed,
                        "metric": metric,
                        "metric_label": METRIC_LEGEND_LABELS[metric],
                        "is_plotted": is_plotted,
                        "outer_iter": outer,
                        "value": histories[seed][metric][outer],
                    }
                )
        for outer in outer_grid:
            values = [histories[seed][metric][outer] for seed in seeds]
            mean, std, sem, ci_low, ci_high = mean_std_ci(values)
            pointwise_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "metric_label": METRIC_LEGEND_LABELS[metric],
                    "is_plotted": is_plotted,
                    "outer_iter": outer,
                    "n_seeds": len(seeds),
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

        if not write_decay_fits:
            continue
        for window in windows:
            window_label = format_window(window)
            per_seed_fits: List[Dict[str, Any]] = []
            for seed in seeds:
                fit = fit_log_linear(histories[seed][metric], window)
                row = {
                    **identity,
                    "seed": seed,
                    "metric": metric,
                    "metric_label": METRIC_LEGEND_LABELS[metric],
                    "is_plotted": is_plotted,
                    "estimate_kind": "relative_L2_error_decay",
                    "fit_window": window_label,
                    "outer_start": window[0],
                    "outer_end": window[1],
                    "is_primary": int(window == primary_window),
                    **fit,
                }
                fit_rows.append(row)
                per_seed_fits.append(row)
            rho = [float(row["rho"]) for row in per_seed_fits]
            log_rho = [float(row["log_rho"]) for row in per_seed_fits]
            rho_stats = mean_std_ci(rho)
            log_stats = mean_std_ci(log_rho)
            decay_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "metric_label": METRIC_LEGEND_LABELS[metric],
                    "is_plotted": is_plotted,
                    "estimate_kind": "relative_L2_error_decay",
                    "fit_window": window_label,
                    "outer_start": window[0],
                    "outer_end": window[1],
                    "is_primary": int(window == primary_window),
                    "n_seeds": len(seeds),
                    "rho_mean": rho_stats[0],
                    "rho_std": rho_stats[1],
                    "rho_sem": rho_stats[2],
                    "rho_ci95_low": rho_stats[3],
                    "rho_ci95_high": rho_stats[4],
                    "log_rho_mean": log_stats[0],
                    "log_rho_std": log_stats[1],
                    "log_rho_sem": log_stats[2],
                    "log_rho_ci95_low": log_stats[3],
                    "log_rho_ci95_high": log_stats[4],
                }
            )
    return trajectory_rows, pointwise_rows, fit_rows, decay_rows


def build_endpoint_summary(
    meta: Mapping[str, Any],
    pointwise_rows: Sequence[Mapping[str, Any]],
    plotted_metrics: Sequence[str],
    outer_end: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in EXPORTED_METRICS:
        by_outer = {
            int(row["outer_iter"]): row
            for row in pointwise_rows
            if row["metric"] == metric
        }
        start_row, end_row = by_outer[1], by_outer[outer_end]
        mean_start = float(start_row["mean"])
        mean_end = float(end_row["mean"])
        rows.append(
            {
                "group": meta["group"],
                "model_type": meta["model_type"],
                "n_assets": meta["n_assets"],
                "m_states": 1,
                "metric": metric,
                "metric_label": METRIC_LEGEND_LABELS[metric],
                "is_plotted": int(metric in plotted_metrics),
                "outer_start": 1,
                "outer_end": outer_end,
                "n_seeds": int(start_row["n_seeds"]),
                "mean_start": mean_start,
                "std_start": float(start_row["std"]),
                "mean_end": mean_end,
                "std_end": float(end_row["std"]),
                "seed_mean_reduction_factor": mean_start / mean_end,
            }
        )
    return rows


def log_sd_band(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact mean +/- SD band where a log axis can display it."""
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if mean_array.shape != std_array.shape:
        raise ValueError("mean and std must have matching shapes")
    if np.any(~np.isfinite(mean_array)) or np.any(mean_array <= 0.0):
        raise ValueError("mean must contain positive finite values")
    if np.any(~np.isfinite(std_array)) or np.any(std_array < 0.0):
        raise ValueError("std must contain nonnegative finite values")
    raw_lower = mean_array - std_array
    return np.where(raw_lower > 0.0, raw_lower, np.nan), mean_array + std_array


def create_figure(
    trajectory_rows: Sequence[Mapping[str, Any]],
    pointwise_rows: Sequence[Mapping[str, Any]],
    plotted_metrics: Sequence[str],
    show_seed_trajectories: bool,
    figure_size: Tuple[float, float],
    font_size: float,
    font_family: str,
    outer_end: int,
    eps_compatible: bool = False,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.ticker import MaxNLocator

    colors = {
        VALUE_METRIC: "#0072B2",
        POLICY_RMS_METRIC: "#D55E00",
        PI_METRIC: "#CC79A7",
        CONSUMPTION_METRIC: "#009E73",
    }

    def lighten(color: str, white_fraction: float) -> Tuple[float, float, float]:
        rgb = np.asarray(to_rgb(color), dtype=float)
        return tuple((1.0 - white_fraction) * rgb + white_fraction)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    for metric in plotted_metrics:
        color = colors[metric]
        if show_seed_trajectories:
            metric_rows = [row for row in trajectory_rows if row["metric"] == metric]
            for seed in sorted({int(row["seed"]) for row in metric_rows}):
                rows = sorted(
                    [
                        row for row in metric_rows
                        if int(row["seed"]) == seed
                        and 1 <= int(row["outer_iter"]) <= outer_end
                    ],
                    key=lambda row: int(row["outer_iter"]),
                )
                ax.plot(
                    [int(row["outer_iter"]) for row in rows],
                    [float(row["value"]) for row in rows],
                    color=lighten(color, 0.65) if eps_compatible else color,
                    alpha=1.0 if eps_compatible else 0.22,
                    linewidth=0.8,
                )

        summary = sorted(
            [
                row for row in pointwise_rows
                if row["metric"] == metric and 1 <= int(row["outer_iter"]) <= outer_end
            ],
            key=lambda row: int(row["outer_iter"]),
        )
        x = np.asarray([row["outer_iter"] for row in summary], dtype=float)
        mean = np.asarray([row["mean"] for row in summary], dtype=float)
        std = np.asarray([row["std"] for row in summary], dtype=float)
        lower, upper = log_sd_band(mean, std)
        ax.fill_between(
            x,
            lower,
            upper,
            color=lighten(color, 0.80) if eps_compatible else color,
            alpha=1.0 if eps_compatible else 0.18,
            linewidth=0.0,
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.2,
            label=METRIC_LEGEND_LABELS[metric],
        )

    label_font = {"fontfamily": font_family} if font_family else {}
    ax.set_yscale("log")
    ax.set_xlim(1, outer_end)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
    ax.set_xlabel("Outer iteration", fontsize=font_size, **label_font)
    ax.set_ylabel(r"Relative $L^2$ error", fontsize=font_size, **label_font)
    ax.tick_params(axis="both", labelsize=0.9 * font_size)
    if font_family:
        for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick_label.set_fontfamily(font_family)
    grid_kwargs: Dict[str, Any] = {
        "which": "both",
        "alpha": 1.0 if eps_compatible else 0.22,
        "linewidth": 0.6,
    }
    if eps_compatible:
        grid_kwargs["color"] = "#D9D9D9"
    ax.grid(True, **grid_kwargs)
    legend_font: Dict[str, Any] = {"size": 0.8 * font_size}
    if font_family:
        legend_font["family"] = font_family
    ax.legend(frameon=False, prop=legend_font, loc="best")
    fig.tight_layout()
    return fig


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Merton PI-PINN value/policy relative-L2 convergence using "
            "pointwise seed means and sample-SD bands."
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--output", default="", help="Default: <out-root>/figure2_empirical_convergence"
    )
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument(
        "--outer-iters", type=int, default=None,
        help="Select one training budget when the sweep root contains multiple budgets.",
    )
    parser.add_argument(
        "--m-states", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-seeds",
        default="",
        help="Exact arbitrary seed set, e.g. '1,2,3,5,7,11,17,23,42,101'.",
    )
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--primary-margin", type=float, default=0.10)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--group-id", default="")
    parser.add_argument(
        "--policy-curve",
        choices=("rms", "pi", "c", "separate"),
        default="rms",
        help=(
            "Policy display: RMS of portfolio/consumption RelL2 (default), "
            "portfolio only, consumption only, or both components."
        ),
    )
    parser.add_argument("--fit-window", default="1-4")
    parser.add_argument(
        "--sensitivity-windows",
        default="1-3,1-5",
        help="Additional optional per-seed decay-fit windows.",
    )
    parser.add_argument(
        "--skip-decay-fits",
        action="store_true",
        help="Do not write the optional per-seed decay-fit and fit-summary CSVs.",
    )
    parser.add_argument(
        "--endpoint-outer",
        type=int,
        default=20,
        help="Final plotted outer and E1/E_end endpoint (paper default: 20).",
    )
    parser.add_argument(
        "--hide-seed-trajectories",
        action="store_true",
        help="Compatibility flag; seed curves are already hidden by default.",
    )
    parser.add_argument(
        "--show-seed-trajectories",
        action="store_true",
        help="Diagnostic opt-in; individual seed curves are omitted by default.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=4.8)
    parser.add_argument("--fig-height", type=float, default=3.2)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--font-family", default="")
    parser.add_argument(
        "--bbox-inches", choices=("tight", "standard"), default="tight"
    )
    args = parser.parse_args(argv)

    if args.m_states is not None and args.m_states != 1:
        raise ValueError("Merton has one PDE state; deprecated --m-states, if used, must be 1")
    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if args.outer_iters is not None and args.outer_iters < 1:
        raise ValueError("--outer-iters must be positive")
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2 for a sample-SD band")
    if not 0.0 <= args.primary_margin < 1.0:
        raise ValueError("--primary-margin must be in [0,1)")
    if args.endpoint_outer <= 1:
        raise ValueError("--endpoint-outer must be greater than 1")
    for name, value in (
        ("--dpi", args.dpi),
        ("--fig-width", args.fig_width),
        ("--fig-height", args.fig_height),
        ("--font-size", args.font_size),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    formats = parse_formats(args.formats)
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    plotted_metrics = list(POLICY_CURVE_METRICS[args.policy_curve])
    primary_window, windows = parse_windows(args.fit_window, args.sensitivity_windows)

    out_root = Path(args.out_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_root / OUTPUT_BASENAME
    )
    groups = discover_groups(
        out_root=out_root,
        n_assets=args.n_assets,
        outer_iters=args.outer_iters,
        primary_margin=args.primary_margin,
        run_name_regex=args.run_name_regex,
    )
    if not groups:
        raise SystemExit("no eligible successful Merton PI-PINN runs were found")
    meta = select_group(groups, args.group_id)
    seeds, histories, run_rows = validate_and_load(
        meta, expected_seeds, args.min_seeds
    )
    available_outer = max(histories[seeds[0]][VALUE_METRIC])
    if args.endpoint_outer > available_outer:
        raise ValueError(
            f"--endpoint-outer={args.endpoint_outer}, but histories end at {available_outer}"
        )
    if not args.skip_decay_fits:
        max_fit_outer = max(window[1] for window in windows)
        if max_fit_outer > available_outer:
            raise ValueError(
                f"fit windows require outer={max_fit_outer}, histories end at {available_outer}"
            )

    trajectory_rows, pointwise_rows, fit_rows, decay_rows = build_tables(
        meta,
        histories,
        seeds,
        plotted_metrics,
        windows,
        primary_window,
        write_decay_fits=not args.skip_decay_fits,
    )
    endpoint_rows = build_endpoint_summary(
        meta, pointwise_rows, plotted_metrics, args.endpoint_outer
    )
    sd_band_masked_points = {
        metric: sum(
            1
            for row in pointwise_rows
            if row["metric"] == metric
            and 1 <= int(row["outer_iter"]) <= args.endpoint_outer
            and float(row["mean"]) - float(row["std"]) <= 0.0
        )
        for metric in plotted_metrics
    }

    prepare_output(output, args.overwrite)
    common_identity_fields = ["group", "model_type", "n_assets", "m_states"]
    write_csv(
        output / "figure2_trajectories.csv",
        trajectory_rows,
        [
            *common_identity_fields, "seed", "metric", "metric_label", "is_plotted",
            "outer_iter", "value",
        ],
    )
    write_csv(
        output / "figure2_pointwise_summary.csv",
        pointwise_rows,
        [
            *common_identity_fields, "metric", "metric_label", "is_plotted",
            "outer_iter", "n_seeds", "mean", "std", "sem", "ci95_low", "ci95_high",
        ],
    )
    write_csv(
        output / "figure2_endpoint_summary.csv",
        endpoint_rows,
        [
            *common_identity_fields, "metric", "metric_label", "is_plotted",
            "outer_start", "outer_end", "n_seeds", "mean_start", "std_start",
            "mean_end", "std_end", "seed_mean_reduction_factor",
        ],
    )
    if not args.skip_decay_fits:
        write_csv(
            output / "figure2_seed_decay_fits.csv",
            fit_rows,
            [
                *common_identity_fields, "seed", "metric", "metric_label", "is_plotted",
                "estimate_kind", "fit_window", "outer_start", "outer_end", "is_primary",
                "n_points", "intercept", "log_rho", "rho", "r_squared", "start_value",
                "end_value", "observed_reduction_factor",
            ],
        )
        write_csv(
            output / "figure2_decay_summary.csv",
            decay_rows,
            [
                *common_identity_fields, "metric", "metric_label", "is_plotted",
                "estimate_kind", "fit_window", "outer_start", "outer_end", "is_primary",
                "n_seeds", "rho_mean", "rho_std", "rho_sem", "rho_ci95_low",
                "rho_ci95_high", "log_rho_mean", "log_rho_std", "log_rho_sem",
                "log_rho_ci95_low", "log_rho_ci95_high",
            ],
        )
    write_csv(
        output / "figure2_runs_used.csv",
        run_rows,
        [
            *common_identity_fields, "seed", "run_dir", "outer_iters", "diag_every",
            "primary_eval_margin", "pi_init_method", "pi_init_scale",
            "policy_bounds_mode", "market_hash",
        ],
    )

    metadata = {
        "arguments": vars(args),
        "selected_group": meta["group"],
        "selected_seeds": seeds,
        "selected_market_hash": run_rows[0]["market_hash"],
        "merton_state_dimension": 1,
        "policy_curve": args.policy_curve,
        "plotted_metrics": plotted_metrics,
        "exported_metrics": list(EXPORTED_METRICS),
        "raw_component_metrics_always_exported": list(RAW_METRICS),
        "metric_definitions": METRIC_DEFINITIONS,
        "trajectory_aggregation": (
            "pointwise arithmetic seed mean and sample SD on each within-seed metric"
        ),
        "endpoint_summary": {
            "outer_start": 1,
            "outer_end": args.endpoint_outer,
            "definition": "seed mean at outer 1 divided by seed mean at endpoint outer",
            "rows": endpoint_rows,
        },
        "decay_fits_written": not args.skip_decay_fits,
        "primary_fit_window": (
            format_window(primary_window) if not args.skip_decay_fits else None
        ),
        "fit_windows": (
            [format_window(window) for window in windows]
            if not args.skip_decay_fits else []
        ),
        "decay_estimation": (
            "optional diagnostic: fit log(error)=alpha+n*log(rho) separately per seed, "
            "then aggregate seed-wise rho; no fit is drawn or annotated"
            if not args.skip_decay_fits else "disabled"
        ),
        "sd_band_log_display": (
            "mean-SD is unmodified where positive; nonpositive lower endpoints are "
            "masked because a logarithmic axis cannot display them; all saved "
            "statistics remain unmodified"
        ),
        "sd_band_masked_lower_points": sd_band_masked_points,
        "e_Xev_used": False,
        "one_step_ratio_used": False,
        "floor_filter_used": False,
        "exact_map_used": False,
        "interpretation_limit": (
            "Figure 2 is empirical relative-L2 convergence, not an X_ev contraction "
            "factor; exact-map evidence is generated by the separate FD workflow"
        ),
        "figure_has_fit_annotations": False,
        "figure_has_individual_seed_trajectories": bool(
            args.show_seed_trajectories and not args.hide_seed_trajectories
        ),
        "figure_style": {
            "width_inches": args.fig_width,
            "height_inches": args.fig_height,
            "dpi": args.dpi,
            "font_size_points": args.font_size,
            "font_family": args.font_family or "matplotlib_default",
            "bbox_inches": args.bbox_inches,
            "formats": formats,
        },
    }
    with (output / "figure2_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if not args.no_plots:
        figure_kwargs = {
            "trajectory_rows": trajectory_rows,
            "pointwise_rows": pointwise_rows,
            "plotted_metrics": plotted_metrics,
            "show_seed_trajectories": bool(
                args.show_seed_trajectories and not args.hide_seed_trajectories
            ),
            "figure_size": (args.fig_width, args.fig_height),
            "font_size": args.font_size,
            "font_family": args.font_family,
            "outer_end": args.endpoint_outer,
        }
        regular_formats = [fmt for fmt in formats if fmt != "eps"]
        if regular_formats:
            fig = create_figure(**figure_kwargs)
            for fmt in regular_formats:
                fig.savefig(
                    output / f"{OUTPUT_BASENAME}.{fmt}",
                    dpi=args.dpi,
                    bbox_inches="tight" if args.bbox_inches == "tight" else None,
                )
            import matplotlib.pyplot as plt
            plt.close(fig)
        if "eps" in formats:
            fig = create_figure(**figure_kwargs, eps_compatible=True)
            fig.savefig(
                output / f"{OUTPUT_BASENAME}.eps",
                dpi=args.dpi,
                bbox_inches="tight" if args.bbox_inches == "tight" else None,
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    print(f"[done] Merton Figure-2 outputs: {output}")
    print(
        f"[info] seeds={seeds}; policy_curve={args.policy_curve}; "
        f"plotted outer=1-{args.endpoint_outer}"
    )
    for row in endpoint_rows:
        if not int(row["is_plotted"]):
            continue
        print(
            f"[endpoint] {row['metric_label']}: {float(row['mean_start']):.3e} -> "
            f"{float(row['mean_end']):.3e} "
            f"({float(row['seed_mean_reduction_factor']):.3f}x reduction)"
        )
    masked_total = sum(sd_band_masked_points.values())
    if masked_total:
        print(
            f"[warn] mean-SD was nonpositive at {masked_total} plotted points; "
            "those lower band endpoints were masked on the log axis"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
