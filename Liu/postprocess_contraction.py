#!/usr/bin/env python3
"""Create the Liu Figure-2 empirical convergence trajectories.

The main figure uses quantities recorded at every PI outer iteration on their
fixed held-out diagnostic or validation sets:

* ``diag_RelL2_V``
* ``diag_RelL2_vartheta`` for the wealth-normalized policy
* ``val_pres`` only when ``--include-val-pres`` is requested

The paper figure places the pointwise arithmetic seed means for Value and
Policy on one logarithmic axis, with matching +/- one sample-SD bands.  It
omits individual seed curves, an early-window highlight, and fitted-rho
annotations.  The transparent seed-mean endpoint reduction E_1/E_20 is
written separately for the main text.

For optional diagnostic CSVs, an early-phase decay factor is estimated by
fitting each seed separately,

    log(e_n) = alpha + n log(rho),

over a fixed outer-iteration window.  Only the resulting seed-wise ``rho``
values are aggregated.  The seed-mean curve is never fitted.

This is an empirical convergence diagnostic in relative L2 error.  It is not
an estimate of the X_ev-norm contraction factor in the theorem, and it is not
an exact policy-iteration map calculation.  In particular, ``e_Xev`` is not
read or used by this script.
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


POLICY_METRIC = "diag_RelL2_vartheta"
LEGACY_RAW_POLICY_METRIC = "diag_RelL2_theta"
MAIN_METRICS = ("diag_RelL2_V", POLICY_METRIC)
OPTIONAL_METRIC = "val_pres"

METRIC_LEGEND_LABELS = {
    "diag_RelL2_V": "Value",
    POLICY_METRIC: "Policy",
    "val_pres": r"$p_{\mathrm{res}}$",
}

SUPPORTED_FORMATS = {"png", "pdf", "svg", "eps"}

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
        raise ValueError(f"invalid fit window {text!r}; use START-END, for example 1-4")
    start, end = int(match.group(1)), int(match.group(2))
    if start < 1 or end < start:
        raise ValueError(f"invalid fit window {text!r}; require 1 <= START <= END")
    if end - start + 1 < 3:
        raise ValueError(f"fit window {text!r} has fewer than three outer iterations")
    return start, end


def parse_windows(primary: str, sensitivity: str) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    primary_window = parse_window(primary)
    sensitivity_windows: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for token in re.split(r"[;,]+", str(sensitivity or "")):
        if not token.strip():
            continue
        window = parse_window(token)
        if window in seen:
            raise ValueError(f"duplicate sensitivity fit window: {token.strip()}")
        seen.add(window)
        sensitivity_windows.append(window)
    windows = [primary_window]
    windows.extend(window for window in sensitivity_windows if window != primary_window)
    windows = sorted(windows, key=lambda item: (item[0], item[1]))
    return primary_window, windows


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_vartheta_overlay(
    path: Path,
    meta: Mapping[str, Any],
) -> Tuple[Dict[int, Dict[int, float]], Dict[str, Any]]:
    """Load a checkpoint-reconstructed policy trajectory without touching runs.

    The companion provenance document and success marker are mandatory.  Each
    selected row is tied to the exact source config, outer history, market
    snapshot, and iterate checkpoint, so a stale derived trajectory cannot be
    silently applied to a newer run.
    """

    path = path.expanduser().resolve()
    provenance_path = path.with_name("figure2_vartheta_provenance.json")
    success_path = path.with_name("_SUCCESS_FIGURE2_VARTTHETA")
    for required in (path, provenance_path, success_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing reconstructed-policy artifact: {required}")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if not isinstance(provenance, dict):
        raise ValueError(f"{provenance_path}: expected a JSON object")
    recorded_csv_hash = str(
        provenance.get("artifact_sha256", {}).get(path.name, "")
    )
    actual_csv_hash = sha256_file(path)
    if not recorded_csv_hash or recorded_csv_hash != actual_csv_hash:
        raise ValueError(
            f"{path}: SHA-256 disagrees with {provenance_path}; derived artifact is "
            "incomplete or has been modified"
        )

    selected_runs = {
        int(seed): Path(run_dir).resolve()
        for seed, run_dir in meta["runs"].items()
    }
    selected_market_hashes = {
        int(seed): str(value) for seed, value in meta["market_hashes"].items()
    }
    series: Dict[int, Dict[int, float]] = {}
    hash_cache: Dict[Path, str] = {path: actual_csv_hash}
    required_fields = {
        "group", "training_seed", "outer_iter", "metric", "value", "run_dir",
        "config_sha256", "outer_history_sha256", "market_hash", "checkpoint",
        "checkpoint_sha256", "market_params", "market_params_file_sha256",
        "closed_form", "closed_form_file_sha256", "closed_form_hash",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_fields - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for row in reader:
            if str(row.get("group", "")) != str(meta["group"]):
                continue
            seed = _int(row.get("training_seed"))
            if seed is None or seed not in selected_runs:
                continue
            if str(row.get("metric", "")) != POLICY_METRIC:
                raise ValueError(
                    f"{path}: unexpected metric for selected seed={seed}: "
                    f"{row.get('metric')!r}"
                )
            recorded_run = Path(str(row["run_dir"])).expanduser().resolve()
            run_dir = selected_runs[seed]
            if recorded_run != run_dir:
                raise ValueError(
                    f"{path}: seed={seed} was reconstructed from {recorded_run}, "
                    f"but the selected newest run is {run_dir}"
                )
            expected_sources = (
                (run_dir / "config.json", str(row["config_sha256"])),
                (run_dir / "outer_history.csv", str(row["outer_history_sha256"])),
                (Path(str(row["market_params"])).expanduser().resolve(),
                 str(row["market_params_file_sha256"])),
                (Path(str(row["closed_form"])).expanduser().resolve(),
                 str(row["closed_form_file_sha256"])),
                (Path(str(row["checkpoint"])).expanduser().resolve(),
                 str(row["checkpoint_sha256"])),
            )
            for source, expected_hash in expected_sources:
                if not source.is_file():
                    raise FileNotFoundError(
                        f"{path}: source recorded for seed={seed} is missing: {source}"
                    )
                if source not in hash_cache:
                    hash_cache[source] = sha256_file(source)
                if not expected_hash or hash_cache[source] != expected_hash:
                    raise ValueError(
                        f"{path}: source SHA-256 changed for seed={seed}: {source}"
                    )
            if str(row["market_hash"]) != selected_market_hashes.get(seed, ""):
                raise ValueError(
                    f"{path}: canonical market hash mismatch for seed={seed}"
                )
            outer = _int(row.get("outer_iter"))
            value = _float(row.get("value"))
            if outer is None or outer < 1 or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{path}: invalid reconstructed value at seed={seed}, outer={outer}"
                )
            if outer in series.setdefault(seed, {}):
                raise ValueError(
                    f"{path}: duplicate reconstructed row for seed={seed}, outer={outer}"
                )
            series[seed][outer] = value
    if not series:
        raise ValueError(
            f"{path}: no rows match selected Figure-2 group={meta['group']}"
        )
    return series, {
        "csv": str(path),
        "csv_sha256": actual_csv_hash,
        "provenance": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "success_marker": str(success_path),
    }


def owned_output_names() -> set[str]:
    names = set(CURRENT_OUTPUT_FILES) | set(LEGACY_OUTPUT_FILES)
    for fmt in SUPPORTED_FORMATS:
        names.add(f"figure2_empirical_convergence.{fmt}")
        names.add(f"figure2_contraction.{fmt}")
        for metric in (*MAIN_METRICS, LEGACY_RAW_POLICY_METRIC, OPTIONAL_METRIC):
            names.add(f"supplemental_diagnostic_{metric}.{fmt}")
    return names


def prepare_output(output: Path, overwrite: bool) -> None:
    """Create a clean, script-owned output directory.

    A nonempty directory is never reused implicitly.  With ``--overwrite``
    only known current/legacy files owned by this post-processor are removed.
    Unrelated files and directories (including Jupyter's
    ``.ipynb_checkpoints``) are left untouched.  This prevents a stale 3-panel
    or legacy ratio artifact from surviving a later 2-panel/no-plot run
    without treating harmless unrelated entries as an error.
    """
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
            f"output directory is not empty: {output}; choose a new --output or pass "
            "--overwrite to replace only known Figure-2 artifacts"
        )
    owned = owned_output_names()
    blocked = [entry.name for entry in entries if entry.name in owned and not entry.is_file()]
    if blocked:
        raise ValueError(
            "refusing --overwrite because a path reserved for a Figure-2 output is not "
            f"a regular file: {blocked}"
        )
    for entry in entries:
        if entry.name in owned:
            entry.unlink()


def read_outer_history(path: Path, metrics: Sequence[str]) -> Dict[str, Dict[int, float]]:
    """Read only the requested diagnostics; e_Xev is deliberately ignored."""
    series: Dict[str, Dict[int, float]] = {metric: {} for metric in metrics}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty outer history: {path}")
        missing_columns = [metric for metric in metrics if metric not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{path}: missing requested columns {missing_columns}")
        if "outer_iter" not in reader.fieldnames:
            raise ValueError(f"{path}: missing outer_iter column")
        for row in reader:
            outer = _int(row.get("outer_iter"))
            if outer is None:
                continue
            for metric in metrics:
                value = _float(row.get(metric))
                if math.isfinite(value):
                    # A rerun should have been archived.  If duplicate rows
                    # remain, the last record is the final one.
                    series[metric][outer] = value
    return series


def discover_groups(
    out_root: Path,
    m_states: int,
    n_assets: int | None,
    primary_margin: float,
    run_name_regex: str,
    theta_init_method: str,
    theta_init_scale: float,
    risk_premium_mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Find newest PI-PINN run per (configuration, seed).

    Deduplication happens before success filtering, so a newer failed rerun
    cannot be hidden by an older successful copy.
    """
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int], Tuple[str, Path, Dict[str, Any], str]] = {}
    group_meta: Dict[str, Dict[str, Any]] = {}

    for run_dir_text in find_runs(str(out_root)):
        run_dir = Path(run_dir_text)
        cfg = load_config_args_raw(str(run_dir))
        if cfg is None or str(cfg.get("model_type", "")) != "pipinn":
            continue
        if _int(cfg.get("m_states")) != m_states:
            continue
        cfg_n_assets = _int(cfg.get("n_assets"))
        if n_assets is not None and cfg_n_assets != n_assets:
            continue
        if pattern and not pattern.search(str(run_dir)):
            continue
        margin = primary_eval_margin(cfg)
        if not math.isclose(margin, primary_margin, rel_tol=0.0, abs_tol=1e-12):
            continue
        # Defaults make pre-option affine configs semantically compatible.
        if str(cfg.get("risk_premium_mode", "affine")) != risk_premium_mode:
            continue
        if str(cfg.get("theta_init_method", "myopic")) != theta_init_method:
            continue
        scale = _float(cfg.get("theta_init_scale", 1.0))
        if not math.isclose(scale, theta_init_scale, rel_tol=0.0, abs_tol=1e-12):
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
            "m_states": m_states,
            "primary_eval_margin": margin,
            "risk_premium_mode": str(cfg.get("risk_premium_mode", "affine")),
            "theta_init_method": str(cfg.get("theta_init_method", "myopic")),
            "theta_init_scale": scale,
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
                f"--group-id={group_id!r} was not found; available groups={sorted(groups)}"
            )
        return groups[group_id]
    if len(groups) != 1:
        details = {
            group: {
                "n_assets": meta["n_assets"],
                "m_states": meta["m_states"],
                "successful_seeds": sorted(meta["runs"]),
            }
            for group, meta in groups.items()
        }
        raise ValueError(
            "expected exactly one eligible training configuration; narrow the selection "
            f"with --group-id or --run-name-regex. Candidates: {details}"
        )
    return next(iter(groups.values()))


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("summary values must be nonempty and finite")
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0, 0.0, float("nan"), float("nan")
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(int(arr.size))
    half = t_crit_95(int(arr.size) - 1) * sem
    return mean, std, sem, mean - half, mean + half


def validate_and_load(
    meta: Dict[str, Any],
    expected_seeds: set[int],
    min_seeds: int,
    metrics: Sequence[str],
    vartheta_overlay: Mapping[int, Mapping[int, float]] | None = None,
) -> Tuple[List[int], Dict[int, Dict[str, Dict[int, float]]], List[Dict[str, Any]]]:
    available = set(meta["runs"])
    if expected_seeds and available != expected_seeds:
        latest_status = {
            seed: str(record["status"])
            for seed, record in sorted(meta["latest"].items())
        }
        raise ValueError(
            f"group={meta['group']}: successful seeds={sorted(available)}, "
            f"expected exactly={sorted(expected_seeds)}; latest statuses={latest_status}"
        )
    seeds = sorted(expected_seeds if expected_seeds else available)
    if len(seeds) < min_seeds:
        raise ValueError(
            f"group={meta['group']}: found {len(seeds)} successful seeds, "
            f"but --min-seeds={min_seeds}"
        )
    if vartheta_overlay is not None:
        missing_overlay = sorted(set(seeds) - set(vartheta_overlay))
        if missing_overlay:
            raise ValueError(
                "checkpoint-reconstructed normalized-control trajectory is missing "
                f"selected seeds {missing_overlay}"
            )

    market_errors = [
        (seed, meta["market_errors"].get(seed, "missing market hash"))
        for seed in seeds
        if not meta["market_hashes"].get(seed)
    ]
    if market_errors:
        raise ValueError(f"invalid market snapshots: {market_errors}")
    hashes = {meta["market_hashes"][seed] for seed in seeds}
    if len(hashes) != 1:
        raise ValueError(
            f"selected seeds have {len(hashes)} distinct canonical market snapshots"
        )

    histories: Dict[int, Dict[str, Dict[int, float]]] = {}
    run_rows: List[Dict[str, Any]] = []
    common_outer_iters: int | None = None
    for seed in seeds:
        run_dir: Path = meta["runs"][seed]
        cfg = meta["configs"][seed]
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
                f"group={meta['group']}: inconsistent outer_iters; seed={seed} has "
                f"{outer_iters}, expected {common_outer_iters}"
            )

        history_path = run_dir / "outer_history.csv"
        if vartheta_overlay is None:
            try:
                history = read_outer_history(history_path, metrics)
            except ValueError as exc:
                if POLICY_METRIC in str(exc):
                    raise ValueError(
                        f"{exc}. This run predates {POLICY_METRIC}; reconstruct it from "
                        "the saved outer checkpoints with reconstruct_vartheta_trajectory.py "
                        "and pass --vartheta-trajectory-csv."
                    ) from exc
                raise
            policy_source = "outer_history"
        else:
            base_metrics = [metric for metric in metrics if metric != POLICY_METRIC]
            history = read_outer_history(history_path, base_metrics)
            reconstructed = {
                int(outer): float(value)
                for outer, value in vartheta_overlay[seed].items()
            }
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                native_fields = set(csv.DictReader(handle).fieldnames or ())
            if POLICY_METRIC in native_fields:
                native = read_outer_history(history_path, [POLICY_METRIC])[POLICY_METRIC]
                if set(native) != set(reconstructed):
                    raise ValueError(
                        f"{run_dir}: native and reconstructed {POLICY_METRIC} outer grids differ"
                    )
                mismatches = [
                    outer for outer in native
                    if not math.isclose(
                        native[outer], reconstructed[outer],
                        rel_tol=5.0e-5, abs_tol=5.0e-7,
                    )
                ]
                if mismatches:
                    raise ValueError(
                        f"{run_dir}: native and reconstructed {POLICY_METRIC} disagree "
                        f"at outer={mismatches}"
                    )
            history[POLICY_METRIC] = reconstructed
            policy_source = "checkpoint_reconstruction"
        expected_outers = list(range(1, outer_iters + 1))
        for metric in metrics:
            actual = sorted(history[metric])
            if actual != expected_outers:
                missing = sorted(set(expected_outers) - set(actual))
                extra = sorted(set(actual) - set(expected_outers))
                raise ValueError(
                    f"{run_dir}: metric={metric} must have one finite value at every "
                    f"outer iteration 1..{outer_iters}; missing={missing}, extra={extra}"
                )
            nonpositive = [outer for outer in actual if history[metric][outer] <= 0.0]
            if nonpositive:
                raise ValueError(
                    f"{run_dir}: metric={metric} is nonpositive at outer={nonpositive}; "
                    "log-scale plotting and log-linear fitting require positive values"
                )
        histories[seed] = history
        run_rows.append(
            {
                "group": meta["group"],
                "model_type": meta["model_type"],
                "n_assets": meta["n_assets"],
                "m_states": meta["m_states"],
                "seed": seed,
                "run_dir": str(run_dir),
                "outer_iters": outer_iters,
                "market_hash": meta["market_hashes"][seed],
                "policy_metric_source": policy_source,
            }
        )
    return seeds, histories, run_rows


def fit_log_linear(
    series: Mapping[int, float], window: Tuple[int, int]
) -> Dict[str, float | int]:
    outers = list(range(window[0], window[1] + 1))
    missing = [outer for outer in outers if outer not in series]
    if missing:
        raise ValueError(f"fit window {format_window(window)} is missing outer={missing}")
    values = np.asarray([series[outer] for outer in outers], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            f"fit window {format_window(window)} contains nonfinite or nonpositive values"
        )
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
    metrics: Sequence[str],
    windows: Sequence[Tuple[int, int]],
    primary_window: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trajectory_rows: List[Dict[str, Any]] = []
    pointwise_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    decay_rows: List[Dict[str, Any]] = []
    identity = {
        "group": meta["group"],
        "model_type": meta["model_type"],
        "n_assets": meta["n_assets"],
        "m_states": meta["m_states"],
    }

    for metric in metrics:
        outer_grid = sorted(histories[seeds[0]][metric])
        for seed in seeds:
            for outer in outer_grid:
                trajectory_rows.append(
                    {
                        **identity,
                        "seed": seed,
                        "metric": metric,
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
                    "outer_iter": outer,
                    "n_seeds": len(seeds),
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

        for window in windows:
            window_label = format_window(window)
            per_seed_fits: List[Dict[str, Any]] = []
            for seed in seeds:
                fit = fit_log_linear(histories[seed][metric], window)
                row = {
                    **identity,
                    "seed": seed,
                    "metric": metric,
                    "estimate_kind": (
                        "validation_residual_decay"
                        if metric == OPTIONAL_METRIC
                        else "relative_L2_error_decay"
                    ),
                    "fit_window": window_label,
                    "outer_start": window[0],
                    "outer_end": window[1],
                    "is_primary": int(window == primary_window),
                    **fit,
                }
                fit_rows.append(row)
                per_seed_fits.append(row)

            rho_values = [float(row["rho"]) for row in per_seed_fits]
            log_rho_values = [float(row["log_rho"]) for row in per_seed_fits]
            rho_mean, rho_std, rho_sem, rho_ci_low, rho_ci_high = mean_std_ci(rho_values)
            log_mean, log_std, log_sem, log_ci_low, log_ci_high = mean_std_ci(
                log_rho_values
            )
            decay_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "estimate_kind": (
                        "validation_residual_decay"
                        if metric == OPTIONAL_METRIC
                        else "relative_L2_error_decay"
                    ),
                    "fit_window": window_label,
                    "outer_start": window[0],
                    "outer_end": window[1],
                    "is_primary": int(window == primary_window),
                    "n_seeds": len(seeds),
                    "rho_mean": rho_mean,
                    "rho_std": rho_std,
                    "rho_sem": rho_sem,
                    "rho_ci95_low": rho_ci_low,
                    "rho_ci95_high": rho_ci_high,
                    "log_rho_mean": log_mean,
                    "log_rho_std": log_std,
                    "log_rho_sem": log_sem,
                    "log_rho_ci95_low": log_ci_low,
                    "log_rho_ci95_high": log_ci_high,
                }
            )
    return trajectory_rows, pointwise_rows, fit_rows, decay_rows


def build_endpoint_summary(
    meta: Mapping[str, Any],
    pointwise_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    outer_start: int,
    outer_end: int,
) -> List[Dict[str, Any]]:
    """Report transparent seed-mean endpoint reductions E_start / E_end."""
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        by_outer = {
            int(row["outer_iter"]): row
            for row in pointwise_rows
            if row["metric"] == metric
        }
        missing = [outer for outer in (outer_start, outer_end) if outer not in by_outer]
        if missing:
            raise ValueError(
                f"metric={metric}: endpoint summary is missing outer iterations {missing}"
            )
        start_row = by_outer[outer_start]
        end_row = by_outer[outer_end]
        mean_start = float(start_row["mean"])
        mean_end = float(end_row["mean"])
        if mean_start <= 0.0 or mean_end <= 0.0:
            raise ValueError(f"metric={metric}: endpoint means must be positive")
        rows.append(
            {
                "group": meta["group"],
                "model_type": meta["model_type"],
                "n_assets": meta["n_assets"],
                "m_states": meta["m_states"],
                "metric": metric,
                "outer_start": outer_start,
                "outer_end": outer_end,
                "n_seeds": int(start_row["n_seeds"]),
                "mean_start": mean_start,
                "std_start": float(start_row["std"]),
                "mean_end": mean_end,
                "std_end": float(end_row["std"]),
                # Ratio of seed means, deliberately not the mean of seed-wise ratios.
                "seed_mean_reduction_factor": mean_start / mean_end,
            }
        )
    return rows


def log_sd_band(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an exact mean +/- SD band that is valid on a logarithmic axis.

    A logarithmic axis cannot represent a nonpositive ``mean - std`` value.
    Such lower endpoints are therefore masked with NaN instead of being
    replaced by an arbitrary positive display floor.  The upper endpoint is
    always the unmodified ``mean + std``.
    """
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if mean_array.shape != std_array.shape:
        raise ValueError("mean and std must have matching shapes")
    if np.any(~np.isfinite(mean_array)) or np.any(mean_array <= 0.0):
        raise ValueError("mean must contain positive finite values")
    if np.any(~np.isfinite(std_array)) or np.any(std_array < 0.0):
        raise ValueError("std must contain nonnegative finite values")
    raw_lower = mean_array - std_array
    lower = np.where(raw_lower > 0.0, raw_lower, np.nan)
    return lower, mean_array + std_array


def create_figure(
    trajectory_rows: Sequence[Mapping[str, Any]],
    pointwise_rows: Sequence[Mapping[str, Any]],
    decay_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    primary_window: Tuple[int, int],
    annotate_decay: bool = False,
    show_seed_trajectories: bool = False,
    figure_size: Tuple[float, float] | None = None,
    font_size: float = 10.0,
    font_family: str = "",
    outer_start: int = 1,
    outer_end: int | None = None,
    eps_compatible: bool = False,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    # ``decay_rows``, ``primary_window`` and ``annotate_decay`` remain in the
    # API for backwards compatibility.  The paper figure deliberately shows
    # neither a fitted-rho annotation nor an early-window highlight.
    _ = (decay_rows, primary_window, annotate_decay)
    size = figure_size if figure_size is not None else (4.8, 3.2)
    fig, ax = plt.subplots(1, 1, figsize=size)
    colors = {
        "diag_RelL2_V": "#0072B2",
        POLICY_METRIC: "#D55E00",
        "val_pres": "#009E73",
    }

    def lighten(color: str, white_fraction: float) -> Tuple[float, float, float]:
        rgb = np.asarray(to_rgb(color), dtype=float)
        return tuple((1.0 - white_fraction) * rgb + white_fraction)

    for metric in metrics:
        color = colors[metric]
        metric_trajectories = [row for row in trajectory_rows if row["metric"] == metric]
        seeds = sorted({int(row["seed"]) for row in metric_trajectories})
        if show_seed_trajectories:
            for seed in seeds:
                rows = sorted(
                    [
                        row
                        for row in metric_trajectories
                        if int(row["seed"]) == seed
                        and int(row["outer_iter"]) >= outer_start
                        and (outer_end is None or int(row["outer_iter"]) <= outer_end)
                    ],
                    key=lambda row: int(row["outer_iter"]),
                )
                ax.plot(
                    [int(row["outer_iter"]) for row in rows],
                    [float(row["value"]) for row in rows],
                    color=lighten(color, 0.65) if eps_compatible else color,
                    alpha=1.0 if eps_compatible else 0.22,
                    linewidth=0.8,
                    label=None,
                )

        summary = sorted(
            [
                row
                for row in pointwise_rows
                if row["metric"] == metric
                and int(row["outer_iter"]) >= outer_start
                and (outer_end is None or int(row["outer_iter"]) <= outer_end)
            ],
            key=lambda row: int(row["outer_iter"]),
        )
        if not summary:
            raise ValueError(f"metric={metric}: no points in requested plotting window")
        x = np.asarray([row["outer_iter"] for row in summary], dtype=float)
        mean = np.asarray([row["mean"] for row in summary], dtype=float)
        std = np.asarray([row["std"] for row in summary], dtype=float)
        lower, upper = log_sd_band(mean, std)
        # PostScript/EPS has no alpha channel.  Use a light, opaque version of
        # the same metric color for EPS so Matplotlib does not silently render
        # the translucent SD region as an opaque saturated polygon.
        band_color = lighten(color, 0.80) if eps_compatible else color
        ax.fill_between(
            x,
            lower,
            upper,
            color=band_color,
            alpha=1.0 if eps_compatible else 0.18,
            linewidth=0.0,
            label=None,
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.2,
            label=METRIC_LEGEND_LABELS[metric],
        )

    ax.set_yscale("log")
    if outer_end is not None:
        ax.set_xlim(outer_start, outer_end)
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
    label_font = {"fontfamily": font_family} if font_family else {}
    ax.set_xlabel("Outer iteration", fontsize=font_size, **label_font)
    ylabel = (
        r"Relative $L^2$ error"
        if OPTIONAL_METRIC not in metrics
        else r"Relative $L^2$ error / residual level"
    )
    ax.set_ylabel(ylabel, fontsize=font_size, **label_font)
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
            "Plot Liu value/policy seed-mean log-RelL2 convergence with SD bands "
            "and report transparent endpoint reductions."
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Default: <out-root>/figure2_empirical_convergence",
    )
    parser.add_argument("--m-states", type=int, default=3)
    parser.add_argument("--n-assets", type=int, default=None)
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
        "--vartheta-trajectory-csv",
        default="",
        help=(
            "Optional derived figure2_vartheta_per_outer.csv reconstructed from "
            "saved per-outer checkpoints. Source outer_history.csv files remain read-only."
        ),
    )
    parser.add_argument(
        "--theta-init-method",
        choices=("myopic", "zero", "closed_form"),
        default="myopic",
    )
    parser.add_argument("--theta-init-scale", type=float, default=1.0)
    parser.add_argument(
        "--risk-premium-mode", choices=("affine", "tanh"), default="affine"
    )
    parser.add_argument("--fit-window", default="1-4")
    parser.add_argument(
        "--sensitivity-windows",
        default="1-3,1-5",
        help="Additional inclusive windows. Default outputs 1-3, 1-4, and 1-5.",
    )
    parser.add_argument(
        "--include-val-pres",
        action="store_true",
        help="Add val_pres as an optional third curve (off in the paper figure).",
    )
    parser.add_argument(
        "--hide-seed-trajectories",
        action="store_true",
        help="Compatibility flag; individual seed curves are hidden by default.",
    )
    parser.add_argument(
        "--show-seed-trajectories",
        action="store_true",
        help="Diagnostic opt-in; the paper figure omits individual seed curves.",
    )
    parser.add_argument("--no-decay-annotations", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--endpoint-outer",
        type=int,
        default=20,
        help="Final outer iteration for the plotted range and E1/E_end summary.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known Figure-2 artifacts in a nonempty output directory.",
    )
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (default: 4.8 for the Liu single panel).",
    )
    parser.add_argument("--fig-height", type=float, default=3.2, help="Figure height in inches.")
    parser.add_argument("--font-size", type=float, default=10.0, help="Base font size in points.")
    parser.add_argument(
        "--font-family",
        default="",
        help="Optional Matplotlib font family, e.g. 'Times New Roman'.",
    )
    parser.add_argument(
        "--bbox-inches",
        choices=("tight", "standard"),
        default="tight",
        help="Use 'standard' to preserve the exact figsize*dpi raster dimensions.",
    )
    args = parser.parse_args(argv)

    if args.m_states < 1:
        raise ValueError("--m-states must be positive")
    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2")
    if not 0.0 <= args.primary_margin < 1.0:
        raise ValueError("--primary-margin must be in [0,1)")
    if not math.isfinite(args.theta_init_scale) or args.theta_init_scale <= 0.0:
        raise ValueError("--theta-init-scale must be positive and finite")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if args.endpoint_outer <= 1:
        raise ValueError("--endpoint-outer must be greater than 1")
    if args.fig_width is not None and (
        not math.isfinite(args.fig_width) or args.fig_width <= 0.0
    ):
        raise ValueError("--fig-width must be positive and finite")
    if not math.isfinite(args.fig_height) or args.fig_height <= 0.0:
        raise ValueError("--fig-height must be positive and finite")
    if not math.isfinite(args.font_size) or args.font_size <= 0.0:
        raise ValueError("--font-size must be positive and finite")

    primary_window, windows = parse_windows(args.fit_window, args.sensitivity_windows)
    formats = parse_formats(args.formats)
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    metrics = list(MAIN_METRICS)
    if args.include_val_pres:
        metrics.append(OPTIONAL_METRIC)
    default_width = 4.8
    figure_size = (
        args.fig_width if args.fig_width is not None else default_width,
        args.fig_height,
    )

    out_root = Path(args.out_root).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_root / "figure2_empirical_convergence"
    )
    groups = discover_groups(
        out_root=out_root,
        m_states=args.m_states,
        n_assets=args.n_assets,
        primary_margin=args.primary_margin,
        run_name_regex=args.run_name_regex,
        theta_init_method=args.theta_init_method,
        theta_init_scale=args.theta_init_scale,
        risk_premium_mode=args.risk_premium_mode,
    )
    if not groups:
        raise SystemExit("no eligible successful PI-PINN runs with outer_history.csv were found")
    meta = select_group(groups, args.group_id)
    vartheta_overlay = None
    vartheta_overlay_provenance: Dict[str, Any] | None = None
    if args.vartheta_trajectory_csv:
        vartheta_overlay, vartheta_overlay_provenance = load_vartheta_overlay(
            Path(args.vartheta_trajectory_csv), meta
        )
    seeds, histories, run_rows = validate_and_load(
        meta, expected_seeds, args.min_seeds, metrics, vartheta_overlay
    )
    max_fit_outer = max(window[1] for window in windows)
    available_outer = max(histories[seeds[0]][metrics[0]])
    if max_fit_outer > available_outer:
        raise ValueError(
            f"fit windows require outer={max_fit_outer}, but histories end at {available_outer}"
        )
    if args.endpoint_outer > available_outer:
        raise ValueError(
            f"--endpoint-outer={args.endpoint_outer}, but histories end at {available_outer}"
        )

    trajectory_rows, pointwise_rows, fit_rows, decay_rows = build_tables(
        meta, histories, seeds, metrics, windows, primary_window
    )
    endpoint_rows = build_endpoint_summary(
        meta,
        pointwise_rows,
        metrics,
        outer_start=1,
        outer_end=args.endpoint_outer,
    )
    sd_band_masked_points = {
        metric: sum(
            1
            for row in pointwise_rows
            if row["metric"] == metric
            and 1 <= int(row["outer_iter"]) <= args.endpoint_outer
            and float(row["mean"]) - float(row["std"]) <= 0.0
        )
        for metric in metrics
    }

    prepare_output(output, args.overwrite)

    write_csv(
        output / "figure2_trajectories.csv",
        trajectory_rows,
        [
            "group", "model_type", "n_assets", "m_states", "seed", "metric",
            "outer_iter", "value",
        ],
    )
    write_csv(
        output / "figure2_pointwise_summary.csv",
        pointwise_rows,
        [
            "group", "model_type", "n_assets", "m_states", "metric", "outer_iter",
            "n_seeds", "mean", "std", "sem", "ci95_low", "ci95_high",
        ],
    )
    write_csv(
        output / "figure2_endpoint_summary.csv",
        endpoint_rows,
        [
            "group", "model_type", "n_assets", "m_states", "metric", "outer_start",
            "outer_end", "n_seeds", "mean_start", "std_start", "mean_end", "std_end",
            "seed_mean_reduction_factor",
        ],
    )
    write_csv(
        output / "figure2_seed_decay_fits.csv",
        fit_rows,
        [
            "group", "model_type", "n_assets", "m_states", "seed", "metric",
            "estimate_kind", "fit_window", "outer_start", "outer_end", "is_primary",
            "n_points", "intercept", "log_rho", "rho", "r_squared", "start_value",
            "end_value", "observed_reduction_factor",
        ],
    )
    write_csv(
        output / "figure2_decay_summary.csv",
        decay_rows,
        [
            "group", "model_type", "n_assets", "m_states", "metric", "estimate_kind",
            "fit_window", "outer_start", "outer_end", "is_primary", "n_seeds",
            "rho_mean", "rho_std", "rho_sem", "rho_ci95_low", "rho_ci95_high",
            "log_rho_mean", "log_rho_std", "log_rho_sem", "log_rho_ci95_low",
            "log_rho_ci95_high",
        ],
    )
    write_csv(
        output / "figure2_runs_used.csv",
        run_rows,
        [
            "group", "model_type", "n_assets", "m_states", "seed", "run_dir",
            "outer_iters", "market_hash", "policy_metric_source",
        ],
    )

    metadata = {
        "arguments": vars(args),
        "selected_group": meta["group"],
        "selected_seeds": seeds,
        "metrics": metrics,
        "policy_metric": {
            "field": POLICY_METRIC,
            "definition": "relative L2 error of vartheta=theta/w on the fixed Q_ev design",
            "legacy_raw_field_not_used": LEGACY_RAW_POLICY_METRIC,
            "source": (
                "checkpoint_reconstruction"
                if vartheta_overlay is not None
                else "outer_history"
            ),
            "derived_artifact": vartheta_overlay_provenance,
        },
        "primary_fit_window": format_window(primary_window),
        "fit_windows": [format_window(window) for window in windows],
        "trajectory_aggregation": (
            "pointwise arithmetic mean and sample SD on untransformed error values"
        ),
        "endpoint_summary": {
            "outer_start": 1,
            "outer_end": args.endpoint_outer,
            "definition": "seed mean at outer 1 divided by seed mean at endpoint outer",
            "rows": endpoint_rows,
        },
        "sd_band_log_display": (
            "mean-SD is rendered without clipping wherever it is positive; nonpositive "
            "lower endpoints are masked because a logarithmic axis cannot display them; "
            "mean+SD and all saved statistics are unmodified"
        ),
        "sd_band_masked_lower_points": sd_band_masked_points,
        "decay_estimation": (
            "fit log(error)=alpha+n*log(rho) separately within each seed, then "
            "aggregate seed-wise rho by mean, sample SD, and Student-t 95% CI"
        ),
        "metric_interpretation": {
            "diag_RelL2_V": (
                "relative L2 value-error trajectory; the main-text reduction is the "
                "ratio of seed means at the two reported endpoints"
            ),
            POLICY_METRIC: (
                "relative L2 wealth-normalized policy error for vartheta=theta/w; "
                "the main-text reduction is the ratio of seed means at the two "
                "reported endpoints"
            ),
            **(
                {
                    "val_pres": (
                        "early validation residual-level decay; p_res combines PDE and "
                        "terminal RMS terms and is not a PI contraction factor"
                    )
                }
                if args.include_val_pres
                else {}
            ),
        },
        "interpretation_limit": (
            "no estimate here is the X_ev-norm contraction factor or an exact PI-map result"
        ),
        "e_Xev_used": False,
        "exact_map_used": False,
        "figure_has_title_or_caption": False,
        "figure_layout": "single Liu panel with Value and Policy curves",
        "figure_shows_decay_fit_or_early_window": False,
        "show_individual_seed_trajectories": (
            args.show_seed_trajectories and not args.hide_seed_trajectories
        ),
        "figure_style": {
            "width_inches": figure_size[0],
            "height_inches": figure_size[1],
            "dpi": args.dpi,
            "font_size_points": args.font_size,
            "font_family": args.font_family or "matplotlib_default",
            "bbox_inches": args.bbox_inches,
            "formats": formats,
            "eps_rendering": (
                "opaque lightened SD fills and grid colors are used because PostScript "
                "does not support alpha transparency"
                if "eps" in formats
                else "not requested"
            ),
        },
    }
    with (output / "figure2_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if not args.no_plots:
        figure_kwargs = {
            "annotate_decay": False,
            "show_seed_trajectories": (
                args.show_seed_trajectories and not args.hide_seed_trajectories
            ),
            "figure_size": figure_size,
            "font_size": args.font_size,
            "font_family": args.font_family,
            "outer_start": 1,
            "outer_end": args.endpoint_outer,
        }
        regular_formats = [fmt for fmt in formats if fmt != "eps"]
        if regular_formats:
            fig = create_figure(
                trajectory_rows,
                pointwise_rows,
                decay_rows,
                metrics,
                primary_window,
                **figure_kwargs,
            )
            for fmt in regular_formats:
                fig.savefig(
                    output / f"figure2_empirical_convergence.{fmt}",
                    dpi=args.dpi,
                    bbox_inches="tight" if args.bbox_inches == "tight" else None,
                )
            import matplotlib.pyplot as plt

            plt.close(fig)
        if "eps" in formats:
            eps_fig = create_figure(
                trajectory_rows,
                pointwise_rows,
                decay_rows,
                metrics,
                primary_window,
                eps_compatible=True,
                **figure_kwargs,
            )
            eps_fig.savefig(
                output / "figure2_empirical_convergence.eps",
                dpi=args.dpi,
                bbox_inches="tight" if args.bbox_inches == "tight" else None,
            )
            import matplotlib.pyplot as plt

            plt.close(eps_fig)

    print(f"[done] Figure-2 empirical convergence outputs: {output}")
    print(f"[info] seeds={seeds}; metrics={metrics}; plotted outer=1-{args.endpoint_outer}")
    for row in endpoint_rows:
        print(
            f"[endpoint] {METRIC_LEGEND_LABELS[str(row['metric'])]}: "
            f"{float(row['mean_start']):.3e} -> {float(row['mean_end']):.3e} "
            f"({float(row['seed_mean_reduction_factor']):.3f}x reduction)"
        )
    masked_total = sum(sd_band_masked_points.values())
    if masked_total:
        print(
            "[warn] mean-SD was nonpositive at "
            f"{masked_total} plotted metric/iteration points; those lower band endpoints "
            "were masked on the logarithmic axis"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
