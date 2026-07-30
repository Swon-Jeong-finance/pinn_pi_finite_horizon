#!/usr/bin/env python3
"""Post-process adjacent empirical Liu ``e_Xev`` ratios.

For each selected successful PI-PINN training run, this script either reads
the training-time ``e_Xev`` trajectory or reevaluates every saved outer
checkpoint on a user-selected tensor grid, then forms every adjacent ratio

    rho_empirical_X(k) = e_Xev(k + 1) / e_Xev(k),  k = 1, ..., K - 1.

Ratios are formed inside each seed before pointwise seed statistics or
seedwise worst-case summaries are computed.  Every finite positive pair is
retained in the raw tables.  For the Merton-style figure, a separate display
classification marks source errors at or below a configurable multiple of a
late-trajectory error scale as floor dominated.

Neither mode performs a finite-difference PDE solve.  The checkpoint mode
reuses the exact-map evaluator's window resolver, neural derivatives,
closed-form reference, and policy-relevant X norm, so its errors equal the
exact-map audit's ``e_input_X`` when the same grid/window options are used.
This remains distinct from the relative-L2 statistic used in main Figure 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import canonical_market_hash, parse_seed_spec
from postprocess_contraction import (
    discover_groups,
    mean_std_ci,
    select_group,
    validate_and_load,
)


METRIC = "e_Xev"
SUPPORTED_FORMATS = {"png", "pdf", "svg"}
PLOT_STEM = "empirical_xev_ratio"
BASE_MANAGED_OUTPUTS = {
    "empirical_xev_ratio_per_seed.csv",
    "empirical_xev_ratio_summary.csv",
    "empirical_xev_ratio_worst_per_seed.csv",
    "empirical_xev_ratio_worst_summary.csv",
    "empirical_xev_ratio_floor_per_seed.csv",
    "empirical_xev_ratio_floor_summary.csv",
    "empirical_xev_ratio_floor_worst_per_seed.csv",
    "empirical_xev_ratio_floor_worst_summary.csv",
    "empirical_xev_ratio_runs_used.csv",
    "empirical_xev_reevaluated_trajectory.csv",
    "empirical_xev_ratio_metadata.json",
    "_SUCCESS_EMPIRICAL_XEV_RATIO",
}

METRIC_SOURCES = ("training-history", "checkpoint-reevaluation")
DEFAULT_FLOOR_MULTIPLIERS = (5.0, 10.0, 20.0)
DEFAULT_MAIN_FLOOR_MULTIPLE = 10.0
EMPIRICAL_RATIO_LABEL = r"Empirical ratio $\widehat{\varrho}_n$"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash an artifact without requiring a newer companion module."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_companion_api() -> None:
    """Reject an incompatible ``postprocess_contraction.py`` explicitly."""

    contracts = {
        "discover_groups": (
            discover_groups,
            {
                "out_root",
                "m_states",
                "n_assets",
                "primary_margin",
                "run_name_regex",
                "theta_init_method",
                "theta_init_scale",
                "risk_premium_mode",
            },
        ),
        "validate_and_load": (
            validate_and_load,
            {"meta", "expected_seeds", "min_seeds", "metrics"},
        ),
    }
    mismatches: List[str] = []
    for name, (function, required) in contracts.items():
        available = set(inspect.signature(function).parameters)
        missing = sorted(required - available)
        if missing:
            mismatches.append(f"{name} missing {missing}")
    if mismatches:
        raise ImportError(
            "incompatible postprocess_contraction.py; replace it with the "
            "matching Liu file distributed alongside "
            "postprocess_empirical_xev_ratio.py. API mismatches: "
            + "; ".join(mismatches)
        )


def parse_formats(text: str) -> List[str]:
    formats = [
        part.lower()
        for part in re.split(r"[\s,]+", str(text or "").strip())
        if part
    ]
    if not formats:
        raise ValueError("--formats must contain at least one format")
    if len(formats) != len(set(formats)):
        raise ValueError(f"duplicate formats in --formats={text!r}")
    invalid = sorted(set(formats) - SUPPORTED_FORMATS)
    if invalid:
        raise ValueError(f"unsupported figure formats: {invalid}")
    return formats


def managed_output_names() -> set[str]:
    names = set(BASE_MANAGED_OUTPUTS)
    names.update(f"{PLOT_STEM}.{suffix}" for suffix in SUPPORTED_FORMATS)
    return names


def _check_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    if not output.exists():
        return
    entries = list(output.iterdir())
    if entries and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; choose a new --output "
            "or pass --overwrite"
        )
    managed = managed_output_names()
    blocked = [
        entry.name
        for entry in entries
        if entry.name in managed and not (entry.is_file() or entry.is_symlink())
    ]
    if blocked:
        raise ValueError(
            "refusing --overwrite because reserved managed paths are not files: "
            f"{blocked}"
        )


def _commit_staged_output(stage: Path, output: Path) -> None:
    """Atomically replace only this script's managed output files."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    backup = Path(
        tempfile.mkdtemp(
            prefix=".empirical-xev-ratio-backup-",
            dir=str(output.parent),
        )
    )
    moved_old: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        for name in sorted(managed_output_names()):
            original = output / name
            if original.exists() or original.is_symlink():
                saved = backup / name
                os.replace(original, saved)
                moved_old.append((saved, original))

        success_name = "_SUCCESS_EMPIRICAL_XEV_RATIO"
        install_names = [
            name for name in sorted(managed_output_names())
            if name != success_name
        ]
        if (stage / success_name).is_file():
            install_names.append(success_name)
        for name in install_names:
            staged = stage / name
            if not (staged.is_file() or staged.is_symlink()):
                continue
            destination = output / name
            os.replace(staged, destination)
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


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _identity(meta: Mapping[str, Any], target_label: str) -> Dict[str, Any]:
    return {
        "target_label": target_label,
        "group": meta["group"],
        "training_group": meta.get("training_group", meta["group"]),
        "model_type": meta["model_type"],
        "n_assets": meta["n_assets"],
        "m_states": meta["m_states"],
        "metric": METRIC,
        "metric_source": meta.get(
            "metric_source", "training-history"
        ),
    }


def build_ratio_tables(
    meta: Mapping[str, Any],
    histories: Mapping[int, Mapping[str, Mapping[int, float]]],
    seeds: Sequence[int],
    run_rows: Sequence[Mapping[str, Any]],
    target_label: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Build seed-first adjacent ratios and their common-sample summaries."""

    if not str(target_label).strip():
        raise ValueError("target_label must be nonempty")
    if not seeds:
        raise ValueError("at least one seed is required")

    run_by_seed = {int(row["seed"]): row for row in run_rows}
    if set(run_by_seed) != set(int(seed) for seed in seeds):
        raise ValueError("run metadata and selected seeds do not match")

    identity = _identity(meta, target_label)
    per_seed: List[Dict[str, Any]] = []
    common_schedule: List[int] | None = None
    for seed in sorted(int(value) for value in seeds):
        try:
            series = histories[seed][METRIC]
        except KeyError as exc:
            raise ValueError(f"seed={seed}: missing metric={METRIC}") from exc
        outers = sorted(int(outer) for outer in series)
        if not outers:
            raise ValueError(f"seed={seed}: empty metric={METRIC} trajectory")
        expected = list(range(outers[0], outers[-1] + 1))
        if outers != expected:
            raise ValueError(
                f"seed={seed}: metric={METRIC} must cover complete outer "
                "1..K or one selected contiguous outer interval; "
                f"observed={outers}"
            )
        if len(outers) < 2:
            raise ValueError(
                f"seed={seed}: at least two outer iterations are required"
            )
        if common_schedule is None:
            common_schedule = outers
        elif outers != common_schedule:
            raise ValueError(
                f"seed={seed}: outer schedule {outers} differs from "
                f"common schedule {common_schedule}"
            )

        run_row = run_by_seed[seed]
        for source_outer in outers[:-1]:
            target_outer = source_outer + 1
            source_value = float(series[source_outer])
            target_value = float(series[target_outer])
            if (
                not math.isfinite(source_value)
                or not math.isfinite(target_value)
                or source_value <= 0.0
                or target_value <= 0.0
            ):
                raise ValueError(
                    f"seed={seed}, pair={source_outer}->{target_outer}: "
                    f"{METRIC} values must be finite and positive"
                )
            ratio = target_value / source_value
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    f"seed={seed}, pair={source_outer}->{target_outer}: "
                    "adjacent ratio must be finite and positive"
                )
            per_seed.append(
                {
                    **identity,
                    "seed": seed,
                    "ratio_iter": source_outer - 1,
                    "source_outer_iter": source_outer,
                    "target_outer_iter": target_outer,
                    "e_Xev_source": source_value,
                    "e_Xev_target": target_value,
                    "rho_empirical_X": ratio,
                    "run_dir": run_row["run_dir"],
                    "market_hash": run_row["market_hash"],
                }
            )

    expected_pairs = len(seeds) * (len(common_schedule or []) - 1)
    if len(per_seed) != expected_pairs:
        raise ValueError(
            f"incomplete adjacent-ratio panel: found {len(per_seed)}, "
            f"expected {expected_pairs}"
        )

    summary: List[Dict[str, Any]] = []
    for source_outer in (common_schedule or [])[:-1]:
        rows = [
            row
            for row in per_seed
            if int(row["source_outer_iter"]) == source_outer
        ]
        observed_seeds = [int(row["seed"]) for row in rows]
        expected_seeds = sorted(int(seed) for seed in seeds)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"source_outer_iter={source_outer}: common seed sample mismatch; "
                f"observed={observed_seeds}, expected={expected_seeds}"
            )
        values = [float(row["rho_empirical_X"]) for row in rows]
        mean, std, sem, ci_low, ci_high = mean_std_ci(values)
        summary.append(
            {
                **identity,
                "ratio_iter": source_outer - 1,
                "source_outer_iter": source_outer,
                "target_outer_iter": source_outer + 1,
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": min(values),
                "max": max(values),
                "seeds": ",".join(str(seed) for seed in expected_seeds),
            }
        )

    worst_per_seed: List[Dict[str, Any]] = []
    for seed in sorted(int(value) for value in seeds):
        rows = [
            row for row in per_seed if int(row["seed"]) == seed
        ]
        # Rows are already in ascending outer order, so an exact tie resolves
        # to the earliest source outer iteration.
        worst = max(rows, key=lambda row: float(row["rho_empirical_X"]))
        ratios = [float(row["rho_empirical_X"]) for row in rows]
        worst_per_seed.append(
            {
                **identity,
                "seed": seed,
                "n_pairs": len(rows),
                "max_rho_empirical_X": float(worst["rho_empirical_X"]),
                "max_rho_source_outer_iter": int(worst["source_outer_iter"]),
                "max_rho_target_outer_iter": int(worst["target_outer_iter"]),
                "min_rho_empirical_X": min(ratios),
                "all_adjacent_ratios_below_one": int(
                    all(value < 1.0 for value in ratios)
                ),
                "run_dir": run_by_seed[seed]["run_dir"],
                "market_hash": run_by_seed[seed]["market_hash"],
            }
        )

    worst_values = [
        float(row["max_rho_empirical_X"]) for row in worst_per_seed
    ]
    worst_mean, worst_std, worst_sem, worst_ci_low, worst_ci_high = (
        mean_std_ci(worst_values)
    )
    global_worst = max(
        worst_per_seed,
        key=lambda row: (
            float(row["max_rho_empirical_X"]),
            -int(row["seed"]),
        ),
    )
    worst_summary = [
        {
            **identity,
            "statistic": "seedwise_max_rho_empirical_X",
            "n_seeds": len(worst_values),
            "mean": worst_mean,
            "std": worst_std,
            "sem": worst_sem,
            "ci95_low": worst_ci_low,
            "ci95_high": worst_ci_high,
            "min": min(worst_values),
            "max": max(worst_values),
            "global_max": float(global_worst["max_rho_empirical_X"]),
            "global_max_seed": int(global_worst["seed"]),
            "global_max_source_outer_iter": int(
                global_worst["max_rho_source_outer_iter"]
            ),
            "global_max_target_outer_iter": int(
                global_worst["max_rho_target_outer_iter"]
            ),
            "all_seedwise_maxima_below_one": int(
                all(value < 1.0 for value in worst_values)
            ),
            "seeds": ",".join(str(seed) for seed in sorted(seeds)),
        }
    ]
    return per_seed, summary, worst_per_seed, worst_summary


def parse_floor_multipliers(text: str) -> List[float]:
    """Parse a unique, nonnegative floor-multiplier schedule."""

    tokens = [
        token
        for token in re.split(r"[\s,]+", str(text or "").strip())
        if token
    ]
    if not tokens:
        raise ValueError("--floor-multipliers must contain at least one value")
    values: List[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"invalid floor multiplier {token!r}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "--floor-multipliers must contain finite nonnegative values"
            )
        if value in values:
            raise ValueError(
                f"duplicate floor multiplier in --floor-multipliers: {value:g}"
            )
        values.append(value)
    return values


def _floor_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "target_label",
            "group",
            "training_group",
            "model_type",
            "n_assets",
            "m_states",
            "metric",
            "metric_source",
        )
        if key in row
    }


def build_floor_tables(
    histories: Mapping[int, Mapping[str, Mapping[int, float]]],
    seeds: Sequence[int],
    raw_per_seed: Sequence[Mapping[str, Any]],
    *,
    floor_multipliers: Sequence[float],
    main_floor_multiple: float,
    floor_value: Optional[float] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Classify raw ratios using the Merton late-error floor convention.

    Raw ratios are never removed.  The returned repeated floor table records
    every raw ratio for every requested multiplier.  Pointwise summaries also
    retain dominated iterations and expose ``common_regular`` so the plot can
    draw eligible segments without bridging a floor-dominated gap.
    """

    ordered_seeds = sorted(int(seed) for seed in seeds)
    if not ordered_seeds or not raw_per_seed:
        raise ValueError("floor classification requires nonempty ratio data")
    multipliers = [float(value) for value in floor_multipliers]
    if (
        not multipliers
        or any(not math.isfinite(value) or value < 0.0 for value in multipliers)
        or len(multipliers) != len(set(multipliers))
    ):
        raise ValueError(
            "floor_multipliers must be unique, finite, and nonnegative"
        )
    main = float(main_floor_multiple)
    if not math.isfinite(main) or main < 0.0:
        raise ValueError("main_floor_multiple must be finite and nonnegative")
    if main not in multipliers:
        raise ValueError(
            "main_floor_multiple must be included in floor_multipliers"
        )
    if floor_value is not None:
        floor_value = float(floor_value)
        if not math.isfinite(floor_value) or floor_value < 0.0:
            raise ValueError("floor_value must be finite and nonnegative")

    floors: Dict[int, float] = {}
    tail_counts: Dict[int, int] = {}
    for seed in ordered_seeds:
        try:
            series = histories[seed][METRIC]
        except KeyError as exc:
            raise ValueError(
                f"seed={seed}: missing {METRIC} trajectory for floor"
            ) from exc
        values = np.asarray(
            [float(series[outer]) for outer in sorted(series)],
            dtype=float,
        )
        if (
            values.size < 2
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                f"seed={seed}: floor trajectory must contain at least two "
                "finite positive values"
            )
        tail_count = max(1, int(math.ceil(0.10 * values.size)))
        tail_counts[seed] = tail_count
        floors[seed] = (
            float(floor_value)
            if floor_value is not None
            else float(np.median(values[-tail_count:]))
        )

    raw_index: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    ratio_iters: set[int] = set()
    for row in raw_per_seed:
        seed = int(row["seed"])
        ratio_iter = int(
            row.get("ratio_iter", int(row["source_outer_iter"]) - 1)
        )
        key = (seed, ratio_iter)
        if key in raw_index:
            raise ValueError(
                f"duplicate raw empirical ratio for seed={seed}, n={ratio_iter}"
            )
        raw_index[key] = row
        ratio_iters.add(ratio_iter)
    expected_keys = {
        (seed, ratio_iter)
        for seed in ordered_seeds
        for ratio_iter in ratio_iters
    }
    if set(raw_index) != expected_keys:
        raise ValueError("floor classification requires a complete seed panel")

    floor_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    worst_per_seed: List[Dict[str, Any]] = []
    worst_summary: List[Dict[str, Any]] = []
    identity = _floor_identity(raw_per_seed[0])
    common_by_multiplier: Dict[float, List[int]] = {}
    dominated_by_multiplier: Dict[float, List[int]] = {}

    for multiplier in multipliers:
        regular: Dict[Tuple[int, int], bool] = {}
        for seed in ordered_seeds:
            for ratio_iter in sorted(ratio_iters):
                row = raw_index[(seed, ratio_iter)]
                regular[(seed, ratio_iter)] = (
                    float(row["e_Xev_source"])
                    > multiplier * floors[seed]
                )
        common = [
            ratio_iter
            for ratio_iter in sorted(ratio_iters)
            if all(
                regular[(seed, ratio_iter)] for seed in ordered_seeds
            )
        ]
        common_set = set(common)
        dominated = sorted(ratio_iters - common_set)
        common_by_multiplier[multiplier] = common
        dominated_by_multiplier[multiplier] = dominated

        for seed in ordered_seeds:
            for ratio_iter in sorted(ratio_iters):
                row = raw_index[(seed, ratio_iter)]
                floor_rows.append({
                    **dict(row),
                    "floor": floors[seed],
                    "floor_source": (
                        "explicit_absolute_base_floor"
                        if floor_value is not None
                        else "median_last_ceil_10pct_error_trajectory"
                    ),
                    "floor_tail_count": tail_counts[seed],
                    "floor_multiple": multiplier,
                    "regular": int(regular[(seed, ratio_iter)]),
                    "common_regular": int(ratio_iter in common_set),
                })

        for ratio_iter in sorted(ratio_iters):
            rows = [
                raw_index[(seed, ratio_iter)] for seed in ordered_seeds
            ]
            values = [float(row["rho_empirical_X"]) for row in rows]
            mean, std, sem, ci_low, ci_high = mean_std_ci(values)
            summary_rows.append({
                **identity,
                "floor_multiple": multiplier,
                "ratio_iter": ratio_iter,
                "source_outer_iter": int(rows[0]["source_outer_iter"]),
                "target_outer_iter": int(rows[0]["target_outer_iter"]),
                "common_regular": int(ratio_iter in common_set),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": min(values),
                "max": max(values),
                "seeds": ",".join(str(seed) for seed in ordered_seeds),
            })

        seed_maxima: List[Dict[str, Any]] = []
        for seed in ordered_seeds:
            eligible = [
                raw_index[(seed, ratio_iter)]
                for ratio_iter in sorted(ratio_iters)
                if regular[(seed, ratio_iter)]
            ]
            row_out: Dict[str, Any] = {
                **identity,
                "floor_multiple": multiplier,
                "seed": seed,
                "floor": floors[seed],
                "n_regular_pairs": len(eligible),
                "max_rho_empirical_X": "",
                "max_rho_ratio_iter": "",
                "max_rho_source_outer_iter": "",
                "max_rho_target_outer_iter": "",
                "run_dir": raw_index[(seed, min(ratio_iters))]["run_dir"],
                "market_hash": raw_index[(seed, min(ratio_iters))][
                    "market_hash"
                ],
            }
            if eligible:
                worst = max(
                    eligible,
                    key=lambda row: (
                        float(row["rho_empirical_X"]),
                        -int(row["source_outer_iter"]),
                    ),
                )
                row_out.update({
                    "max_rho_empirical_X": float(
                        worst["rho_empirical_X"]
                    ),
                    "max_rho_ratio_iter": int(
                        worst.get(
                            "ratio_iter",
                            int(worst["source_outer_iter"]) - 1,
                        )
                    ),
                    "max_rho_source_outer_iter": int(
                        worst["source_outer_iter"]
                    ),
                    "max_rho_target_outer_iter": int(
                        worst["target_outer_iter"]
                    ),
                })
                seed_maxima.append(row_out)
            worst_per_seed.append(row_out)

        status = (
            "pass"
            if len(seed_maxima) == len(ordered_seeds) and common
            else "skipped_empty_regular_support"
        )
        summary_out: Dict[str, Any] = {
            **identity,
            "floor_multiple": multiplier,
            "status": status,
            "n_common_iterations": len(common),
            "first_common_ratio_iter": common[0] if common else "",
            "last_common_ratio_iter": common[-1] if common else "",
            "max_of_pointwise_mean_rho": "",
            "n_seed_maxima": len(seed_maxima),
            "mean": "",
            "std": "",
            "sem": "",
            "ci95_low": "",
            "ci95_high": "",
            "min": "",
            "max": "",
            "seeds": ",".join(str(seed) for seed in ordered_seeds),
        }
        if status == "pass":
            maxima = [
                float(row["max_rho_empirical_X"]) for row in seed_maxima
            ]
            mean, std, sem, ci_low, ci_high = mean_std_ci(maxima)
            common_means = [
                float(row["mean"])
                for row in summary_rows
                if float(row["floor_multiple"]) == multiplier
                and int(row["common_regular"]) == 1
            ]
            summary_out.update({
                "max_of_pointwise_mean_rho": max(common_means),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": min(maxima),
                "max": max(maxima),
            })
        worst_summary.append(summary_out)

    main_common = common_by_multiplier[main]
    if not main_common:
        raise ValueError(
            "main empirical floor multiple leaves no common regular "
            f"iteration: main_floor_multiple={main:g}"
        )
    main_seed_empty = [
        seed
        for seed in ordered_seeds
        if not any(
            bool(row["regular"])
            for row in floor_rows
            if int(row["seed"]) == seed
            and float(row["floor_multiple"]) == main
        )
    ]
    if main_seed_empty:
        raise ValueError(
            "main empirical floor multiple leaves empty seed support: "
            f"{main_seed_empty}"
        )

    evidence = {
        "definition": (
            "source e_Xev is regular iff "
            "e_Xev(source) > floor_multiple * seed_floor"
        ),
        "comparison": "strict_greater_than",
        "tail_fraction": 0.10,
        "floor_source": (
            "explicit_absolute_base_floor"
            if floor_value is not None
            else "median_last_ceil_10pct_error_trajectory"
        ),
        "explicit_floor_value": floor_value,
        "floor_is_not_fd_discretization_floor": True,
        "floor_multipliers": multipliers,
        "main_floor_multiple": main,
        "floors_by_seed": {
            str(seed): floors[seed] for seed in ordered_seeds
        },
        "tail_counts_by_seed": {
            str(seed): tail_counts[seed] for seed in ordered_seeds
        },
        "common_ratio_iters_by_multiplier": {
            f"{multiplier:g}": common_by_multiplier[multiplier]
            for multiplier in multipliers
        },
        "floor_dominated_ratio_iters_by_multiplier": {
            f"{multiplier:g}": dominated_by_multiplier[multiplier]
            for multiplier in multipliers
        },
        "main_common_ratio_iters": main_common,
        "main_floor_dominated_ratio_iters": (
            dominated_by_multiplier[main]
        ),
        "raw_ratio_rows_retained": True,
    }
    return (
        floor_rows,
        summary_rows,
        worst_per_seed,
        worst_summary,
        evidence,
    )


def parse_checkpoint_spec(text: Optional[str]) -> Optional[List[int]]:
    """Parse ``all`` or one contiguous checkpoint interval."""

    if text is None or str(text).strip().lower() == "all":
        return None
    raw = str(text).strip()
    tokens = raw.split(",")
    if (
        not raw
        or raw.startswith(",")
        or raw.endswith(",")
        or any(not re.fullmatch(r"[1-9]\d*", token.strip()) for token in tokens)
    ):
        raise ValueError(
            "--checkpoints must be 'all' or a contiguous list such as 1,2,3"
        )
    values = [int(token.strip()) for token in tokens]
    if (
        any(value < 1 for value in values)
        or len(values) != len(set(values))
        or values != list(range(values[0], values[-1] + 1))
    ):
        raise ValueError(
            "--checkpoints must be 'all' or one positive contiguous interval"
        )
    return values


def validate_checkpoint_run_selection(
    meta: Mapping[str, Any],
    expected_seeds: set[int],
    min_seeds: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Select custom-reevaluation runs without requiring stored ``e_Xev``."""

    available = set(int(seed) for seed in meta["runs"])
    if expected_seeds and available != expected_seeds:
        latest_status = {
            int(seed): str(record["status"])
            for seed, record in sorted(meta["latest"].items())
        }
        raise ValueError(
            f"group={meta['group']}: successful seeds={sorted(available)}, "
            f"expected exactly={sorted(expected_seeds)}; "
            f"latest statuses={latest_status}"
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
        raise ValueError(f"invalid market snapshots: {market_errors}")
    market_hashes = {str(meta["market_hashes"][seed]) for seed in seeds}
    if len(market_hashes) != 1:
        raise ValueError(
            f"selected seeds have {len(market_hashes)} distinct canonical "
            "market snapshots"
        )

    run_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run_dir = Path(meta["runs"][seed]).resolve()
        cfg = meta["configs"][seed]
        outer_iters = int(cfg.get("outer_iters", 0))
        if outer_iters < 2:
            raise ValueError(
                f"{run_dir}: checkpoint reevaluation requires at least two "
                f"outer iterations; got {outer_iters}"
            )
        run_rows.append(
            {
                "group": meta["group"],
                "model_type": meta["model_type"],
                "n_assets": meta["n_assets"],
                "m_states": meta["m_states"],
                "seed": seed,
                "run_dir": str(run_dir),
                "outer_iters": outer_iters,
                "market_hash": str(meta["market_hashes"][seed]),
                "policy_metric_source": "checkpoint_reevaluation",
            }
        )
    return seeds, run_rows


def _grid_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def reevaluate_checkpoint_errors(
    meta: Mapping[str, Any],
    seeds: Sequence[int],
    run_rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    checkpoint_subset: Optional[Sequence[int]],
    eval_margin: Optional[float],
    eval_w_min: Optional[float],
    eval_w_max: Optional[float],
    eval_x_margin: Optional[float],
    eval_nt: int,
    eval_ny: int,
    eval_nx: int,
    eval_chunk: int,
) -> Tuple[
    Dict[str, Any],
    Dict[int, Dict[str, Dict[int, float]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Recompute exact-reference checkpoint errors without any FD solve."""

    # Importing these evaluation helpers is safe: PyTorch is loaded lazily by
    # TorchCheckpointEvaluator, and no frozen-policy FD routine is called.
    from liu_exact_map_core import x_norm_components
    from liu_exact_map_fd import (
        TorchCheckpointEvaluator,
        load_run,
        stable_hash,
    )

    selected_rows = {int(row["seed"]): dict(row) for row in run_rows}
    if set(selected_rows) != {int(seed) for seed in seeds}:
        raise ValueError("selected run rows do not match the requested seeds")

    histories: Dict[int, Dict[str, Dict[int, float]]] = {}
    evidence_rows: List[Dict[str, Any]] = []
    enriched_runs: List[Dict[str, Any]] = []
    common_schedule: Optional[List[int]] = None
    common_group: Optional[str] = None
    common_market_hash: Optional[str] = None
    common_training_protocol_hash: Optional[str] = None
    common_window_json: Optional[str] = None
    common_grid_hash: Optional[str] = None
    common_window: Optional[Dict[str, Any]] = None
    network_dtypes: set[str] = set()
    implementation_hashes = {
        "postprocessor": sha256_file(Path(__file__).resolve()),
        "exact_map_evaluator": sha256_file(
            Path(__file__).with_name("liu_exact_map_fd.py").resolve()
        ),
        "x_norm_core": sha256_file(
            Path(__file__).with_name("liu_exact_map_core.py").resolve()
        ),
    }

    for seed in sorted(int(value) for value in seeds):
        row = selected_rows[seed]
        run_dir = Path(str(row["run_dir"])).resolve()
        run = load_run(
            run_dir,
            checkpoint_subset=checkpoint_subset,
            eval_margin_override=eval_margin,
            eval_w_min_override=eval_w_min,
            eval_w_max_override=eval_w_max,
            eval_x_margin_override=eval_x_margin,
            allow_sparse_subset=True,
        )
        if int(run.seed) != seed:
            raise ValueError(
                f"{run_dir}: loaded seed={run.seed}, expected seed={seed}"
            )
        if int(run.problem.n_assets) != int(meta["n_assets"]):
            raise ValueError(
                f"{run_dir}: loaded n_assets={run.problem.n_assets}, "
                f"expected {meta['n_assets']}"
            )
        current_discovery_market_hash = canonical_market_hash(
            str(run.market_path)
        )
        if current_discovery_market_hash != str(row["market_hash"]):
            raise ValueError(
                f"{run_dir}: market snapshot changed between run discovery "
                "and checkpoint reevaluation"
            )

        schedule = [int(outer) for outer, _path in run.checkpoints]
        if len(schedule) < 2:
            raise ValueError(
                f"{run_dir}: at least two selected checkpoints are required"
            )
        if schedule != list(range(schedule[0], schedule[-1] + 1)):
            raise ValueError(
                f"{run_dir}: selected checkpoints must be contiguous; "
                f"observed={schedule}"
            )
        if common_schedule is None:
            common_schedule = schedule
        elif schedule != common_schedule:
            raise ValueError(
                f"{run_dir}: checkpoint schedule {schedule} differs from "
                f"common schedule {common_schedule}"
            )

        window = dict(run.evaluation_window)
        window_json = json.dumps(
            window, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if common_group is None:
            common_group = str(run.group)
            common_market_hash = str(run.market_hash)
            common_training_protocol_hash = str(run.training_protocol_hash)
            common_window_json = window_json
            common_window = window
        else:
            if str(run.group) != common_group:
                raise ValueError(
                    f"{run_dir}: custom evaluation group differs across seeds"
                )
            if str(run.market_hash) != common_market_hash:
                raise ValueError(
                    f"{run_dir}: exact-evaluator market hash differs across seeds"
                )
            if (
                str(run.training_protocol_hash)
                != common_training_protocol_hash
            ):
                raise ValueError(
                    f"{run_dir}: training protocol differs across seeds"
                )
            if window_json != common_window_json:
                raise ValueError(
                    f"{run_dir}: resolved evaluation window differs across seeds"
                )

        ev_w_min, ev_w_max = (
            float(run.eval_w_bounds[0]),
            float(run.eval_w_bounds[1]),
        )
        ev_x_min, ev_x_max = (
            float(run.eval_x_bounds[0]),
            float(run.eval_x_bounds[1]),
        )
        ev_y = np.linspace(
            math.log(ev_w_min), math.log(ev_w_max), int(eval_ny)
        )
        ev_x = np.linspace(ev_x_min, ev_x_max, int(eval_nx))
        ev_tau = np.linspace(
            0.0, float(run.problem.horizon), int(eval_nt) + 1
        )[1:]
        tt, yy, xx = np.meshgrid(ev_tau, ev_y, ev_x, indexing="ij")
        grid_hash = _grid_sha256(ev_tau, ev_y, ev_x)
        if common_grid_hash is None:
            common_grid_hash = grid_hash
        elif grid_hash != common_grid_hash:
            raise ValueError(
                f"{run_dir}: resolved evaluation grid differs across seeds"
            )
        reference = run.closed_form.wealth_bundle(tt, yy, xx)

        series: Dict[int, float] = {}
        checkpoint_manifest = hashlib.sha256()
        for outer, checkpoint in run.checkpoints:
            checkpoint = Path(checkpoint).resolve()
            checkpoint_hash = sha256_file(checkpoint)
            evaluator = TorchCheckpointEvaluator(checkpoint, run, device)
            network_dtype = str(evaluator.dtype).replace("torch.", "")
            network_dtypes.add(network_dtype)
            flat_bundle = evaluator.bundle_at_points(
                tt.ravel(),
                yy.ravel(),
                xx.ravel(),
                chunk=int(eval_chunk),
            )
            input_bundle = tuple(
                np.asarray(item, dtype=np.float64).reshape(tt.shape)
                for item in flat_bundle
            )
            metric = x_norm_components(*input_bundle, reference)
            error_x = float(metric["x_norm"])
            if not math.isfinite(error_x) or error_x <= 0.0:
                raise ValueError(
                    f"{run_dir}, outer={outer}: reevaluated e_Xev must be "
                    f"finite and positive; got {error_x}"
                )
            series[int(outer)] = error_x
            checkpoint_manifest.update(
                f"{int(outer)}:{checkpoint_hash}\n".encode("ascii")
            )
            evidence_rows.append(
                {
                    "target_label": "",
                    "group": str(run.group),
                    "training_group": str(meta["group"]),
                    "model_type": "pipinn",
                    "n_assets": int(run.problem.n_assets),
                    "m_states": 1,
                    "metric": METRIC,
                    "metric_source": "checkpoint-reevaluation",
                    "seed": seed,
                    "outer_iter": int(outer),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "network_dtype": network_dtype,
                    "eval_margin": float(run.eval_margin),
                    "eval_x_margin": float(run.eval_x_margin),
                    "eval_w_min_override": (
                        "" if run.eval_w_min_override is None
                        else float(run.eval_w_min_override)
                    ),
                    "eval_w_max_override": (
                        "" if run.eval_w_max_override is None
                        else float(run.eval_w_max_override)
                    ),
                    "ev_w_min": ev_w_min,
                    "ev_w_max": ev_w_max,
                    "ev_x_min": ev_x_min,
                    "ev_x_max": ev_x_max,
                    "eval_nt": int(eval_nt),
                    "eval_ny": int(eval_ny),
                    "eval_nx": int(eval_nx),
                    "eval_points": int(tt.size),
                    "evaluation_grid_sha256": grid_hash,
                    "e_input_value": float(metric["value_sup"]),
                    "e_input_vw": float(metric["vw_sup"]),
                    "e_input_vww": float(metric["vww_sup"]),
                    "e_input_vwx": float(metric["vwx_sup"]),
                    "e_input_bundle": float(metric["bundle_sup"]),
                    "e_input_X": error_x,
                    "e_Xev": error_x,
                    "run_dir": str(run_dir),
                    "market_hash": str(run.market_hash),
                    "training_protocol_hash": str(
                        run.training_protocol_hash
                    ),
                }
            )
            del evaluator, flat_bundle, input_bundle
        histories[seed] = {METRIC: series}

        history_path = run_dir / "outer_history.csv"
        config_path = run_dir / "config.json"
        market_path = run_dir / "market_params.npz"
        closed_form_path = run_dir / "closed_form_ode.npz"
        enriched_runs.append(
            {
                "target_label": "",
                **row,
                "group": str(run.group),
                "training_group": str(meta["group"]),
                "metric": METRIC,
                "metric_source": "checkpoint-reevaluation",
                "policy_metric_source": "checkpoint_reevaluation",
                "market_hash": str(run.market_hash),
                "training_protocol_hash": str(
                    run.training_protocol_hash
                ),
                "weight_dir": str(run.weight_dir),
                "checkpoint_selection": str(run.checkpoint_selection),
                "terminal_state_hash": str(run.terminal_state_hash),
                "selected_checkpoints": ",".join(
                    str(value) for value in schedule
                ),
                "checkpoint_manifest_sha256": (
                    checkpoint_manifest.hexdigest()
                ),
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "market_params": str(market_path),
                "market_params_sha256": sha256_file(market_path),
                "closed_form": str(closed_form_path),
                "closed_form_sha256": sha256_file(closed_form_path),
                "outer_history": str(history_path),
                "outer_history_sha256": sha256_file(history_path),
                "evaluation_grid_sha256": grid_hash,
            }
        )

    if common_group is None or common_window is None or common_schedule is None:
        raise ValueError("no checkpoints were reevaluated")
    if len(network_dtypes) != 1:
        raise ValueError(
            f"network dtypes differ across selected runs: {network_dtypes}"
        )
    evaluation_grid = {
        "coordinate_system": "tensor grid uniform in tau, log-wealth, x",
        "tau_zero_excluded": True,
        "eval_nt": int(eval_nt),
        "eval_ny": int(eval_ny),
        "eval_nx": int(eval_nx),
        "n_points": int(eval_nt) * int(eval_ny) * int(eval_nx),
        "sha256": common_grid_hash,
    }
    protocol_payload = {
        "problem": "liu",
        "metric_source": "checkpoint-reevaluation",
        "metric": METRIC,
        "norm": "value_sup_plus_joint_vw_vww_vwx_bundle_sup",
        "exact_map_group": common_group,
        "training_group": str(meta["group"]),
        "evaluation_window": common_window,
        "evaluation_grid": evaluation_grid,
        "checkpoint_schedule": common_schedule,
        "network_dtype": next(iter(network_dtypes)),
        "market_hash": common_market_hash,
        "training_protocol_hash": common_training_protocol_hash,
        "implementation_hashes": implementation_hashes,
    }
    evaluation_protocol_hash = stable_hash(protocol_payload)
    evaluation_group = evaluation_protocol_hash[:12]
    for output_row in evidence_rows:
        output_row["group"] = evaluation_group
        output_row["exact_map_group"] = common_group
        output_row["evaluation_protocol_hash"] = evaluation_protocol_hash
    for output_row in enriched_runs:
        output_row["group"] = evaluation_group
        output_row["exact_map_group"] = common_group
        output_row["evaluation_protocol_hash"] = evaluation_protocol_hash
    custom_meta = {
        **dict(meta),
        "group": evaluation_group,
        "training_group": str(meta["group"]),
        "exact_map_group": common_group,
        "evaluation_protocol_hash": evaluation_protocol_hash,
        "metric_source": "checkpoint-reevaluation",
    }
    provenance = {
        "evaluation_protocol_hash": evaluation_protocol_hash,
        "evaluation_group": evaluation_group,
        "exact_map_group": common_group,
        "evaluation_window": common_window,
        "checkpoint_schedule": common_schedule,
        "evaluation_grid": evaluation_grid,
        "device": str(device),
        "eval_chunk": int(eval_chunk),
        "network_dtype": next(iter(network_dtypes)),
        "market_hash": common_market_hash,
        "training_protocol_hash": common_training_protocol_hash,
        "finite_difference_solver_called": False,
        "finite_difference_outputs_used": False,
        "same_definition_as_exact_map_e_input_X": True,
        "matched_to_specific_fd_artifact": False,
        "implementation_hashes": implementation_hashes,
    }
    return (
        custom_meta,
        histories,
        enriched_runs,
        evidence_rows,
        provenance,
    )


def create_figure(
    floor_summary_rows: Sequence[Mapping[str, Any]],
    *,
    main_floor_multiple: float = DEFAULT_MAIN_FLOOR_MULTIPLE,
    y_scale: str = "linear",
    figure_size: Tuple[float, float] = (6.4, 4.2),
    font_size: float = 10.0,
    font_family: str = "",
    line_width: float = 2.0,
    marker_size: float = 6.0,
    band_alpha: float = 0.18,
    floor_alpha: float = 0.80,
    grid_alpha: float = 0.22,
):
    """Create the Merton empirical-ratio floor-aware figure."""

    if y_scale not in {"linear", "log"}:
        raise ValueError("y_scale must be 'linear' or 'log'")
    if not floor_summary_rows:
        raise ValueError("cannot plot an empty empirical-ratio summary")
    if not 0.0 <= float(band_alpha) <= 1.0:
        raise ValueError("band_alpha must lie in [0, 1]")
    if not 0.0 <= float(floor_alpha) <= 1.0:
        raise ValueError("floor_alpha must lie in [0, 1]")
    if not 0.0 <= float(grid_alpha) <= 1.0:
        raise ValueError("grid_alpha must lie in [0, 1]")

    main = float(main_floor_multiple)
    selected = [
        row
        for row in floor_summary_rows
        if math.isclose(
            float(row["floor_multiple"]),
            main,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if not selected:
        raise ValueError(
            f"no floor summary rows for main_floor_multiple={main:g}"
        )

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ordered = sorted(
        selected,
        key=lambda row: int(row["ratio_iter"]),
    )
    x = np.asarray(
        [int(row["ratio_iter"]) for row in ordered],
        dtype=float,
    )
    mean = np.asarray([float(row["mean"]) for row in ordered], dtype=float)
    std = np.asarray([float(row["std"]) for row in ordered], dtype=float)
    common = np.asarray(
        [int(row["common_regular"]) == 1 for row in ordered],
        dtype=bool,
    )
    if (
        np.any(~np.isfinite(mean))
        or np.any(mean <= 0.0)
        or np.any(~np.isfinite(std))
        or np.any(std < 0.0)
    ):
        raise ValueError("plot summary must contain positive means and finite SDs")
    if not np.any(common):
        raise ValueError("main floor support is empty")

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

    rc: Dict[str, Any] = {"font.size": font_size}
    if font_family:
        rc["font.family"] = font_family
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figure_size)
        blue = "#0072B2"
        eligible_label_used = False
        for segment in contiguous_segments(common):
            segment_mean = mean[segment]
            segment_std = std[segment]
            lower = segment_mean - segment_std
            if y_scale == "linear":
                lower = np.maximum(lower, 0.0)
            else:
                lower = np.where(lower > 0.0, lower, np.nan)
            ax.fill_between(
                x[segment],
                lower,
                segment_mean + segment_std,
                color=blue,
                alpha=float(band_alpha),
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                x[segment],
                segment_mean,
                color=blue,
                linestyle="-",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label=(
                    EMPIRICAL_RATIO_LABEL
                    if not eligible_label_used
                    else None
                ),
                zorder=3,
            )
            eligible_label_used = True
        dominated = ~common
        if np.any(dominated):
            ax.scatter(
                x[dominated],
                mean[dominated],
                color="#9E9E9E",
                marker="x",
                s=max(20.0, float(marker_size) ** 2),
                linewidths=max(1.0, 0.75 * float(line_width)),
                alpha=float(floor_alpha),
                label="Floor-dominated",
                zorder=4,
            )
        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=max(1.0, 0.8 * line_width),
            label="Contraction threshold",
            zorder=1,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(EMPIRICAL_RATIO_LABEL)
        ax.set_yscale(y_scale)
        tick_step = 3
        ax.set_xticks(x[::tick_step])
        ax.grid(
            True,
            which="both",
            alpha=float(grid_alpha),
            linewidth=0.8,
        )
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate adjacent Liu e_Xev ratios within seed from either the "
            "saved training history or a no-FD custom-window checkpoint "
            "reevaluation, then draw seed mean +/- sample SD."
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric-source",
        choices=METRIC_SOURCES,
        default="training-history",
        help=(
            "training-history reads saved e_Xev; checkpoint-reevaluation "
            "recomputes the exact-reference X norm on a custom tensor grid "
            "without solving any FD PDE"
        ),
    )
    parser.add_argument(
        "--run-name-regex",
        required=True,
        help=(
            "Required regex that isolates one PI-PINN run family. A p_res "
            "target is not required; main-run names are supported."
        ),
    )
    parser.add_argument(
        "--target-label",
        required=True,
        help="Verbatim target label recorded in every output row and metadata.",
    )
    parser.add_argument(
        "--expected-seeds",
        required=True,
        help="Exact comma/space/range seed set.",
    )
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--m-states", type=int, default=1)
    parser.add_argument("--n-assets", type=int, default=30)
    parser.add_argument(
        "--primary-margin",
        type=float,
        default=0.10,
        help=(
            "Training-run selector: the primary eval_margin saved in config. "
            "This is distinct from checkpoint-reevaluation --eval-margin."
        ),
    )
    parser.add_argument("--group-id", default="")
    parser.add_argument(
        "--theta-init-method",
        choices=("myopic", "zero", "closed_form"),
        default="myopic",
    )
    parser.add_argument("--theta-init-scale", type=float, default=1.0)
    parser.add_argument(
        "--risk-premium-mode",
        choices=("affine", "tanh"),
        default="affine",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help=(
            "Checkpoint mode only: 'all' (default) or one contiguous interval "
            "such as 1,2,3 or 5,6,7."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Checkpoint mode only; defaults to cpu. Example: cuda:1.",
    )
    parser.add_argument(
        "--eval-margin",
        type=float,
        default=None,
        help=(
            "Checkpoint mode only: symmetric half-width margin applied to the "
            "saved wealth and factor domains. When omitted, reuse the run's "
            "saved primary margin."
        ),
    )
    parser.add_argument(
        "--eval-w-min",
        type=float,
        default=None,
        help=(
            "Checkpoint mode only: replace the effective wealth lower endpoint "
            "after --eval-margin is applied, matching liu_exact_map_fd.py."
        ),
    )
    parser.add_argument(
        "--eval-w-max",
        type=float,
        default=None,
        help=(
            "Checkpoint mode only: replace the effective wealth upper endpoint "
            "after --eval-margin is applied, matching liu_exact_map_fd.py."
        ),
    )
    parser.add_argument(
        "--eval-x-margin",
        type=float,
        default=None,
        help=(
            "Checkpoint mode only: optional factor-only half-width margin; "
            "otherwise --eval-margin is used for the factor interval."
        ),
    )
    parser.add_argument(
        "--eval-nt",
        type=int,
        default=None,
        help=(
            "Checkpoint mode only: positive-time nodes (default 80; matches "
            "exact-map --base-nt 80)."
        ),
    )
    parser.add_argument(
        "--eval-ny",
        type=int,
        default=None,
        help="Checkpoint mode only: log-wealth grid nodes (default 41).",
    )
    parser.add_argument(
        "--eval-nx",
        type=int,
        default=None,
        help="Checkpoint mode only: factor grid nodes (default 41).",
    )
    parser.add_argument(
        "--eval-chunk",
        type=int,
        default=None,
        help=(
            "Checkpoint mode only: neural/autograd points per chunk "
            "(default 32768; lower this after a GPU OOM)."
        ),
    )
    parser.add_argument(
        "--ratio-y-scale",
        choices=("linear", "log"),
        default="linear",
    )
    parser.add_argument(
        "--floor-multipliers",
        default="5,10,20",
        help=(
            "Merton-style nonnegative sensitivity schedule. A source point is "
            "regular when e_Xev(source) is strictly above multiplier times "
            "the seed floor."
        ),
    )
    parser.add_argument(
        "--main-floor-multiple",
        type=float,
        default=DEFAULT_MAIN_FLOOR_MULTIPLE,
        help="Multiplier used by the rendered figure (default: 10).",
    )
    parser.add_argument(
        "--floor-value",
        type=float,
        default=None,
        help=(
            "Optional absolute base floor shared by all seeds. By default each "
            "seed floor is the median of the final ceil(10%%) error values."
        ),
    )
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=5.9)
    parser.add_argument("--fig-height", type=float, default=4.0)
    parser.add_argument("--font-size", type=float, default=10.0)
    parser.add_argument("--font-family", default="")
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--marker-size", type=float, default=6.0)
    parser.add_argument("--band-alpha", type=float, default=0.18)
    parser.add_argument("--floor-alpha", type=float, default=0.80)
    parser.add_argument("--grid-alpha", type=float, default=0.22)
    parser.add_argument(
        "--capsize",
        type=float,
        default=4.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bbox-inches",
        choices=("tight", "standard"),
        default="tight",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> List[str]:
    if not str(args.run_name_regex).strip():
        raise ValueError("--run-name-regex must be nonempty")
    try:
        re.compile(args.run_name_regex)
    except re.error as exc:
        raise ValueError(f"invalid --run-name-regex: {exc}") from exc
    if not str(args.target_label).strip():
        raise ValueError("--target-label must be nonempty")
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2")
    if args.m_states < 1:
        raise ValueError("--m-states must be positive")
    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if not 0.0 <= args.primary_margin < 1.0:
        raise ValueError("--primary-margin must be in [0,1)")
    if not math.isfinite(args.theta_init_scale) or args.theta_init_scale <= 0.0:
        raise ValueError("--theta-init-scale must be positive and finite")
    checkpoint_only = {
        "--checkpoints": args.checkpoints,
        "--device": args.device,
        "--eval-margin": args.eval_margin,
        "--eval-w-min": args.eval_w_min,
        "--eval-w-max": args.eval_w_max,
        "--eval-x-margin": args.eval_x_margin,
        "--eval-nt": args.eval_nt,
        "--eval-ny": args.eval_ny,
        "--eval-nx": args.eval_nx,
        "--eval-chunk": args.eval_chunk,
    }
    if args.metric_source == "training-history":
        supplied = [
            name for name, value in checkpoint_only.items()
            if value is not None
        ]
        if supplied:
            raise ValueError(
                f"{', '.join(supplied)} require "
                "--metric-source checkpoint-reevaluation"
            )
    else:
        if args.m_states != 1:
            raise ValueError(
                "checkpoint reevaluation currently supports only m_states=1"
            )
        if args.risk_premium_mode != "affine":
            raise ValueError(
                "checkpoint reevaluation requires the affine benchmark"
            )
        for name in ("eval_margin", "eval_x_margin"):
            value = getattr(args, name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value < 1.0
            ):
                raise ValueError(
                    f"--{name.replace('_', '-')} must lie in [0,1)"
                )
        for name in ("eval_w_min", "eval_w_max"):
            value = getattr(args, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"--{name.replace('_', '-')} must be finite"
                )
        if (
            args.eval_w_min is not None
            and args.eval_w_max is not None
            and not args.eval_w_min < args.eval_w_max
        ):
            raise ValueError("--eval-w-min must be below --eval-w-max")
        for name, default in (
            ("eval_nt", 80),
            ("eval_ny", 41),
            ("eval_nx", 41),
            ("eval_chunk", 32768),
        ):
            if getattr(args, name) is None:
                setattr(args, name, default)
            if int(getattr(args, name)) < 1:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be positive"
                )
        if args.eval_nt < 2:
            raise ValueError("--eval-nt must be at least 2")
        if args.eval_ny < 3 or args.eval_nx < 3:
            raise ValueError("--eval-ny and --eval-nx must each be at least 3")
        if args.device is None:
            args.device = "cpu"
        elif not str(args.device).strip():
            raise ValueError("--device must be nonempty")
        else:
            args.device = str(args.device).strip()
        parse_checkpoint_spec(args.checkpoints)
    for name in (
        "fig_width",
        "fig_height",
        "font_size",
        "line_width",
        "marker_size",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    floor_multipliers = parse_floor_multipliers(args.floor_multipliers)
    if (
        not math.isfinite(args.main_floor_multiple)
        or args.main_floor_multiple < 0.0
    ):
        raise ValueError(
            "--main-floor-multiple must be finite and nonnegative"
        )
    if args.main_floor_multiple not in floor_multipliers:
        raise ValueError(
            "--main-floor-multiple must appear in --floor-multipliers"
        )
    if args.floor_value is not None and (
        not math.isfinite(args.floor_value) or args.floor_value < 0.0
    ):
        raise ValueError("--floor-value must be finite and nonnegative")
    for name in ("band_alpha", "floor_alpha", "grid_alpha"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must lie in [0,1]"
            )
    if not math.isfinite(args.capsize) or args.capsize < 0.0:
        raise ValueError("--capsize must be finite and nonnegative")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    formats = parse_formats(args.formats)
    expected = parse_seed_spec(args.expected_seeds)
    if not expected:
        raise ValueError("--expected-seeds must be nonempty")
    if len(expected) < args.min_seeds:
        raise ValueError(
            f"--expected-seeds contains {len(expected)} seeds, fewer than "
            f"--min-seeds={args.min_seeds}"
        )
    return formats


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_companion_api()
    formats = _validate_args(args)
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    out_root = Path(args.out_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    _check_output(output, args.overwrite)

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
        raise ValueError(
            "no eligible successful PI-PINN runs match --run-name-regex "
            "and the requested configuration"
        )
    training_meta = select_group(groups, args.group_id)
    evidence_rows: List[Dict[str, Any]] = []
    reevaluation: Optional[Dict[str, Any]] = None
    if args.metric_source == "training-history":
        meta = {
            **training_meta,
            "training_group": training_meta["group"],
            "metric_source": "training-history",
        }
        seeds, histories, run_rows = validate_and_load(
            training_meta,
            expected_seeds,
            args.min_seeds,
            [METRIC],
        )
        enriched_run_rows: List[Dict[str, Any]] = []
        for row in run_rows:
            run_dir = Path(str(row["run_dir"]))
            history_path = run_dir / "outer_history.csv"
            enriched_run_rows.append(
                {
                    "target_label": args.target_label,
                    **row,
                    "training_group": training_meta["group"],
                    "metric": METRIC,
                    "metric_source": "training-history",
                    "outer_history": str(history_path),
                    "outer_history_sha256": sha256_file(history_path),
                    "selected_checkpoints": "",
                    "checkpoint_manifest_sha256": "",
                    "training_protocol_hash": "",
                    "config": str(run_dir / "config.json"),
                    "config_sha256": sha256_file(
                        run_dir / "config.json"
                    ),
                    "market_params": str(run_dir / "market_params.npz"),
                    "market_params_sha256": sha256_file(
                        run_dir / "market_params.npz"
                    ),
                    "closed_form": str(run_dir / "closed_form_ode.npz"),
                    "closed_form_sha256": (
                        sha256_file(run_dir / "closed_form_ode.npz")
                        if (run_dir / "closed_form_ode.npz").is_file()
                        else ""
                    ),
                    "evaluation_grid_sha256": "",
                }
            )
    else:
        seeds, selected_rows = validate_checkpoint_run_selection(
            training_meta,
            expected_seeds,
            args.min_seeds,
        )
        (
            meta,
            histories,
            enriched_run_rows,
            evidence_rows,
            reevaluation,
        ) = reevaluate_checkpoint_errors(
            training_meta,
            seeds,
            selected_rows,
            device=args.device,
            checkpoint_subset=parse_checkpoint_spec(args.checkpoints),
            eval_margin=args.eval_margin,
            eval_w_min=args.eval_w_min,
            eval_w_max=args.eval_w_max,
            eval_x_margin=args.eval_x_margin,
            eval_nt=args.eval_nt,
            eval_ny=args.eval_ny,
            eval_nx=args.eval_nx,
            eval_chunk=args.eval_chunk,
        )
        for row in enriched_run_rows:
            row["target_label"] = args.target_label
        for row in evidence_rows:
            row["target_label"] = args.target_label
        run_rows = enriched_run_rows

    per_seed, summary, worst_per_seed, worst_summary = build_ratio_tables(
        meta,
        histories,
        seeds,
        run_rows,
        args.target_label,
    )
    (
        floor_per_seed,
        floor_summary,
        floor_worst_per_seed,
        floor_worst_summary,
        floor_evidence,
    ) = build_floor_tables(
        histories,
        seeds,
        per_seed,
        floor_multipliers=parse_floor_multipliers(
            args.floor_multipliers
        ),
        main_floor_multiple=args.main_floor_multiple,
        floor_value=args.floor_value,
    )

    evidence_index = {
        (int(row["seed"]), int(row["outer_iter"])): row
        for row in evidence_rows
    }
    for row in per_seed:
        source = evidence_index.get(
            (int(row["seed"]), int(row["source_outer_iter"]))
        )
        target = evidence_index.get(
            (int(row["seed"]), int(row["target_outer_iter"]))
        )
        row["source_checkpoint"] = (
            "" if source is None else source["checkpoint"]
        )
        row["source_checkpoint_sha256"] = (
            "" if source is None else source["checkpoint_sha256"]
        )
        row["target_checkpoint"] = (
            "" if target is None else target["checkpoint"]
        )
        row["target_checkpoint_sha256"] = (
            "" if target is None else target["checkpoint_sha256"]
        )
    checkpoint_fields = (
        "source_checkpoint",
        "source_checkpoint_sha256",
        "target_checkpoint",
        "target_checkpoint_sha256",
    )
    checkpoint_index = {
        (int(row["seed"]), int(row["ratio_iter"])): row
        for row in per_seed
    }
    for row in floor_per_seed:
        raw = checkpoint_index[(int(row["seed"]), int(row["ratio_iter"]))]
        for field in checkpoint_fields:
            row[field] = raw[field]

    outer_schedule = sorted(
        int(value) for value in histories[seeds[0]][METRIC]
    )
    market_hash = str(run_rows[0]["market_hash"])
    masked_log_lower = sum(
        1
        for row in floor_summary
        if math.isclose(
            float(row["floor_multiple"]),
            float(args.main_floor_multiple),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and int(row["common_regular"]) == 1
        and float(row["mean"]) - float(row["std"]) <= 0.0
    )
    plot_files = [] if args.no_plot else [
        f"{PLOT_STEM}.{suffix}" for suffix in formats
    ]
    metadata = {
        "schema_version": 3,
        "status": "success",
        "arguments": vars(args),
        "target_label": args.target_label,
        "selected_group": meta["group"],
        "selected_training_group": training_meta["group"],
        "selected_seeds": seeds,
        "market_hash": market_hash,
        "metric": METRIC,
        "metric_source": args.metric_source,
        "outer_schedule": outer_schedule,
        "adjacent_source_schedule": outer_schedule[:-1],
        "ratio_iteration_schedule": [
            int(value) - 1 for value in outer_schedule[:-1]
        ],
        "ratio_definition": "e_Xev(k+1) / e_Xev(k), formed within seed",
        "aggregation": (
            "pointwise arithmetic mean, sample SD, SEM, and Student-t 95% CI "
            "of seed-wise adjacent ratios"
        ),
        "worst_summary": (
            "maximum adjacent ratio within each seed, followed by seed "
            "mean/sample-SD/SEM/Student-t 95% CI and global maximum"
        ),
        "raw_ratio_rows_retained": True,
        "floor_classification": floor_evidence,
        "interpretation": (
            "training-window empirical-only e_Xev adjacent-ratio diagnostic"
            if args.metric_source == "training-history"
            else (
                "custom-window checkpoint-reevaluated empirical e_Xev "
                "adjacent-ratio diagnostic"
            )
        ),
        "not_matched_to_custom_fd_X_ev": True,
        "matched_to_specific_fd_artifact": False,
        "same_definition_as_exact_map_e_input_X": (
            args.metric_source == "checkpoint-reevaluation"
        ),
        "reevaluation": reevaluation,
        "finite_difference_outputs_used": False,
        "finite_difference_solver_called": False,
        "not_main_figure2_relative_L2": True,
        "main_figure2_metric_used": False,
        "claim_limit": (
            "This output contains no frozen-policy FD map ratio and is not "
            "the main Figure-2 relative-L2 convergence statistic."
        ),
        "runs": {
            str(row["seed"]): {
                "run_dir": row["run_dir"],
                "outer_history": row["outer_history"],
                "outer_history_sha256": row["outer_history_sha256"],
                "selected_checkpoints": row.get(
                    "selected_checkpoints", ""
                ),
                "checkpoint_manifest_sha256": row.get(
                    "checkpoint_manifest_sha256", ""
                ),
                "evaluation_grid_sha256": row.get(
                    "evaluation_grid_sha256", ""
                ),
                "evaluation_protocol_hash": row.get(
                    "evaluation_protocol_hash", ""
                ),
                "exact_map_group": row.get("exact_map_group", ""),
                "weight_dir": row.get("weight_dir", ""),
                "checkpoint_selection": row.get(
                    "checkpoint_selection", ""
                ),
                "terminal_state_hash": row.get(
                    "terminal_state_hash", ""
                ),
            }
            for row in enriched_run_rows
        },
        "figure": {
            "files": plot_files,
            "y_scale": args.ratio_y_scale,
            "mean_line": (
                "blue solid with circle markers on main common regular support"
            ),
            "uncertainty_band": "fill_between mean plus/minus one sample SD",
            "floor_dominated": (
                "gray x at the all-seed raw-ratio mean; no uncertainty band"
            ),
            "reference_line": "black dashed varrho=1",
            "main_floor_multiple": args.main_floor_multiple,
            "log_scale_nonpositive_mean_minus_sd_lower_endpoints_omitted": (
                masked_log_lower if args.ratio_y_scale == "log" else 0
            ),
            "width_inches": args.fig_width,
            "height_inches": args.fig_height,
            "font_size_points": args.font_size,
            "font_family": args.font_family or "matplotlib_default",
            "line_width": args.line_width,
            "marker_size": args.marker_size,
            "band_alpha": args.band_alpha,
            "floor_alpha": args.floor_alpha,
            "grid_alpha": args.grid_alpha,
            "deprecated_capsize_ignored": args.capsize,
            "dpi": args.dpi,
            "formats": formats,
            "bbox_inches": args.bbox_inches,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".empirical-xev-ratio-stage-",
        dir=str(output.parent),
    ) as stage_text:
        stage = Path(stage_text)
        write_csv(
            stage / "empirical_xev_ratio_per_seed.csv",
            per_seed,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "seed",
                "ratio_iter",
                "source_outer_iter",
                "target_outer_iter",
                "e_Xev_source",
                "e_Xev_target",
                "rho_empirical_X",
                "run_dir",
                "market_hash",
                "training_group",
                "metric_source",
                "source_checkpoint",
                "source_checkpoint_sha256",
                "target_checkpoint",
                "target_checkpoint_sha256",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_summary.csv",
            summary,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "ratio_iter",
                "source_outer_iter",
                "target_outer_iter",
                "n_seeds",
                "mean",
                "std",
                "sem",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
                "seeds",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_floor_per_seed.csv",
            floor_per_seed,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "seed",
                "ratio_iter",
                "source_outer_iter",
                "target_outer_iter",
                "e_Xev_source",
                "e_Xev_target",
                "rho_empirical_X",
                "floor",
                "floor_source",
                "floor_tail_count",
                "floor_multiple",
                "regular",
                "common_regular",
                "run_dir",
                "market_hash",
                "training_group",
                "metric_source",
                "source_checkpoint",
                "source_checkpoint_sha256",
                "target_checkpoint",
                "target_checkpoint_sha256",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_floor_summary.csv",
            floor_summary,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "floor_multiple",
                "ratio_iter",
                "source_outer_iter",
                "target_outer_iter",
                "common_regular",
                "n_seeds",
                "mean",
                "std",
                "sem",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
                "seeds",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_floor_worst_per_seed.csv",
            floor_worst_per_seed,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "floor_multiple",
                "seed",
                "floor",
                "n_regular_pairs",
                "max_rho_empirical_X",
                "max_rho_ratio_iter",
                "max_rho_source_outer_iter",
                "max_rho_target_outer_iter",
                "run_dir",
                "market_hash",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_floor_worst_summary.csv",
            floor_worst_summary,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "floor_multiple",
                "status",
                "n_common_iterations",
                "first_common_ratio_iter",
                "last_common_ratio_iter",
                "max_of_pointwise_mean_rho",
                "n_seed_maxima",
                "mean",
                "std",
                "sem",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
                "seeds",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_worst_per_seed.csv",
            worst_per_seed,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "seed",
                "n_pairs",
                "max_rho_empirical_X",
                "max_rho_source_outer_iter",
                "max_rho_target_outer_iter",
                "min_rho_empirical_X",
                "all_adjacent_ratios_below_one",
                "run_dir",
                "market_hash",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_worst_summary.csv",
            worst_summary,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "metric",
                "statistic",
                "n_seeds",
                "mean",
                "std",
                "sem",
                "ci95_low",
                "ci95_high",
                "min",
                "max",
                "global_max",
                "global_max_seed",
                "global_max_source_outer_iter",
                "global_max_target_outer_iter",
                "all_seedwise_maxima_below_one",
                "seeds",
                "training_group",
                "metric_source",
            ],
        )
        write_csv(
            stage / "empirical_xev_ratio_runs_used.csv",
            enriched_run_rows,
            [
                "target_label",
                "group",
                "model_type",
                "n_assets",
                "m_states",
                "seed",
                "run_dir",
                "outer_iters",
                "market_hash",
                "policy_metric_source",
                "metric",
                "outer_history",
                "outer_history_sha256",
                "training_group",
                "metric_source",
                "exact_map_group",
                "evaluation_protocol_hash",
                "selected_checkpoints",
                "checkpoint_manifest_sha256",
                "training_protocol_hash",
                "weight_dir",
                "checkpoint_selection",
                "terminal_state_hash",
                "config",
                "config_sha256",
                "market_params",
                "market_params_sha256",
                "closed_form",
                "closed_form_sha256",
                "evaluation_grid_sha256",
            ],
        )
        if evidence_rows:
            write_csv(
                stage / "empirical_xev_reevaluated_trajectory.csv",
                evidence_rows,
                [
                    "target_label",
                    "group",
                    "training_group",
                    "exact_map_group",
                    "evaluation_protocol_hash",
                    "model_type",
                    "n_assets",
                    "m_states",
                    "metric",
                    "metric_source",
                    "seed",
                    "outer_iter",
                    "checkpoint",
                    "checkpoint_sha256",
                    "network_dtype",
                    "eval_margin",
                    "eval_x_margin",
                    "eval_w_min_override",
                    "eval_w_max_override",
                    "ev_w_min",
                    "ev_w_max",
                    "ev_x_min",
                    "ev_x_max",
                    "eval_nt",
                    "eval_ny",
                    "eval_nx",
                    "eval_points",
                    "evaluation_grid_sha256",
                    "e_input_value",
                    "e_input_vw",
                    "e_input_vww",
                    "e_input_vwx",
                    "e_input_bundle",
                    "e_input_X",
                    "e_Xev",
                    "run_dir",
                    "market_hash",
                    "training_protocol_hash",
                ],
            )
        write_json(stage / "empirical_xev_ratio_metadata.json", metadata)

        if not args.no_plot:
            fig = create_figure(
                floor_summary,
                main_floor_multiple=args.main_floor_multiple,
                y_scale=args.ratio_y_scale,
                figure_size=(args.fig_width, args.fig_height),
                font_size=args.font_size,
                font_family=args.font_family,
                line_width=args.line_width,
                marker_size=args.marker_size,
                band_alpha=args.band_alpha,
                floor_alpha=args.floor_alpha,
                grid_alpha=args.grid_alpha,
            )
            try:
                for suffix in formats:
                    fig.savefig(
                        stage / f"{PLOT_STEM}.{suffix}",
                        dpi=args.dpi,
                        bbox_inches=(
                            "tight" if args.bbox_inches == "tight" else None
                        ),
                    )
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)

        (stage / "_SUCCESS_EMPIRICAL_XEV_RATIO").touch()
        _commit_staged_output(stage, output)

    print(f"[done] empirical e_Xev ratio outputs: {output}")
    print(
        f"[info] target={args.target_label}; seeds={seeds}; "
        f"metric_source={args.metric_source}; "
        f"adjacent pairs per seed={len(outer_schedule) - 1}"
    )
    if reevaluation is not None:
        window = reevaluation["evaluation_window"]
        print(
            "[info] resolved checkpoint evaluation window: "
            f"w=[{window['ev_w_min']:.12g}, {window['ev_w_max']:.12g}], "
            f"x=[{window['ev_x_min']:.12g}, {window['ev_x_max']:.12g}]"
        )
        print("[info] finite-difference PDE solves: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
