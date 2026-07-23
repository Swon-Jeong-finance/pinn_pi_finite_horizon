#!/usr/bin/env python3
"""Aggregate Liu E1 diagnostics without modifying training artifacts.

The aggregation unit is a training seed.  Each seed is first reduced over
outer iterations using the assumption-relevant worst direction (for example,
minimum ellipticity and maximum guard fraction).  Mean, sample standard
deviation, and Student-t 95% intervals are then computed across those seed
extrema, avoiding pseudo-replication of outer rows.

PI-PINN indexing is kept explicit: ellipticity belongs to the frozen policy
``alpha_{n-1}``, while curvature/numerator/normalized-control diagnostics
belong to the improved policy ``alpha_n`` represented by the completed value
iterate.  Direct PINN rows are labelled as greedy-policy diagnostics at the
pseudo-outer training block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from audit_run_artifacts import (
        AuditError,
        discover_runs,
        inspect_run,
        parse_int,
        parse_seed_spec,
        read_json,
        read_outer_history,
        sha256_file,
        utc_now,
    )
    from aggregate_seeds import t_crit_95
except ImportError:  # package-style import during tests
    from .audit_run_artifacts import (  # type: ignore
        AuditError,
        discover_runs,
        inspect_run,
        parse_int,
        parse_seed_spec,
        read_json,
        read_outer_history,
        sha256_file,
        utc_now,
    )
    from .aggregate_seeds import t_crit_95  # type: ignore


SCHEMA_VERSION = 1
OUTPUT_FILES = (
    "diagnostics_raw.csv",
    "diagnostics_per_seed.csv",
    "diagnostics_setting_summary.csv",
    "diagnostics_assumption_summary.csv",
    "diagnostic_groups.json",
    "diagnostics_manifest.json",
)

# These values cannot alter the learned network or the training-time
# diagnostic design.  In particular, e3b_checkpoints is only a saving knob.
GROUP_IGNORE_KEYS = {
    "seed", "run_tag", "device", "output_root", "weight_root",
    "stop_flag_path", "eval_only", "skip_plots", "skip_figures", "skip_eval",
    "print_every", "print_every_outer", "print_every_eval", "verbose_detail",
    "save_iterate_every", "e3b_checkpoints", "w_levels", "n_tau", "n_x",
    "test_points",
}

METRIC_REDUCTIONS: Dict[str, str] = {
    "lambda_min_sigma": "min",
    "lambda_max_sigma": "max",
    "m_ww": "min",
    "M_num": "max",
    "implied_control_bound": "max",
    "vartheta_l2_min": "min",
    "vartheta_l2_max": "max",
    "vartheta_component_min": "min",
    "vartheta_component_max": "max",
    "vartheta_abs_max": "max",
    "guard_frac_ev": "max",
    "clip_frac": "max",
}

COMMON_REQUIRED = (
    "m_ww", "M_num", "guard_frac_ev",
    "vartheta_l2_min", "vartheta_l2_max",
    "vartheta_component_min", "vartheta_component_max", "vartheta_abs_max",
)


class DiagnosticsError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_group(args: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    payload = {key: args[key] for key in sorted(args) if key not in GROUP_IGNORE_KEYS}
    text = canonical_json(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], payload


def parse_csv_list(text: str) -> List[str]:
    values = [item.strip() for item in re.split(r"[,\s]+", str(text or "")) if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate values: {text!r}")
    return values


def primary_margin(args: Mapping[str, Any]) -> float:
    raw = str(args.get("eval_margin", "0.10"))
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise DiagnosticsError("eval_margin has no values")
    margin = values[0]
    if not (0.0 <= margin < 1.0):
        raise DiagnosticsError(f"invalid primary eval margin {margin}")
    return margin


def evaluation_w_min(args: Mapping[str, Any]) -> float:
    lo = float(args.get("w_min"))
    hi = float(args.get("w_max"))
    margin = primary_margin(args)
    value = lo + 0.5 * margin * (hi - lo)
    if not math.isfinite(value) or value <= 0.0 or hi <= lo:
        raise DiagnosticsError(f"invalid wealth window [{lo}, {hi}] at margin {margin}")
    return value


def optional_float(value: Any, label: str, *, required: bool = True) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            raise DiagnosticsError(f"missing diagnostic value: {label}")
        return None
    try:
        number = float(text)
    except Exception as exc:
        raise DiagnosticsError(f"invalid diagnostic value {label}={value!r}") from exc
    if not math.isfinite(number):
        raise DiagnosticsError(f"non-finite diagnostic value {label}={value!r}")
    return number


def _bool_config(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clipping_enabled(args: Mapping[str, Any], model_type: str) -> bool:
    if model_type != "pipinn":
        return False
    value = args.get("theta_clip_abs")
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return False
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DiagnosticsError(f"invalid theta_clip_abs={value!r}")
    return True


def _validate_ranges(row: Mapping[str, Any], context: str) -> None:
    lmin = float(row["lambda_min_sigma"])
    lmax = float(row["lambda_max_sigma"])
    if lmax < lmin:
        raise DiagnosticsError(f"{context}: lambda_max < lambda_min")
    if float(row["M_num"]) < 0:
        raise DiagnosticsError(f"{context}: M_num < 0")
    l2_min = float(row["vartheta_l2_min"])
    l2_max = float(row["vartheta_l2_max"])
    comp_min = float(row["vartheta_component_min"])
    comp_max = float(row["vartheta_component_max"])
    abs_max = float(row["vartheta_abs_max"])
    if l2_min < 0 or l2_max < l2_min:
        raise DiagnosticsError(f"{context}: invalid vartheta L2 range")
    if comp_max < comp_min or abs_max < 0:
        raise DiagnosticsError(f"{context}: invalid vartheta component range")
    component_abs = max(abs(comp_min), abs(comp_max))
    tolerance = 1e-7 * max(1.0, abs_max, component_abs)
    if abs(abs_max - component_abs) > tolerance:
        raise DiagnosticsError(
            f"{context}: vartheta_abs_max={abs_max} disagrees with component extrema {component_abs}"
        )
    if l2_max + tolerance < abs_max:
        raise DiagnosticsError(f"{context}: vartheta_l2_max < vartheta_abs_max")
    for field in ("guard_frac_ev", "clip_frac"):
        value = row.get(field)
        if value is not None and not (0.0 <= float(value) <= 1.0):
            raise DiagnosticsError(f"{context}: {field} outside [0,1]")


def normalized_diagnostic_rows(
    run_dir: Path,
    args: Mapping[str, Any],
    market_hash: str,
    group_hash: str,
    rows: Sequence[Mapping[str, str]],
    *,
    allow_sparse: bool = False,
) -> List[Dict[str, Any]]:
    model_type = str(args["model_type"])
    seed = parse_int(args["seed"], "seed")
    n_assets = parse_int(args["n_assets"], "n_assets")
    m_states = parse_int(args["m_states"], "m_states")
    risk_mode = str(args.get("risk_premium_mode", "affine"))
    epsilon = float(args.get("nonaffine_eps", 0.0) or 0.0)
    w_min = evaluation_w_min(args)
    margin = primary_margin(args)
    clip_enabled = clipping_enabled(args, model_type)
    if parse_int(args.get("diag_points", 0), "diag_points") <= 0:
        raise DiagnosticsError(f"{run_dir}: diag_points must be positive")
    diag_every = parse_int(args.get("diag_every", 1), "diag_every")
    if not allow_sparse and diag_every != 1:
        raise DiagnosticsError(f"{run_dir}: E1 requires diag_every=1, got {diag_every}")

    if model_type == "pipinn":
        lambda_min_field = "lam_min_sigma_frozen"
        lambda_max_field = "lam_max_sigma_frozen"
        policy_kind = "frozen"
    elif model_type == "pinn":
        lambda_min_field = "lam_min_sigma_greedy"
        lambda_max_field = "lam_max_sigma_greedy"
        policy_kind = "greedy"
    else:
        raise DiagnosticsError(f"{run_dir}: unsupported model_type={model_type!r}")

    out: List[Dict[str, Any]] = []
    for row_number, source in enumerate(rows, 1):
        outer = parse_int(source.get("outer_iter"), f"outer_iter row {row_number}")
        required_values = list(COMMON_REQUIRED) + [lambda_min_field, lambda_max_field]
        if allow_sparse and any(not str(source.get(field, "")).strip() for field in required_values):
            continue
        values = {
            field: optional_float(source.get(field), f"{run_dir}: outer {outer} {field}")
            for field in COMMON_REQUIRED
        }
        lambda_min = optional_float(
            source.get(lambda_min_field), f"{run_dir}: outer {outer} {lambda_min_field}"
        )
        lambda_max = optional_float(
            source.get(lambda_max_field), f"{run_dir}: outer {outer} {lambda_max_field}"
        )

        clip_value: Optional[float] = None
        if clip_enabled:
            clip_value = optional_float(
                source.get("clip_frac_frozen"),
                f"{run_dir}: outer {outer} clip_frac_frozen",
            )
        # A disabled/unavailable clipping diagnostic remains NA, never a fake zero.
        if model_type == "pipinn":
            frozen_index = parse_int(source.get("frozen_policy_iter"), "frozen_policy_iter")
            improved_index = parse_int(source.get("improved_policy_iter"), "improved_policy_iter")
            if frozen_index != outer - 1 or improved_index != outer:
                raise DiagnosticsError(
                    f"{run_dir}: outer {outer} has frozen={frozen_index}, improved={improved_index}"
                )
            lambda_policy_iter: Any = frozen_index
            margin_policy_iter: Any = improved_index
            outer_semantics = "policy_evaluation_iterate"
        else:
            frozen_index = ""
            improved_index = ""
            lambda_policy_iter = outer
            margin_policy_iter = outer
            outer_semantics = "direct_training_block"

        m_ww = float(values["m_ww"])
        M_num = float(values["M_num"])
        implied_bound = M_num / (w_min * m_ww) if m_ww > 0 else math.inf
        normalized: Dict[str, Any] = {
            "group": group_hash,
            "run_dir": str(run_dir.resolve()),
            "run_tag": str(args["run_tag"]),
            "model_type": model_type,
            "n_assets": n_assets,
            "m_states": m_states,
            "risk_premium_mode": risk_mode,
            "nonaffine_eps": epsilon,
            "seed": seed,
            "outer_iter": outer,
            "outer_semantics": outer_semantics,
            "policy_kind": policy_kind,
            "frozen_policy_iter": frozen_index,
            "improved_policy_iter": improved_index,
            "lambda_policy_iter": lambda_policy_iter,
            "margin_policy_iter": margin_policy_iter,
            "primary_eval_margin": margin,
            "w_min": w_min,
            "market_hash": market_hash,
            "lambda_min_sigma": float(lambda_min),
            "lambda_max_sigma": float(lambda_max),
            **{key: float(value) for key, value in values.items()},
            "implied_control_bound": implied_bound,
            "clipping_enabled": int(clip_enabled),
            "clip_frac": clip_value,
            "ellipticity_positive": int(float(lambda_min) > 0.0),
            "concavity_margin_positive": int(m_ww > 0.0),
        }
        _validate_ranges(normalized, f"{run_dir}: outer {outer}")
        out.append(normalized)
    if not out:
        raise DiagnosticsError(f"{run_dir}: no complete diagnostic rows")
    if not allow_sparse and len(out) != len(rows):
        raise DiagnosticsError(f"{run_dir}: not every outer row has diagnostics")
    return out


def reduce_values(values: Sequence[float], direction: str) -> float:
    if not values:
        raise DiagnosticsError("cannot reduce an empty metric")
    return min(values) if direction == "min" else max(values)


def per_seed_extrema(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["group"]), int(row["seed"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (group_hash, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["outer_iter"]))
        first = rows[0]
        item: Dict[str, Any] = {
            key: first[key]
            for key in (
                "group", "model_type", "n_assets", "m_states", "risk_premium_mode",
                "nonaffine_eps", "seed", "run_tag", "run_dir", "market_hash",
                "policy_kind", "primary_eval_margin", "w_min", "clipping_enabled",
            )
        }
        item.update(
            n_outer=len(rows),
            outer_first=int(rows[0]["outer_iter"]),
            outer_last=int(rows[-1]["outer_iter"]),
        )
        for metric, direction in METRIC_REDUCTIONS.items():
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            item[metric] = reduce_values(values, direction) if values else None
        item["all_elliptic"] = int(all(float(row["lambda_min_sigma"]) > 0 for row in rows))
        item["all_concave_margin"] = int(all(float(row["m_ww"]) > 0 for row in rows))
        item["guard_identically_zero"] = int(all(float(row["guard_frac_ev"]) == 0 for row in rows))
        item["clip_identically_zero"] = (
            int(all(float(row["clip_frac"]) == 0 for row in rows))
            if int(first["clipping_enabled"]) else None
        )
        output.append(item)
    return output


def summarize_seed_values(values: Sequence[float]) -> Dict[str, Any]:
    total_n = len(values)
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size != total_n or total_n == 0:
        return {
            "n": total_n, "finite_n": int(finite.size), "mean": None, "std": None,
            "sem": None, "ci95_lo": None, "ci95_hi": None, "t_crit": None,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    mean = float(np.mean(finite))
    if total_n > 1:
        std = float(np.std(finite, ddof=1))
        sem = std / math.sqrt(total_n)
        tc = float(t_crit_95(total_n - 1))
        lo, hi = mean - tc * sem, mean + tc * sem
    else:
        # A one-seed pilot has no estimable sample variance.  Do not encode a
        # fictitious zero-SD result that could leak into a paper table.
        std, sem, tc, lo, hi = None, None, None, None, None
    return {
        "n": total_n, "finite_n": int(finite.size), "mean": mean, "std": std,
        "sem": sem, "ci95_lo": lo, "ci95_hi": hi, "t_crit": tc,
        "min": float(np.min(finite)), "max": float(np.max(finite)),
    }


def setting_summaries(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["group"])].append(row)
    output: List[Dict[str, Any]] = []
    for group_hash, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        first = rows[0]
        seeds = ";".join(str(row["seed"]) for row in rows)
        for metric, reduction in METRIC_REDUCTIONS.items():
            present = [row[metric] for row in rows if row.get(metric) is not None]
            observed_values = [float(value) for value in present]
            stats = summarize_seed_values(observed_values)
            output.append({
                "group": group_hash,
                "model_type": first["model_type"],
                "n_assets": first["n_assets"],
                "m_states": first["m_states"],
                "risk_premium_mode": first["risk_premium_mode"],
                "nonaffine_eps": first["nonaffine_eps"],
                "policy_kind": first["policy_kind"],
                "metric": metric,
                "outer_reduction": reduction,
                "applicability": (
                    "not_applicable" if metric == "clip_frac" and not int(first["clipping_enabled"])
                    else "observed"
                ),
                **stats,
                "global_worst": (
                    reduce_values(observed_values, reduction) if observed_values else None
                ),
                "seeds": seeds,
            })
    return output


def assumption_summaries(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[str(row["group"])].append(row)
    output: List[Dict[str, Any]] = []
    flags = (
        "all_elliptic", "all_concave_margin", "guard_identically_zero",
        "clip_identically_zero",
    )
    for group_hash, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        first = rows[0]
        for flag in flags:
            applicable = [int(row[flag]) for row in rows if row.get(flag) is not None]
            output.append({
                "group": group_hash,
                "model_type": first["model_type"],
                "n_assets": first["n_assets"],
                "m_states": first["m_states"],
                "risk_premium_mode": first["risk_premium_mode"],
                "nonaffine_eps": first["nonaffine_eps"],
                "assumption_check": flag,
                "n_applicable": len(applicable),
                "n_pass": sum(applicable),
                "fraction_pass": (sum(applicable) / len(applicable)) if applicable else None,
                "seeds": ";".join(str(row["seed"]) for row in rows),
            })
    return output


def _run_updated_key(run_dir: Path) -> Tuple[str, int, str]:
    try:
        status = read_json(run_dir / "status.json")
        updated = str(status.get("updated_at", ""))
    except Exception:
        updated = ""
    try:
        mtime = int((run_dir / "config.json").stat().st_mtime_ns)
    except OSError:
        mtime = 0
    return updated, mtime, str(run_dir)


def select_latest_candidates(
    run_dirs: Sequence[Path],
    *,
    models: Sequence[str],
    m_states: Sequence[int],
    expected_n_assets: Optional[int],
) -> List[Tuple[Path, Dict[str, Any], str, Dict[str, Any]]]:
    newest: Dict[Tuple[str, str, int], Tuple[Path, Dict[str, Any], str, Dict[str, Any]]] = {}
    for run_dir in run_dirs:
        try:
            config = read_json(run_dir / "config.json")
            args = config.get("args")
            if not isinstance(args, dict):
                continue
            model = str(args.get("model_type", ""))
            m_value = parse_int(args.get("m_states"), "m_states")
            n_value = parse_int(args.get("n_assets"), "n_assets")
            seed = parse_int(args.get("seed"), "seed")
            if models and model not in models:
                continue
            if m_states and m_value not in m_states:
                continue
            if expected_n_assets is not None and n_value != expected_n_assets:
                continue
            group_hash, payload = config_group(args)
            key = (group_hash, model, seed)
            item = (run_dir, config, group_hash, payload)
            if key not in newest or _run_updated_key(run_dir) > _run_updated_key(newest[key][0]):
                newest[key] = item
        except Exception:
            continue
    return sorted(newest.values(), key=lambda item: str(item[0]))


def collect_diagnostics(
    out_root: Path,
    *,
    models: Sequence[str] = ("pinn", "pipinn"),
    m_states: Sequence[int] = (),
    expected_n_assets: Optional[int] = None,
    expected_seeds: Sequence[int] = (),
    min_seeds: int = 1,
    allow_sparse: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    candidates = select_latest_candidates(
        discover_runs(out_root), models=models, m_states=m_states,
        expected_n_assets=expected_n_assets,
    )
    if not candidates:
        raise DiagnosticsError("no matching Liu runs found")
    raw: List[Dict[str, Any]] = []
    run_index: List[Dict[str, Any]] = []
    groups: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for run_dir, config, config_hash, payload in candidates:
        try:
            observation = inspect_run(run_dir, inspect_checkpoints=False)
            args = config["args"]
            # Include market identity in the final diagnostic group.
            combined = canonical_json({"config": payload, "market_hash": observation["market_canonical_sha256"]})
            group_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
            groups[group_hash] = {
                "config": payload,
                "market_hash": observation["market_canonical_sha256"],
            }
            _, rows, _ = read_outer_history(run_dir / "outer_history.csv")
            normalized = normalized_diagnostic_rows(
                run_dir, args, observation["market_canonical_sha256"], group_hash,
                rows, allow_sparse=allow_sparse,
            )
            raw.extend(normalized)
            run_index.append({
                "group": group_hash,
                "run_dir": str(run_dir.resolve()),
                "model_type": args["model_type"],
                "n_assets": int(args["n_assets"]),
                "m_states": int(args["m_states"]),
                "seed": int(args["seed"]),
                "market_hash": observation["market_canonical_sha256"],
                "outer_rows": len(normalized),
            })
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")
    if errors:
        raise DiagnosticsError("invalid selected runs:\n  " + "\n  ".join(errors))

    cells: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_index:
        cells[(str(row["model_type"]), int(row["m_states"]))].append(row)
    if m_states:
        requested_cells = {(str(model), int(m)) for model in models for m in m_states}
        missing_cells = requested_cells - set(cells)
        if missing_cells:
            raise DiagnosticsError(f"missing requested method/M cells: {sorted(missing_cells)}")
    expected_set = set(int(seed) for seed in expected_seeds)
    for cell, entries in sorted(cells.items()):
        group_ids = {str(entry["group"]) for entry in entries}
        if len(group_ids) != 1:
            raise DiagnosticsError(f"cell {cell} contains multiple training/config groups: {sorted(group_ids)}")
        seeds = {int(entry["seed"]) for entry in entries}
        if expected_set and seeds != expected_set:
            raise DiagnosticsError(
                f"cell {cell} expected seeds {sorted(expected_set)}, observed {sorted(seeds)}"
            )
        if len(seeds) < min_seeds:
            raise DiagnosticsError(f"cell {cell} has {len(seeds)} seeds, minimum is {min_seeds}")

    # Both methods and all seeds for a fixed M must refer to one market.
    markets_by_m: Dict[int, set[str]] = defaultdict(set)
    for row in run_index:
        markets_by_m[int(row["m_states"])].add(str(row["market_hash"]))
    bad_markets = {m: hashes for m, hashes in markets_by_m.items() if len(hashes) != 1}
    if bad_markets:
        raise DiagnosticsError(
            "market mismatch within M: "
            + "; ".join(f"M={m}: {sorted(hashes)}" for m, hashes in sorted(bad_markets.items()))
        )
    return raw, run_index, groups


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return ""
        return f"{value:.17g}"
    return value


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_outputs(
    output: Path,
    raw: Sequence[Mapping[str, Any]],
    per_seed: Sequence[Mapping[str, Any]],
    setting: Sequence[Mapping[str, Any]],
    assumptions: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Any],
    run_index: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing and not overwrite:
        raise DiagnosticsError(f"output files exist; pass --overwrite: {existing}")

    raw_fields = [
        "group", "run_dir", "run_tag", "model_type", "n_assets", "m_states",
        "risk_premium_mode", "nonaffine_eps", "seed", "outer_iter", "outer_semantics",
        "policy_kind", "frozen_policy_iter", "improved_policy_iter", "lambda_policy_iter",
        "margin_policy_iter", "primary_eval_margin", "w_min", "market_hash",
        *METRIC_REDUCTIONS.keys(), "clipping_enabled", "ellipticity_positive",
        "concavity_margin_positive",
    ]
    seed_fields = [
        "group", "model_type", "n_assets", "m_states", "risk_premium_mode",
        "nonaffine_eps", "seed", "run_tag", "run_dir", "market_hash", "policy_kind",
        "primary_eval_margin", "w_min", "clipping_enabled", "n_outer", "outer_first",
        "outer_last", *METRIC_REDUCTIONS.keys(), "all_elliptic", "all_concave_margin",
        "guard_identically_zero", "clip_identically_zero",
    ]
    setting_fields = [
        "group", "model_type", "n_assets", "m_states", "risk_premium_mode",
        "nonaffine_eps", "policy_kind", "metric", "outer_reduction", "applicability",
        "n", "finite_n", "mean", "std", "sem", "ci95_lo", "ci95_hi", "t_crit",
        "min", "max", "global_worst", "seeds",
    ]
    assumption_fields = [
        "group", "model_type", "n_assets", "m_states", "risk_premium_mode",
        "nonaffine_eps", "assumption_check", "n_applicable", "n_pass",
        "fraction_pass", "seeds",
    ]
    atomic_write_csv(output / "diagnostics_raw.csv", raw, raw_fields)
    atomic_write_csv(output / "diagnostics_per_seed.csv", per_seed, seed_fields)
    atomic_write_csv(output / "diagnostics_setting_summary.csv", setting, setting_fields)
    atomic_write_csv(output / "diagnostics_assumption_summary.csv", assumptions, assumption_fields)
    atomic_write_json(output / "diagnostic_groups.json", groups)
    atomic_write_json(output / "diagnostics_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "liu_e1_diagnostics",
        "generated_at": utc_now(),
        "tool_path": str(Path(__file__).resolve()),
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
        "run_count": len(run_index),
        "raw_row_count": len(raw),
        "seed_row_count": len(per_seed),
        "setting_row_count": len(setting),
        "runs": list(run_index),
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", default="pinn,pipinn")
    parser.add_argument("--m-states", default="", help="Optional comma-separated M values.")
    parser.add_argument("--expected-n-assets", type=int)
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument(
        "--allow-sparse-diagnostics", action="store_true",
        help="Exploratory mode: accept rows omitted according to diag_every.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    models = parse_csv_list(args.models)
    invalid_models = set(models) - {"pinn", "pipinn"}
    if invalid_models:
        raise SystemExit(f"invalid models: {sorted(invalid_models)}")
    m_values = [int(value) for value in parse_csv_list(args.m_states)]
    seeds = parse_seed_spec(args.expected_seeds)
    out_root = args.out_root.resolve()
    output = (args.output or (out_root / "diagnostics_summary")).resolve()
    try:
        raw, run_index, groups = collect_diagnostics(
            out_root,
            models=models,
            m_states=m_values,
            expected_n_assets=args.expected_n_assets,
            expected_seeds=seeds,
            min_seeds=args.min_seeds,
            allow_sparse=args.allow_sparse_diagnostics,
        )
        per_seed = per_seed_extrema(raw)
        setting = setting_summaries(per_seed)
        assumptions = assumption_summaries(per_seed)
        for run in run_index:
            run_dir = Path(str(run["run_dir"])).resolve()
            try:
                output.relative_to(run_dir)
            except ValueError:
                continue
            raise DiagnosticsError(
                f"output must be separate from every training run: {output} is under {run_dir}"
            )
        write_outputs(
            output, raw, per_seed, setting, assumptions, groups, run_index,
            overwrite=args.overwrite,
        )
    except (AuditError, DiagnosticsError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    print(
        f"[diagnostics] wrote {output}: {len(run_index)} runs, "
        f"{len(raw)} outer rows, {len(per_seed)} seed summaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
