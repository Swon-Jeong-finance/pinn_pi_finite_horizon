#!/usr/bin/env python3
"""Aggregate the Merton E4 residual-to-approximation diagnostic.

For each residual-sweep run this script reads ``exact_map_defects.csv`` plus
its independent defect-level FD refinement table and forms

    p_hat_X = max_n ||v_tilde_n - v^{alpha_n}||_Xev.

The maximum includes ``delta_0`` and every available adjacent-checkpoint
defect.  Paper aggregation requires passing grid/domain/boundary sensitivity
evidence for ``delta_0``, the first and last adjacent defects, and whichever
primary defect attains the run-wise maximum.

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


FORMATS = {"png", "pdf", "svg", "eps"}


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


def exact_protocol_group(training_group: str, exact_cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """Separate E4 regressions whenever the numerical FD protocol differs."""
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
        # The producer hash now includes the exact primary evaluation window.
        # Keep the explicit window here as a backward-safe guard for legacy
        # configs whose recorded protocol hash predated that invariant.
        "recorded_protocol_hash": exact_cfg.get("protocol_hash"),
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
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    protocol = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{training_group}-{protocol[:8]}", protocol


def required_refinement_iterations(
    defect_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Minimum paper evidence: delta_0, first/last adjacent, and worst."""
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


def validate_defect_refinement_evidence(
    result_dir: Path,
    exact_cfg: Mapping[str, Any],
    defect_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    """Verify that required E4 statuses are backed by complete FD variants."""
    path = result_dir / "exact_map_defect_refinement.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: E4 requires defect-level FD refinement evidence"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        refinement_rows = list(csv.DictReader(handle))
    required_fields = {
        "protocol_hash", "defect_iter", "grid_factor", "fd_margin",
        "boundary", "is_primary", "is_verification", "delta_X",
        "defect_grid_abs_change", "defect_domain_abs_change",
        "defect_boundary_abs_change", "defect_sensitivity_envelope",
        "refinement_status",
    }
    if not refinement_rows:
        raise ValueError(f"{path}: empty defect-refinement table")
    missing = required_fields - set(refinement_rows[0])
    if missing:
        raise ValueError(
            f"{path}: missing defect-refinement fields {sorted(missing)}"
        )

    grid = exact_cfg.get("grid")
    grid = grid if isinstance(grid, Mapping) else {}
    factor_sequence = [int(value) for value in grid.get("grid_factors", [])]
    margin_sequence = [float(value) for value in grid.get("fd_margins", [])]
    boundary_sequence = [str(value) for value in grid.get("boundaries", [])]
    factors = set(factor_sequence)
    margins = set(margin_sequence)
    boundaries = set(boundary_sequence)
    if not factors or not margins or not boundaries:
        raise ValueError(f"{result_dir}: incomplete exact-map FD grid protocol")
    if (
        len(factors) != len(factor_sequence)
        or len(margins) != len(margin_sequence)
        or len(boundaries) != len(boundary_sequence)
    ):
        raise ValueError(f"{result_dir}: duplicate exact-map FD variants")
    expected_variants = {
        (factor, margin, boundary)
        for factor in factors
        for margin in margins
        for boundary in boundaries
    }
    expected_primary = (
        max(factors),
        min(margins),
        boundary_sequence[0],
    )
    declared_protocol = str(exact_cfg.get("protocol_hash", ""))
    if not declared_protocol:
        raise ValueError(f"{result_dir}: missing exact protocol_hash")

    required = required_refinement_iterations(defect_rows)
    primary_by_iter = {
        int(float(row["defect_iter"])): row for row in defect_rows
    }
    for defect_iter in required:
        candidates = [
            row for row in refinement_rows
            if int(float(row["defect_iter"])) == defect_iter
        ]
        if len(candidates) != len(expected_variants):
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} has "
                f"{len(candidates)} refinement rows, expected "
                f"{len(expected_variants)}"
            )
        observed_variants = {
            (
                int(float(row["grid_factor"])),
                float(row["fd_margin"]),
                str(row["boundary"]),
            )
            for row in candidates
        }
        if observed_variants != expected_variants:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} refinement variants "
                f"do not match the exact protocol"
            )
        primaries = [
            row for row in candidates if int(float(row["is_primary"])) == 1
        ]
        if len(primaries) != 1:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} requires one "
                "primary defect-refinement row"
            )
        primary = primaries[0]
        if any(
            str(row["protocol_hash"]) != declared_protocol
            for row in candidates
        ):
            raise ValueError(
                f"{result_dir}: defect refinement protocol hash mismatch"
            )
        primary_variant = (
            int(float(primary["grid_factor"])),
            float(primary["fd_margin"]),
            str(primary["boundary"]),
        )
        if primary_variant != expected_primary:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} primary variant "
                f"is {primary_variant}, expected {expected_primary}"
            )
        if any(
            int(float(row["is_verification"])) != 1
            for row in candidates
        ):
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} contains "
                "non-verification FD variants"
            )
        if any(not _float(row["delta_X"]) >= 0.0 for row in candidates):
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} contains invalid "
                "variant delta_X"
            )
        if int(float(primary["is_verification"])) != 1:
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} was not selected "
                "for FD refinement"
            )
        if str(primary["refinement_status"]) != "pass":
            raise ValueError(
                f"{result_dir}: defect_iter={defect_iter} refinement status "
                f"is {primary['refinement_status']!r}, expected 'pass'"
            )
        source = primary_by_iter[defect_iter]
        if str(source.get("refinement_status", "")) != "pass":
            raise ValueError(
                f"{result_dir}: primary defect row lacks passing evidence at "
                f"defect_iter={defect_iter}"
            )
        if not math.isclose(
            _float(primary["delta_X"]),
            _float(source["delta_X"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"{result_dir}: primary defect/refinement delta_X mismatch at "
                f"defect_iter={defect_iter}"
            )
        for field in (
            "defect_grid_abs_change",
            "defect_domain_abs_change",
            "defect_boundary_abs_change",
            "defect_sensitivity_envelope",
        ):
            if not _float(primary[field]) >= 0.0:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} has no finite "
                    f"{field} evidence"
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
    for marker in sorted(root.rglob("_SUCCESS_EXACT_MAP")):
        result = marker.parent
        if (result / "exact_map_defects.csv").is_file() and (
            result / "exact_map_config.json"
        ).is_file():
            yield result


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
        required_refinement = validate_defect_refinement_evidence(
            result_dir, exact_cfg, defect_rows
        )
        exact_status = load_json(result_dir / "exact_map_status.json")
        if exact_status.get("defect_refinement_evidence_status") != "pass":
            raise ValueError(
                f"{result_dir}: exact-map status does not certify passing "
                "defect-level refinement evidence"
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
        exact_schedule = [
            int(value) for value in exact_cfg.get("checkpoint_schedule_outer", [])
        ]
        status = load_json(run_dir / "status.json")
        completed_outer = int(
            status.get(
                "completed_outer_iters",
                status.get("final_outer_iter", 0),
            )
        )
        full_schedule = list(range(1, completed_outer + 1))
        if completed_outer < 2 or exact_schedule != full_schedule:
            raise ValueError(
                f"{result_dir}: E4 requires every outer checkpoint 1.."
                f"{completed_outer}; found {exact_schedule}. Train with "
                "e3b_checkpoints=false and save_iterate_every=1."
            )
        if len(defect_rows) != completed_outer:
            raise ValueError(
                f"{result_dir}: expected {completed_outer} complete E4 "
                f"defects, found {len(defect_rows)}"
            )
        expected_defect_iters = list(range(0, completed_outer))
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
            if target_outer != defect_iter + 1:
                raise ValueError(
                    f"{result_dir}: defect_iter={defect_iter} is attached to "
                    f"outer={target_outer}, expected {defect_iter + 1}"
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
            "defect_coverage": "complete_all_outer_iterations_including_delta0",
            "defect_refinement_statuses": ";".join(defect_refinement),
            "defect_refinement_required_iterations": ";".join(
                str(value) for value in required_refinement
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
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for color, fit_row in zip(colors, fits):
        group = str(fit_row["group"])
        rows = sorted(
            [row for row in per_target if str(row["group"]) == group],
            key=lambda row: float(row["achieved_pres_mean"]),
        )
        x = np.asarray([float(row["achieved_pres_mean"]) for row in rows])
        y = np.asarray([float(row["p_hat_X_mean"]) for row in rows])
        yerr = np.asarray([float(row["p_hat_X_std"]) for row in rows])
        ax.errorbar(x, y, yerr=yerr, marker="o", color=color, capsize=2.5, label=group)
        c_num = float(fit_row["C_num_empirical_upper"])
        xx = np.geomspace(float(x.min()), float(x.max()), 100)
        ax.plot(xx, c_num * xx, linestyle="--", color=color, linewidth=1.0,
                label=rf"$C_{{\rm num}}={c_num:.2g}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Official post-restore residual $p_{\mathrm{res}}$")
    ax.set_ylabel(r"Measured tolerance $\widehat p_X$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(output / f"regularity_transfer.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Merton E4 regularity-transfer evidence.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--n-assets", type=int, default=None)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2")
    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    formats = parse_formats(args.formats)
    root = Path(args.out_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "regularity_transfer"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(root, n_assets=args.n_assets, run_name_regex=args.run_name_regex)
    expected = set(parse_seed_spec(args.expected_seeds))
    validate_panel(rows, expected, args.min_seeds)
    per_target, fits = build_summaries(rows)
    write_csv(output / "regularity_transfer_runs.csv", rows, [
        "group", "training_group", "exact_protocol_hash",
        "source_protocol_hash", "run_dir", "result_dir",
        "n_assets", "seed", "market_seed",
        "market_hash", "pres_target", "achieved_pres_post_restore", "p_hat_X",
        "C_num_run", "n_outer_residuals", "n_adjacent_defects",
        "completed_outer_iters", "defect_coverage",
        "defect_refinement_statuses",
        "defect_refinement_required_iterations",
        "eval_margin", "ev_y_min", "ev_y_max",
        "defect_iter_min", "defect_iter_max", "residual_semantics", "updated_at",
    ])
    write_csv(output / "regularity_transfer_per_target.csv", per_target, [
        "group", "pres_target", "n_seeds", "seeds", "achieved_pres_mean",
        "achieved_pres_std", "achieved_pres_sem", "achieved_pres_ci95_low",
        "achieved_pres_ci95_high", "p_hat_X_mean", "p_hat_X_std", "p_hat_X_sem",
        "p_hat_X_ci95_low", "p_hat_X_ci95_high", "C_num_target_max",
    ])
    write_csv(output / "regularity_transfer_fit.csv", fits, [
        "group", "n_points", "n_seeds", "slope", "intercept", "r2",
        "cluster_slope_se", "cluster_ci95_low", "cluster_ci95_high",
        "C_num_empirical_upper", "fit_definition",
    ])
    with (output / "regularity_transfer_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "arguments": vars(args),
            "residual_semantics": "max over official post-restore outer states",
            "p_hat_definition": (
                "max delta_X within each run, including delta_0 and every "
                "adjacent-checkpoint defect"
            ),
            "C_num_definition": "max_run p_hat_X / p_res_post_restore",
            "fd_interpretation": (
                "Primary delta_X uses one fixed FD protocol. A separate "
                "defect-level table recomputes delta_X over the configured "
                "grid/domain/boundary variants; E4 requires passing evidence "
                "for delta_0, the first and last adjacent defects, and the "
                "worst primary delta_X. Exact-ratio sensitivity status is "
                "never reused as defect evidence."
            ),
            "n_runs": len(rows),
        }, handle, indent=2, sort_keys=True)
    make_figure(per_target, fits, output, formats, args.dpi)
    print(f"[done] Merton E4 regularity-transfer outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
