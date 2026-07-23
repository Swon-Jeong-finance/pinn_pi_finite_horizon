"""Aggregate iteration-level Merton E1 diagnostics without pseudo-replication.

The training programs write one ``outer_history.csv`` row per outer block.
This script first reduces every seed over its outer iterations, then treats
the resulting seed-level extrema as the independent observations used for
sample SD, standard error, and Student-t 95% confidence intervals.

Merton-specific manuscript mapping
-----------------------------------
* ``diffusion_var_{min,max}_frozen`` is the sole eigenvalue range of the
  one-dimensional frozen state covariance ``pi.T @ Sigma @ pi`` on Q_col.
  Direct PINN has no frozen PI policy, so its greedy fields are used.
* ``m_y``, ``M_y``, and ``m_c`` are the Merton derivative margins on Q_ev.
* ``pi`` is already the wealth-normalized portfolio ``vartheta = theta / w``;
  ``chi`` is the wealth-normalized consumption ``c / w``.

The Kim--Omberg-only quantities ``m_ww``, ``M_num``, and
``M_num/(w_min*m_ww)`` are deliberately not fabricated for Merton.  Their
not-applicable status is written to ``e1_diagnostics_coverage.csv``.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (
    canonical_market_hash,
    find_runs,
    group_key,
    load_config_args_raw,
    parse_int_spec,
    parse_seed_spec,
    run_status,
    run_updated_at,
    t_crit_95,
)


@dataclass(frozen=True)
class MetricSpec:
    source: str
    reducer: str
    scope: str
    description: str


def primary_diag_margin(config: Mapping[str, Any]) -> float:
    """Return the first eval margin, which defines the training-time Q_ev."""
    raw = config.get("diag_margin", config.get("eval_margin", ""))
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("eval_margin is empty")
        raw = raw[0]
    else:
        text = str(raw).strip()
        if not text:
            raise ValueError("eval_margin/diag_margin is missing")
        raw = text.split(",", 1)[0]
    margin = _finite(raw, label="primary diagnostic margin")
    if margin < 0.0 or margin >= 0.5:
        raise ValueError(f"primary diagnostic margin must be in [0,0.5), got {margin}")
    return margin


def e1_group_key(config: Mapping[str, Any]) -> Tuple[str, str]:
    """Seed-independent training group augmented by E1's primary Q_ev margin.

    ``aggregate_seeds.group_key`` intentionally ignores all evaluation margins.
    E1 cannot: the first margin is used during training for fixed-Q_ev outer
    diagnostics.  Later margins remain evaluation-only and do not split E1.
    """
    _base_hash, base_canon = group_key(dict(config))
    core = json.loads(base_canon)
    core["_e1_primary_diag_margin"] = primary_diag_margin(config)
    canon = json.dumps(core, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12], canon


def metric_specs(model_type: str) -> Dict[str, MetricSpec]:
    """Return the exact CSV-to-paper mapping for one Merton method."""
    if model_type not in {"pinn", "pipinn"}:
        raise ValueError(f"unsupported model_type={model_type!r}; expected pinn or pipinn")
    policy_suffix = "frozen" if model_type == "pipinn" else "greedy"

    def policy_clip_source(base: str) -> str:
        return f"{base}_frozen" if model_type == "pipinn" else base

    policy_description = (
        "frozen alpha_n used by the PI policy-evaluation PDE"
        if model_type == "pipinn"
        else "direct-PINN greedy policy (no frozen PI policy exists)"
    )
    specs = {
        "diffusion_covariance_lambda_min": MetricSpec(
            f"diffusion_var_min_{policy_suffix}", "min", "Q_col",
            f"minimum pi^T Sigma pi for the {policy_description}"),
        "diffusion_covariance_lambda_max": MetricSpec(
            f"diffusion_var_max_{policy_suffix}", "max", "Q_col",
            f"maximum pi^T Sigma pi for the {policy_description}"),
        "m_y": MetricSpec("m_y", "min", "Q_ev", "minimum V_y"),
        "M_y": MetricSpec("M_y", "max", "Q_ev", "maximum V_y"),
        "m_c": MetricSpec(
            "m_c", "min", "Q_ev", "minimum log-wealth curvature V_y - V_yy"),
        "m_Vw": MetricSpec("m_Vw", "min", "Q_ev", "minimum wealth derivative V_w"),
        "m_minus_Vww": MetricSpec(
            "m_minus_Vww", "min", "Q_ev", "minimum -V_ww (auxiliary audit)"),
        "m_curvature_y": MetricSpec(
            "m_curvature_y", "min", "Q_ev", "minimum V_y - V_yy (alias audit)"),
        "vartheta_component_min": MetricSpec(
            f"pi_component_min_{policy_suffix}", "min", "Q_ev",
            f"minimum over grid points and asset components of {policy_description}"),
        "vartheta_component_max": MetricSpec(
            f"pi_component_max_{policy_suffix}", "max", "Q_ev",
            f"maximum over grid points and asset components of {policy_description}"),
        "vartheta_l2_min": MetricSpec(
            f"pi_l2_min_{policy_suffix}", "min", "Q_ev",
            f"minimum ||vartheta||_2 for the {policy_description}"),
        "vartheta_l2_max": MetricSpec(
            f"pi_l2_max_{policy_suffix}", "max", "Q_ev",
            f"maximum ||vartheta||_2 for the {policy_description}"),
        "chi_min": MetricSpec(
            f"chi_min_{policy_suffix}", "min", "Q_ev",
            f"minimum c/w for the {policy_description}"),
        "chi_max": MetricSpec(
            f"chi_max_{policy_suffix}", "max", "Q_ev",
            f"maximum c/w for the {policy_description}"),
        "policy_clip_fraction_max": MetricSpec(
            f"clip_frac_pi_{policy_suffix}", "max", "Q_ev",
            f"maximum portfolio-bound activation fraction for the {policy_description}"),
        "kappa_lower_clip_fraction_max": MetricSpec(
            policy_clip_source("clip_frac_kappa_low"), "max", "Q_ev",
            f"maximum lower kappa-bound activation fraction for the {policy_description}"),
        "kappa_upper_clip_fraction_max": MetricSpec(
            policy_clip_source("clip_frac_kappa_high"), "max", "Q_ev",
            f"maximum upper kappa-bound activation fraction for the {policy_description}"),
        "consumption_level_lower_clip_fraction_max": MetricSpec(
            policy_clip_source("clip_frac_c_level_low"), "max", "Q_ev",
            f"maximum lower consumption-level bound activation for the {policy_description}"),
        "consumption_level_upper_clip_fraction_max": MetricSpec(
            policy_clip_source("clip_frac_c_level_high"), "max", "Q_ev",
            f"maximum upper consumption-level bound activation for the {policy_description}"),
        "guard_Vw_fraction_max": MetricSpec(
            "guard_frac_Vw", "max", "Q_ev", "maximum V_w guard activation fraction"),
        "guard_curvature_fraction_max": MetricSpec(
            "guard_frac_curvature", "max", "Q_ev",
            "maximum V_y-V_yy guard activation fraction"),
        # Keep greedy diagnostics separately for PI-PINN because the final
        # improved alpha_{n+1} is not among alpha_0,...,alpha_{K-1} frozen in
        # the K policy-evaluation PDEs.
        "greedy_vartheta_component_min": MetricSpec(
            "pi_component_min_greedy", "min", "Q_ev",
            "minimum component of the improved/greedy policy"),
        "greedy_vartheta_component_max": MetricSpec(
            "pi_component_max_greedy", "max", "Q_ev",
            "maximum component of the improved/greedy policy"),
        "greedy_vartheta_l2_min": MetricSpec(
            "pi_l2_min_greedy", "min", "Q_ev", "minimum greedy portfolio L2 norm"),
        "greedy_vartheta_l2_max": MetricSpec(
            "pi_l2_max_greedy", "max", "Q_ev", "maximum greedy portfolio L2 norm"),
        "greedy_chi_min": MetricSpec(
            "chi_min_greedy", "min", "Q_ev", "minimum greedy c/w"),
        "greedy_chi_max": MetricSpec(
            "chi_max_greedy", "max", "Q_ev", "maximum greedy c/w"),
        "greedy_policy_clip_fraction_max": MetricSpec(
            "clip_frac_pi_greedy", "max", "Q_ev",
            "maximum greedy portfolio-bound activation fraction"),
        "greedy_kappa_lower_clip_fraction_max": MetricSpec(
            "clip_frac_kappa_low", "max", "Q_ev",
            "maximum lower kappa-bound activation fraction for the greedy policy"),
        "greedy_kappa_upper_clip_fraction_max": MetricSpec(
            "clip_frac_kappa_high", "max", "Q_ev",
            "maximum upper kappa-bound activation fraction for the greedy policy"),
        "greedy_consumption_level_lower_clip_fraction_max": MetricSpec(
            "clip_frac_c_level_low", "max", "Q_ev",
            "maximum lower consumption-level bound activation for the greedy policy"),
        "greedy_consumption_level_upper_clip_fraction_max": MetricSpec(
            "clip_frac_c_level_high", "max", "Q_ev",
            "maximum upper consumption-level bound activation for the greedy policy"),
    }
    if model_type == "pinn":
        # These aliases would duplicate the direct-PINN canonical policy
        # columns exactly; they are useful only for PI-PINN, where frozen and
        # newly improved policies are genuinely different sequences.
        specs = {key: value for key, value in specs.items()
                 if not key.startswith("greedy_")}
    return specs


@dataclass
class RunSummary:
    run_dir: str
    updated_at: str
    group: str
    config: Dict[str, Any]
    model_type: str
    n_assets: int
    seed: int
    status: str
    outer_first: int
    outer_last: int
    n_outer_rows: int
    metrics: Dict[str, float]
    sources: Dict[str, str]


def _atomic_csv(path: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-e1-", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-e1-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: not numeric ({value!r})") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite ({value!r})")
    return number


def _read_outer_rows(run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        raise ValueError("missing outer_history.csv")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("outer_history.csv has no data rows")
    return rows


def _summarize_run(
    run_dir: str,
    config: Dict[str, Any],
    status: str,
    *,
    allow_incomplete: bool,
) -> RunSummary:
    model = str(config.get("model_type", ""))
    specs = metric_specs(model)
    rows = _read_outer_rows(run_dir)
    outer: List[int] = []
    seen = set()
    for row_no, row in enumerate(rows, start=2):
        try:
            iteration = int(row.get("outer_iter", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {row_no}: invalid outer_iter={row.get('outer_iter')!r}") from exc
        if iteration <= 0 or iteration in seen:
            raise ValueError(f"row {row_no}: nonpositive or duplicate outer_iter={iteration}")
        seen.add(iteration)
        outer.append(iteration)
        row_model = str(row.get("model_type", model) or model)
        if row_model != model:
            raise ValueError(
                f"row {row_no}: model_type={row_model!r} disagrees with config {model!r}")
    order = np.argsort(np.asarray(outer))
    rows = [rows[int(i)] for i in order]
    outer = sorted(outer)
    if outer != list(range(1, outer[-1] + 1)):
        raise ValueError(f"outer iterations are not contiguous 1..{outer[-1]}: {outer}")
    expected = int(config.get("outer_iters", outer[-1]))
    if status == "success" and outer[-1] != expected:
        raise ValueError(
            f"successful run has outer iterations 1..{outer[-1]}, expected 1..{expected}")
    if int(config.get("diag_points", 0) or 0) <= 0:
        raise ValueError("diag_points must be positive for E1")
    if int(config.get("diag_every", 1) or 1) != 1 and not allow_incomplete:
        raise ValueError("paper E1 requires diag_every=1 (every outer iteration)")

    reduced: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    missing: List[str] = []
    for metric, spec in specs.items():
        vals: List[float] = []
        for iteration, row in zip(outer, rows):
            raw = row.get(spec.source, "")
            if str(raw).strip() == "":
                missing.append(f"outer={iteration}:{spec.source}")
                continue
            try:
                vals.append(_finite(raw, label=f"outer={iteration}:{spec.source}"))
            except ValueError as exc:
                missing.append(str(exc))
        if len(vals) != len(rows):
            continue
        reduced[metric] = min(vals) if spec.reducer == "min" else max(vals)
        sources[metric] = spec.source
    if missing and not allow_incomplete:
        preview = "; ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f"; ... ({len(missing)} total)"
        raise ValueError(f"incomplete E1 diagnostics: {preview}{extra}")
    if not reduced:
        raise ValueError("no complete E1 diagnostic field is available")
    scopes = {str(row.get("control_metric_scope", "")) for row in rows}
    if scopes != {"fixed_qev"}:
        raise ValueError(
            f"every control diagnostic must use fixed_qev; observed scopes={sorted(scopes)}")
    try:
        point_counts = {int(row.get("control_metric_points", "")) for row in rows}
    except (TypeError, ValueError) as exc:
        raise ValueError("control_metric_points is missing or invalid") from exc
    if len(point_counts) != 1 or next(iter(point_counts)) <= 0:
        raise ValueError(
            f"fixed_qev point count must be one positive constant; got {sorted(point_counts)}")

    group, _ = e1_group_key(config)
    return RunSummary(
        run_dir=run_dir,
        updated_at=run_updated_at(run_dir),
        group=group,
        config=config,
        model_type=model,
        n_assets=int(config["n_assets"]),
        seed=int(config["seed"]),
        status=status,
        outer_first=outer[0],
        outer_last=outer[-1],
        n_outer_rows=len(outer),
        metrics=reduced,
        sources=sources,
    )


def _seed_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")
    sem = sd / math.sqrt(n) if n >= 2 else float("nan")
    crit = t_crit_95(n - 1) if n >= 2 else float("nan")
    return {
        "mean": mean,
        "std": sd,
        "sem": sem,
        "ci95_lo": mean - crit * sem if n >= 2 else float("nan"),
        "ci95_hi": mean + crit * sem if n >= 2 else float("nan"),
    }


def aggregate_diagnostics(
    out_root: str,
    output: Optional[str] = None,
    *,
    expected_seeds: Sequence[int] = (),
    expected_n_assets: Sequence[int] = (),
    expected_models: Sequence[str] = (),
    min_seeds: int = 1,
    include_stopped: bool = False,
    allow_incomplete: bool = False,
    strict_market_snapshots: bool = False,
) -> Dict[str, str]:
    """Validate, aggregate, and write E1 artifacts; raise on paper-contract errors."""
    out_root = os.path.abspath(out_root)
    output = os.path.abspath(output or os.path.join(out_root, "diagnostic_summary"))
    os.makedirs(output, exist_ok=True)
    expected_seed_set = {int(x) for x in expected_seeds}
    expected_n_set = {int(x) for x in expected_n_assets}
    expected_model_set = {str(x) for x in expected_models}
    accepted = {"success"}
    if include_stopped:
        accepted.add("stopped_early")

    candidates: Dict[Tuple[str, str, int], Tuple[str, str, Dict[str, Any], str]] = {}
    index_rows: List[Dict[str, Any]] = []
    configs: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for run_dir in find_runs(out_root):
        config = load_config_args_raw(run_dir)
        if config is None or str(config.get("model_type", "")) not in {"pinn", "pipinn"}:
            continue
        if bool(config.get("timing_mode", False)):
            index_rows.append({
                "run_dir": os.path.relpath(run_dir, out_root), "selected": 0,
                "used": 0, "reason": "timing_mode_excluded",
            })
            continue
        status = run_status(run_dir)
        try:
            group, canon = e1_group_key(config)
        except ValueError as exc:
            errors.append(f"{run_dir}: {exc}")
            continue
        configs[group] = json.loads(canon)
        try:
            seed = int(config["seed"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{run_dir}: missing/invalid seed")
            continue
        key = (group, str(config["model_type"]), seed)
        updated = run_updated_at(run_dir)
        current = candidates.get(key)
        if current is None or (updated, run_dir) >= (current[0], current[1]):
            candidates[key] = (updated, run_dir, config, status)

    if not candidates:
        raise ValueError(f"no non-timing Merton runs found under {out_root}")

    summaries: List[RunSummary] = []
    market_by_n: Dict[int, set] = defaultdict(set)
    for (group, model, seed), (updated, run_dir, config, status) in sorted(candidates.items()):
        used = status in accepted
        idx = {
            "run_dir": os.path.relpath(run_dir, out_root), "updated_at": updated,
            "group": group, "model_type": model, "n_assets": config.get("n_assets"),
            "seed": seed, "status": status, "selected": 1, "used": int(used),
            "reason": "" if used else "status_not_accepted",
        }
        index_rows.append(idx)
        if not used:
            continue
        if strict_market_snapshots or expected_seed_set:
            try:
                market_hash = canonical_market_hash(os.path.join(run_dir, "market_params.npz"))
                market_by_n[int(config["n_assets"])].add(market_hash)
            except Exception as exc:
                errors.append(f"{run_dir}: invalid market_params.npz: {exc}")
        try:
            summaries.append(_summarize_run(
                run_dir, config, status, allow_incomplete=allow_incomplete))
        except Exception as exc:
            idx["used"] = 0
            idx["reason"] = f"diagnostic_validation_failed: {exc}"
            errors.append(f"{run_dir}: {exc}")

    groups: Dict[str, List[RunSummary]] = defaultdict(list)
    for summary in summaries:
        groups[summary.group].append(summary)
    if not groups:
        errors.append("no run passed E1 diagnostic validation")

    for group, runs in sorted(groups.items()):
        seeds = {run.seed for run in runs}
        if len(seeds) < int(min_seeds):
            errors.append(f"group={group}: {len(seeds)} seeds < min_seeds={min_seeds}")
        if expected_seed_set and seeds != expected_seed_set:
            errors.append(
                f"group={group}: seeds={sorted(seeds)}, expected={sorted(expected_seed_set)}")
    observed_n = {run.n_assets for run in summaries}
    observed_models = {run.model_type for run in summaries}
    if expected_n_set and observed_n != expected_n_set:
        errors.append(f"observed N={sorted(observed_n)}, expected exactly={sorted(expected_n_set)}")
    if expected_model_set and observed_models != expected_model_set:
        errors.append(
            f"observed models={sorted(observed_models)}, expected exactly={sorted(expected_model_set)}")
    if expected_model_set and expected_n_set:
        observed_pairs = {(run.model_type, run.n_assets) for run in summaries}
        expected_pairs = {
            (model, n_assets)
            for model in expected_model_set for n_assets in expected_n_set
        }
        missing_pairs = expected_pairs - observed_pairs
        if missing_pairs:
            errors.append(
                "missing method/N panels=" + str(sorted(missing_pairs, key=str)))
    if strict_market_snapshots or expected_seed_set:
        for n_assets in sorted(expected_n_set or observed_n):
            hashes = market_by_n.get(n_assets, set())
            if len(hashes) != 1:
                errors.append(
                    f"N={n_assets}: expected one canonical market hash, found {sorted(hashes)}")

    metric_names = sorted({name for run in summaries for name in run.metrics})
    per_run_rows: List[Dict[str, Any]] = []
    for run in sorted(summaries, key=lambda item: (item.group, item.seed)):
        row: Dict[str, Any] = {
            "group": run.group, "model_type": run.model_type,
            "n_assets": run.n_assets, "seed": run.seed, "status": run.status,
            "run_dir": os.path.relpath(run.run_dir, out_root),
            "outer_first": run.outer_first, "outer_last": run.outer_last,
            "n_outer_rows": run.n_outer_rows,
            "policy_bounds_mode": run.config.get("policy_bounds_mode", ""),
            "diag_points": run.config.get("diag_points", ""),
            "diag_margin": primary_diag_margin(run.config),
        }
        row.update(run.metrics)
        per_run_rows.append(row)

    long_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for group, runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda item: item.seed)
        model = runs[0].model_type
        specs = metric_specs(model)
        table: Dict[str, Any] = {
            "group": group, "model_type": model, "n_assets": runs[0].n_assets,
            "n_seeds": len(runs), "seeds": ",".join(str(run.seed) for run in runs),
            "policy_bounds_mode": runs[0].config.get("policy_bounds_mode", ""),
        }
        for metric, spec in specs.items():
            available = [run for run in runs if metric in run.metrics]
            coverage_rows.append({
                "group": group, "model_type": model, "n_assets": runs[0].n_assets,
                "concept": metric, "status": (
                    "available" if len(available) == len(runs) else "incomplete"),
                "source": spec.source, "scope": spec.scope,
                "note": spec.description,
            })
            if len(available) != len(runs):
                if not allow_incomplete:
                    errors.append(
                        f"group={group} metric={metric}: available for "
                        f"{len(available)}/{len(runs)} seeds")
                continue
            values = [run.metrics[metric] for run in runs]
            stats = _seed_stats(values)
            paper_extreme = min(values) if spec.reducer == "min" else max(values)
            long_rows.append({
                "group": group, "model_type": model, "n_assets": runs[0].n_assets,
                "metric": metric, "source": spec.source, "scope": spec.scope,
                "outer_reducer": spec.reducer, "setting_reducer": spec.reducer,
                "n_seeds": len(values), "seeds": ",".join(str(run.seed) for run in runs),
                "paper_extreme_across_seed_outer": paper_extreme,
                "seed_extrema_mean": stats["mean"], "seed_extrema_std": stats["std"],
                "seed_extrema_sem": stats["sem"],
                "seed_extrema_ci95_lo": stats["ci95_lo"],
                "seed_extrema_ci95_hi": stats["ci95_hi"],
                "description": spec.description,
            })
            table[f"{metric}__paper_extreme"] = paper_extreme
            table[f"{metric}__seed_mean"] = stats["mean"]
            table[f"{metric}__seed_sd"] = stats["std"]
            table[f"{metric}__ci95_lo"] = stats["ci95_lo"]
            table[f"{metric}__ci95_hi"] = stats["ci95_hi"]
        for concept, note in (
            ("m_ww", "Kim--Omberg-only derivative margin; not a Merton E1 quantity"),
            ("M_num", "Kim--Omberg-only numerator bound; not a Merton E1 quantity"),
            ("M_num_over_w_min_m_ww", "Kim--Omberg-only implied bound; not defined for Merton"),
        ):
            coverage_rows.append({
                "group": group, "model_type": model, "n_assets": runs[0].n_assets,
                "concept": concept, "status": "not_applicable_merton",
                "source": "", "scope": "", "note": note,
            })
        coverage_rows.append({
            "group": group, "model_type": model, "n_assets": runs[0].n_assets,
            "concept": "vartheta_per_asset_component_ranges",
            "status": "not_recorded",
            "source": "pi_component_min/max_*",
            "scope": "Q_ev",
            "note": (
                "CSV stores the global range over all asset components; exact per-asset "
                "ranges would require vector-valued trainer logging"),
        })
        table_rows.append(table)

    paths = {
        "runs_index": os.path.join(output, "e1_runs_index.csv"),
        "per_run": os.path.join(output, "e1_diagnostics_per_run.csv"),
        "summary_long": os.path.join(output, "e1_diagnostics_summary_long.csv"),
        "table": os.path.join(output, "e1_diagnostics_table.csv"),
        "coverage": os.path.join(output, "e1_diagnostics_coverage.csv"),
        "metadata": os.path.join(output, "e1_diagnostics_metadata.json"),
        "settings": os.path.join(output, "e1_diagnostics_settings.json"),
    }
    index_fields = [
        "run_dir", "updated_at", "group", "model_type", "n_assets", "seed",
        "status", "selected", "used", "reason",
    ]
    per_run_fields = [
        "group", "model_type", "n_assets", "seed", "status", "run_dir",
        "outer_first", "outer_last", "n_outer_rows", "policy_bounds_mode",
        "diag_points", "diag_margin", *metric_names,
    ]
    long_fields = [
        "group", "model_type", "n_assets", "metric", "source", "scope",
        "outer_reducer", "setting_reducer", "n_seeds", "seeds",
        "paper_extreme_across_seed_outer", "seed_extrema_mean", "seed_extrema_std",
        "seed_extrema_sem", "seed_extrema_ci95_lo", "seed_extrema_ci95_hi",
        "description",
    ]
    table_metric_names = sorted({key for row in table_rows for key in row if key not in {
        "group", "model_type", "n_assets", "n_seeds", "seeds", "policy_bounds_mode"}})
    table_fields = [
        "group", "model_type", "n_assets", "n_seeds", "seeds",
        "policy_bounds_mode", *table_metric_names,
    ]
    coverage_fields = [
        "group", "model_type", "n_assets", "concept", "status", "source", "scope", "note",
    ]
    _atomic_csv(paths["runs_index"], index_rows, index_fields)
    _atomic_csv(paths["per_run"], per_run_rows, per_run_fields)
    _atomic_csv(paths["summary_long"], long_rows, long_fields)
    _atomic_csv(paths["table"], table_rows, table_fields)
    _atomic_csv(paths["coverage"], coverage_rows, coverage_fields)
    _atomic_json(paths["settings"], configs)
    _atomic_json(paths["metadata"], {
        "schema_version": 1,
        "out_root": out_root,
        "n_selected_runs": len(summaries),
        "n_groups": len(groups),
        "inference_unit": "training_seed",
        "iteration_handling": (
            "reduce within each seed over outer iterations, then compute seed-level statistics"),
        "ci95": "two-sided Student-t interval over seed-level extrema",
        "paper_extreme": "global min/max across seeds and outer iterations",
        "merton_mapping": {
            "ellipticity": "scalar pi^T Sigma pi is the sole state-covariance eigenvalue",
            "derivative_margins": ["m_y", "M_y", "m_c"],
            "vartheta": "pi (already theta/w)",
            "chi": "c/w",
            "kim_omberg_only_not_computed": ["m_ww", "M_num", "M_num/(w_min*m_ww)"],
        },
        "validation_errors": errors,
    })
    if errors:
        raise ValueError("E1 aggregation validation failed:\n- " + "\n- ".join(errors))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Merton E1 iteration diagnostics by setting and seed.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--expected-n-assets", default="")
    parser.add_argument("--expected-models", default="")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--include-stopped", action="store_true")
    parser.add_argument(
        "--allow-incomplete-diagnostics", action="store_true",
        help="Write explicit coverage gaps instead of requiring every metric at every outer.")
    parser.add_argument("--strict-market-snapshots", action="store_true")
    args = parser.parse_args()
    try:
        paths = aggregate_diagnostics(
            args.out_root, args.output,
            expected_seeds=parse_seed_spec(args.expected_seeds),
            expected_n_assets=parse_int_spec(
                args.expected_n_assets, label="--expected-n-assets"),
            expected_models=[x.strip() for x in args.expected_models.split(",") if x.strip()],
            min_seeds=args.min_seeds,
            include_stopped=args.include_stopped,
            allow_incomplete=args.allow_incomplete_diagnostics,
            strict_market_snapshots=args.strict_market_snapshots,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for label, path in paths.items():
        print(f"[aggregate-e1] {label}: {path}")


if __name__ == "__main__":
    main()
