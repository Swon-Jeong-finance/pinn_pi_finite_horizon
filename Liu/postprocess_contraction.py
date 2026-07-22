#!/usr/bin/env python3
"""Post-process Liu outer histories into paper contraction trajectories.

Figure 2 is PI-PINN only.  Its empirical ratio is formed within each training
seed first,

    rho_n^(s) = e_Xev[n+1] / e_Xev[n],

then aggregated across seeds.  Ratios of seed-mean errors are never used.
The main support is the intersection of seed-wise regular regions
e_n > floor_multiple * median(last 10% of e_n).

This ratio is an empirical combined contraction–perturbation trajectory, not
an exact policy-iteration contraction constant.  Direct PINN can be included
only in the separate common-diagnostic supplement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
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


DIAGNOSTIC_METRICS = ("diag_RelL2_V", "diag_RelL2_theta", "val_pres")


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
    text = str(cfg.get("eval_margin", "0.10"))
    vals = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not vals:
        vals = [0.0]
    if any(not 0.0 <= margin < 1.0 for margin in vals):
        raise ValueError(f"invalid eval_margin={text!r}")
    return vals[0]


def contraction_group_key(cfg: Mapping[str, Any]) -> str:
    """Training group plus the primary Q_ev window defining e_Xev."""
    training_group, _canon = group_key(dict(cfg))
    payload = json.dumps(
        {"training_group": training_group, "primary_eval_margin": primary_eval_margin(cfg)},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def read_outer_history(path: Path) -> Dict[str, Dict[int, float]]:
    series: Dict[str, Dict[int, float]] = {
        "e_Xev": {},
        **{metric: {} for metric in DIAGNOSTIC_METRICS},
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            outer = _int(row.get("outer_iter"))
            if outer is None:
                continue
            for metric in series:
                value = _float(row.get(metric))
                if math.isfinite(value):
                    # A rerun should have been archived.  If duplicate rows
                    # nevertheless exist, the last record is the final one.
                    series[metric][outer] = value
    return series


def discover_groups(
    out_root: Path,
    model_types: set[str],
    m_states: set[int],
    run_name_regex: str,
) -> Dict[str, Dict[str, Any]]:
    pattern = re.compile(run_name_regex) if run_name_regex else None
    newest: Dict[Tuple[str, int], Tuple[str, Path, Dict[str, Any], str]] = {}
    group_meta: Dict[str, Dict[str, Any]] = {}

    for run_dir_text in find_runs(str(out_root)):
        run_dir = Path(run_dir_text)
        cfg = load_config_args_raw(str(run_dir))
        model_type = str(cfg.get("model_type")) if cfg is not None else ""
        if cfg is None or model_type not in model_types:
            continue
        m = _int(cfg.get("m_states"))
        seed = _int(cfg.get("seed"))
        if m is None or seed is None or (m_states and m not in m_states):
            continue
        if pattern and not pattern.search(run_dir.name):
            continue
        group = contraction_group_key(cfg)
        updated = run_updated_at(str(run_dir))
        status = run_status(str(run_dir))
        key = (group, seed)
        if key not in newest or updated >= newest[key][0]:
            # Deduplicate before filtering on status.  Otherwise a newer
            # failed rerun in another directory can be silently masked by an
            # older successful copy of the same configuration/seed.
            newest[key] = (updated, run_dir, cfg, status)
        group_meta[group] = {
            "group": group,
            "model_type": model_type,
            "n_assets": _int(cfg.get("n_assets")),
            "m_states": m,
            "primary_eval_margin": primary_eval_margin(cfg),
        }

    groups: Dict[str, Dict[str, Any]] = {}
    for (group, seed), (_updated, run_dir, cfg, status) in newest.items():
        if status != "success":
            continue
        if not (run_dir / "outer_history.csv").is_file():
            raise ValueError(
                f"newest successful run is missing outer_history.csv: {run_dir}"
            )
        entry = groups.setdefault(
            group,
            {**group_meta[group], "runs": {}, "configs": {}, "market_hashes": {}, "market_errors": {}},
        )
        entry["runs"][seed] = run_dir
        entry["configs"][seed] = cfg
        try:
            entry["market_hashes"][seed] = canonical_market_hash(str(run_dir / "market_params.npz"))
            entry["market_errors"][seed] = ""
        except Exception as exc:
            entry["market_hashes"][seed] = ""
            entry["market_errors"][seed] = str(exc)
    return groups


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(np.std(arr, ddof=1))
    half = t_crit_95(int(arr.size) - 1) * std / math.sqrt(int(arr.size))
    return mean, std, mean - half, mean + half


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    groups: Dict[str, Dict[str, Any]],
    expected_seeds: set[int],
    floor_multipliers: Sequence[float],
    primary_margin: float,
    allow_incomplete: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    ratio_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    worst_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    groups_by_method_m: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for group, meta in groups.items():
        groups_by_method_m[(str(meta["model_type"]), int(meta["m_states"]))].append(group)
    ambiguous = {key: gs for key, gs in groups_by_method_m.items() if len(gs) != 1}
    if ambiguous:
        raise ValueError(
            "more than one training configuration / primary window found for a paper method/dimension; "
            f"narrow with --run-name-regex: {ambiguous}"
        )

    selected_seeds: Dict[str, List[int]] = {}
    market_by_m: Dict[int, List[Tuple[str, int, str, str]]] = defaultdict(list)
    for group, meta in groups.items():
        margin = float(meta["primary_eval_margin"])
        if not math.isclose(margin, float(primary_margin), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"group={group}, model={meta['model_type']}, M={meta['m_states']}: "
                f"primary eval margin={margin:g}, required={primary_margin:g}"
            )
        available = set(meta["runs"])
        if expected_seeds and available != expected_seeds and not allow_incomplete:
            raise ValueError(
                f"group={group}, M={meta['m_states']}: seeds={sorted(available)}, "
                f"expected={sorted(expected_seeds)}"
            )
        seeds = sorted(expected_seeds & available if expected_seeds else available)
        if len(seeds) < 2:
            raise ValueError(f"group={group}: at least two seeds are required")
        selected_seeds[group] = seeds
        for seed in seeds:
            market_by_m[int(meta["m_states"])].append((
                str(meta["model_type"]), seed,
                str(meta["market_hashes"].get(seed, "")),
                str(meta["market_errors"].get(seed, "")),
            ))

    for m_states, rows in sorted(market_by_m.items()):
        errors = [(model, seed, error) for model, seed, market_hash, error in rows
                  if error or not market_hash]
        hashes = {market_hash for _model, _seed, market_hash, _error in rows if market_hash}
        if errors:
            raise ValueError(f"M={m_states}: missing/invalid market snapshots: {errors}")
        if len(hashes) != 1:
            raise ValueError(
                f"M={m_states}: selected methods/seeds have {len(hashes)} canonical market hashes"
            )

    for group, meta in sorted(
        groups.items(),
        key=lambda item: (str(item[1]["model_type"]), int(item[1]["m_states"])),
    ):
        seeds = selected_seeds[group]
        model_type = str(meta["model_type"])

        histories: Dict[int, Dict[str, Dict[int, float]]] = {}
        floors: Dict[int, float] = {}
        ratios: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
        group_expected_outers: List[int] | None = None
        for seed in seeds:
            cfg = meta["configs"][seed]
            diag_every = _int(cfg.get("diag_every"))
            outer_iters = _int(cfg.get("outer_iters"))
            if diag_every != 1:
                raise ValueError(
                    f"{meta['runs'][seed]}: diag_every={diag_every}; complete per-outer "
                    "diagnostic histories require diag_every=1"
                )
            if outer_iters is None or outer_iters < 1:
                raise ValueError(f"{meta['runs'][seed]}: invalid outer_iters={outer_iters}")
            if model_type == "pipinn" and outer_iters < 2:
                raise ValueError(
                    f"{meta['runs'][seed]}: Figure 2 requires outer_iters>=2, got {outer_iters}"
                )
            history = read_outer_history(meta["runs"][seed] / "outer_history.csv")
            expected_outers = list(range(1, outer_iters + 1))
            if group_expected_outers is None:
                group_expected_outers = expected_outers
            elif expected_outers != group_expected_outers:
                raise ValueError(
                    f"group={group}: inconsistent outer_iters across seeds; "
                    f"seed={seed} has {outer_iters}"
                )
            for metric in DIAGNOSTIC_METRICS:
                metric_outers = sorted(history[metric])
                if metric_outers != expected_outers:
                    raise ValueError(
                        f"{meta['runs'][seed]}: supplemental metric={metric} has finite "
                        f"indices={metric_outers[:3]}..."
                        f"{metric_outers[-3:] if metric_outers else []} "
                        f"(n={len(metric_outers)}), expected exactly 1..{outer_iters}"
                    )
            histories[seed] = history

            # Direct PINN participates only in the supplemental diagnostics.
            if model_type != "pipinn":
                continue

            e_series = history["e_Xev"]
            outers = sorted(e_series)
            if outers != expected_outers:
                raise ValueError(
                    f"{meta['runs'][seed]}: e_Xev indices={outers[:3]}..."
                    f"{outers[-3:] if outers else []} (n={len(outers)}), "
                    f"expected exactly 1..{outer_iters}"
                )
            if any(e_series[outer] < 0.0 for outer in outers):
                raise ValueError(f"{meta['runs'][seed]}: e_Xev must be nonnegative")
            zero_denominators = [outer for outer in outers[:-1] if e_series[outer] <= 0.0]
            if zero_denominators:
                raise ValueError(
                    f"{meta['runs'][seed]}: e_Xev is zero at ratio denominator "
                    f"indices={zero_denominators}; rho=e_(n+1)/e_n is undefined"
                )
            tail_len = max(1, int(math.ceil(0.1 * len(outers))))
            floor = float(np.median([e_series[n] for n in outers[-tail_len:]]))
            if not math.isfinite(floor) or floor < 0.0:
                raise ValueError(f"{meta['runs'][seed]}: invalid empirical floor {floor}")
            floors[seed] = floor
            ratios[seed] = {
                n: (e_series[n], e_series[n + 1], e_series[n + 1] / e_series[n])
                for n in outers[:-1]
            }

        if model_type == "pipinn":
            for multiple in floor_multipliers:
                regular_by_seed = {
                    seed: {
                        n for n, (e_n, _e_np1, _rho) in ratios[seed].items()
                        if e_n > float(multiple) * floors[seed]
                    }
                    for seed in seeds
                }
                empty_seeds = [seed for seed in seeds if not regular_by_seed[seed]]
                if empty_seeds:
                    raise ValueError(
                        f"group={group}, M={meta['m_states']}, floor_multiple={multiple:g}: "
                        f"empty regular region for seeds={empty_seeds}; "
                        "refusing a partial-seed summary"
                    )
                common = set.intersection(*(regular_by_seed[seed] for seed in seeds))
                if not common:
                    raise ValueError(
                        f"group={group}, M={meta['m_states']}, floor_multiple={multiple:g}: "
                        "common regular support is empty"
                    )

                for seed in seeds:
                    for outer, (e_n, e_np1, rho) in sorted(ratios[seed].items()):
                        ratio_rows.append({
                            "group": group,
                            "model_type": model_type,
                            "M": meta["m_states"],
                            "seed": seed,
                            "outer_iter": outer,
                            "e_n": e_n,
                            "e_np1": e_np1,
                            "rho": rho,
                            "floor": floors[seed],
                            "floor_multiple": multiple,
                            "regular": int(outer in regular_by_seed[seed]),
                            "common_regular": int(outer in common),
                        })

                for outer in sorted(common):
                    vals = [ratios[seed][outer][2] for seed in seeds]
                    mean, std, ci_lo, ci_hi = mean_std_ci(vals)
                    summary_rows.append({
                        "group": group,
                        "model_type": model_type,
                        "M": meta["m_states"],
                        "floor_multiple": multiple,
                        "outer_iter": outer,
                        "n_seeds": len(vals),
                        "rho_mean": mean,
                        "rho_std": std,
                        "rho_ci_low": ci_lo,
                        "rho_ci_high": ci_hi,
                    })

                seed_maxima = []
                for seed in seeds:
                    regular_ratios = [ratios[seed][n][2] for n in regular_by_seed[seed]]
                    seed_maxima.append(max(regular_ratios))
                common_means = [
                    float(np.mean([ratios[seed][n][2] for seed in seeds])) for n in common
                ]
                worst_rows.append({
                    "group": group,
                    "model_type": model_type,
                    "M": meta["m_states"],
                    "floor_multiple": multiple,
                    "n_common_iterations": len(common),
                    "max_of_seed_mean_rho": max(common_means) if common_means else float("nan"),
                    "n_seed_maxima": len(seed_maxima),
                    "mean_of_seed_max_rho": float(np.mean(seed_maxima)),
                    "std_of_seed_max_rho": float(np.std(seed_maxima, ddof=1)),
                })

        # Supplement convergence trajectories: aggregate the same diagnostic
        # at the same outer index, never mix PINN training loss with PI eval loss.
        if group_expected_outers is None:
            raise AssertionError(f"group={group}: no selected histories")
        for metric in DIAGNOSTIC_METRICS:
            for outer in group_expected_outers:
                vals = [histories[seed][metric][outer] for seed in seeds]
                mean, std, ci_lo, ci_hi = mean_std_ci(vals)
                diagnostic_rows.append({
                    "group": group,
                    "model_type": model_type,
                    "M": meta["m_states"],
                    "metric": metric,
                    "outer_iter": outer,
                    "n_seeds": len(vals),
                    "mean": mean,
                    "std": std,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                })

    return ratio_rows, summary_rows, worst_rows, diagnostic_rows


def make_plots(
    output: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    main_floor_multiple: float,
    fmt: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    non_pi_rows = [row for row in summary_rows if str(row["model_type"]) != "pipinn"]
    if non_pi_rows:
        raise ValueError("Figure 2 received non-PI-PINN summary rows")
    m_values = sorted({int(row["M"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in m_values:
        rows = sorted(
            [row for row in summary_rows
             if int(row["M"]) == m
             and math.isclose(float(row["floor_multiple"]), main_floor_multiple)],
            key=lambda row: int(row["outer_iter"]),
        )
        if not rows:
            continue
        x = np.asarray([row["outer_iter"] for row in rows], dtype=float)
        mean = np.asarray([row["rho_mean"] for row in rows], dtype=float)
        std = np.asarray([row["rho_std"] for row in rows], dtype=float)
        ax.plot(x, mean, linewidth=1.8, label=f"M={m}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\widehat{\rho}_n$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output / f"figure2_contraction.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    for metric in DIAGNOSTIC_METRICS:
        method_m_diag = sorted({
            (str(row["model_type"]), int(row["M"]))
            for row in diagnostic_rows if row["metric"] == metric
        })
        show_method_diag = len({model for model, _m in method_m_diag}) > 1
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for model_type, m in method_m_diag:
            rows = sorted(
                [row for row in diagnostic_rows
                 if int(row["M"]) == m and str(row["model_type"]) == model_type
                 and row["metric"] == metric],
                key=lambda row: int(row["outer_iter"]),
            )
            if not rows:
                continue
            x = np.asarray([row["outer_iter"] for row in rows], dtype=float)
            mean = np.asarray([row["mean"] for row in rows], dtype=float)
            std = np.asarray([row["std"] for row in rows], dtype=float)
            method = "PINN" if model_type == "pinn" else "PI-PINN"
            label = f"{method}, M={m}" if show_method_diag else f"M={m}"
            ax.plot(x, mean, linewidth=1.8, label=label)
            ax.fill_between(x, np.maximum(mean - std, np.finfo(float).tiny), mean + std, alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.savefig(
            output / f"supplemental_diagnostic_{metric}.{fmt}",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create PI-PINN-only Figure-2 contraction outputs and diagnostic supplements."
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--output", default="", help="Default: <out-root>/paper_postprocess")
    parser.add_argument(
        "--diagnostic-models", choices=("pipinn", "both"), default="pipinn",
        help="Models in supplemental common-diagnostic outputs. Figure 2 is always PI-PINN only.",
    )
    parser.add_argument("--m-states", default="1,3,5")
    parser.add_argument("--expected-seeds", default="1-10")
    parser.add_argument("--primary-margin", type=float, default=0.10)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--floor-multipliers", default="5,10,20")
    parser.add_argument("--main-floor-multiple", type=float, default=10.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--format", choices=("png", "pdf", "svg", "eps"), default="png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else out_root / "paper_postprocess"
    output.mkdir(parents=True, exist_ok=True)
    expected = set(parse_seed_spec(args.expected_seeds))
    m_states = set(parse_seed_spec(args.m_states))
    multipliers = [float(x) for x in re.split(r"[\s,]+", args.floor_multipliers) if x]
    if not 0.0 <= args.primary_margin < 1.0:
        raise ValueError("--primary-margin must be in [0,1)")
    if (not multipliers or any(not math.isfinite(value) or value <= 0.0 for value in multipliers)
            or len(set(multipliers)) != len(multipliers)):
        raise ValueError("--floor-multipliers must be unique positive finite values")
    if not any(math.isclose(args.main_floor_multiple, value, rel_tol=0.0, abs_tol=1e-12)
               for value in multipliers):
        raise ValueError("--main-floor-multiple must be included in --floor-multipliers")

    model_types = {"pipinn"}
    if args.diagnostic_models == "both":
        model_types.add("pinn")
    groups = discover_groups(out_root, model_types, m_states, args.run_name_regex)
    if not groups:
        raise SystemExit("no eligible successful runs with outer_history.csv were found")
    if m_states and not args.allow_incomplete:
        for model_type in sorted(model_types):
            found_m = {
                int(meta["m_states"]) for meta in groups.values()
                if str(meta["model_type"]) == model_type
            }
            if found_m != m_states:
                raise ValueError(
                    f"model={model_type}: found M={sorted(found_m)}, expected M={sorted(m_states)}"
                )

    ratio_rows, summary_rows, worst_rows, diagnostic_rows = build_outputs(
        groups, expected, multipliers, args.primary_margin, args.allow_incomplete
    )
    figure2_rows = ratio_rows + summary_rows + worst_rows
    if any(str(row.get("model_type")) != "pipinn" for row in figure2_rows):
        raise AssertionError("internal error: Figure-2 outputs must contain only PI-PINN rows")
    write_csv(output / "figure2_ratios.csv", ratio_rows, [
        "group", "model_type", "M", "seed", "outer_iter", "e_n", "e_np1", "rho",
        "floor", "floor_multiple", "regular", "common_regular",
    ])
    write_csv(output / "figure2_summary.csv", summary_rows, [
        "group", "model_type", "M", "floor_multiple", "outer_iter", "n_seeds",
        "rho_mean", "rho_std", "rho_ci_low", "rho_ci_high",
    ])
    write_csv(output / "figure2_worst_summary.csv", worst_rows, [
        "group", "model_type", "M", "floor_multiple", "n_common_iterations",
        "max_of_seed_mean_rho", "n_seed_maxima", "mean_of_seed_max_rho",
        "std_of_seed_max_rho",
    ])
    write_csv(output / "supplemental_diagnostic_summary.csv", diagnostic_rows, [
        "group", "model_type", "M", "metric", "outer_iter", "n_seeds",
        "mean", "std", "ci95_low", "ci95_high",
    ])
    with (output / "postprocess_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    if not args.no_plots:
        make_plots(output, summary_rows, diagnostic_rows,
                   args.main_floor_multiple, args.format, args.dpi)
    print(f"[done] contraction outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
