#!/usr/bin/env python3
"""Aggregate Merton PI-PINN control trajectories for paper Figure 1.

Each row of ``outer_history.csv`` stores mean-squared control discrepancies
on the run's fixed Q_ev diagnostic set.  At recorded outer iteration ``k``,

* ``c_diff`` and ``pi_diff`` compare the improved policy alpha_k with the
  frozen policy alpha_{k-1} used by that iteration's linear PDE; and
* ``c_vs_closed_form`` and ``pi_vs_closed_form`` compare alpha_k with the
  closed-form Merton policy.

The two separate paper panels show pointwise arithmetic means across training
seeds with plus/minus one sample-standard-deviation bands.  They never treat
outer iterations as independent replicates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from aggregate_seeds import parse_seed_spec
from postprocess_contraction import (
    convergence_group_key,
    discover_groups,
    mean_std_ci,
    parse_formats,
    primary_eval_margin,
    select_group,
)


C_DIFF = "c_diff"
PI_DIFF = "pi_diff"
C_CLOSED_FORM = "c_vs_closed_form"
PI_CLOSED_FORM = "pi_vs_closed_form"
METRICS = (C_DIFF, PI_DIFF, C_CLOSED_FORM, PI_CLOSED_FORM)
PANELS = {
    "successive": (C_DIFF, PI_DIFF),
    "closed_form": (C_CLOSED_FORM, PI_CLOSED_FORM),
}
METRIC_LABELS = {
    C_DIFF: r"$\|c_{n+1}-c_n\|_2^2$",
    PI_DIFF: r"$\|\pi_{n+1}-\pi_n\|_2^2$",
    C_CLOSED_FORM: r"$\|c_n-c^*\|_2^2$",
    PI_CLOSED_FORM: r"$\|\pi_n-\pi^*\|_2^2$",
}
METRIC_DEFINITIONS = {
    C_DIFF: (
        "trainer c_diff: elementwise mean squared discrepancy between the "
        "improved consumption at outer k and the frozen consumption at outer k-1"
    ),
    PI_DIFF: (
        "trainer pi_diff: elementwise mean squared discrepancy between the "
        "improved portfolio at outer k and the frozen portfolio at outer k-1"
    ),
    C_CLOSED_FORM: (
        "trainer c_vs_closed_form: elementwise mean squared discrepancy between "
        "the improved consumption at outer k and the closed-form consumption"
    ),
    PI_CLOSED_FORM: (
        "trainer pi_vs_closed_form: elementwise mean squared discrepancy between "
        "the improved portfolio at outer k and the closed-form portfolio"
    ),
}

OUTPUT_BASENAME = "figure1_pipinn_control_convergence"
PANEL_TAGS = {
    "successive": "diff",
    "closed_form": "cf",
}
PANEL_OUTPUT_BASENAMES = {
    panel: f"{OUTPUT_BASENAME}_{tag}" for panel, tag in PANEL_TAGS.items()
}
OWNED_OUTPUTS = {
    "figure1_control_trajectories.csv",
    "figure1_pointwise_summary.csv",
    "figure1_runs_used.csv",
    "figure1_metadata.json",
    # The old combined two-panel artifact is cleanup-only.  New runs write
    # one paper-ready file per panel.
    *(f"{OUTPUT_BASENAME}.{fmt}" for fmt in ("png", "pdf", "svg", "eps")),
    *(
        f"{basename}.{fmt}"
        for basename in PANEL_OUTPUT_BASENAMES.values()
        for fmt in ("png", "pdf", "svg", "eps")
    ),
}


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def prepare_output(output: Path, overwrite: bool) -> None:
    """Create output safely and replace only files owned by this script."""
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        return
    entries = list(output.iterdir())
    if entries and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --overwrite to replace "
            "only known Figure-1 artifacts"
        )
    blocked = [
        entry.name
        for entry in entries
        if entry.name in OWNED_OUTPUTS and not entry.is_file()
    ]
    if blocked:
        raise ValueError(
            "refusing --overwrite because reserved output paths are not regular files: "
            f"{blocked}"
        )
    for entry in entries:
        if entry.name in OWNED_OUTPUTS:
            entry.unlink()


def read_outer_history(path: Path) -> Tuple[Dict[str, Dict[int, float]], int]:
    """Read fixed-Q_ev control metrics and reject incomplete or ambiguous rows."""
    series: Dict[str, Dict[int, float]] = {metric: {} for metric in METRICS}
    seen_outer: set[int] = set()
    point_counts: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"empty outer history: {path}")
        required = [
            "outer_iter",
            *METRICS,
            "control_metric_scope",
            "control_metric_points",
        ]
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")

        for row_number, row in enumerate(reader, start=2):
            outer = _int(row.get("outer_iter"))
            if outer is None:
                raise ValueError(f"{path}: invalid outer_iter on CSV row {row_number}")
            if outer in seen_outer:
                raise ValueError(f"{path}: duplicate outer_iter={outer}")
            seen_outer.add(outer)

            scope = str(row.get("control_metric_scope", "")).strip()
            if scope != "fixed_qev":
                raise ValueError(
                    f"{path}: outer_iter={outer} has control_metric_scope={scope!r}; "
                    "paper Figure 1 requires fixed_qev"
                )
            points = _int(row.get("control_metric_points"))
            if points is None or points <= 0:
                raise ValueError(
                    f"{path}: invalid control_metric_points at outer_iter={outer}"
                )
            point_counts.add(points)

            for metric in METRICS:
                value = _float(row.get(metric))
                if not math.isfinite(value):
                    raise ValueError(
                        f"{path}: metric={metric} is nonfinite at outer_iter={outer}"
                    )
                if value < 0.0:
                    raise ValueError(
                        f"{path}: metric={metric} is negative at outer_iter={outer}; "
                        "mean-squared control discrepancies must be nonnegative"
                    )
                series[metric][outer] = value

    if len(point_counts) != 1:
        raise ValueError(
            f"{path}: control_metric_points changed across outer iterations: "
            f"{sorted(point_counts)}"
        )
    if not point_counts:
        raise ValueError(f"{path}: outer history contains no data rows")
    return series, next(iter(point_counts))


def validate_and_load(
    meta: Dict[str, Any],
    expected_seeds: set[int],
    min_seeds: int,
) -> Tuple[
    List[int],
    Dict[int, Dict[str, Dict[int, float]]],
    List[Dict[str, Any]],
]:
    """Validate one exact configuration and return its per-seed trajectories."""
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
    common_points: int | None = None
    for seed in seeds:
        run_dir: Path = meta["runs"][seed]
        cfg = meta["configs"][seed]
        if convergence_group_key(cfg) != meta["group"]:
            raise ValueError(f"{run_dir}: configuration changed during selection")
        if _int(cfg.get("m_states", 1)) != 1:
            raise ValueError(f"{run_dir}: Merton requires m_states=1")
        outer_iters = _int(cfg.get("outer_iters"))
        if outer_iters is None or outer_iters < 1:
            raise ValueError(f"{run_dir}: invalid outer_iters={outer_iters}")
        if common_outer_iters is None:
            common_outer_iters = outer_iters
        elif outer_iters != common_outer_iters:
            raise ValueError(
                f"group={meta['group']}: seed={seed} has outer_iters={outer_iters}, "
                f"expected {common_outer_iters}"
            )

        history, metric_points = read_outer_history(run_dir / "outer_history.csv")
        # ``diag_points`` is the requested budget.  The trainer constructs a
        # tensor grid, so ``control_metric_points`` is the realized cardinality
        # (for example, 8192 requested points become a 91 x 91 = 8281 grid).
        # The realized count, not equality with the requested budget, is the
        # fixed-Q_ev comparability contract.
        requested_diag_points = _int(cfg.get("diag_points"))
        expected_outers = list(range(1, outer_iters + 1))
        for metric in METRICS:
            actual = sorted(history[metric])
            if actual != expected_outers:
                missing = sorted(set(expected_outers) - set(actual))
                extra = sorted(set(actual) - set(expected_outers))
                raise ValueError(
                    f"{run_dir}: metric={metric} must cover outer 1..{outer_iters} "
                    f"exactly; missing={missing}, extra={extra}"
                )
        if common_points is None:
            common_points = metric_points
        elif metric_points != common_points:
            raise ValueError(
                f"group={meta['group']}: seed={seed} uses {metric_points} fixed-Q_ev "
                f"points, expected {common_points}"
            )

        histories[seed] = history
        run_rows.append(
            {
                "group": meta["group"],
                "model_type": "pipinn",
                "n_assets": meta["n_assets"],
                "m_states": 1,
                "seed": seed,
                "run_dir": str(run_dir),
                "outer_iters": outer_iters,
                "primary_eval_margin": primary_eval_margin(cfg),
                "control_metric_scope": "fixed_qev",
                "control_metric_points": metric_points,
                "diag_points_requested": (
                    requested_diag_points
                    if requested_diag_points is not None
                    else ""
                ),
                "diag_points_actual": metric_points,
                "pi_init_method": str(cfg.get("pi_init_method", "")),
                "pi_init_scale": cfg.get("pi_init_scale", ""),
                "policy_bounds_mode": str(cfg.get("policy_bounds_mode", "")),
                "market_hash": meta["market_hashes"][seed],
            }
        )
    return seeds, histories, run_rows


def build_tables(
    meta: Mapping[str, Any],
    seeds: Sequence[int],
    histories: Mapping[int, Mapping[str, Mapping[int, float]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    identity = {
        "group": meta["group"],
        "model_type": "pipinn",
        "n_assets": meta["n_assets"],
        "m_states": 1,
    }
    trajectory_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for panel, metrics in PANELS.items():
        for metric in metrics:
            outer_grid = sorted(histories[seeds[0]][metric])
            for seed in seeds:
                for outer in outer_grid:
                    trajectory_rows.append(
                        {
                            **identity,
                            "panel": panel,
                            "seed": seed,
                            "outer_iter": outer,
                            "metric": metric,
                            "metric_label": METRIC_LABELS[metric],
                            "value": histories[seed][metric][outer],
                        }
                    )
            for outer in outer_grid:
                values = [histories[seed][metric][outer] for seed in seeds]
                mean, std, sem, ci_low, ci_high = mean_std_ci(values)
                summary_rows.append(
                    {
                        **identity,
                        "panel": panel,
                        "outer_iter": outer,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "n_seeds": len(seeds),
                        "mean": mean,
                        "sample_sd": std,
                        "sem": sem,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "sd_band_lower_raw": mean - std,
                        "sd_band_upper": mean + std,
                    }
                )
    return trajectory_rows, summary_rows


def figure_log_floor(summary_rows: Sequence[Mapping[str, Any]]) -> float:
    """Choose a display-only floor without changing any exported raw value."""
    positive_plot_values = [
        float(row[key])
        for row in summary_rows
        for key in ("mean", "sd_band_upper")
        if math.isfinite(float(row[key])) and float(row[key]) > 0.0
    ]
    if not positive_plot_values:
        return 1.0e-16
    return max(np.finfo(float).tiny, min(positive_plot_values) * 1.0e-3)


def create_panel_figure(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    figure_size: Tuple[float, float] = (6.0, 4.0),
    font_size: float = 22.0,
    font_family: str = "",
    eps_compatible: bool = False,
    line_width: float = 1.5,
    line_alpha: float = 1.0,
):
    """Create one legacy-style paper panel with a seed mean +/- sample-SD band."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.ticker import MaxNLocator

    if panel not in PANELS:
        raise ValueError(f"unknown Figure-1 panel {panel!r}; choose from {list(PANELS)}")

    colors = {
        C_DIFF: "b",
        C_CLOSED_FORM: "b",
        PI_DIFF: "r",
        PI_CLOSED_FORM: "r",
    }
    plot_floor = figure_log_floor(summary_rows)

    def lighten(color: str, white_fraction: float) -> Tuple[float, float, float]:
        rgb = np.asarray(to_rgb(color), dtype=float)
        return tuple((1.0 - white_fraction) * rgb + white_fraction)

    rc_params: Dict[str, Any] = {"font.size": font_size}
    if font_family:
        rc_params["font.family"] = font_family
    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=figure_size)
        metrics = PANELS[panel]
        for metric in metrics:
            rows = sorted(
                [row for row in summary_rows if row["metric"] == metric],
                key=lambda row: int(row["outer_iter"]),
            )
            x = np.asarray([row["outer_iter"] for row in rows], dtype=float)
            mean = np.asarray([row["mean"] for row in rows], dtype=float)
            std = np.asarray([row["sample_sd"] for row in rows], dtype=float)
            raw_lower = mean - std
            mean_plot = np.maximum(mean, plot_floor)
            lower = np.where(raw_lower > 0.0, raw_lower, plot_floor)
            upper = np.maximum(mean + std, plot_floor)
            color = colors[metric]
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
                mean_plot,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                label=METRIC_LABELS[metric],
            )

        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=5))
        ax.set_xlabel("Iteration", fontsize=font_size)
        # Paper styling intentionally has neither a panel title nor a y label.
        ax.set_title("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=font_size)
        if font_family:
            for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                tick_label.set_fontfamily(font_family)
        grid_kwargs: Dict[str, Any] = {
            "alpha": 1.0 if eps_compatible else 0.3,
        }
        if eps_compatible:
            # Black at alpha=.3 over white, preblended because EPS has no alpha.
            grid_kwargs["color"] = "#B2B2B2"
        ax.grid(True, **grid_kwargs)
        legend_font: Dict[str, Any] = {"size": font_size}
        if font_family:
            legend_font["family"] = font_family
        ax.legend(frameon=False, prop=legend_font, loc="best")

        fig.tight_layout()
    return fig


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create two separate legacy-style Merton PI-PINN Figure-1 panels "
            "from fixed-Q_ev outer-history control diagnostics."
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Default: <out-root>/figure1_pipinn_control_convergence",
    )
    parser.add_argument(
        "--n-assets",
        "--expected-n-assets",
        dest="n_assets",
        type=int,
        default=None,
        help="Select one risky-asset dimension.",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=None,
        help="Select one training budget when the sweep root contains several.",
    )
    parser.add_argument(
        "--expected-seeds",
        default="",
        help="Exact seed set, e.g. '1,2,3,5,7,11,17,23,42,101'.",
    )
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--primary-margin", type=float, default=0.10)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--group-id", default="")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.0,
        help="Width in inches of each separate panel (default: 6.0).",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=4.0,
        help="Height in inches of each separate panel (default: 4.0).",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=22.0,
        help="Global paper font size in points (default: 22).",
    )
    parser.add_argument("--font-family", default="")
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.5,
        help="Seed-mean line width (default: 1.5).",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=1.0,
        help="Seed-mean line opacity in (0,1] (default: 1.0).",
    )
    parser.add_argument(
        "--bbox-inches", choices=("tight", "standard"), default="tight"
    )
    args = parser.parse_args(argv)

    if args.n_assets is not None and args.n_assets < 1:
        raise ValueError("--n-assets must be positive")
    if args.outer_iters is not None and args.outer_iters < 1:
        raise ValueError("--outer-iters must be positive")
    if args.min_seeds < 2:
        raise ValueError("--min-seeds must be at least 2 for a sample-SD band")
    if not 0.0 <= args.primary_margin < 0.5:
        raise ValueError("--primary-margin must be in [0,0.5)")
    for name, value in (
        ("--dpi", args.dpi),
        ("--fig-width", args.fig_width),
        ("--fig-height", args.fig_height),
        ("--font-size", args.font_size),
        ("--line-width", args.line_width),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if (
        not math.isfinite(float(args.line_alpha))
        or not 0.0 < float(args.line_alpha) <= 1.0
    ):
        raise ValueError("--line-alpha must be finite and lie in (0,1]")

    formats = parse_formats(args.formats)
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
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
    trajectory_rows, summary_rows = build_tables(meta, seeds, histories)
    masked_sd_points = {
        metric: sum(
            1
            for row in summary_rows
            if row["metric"] == metric
            and float(row["sd_band_lower_raw"]) <= 0.0
        )
        for metric in METRICS
    }

    prepare_output(output, args.overwrite)
    identity_fields = ["group", "model_type", "n_assets", "m_states"]
    write_csv(
        output / "figure1_control_trajectories.csv",
        trajectory_rows,
        [
            *identity_fields,
            "panel",
            "seed",
            "outer_iter",
            "metric",
            "metric_label",
            "value",
        ],
    )
    write_csv(
        output / "figure1_pointwise_summary.csv",
        summary_rows,
        [
            *identity_fields,
            "panel",
            "outer_iter",
            "metric",
            "metric_label",
            "n_seeds",
            "mean",
            "sample_sd",
            "sem",
            "ci95_low",
            "ci95_high",
            "sd_band_lower_raw",
            "sd_band_upper",
        ],
    )
    write_csv(
        output / "figure1_runs_used.csv",
        run_rows,
        [
            *identity_fields,
            "seed",
            "run_dir",
            "outer_iters",
            "primary_eval_margin",
            "control_metric_scope",
            "control_metric_points",
            "diag_points_requested",
            "diag_points_actual",
            "pi_init_method",
            "pi_init_scale",
            "policy_bounds_mode",
            "market_hash",
        ],
    )

    figure_files: List[str] = []
    if not args.no_plots:
        for fmt in formats:
            for panel in PANELS:
                figure = create_panel_figure(
                    summary_rows,
                    panel=panel,
                    figure_size=(args.fig_width, args.fig_height),
                    font_size=args.font_size,
                    font_family=args.font_family,
                    eps_compatible=(fmt == "eps"),
                    line_width=args.line_width,
                    line_alpha=args.line_alpha,
                )
                path = output / f"{PANEL_OUTPUT_BASENAMES[panel]}.{fmt}"
                figure.savefig(
                    path,
                    dpi=args.dpi,
                    bbox_inches=None if args.bbox_inches == "standard" else "tight",
                )
                import matplotlib.pyplot as plt

                plt.close(figure)
                figure_files.append(path.name)

    metadata = {
        "artifact": "Merton PI-PINN paper Figure 1",
        "group": meta["group"],
        "model_type": "pipinn",
        "n_assets": meta["n_assets"],
        "m_states": 1,
        "outer_iters": meta["outer_iters"],
        "seeds": seeds,
        "n_seeds": len(seeds),
        "aggregation": (
            "pointwise arithmetic seed mean with plus/minus one sample-standard-"
            "deviation band; outer iterations are not replicates"
        ),
        "source": "outer_history.csv",
        "source_metrics": list(METRICS),
        "metric_definitions": METRIC_DEFINITIONS,
        "indexing_contract": (
            "At outer_iter=k, *_diff compares alpha_k against frozen alpha_(k-1); "
            "*_vs_closed_form compares alpha_k against alpha_star."
        ),
        "normalization": (
            "The trainer columns are elementwise means of squared discrepancies "
            "over fixed Q_ev and, for pi, over portfolio components."
        ),
        "control_metric_scope": "fixed_qev",
        "diag_points_requested": run_rows[0]["diag_points_requested"],
        "diag_points_actual": run_rows[0]["diag_points_actual"],
        # Retained as a compatibility alias for older consumers.
        "control_metric_points": run_rows[0]["control_metric_points"],
        "primary_eval_margin": meta["primary_eval_margin"],
        "market_hash": run_rows[0]["market_hash"],
        "sample_sd_ddof": 1,
        "log_y_axis": True,
        "log_plot_floor": figure_log_floor(summary_rows),
        "zero_mean_points_plot_floored": {
            metric: sum(
                1
                for row in summary_rows
                if row["metric"] == metric and float(row["mean"]) == 0.0
            )
            for metric in METRICS
        },
        "sd_band_nonpositive_lower_points_plot_floored": masked_sd_points,
        "raw_zero_values_preserved": True,
        "figure_layout": "two_separate_panels",
        "panel_tags": PANEL_TAGS,
        "panel_output_basenames": PANEL_OUTPUT_BASENAMES,
        "legacy_combined_output_cleanup_only": OUTPUT_BASENAME,
        "plot_style": {
            "figure_size_inches_each": [args.fig_width, args.fig_height],
            "font_size_pt": args.font_size,
            "font_family": args.font_family,
            "consumption_color": "blue",
            "portfolio_color": "red",
            "line_width": args.line_width,
            "line_alpha": args.line_alpha,
            "x_label": "Iteration",
            "title": None,
            "y_label": None,
            "grid_alpha": 0.3,
            "legend_frame": False,
        },
        "figure_files": figure_files,
        "formats_requested": formats,
        "dpi": args.dpi,
        "no_plots": bool(args.no_plots),
    }
    (output / "figure1_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[figure1] wrote: {output}")
    print(
        f"[figure1] group={meta['group']} | N={meta['n_assets']} | "
        f"seeds={seeds} | outer_iters={meta['outer_iters']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
