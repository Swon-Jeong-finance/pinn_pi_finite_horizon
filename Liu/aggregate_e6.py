"""E6 aggregation: residual-tolerance sweep -> error-floor scaling.

Collects, per run: the pres target, the ACHIEVED residual level
p_res = max_n p_res,n (status.json `pres_max`, with an outer_history.csv
fallback), target-reached status, total inner optimizer steps, and the final
error metrics from metrics.csv. Then, per configuration group (everything
shared except training seed and pres_target):

  per_target.csv   n_runs, n_target_reached, achieved p_res/error/steps
                   mean, sample SD, SEM, and Student-t 95% CI per target
  points.csv       one row per run (x = achieved p_res, y = error)
  fit.csv          log10(error) ~ log10(achieved p_res), including the
                   primary seed-cluster-robust slope interval

Per the protocol: runs that did NOT reach their target enter the fit with
their ACHIEVED residual (never the nominal target); reached runs also use
the achieved value, which is <= target by construction, so the x-axis is
uniformly "achieved p_res".

Standalone: stdlib + numpy (+ scipy if available for exact t quantiles).

Usage:
    python3 aggregate_e6.py --out-root <OUT_ROOT>
    python3 aggregate_e6.py --out-root <OUT_ROOT> --metrics RelL2_V,RelL2_theta
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aggregate_seeds import (  # reuse shared helpers
    GROUP_IGNORE_KEYS, canonical_market_hash, find_runs, load_config_args,
    load_config_args_raw, parse_seed_spec, run_status, run_updated_at,
    t_crit_95, fmt,
)

# E6 groups additionally collapse over the tolerance itself.
E6_IGNORE_KEYS = set(GROUP_IGNORE_KEYS) | {"pres_target"}


def e6_group_key(args: Dict[str, Any]) -> str:
    import hashlib
    core = {k: args[k] for k in sorted(args) if k not in E6_IGNORE_KEYS}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def read_status(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "status.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def achieved_pres_from_outer_history(run_dir: str) -> Optional[float]:
    """Fallback: max of the per-outer reported val_pres column."""
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None
    best = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("val_pres", "")).strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            best = v if best is None else max(best, v)
    return best


def total_steps_from_outer_history(run_dir: str) -> Optional[float]:
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None
    tot = 0.0
    seen = False
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = str(row.get("inner_epochs_used", "")).strip()
            if not raw:
                continue
            try:
                tot += float(raw)
                seen = True
            except ValueError:
                continue
    return tot if seen else None


def pick_metric_value(run_dir: str, metric: str) -> Optional[Tuple[float, float]]:
    """Return (value, eval_margin_used) for the requested metric.

    FULL-DIMENSIONAL rows only, at the run's PRIMARY margin (first listed in
    config; smallest recorded margin as a fallback). Legacy slice rows are
    ignored.
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("metric", "")) != metric:
                continue
            if str(row.get("scope", "") or "slice") != "fulldim":
                continue
            try:
                val = float(row["value"])
            except (KeyError, ValueError):
                continue
            mraw = str(row.get("eval_margin", "")).strip()
            margin = float(mraw) if mraw else 0.0
            rows.append((margin, val))
    if not rows:
        return None

    cfg = load_config_args(run_dir) or {}
    primary = None
    try:
        margins = [float(x) for x in str(cfg.get("eval_margin", "")).split(",") if str(x).strip()]
        primary = margins[0] if margins else None
    except ValueError:
        primary = None

    if primary is not None:
        exact = [r for r in rows if abs(r[0] - primary) < 1e-12]
        if exact:
            m, v = exact[-1]
            return (v, m)
    m, v = sorted(rows, key=lambda z: z[0])[0]
    return (v, m)


def mean_std(vals: List[float]) -> Tuple[float, float, int]:
    a = np.asarray(vals, dtype=float)
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if n > 1 else float("nan"), n


def parse_float_spec(text: str) -> List[float]:
    """Parse a comma/space separated exact tolerance set."""

    values: List[float] = []
    for token in re.split(r"[\s,]+", str(text or "").strip()):
        if not token:
            continue
        value = float(token)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"tolerances must be positive finite values, got {token!r}")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate tolerances in {text!r}")
    return sorted(values)


def _float_key(value: float) -> str:
    return f"{float(value):.17g}"


def cluster_robust_slope_se(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    """Seed-cluster-robust (CR1) standard error for the log-log OLS slope.

    Sandwich estimator with the small-G correction G/(G-1) * (n-1)/(n-k);
    CI uses t with G-1 degrees of freedom.
    """
    lx, ly = np.log10(x), np.log10(y)
    n = int(lx.size)
    labels = np.unique(clusters)
    G = int(labels.size)
    out = {"G": G, "se": float("nan"), "ci_lo": float("nan"),
           "ci_hi": float("nan"), "t_crit": float("nan")}
    if n < 2 or G < 2:
        return out
    X = np.vstack([lx, np.ones(n)]).T
    # Degenerate design (all achieved residuals identical -- e.g. every run
    # hit the same floor): the slope is NOT identified. Report NaN instead of
    # crashing or faking a number via pseudoinverse.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return out
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ ly)
    u = ly - X @ beta
    meat = np.zeros((2, 2))
    for gname in labels:
        m_ = clusters == gname
        s_g = X[m_].T @ u[m_]
        meat += np.outer(s_g, s_g)
    k = 2
    corr = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    V = corr * XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[0, 0], 0.0)))
    tc = t_crit_95(G - 1)
    slope = float(beta[0])
    out.update({"se": se, "ci_lo": slope - tc * se, "ci_hi": slope + tc * se, "t_crit": tc})
    return out


def ols_loglog(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS of log10(y) on log10(x): slope, intercept, stderr(slope), 95% CI, R^2."""
    lx, ly = np.log10(x), np.log10(y)
    n = int(lx.size)
    out: Dict[str, float] = {"n": n}
    if n < 2:
        for k in ["slope", "intercept", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "r2"]:
            out[k] = float("nan")
        return out
    X = np.vstack([lx, np.ones(n)]).T
    if np.linalg.matrix_rank(X) < X.shape[1]:
        # Constant x: slope unidentified.
        for k in ["slope", "intercept", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "r2"]:
            out[k] = float("nan")
        return out
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    resid = ly - X @ beta
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(float(np.sum((ly - ly.mean()) ** 2)), 1e-300)
    if n > 2:
        s2 = float(np.sum(resid ** 2)) / (n - 2)
        sxx = float(np.sum((lx - lx.mean()) ** 2))
        se = math.sqrt(s2 / max(sxx, 1e-300))
        tc = t_crit_95(n - 2)
        out.update({"slope_se": se, "ci95_lo": slope - tc * se,
                    "ci95_hi": slope + tc * se, "t_crit": tc})
    else:
        out.update({"slope_se": float("nan"), "ci95_lo": float("nan"),
                    "ci95_hi": float("nan"), "t_crit": float("nan")})
    out.update({"slope": slope, "intercept": intercept, "r2": r2})
    return out


def _summary_stats(values: List[float]) -> Dict[str, float]:
    mean, std, n = mean_std(values)
    if n > 1 and math.isfinite(std):
        sem = std / math.sqrt(n)
        half = t_crit_95(n - 1) * sem
        lo, hi = mean - half, mean + half
    else:
        sem = lo = hi = float("nan")
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "ci95_lo": lo, "ci95_hi": hi,
    }


def _parse_formats(text: str) -> List[str]:
    formats = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    allowed = {"png", "pdf", "svg", "eps"}
    if not formats or len(formats) != len(set(formats)) or any(x not in allowed for x in formats):
        raise ValueError(
            f"--formats must be a unique comma-separated subset of {sorted(allowed)}"
        )
    return formats


def _owned_e6_paths(output: Path) -> List[Path]:
    fixed = [
        output / "points.csv", output / "per_target.csv", output / "fit.csv",
        output / "e6_metadata.json", output / "_SUCCESS_E6",
    ]
    return fixed + sorted(output.glob("e6_error_floor_*.*")) if output.is_dir() else fixed


def _prepare_e6_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output exists and is not a directory: {output}")
    existing = [path for path in _owned_e6_paths(output) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "E6 output already contains managed artifacts; pass --overwrite to replace them: "
            + ", ".join(path.name for path in existing)
        )


def _plot_e6_cells(
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    output: Path,
    args: argparse.Namespace,
    formats: List[str],
) -> List[str]:
    if args.skip_plot:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required unless --skip-plot is used") from exc

    plt.rcParams.update({"font.size": args.font_size})
    written: List[str] = []
    for (group, metric, margin), rows in sorted(points.items()):
        by_target: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_target[float(row["pres_target"])].append(row)
        x_mean: List[float] = []
        y_mean: List[float] = []
        y_lo: List[float] = []
        y_hi: List[float] = []
        for target in sorted(by_target):
            cell = by_target[target]
            xs = np.asarray([float(row["achieved_pres"]) for row in cell], dtype=float)
            ys = np.asarray([float(row["error"]) for row in cell], dtype=float)
            xm = float(xs.mean())
            ym = float(ys.mean())
            sd = float(ys.std(ddof=1)) if ys.size > 1 else float("nan")
            x_mean.append(xm)
            y_mean.append(ym)
            if math.isfinite(sd):
                y_lo.append(max(ym - sd, np.finfo(float).tiny))
                y_hi.append(ym + sd)
            else:
                y_lo.append(ym)
                y_hi.append(ym)

        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        for row in rows:
            ax.scatter(
                float(row["achieved_pres"]), float(row["error"]),
                s=14, color="0.65", alpha=0.55, linewidths=0, zorder=1,
            )
        order = np.argsort(np.asarray(x_mean, dtype=float))
        xx = np.asarray(x_mean, dtype=float)[order]
        yy = np.asarray(y_mean, dtype=float)[order]
        lo = np.asarray(y_lo, dtype=float)[order]
        hi = np.asarray(y_hi, dtype=float)[order]
        ax.plot(xx, yy, color="#1f77b4", marker="o", linewidth=2.0, zorder=3)
        ax.fill_between(xx, lo, hi, color="#1f77b4", alpha=0.20, linewidth=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Achieved $p_{res}$")
        ax.set_ylabel(metric)
        ax.grid(True, which="both", alpha=0.22, linewidth=0.6)
        fig.tight_layout()
        margin_token = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{margin:g}")
        metric_token = re.sub(r"[^0-9A-Za-z_.-]+", "_", metric)
        stem = f"e6_error_floor_{group}_{metric_token}_margin_{margin_token}"
        for extension in formats:
            path = output / f"{stem}.{extension}"
            save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
            if extension == "png":
                save_kwargs["dpi"] = args.dpi
            fig.savefig(path, format=extension, **save_kwargs)
            written.append(path.name)
        plt.close(fig)
    return written


def _write_e6_outputs(
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    stage: Path,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    formats: List[str],
) -> None:
    stage.mkdir(parents=True, exist_ok=True)

    pts_path = stage / "points.csv"
    pts_fields = [
        "group", "metric", "run_dir", "seed", "pres_target", "achieved_pres",
        "target_reached", "total_inner_steps", "error", "eval_margin", "market_hash",
    ]
    with pts_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pts_fields)
        writer.writeheader()
        for (group, metric, _margin), rows in sorted(points.items()):
            for row in sorted(rows, key=lambda item: (item["pres_target"], item["seed"])):
                writer.writerow({"group": group, "metric": metric, **row})

    per_path = stage / "per_target.csv"
    quantities = ("achieved_pres", "error", "total_inner_steps")
    per_fields = [
        "group", "metric", "eval_margin", "pres_target", "n_runs",
        "n_target_reached", "seeds",
    ] + [f"{name}_{field}" for name in quantities
         for field in ("mean", "std", "sem", "ci95_lo", "ci95_hi")]
    with per_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_fields)
        writer.writeheader()
        for (group, metric, margin), rows in sorted(points.items()):
            by_target: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_target[float(row["pres_target"])].append(row)
            for target in sorted(by_target):
                cell = by_target[target]
                output_row: Dict[str, Any] = {
                    "group": group, "metric": metric, "eval_margin": margin,
                    "pres_target": target, "n_runs": len(cell),
                    "n_target_reached": sum(int(row["target_reached"]) for row in cell),
                    "seeds": ";".join(str(seed) for seed in sorted(row["seed"] for row in cell)),
                }
                for name in quantities:
                    values = [float(row[name]) for row in cell if row[name] != ""]
                    stats = _summary_stats(values)
                    for field in ("mean", "std", "sem", "ci95_lo", "ci95_hi"):
                        output_row[f"{name}_{field}"] = fmt(stats[field])
                writer.writerow(output_row)

    fit_path = stage / "fit.csv"
    fit_fields = [
        "group", "metric", "eval_margin", "se_type", "n_points", "n_clusters",
        "slope", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "intercept", "r2",
    ]
    with fit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fit_fields)
        writer.writeheader()
        for (group, metric, margin), rows in sorted(points.items()):
            x = np.asarray([row["achieved_pres"] for row in rows], dtype=float)
            y = np.asarray([row["error"] for row in rows], dtype=float)
            seeds = np.asarray([str(row["seed"]) for row in rows])
            tolerances = np.asarray([row["pres_target"] for row in rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y, seeds, tolerances = x[ok], y[ok], seeds[ok], tolerances[ok]
            result = ols_loglog(x, y)
            clustered = cluster_robust_slope_se(x, y, seeds)
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "cluster_seed", "n_points": result["n"],
                "n_clusters": clustered["G"], "slope": fmt(result["slope"]),
                "slope_se": fmt(clustered["se"]), "ci95_lo": fmt(clustered["ci_lo"]),
                "ci95_hi": fmt(clustered["ci_hi"]),
                "t_crit": fmt(clustered["t_crit"]),
                "intercept": fmt(result["intercept"]), "r2": fmt(result["r2"], 4),
            })
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "ols_iid", "n_points": result["n"], "n_clusters": "",
                "slope": fmt(result["slope"]), "slope_se": fmt(result["slope_se"]),
                "ci95_lo": fmt(result["ci95_lo"]), "ci95_hi": fmt(result["ci95_hi"]),
                "t_crit": fmt(result["t_crit"]), "intercept": fmt(result["intercept"]),
                "r2": fmt(result["r2"], 4),
            })
            target_x: List[float] = []
            target_y: List[float] = []
            for target in sorted(set(tolerances.tolist())):
                mask = tolerances == target
                target_x.append(float(np.exp(np.mean(np.log(x[mask])))))
                target_y.append(float(np.exp(np.mean(np.log(y[mask])))))
            target_result = ols_loglog(np.asarray(target_x), np.asarray(target_y))
            writer.writerow({
                "group": group, "metric": metric, "eval_margin": margin,
                "se_type": "tolerance_mean", "n_points": target_result["n"],
                "n_clusters": "", "slope": fmt(target_result["slope"]),
                "slope_se": fmt(target_result["slope_se"]),
                "ci95_lo": fmt(target_result["ci95_lo"]),
                "ci95_hi": fmt(target_result["ci95_hi"]),
                "t_crit": fmt(target_result["t_crit"]),
                "intercept": fmt(target_result["intercept"]),
                "r2": fmt(target_result["r2"], 4),
            })

    metadata["figure_files"] = _plot_e6_cells(points, stage, args, formats)
    (stage / "e6_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "_SUCCESS_E6").write_text("success\n", encoding="utf-8")


def _commit_e6_outputs(stage: Path, output: Path, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in _owned_e6_paths(output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("managed E6 output appeared during staged generation")
    backup = Path(tempfile.mkdtemp(prefix=".e6_backup_", dir=str(output.parent)))
    moved_old: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    marker = stage / "_SUCCESS_E6"
    try:
        for path in existing:
            saved = backup / path.name
            os.replace(path, saved)
            moved_old.append((saved, path))
        ordered = [path for path in sorted(stage.iterdir()) if path != marker]
        ordered.append(marker)
        for path in ordered:
            destination = output / path.name
            os.replace(path, destination)
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


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Aggregate the E6 residual-tolerance sweep.")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--output", type=str, default=None, help="Default: <out-root>/e6_summary")
    ap.add_argument("--model-type", type=str, default="pipinn", help="Which model to aggregate (E6 is PI-PINN).")
    ap.add_argument("--metrics", type=str, default="RelL2_V,RelL2_theta",
                    help="Comma-separated metric names to fit against achieved p_res.")
    ap.add_argument("--include-stopped", action="store_true")
    ap.add_argument("--expected-seeds", default="",
                    help="Exact seed set required at every tolerance, e.g. 1-10 or 1,3,7.")
    ap.add_argument("--expected-tolerances", default="",
                    help="Exact positive pres_target set, e.g. 1e-2,1e-3,1e-4.")
    ap.add_argument("--min-runs-per-tolerance", type=int, default=1)
    ap.add_argument("--expected-n-assets", type=int, default=None)
    ap.add_argument("--expected-m-states", type=int, default=None)
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--fig-width", type=float, default=6.4)
    ap.add_argument("--fig-height", type=float, default=4.2)
    ap.add_argument("--font-size", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.min_runs_per_tolerance < 1:
        raise ValueError("--min-runs-per-tolerance must be positive")
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    expected_tolerances = parse_float_spec(args.expected_tolerances)
    expected_tolerance_keys = {_float_key(value) for value in expected_tolerances}
    formats = _parse_formats(args.formats)
    if args.fig_width <= 0 or args.fig_height <= 0 or args.font_size <= 0 or args.dpi <= 0:
        raise ValueError("figure dimensions, font size, and dpi must be positive")

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "e6_summary")
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise ValueError("--metrics must contain at least one metric")
    accepted = {"success"} | ({"stopped_early"} if args.include_stopped else set())

    # Select the newest attempt before inspecting its status. This prevents a
    # failed rerun from reviving an older success for the same tolerance/seed.
    newest: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for run_dir in find_runs(out_root):
        cfg = load_config_args_raw(run_dir)
        if cfg is None or str(cfg.get("model_type", "")) != args.model_type:
            continue
        try:
            n_assets = int(cfg.get("n_assets", -1))
            m_states = int(cfg.get("m_states", -1))
        except (TypeError, ValueError):
            continue
        if args.expected_n_assets is not None and n_assets != args.expected_n_assets:
            continue
        if args.expected_m_states is not None and m_states != args.expected_m_states:
            continue
        target_raw = cfg.get("pres_target", None)
        try:
            target = float(target_raw) if target_raw not in (None, "") else None
            seed = int(cfg.get("seed"))
        except (TypeError, ValueError):
            continue
        if target is None or not math.isfinite(target) or target <= 0.0:
            continue
        group = e6_group_key(cfg)
        record = {
            "group": group,
            "config": cfg,
            "run_dir_abs": os.path.abspath(run_dir),
            "run_dir": os.path.relpath(run_dir, out_root),
            "seed": seed,
            "pres_target": target,
            "status": run_status(run_dir),
            "updated_at": run_updated_at(run_dir),
        }
        key = (group, args.model_type, _float_key(target), seed)
        previous = newest.get(key)
        if previous is None or (record["updated_at"], record["run_dir_abs"]) >= (
            previous["updated_at"], previous["run_dir_abs"]
        ):
            newest[key] = record

    if not newest:
        raise ValueError("no E6 runs with a positive pres_target match the requested filters")

    validation_errors: List[str] = []
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in newest.values():
        by_group[record["group"]].append(record)

    selected_records: List[Dict[str, Any]] = []
    for group, records in sorted(by_group.items()):
        observed_tolerance_keys = {
            _float_key(record["pres_target"])
            for record in records if record["status"] in accepted
        }
        if expected_tolerance_keys and observed_tolerance_keys != expected_tolerance_keys:
            validation_errors.append(
                f"group={group}: tolerances={sorted(observed_tolerance_keys)} != "
                f"expected={sorted(expected_tolerance_keys)}"
            )
        tolerance_keys = expected_tolerance_keys or observed_tolerance_keys
        group_selected: List[Dict[str, Any]] = []
        for tolerance_key in sorted(tolerance_keys, key=float):
            attempts = [
                record for record in records
                if _float_key(record["pres_target"]) == tolerance_key
            ]
            accepted_rows = [record for record in attempts if record["status"] in accepted]
            accepted_seed_set = {record["seed"] for record in accepted_rows}
            if expected_seeds and accepted_seed_set != expected_seeds:
                statuses = {record["seed"]: record["status"] for record in attempts}
                validation_errors.append(
                    f"group={group}, tolerance={tolerance_key}: accepted seeds="
                    f"{sorted(accepted_seed_set)} != expected={sorted(expected_seeds)}; "
                    f"newest statuses={statuses}"
                )
            if len(accepted_rows) < args.min_runs_per_tolerance:
                validation_errors.append(
                    f"group={group}, tolerance={tolerance_key}: {len(accepted_rows)} accepted "
                    f"runs < min-runs-per-tolerance={args.min_runs_per_tolerance}"
                )
            group_selected.extend(accepted_rows)

        market_hashes: set[str] = set()
        for record in group_selected:
            market_path = os.path.join(record["run_dir_abs"], "market_params.npz")
            try:
                record["market_hash"] = canonical_market_hash(market_path)
                market_hashes.add(record["market_hash"])
            except Exception as exc:
                validation_errors.append(f"{record['run_dir_abs']}: invalid market snapshot: {exc}")
        if len(market_hashes) != 1:
            validation_errors.append(
                f"group={group}: expected one canonical market snapshot, found "
                f"{sorted(market_hashes)}"
            )
        selected_records.extend(group_selected)

    if validation_errors:
        raise ValueError("E6 support validation failed:\n  - " + "\n  - ".join(validation_errors))

    # points[(group, metric, primary margin)] -> one row per accepted run.
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        run_dir = record["run_dir_abs"]
        target = float(record["pres_target"])
        status = read_status(run_dir)
        if str(status.get("status", "")) != record["status"]:
            raise ValueError(
                f"{run_dir}: terminal marker/status.json disagreement: "
                f"{record['status']!r} vs {status.get('status')!r}"
            )
        achieved = status.get("pres_max", None)
        if not isinstance(achieved, (int, float)):
            achieved = achieved_pres_from_outer_history(run_dir)
        if achieved is None or not math.isfinite(float(achieved)) or not (float(achieved) > 0):
            raise ValueError(f"{run_dir}: missing positive achieved p_res")
        steps = status.get("total_inner_steps", None)
        if not isinstance(steps, (int, float)):
            steps = total_steps_from_outer_history(run_dir)
        reached = bool(achieved <= target * (1.0 + 1e-9))
        for metric in metrics:
            picked = pick_metric_value(run_dir, metric)
            if picked is None:
                raise ValueError(f"{run_dir}: required fulldim metric {metric!r} is missing")
            val, margin_used = picked
            if not math.isfinite(float(val)) or float(val) <= 0.0:
                raise ValueError(f"{run_dir}: required metric {metric!r} must be positive and finite")
            points[(record["group"], metric, margin_used)].append({
                "run_dir": record["run_dir"],
                "seed": record["seed"],
                "pres_target": target,
                "achieved_pres": float(achieved),
                "target_reached": int(reached),
                "total_inner_steps": steps if steps is not None else "",
                "error": float(val),
                "eval_margin": margin_used,
                "market_hash": record["market_hash"],
            })

    if not points:
        raise ValueError("no complete E6 metric rows were selected")

    # ``eval_margin`` is deliberately excluded from the training group hash,
    # but a paper-facing residual sweep cannot pool runs whose official metric
    # rows use different primary windows.  Validate support again after the
    # metric/margin split so a nominally balanced seed cell cannot fragment
    # into several one-seed summaries.
    support_errors: List[str] = []
    margins_by_group: Dict[str, set[float]] = defaultdict(set)
    for (group, _metric, margin) in points:
        margins_by_group[group].add(float(margin))
    for group, margins in sorted(margins_by_group.items()):
        if len(margins) != 1:
            support_errors.append(
                f"group={group}: mixed primary eval margins {sorted(margins)}"
            )
    for (group, metric, margin), rows in sorted(points.items()):
        by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_target[_float_key(float(row["pres_target"]))].append(row)
        for tolerance_key, cell in sorted(by_target.items(), key=lambda item: float(item[0])):
            seeds = {int(row["seed"]) for row in cell}
            if expected_seeds and seeds != expected_seeds:
                support_errors.append(
                    f"group={group}, metric={metric}, margin={margin}, "
                    f"tolerance={tolerance_key}: seeds={sorted(seeds)} != "
                    f"expected={sorted(expected_seeds)}"
                )
            if len(cell) < args.min_runs_per_tolerance:
                support_errors.append(
                    f"group={group}, metric={metric}, margin={margin}, "
                    f"tolerance={tolerance_key}: {len(cell)} metric rows < "
                    f"min-runs-per-tolerance={args.min_runs_per_tolerance}"
                )
    if support_errors:
        raise ValueError(
            "E6 metric-window support validation failed:\n  - "
            + "\n  - ".join(support_errors)
        )

    output = Path(summary_dir).resolve()
    _prepare_e6_output(output, args.overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "schema_version": 2,
        "out_root": out_root,
        "model_type": args.model_type,
        "metrics": metrics,
        "expected_seeds": sorted(expected_seeds),
        "expected_tolerances": expected_tolerances,
        "min_runs_per_tolerance": args.min_runs_per_tolerance,
        "expected_n_assets": args.expected_n_assets,
        "expected_m_states": args.expected_m_states,
        "n_groups": len({key[0] for key in points}),
        "n_metric_cells": len(points),
        "n_point_rows": sum(len(rows) for rows in points.values()),
        "market_hashes": sorted({row["market_hash"] for rows in points.values() for row in rows}),
        "uncertainty": "sample SD and Student-t 95% CI across training seeds within tolerance",
        "fit_x": "achieved p_res (never nominal target)",
    }
    stage = Path(tempfile.mkdtemp(prefix=".e6_stage_", dir=str(output.parent)))
    try:
        _write_e6_outputs(points, stage, args, metadata, formats)
        _commit_e6_outputs(stage, output, args.overwrite)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(f"[e6] wrote validated artifacts to: {output}")
    return metadata


if __name__ == "__main__":
    main()
