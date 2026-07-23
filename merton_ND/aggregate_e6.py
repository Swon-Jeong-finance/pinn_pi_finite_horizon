"""E6 aggregation: residual-tolerance sweep -> error-floor scaling.

Collects, per run: the pres target, the official post-restore ACHIEVED
residual level p_res = max_n p_res,n, target-reached status, total inner
optimizer steps, and the final error metrics.  The default paper error is the
final fixed-Q_ev ``e_Xev`` diagnostic; RelL2 policy errors are secondary.
Then, per configuration group (everything
shared except training seed and pres_target):

  per_target.csv   n_runs, n_target_reached, achieved p_res mean+-std,
                   error mean+-std, total inner steps mean+-std, per target
  points.csv       one row per run (x = achieved p_res, y = error)
  fit.csv          log10(error) ~ log10(achieved p_res) OLS slope with
                   standard error, 95% CI (t, df=n-2), intercept, R^2
  settings.csv     human-readable N/outer-budget/configuration manifest
  e6_residual_error_scaling_*.{png,pdf}
                   existing seed scatter/geometric-mean/pooled-fit figure
  e6_residual_error_scaling_mean_sd_*.{png,pdf}
                   per-target arithmetic mean +/- sample SD paper figure
                   with a slope-one reference line

Per the protocol: runs that did NOT reach their target enter the fit with
their ACHIEVED residual (never the nominal target); reached runs also use
the achieved value, which is <= target by construction, so the x-axis is
uniformly "achieved p_res".

The paper plot is generated configuration-by-configuration (configurations
are never pooled).  Its x coordinate is the official post-restore achieved
``p_res``; individual seed points, target-level geometric means, and the
pooled log-log fit are shown for ``e_Xev`` and the requested secondary
control metrics.

The trajectory diagnostic ``e_Xev`` always takes its primary-margin
provenance from the raw training config/status.  Successful eval-only
overlays may select the primary margin for final metrics.csv values only.
Every output row records this provenance, and a series that would mix
provenance is rejected rather than pooled.

Standalone: stdlib + numpy + matplotlib (+ scipy if available for exact t
quantiles).

Usage:
    python3 aggregate_e6.py --out-root <OUT_ROOT>
    python3 aggregate_e6.py --out-root <OUT_ROOT> --metrics e_Xev,RelL2_pi,RelL2_c
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from aggregate_seeds import (  # reuse shared helpers
    GROUP_IGNORE_KEYS, canonical_market_hash, find_runs, load_config_args,
    load_config_args_raw, parse_int_spec, parse_seed_spec, run_status,
    t_crit_95, fmt,
)

# E6 groups additionally collapse over the tolerance itself.
E6_IGNORE_KEYS = set(GROUP_IGNORE_KEYS) | {"pres_target"}


def e6_group_key(args: Dict[str, Any]) -> str:
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


def achieved_pres_from_outer_history(
    run_dir: str,
    *,
    allow_legacy: bool = False,
) -> Tuple[Optional[float], str]:
    """Return max per-outer residual and its state semantics.

    Paper E6 requires the residual of the exact checkpoint state used for the
    error measurement, after any held-out model+optimizer restore.  Legacy
    ``val_pres`` is accepted only behind an explicit diagnostic opt-in because
    old runs can attach it to a pre-restore training crossing.
    """
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None, "missing"
    field = ""
    semantics = "missing"
    best = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        if "val_pres_post_restore" in fields:
            field = "val_pres_post_restore"
            semantics = "max_outer_post_restore_fixed_qres"
        elif allow_legacy and "val_pres" in fields:
            field = "val_pres"
            semantics = "legacy_val_pres_explicitly_allowed"
        else:
            return None, "missing_post_restore_residual"
        for row in reader:
            raw = str(row.get(field, "")).strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if not math.isfinite(v) or v <= 0.0:
                continue
            best = v if best is None else max(best, v)
    return best, semantics


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


EVAL_PROVENANCE_KEYS = (
    "eval_margin",
    "test_points",
    "n_tau",
    "n_x",
    "w_levels",
)


def _parse_primary_margin(config: Mapping[str, Any]) -> Optional[float]:
    try:
        margins = [
            float(part)
            for part in str(config.get("eval_margin", "")).split(",")
            if str(part).strip()
        ]
    except (TypeError, ValueError):
        return None
    if not margins or not math.isfinite(margins[0]):
        return None
    return float(margins[0])


def training_primary_margin(
    run_dir: str,
    *,
    raw_config: Optional[Mapping[str, Any]] = None,
    status: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, str]:
    """Resolve the diagnostic margin from training artifacts only.

    ``e_Xev`` is measured while training and written to outer_history.csv.
    A later successful eval-only run may replace metrics.csv and write a new
    ``config_eval.json``; it cannot retroactively change that trajectory
    diagnostic's window.  Prefer the raw training config and use status.json
    as an independent consistency check/fallback.  Never call the overlaid
    ``load_config_args`` here.
    """
    raw = dict(raw_config or load_config_args_raw(run_dir) or {})
    training_status = dict(status or read_status(run_dir))
    config_margin = _parse_primary_margin(raw)
    status_margin: Optional[float] = None
    try:
        candidate = training_status.get("primary_margin")
        if candidate not in (None, ""):
            parsed = float(candidate)
            if math.isfinite(parsed):
                status_margin = parsed
    except (TypeError, ValueError):
        status_margin = None

    if config_margin is not None and status_margin is not None:
        if not math.isclose(config_margin, status_margin, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{run_dir}: training config primary margin={config_margin:g} "
                f"conflicts with status primary_margin={status_margin:g}"
            )
        return config_margin, "raw_training_config+training_status"
    if config_margin is not None:
        return config_margin, "raw_training_config"
    if status_margin is not None:
        return status_margin, "training_status"
    raise ValueError(
        f"{run_dir}: cannot determine the training primary margin for e_Xev"
    )


def successful_eval_overlay_active(run_dir: str) -> bool:
    """Mirror aggregate_seeds' guarded eval-only overlay activation test."""
    config_eval = os.path.join(run_dir, "config_eval.json")
    if not os.path.exists(config_eval):
        return False
    if not os.path.exists(os.path.join(run_dir, "_SUCCESS_EVAL")):
        return False
    try:
        with open(
            os.path.join(run_dir, "status_eval.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            if str(json.load(handle).get("status", "")) != "success":
                return False
    except Exception:
        return False
    try:
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if (
            os.path.exists(metrics_path)
            and os.path.getmtime(metrics_path) + 1e-6
            < os.path.getmtime(config_eval)
        ):
            return False
    except OSError:
        pass
    try:
        with open(config_eval, "r", encoding="utf-8") as handle:
            eval_args = json.load(handle).get("args", {})
        return isinstance(eval_args, dict)
    except Exception:
        return False


def metric_provenance(
    run_dir: str,
    metric: str,
    margin: float,
    *,
    raw_config: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    training_margin_source: Optional[str] = None,
) -> Dict[str, str]:
    """Return a canonical, hashable provenance contract for one error value."""
    if metric == "e_Xev":
        payload: Dict[str, Any] = {
            "metric_value_source": "outer_history.csv:final_outer",
            "primary_margin_source": (
                training_margin_source or "raw_training_config_or_status"
            ),
            "primary_margin": float(margin),
        }
        source = "training_trajectory"
    else:
        overlay = successful_eval_overlay_active(run_dir)
        source = (
            "successful_eval_only_overlay"
            if overlay else "raw_training_evaluation"
        )
        payload = {
            "metric_value_source": "metrics.csv:fulldim",
            "primary_margin_source": source,
            "primary_margin": float(margin),
            "effective_eval_config": {
                key: effective_config.get(key)
                for key in EVAL_PROVENANCE_KEYS
            },
        }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return {
        "metric_provenance_id": hashlib.sha1(
            canonical.encode("utf-8")
        ).hexdigest()[:12],
        "metric_provenance_source": source,
        "primary_margin_source": str(payload["primary_margin_source"]),
        "metric_provenance_json": canonical,
    }


def pick_metric_value(
    run_dir: str,
    metric: str,
    *,
    effective_config: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    """Return (value, eval_margin_used) for the requested metric.

    FULL-DIMENSIONAL rows only, at the run's PRIMARY margin (first listed in
    the explicitly supplied effective config).  The smallest recorded margin
    is a legacy fallback only when no primary margin can be recovered.  If a
    declared primary margin is absent, fail rather than relabel another row.
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

    cfg = dict(effective_config or load_config_args(run_dir) or {})
    primary = _parse_primary_margin(cfg)

    if primary is not None:
        exact = [r for r in rows if abs(r[0] - primary) < 1e-12]
        if exact:
            m, v = exact[-1]
            return (v, m)
        return None
    m, v = sorted(rows, key=lambda z: z[0])[0]
    return (v, m)


def pick_outer_metric_value(
    run_dir: str,
    metric: str,
    *,
    raw_config: Optional[Mapping[str, Any]] = None,
    status: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    """Return the final finite outer-history diagnostic at the primary margin.

    ``e_Xev`` is a trajectory diagnostic rather than a generic random-test
    metric, so it lives in outer_history.csv.  Selecting the largest outer
    index makes the official final value explicit and avoids silently using a
    diagnostic-best checkpoint.
    """
    path = os.path.join(run_dir, "outer_history.csv")
    if not os.path.exists(path):
        return None
    rows_by_outer: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in set(reader.fieldnames or ()):
            return None
        for row in reader:
            try:
                outer = int(float(str(row.get("outer_iter", ""))))
            except (TypeError, ValueError):
                continue
            if outer in rows_by_outer:
                raise ValueError(
                    f"{path}: duplicate outer_iter={outer}; final e_Xev is ambiguous"
                )
            rows_by_outer[outer] = str(row.get(metric, ""))
    if not rows_by_outer:
        return None
    status = read_status(run_dir)
    declared_final = status.get("final_outer_iter")
    try:
        final_outer = int(declared_final) if declared_final not in (None, "") else max(rows_by_outer)
    except (TypeError, ValueError):
        return None
    if final_outer != max(rows_by_outer) or final_outer not in rows_by_outer:
        raise ValueError(
            f"{path}: final_outer_iter={final_outer} does not match outer-history "
            f"maximum={max(rows_by_outer)}"
        )
    try:
        value = float(rows_by_outer[final_outer])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    margin, _source = training_primary_margin(
        run_dir,
        raw_config=raw_config,
        status=status,
    )
    return value, float(margin)


def pick_error_value(
    run_dir: str,
    metric: str,
    *,
    raw_config: Optional[Mapping[str, Any]] = None,
    effective_config: Optional[Mapping[str, Any]] = None,
    status: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    if metric == "e_Xev":
        return pick_outer_metric_value(
            run_dir,
            metric,
            raw_config=raw_config,
            status=status,
        )
    return pick_metric_value(
        run_dir,
        metric,
        effective_config=effective_config,
    )


def mean_std(vals: List[float]) -> Tuple[float, float, int]:
    a = np.asarray(vals, dtype=float)
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    # A sample standard deviation is not estimable from one seed.  Keeping it
    # missing (rather than reporting a misleading zero) matches the paper
    # aggregation convention used by the E1/E8 tables.
    return float(a.mean()), float(a.std(ddof=1)) if n > 1 else float("nan"), n


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


SUPPORTED_FIGURE_FORMATS = {"png", "pdf", "eps", "svg"}
STATIC_OUTPUT_NAMES = {
    "points.csv",
    "per_target.csv",
    "fit.csv",
    "settings.csv",
    "e6_metadata.json",
    "validation_errors.txt",
}


def parse_formats(text: str) -> List[str]:
    formats = [
        part.lower().lstrip(".")
        for part in re.split(r"[\s,]+", str(text))
        if part
    ]
    if not formats:
        raise ValueError("--formats must contain at least one format")
    if len(set(formats)) != len(formats):
        raise ValueError(f"duplicate formats in --formats={text!r}")
    invalid = sorted(set(formats) - SUPPORTED_FIGURE_FORMATS)
    if invalid:
        raise ValueError(
            f"unsupported figure formats {invalid}; choose from "
            f"{sorted(SUPPORTED_FIGURE_FORMATS)}"
        )
    return formats


def prepare_output(path: Path, *, overwrite: bool) -> None:
    """Create the summary directory without deleting unrelated artifacts."""
    if path.exists() and not path.is_dir():
        raise ValueError(f"output exists and is not a directory: {path}")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return
    entries = list(path.iterdir())
    if entries and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {path}; pass --overwrite to replace "
            "E6-owned artifacts"
        )
    if not overwrite:
        return
    for entry in entries:
        owned = (
            entry.name in STATIC_OUTPUT_NAMES
            or entry.name.startswith("e6_residual_error_scaling_")
        )
        if not owned:
            continue
        if not entry.is_file():
            raise ValueError(
                f"refusing --overwrite because reserved output is not a file: {entry}"
            )
        entry.unlink()


def _as_int_or_blank(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return ""


def setting_metadata(config: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """Paper-readable identity for one pres-target-independent setting."""
    model_type = str(config.get("model_type", ""))
    n_assets = _as_int_or_blank(config.get("n_assets"))
    outer_iters = _as_int_or_blank(config.get("outer_iters"))
    method_label = "PI-PINN" if model_type == "pipinn" else model_type.upper()
    n_label = str(n_assets) if n_assets != "" else "?"
    k_label = str(outer_iters) if outer_iters != "" else "?"
    core = {
        key: config[key]
        for key in sorted(config)
        if key not in E6_IGNORE_KEYS
    }
    return {
        "group": group,
        "model_type": model_type,
        "n_assets": n_assets,
        "outer_iters": outer_iters,
        "setting_label": f"{method_label}, N={n_label}, K={k_label}",
        "setting_config_json": json.dumps(core, sort_keys=True, default=str),
    }


def _margin_slug(margin: float) -> str:
    text = f"{float(margin):g}".replace("-", "m").replace(".", "p")
    return f"margin{text}"


def plot_e6(
    summary_dir: Path,
    points: Mapping[Tuple[str, str, float], List[Dict[str, Any]]],
    group_meta: Mapping[str, Mapping[str, Any]],
    requested_metrics: Sequence[str],
    *,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    """Render one non-pooled paper figure per configuration and margin."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metric_labels = {
        "e_Xev": r"Final $e_{X_{ev}}$",
        "RelL2_pi": r"Final relative-$L^2$ portfolio error",
        "RelL2_c": r"Final relative-$L^2$ consumption error",
    }
    group_margins = sorted({(g, float(margin)) for g, _metric, margin in points})
    written: List[str] = []
    for group, margin in group_margins:
        available = {
            metric for g, metric, m in points
            if g == group and float(m) == margin
        }
        metrics = [metric for metric in requested_metrics if metric in available]
        if not metrics:
            continue
        fig, axes = plt.subplots(
            1, len(metrics), figsize=(4.2 * len(metrics), 3.65), squeeze=False
        )
        for ax, metric in zip(axes[0], metrics):
            rows = points[(group, metric, margin)]
            valid = [
                row for row in rows
                if float(row["achieved_pres"]) > 0.0 and float(row["error"]) > 0.0
            ]
            reached = [row for row in valid if int(row["target_reached"]) == 1]
            unreached = [row for row in valid if int(row["target_reached"]) == 0]
            if reached:
                ax.scatter(
                    [row["achieved_pres"] for row in reached],
                    [row["error"] for row in reached],
                    s=24, alpha=0.38, color="#4C78A8", marker="o",
                    label="seed (target reached)",
                )
            if unreached:
                ax.scatter(
                    [row["achieved_pres"] for row in unreached],
                    [row["error"] for row in unreached],
                    s=34, alpha=0.70, color="#E45756", marker="x",
                    label="seed (target not reached)",
                )

            # Both axes and the fit are logarithmic, so target summaries are
            # geometric means.  Their x values remain achieved residuals.
            gx: List[float] = []
            gy: List[float] = []
            for target in sorted({float(row["pres_target"]) for row in valid}):
                subset = [row for row in valid if float(row["pres_target"]) == target]
                gx.append(float(np.exp(np.mean(np.log([
                    float(row["achieved_pres"]) for row in subset
                ])))))
                gy.append(float(np.exp(np.mean(np.log([
                    float(row["error"]) for row in subset
                ])))))
            if gx:
                order = np.argsort(np.asarray(gx))
                gx_arr, gy_arr = np.asarray(gx)[order], np.asarray(gy)[order]
                ax.plot(
                    gx_arr, gy_arr, "o-", color="#1F4E79", linewidth=1.5,
                    markersize=5, label="target geometric mean",
                )

            x = np.asarray([float(row["achieved_pres"]) for row in valid])
            y = np.asarray([float(row["error"]) for row in valid])
            fit = ols_loglog(x, y)
            if x.size >= 2 and math.isfinite(fit.get("slope", float("nan"))):
                xx = np.geomspace(float(x.min()), float(x.max()), 160)
                yy = 10.0 ** (
                    float(fit["intercept"])
                    + float(fit["slope"]) * np.log10(xx)
                )
                ax.plot(
                    xx, yy, "--", color="#333333", linewidth=1.25,
                    label=rf"pooled fit: slope={fit['slope']:.2f}",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Official achieved $p_{res}$ (post-restore)")
            ax.set_ylabel(metric_labels.get(metric, metric))
            title = "Primary" if metric == "e_Xev" else "Secondary"
            ax.set_title(f"{title}: {metric}")
            ax.grid(True, which="both", alpha=0.24)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7.5, loc="best")

        meta = group_meta[group]
        fig.suptitle(
            f"E6 residual-error scaling: {meta['setting_label']} "
            f"(evaluation margin={margin:g})",
            y=1.02,
        )
        fig.tight_layout()
        base = (
            f"e6_residual_error_scaling_N{meta['n_assets']}_K{meta['outer_iters']}_"
            f"{group}_{_margin_slug(margin)}"
        )
        for figure_format in formats:
            path = summary_dir / f"{base}.{figure_format}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path.name)
        plt.close(fig)
    return written


def _log_safe_sd_errors(
    means: np.ndarray,
    sample_sds: np.ndarray,
) -> np.ndarray:
    """Return asymmetric log-safe whiskers representing one sample SD.

    The upper whisker is the full sample SD.  Only a nonpositive lower
    endpoint is clipped to the nearest positive display value; the underlying
    arithmetic means and sample SDs remain unchanged in per_target.csv.
    """
    lower = np.asarray(sample_sds, dtype=float).copy()
    upper = np.asarray(sample_sds, dtype=float).copy()
    finite = np.isfinite(lower)
    lower[~finite] = np.nan
    upper[~finite] = np.nan
    lower[finite] = np.minimum(
        lower[finite],
        means[finite] * (1.0 - 1e-9),
    )
    return np.vstack([lower, upper])


def plot_e6_mean_sd(
    summary_dir: Path,
    points: Mapping[Tuple[str, str, float], List[Dict[str, Any]]],
    group_meta: Mapping[str, Mapping[str, Any]],
    requested_metrics: Sequence[str],
    *,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    """Render the paper summary: per-target mean +/- sample SD and slope 1."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metric_labels = {
        "e_Xev": r"Final $e_{X_{ev}}$",
        "RelL2_pi": r"Final relative-$L^2$ portfolio error",
        "RelL2_c": r"Final relative-$L^2$ consumption error",
    }
    group_margins = sorted({(g, float(margin)) for g, _metric, margin in points})
    written: List[str] = []
    for group, margin in group_margins:
        available = {
            metric for g, metric, m in points
            if g == group and float(m) == margin
        }
        metrics = [metric for metric in requested_metrics if metric in available]
        if not metrics:
            continue
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(4.2 * len(metrics), 3.65),
            squeeze=False,
        )
        for ax, metric in zip(axes[0], metrics):
            rows = [
                row
                for row in points[(group, metric, margin)]
                if float(row["achieved_pres"]) > 0.0
                and float(row["error"]) > 0.0
            ]
            summaries: List[Tuple[float, float, float, float, float, int]] = []
            for target in sorted({float(row["pres_target"]) for row in rows}):
                subset = [
                    row
                    for row in rows
                    if float(row["pres_target"]) == target
                ]
                x_mean, x_sd, n = mean_std([
                    float(row["achieved_pres"]) for row in subset
                ])
                y_mean, y_sd, _ = mean_std([
                    float(row["error"]) for row in subset
                ])
                summaries.append((target, x_mean, x_sd, y_mean, y_sd, n))

            if summaries:
                summaries.sort(key=lambda item: item[1])
                x_mean = np.asarray([item[1] for item in summaries], dtype=float)
                x_sd = np.asarray([item[2] for item in summaries], dtype=float)
                y_mean = np.asarray([item[3] for item in summaries], dtype=float)
                y_sd = np.asarray([item[4] for item in summaries], dtype=float)
                ax.errorbar(
                    x_mean,
                    y_mean,
                    xerr=_log_safe_sd_errors(x_mean, x_sd),
                    yerr=_log_safe_sd_errors(y_mean, y_sd),
                    fmt="o-",
                    color="#1F4E79",
                    ecolor="#6B8EAD",
                    linewidth=1.5,
                    elinewidth=1.0,
                    capsize=3,
                    markersize=5,
                    label=r"target mean $\pm$ sample SD",
                )

                # A slope-one guide requires a vertical placement.  Anchor it
                # at the geometric mean of y_bar/x_bar across target levels.
                anchor = float(np.exp(np.mean(np.log(y_mean / x_mean))))
                x_lo = float(x_mean.min())
                x_hi = float(x_mean.max())
                if math.isclose(x_lo, x_hi, rel_tol=1e-12, abs_tol=0.0):
                    x_lo *= 0.8
                    x_hi *= 1.25
                xx = np.geomspace(x_lo, x_hi, 160)
                ax.plot(
                    xx,
                    anchor * xx,
                    ":",
                    color="#333333",
                    linewidth=1.35,
                    label="slope-1 reference",
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Mean achieved $p_{res}$ (post-restore)")
            ax.set_ylabel(metric_labels.get(metric, metric))
            title = "Primary" if metric == "e_Xev" else "Secondary"
            ax.set_title(f"{title}: {metric}")
            ax.grid(True, which="both", alpha=0.24)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=7.5, loc="best")

        meta = group_meta[group]
        fig.suptitle(
            f"E6 mean +/- sample SD: {meta['setting_label']} "
            f"(evaluation margin={margin:g})",
            y=1.02,
        )
        fig.tight_layout()
        base = (
            f"e6_residual_error_scaling_mean_sd_"
            f"N{meta['n_assets']}_K{meta['outer_iters']}_"
            f"{group}_{_margin_slug(margin)}"
        )
        for figure_format in formats:
            path = summary_dir / f"{base}.{figure_format}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path.name)
        plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aggregate the E6 residual-tolerance sweep.")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--output", type=str, default=None, help="Default: <out-root>/e6_summary")
    ap.add_argument("--model-type", type=str, default="pipinn", help="Which model to aggregate (E6 is PI-PINN).")
    ap.add_argument("--metrics", type=str, default="e_Xev,RelL2_pi,RelL2_c",
                    help="Comma-separated metric names to fit against achieved p_res.")
    ap.add_argument("--include-stopped", action="store_true")
    ap.add_argument("--expected-seeds", type=str, default="",
                    help="Exact seed set required at every residual target, e.g. 1-10.")
    ap.add_argument("--expected-n-assets", type=str, default="",
                    help="Exact asset dimensions required, e.g. 10,50.")
    ap.add_argument(
        "--outer-iters", type=str, default="",
        help="Optional exact outer-budget filter, e.g. 20 or 20,30.",
    )
    ap.add_argument("--strict-market-snapshots", action="store_true",
                    help="Require one canonical Merton market per N. Enabled by --expected-seeds.")
    ap.add_argument(
        "--allow-legacy-residual-semantics",
        action="store_true",
        help=(
            "Diagnostic-only compatibility: permit outer_history val_pres when "
            "val_pres_post_restore is absent. Never use this for the paper E6 fit."
        ),
    )
    ap.add_argument("--formats", type=str, default="png,pdf")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "e6_summary")
    summary_path = Path(summary_dir).expanduser().resolve()
    if args.dpi < 36:
        raise ValueError("--dpi must be at least 36")
    formats = parse_formats(args.formats)
    prepare_output(summary_path, overwrite=bool(args.overwrite))
    summary_dir = str(summary_path)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("--metrics must contain unique non-empty names")
    accepted = {"success"} | ({"stopped_early"} if args.include_stopped else set())
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    expected_n_assets = set(parse_int_spec(args.expected_n_assets, label="--expected-n-assets"))
    selected_outer_iters = set(parse_int_spec(args.outer_iters, label="--outer-iters"))
    strict_market = bool(args.strict_market_snapshots or expected_seeds)
    validation_errors: List[str] = []
    market_rows: List[Dict[str, Any]] = []
    group_meta: Dict[str, Dict[str, Any]] = {}

    # points[(group, metric, primary_margin)] -> per-run rows. The group hash
    # is always derived from the RAW TRAINING config and includes every
    # training-relevant setting except seed and pres_target.  Metric-level
    # provenance is recorded on every row and must be unique within a series.
    points: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = defaultdict(list)

    for run_dir in find_runs(out_root):
        raw_cfg = load_config_args_raw(run_dir)
        if raw_cfg is None or str(raw_cfg.get("model_type", "")) != args.model_type:
            continue
        # Only final metrics.csv rows may use a successful eval-only overlay.
        # The training group, target, residual, and e_Xev provenance never do.
        effective_eval_cfg = load_config_args(run_dir) or raw_cfg
        if selected_outer_iters:
            try:
                outer_budget = int(raw_cfg.get("outer_iters"))
            except (TypeError, ValueError):
                validation_errors.append(
                    f"{run_dir}: --outer-iters filter requested but config "
                    f"outer_iters is invalid: {raw_cfg.get('outer_iters')!r}"
                )
                continue
            if outer_budget not in selected_outer_iters:
                continue
        if run_status(run_dir) not in accepted:
            continue
        target_raw = raw_cfg.get("pres_target", None)
        try:
            target = float(target_raw) if target_raw is not None and str(target_raw) != "" else None
        except (TypeError, ValueError):
            target = None
        if target is None:
            continue  # not an E6 run

        status = read_status(run_dir)
        status_semantics = str(status.get("pres_max_semantics", ""))
        status_achieved = status.get("pres_max", None)
        achieved, residual_semantics = achieved_pres_from_outer_history(
            run_dir,
            allow_legacy=bool(args.allow_legacy_residual_semantics),
        )
        if (status_semantics == "max_outer_post_restore_fixed_qres"
                and isinstance(status_achieved, (int, float))
                and achieved is not None
                and not math.isclose(
                    float(status_achieved), float(achieved),
                    rel_tol=1e-10, abs_tol=0.0,
                )):
            validation_errors.append(
                f"{run_dir}: status pres_max={float(status_achieved):.12g} "
                f"does not match recomputed outer-history max={float(achieved):.12g}"
            )
        if achieved is None or not (achieved > 0):
            validation_errors.append(
                f"{run_dir}: missing official post-restore p_res "
                f"(status semantics={status_semantics!r}, "
                f"outer-history semantics={residual_semantics!r})"
            )
            continue
        steps = status.get("total_inner_steps", None)
        if not isinstance(steps, (int, float)):
            steps = total_steps_from_outer_history(run_dir)
        # This definition is state-consistent even if an old sticky
        # training-crossing flag happens to be present in status.json.
        reached = bool(achieved <= target * (1.0 + 1e-9))
        reached_raw = status.get("target_reached", None)
        reached_semantics = str(status.get("target_reached_semantics", ""))
        if (reached_semantics
                == "all_outer_post_restore_fixed_qres_at_or_below_target"
                and isinstance(reached_raw, (bool, int, float))
                and bool(reached_raw) != reached):
            validation_errors.append(
                f"{run_dir}: status target_reached={bool(reached_raw)} conflicts "
                f"with official post-restore pres_max={float(achieved):.6g} and "
                f"target={target:.6g}"
            )

        g = e6_group_key(raw_cfg)
        current_meta = setting_metadata(raw_cfg, g)
        if g in group_meta and group_meta[g] != current_meta:
            validation_errors.append(
                f"{run_dir}: hash collision/inconsistent metadata for E6 group={g}"
            )
        else:
            group_meta[g] = current_meta
        ts = str(status.get("updated_at", ""))
        market_hash = ""
        market_error = ""
        try:
            market_hash = canonical_market_hash(os.path.join(run_dir, "market_params.npz"))
        except Exception as exc:
            market_error = str(exc)
        market_rows.append({
            "group": g, "run_dir": os.path.relpath(run_dir, out_root),
            "seed": raw_cfg.get("seed"), "pres_target": target,
            "n_assets": raw_cfg.get("n_assets"), "market_hash": market_hash,
            "market_error": market_error, "updated_at": ts,
        })
        for metric in metrics:
            try:
                picked = pick_error_value(
                    run_dir,
                    metric,
                    raw_config=raw_cfg,
                    effective_config=effective_eval_cfg,
                    status=status,
                )
            except ValueError as exc:
                validation_errors.append(str(exc))
                continue
            if picked is None:
                location = "outer_history final diagnostic" if metric == "e_Xev" else "fulldim metric"
                validation_errors.append(
                    f"{run_dir}: {location} {metric} missing/nonfinite on the "
                    "official final state"
                )
                continue
            val, margin_used = picked
            training_margin_source = None
            if metric == "e_Xev":
                try:
                    _margin, training_margin_source = training_primary_margin(
                        run_dir,
                        raw_config=raw_cfg,
                        status=status,
                    )
                except ValueError as exc:
                    validation_errors.append(str(exc))
                    continue
            provenance = metric_provenance(
                run_dir,
                metric,
                margin_used,
                raw_config=raw_cfg,
                effective_config=effective_eval_cfg,
                training_margin_source=training_margin_source,
            )
            points[(g, metric, margin_used)].append({
                "_ts": ts,
                "run_dir": os.path.relpath(run_dir, out_root),
                "seed": raw_cfg.get("seed"),
                "model_type": current_meta["model_type"],
                "n_assets": raw_cfg.get("n_assets"),
                "outer_iters": current_meta["outer_iters"],
                "setting_label": current_meta["setting_label"],
                "pres_target": target,
                "achieved_pres": float(achieved),
                "residual_semantics": residual_semantics,
                "target_reached": int(reached),
                "total_inner_steps": steps if steps is not None else "",
                "error": float(val),
                "eval_margin": margin_used,
                **provenance,
            })

    if not points:
        message = "no E6 runs (pres_target set) found"
        if validation_errors:
            error_path = os.path.join(summary_dir, "validation_errors.txt")
            with open(error_path, "w", encoding="utf-8") as f:
                f.write("\n".join(validation_errors) + "\n")
            raise SystemExit(
                f"E6 aggregation validation failed; see {error_path}"
            )
        if expected_seeds or expected_n_assets or strict_market:
            raise SystemExit(f"E6 aggregation validation failed: {message}")
        print(f"[warn] {message}.")
        return 0

    # Rerun dedup UPSTREAM: keep the newest run per (pres_target, seed) so
    # points.csv, per_target.csv and fit.csv all see the SAME data.
    for k in list(points.keys()):
        best: Dict[Tuple[float, Any], Dict[str, Any]] = {}
        for r in points[k]:
            kk = (r["pres_target"], r["seed"])
            if kk not in best or r["_ts"] >= best[kk]["_ts"]:
                best[kk] = r
        points[k] = list(best.values())

    # A deduplicated series must never silently pool training diagnostics,
    # original final metrics, and successful eval-only replacements.
    # Different primary margins already occupy different keys; any remaining
    # provenance split within one key is a hard validation error.
    for (group, metric, margin), rows in sorted(points.items()):
        provenance_ids = {
            str(row["metric_provenance_id"])
            for row in rows
        }
        if len(provenance_ids) > 1:
            provenance_sources = sorted({
                str(row["metric_provenance_source"])
                for row in rows
            })
            validation_errors.append(
                f"group={group} metric={metric} margin={margin:g}: "
                f"mixed metric provenance ids={sorted(provenance_ids)} "
                f"sources={provenance_sources}"
            )

    newest_panel: Dict[Tuple[str, float, Any], Dict[str, Any]] = {}
    for row in market_rows:
        key = (str(row["group"]), float(row["pres_target"]), row["seed"])
        if key not in newest_panel or row["updated_at"] >= newest_panel[key]["updated_at"]:
            newest_panel[key] = row

    if expected_seeds:
        panel_seeds: Dict[Tuple[str, float], set] = defaultdict(set)
        for (group, target, seed), _row in newest_panel.items():
            panel_seeds[(group, target)].add(int(seed))
        for (group, target), seeds in sorted(panel_seeds.items()):
            if seeds != expected_seeds:
                validation_errors.append(
                    f"run panel group={group} target={target:g}: seeds={sorted(seeds)}, "
                    f"expected={sorted(expected_seeds)}"
                )

        panel_targets: Dict[str, set] = defaultdict(set)
        for group, target, _seed in newest_panel:
            panel_targets[group].add(target)
        for group, targets in panel_targets.items():
            for metric in metrics:
                keys = [k for k in points if k[0] == group and k[1] == metric]
                if len(keys) != 1:
                    validation_errors.append(
                        f"group={group}: metric={metric} must have exactly one primary-margin "
                        f"series; found {keys}"
                    )
                    continue
                present_targets = {float(r["pres_target"]) for r in points[keys[0]]}
                if present_targets != targets:
                    validation_errors.append(
                        f"group={group} metric={metric}: targets={sorted(present_targets)}, "
                        f"expected={sorted(targets)}"
                    )

    observed_n_assets = {
        int(r["n_assets"])
        for rows in points.values() for r in rows
        if r.get("n_assets") is not None
    }
    if expected_n_assets and observed_n_assets != expected_n_assets:
        validation_errors.append(
            f"observed N={sorted(observed_n_assets)}, expected exactly={sorted(expected_n_assets)}"
        )

    if expected_seeds:
        # Every (configuration, metric, target) must contain the same exact
        # seed panel. This is stricter than merely checking the pooled fit.
        for (group, metric, margin), rows in sorted(points.items()):
            by_target: Dict[float, set] = defaultdict(set)
            for row in rows:
                by_target[float(row["pres_target"])].add(int(row["seed"]))
            for target, seeds in sorted(by_target.items()):
                if seeds != expected_seeds:
                    validation_errors.append(
                        f"group={group} metric={metric} margin={margin:g} target={target:g}: "
                        f"seeds={sorted(seeds)}, expected={sorted(expected_seeds)}"
                    )

    if strict_market:
        # Deduplicate reruns consistently with the metric points, then compare
        # all E6 tolerances/method instances within the same asset dimension.
        by_n: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in newest_panel.values():
            by_n[row["n_assets"]].append(row)
        expected_ns = expected_n_assets or set(by_n)
        for n_assets in sorted(expected_ns, key=str):
            rows = by_n.get(n_assets, [])
            invalid = [r for r in rows if r["market_error"] or not r["market_hash"]]
            hashes = {r["market_hash"] for r in rows if r["market_hash"]}
            if invalid:
                validation_errors.append(
                    f"market N={n_assets}: {len(invalid)} missing/invalid snapshot(s)"
                )
            if len(hashes) != 1:
                validation_errors.append(
                    f"market N={n_assets}: expected one canonical hash, found {sorted(hashes)}"
                )

    error_path = os.path.join(summary_dir, "validation_errors.txt")
    if validation_errors:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("\n".join(validation_errors) + "\n")
        for message in validation_errors[:25]:
            print(f"[validation error] {message}")
        raise SystemExit(f"E6 aggregation validation failed; see {error_path}")
    if os.path.exists(error_path):
        os.remove(error_path)

    # ---- settings.csv: human-readable, non-pooled configuration manifest ----
    settings_path = os.path.join(summary_dir, "settings.csv")
    settings_fields = [
        "group", "setting_label", "model_type", "n_assets", "outer_iters",
        "n_runs", "n_targets", "seeds", "pres_targets", "eval_margins",
        "metrics", "metric_provenance_ids", "metric_provenance_sources",
        "setting_config_json",
    ]
    with open(settings_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=settings_fields)
        w.writeheader()
        for group in sorted({key[0] for key in points}):
            meta = group_meta[group]
            panel = [row for row in newest_panel.values() if row["group"] == group]
            group_keys = [key for key in points if key[0] == group]
            w.writerow({
                **meta,
                "n_runs": len(panel),
                "n_targets": len({float(row["pres_target"]) for row in panel}),
                "seeds": ";".join(
                    str(seed) for seed in sorted({int(row["seed"]) for row in panel})
                ),
                "pres_targets": ";".join(
                    f"{target:g}" for target in sorted({
                        float(row["pres_target"]) for row in panel
                    })
                ),
                "eval_margins": ";".join(
                    f"{margin:g}" for margin in sorted({float(key[2]) for key in group_keys})
                ),
                "metrics": ";".join(sorted({key[1] for key in group_keys})),
                "metric_provenance_ids": ";".join(sorted({
                    str(row["metric_provenance_id"])
                    for key in group_keys for row in points[key]
                })),
                "metric_provenance_sources": ";".join(sorted({
                    str(row["metric_provenance_source"])
                    for key in group_keys for row in points[key]
                })),
            })

    # ---- points.csv ----
    pts_path = os.path.join(summary_dir, "points.csv")
    pts_fields = [
        "group", "setting_label", "model_type", "n_assets", "outer_iters",
        "metric", "run_dir", "seed", "pres_target", "achieved_pres",
        "residual_semantics", "target_reached", "total_inner_steps", "error",
        "eval_margin", "metric_provenance_id", "metric_provenance_source",
        "primary_margin_source", "metric_provenance_json",
    ]
    with open(pts_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pts_fields)
        w.writeheader()
        for (g, metric, margin), rows in sorted(points.items()):
            for r in rows:
                w.writerow({"group": g, "metric": metric,
                            **{k: v for k, v in r.items() if k != "_ts"}})

    # ---- per_target.csv ----
    per_path = os.path.join(summary_dir, "per_target.csv")
    per_fields = [
        "group", "setting_label", "model_type", "n_assets", "outer_iters",
        "metric", "eval_margin", "pres_target", "n_runs", "n_target_reached",
        "achieved_pres_mean", "achieved_pres_std", "error_mean", "error_std",
        "total_inner_steps_mean", "total_inner_steps_std", "seeds",
        "metric_provenance_id", "metric_provenance_source",
        "primary_margin_source", "metric_provenance_json",
    ]
    with open(per_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_fields)
        w.writeheader()
        for (g, metric, margin), rows in sorted(points.items()):
            by_t: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_t[r["pres_target"]].append(r)
            for t_val in sorted(by_t):
                rs = by_t[t_val]
                am, asd, n = mean_std([r["achieved_pres"] for r in rs])
                em, esd, _ = mean_std([r["error"] for r in rs])
                st = [float(r["total_inner_steps"]) for r in rs if r["total_inner_steps"] != ""]
                sm, ssd, _ = mean_std(st) if st else (float("nan"), float("nan"), 0)
                meta = group_meta[g]
                w.writerow({
                    "group": g, "setting_label": meta["setting_label"],
                    "model_type": meta["model_type"], "n_assets": meta["n_assets"],
                    "outer_iters": meta["outer_iters"], "metric": metric,
                    "eval_margin": margin, "pres_target": t_val,
                    "n_runs": n, "n_target_reached": sum(r["target_reached"] for r in rs),
                    "achieved_pres_mean": fmt(am), "achieved_pres_std": fmt(asd),
                    "error_mean": fmt(em), "error_std": fmt(esd),
                    "total_inner_steps_mean": fmt(sm), "total_inner_steps_std": fmt(ssd),
                    "seeds": ";".join(str(r["seed"]) for r in rs),
                    "metric_provenance_id":
                        rs[0]["metric_provenance_id"],
                    "metric_provenance_source":
                        rs[0]["metric_provenance_source"],
                    "primary_margin_source":
                        rs[0]["primary_margin_source"],
                    "metric_provenance_json":
                        rs[0]["metric_provenance_json"],
                })

    # ---- fit.csv ----
    fit_path = os.path.join(summary_dir, "fit.csv")
    fit_fields = [
        "group", "setting_label", "model_type", "n_assets", "outer_iters",
        "metric", "eval_margin", "se_type", "n_points", "n_clusters",
        "slope", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "intercept", "r2",
        "metric_provenance_id", "metric_provenance_source",
        "primary_margin_source", "metric_provenance_json",
    ]
    with open(fit_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fit_fields)
        w.writeheader()
        for (g, metric, margin), rows in sorted(points.items()):
            rr = rows  # already deduped upstream
            x = np.asarray([r["achieved_pres"] for r in rr], dtype=float)
            y = np.asarray([r["error"] for r in rr], dtype=float)
            seeds = np.asarray([str(r["seed"]) for r in rr])
            tols = np.asarray([r["pres_target"] for r in rr], dtype=float)
            ok = (x > 0) & (y > 0)
            x, y, seeds, tols = x[ok], y[ok], seeds[ok], tols[ok]

            # (a) pooled OLS + seed-cluster-robust SE (PRIMARY: the same
            #     seeds recur across tolerances, so iid SEs are too small).
            res = ols_loglog(x, y)
            cl = cluster_robust_slope_se(x, y, seeds)
            meta = group_meta[g]
            identity = {
                "group": g, "setting_label": meta["setting_label"],
                "model_type": meta["model_type"], "n_assets": meta["n_assets"],
                "outer_iters": meta["outer_iters"], "metric": metric,
                "eval_margin": margin,
                "metric_provenance_id": rr[0]["metric_provenance_id"],
                "metric_provenance_source":
                    rr[0]["metric_provenance_source"],
                "primary_margin_source": rr[0]["primary_margin_source"],
                "metric_provenance_json": rr[0]["metric_provenance_json"],
            }
            w.writerow({
                **identity,
                "se_type": "cluster_seed", "n_points": res["n"], "n_clusters": cl["G"],
                "slope": fmt(res["slope"]), "slope_se": fmt(cl["se"]),
                "ci95_lo": fmt(cl["ci_lo"]), "ci95_hi": fmt(cl["ci_hi"]),
                "t_crit": "" if math.isnan(cl["t_crit"]) else f"{cl['t_crit']:.4f}",
                "intercept": fmt(res["intercept"]), "r2": fmt(res["r2"], 4),
            })
            # (b) plain OLS SE, for reference.
            w.writerow({
                **identity,
                "se_type": "ols_iid", "n_points": res["n"], "n_clusters": "",
                "slope": fmt(res["slope"]), "slope_se": fmt(res["slope_se"]),
                "ci95_lo": fmt(res["ci95_lo"]), "ci95_hi": fmt(res["ci95_hi"]),
                "t_crit": "" if math.isnan(res["t_crit"]) else f"{res['t_crit']:.4f}",
                "intercept": fmt(res["intercept"]), "r2": fmt(res["r2"], 4),
            })
            # (c) tolerance-level seed means (one point per tolerance).
            tol_x, tol_y = [], []
            for t_val in sorted(set(tols.tolist())):
                m_ = tols == t_val
                tol_x.append(float(np.exp(np.mean(np.log(x[m_])))))
                tol_y.append(float(np.exp(np.mean(np.log(y[m_])))))
            res_t = ols_loglog(np.asarray(tol_x), np.asarray(tol_y))
            w.writerow({
                **identity,
                "se_type": "tolerance_mean", "n_points": res_t["n"], "n_clusters": "",
                "slope": fmt(res_t["slope"]), "slope_se": fmt(res_t["slope_se"]),
                "ci95_lo": fmt(res_t["ci95_lo"]), "ci95_hi": fmt(res_t["ci95_hi"]),
                "t_crit": "" if math.isnan(res_t["t_crit"]) else f"{res_t['t_crit']:.4f}",
                "intercept": fmt(res_t["intercept"]), "r2": fmt(res_t["r2"], 4),
            })
            print(f"[fit] group={g} metric={metric} margin={margin}: slope={res['slope']:.4f} "
                  f"(cluster se={cl['se']:.4f}, 95% CI [{cl['ci_lo']:.4f}, {cl['ci_hi']:.4f}], "
                  f"G={cl['G']} seeds, n={res['n']}, R^2={res['r2']:.4f})")

    # Mixed-margin sanity: a (group, metric) spanning several margins means
    # runs were evaluated on different primary windows -- flag it.
    _gm: Dict[Tuple[str, str], set] = defaultdict(set)
    for (g, metric, margin) in points:
        _gm[(g, metric)].add(margin)
    for (g, metric), ms in sorted(_gm.items()):
        if len(ms) > 1:
            print(f"[warn] (group={g}, metric={metric}) mixes primary margins {sorted(ms)}; "
                  f"fits are per-margin, compare within one margin only.")

    scatter_fit_figure_files: List[str] = []
    mean_sd_figure_files: List[str] = []
    if not args.no_plots:
        scatter_fit_figure_files = plot_e6(
            summary_path,
            points,
            group_meta,
            metrics,
            formats=formats,
            dpi=int(args.dpi),
        )
        mean_sd_figure_files = plot_e6_mean_sd(
            summary_path,
            points,
            group_meta,
            metrics,
            formats=formats,
            dpi=int(args.dpi),
        )
    figure_files = scatter_fit_figure_files + mean_sd_figure_files

    metadata_path = os.path.join(summary_dir, "e6_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 3,
            "residual_definition": (
                "max over outers of fixed-Q_res p_res measured on the official "
                "post-restore checkpoint state"
            ),
            "target_reached_definition": "achieved_pres <= pres_target",
            "primary_error_metric": "e_Xev",
            "secondary_error_metrics": [m for m in metrics if m != "e_Xev"],
            "metrics_requested": metrics,
            "legacy_residual_semantics_allowed": bool(
                args.allow_legacy_residual_semantics
            ),
            "fit_x_axis": "achieved_pres (never nominal pres_target)",
            "fit_primary_se": "seed-cluster-robust CR1",
            "target_summary": "geometric means of achieved_pres and error",
            "paper_summary_figure": (
                "per-target arithmetic mean +/- sample SD in both axes, "
                "with slope-1 reference anchored at geometric mean(error/achieved_pres)"
            ),
            "metric_provenance_contract": (
                "training e_Xev uses raw config/status primary margin and "
                "outer_history; final metrics.csv may use only the guarded "
                "successful eval-only overlay; mixed provenance within a "
                "(group, metric, margin) series is fatal"
            ),
            "configuration_pooling": (
                "none; each group hash fixes every training-relevant setting "
                "except seed and pres_target"
            ),
            "outer_iters_filter": sorted(selected_outer_iters),
            "sample_sd": "ddof=1; blank/NA when n=1",
            "figures": figure_files,
            "scatter_fit_figures": scatter_fit_figure_files,
            "mean_sd_figures": mean_sd_figure_files,
            "figure_formats": [] if args.no_plots else formats,
            "figure_dpi": None if args.no_plots else int(args.dpi),
            "settings": [group_meta[group] for group in sorted({key[0] for key in points})],
        }, f, indent=2, sort_keys=True)

    print(f"[e6] wrote: {settings_path}")
    print(f"[e6] wrote: {pts_path}")
    print(f"[e6] wrote: {per_path}")
    print(f"[e6] wrote: {fit_path}")
    for figure_file in figure_files:
        print(f"[e6] wrote: {os.path.join(summary_dir, figure_file)}")
    print(f"[e6] wrote: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
