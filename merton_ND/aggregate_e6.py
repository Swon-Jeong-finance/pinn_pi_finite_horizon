"""E6 aggregation: residual-tolerance sweep -> error-floor scaling.

Collects, per run: the pres target, the ACHIEVED residual level
p_res = max_n p_res,n (status.json `pres_max`, with an outer_history.csv
fallback), target-reached status, total inner optimizer steps, and the final
error metrics from metrics.csv. Then, per configuration group (everything
shared except training seed and pres_target):

  per_target.csv   n_runs, n_target_reached, achieved p_res mean+-std,
                   error mean+-std, total inner steps mean+-std, per target
  points.csv       one row per run (x = achieved p_res, y = error)
  fit.csv          log10(error) ~ log10(achieved p_res) OLS slope with
                   standard error, 95% CI (t, df=n-2), intercept, R^2

Per the protocol: runs that did NOT reach their target enter the fit with
their ACHIEVED residual (never the nominal target); reached runs also use
the achieved value, which is <= target by construction, so the x-axis is
uniformly "achieved p_res".

Standalone: stdlib + numpy (+ scipy if available for exact t quantiles).

Usage:
    python3 aggregate_e6.py --out-root <OUT_ROOT>
    python3 aggregate_e6.py --out-root <OUT_ROOT> --metrics RelRMSE_V,RelRMSE_theta
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aggregate_seeds import (  # reuse shared helpers
    GROUP_IGNORE_KEYS, find_runs, load_config_args, run_status, t_crit_95, fmt,
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
    return float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0, n


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the E6 residual-tolerance sweep.")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--output", type=str, default=None, help="Default: <out-root>/e6_summary")
    ap.add_argument("--model-type", type=str, default="pipinn", help="Which model to aggregate (E6 is PI-PINN).")
    ap.add_argument("--metrics", type=str, default="RelL2_V,RelL2_theta",
                    help="Comma-separated metric names to fit against achieved p_res.")
    ap.add_argument("--include-stopped", action="store_true")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "e6_summary")
    os.makedirs(summary_dir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    accepted = {"success"} | ({"stopped_early"} if args.include_stopped else set())

    # points[(group, metric)] -> list of per-run dicts
    points: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for run_dir in find_runs(out_root):
        cfg = load_config_args(run_dir)
        if cfg is None or str(cfg.get("model_type", "")) != args.model_type:
            continue
        if run_status(run_dir) not in accepted:
            continue
        target_raw = cfg.get("pres_target", None)
        try:
            target = float(target_raw) if target_raw is not None and str(target_raw) != "" else None
        except (TypeError, ValueError):
            target = None
        if target is None:
            continue  # not an E6 run

        status = read_status(run_dir)
        achieved = status.get("pres_max", None)
        if not isinstance(achieved, (int, float)):
            achieved = achieved_pres_from_outer_history(run_dir)
        if achieved is None or not (achieved > 0):
            print(f"[warn] no achieved p_res for {run_dir}; skipped")
            continue
        steps = status.get("total_inner_steps", None)
        if not isinstance(steps, (int, float)):
            steps = total_steps_from_outer_history(run_dir)
        reached = bool(achieved <= target * (1.0 + 1e-9))

        g = e6_group_key(cfg)
        ts = str(status.get("updated_at", ""))
        for metric in metrics:
            picked = pick_metric_value(run_dir, metric)
            if picked is None:
                print(f"[warn] fulldim metric {metric} missing in {run_dir}")
                continue
            val, margin_used = picked
            points[(g, metric, margin_used)].append({
                "_ts": ts,
                "run_dir": os.path.relpath(run_dir, out_root),
                "seed": cfg.get("seed"),
                "pres_target": target,
                "achieved_pres": float(achieved),
                "target_reached": int(reached),
                "total_inner_steps": steps if steps is not None else "",
                "error": float(val),
                "eval_margin": margin_used,
            })

    if not points:
        print("[warn] no E6 runs (pres_target set) found.")
        return

    # Rerun dedup UPSTREAM: keep the newest run per (pres_target, seed) so
    # points.csv, per_target.csv and fit.csv all see the SAME data.
    for k in list(points.keys()):
        best: Dict[Tuple[float, Any], Dict[str, Any]] = {}
        for r in points[k]:
            kk = (r["pres_target"], r["seed"])
            if kk not in best or r["_ts"] >= best[kk]["_ts"]:
                best[kk] = r
        points[k] = list(best.values())

    # ---- points.csv ----
    pts_path = os.path.join(summary_dir, "points.csv")
    pts_fields = ["group", "metric", "run_dir", "seed", "pres_target", "achieved_pres",
                  "target_reached", "total_inner_steps", "error", "eval_margin"]
    with open(pts_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pts_fields)
        w.writeheader()
        for (g, metric, margin), rows in sorted(points.items()):
            for r in rows:
                w.writerow({"group": g, "metric": metric,
                            **{k: v for k, v in r.items() if k != "_ts"}})

    # ---- per_target.csv ----
    per_path = os.path.join(summary_dir, "per_target.csv")
    per_fields = ["group", "metric", "eval_margin", "pres_target", "n_runs", "n_target_reached",
                  "achieved_pres_mean", "achieved_pres_std",
                  "error_mean", "error_std",
                  "total_inner_steps_mean", "total_inner_steps_std", "seeds"]
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
                w.writerow({
                    "group": g, "metric": metric, "eval_margin": margin, "pres_target": t_val,
                    "n_runs": n, "n_target_reached": sum(r["target_reached"] for r in rs),
                    "achieved_pres_mean": fmt(am), "achieved_pres_std": fmt(asd),
                    "error_mean": fmt(em), "error_std": fmt(esd),
                    "total_inner_steps_mean": fmt(sm), "total_inner_steps_std": fmt(ssd),
                    "seeds": ";".join(str(r["seed"]) for r in rs),
                })

    # ---- fit.csv ----
    fit_path = os.path.join(summary_dir, "fit.csv")
    fit_fields = ["group", "metric", "eval_margin", "se_type", "n_points", "n_clusters",
                  "slope", "slope_se", "ci95_lo", "ci95_hi", "t_crit", "intercept", "r2"]
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
            w.writerow({
                "group": g, "metric": metric, "eval_margin": margin,
                "se_type": "cluster_seed", "n_points": res["n"], "n_clusters": cl["G"],
                "slope": fmt(res["slope"]), "slope_se": fmt(cl["se"]),
                "ci95_lo": fmt(cl["ci_lo"]), "ci95_hi": fmt(cl["ci_hi"]),
                "t_crit": "" if math.isnan(cl["t_crit"]) else f"{cl['t_crit']:.4f}",
                "intercept": fmt(res["intercept"]), "r2": fmt(res["r2"], 4),
            })
            # (b) plain OLS SE, for reference.
            w.writerow({
                "group": g, "metric": metric, "eval_margin": margin,
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
                "group": g, "metric": metric, "eval_margin": margin,
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

    print(f"[e6] wrote: {pts_path}")
    print(f"[e6] wrote: {per_path}")
    print(f"[e6] wrote: {fit_path}")


if __name__ == "__main__":
    main()
