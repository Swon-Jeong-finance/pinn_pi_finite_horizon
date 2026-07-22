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
    python3 aggregate_e6.py --out-root <OUT_ROOT> --metrics RelL2_V,RelL2_pi,RelL2_c
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
    GROUP_IGNORE_KEYS, canonical_market_hash, find_runs, load_config_args,
    parse_int_spec, parse_seed_spec, run_status, t_crit_95, fmt,
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
    ap.add_argument("--metrics", type=str, default="RelL2_V,RelL2_pi,RelL2_c",
                    help="Comma-separated metric names to fit against achieved p_res.")
    ap.add_argument("--include-stopped", action="store_true")
    ap.add_argument("--expected-seeds", type=str, default="",
                    help="Exact seed set required at every residual target, e.g. 1-10.")
    ap.add_argument("--expected-n-assets", type=str, default="",
                    help="Exact asset dimensions required, e.g. 10,50.")
    ap.add_argument("--strict-market-snapshots", action="store_true",
                    help="Require one canonical Merton market per N. Enabled by --expected-seeds.")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "e6_summary")
    os.makedirs(summary_dir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    accepted = {"success"} | ({"stopped_early"} if args.include_stopped else set())
    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    expected_n_assets = set(parse_int_spec(args.expected_n_assets, label="--expected-n-assets"))
    strict_market = bool(args.strict_market_snapshots or expected_seeds)
    validation_errors: List[str] = []
    market_rows: List[Dict[str, Any]] = []

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
        reached_raw = status.get("target_reached", None)
        if isinstance(reached_raw, bool):
            reached = reached_raw
        elif isinstance(reached_raw, (int, float)) and reached_raw in (0, 1):
            reached = bool(reached_raw)
        else:
            # Legacy runs did not record the sticky training-time crossing
            # fact; only for those runs infer it from the achieved residual.
            reached = bool(achieved <= target * (1.0 + 1e-9))

        g = e6_group_key(cfg)
        ts = str(status.get("updated_at", ""))
        market_hash = ""
        market_error = ""
        try:
            market_hash = canonical_market_hash(os.path.join(run_dir, "market_params.npz"))
        except Exception as exc:
            market_error = str(exc)
        market_rows.append({
            "group": g, "run_dir": os.path.relpath(run_dir, out_root),
            "seed": cfg.get("seed"), "pres_target": target,
            "n_assets": cfg.get("n_assets"), "market_hash": market_hash,
            "market_error": market_error, "updated_at": ts,
        })
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
                "n_assets": cfg.get("n_assets"),
                "pres_target": target,
                "achieved_pres": float(achieved),
                "target_reached": int(reached),
                "total_inner_steps": steps if steps is not None else "",
                "error": float(val),
                "eval_margin": margin_used,
            })

    if not points:
        message = "no E6 runs (pres_target set) found"
        if expected_seeds or expected_n_assets or strict_market:
            raise SystemExit(f"E6 aggregation validation failed: {message}")
        print(f"[warn] {message}.")
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

    # ---- points.csv ----
    pts_path = os.path.join(summary_dir, "points.csv")
    pts_fields = ["group", "metric", "run_dir", "seed", "n_assets", "pres_target", "achieved_pres",
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
