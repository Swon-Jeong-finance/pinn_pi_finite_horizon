"""Aggregate per-seed Liu ND PINN / PI-PINN runs into mean / std / 95% CI tables.

Usage:
    python3 aggregate_seeds.py --out-root <OUT_ROOT>
    python3 aggregate_seeds.py --out-root <OUT_ROOT> --include-stopped
    python3 aggregate_seeds.py --out-root <OUT_ROOT> --output <dir>

The script walks OUT_ROOT for run directories (anything containing a
config.json written by ExperimentRecorder plus a metrics.csv), groups runs
that share every hyperparameter EXCEPT the seed (and bookkeeping-only keys),
and writes, per (configuration, wealth level w, metric):

    n, mean, std (ddof=1), sem, ci95_lo, ci95_hi, seeds

The 95% CI uses the Student-t critical value with df = n - 1 (scipy when
available, a built-in t-table fallback otherwise), matching the paper
protocol "mean +- std over seeds; 95% CIs in the supplementary material".

Outputs (under <OUT_ROOT>/seed_summary by default):
    runs_index.csv        every run found, with status and group hash
    groups.json           group hash -> shared configuration
    summary_long.csv      one row per (group, model_type, w, metric)
    summary_headline.csv  compact table of headline metrics with
                          "mean +- std" and "[ci_lo, ci_hi]" strings

This file is standalone on purpose: it does not import torch or the
training scripts, so it can run on any machine that has the outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Keys that must NOT distinguish groups: the seed itself plus pure
# bookkeeping / logging / saving knobs that cannot change the learned model.
GROUP_IGNORE_KEYS = {
    "seed",
    "run_tag",
    "device",
    "output_root",
    "weight_root",
    "stop_flag_path",
    "eval_only",
    "skip_plots",
    "print_every",
    "print_every_outer",
    "print_every_eval",
    "verbose_detail",
    "save_iterate_every",
    "w_levels",
    "n_tau",
    "n_x",
    # eval_margin is evaluation-only and carried per metrics row instead.
    "eval_margin",
}

HEADLINE_METRICS = [
    "RelL2_V",
    "RelL2_theta",
    "RelL2_myopic",
    "RelL2_hedging",
    "StdNRMSE_V",
    "StdNRMSE_theta",
    "MSE_V",
    "MSE_theta",
]

# Two-sided 95% t critical values (df -> t_{0.975, df}); fallback when scipy
# is unavailable. df > 30 falls back to interpolation anchors / 1.96.
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t as _t  # type: ignore
        return float(_t.ppf(0.975, df))
    except Exception:
        if df in _T_TABLE:
            return _T_TABLE[df]
        keys = sorted(_T_TABLE)
        if df > keys[-1]:
            return 1.96
        lo = max(k for k in keys if k <= df)
        hi = min(k for k in keys if k >= df)
        if lo == hi:
            return _T_TABLE[lo]
        w = (df - lo) / (hi - lo)
        return _T_TABLE[lo] * (1 - w) + _T_TABLE[hi] * w


def _overlay_eval_config(run_dir: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """If a SUCCESSFUL eval-only rerun rewrote metrics.csv, its evaluation
    settings live in config_eval.json; overlay the eval-affecting keys so
    grouping and primary-margin selection match the metrics actually on disk.

    Guards: a failed/aborted eval-only leaves a config_eval.json that does
    NOT describe the metrics on disk, so the overlay requires
    (a) _SUCCESS_EVAL to exist,
    (b) status_eval.json to report success, and
    (c) metrics.csv to be at least as new as config_eval.json."""
    path = os.path.join(run_dir, "config_eval.json")
    if not os.path.exists(path):
        return args
    if not os.path.exists(os.path.join(run_dir, "_SUCCESS_EVAL")):
        return args
    try:
        with open(os.path.join(run_dir, "status_eval.json"), "r", encoding="utf-8") as f:
            if str(json.load(f).get("status", "")) != "success":
                return args
    except Exception:
        return args
    try:
        m_path = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(m_path) and os.path.getmtime(m_path) + 1e-6 < os.path.getmtime(path):
            return args  # metrics predate the eval config: not its output
    except OSError:
        pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            eval_args = json.load(f).get("args", {})
    except Exception:
        return args
    out = dict(args)
    for k in ("test_points", "eval_margin", "n_tau", "n_x", "w_levels"):
        if k in eval_args:
            out[k] = eval_args[k]
    return out


def run_updated_at(run_dir: str) -> str:
    """status.json updated_at (fallback: config.json mtime) for rerun dedup."""
    path = os.path.join(run_dir, "status.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            ts = str(json.load(f).get("updated_at", ""))
            if ts:
                return ts
    except Exception:
        pass
    try:
        return datetime.utcfromtimestamp(
            os.path.getmtime(os.path.join(run_dir, "config.json"))).isoformat()
    except Exception:
        return ""


def run_status(run_dir: str) -> str:
    if os.path.exists(os.path.join(run_dir, "_SUCCESS")):
        return "success"
    if os.path.exists(os.path.join(run_dir, "_STOPPED_EARLY")):
        return "stopped_early"
    if os.path.exists(os.path.join(run_dir, "_FAILED")):
        return "failed"
    return "unknown"


def find_runs(out_root: str) -> List[str]:
    runs = []
    for dirpath, dirnames, filenames in os.walk(out_root):
        if "config.json" in filenames:
            runs.append(dirpath)
            # Do not descend into a run directory looking for nested runs.
            dirnames[:] = [d for d in dirnames if d not in ("plots",)]
    return sorted(runs)


def load_config_args(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        args = data.get("args", {})
        if not isinstance(args, dict) or "model_type" not in args:
            return None
        return _overlay_eval_config(run_dir, args)
    except Exception as exc:
        print(f"[warn] could not read {path}: {exc}")
        return None


def load_config_args_raw(run_dir: str) -> Optional[Dict[str, Any]]:
    """Training config.json args WITHOUT any eval-only overlay (used for
    success-rate grouping so a partial eval-only re-evaluation cannot split
    successes and failures into different groups)."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            args = json.load(f).get("args", {})
        if not isinstance(args, dict) or "model_type" not in args:
            return None
        return args
    except Exception:
        return None


def group_key(args: Dict[str, Any]) -> Tuple[str, str]:
    """Return (hash, canonical json) of the seed-independent configuration."""
    core = {k: args[k] for k in sorted(args) if k not in GROUP_IGNORE_KEYS}
    canon = json.dumps(core, sort_keys=True, default=str)
    h = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    return h, canon


def load_metrics_rows(run_dir: str) -> List[Dict[str, Any]]:
    """Read FULL-DIMENSIONAL metric rows only.

    Slice metrics (legacy fixed-w (tau, x_0) grid rows, scope=slice or files
    without a scope column) are visualization-era output and are skipped;
    all reported numbers come from the fulldim Omega_ev test set.
    """
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    rows = []
    n_slice_skipped = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                scope = str(row.get("scope", "") or "slice")
                if scope != "fulldim":
                    n_slice_skipped += 1
                    continue
                margin_raw = row.get("eval_margin", "")
                margin = float(margin_raw) if str(margin_raw).strip() != "" else 0.0
                rows.append({
                    "eval_margin": margin,
                    "metric": str(row["metric"]),
                    "value": float(row["value"]),
                })
            except (KeyError, ValueError):
                continue
    if n_slice_skipped and not rows:
        print(f"[warn] {run_dir}: only legacy slice metrics found "
              f"({n_slice_skipped} rows skipped); re-run evaluation with --test-points > 0.")
    return rows


def fmt(x: float, digits: int = 6) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{digits}e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-seed metrics into mean/std/95% CI tables.")
    ap.add_argument("--out-root", type=str, required=True, help="Sweep output root (the OUT_ROOT of tune_pipinn.sh).")
    ap.add_argument("--output", type=str, default=None, help="Summary directory (default: <out-root>/seed_summary).")
    ap.add_argument("--include-stopped", action="store_true",
                    help="Also include runs marked _STOPPED_EARLY (default: success only).")
    ap.add_argument("--min-runs", type=int, default=1, help="Minimum runs per group to report (default 1).")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "seed_summary")
    os.makedirs(summary_dir, exist_ok=True)

    accepted_status = {"success"}
    if args.include_stopped:
        accepted_status.add("stopped_early")

    run_dirs = find_runs(out_root)
    if not run_dirs:
        print(f"[warn] no runs (config.json) found under {out_root}")
        return

    runs_index_rows = []
    groups_config: Dict[str, str] = {}
    # values[(ghash, model_type, w, metric)] -> list of (seed, value)
    values: Dict[Tuple[str, str, float, str], List[Tuple[Any, float]]] = defaultdict(list)
    group_dims: Dict[str, Tuple[Any, Any]] = {}

    n_used = 0
    for run_dir in run_dirs:
        cfg = load_config_args(run_dir)
        if cfg is None:
            continue
        status = run_status(run_dir)
        ghash, canon = group_key(cfg)
        groups_config.setdefault(ghash, canon)
        group_dims.setdefault(ghash, (cfg.get("n_assets"), cfg.get("m_states")))
        # Success rates group on the TRAINING config only: a partial
        # eval-only re-evaluation must not split successes and failures of
        # the same training configuration into different groups.
        raw_cfg = load_config_args_raw(run_dir) or cfg
        ghash_train, canon_train = group_key(raw_cfg)
        groups_config.setdefault(ghash_train, canon_train)
        model_type = str(cfg.get("model_type", ""))
        seed = cfg.get("seed")

        runs_index_rows.append({
            "run_dir": os.path.relpath(run_dir, out_root),
            "updated_at": run_updated_at(run_dir),
            "group_train": ghash_train,
            "group": ghash,
            "model_type": model_type,
            "n_assets": cfg.get("n_assets"),
            "m_states": cfg.get("m_states"),
            "seed": seed,
            "status": status,
            "used": int(status in accepted_status),
        })
        if status not in accepted_status:
            continue

        rows = load_metrics_rows(run_dir)
        if not rows:
            print(f"[warn] no metrics.csv rows in accepted run: {run_dir}")
            continue
        n_used += 1
        ts = run_updated_at(run_dir)
        for r in rows:
            values[(ghash, model_type, r["eval_margin"], r["metric"])].append((seed, r["value"], ts))

    # ---- success rates per (group, model_type), on UNIQUE SEEDS: the same
    #      seed rerun keeps only its NEWEST run's status, so a failed old
    #      attempt followed by a successful rerun counts as 1/1, not 1/2.
    #      Divergence-stop censoring is reported, never silently dropped ----
    _newest: Dict[Tuple[str, str, Any], Tuple[str, str]] = {}  # (ts, status)
    for row in runs_index_rows:
        k = (row["group_train"], row["model_type"], row["seed"])
        ts = row.get("updated_at", "")
        if k not in _newest or ts >= _newest[k][0]:
            _newest[k] = (ts, row["status"])
    _cnt: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (g, m, _seed), (_ts, st) in _newest.items():
        _cnt[(g, m)]["n_seeds"] += 1
        if st == "success":
            _cnt[(g, m)]["n_success"] += 1
        elif st == "stopped_early":
            _cnt[(g, m)]["n_stopped"] += 1
        elif st == "failed":
            _cnt[(g, m)]["n_failed"] += 1
        else:
            _cnt[(g, m)]["n_other"] += 1
    sr_path = os.path.join(summary_dir, "success_rates.csv")
    with open(sr_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["group", "model_type", "n_seeds", "n_success",
                                            "n_stopped", "n_failed", "n_other", "success_rate"])
        wtr.writeheader()
        for k in sorted(_cnt):
            c = _cnt[k]
            wtr.writerow({"group": k[0], "model_type": k[1],
                          "n_seeds": c["n_seeds"], "n_success": c["n_success"],
                          "n_stopped": c["n_stopped"], "n_failed": c["n_failed"],
                          "n_other": c["n_other"],
                          "success_rate": f"{c['n_success']/max(c['n_seeds'],1):.3f}"})
    print(f"[aggregate] wrote: {sr_path}")

    # ---- runs_index.csv ----
    idx_path = os.path.join(summary_dir, "runs_index.csv")
    with open(idx_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=[
            "run_dir", "updated_at", "group_train", "group", "model_type", "n_assets", "m_states", "seed", "status", "used"])
        wtr.writeheader()
        for row in runs_index_rows:
            wtr.writerow(row)

    # ---- groups.json ----
    with open(os.path.join(summary_dir, "groups.json"), "w", encoding="utf-8") as f:
        json.dump({h: json.loads(c) for h, c in groups_config.items()}, f, indent=2, sort_keys=True)

    # ---- summary_long.csv ----
    long_path = os.path.join(summary_dir, "summary_long.csv")
    long_fields = ["group", "model_type", "n_assets", "m_states", "eval_margin", "metric",
                   "n", "mean", "std", "sem", "ci95_lo", "ci95_hi", "t_crit", "seeds"]
    long_rows = []
    for (ghash, model_type, eval_margin, metric), pairs in sorted(values.items()):
        # Deduplicate seeds (e.g. a rerun overwriting the same seed): keep last.
        # Rerun dedup: keep the NEWEST run per seed (status updated_at), not
        # the last one in path order.
        by_seed: Dict[Any, float] = {}
        by_seed_ts: Dict[Any, str] = {}
        for seed, val, ts in pairs:
            if seed not in by_seed or ts >= by_seed_ts.get(seed, ""):
                by_seed[seed] = val
                by_seed_ts[seed] = ts
        vals = np.asarray(list(by_seed.values()), dtype=float)
        n = int(vals.size)
        if n < args.min_runs:
            continue
        mean = float(np.mean(vals))
        if n > 1:
            std = float(np.std(vals, ddof=1))
            sem = std / math.sqrt(n)
            tc = t_crit_95(n - 1)
            ci_lo, ci_hi = mean - tc * sem, mean + tc * sem
        else:
            std, sem, tc, ci_lo, ci_hi = 0.0, float("nan"), float("nan"), float("nan"), float("nan")
        na, ms = group_dims.get(ghash, ("", ""))
        long_rows.append({
            "group": ghash, "model_type": model_type, "n_assets": na, "m_states": ms,
            "eval_margin": eval_margin, "metric": metric, "n": n,
            "mean": fmt(mean), "std": fmt(std), "sem": fmt(sem),
            "ci95_lo": fmt(ci_lo), "ci95_hi": fmt(ci_hi),
            "t_crit": "" if math.isnan(tc) else f"{tc:.4f}",
            "seeds": ";".join(str(s) for s in sorted(by_seed, key=lambda z: (str(z)))),
        })
    with open(long_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=long_fields)
        wtr.writeheader()
        for row in long_rows:
            wtr.writerow(row)

    # ---- summary_headline.csv (paper-facing compact view) ----
    head_path = os.path.join(summary_dir, "summary_headline.csv")
    head_fields = ["group", "model_type", "n_assets", "m_states", "eval_margin", "metric",
                   "n", "mean_pm_std", "ci95"]
    with open(head_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=head_fields)
        wtr.writeheader()
        for row in long_rows:
            if row["metric"] not in HEADLINE_METRICS:
                continue
            mean_s, std_s = row["mean"], row["std"]
            pm = f"{mean_s} +- {std_s}" if std_s else mean_s
            ci = f"[{row['ci95_lo']}, {row['ci95_hi']}]" if row["ci95_lo"] else ""
            wtr.writerow({
                "group": row["group"], "model_type": row["model_type"],
                "n_assets": row["n_assets"], "m_states": row["m_states"],
                "eval_margin": row["eval_margin"],
                "metric": row["metric"], "n": row["n"],
                "mean_pm_std": pm, "ci95": ci,
            })

    n_groups = len({r["group"] for r in long_rows})
    print(f"[aggregate] runs found: {len(runs_index_rows)} | used: {n_used} | groups: {n_groups}")
    print(f"[aggregate] wrote: {idx_path}")
    print(f"[aggregate] wrote: {os.path.join(summary_dir, 'groups.json')}")
    print(f"[aggregate] wrote: {long_path}")
    print(f"[aggregate] wrote: {head_path}")


if __name__ == "__main__":
    main()
