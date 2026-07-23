"""Aggregate per-seed Liu ND PINN / PI-PINN runs into mean / std / 95% CI tables.

Usage:
    python3 aggregate_seeds.py --out-root <OUT_ROOT>
    python3 aggregate_seeds.py --out-root <OUT_ROOT> --include-stopped
    python3 aggregate_seeds.py --out-root <OUT_ROOT> --output <dir>

The script walks OUT_ROOT for run directories (anything containing a
config.json written by ExperimentRecorder plus a metrics.csv), groups runs
that share every hyperparameter EXCEPT the seed (and bookkeeping-only keys),
and writes, per (configuration, full-dimensional evaluation margin, metric):

    n, mean, std (ddof=1), sem, ci95_lo, ci95_hi, seeds

The 95% CI uses the Student-t critical value with df = n - 1 (scipy when
available, a built-in t-table fallback otherwise), matching the paper
protocol "mean +- std over seeds; 95% CIs in the supplementary material".

Outputs (under <OUT_ROOT>/seed_summary by default):
    runs_index.csv        every run found, with status and group hash
    groups.json           group hash -> shared configuration
    summary_long.csv      one row per (group, model_type, eval_margin, metric)
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
from datetime import datetime, timezone
import math
import os
import re
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
    "skip_figures",
    "skip_eval",
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
]

MARKET_HASH_KEYS = (
    "K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0", "lam0",
    "X_min", "X_max", "eta", "gamma", "r", "tau_max", "W_min", "W_max",
    "market_seed",
)


def parse_seed_spec(text: str) -> List[int]:
    """Parse comma/space-separated seeds and inclusive ranges such as 1-10."""
    out: List[int] = []
    for token in re.split(r"[\s,]+", str(text or "").strip()):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", token)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(token))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate seeds in --expected-seeds: {text!r}")
    return sorted(out)


def canonical_market_hash(path: str) -> str:
    """Hash the economic market snapshot, excluding the training seed.

    Key name, normalized dtype, shape and C-order bytes are all included so
    arrays with the same raw bytes but different meanings cannot collide.
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in MARKET_HASH_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"missing market keys {missing}")
        digest = hashlib.sha256()
        for key in MARKET_HASH_KEYS:
            arr = np.asarray(data[key])
            if arr.dtype.hasobject:
                raise ValueError(f"object dtype is not canonical: {key}")
            # Normalize byte order so the hash is machine-independent.
            dtype = arr.dtype
            if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
                arr = arr.byteswap().view(dtype.newbyteorder("<"))
            else:
                arr = arr.astype(dtype.newbyteorder("<"), copy=False)
            arr = np.ascontiguousarray(arr)
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(arr.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(arr.shape).encode("ascii") + b"\0")
            digest.update(arr.tobytes(order="C"))
        return digest.hexdigest()

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
        return datetime.fromtimestamp(
            os.path.getmtime(os.path.join(run_dir, "config.json")),
            tz=timezone.utc,
        ).isoformat()
    except Exception:
        return ""


def run_status(run_dir: str) -> str:
    present = [
        name for name in ("_SUCCESS", "_STOPPED_EARLY", "_FAILED")
        if os.path.exists(os.path.join(run_dir, name))
    ]
    if len(present) > 1:
        # Never let a stale success marker outrank a newer failure marker.
        return "conflicting_markers"
    if present == ["_SUCCESS"]:
        return "success"
    if present == ["_STOPPED_EARLY"]:
        return "stopped_early"
    if present == ["_FAILED"]:
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
                value = float(row["value"])
                if not math.isfinite(margin) or not math.isfinite(value):
                    continue
                rows.append({
                    "eval_margin": margin,
                    "metric": str(row["metric"]),
                    "value": value,
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


def as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-seed metrics into mean/std/95% CI tables.")
    ap.add_argument("--out-root", type=str, required=True, help="Sweep output root (the OUT_ROOT of tune_pipinn.sh).")
    ap.add_argument("--output", type=str, default=None, help="Summary directory (default: <out-root>/seed_summary).")
    ap.add_argument("--include-stopped", action="store_true",
                    help="Also include runs marked _STOPPED_EARLY (default: success only).")
    ap.add_argument("--min-runs", type=int, default=1, help="Minimum runs per group to report (default 1).")
    ap.add_argument(
        "--expected-seeds", type=str, default="",
        help="Exact successful training-seed set required per group, e.g. 1-10 or 1,2,3. "
             "When set, missing/extra seeds and missing per-seed metrics are fatal.",
    )
    ap.add_argument(
        "--strict-market-snapshots", action="store_true",
        help="Fail unless every selected run with the same M has one canonical market hash. "
             "Automatically enabled by --expected-seeds.",
    )
    ap.add_argument(
        "--headline-margin", type=float, default=0.10,
        help="Evaluation margin used in summary_headline.csv (default: 0.10).",
    )
    ap.add_argument(
        "--expected-m-states", type=str, default="",
        help="Optional exact dimensions across discovered groups, e.g. 1,3,5; "
             "enforced independently of --expected-models.",
    )
    ap.add_argument(
        "--expected-n-assets", type=int, default=None,
        help="Optional exact risky-asset dimension required in every discovered group "
             "(paper Table 3 uses 30).",
    )
    ap.add_argument(
        "--expected-models", type=str, default="",
        help="Optional exact methods across discovered groups, e.g. pinn,pipinn; "
             "enforced independently of --expected-m-states.",
    )
    args = ap.parse_args()

    expected_seeds = set(parse_seed_spec(args.expected_seeds))
    expected_m_states = set(parse_seed_spec(args.expected_m_states))
    expected_models = {x.strip() for x in args.expected_models.split(",") if x.strip()}
    strict_market = bool(args.strict_market_snapshots or expected_seeds)

    out_root = os.path.abspath(args.out_root)
    summary_dir = args.output or os.path.join(out_root, "seed_summary")
    os.makedirs(summary_dir, exist_ok=True)

    accepted_status = {"success"}
    if args.include_stopped:
        accepted_status.add("stopped_early")

    run_dirs = find_runs(out_root)
    if not run_dirs:
        message = f"no runs (config.json) found under {out_root}"
        if (expected_seeds or expected_m_states or expected_models
                or args.expected_n_assets is not None or strict_market):
            raise SystemExit(f"paper aggregation validation failed: {message}")
        print(f"[warn] {message}")
        return

    runs_index_rows = []
    groups_config: Dict[str, str] = {}
    # values[(ghash, model_type, eval_margin, metric)] ->
    #     list of (seed, value, run_updated_at)
    values: Dict[Tuple[str, str, float, str], List[Tuple[Any, float, str]]] = defaultdict(list)
    group_dims: Dict[str, Tuple[Any, Any]] = {}
    market_rows: List[Dict[str, Any]] = []
    validation_errors: List[str] = []

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
        group_dims.setdefault(ghash_train, (raw_cfg.get("n_assets"), raw_cfg.get("m_states")))
        model_type = str(cfg.get("model_type", ""))
        seed = cfg.get("seed")
        market_path = os.path.join(run_dir, "market_params.npz")
        market_hash = ""
        market_error = ""
        try:
            market_hash = canonical_market_hash(market_path)
        except Exception as exc:
            market_error = str(exc)

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
            # Set after newest-per-(training group, method, seed) selection.
            "used": 0,
            "market_hash": market_hash,
            "market_error": market_error,
        })

    # ---- success rates per (group, model_type), on UNIQUE SEEDS: the same
    #      seed rerun keeps only its NEWEST run's status, so a failed old
    #      attempt followed by a successful rerun counts as 1/1, not 1/2.
    #      Divergence-stop censoring is reported, never silently dropped ----
    _newest: Dict[Tuple[str, str, Any], Tuple[str, str]] = {}  # (ts, status)
    _newest_row: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    for row in runs_index_rows:
        k = (row["group_train"], row["model_type"], row["seed"])
        ts = row.get("updated_at", "")
        if k not in _newest or ts >= _newest[k][0]:
            _newest[k] = (ts, row["status"])
            _newest_row[k] = row
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

    # Metrics and market snapshots must come from the SAME newest selected
    # run.  Reading every historical success first and deduplicating each
    # metric key later can silently backfill a missing new metric from an old
    # rerun, which makes a broken Table-3 evaluation look complete.
    for row in _newest_row.values():
        if row["status"] not in accepted_status:
            continue
        row["used"] = 1
        run_dir = os.path.join(out_root, str(row["run_dir"]))
        market_rows.append({
            "run_dir": row["run_dir"],
            "model_type": row["model_type"],
            "n_assets": row["n_assets"],
            "m_states": row["m_states"],
            "seed": row["seed"],
            "status": row["status"],
            "market_hash": row["market_hash"],
            "market_error": row["market_error"],
        })
        metric_rows = load_metrics_rows(run_dir)
        if not metric_rows:
            print(f"[warn] no metrics.csv rows in selected run: {run_dir}")
            continue
        n_used += 1
        for metric_row in metric_rows:
            values[(
                row["group"], row["model_type"], metric_row["eval_margin"], metric_row["metric"]
            )].append((row["seed"], metric_row["value"], row["updated_at"]))

    if expected_seeds:
        for (g, model_type), _counts in sorted(_cnt.items()):
            successful = {
                int(seed) for (gg, mm, seed), (_ts, status) in _newest.items()
                if gg == g and mm == model_type and status == "success"
            }
            if successful != expected_seeds:
                missing = sorted(expected_seeds - successful)
                extra = sorted(successful - expected_seeds)
                validation_errors.append(
                    f"group={g} model={model_type}: successful seeds={sorted(successful)}, "
                    f"expected={sorted(expected_seeds)}, missing={missing}, extra={extra}"
                )

    observed_m_states = {
        as_int(group_dims.get(group, (None, None))[1])
        for group, _model_type in _cnt
    }
    observed_n_assets = {
        as_int(group_dims.get(group, (None, None))[0])
        for group, _model_type in _cnt
    }
    observed_models = {model_type for _group, model_type in _cnt}

    if args.expected_n_assets is not None:
        if args.expected_n_assets <= 0:
            validation_errors.append(
                f"--expected-n-assets must be positive, got {args.expected_n_assets}"
            )
        elif observed_n_assets != {args.expected_n_assets}:
            validation_errors.append(
                f"paper groups: observed N={sorted(observed_n_assets, key=str)}, "
                f"expected exactly={[args.expected_n_assets]}"
            )

    if expected_m_states and observed_m_states != expected_m_states:
        validation_errors.append(
            f"paper groups: observed M={sorted(observed_m_states, key=str)}, "
            f"expected exactly={sorted(expected_m_states)}"
        )

    if expected_models and observed_models != expected_models:
        validation_errors.append(
            f"paper groups: observed models={sorted(observed_models)}, "
            f"expected exactly={sorted(expected_models)}"
        )

    if expected_m_states and expected_models:
        expected_pairs = {
            (model_type, m_states)
            for model_type in expected_models
            for m_states in expected_m_states
        }
        observed_pairs = {
            (model_type, as_int(group_dims.get(group, (None, None))[1]))
            for group, model_type in _cnt
        }
        if observed_pairs != expected_pairs:
            validation_errors.append(
                f"paper groups: observed method/M={sorted(observed_pairs, key=str)}, "
                f"expected exactly={sorted(expected_pairs, key=str)}"
            )
        for m_states in sorted(expected_m_states):
            for model_type in sorted(expected_models):
                matching = [
                    g for (g, model) in _cnt
                    if model == model_type and as_int(group_dims.get(g, (None, None))[1]) == m_states
                ]
                if len(matching) != 1:
                    validation_errors.append(
                        f"expected exactly one group for model={model_type}, M={m_states}; "
                        f"found {matching}"
                    )

    if strict_market:
        # Check only the newest selected run for each (configuration, method,
        # seed); archived reruns must not override the same deduplication rule
        # used for metrics.  Group by M (not by N,M): the paper requires one
        # identical market snapshot across both methods and all seeds at a
        # given state dimension, and the hash itself also detects a changed N.
        market_by_m: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in market_rows:
            if row["status"] == "success":
                market_by_m[row.get("m_states")].append(row)
        for m_states, rows in sorted(market_by_m.items(), key=lambda item: str(item[0])):
            errors = [r for r in rows if r["market_error"] or not r["market_hash"]]
            hashes = {r["market_hash"] for r in rows if r["market_hash"]}
            if errors:
                validation_errors.append(
                    f"market M={m_states}: {len(errors)} selected run(s) have missing/invalid snapshots"
                )
            if len(hashes) != 1:
                validation_errors.append(
                    f"market M={m_states}: expected one canonical hash, found {sorted(hashes)}"
                )

    if expected_seeds:
        for (ghash, model_type, eval_margin, metric), pairs in sorted(values.items()):
            metric_seeds = {int(seed) for seed, _value, _ts in pairs}
            if metric_seeds != expected_seeds:
                validation_errors.append(
                    f"metric group={ghash} model={model_type} margin={eval_margin:g} "
                    f"metric={metric}: seeds={sorted(metric_seeds)}, expected={sorted(expected_seeds)}"
                )

        # Existing metric keys alone are not sufficient validation: if a
        # headline metric (or the entire 0.10 window) is absent everywhere,
        # there is no key to inspect above.  Require the complete Table-3
        # pair explicitly for each selected paper (method, M) group.
        if expected_m_states and expected_models:
            targets = [
                (model_type, m_states)
                for m_states in sorted(expected_m_states)
                for model_type in sorted(expected_models)
            ]
        else:
            targets = sorted({
                (model_type, as_int(group_dims.get(group, (None, None))[1]))
                for group, model_type in _cnt
            }, key=str)

        for model_type, m_states in targets:
            candidate_groups = {
                ghash for (ghash, model, _margin, _metric) in values
                if model == model_type
                and as_int(group_dims.get(ghash, (None, None))[1]) == m_states
            }
            if len(candidate_groups) != 1:
                validation_errors.append(
                    f"Table 3 model={model_type}, M={m_states}: expected exactly one metric group, "
                    f"found {sorted(candidate_groups)}"
                )
                continue
            ghash = next(iter(candidate_groups))
            for metric in HEADLINE_METRICS:
                matching_pairs = [
                    pairs for (group, model, margin, name), pairs in values.items()
                    if group == ghash and model == model_type and name == metric
                    and math.isclose(float(margin), float(args.headline_margin),
                                     rel_tol=0.0, abs_tol=1e-12)
                ]
                if len(matching_pairs) != 1:
                    validation_errors.append(
                        f"Table 3 group={ghash} model={model_type}, M={m_states}: "
                        f"missing/ambiguous metric={metric} at eval_margin={args.headline_margin:g}"
                    )
                    continue
                metric_seeds = {int(seed) for seed, _value, _ts in matching_pairs[0]}
                if metric_seeds != expected_seeds:
                    validation_errors.append(
                        f"Table 3 group={ghash} model={model_type}, M={m_states}, metric={metric}: "
                        f"seeds={sorted(metric_seeds)}, expected={sorted(expected_seeds)}"
                    )

    error_path = os.path.join(summary_dir, "validation_errors.txt")
    if validation_errors:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write("\n".join(validation_errors) + "\n")
        for msg in validation_errors[:25]:
            print(f"[validation error] {msg}")
        if len(validation_errors) > 25:
            print(f"[validation error] ... {len(validation_errors) - 25} more; see {error_path}")
        raise SystemExit(f"paper aggregation validation failed; see {error_path}")
    if os.path.exists(error_path):
        os.remove(error_path)
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
            "run_dir", "updated_at", "group_train", "group", "model_type", "n_assets", "m_states", "seed", "status", "used",
            "market_hash", "market_error"])
        wtr.writeheader()
        for row in runs_index_rows:
            wtr.writerow(row)

    market_path = os.path.join(summary_dir, "market_hashes.csv")
    with open(market_path, "w", encoding="utf-8", newline="") as f:
        fields = ["run_dir", "model_type", "n_assets", "m_states", "seed", "status",
                  "market_hash", "market_error"]
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for row in market_rows:
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
            "seeds": ";".join(str(s) for s in sorted(
                by_seed,
                key=lambda z: (as_int(z) is None, as_int(z) if as_int(z) is not None else str(z)),
            )),
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
            if not math.isclose(float(row["eval_margin"]), float(args.headline_margin),
                                rel_tol=0.0, abs_tol=1e-12):
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
    print(f"[aggregate] wrote: {market_path}")
    print(f"[aggregate] wrote: {long_path}")
    print(f"[aggregate] wrote: {head_path}")


if __name__ == "__main__":
    main()
