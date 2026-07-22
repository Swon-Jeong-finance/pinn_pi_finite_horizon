#!/usr/bin/env python3
"""Validate and classify the fractional-myopic Liu contraction pilot.

The intended pilot is the complete Cartesian grid

    theta_0 = a * theta_myopic,  a in {0.5, 1.5},  one or more seeds.

Before looking at trajectories, this script verifies that every run completed,
that outer_history.csv contains exactly outer_iter=1,...,outer_iters, that the
same training protocol and market snapshot were used, and that clipping was
disabled.  It then checks, on the recorded finite diagnostic sets:

  C1  empirical nondegeneracy:
          min_n lam_min_sigma_frozen(n) > ellipticity_floor;

  C2  useful dynamic range in the manuscript X_ev proxy:
          e_Xev(1) / tail_floor >= range_threshold;

  C2b actual derivative-bundle decay:
          e_bundle_sup(1) / tail_floor >= decay_threshold.

Tail floors use the final 10% with at least three points.  Decisions must be
unchanged when 10%, 20%, and 25% tails are used; threshold-straddling cases are
MIXED rather than forced into PASS or FLAT.

Per-run verdicts are diagnostic only.  The global verdict is conservative:

  PASS        every seed x scale run satisfies C1, C2, and C2b;
  NORM-PROXY  every run is nondegenerate, the absolute X_ev proxy is flat,
              and both fixed-reference RelL2 trajectories decay;
  DEGENERATE  at least one run violates C1;
  FLAT        every run is nondegenerate and all fixed-reference errors are flat;
  MIXED       valid pilot, but the evidence is inconsistent across criteria/runs;
  INVALID     incomplete, stale, non-comparable, or malformed pilot artifacts.

The eigenvalue result is empirical evidence on the sampled Q_col points, not a
proof of uniform ellipticity over the continuous domain.  val_pres is printed
only as policy-evaluation health; it is never treated as contraction evidence.

Exit status: 0=PASS, 1=valid non-PASS scientific outcome, 2=INVALID artifacts.
Torch-free; requires only NumPy plus config/status/CSV/NPZ run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MANDATORY_SERIES = (
    "lam_min_sigma_frozen",
    "lam_max_sigma_frozen",
    "e_V_sup",
    "e_bundle_sup",
    "e_Xev",
    "e_Vw_sup",
    "e_Vww_sup",
    "e_Vwx_sup",
    "diag_RelL2_V",
    "diag_RelL2_theta",
    "val_pres",
    "guard_frac_ev",
)

# Keys that must agree across every seed/scale run.  seed, theta_init_scale,
# paths, device, and run_tag are intentionally excluded.
PROTOCOL_KEYS = (
    "model_type",
    "n_assets",
    "m_states",
    "market_seed",
    "tau_max",
    "w_min",
    "w_max",
    "gamma",
    "r",
    "x_range_scale",
    "dirichlet_concentration",
    "alpha_scale",
    "value_hidden",
    "value_depth",
    "batch_size",
    "terminal_frac",
    "lr",
    "w_terminal",
    "w_shape",
    "w_rra",
    "eval_epochs",
    "outer_iters",
    "theta_init_method",
    "theta_clip_abs",
    "lr_schedule",
    "adam_reset",
    "scheduler_patience",
    "scheduler_factor",
    "scheduler_min_lr",
    "carry_lr_min",
    "carry_lr_max",
    "pe_resample_every",
    "inner_best_restore",
    "sel_points",
    "sel_terminal_points",
    "sel_every",
    "sel_patience",
    "pres_target",
    "val_points",
    "val_terminal_points",
    "val_every",
    "diag_points",
    "diag_every",
    "eval_margin",
    "timing_mode",
)


class InvalidPilot(RuntimeError):
    """Raised when artifacts cannot support a scientific pilot verdict."""


@dataclass
class Run:
    directory: str
    args: Dict[str, Any]
    rows: List[Dict[str, str]]
    seed: int
    scale: float
    market_hash: str
    protocol: Tuple[Tuple[str, str], ...]


@dataclass
class Span:
    first: float
    floor: float
    ratio: float
    sensitivity_ratios: Tuple[float, ...]
    decision: Optional[bool]
    tail_points: int
    tail_spread: float
    tail_log_slope: float


def parse_int_list(raw: str) -> Optional[List[int]]:
    if not raw.strip():
        return None
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise InvalidPilot(f"invalid --seeds value: {raw!r}") from exc
    if not values:
        raise InvalidPilot("the pilot requires at least one expected seed")
    if len(set(values)) != len(values):
        raise InvalidPilot("--seeds contains duplicates")
    return values


def parse_float_list(raw: str) -> List[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise InvalidPilot(f"invalid --expected-scales value: {raw!r}") from exc
    if len(values) != 2 or len(set(values)) != 2:
        raise InvalidPilot("--expected-scales must contain two distinct values")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise InvalidPilot("expected scales must be finite and > 0")
    return sorted(values)


def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidPilot(f"cannot read valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidPilot(f"JSON root must be an object: {path}")
    return value


def clip_is_disabled(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == "none")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def protocol_signature(args: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple((key, canonical_json(args.get(key, "<missing>"))) for key in PROTOCOL_KEYS)


def market_snapshot_hash(path: str) -> str:
    if not os.path.isfile(path):
        raise InvalidPilot(f"market snapshot missing: {path}")
    digest = hashlib.sha256()
    try:
        with np.load(path, allow_pickle=False) as snapshot:
            keys = sorted(key for key in snapshot.files if key != "seed")
            if not keys:
                raise InvalidPilot(f"empty market snapshot: {path}")
            for key in keys:
                arr = np.ascontiguousarray(np.asarray(snapshot[key]))
                digest.update(key.encode("utf-8"))
                digest.update(str(arr.dtype).encode("ascii"))
                digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
                digest.update(arr.tobytes())
    except (OSError, ValueError) as exc:
        raise InvalidPilot(f"cannot read market snapshot {path}: {exc}") from exc
    return digest.hexdigest()


def scale_index(value: float, expected: Sequence[float]) -> Optional[int]:
    for index, target in enumerate(expected):
        if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12):
            return index
    return None


def read_outer_history(path: str, expected_outer: int) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise InvalidPilot(f"cannot read {path}: {exc}") from exc
    if len(rows) != expected_outer:
        raise InvalidPilot(
            f"{path}: expected {expected_outer} outer rows, found {len(rows)}"
        )
    try:
        indices = [int(row.get("outer_iter", "")) for row in rows]
    except ValueError as exc:
        raise InvalidPilot(f"{path}: non-integer outer_iter") from exc
    wanted = list(range(1, expected_outer + 1))
    if indices != wanted:
        raise InvalidPilot(
            f"{path}: outer_iter must be exactly 1..{expected_outer} in order; got {indices}"
        )
    return rows


def require_success(run_dir: str) -> None:
    bad = [name for name in ("_FAILED", "_STOPPED_EARLY")
           if os.path.exists(os.path.join(run_dir, name))]
    if bad:
        raise InvalidPilot(f"{run_dir}: terminal failure marker(s): {', '.join(bad)}")
    if not os.path.isfile(os.path.join(run_dir, "_SUCCESS")):
        raise InvalidPilot(f"{run_dir}: _SUCCESS is missing")
    status = load_json(os.path.join(run_dir, "status.json"))
    if status.get("status") != "success":
        raise InvalidPilot(f"{run_dir}: status.json is not success")


def strict_series(rows: Sequence[Dict[str, str]], name: str,
                  *, nonnegative: bool = True) -> np.ndarray:
    values: List[float] = []
    for offset, row in enumerate(rows, start=1):
        raw = row.get(name, "")
        if raw is None or not str(raw).strip():
            raise InvalidPilot(f"outer {offset}: mandatory column {name!r} is blank")
        try:
            value = float(raw)
        except ValueError as exc:
            raise InvalidPilot(f"outer {offset}: invalid {name}={raw!r}") from exc
        if not math.isfinite(value):
            raise InvalidPilot(f"outer {offset}: non-finite {name}={raw!r}")
        if nonnegative and value < 0.0:
            raise InvalidPilot(f"outer {offset}: negative {name}={value}")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def validate_series_consistency(series: Dict[str, np.ndarray]) -> None:
    e_xev = series["e_Xev"]
    expected = series["e_V_sup"] + series["e_bundle_sup"]
    if not np.allclose(e_xev, expected, rtol=1e-8, atol=1e-10):
        raise InvalidPilot("e_Xev is inconsistent with e_V_sup + e_bundle_sup")
    for name in ("e_Vw_sup", "e_Vww_sup", "e_Vwx_sup"):
        if np.any(series[name] > series["e_bundle_sup"] + 1e-8 * (1.0 + series[name])):
            raise InvalidPilot(f"{name} exceeds the recorded joint derivative-bundle sup")
    if np.any(series["lam_max_sigma_frozen"] < series["lam_min_sigma_frozen"]):
        raise InvalidPilot("lam_max_sigma_frozen is smaller than lam_min_sigma_frozen")
    for name in ("guard_frac_ev",):
        if np.any((series[name] < 0.0) | (series[name] > 1.0)):
            raise InvalidPilot(f"{name} must lie in [0,1]")


def discover_runs(out_root: str, m_states: int, requested_seeds: Optional[List[int]],
                  expected_scales: Sequence[float]) -> Tuple[List[Run], List[int]]:
    candidates: List[Tuple[str, Dict[str, Any], int, float]] = []
    pattern = os.path.join(os.path.abspath(out_root), "**", "config.json")
    for config_path in sorted(glob.glob(pattern, recursive=True)):
        payload = load_json(config_path)
        args = payload.get("args", {})
        if not isinstance(args, dict):
            continue
        if args.get("model_type") != "pipinn":
            continue
        try:
            if int(args.get("m_states", -1)) != m_states:
                continue
            seed = int(args["seed"])
            scale = float(args["theta_init_scale"])
        except (KeyError, TypeError, ValueError):
            continue
        if requested_seeds is not None and seed not in requested_seeds:
            continue
        if scale_index(scale, expected_scales) is None:
            continue
        candidates.append((os.path.dirname(config_path), args, seed, scale))

    if not candidates:
        raise InvalidPilot(f"no matching scale-pilot runs found below {out_root}")
    seeds = requested_seeds or sorted({item[2] for item in candidates})
    if not seeds:
        raise InvalidPilot("the pilot requires at least one seed")

    grid: Dict[Tuple[int, int], Run] = {}
    for run_dir, args, seed, scale in candidates:
        index = scale_index(scale, expected_scales)
        assert index is not None
        key = (seed, index)
        if key in grid:
            raise InvalidPilot(
                f"duplicate run for seed={seed}, scale={expected_scales[index]}: "
                f"{grid[key].directory} and {run_dir}"
            )
        require_success(run_dir)
        if args.get("theta_init_method") != "myopic":
            raise InvalidPilot(f"{run_dir}: theta_init_method must be myopic")
        if not clip_is_disabled(args.get("theta_clip_abs")):
            raise InvalidPilot(
                f"{run_dir}: clipping is active, so theta_0 is not exactly a*theta_myopic"
            )
        if int(args.get("diag_every", -1)) != 1:
            raise InvalidPilot(f"{run_dir}: diag_every must be 1 for an all-outer gate")
        if int(args.get("diag_points", 0)) <= 0 or int(args.get("val_points", 0)) <= 0:
            raise InvalidPilot(f"{run_dir}: positive diag_points and val_points are required")
        if bool(args.get("timing_mode", False)):
            raise InvalidPilot(f"{run_dir}: timing_mode disables required diagnostics")
        try:
            expected_outer = int(args["outer_iters"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPilot(f"{run_dir}: invalid outer_iters") from exc
        rows = read_outer_history(os.path.join(run_dir, "outer_history.csv"), expected_outer)
        run = Run(
            directory=run_dir,
            args=args,
            rows=rows,
            seed=seed,
            scale=float(expected_scales[index]),
            market_hash=market_snapshot_hash(os.path.join(run_dir, "market_params.npz")),
            protocol=protocol_signature(args),
        )
        grid[key] = run

    missing = [
        (seed, scale)
        for seed in seeds
        for scale in expected_scales
        if (seed, scale_index(scale, expected_scales)) not in grid
    ]
    if missing:
        formatted = ", ".join(f"seed={seed}/scale={scale:g}" for seed, scale in missing)
        raise InvalidPilot(f"incomplete seed x scale pilot grid: missing {formatted}")

    runs = [grid[(seed, index)] for seed in seeds for index in range(len(expected_scales))]
    market_hashes = {run.market_hash for run in runs}
    if len(market_hashes) != 1:
        raise InvalidPilot("market_params.npz differs across seed/scale runs")
    protocols = {run.protocol for run in runs}
    if len(protocols) != 1:
        baseline = dict(runs[0].protocol)
        differences: List[str] = []
        for run in runs[1:]:
            current = dict(run.protocol)
            changed = [key for key in PROTOCOL_KEYS if baseline[key] != current[key]]
            if changed:
                differences.append(
                    f"seed={run.seed}/scale={run.scale:g}: {','.join(changed)}"
                )
        raise InvalidPilot("training protocol mismatch: " + "; ".join(differences))
    return runs, seeds


def floor_ratio(first: float, floor: float) -> float:
    if floor > 0.0:
        return first / floor
    return math.inf if first > 0.0 else 1.0


def tail_size(length: int, fraction: float, minimum: int) -> int:
    return min(length - 1, max(minimum, int(math.ceil(fraction * length))))


def span(series: np.ndarray, threshold: float, tail_fraction: float,
         min_tail_points: int) -> Span:
    if len(series) <= min_tail_points:
        raise InvalidPilot(
            f"need more than {min_tail_points} outer points; found {len(series)}"
        )
    if np.any(series < 0.0) or not np.all(np.isfinite(series)):
        raise InvalidPilot("trajectory contains negative or non-finite values")

    fractions = tuple(sorted(set((tail_fraction, 0.10, 0.20, 0.25))))
    ratios: List[float] = []
    floors: Dict[float, float] = {}
    for fraction in fractions:
        count = tail_size(len(series), fraction, min_tail_points)
        value = float(np.median(series[-count:]))
        floors[fraction] = value
        ratios.append(floor_ratio(float(series[0]), value))
    primary_count = tail_size(len(series), tail_fraction, min_tail_points)
    primary_floor = floors[tail_fraction]
    tail = series[-primary_count:]
    if primary_floor == 0.0:
        spread = 0.0 if np.max(tail) == 0.0 else math.inf
    else:
        spread = float((np.percentile(tail, 90) - np.percentile(tail, 10)) / primary_floor)
    if np.all(tail > 0.0) and len(tail) >= 2:
        slope = float(np.polyfit(np.arange(len(tail)), np.log(tail), 1)[0])
    else:
        slope = 0.0

    if all(ratio >= threshold for ratio in ratios):
        decision: Optional[bool] = True
    elif all(ratio < threshold for ratio in ratios):
        decision = False
    else:
        decision = None
    return Span(
        first=float(series[0]),
        floor=primary_floor,
        ratio=floor_ratio(float(series[0]), primary_floor),
        sensitivity_ratios=tuple(ratios),
        decision=decision,
        tail_points=primary_count,
        tail_spread=spread,
        tail_log_slope=slope,
    )


def decision_text(value: Optional[bool]) -> str:
    if value is True:
        return "OK"
    if value is False:
        return "FAIL"
    return "MIXED(tail-sensitive)"


def print_span(label: str, item: Span, threshold: float) -> None:
    sens_lo = min(item.sensitivity_ratios)
    sens_hi = max(item.sensitivity_ratios)
    print(
        f"  {label:<20}: first={item.first:.3e} floor={item.floor:.3e} "
        f"ratio={item.ratio:.2f}x [tail sensitivity {sens_lo:.2f},{sens_hi:.2f}] "
        f"need>={threshold:g} -> {decision_text(item.decision)}"
    )
    print(
        f"  {'tail diagnostic':<20}: points={item.tail_points} "
        f"relative_spread={item.tail_spread:.2e} log_slope={item.tail_log_slope:.2e}"
    )


def classify_run(run: Run, range_threshold: float, decay_threshold: float,
                 ellipticity_floor: float, tail_fraction: float,
                 min_tail_points: int) -> Tuple[str, Dict[str, Any]]:
    series = {
        name: strict_series(
            run.rows,
            name,
            nonnegative=(name not in ("lam_min_sigma_frozen", "lam_max_sigma_frozen")),
        )
        for name in MANDATORY_SERIES
    }
    validate_series_consistency(series)

    lam_min = float(np.min(series["lam_min_sigma_frozen"]))
    positive_lmax = np.maximum(series["lam_max_sigma_frozen"], np.finfo(float).tiny)
    condition_ratio = float(np.min(series["lam_min_sigma_frozen"] / positive_lmax))
    c1 = lam_min > ellipticity_floor

    x_span = span(series["e_Xev"], range_threshold, tail_fraction, min_tail_points)
    bundle_span = span(
        series["e_bundle_sup"], decay_threshold, tail_fraction, min_tail_points
    )
    rel_v_span = span(
        series["diag_RelL2_V"], decay_threshold, tail_fraction, min_tail_points
    )
    rel_theta_span = span(
        series["diag_RelL2_theta"], decay_threshold, tail_fraction, min_tail_points
    )
    pres_span = span(series["val_pres"], decay_threshold, tail_fraction, min_tail_points)

    print("=" * 100)
    print(
        f"seed={run.seed} M={run.args.get('m_states')} init=myopic "
        f"scale={run.scale:g} outers={len(run.rows)} [{run.directory}]"
    )
    print(
        f"  C1 sampled covariance: min lambda={lam_min:.4e}, "
        f"min(lambda_min/lambda_max)={condition_ratio:.4e}, "
        f"floor>{ellipticity_floor:.1e} -> {'OK' if c1 else 'FAIL'}"
    )
    print_span("C2 e_Xev range", x_span, range_threshold)
    print_span("C2b bundle decay", bundle_span, decay_threshold)

    print(f"  {'component':>12} {'first':>11} {'tail floor':>11} {'ratio':>10}")
    for name in ("e_V_sup", "e_Vw_sup", "e_Vww_sup", "e_Vwx_sup"):
        item = span(series[name], decay_threshold, tail_fraction, min_tail_points)
        print(f"  {name:>12} {item.first:>11.3e} {item.floor:>11.3e} {item.ratio:>9.2f}x")

    print_span("ref RelL2_V", rel_v_span, decay_threshold)
    print_span("ref RelL2_theta", rel_theta_span, decay_threshold)
    print_span("health val_pres", pres_span, decay_threshold)
    guard_max = float(np.max(series["guard_frac_ev"]))
    if guard_max > 0.0:
        print(
            f"  [warn] V_ww guard active on up to {guard_max:.2%} of Q_ev; "
            "RelL2_theta describes the guarded numerical map."
        )

    ref_both_decay = rel_v_span.decision is True and rel_theta_span.decision is True
    all_fixed_flat = (
        x_span.decision is False
        and bundle_span.decision is False
        and rel_v_span.decision is False
        and rel_theta_span.decision is False
    )
    if not c1:
        verdict = "DEGENERATE"
    elif x_span.decision is True and bundle_span.decision is True:
        verdict = "PASS"
    elif x_span.decision is False and bundle_span.decision is False and ref_both_decay:
        verdict = "NORM-PROXY-CANDIDATE"
    elif all_fixed_flat:
        verdict = "FLAT"
    else:
        verdict = "MIXED"
    print(f"  RUN VERDICT: {verdict}")
    return verdict, {
        "c1": c1,
        "x_decision": x_span.decision,
        "bundle_decision": bundle_span.decision,
        "rel_v_decision": rel_v_span.decision,
        "rel_theta_decision": rel_theta_span.decision,
        "guard_max": guard_max,
    }


def global_verdict(verdicts: Iterable[str]) -> str:
    values = list(verdicts)
    if any(value == "DEGENERATE" for value in values):
        return "DEGENERATE"
    if values and all(value == "PASS" for value in values):
        return "PASS"
    if values and all(value == "NORM-PROXY-CANDIDATE" for value in values):
        return "NORM-PROXY"
    if values and all(value == "FLAT" for value in values):
        return "FLAT"
    return "MIXED"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--m-states", type=int, default=3)
    parser.add_argument(
        "--seeds", default="",
        help="Expected comma-separated seeds (any positive count); blank infers from matching runs.",
    )
    parser.add_argument("--expected-scales", default="0.5,1.5")
    parser.add_argument("--range-threshold", type=float, default=5.0)
    parser.add_argument("--decay-threshold", type=float, default=2.0)
    parser.add_argument(
        "--ellipticity-floor", type=float, default=1e-10,
        help="Numerical floor for the sampled minimum covariance eigenvalue.",
    )
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--min-tail-points", type=int, default=3)
    options = parser.parse_args(argv)

    try:
        if not (0.0 < options.tail_fraction < 1.0):
            raise InvalidPilot("--tail-fraction must lie in (0,1)")
        if options.min_tail_points < 2:
            raise InvalidPilot("--min-tail-points must be >= 2")
        for name in ("range_threshold", "decay_threshold", "ellipticity_floor"):
            value = float(getattr(options, name))
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidPilot(f"--{name.replace('_', '-')} must be finite and > 0")
        seeds = parse_int_list(options.seeds)
        scales = parse_float_list(options.expected_scales)
        runs, resolved_seeds = discover_runs(
            options.out_root, options.m_states, seeds, scales
        )
        print(
            f"[integrity] complete grid: seeds={resolved_seeds}, scales={scales}; "
            f"runs={len(runs)}, common market={runs[0].market_hash[:12]}..., protocol=matched"
        )
        per_run: List[str] = []
        for run in runs:
            verdict, _details = classify_run(
                run,
                options.range_threshold,
                options.decay_threshold,
                options.ellipticity_floor,
                options.tail_fraction,
                options.min_tail_points,
            )
            per_run.append(verdict)
        verdict = global_verdict(per_run)
    except InvalidPilot as exc:
        print(f"[INVALID] {exc}", file=sys.stderr)
        return 2

    print("=" * 100)
    if verdict == "PASS":
        message = "all pilot runs pass; extending this diagnostic to 10 seeds is supported"
    elif verdict == "NORM-PROXY":
        message = (
            "both scales/seeds consistently show fixed-reference RelL2 decay but no "
            "absolute X_ev range; revisit derivative normalization before more scale tuning"
        )
    elif verdict == "DEGENERATE":
        message = "at least one scale loses sampled nondegeneracy; move that scale toward 1"
    elif verdict == "FLAT":
        message = "all fixed-reference trajectories are flat; inspect training health"
    else:
        message = "criteria or runs disagree; do not extend or redefine the norm yet"
    print(f"GLOBAL VERDICT: {verdict} -- {message}")
    print(
        "[scope] This is a finite-set pilot gate, not a continuum "
        "ellipticity proof or a 10-seed statistical conclusion."
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
