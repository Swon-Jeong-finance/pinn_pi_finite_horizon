#!/usr/bin/env python3
"""Reconstruct Liu Figure-2 normalized-policy errors from saved PI iterates.

This is a read-only bridge for runs completed before ``outer_history.csv``
recorded ``diag_RelL2_vartheta``.  It selects the newest successful affine
PI-PINN attempt with the same semantics as E9, verifies config/market/closed-
form/final-checkpoint provenance, recreates the trainers' deterministic
diagnostic design, and evaluates every saved outer checkpoint.

No source artifact is modified.  The resulting long CSV can be supplied to
``postprocess_contraction.py --vartheta-trajectory-csv ...``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from evaluate_margin_bundle import (
        RunRecord,
        analytic_reference,
        build_value_network,
        canonical_array_hash,
        discover_runs,
        ensure_separate_output,
        evaluate_model_bundle,
        import_torch,
        map_base_design,
        parse_int_list,
        relative_l2,
        sha256_file,
        torch_base_design,
        utc_now,
        validate_run_provenance,
        write_csv_atomic,
        write_json_atomic,
    )
except ModuleNotFoundError:  # package-style import from the repository root
    from .evaluate_margin_bundle import (
        RunRecord,
        analytic_reference,
        build_value_network,
        canonical_array_hash,
        discover_runs,
        ensure_separate_output,
        evaluate_model_bundle,
        import_torch,
        map_base_design,
        parse_int_list,
        relative_l2,
        sha256_file,
        torch_base_design,
        utc_now,
        validate_run_provenance,
        write_csv_atomic,
        write_json_atomic,
    )


POLICY_METRIC = "diag_RelL2_vartheta"
LEGACY_RAW_POLICY_METRIC = "diag_RelL2_theta"
PER_OUTER_FILE = "figure2_vartheta_per_outer.csv"
PROVENANCE_FILE = "figure2_vartheta_provenance.json"
SUCCESS_MARKER = "_SUCCESS_FIGURE2_VARTTHETA"
OUTPUT_FILES = {PER_OUTER_FILE, PROVENANCE_FILE, SUCCESS_MARKER}


def primary_eval_margin(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        margins = [float(item) for item in value]
    else:
        margins = [
            float(token.strip()) for token in str(value).split(",") if token.strip()
        ]
    if not margins:
        raise ValueError("training config has no primary eval_margin")
    margin = margins[0]
    if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
        raise ValueError(f"invalid primary eval_margin={margin}")
    return margin


def convergence_group_key(training_group: str, primary_margin: float) -> str:
    """Match ``postprocess_contraction.convergence_group_key`` exactly."""

    payload = json.dumps(
        {
            "training_group": str(training_group),
            "primary_eval_margin": float(primary_margin),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def read_outer_grid(
    path: Path,
) -> Tuple[List[int], Dict[int, float], Dict[int, float]]:
    """Return the complete outer grid plus value/raw-policy legacy diagnostics."""

    outers: List[int] = []
    raw_values: Dict[int, float] = {}
    value_errors: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"outer_iter", "diag_RelL2_V", LEGACY_RAW_POLICY_METRIC}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for row in reader:
            try:
                outer = int(row["outer_iter"])
                raw_value = float(row[LEGACY_RAW_POLICY_METRIC])
                value_error = float(row["diag_RelL2_V"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid policy row {row}") from exc
            if (
                outer < 1
                or not math.isfinite(raw_value)
                or raw_value < 0.0
                or not math.isfinite(value_error)
                or value_error < 0.0
            ):
                raise ValueError(f"{path}: invalid diagnostic metric at outer={outer}")
            if outer in raw_values:
                raise ValueError(f"{path}: duplicate outer iteration {outer}")
            outers.append(outer)
            raw_values[outer] = raw_value
            value_errors[outer] = value_error
    ordered = sorted(outers)
    if ordered != list(range(1, len(ordered) + 1)):
        raise ValueError(f"{path}: outer grid is not exactly 1..K: {ordered}")
    return ordered, raw_values, value_errors


def resolve_iterate_checkpoints(record: RunRecord, outers: Sequence[int]) -> Dict[int, Path]:
    if record.checkpoint is None:
        raise ValueError(f"{record.run_dir}: final-checkpoint provenance was not resolved")
    iterate_dir = record.checkpoint.parent / "iterates"
    checkpoints = {
        int(outer): iterate_dir / f"value_net_iter{int(outer):04d}.pt"
        for outer in outers
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{record.run_dir}: normalized-control reconstruction needs every outer "
            "checkpoint; missing:\n  " + "\n  ".join(missing)
        )
    expected_names = {path.name for path in checkpoints.values()}
    actual_names = {path.name for path in iterate_dir.glob("value_net_iter*.pt")}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError(
            f"{record.run_dir}: iterate directory contains checkpoints outside the "
            f"completed 1..{len(outers)} grid: {unexpected}"
        )
    return checkpoints


def load_checkpoint_state(checkpoint: Path, *, torch: Any, map_location: Any) -> Dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    except TypeError:  # legacy torch
        state = torch.load(checkpoint, map_location=map_location)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint}: checkpoint is not a state dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def canonical_state_hash(state: Mapping[str, Any]) -> str:
    """Hash checkpoint tensors by sorted key, dtype, shape, and raw values."""

    return canonical_array_hash(
        (str(key), tensor.detach().cpu().numpy())
        for key, tensor in sorted(state.items(), key=lambda item: str(item[0]))
    )


def verify_final_outer_state(
    final_checkpoint: Path,
    final_outer_checkpoint: Path,
    *,
    torch: Any,
) -> str:
    final_state = load_checkpoint_state(
        final_checkpoint, torch=torch, map_location=torch.device("cpu")
    )
    last_state = load_checkpoint_state(
        final_outer_checkpoint, torch=torch, map_location=torch.device("cpu")
    )
    final_state_hash = canonical_state_hash(final_state)
    last_state_hash = canonical_state_hash(last_state)
    if final_state_hash != last_state_hash:
        raise ValueError(
            "official final checkpoint does not equal the final outer iterate"
        )
    return final_state_hash


def load_iterate_model(
    record: RunRecord,
    checkpoint: Path,
    *,
    torch: Any,
    nn: Any,
    device: Any,
) -> Any:
    missing_architecture = [
        key for key in ("value_hidden", "value_depth")
        if key not in record.config_args or record.config_args[key] is None
    ]
    if missing_architecture:
        raise ValueError(
            f"{record.run_dir}: config is missing network architecture keys "
            f"{missing_architecture}"
        )
    hidden = int(record.config_args["value_hidden"])
    depth = int(record.config_args["value_depth"])
    model = build_value_network(torch, nn, record.m_states, hidden, depth).to(device)
    state = load_checkpoint_state(checkpoint, torch=torch, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_record(
    record: RunRecord,
    market: Any,
    closed_form: Any,
    *,
    torch: Any,
    nn: Any,
    device: Any,
    chunk_size: int,
    crosscheck_rtol: float,
    crosscheck_atol: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = record.config_args
    diag_points = int(config.get("diag_points", 0) or 0)
    diag_every = int(config.get("diag_every", 0) or 0)
    if diag_points < 1:
        raise ValueError(f"{record.run_dir}: diag_points must be positive")
    if diag_every != 1:
        raise ValueError(
            f"{record.run_dir}: diag_every={diag_every}; a complete Figure-2 trajectory "
            "requires diag_every=1"
        )
    margin = primary_eval_margin(config.get("eval_margin", ""))
    outer_path = record.run_dir / "outer_history.csv"
    outers, legacy_raw, legacy_value = read_outer_grid(outer_path)
    configured_outer = int(config.get("outer_iters", 0) or 0)
    if outers != list(range(1, configured_outer + 1)):
        raise ValueError(
            f"{record.run_dir}: completed outer grid {outers} disagrees with "
            f"configured outer_iters={configured_outer}"
        )
    checkpoints = resolve_iterate_checkpoints(record, outers)
    try:
        final_state_hash = verify_final_outer_state(
            record.checkpoint, checkpoints[outers[-1]], torch=torch
        )
    except ValueError as exc:
        raise ValueError(f"{record.run_dir}: {exc}") from exc
    last_state_hash = final_state_hash

    diag_seed = int(market.market_seed) * 1_000_003 + 7
    unit = torch_base_design(torch, diag_points, record.m_states, diag_seed)
    tau, wealth, state = map_base_design(
        unit,
        margin=margin,
        w_min=market.W_min,
        w_max=market.W_max,
        x_min=market.X_min,
        x_max=market.X_max,
        tau_max=market.tau_max,
        tau_epsilon=1.0e-3,
    )
    reference = analytic_reference(
        tau,
        wealth,
        state,
        closed_form=closed_form,
        gamma=market.gamma,
        r=market.r,
        lam0=market.lam0,
        Lam=market.Lam,
        Gamma=market.Gamma,
    )
    design_hash = canonical_array_hash((("unit_points", unit),))
    config_hash = sha256_file(record.run_dir / "config.json")
    outer_hash = sha256_file(outer_path)
    figure_group = convergence_group_key(record.group, margin)
    rows: List[Dict[str, Any]] = []
    crosscheck_failures: List[str] = []

    for outer in outers:
        checkpoint = checkpoints[outer].resolve()
        checkpoint_hash = sha256_file(checkpoint)
        model = load_iterate_model(
            record, checkpoint, torch=torch, nn=nn, device=device
        )
        estimate = evaluate_model_bundle(
            model,
            tau=tau,
            wealth=wealth,
            state=state,
            market=market,
            torch=torch,
            device=device,
            chunk_size=chunk_size,
        )
        vartheta_error = relative_l2(estimate["vartheta"], reference.vartheta)
        # Reproduce the legacy diagnostic as an integrity check.  It is a
        # wealth-squared-weighted vartheta error, not the paper policy metric.
        raw_error = relative_l2(
            estimate["vartheta"] * wealth[:, None],
            reference.vartheta * wealth[:, None],
        )
        raw_status = (
            "pass"
            if math.isclose(
                raw_error,
                legacy_raw[outer],
                rel_tol=crosscheck_rtol,
                abs_tol=crosscheck_atol,
            )
            else "mismatch"
        )
        value_error = relative_l2(estimate["value"], reference.value)
        value_status = (
            "pass"
            if math.isclose(
                value_error,
                legacy_value[outer],
                rel_tol=crosscheck_rtol,
                abs_tol=crosscheck_atol,
            )
            else "mismatch"
        )
        if raw_status != "pass":
            crosscheck_failures.append(
                f"outer={outer} raw-policy: saved={legacy_raw[outer]:.9e}, "
                f"recomputed={raw_error:.9e}"
            )
        if value_status != "pass":
            crosscheck_failures.append(
                f"outer={outer} value: saved={legacy_value[outer]:.9e}, "
                f"recomputed={value_error:.9e}"
            )
        rows.append({
            "group": figure_group,
            "training_group": record.group,
            "model_type": record.model_type,
            "n_assets": record.n_assets,
            "m_states": record.m_states,
            "training_seed": record.seed,
            "outer_iter": outer,
            "metric": POLICY_METRIC,
            "value": vartheta_error,
            "legacy_raw_value": legacy_raw[outer],
            "legacy_raw_recomputed": raw_error,
            "legacy_raw_crosscheck_status": raw_status,
            "legacy_value": legacy_value[outer],
            "legacy_value_recomputed": value_error,
            "legacy_value_crosscheck_status": value_status,
            "diag_points": diag_points,
            "diag_margin": margin,
            "diag_seed": diag_seed,
            "design_hash": design_hash,
            "run_dir": str(record.run_dir),
            "config_sha256": config_hash,
            "outer_history_sha256": outer_hash,
            "market_params": str(record.run_dir / "market_params.npz"),
            "market_params_file_sha256": record.market_file_sha256,
            "market_hash": record.market_hash,
            "closed_form": str(record.run_dir / "closed_form_ode.npz"),
            "closed_form_file_sha256": record.closed_form_file_sha256,
            "closed_form_hash": record.closed_form_hash,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if crosscheck_failures:
        raise ValueError(
            f"{record.run_dir}: iterate/design integrity cross-check failed:\n  "
            + "\n  ".join(crosscheck_failures)
        )
    return rows, {
        "run_dir": str(record.run_dir),
        "group": figure_group,
        "training_group": record.group,
        "training_seed": record.seed,
        "config_sha256": config_hash,
        "outer_history_sha256": outer_hash,
        "market_params_file_sha256": record.market_file_sha256,
        "canonical_market_hash": record.market_hash,
        "closed_form_file_sha256": record.closed_form_file_sha256,
        "canonical_closed_form_hash": record.closed_form_hash,
        "official_final_checkpoint": str(record.checkpoint),
        "official_final_checkpoint_sha256": record.checkpoint_sha256,
        "official_final_state_hash": final_state_hash,
        "final_outer_state_hash": last_state_hash,
        "outer_iterations": outers,
        "diag_points": diag_points,
        "diag_margin": margin,
        "diag_seed": diag_seed,
        "design_hash": design_hash,
    }


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output exists and is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILES if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"derived normalized-policy output already exists: {sorted(existing)}; "
            "pass --overwrite"
        )
    blocked = [
        name for name in OUTPUT_FILES
        if (output / name).exists() and not (output / name).is_file()
    ]
    if blocked:
        raise ValueError(f"reserved output paths are not regular files: {blocked}")


def commit_staged_output(stage: Path, output: Path) -> None:
    """Install the complete derived set and restore old files on any failure."""

    backup = Path(tempfile.mkdtemp(
        prefix=".figure2_vartheta_backup_", dir=str(output.parent)
    ))
    moved: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        missing = sorted(name for name in OUTPUT_FILES if not (stage / name).is_file())
        if missing:
            raise RuntimeError(f"staged normalized-policy output is incomplete: {missing}")
        for name in sorted(OUTPUT_FILES):
            destination = output / name
            if destination.exists():
                saved = backup / name
                os.replace(destination, saved)
                moved.append((saved, destination))
        for name in sorted(OUTPUT_FILES):
            destination = output / name
            os.replace(stage / name, destination)
            installed.append(destination)
    except Exception:
        for destination in reversed(installed):
            if destination.exists():
                destination.unlink()
        for saved, destination in reversed(moved):
            if saved.exists():
                os.replace(saved, destination)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct per-outer Liu vartheta=theta/w errors from saved PI-PINN "
            "iterate checkpoints without modifying training runs."
        )
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Default: OUT_ROOT/derived/figure2_vartheta_trajectory",
    )
    parser.add_argument("--n-assets", default="")
    parser.add_argument("--m-states", default="3")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--expected-seeds", default="")
    parser.add_argument("--strict-seed-set", action="store_true")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--run-name-regex", default="")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--crosscheck-rtol", type=float, default=5.0e-5)
    parser.add_argument("--crosscheck-atol", type=float, default=5.0e-7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_seeds < 1 or args.chunk_size < 1:
        raise SystemExit("--min-seeds and --chunk-size must be positive")
    if args.crosscheck_rtol < 0.0 or args.crosscheck_atol < 0.0:
        raise SystemExit("cross-check tolerances must be nonnegative")
    out_root = Path(args.out_root).expanduser().resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out-root is not a directory: {out_root}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else out_root / "derived" / "figure2_vartheta_trajectory"
    )
    stage: Optional[Path] = None
    try:
        if args.seeds and args.expected_seeds:
            selected_seeds = parse_int_list(args.seeds)
            expected_seeds = parse_int_list(args.expected_seeds)
            if selected_seeds != expected_seeds:
                raise ValueError(
                    "--seeds and --expected-seeds cannot select different sets"
                )
            seeds = selected_seeds
        else:
            seeds = parse_int_list(args.expected_seeds or args.seeds)
        strict_seed_set = bool(args.strict_seed_set or args.expected_seeds)
        selected = discover_runs(
            out_root,
            models=["pipinn"],
            n_assets=parse_int_list(args.n_assets),
            m_states=parse_int_list(args.m_states),
            seeds=seeds,
            run_name_regex=args.run_name_regex,
            strict_seed_set=strict_seed_set,
            min_seeds=args.min_seeds,
        )
        selected, markets, closed_forms = validate_run_provenance(
            selected, out_root
        )
        records = [record for values in selected.values() for record in values]
        ensure_separate_output(output, records)
        torch, nn, device = import_torch(args.device, args.torch_num_threads)

        rows: List[Dict[str, Any]] = []
        provenance_runs: List[Dict[str, Any]] = []
        for cell, cell_records in sorted(selected.items()):
            _ = cell
            for record in cell_records:
                run_rows, run_provenance = evaluate_record(
                    record,
                    markets[record.run_dir],
                    closed_forms[record.run_dir],
                    torch=torch,
                    nn=nn,
                    device=device,
                    chunk_size=args.chunk_size,
                    crosscheck_rtol=args.crosscheck_rtol,
                    crosscheck_atol=args.crosscheck_atol,
                )
                rows.extend(run_rows)
                provenance_runs.append(run_provenance)

        # Source runs are untouched until every checkpoint and legacy metric
        # has passed.  Only then stage and atomically install derived artifacts.
        prepare_output(output, args.overwrite)
        stage = Path(tempfile.mkdtemp(
            prefix=".figure2_vartheta_stage_", dir=str(output.parent)
        ))
        fields = (
            "group", "training_group", "model_type", "n_assets", "m_states",
            "training_seed",
            "outer_iter", "metric", "value", "legacy_raw_value",
            "legacy_raw_recomputed", "legacy_raw_crosscheck_status",
            "legacy_value", "legacy_value_recomputed",
            "legacy_value_crosscheck_status", "diag_points",
            "diag_margin", "diag_seed", "design_hash", "run_dir",
            "config_sha256", "outer_history_sha256", "market_params",
            "market_params_file_sha256", "market_hash", "closed_form",
            "closed_form_file_sha256", "closed_form_hash", "checkpoint",
            "checkpoint_sha256",
        )
        csv_path = stage / PER_OUTER_FILE
        write_csv_atomic(csv_path, rows, fields)
        csv_hash = sha256_file(csv_path)
        write_json_atomic(stage / PROVENANCE_FILE, {
            "schema_version": 1,
            "created_at": utc_now(),
            "host": socket.gethostname(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_out_root": str(out_root),
            "output": str(output),
            "source_runs_mutated": False,
            "selection": {
                "newest_attempt_before_status_filter": True,
                "completed_success_only": True,
                "affine_pi_pinn_only": True,
                "requested_seeds": seeds,
                "strict_seed_set": strict_seed_set,
                "min_seeds": args.min_seeds,
            },
            "metric": {
                "name": POLICY_METRIC,
                "definition": "||theta_hat/w-theta_star/w||_2 / ||theta_star/w||_2",
                "legacy_raw_crosscheck": LEGACY_RAW_POLICY_METRIC,
                "vww_guard": 1.0e-8,
                "crosscheck_rtol": args.crosscheck_rtol,
                "crosscheck_atol": args.crosscheck_atol,
            },
            "evaluation": {
                "chunk_size": args.chunk_size,
                "device": str(device),
                "tau_epsilon": 1.0e-3,
                "diagnostic_seed_rule": "market_seed*1000003+7",
            },
            "runs": provenance_runs,
            "artifact_sha256": {PER_OUTER_FILE: csv_hash},
        })
        (stage / SUCCESS_MARKER).touch()
        commit_staged_output(stage, output)
    except (
        ValueError, OSError, FileExistsError, RuntimeError, FloatingPointError
    ) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)

    print(
        f"[done] reconstructed {len(rows)} normalized-policy point(s) from "
        f"{len(records)} PI-PINN run(s); wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
