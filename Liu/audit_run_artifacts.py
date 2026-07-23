#!/usr/bin/env python3
"""Read-only provenance audit for completed Liu experiment runs.

The training directories are treated as immutable observations.  This tool
reads their configuration, completion status, market snapshot, outer-history
rows, and checkpoints, then writes one ``posthoc_observed.json`` manifest to a
*separate* output directory.  It never creates, removes, or rewrites anything
inside a run or weight directory.

Checkpoint equality is based on a canonical hash of the state-dict tensors,
not on the serialization bytes.  Consequently ``final``, ``last``, and the
last scheduled iterate may be compared even when ``torch.save`` produced
different container bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from joint_market_setup_dirichlet import validate_market_snapshot


SCHEMA_VERSION = 1
MANIFEST_NAME = "posthoc_observed.json"
ITERATE_RE = re.compile(r"^value_net_iter(\d+)\.pt$")
COMPLETION_MARKERS = ("_SUCCESS", "_STOPPED_EARLY", "_FAILED")
MARKET_HASH_KEYS = (
    "K", "xbar", "SigmaX", "rho", "Lam", "Q", "Gamma", "k0", "lam0",
    "X_min", "X_max", "eta", "gamma", "r", "tau_max", "W_min", "W_max",
    "market_seed",
)


class AuditError(RuntimeError):
    """Raised for an invalid or incomplete observed artifact set."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_array_bytes(value: Any) -> Tuple[str, Tuple[int, ...], bytes]:
    """Return canonical dtype, shape and bytes for a tensor-like value.

    NumPy arrays are supported directly so the hashing contract can be tested
    without PyTorch.  Real checkpoints pass torch tensors, which are detached,
    moved to CPU and converted to a contiguous NumPy representation.  Object,
    string and structured dtypes are rejected rather than silently hashing a
    Python representation.
    """

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if getattr(tensor, "layout", None) is not None:
            layout = str(tensor.layout)
            if layout not in {"torch.strided", "strided"}:
                raise TypeError(f"non-strided tensor is not canonical: {layout}")
        try:
            value = tensor.contiguous().numpy()
        except Exception as exc:  # notably bfloat16 on older NumPy builds
            try:
                raw = tensor.contiguous().view(np.uint8).numpy()
            except Exception:
                try:
                    import torch  # type: ignore

                    raw = tensor.contiguous().view(torch.uint8).numpy()
                except Exception as raw_exc:  # pragma: no cover - exotic dtype
                    raise TypeError(
                        f"cannot canonicalize tensor dtype={getattr(tensor, 'dtype', None)}"
                    ) from raw_exc
            return str(getattr(tensor, "dtype", "unknown")), tuple(tensor.shape), bytes(raw)

    arr = np.asarray(value)
    if arr.dtype.hasobject or arr.dtype.fields is not None or arr.dtype.kind in {"U", "S", "V"}:
        raise TypeError(f"unsupported canonical tensor dtype: {arr.dtype}")
    dtype = arr.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        arr = arr.byteswap().view(dtype.newbyteorder("<"))
    else:
        arr = arr.astype(dtype.newbyteorder("<"), copy=False)
    arr = np.ascontiguousarray(arr)
    return arr.dtype.str, tuple(int(x) for x in arr.shape), arr.tobytes(order="C")


def canonical_tensor_hash(state_dict: Mapping[str, Any]) -> str:
    """Hash state-dict keys, dtypes, shapes and tensor values canonically."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("checkpoint state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for key in sorted(state_dict, key=str):
        if not isinstance(key, str):
            raise TypeError(f"state_dict key is not a string: {key!r}")
        dtype, shape, payload = _canonical_array_bytes(state_dict[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(dtype.encode("ascii", errors="strict") + b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for wrapper in ("state_dict", "model_state_dict", "value_net_state_dict", "model"):
            candidate = payload.get(wrapper)
            if isinstance(candidate, Mapping) and candidate:
                return candidate
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise TypeError("checkpoint does not contain a non-empty state_dict")


def load_checkpoint_state_dict(path: Path) -> Mapping[str, Any]:
    """Safely load a weights-only PyTorch checkpoint on CPU."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on audit host
        raise AuditError(
            "PyTorch is required to hash checkpoint tensors; install it on the audit host"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch predating weights_only
        payload = torch.load(path, map_location="cpu")
    return _extract_state_dict(payload)


def checkpoint_observation(
    path: Path,
    state_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_state_dict,
) -> Dict[str, Any]:
    state = state_loader(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
        "file_sha256": sha256_file(path),
        "tensor_sha256": canonical_tensor_hash(state),
        "tensor_count": int(len(state)),
    }


def canonical_npz_hash(path: Path, keys: Sequence[str] = MARKET_HASH_KEYS) -> str:
    """Hash named NPZ arrays with normalized byte order and explicit metadata."""

    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise AuditError(f"{path}: missing market keys {missing}")
        for key in keys:
            dtype, shape, payload = _canonical_array_bytes(np.asarray(data[key]))
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(dtype.encode("ascii") + b"\0")
            digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(payload)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def parse_int(value: Any, label: str) -> int:
    try:
        number = float(str(value).strip())
    except Exception as exc:
        raise AuditError(f"{label} is not an integer: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise AuditError(f"{label} is not an integer: {value!r}")
    return int(number)


def read_outer_history(path: Path) -> Tuple[List[str], List[Dict[str, str]], List[int]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except Exception as exc:
        raise AuditError(f"cannot read outer history {path}: {exc}") from exc
    if "outer_iter" not in fields:
        raise AuditError(f"{path}: missing outer_iter column")
    if len(fields) != len(set(fields)):
        raise AuditError(f"{path}: duplicate CSV header names")
    if not rows:
        raise AuditError(f"{path}: no data rows")
    outer = [parse_int(row.get("outer_iter"), f"{path}: outer_iter row {i}") for i, row in enumerate(rows, 1)]
    if len(set(outer)) != len(outer):
        raise AuditError(f"{path}: duplicate outer_iter values")
    if outer != sorted(outer):
        raise AuditError(f"{path}: outer_iter values are not increasing")
    expected = list(range(1, outer[-1] + 1))
    if outer != expected:
        raise AuditError(f"{path}: outer_iter must be contiguous 1..{outer[-1]}, got {outer}")
    return fields, rows, outer


def completion_status(run_dir: Path, status: Mapping[str, Any]) -> str:
    markers = [name for name in COMPLETION_MARKERS if (run_dir / name).exists()]
    if len(markers) != 1:
        raise AuditError(
            f"{run_dir}: expected exactly one completion marker from {COMPLETION_MARKERS}, got {markers}"
        )
    marker_status = {
        "_SUCCESS": "success",
        "_STOPPED_EARLY": "stopped_early",
        "_FAILED": "failed",
    }[markers[0]]
    json_status = str(status.get("status", "")).strip()
    if json_status != marker_status:
        raise AuditError(
            f"{run_dir}: marker says {marker_status!r}, status.json says {json_status!r}"
        )
    if marker_status != "success":
        raise AuditError(f"{run_dir}: run is not successfully completed ({marker_status})")
    return marker_status


def resolve_weight_dir(
    run_dir: Path,
    config: Mapping[str, Any],
    weight_root: Optional[Path] = None,
) -> Path:
    args = config.get("args", {})
    if not isinstance(args, Mapping):
        args = {}
    run_tag = str(args.get("run_tag", run_dir.name))
    model_type = str(args.get("model_type", ""))
    model_folder = "pi-pinn" if model_type == "pipinn" else "pinn"

    candidates: List[Path] = []
    declared = config.get("weight_dir") or args.get("weight_root")
    if declared:
        candidates.append(Path(str(declared)).expanduser())
    if weight_root is not None:
        candidates.extend((weight_root / model_folder / run_tag, weight_root / run_tag, weight_root))
    # Portable fallback for the launcher's OUT_ROOT/{method,weights/method}/tag layout.
    if len(run_dir.parents) >= 2:
        experiment_root = run_dir.parent.parent
        candidates.append(experiment_root / "weights" / model_folder / run_tag)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    raise AuditError(
        f"{run_dir}: cannot locate weight directory; checked {[str(x) for x in candidates]}"
    )


def expected_iterate_indices(args: Mapping[str, Any], completed_outer: int) -> List[int]:
    model_type = str(args.get("model_type", ""))
    timing_mode = bool(args.get("timing_mode", False))
    if timing_mode:
        return []
    if model_type == "pipinn" and bool(args.get("e3b_checkpoints", False)):
        return list(range(1, completed_outer + 1))
    every = parse_int(args.get("save_iterate_every", 0), "save_iterate_every")
    if every < 0:
        raise AuditError(f"save_iterate_every must be non-negative, got {every}")
    if every == 0:
        return []
    return [index for index in range(1, completed_outer + 1) if index % every == 0]


def _validate_identity(
    run_dir: Path,
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    completed_outer: int,
) -> Dict[str, Any]:
    args = config.get("args")
    if not isinstance(args, Mapping):
        raise AuditError(f"{run_dir}: config.json has no args object")
    required = ("run_tag", "model_type", "seed", "n_assets", "m_states", "outer_iters")
    missing = [key for key in required if key not in args]
    if missing:
        raise AuditError(f"{run_dir}: config args missing {missing}")
    run_tag = str(args["run_tag"])
    model_type = str(args["model_type"])
    if model_type not in {"pinn", "pipinn"}:
        raise AuditError(f"{run_dir}: unsupported model_type={model_type!r}")
    for key, expected in (("run_tag", run_tag), ("model_type", model_type)):
        observed = str(status.get(key, ""))
        if observed != expected:
            raise AuditError(f"{run_dir}: status {key}={observed!r}, config has {expected!r}")
    for index, row in enumerate(rows, 1):
        for key, expected in (("run_tag", run_tag), ("model_type", model_type)):
            if key in row and str(row.get(key, "")) != expected:
                raise AuditError(
                    f"{run_dir}: outer row {index} {key}={row.get(key)!r}, expected {expected!r}"
                )

    target_reached = bool(status.get("target_reached", False))
    configured_outer = parse_int(args["outer_iters"], "outer_iters")
    if not target_reached and completed_outer != configured_outer:
        raise AuditError(
            f"{run_dir}: success has {completed_outer} completed outer rows, expected {configured_outer}"
        )
    if completed_outer > configured_outer:
        raise AuditError(
            f"{run_dir}: completed outer {completed_outer} exceeds configured {configured_outer}"
        )

    if model_type == "pipinn":
        required_policy_fields = ("frozen_policy_iter", "improved_policy_iter")
        for field in required_policy_fields:
            if field not in rows[0]:
                raise AuditError(f"{run_dir}: PI outer history missing {field}")
        for row in rows:
            outer = parse_int(row["outer_iter"], "outer_iter")
            frozen = parse_int(row.get("frozen_policy_iter"), "frozen_policy_iter")
            improved = parse_int(row.get("improved_policy_iter"), "improved_policy_iter")
            if frozen != outer - 1 or improved != outer:
                raise AuditError(
                    f"{run_dir}: outer {outer} policy indices are frozen={frozen}, improved={improved}"
                )
    return {
        "run_tag": run_tag,
        "model_type": model_type,
        "seed": parse_int(args["seed"], "seed"),
        "n_assets": parse_int(args["n_assets"], "n_assets"),
        "m_states": parse_int(args["m_states"], "m_states"),
        "configured_outer_iters": configured_outer,
        "target_reached": target_reached,
    }


def inspect_run(
    run_dir: Path,
    *,
    weight_root: Optional[Path] = None,
    state_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_state_dict,
    inspect_checkpoints: bool = True,
    require_canonical_market: bool = False,
) -> Dict[str, Any]:
    """Inspect one completed run and return a JSON-serializable observation."""

    run_dir = run_dir.resolve()
    config_path = run_dir / "config.json"
    status_path = run_dir / "status.json"
    market_path = run_dir / "market_params.npz"
    outer_path = run_dir / "outer_history.csv"
    for path in (config_path, status_path, market_path, outer_path):
        if not path.is_file():
            raise AuditError(f"{run_dir}: missing required artifact {path.name}")

    config = read_json(config_path)
    status = read_json(status_path)
    completion_status(run_dir, status)
    fields, rows, outer_indices = read_outer_history(outer_path)
    identity = _validate_identity(run_dir, config, status, rows, outer_indices[-1])
    args = config["args"]
    market_hash = canonical_npz_hash(market_path)
    try:
        with np.load(market_path, allow_pickle=False) as market_values:
            market_diagnostics = validate_market_snapshot(
                market_values,
                require_canonical_metadata=require_canonical_market,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"{market_path}: invalid market snapshot: {exc}") from exc

    observation: Dict[str, Any] = {
        **identity,
        "run_dir": str(run_dir),
        "completion_status": "success",
        "config_path": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "status_path": str(status_path),
        "status_file_sha256": sha256_file(status_path),
        "market_path": str(market_path),
        "market_file_sha256": sha256_file(market_path),
        "market_canonical_sha256": market_hash,
        "market_schema_version": market_diagnostics["market_schema_version"],
        "rho_convention": market_diagnostics["rho_convention"],
        "rho_spectral_norm": market_diagnostics["rho_spectral_norm"],
        "min_eig_joint_innovation": market_diagnostics["min_eig"],
        "min_eig_Q": market_diagnostics["min_eig_Q"],
        "outer_history_path": str(outer_path),
        "outer_history_file_sha256": sha256_file(outer_path),
        "outer_columns": fields,
        "outer_row_count": len(rows),
        "outer_indices": outer_indices,
        "outer_first": outer_indices[0],
        "outer_last": outer_indices[-1],
        "checkpoint_audit": None,
    }
    if not inspect_checkpoints:
        return observation

    weight_dir = resolve_weight_dir(run_dir, config, weight_root)
    final_path = weight_dir / "value_net_final.pt"
    last_path = weight_dir / "value_net_last.pt"
    for path in (final_path, last_path):
        if not path.is_file():
            raise AuditError(f"{run_dir}: missing checkpoint {path}")

    iterate_dir = weight_dir / "iterates"
    observed_iterates: Dict[int, Path] = {}
    if iterate_dir.is_dir():
        for entry in iterate_dir.iterdir():
            match = ITERATE_RE.fullmatch(entry.name)
            if match and entry.is_file():
                index = int(match.group(1))
                if index in observed_iterates:
                    raise AuditError(f"{run_dir}: duplicate iterate index {index}")
                observed_iterates[index] = entry
    expected_iterates = expected_iterate_indices(args, outer_indices[-1])
    observed_indices = sorted(observed_iterates)
    if observed_indices != expected_iterates:
        raise AuditError(
            f"{run_dir}: iterate schedule mismatch; expected {expected_iterates}, got {observed_indices}"
        )

    iterate_observations = [
        {
            "outer_iter": index,
            **checkpoint_observation(observed_iterates[index], state_loader),
        }
        for index in observed_indices
    ]
    checkpoints: Dict[str, Any] = {
        "final": checkpoint_observation(final_path, state_loader),
        "last": checkpoint_observation(last_path, state_loader),
        "iterates": iterate_observations,
        "last_iterate": None,
    }
    if iterate_observations:
        checkpoints["last_iterate"] = dict(iterate_observations[-1])
    final_hash = checkpoints["final"]["tensor_sha256"]
    last_hash = checkpoints["last"]["tensor_sha256"]
    if final_hash != last_hash:
        raise AuditError(f"{run_dir}: final and last checkpoints differ by canonical tensor hash")

    last_iterate_matches_final: Optional[bool] = None
    last_iterate_is_final_outer = False
    if checkpoints["last_iterate"] is not None:
        last_iterate_is_final_outer = (
            int(checkpoints["last_iterate"]["outer_iter"]) == outer_indices[-1]
        )
        last_iterate_matches_final = (
            checkpoints["last_iterate"]["tensor_sha256"] == final_hash
        )
        if last_iterate_is_final_outer and not last_iterate_matches_final:
            raise AuditError(
                f"{run_dir}: final-outer iterate and final checkpoint tensor hashes differ"
            )

    observation["checkpoint_audit"] = {
        "weight_dir": str(weight_dir.resolve()),
        "expected_iterate_indices": expected_iterates,
        "observed_iterate_indices": observed_indices,
        "final_equals_last": True,
        "last_iterate_is_final_outer": last_iterate_is_final_outer,
        "last_iterate_matches_final": last_iterate_matches_final,
        "checkpoints": checkpoints,
    }
    return observation


def discover_runs(out_root: Path) -> List[Path]:
    return sorted(path.parent.resolve() for path in out_root.rglob("config.json") if path.is_file())


def parse_seed_spec(text: str) -> List[int]:
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
    if len(out) != len(set(out)):
        raise ValueError(f"duplicate seed in specification: {text!r}")
    return sorted(out)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def build_manifest(
    run_dirs: Sequence[Path],
    *,
    weight_root: Optional[Path] = None,
    state_loader: Callable[[Path], Mapping[str, Any]] = load_checkpoint_state_dict,
    require_canonical_market: bool = False,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            runs.append(
                inspect_run(
                    run_dir,
                    weight_root=weight_root,
                    state_loader=state_loader,
                    require_canonical_market=require_canonical_market,
                )
            )
        except Exception as exc:
            failures.append({"run_dir": str(run_dir.resolve()), "error": str(exc)})
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "posthoc_observed",
        "generated_at": utc_now(),
        "tool_path": str(Path(__file__).resolve()),
        "tool_file_sha256": sha256_file(Path(__file__).resolve()),
        "run_count": len(runs),
        "failure_count": len(failures),
        "valid": not failures and bool(runs),
        "runs": sorted(
            runs,
            key=lambda row: (
                int(row["n_assets"]), int(row["m_states"]),
                str(row["model_type"]), int(row["seed"]), str(row["run_tag"]),
            ),
        ),
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--out-root", type=Path, help="Discover every config.json below this root.")
    source.add_argument(
        "--run-dir", type=Path, action="append",
        help="Audit one run directory; repeat for multiple runs.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Separate manifest directory (default: OUT_ROOT/posthoc_audit).",
    )
    parser.add_argument(
        "--weight-root", type=Path,
        help="Optional relocated weight root; declared config paths are tried first.",
    )
    parser.add_argument(
        "--expected-seeds", default="",
        help="Optional exact seed set, e.g. '1,2,3,5,7,11,17,23,42,101'.",
    )
    parser.add_argument(
        "--allow-legacy-market",
        action="store_true",
        help=(
            "Permit an unlabeled pre-schema-2 market snapshot when its saved "
            "economic identities and identity-block covariance are valid. "
            "Paper audits require canonical metadata by default."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing manifest file.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out_root is not None:
        out_root = args.out_root.resolve()
        run_dirs = discover_runs(out_root)
        output = (args.output or (out_root / "posthoc_audit")).resolve()
    else:
        run_dirs = [path.resolve() for path in (args.run_dir or [])]
        common = Path(os.path.commonpath([str(path.parent) for path in run_dirs]))
        output = (args.output or (common / "posthoc_audit")).resolve()
    if not run_dirs:
        raise SystemExit("no run directories found")
    for run_dir in run_dirs:
        try:
            output.relative_to(run_dir)
        except ValueError:
            continue
        raise SystemExit(f"output must be separate from every run directory: {output} is under {run_dir}")

    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"manifest exists; pass --overwrite: {manifest_path}")
    manifest = build_manifest(
        run_dirs,
        weight_root=args.weight_root,
        require_canonical_market=not args.allow_legacy_market,
    )

    expected = parse_seed_spec(args.expected_seeds)
    if expected:
        cells: Dict[Tuple[int, int, str], set[int]] = {}
        for row in manifest["runs"]:
            key = (int(row["n_assets"]), int(row["m_states"]), str(row["model_type"]))
            cells.setdefault(key, set()).add(int(row["seed"]))
        for cell, seeds in sorted(cells.items()):
            observed = sorted(seeds)
            if observed != expected:
                manifest["valid"] = False
                manifest["failures"].append(
                    {
                        "run_dir": "<collection>",
                        "error": f"cell {cell} expected seeds {expected}, observed {observed}",
                    }
                )
        manifest["failure_count"] = len(manifest["failures"])

    atomic_write_json(manifest_path, manifest)
    print(
        f"[audit] wrote {manifest_path} "
        f"({manifest['run_count']} valid runs, {manifest['failure_count']} failures)"
    )
    if not manifest["valid"]:
        for failure in manifest["failures"]:
            print(f"[invalid] {failure['run_dir']}: {failure['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
