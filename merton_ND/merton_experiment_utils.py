"""Experiment utilities for the Merton (with-consumption) PINN / PI-PINN sweep.

Independent copy of the Liu-track experiment_utils: identical infrastructure
(CSV/JSON persistence, per-run log rotation, shared PDE-loss early stopping,
device / reproducibility helpers) but WITHOUT add_common_experiment_args --
the Merton training scripts define their own argparse (single state W, market
from market_setup, consumption controls). save_closed_form_solution is kept
for API parity though the Merton closed form is analytic (no ODE npz).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def none_or_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null", "nan", ""}:
        return None
    return float(value)


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return str(obj)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=json_default)


def save_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Atomically replace a JSON artifact after writing it completely."""
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=json_default)
    os.replace(tmp_path, path)


def sha256_file(path: str) -> str:
    """Hash file bytes for corruption detection (not model-state identity)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor content independently of ``torch.save`` container bytes.

    PyTorch's serialization container is not the identity of a model state:
    independently saving the same tensors need not be assumed to yield the
    same file bytes.  This digest uses sorted tensor names, dtype, shape, and
    contiguous numerical bytes, matching the canonical-array convention used
    by the exact-map post-processor.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def append_csv_rows(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            clean = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy().tolist()
                elif isinstance(v, np.generic):
                    v = v.item()
                clean[k] = v
            writer.writerow(clean)


def resolve_device(device_spec: str) -> torch.device:
    spec = str(device_spec or "auto").strip()
    if spec.lower() in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={spec}, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(spec)


def set_reproducibility(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Keep the original code's eager CUDA initialization behavior.
        if str(device).startswith("cuda"):
            torch.cuda.init()
            _ = torch.zeros(1, device=device)


def parse_w_levels(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_eval_margins(text: str) -> List[float]:
    """Parse comma-separated HALF-WIDTH eval margins. First entry = primary.

    Convention (paper): an axis [c-h, c+h] shrinks to [c-(1-m)h, c+(1-m)h],
    i.e. margin m removes fraction m of the HALF-width on each side and keeps
    fraction (1-m) of the original length. m=0 is the full-window stress test.
    """
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        vals = [0.0]
    for m in vals:
        if not (0.0 <= m < 1.0):
            raise ValueError(f"eval margin (half-width fraction) must be in [0, 1), got {m}")
    return vals


def shrink_bounds(lo, hi, margin: float):
    """HALF-WIDTH shrink: [c-h, c+h] -> [c-(1-m)h, c+(1-m)h].

    Equivalently each side moves inward by margin*(hi-lo)/2, keeping (1-m)
    of the original length. Works on scalars and numpy arrays.
    NOTE: this replaces the earlier full-span-per-side convention; the old
    per-side value p corresponds to half-width margin m = 2p.
    """
    half_removed = 0.5 * margin * (hi - lo)
    return lo + half_removed, hi - half_removed


def pres_from_mse(pde_mse: float, terminal_mse: float) -> float:
    """Paper-style residual level: RMS(PDE residual) + RMS(terminal mismatch).

    The logged training components are MSEs, so p_res = sqrt(pde) + sqrt(term)
    (NOT their raw sum).
    """
    return float(np.sqrt(max(pde_mse, 0.0)) + np.sqrt(max(terminal_mse, 0.0)))


class ExperimentRecorder:
    def __init__(self, output_dir: str, weight_dir: str, args: argparse.Namespace):
        self.output_dir = output_dir
        self.weight_dir = weight_dir
        self.args = args
        ensure_dir(self.output_dir)
        ensure_dir(self.weight_dir)
        ensure_dir(os.path.join(self.output_dir, "plots"))
        self.train_csv = os.path.join(self.output_dir, "train_history.csv")
        self.outer_csv = os.path.join(self.output_dir, "outer_history.csv")
        self.metrics_csv = os.path.join(self.output_dir, "metrics.csv")
        self.config_json = os.path.join(self.output_dir, "config.json")
        self.status_json = os.path.join(self.output_dir, "status.json")
        # Eval-only artifacts: NEVER touch the training-time provenance files.
        self.config_eval_json = os.path.join(self.output_dir, "config_eval.json")
        self.status_eval_json = os.path.join(self.output_dir, "status_eval.json")

    def rotate_training_logs(self) -> None:
        """Archive any pre-existing per-run CSV logs before a NEW training run.

        The CSV writers append (so one run's epochs accumulate incrementally),
        which means re-running the SAME run tag into the same output dir would
        interleave two experiments in one file and silently corrupt Figure-2
        ratios / inner-best statistics. Called only at TRAINING start --
        eval-only reruns must never touch these files.
        Existing files are renamed to <name>.old.<timestamp>, not deleted.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in (self.train_csv, self.outer_csv, self.metrics_csv):
            if os.path.exists(path):
                dst = f"{path}.old.{stamp}"
                os.replace(path, dst)
                print(f"[recorder] previous log archived: {os.path.basename(path)} "
                      f"-> {os.path.basename(dst)}")

    def rotate_training_checkpoints(self) -> None:
        """Archive checkpoint artifacts before a new same-tag training run.

        Exact-map checkpoint discovery is filename based.  Leaving an older
        ``iterates/`` directory in place can silently mix two trajectories
        when a forced rerun stops earlier or changes its save schedule.  A
        timestamped rename preserves the old artifacts while guaranteeing
        that the new manifest describes only the new run.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        names = (
            "iterates",
            "checkpoint_manifest.json",
            "value_net_final.pt",
            "value_net_last.pt",
            "value_net_best_diag.pt",
        )
        for name in names:
            path = os.path.join(self.weight_dir, name)
            if not os.path.exists(path):
                continue
            dst = f"{path}.old.{stamp}"
            os.replace(path, dst)
            print(f"[recorder] previous checkpoint artifact archived: {name} "
                  f"-> {os.path.basename(dst)}")

    def save_config(self, extra: Optional[Dict[str, Any]] = None) -> None:
        data = {
            "created_at": now_iso(),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "args": vars(self.args),
            "output_dir": self.output_dir,
            "weight_dir": self.weight_dir,
        }
        if extra:
            data.update(extra)
        save_json(self.config_json, data)

    def write_status(self, status: str, **kwargs: Any) -> None:
        if status == "running":
            self._clear_terminal_markers(eval_only=False)
        data = {
            "updated_at": now_iso(),
            "status": status,
            "run_tag": getattr(self.args, "run_tag", ""),
            "model_type": getattr(self.args, "model_type", ""),
        }
        data.update(kwargs)
        save_json(self.status_json, data)

    def save_config_eval(self, extra: Optional[Dict[str, Any]] = None) -> None:
        data = {
            "created_at": now_iso(),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "args": vars(self.args),
            "output_dir": self.output_dir,
            "weight_dir": self.weight_dir,
            "mode": "eval_only",
        }
        if extra:
            data.update(extra)
        save_json(self.config_eval_json, data)

    def write_status_eval(self, status: str, **kwargs: Any) -> None:
        if status == "running":
            self._clear_terminal_markers(eval_only=True)
        data = {
            "updated_at": now_iso(),
            "status": status,
            "mode": "eval_only",
            "run_tag": getattr(self.args, "run_tag", ""),
            "model_type": getattr(self.args, "model_type", ""),
        }
        data.update(kwargs)
        save_json(self.status_eval_json, data)

    def _clear_terminal_markers(self, *, eval_only: bool) -> None:
        names = ("_SUCCESS_EVAL", "_FAILED_EVAL") if eval_only else (
            "_SUCCESS", "_STOPPED_EARLY", "_FAILED",
        )
        for name in names:
            try:
                os.remove(os.path.join(self.output_dir, name))
            except FileNotFoundError:
                pass

    def _set_terminal_marker(self, marker: str, *, eval_only: bool = False) -> None:
        """Expose one terminal state, never a conflicting marker set.

        The launcher also clears markers before a retry, but keeping this
        invariant in the recorder protects manual runs as well.
        """
        names = ("_SUCCESS_EVAL", "_FAILED_EVAL") if eval_only else (
            "_SUCCESS", "_STOPPED_EARLY", "_FAILED",
        )
        for name in names:
            if name == marker:
                continue
            try:
                os.remove(os.path.join(self.output_dir, name))
            except FileNotFoundError:
                pass
        open(os.path.join(self.output_dir, marker), "a").close()

    def mark_success_eval(self, **kwargs: Any) -> None:
        self._set_terminal_marker("_SUCCESS_EVAL", eval_only=True)
        self.write_status_eval("success", **kwargs)

    def mark_failed_eval(self, **kwargs: Any) -> None:
        self._set_terminal_marker("_FAILED_EVAL", eval_only=True)
        self.write_status_eval("failed", **kwargs)

    def mark_success(self, **kwargs: Any) -> None:
        self._set_terminal_marker("_SUCCESS")
        self.write_status("success", **kwargs)

    def mark_stopped_early(self, **kwargs: Any) -> None:
        self._set_terminal_marker("_STOPPED_EARLY")
        self.write_status("stopped_early", **kwargs)

    def mark_failed(self, **kwargs: Any) -> None:
        self._set_terminal_marker("_FAILED")
        self.write_status("failed", **kwargs)

    def save_market_snapshot(self, **arrays: Any) -> None:
        path = os.path.join(self.output_dir, "market_params.npz")
        np.savez(path, **arrays)

    def save_closed_form_solution(self, sol: Any) -> None:
        path = os.path.join(self.output_dir, "closed_form_ode.npz")
        np.savez(path, t=sol.t, y=sol.y, success=np.array([int(bool(sol.success))]))


class PDEEarlyStopper:
    """Shared PDE-loss early stopper for paired PINN / PI-PINN jobs."""

    def __init__(
        self,
        *,
        threshold: float,
        start_outer: int,
        patience: int,
        stop_flag_path: str,
        recorder: ExperimentRecorder,
        run_tag: str,
        model_type: str,
    ):
        self.threshold = float(threshold)
        self.start_outer = int(start_outer)
        self.patience = int(patience)
        self.stop_flag_path = str(stop_flag_path or "")
        self.recorder = recorder
        self.run_tag = run_tag
        self.model_type = model_type
        self.bad_count = 0

    def shared_stop_exists(self) -> bool:
        return bool(self.stop_flag_path) and os.path.exists(self.stop_flag_path)

    def mark_from_existing_flag(self, outer_iter: int, pde_loss: Optional[float] = None) -> Dict[str, Any]:
        info = {
            "reason": "shared_stop_flag_exists",
            "outer_iter": int(outer_iter),
            "pde_loss": pde_loss,
            "threshold": self.threshold,
            "patience": self.patience,
            "stop_flag_path": self.stop_flag_path,
        }
        self.recorder.mark_stopped_early(**info)
        return info

    def write_shared_stop_flag(self, outer_iter: int, pde_loss: float, reason: str) -> None:
        if not self.stop_flag_path:
            return
        ensure_dir(os.path.dirname(self.stop_flag_path))
        exists = os.path.exists(self.stop_flag_path) and os.path.getsize(self.stop_flag_path) > 0
        with open(self.stop_flag_path, "a", encoding="utf-8") as f:
            if not exists:
                f.write("timestamp\trun_tag\tmodel_type\treason\touter_iter\tpde_loss\tthreshold\tpatience\n")
            f.write(
                f"{now_iso()}\t{self.run_tag}\t{self.model_type}\t{reason}\t{int(outer_iter)}\t"
                f"{float(pde_loss):.12g}\t{self.threshold:.12g}\t{self.patience}\n"
            )

    def update(self, outer_iter: int, pde_loss: float) -> Tuple[bool, Dict[str, Any]]:
        pde_loss = float(pde_loss)
        if self.shared_stop_exists():
            return True, self.mark_from_existing_flag(outer_iter, pde_loss)

        active = int(outer_iter) >= self.start_outer
        is_bad = active and (pde_loss > self.threshold)
        self.bad_count = self.bad_count + 1 if is_bad else 0

        info = {
            "active": active,
            "is_bad": is_bad,
            "bad_count": self.bad_count,
            "threshold": self.threshold,
            "start_outer": self.start_outer,
            "patience": self.patience,
        }

        if self.bad_count >= self.patience:
            reason = "pde_loss_above_threshold_patience"
            self.write_shared_stop_flag(outer_iter, pde_loss, reason)
            stop_info = {
                "reason": reason,
                "outer_iter": int(outer_iter),
                "pde_loss": pde_loss,
                "bad_count": self.bad_count,
                "threshold": self.threshold,
                "patience": self.patience,
                "stop_flag_path": self.stop_flag_path,
            }
            self.recorder.mark_stopped_early(**stop_info)
            info.update(stop_info)
            return True, info

        return False, info
