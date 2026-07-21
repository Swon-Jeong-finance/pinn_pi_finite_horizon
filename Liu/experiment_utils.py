"""Experiment utilities for Liu ND PINN / PI-PINN sweeps.

The helpers in this file are intentionally lightweight: they add CLI parsing,
CSV/JSON persistence, and shared PDE-loss early stopping without changing the
core training equations in the original scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def add_common_experiment_args(parser: argparse.ArgumentParser, *, model_type_default: str) -> argparse.ArgumentParser:
    # Run bookkeeping / paths
    parser.add_argument("--run-tag", type=str, default=f"{model_type_default}_manual")
    parser.add_argument("--model-type", type=str, default=model_type_default, choices=["pinn", "pipinn"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--weight-root", type=str, default=None)
    parser.add_argument("--stop-flag-path", type=str, default="")

    # Shared PDE early-stop policy
    parser.add_argument("--pde-stop-threshold", type=float, default=5.0)
    parser.add_argument("--pde-stop-start-outer", type=int, default=100)
    parser.add_argument("--pde-stop-patience", type=int, default=20)

    # Problem / market parameters
    parser.add_argument("--n-assets", type=int, default=30)
    parser.add_argument("--m-states", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12,
                        help="Training randomness: network init, collocation sampling, optimizer.")
    parser.add_argument("--market-seed", type=int, default=None,
                        help="Benchmark market generation seed. Fixed across a seed sweep so all "
                             "runs solve the SAME problem. Default None = use --seed (legacy).")
    parser.add_argument("--tau-max", type=float, default=3.0)
    parser.add_argument("--w-min", type=float, default=0.1)
    parser.add_argument("--w-max", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--x-range-scale", type=float, default=1.0)
    parser.add_argument("--dirichlet-concentration", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=0.25)

    # Shared network / optimization parameters
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--value-depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--terminal-frac", type=float, default=0.5,
                        help="Terminal sample count = max(1, int(batch_size * terminal_frac)). 0.5 reproduces batch_size//2.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--w-terminal", type=float, default=20.0)
    parser.add_argument("--w-shape", type=float, default=1.0)
    parser.add_argument("--w-rra", type=float, default=0.0,
                        help="CRRA homogeneity penalty weight: eta=-w*V_ww/V_w -> gamma. 0 disables.")
    parser.add_argument("--eval-epochs", type=int, default=200)
    parser.add_argument("--outer-iters", type=int, default=1000)

    # Weight-saving policy.
    # The FINAL iterate (value_net_final.pt) is the official reported model.
    # best/last checkpoints are still written, but only as diagnostics.
    # Every save-iterate-every outer iterations, the current iterate is also
    # snapshotted to <weight_root>/iterates/value_net_iter{NNNN}.pt so that
    # post-hoc contraction-ratio (rho_n) analyses can reload every iterate.
    parser.add_argument("--save-iterate-every", type=int, default=1,
                        help="Save per-outer-iteration checkpoints every k iterations (0 disables).")

    # Evaluation-window separation (paper: Omega_ev strictly inside Omega_col).
    # Comma-separated per-side margins; the FIRST value is the primary window
    # used for headline metrics and plots, the rest are re-evaluated for the
    # E9 window-sensitivity study. 0.0 reproduces the legacy full-window eval.
    parser.add_argument("--eval-margin", type=str, default="0.0",
                        help="HALF-WIDTH shrink fraction(s) per spatial axis, e.g. '0.10' or '0.10,0.0,0.05'. "
                             "m keeps (1-m) of each axis length. First = primary.")

    # Held-out validation set (paper: residual levels on a held-out collocation
    # set) and residual-target early stopping of the inner evaluation loop.
    parser.add_argument("--pres-target", type=none_or_float, default=None,
                        help="Target held-out p_res = RMS(PDE)+RMS(terminal). Empty/none disables early stopping.")
    parser.add_argument("--val-points", type=int, default=100000,
                        help="Held-out interior points sampled once from Q_col (dedicated RNG stream).")
    parser.add_argument("--val-terminal-points", type=int, default=10000,
                        help="Held-out terminal points sampled once from Omega_col.")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Check held-out p_res against the target every k inner epochs.")

    # Independent FULL-DIMENSIONAL test set on Omega_ev: uniform points over
    # (0,T] x [W_ev] x prod[X_ev,i] (all state dims vary, unlike the fixed-w
    # (tau, x_0) visualization slice). One base unit-cube sample (fixed RNG,
    # identical across runs) is mapped into every eval window, so nested-
    # window (E9) evaluations use corresponding points. 0 disables.
    parser.add_argument("--test-points", type=int, default=20000,
                        help="Number of full-dimensional Omega_ev test points for Table metrics (0 = off).")

    # Fixed Q_ev diagnostic set: per-outer-iteration e_n = |v~_n - V|_sup +
    # |Dv~_n - DV|_sup (reduced bundle, pointwise Euclidean norm) plus the
    # stability margins (m_ww, M_num, lambda_min(Sigma), guard fraction) are
    # recorded to outer_history.csv, replacing per-iteration weight dumps for
    # the E3-a contraction-ratio analysis. Sampled once from Q_ev at the
    # primary margin with a MARKET-seed-derived RNG (identical across
    # training seeds). 0 disables.
    parser.add_argument("--diag-points", type=int, default=4096,
                        help="Fixed Q_ev diagnostic-set size for per-iteration e_n / stability margins (0 = off).")
    parser.add_argument("--diag-every", type=int, default=1,
                        help="Run the diagnostic pass every k outer iterations (E3-a needs 1; iteration 1 always runs).")

    # E8 timing mode: strip everything the algorithm itself does not need --
    # closed-form comparisons, best-state copies/saves, iterate snapshots,
    # diagnostic-set passes, and (when no pres-target is set) held-out
    # validation -- so wall-clock reflects core computation only.
    parser.add_argument("--timing-mode", action="store_true",
                        help="Disable all diagnostics for clean E8 wall-clock / memory measurement.")

    # Plotting / eval controls.
    #   --skip-figures : skip only the per-run figures; full-dimensional
    #                    metrics are STILL computed and written to metrics.csv.
    #   --skip-eval    : skip the evaluation stage entirely (no metrics.csv).
    #   --skip-plots   : BACK-COMPAT ALIAS for --skip-figures. Historically this
    #                    also skipped evaluation, which silently produced runs
    #                    with _SUCCESS but no metrics.csv; it now means
    #                    figures-only so a main sweep always yields metrics.
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip per-run figures only; full-dim metrics still computed.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the evaluation stage entirely (no metrics.csv).")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Back-compat alias for --skip-figures (figures only; eval still runs).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; load saved weights and re-run evaluation only.")
    parser.add_argument("--n-tau", type=int, default=100)
    parser.add_argument("--n-x", type=int, default=100)
    parser.add_argument("--w-levels", type=str, default="0.5", help="Comma-separated wealth levels for plots.")
    return parser


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
        data = {
            "updated_at": now_iso(),
            "status": status,
            "mode": "eval_only",
            "run_tag": getattr(self.args, "run_tag", ""),
            "model_type": getattr(self.args, "model_type", ""),
        }
        data.update(kwargs)
        save_json(self.status_eval_json, data)

    def mark_success_eval(self, **kwargs: Any) -> None:
        open(os.path.join(self.output_dir, "_SUCCESS_EVAL"), "a").close()
        self.write_status_eval("success", **kwargs)

    def mark_failed_eval(self, **kwargs: Any) -> None:
        open(os.path.join(self.output_dir, "_FAILED_EVAL"), "a").close()
        self.write_status_eval("failed", **kwargs)

    def mark_success(self, **kwargs: Any) -> None:
        open(os.path.join(self.output_dir, "_SUCCESS"), "a").close()
        self.write_status("success", **kwargs)

    def mark_stopped_early(self, **kwargs: Any) -> None:
        open(os.path.join(self.output_dir, "_STOPPED_EARLY"), "a").close()
        self.write_status("stopped_early", **kwargs)

    def mark_failed(self, **kwargs: Any) -> None:
        open(os.path.join(self.output_dir, "_FAILED"), "a").close()
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
