#!/usr/bin/env python3
"""Affine closed-form substitution gate for the current Liu HJB residual.

This is deliberately a source-level, standalone check.  Importing either
training script would parse its CLI and start constructing an experiment, so
the gate extracts only ``compute_derivatives_nd`` and ``hjb_residual_nd`` from
the selected current source file.  The extraction namespace explicitly
supplies every external dependency used by the residual, including
``actual_risk_premium_torch`` and ``safe_concave_vww``.

Stage 1 is a NumPy algebra check.  It substitutes the exponential-quadratic
affine solution and the Riccati right-hand side into the HJB.  Stage 2 feeds a
locally exact differentiable version of the same solution through the actual
Torch HJB residual extracted from each training source.  For PI-PINN, Stage 3
also feeds the analytic, raw (wealth-scaled) optimal policy and the same value
through the actual frozen-policy linear PDE residual.  No V_ww guard or policy
clipping is allowed in that linear-operator check.  Torch stages are reported
as clean skips when PyTorch is unavailable, unless ``--require-torch`` is
requested.

The gate is intentionally affine-only.  It is not evidence that the affine
closed form solves a non-affine perturbation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_BY_SOLVER = {
    "pinn": SCRIPT_DIR / "Liu_nd_pinn.py",
    "pipinn": SCRIPT_DIR / "Liu_nd_pi_pinn.py",
}
EXPECTED_RESIDUAL_ARGS = (
    "model", "w", "x", "tau", "M", "N", "gamma", "r", "K_t",
    "k0_t", "Q_t", "Gamma_t", "lam0_t", "Lam_t",
)
EXPECTED_LINEAR_RESIDUAL_ARGS = (
    "value_net", "theta_n", "w", "x", "tau", "M", "N", "gamma", "r",
    "K_t", "k0_t", "Q_t", "Gamma_t", "lam0_t", "Lam_t",
)
ODE_RTOL = 1.0e-12
ODE_ATOL = 1.0e-14
ODE_NODES = 8001
VWW_GUARD = 1.0e-8


class GateError(RuntimeError):
    """A source-contract or numerical substitution failure."""


@dataclass(frozen=True)
class SourceContract:
    solver: str
    path: str
    sha256: str
    residual_lineno: int
    residual_end_lineno: int
    residual_calls: tuple[str, ...]
    linear_residual_lineno: int | None
    linear_residual_end_lineno: int | None
    linear_residual_calls: tuple[str, ...]
    ode_rtol: float
    ode_atol: float
    ode_nodes: int


@dataclass(frozen=True)
class NumericalResult:
    stage: str
    status: str
    max_abs_residual: float | None
    max_scaled_residual: float | None
    tolerance: float
    detail: str
    conditions: Mapping[str, Any] | None = None


def _single_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise GateError(f"expected exactly one top-level {name} definition; found {len(matches)}")
    return matches[0]


def _literal_default(node: ast.expr, label: str) -> float:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise GateError(f"{label} must have a literal default") from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateError(f"{label} must have a numeric literal default")
    return float(value)


def _function_defaults(node: ast.FunctionDef) -> dict[str, ast.expr]:
    positional = [*node.args.posonlyargs, *node.args.args]
    default_names = [arg.arg for arg in positional[-len(node.args.defaults):]] if node.args.defaults else []
    defaults = dict(zip(default_names, node.args.defaults))
    defaults.update(zip((arg.arg for arg in node.args.kwonlyargs), node.args.kw_defaults))
    return {key: value for key, value in defaults.items() if value is not None}


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def inspect_source_contract(path: Path, solver: str) -> SourceContract:
    """Validate that extraction targets the current, expected source contract."""

    raw = path.read_bytes()
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise GateError(f"cannot parse solver source {path}: {exc}") from exc

    deriv = _single_function(tree, "compute_derivatives_nd")
    residual = _single_function(tree, "hjb_residual_nd")
    ode = _single_function(tree, "solve_closed_form_ode")

    residual_args = tuple(arg.arg for arg in [*residual.args.posonlyargs, *residual.args.args])
    if residual_args != EXPECTED_RESIDUAL_ARGS:
        raise GateError(
            "hjb_residual_nd signature changed; update the substitution gate explicitly: "
            f"{residual_args!r}"
        )
    deriv_args = tuple(arg.arg for arg in [*deriv.args.posonlyargs, *deriv.args.args])
    if deriv_args != ("model", "w", "x", "tau", "M"):
        raise GateError(f"compute_derivatives_nd signature changed: {deriv_args!r}")

    calls = tuple(sorted({name for call in ast.walk(residual)
                          if isinstance(call, ast.Call)
                          if (name := _call_name(call)) is not None}))
    required = {"compute_derivatives_nd", "safe_concave_vww"}
    if solver == "pipinn":
        required.add("actual_risk_premium_torch")
    missing = required.difference(calls)
    if missing:
        raise GateError(
            f"{solver} residual no longer calls required dependencies {sorted(missing)}; "
            "the gate refuses to validate a stale source fragment"
        )

    linear_lineno: int | None = None
    linear_end_lineno: int | None = None
    linear_calls: tuple[str, ...] = ()
    if solver == "pipinn":
        linear = _single_function(tree, "linear_pde_residual_nd")
        linear_args = tuple(
            arg.arg for arg in [*linear.args.posonlyargs, *linear.args.args]
        )
        if linear_args != EXPECTED_LINEAR_RESIDUAL_ARGS:
            raise GateError(
                "linear_pde_residual_nd signature changed; update the substitution "
                f"gate explicitly: {linear_args!r}"
            )
        linear_calls = tuple(sorted({
            name for call in ast.walk(linear)
            if isinstance(call, ast.Call)
            if (name := _call_name(call)) is not None
        }))
        required_linear = {"compute_derivatives_nd", "actual_risk_premium_torch"}
        missing_linear = required_linear.difference(linear_calls)
        if missing_linear:
            raise GateError(
                "PI-PINN linear residual no longer calls required dependencies "
                f"{sorted(missing_linear)}; the gate refuses to validate a stale "
                "source fragment"
            )
        forbidden_linear = {"safe_concave_vww", "clamp", "clip"}
        present_forbidden = forbidden_linear.intersection(linear_calls)
        if present_forbidden:
            raise GateError(
                "PI-PINN frozen-policy linear residual must use raw V_ww and the "
                "supplied raw policy without a guard or clip; found calls "
                f"{sorted(present_forbidden)}"
            )
        linear_lineno = int(linear.lineno)
        linear_end_lineno = int(linear.end_lineno or linear.lineno)

    defaults = _function_defaults(ode)
    try:
        rtol = _literal_default(defaults["rtol"], "solve_closed_form_ode.rtol")
        atol = _literal_default(defaults["atol"], "solve_closed_form_ode.atol")
    except KeyError as exc:
        raise GateError(f"solve_closed_form_ode is missing default {exc.args[0]}") from exc
    if rtol != ODE_RTOL or atol != ODE_ATOL:
        raise GateError(
            f"ODE defaults changed: rtol={rtol:g}, atol={atol:g}; expected "
            f"{ODE_RTOL:g}, {ODE_ATOL:g}"
        )

    node_counts: list[int] = []
    forwards_tolerances = False
    for item in ast.walk(ode):
        if not isinstance(item, ast.Call):
            continue
        name = _call_name(item)
        if name == "linspace" and len(item.args) >= 3:
            try:
                node_counts.append(int(ast.literal_eval(item.args[2])))
            except (ValueError, TypeError):
                pass
        if name == "solve_ivp":
            keyword_names = {kw.arg: kw.value for kw in item.keywords if kw.arg is not None}
            forwards_tolerances = (
                isinstance(keyword_names.get("rtol"), ast.Name)
                and keyword_names["rtol"].id == "rtol"
                and isinstance(keyword_names.get("atol"), ast.Name)
                and keyword_names["atol"].id == "atol"
            )
    if ODE_NODES not in node_counts:
        raise GateError(f"closed-form default grid is not {ODE_NODES} nodes")
    if not forwards_tolerances:
        raise GateError("solve_closed_form_ode does not forward its rtol/atol to solve_ivp")

    return SourceContract(
        solver=solver,
        path=str(path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        residual_lineno=int(residual.lineno),
        residual_end_lineno=int(residual.end_lineno or residual.lineno),
        residual_calls=calls,
        linear_residual_lineno=linear_lineno,
        linear_residual_end_lineno=linear_end_lineno,
        linear_residual_calls=linear_calls,
        ode_rtol=rtol,
        ode_atol=atol,
        ode_nodes=ODE_NODES,
    )


def compile_named_functions(
    path: Path,
    names: Iterable[str],
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile only named top-level functions into an explicit namespace."""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    selected = [_single_function(tree, name) for name in names]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(namespace)
    exec(compile(module, str(path), "exec"), ns, ns)
    return ns


def validate_affine_request(mode: str, eps: float) -> None:
    if str(mode).strip().lower() != "affine" or float(eps) != 0.0:
        raise GateError(
            "closed-form residual substitution is affine-only: require "
            "--risk-premium-mode affine --nonaffine-eps 0 exactly"
        )


def deterministic_problem() -> dict[str, np.ndarray | float | int]:
    """Small, stable affine instance used only by this implementation gate."""

    gamma = 3.0
    r = 0.02
    K = np.array([[0.31, 0.04], [-0.02, 0.24]], dtype=np.float64)
    xbar = np.array([0.03, -0.02], dtype=np.float64)
    SigmaX = np.array([[0.16, 0.00], [0.03, 0.12]], dtype=np.float64)
    rho = np.array(
        [[0.12, -0.04], [-0.08, 0.10], [0.05, 0.03], [-0.02, -0.06]],
        dtype=np.float64,
    )
    lam0 = np.array([0.11, 0.08, 0.13, 0.09], dtype=np.float64)
    Lam = np.array(
        [[0.20, -0.08], [0.05, 0.16], [-0.10, 0.07], [0.12, 0.04]],
        dtype=np.float64,
    )
    return {
        "M": 2,
        "N": 4,
        "gamma": gamma,
        "r": r,
        "K": K,
        "k0": K @ xbar,
        "Q": SigmaX @ SigmaX.T,
        "Gamma": rho @ SigmaX.T,
        "lam0": lam0,
        "Lam": Lam,
    }


def riccati_rhs_batch(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    problem: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current affine exponential-quadratic coefficient equations."""

    del a  # The autonomous right-hand side does not depend on the level a.
    gamma = float(problem["gamma"])
    K = np.asarray(problem["K"])
    k0 = np.asarray(problem["k0"])
    Q = np.asarray(problem["Q"])
    Gamma = np.asarray(problem["Gamma"])
    lam0 = np.asarray(problem["lam0"])
    Lam = np.asarray(problem["Lam"])
    alpha = (1.0 - gamma) / gamma

    B = b.shape[0]
    da = np.empty(B, dtype=np.float64)
    db = np.empty_like(b)
    dC = np.empty_like(C)
    for j in range(B):
        Cj = 0.5 * (C[j] + C[j].T)
        A = Lam + Gamma @ Cj
        m = lam0 + Gamma @ b[j]
        dC[j] = -(K.T @ Cj + Cj @ K) + Cj @ Q @ Cj + alpha * (A.T @ A)
        db[j] = Cj @ k0 - K.T @ b[j] + Cj @ Q @ b[j] + alpha * (A.T @ m)
        da[j] = (
            k0 @ b[j]
            + 0.5 * np.trace(Q @ Cj)
            + 0.5 * (b[j] @ Q @ b[j])
            + 0.5 * alpha * (m @ m)
        )
    return da, db, dC


def make_substitution_batch(seed: int, batch_size: int) -> dict[str, np.ndarray | Mapping[str, Any]]:
    if batch_size < 2:
        raise GateError("batch-size must be at least 2")
    rng = np.random.default_rng(seed)
    problem = deterministic_problem()
    M = int(problem["M"])
    a = rng.normal(scale=0.05, size=batch_size)
    b = rng.normal(scale=0.06, size=(batch_size, M))
    raw_C = rng.normal(scale=0.025, size=(batch_size, M, M))
    C = 0.5 * (raw_C + np.swapaxes(raw_C, 1, 2))
    da, db, dC = riccati_rhs_batch(a, b, C, problem)
    return {
        "problem": problem,
        "tau": rng.uniform(0.05, 1.8, size=(batch_size, 1)),
        "w": rng.uniform(0.35, 1.25, size=(batch_size, 1)),
        "x": rng.uniform(-0.35, 0.35, size=(batch_size, M)),
        "a": a,
        "b": b,
        "C": C,
        "da": da,
        "db": db,
        "dC": dC,
    }


def linear_policy_conditions(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the unmodified analytic policy used by the linear-PDE gate."""

    problem = batch["problem"]
    gamma = float(problem["gamma"])
    r = float(problem["r"])
    tau = np.asarray(batch["tau"], dtype=np.float64)[:, 0]
    w = np.asarray(batch["w"], dtype=np.float64)[:, 0]
    x = np.asarray(batch["x"], dtype=np.float64)
    a = np.asarray(batch["a"], dtype=np.float64)
    b = np.asarray(batch["b"], dtype=np.float64)
    C = np.asarray(batch["C"], dtype=np.float64)

    p = 1.0 - gamma
    exponent = p * r * tau + a + np.einsum("bi,bi->b", b, x)
    exponent += 0.5 * np.einsum("bi,bij,bj->b", x, C, x)
    V = np.exp(exponent) * np.power(w, p) / p
    V_ww = p * (p - 1.0) * V / np.square(w)
    guard_would_modify = bool(np.any(V_ww > -VWW_GUARD))
    return {
        "policy_source": "analytic_affine_foc",
        "policy_representation": "raw_theta_not_theta_over_w",
        "theta_clipping_applied": False,
        "vww_guard_applied": False,
        "vww_guard_threshold": VWW_GUARD,
        "vww_guard_inactive_on_reference": not guard_would_modify,
        "analytic_vww_min": float(np.min(V_ww)),
        "analytic_vww_max": float(np.max(V_ww)),
    }


def numpy_substitution(batch: Mapping[str, Any], tolerance: float) -> NumericalResult:
    problem = batch["problem"]
    gamma = float(problem["gamma"])
    r = float(problem["r"])
    K = np.asarray(problem["K"])
    k0 = np.asarray(problem["k0"])
    Q = np.asarray(problem["Q"])
    Gamma = np.asarray(problem["Gamma"])
    lam0 = np.asarray(problem["lam0"])
    Lam = np.asarray(problem["Lam"])
    tau = np.asarray(batch["tau"])[:, 0]
    w = np.asarray(batch["w"])[:, 0]
    x = np.asarray(batch["x"])
    a = np.asarray(batch["a"])
    b = np.asarray(batch["b"])
    C = np.asarray(batch["C"])
    da = np.asarray(batch["da"])
    db = np.asarray(batch["db"])
    dC = np.asarray(batch["dC"])

    p = 1.0 - gamma
    g = b + np.einsum("bij,bj->bi", C, x)
    q = p * r * tau + a + np.einsum("bi,bi->b", b, x)
    q += 0.5 * np.einsum("bi,bij,bj->b", x, C, x)
    V = np.exp(q) * np.power(w, p) / p
    q_tau = p * r + da + np.einsum("bi,bi->b", db, x)
    q_tau += 0.5 * np.einsum("bi,bij,bj->b", x, dC, x)
    V_tau = V * q_tau
    V_w = p * V / w
    V_ww = p * (p - 1.0) * V / np.square(w)
    V_x = V[:, None] * g
    V_wx = V_w[:, None] * g
    V_xx = V[:, None, None] * (C + np.einsum("bi,bj->bij", g, g))

    drift = k0[None, :] - np.einsum("ij,bj->bi", K, x)
    lam_x = lam0[None, :] + np.einsum("ij,bj->bi", Lam, x)
    combined = lam_x * V_w[:, None] + np.einsum("ij,bj->bi", Gamma, V_wx)
    terms = np.column_stack(
        [
            -V_tau,
            r * w * V_w,
            np.einsum("bi,bi->b", drift, V_x),
            0.5 * np.einsum("ij,bij->b", Q, V_xx),
            -np.einsum("bi,bi->b", combined, combined) / (2.0 * V_ww),
        ]
    )
    residual = terms.sum(axis=1)
    scale = np.maximum(np.max(np.abs(terms), axis=1), np.finfo(np.float64).tiny)
    max_abs = float(np.max(np.abs(residual)))
    max_scaled = float(np.max(np.abs(residual) / scale))
    status = "pass" if max_scaled <= tolerance else "fail"
    return NumericalResult(
        stage="numpy_affine_substitution",
        status=status,
        max_abs_residual=max_abs,
        max_scaled_residual=max_scaled,
        tolerance=tolerance,
        detail="independent exponential-quadratic derivative/Riccati substitution",
    )


def torch_substitution(
    batch: Mapping[str, Any],
    source: Path,
    solver: str,
    tolerance: float,
) -> NumericalResult:
    try:
        import torch
    except ModuleNotFoundError:
        return NumericalResult(
            stage=f"torch_current_residual_{solver}",
            status="skip",
            max_abs_residual=None,
            max_scaled_residual=None,
            tolerance=tolerance,
            detail="PyTorch is not installed in this environment",
        )

    def actual_risk_premium_torch(x, lam0_t, Lam_t):
        return lam0_t.unsqueeze(0) + torch.einsum("ij,bj->bi", Lam_t, x)

    def safe_concave_vww(V_ww):
        return torch.clamp(V_ww, max=-1.0e-8)

    ns = compile_named_functions(
        source,
        ("compute_derivatives_nd", "hjb_residual_nd"),
        {
            "torch": torch,
            "actual_risk_premium_torch": actual_risk_premium_torch,
            "safe_concave_vww": safe_concave_vww,
        },
    )

    problem = batch["problem"]
    dtype = torch.float64
    t = lambda value: torch.as_tensor(value, dtype=dtype)  # noqa: E731
    tau0 = t(batch["tau"])
    w = t(batch["w"]).clone().requires_grad_(True)
    x = t(batch["x"]).clone().requires_grad_(True)
    tau = tau0.clone().requires_grad_(True)
    a0, b0, C0 = t(batch["a"]), t(batch["b"]), t(batch["C"])
    da, db, dC = t(batch["da"]), t(batch["db"]), t(batch["dC"])
    gamma, r = float(problem["gamma"]), float(problem["r"])

    class LocalExactValue:
        def __call__(self, w_arg, x_arg, tau_arg):
            dt = tau_arg - tau0
            a = a0 + da * dt[:, 0]
            b = b0 + db * dt
            C = C0 + dC * dt[:, :, None]
            exponent = (1.0 - gamma) * r * tau_arg[:, 0] + a
            exponent = exponent + torch.einsum("bi,bi->b", b, x_arg)
            exponent = exponent + 0.5 * torch.einsum("bi,bij,bj->b", x_arg, C, x_arg)
            return (
                torch.exp(exponent[:, None])
                * torch.pow(w_arg, 1.0 - gamma)
                / (1.0 - gamma)
            )

    args = (
        LocalExactValue(), w, x, tau, int(problem["M"]), int(problem["N"]),
        gamma, r, t(problem["K"]), t(problem["k0"]), t(problem["Q"]),
        t(problem["Gamma"]), t(problem["lam0"]), t(problem["Lam"]),
    )
    residual = ns["hjb_residual_nd"](*args)[0]
    max_abs = float(torch.max(torch.abs(residual)).detach().cpu())

    # Scale by the value magnitude.  The NumPy stage already checks a strict
    # termwise normalization; this stage's purpose is source-path parity.
    V = LocalExactValue()(w, x, tau)
    scale = torch.clamp(torch.max(torch.abs(V)), min=torch.finfo(dtype).tiny)
    max_scaled = float((torch.max(torch.abs(residual)) / scale).detach().cpu())
    status = "pass" if max_scaled <= tolerance else "fail"
    return NumericalResult(
        stage=f"torch_current_residual_{solver}",
        status=status,
        max_abs_residual=max_abs,
        max_scaled_residual=max_scaled,
        tolerance=tolerance,
        detail=f"AST-extracted current residual from {source.name}",
    )


def torch_linear_substitution(
    batch: Mapping[str, Any],
    source: Path,
    tolerance: float,
) -> NumericalResult:
    """Run T[V*, theta*] through the current AST-extracted PI-PINN operator."""

    conditions = linear_policy_conditions(batch)
    try:
        import torch
    except ModuleNotFoundError:
        return NumericalResult(
            stage="torch_current_linear_residual_pipinn",
            status="skip",
            max_abs_residual=None,
            max_scaled_residual=None,
            tolerance=tolerance,
            detail="PyTorch is not installed in this environment",
            conditions=conditions,
        )

    def actual_risk_premium_torch(x, lam0_t, Lam_t):
        return lam0_t.unsqueeze(0) + torch.einsum("ij,bj->bi", Lam_t, x)

    ns = compile_named_functions(
        source,
        ("compute_derivatives_nd", "linear_pde_residual_nd"),
        {
            "torch": torch,
            "actual_risk_premium_torch": actual_risk_premium_torch,
        },
    )

    problem = batch["problem"]
    dtype = torch.float64
    t = lambda value: torch.as_tensor(value, dtype=dtype)  # noqa: E731
    tau0 = t(batch["tau"])
    w = t(batch["w"]).clone().requires_grad_(True)
    x = t(batch["x"]).clone().requires_grad_(True)
    tau = tau0.clone().requires_grad_(True)
    a0, b0, C0 = t(batch["a"]), t(batch["b"]), t(batch["C"])
    da, db, dC = t(batch["da"]), t(batch["db"]), t(batch["dC"])
    gamma, r = float(problem["gamma"]), float(problem["r"])

    class LocalExactValue:
        def __call__(self, w_arg, x_arg, tau_arg):
            dt = tau_arg - tau0
            a = a0 + da * dt[:, 0]
            b = b0 + db * dt
            C = C0 + dC * dt[:, :, None]
            exponent = (1.0 - gamma) * r * tau_arg[:, 0] + a
            exponent = exponent + torch.einsum("bi,bi->b", b, x_arg)
            exponent = exponent + 0.5 * torch.einsum(
                "bi,bij,bj->b", x_arg, C, x_arg
            )
            return (
                torch.exp(exponent[:, None])
                * torch.pow(w_arg, 1.0 - gamma)
                / (1.0 - gamma)
            )

    value_model = LocalExactValue()
    V = value_model(w, x, tau)
    p = 1.0 - gamma
    V_w = p * V / w
    V_ww = p * (p - 1.0) * V / torch.square(w)
    grad_log_phi = b0 + torch.einsum("bij,bj->bi", C0, x)
    V_wx = V_w * grad_log_phi
    lam_x = actual_risk_premium_torch(x, t(problem["lam0"]), t(problem["Lam"]))
    Gamma_Vwx = torch.einsum("ij,bj->bi", t(problem["Gamma"]), V_wx)

    # This is the raw dollar policy theta*, not vartheta=theta*/w.  The
    # frozen-policy operator must receive it without either a V_ww guard or a
    # componentwise policy clip.
    theta_star = -(lam_x * V_w + Gamma_Vwx) / V_ww
    theta_closed_form = (
        w / gamma
        * (
            lam_x
            + torch.einsum("ij,bj->bi", t(problem["Gamma"]), grad_log_phi)
        )
    )
    theta_scale = torch.clamp(
        torch.max(torch.abs(theta_closed_form)), min=torch.finfo(dtype).tiny
    )
    theta_formula_error = float(
        (torch.max(torch.abs(theta_star - theta_closed_form)) / theta_scale)
        .detach()
        .cpu()
    )

    residual = ns["linear_pde_residual_nd"](
        value_model,
        theta_star.detach(),
        w,
        x,
        tau,
        int(problem["M"]),
        int(problem["N"]),
        gamma,
        r,
        t(problem["K"]),
        t(problem["k0"]),
        t(problem["Q"]),
        t(problem["Gamma"]),
        t(problem["lam0"]),
        t(problem["Lam"]),
    )[0]
    max_abs = float(torch.max(torch.abs(residual)).detach().cpu())
    scale = torch.clamp(torch.max(torch.abs(V)), min=torch.finfo(dtype).tiny)
    max_scaled = float((torch.max(torch.abs(residual)) / scale).detach().cpu())

    conditions = dict(conditions)
    conditions["max_scaled_foc_vs_closed_form_theta"] = theta_formula_error
    conditions_ok = (
        bool(conditions["vww_guard_inactive_on_reference"])
        and not bool(conditions["vww_guard_applied"])
        and not bool(conditions["theta_clipping_applied"])
        and theta_formula_error <= tolerance
    )
    status = "pass" if max_scaled <= tolerance and conditions_ok else "fail"
    return NumericalResult(
        stage="torch_current_linear_residual_pipinn",
        status=status,
        max_abs_residual=max_abs,
        max_scaled_residual=max_scaled,
        tolerance=tolerance,
        detail=(
            "AST-extracted frozen-policy residual with analytic raw theta*, "
            "without a V_ww guard or policy clipping"
        ),
        conditions=conditions,
    )


def _solver_list(value: str) -> tuple[str, ...]:
    return ("pinn", "pipinn") if value == "both" else (value,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("pinn", "pipinn", "both"), default="both")
    parser.add_argument("--risk-premium-mode", choices=("affine", "tanh"), default="affine")
    parser.add_argument("--nonaffine-eps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--numpy-tol", type=float, default=5.0e-12)
    parser.add_argument("--torch-tol", type=float, default=5.0e-11)
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise GateError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_ready) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validate_affine_request(args.risk_premium_mode, args.nonaffine_eps)
    if args.numpy_tol <= 0.0 or args.torch_tol <= 0.0:
        raise GateError("numerical tolerances must be positive")

    solvers = _solver_list(args.solver)
    contracts = [inspect_source_contract(SOURCE_BY_SOLVER[name], name) for name in solvers]
    batch = make_substitution_batch(args.seed, args.batch_size)
    numerical = [numpy_substitution(batch, args.numpy_tol)]
    for name in solvers:
        numerical.append(
            torch_substitution(
                batch, SOURCE_BY_SOLVER[name], name, args.torch_tol
            )
        )
        if name == "pipinn":
            numerical.append(
                torch_linear_substitution(
                    batch, SOURCE_BY_SOLVER[name], args.torch_tol
                )
            )

    failed = [item for item in numerical if item.status == "fail"]
    skipped = [item for item in numerical if item.status == "skip"]
    if args.require_torch and skipped:
        failed.extend(skipped)
    overall = "fail" if failed else ("pass_with_torch_skip" if skipped else "pass")
    payload = {
        "schema_version": 2,
        "gate": "liu_affine_residual_substitution",
        "affine_only": True,
        "risk_premium_mode": args.risk_premium_mode,
        "nonaffine_eps": args.nonaffine_eps,
        "ode_reference_defaults": {
            "rtol": ODE_RTOL,
            "atol": ODE_ATOL,
            "nodes": ODE_NODES,
        },
        "source_contracts": [asdict(item) for item in contracts],
        "results": [asdict(item) for item in numerical],
        "overall_status": overall,
    }
    return payload, 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, code = run(args)
        if args.json is not None:
            write_report(args.json, payload, args.overwrite)
    except (GateError, OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_ready))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
