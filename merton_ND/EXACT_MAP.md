# Merton exact PI-map diagnostic

`merton_exact_map_fd.py` evaluates the exact policy-iteration map independently
of the neural policy-evaluation step. For each saved PI-PINN value checkpoint
it extracts the trainer's greedy policy, freezes that policy, solves the
resulting linear PDE by finite differences, and reports

\[
\widehat\rho_n^{\mathrm{exact}}
=
\frac{\|E(G(\widetilde v_n))-V^*\|_{X_{\mathrm{ev}}}}
     {\|\widetilde v_n-V^*\|_{X_{\mathrm{ev}}}}.
\]

The numerator is the FD solution of the frozen-policy PDE. It is never the
next learned neural iterate, so these files remain separate from the empirical
combined contraction--perturbation diagnostic.

## Checkpoint indexing

The trainer numbers completed policy-evaluation outer loops from one, whereas
the paper numbers the corresponding value iterates from zero.

| Saved artifact | Value represented | Greedy policy evaluated by this tool |
| --- | --- | --- |
| `value_net_iter0001.pt` | `v_tilde_0 = E(alpha_0)` | `alpha_1 = G(v_tilde_0)` |
| `value_net_iter0002.pt` | `v_tilde_1 = E(alpha_1)` | `alpha_2 = G(v_tilde_1)` |
| `value_net_iterKKKK.pt` | `v_tilde_(K-1)` | `alpha_K = G(v_tilde_(K-1))` |

Accordingly, every output row records all three fields:
`checkpoint_outer_iter=K`, `source_iter=K-1`, and `target_policy_iter=K`.
The final checkpoint can still be mapped even though its resulting greedy
policy is not used by another training outer loop.

E4 also solves the configured initial policy
`pi_0=pi_init_scale*pi_myopic`, `c_0=rho_discount*W` independently and compares
it with `value_net_iter0001.pt`, producing `defect_iter=0`. Adjacent
checkpoint pairs then produce `defect_iter=1,...,N-1`. Thus a complete
1--N checkpoint schedule yields exactly N defect rows. Every row is attached
to outer `n+1`'s official post-restore fixed-Q_res value.

Defect refinement is not inferred from the exact-map-ratio audit. The
evaluator separately writes `exact_map_defect_refinement.csv`, recomputing
each selected \(\delta_n\) over the FD grid/domain/boundary variants while
holding the next neural bundle fixed. `delta_0` is always audited; adjacent
defects follow `--verify-checkpoints` (paper default: `all`). The E4
postprocessor requires passing evidence for `delta_0`, the first and last
adjacent defects, and the defect attaining the run-wise maximum
\(\widehat p_X\).

## Producing exact-map and E4 training artifacts

The empirical paper Figure 2 is now produced from relative L2 outer
trajectories by `postprocess_contraction.py`; it does not use this FD ratio.
The exact-map result is a separate numerical experiment.

For the complete E4 maximum over every policy-evaluation error, save every
PI-PINN outer iterate and keep the sparse E3b schedule off:

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
N_ASSETS_LIST="10,50" \
PIPINN_OVERRIDES="e3b_checkpoints=false save_iterate_every=1" \
bash tune_merton.sh /path/to/merton_sweep 2
```

`e3b_checkpoints=true` remains available for the older sparse schedule
(outers 1--10, every tenth, and final), but that schedule is insufficient for
the all-iteration E4 maximum. A checkpoint is taken after policy evaluation
and after optional held-out model+optimizer restoration. The manifest records
this timing and the paper-index offset.

The launcher stores run data and weights separately:

```text
/path/to/merton_sweep/
  pipinn/<run-tag>/
    config.json
    market_params.npz
    status.json
    train_history.csv
    outer_history.csv
    metrics.csv
    plots/
  weights/pipinn/<run-tag>/
    checkpoint_manifest.json
    value_net_final.pt
    value_net_last.pt
    value_net_best_diag.pt
    iterates/value_net_iterNNNN.pt
```

`config.json` records the network coordinate/order/activation/dtype, resolved
policy bounds, current guard contract, initialization scale, selection and
resampling protocol, and trainer source hash. `market_params.npz` includes the
actual market arrays and `Sigma_inv_mu`. The manifest stores both file hashes
and canonical tensor-state hashes. The exact-map loader verifies the complete
schedule and verifies that the official final, last, and final-iterate files
represent the same tensor state; it does not assume independently serialized
PyTorch files have identical bytes.

Current launchers canonicalize the sweep root before recording paths. For
legacy runs that stored an `OUT_ROOT`-relative `weight_dir`, the loader first
resolves that path against the `cwd` recorded in `config.json`, then checks
run-directory and invocation-directory compatibility fallbacks. Thus a
relative path such as `outputs/sweep/weights/...` is not incorrectly appended
to the already nested run directory.

Legacy checkpoints without `policy_guard_mode` are not guessed. They require
an audited `--policy-mode` override; this prevents an older sign-preserving
guard from being silently evaluated as the current one-sided map.

Starting a new same-tag training run archives old logs and checkpoint
trajectories with timestamped names. The launcher also rejects duplicate
resolved run tags before parallel workers can write the same directory.

## Running the exact map

PyTorch is required to differentiate a saved network. NumPy and SciPy perform
the independent frozen-policy solve.

Single run:

```bash
python3 merton_exact_map_fd.py \
  --run-dir /path/to/merton_sweep/pipinn/<run-tag> \
  --device cuda:0 \
  --base-ny 401 \
  --base-nt 400 \
  --grid-factors 1,2 \
  --fd-margins=-1.0,-0.5 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all
```

Using a smaller verification subset remains possible for exploratory
exact-map ratios, but the E4 paper aggregation will fail if that subset omits
one of its required defect-refinement iterations.

Multiple runs discovered below a sweep root can be selected with
`--out-root` and `--run-name-regex`. Seed completeness is not forced by
default. Pass
`--expected-seeds "1,2,3,5,7,11,17,23,42,101"` explicitly for the paper
sweep; smaller pilots can use their own list or omit the option.

Before a large run, execute the PyTorch-free manufactured checks:

```bash
python3 merton_exact_map_fd.py --self-test
```

The self-test includes both the closed-form optimal frozen policy and an
independent nonoptimal constant-proportional policy
`pi(t,w)=pi_0`, `c(t,w)=chi_0 w` with non-unit bequest. The second oracle
reduces the PDE to a scalar homothetic ODE and therefore checks the frozen
source/drift signs, Robin closure, and remaining-time orientation without
reusing the optimal first-order conditions.

## Trainer map reproduced by the evaluator

For the current `trainer-one-sided` / `merton-logw-v1` contract, the evaluator
reconstructs the implemented greedy map exactly:

\[
V_w=e^{-y}v_y,
\qquad
c_{\rm raw}=\max(V_w,\varepsilon_c)^{-1/\gamma},
\]

followed by the recorded `kappa=c/W` and consumption-level bounds, and

\[
\pi_{\rm raw}
=
\frac{\max(v_y,\varepsilon_{\rm num})}
     {\max(v_y-v_{yy},\varepsilon_{\rm den})}
\Sigma^{-1}\mu,
\]

followed by the recorded optional componentwise portfolio bounds. The default
`pi_clip_abs=2` means `[-2,2]`; `pi_clip_abs=none` removes only the portfolio
projection.  The global `policy_bounds_mode=none` audit setting removes every
finite portfolio, kappa, and consumption-level projection at once.  It does
not remove the one-sided derivative guards, whose activation is reported
separately.
The synthetic-market argument `kappa_max=30` is not confused with the policy
bound `kappa_max_bound=3`.

Guard and clipping fractions are reported on both the evaluation grid and the
full enlarged FD domain, including separate portfolio numerator and curvature
guards, positive curvature, kappa clipping, consumption-level clipping, and
portfolio clipping. Guards modify only policy extraction `G`; the resulting
fixed coefficients are then used directly in the linear PDE. Rows with any
sampled activation are labelled as a modified implemented map rather than an
unconstrained theoretical map.

The paper default is `policy_bounds_mode=stabilized`: finite safety bounds are
retained and their raw activation fractions are reported.  A fully unbounded
action-box robustness run can be launched with
`PIPINN_OVERRIDES="policy_bounds_mode=none"`; the resolved null bounds are
stored in both `config.json` and the checkpoint manifest and are verified by
the exact-map loader.

## Manuscript norm

Both numerator and denominator use the wealth-coordinate norm from the
manuscript. Candidate and reference derivatives are initially available in
log wealth and are transformed by

\[
e_w=e^{-y}e_y,
\qquad
e_{ww}=e^{-2y}(e_{yy}-e_y),
\]

then

\[
\|e\|_{X_{\rm ev}}
=
\sup|e|
+
\sup\sqrt{e_w^2+e_{ww}^2}.
\]

The raw `e_y` and `e_yy` sup errors remain in CSV files only as numerical
diagnostics. Ratios are never clipped. A denominator below the configured
tolerance is marked undefined.

## FD and sensitivity interpretation

The frozen PDE is solved forward in remaining time with a theta-method
tridiagonal scheme. The compatibility option `--rannacher-steps` denotes
initial full-step backward-Euler damping, not classical two-half-step
Rannacher smoothing. The primary boundary is the homogeneous CRRA Robin
condition `u_y=(1-gamma)u`. The compatibility label `exact-dirichlet` means
an **optimal-reference Dirichlet sensitivity audit**: it injects the
closed-form optimal value `V*` at the lateral boundary. It is exact for the
optimal-policy manufactured check, but it is not an exact boundary oracle for
a nonoptimal frozen neural policy and must not be used as the paper primary.
This distinction is emitted in `exact_map_protocol.json` and in each
refinement CSV row's `boundary_semantics` field.

The primary evaluation window follows the trainer's calendar-time `[0,T)`
convention: the terminal face `t=T` is excluded, while `t=0` is retained.
This prevents a neural terminal residual from entering only the denominator
when the FD numerator enforces the terminal condition exactly. The margin
shrinks only the log-wealth axis. Negative FD margins enlarge the spatial domain beyond
the training interval. The program reports grid, domain, and boundary
sensitivity, Péclet/upwind diagnostics, linear residuals, and both
`pi^T Sigma pi` and the PDE coefficient `0.5*pi^T Sigma pi`.

`rho_sensitivity_envelope` is the primary ratio plus observed changes across
the requested numerical variants. It is a transparent sensitivity diagnostic,
not a rigorous discretization bound and not a proof about the whole-space map.
Every row therefore states
`whole_space_map_claim=not_verified_by_finite_domain`.

## Aggregation

After all seed runs have successful exact-map outputs:

```bash
python3 merton_exact_map_fd.py \
  --aggregate-only \
  --out-root /path/to/merton_sweep \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10
```

The separate E4 residual-transfer aggregation is:

```bash
python3 postprocess_regularity_transfer.py \
  --out-root /path/to/merton_residual_sweep \
  --n-assets 50 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --formats png,pdf \
  --overwrite
```

Aggregation checks the seed set, fixed market, checkpoint support, training
group, and FD/evaluation protocol. Ratios are computed within each seed before
the mean, sample standard deviation, and Student-t 95% interval are formed.
The seed-set equality check is enabled only when `--expected-seeds` is
nonempty; `--allow-incomplete` then controls whether a mismatch is tolerated.
Independently, `--min-seeds` controls the minimum usable seed count per group
and defaults to 2 for the seed-level uncertainty summary.
The paper default `--floor-multiple 0` retains every finite checkpoint. A
positive value enables an exploratory cutoff relative to a late neural
input-error scale; that scale is explicitly not an FD discretization floor and
should not be used to hide late non-contraction.

Aggregation also preserves the minimum and maximum `pi^T Sigma pi` across
seeds and the full regular region. By default, any sampled value at or below
`--ellipticity-tolerance` (default zero) is rejected. A positive tolerance can
state a stronger numerical lower bound. `--allow-degenerate-diffusion` is an
exploratory override only.

Use `--require-locally-unmodified-map` only when an activation-free sampled
map is required. Without it, guarded/clipped results remain in the aggregate
with explicit activation maxima and map-status labels. `--allow-unverified`,
`--allow-incomplete`, `--allow-degenerate-diffusion`, and a positive
`--floor-multiple` are exploratory overrides and should not be used for the
paper sweep.

Outputs:

```text
<run>/exact_map_fd/
  exact_map_ratios.csv
  exact_map_refinement.csv
  exact_map_defects.csv
  evaluated_bundles/*.npz
  exact_map_config.json
  exact_map_status.json
  _SUCCESS_EXACT_MAP

<sweep>/exact_map_paper/
  exact_map_ratios_by_seed.csv
  exact_map_ratio_summary.csv
  exact_map_floor_summary.csv
  exact_map_worst_summary.csv
  exact_map_contraction.{png,pdf,svg,eps}
```

For E4 residual-transfer evidence, run `postprocess_regularity_transfer.py`
after exact-map evaluation. It refuses sparse checkpoint schedules, legacy
pre-restore residual semantics, missing or hash-mismatched evaluated bundles,
incomplete defect indices, and mixed FD protocols.
