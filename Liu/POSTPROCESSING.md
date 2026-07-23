# Liu paper post-processing

All commands below are read-only with respect to trained checkpoints. They
write separately named derived CSV/figure files and never resume training.
The scripts create no manuscript captions; captions remain in the paper's
LaTeX source.

The examples use the current ten training seeds. A different replication may
use any unique integer seed set, but the same explicit set should be passed to
every paper-facing aggregator.

## Market snapshot schema

Current training runs use the identity-block Brownian convention

\[
\rho_{\mathrm{HJB}}=\Psi^{-1/2}\rho_{\mathrm{raw}}\Phi_Z^{-1/2},
\qquad
\begin{bmatrix}I&\rho_{\mathrm{HJB}}\\
\rho_{\mathrm{HJB}}^\top&I\end{bmatrix}\succ0.
\]

The saved `rho` and `Gamma` are the coefficients actually used by the HJB.
Schema-2 snapshots additionally retain `Psi`, `Phi_Z`, `rho_raw`,
`rho_convention`, and the joint-covariance diagnostics so post-processors can
reconstruct and verify the transformation. No clipping, jitter, or
post-training snapshot repair is performed.

Runs trained before this convention change must not be mixed with schema-2
runs or evaluated with regenerated schema-2 coefficients. The launcher puts
`rho_canonical_v1` in automatic run tags, but a completely new output root is
still required for the new paper run because aggregators inspect every
completed run below the selected root.

```bash
SEED_SET='1,2,3,5,7,11,17,23,42,101'
MAIN_ROOT=/path/to/main_10seed
```

## 0. Freeze and audit the completed training artifacts

Before a long post-hoc evaluation, record a semantic checkpoint manifest in a
directory outside the training runs:

```bash
python3 Liu/audit_run_artifacts.py \
  --out-root "$MAIN_ROOT" \
  --expected-seeds "$SEED_SET" \
  --output "$MAIN_ROOT/derived/posthoc_audit" \
  --overwrite
```

The audit requires consistent success markers/status, the complete saved
outer-iterate schedule, and canonical tensor-state equality among
`value_net_final.pt`, `value_net_last.pt`, and the last iterate. It hashes
tensor contents rather than `torch.save` container bytes and never writes into
a run or weight directory. It also validates the identity-block innovation
covariance and, for schema-2 snapshots, reconstructs `rho` from the saved raw
correlation blocks. Canonical schema-2 metadata is required by default.
`--allow-legacy-market` exists only for exploratory audits of an older
already-valid snapshot; it must not be used to certify the former non-elliptic
\(M=5\) market.

## 1. Validate and aggregate final-error tables

```bash
python3 Liu/aggregate_seeds.py \
  --out-root "$MAIN_ROOT" \
  --expected-seeds "$SEED_SET" \
  --min-runs 10 \
  --expected-n-assets 30 \
  --expected-m-states 1,3,5 \
  --expected-models pinn,pipinn \
  --headline-margin 0.10
```

`summary_headline.csv` contains only `scope=fulldim`, margin `0.10`, and the
two headline metrics `RelL2_V` and `RelL2_theta`.  `summary_long.csv` keeps the
other robustness margins separate.  The command fails if the exact seed set,
required metrics, paper method/dimension cells, or canonical market snapshots
do not match.  `--expected-n-assets 30` prevents an internally consistent but
wrong asset dimension from passing.  `--expected-m-states` and
`--expected-models` are each exact, independent checks when supplied alone;
when both are supplied, every cell in their Cartesian product must appear
exactly once.

## 2. Aggregate E1 assumption diagnostics

The iteration-level ellipticity, curvature, Hamiltonian-numerator, and
normalized-control extrema already stored in `outer_history.csv` are reduced
within each seed first and only then summarized across seeds:

```bash
python3 Liu/aggregate_diagnostics.py \
  --out-root "$MAIN_ROOT" \
  --models pinn,pipinn \
  --m-states 1,3,5 \
  --expected-n-assets 30 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --output "$MAIN_ROOT/derived/e1_diagnostics" \
  --overwrite
```

PI-PINN rows retain the distinction between the frozen policy
`alpha_(n-1)` and improved policy `alpha_n`; direct-PINN rows are explicitly
labelled as greedy pseudo-outer diagnostics. The summary uses seed-level
extrema, sample SD, and Student-t 95% intervals, so outer iterations are not
treated as independent replications.

## 3. Create Figure 2 empirical convergence trajectories

Run the post-processor on the completed main myopic PI-PINN runs.  Pass the
actual seed IDs; they do not need to be consecutive.

```bash
python3 Liu/postprocess_contraction.py \
  --out-root "$MAIN_ROOT" \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --primary-margin 0.10 \
  --endpoint-outer 20 \
  --fig-width 4.8 \
  --fig-height 3.2 \
  --font-size 10 \
  --dpi 300 \
  --formats png,eps
```

The paper output is one Liu/Kim--Omberg panel.  It places
`diag_RelL2_V` (Value, blue) and `diag_RelL2_vartheta` (Policy, orange), where
`vartheta=theta/w`, on the same logarithmic y-axis.  Thick curves are
pointwise arithmetic seed means
and the matching shaded regions are sample mean +/- one sample SD.  The
legend contains only `Value` and `Policy`.  Individual seed curves, the gray
early window, fitted-rho annotations, and separate `Seed mean` / `Mean +/- SD`
legend entries are absent.  Their statistical meaning belongs in the LaTeX
caption.

`diag_RelL2_theta` is retained in new training logs only as a legacy raw-dollar
control diagnostic.  Because its pointwise norm weights normalized-control
errors by wealth squared, it is not used in Figure 2 and must not be relabeled
as `vartheta`.

Runs completed before `diag_RelL2_vartheta` was added can be repaired without
retraining when every outer checkpoint was saved:

```bash
python3 Liu/reconstruct_vartheta_trajectory.py \
  --out-root "$MAIN_ROOT" \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --device cuda:0 \
  --output "$MAIN_ROOT/derived/figure2_vartheta_trajectory" \
  --overwrite

python3 Liu/postprocess_contraction.py \
  --out-root "$MAIN_ROOT" \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --vartheta-trajectory-csv \
    "$MAIN_ROOT/derived/figure2_vartheta_trajectory/figure2_vartheta_per_outer.csv" \
  --overwrite
```

The reconstruction uses the newest-attempt-before-status-filter run selection,
requires a successful affine PI-PINN run, reproduces the original deterministic
diagnostic seed `market_seed*1000003+7`, and requires the exact checkpoint grid
`1,...,outer_iters`.  It cross-checks both the saved value error and legacy raw
policy error, verifies the final outer tensor state against
`value_net_final.pt`, and binds every derived row to the source config, outer
history, market, closed-form file, and checkpoint hashes.  Source run files are
never modified.

`figure2_endpoint_summary.csv` reports the seed-mean errors at outer 1 and
`--endpoint-outer` (20 by default), together with their transparent reduction
factor `mean(E_1) / mean(E_20)`.  This is deliberately a ratio of seed means,
not the mean of seed-wise ratios, and is the preferred statistic for the main
text.  The optional early-window fits remain in the decay CSVs for supplement
or diagnostics only; they do not appear in the figure.

Add `--include-val-pres` only for a supplementary third curve.  Here `p_res`
combines the held-out PDE RMS and terminal-mismatch RMS; it is off in the
paper figure.  `--show-seed-trajectories` is also a diagnostic opt-in and is
off by default.

Figure width/height are in inches (`--fig-width`, `--fig-height`),
`--font-size` is in points, and `--dpi` controls raster resolution.
`--formats` accepts any comma-separated subset of `png,pdf,svg,eps`; the
default is `png,pdf`.  EPS uses light opaque versions of the SD-band colors
because PostScript has no alpha transparency; PDF and SVG preserve the
translucent shading.
`--font-family 'Times New Roman'` can select an installed paper font.  The
default `--bbox-inches tight` trims surrounding whitespace; use
`--bbox-inches standard` when exact `figsize * dpi` pixel dimensions are
required.

For every metric and every requested early window, the diagnostic CSV path fits
`log(e_n) = alpha + n log(rho)` **inside each seed first**.  It then reports the
mean, sample SD, and Student-t 95% CI of the seed-wise `rho` estimates.  It
never fits the seed-mean curve.  The primary window is `1-4`; the defaults also
write the `1-3` and `1-5` sensitivity results.

These are **early-phase empirical decay factors in relative L2 error**.  They
are not estimates of the theorem's `X_ev`-norm contraction factor.  In
particular, the script does not read `e_Xev`, does not form pointwise
`e_(n+1)/e_n` ratios, and does not claim a bound below one.  The optional
`val_pres` slope is a residual-level decay diagnostic, not a PI contraction
factor.

The script requires `diag_every=1`, complete positive finite diagnostics at
every outer iteration, the exact requested seed set, and identical canonical
market snapshots.  It fails rather than silently intersecting incomplete
histories.  If more than one eligible training configuration is present, use
`--run-name-regex` or `--group-id` to select it.

Outputs include `figure2_empirical_convergence.{png,pdf,svg,eps}` for the
requested formats,
`figure2_trajectories.csv`, `figure2_pointwise_summary.csv`,
`figure2_endpoint_summary.csv`,
`figure2_seed_decay_fits.csv`, `figure2_decay_summary.csv`,
`figure2_runs_used.csv`, and `figure2_metadata.json`.  The figure has no title
or caption; the paper's LaTeX supplies the caption.  A nonempty output
directory is rejected by default; use `--overwrite` to replace files owned by
this post-processor without leaving a stale panel or legacy ratio.  Unrelated
entries such as `.ipynb_checkpoints` are retained and do not block overwrite.

## 4. Evaluate E9 value/bundle/control errors on nested windows

E9 is a separate deterministic evaluation of the official final affine
checkpoints. The same base design is remapped to every requested margin:

```bash
python3 Liu/evaluate_margin_bundle.py \
  --out-root "$MAIN_ROOT" \
  --models both \
  --n-assets 30 \
  --m-states 1,3,5 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --margins 0.05,0.10,0.20,0.30 \
  --n-points 100000 \
  --base-seed 727 \
  --device cuda:0 \
  --strict-crosscheck \
  --output "$MAIN_ROOT/derived/e9_margin_bundle" \
  --overwrite
```

For each margin it reports relative-L2 and sup errors for value, the reduced
derivative bundle `(V_w,V_ww,V_wx)`, and normalized control
`vartheta=theta/w`, together with componentwise bundle sup errors and guard
fractions. It requires `value_net_final.pt`, verifies market/closed-form
identity, and cross-checks matching value/policy rows already present in
`metrics.csv`. Seed subsets are exploratory via `--seeds`; the paper command
uses `--expected-seeds` as an exact set. Run attempts are deduplicated before
status filtering, so a newer failed rerun invalidates an older success rather
than silently resurrecting it. In strict mode every requested comparison must
exist, use the same 100,000-point/seed-727 design, and agree numerically;
`mismatch`, `not_available`, and `not_comparable_design` all fail.

## 5. Validate and evaluate CE/WL

Run the cheap provenance check first:

```bash
python3 Liu/evaluate_welfare.py \
  --out-root "$MAIN_ROOT" \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds "$SEED_SET" \
  --validate-only
```

Then run the projected-extension main evaluation in the training environment
with PyTorch and the desired GPU:

```bash
python3 Liu/evaluate_welfare.py \
  --out-root "$MAIN_ROOT" \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds "$SEED_SET" \
  --device cuda:0
```

The defaults are `w0=0.5`, `x0=xbar`, 100,000 paths, 1,000 log-wealth Euler
steps, and `mc_seed=2718`.  Add `--include-raw` for the unprojected sensitivity
run.  Official paper runs require `value_net_final.pt`; legacy checkpoint
fallback is available only through the explicit exploratory flag
`--allow-checkpoint-fallback`.

Long evaluations resume safely by default.  The resume signature covers the
protocol, selected run groups, canonical markets, closed-form files, evaluator
implementation, and final-checkpoint content hashes; completed seed/extension
rows and the per-dimension optimal-path cache are reused only after that
signature matches.  `--no-resume` deliberately discards partial derived
outputs and starts over.  The requested device spelling is not part of the
signature, so a torch-free `--validate-only` call using `auto` can be followed
by the actual `--device cuda:0` run; once simulation starts, the effective
device and NumPy/PyTorch versions are recorded and protected against mixing.

Because `--include-raw` changes the requested output set, decide it before a
long run.  To add raw sensitivity later without replacing the projected-main
directory, use a separate `--output` directory (that sensitivity job will
produce both projected and raw rows).  Run only one evaluator process per
`--output` directory at a time; resume protects sequential restarts, not
concurrent writers.

The evaluator writes `welfare_metrics.csv` per training seed,
`welfare_seed_summary.csv` with seed-level mean/sample-SD/Student-t intervals,
and `welfare_validation.csv` comparing the optimal Euler Monte Carlo CE with
the exact continuous-time closed-form CE.  `welfare_config.json` records the
full provenance/signature, while `optimal_paths_M*.npz` stores the paired CRN
reference paths used for resumable evaluation.  WL uses the paired
common-random-number delta-method standard error and is never clipped at zero.

This evaluator is affine-only: any nonzero `nonaffine_eps`, including a tiny
one, is rejected rather than rounded to zero.

## 6. Post-process the non-affine perturbation experiment

The paper perturbation curves pair each epsilon run with the same-seed affine
baseline before averaging:

```bash
python3 Liu/postprocess_nonaffine.py \
  --out-root /path/to/nonaffine_n30_m3 \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --eps 0.1,1,2,3,4,5 \
  --device cuda:0
```

`epsilon=0` need not appear in `--eps`; the script locates and adds the unique
baseline automatically. Paper mode uses only `value_net_final.pt`. The
`--allow-checkpoint-fallback` option is explicitly legacy/exploratory. For a
single-seed pilot, sample SD is undefined and remains `NaN` rather than being
reported as zero.

## 7. Run and aggregate E8 computation measurements

E8 requires a separate timing profile because ordinary main runs include
diagnostics and iterate checkpoint I/O. For the previously planned one-seed
measurement:

```bash
SWEEP_PROFILE=timing \
TIMING_M_STATES_LIST='1,3,5' \
SEEDS='1' \
DEVICE_LIST='cuda:0' \
bash Liu/tune_pipinn.sh /path/to/liu_timing
```

The timing profile runs both PINN and PI-PINN, evaluates the final models, and
automatically calls `aggregate_compute.py`. The paper wall-clock field is the
CUDA-synchronized `core_train_wall_sec`; final checkpoint serialization is
excluded. End-to-end `train_wall_sec` remains in `status.json` as a separate
observation. Optimizer steps and both training/evaluation peak allocated GPU
memory are also retained.

To rerun only the aggregation:

```bash
python3 Liu/aggregate_compute.py \
  --out-root /path/to/liu_timing \
  --models pinn,pipinn \
  --m-states 1,3,5 \
  --n-assets 30 \
  --expected-seeds 1 \
  --min-runs 1 \
  --formats png,pdf \
  --overwrite
```

Paper mode rejects incomplete CUDA/software identity, mismatched success
status, missing evaluation memory, mixed GPU capacities, and mixed markets.
With one seed, SD and CI fields are `NaN`; use multiple timing seeds if the
paper will report timing variability rather than a single-run measurement.

## 8. Optional Liu M=1 exact PI-map and E4 FD audit

For each saved PI-PINN checkpoint `k`, the Liu evaluator differentiates the
network to form the greedy policy `alpha_k`, freezes that policy, and solves
the two-dimensional `(log wealth, factor)` linear PDE by finite differences.
The exact-map numerator is therefore `E(alpha_k)`, not neural checkpoint
`k+1`. The same solves provide the shifted E4 reference
`E(alpha_(k-1))` for checkpoint `k`.

Run a full one-seed audit only for an affine `M=1` PI-PINN run with all outer
checkpoints:

```bash
python3 Liu/liu_exact_map_fd.py \
  --run-dir /path/to/M1_seed_run \
  --output /path/to/derived/exact_map_seed1 \
  --device cuda:0 \
  --grid-factors 1,2 \
  --domain-factors 1.5,2.0 \
  --boundaries linearity,exact-dirichlet \
  --verify-checkpoints all
```

After running every seed, aggregate only common, fully checked schedules:

```bash
python3 Liu/aggregate_liu_exact_map.py \
  --out-root /path/to/derived/exact_map_all_seeds \
  --expected-seeds "$SEED_SET" \
  --min-seeds 10 \
  --output /path/to/derived/exact_map_paper \
  --overwrite
```

The norm is identical in numerator and denominator:
`sup|V| + sup sqrt(V_w^2+V_ww^2+V_wx^2)`, with derivatives converted back to
the original wealth coordinate. Grid, enlarged-domain, boundary, ellipticity,
Peclet, guard, clipping, and linear-solve diagnostics are preserved. A guard
or clip activation is labelled as the implemented modified map, and no
finite-domain result is promoted to a whole-space proof. See
`Liu/LIU_EXACT_MAP.md` for the indexing and numerical protocol.

For paper-facing worst-case reporting, use `exact_map_worst_per_seed.csv` and
`exact_map_worst_summary.csv`. They contain each seed's maximum primary ratio
and maximum sensitivity envelope, seedwise-max mean/sample-SD/Student-t 95%
CI, and the global maximum with its seed/outer location. The aggregate status
sets `finite_domain_all_tested_ratios_below_one` only if all denominators are
defined, all requested sensitivity checks pass, every sampled seed/outer map
is locally unmodified, and both the global primary exact ratio and global
sensitivity envelope are below one. A passing envelope is also validated to
be no smaller than its corresponding primary ratio.
This supports only a statement about the tested finite-domain sampled-map
audit, not a whole-space contraction claim.

This exact-map audit supplements, and does not replace, Figure 2's empirical
relative-L2 trajectories. The Merton exact-map implementation remains a
separate diagnostic under `merton_ND/EXACT_MAP.md`.

## 9. Check affine closed-form substitution in the current residual

Before relying on the affine reference, run the standalone substitution gate
in the same environment used for training:

```bash
python3 Liu/check_residual_substitution.py \
  --solver both \
  --risk-premium-mode affine \
  --nonaffine-eps 0 \
  --require-torch \
  --json /path/to/derived/liu_residual_substitution.json \
  --overwrite
```

Stage 1 independently substitutes the exponential-quadratic derivatives and
Riccati right-hand side into the HJB using NumPy. Stage 2 uses autograd and the
actual `hjb_residual_nd` functions extracted from the current PINN and PI-PINN
sources. For PI-PINN, Stage 3 separately extracts
`linear_pde_residual_nd` and verifies
\(T[V^*,\theta^*]\simeq0\) with the analytic raw policy
\(\theta^*=-(\lambda V_w+\Gamma V_{wx})/V_{ww}\). The policy passed to this
operator is raw \(\theta\), not \(\theta/w\), and neither a `V_ww` guard nor
policy clipping is applied. The report records those conditions and verifies
that the analytic reference lies outside the guard region.

Importing the training modules is deliberately avoided because they parse CLI
arguments and construct runs at module import time. The extraction namespace
explicitly supplies `actual_risk_premium_torch` and, for the nonlinear HJB,
`safe_concave_vww`, so a missing residual dependency cannot be mistaken for a
mathematical failure. The PI-PINN source contract also fixes the linear
operator signature and rejects a guard or clip inside that frozen-policy
operator.

This is strictly an affine gate: `mode=tanh`, even with epsilon zero, and any
nonzero epsilon are rejected. Without PyTorch, Stage 1 still runs and Stage 2
is reported as `skip`; `--require-torch` converts that skip into a failing exit
status for a paper audit. The source contract also verifies the current
closed-form ODE reference defaults: `rtol=1e-12`, `atol=1e-14`, and 8,001
saved time nodes. These tests are local residual/implementation consistency
checks; they do not replace a saved-grid interpolation/terminal-condition
audit or an FD exact-map calculation.

## 10. Aggregate E4 across residual-tolerance cells

Once every tolerance/seed cell has both a successful training run and a
successful full Liu FD audit, aggregate the existing E4 artifacts without
rerunning the solver. As with the underlying exact-map evaluator, this path is
currently restricted to the affine `M=1` reference problem:

```bash
python3 Liu/aggregate_e4_tolerance.py \
  --out-root /path/to/derived/exact_map_residual_sweep \
  --expected-tolerances 1e-2,3e-3,1e-3 \
  --expected-seeds 1,2,3,5,7,11,17,23,42,101 \
  --min-runs-per-tolerance 10 \
  --checkpoints all \
  --output /path/to/derived/e4_tolerance_paper \
  --plot --formats png,pdf \
  --overwrite
```

`e4_tolerance_per_seed.csv` first takes the worst E4 value, derivative-bundle,
and combined X-norm error over the requested outer checkpoints. Control error
is included automatically only if the FD artifacts actually provide it.
Source-policy ellipticity extrema and their outer locations remain visible.
`e4_tolerance_summary.csv` then reports mean, sample SD, and Student-t 95% CI
across the seedwise extrema for each tolerance. The optional log-log plot uses
the achieved held-out residual, not merely the requested target.

The aggregator validates the exact-map artifact manifest and provenance,
requires passing refinement and ellipticity checks, enforces a common market
snapshot, balanced seed support, and identical training/FD protocols after
removing only `pres_target`. Exact expected seed and tolerance sets should be
supplied for paper output. A discovered failed exact-map attempt is fatal;
failed attempts are never filtered out in a way that can revive an older
success. The resulting E4 table is an approximation-versus-residual audit,
not an exact-map contraction-ratio claim.

## 11. Aggregate a Liu residual-tolerance sweep (E6)

`aggregate_e6.py` is for final relative-L2 error versus the **achieved** held-out
residual level. It does not substitute for E4's FD approximation-error audit.
For a balanced paper sweep, state both sets explicitly:

```bash
python3 Liu/aggregate_e6.py \
  --out-root /path/to/liu_residual_sweep \
  --model-type pipinn \
  --expected-n-assets 30 \
  --expected-m-states 3 \
  --expected-seeds "$SEED_SET" \
  --expected-tolerances 1e-2,3e-3,1e-3 \
  --min-runs-per-tolerance 10 \
  --metrics RelL2_V,RelL2_theta \
  --formats png,pdf \
  --overwrite
```

The aggregator selects the newest attempt for each
`(configuration,tolerance,seed)` before checking status, requires the exact
successful seed and tolerance sets when supplied, and verifies one canonical
market snapshot across the sweep. A newer failed rerun therefore cannot revive
an older success. `per_target.csv` reports seed mean, sample SD, SEM, and
Student-t 95% CI; one seed leaves SD and CI undefined rather than writing zero.
`fit.csv` includes seed-cluster-robust, ordinary OLS, and tolerance-mean log-log
fits. The generated figures show seed points and tolerance-wise mean +/- one SD
against achieved `p_res`; no caption is embedded.

`--overwrite` replaces only artifacts owned by this script and preserves
unrelated entries such as notebook checkpoints. Validation finishes before a
staged result replaces the previous successful output.
