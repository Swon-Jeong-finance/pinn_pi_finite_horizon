# Liu affine exact PI-map and E4 diagnostic

`liu_exact_map_fd.py` is an independent finite-difference policy-evaluation
audit for the one-factor affine Liu/Kim--Omberg experiment. It is a post-hoc
calculation: it reads completed PI-PINN checkpoints and does not retrain or
modify the neural networks.

This diagnostic is intentionally separate from the empirical relative-
\(L^2\) convergence figure. The latter follows the learned iterates, whereas
the exact-map numerator below is the result of an independent frozen-policy FD
solve.

## Objects and checkpoint indexing

Write \(G\) for greedy policy improvement, \(E\) for exact frozen-policy
evaluation, and \(P=E\circ G\). With the indexing used by the Liu trainer,

\[
\widetilde v_k \approx E(\alpha_{k-1}),
\qquad
\alpha_k=G(\widetilde v_k).
\]

Thus `value_net_iter000k.pt` is the learned evaluation of
\(\alpha_{k-1}\). For that checkpoint the FD program extracts \(\alpha_k\),
freezes it, and computes

\[
u_k^{\mathrm{FD}}\approx E(\alpha_k)=P(\widetilde v_k).
\]

The exact-map diagnostic is

\[
\widehat\rho_k^{\mathrm{exact}}
=
\frac{\|u_k^{\mathrm{FD}}-V^*\|_{X_{\mathrm{ev}}}}
     {\|\widetilde v_k-V^*\|_{X_{\mathrm{ev}}}}.
\]

The numerator is **not** checkpoint \(k+1\). Accordingly, the primary CSV
records all of `source_outer_iter=k`, `greedy_policy_iter=k`, and
`target_value_outer_iter=k+1`. Its `frozen_policy_iter=k-1` field records the
policy that produced the source checkpoint during training; it is not the
policy used in the new exact-map solve.

The same solves also produce the shifted E4 evaluation-error diagnostic

\[
e_k^{\mathrm{approx}}
=\|\widetilde v_k-E(\alpha_{k-1})\|_{X_{\mathrm{ev}}}.
\]

For \(k=1\), the program constructs the configured initial policy
\(\alpha_0\) and performs an additional FD solve. For \(k\ge 2\), it reuses
the exact-map solution generated from checkpoint \(k-1\); it does not solve
the same frozen-policy equation twice. This is why every training outer
iterate must have a checkpoint. An exploratory `--checkpoints` selection, if
used, must be a contiguous prefix `1,...,k`.

## Frozen-policy equation

The implementation does not import the Merton numerical core. It solves the
Liu equation on the two-dimensional spatial domain \((y,x)\), even though the
factor dimension is \(M=1\). Here \(y=\log w\), remaining time is \(\tau\),
and the frozen normalized risky position is
\(\vartheta=\theta/w\in\mathbb R^N\). For a supplied \(\vartheta(\tau,y,x)\),

\[
\begin{aligned}
u_\tau={}&\tfrac12\|\vartheta\|^2u_{yy}
+(\vartheta^\top\Gamma)u_{yx}
+\tfrac12Q u_{xx}\\
&+\left(r+\vartheta^\top\lambda(x)
-\tfrac12\|\vartheta\|^2\right)u_y
+(k_0-Kx)u_x,
\end{aligned}
\]

with the saved CRRA terminal utility at \(\tau=0\). The mixed derivative is
retained with a centered nine-point stencil. Time stepping uses a sparse
theta scheme (Crank--Nicolson by default) and full backward-Euler startup
steps. Drift differencing can be `central`, `monotone`, or `adaptive`; the
adaptive default switches according to the sampled cell Peclet number.

No artificial diffusion, curvature guard, or ellipticity repair is inserted
into this linear PDE. The code reports the sampled joint-diffusion eigenvalues,
nonpositive-eigenvalue fraction, Peclet numbers, upwind fractions, and linear
solve residual so that numerical degeneracy is visible.
`--ellipticity-tolerance` is a hard lower-bound check. Its default zero
requires strict positivity and therefore rejects a degenerate frozen policy
without changing the sampled operator. In particular, a zero initial policy
cannot be silently presented as satisfying the ellipticity assumption.

The affine closed-form reference \(V^*\) is recomputed with an independent
one-factor Riccati solve. The run must contain a successful, well-formed
`closed_form_ode.npz`; its saved and recomputed coefficients are checked for
agreement before any FD work.

The driver also validates the market's identity-block Brownian correlation.
For current schema-2 runs it reconstructs
\(\rho=\Psi^{-1/2}\rho_{\rm raw}\Phi_Z^{-1/2}\) from the saved source blocks
and rejects any mismatch. Legacy snapshots remain readable only when their
saved identity-block covariance is already strictly positive; a checkpoint is
never paired with post-hoc modified market coefficients.

## Norm and original-wealth derivatives

The same policy-relevant norm is used in the exact-map numerator,
denominator, and E4 error:

\[
\|e\|_{X_{\mathrm{ev}}}
=
\sup_{X_{\mathrm{ev}}}|e|
+
\sup_{X_{\mathrm{ev}}}
\sqrt{e_w^2+e_{ww}^2+e_{wx}^2}.
\]

Although the FD solve is performed in log wealth, the derivative bundle is
converted back to the original wealth coordinate before taking the norm:

\[
V_w=e^{-y}u_y,
\qquad
V_{ww}=e^{-2y}(u_{yy}-u_y),
\qquad
V_{wx}=e^{-y}u_{yx}.
\]

The CSV retains value, \(V_w\), \(V_{ww}\), \(V_{wx}\), pointwise bundle,
and total \(X\)-norm components separately. The derivative-bundle term is the
supremum of the pointwise Euclidean norm, not the sum of three independently
attained suprema.

Ratios above one are preserved. If the denominator is nonfinite or no larger
than `--denominator-tolerance`, `rho_exact` is written as undefined and the
row is marked `undefined_denominator`; it is never regularized, replaced, or
silently dropped. Always inspect `all_denominators_defined` and
`undefined_denominator_outers` in `exact_map_status.json` before using a run.

## Evaluation window, FD domain, and boundaries

`X_ev` uses the first saved `eval_margin` unless `--eval-margin` is supplied.
The saved wealth and factor intervals are shrunk in their original
coordinates; the resulting wealth endpoints are then transformed to log
wealth. Time zero, where every checkpoint satisfies the same terminal
condition, is excluded from the tensor-grid norm.

The FD rectangle is centered on the saved training interval in \((y,x)\),
with each half-width multiplied by `--domain-factors`. Every factor must be
strictly larger than one, so the FD solve surrounds the evaluation window.
Increasing a domain factor also increases the grid-point count to keep the
spacing approximately fixed at a given refinement factor.

Two boundary closures are available:

- `linearity` uses \(V_{ww}=0\), equivalently \(u_{yy}-u_y=0\), on the
  log-wealth edges and \(u_{xx}=0\) on the factor edges. This is the default
  primary closure.
- `exact-dirichlet` places the affine optimal value \(V^*\) on the artificial
  FD boundary. It is a boundary-sensitivity audit, not the true boundary value
  of a general frozen-policy solution, and should not be presented as such.

The primary row is the combination of the largest grid factor, largest domain
factor, and the **first** requested boundary. With the defaults this is the
fine, largest-domain, linearity solve. For verification checkpoints, the
program additionally changes grid spacing, FD-domain size, and boundary
closure. Every requested Cartesian variant participates in the pass/fail
check. `rho_sensitivity_envelope` is the primary ratio plus the largest
observed absolute Cartesian change; `approx_sensitivity_envelope` reports the
largest corresponding E4 change. These are transparent two-level sensitivity diagnostics, not
rigorous discretization-error bounds or a proof on an unbounded domain.
Each axis passes when its change is no larger than
`--refinement-abs-tolerance + --refinement-rel-tolerance * abs(primary)`;
the defaults are (10^{-2}) and (2\times10^{-2}). Paper aggregation requires
`refinement_status=pass` unless the exploratory
`--allow-partial-sensitivity` flag is supplied.

`--verify-checkpoints all` performs the full Cartesian set of requested
variants at every iterate. `first,middle,last` is a useful preliminary check;
`none` computes primary rows only. All checkpoints still receive a primary
solve.

## Greedy guard and clipping interpretation

Checkpoint differentiation uses the same concavity guard as current Liu
training:

\[
V_{ww}^{\mathrm{safe}}=\min(V_{ww},-10^{-8}),
\qquad
\theta
=-\frac{\lambda(x)V_w+\Gamma V_{wx}}
        {V_{ww}^{\mathrm{safe}}},
\qquad
\vartheta=\theta/w.
\]

If the saved run configured a raw-dollar `theta_clip_abs`, that clipping is
also applied before division by wealth. The program reports guard,
positive-curvature, any-component clip, and componentwise clip fractions on
both `X_ev` and the enlarged FD domain, together with extrema of
\(\vartheta\).

If any guard or clipping activates, the row is labelled
`sampled_guarded_clipped`: it measures the actually implemented modified
greedy map. With no sampled activation it is labelled
`locally_unmodified_on_sampled_xfd`. That label is deliberately local. Neural
feedback is sampled only on a finite enlarged rectangle, possibly outside the
training collocation domain, and every row therefore records
`whole_space_map_claim=not_verified_by_finite_domain`.

## Required input artifacts

The evaluator accepts only a completed affine PI-PINN run with \(M=1\) and
`nonaffine_eps=0`. The asset count \(N\) is unrestricted. The run must contain

```text
RUN_DIR/
  _SUCCESS
  config.json
  status.json
  market_params.npz
  closed_form_ode.npz
  outer_history.csv
```

`status.json` must report `success`, and `_SUCCESS` must be the unique terminal
marker: a simultaneous `_FAILED` or `_STOPPED_EARLY` marker is rejected.

and its weight directory must contain `value_net_final.pt` plus the complete,
contiguous schedule
`iterates/value_net_iter0001.pt,...,value_net_iterKKKK.pt`, together with
`value_net_last.pt`. The tensor states in `value_net_final.pt`,
`value_net_last.pt`, and the last iterate snapshot must agree. This comparison
uses a canonical name/dtype/shape/tensor-content hash, not serialized file
bytes (whose archive metadata can differ for equal states). The evaluator also
verifies saved market identities and policy indices in `outer_history.csv`.
It records each checkpoint file hash, the canonical market hash, and
implementation/protocol hashes in the derived artifacts.
The protocol fingerprint includes the saved training arguments (including
initial-policy method/scale, optimizer, architecture, and iteration settings)
while excluding seed/device/run-location fields, so heterogeneous trajectories
cannot be pooled merely because their market snapshot is the same.

Bare state dictionaries, partial/failed runs, direct PINNs, \(M>1\), and
non-affine perturbations are rejected rather than interpreted heuristically.

## One-seed evaluation

Run from the repository root in the environment used for training. PyTorch is
needed for checkpoint differentiation; the FD core itself has no PyTorch
dependency.

```bash
python3 Liu/liu_exact_map_fd.py \
  --run-dir /path/to/liu_main/pipinn_N30_M1_seed1 \
  --device cuda:0 \
  --base-ny 41 \
  --base-nx 41 \
  --base-nt 80 \
  --eval-ny 41 \
  --eval-nx 41 \
  --grid-factors 1,2 \
  --domain-factors 1.5,2.0 \
  --boundaries linearity,exact-dirichlet \
  --verify-checkpoints all
```

Unless `--output` is given, derived files are written to
`RUN_DIR/liu_exact_map_fd`. Existing managed results are not replaced unless
`--overwrite` is supplied; unrelated files in that directory are left alone.
A fast plumbing check is:

```bash
python3 Liu/liu_exact_map_fd.py \
  --run-dir /path/to/liu_main/pipinn_N30_M1_seed1 \
  --device cuda:0 \
  --grid-factors 1 \
  --domain-factors 1.5 \
  --boundaries linearity \
  --verify-checkpoints none
```

That reduced command is exploratory and has no grid/domain/boundary
sensitivity assessment. For paper-facing results, evaluate all checkpoints
and all requested sensitivity variants with one frozen numerical protocol
across seeds.

## Multi-seed aggregation

After evaluating every seed independently, aggregate the derived directories
with the separate Liu aggregator. For example:

```bash
python3 Liu/aggregate_liu_exact_map.py \
  --out-root /path/to/liu_main \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --output /path/to/liu_main/liu_exact_map_paper \
  --overwrite
```

The strict defaults enforce the requested seed set, common market snapshot,
common checkpoint schedule, common FD/evaluation protocol, defined
denominators, and passed sensitivity checks before computing seed
statistics. A mismatch fails aggregation rather than producing a shorter,
apparently complete paper trajectory. Use
`python3 Liu/aggregate_liu_exact_map.py --help` to inspect any explicitly
requested exploratory opt-outs.
Aggregation also re-hashes the current driver/core sources, training config,
market snapshot, and every checkpoint, and checks the shifted-E4 source-policy
hash chain before trusting a derived CSV. It also verifies the manifest of all
four derived CSVs plus `exact_map_config.json`, requires the recorded schedule
to equal exactly `1,...,outer_iters`, and rejects exploratory checkpoint
subsets from paper aggregation.

## Per-run outputs

```text
liu_exact_map_fd/
  exact_map_ratios.csv              # one primary exact-map row per checkpoint
  exact_map_refinement.csv          # all grid/domain/boundary variants
  e4_approximation_errors.csv       # one primary shifted-E4 row per checkpoint
  e4_approximation_refinement.csv   # all shifted-E4 variants
  exact_map_config.json             # inputs, hashes, indexing, norm, protocol
  exact_map_status.json             # completion and denominator/sensitivity status
  _SUCCESS_EXACT_MAP
```

On a first-run failure, `exact_map_status.json` records the exception and
`_FAILED_EXACT_MAP` is created. An unsuccessful `--overwrite` attempt leaves a
previous completed audit intact because the replacement is staged and
committed only after all FD work succeeds. Treat the status file and marker as
authoritative; do not infer success from a partial CSV set.

The multi-seed command above writes:

```text
liu_exact_map_paper/
  exact_map_per_seed.csv
  exact_map_summary.csv
  exact_map_worst_per_seed.csv
  exact_map_worst_summary.csv
  e4_per_seed.csv
  e4_summary.csv
  exact_map_aggregate_status.json
  _SUCCESS_EXACT_MAP_AGG
```

`exact_map_worst_per_seed.csv` records, for each seed, the maximum primary
`rho_exact` and maximum `rho_sensitivity_envelope` over the complete outer
schedule together with their outer iterations. `exact_map_worst_summary.csv`
then reports the mean, sample SD, Student-t 95% CI, and global maximum of those
seedwise maxima. The global-maximum seed and outer iteration are included;
ties are resolved deterministically by the lowest seed and then the lowest
outer iteration. A missing or non-finite checkpoint makes that metric's
common-seed worst summary unavailable rather than allowing a shorter sample.

The aggregate JSON exposes
`finite_domain_all_tested_ratios_below_one=true` only when every denominator
is defined, every requested exact-map/E4 sensitivity assessment passes, every
seed/outer row is locally unmodified by the guard or clipping, and the global
maximum primary ratio **and** global maximum sensitivity envelope are strictly
below one. A passing envelope is additionally required to be no smaller than
its associated primary ratio. Only in that case is the finite-domain
paper-facing claim text populated. This is a conservative status for the
tested sampled FD audit, not a whole-space theorem.

## Scope and claims

- This implementation is for the **affine \(M=1\)** Liu benchmark only. It
  does not provide an exact reference for the non-affine perturbation sweep.
- It is a finite-domain approximation to the frozen-policy map, evaluated on
  an interior window. Domain, boundary, and grid sensitivity are necessary
  diagnostics but do not establish a whole-space contraction theorem.
- A guarded or clipped result is evidence about the implemented modified map,
  not automatically about the ideal unconstrained greedy operator.
- `rho_exact < 1` at sampled checkpoints is an empirical exact-map diagnostic.
  `sensitivity_stable_below_one` means the requested numerical perturbations
  also stayed below one; it is not a mathematical proof.
- Use `exact_map_worst_summary.csv` and the aggregate JSON, rather than the
  outer-wise mean trajectory alone, before stating that all tested ratios
  remained below one. The statement must retain the finite-domain,
  sampled-map qualification.
- The exact-map/E4 audit supplements rather than replaces the main seed-mean
  relative-\(L^2\) convergence figure.
