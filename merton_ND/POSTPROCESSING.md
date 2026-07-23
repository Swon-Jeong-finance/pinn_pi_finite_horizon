# Merton paper post-processing workflow

The empirical convergence figure and the exact PI-map experiment are separate
outputs with different interpretations.

## 1. Training artifacts

For paper PI-PINN runs that will also enter E4, retain every outer checkpoint:

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
PIPINN_OVERRIDES="e3b_checkpoints=false save_iterate_every=1" \
bash tune_merton.sh /path/to/merton_sweep 2
```

`e3b_checkpoints=true` selects the sparse historical schedule and overrides
periodic checkpointing; do not use it for the complete E4 maximum.

The main launcher default is `theta_init_scale=1.0`, i.e. the closed-form
myopic portfolio. Fractional `0.5` and `1.5` settings remain explicit
diagnostic pilots; they are not part of the 10-seed main sweep.

For Direct PINN, the Q_sel rescue still restores the last admissible
model+Adam state, resets the scheduler, and refreshes the collocation batch.
The paper launcher now uses `qsel_rollback_lr_factor=1.0`, so a rescue adds no
extra multiplicative LR reduction; the restored LR remains capped by the
lower of the pre-rollback and restored optimizer LRs.

## 2. E1 iteration diagnostics

E1 is aggregated from every `outer_history.csv` row, not from the final-only
`metrics.csv`. For a paper panel, keep `diag_every=1` and run:

```bash
python3 aggregate_diagnostics.py \
  --out-root /path/to/merton_sweep \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --expected-n-assets "10,50" \
  --expected-models "pinn,pipinn" \
  --min-seeds 10
```

Within each training seed, the script first takes the required minimum or
maximum over outer iterations. It then reports the mean, sample SD, standard
error, and Student-t 95% CI over those seed-level extrema; iterations are
never treated as independent replicates. A separate `paper_extreme` column is
the global minimum or maximum across both seeds and iterations.

For Merton, the reported derivative margins are (m_y,M_y,m_c). The scalar
`pi.T @ Sigma @ pi` is the sole eigenvalue of the one-dimensional frozen
state covariance. In PI-PINN, `pi_*_frozen` records the policy used by each
linear PDE; in direct PINN, the greedy policy is used because no frozen
policy exists. `pi` is already the normalized portfolio
\(\vartheta=\theta/w\), and `chi` is \(c/w\). The Kim--Omberg-only quantities
\(m_{ww}\), \(M_{num}\), and \(M_{num}/(w_{min}m_{ww})\) are explicitly marked
not applicable rather than being assigned Merton proxies.

Outputs under `diagnostic_summary/` include per-run seed extrema, a long table
with uncertainty estimates, a compact setting table, source/coverage status,
the selected-run index, and the exact grouped configurations.

### Explicit recovery merge for split seed groups

`aggregate_seeds.py` normally requires every distinct training configuration
to contain the complete expected seed panel. This remains the paper-safe
default. If interrupted/restarted jobs with intentionally equivalent settings
have already been written as different configuration groups under one output
root, they can be combined explicitly:

```bash
python3 aggregate_seeds.py \
  --out-root /path/to/merton_sweep \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --expected-n-assets "10,50" \
  --expected-m-states 1 \
  --expected-models "pinn,pipinn" \
  --strict-market-snapshots \
  --merge-config-groups
```

The merge is confined to each `model_type x n_assets x m_states` cell; methods
and dimensions are never combined. If one seed occurs in multiple source
groups, the newest run wins under the same conservative rule as ordinary
rerun deduplication: a newer failed run is not backfilled from an older
success. Canonical market validation and complete metric/seed validation
remain active. `runs_index.csv` and `groups.json` retain the source group
hashes/configurations so the forced merge is auditable.

The launcher exposes the same opt-in mode:

```bash
MERGE_CONFIG_GROUPS=1 bash tune_merton.sh /path/to/merton_sweep 2
```

## 3. PI-PINN control convergence (Figure 1)

Figure 1 is generated directly from the four control diagnostics recorded in
each PI-PINN `outer_history.csv` row:

- successive iterates: `c_diff`, `pi_diff`;
- distance to the closed-form policy: `c_vs_closed_form`,
  `pi_vs_closed_form`.

All four quantities are elementwise mean-squared discrepancies on the same
fixed \(Q_{\rm ev}\) set. At recorded `outer_iter=k`, the successive-iterate
columns compare the improved policy \(\alpha_k\) with the frozen policy
\(\alpha_{k-1}\); the closed-form columns compare \(\alpha_k\) with
\(\alpha^*\). The postprocessor requires `control_metric_scope=fixed_qev`,
complete outer coverage, one canonical market, and one exact training
configuration. It plots pointwise seed means with plus/minus one sample-SD
bands on logarithmic y axes:

```bash
python3 postprocess_pipinn_figure1.py \
  --out-root /path/to/merton_sweep \
  --n-assets 50 \
  --outer-iters 20 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --formats png,eps \
  --dpi 300 \
  --overwrite
```

Outputs are written under
`figure1_pipinn_control_convergence/`: the two-panel figure, raw seed
trajectories, pointwise summaries, the selected-run audit table, and
metadata. The configured `diag_points` budget and the realized tensor-grid
cardinality are intentionally distinct: for example, 8192 requested points
produce a 91-by-91 grid with 8281 actual points. They are recorded separately
as `diag_points_requested` and `diag_points_actual`; only the positive actual
count and its seed/outer consistency are required.

Exact zero discrepancies remain zero in every raw CSV. For the logarithmic
figure only, zero means and nonpositive mean-minus-SD endpoints are displayed
at a data-dependent positive floor. The floor and affected-point counts are
recorded in `figure1_metadata.json`.

## 4. Empirical Figure 2

Figure 2 uses outer 1--20 relative L2 errors, not `e_Xev`, a one-step
ratio, a floor filter, or an FD exact map. The default Policy curve is formed
inside each seed and outer iteration as

\[
\sqrt{(\operatorname{RelL2}_\pi^2+\operatorname{RelL2}_c^2)/2}.
\]

It then plots the pointwise arithmetic seed mean with a plus/minus one sample
SD band. Individual seed curves are hidden by default.

```bash
python3 postprocess_contraction.py \
  --out-root /path/to/merton_sweep \
  --n-assets 50 \
  --outer-iters 20 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --endpoint-outer 20 \
  --policy-curve rms \
  --formats png,eps \
  --dpi 300 \
  --overwrite \
  --hide-seed-trajectories
```

The CSV output always retains Value, portfolio, consumption, and derived
Policy-RMS trajectories, regardless of which curves are drawn. The endpoint
table records the seed-mean (E_1/E_{20}) reduction.

## 5. Total-lifetime welfare (E2)

```bash
python3 evaluate_welfare.py \
  --out-root /path/to/merton_sweep \
  --n-assets 10,50 \
  --outer-iters 20 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --w0 0.5 \
  --n-paths 100000 \
  --n-steps 1000 \
  --device cuda:0
```

This evaluates discounted running consumption plus terminal bequest. It uses
common random numbers and the optimal-policy Monte Carlo objective under the
same discretization as the CE/WL denominator. The default empty
`--expected-seeds` discovers all successful seeds; the paper command above
passes the exact 10-seed contract explicitly, with `--min-seeds 10` as an
additional minimum-count guard. See `WELFARE.md`.

## 6. Nested-window derivative-bundle sensitivity (E9)

Both trainers write the same metrics for every configured `eval_margin`:

- value: `RelL2_V` and `MaxErr_V`;
- wealth-coordinate bundle \(D V=(V_w,V_{ww})\): `RelL2_D` and `e_D_sup`;
- controls: `RelL2_pi`, `RelL2_c` and their max errors.

There is no exogenous factor state in this reduced Merton problem, so the
bundle does not contain an invented `V_wx` component. The bundle metrics are

\[
\operatorname{RelL2}_D=
\left(
\frac{\sum_i[(\Delta V_w^i)^2+(\Delta V_{ww}^i)^2]}
{\sum_i[(V_w^{*,i})^2+(V_{ww}^{*,i})^2]}
\right)^{1/2},
\qquad
e_D^{\rm sup}=\max_i\sqrt{(\Delta V_w^i)^2+(\Delta V_{ww}^i)^2}.
\]

`summary_long.csv` retains every configured margin. To produce a strict,
wide E9 table for the supplement's 5%, 10%, 20%, and 30% windows, run:

```bash
python3 aggregate_seeds.py \
  --out-root /path/to/merton_sweep \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --expected-n-assets "10,50" \
  --expected-m-states 1 \
  --expected-models "pinn,pipinn" \
  --e9-margins "0.05,0.10,0.20,0.30"
```

This writes `seed_summary/summary_e9.csv` and fails if any requested
value/bundle/control metric or seed panel is missing. Omitting `--e9-margins`
keeps all configured windows in that table, including 0%, 15%, and 25%.

## 7. Separate exact-map experiment (E3)

Run `merton_exact_map_fd.py` as documented in `EXACT_MAP.md`. Its

\[
\|E(G(\widetilde v_n))-V^*\|_{X_{\rm ev}}
/
\|\widetilde v_n-V^*\|_{X_{\rm ev}}
\]

is independent FD evidence. It is not drawn in the empirical Figure 2. Raw
finite ratios are retained by default (`floor_multiple=0`); any positive floor
filter is exploratory only.

## 8. Regularity transfer (E4)

The exact-map evaluator writes `delta_0` from the configured initial policy
and `delta_n`, (n\ge1), from adjacent checkpoints, together with hashed
evaluated bundles. It also writes `exact_map_defect_refinement.csv`, in which
the FD map entering each selected defect is recomputed across the configured
grid/domain/boundary variants. `delta_0` is always checked; adjacent defects
follow `--verify-checkpoints`, whose paper default is `all`.

The E4 aggregator requires passing defect-specific evidence for `delta_0`,
the first and last adjacent defects, and the run-wise worst `delta_X`. It
rejects missing variants, failed refinement, mixed exact-protocol hashes,
mixed primary \(Q_{\rm ev}\) windows, and eval-only relabelling. Then run:

```bash
python3 postprocess_regularity_transfer.py \
  --out-root /path/to/merton_residual_sweep \
  --n-assets 50 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --formats png,pdf \
  --overwrite
```

The script reports

\[
\widehat p_X=\max_{n=0,\ldots,N-1}\|\delta_n\|_{X_{\rm ev}}
\quad\text{against}\quad
p_{\rm res}=\max_n p_{{\rm res},n},
\]

using only official post-restore residuals. It also reports the log-log fit
and empirical upper envelope (C_{\rm num}=\max \widehat p_X/p_{\rm res}).

## 9. Residual sweep (E6)

```bash
python3 aggregate_e6.py \
  --out-root /path/to/merton_residual_sweep \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --expected-n-assets "10,50" \
  --formats png,pdf \
  --dpi 300 \
  --overwrite
```

The primary error is final `e_Xev`; `RelL2_pi` and `RelL2_c` are secondary.
The x-axis is the recomputed maximum official post-restore fixed-Q_res value,
never a sticky training-time crossing or the nominal target. Legacy residual
semantics are rejected by default. The `e_Xev` margin is taken only from the
raw training config/status, while a guarded successful eval-only overlay may
select the margin for final `metrics.csv` values. Provenance is written into
every CSV series and mixed provenance is fatal.

Two figure families are produced per configuration and margin:

- `e6_residual_error_scaling_*`: seed scatter, target geometric means, and
  the pooled fitted line;
- `e6_residual_error_scaling_mean_sd_*`: the manuscript-style target
  arithmetic mean with x/y sample-SD bars and an anchored slope-one reference.

## 10. Compute cost (E8)

Run timing jobs in a separate output root with evaluation enabled and plotting
disabled. Timing mode removes training diagnostics and checkpoint I/O, while
the fixed held-out scheduler/selection operations that belong to each method
remain part of the measured optimizer path.

```bash
SEEDS="1,2" \
PINN_OVERRIDES="timing_mode=true skip_figures=true save_iterate_every=0" \
PIPINN_OVERRIDES="timing_mode=true skip_figures=true save_iterate_every=0 e3b_checkpoints=false" \
AGGREGATE=0 \
bash tune_merton.sh /path/to/merton_timing 2

python3 aggregate_compute.py \
  --out-root /path/to/merton_timing \
  --expected-seeds "1,2" \
  --expected-n-assets "10,50" \
  --expected-methods "pinn,pipinn" \
  --require-sample-sd \
  --formats png,pdf \
  --overwrite
```

The postprocessor requires `timing_mode=true`, an unambiguous successful
training marker/status, and a persisted effective `skip_figures=true` flag.
Both trainers now force that effective flag whenever `timing_mode=true`;
spelling it out in the launcher command above is retained only for readability
and legacy scripts. It also requires one canonical market per asset dimension,
one exact method configuration across seeds, and all four status fields:

- `train_wall_sec`;
- `total_optimizer_steps`;
- `train_gpu_peak_mem_bytes`;
- `eval_gpu_peak_mem_bytes`.

The launcher's outer-20 and outer-30 jobs are treated as distinct E8 settings
(`outer_iters` is shown explicitly in every table and plot), so they are not
mistaken for duplicate seeds.

Final errors are read only from `metrics.csv` rows with `scope=fulldim` at the
recorded primary margin. The default errors are `RelL2_V`, `RelL2_D`,
`RelL2_pi`, and `RelL2_c`. To use `e_Xev`, request
`--error-metrics e_Xev` and first ensure
that the trainer writes an official full-dimensional `e_Xev` metric; the
postprocessor never substitutes an `outer_history.csv` diagnostic.

Outputs include a per-run audit table, mean plus/minus sample-SD compute and
error summaries, a wide compute table, error-versus-cost data, a four-panel
compute figure, and one four-panel error-versus-cost figure per error metric.
With one timing seed the mean is valid but sample SD is reported as `NA`;
use at least two timing seeds and `--require-sample-sd` for a paper table.
CUDA peak bytes do not identify the accelerator model, so paper timing runs
must use homogeneous GPUs and report that model separately.

## Required environment checks

The lightweight unit suite cannot execute Torch checkpoint parity. Before the
paper post-processing, run all tests in the training environment and perform
one small end-to-end exact-map and welfare smoke run on real checkpoints.
