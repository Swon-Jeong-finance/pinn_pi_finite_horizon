# Merton integration status

The current package covers the training-time changes and the paper-facing
post-processing requested for the Merton experiment.

## Training contract

- Direct PINN uses a smooth one-sided HJB guard plus the existing sign/shape
  penalties. Its plateau scheduler is driven only by a deterministic fixed
  `Q_sel` score; the independent `Q_res` stream is reserved for target and
  official residual reporting. The Q_sel rescue restores model+Adam, resets
  the scheduler, and refreshes the batch. Its paper-launcher LR factor is
  `1.0`, meaning no additional emergency LR multiplication while retaining
  the non-increasing LR cap.
- PI-PINN held-out selection restores model and Adam state together. When a
  residual target is active, only checkpoints that meet that target on the
  same fixed `Q_res` state are selection-eligible.
- PI-PINN accepts `print_every_outer=0`: outer iterations 1--3 are still
  printed, while later periodic outer-loop logging is disabled without a
  modulo-by-zero path.
- Official PI-PINN residuals and `target_reached` are measured after restore.
  Training-time crossings are retained in separate diagnostic fields.
- `policy_bounds_mode=stabilized` keeps the safety bounds and records all
  activation fractions. `policy_bounds_mode=none` nulls every finite action
  bound, including the portfolio box, while retaining derivative guards.
- E1 records raw portfolio/consumption ranges, separate kappa and level-bound
  activation, guard fractions, and min/max `pi.T Sigma pi` on fixed diagnostic
  sets. CUDA training and evaluation peak allocation are written to status.
- Official models are final iterates. Diagnostic-best states are never used as
  the paper model.

## Post-processing contract

- `postprocess_contraction.py`: empirical Figure 2 from outer 1--20
  relative L2 Value/Policy errors, seed mean plus/minus one sample SD, no
  `e_Xev` ratio or floor filter.
- `postprocess_pipinn_figure1.py`: two-panel PI-PINN control convergence from
  the four already-squared fixed-Q_ev control diagnostics, using pointwise
  seed mean plus/minus one sample SD. Requested diagnostic points and realized
  tensor-grid points are recorded separately; exact raw zeros are retained and
  floored only for logarithmic display.
- `evaluate_welfare.py`: total discounted consumption-plus-bequest objective,
  optimal-MC denominator, CRN, CE0/WL, paired SEs, and seed t-intervals.
  Seed discovery is non-prescriptive by default; an explicit
  `--expected-seeds` list enables exact-set validation and `--min-seeds`
  independently controls the minimum selected panel size.
- `merton_exact_map_fd.py`: independent FD exact-map ratio, kept separate from
  Figure 2. It uses the manuscript wealth-coordinate norm and audits numerical
  grid/domain/boundary sensitivity. Its aggregate artifacts are named
  `exact_map_ratio_summary.csv` and `exact_map_contraction.<format>`; seed-set
  validation is opt-in and the independent minimum defaults to two seeds.
- The exact-map loader resolves legacy relative checkpoint paths against the
  launch `cwd` recorded in `config.json`. Its public self-test now includes a
  nonoptimal constant-proportional frozen policy with non-unit bequest, so
  source/drift signs and time orientation are checked independently of the
  optimal-policy formula.
- The FD paper primary is the homogeneous CRRA Robin closure. The retained
  `exact-dirichlet` CLI label is explicitly metadata-labelled as an
  optimal-reference Dirichlet sensitivity audit, not an exact boundary oracle
  for a nonoptimal neural policy.
- E4 now includes `delta_0` from the configured initial policy and every
  adjacent `delta_n`, saves hash-verified evaluated bundles, and attaches each
  defect to outer `n+1`'s official post-restore residual. A separate
  `exact_map_defect_refinement.csv` recomputes the defects across FD
  grid/domain/boundary variants; paper aggregation requires passing evidence
  for delta0, first/last adjacent, and the worst defect.
- `postprocess_regularity_transfer.py` requires checkpoints 1--N and defect
  indices 0--N-1, rejects mixed FD protocols and legacy residual semantics,
  rejects mixed primary evaluation windows or eval-only relabelling, and
  reports `p_hat_X`, its residual scaling, and `C_num`.
- `aggregate_e6.py` uses final `e_Xev` as primary and RelL2 control errors as
  secondary. It recomputes `pres_max` from post-restore outer history and
  refuses a missing/nonfinite final diagnostic. Training-time `e_Xev` uses
  only raw config/status margin provenance; guarded eval-only overlays are
  confined to final `metrics.csv` values, and mixed provenance is fatal. It
  preserves the non-pooled scatter/geometric-mean/fitted-slope figure and
  additionally writes the manuscript-style target mean plus/minus sample-SD
  figure with a slope-one reference. `settings.csv` exposes `N`, outer budget,
  and the full target-independent configuration; `--outer-iters` selects a
  paper panel.
- `aggregate_compute.py` turns successful clean timing-mode runs into the E8
  compute table and error-versus-wall-clock/steps/memory figures. It requires
  the four named status measurements, exact primary-margin `metrics.csv`
  errors, canonical market/config panels, and never backfills `e_Xev` from
  outer-history diagnostics.
  Outer-20 and outer-30 timing budgets remain separate settings.
- Direct PINN and PI-PINN use the shared `merton_evaluation_metrics.py`
  formulas at every configured evaluation margin. E9 records value,
  wealth-coordinate derivative-bundle `(V_w,V_ww)`, and control errors;
  `aggregate_seeds.py --e9-margins 0.05,0.10,0.20,0.30` creates the strict
  nested-window `summary_e9.csv` paper table while the long table retains all
  configured margins.
- Seed aggregation remains configuration-strict by default. The explicit
  `--merge-config-groups` recovery option combines split groups only within a
  common method/N/M cell, keeps newest-run deduplication and market/metric
  validation, and records every source group/configuration for audit.

## Checkpoint rule

For complete E4 evidence train with:

```text
e3b_checkpoints=false
save_iterate_every=1
```

The sparse E3b schedule is still supported for exploratory exact-map ratios,
but cannot be described as `max_n` E4 evidence.

## Validation in this workspace

- Auxiliary regression files live under `auxiliary_tests/`; the source root
  contains only experiment, evaluation, aggregation, and documentation files.
  Run them from this directory with
  `python3 -m unittest discover -s auxiliary_tests -t . -p 'test_*.py' -q`.
- Python compilation passed for trainers and all post-processors.
- `bash -n tune_merton.sh` passed.
- `tune_merton.sh` exports `PYTHONUNBUFFERED=1`, so redirected Python training
  logs are flushed without waiting for a large I/O buffer.
- 144 tests were discovered: 138 passed and 6 Torch-only checkpoint/parity
  tests were skipped because PyTorch is unavailable in this environment.
- The PyTorch-free FD manufactured refinement self-tests passed for both the
  optimal policy (`fine/coarse X-error = 0.0818`) and an independent
  nonoptimal constant-proportional policy (`0.2040`).
- The user-approved `tune_merton.sh` baseline and active run list are protected
  by a regression snapshot test; a two-job `/bin/true` launcher dry-run passed.
- Figure PNG/EPS rendering, manufactured FD checks, welfare objective oracles,
  and deterministic optimal-discretization convergence were exercised.

## Required training-environment checks

Before the main paper sweep or post-processing, run the complete test suite in
the Torch environment and execute one small real-checkpoint smoke for:

1. PI-PINN target-eligible restore and post-restore `Q_res`;
2. exact-map `delta_0` plus one adjacent defect bundle;
3. welfare policy reconstruction and CRN simulation;
4. Figure 2 discovery against a real 10-seed output tree.

FD results remain finite-domain numerical evidence, not a proof of a
whole-space contraction theorem. The exact-map defect table currently labels
its `delta_X` values as primary-grid measurements; the exact-ratio sensitivity
audit is not silently reused as a defect-specific error bound.
