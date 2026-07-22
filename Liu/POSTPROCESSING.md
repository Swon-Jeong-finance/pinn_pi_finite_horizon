# Liu paper post-processing

All commands below are read-only with respect to trained checkpoints.  They
write derived CSV/figure files under the selected output root and never resume
or modify training.

## 1. Validate and aggregate Table 3

```bash
python3 aggregate_seeds.py \
  --out-root /path/to/main10seed \
  --expected-seeds 1-10 \
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

## 2. Create Figure 2 and convergence supplements

Main PI-PINN contraction output:

```bash
python3 postprocess_contraction.py \
  --out-root /path/to/main10seed \
  --diagnostic-models pipinn \
  --m-states 1,3,5 \
  --expected-seeds 1-10 \
  --primary-margin 0.10 \
  --floor-multipliers 5,10,20 \
  --main-floor-multiple 10
```

Figure 2 and all `figure2_*` CSVs are **PI-PINN only**.  The plotted ratio is
always computed within a seed first as `e_Xev[n+1] / e_Xev[n]`; only then are
mean and sample SD computed across seeds.  It is an **empirical combined
contraction–perturbation trajectory**: it reflects the realized PI update
together with approximation, optimization, and evaluation perturbations.  It
is not the exact PI contraction constant and must not be interpreted as one.

For a direct PINN/PI-PINN comparison of common diagnostic quantities only, run
the same command with `--diagnostic-models both` and a separate `--output`
directory.  Direct PINN then appears only in
`supplemental_diagnostic_summary.csv` and the
`supplemental_diagnostic_*.{png,pdf,svg,eps}` plots; it never enters
`figure2_contraction` or the Figure-2 ratio, summary, or worst-case CSVs.

The script requires `diag_every=1` and a complete finite `1..outer_iters`
history for every supplemental diagnostic metric in every requested seed;
missing indices are fatal rather than intersected or truncated.  PI-PINN also
requires a complete `e_Xev` history, a nonempty common regular support, and
identical markets across the selected methods and seeds.

Outputs include `figure2_ratios.csv`, `figure2_summary.csv`,
`figure2_worst_summary.csv`, `supplemental_diagnostic_summary.csv`, and
`postprocess_config.json`.  Generated figures have axis labels and legends
only; captions remain in the paper's LaTeX.

## 3. Validate and evaluate CE/WL

Run the cheap provenance check first:

```bash
python3 evaluate_welfare.py \
  --out-root /path/to/main10seed \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds 1-10 \
  --validate-only
```

Then run the projected-extension main evaluation in the training environment
with PyTorch and the desired GPU:

```bash
python3 evaluate_welfare.py \
  --out-root /path/to/main10seed \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds 1-10 \
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
