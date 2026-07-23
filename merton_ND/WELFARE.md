# Merton total-lifetime welfare evaluation

`evaluate_welfare.py` is a post-training evaluator. It does not alter a
checkpoint or trainer. For each successful training seed it loads only the
official `value_net_final.pt`, reconstructs the trainer's final greedy policy,
and estimates

\[
J^\pi(0,w_0)=\mathbb E\!\left[
\int_0^T e^{-\rho t}U(c_t^\pi)\,dt
+e^{-\rho T}\epsilon U(W_T^\pi)
\right].
\]

This is intentionally different from terminal-wealth CE. Merton's running
consumption and terminal bequest are both included.

## Paper protocol

The defaults are:

- `w0=0.5`;
- 100,000 paths and 1,000 log-Euler steps;
- left-Riemann integration of discounted running utility;
- common asset-Brownian draws for PINN, PI-PINN, and the optimal policy;
- the optimal-policy Monte Carlo objective under the identical time grid as
  the primary denominator;
- the analytic continuous-time value only as a discretization/MC diagnostic
  (the reported z-score is not a hypothesis test because it also contains
  deterministic time-discretization bias);
- official `value_net_final.pt` checkpoints, with file hashes and the full
  resolved greedy-policy contract in `welfare_config.json`.

For CRRA homogeneity the reported initial-wealth equivalent is

\[
q^\pi=\left(J^\pi/J^*_{\rm MC}\right)^{1/(1-\gamma)},\qquad
CE_0^\pi=w_0q^\pi,\qquad WL^\pi=1-q^\pi.
\]

`se_q`, `se_ce0`, `se_wl`, and `se_utility_gap` are paired Monte Carlo
standard errors using the common random numbers. Seed summaries first compute
one metric per training seed, then report mean, sample SD, SEM, and a
Student-t 95% interval.

Seed discovery is deliberately non-prescriptive by default:
`--expected-seeds ""` uses every successful seed in the single selected
configuration, while `--min-seeds` (default 1) sets the minimum sample count
per method/dimension. Paper runs must pass the intended seed list explicitly;
a nonempty `--expected-seeds` remains an exact-set validation contract.

## Running

Validate all runs, market snapshots, exact seed sets, policy metadata, and
official checkpoint hashes without importing PyTorch:

```bash
python3 evaluate_welfare.py \
  --out-root /path/to/merton_sweep \
  --n-assets 10,50 \
  --outer-iters 20 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --validate-only
```

Validation is intentionally strict: the run must record the current trainer
source hash/marker and the `t,y` identity-input, tanh, float32 log-wealth
network contract. Ambiguous legacy checkpoints are not silently interpreted
as the current architecture.

Run the paper evaluation on a GPU:

```bash
python3 evaluate_welfare.py \
  --out-root /path/to/merton_sweep \
  --n-assets 10,50 \
  --outer-iters 20 \
  --expected-seeds "1,2,3,5,7,11,17,23,42,101" \
  --min-seeds 10 \
  --n-paths 100000 \
  --n-steps 1000 \
  --w0 0.5 \
  --device cuda:0
```

The computation resumes safely. Its signature includes the evaluator source,
protocol, selected run groups, market hashes, checkpoint hashes, and resolved
policy bounds/guards. Use `--no-resume` only to intentionally replace an
incompatible partial result.

Outputs under `<out-root>/welfare_summary/` are:

- `welfare_metrics.csv`: one optimal row per asset dimension and one learned
  row per method/training seed;
- `welfare_seed_summary.csv`: mean/SD/Student-t CI over training seeds;
- `welfare_validation.csv`: optimal MC objective versus analytic `V*(0,w0)`;
- `welfare_config.json`: complete protocol and provenance;
- `optimal_paths_N*.npz`: resumable pathwise optimal objective used for paired
  comparisons.

## Domain, guards, and bounds

Wealth is never projected or clipped back into the training window. The final
network is evaluated at the actual simulated log wealth. The output records
path-time and path-level wealth exits and fails immediately on nonfinite
wealth, policy, or utility. The path-time fraction uses the (n_{steps})
pre-step policy-evaluation states; the path-level indicator also includes the
terminal state.

The learned control map matches the recorded trainer contract:

- `V_w` and `V_y` numerator one-sided guards;
- the `V_y-V_yy` one-sided curvature guard;
- optional two-stage `kappa=c/W` and consumption-level bounds;
- optional componentwise portfolio bounds.

Every activation rate is reported. `policy_bounds_mode=none` is accepted only
when all resolved finite action bounds in `config.json` are null; `pi_clip_abs`
is then irrelevant exactly as in the trainer. The direct PINN's smooth/abs HJB
training guard is not reused here: its recorded evaluation policy is the same
hard one-sided FOC map used by the trainer's final control evaluation.

The analytic formula implements the stated bequest `epsilon*U(W_T)`, hence its
terminal homogeneity factor is `epsilon**(1/gamma)`. With the paper default
`epsilon=1`, this is also identical to the historical trainer benchmark
formula.
