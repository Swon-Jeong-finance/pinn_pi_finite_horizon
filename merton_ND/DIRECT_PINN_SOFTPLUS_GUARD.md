# Direct PINN softplus HJB guard

This variant changes only the positivity continuation inside the **direct
PINN nonlinear HJB residual**.  The PI-PINN policy-improvement map and the
exact-map postprocessor are unchanged.

The default guard is

\[
g_{\tau,\varepsilon}(x)
=\varepsilon+\tau\,\operatorname{softplus}
\left(\frac{x-\varepsilon}{\tau}\right),
\]

applied separately to \(V_y\) and \(D=V_y-V_{yy}\).  The raw sign penalties
remain

\[
\operatorname{ReLU}(-V_y)^2,
\qquad
\operatorname{ReLU}(-D)^2.
\]

## Default pilot

`tune_merton.sh` now launches the direct PINN with:

```text
hjb_guard_mode=softplus
hjb_vy_guard_tau=1e-2
hjb_denom_guard_tau=1e-2
hjb_guard_anneal_every=0
```

Thus the first pilot uses a fixed temperature.  This avoids changing the
surrogate loss while checking whether softplus itself resolves the hard-clamp
failure.  Blockwise continuation is available but opt-in:

```bash
PINN_OVERRIDES="hjb_guard_anneal_every=5 hjb_guard_anneal_factor=0.5" \
  bash tune_merton.sh OUTPUT_ROOT MAX_PARALLEL
```

Historical controls remain reproducible:

```bash
PINN_OVERRIDES="hjb_guard_mode=abs"  bash tune_merton.sh OUTPUT_ROOT MAX_PARALLEL
PINN_OVERRIDES="hjb_guard_mode=hard" bash tune_merton.sh OUTPUT_ROOT MAX_PARALLEL
```

Baseline guard values are intentionally omitted from run tags, keeping names
such as `pinn_n_assets10_outer_iters30_seed1`. Guard settings supplied through
`PINN_OVERRIDES` or directly on a `run_pinn` line are treated as explicit
overrides and therefore appear in the tag. Use a new output root after editing
the baseline guard values in the launcher itself.

The active mode, temperatures, and raw sign-violation fractions are written to
`config.json`, `train_history.csv`, and `outer_history.csv`.

## Direct-PINN scheduler held-out contract

`ReduceLROnPlateau` is driven only by the fixed held-out selection score

\[
s_{\rm sel}=\operatorname{RMS}_{Q_{\rm sel}}(r)
 +\operatorname{RMS}_{\rm terminal}(\eta).
\]

It is evaluated every `val_every` optimizer steps. `scheduler_patience` is
therefore measured in **Q_sel checks**, not stochastic training steps. The
optional `pres_target` continues to use a distinct fixed set (Q_{\rm res});
its score is never sent to the scheduler.

Both sets use `val_points` and `val_terminal_points`. Their RNG streams are
deterministic for a fixed market seed: `Q_res` uses `market_seed`, while
`Q_sel` uses `market_seed + 1000003`. The roles, seeds, score formula, check
count, and last selection score are recorded in `config.json` and
`status.json`; selection scores also appear in the two history CSVs.

## Deliberately unchanged

- The auxiliary eta penalty still uses its historical `abs(V_y)` denominator.
- Evaluation-only FOC policy extraction still uses one-sided hard clamps.
- PI-PINN and exact-map policy guards remain one-sided hard clamps.
