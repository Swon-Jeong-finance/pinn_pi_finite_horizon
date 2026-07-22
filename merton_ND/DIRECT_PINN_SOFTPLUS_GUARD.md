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

## Deliberately unchanged

- The auxiliary eta penalty still uses its historical `abs(V_y)` denominator.
- Evaluation-only FOC policy extraction still uses one-sided hard clamps.
- PI-PINN and exact-map policy guards remain one-sided hard clamps.
- The direct PINN plateau scheduler still observes training total loss.  Moving
  it to a fixed held-out score requires a separate cadence/patience change and
  is not part of this guard-only variant.
