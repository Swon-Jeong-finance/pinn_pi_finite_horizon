# Merton exact-map integration status

The exact-map pipeline is integrated with the current Merton PI-PINN trainer
and launcher in this package.

## Integrated contract

- `tune_merton.sh` applies global overrides, gives per-run overrides higher
  precedence, supports boolean `e3b_checkpoints`, honors `N_ASSETS_LIST`, and
  rejects duplicate resolved run tags.
- `merton_nd_consumption_pi_pinn.py` records the `trainer-one-sided` /
  `merton-logw-v1` policy contract, network metadata, resolved policy bounds,
  full training arguments, trainer source hash, and `Sigma_inv_mu`.
- E3b checkpoints are saved after optional held-out model+optimizer restore.
  The manifest records outer `K` as paper source `K-1` and target policy `K`.
- New same-tag training archives stale logs/checkpoints. A pre-existing shared
  stop flag is checked before archive rotation.
- The manifest and `status.json` expose file hashes and canonical tensor-state
  hashes. The exact loader validates schedule, completion, per-checkpoint
  provenance, and final/last/final-iterate state equality.
- `merton_exact_map_fd.py` reproduces the current guarded/clipped greedy map,
  solves the independent frozen-policy FD equation, applies the manuscript's
  wealth-coordinate norm, audits grid/domain/boundary sensitivity, and
  aggregates seedwise ratios.
- The exact evaluation grid excludes the terminal face `t=T`, matching the
  trainer's `[0,T)` convention. Paper aggregation retains every finite
  checkpoint by default and carries an explicit ellipticity gate/summary.

Required artifacts for a new paper-facing run are:

```text
run/config.json
run/market_params.npz
run/status.json
weights/checkpoint_manifest.json
weights/value_net_final.pt
weights/value_net_last.pt
weights/iterates/value_net_iterNNNN.pt
```

## Validation performed in this workspace

- Python compile checks passed for trainer, utility, policy, FD core, driver,
  and test modules.
- `bash -n tune_merton.sh` passed.
- Launcher dry-runs verified E3b boolean propagation, global/per-run override
  precedence, `N_ASSETS_LIST` filtering, and duplicate-tag rejection.
- The exact-map unit suite contains 17 tests. In the lightweight environment,
  12 CPU tests pass and 5 Torch-only parity/provenance tests are skipped because
  PyTorch is not installed.
- The independent FD self-test passes; the 161/81 wealth-norm error ratio is
  approximately `0.0818`.

## Remaining run-time checks

No production checkpoint was trained or differentiated in this lightweight
environment. Before treating Figure 2 as final, run the Torch parity tests in
the training environment, then inspect each exact-map output for:

- a complete manifest and successful training marker;
- nonzero guard/clip activation fractions and the resulting `map_variant`;
- positive frozen diffusion variance over the sampled FD/evaluation domains;
- passed grid/domain/boundary sensitivity on every regular checkpoint;
- a defined denominator and stable ratio under the requested numerical
  variants;
- a sampled diffusion-variance minimum above the declared ellipticity
  tolerance.

The FD calculation is finite-domain evidence. Even a passed sensitivity audit
does not certify a whole-space contraction theorem, and the wealth-coordinate
second-derivative norm can have a more demanding numerical floor near small
wealth than the former log-coordinate diagnostic.
