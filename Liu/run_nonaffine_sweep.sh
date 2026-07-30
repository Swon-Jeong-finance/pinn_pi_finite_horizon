#!/usr/bin/env bash
set -euo pipefail

# Dedicated Figure-4 launcher for the Liu non-affine risk-premium experiment.
# It reuses tune_pipinn.sh's production queue, marker, manifest, resume, and
# failure-propagation logic while keeping every artifact under a separate root.
#
# Default paper run (N=30, M=3, eps=0 baseline + six perturbations, 10 seeds):
#   DEVICE_LIST="cuda:0,cuda:1" \
#     bash run_nonaffine_sweep.sh outputs/nonaffine_n30_m3
#
# Choose M after the code is installed (one or several of 1,3,5):
#   M_STATES_LIST="1" SEEDS="1,2,3" \
#   EPS_LIST="0,0.1,1,2,3,4,5" \
#   DEVICE_LIST="cuda:1,cuda:2,cuda:3" \
#     bash run_nonaffine_sweep.sh outputs/nonaffine_n30_m1
#
# The optional second positional argument caps the number of workers, exactly
# as in tune_pipinn.sh.  FORCE_RERUN, RERUN_STOPPED, EVAL_ONLY, JOBS_PER_GPU,
# PYTHON_BIN, and all device variables are forwarded unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${1:-$(pwd)/outputs/nonaffine_n30_$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL="${2:-}"

export SWEEP_PROFILE=nonaffine
export M_STATES_LIST="${M_STATES_LIST:-3}"
export EPS_LIST="${EPS_LIST:-0,0.1,1,2,3,4,5}"
export SEEDS="${SEEDS:-1,2,3,5,7,11,17,23,42,101}"
export NONAFFINE_LOADING_SCALE="${NONAFFINE_LOADING_SCALE:-1.0}"
export NONAFFINE_SKIP_FIGURES="${NONAFFINE_SKIP_FIGURES:-0}"

# The main aggregator expects affine RelL2 tables for both PINN and PI-PINN.
# Non-affine uses paired deformation/residual aggregation instead.
export AGGREGATE=0
export STRICT_PAPER_AGGREGATION=0

echo "[nonaffine] N_ASSETS=30"
echo "[nonaffine] M_STATES_LIST=$M_STATES_LIST"
echo "[nonaffine] EPS_LIST=$EPS_LIST"
echo "[nonaffine] SEEDS=$SEEDS"
echo "[nonaffine] OUT_ROOT=$OUT_ROOT"

_args=("$OUT_ROOT")
if [[ -n "$MAX_PARALLEL" ]]; then
  _args+=("$MAX_PARALLEL")
fi
bash "$SCRIPT_DIR/tune_pipinn.sh" "${_args[@]}"

echo ""
echo "[nonaffine] training complete"
echo "[nonaffine] post-process with:"
printf "  %q %q --out-root %q --n-assets 30 --m-states %q --eps %q --expected-seeds %q --min-seeds 1\n" \
  "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/postprocess_nonaffine.py" "$OUT_ROOT" \
  "$M_STATES_LIST" "$EPS_LIST" "$SEEDS"
