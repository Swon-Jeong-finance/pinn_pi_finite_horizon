#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Liu E6 common-warm-start launcher.
#
# For every requested training seed, phase 1 performs exactly one policy
# evaluation with p_res target 1 and writes a seed-specific model/Adam/LR/RNG
# bundle.  Phase 2 is not queued until every warm-up process has succeeded and
# every expected bundle exists and is nonempty.  Each residual target then
# branches from that same seed-specific bundle.
#
# Usage:
#   DEVICE_LIST="cuda:0,cuda:1" \
#     bash tune_pipinn_e6.sh outputs/liu_e6_m1
#
# Paper/pilot defaults:
#   M=1, N=30
#   seeds = 1,2,3,5,7
#   targets = 0.2,0.1,0.05,0.02,0.01
#   one target-1 warm-up evaluation per seed
#   carry_plateau + Adam keep; target outers reset LR to 3e-4
#   no within-evaluation collocation resampling (paper/main default)
#   30,000 inner epochs as the per-evaluation cap
#
# Useful overrides:
#   SEEDS="1,2"
#   E6_TARGETS="0.2,0.1"
#   E6_BRANCH_MAX_EPOCHS=30000
#   E6_WARMUP_MAX_EPOCHS=30000
#   E6_OUTER_ITERS=20
#   E6_PE_RESAMPLE_EVERY=0
#   E6_CARRY_LR_MIN=1e-5
#   E6_CARRY_LR_MAX=3e-4
#   E6_RESET_LR_EACH_OUTER=1
#   AGGREGATE_E6=0
#   STRICT_E6_AGGREGATION=0
#   FORCE_RERUN=1

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  DEVICE_LIST="cuda:0,cuda:1" \
    bash tune_pipinn_e6.sh [OUT_ROOT] [MAX_PARALLEL]

Default paper pilot:
  N=30, M=1
  seeds=1,2,3,5,7
  common warm-up: one outer solve with p_res target 1
  target branches: 0.2,0.1,0.05,0.02,0.01

Useful environment overrides:
  SEEDS, E6_N_ASSETS, E6_M_STATES, E6_TARGETS, E6_OUTER_ITERS
  E6_WARMUP_MAX_EPOCHS, E6_BRANCH_MAX_EPOCHS
  E6_PE_RESAMPLE_EVERY, E6_CARRY_LR_MIN, E6_CARRY_LR_MAX
  E6_RESET_LR_EACH_OUTER, E6_BUNDLE_ROOT
  DEVICE_LIST, JOBS_PER_GPU, FORCE_RERUN
  AGGREGATE_E6, STRICT_E6_AGGREGATION, E6_FORMATS, E6_DPI
EOF
  exit 0
fi

E6_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E6_OUT_ROOT="${1:-$(pwd)/outputs/liu_e6_m1_$(date +%Y%m%d_%H%M%S)}"
E6_MAX_PARALLEL="${2:-}"

# These must be resolved before sourcing the production launcher because it
# parses SEEDS and constructs the worker pool at source time.
export SEEDS="${SEEDS:-1,2,3,5,7}"
export SWEEP_PROFILE="e6"

# Source, rather than copy, the production queue/marker/default machinery.
# tune_pipinn.sh has a BASH_SOURCE guard, so this does not enqueue its ordinary
# main/non-affine/timing sweep.
# shellcheck source=tune_pipinn.sh
source "$E6_SCRIPT_DIR/tune_pipinn.sh" "$E6_OUT_ROOT" "$E6_MAX_PARALLEL"

E6_N_ASSETS="${E6_N_ASSETS:-30}"
E6_M_STATES="${E6_M_STATES:-1}"
E6_WARMUP_TARGET="${E6_WARMUP_TARGET:-1}"
E6_TARGETS="${E6_TARGETS:-0.2,0.1,0.05,0.02,0.01}"
E6_OUTER_ITERS="${E6_OUTER_ITERS:-20}"
E6_BRANCH_MAX_EPOCHS="${E6_BRANCH_MAX_EPOCHS:-30000}"
E6_WARMUP_MAX_EPOCHS="${E6_WARMUP_MAX_EPOCHS:-$E6_BRANCH_MAX_EPOCHS}"
E6_PE_RESAMPLE_EVERY="${E6_PE_RESAMPLE_EVERY:-0}"
E6_CARRY_LR_MIN="${E6_CARRY_LR_MIN:-1e-5}"
E6_CARRY_LR_MAX="${E6_CARRY_LR_MAX:-3e-4}"
E6_RESET_LR_EACH_OUTER="${E6_RESET_LR_EACH_OUTER:-1}"
E6_BUNDLE_ROOT="${E6_BUNDLE_ROOT:-$OUT_ROOT/e6_warm_starts}"
AGGREGATE_E6="${AGGREGATE_E6:-1}"
STRICT_E6_AGGREGATION="${STRICT_E6_AGGREGATION:-1}"
E6_FORMATS="${E6_FORMATS:-png,pdf}"
E6_DPI="${E6_DPI:-300}"

if [[ "$EVAL_ONLY" != "0" ]]; then
  echo "[error] tune_pipinn_e6.sh is a two-phase training launcher; EVAL_ONLY must be 0" >&2
  exit 2
fi
for _toggle_name in AGGREGATE_E6 STRICT_E6_AGGREGATION E6_RESET_LR_EACH_OUTER; do
  _toggle_value="${!_toggle_name}"
  case "$_toggle_value" in
    0|1) ;;
    *)
      echo "[error] $_toggle_name must be 0 or 1; got: $_toggle_value" >&2
      exit 2
      ;;
  esac
done
for _name in E6_N_ASSETS E6_M_STATES E6_OUTER_ITERS \
             E6_BRANCH_MAX_EPOCHS E6_WARMUP_MAX_EPOCHS; do
  _value="${!_name}"
  if [[ ! "$_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] $_name must be a positive integer; got: $_value" >&2
    exit 2
  fi
done
if [[ ! "$E6_PE_RESAMPLE_EVERY" =~ ^[0-9]+$ ]]; then
  echo "[error] E6_PE_RESAMPLE_EVERY must be a non-negative integer; got: $E6_PE_RESAMPLE_EVERY" >&2
  exit 2
fi
if [[ ! "$E6_DPI" =~ ^[0-9]+$ ]] || (( E6_DPI < 36 )); then
  echo "[error] E6_DPI must be an integer >= 36; got: $E6_DPI" >&2
  exit 2
fi
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "[error] E6 requires at least one training seed" >&2
  exit 2
fi

canonical_positive_float() {
  local raw="$1"
  [[ "$raw" =~ ^[+]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]] || return 1
  local canonical
  canonical="$(awk -v value="$raw" '
    BEGIN {
      numeric = value + 0.0
      if (!(numeric > 0.0)) exit 1
      printf "%.12g", numeric
    }
  ')" || return 1
  case "${canonical,,}" in
    *inf*|*nan*) return 1 ;;
  esac
  printf "%s" "$canonical"
}

warm_target="$(canonical_positive_float "$E6_WARMUP_TARGET")" || {
  echo "[error] E6_WARMUP_TARGET must be finite and positive; got: $E6_WARMUP_TARGET" >&2
  exit 2
}
if [[ "$warm_target" != "1" ]]; then
  echo "[error] the E6 common warm-up target is fixed at 1; got: $E6_WARMUP_TARGET" >&2
  exit 2
fi

carry_lr_min="$(canonical_positive_float "$E6_CARRY_LR_MIN")" || {
  echo "[error] E6_CARRY_LR_MIN must be finite and positive; got: $E6_CARRY_LR_MIN" >&2
  exit 2
}
carry_lr_max="$(canonical_positive_float "$E6_CARRY_LR_MAX")" || {
  echo "[error] E6_CARRY_LR_MAX must be finite and positive; got: $E6_CARRY_LR_MAX" >&2
  exit 2
}
if ! awk -v lo="$carry_lr_min" -v hi="$carry_lr_max" \
    'BEGIN { exit !(lo <= hi) }'; then
  echo "[error] E6_CARRY_LR_MIN must be <= E6_CARRY_LR_MAX" >&2
  exit 2
fi

E6_TARGET_VALUES=()
IFS=', ' read -ra _raw_targets <<< "$E6_TARGETS"
declare -A _seen_targets=()
for _raw_target in "${_raw_targets[@]}"; do
  [[ -n "$_raw_target" ]] || continue
  _target="$(canonical_positive_float "$_raw_target")" || {
    echo "[error] every E6 target must be finite and positive; got: $_raw_target" >&2
    exit 2
  }
  if [[ -n "${_seen_targets[$_target]+x}" ]]; then
    echo "[error] duplicate numerical E6 target: $_raw_target -> $_target" >&2
    exit 2
  fi
  _seen_targets["$_target"]=1
  E6_TARGET_VALUES+=("$_target")
done
if (( ${#E6_TARGET_VALUES[@]} == 0 )); then
  echo "[error] E6_TARGETS must contain at least one positive target" >&2
  exit 2
fi
E6_TARGETS_CANONICAL="$(IFS=,; echo "${E6_TARGET_VALUES[*]}")"

E6_BUNDLE_ROOT="$(realpath -m "$E6_BUNDLE_ROOT")"
mkdir -p "$E6_BUNDLE_ROOT"

bundle_path() {
  local seed="$1"
  printf "%s/n_assets%s/m_states%s/seed%s/e6_warm_start.pt" \
    "$E6_BUNDLE_ROOT" "$E6_N_ASSETS" "$E6_M_STATES" "$seed"
}

reset_phase_queue() {
  : > "$JOB_QUEUE"
  : > "$JOB_FAILURES"
  echo 0 > "$JOB_CURSOR"
  rm -f "$JOB_LOCK" "$JOB_FAILURE_LOCK"
}

WARMUP_RUN_DIRS=()
WARMUP_BUNDLES=()

echo "[e6] phase 1/2: common target-1 warm-up per seed"
echo "[e6] N=$E6_N_ASSETS M=$E6_M_STATES | seeds=${SEED_LIST[*]}"
echo "[e6] warm-up inner cap=$E6_WARMUP_MAX_EPOCHS | resample every $E6_PE_RESAMPLE_EVERY"
echo "[e6] warm-up LR protocol=carry (target-only reset disabled)"
reset_phase_queue
for seed in "${SEED_LIST[@]}"; do
  bundle="$(bundle_path "$seed")"
  mkdir -p "$(dirname "$bundle")"
  warm_args=(
    "n_assets=$E6_N_ASSETS"
    "m_states=$E6_M_STATES"
    "seed=$seed"
    "e6_role=warmup"
    "e6_warmup_bundle=$bundle"
    "outer_iters=1"
    "pres_target=$warm_target"
    "eval_epochs=$E6_WARMUP_MAX_EPOCHS"
    "lr_schedule=carry_plateau"
    "adam_reset=keep"
    "carry_lr_min=$carry_lr_min"
    "carry_lr_max=$carry_lr_max"
    "e6_reset_lr_each_outer=0"
    "pe_resample_every=$E6_PE_RESAMPLE_EVERY"
    "e3b_checkpoints=0"
    "save_iterate_every=0"
    "diag_points=0"
    "test_points=0"
    "skip_figures=1"
    "skip_eval=1"
  )
  warm_tag="$(auto_tag pipinn "${warm_args[@]}")"
  WARMUP_RUN_DIRS+=("$OUT_ROOT/pi-pinn/$warm_tag")
  WARMUP_BUNDLES+=("$bundle")
  run_pipinn auto "${warm_args[@]}"
done

if ! run_all_jobs; then
  echo "[error] at least one E6 warm-up failed; no target branch was launched." >&2
  exit 1
fi

# Strong phase barrier: a shell-level success, an unambiguous success marker
# and a nonempty bundle are all required.  In particular, an old _SUCCESS
# without its bundle cannot release the target phase.
_warmup_incomplete=0
for _index in "${!WARMUP_RUN_DIRS[@]}"; do
  _run_dir="${WARMUP_RUN_DIRS[$_index]}"
  _bundle="${WARMUP_BUNDLES[$_index]}"
  if [[ ! -f "$_run_dir/_SUCCESS" \
        || -f "$_run_dir/_FAILED" \
        || -f "$_run_dir/_STOPPED_EARLY" ]]; then
    echo "[error] warm-up run is not uniquely successful: $_run_dir" >&2
    _warmup_incomplete=1
  fi
  if [[ ! -s "$_bundle" ]]; then
    echo "[error] missing/empty successful warm-up bundle: $_bundle" >&2
    _warmup_incomplete=1
  fi
  if [[ ! -s "$_run_dir/status.json" ]] \
      || ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"success"' "$_run_dir/status.json"; then
    echo "[error] warm-up status.json is missing or not successful: $_run_dir/status.json" >&2
    _warmup_incomplete=1
  fi
done
if (( _warmup_incomplete != 0 )); then
  echo "[error] warm-up panel is incomplete; target branches were not queued." >&2
  exit 1
fi

echo "[e6] phase 2/2: branch every target from each seed's common bundle"
echo "[e6] targets=${E6_TARGET_VALUES[*]} | target-phase outers=$E6_OUTER_ITERS"
echo "[e6] branch inner cap=$E6_BRANCH_MAX_EPOCHS | resample every $E6_PE_RESAMPLE_EVERY"
echo "[e6] target outer-start LR reset=$E6_RESET_LR_EACH_OUTER | carry_lr_max=$carry_lr_max"
reset_phase_queue
for seed in "${SEED_LIST[@]}"; do
  bundle="$(bundle_path "$seed")"
  for target in "${E6_TARGET_VALUES[@]}"; do
    branch_args=(
      "n_assets=$E6_N_ASSETS"
      "m_states=$E6_M_STATES"
      "seed=$seed"
      "e6_role=target_branch"
      "e6_warm_start=$bundle"
      "outer_iters=$E6_OUTER_ITERS"
      "pres_target=$target"
      "eval_epochs=$E6_BRANCH_MAX_EPOCHS"
      "lr_schedule=carry_plateau"
      "adam_reset=keep"
      "carry_lr_min=$carry_lr_min"
      "carry_lr_max=$carry_lr_max"
      "e6_reset_lr_each_outer=$E6_RESET_LR_EACH_OUTER"
      "pe_resample_every=$E6_PE_RESAMPLE_EVERY"
      "e3b_checkpoints=0"
      "save_iterate_every=0"
      "skip_figures=1"
      "skip_eval=0"
    )
    run_pipinn auto "${branch_args[@]}"
  done
done

if ! run_all_jobs; then
  echo "[error] at least one E6 target branch failed; aggregation was not run." >&2
  exit 1
fi

if [[ "$AGGREGATE_E6" == "1" ]]; then
  echo "[e6] aggregating validated common-warm-start branches -> $OUT_ROOT/e6_summary"
  aggregate_args=(
    --out-root "$OUT_ROOT"
    --expected-seeds "$SEEDS"
    --expected-tolerances "$E6_TARGETS_CANONICAL"
    --min-runs-per-tolerance "${#SEED_LIST[@]}"
    --expected-n-assets "$E6_N_ASSETS"
    --expected-m-states "$E6_M_STATES"
    --require-common-warm-start
    --expected-e6-reset-lr-each-outer "$E6_RESET_LR_EACH_OUTER"
    --formats "$E6_FORMATS"
    --dpi "$E6_DPI"
    --overwrite
  )
  if ! "$PYTHON_BIN" "$E6_SCRIPT_DIR/aggregate_e6.py" "${aggregate_args[@]}"; then
    if [[ "$STRICT_E6_AGGREGATION" == "1" ]]; then
      echo "[error] strict E6 aggregation failed: $OUT_ROOT" >&2
      exit 1
    fi
    echo "[warn] E6 aggregation failed; target runs remain available under $OUT_ROOT" >&2
  fi
fi

echo "[done] Liu E6 common-warm-start sweep complete: $OUT_ROOT"
