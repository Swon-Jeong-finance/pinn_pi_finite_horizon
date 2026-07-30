#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Merton E6 common-warm-start launcher.
#
# For every (N, seed), this launcher first runs exactly one policy evaluation
# with p_res target 1.0 and writes a model+optimizer+RNG bundle.  Only after
# every requested warm-up has succeeded does it branch the five residual
# targets from the corresponding seed-specific bundle.
#
# With the default adam_reset=keep, residual-target runs retain the restored
# model and Adam moments, but every target-phase outer solve restarts at
# BASE_PIPINN[carry_lr_max] with a fresh within-PDE plateau scheduler.
# Ordinary PI-PINN runs keep the carry-LR rule.
#
# Usage:
#   bash tune_merton_e6.sh [OUT_ROOT] [MAX_PARALLEL]
#
# Common overrides retain the ordinary launcher's exact BASE_PIPINN values:
#   SEEDS="1,2,3,5,7,11,17,23,42,101"
#   N_ASSETS_LIST="10,50"
#   E6_TARGETS="1,0.5,0.1,0.05,0.01"
#   E6_OUTER_ITERS=20
#   PIPINN_OVERRIDES="eval_epochs=2000 ..."
#   DEVICE_LIST="cuda:0,cuda:1"
#   FORCE_RERUN=1
#   AGGREGATE=0

E6_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E6_OUT_ROOT="${1:-$(pwd)/outputs/merton_e6_$(date +%Y%m%d_%H%M%S)}"
E6_MAX_PARALLEL="${2:-}"

# Source, rather than copy, the general launcher.  Its source guard leaves all
# BASE dictionaries and queue helpers available without enqueuing the normal
# PINN sweep.
# shellcheck source=tune_merton.sh
source "$E6_SCRIPT_DIR/tune_merton.sh" "$E6_OUT_ROOT" "$E6_MAX_PARALLEL"

E6_WARMUP_TARGET="${E6_WARMUP_TARGET:-1}"
E6_TARGETS="${E6_TARGETS:-1,0.5,0.1,0.05,0.01}"
E6_OUTER_ITERS="${E6_OUTER_ITERS:-${BASE_PIPINN[outer_iters]}}"
E6_BUNDLE_ROOT="${E6_BUNDLE_ROOT:-$OUT_ROOT/e6_warm_starts}"
E6_FORMATS="${E6_FORMATS:-png,pdf}"
E6_DPI="${E6_DPI:-300}"
# Eval-only mode reuses every target branch's official final checkpoint.  It
# changes no Q_res/Q_sel set and performs no policy-evaluation optimizer step.
E6_EVAL_MARGIN="${E6_EVAL_MARGIN:-0.1}"
E6_EVAL_W_MIN="${E6_EVAL_W_MIN:-0.5}"
E6_EVAL_TEST_POINTS="${E6_EVAL_TEST_POINTS:-0}"
E6_EVAL_N_TAU="${E6_EVAL_N_TAU:-100}"
E6_EVAL_N_X="${E6_EVAL_N_X:-100}"
case "$AGGREGATE" in
  0|1) ;;
  *) echo "[error] AGGREGATE must be 0 or 1; got: $AGGREGATE" >&2; exit 2 ;;
esac
[[ "$E6_OUTER_ITERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[error] E6_OUTER_ITERS must be a positive integer; got: $E6_OUTER_ITERS" >&2
  exit 2
}
[[ "$E6_DPI" =~ ^[0-9]+$ ]] && (( E6_DPI >= 36 )) || {
  echo "[error] E6_DPI must be an integer >= 36; got: $E6_DPI" >&2
  exit 2
}

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
  echo "[error] the paper E6 common warm-up target is fixed at 1; got: $E6_WARMUP_TARGET" >&2
  exit 2
fi

E6_TARGET_VALUES=()
IFS=', ' read -ra raw_targets <<< "$E6_TARGETS"
declare -A seen_targets=()
for raw_target in "${raw_targets[@]}"; do
  [[ -n "$raw_target" ]] || continue
  target="$(canonical_positive_float "$raw_target")" || {
    echo "[error] every E6 target must be finite and positive; got: $raw_target" >&2
    exit 2
  }
  if [[ -n "${seen_targets[$target]+x}" ]]; then
    echo "[error] duplicate numerical E6 target: $raw_target -> $target" >&2
    exit 2
  fi
  seen_targets["$target"]=1
  E6_TARGET_VALUES+=("$target")
done
(( ${#E6_TARGET_VALUES[@]} > 0 )) || {
  echo "[error] E6_TARGETS must contain at least one positive target" >&2
  exit 2
}

for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "[error] every E6 seed must be a nonnegative integer; got: $seed" >&2
    exit 2
  }
done

E6_BUNDLE_ROOT="$(realpath -m "$E6_BUNDLE_ROOT")"
mkdir -p "$E6_BUNDLE_ROOT"

bundle_path() {
  local n_assets="$1" seed="$2"
  printf "%s/n_assets%s/seed%s/e6_warm_start.pt" \
    "$E6_BUNDLE_ROOT" "$n_assets" "$seed"
}

reset_phase_queue() {
  : > "$JOB_QUEUE"
  echo 0 > "$JOB_CURSOR"
  rm -f "$JOB_LOCK"
  ENQUEUED_TAGS=()
  ENQUEUED_N=()
}

if [[ "$EVAL_ONLY" == "1" ]]; then
  echo "[e6-eval] re-evaluating official final checkpoints only"
  echo "[e6-eval] eval_margin=$E6_EVAL_MARGIN, eval_w_min=$E6_EVAL_W_MIN"
  reset_phase_queue
  for n_assets in "${N_ASSET_VALUES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      for target in "${E6_TARGET_VALUES[@]}"; do
        run_pipinn \
          n_assets="$n_assets" seed="$seed" \
          e6_role=target_branch \
          outer_iters="$E6_OUTER_ITERS" pres_target="$target" \
          eval_margin="$E6_EVAL_MARGIN" eval_w_min="$E6_EVAL_W_MIN" \
          test_points="$E6_EVAL_TEST_POINTS" \
          n_tau="$E6_EVAL_N_TAU" n_x="$E6_EVAL_N_X" \
          skip_figures=1
      done
    done
  done
  if ! run_all_jobs; then
    echo "[error] at least one E6 final-checkpoint evaluation failed." >&2
    exit 1
  fi
  if [[ "$AGGREGATE" == "1" ]]; then
    expected_n_assets="$(printf '%s\n' "${N_ASSET_VALUES[@]}" | paste -sd, -)"
    echo "[e6-eval] aggregating final-window metrics -> $OUT_ROOT/e6_summary"
    "$PYTHON_BIN" "$E6_SCRIPT_DIR/aggregate_e6.py" \
      --out-root "$OUT_ROOT" \
      --expected-seeds "$SEEDS" \
      --expected-n-assets "$expected_n_assets" \
      --expected-targets "$E6_TARGETS" \
      --outer-iters "$E6_OUTER_ITERS" \
      --require-common-warm-start \
      --strict-market-snapshots \
      --e-xev-source final-metrics \
      --expected-eval-w-min "$E6_EVAL_W_MIN" \
      --formats "$E6_FORMATS" \
      --dpi "$E6_DPI" \
      --overwrite
  fi
  echo "[done] Merton E6 final-window evaluation complete: $OUT_ROOT"
  exit 0
fi

echo "[e6] phase 1/2: one common warm-up policy evaluation per (N, seed)"
echo "[e6] warm-up target: $warm_target"
reset_phase_queue
for n_assets in "${N_ASSET_VALUES[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    bundle="$(bundle_path "$n_assets" "$seed")"
    mkdir -p "$(dirname "$bundle")"
    run_pipinn \
      n_assets="$n_assets" seed="$seed" \
      e6_role=warmup e6_warmup_bundle="$bundle" \
      outer_iters=1 pres_target="$warm_target" \
      skip_figures=1 skip_eval=1
  done
done

if ! run_all_jobs; then
  echo "[error] at least one E6 warm-up failed; no target branch was launched." >&2
  exit 1
fi

# A stale _SUCCESS without its bundle must never let the target phase start.
missing_bundle=0
for n_assets in "${N_ASSET_VALUES[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    bundle="$(bundle_path "$n_assets" "$seed")"
    if [[ ! -s "$bundle" ]]; then
      echo "[error] missing/empty successful warm-up bundle: $bundle" >&2
      missing_bundle=1
    fi
  done
done
if (( missing_bundle != 0 )); then
  echo "[error] warm-up panel is incomplete; use FORCE_RERUN=1 after resolving it." >&2
  exit 1
fi

echo "[e6] phase 2/2: branch every target from the common seed-specific bundle"
echo "[e6] targets: ${E6_TARGET_VALUES[*]} | post-warm-up outer evaluations: $E6_OUTER_ITERS"
reset_phase_queue
for n_assets in "${N_ASSET_VALUES[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    bundle="$(bundle_path "$n_assets" "$seed")"
    for target in "${E6_TARGET_VALUES[@]}"; do
      run_pipinn \
        n_assets="$n_assets" seed="$seed" \
        e6_role=target_branch e6_warm_start="$bundle" \
        outer_iters="$E6_OUTER_ITERS" pres_target="$target"
    done
  done
done

if ! run_all_jobs; then
  echo "[error] at least one E6 target branch failed; aggregation is not run." >&2
  exit 1
fi

if [[ "$AGGREGATE" == "1" ]]; then
  expected_n_assets="$(printf '%s\n' "${N_ASSET_VALUES[@]}" | paste -sd, -)"
  echo "[e6] aggregating common-warm-start target branches -> $OUT_ROOT/e6_summary"
  "$PYTHON_BIN" "$E6_SCRIPT_DIR/aggregate_e6.py" \
    --out-root "$OUT_ROOT" \
    --expected-seeds "$SEEDS" \
    --expected-n-assets "$expected_n_assets" \
    --expected-targets "$E6_TARGETS" \
    --outer-iters "$E6_OUTER_ITERS" \
    --require-common-warm-start \
    --strict-market-snapshots \
    --formats "$E6_FORMATS" \
    --dpi "$E6_DPI" \
    --overwrite
fi

echo "[done] Merton E6 common-warm-start sweep complete: $OUT_ROOT"
