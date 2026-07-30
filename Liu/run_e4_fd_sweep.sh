#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for the Liu exact-map/E4 FD post-evaluation sweep.
#
# This launcher parallelizes independent experiment-cell-by-seed runs.  The
# default ``pres-target`` family forms one cell per residual target; ``main``
# forms one p_res=None cell from the ordinary M=1 checkpoint runs.  Each
# liu_exact_map_fd.py process still executes its checkpoint/grid/domain
# refinements and matched boundary-BVP sensitivity solves serially. Boundary
# replacement is reported separately and never enters refinement pass/fail.
#
# Recommended starting point for a 64-core server with a ~50-core budget when
# one uncapped job has been observed to consume roughly 10--15 core-equivalents:
#
#   E4_MAX_WORKERS=5 \
#   E4_CPU_THREADS=10 \
#   E4_CPU_BUDGET=50 \
#   E4_DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
#     bash run_e4_fd_sweep.sh outputs/pres_5seed
#
# GPU assignment is round-robin and does not limit one worker per GPU:
#   E4_DEVICE_LIST="cuda:2"          # all workers share cuda:2
#   E4_DEVICE_LIST="cuda:1,cuda:2"   # five workers split 3/2
#
# If measured CPU use stays well below the budget, keep the same output root
# and try 10 workers with 5 threads each.  A more aggressive third step is
# 20 workers with 2 threads each; the nominal caps are 50, 50, and 40,
# respectively, but SuperLU is largely single-threaded so more processes often
# improve throughput more than more threads.
#
# Usage:
#   bash run_e4_fd_sweep.sh [TRAINING_ROOT] [FD_OUTPUT_ROOT]
#
# Useful controls:
#   E4_RUN_FAMILY=...          pres-target (default) or main
#   E4_TARGETS=...             default: 0.2,0.1,0.05,0.02,0.01
#                              used only by pres-target
#   E4_MAIN_RUN_STEM=...       main run basename before _seedN
#   E4_MAIN_LABEL=...          main output/plot label (default: main_m1)
#   E4_SEEDS=...               pres-target default: 1,11,23,42,101
#                              main default: 1,2,3,5,7,11,17,23,42,101
#   E4_CHECKPOINTS=...         default: all; pilot: contiguous prefix 1,2,...,k
#   E4_REFINEMENT_RULE=...     default: cartesian; or merton-axis
#   E4_MIN_PAPER_CHECKPOINT=N  default: 0 (exclude no E4 target)
#   E4_MAX_WORKERS=N           default: 5
#   E4_CPU_THREADS=N           default: 10
#   E4_CPU_BUDGET=N            default: 50 (warning threshold)
#   E4_DEVICE_LIST=...         default: DEVICE_LIST or cuda:2
#   E4_FD_W_MINS=...           optional absolute FD lower endpoint schedule
#   E4_FD_W_MAXS=...           optional matching FD upper endpoint schedule
#   E4_FD_W_MIN/MAX            backward-compatible singular aliases
#   E4_FORCE_RERUN=1           recompute successful output directories
#   E4_DRY_RUN=1               print commands without running the solver
#   E4_RATIO_MODE=...           none (default), empirical, exact, or both
#   E4_RATIO_OUTPUT_ROOT=...    optional ratio-table/figure root
#   E4_RATIO_ALLOW_PARTIAL_SENSITIVITY=1
#                              keep failed exact refinements as exploratory
#   E4_EMPIRICAL_FLOOR_MULTIPLIERS=5,10,20
#                              Merton-style empirical floor sensitivity
#   E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE=10
#                              plotted empirical floor multiple
#   E4_EMPIRICAL_FLOOR_VALUE=... optional shared empirical floor override
#   E4_EXACT_FLOOR_MULTIPLE=0  exact-map display floor (0 keeps every point)
#   E4_EXACT_FLOOR_VALUE=...   optional shared exact-map floor override
#
# Absolute FD wealth mode (narrowest comparison, then widest primary):
#   E4_FD_W_MINS="0.08,0.05"
#   E4_FD_W_MAXS="16,32"
# When both are set, E4_WEALTH_DOMAIN_FACTORS is intentionally not forwarded.
# Resume skips a completed job only when its saved launcher protocol signature
# matches the current solver source and all numerical FD options.
#
# The four BLAS/OpenMP limits are set before Python starts.  E4-specific
# per-library overrides are also available:
#   E4_OMP_NUM_THREADS
#   E4_MKL_NUM_THREADS
#   E4_OPENBLAS_NUM_THREADS
#   E4_NUMEXPR_NUM_THREADS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

E4_RUN_FAMILY="${E4_RUN_FAMILY:-pres-target}"
E4_TARGETS="${E4_TARGETS:-0.2,0.1,0.05,0.02,0.01}"
E4_MAIN_RUN_STEM="${E4_MAIN_RUN_STEM:-pipinn_m_states1_e3b_checkpoints1}"
E4_MAIN_LABEL="${E4_MAIN_LABEL:-main_m1}"
E4_SEEDS="${E4_SEEDS:-1,11,23,42,101}"
E4_CHECKPOINTS="${E4_CHECKPOINTS:-all}"
E4_REFINEMENT_RULE="${E4_REFINEMENT_RULE:-cartesian}"
E4_MIN_PAPER_CHECKPOINT="${E4_MIN_PAPER_CHECKPOINT:-0}"
E4_MAX_WORKERS="${E4_MAX_WORKERS:-25}"
E4_CPU_THREADS="${E4_CPU_THREADS:-2}"
E4_CPU_BUDGET="${E4_CPU_BUDGET:-50}"
E4_DEVICE_LIST="${E4_DEVICE_LIST:-${DEVICE_LIST:-cuda:1,cuda:2}}"
E4_FORCE_RERUN="${E4_FORCE_RERUN:-0}"
E4_DRY_RUN="${E4_DRY_RUN:-0}"
E4_RATIO_MODE="${E4_RATIO_MODE:-none}"
E4_RATIO_OUTPUT_ROOT="${E4_RATIO_OUTPUT_ROOT:-}"
E4_RATIO_ALLOW_PARTIAL_SENSITIVITY="${E4_RATIO_ALLOW_PARTIAL_SENSITIVITY:-0}"
E4_RATIO_FORMATS="${E4_RATIO_FORMATS:-png,pdf}"
E4_RATIO_Y_SCALE="${E4_RATIO_Y_SCALE:-}"
E4_RATIO_PLOT_SENSITIVITY_ENVELOPE="${E4_RATIO_PLOT_SENSITIVITY_ENVELOPE:-0}"
E4_RATIO_FIG_WIDTH="${E4_RATIO_FIG_WIDTH:-6.5}"
E4_RATIO_FIG_HEIGHT="${E4_RATIO_FIG_HEIGHT:-4.2}"
E4_RATIO_FONT_SIZE="${E4_RATIO_FONT_SIZE:-18}"
E4_RATIO_DPI="${E4_RATIO_DPI:-300}"
E4_EMPIRICAL_PRIMARY_MARGIN="${E4_EMPIRICAL_PRIMARY_MARGIN:-0.10}"
E4_EMPIRICAL_FLOOR_MULTIPLIERS="${E4_EMPIRICAL_FLOOR_MULTIPLIERS:-1.5,2,3}"
E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE="${E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE:-1.5}"
E4_EMPIRICAL_FLOOR_VALUE="${E4_EMPIRICAL_FLOOR_VALUE:-}"
E4_EXACT_FLOOR_MULTIPLE="${E4_EXACT_FLOOR_MULTIPLE:-0}"
E4_EXACT_FLOOR_VALUE="${E4_EXACT_FLOOR_VALUE:-}"
if [[ -z "$E4_RATIO_Y_SCALE" ]]; then
  if [[ "$E4_RATIO_MODE" == "empirical" ]]; then
    E4_RATIO_Y_SCALE="linear"
  else
    E4_RATIO_Y_SCALE="log"
  fi
fi

# Frozen FD protocol matching the accepted pilot command.  Changing any of
# these values should be accompanied by a distinct FD_OUTPUT_ROOT.
E4_EVAL_W_MIN="${E4_EVAL_W_MIN:-1.0}"
E4_EVAL_MARGIN="${E4_EVAL_MARGIN:-0.10}"
E4_BASE_NY="${E4_BASE_NY:-41}"
E4_BASE_NX="${E4_BASE_NX:-41}"
E4_BASE_NT="${E4_BASE_NT:-80}"
E4_EVAL_NY="${E4_EVAL_NY:-41}"
E4_EVAL_NX="${E4_EVAL_NX:-41}"
E4_GRID_FACTORS="${E4_GRID_FACTORS:-1,2}"
E4_WEALTH_DOMAIN_FACTORS="${E4_WEALTH_DOMAIN_FACTORS:-3.5,4.0}"
E4_FACTOR_DOMAIN_FACTORS="${E4_FACTOR_DOMAIN_FACTORS:-1.25,1.50}"
E4_FD_W_MIN="${E4_FD_W_MIN:-}"
E4_FD_W_MAX="${E4_FD_W_MAX:-}"
E4_FD_W_MINS="${E4_FD_W_MINS:-}"
E4_FD_W_MAXS="${E4_FD_W_MAXS:-}"
if [[ -n "$E4_FD_W_MIN" && -n "$E4_FD_W_MINS" \
      && "$E4_FD_W_MIN" != "$E4_FD_W_MINS" ]]; then
  echo "[error] E4_FD_W_MIN and E4_FD_W_MINS disagree" >&2
  exit 2
fi
if [[ -n "$E4_FD_W_MAX" && -n "$E4_FD_W_MAXS" \
      && "$E4_FD_W_MAX" != "$E4_FD_W_MAXS" ]]; then
  echo "[error] E4_FD_W_MAX and E4_FD_W_MAXS disagree" >&2
  exit 2
fi
E4_FD_W_MIN="${E4_FD_W_MIN:-$E4_FD_W_MINS}"
E4_FD_W_MAX="${E4_FD_W_MAX:-$E4_FD_W_MAXS}"
E4_BOUNDARIES="${E4_BOUNDARIES:-linearity,exact-dirichlet}"
E4_VERIFY_CHECKPOINTS="${E4_VERIFY_CHECKPOINTS:-all}"
E4_POLICY_EXTENSION="${E4_POLICY_EXTENSION:-boundary-projection}"
E4_DRIFT_SCHEME="${E4_DRIFT_SCHEME:-adaptive}"
E4_LINEAR_RESIDUAL_TOLERANCE="${E4_LINEAR_RESIDUAL_TOLERANCE:-1e-8}"
E4_BOUNDARY_CONDITION_LIMIT="${E4_BOUNDARY_CONDITION_LIMIT:-1e12}"

export E4_RUN_FAMILY E4_TARGETS E4_MAIN_RUN_STEM E4_MAIN_LABEL
export E4_SEEDS E4_CHECKPOINTS
export E4_REFINEMENT_RULE E4_MIN_PAPER_CHECKPOINT
export E4_MAX_WORKERS E4_CPU_THREADS E4_CPU_BUDGET
export E4_DEVICE_LIST E4_FORCE_RERUN E4_DRY_RUN
export E4_RATIO_MODE E4_RATIO_OUTPUT_ROOT
export E4_RATIO_ALLOW_PARTIAL_SENSITIVITY E4_RATIO_FORMATS
export E4_RATIO_Y_SCALE E4_RATIO_PLOT_SENSITIVITY_ENVELOPE
export E4_RATIO_FIG_WIDTH E4_RATIO_FIG_HEIGHT E4_RATIO_FONT_SIZE
export E4_RATIO_DPI E4_EMPIRICAL_PRIMARY_MARGIN
export E4_EMPIRICAL_FLOOR_MULTIPLIERS
export E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE E4_EMPIRICAL_FLOOR_VALUE
export E4_EXACT_FLOOR_MULTIPLE E4_EXACT_FLOOR_VALUE
export E4_EVAL_W_MIN E4_EVAL_MARGIN E4_BASE_NY E4_BASE_NX E4_BASE_NT
export E4_EVAL_NY E4_EVAL_NX E4_GRID_FACTORS
export E4_WEALTH_DOMAIN_FACTORS E4_FACTOR_DOMAIN_FACTORS E4_BOUNDARIES
export E4_FD_W_MIN E4_FD_W_MAX E4_FD_W_MINS E4_FD_W_MAXS
export E4_VERIFY_CHECKPOINTS E4_POLICY_EXTENSION E4_DRIFT_SCHEME
export E4_LINEAR_RESIDUAL_TOLERANCE E4_BOUNDARY_CONDITION_LIMIT
export PYTHON_BIN

export OMP_NUM_THREADS="${E4_OMP_NUM_THREADS:-$E4_CPU_THREADS}"
export MKL_NUM_THREADS="${E4_MKL_NUM_THREADS:-$E4_CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${E4_OPENBLAS_NUM_THREADS:-$E4_CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${E4_NUMEXPR_NUM_THREADS:-$E4_CPU_THREADS}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_binary_flag() {
  [[ "$1" == "0" || "$1" == "1" ]]
}

is_nonnegative_finite_number() {
  [[ "$1" =~ ^\+?(([0-9]+(\.[0-9]*)?)|(\.[0-9]+))([eE][+-]?[0-9]+)?$ ]] \
    && awk -v value="$1" \
      'BEGIN {
         rendered = tolower(sprintf("%.17g", value + 0.0))
         exit (rendered ~ /inf|nan/) ? 1 : 0
       }'
}

numbers_equal() {
  awk -v lhs="$1" -v rhs="$2" \
    'BEGIN { exit !((lhs + 0.0) == (rhs + 0.0)) }'
}

normalize_empirical_floor_controls() {
  local compact
  local value
  local main_in_list=0
  local -a values=()

  compact="${E4_EMPIRICAL_FLOOR_MULTIPLIERS//[[:space:]]/}"
  if [[ -z "$compact" || "$compact" == *, || "$compact" == ,* \
        || "$compact" == *,,* ]]; then
    return 1
  fi
  IFS=',' read -r -a values <<<"$compact"
  for value in "${values[@]}"; do
    if ! is_nonnegative_finite_number "$value"; then
      return 1
    fi
    if numbers_equal "$value" "$E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE"; then
      main_in_list=1
    fi
  done
  if (( main_in_list == 0 )); then
    return 2
  fi
  E4_EMPIRICAL_FLOOR_MULTIPLIERS="$(IFS=,; echo "${values[*]}")"
}

normalize_checkpoints() {
  local compact
  local -a values=()
  local index
  compact="${E4_CHECKPOINTS//[[:space:]]/}"
  if [[ "${compact,,}" == "all" ]]; then
    E4_CHECKPOINTS="all"
    return 0
  fi
  if [[ -z "$compact" || "$compact" == *, || "$compact" == ,* \
        || "$compact" == *,,* ]]; then
    return 1
  fi
  IFS=',' read -r -a values <<<"$compact"
  for index in "${!values[@]}"; do
    if ! is_positive_integer "${values[$index]}" \
        || (( 10#${values[$index]} != index + 1 )); then
      return 1
    fi
  done
  E4_CHECKPOINTS="$(IFS=,; echo "${values[*]}")"
}

RENDERED_COMMAND=""
render_command() {
  local rendered
  local item
  printf -v rendered "%q" "$1"
  shift
  for item in "$@"; do
    printf -v rendered "%s %q" "$rendered" "$item"
  done
  RENDERED_COMMAND="$rendered"
}

run_empirical_ratio_target() {
  local training_root="$1"
  local ratio_root="$2"
  local target="$3"
  local tag="${target//./p}"
  local escaped_target="${target//./\\.}"
  local escaped_main_stem="${E4_MAIN_RUN_STEM//./\\.}"
  local seed_alternation
  local output
  local run_name_regex
  local target_label
  seed_alternation="$(IFS='|'; echo "${SEEDS[*]}")"
  if [[ "$E4_RUN_FAMILY" == "main" ]]; then
    output="${ratio_root}/${E4_MAIN_LABEL}/empirical_training_xev"
    run_name_regex="(?:^|/)${escaped_main_stem}_seed(${seed_alternation})$"
    target_label="$E4_MAIN_LABEL"
  else
    output="${ratio_root}/pres_${tag}/empirical_training_xev"
    run_name_regex="(?:^|/)pipinn_rho_canonical_v1_m_states1_eval_epochs50000_pres_target${escaped_target}_seed(${seed_alternation})$"
    target_label="$target"
  fi
  local -a command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/postprocess_empirical_xev_ratio.py"
    --out-root "$training_root"
    --output "$output"
    --m-states 1
    --n-assets 30
    --expected-seeds "$E4_SEEDS"
    --min-seeds "${#SEEDS[@]}"
    --primary-margin "$E4_EMPIRICAL_PRIMARY_MARGIN"
    --floor-multipliers "$E4_EMPIRICAL_FLOOR_MULTIPLIERS"
    --main-floor-multiple "$E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE"
    --run-name-regex "$run_name_regex"
    --target-label "$target_label"
    --formats "$E4_RATIO_FORMATS"
    --ratio-y-scale "$E4_RATIO_Y_SCALE"
    --fig-width "$E4_RATIO_FIG_WIDTH"
    --fig-height "$E4_RATIO_FIG_HEIGHT"
    --font-size "$E4_RATIO_FONT_SIZE"
    --dpi "$E4_RATIO_DPI"
    --overwrite
  )
  if [[ -n "$E4_EMPIRICAL_FLOOR_VALUE" ]]; then
    command+=(--floor-value "$E4_EMPIRICAL_FLOOR_VALUE")
  fi
  render_command "${command[@]}"
  if [[ "$E4_DRY_RUN" == "1" ]]; then
    printf "[dry-run ratio empirical] target=%s\n%s\n" \
      "$target" "$RENDERED_COMMAND"
    return 0
  fi
  echo "[ratio empirical start] target=$target output=$output"
  "${command[@]}"
  echo "[ratio empirical done] target=$target output=$output"
}

run_exact_ratio_target() {
  local fd_root="$1"
  local ratio_root="$2"
  local target="$3"
  local series="$4"
  local tag="${target//./p}"
  local cell_dir
  local target_label
  if [[ "$E4_RUN_FAMILY" == "main" ]]; then
    cell_dir="$E4_MAIN_LABEL"
    target_label="$E4_MAIN_LABEL"
  else
    cell_dir="pres_${tag}"
    target_label="$target"
  fi
  local output="${ratio_root}/${cell_dir}/exact_audit"
  local -a command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_liu_exact_map.py"
    --out-root "${fd_root}/${cell_dir}"
    --output "$output"
    --expected-seeds "$E4_SEEDS"
    --min-seeds "${#SEEDS[@]}"
    --plot-ratios
    --ratio-series "$series"
    --floor-multiple "$E4_EXACT_FLOOR_MULTIPLE"
    --target-label "$target_label"
    --formats "$E4_RATIO_FORMATS"
    --ratio-y-scale "$E4_RATIO_Y_SCALE"
    --fig-width "$E4_RATIO_FIG_WIDTH"
    --fig-height "$E4_RATIO_FIG_HEIGHT"
    --font-size "$E4_RATIO_FONT_SIZE"
    --dpi "$E4_RATIO_DPI"
    --overwrite
  )
  if [[ -n "$E4_EXACT_FLOOR_VALUE" ]]; then
    command+=(--floor-value "$E4_EXACT_FLOOR_VALUE")
  fi
  if [[ "$E4_RATIO_ALLOW_PARTIAL_SENSITIVITY" == "1" ]]; then
    command+=(--allow-partial-sensitivity)
  fi
  if [[ "$E4_RATIO_PLOT_SENSITIVITY_ENVELOPE" == "1" ]]; then
    command+=(--plot-sensitivity-envelope)
  fi
  render_command "${command[@]}"
  if [[ "$E4_DRY_RUN" == "1" ]]; then
    printf "[dry-run ratio %s] target=%s\n%s\n" \
      "$series" "$target" "$RENDERED_COMMAND"
    return 0
  fi
  echo "[ratio $series start] target=$target output=$output"
  "${command[@]}"
  echo "[ratio $series done] target=$target output=$output"
}

worker_main() {
  local training_root="$1"
  local fd_root="$2"
  local target="$3"
  local seed="$4"
  local device="$5"
  local tag="${target//./p}"
  local cell_dir
  local run_dir
  local out_dir
  local log_dir
  local log_file
  local lock_dir
  local lock_file
  local protocol_signature
  local signature_file
  local signature_tmp
  local solver_sha256
  local core_sha256
  local stored_signature

  if [[ "$E4_RUN_FAMILY" == "main" ]]; then
    run_dir="${training_root}/pi-pinn/${E4_MAIN_RUN_STEM}_seed${seed}"
    cell_dir="$E4_MAIN_LABEL"
  else
    run_dir="${training_root}/pi-pinn/pipinn_rho_canonical_v1_m_states1_eval_epochs50000_pres_target${target}_seed${seed}"
    cell_dir="pres_${tag}"
  fi
  out_dir="${fd_root}/${cell_dir}/seed${seed}"
  log_dir="${fd_root}/logs"
  log_file="${log_dir}/${cell_dir}_seed${seed}.log"
  lock_dir="${fd_root}/locks"
  lock_file="${lock_dir}/${cell_dir}_seed${seed}.lock"

  local -a command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/liu_exact_map_fd.py"
    --run-dir "$run_dir"
    --output "$out_dir"
    --device "$device"
    --eval-w-min "$E4_EVAL_W_MIN"
    --eval-margin "$E4_EVAL_MARGIN"
    --base-ny "$E4_BASE_NY"
    --base-nx "$E4_BASE_NX"
    --base-nt "$E4_BASE_NT"
    --eval-ny "$E4_EVAL_NY"
    --eval-nx "$E4_EVAL_NX"
    --grid-factors "$E4_GRID_FACTORS"
    --refinement-rule "$E4_REFINEMENT_RULE"
    --min-paper-checkpoint "$E4_MIN_PAPER_CHECKPOINT"
  )
  if [[ "$E4_CHECKPOINTS" != "all" ]]; then
    command+=(--checkpoints "$E4_CHECKPOINTS")
  fi
  if [[ -n "$E4_FD_W_MIN" ]]; then
    command+=(
      --fd-w-min "$E4_FD_W_MIN"
      --fd-w-max "$E4_FD_W_MAX"
    )
  else
    command+=(--wealth-domain-factors "$E4_WEALTH_DOMAIN_FACTORS")
  fi
  command+=(
    --factor-domain-factors "$E4_FACTOR_DOMAIN_FACTORS"
    --boundaries "$E4_BOUNDARIES"
    --verify-checkpoints "$E4_VERIFY_CHECKPOINTS"
    --policy-extension "$E4_POLICY_EXTENSION"
    --drift-scheme "$E4_DRIFT_SCHEME"
    --linear-residual-tolerance "$E4_LINEAR_RESIDUAL_TOLERANCE"
    --boundary-condition-limit "$E4_BOUNDARY_CONDITION_LIMIT"
    --overwrite
  )

  render_command "${command[@]}"
  if [[ "$E4_DRY_RUN" == "1" ]]; then
    printf "[dry-run] target=%s seed=%s device=%s\n%s\n" \
      "$target" "$seed" "$device" "$RENDERED_COMMAND"
    return 0
  fi

  solver_sha256="$(sha256sum "$SCRIPT_DIR/liu_exact_map_fd.py")"
  solver_sha256="${solver_sha256%% *}"
  core_sha256="$(sha256sum "$SCRIPT_DIR/liu_exact_map_core.py")"
  core_sha256="${core_sha256%% *}"
  protocol_signature="$({
    printf "solver_sha256=%s\n" "$solver_sha256"
    printf "core_sha256=%s\n" "$core_sha256"
    printf "run_family=%s\n" "$E4_RUN_FAMILY"
    printf "training_run_dir=%s\n" "$run_dir"
    printf "cell_dir=%s\n" "$cell_dir"
    printf "eval_w_min=%s\n" "$E4_EVAL_W_MIN"
    printf "eval_margin=%s\n" "$E4_EVAL_MARGIN"
    printf "base_grid=%s,%s,%s\n" "$E4_BASE_NY" "$E4_BASE_NX" "$E4_BASE_NT"
    printf "eval_grid=%s,%s\n" "$E4_EVAL_NY" "$E4_EVAL_NX"
    printf "grid_factors=%s\n" "$E4_GRID_FACTORS"
    printf "checkpoints=%s\n" "$E4_CHECKPOINTS"
    printf "refinement_rule=%s\n" "$E4_REFINEMENT_RULE"
    printf "min_paper_checkpoint=%s\n" "$E4_MIN_PAPER_CHECKPOINT"
    if [[ -n "$E4_FD_W_MIN" ]]; then
      printf "wealth_mode=absolute\n"
      printf "fd_w_min=%s\n" "$E4_FD_W_MIN"
      printf "fd_w_max=%s\n" "$E4_FD_W_MAX"
    else
      printf "wealth_mode=factor\n"
      printf "wealth_domain_factors=%s\n" "$E4_WEALTH_DOMAIN_FACTORS"
    fi
    printf "factor_domain_factors=%s\n" "$E4_FACTOR_DOMAIN_FACTORS"
    printf "boundaries=%s\n" "$E4_BOUNDARIES"
    printf "verify_checkpoints=%s\n" "$E4_VERIFY_CHECKPOINTS"
    printf "policy_extension=%s\n" "$E4_POLICY_EXTENSION"
    printf "drift_scheme=%s\n" "$E4_DRIFT_SCHEME"
    printf "linear_residual_tolerance=%s\n" \
      "$E4_LINEAR_RESIDUAL_TOLERANCE"
    printf "boundary_condition_limit=%s\n" \
      "$E4_BOUNDARY_CONDITION_LIMIT"
  } | sha256sum)"
  protocol_signature="${protocol_signature%% *}"
  signature_file="${out_dir}/launcher_protocol.sha256"

  mkdir -p "$log_dir" "$lock_dir"
  exec 9>"$lock_file"
  flock 9

  # Another launcher may have completed the same job while this worker was
  # waiting for the lock, so re-check only after acquiring it.
  if [[ "$E4_FORCE_RERUN" != "1" && -f "${out_dir}/_SUCCESS_EXACT_MAP" ]]; then
    stored_signature=""
    if [[ -f "$signature_file" ]]; then
      IFS= read -r stored_signature <"$signature_file" || true
    fi
    if [[ "$stored_signature" == "$protocol_signature" ]]; then
      echo "[skip] target=$target seed=$seed output=$out_dir"
      return 0
    fi
    echo "[rerun] target=$target seed=$seed: completed output has a missing or different FD protocol signature"
  fi

  echo "[start] target=$target seed=$seed device=$device log=$log_file"
  {
    printf "[command] %s\n" "$RENDERED_COMMAND"
    printf "[threads] OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s\n" \
      "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" \
      "$OPENBLAS_NUM_THREADS" "$NUMEXPR_NUM_THREADS"
  } >"$log_file"

  local status
  set +e
  "${command[@]}" >>"$log_file" 2>&1
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[failed] target=$target seed=$seed status=$status log=$log_file" >&2
    return "$status"
  fi
  if [[ ! -f "${out_dir}/_SUCCESS_EXACT_MAP" ]]; then
    echo "[failed] target=$target seed=$seed: success marker missing; log=$log_file" >&2
    return 1
  fi
  signature_tmp="${out_dir}/.launcher_protocol.sha256.tmp.$$"
  printf "%s\n" "$protocol_signature" >"$signature_tmp"
  mv -f -- "$signature_tmp" "$signature_file"

  echo "[done] target=$target seed=$seed output=$out_dir"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  if [[ "$#" -ne 5 ]]; then
    echo "[error] internal worker expected 5 arguments, received $#" >&2
    exit 2
  fi
  worker_main "$@"
  exit $?
fi

if [[ "$E4_RUN_FAMILY" != "pres-target" \
      && "$E4_RUN_FAMILY" != "main" ]]; then
  echo "[error] E4_RUN_FAMILY must be pres-target or main: $E4_RUN_FAMILY" >&2
  exit 2
fi
if [[ ! "$E4_MAIN_RUN_STEM" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ \
      || "$E4_MAIN_RUN_STEM" == "." \
      || "$E4_MAIN_RUN_STEM" == ".." ]]; then
  echo "[error] E4_MAIN_RUN_STEM must be one safe run-name stem: $E4_MAIN_RUN_STEM" >&2
  exit 2
fi
if [[ ! "$E4_MAIN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ \
      || "$E4_MAIN_LABEL" == "." \
      || "$E4_MAIN_LABEL" == ".." ]]; then
  echo "[error] E4_MAIN_LABEL must be one safe output label: $E4_MAIN_LABEL" >&2
  exit 2
fi

if ! is_positive_integer "$E4_MAX_WORKERS"; then
  echo "[error] E4_MAX_WORKERS must be a positive integer: $E4_MAX_WORKERS" >&2
  exit 2
fi
if ! is_positive_integer "$E4_CPU_THREADS"; then
  echo "[error] E4_CPU_THREADS must be a positive integer: $E4_CPU_THREADS" >&2
  exit 2
fi
if ! is_positive_integer "$E4_CPU_BUDGET"; then
  echo "[error] E4_CPU_BUDGET must be a positive integer: $E4_CPU_BUDGET" >&2
  exit 2
fi
if [[ ! "$E4_MIN_PAPER_CHECKPOINT" =~ ^[0-9]+$ ]]; then
  echo "[error] E4_MIN_PAPER_CHECKPOINT must be a nonnegative integer: $E4_MIN_PAPER_CHECKPOINT" >&2
  exit 2
fi
if [[ "$E4_REFINEMENT_RULE" != "cartesian" \
      && "$E4_REFINEMENT_RULE" != "merton-axis" ]]; then
  echo "[error] E4_REFINEMENT_RULE must be cartesian or merton-axis: $E4_REFINEMENT_RULE" >&2
  exit 2
fi
if ! is_binary_flag "$E4_FORCE_RERUN"; then
  echo "[error] E4_FORCE_RERUN must be 0 or 1: $E4_FORCE_RERUN" >&2
  exit 2
fi
if ! is_binary_flag "$E4_DRY_RUN"; then
  echo "[error] E4_DRY_RUN must be 0 or 1: $E4_DRY_RUN" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "none" \
      && "$E4_RATIO_MODE" != "empirical" \
      && "$E4_RATIO_MODE" != "exact" \
      && "$E4_RATIO_MODE" != "both" ]]; then
  echo "[error] E4_RATIO_MODE must be none, empirical, exact, or both: $E4_RATIO_MODE" >&2
  exit 2
fi
if ! is_binary_flag "$E4_RATIO_ALLOW_PARTIAL_SENSITIVITY"; then
  echo "[error] E4_RATIO_ALLOW_PARTIAL_SENSITIVITY must be 0 or 1: $E4_RATIO_ALLOW_PARTIAL_SENSITIVITY" >&2
  exit 2
fi
if ! is_binary_flag "$E4_RATIO_PLOT_SENSITIVITY_ENVELOPE"; then
  echo "[error] E4_RATIO_PLOT_SENSITIVITY_ENVELOPE must be 0 or 1: $E4_RATIO_PLOT_SENSITIVITY_ENVELOPE" >&2
  exit 2
fi
if [[ "$E4_RATIO_Y_SCALE" != "linear" \
      && "$E4_RATIO_Y_SCALE" != "log" ]]; then
  echo "[error] E4_RATIO_Y_SCALE must be linear or log: $E4_RATIO_Y_SCALE" >&2
  exit 2
fi
if ! is_nonnegative_finite_number "$E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE"; then
  echo "[error] E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE must be finite and nonnegative: $E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE" >&2
  exit 2
fi
if [[ -n "$E4_EMPIRICAL_FLOOR_VALUE" ]] \
    && ! is_nonnegative_finite_number "$E4_EMPIRICAL_FLOOR_VALUE"; then
  echo "[error] E4_EMPIRICAL_FLOOR_VALUE must be finite and nonnegative: $E4_EMPIRICAL_FLOOR_VALUE" >&2
  exit 2
fi
set +e
normalize_empirical_floor_controls
empirical_floor_status=$?
set -e
if [[ "$empirical_floor_status" -eq 1 ]]; then
  echo "[error] E4_EMPIRICAL_FLOOR_MULTIPLIERS must be a nonempty comma-separated list of finite nonnegative values: $E4_EMPIRICAL_FLOOR_MULTIPLIERS" >&2
  exit 2
fi
if [[ "$empirical_floor_status" -eq 2 ]]; then
  echo "[error] E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE must appear in E4_EMPIRICAL_FLOOR_MULTIPLIERS" >&2
  exit 2
fi
if ! is_nonnegative_finite_number "$E4_EXACT_FLOOR_MULTIPLE"; then
  echo "[error] E4_EXACT_FLOOR_MULTIPLE must be finite and nonnegative: $E4_EXACT_FLOOR_MULTIPLE" >&2
  exit 2
fi
if [[ -n "$E4_EXACT_FLOOR_VALUE" ]] \
    && ! is_nonnegative_finite_number "$E4_EXACT_FLOOR_VALUE"; then
  echo "[error] E4_EXACT_FLOOR_VALUE must be finite and nonnegative: $E4_EXACT_FLOOR_VALUE" >&2
  exit 2
fi
if ! normalize_checkpoints; then
  echo "[error] E4_CHECKPOINTS must be 'all' or a contiguous prefix such as 1,2" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "none" \
      && "$E4_RATIO_MODE" != "empirical" \
      && "$E4_CHECKPOINTS" != "all" ]]; then
  echo "[error] exact/both ratio modes require E4_CHECKPOINTS=all" >&2
  exit 2
fi
if [[ -n "$E4_FD_W_MIN" && -z "$E4_FD_W_MAX" ]] \
    || [[ -z "$E4_FD_W_MIN" && -n "$E4_FD_W_MAX" ]]; then
  echo "[error] E4_FD_W_MIN and E4_FD_W_MAX must be set together" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "empirical" ]]; then
  if [[ ! -f "$SCRIPT_DIR/liu_exact_map_fd.py" ]]; then
    echo "[error] missing solver: $SCRIPT_DIR/liu_exact_map_fd.py" >&2
    exit 2
  fi
  if [[ ! -f "$SCRIPT_DIR/liu_exact_map_core.py" ]]; then
    echo "[error] missing FD core: $SCRIPT_DIR/liu_exact_map_core.py" >&2
    exit 2
  fi
fi
if [[ "$E4_RATIO_MODE" == "empirical" \
      && ! -f "$SCRIPT_DIR/postprocess_empirical_xev_ratio.py" ]]; then
  echo "[error] missing empirical-ratio postprocessor: $SCRIPT_DIR/postprocess_empirical_xev_ratio.py" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "none" \
      && "$E4_RATIO_MODE" != "empirical" \
      && ! -f "$SCRIPT_DIR/aggregate_liu_exact_map.py" ]]; then
  echo "[error] missing exact-ratio aggregator: $SCRIPT_DIR/aggregate_liu_exact_map.py" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "empirical" ]] \
    && ! command -v xargs >/dev/null 2>&1; then
  echo "[error] xargs is required for the process pool" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "empirical" \
      && "$E4_DRY_RUN" != "1" ]] \
    && ! command -v flock >/dev/null 2>&1; then
  echo "[error] flock is required to prevent duplicate output writers" >&2
  exit 2
fi
if [[ "$E4_RATIO_MODE" != "empirical" \
      && "$E4_DRY_RUN" != "1" ]] \
    && ! command -v sha256sum >/dev/null 2>&1; then
  echo "[error] sha256sum is required for protocol-aware resume" >&2
  exit 2
fi

TRAINING_ROOT="${1:-outputs/pres_5seed}"
FD_OUTPUT_ROOT="${2:-${TRAINING_ROOT}/derived/e4_fd_sweep_wmin1p0}"
E4_RATIO_OUTPUT_ROOT="${E4_RATIO_OUTPUT_ROOT:-${FD_OUTPUT_ROOT}/ratio_outputs}"
export E4_RATIO_OUTPUT_ROOT

declare -a TARGETS=()
declare -a SEEDS=()
declare -a DEVICES=()
if [[ "$E4_RUN_FAMILY" == "main" ]]; then
  TARGETS=("$E4_MAIN_LABEL")
else
  IFS=', ' read -r -a TARGETS <<<"$E4_TARGETS"
fi
IFS=', ' read -r -a SEEDS <<<"$E4_SEEDS"
IFS=', ' read -r -a DEVICES <<<"$E4_DEVICE_LIST"

if [[ "${#TARGETS[@]}" -eq 0 || "${#SEEDS[@]}" -eq 0 || "${#DEVICES[@]}" -eq 0 ]]; then
  echo "[error] experiment cells, E4_SEEDS, and E4_DEVICE_LIST must be nonempty" >&2
  exit 2
fi

declare -A target_seen=()
declare -A seed_seen=()
for target in "${TARGETS[@]}"; do
  if [[ -z "$target" || -n "${target_seen[$target]+x}" ]]; then
    echo "[error] targets must be nonempty and unique: $target" >&2
    exit 2
  fi
  target_seen["$target"]=1
done
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ || -n "${seed_seen[$seed]+x}" ]]; then
    echo "[error] seeds must be unique nonnegative integers: $seed" >&2
    exit 2
  fi
  seed_seen["$seed"]=1
done
for device in "${DEVICES[@]}"; do
  if [[ -z "$device" ]]; then
    echo "[error] devices must be nonempty" >&2
    exit 2
  fi
done

missing=0
for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if [[ "$E4_RUN_FAMILY" == "main" ]]; then
      run_dir="${TRAINING_ROOT}/pi-pinn/${E4_MAIN_RUN_STEM}_seed${seed}"
    else
      run_dir="${TRAINING_ROOT}/pi-pinn/pipinn_rho_canonical_v1_m_states1_eval_epochs50000_pres_target${target}_seed${seed}"
    fi
    if [[ ! -d "$run_dir" ]]; then
      echo "[error] missing training run: $run_dir" >&2
      missing=1
    fi
  done
done
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi

if [[ "$E4_RATIO_MODE" == "empirical" ]]; then
  echo "[ratio] mode:          empirical-only (training outer_history.csv)"
  echo "[ratio] run family:    $E4_RUN_FAMILY"
  if [[ "$E4_RUN_FAMILY" == "main" ]]; then
    echo "[ratio] main run stem: $E4_MAIN_RUN_STEM"
    echo "[ratio] main label:    $E4_MAIN_LABEL"
  fi
  echo "[ratio] output root:   $E4_RATIO_OUTPUT_ROOT"
  echo "[ratio] FD solves:     skipped"
  echo "[ratio] floor sweep:   $E4_EMPIRICAL_FLOOR_MULTIPLIERS"
  echo "[ratio] main floor:    $E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE"
  if [[ -n "$E4_EMPIRICAL_FLOOR_VALUE" ]]; then
    echo "[ratio] floor value:   $E4_EMPIRICAL_FLOOR_VALUE (explicit)"
  else
    echo "[ratio] floor value:   per-seed late-tail median"
  fi
  for target in "${TARGETS[@]}"; do
    run_empirical_ratio_target \
      "$TRAINING_ROOT" "$E4_RATIO_OUTPUT_ROOT" "$target"
  done
  if [[ "$E4_DRY_RUN" == "1" ]]; then
    echo "[done] dry-run expanded ${#TARGETS[@]} empirical-only ratio commands"
  else
    echo "[done] empirical-only ratio tables and figures completed"
  fi
  exit 0
fi

if [[ "$E4_DRY_RUN" != "1" ]]; then
  mkdir -p "$FD_OUTPUT_ROOT/logs" "$FD_OUTPUT_ROOT/locks"
fi

task_count=$((${#TARGETS[@]} * ${#SEEDS[@]}))
effective_workers="$E4_MAX_WORKERS"
if (( effective_workers > task_count )); then
  effective_workers="$task_count"
fi
nominal_threads=$((effective_workers * E4_CPU_THREADS))
available_cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf "unknown")"
workers_per_device=$(((effective_workers + ${#DEVICES[@]} - 1) / ${#DEVICES[@]}))

echo "[e4] training root: $TRAINING_ROOT"
echo "[e4] output root:   $FD_OUTPUT_ROOT"
echo "[e4] run family:    $E4_RUN_FAMILY"
if [[ "$E4_RUN_FAMILY" == "main" ]]; then
  echo "[e4] main run stem: $E4_MAIN_RUN_STEM"
  echo "[e4] main label:    $E4_MAIN_LABEL"
  echo "[e4] cells:         ${TARGETS[*]} (E4_TARGETS ignored)"
else
  echo "[e4] targets:       ${TARGETS[*]}"
fi
echo "[e4] seeds:         ${SEEDS[*]}"
if [[ "$E4_CHECKPOINTS" == "all" ]]; then
  echo "[e4] checkpoints:   all (paper schedule)"
else
  echo "[e4] checkpoints:   $E4_CHECKPOINTS (contiguous-prefix pilot)"
fi
echo "[e4] devices:       ${DEVICES[*]}"
echo "[e4] tasks/workers: $task_count/$effective_workers"
echo "[e4] CPU threads:   $E4_CPU_THREADS per worker; nominal cap=$nominal_threads"
echo "[e4] CPU budget:    $E4_CPU_BUDGET; online cores=$available_cores"
echo "[e4] device load:   at most about $workers_per_device workers/device"
echo "[e4] drift scheme:  $E4_DRIFT_SCHEME"
echo "[e4] refinement:    $E4_REFINEMENT_RULE"
echo "[e4] boundary role: report-only matched-BVP sensitivity"
echo "[e4] paper floor:   $E4_MIN_PAPER_CHECKPOINT (0 excludes nothing)"
echo "[e4] ratio mode:    $E4_RATIO_MODE"
if [[ "$E4_RATIO_MODE" != "none" ]]; then
  echo "[e4] ratio output:  $E4_RATIO_OUTPUT_ROOT"
fi
if [[ "$E4_RATIO_MODE" == "exact" || "$E4_RATIO_MODE" == "both" ]]; then
  echo "[e4] exact floor:   multiple=$E4_EXACT_FLOOR_MULTIPLE"
  if [[ -n "$E4_EXACT_FLOOR_VALUE" ]]; then
    echo "[e4] exact value:   $E4_EXACT_FLOOR_VALUE (explicit)"
  else
    echo "[e4] exact value:   per-seed late-input median"
  fi
fi

if (( nominal_threads > E4_CPU_BUDGET )); then
  echo "[warning] nominal worker-thread cap $nominal_threads exceeds E4_CPU_BUDGET=$E4_CPU_BUDGET" >&2
fi
if [[ "${#DEVICES[@]}" -eq 1 && "$effective_workers" -gt 8 && "${DEVICES[0]}" == cuda:* ]]; then
  echo "[warning] $effective_workers processes share ${DEVICES[0]}; monitor GPU memory or provide more devices" >&2
fi

task_file="$(mktemp)"
cleanup() {
  rm -f -- "$task_file"
}
trap cleanup EXIT

job_index=0
for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    device="${DEVICES[$((job_index % ${#DEVICES[@]}))]}"
    printf "%s\0%s\0%s\0" "$target" "$seed" "$device" >>"$task_file"
    job_index=$((job_index + 1))
  done
done

if ! xargs -0 -n 3 -P "$effective_workers" \
  bash "$SCRIPT_PATH" --worker "$TRAINING_ROOT" "$FD_OUTPUT_ROOT" \
  <"$task_file"; then
  echo "[error] one or more E4 jobs failed; inspect $FD_OUTPUT_ROOT/logs" >&2
  exit 1
fi

if [[ "$E4_RATIO_MODE" == "exact" || "$E4_RATIO_MODE" == "both" ]]; then
  for target in "${TARGETS[@]}"; do
    run_exact_ratio_target \
      "$FD_OUTPUT_ROOT" "$E4_RATIO_OUTPUT_ROOT" \
      "$target" "$E4_RATIO_MODE"
  done
fi

if [[ "$E4_DRY_RUN" == "1" ]]; then
  echo "[done] dry-run expanded all $task_count commands"
else
  echo "[done] all E4 jobs completed or were already successful"
  if [[ "$E4_CHECKPOINTS" == "all" ]]; then
    if [[ "$E4_RUN_FAMILY" == "main" ]]; then
      if [[ "$E4_RATIO_MODE" == "exact" \
            || "$E4_RATIO_MODE" == "both" ]]; then
        echo "[done] main exact-map ratio tables/figure:"
        echo "       ${E4_RATIO_OUTPUT_ROOT}/${E4_MAIN_LABEL}/exact_audit"
      else
        echo "[next] aggregate and plot the main exact-map cell with:"
        printf "  %q %q --out-root %q --expected-seeds %q --min-seeds %q --plot-ratios --ratio-series exact --floor-multiple %q --formats %q --ratio-y-scale %q --target-label %q --output %q --overwrite\n" \
          "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_liu_exact_map.py" \
          "${FD_OUTPUT_ROOT}/${E4_MAIN_LABEL}" \
          "$E4_SEEDS" "${#SEEDS[@]}" "$E4_EXACT_FLOOR_MULTIPLE" \
          "$E4_RATIO_FORMATS" "$E4_RATIO_Y_SCALE" "$E4_MAIN_LABEL" \
          "${E4_RATIO_OUTPUT_ROOT}/${E4_MAIN_LABEL}/exact_audit"
        echo "       add --allow-partial-sensitivity only for an explicitly exploratory failed-refinement report"
      fi
    else
      echo "[next] aggregate with:"
      printf "  %q %q --out-root %q --expected-tolerances %q --expected-seeds %q --min-runs-per-tolerance %q --checkpoints all --refinement-failure-mode report --output %q --plot --plot-metric X --formats png,pdf --overwrite\n" \
        "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_e4_tolerance.py" "$FD_OUTPUT_ROOT" \
        "$E4_TARGETS" "$E4_SEEDS" "${#SEEDS[@]}" \
        "${FD_OUTPUT_ROOT}_paper"
    fi
  else
    echo "[pilot] explicit checkpoint prefixes are diagnostic only:"
    echo "        analysis_mode=contiguous_prefix_exact_map_and_e4_pilot"
    echo "        paper_aggregation_eligible=false"
    echo "        rerun with E4_CHECKPOINTS=all in a full-protocol output root before aggregation"
  fi
fi
