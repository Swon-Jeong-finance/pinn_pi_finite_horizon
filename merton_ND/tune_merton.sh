#!/usr/bin/env bash
set -euo pipefail

# Merton (with-consumption) PINN / PI-PINN sweep driver.
# Independent of the Liu tune script; mirrors its UX (BASE dicts, seed sweep,
# CPU thread caps, per-run tags/dirs, skip/rerun policy, parallel device
# slots, automatic aggregation) but for the Merton argparse.
#
# Usage:
#   bash tune_merton.sh [OUT_ROOT] [MAX_PARALLEL]
# Examples:
#   DEVICE_LIST="cuda:0,cuda:1" bash tune_merton.sh outputs_merton/run 2
#   SEEDS="1,2,3,4,5" DEVICE_LIST="cuda:0,cuda:1,cuda:2" bash tune_merton.sh out 3
#   AGGREGATE=0 bash tune_merton.sh ...        # skip the seed-aggregation step
#   FORCE_RERUN=1 bash tune_merton.sh ...      # ignore previous _SUCCESS
#   EVAL_ONLY=1 bash tune_merton.sh ...        # re-evaluate from saved weights

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${1:-$(pwd)/outputs/tune_merton_$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL_ARG="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PINN_SCRIPT="${PINN_SCRIPT:-$SCRIPT_DIR/merton_nd_consumption_pinn.py}"
PIPINN_SCRIPT="${PIPINN_SCRIPT:-$SCRIPT_DIR/merton_nd_consumption_pi_pinn.py}"
mkdir -p "$OUT_ROOT"
LOG_DIR="$OUT_ROOT/logs"; mkdir -p "$LOG_DIR"

FORCE_RERUN="${FORCE_RERUN:-0}"      # 1: rerun regardless of previous status
EVAL_ONLY="${EVAL_ONLY:-0}"          # 1: skip training, re-evaluate from weights
AGGREGATE="${AGGREGATE:-1}"          # 1: run aggregate_seeds after the sweep

# Multi-seed sweep. When SEEDS is set, every run_* call WITHOUT an explicit
# seed=... override expands into one job per seed (seed goes into the tag, so
# each seed gets its own output/weight dir). market_seed stays fixed so all
# seeds solve the SAME market.
SEEDS="${SEEDS:-}"

# Cap CPU thread pools: parallel workers otherwise oversubscribe cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"

SEED_LIST=()
if [[ -n "$SEEDS" ]]; then
  IFS=', ' read -ra SEED_LIST <<< "$SEEDS"
  echo "[info] multi-seed mode: seeds = ${SEED_LIST[*]}"
fi

# Device worker queue.
DEVICE="${DEVICE:-cuda:0}"
DEVICE_LIST="${DEVICE_LIST:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
if [[ -n "$DEVICE_LIST" ]]; then
  IFS=',' read -ra DEVICES <<< "$DEVICE_LIST"
else
  DEVICES=("$DEVICE")
fi
(( JOBS_PER_GPU < 1 )) && { echo "[error] JOBS_PER_GPU must be >= 1" >&2; exit 2; }

SLOTS=()
for dev in "${DEVICES[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do SLOTS+=("$dev"); done
done
if [[ -n "$MAX_PARALLEL_ARG" ]]; then
  MAX_WORKERS="$MAX_PARALLEL_ARG"
  (( MAX_WORKERS < 1 )) && { echo "[error] MAX_PARALLEL must be >= 1" >&2; exit 2; }
  (( MAX_WORKERS > ${#SLOTS[@]} )) && MAX_WORKERS="${#SLOTS[@]}"
else
  MAX_WORKERS="${#SLOTS[@]}"
fi

JOB_QUEUE="$OUT_ROOT/_jobs.tsv"
JOB_CURSOR="$OUT_ROOT/_jobs.cursor"
JOB_LOCK="$OUT_ROOT/_jobs.lock"
: > "$JOB_QUEUE"; echo 0 > "$JOB_CURSOR"

sanitize() { echo "$1" | tr ' /:=,|' '______' | tr '-' '_'; }

auto_tag() {
  local model="$1"; shift
  if [[ $# -eq 0 ]]; then echo "${model}_baseline"; return; fi
  local parts=()
  for kv in "$@"; do
    local k=${kv%%=*}; local v=${kv#*=}
    parts+=("$(sanitize "${k}${v}")")
  done
  printf "%s_%s" "$model" "$(IFS=_; echo "${parts[*]}")"
}

# ==============================
# Baselines (change here once). Mirrors the Merton ipynb / argparse defaults.
# N_ASSETS 50, market fixed by market_seed; state is wealth-only (m_states=1).
# ==============================
declare -A BASE_PINN=(
  [n_assets]=50
  [seed]=12
  [market_seed]=12
  [gamma]=2.0
  [rho_discount]=0.04
  [r]=0.03
  [epsilon_bequest]=1.0
  [tau_max]=1.0
  [w_min]=0.1
  [w_max]=2.0
  [sigma_lo]=0.10
  [sigma_hi]=0.25
  [rho_max]=1.0
  [kappa_max]=30.0
  [pi_scale]=0.6
  [mu_noise_rel]=0.02
  [value_hidden]=256
  [value_depth]=3
  [batch_size]=3000
  [lr]=5e-4
  [outer_iters]=500
  [eval_epochs]=200
  [resample_every]=200
  [w_terminal]=10.0
  [w_shape]=1.0
  [w_eta]=1.5
  [eta_clip]=10.0
  [print_every]=5000
  # First margin = PRIMARY (diagnostic + representative metric).
  [eval_margin]="0.10,0.0,0.05,0.15,0.20"
  [test_points]=100000
  [n_tau]=100
  [n_x]=100
)

declare -A BASE_PIPINN=(
  [n_assets]=50
  [seed]=12
  [market_seed]=12
  [gamma]=2.0
  [rho_discount]=0.04
  [r]=0.03
  [epsilon_bequest]=1.0
  [tau_max]=1.0
  [w_min]=0.1
  [w_max]=2.0
  [sigma_lo]=0.10
  [sigma_hi]=0.25
  [rho_max]=1.0
  [kappa_max]=30.0
  [pi_scale]=0.6
  [mu_noise_rel]=0.02
  [pi_min]=-2.0
  [pi_max]=2.0
  [kappa_max_bound]=3.0
  [utility_cap]=1e3
  [value_hidden]=256
  [value_depth]=3
  [batch_size]=3000
  [lr]=5e-4
  [outer_iters]=500
  [eval_epochs]=200
  [scheduler_patience]=10
  [scheduler_factor]=0.5
  [scheduler_min_lr]=1e-8
  # Paper alignment: nondegenerate initial policy (Assumption 1). Override
  # pi_init_method=zero c_init_method=zero to reproduce the ipynb runs.
  [pi_init_method]=myopic
  [c_init_method]=proportional
  [w_terminal]=10.0
  [w_eta]=3.0
  [eta_clip]=10.0
  [eta_focus_w]=none
  [print_every_outer]=10
  [print_every_eval]=0
  [eval_margin]="0.10,0.0,0.05,0.15,0.20"
  [test_points]=100000
  [n_tau]=100
  [n_x]=100
)

# Build one --flag string from a BASE dict plus overrides (key=val ...).
# Overrides win; keys are the dict keys with _ -> - for the CLI.
build_flags() {
  local -n BASE=$1; shift
  declare -A OVR=()
  for kv in "$@"; do OVR[${kv%%=*}]="${kv#*=}"; done
  local out=""
  local k
  for k in "${!BASE[@]}"; do
    local v="${OVR[$k]:-${BASE[$k]}}"
    out+=" --${k//_/-} ${v}"
  done
  # any override key NOT in BASE (e.g. m_states passthrough) is appended too
  for k in "${!OVR[@]}"; do
    if [[ -z "${BASE[$k]+x}" ]]; then
      out+=" --${k//_/-} ${OVR[$k]}"
    fi
  done
  printf "%s" "$out"
}

enqueue() {
  local model="$1"; shift
  local script tag flags
  if [[ "$model" == "pinn" ]]; then
    script="$PINN_SCRIPT"; flags="$(build_flags BASE_PINN "$@")"
  else
    script="$PIPINN_SCRIPT"; flags="$(build_flags BASE_PIPINN "$@")"
  fi
  tag="$(auto_tag "$model" "$@")"
  local out_dir="$OUT_ROOT/$model/$tag"
  local weight_dir="$out_dir/weights"
  local log="$LOG_DIR/${tag}.log"

  # Skip policy: a completed run has _SUCCESS unless FORCE_RERUN.
  if [[ "$FORCE_RERUN" != "1" && "$EVAL_ONLY" != "1" && -f "$out_dir/_SUCCESS" ]]; then
    echo "[skip] $tag (already _SUCCESS)"
    return
  fi
  local eval_flag=""
  [[ "$EVAL_ONLY" == "1" ]] && eval_flag=" --eval-only"

  # Placeholder DEVICE token replaced per worker at launch time.
  local cmd="$PYTHON_BIN $script${flags} --run-tag $tag --model-type $model"
  cmd+=" --output-root $OUT_ROOT/$model --weight-root $weight_dir${eval_flag} --device __DEV__"
  printf "%s\t%s\t%s\t%s\n" "$tag" "$out_dir" "$log" "$cmd" >> "$JOB_QUEUE"
}

# Expand a run into per-seed jobs when SEEDS is set and no explicit seed given.
run_pinn()   { _run_model pinn   "$@"; }
run_pipinn() { _run_model pipinn "$@"; }
_run_model() {
  local model="$1"; shift
  local has_seed=0
  for kv in "$@"; do [[ "${kv%%=*}" == "seed" ]] && has_seed=1; done
  if [[ ${#SEED_LIST[@]} -gt 0 && $has_seed -eq 0 ]]; then
    for sd in "${SEED_LIST[@]}"; do enqueue "$model" "$@" "seed=$sd"; done
  else
    enqueue "$model" "$@"
  fi
}

next_job_index() {
  local idx
  { flock -x 9; idx="$(cat "$JOB_CURSOR")"; echo $((idx + 1)) > "$JOB_CURSOR"; } 9>"$JOB_LOCK"
  echo "$idx"
}

worker_loop() {
  local worker_id="$1" dev="$2"
  while true; do
    local idx line; idx="$(next_job_index)"
    line="$(sed -n "$((idx + 1))p" "$JOB_QUEUE" || true)"
    [[ -z "$line" ]] && break
    local tag out_dir log cmd
    IFS=$'\t' read -r tag out_dir log cmd <<< "$line"
    cmd="${cmd//__DEV__/$dev}"
    echo "[run ] $tag on $dev (worker $worker_id)"
    mkdir -p "$out_dir"
    if bash -c "$cmd" >"$log" 2>&1; then
      echo "[ok  ] $tag"
    else
      echo "[FAIL] $tag (see $log)"
    fi
  done
}

run_all_jobs() {
  local total; total="$(wc -l < "$JOB_QUEUE" | tr -d ' ')"
  if (( total == 0 )); then echo "[done] no queued jobs."; return; fi
  echo "[info] queued jobs: $total | devices: ${DEVICES[*]} | workers: $MAX_WORKERS"
  local pids=()
  for ((i=0; i<MAX_WORKERS; i++)); do
    worker_loop "$((i + 1))" "${SLOTS[$i]}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

# =========================================================================
# Sweep definition. Merton state is wealth-only, so there is NO m_states
# sweep (fixed at 1, recorded by the scripts). The paper needs both methods
# on the same market across the seed sweep; add tuning variants as extra
# run_* lines (e.g. run_pipinn w_eta=1.0).
# =========================================================================
run_pinn   outer_iters=3
run_pipinn outer_iters=3

run_all_jobs

if [[ "$AGGREGATE" == "1" && "$EVAL_ONLY" != "1" ]]; then
  echo "[info] aggregating seeds -> $OUT_ROOT"
  "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_seeds.py" --out-root "$OUT_ROOT" || \
    echo "[warn] aggregation step failed (non-fatal)."
fi

echo "[done] Merton sweep complete: $OUT_ROOT"
