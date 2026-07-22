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
#   SEEDS="1,2" N_ASSETS_LIST="10" DEVICE_LIST="cuda:0,cuda:1" bash tune_merton.sh out 2
#   PINN_OVERRIDES="policy_bounds_mode=none" PIPINN_OVERRIDES="policy_bounds_mode=none" \
#     bash tune_merton.sh out  # no finite action boxes in either method
#   PIPINN_OVERRIDES="e3b_checkpoints=1" bash tune_merton.sh out  # exact-map trajectory
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
ALLOW_LEGACY_BEST_EVAL="${ALLOW_LEGACY_BEST_EVAL:-0}"  # eval-only diagnostic/legacy fallback
AGGREGATE="${AGGREGATE:-1}"          # 1: run aggregate_seeds after the sweep

# Paper sweep defaults: N={10,50}, methods={PINN,PI-PINN}, and ten fixed
# training seeds {1,2,3,5,7,11,17,23,42,101}.
# When SEEDS is set, every run_* call WITHOUT an explicit
# seed=... override expands into one job per seed (seed goes into the tag, so
# each seed gets its own output/weight dir). market_seed stays fixed so all
# seeds solve the SAME market.
SEEDS="${SEEDS:-1,2,3,5,7,11,17,23,42,101}"
N_ASSETS_LIST="${N_ASSETS_LIST:-10,50}"
PINN_OVERRIDES="${PINN_OVERRIDES:-}"
PIPINN_OVERRIDES="${PIPINN_OVERRIDES:-}"

# Cap CPU thread pools: parallel workers otherwise oversubscribe cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"

SEED_LIST=()
IFS=', ' read -ra SEED_LIST <<< "$SEEDS"
N_ASSET_VALUES=()
IFS=', ' read -ra N_ASSET_VALUES <<< "$N_ASSETS_LIST"
PINN_OVERRIDE_ARGS=()
PIPINN_OVERRIDE_ARGS=()
[[ -n "$PINN_OVERRIDES" ]] && read -r -a PINN_OVERRIDE_ARGS <<< "$PINN_OVERRIDES"
[[ -n "$PIPINN_OVERRIDES" ]] && read -r -a PIPINN_OVERRIDE_ARGS <<< "$PIPINN_OVERRIDES"
(( ${#SEED_LIST[@]} > 0 )) || { echo "[error] SEEDS must not be empty" >&2; exit 2; }
(( ${#N_ASSET_VALUES[@]} > 0 )) || { echo "[error] N_ASSETS_LIST must not be empty" >&2; exit 2; }
echo "[info] seeds: ${SEED_LIST[*]}"
echo "[info] asset dimensions: ${N_ASSET_VALUES[*]}"

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
    # Evaluation/output-only overrides must keep the training identity so an
    # eval-only rerun finds the existing official weights.
    case "$k" in
      allow_legacy_best_eval|eval_only|test_points|eval_margin|n_tau|n_x|skip_figures|skip_eval|skip_plots)
        continue
        ;;
    esac
    parts+=("$(sanitize "${k}${v}")")
  done
  if [[ ${#parts[@]} -eq 0 ]]; then echo "${model}_baseline"; return; fi
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
  [batch_size]=10000
  [terminal_frac]=0.25
  [lr]=5e-4
  [outer_iters]=20
  [eval_epochs]=2000
  [resample_every]=200
  [scheduler_patience]=5000
  [scheduler_factor]=0.5
  [scheduler_min_lr]=1e-8
  [lr_schedule]=plateau
  # Direct nonlinear-HJB guard. Use PINN_OVERRIDES or an explicit run_pinn
  # override for comparisons with the historical abs/hard continuations.
  [hjb_guard_mode]=softplus
  [hjb_vy_guard_eps]=1e-8
  [hjb_denom_guard_eps]=1e-8
  [hjb_vy_guard_tau]=0.05
  [hjb_denom_guard_tau]=0.05
  [hjb_vy_guard_tau_min]=1e-4
  [hjb_denom_guard_tau_min]=1e-4
  # First softplus pilot keeps tau fixed. Set a positive number of outer
  # blocks here (or via PINN_OVERRIDES) to enable staged continuation.
  [hjb_guard_anneal_every]=0
  [hjb_guard_anneal_factor]=0.5
  [w_terminal]=10.0
  [w_shape]=1.0
  [w_eta]=1.5
  [eta_clip]=10.0
  [policy_bounds_mode]=stabilized
  [pi_clip_abs]="${PI_CLIP_ABS:-2.0}"
  [pres_target]=none
  [val_points]=50000
  [val_terminal_points]=10000
  [val_every]=25
  [save_iterate_every]=0
  [diag_points]=8192
  [diag_every]=1
  [print_every]=1000
  # First margin = PRIMARY (diagnostic + representative metric).
  [eval_margin]="0.10,0.0,0.05,0.15,0.20,0.25,0.30"
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
  [policy_bounds_mode]=stabilized
  [pi_clip_abs]="${PI_CLIP_ABS:-2.0}"
  [kappa_max_bound]=3.0
  [utility_cap]=1e3
  [value_hidden]=256
  [value_depth]=3
  [batch_size]=10000
  [terminal_frac]=0.5
  [lr]=5e-4
  [outer_iters]=20
  [eval_epochs]=2000
  [scheduler_patience]=20
  [scheduler_factor]=0.5
  [scheduler_min_lr]=1e-8
  [lr_schedule]=carry_plateau
  [adam_reset]=keep
  [carry_lr_min]=1e-8
  [carry_lr_max]=5e-4
  # Paper alignment: nondegenerate initial policy (Assumption 1). Override
  # pi_init_method=zero c_init_method=zero to reproduce the ipynb runs.
  [pi_init_method]=myopic
  # Bash/public name intentionally follows the Liu experiment vocabulary;
  # build_flags maps it to Python's Merton-specific --pi-init-scale.
  [theta_init_scale]="${THETA_INIT_SCALE:-0.5}"
  [c_init_method]=proportional
  [w_terminal]=10.0
  [w_shape]=1.0
  [w_eta]=3.0
  [eta_clip]=10.0
  [eta_focus_w]=none
  [pres_target]=none
  [val_points]=50000
  [val_terminal_points]=10000
  [val_every]=25
  [inner_best_restore]=1
  [sel_points]=10000
  [sel_terminal_points]=2000
  [sel_every]=50
  [sel_patience]=0
  [pe_resample_every]="${PE_RESAMPLE_EVERY:-0}"
  [save_iterate_every]=0
  [diag_points]=8192
  [diag_every]=1
  [print_every_outer]=1
  [print_every_eval]=0
  [eval_margin]="0.10,0.0,0.05,0.15,0.20,0.25,0.30"
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
    local cli_key="${k//_/-}"
    [[ "$k" == "theta_init_scale" ]] && cli_key="pi-init-scale"
    case "$k" in
      timing_mode|skip_figures|skip_eval|skip_plots|e3b_checkpoints|eval_only|allow_legacy_best_eval)
        case "${v,,}" in
          1|true|yes|on) printf -v out '%s %q' "$out" "--$cli_key" ;;
          0|false|no|off|"") ;;
          *) echo "[error] boolean override $k must be 0/1/true/false; got: $v" >&2; return 2 ;;
        esac
        ;;
      *) printf -v out '%s %q %q' "$out" "--$cli_key" "$v" ;;
    esac
  done
  # any override key NOT in BASE (e.g. m_states passthrough) is appended too
  for k in "${!OVR[@]}"; do
    if [[ -z "${BASE[$k]+x}" ]]; then
      local cli_key="${k//_/-}"
      [[ "$k" == "theta_init_scale" ]] && cli_key="pi-init-scale"
      local v="${OVR[$k]}"
      case "$k" in
        timing_mode|skip_figures|skip_eval|skip_plots|e3b_checkpoints|eval_only|allow_legacy_best_eval)
          case "${v,,}" in
            1|true|yes|on) printf -v out '%s %q' "$out" "--$cli_key" ;;
            0|false|no|off|"") ;;
            *) echo "[error] boolean override $k must be 0/1/true/false; got: $v" >&2; return 2 ;;
          esac
          ;;
        *) printf -v out '%s %q %q' "$out" "--$cli_key" "$v" ;;
      esac
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
  if [[ -n "${ENQUEUED_TAGS[$tag]+x}" ]]; then
    echo "[error] duplicate resolved run tag: $tag" >&2
    echo "        Check global overrides versus explicit run_* variants." >&2
    return 2
  fi
  ENQUEUED_TAGS["$tag"]=1
  local out_dir="$OUT_ROOT/$model/$tag"
  local weight_dir="$OUT_ROOT/weights/$model/$tag"
  local log="$LOG_DIR/${tag}.log"

  # Never trust mutually inconsistent terminal markers.
  local marker_count=0 marker
  for marker in _SUCCESS _STOPPED_EARLY _FAILED; do
    [[ -f "$out_dir/$marker" ]] && marker_count=$((marker_count + 1))
  done
  if (( marker_count > 1 )); then
    echo "[warn] $tag has conflicting terminal markers; queuing a clean rerun"
  fi

  # Skip policy: only one unambiguous success is complete unless forced.
  if [[ "$FORCE_RERUN" != "1" && "$EVAL_ONLY" != "1" && -f "$out_dir/_SUCCESS" ]]; then
    if (( marker_count == 1 )); then
      echo "[skip] $tag (already _SUCCESS)"
      return
    fi
  fi

  # Placeholder DEVICE token replaced per worker at launch time.
  local cmd tail
  printf -v cmd '%q %q' "$PYTHON_BIN" "$script"
  cmd+="$flags"
  printf -v tail ' %q %q %q %q %q %q %q %q %q %q' \
    --run-tag "$tag" --model-type "$model" \
    --output-root "$OUT_ROOT/$model" --weight-root "$weight_dir" \
    --device __DEV__
  cmd+="$tail"
  [[ "$EVAL_ONLY" == "1" ]] && cmd+=" --eval-only"
  if [[ "$EVAL_ONLY" == "1" && "$ALLOW_LEGACY_BEST_EVAL" == "1" ]]; then
    cmd+=" --allow-legacy-best-eval"
  fi
  printf "%s\t%s\t%s\t%s\n" "$tag" "$out_dir" "$log" "$cmd" >> "$JOB_QUEUE"
}

# Expand a run into per-seed jobs when SEEDS is set and no explicit seed given.
run_pinn()   { _run_model pinn   "$@"; }
run_pipinn() { _run_model pipinn "$@"; }
declare -A ENQUEUED_N=()
declare -A ENQUEUED_TAGS=()
_run_model() {
  local model="$1"; shift
  # Environment-level overrides apply to every manual run line.  Resolve
  # duplicate keys once, with the per-run arguments appended last so a tuning
  # line can intentionally specialize a global setting without emitting
  # duplicate CLI flags or a misleading auto-tag.
  local -a all_overrides=()
  if [[ "$model" == "pinn" ]]; then
    all_overrides=("${PINN_OVERRIDE_ARGS[@]}" "$@")
  else
    all_overrides=("${PIPINN_OVERRIDE_ARGS[@]}" "$@")
  fi
  declare -A resolved_overrides=()
  local -a override_order=()
  local kv key value
  for kv in "${all_overrides[@]}"; do
    [[ "$kv" == *=* ]] || {
      echo "[error] override must have key=value form: $kv" >&2
      exit 2
    }
    key="${kv%%=*}"
    value="${kv#*=}"
    [[ -n "$key" ]] || {
      echo "[error] override key must not be empty: $kv" >&2
      exit 2
    }
    # Both spellings map to Python's --pi-init-scale. Canonicalize before
    # deduplication so a later per-run value always wins over the global one.
    [[ "$key" == "pi_init_scale" ]] && key="theta_init_scale"
    if [[ -z "${resolved_overrides[$key]+x}" ]]; then
      override_order+=("$key")
    fi
    resolved_overrides["$key"]="$value"
  done
  local -a merged_overrides=()
  for key in "${override_order[@]}"; do
    merged_overrides+=("$key=${resolved_overrides[$key]}")
  done

  local has_seed=0 n_val=""
  for kv in "${merged_overrides[@]}"; do
    [[ "${kv%%=*}" == "seed" ]] && has_seed=1
    [[ "${kv%%=*}" == "n_assets" ]] && n_val="${kv#*=}"
  done
  if [[ -z "$n_val" ]]; then
    if [[ "$model" == "pinn" ]]; then n_val="${BASE_PINN[n_assets]}"; else n_val="${BASE_PIPINN[n_assets]}"; fi
  fi
  [[ "$n_val" =~ ^[1-9][0-9]*$ ]] || { echo "[error] invalid n_assets: $n_val" >&2; exit 2; }
  ENQUEUED_N["$n_val"]=1
  if [[ ${#SEED_LIST[@]} -gt 0 && $has_seed -eq 0 ]]; then
    for sd in "${SEED_LIST[@]}"; do
      enqueue "$model" "${merged_overrides[@]}" "seed=$sd"
    done
  else
    enqueue "$model" "${merged_overrides[@]}"
  fi
}

next_job_index() {
  local idx
  { flock -x 9; idx="$(cat "$JOB_CURSOR")"; echo $((idx + 1)) > "$JOB_CURSOR"; } 9>"$JOB_LOCK"
  echo "$idx"
}

worker_loop() {
  local worker_id="$1" dev="$2"
  local failures=0
  while true; do
    local idx line; idx="$(next_job_index)"
    line="$(sed -n "$((idx + 1))p" "$JOB_QUEUE" || true)"
    [[ -z "$line" ]] && break
    local tag out_dir log cmd
    IFS=$'\t' read -r tag out_dir log cmd <<< "$line"
    cmd="${cmd//__DEV__/$dev}"
    echo "[run ] $tag on $dev (worker $worker_id)"
    mkdir -p "$out_dir"
    if [[ "$EVAL_ONLY" == "1" ]]; then
      rm -f "$out_dir/_SUCCESS_EVAL" "$out_dir/_FAILED_EVAL"
    else
      # Clear stale/conflicting markers immediately before launch. If Python
      # fails before constructing its recorder, the shell creates _FAILED.
      rm -f "$out_dir/_SUCCESS" "$out_dir/_STOPPED_EARLY" "$out_dir/_FAILED"
    fi
    if bash -c "$cmd" >"$log" 2>&1; then
      echo "[ok  ] $tag"
    else
      echo "[FAIL] $tag (see $log)"
      failures=$((failures + 1))
      if [[ "$EVAL_ONLY" == "1" ]]; then
        touch "$out_dir/_FAILED_EVAL"
      else
        touch "$out_dir/_FAILED"
      fi
    fi
  done
  (( failures == 0 ))
}

run_all_jobs() {
  local total; total="$(wc -l < "$JOB_QUEUE" | tr -d ' ')"
  if (( total == 0 )); then echo "[done] no queued jobs."; return 0; fi
  echo "[info] queued jobs: $total | devices: ${DEVICES[*]} | workers: $MAX_WORKERS"
  local pids=()
  for ((i=0; i<MAX_WORKERS; i++)); do
    worker_loop "$((i + 1))" "${SLOTS[$i]}" & pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

# Paper default: N in {10, 50} x both methods. Edit/add run_* lines freely;
# tuning variants get their own tag (=own output/weight dir) automatically.
#   run_pipinn n_assets=10 theta_init_scale=0.5
#   run_pipinn n_assets=50 w_eta=1.0 sel_patience=3
#   run_pinn   n_assets=10 seed=1            # single-seed (no SEEDS expansion)

for n_assets in "${N_ASSET_VALUES[@]}"; do
  run_pinn   "n_assets=$n_assets"
  run_pinn   "n_assets=$n_assets" outer_iters=30
  run_pipinn "n_assets=$n_assets"
  run_pipinn "n_assets=$n_assets" outer_iters=30
done





if ! run_all_jobs; then
  echo "[error] at least one Merton job failed; aggregation is not run." >&2
  exit 1
fi

if [[ "$AGGREGATE" == "1" && "$EVAL_ONLY" != "1" ]]; then
  echo "[info] aggregating seeds -> $OUT_ROOT"
  "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_seeds.py" \
    --out-root "$OUT_ROOT" \
    --expected-seeds "$SEEDS" \
    --expected-n-assets "$(printf '%s\n' "${!ENQUEUED_N[@]}" | sort -n | paste -sd, -)" \
    --expected-m-states "1" \
    --expected-models "pinn,pipinn" \
    --strict-market-snapshots
fi

echo "[done] Merton sweep complete: $OUT_ROOT"
