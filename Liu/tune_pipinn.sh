#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tune_pipinn.sh [OUT_ROOT] [MAX_PARALLEL]
#
# Examples:
#   DEVICE_LIST="cuda:0,cuda:1,cuda:2,cuda:3" bash tune_pipinn.sh outputs 4
#   DEVICE_LIST="cuda:1,cuda:2,cuda:3" bash tune_pipinn.sh /workspace/outputs/my_run 3
#   JOBS_PER_GPU=2 DEVICE_LIST="cuda:1,cuda:2,cuda:3" bash tune_pipinn.sh /workspace/outputs/my_run
#   FORCE_RERUN=1 DEVICE_LIST="cuda:1,cuda:2,cuda:3" bash tune_pipinn.sh /workspace/outputs/my_run
#   RERUN_STOPPED=1 DEVICE_LIST="cuda:1,cuda:2,cuda:3" bash tune_pipinn.sh /workspace/outputs/my_run
#   SEEDS="1,2,3,4,5,6,7,8,9,10" DEVICE_LIST="cuda:0,cuda:1" bash tune_pipinn.sh /workspace/outputs/main10seed
#   AGGREGATE=0 bash tune_pipinn.sh ...   # skip the automatic seed aggregation step
#   STRICT_PAPER_AGGREGATION=0 SEEDS="1,2" bash tune_pipinn.sh ...  # do not fail on an incomplete aggregate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${1:-$(pwd)/outputs/tune_liu_$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL_ARG="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SWEEP_PROFILE="${SWEEP_PROFILE:-main}"  # main | nonaffine
mkdir -p "$OUT_ROOT"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"
STOP_FLAG_ROOT="${STOP_FLAG_ROOT:-$OUT_ROOT/_STOP_FLAGS}"
mkdir -p "$STOP_FLAG_ROOT"

# Re-run / skip policy.
FORCE_RERUN="${FORCE_RERUN:-0}"       # 1: rerun regardless of previous status
RERUN_STOPPED="${RERUN_STOPPED:-0}"   # 1: rerun only runs marked _STOPPED_EARLY
EVAL_ONLY="${EVAL_ONLY:-0}"           # 1: skip training, re-evaluate from existing weights

# PDE early-stop policy shared by PINN and PI-PINN.
PDE_STOP_THRESHOLD="${PDE_STOP_THRESHOLD:-10.0}"
PDE_STOP_START_OUTER="${PDE_STOP_START_OUTER:-100}"
PDE_STOP_PATIENCE="${PDE_STOP_PATIENCE:-20}"

# Multi-seed sweep.
#   SEEDS="1,2,3" bash tune_pipinn.sh ...
# When SEEDS is set, every run_pinn/run_pipinn call WITHOUT an explicit
# seed=... override is expanded into one job per seed (seed goes into the
# auto tag, so each seed gets its own output/weight directory). Calls that
# pass seed=... explicitly are left untouched. When SEEDS is empty, behavior
# is identical to the original single-seed script (BASE seed).
SEEDS="${SEEDS:-}"
# A strict aggregate requires every seed requested by the caller, but does not
# prescribe either the seed values or their count.  This keeps the launcher
# usable for pilots and custom replication designs while still detecting a
# missing run.  Disable only if an incomplete aggregate is acceptable.
STRICT_PAPER_AGGREGATION="${STRICT_PAPER_AGGREGATION:-1}"

# Cap CPU thread pools: parallel workers otherwise oversubscribe cores
# (each process would spawn nproc-sized OMP/MKL pools).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"

SEED_LIST=()
if [[ -n "$SEEDS" ]]; then
  IFS=', ' read -ra SEED_LIST <<< "$SEEDS"
  echo "[info] multi-seed mode: seeds = ${SEED_LIST[*]}"
  # Validate uniqueness for every profile.  Duplicate seeds enqueue the
  # same tag/output directory twice and can corrupt CSV/checkpoint writes
  # when workers run concurrently.
  declare -A _seed_seen=()
  for _seed in "${SEED_LIST[@]}"; do
    if [[ ! "$_seed" =~ ^[0-9]+$ ]]; then
      echo "[error] training seeds must be non-negative integers; got: $_seed" >&2
      exit 2
    fi
    _seed_key=$((10#$_seed))
    if [[ -n "${_seed_seen[$_seed_key]+x}" ]]; then
      echo "[error] duplicate training seed: $_seed" >&2
      exit 2
    fi
    _seed_seen[$_seed_key]=1
  done
fi

# Device worker queue.
# DEVICE_LIST="cuda:1,cuda:2,cuda:3" with JOBS_PER_GPU=1 means one job per GPU.
# Increase JOBS_PER_GPU to allow multiple simultaneous jobs on each GPU.
DEVICE="${DEVICE:-cuda}"
DEVICE_LIST="${DEVICE_LIST:-}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
if [[ -n "$DEVICE_LIST" ]]; then
  IFS=',' read -ra DEVICES <<< "$DEVICE_LIST"
else
  DEVICES=("$DEVICE")
fi
if (( JOBS_PER_GPU < 1 )); then
  echo "[error] JOBS_PER_GPU must be >= 1" >&2
  exit 2
fi

SLOTS=()
for dev in "${DEVICES[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do
    SLOTS+=("$dev")
  done
done
if [[ -n "$MAX_PARALLEL_ARG" ]]; then
  MAX_WORKERS="$MAX_PARALLEL_ARG"
  if (( MAX_WORKERS < 1 )); then
    echo "[error] MAX_PARALLEL must be >= 1" >&2
    exit 2
  fi
  if (( MAX_WORKERS > ${#SLOTS[@]} )); then
    MAX_WORKERS="${#SLOTS[@]}"
  fi
else
  MAX_WORKERS="${#SLOTS[@]}"
fi

JOB_QUEUE="$OUT_ROOT/_jobs.tsv"
JOB_CURSOR="$OUT_ROOT/_jobs.cursor"
JOB_LOCK="$OUT_ROOT/_jobs.lock"
JOB_FAILURES="$OUT_ROOT/_job_failures.tsv"
JOB_FAILURE_LOCK="$OUT_ROOT/_job_failures.lock"
MANIFEST_LOCK="$OUT_ROOT/_manifest.lock"
: > "$JOB_QUEUE"
: > "$JOB_FAILURES"
echo 0 > "$JOB_CURSOR"

MANIFEST="$OUT_ROOT/_manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf "tag\tmodel\tdevice\toverrides\tlog_path\toutput_root\tweight_root\n" > "$MANIFEST"
fi

sanitize() {
  echo "$1" | tr ' /:=,|' '______' | tr '-' '_'
}

normalize_key() {
  local k="$1"
  k="${k//-/_}"
  echo "$k"
}

auto_tag() {
  local model="$1"; shift
  if [[ $# -eq 0 ]]; then
    echo "${model}_baseline"
    return
  fi
  local parts=()
  for kv in "$@"; do
    local k=${kv%%=*}
    local v=${kv#*=}
    parts+=("$(sanitize "${k}${v}")")
  done
  printf "%s_%s" "$model" "$(IFS=_; echo "${parts[*]}")"
}

stop_flag_for_shared_hparams() {
  # BUGFIX: terminal_frac is now part of the shared-hparams key. Previously
  # runs differing only in terminal_frac hashed to the SAME stop flag, so an
  # early stop in one configuration silently skipped the others.
  local n_assets="$1" m_states="$2" seed="$3" tau_max="$4" w_min="$5" w_max="$6"
  local gamma="$7" r="$8" x_range_scale="$9" dirc="${10}" alpha_scale="${11}"
  local value_hidden="${12}" value_depth="${13}" batch_size="${14}" lr="${15}"
  local w_terminal="${16}" w_shape="${17}" eval_epochs="${18}" outer_iters="${19}" w_rra="${20}"
  local terminal_frac="${21}"
  local pres_target="${22}" val_points="${23}" val_terminal_points="${24}" val_every="${25}"
  local market_seed="${26}"
  # Model type is part of the key: PINN and PI-PINN monitor DIFFERENT
  # residuals (nonlinear HJB vs frozen linear PDE), so one method's
  # divergence must never censor the other's sample. The flag now only
  # prevents re-running the SAME diverging configuration.
  local model_type="${27}"
  # 28th arg: RESOLVED model-specific settings string (always non-empty).
  local variant="${28:-}"

  local key="model_type=${model_type}|pde_stop_threshold=${PDE_STOP_THRESHOLD}|pde_stop_start_outer=${PDE_STOP_START_OUTER}|pde_stop_patience=${PDE_STOP_PATIENCE}|n_assets=${n_assets}|m_states=${m_states}|seed=${seed}|tau_max=${tau_max}|w_min=${w_min}|w_max=${w_max}|gamma=${gamma}|r=${r}|x_range_scale=${x_range_scale}|dirichlet_concentration=${dirc}|alpha_scale=${alpha_scale}|value_hidden=${value_hidden}|value_depth=${value_depth}|batch_size=${batch_size}|lr=${lr}|w_terminal=${w_terminal}|w_shape=${w_shape}|eval_epochs=${eval_epochs}|outer_iters=${outer_iters}|w_rra=${w_rra}|terminal_frac=${terminal_frac}|pres_target=${pres_target}|val_points=${val_points}|val_terminal_points=${val_terminal_points}|val_every=${val_every}|market_seed=${market_seed}"
  if [[ -n "$variant" ]]; then
    key+="|variant=${variant}"
  fi
  local key_hash
  key_hash="$(printf "%s" "$key" | sha256sum | awk '{print $1}')"
  printf "%s/%s.tsv" "$STOP_FLAG_ROOT" "$key_hash"
}

shell_quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    local q
    printf -v q '%q' "$arg"
    out+="$q "
  done
  printf "%s" "$out"
}

append_manifest() {
  local tag="$1" model="$2" dev="$3" overrides="$4" log="$5" out_dir="$6" weight_dir="$7"
  {
    flock -x 9
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$model" "$dev" "$overrides" "$log" "$out_dir" "$weight_dir" >> "$MANIFEST"
  } 9>"$MANIFEST_LOCK"
}

append_job_failure() {
  local tag="$1" model="$2" mode="$3" log="$4"
  {
    flock -x 9
    printf "%s\t%s\t%s\t%s\n" "$tag" "$model" "$mode" "$log" >> "$JOB_FAILURES"
  } 9>"$JOB_FAILURE_LOCK"
}

remove_run_markers() {
  local out_dir="$1"
  rm -f \
    "$out_dir/_DONE" "$out_dir/_SUCCESS" "$out_dir/_STOPPED_EARLY" "$out_dir/_FAILED" \
    "$out_dir/_DONE_EVAL" "$out_dir/_SUCCESS_EVAL" "$out_dir/_FAILED_EVAL"
  # Keep CSV/status provenance here.  Once Python starts, the recorder
  # archives the previous CSVs as *.old.<timestamp> before writing a fresh
  # run.  If Python fails before recorder construction, the old evidence is
  # still available and the worker's _FAILED marker prevents its reuse.
}

should_skip_run() {
  local tag="$1" model="$2" out_dir="$3" stop_flag_path="$4"

  if [[ "$EVAL_ONLY" == "1" ]]; then
    # Eval-only NEVER touches training provenance: markers, status.json,
    # config.json and metrics backups stay intact (Python handles the
    # metrics re-record with a one-time backup). Skip logic uses the
    # eval-scoped markers only.
    if [[ "$FORCE_RERUN" == "1" ]]; then
      rm -f "$out_dir/_DONE_EVAL" "$out_dir/_SUCCESS_EVAL" "$out_dir/_FAILED_EVAL"
      return 1
    fi
    if [[ -f "$out_dir/_SUCCESS_EVAL" && -f "$out_dir/_FAILED_EVAL" ]]; then
      echo "[warn] $tag has conflicting eval markers; queuing a clean eval rerun"
      return 1
    fi
    if [[ -f "$out_dir/_SUCCESS_EVAL" ]]; then
      echo "[skip] $tag (eval success flag exists: $out_dir/_SUCCESS_EVAL)"
      return 0
    fi
    rm -f "$out_dir/_FAILED_EVAL"
    return 1
  fi

  if [[ "$FORCE_RERUN" == "1" ]]; then
    remove_run_markers "$out_dir"
    [[ -n "$stop_flag_path" ]] && rm -f "$stop_flag_path"
    return 1
  fi

  if [[ -f "$out_dir/_STOPPED_EARLY" && "$RERUN_STOPPED" == "1" ]]; then
    remove_run_markers "$out_dir"
    [[ -n "$stop_flag_path" ]] && rm -f "$stop_flag_path"
    return 1
  fi

  # Mutually exclusive terminal markers indicate an interrupted or stale
  # rerun.  Never let an old _SUCCESS win the skip order; queue a clean
  # attempt and let worker_loop clear the inconsistent marker set.
  local terminal_marker_count=0 terminal_marker
  for terminal_marker in _SUCCESS _STOPPED_EARLY _FAILED; do
    if [[ -f "$out_dir/$terminal_marker" ]]; then
      terminal_marker_count=$((terminal_marker_count + 1))
    fi
  done
  if (( terminal_marker_count > 1 )); then
    echo "[warn] $tag has conflicting terminal markers; queuing a clean rerun"
    return 1
  fi

  if [[ -f "$out_dir/_SUCCESS" ]]; then
    echo "[skip] $tag (success flag exists: $out_dir/_SUCCESS)"
    return 0
  fi
  
  if [[ -f "$out_dir/_STOPPED_EARLY" ]]; then
    echo "[skip] $tag (stopped-early flag exists: $out_dir/_STOPPED_EARLY)"
    return 0
  fi
  if [[ -f "$out_dir/_FAILED" ]]; then
    # A retained figure from an earlier successful attempt must never make a
    # newer failed attempt look complete.  Queue it; worker_loop clears the
    # failure marker immediately before launch.
    echo "[retry] $tag (failed marker exists: $out_dir/_FAILED)"
    return 1
  fi
  if [[ -f "$out_dir/_DONE" ]]; then
    echo "[skip] $tag (legacy done flag exists: $out_dir/_DONE)"
    return 0
  fi

  # Backward-compatible skip rule using representative final figures in output dir.
  # NOTE: intentionally do NOT use weight files for skip, since they are updated during training.
  if [[ "$model" == "pinn" ]]; then
    if find "$out_dir" -type f \( -name 'loss_history_*.png' -o -name 'portfolio_w*.png' \) 2>/dev/null | grep -q .; then
      echo "[skip] $tag (final PINN figures exist in: $out_dir)"
      return 0
    fi
  elif [[ "$model" == "pipinn" ]]; then
    if find "$out_dir" -type f \( -name 'pi_pinn_convergence.png' -o -name 'portfolio_tauX_w*.png' \) 2>/dev/null | grep -q .; then
      echo "[skip] $tag (final PI-PINN figures exist in: $out_dir)"
      return 0
    fi
  fi
  return 1
}

run_job() {
  local tag="$1" model="$2" overrides="$3" out_dir="$4" weight_dir="$5" stop_flag_path="$6"; shift 6
  local log="$LOG_DIR/${tag}.log"
  if [[ "$EVAL_ONLY" == "1" ]]; then
    log="$LOG_DIR/${tag}.eval.log"
  fi
  mkdir -p "$out_dir" "$weight_dir"

  if should_skip_run "$tag" "$model" "$out_dir" "$stop_flag_path"; then
    return
  fi

  local cmd
  cmd="$(shell_quote_cmd "$@")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$tag" "$model" "$overrides" "$out_dir" "$weight_dir" "$stop_flag_path" "$log" "$cmd" >> "$JOB_QUEUE"
  echo "[queue] $tag ($model) overrides=${overrides:-baseline}"
}

next_job_line() {
  local line=""
  {
    flock -x 9
    local idx
    idx="$(cat "$JOB_CURSOR")"
    line="$(sed -n "$((idx + 1))p" "$JOB_QUEUE" || true)"
    if [[ -n "$line" ]]; then
      echo "$((idx + 1))" > "$JOB_CURSOR"
    fi
    printf "%s\n" "$line"
  } 9>"$JOB_LOCK"
}

worker_loop() {
  local worker_id="$1" dev="$2"
  local line
  while true; do
    line="$(next_job_line)"
    [[ -z "$line" ]] && break

    local tag model overrides out_dir weight_dir stop_flag_path log cmd
    IFS=$'\t' read -r tag model overrides out_dir weight_dir stop_flag_path log cmd <<< "$line"
    mkdir -p "$out_dir" "$weight_dir"
    if [[ "$EVAL_ONLY" == "1" ]]; then
      # A queued eval attempt owns only the eval-scoped markers.  Clear all
      # mutually exclusive markers up front so an argparse/import failure
      # cannot leave a stale success marker behind.
      rm -f \
        "$out_dir/_DONE_EVAL" "$out_dir/_SUCCESS_EVAL" "$out_dir/_FAILED_EVAL"
    else
      # Do this immediately before launch as a second line of defence in
      # addition to the recorder's marker rotation.  In particular, a
      # failure before Python constructs the recorder must not inherit an
      # earlier _SUCCESS marker.
      rm -f \
        "$out_dir/_DONE" "$out_dir/_SUCCESS" \
        "$out_dir/_STOPPED_EARLY" "$out_dir/_FAILED"
    fi

    append_manifest "$tag" "$model" "$dev" "$overrides" "$log" "$out_dir" "$weight_dir"

    # A pre-existing stop flag for this configuration means this exact
    # (model, config) diverged before: do not relaunch TRAINING. Evaluation
    # is unrelated to divergence monitoring and always proceeds.
    if [[ "$EVAL_ONLY" != "1" && -n "$stop_flag_path" && -f "$stop_flag_path" ]]; then
      touch "$out_dir/_STOPPED_EARLY" "$out_dir/_DONE"
      cat > "$out_dir/status.json" <<EOF
{
  "status": "stopped_early",
  "reason": "shared_stop_flag_exists_before_launch",
  "run_tag": "$tag",
  "model_type": "$model",
  "stop_flag_path": "$stop_flag_path"
}
EOF
      echo "[skip-stop] $tag (shared PDE stop flag exists: $stop_flag_path)"
      continue
    fi

    echo "[run ] $tag on $dev (worker $worker_id)"

    # Preserve the previous attempt's console provenance.  Training and
    # eval-only logs use separate canonical names, and a rerun rotates rather
    # than truncates the existing file.
    if [[ -f "$log" ]]; then
      local log_stamp
      log_stamp="$(date +%Y%m%d-%H%M%S-%N)-p$$-w${worker_id}"
      mv "$log" "${log}.old.${log_stamp}"
    fi

    local dev_q
    printf -v dev_q '%q' "$dev"
    if PYTHONUNBUFFERED=1 bash -lc "$cmd --device $dev_q" >"$log" 2>&1; then
      if [[ "$EVAL_ONLY" == "1" ]]; then
        touch "$out_dir/_DONE_EVAL"
        # Python normally creates _SUCCESS_EVAL; fallback for safety.
        [[ -f "$out_dir/_SUCCESS_EVAL" ]] || touch "$out_dir/_SUCCESS_EVAL"
        echo "[ok  ] $tag (eval; log: $log)"
      else
        touch "$out_dir/_DONE"
        if [[ -f "$out_dir/_STOPPED_EARLY" ]]; then
          echo "[stop] $tag (early-stopped; log: $log)"
        else
          # The Python script normally creates _SUCCESS; keep this as a fallback.
          touch "$out_dir/_SUCCESS"
          echo "[ok  ] $tag (log: $log)"
        fi
      fi
    else
      if [[ "$EVAL_ONLY" == "1" ]]; then
        touch "$out_dir/_FAILED_EVAL"
        append_job_failure "$tag" "$model" "eval" "$log"
        echo "[fail] $tag (eval; training markers untouched; log: $log)"
      else
        touch "$out_dir/_FAILED"
        append_job_failure "$tag" "$model" "train" "$log"
        echo "[fail] $tag (log: $log)"
      fi
    fi
  done
}

run_all_jobs() {
  local total_jobs
  total_jobs="$(wc -l < "$JOB_QUEUE" | tr -d ' ')"
  if (( total_jobs == 0 )); then
    echo "[done] no queued jobs."
    return
  fi

  echo "[info] queued jobs: $total_jobs"
  echo "[info] devices: ${DEVICES[*]} | JOBS_PER_GPU=$JOBS_PER_GPU | workers=$MAX_WORKERS"

  local pids=()
  for ((i=0; i<MAX_WORKERS; i++)); do
    worker_loop "$((i + 1))" "${SLOTS[$i]}" &
    pids+=("$!")
  done

  local pid worker_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      worker_failed=1
    fi
  done
  if (( worker_failed != 0 )) || [[ -s "$JOB_FAILURES" ]]; then
    local n_failures
    n_failures="$(wc -l < "$JOB_FAILURES" | tr -d ' ')"
    if (( n_failures > 0 )); then
      echo "[error] $n_failures queued job(s) failed; see $JOB_FAILURES" >&2
    else
      echo "[error] a worker process failed before recording its job; inspect $LOG_DIR" >&2
    fi
    return 1
  fi
}

# ==============================
# Baselines (change here once)
# ==============================
declare -A BASE_PINN=(
  [n_assets]=30
  [m_states]=10
  [seed]=12
  [tau_max]=3.0
  [w_min]=0.1
  [w_max]=2.0
  [gamma]=2.0
  [r]=0.03
  [x_range_scale]=1.0
  [dirichlet_concentration]=1.0
  [alpha_scale]=0.25
  [value_hidden]=256
  [value_depth]=3
  # FINAL paper config (Table 3): larger batch, longer inner solves, fewer
  # outer blocks; lr stays 3e-4.
  [batch_size]=10000
  [lr]=3e-4
  [w_terminal]=20.0
  [w_shape]=1.0
  [w_rra]=0.0
  [eval_epochs]=2000
  [outer_iters]=20
  # Paper default (matches the pre-fix effective protocol: the old argparse
  # default 0.5 is what earlier runs actually used).
  [terminal_frac]=0.5
  [print_every]=1000
  # plateau = legacy ReduceLROnPlateau on training loss; fixed = scheduler off
  # (mirrors PI-PINN's lr_schedule=fixed for scheduler-free tests).
  [lr_schedule]=plateau
  [scheduler_patience]=3000
  [scheduler_factor]=0.5
  [scheduler_min_lr]=1e-5
  # Main runs save NO per-iteration weights (final/best only); the E3-a
  # contraction data comes from the per-iteration e_n columns in
  # outer_history.csv. Use e3b_checkpoints=1 (PI-PINN) for the M=1 FD
  # reference runs; it retains every completed outer iterate.
  [save_iterate_every]=0
  # HALF-WIDTH margins: m keeps (1-m) of each axis length. FIRST = PRIMARY:
  # the fixed diagnostic set (e_n, stability margins) and the run-level
  # representative full-dim metric use it, so keep 0.10 first to match the
  # paper convention. The rest are the E9 window-sensitivity list,
  # re-evaluated for free on the same trained network; 0.0 = full-window
  # stress test.
  [eval_margin]="0.10,0.0,0.05,0.15,0.20,0.25,0.30"
  # Held-out validation set + residual-target inner early stopping.
  # pres_target empty = disabled (fixed epoch counts, legacy behavior).
  [pres_target]=""
  [val_points]=50000
  [val_terminal_points]=10000
  [val_every]=25
  # Benchmark market seed: FIXED across the SEEDS sweep so every training
  # seed solves the SAME market problem (paper protocol). The training seed
  # controls only network init / collocation / optimizer randomness.
  [market_seed]=12
  # Full-dimensional Omega_ev test points for Table metrics (0 = off).
  [test_points]=100000
  # Fixed Q_ev diagnostic set for per-iteration e_n / stability margins.
  [diag_points]=8192
  # Diagnostic pass every k outer iterations (E3-a contraction needs 1).
  [diag_every]=1
  # 1 = E8 timing mode (all diagnostics off; wall-clock reflects core work).
  [timing_mode]=0
  # Paper main default: evaluate and generate each run's policy figures.
  # Set skip_figures=1 to retain metrics while suppressing only figures;
  # set skip_eval=1 only when no post-training evaluation is wanted.
  [skip_figures]=0
  [skip_eval]=0
)

declare -A BASE_PIPINN=(
  [n_assets]=30
  [m_states]=10
  [seed]=12
  [tau_max]=3.0
  [w_min]=0.1
  [w_max]=2.0
  [gamma]=2.0
  [r]=0.03
  [x_range_scale]=1.0
  [dirichlet_concentration]=1.0
  [alpha_scale]=0.25
  [value_hidden]=256
  [value_depth]=3
  # FINAL paper config (Table 3): lr stays 3e-4.
  [eval_epochs]=2000
  [outer_iters]=20
  [batch_size]=10000
  [lr]=3e-4
  [w_terminal]=20.0
  [w_shape]=1.0
  [w_rra]=0.0
  # Paper alignment: "nondegenerate initial policy" -> myopic (theta_0 = 0
  # makes the wealth diffusion degenerate at iter 1, violating Assumption 1),
  # and "greedy updates are used unconstrained" -> no clipping. Override
  # theta_init_method=zero theta_clip_abs=3.0 to reproduce older runs.
  [theta_init_method]=myopic
  [theta_init_scale]=1.0
  [theta_clip_abs]="none"
  # Actual risk-premium specification.  The non-affine profile overrides mode
  # and epsilon while retaining every other production training default.
  [risk_premium_mode]=affine
  [nonaffine_eps]=0.0
  [nonaffine_loading_scale]=1.0
  [print_every_outer]=1
  # 0 = no inner prints (default). Set print_every_eval=1 to log every inner
  # (policy-evaluation) epoch, with p_res shown on validation-check epochs.
  [print_every_eval]=0
  # Paper default (matches the pre-fix effective protocol: the old argparse
  # default 0.5 is what earlier runs actually used).
  [terminal_frac]=0.5
  # LR schedules (PI-PINN):
  #   inner_plateau  at every outer: LR resets to [lr] and a FRESH
  #                  ReduceLROnPlateau (patience in INNER epochs, stepped on
  #                  the per-epoch training loss) starts.
  #   carry_plateau  monotone-LR mode. The scheduler steps ONLY on the
  #                  held-out SELECTION residual (one step per sel_every-epoch
  #                  check, so scheduler_patience is in CHECKS). At the end of
  #                  each evaluation the inner-BEST checkpoint (weights +
  #                  optimizer + its LR) is restored, and the next outer
  #                  starts from that best LR, capped by carry_lr_max --
  #                  the LR never increases across outers (outer 1 uses [lr]).
  #                  Use adam_reset=keep. Recommended with carry:
  #                    lr=3e-4 scheduler_factor=0.5 scheduler_patience=3
  #                    scheduler_min_lr=3e-5
  #                  NOTE: with the current BASE (patience 30) a carry run
  #                  would need 30 non-improving CHECKS (= 1500 epochs) per
  #                  decay -- override scheduler_patience for carry runs.
  #   outer_plateau  legacy: ONE persistent scheduler stepped once per outer
  #                  on the end-of-evaluation loss (patience in OUTER iters).
  #   fixed          no scheduler, constant [lr].
  # (PINN side: lr_schedule = plateau | fixed, single training loop.)
  [lr_schedule]=carry_plateau
  [carry_lr_min]=1e-5
  [carry_lr_max]=3e-4
  [adam_reset]=keep
  [scheduler_patience]=20
  [scheduler_factor]=0.5
  [scheduler_min_lr]=1e-8
  # Main runs save NO per-iteration weights (final/best only); the E3-a
  # contraction data comes from the per-iteration e_n columns in
  # outer_history.csv. Use e3b_checkpoints=1 (PI-PINN) for the M=1 FD
  # reference runs; it retains every completed outer iterate.
  [save_iterate_every]=0
  # HALF-WIDTH margins: FIRST = PRIMARY (diagnostic set + representative
  # full-dim metric); keep 0.10 first for the paper convention. Others are
  # the E9 list, re-evaluated for free; 0.0 = full-window stress test.
  [eval_margin]="0.10,0.0,0.05,0.15,0.20,0.25,0.30"
  [pres_target]=""
  [val_points]=50000
  [val_terminal_points]=10000
  [val_every]=25
  [market_seed]=12
  [test_points]=100000
  [diag_points]=8192
  [diag_every]=1
  [timing_mode]=0
  # Paper main default: evaluate and generate each run's policy figures.
  [skip_figures]=0
  [skip_eval]=0
  # Within-evaluation collocation resampling (inner epochs): each policy
  # evaluation redraws a fresh batch every K epochs while the POLICY FUNCTION
  # stays frozen (theta recomputed from a frozen copy of the previous
  # iterate; analytic init at it=1). Avoids overfitting a single batch when
  # eval_epochs is large. 0 = legacy single fixed batch.
  [pe_resample_every]=0
  # Inner held-out BEST selection (within ONE frozen-PDE solve): a small
  # dedicated selection set is checked at inner epoch 0 (the warm start is a
  # candidate, so "do nothing" can win) and every sel_every epochs; the
  # lowest-p_res checkpoint -- MODEL + OPTIMIZER + its LR, post-update
  # aligned -- is RESTORED at the end of the evaluation, before policy
  # improvement and the big audit measurement. sel_patience consecutive
  # non-improving checks end that policy evaluation early.
  # inner_best=0 -> legacy final inner state.
  [inner_best]=1
  [sel_points]=10000
  [sel_terminal_points]=2000
  [sel_every]=50
  [sel_patience]=0
  # 1 = FD-reference checkpoint schedule (every completed outer iterate).
  [e3b_checkpoints]=0
)

# run_pinn_single <tag|auto> key=val ...   (one fully-resolved job)
run_pinn_single() {
  local tag="$1"; shift
  declare -A OVR=()
  local norm_args=()
  local kv k v
  for kv in "$@"; do
    if [[ "$kv" != *=* ]]; then
      echo "[error] invalid override for PINN: $kv (expected key=value)" >&2
      exit 2
    fi
    k="$(normalize_key "${kv%%=*}")"
    v="${kv#*=}"
    if [[ -z "${BASE_PINN[$k]+x}" ]]; then
      echo "[error] unknown PINN override key: ${kv%%=*} (normalized: $k)" >&2
      exit 2
    fi
    OVR[$k]="$v"
    norm_args+=("$k=$v")
  done
  if [[ "$tag" == "baseline" ]]; then tag="pinn_baseline"; fi
  if [[ "$tag" == "auto" ]]; then tag="$(auto_tag pinn "${norm_args[@]}")"; fi

  local n_assets="${OVR[n_assets]:-${BASE_PINN[n_assets]}}"
  local m_states="${OVR[m_states]:-${BASE_PINN[m_states]}}"
  local seed="${OVR[seed]:-${BASE_PINN[seed]}}"
  local tau_max="${OVR[tau_max]:-${BASE_PINN[tau_max]}}"
  local w_min="${OVR[w_min]:-${BASE_PINN[w_min]}}"
  local w_max="${OVR[w_max]:-${BASE_PINN[w_max]}}"
  local gamma="${OVR[gamma]:-${BASE_PINN[gamma]}}"
  local r="${OVR[r]:-${BASE_PINN[r]}}"
  local x_range_scale="${OVR[x_range_scale]:-${BASE_PINN[x_range_scale]}}"
  local dirc="${OVR[dirichlet_concentration]:-${BASE_PINN[dirichlet_concentration]}}"
  local alpha_scale="${OVR[alpha_scale]:-${BASE_PINN[alpha_scale]}}"
  local value_hidden="${OVR[value_hidden]:-${BASE_PINN[value_hidden]}}"
  local value_depth="${OVR[value_depth]:-${BASE_PINN[value_depth]}}"
  local batch_size="${OVR[batch_size]:-${BASE_PINN[batch_size]}}"
  local lr="${OVR[lr]:-${BASE_PINN[lr]}}"
  local w_terminal="${OVR[w_terminal]:-${BASE_PINN[w_terminal]}}"
  local w_shape="${OVR[w_shape]:-${BASE_PINN[w_shape]}}"
  local w_rra="${OVR[w_rra]:-${BASE_PINN[w_rra]}}"
  local eval_epochs="${OVR[eval_epochs]:-${BASE_PINN[eval_epochs]}}"
  local outer_iters="${OVR[outer_iters]:-${BASE_PINN[outer_iters]}}"
  local terminal_frac="${OVR[terminal_frac]:-${BASE_PINN[terminal_frac]}}"
  local print_every="${OVR[print_every]:-${BASE_PINN[print_every]}}"
  local lr_schedule="${OVR[lr_schedule]:-${BASE_PINN[lr_schedule]}}"
  local scheduler_patience="${OVR[scheduler_patience]:-${BASE_PINN[scheduler_patience]}}"
  local scheduler_factor="${OVR[scheduler_factor]:-${BASE_PINN[scheduler_factor]}}"
  local scheduler_min_lr="${OVR[scheduler_min_lr]:-${BASE_PINN[scheduler_min_lr]}}"
  local save_iterate_every="${OVR[save_iterate_every]:-${BASE_PINN[save_iterate_every]}}"
  local eval_margin="${OVR[eval_margin]:-${BASE_PINN[eval_margin]}}"
  local pres_target="${OVR[pres_target]:-${BASE_PINN[pres_target]}}"
  local val_points="${OVR[val_points]:-${BASE_PINN[val_points]}}"
  local val_terminal_points="${OVR[val_terminal_points]:-${BASE_PINN[val_terminal_points]}}"
  local val_every="${OVR[val_every]:-${BASE_PINN[val_every]}}"
  local market_seed="${OVR[market_seed]:-${BASE_PINN[market_seed]}}"
  local test_points="${OVR[test_points]:-${BASE_PINN[test_points]}}"
  local diag_points="${OVR[diag_points]:-${BASE_PINN[diag_points]}}"
  local diag_every="${OVR[diag_every]:-${BASE_PINN[diag_every]}}"
  local timing_mode="${OVR[timing_mode]:-${BASE_PINN[timing_mode]}}"
  local skip_figures="${OVR[skip_figures]:-${BASE_PINN[skip_figures]}}"
  local skip_eval="${OVR[skip_eval]:-${BASE_PINN[skip_eval]}}"
  local timing_flag=()
  [[ "$timing_mode" == "1" ]] && timing_flag=(--timing-mode)
  local skip_figures_flag=()
  [[ "$skip_figures" == "1" ]] && skip_figures_flag=(--skip-figures)
  local skip_eval_flag=()
  [[ "$skip_eval" == "1" ]] && skip_eval_flag=(--skip-eval)

  # Stop-flag key uses RESOLVED model-specific values (not BASE-relative
  # diffs): changing BASE defaults over time in the same OUT_ROOT can never
  # alias two different configurations onto one flag.
  local variant="ls:${lr_schedule};sp:${scheduler_patience};sf:${scheduler_factor};sml:${scheduler_min_lr};"

  local run_output_root="$OUT_ROOT/pinn/$tag"
  local run_weight_root="$OUT_ROOT/weights/pinn/$tag"
  local stop_flag_path
  stop_flag_path="$(stop_flag_for_shared_hparams "$n_assets" "$m_states" "$seed" "$tau_max" "$w_min" "$w_max" "$gamma" "$r" "$x_range_scale" "$dirc" "$alpha_scale" "$value_hidden" "$value_depth" "$batch_size" "$lr" "$w_terminal" "$w_shape" "$eval_epochs" "$outer_iters" "$w_rra" "$terminal_frac" "$pres_target" "$val_points" "$val_terminal_points" "$val_every" "$market_seed" "pinn" "$variant")"
  local overrides_str
  if (( ${#norm_args[@]} == 0 )); then
    overrides_str="baseline"
  else
    overrides_str="$(IFS=','; echo "${norm_args[*]}")"
  fi

  local eval_only_flag=()
  if [[ "$EVAL_ONLY" == "1" ]]; then
    eval_only_flag=(--eval-only)
  fi

  # BUGFIX: --terminal-frac was extracted above but never passed to Python,
  # so every run (including the terminal_frac sweep rows) silently used the
  # Python default 0.5. It is now forwarded, together with the newly exposed
  # print/scheduler/iterate-saving knobs.
  run_job "$tag" "pinn" "$overrides_str" "$run_output_root" "$run_weight_root" "$stop_flag_path" \
    "$PYTHON_BIN" "$SCRIPT_DIR/Liu_nd_pinn.py" \
    --run-tag "$tag" --model-type "pinn" \
    --pde-stop-threshold "$PDE_STOP_THRESHOLD" --pde-stop-start-outer "$PDE_STOP_START_OUTER" --pde-stop-patience "$PDE_STOP_PATIENCE" \
    --n-assets "$n_assets" --m-states "$m_states" --seed "$seed" \
    --tau-max "$tau_max" --w-min "$w_min" --w-max "$w_max" --gamma "$gamma" --r "$r" \
    --x-range-scale "$x_range_scale" --dirichlet-concentration "$dirc" --alpha-scale "$alpha_scale" \
    --value-hidden "$value_hidden" --value-depth "$value_depth" --batch-size "$batch_size" --lr "$lr" \
    --terminal-frac "$terminal_frac" \
    --w-terminal "$w_terminal" --w-shape "$w_shape" --w-rra "$w_rra" --eval-epochs "$eval_epochs" \
    --outer-iters "$outer_iters" --stop-flag-path "$stop_flag_path" \
    --print-every "$print_every" \
    --scheduler-patience "$scheduler_patience" --scheduler-factor "$scheduler_factor" --scheduler-min-lr "$scheduler_min_lr" --lr-schedule "$lr_schedule" \
    --save-iterate-every "$save_iterate_every" \
    --eval-margin "$eval_margin" --pres-target "$pres_target" \
    --val-points "$val_points" --val-terminal-points "$val_terminal_points" --val-every "$val_every" \
    --market-seed "$market_seed" --test-points "$test_points" --diag-points "$diag_points" --diag-every "$diag_every" \
    --output-root "$run_output_root" --weight-root "$run_weight_root" \
    "${timing_flag[@]}" "${skip_figures_flag[@]}" "${skip_eval_flag[@]}" "${eval_only_flag[@]}"
}

# run_pinn <tag|auto> key=val ...
# Seed-loop wrapper: expands over SEED_LIST unless seed=... is given explicitly.
run_pinn() {
  local tag="$1"; shift
  if (( ${#SEED_LIST[@]} == 0 )); then
    run_pinn_single "$tag" "$@"
    return
  fi
  local kv
  for kv in "$@"; do
    if [[ "$kv" == *=* && "$(normalize_key "${kv%%=*}")" == "seed" ]]; then
      run_pinn_single "$tag" "$@"
      return
    fi
  done
  # With multiple seeds a fixed literal tag would collide across seeds;
  # force auto-tagging so the seed lands in the directory name.
  if [[ "$tag" != "auto" ]]; then tag="auto"; fi
  local s
  for s in "${SEED_LIST[@]}"; do
    run_pinn_single "$tag" "$@" seed="$s"
  done
}

# run_pipinn_single <tag|auto> key=val ...   (one fully-resolved job)
run_pipinn_single() {
  local tag="$1"; shift
  declare -A OVR=()
  local norm_args=()
  local kv k v
  for kv in "$@"; do
    if [[ "$kv" != *=* ]]; then
      echo "[error] invalid override for PI-PINN: $kv (expected key=value)" >&2
      exit 2
    fi
    k="$(normalize_key "${kv%%=*}")"
    v="${kv#*=}"
    if [[ -z "${BASE_PIPINN[$k]+x}" ]]; then
      echo "[error] unknown PI-PINN override key: ${kv%%=*} (normalized: $k)" >&2
      exit 2
    fi
    OVR[$k]="$v"
    norm_args+=("$k=$v")
  done
  if [[ "$tag" == "baseline" ]]; then tag="pipinn_baseline"; fi
  if [[ "$tag" == "auto" ]]; then tag="$(auto_tag pipinn "${norm_args[@]}")"; fi

  local n_assets="${OVR[n_assets]:-${BASE_PIPINN[n_assets]}}"
  local m_states="${OVR[m_states]:-${BASE_PIPINN[m_states]}}"
  local seed="${OVR[seed]:-${BASE_PIPINN[seed]}}"
  local tau_max="${OVR[tau_max]:-${BASE_PIPINN[tau_max]}}"
  local w_min="${OVR[w_min]:-${BASE_PIPINN[w_min]}}"
  local w_max="${OVR[w_max]:-${BASE_PIPINN[w_max]}}"
  local gamma="${OVR[gamma]:-${BASE_PIPINN[gamma]}}"
  local r="${OVR[r]:-${BASE_PIPINN[r]}}"
  local x_range_scale="${OVR[x_range_scale]:-${BASE_PIPINN[x_range_scale]}}"
  local dirc="${OVR[dirichlet_concentration]:-${BASE_PIPINN[dirichlet_concentration]}}"
  local alpha_scale="${OVR[alpha_scale]:-${BASE_PIPINN[alpha_scale]}}"
  local value_hidden="${OVR[value_hidden]:-${BASE_PIPINN[value_hidden]}}"
  local value_depth="${OVR[value_depth]:-${BASE_PIPINN[value_depth]}}"
  local outer_iters="${OVR[outer_iters]:-${BASE_PIPINN[outer_iters]}}"
  local eval_epochs="${OVR[eval_epochs]:-${BASE_PIPINN[eval_epochs]}}"
  local batch_size="${OVR[batch_size]:-${BASE_PIPINN[batch_size]}}"
  local lr="${OVR[lr]:-${BASE_PIPINN[lr]}}"
  local w_terminal="${OVR[w_terminal]:-${BASE_PIPINN[w_terminal]}}"
  local w_shape="${OVR[w_shape]:-${BASE_PIPINN[w_shape]}}"
  local w_rra="${OVR[w_rra]:-${BASE_PIPINN[w_rra]}}"
  local theta_init_method="${OVR[theta_init_method]:-${BASE_PIPINN[theta_init_method]}}"
  local theta_init_scale="${OVR[theta_init_scale]:-${BASE_PIPINN[theta_init_scale]}}"
  local theta_clip_abs="${OVR[theta_clip_abs]:-${BASE_PIPINN[theta_clip_abs]}}"
  local risk_premium_mode="${OVR[risk_premium_mode]:-${BASE_PIPINN[risk_premium_mode]}}"
  local nonaffine_eps="${OVR[nonaffine_eps]:-${BASE_PIPINN[nonaffine_eps]}}"
  local nonaffine_loading_scale="${OVR[nonaffine_loading_scale]:-${BASE_PIPINN[nonaffine_loading_scale]}}"
  local print_every_outer="${OVR[print_every_outer]:-${BASE_PIPINN[print_every_outer]}}"
  local print_every_eval="${OVR[print_every_eval]:-${BASE_PIPINN[print_every_eval]}}"
  local terminal_frac="${OVR[terminal_frac]:-${BASE_PIPINN[terminal_frac]}}"
  local scheduler_patience="${OVR[scheduler_patience]:-${BASE_PIPINN[scheduler_patience]}}"
  local scheduler_factor="${OVR[scheduler_factor]:-${BASE_PIPINN[scheduler_factor]}}"
  local scheduler_min_lr="${OVR[scheduler_min_lr]:-${BASE_PIPINN[scheduler_min_lr]}}"
  local save_iterate_every="${OVR[save_iterate_every]:-${BASE_PIPINN[save_iterate_every]}}"
  local lr_schedule="${OVR[lr_schedule]:-${BASE_PIPINN[lr_schedule]}}"
  local adam_reset="${OVR[adam_reset]:-${BASE_PIPINN[adam_reset]}}"
  local eval_margin="${OVR[eval_margin]:-${BASE_PIPINN[eval_margin]}}"
  local pres_target="${OVR[pres_target]:-${BASE_PIPINN[pres_target]}}"
  local val_points="${OVR[val_points]:-${BASE_PIPINN[val_points]}}"
  local val_terminal_points="${OVR[val_terminal_points]:-${BASE_PIPINN[val_terminal_points]}}"
  local val_every="${OVR[val_every]:-${BASE_PIPINN[val_every]}}"
  local market_seed="${OVR[market_seed]:-${BASE_PIPINN[market_seed]}}"
  local test_points="${OVR[test_points]:-${BASE_PIPINN[test_points]}}"
  local diag_points="${OVR[diag_points]:-${BASE_PIPINN[diag_points]}}"
  local diag_every="${OVR[diag_every]:-${BASE_PIPINN[diag_every]}}"
  local timing_mode="${OVR[timing_mode]:-${BASE_PIPINN[timing_mode]}}"
  local skip_figures="${OVR[skip_figures]:-${BASE_PIPINN[skip_figures]}}"
  local skip_eval="${OVR[skip_eval]:-${BASE_PIPINN[skip_eval]}}"
  local e3b_checkpoints="${OVR[e3b_checkpoints]:-${BASE_PIPINN[e3b_checkpoints]}}"
  local pe_resample_every="${OVR[pe_resample_every]:-${BASE_PIPINN[pe_resample_every]}}"
  local inner_best="${OVR[inner_best]:-${BASE_PIPINN[inner_best]}}"
  local sel_points="${OVR[sel_points]:-${BASE_PIPINN[sel_points]}}"
  local sel_terminal_points="${OVR[sel_terminal_points]:-${BASE_PIPINN[sel_terminal_points]}}"
  local sel_every="${OVR[sel_every]:-${BASE_PIPINN[sel_every]}}"
  local sel_patience="${OVR[sel_patience]:-${BASE_PIPINN[sel_patience]}}"
  local carry_lr_min="${OVR[carry_lr_min]:-${BASE_PIPINN[carry_lr_min]}}"
  local carry_lr_max="${OVR[carry_lr_max]:-${BASE_PIPINN[carry_lr_max]}}"
  local timing_flag=()
  [[ "$timing_mode" == "1" ]] && timing_flag=(--timing-mode)
  local skip_figures_flag=()
  [[ "$skip_figures" == "1" ]] && skip_figures_flag=(--skip-figures)
  local skip_eval_flag=()
  [[ "$skip_eval" == "1" ]] && skip_eval_flag=(--skip-eval)
  local e3b_flag=()
  [[ "$e3b_checkpoints" == "1" ]] && e3b_flag=(--e3b-checkpoints)

  # Stop-flag key uses RESOLVED model-specific values (not BASE-relative
  # diffs): changing BASE defaults over time in the same OUT_ROOT can never
  # alias two different configurations onto one flag.
  local variant="ls:${lr_schedule};ar:${adam_reset};sp:${scheduler_patience};sf:${scheduler_factor};sml:${scheduler_min_lr};ti:${theta_init_method};tis:${theta_init_scale};tc:${theta_clip_abs};rpm:${risk_premium_mode};eps:${nonaffine_eps};nls:${nonaffine_loading_scale};pr:${pe_resample_every};ib:${inner_best};sel:${sel_points}/${sel_terminal_points}/${sel_every}/${sel_patience};cl:${carry_lr_min}/${carry_lr_max};"

  local run_output_root="$OUT_ROOT/pi-pinn/$tag"
  local run_weight_root="$OUT_ROOT/weights/pi-pinn/$tag"
  local stop_flag_path
  stop_flag_path="$(stop_flag_for_shared_hparams "$n_assets" "$m_states" "$seed" "$tau_max" "$w_min" "$w_max" "$gamma" "$r" "$x_range_scale" "$dirc" "$alpha_scale" "$value_hidden" "$value_depth" "$batch_size" "$lr" "$w_terminal" "$w_shape" "$eval_epochs" "$outer_iters" "$w_rra" "$terminal_frac" "$pres_target" "$val_points" "$val_terminal_points" "$val_every" "$market_seed" "pipinn" "$variant")"
  local overrides_str
  if (( ${#norm_args[@]} == 0 )); then
    overrides_str="baseline"
  else
    overrides_str="$(IFS=','; echo "${norm_args[*]}")"
  fi

  local eval_only_flag=()
  if [[ "$EVAL_ONLY" == "1" ]]; then
    eval_only_flag=(--eval-only)
  fi

  # BUGFIX: --terminal-frac was accepted as an override key (BASE_PIPINN has
  # it) but never extracted nor passed to Python, so every run silently used
  # the Python default 0.5. It is now forwarded, together with the newly
  # exposed scheduler/iterate-saving knobs.
  run_job "$tag" "pipinn" "$overrides_str" "$run_output_root" "$run_weight_root" "$stop_flag_path" \
    "$PYTHON_BIN" "$SCRIPT_DIR/Liu_nd_pi_pinn.py" \
    --run-tag "$tag" --model-type "pipinn" \
    --pde-stop-threshold "$PDE_STOP_THRESHOLD" --pde-stop-start-outer "$PDE_STOP_START_OUTER" --pde-stop-patience "$PDE_STOP_PATIENCE" \
    --n-assets "$n_assets" --m-states "$m_states" --seed "$seed" \
    --tau-max "$tau_max" --w-min "$w_min" --w-max "$w_max" --gamma "$gamma" --r "$r" \
    --x-range-scale "$x_range_scale" --dirichlet-concentration "$dirc" --alpha-scale "$alpha_scale" \
    --value-hidden "$value_hidden" --value-depth "$value_depth" --outer-iters "$outer_iters" \
    --eval-epochs "$eval_epochs" --batch-size "$batch_size" --lr "$lr" \
    --terminal-frac "$terminal_frac" \
    --w-terminal "$w_terminal" --w-shape "$w_shape" --w-rra "$w_rra" \
    --theta-init-method "$theta_init_method" --theta-init-scale "$theta_init_scale" \
    --theta-clip-abs "$theta_clip_abs" --print-every-outer "$print_every_outer" \
    --risk-premium-mode "$risk_premium_mode" --nonaffine-eps "$nonaffine_eps" \
    --nonaffine-loading-scale "$nonaffine_loading_scale" \
    --print-every-eval "$print_every_eval" --stop-flag-path "$stop_flag_path" \
    --scheduler-patience "$scheduler_patience" --scheduler-factor "$scheduler_factor" --scheduler-min-lr "$scheduler_min_lr" \
    --save-iterate-every "$save_iterate_every" \
    --lr-schedule "$lr_schedule" --adam-reset "$adam_reset" \
    --eval-margin "$eval_margin" --pres-target "$pres_target" \
    --val-points "$val_points" --val-terminal-points "$val_terminal_points" --val-every "$val_every" \
    --market-seed "$market_seed" --test-points "$test_points" --diag-points "$diag_points" --diag-every "$diag_every" \
    --pe-resample-every "$pe_resample_every" \
    --inner-best-restore "$inner_best" --sel-points "$sel_points" --sel-terminal-points "$sel_terminal_points" \
    --sel-every "$sel_every" --sel-patience "$sel_patience" \
    --carry-lr-min "$carry_lr_min" --carry-lr-max "$carry_lr_max" \
    --output-root "$run_output_root" --weight-root "$run_weight_root" \
    "${timing_flag[@]}" "${skip_figures_flag[@]}" "${skip_eval_flag[@]}" "${e3b_flag[@]}" "${eval_only_flag[@]}"
}

# run_pipinn <tag|auto> key=val ...
# Seed-loop wrapper: expands over SEED_LIST unless seed=... is given explicitly.
run_pipinn() {
  local tag="$1"; shift
  if (( ${#SEED_LIST[@]} == 0 )); then
    run_pipinn_single "$tag" "$@"
    return
  fi
  local kv
  for kv in "$@"; do
    if [[ "$kv" == *=* && "$(normalize_key "${kv%%=*}")" == "seed" ]]; then
      run_pipinn_single "$tag" "$@"
      return
    fi
  done
  # With multiple seeds a fixed literal tag would collide across seeds;
  # force auto-tagging so the seed lands in the directory name.
  if [[ "$tag" != "auto" ]]; then tag="auto"; fi
  local s
  for s in "${SEED_LIST[@]}"; do
    run_pipinn_single "$tag" "$@" seed="$s"
  done
}

# =============================================================================
# Sweep definition
#   Use either batch_size=1000 or batch-size=1000; keys are normalized internally.
#   Unknown keys now fail fast instead of being silently ignored.
#
#   PAPER MAIN (Table 3 / E1-a): N=30, M in {1,3,5}, both methods, BASE
#   hyperparameters. Run with SEEDS="1,2,3,4,5,6,7,8,9,10" (10 seeds);
#   the market is pinned by market_seed=12 regardless of the training seed.
# =============================================================================

if [[ "$SWEEP_PROFILE" == "main" ]]; then
  run_pipinn auto m_states=5
  run_pinn   auto m_states=5

  run_pipinn auto m_states=3
  run_pinn   auto m_states=3

  run_pipinn auto m_states=1 e3b_checkpoints=1
  run_pinn   auto m_states=1
elif [[ "$SWEEP_PROFILE" == "nonaffine" ]]; then
  # Figure 4 profile: N=30 is fixed; choose one or more M values from 1,3,5.
  # Every epsilon uses the SAME market_seed and training-seed set.  eps=0 is
  # mandatory because postprocess_nonaffine.py forms paired within-seed
  # deformations relative to that baseline.
  _m_text="${M_STATES_LIST:-3}"
  _eps_text="${EPS_LIST:-0,0.1,1,2,3,4,5}"
  _loading_scale="${NONAFFINE_LOADING_SCALE:-1.0}"
  _skip_figures="${NONAFFINE_SKIP_FIGURES:-0}"
  IFS=', ' read -ra _m_values <<< "$_m_text"
  IFS=', ' read -ra _eps_values <<< "$_eps_text"
  if (( ${#_m_values[@]} == 0 || ${#_eps_values[@]} == 0 )); then
    echo "[error] nonaffine profile requires non-empty M_STATES_LIST and EPS_LIST" >&2
    exit 2
  fi
  _has_eps_zero=0
  declare -A _nonaff_eps_seen=()
  for _eps in "${_eps_values[@]}"; do
    if [[ ! "$_eps" =~ ^[+]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?$ ]]; then
      echo "[error] invalid non-negative epsilon: $_eps" >&2
      exit 2
    fi
    _eps_key="$(awk -v value="$_eps" 'BEGIN { printf "%.17g", value + 0.0 }')"
    if [[ -n "${_nonaff_eps_seen[$_eps_key]+x}" ]]; then
      echo "[error] duplicate epsilon value: $_eps" >&2
      exit 2
    fi
    _nonaff_eps_seen[$_eps_key]=1
    if [[ "$_eps" =~ ^[+]?0*\.?0+([eE][+-]?[0-9]+)?$ ]]; then
      _has_eps_zero=1
    fi
  done
  if (( _has_eps_zero == 0 )); then
    echo "[error] EPS_LIST must include 0 for the paired affine baseline" >&2
    exit 2
  fi
  declare -A _nonaff_m_seen=()
  for _m_state in "${_m_values[@]}"; do
    case "$_m_state" in
      1|3|5) ;;
      *)
        echo "[error] nonaffine M_STATES_LIST supports only 1, 3, or 5; got $_m_state" >&2
        exit 2
        ;;
    esac
    if [[ -n "${_nonaff_m_seen[$_m_state]+x}" ]]; then
      echo "[error] duplicate M_STATES_LIST value: $_m_state" >&2
      exit 2
    fi
    _nonaff_m_seen[$_m_state]=1
  done
  for _m_state in "${_m_values[@]}"; do
    for _eps in "${_eps_values[@]}"; do
      run_pipinn auto \
        n_assets=30 m_states="$_m_state" \
        risk_premium_mode=tanh nonaffine_eps="$_eps" \
        nonaffine_loading_scale="$_loading_scale" \
        theta_init_scale=1.0 e3b_checkpoints=0 \
        skip_figures="$_skip_figures" skip_eval=0
    done
  done
else
  echo "[error] unknown SWEEP_PROFILE=$SWEEP_PROFILE (expected main or nonaffine)" >&2
  exit 2
fi


# --- E6 residual-tolerance sweep (PI-PINN only; enable when running E6) ----
# NOTE: tight tolerances may need a larger inner cap; adjust eval_epochs.
# for PT in 1e-2 1e-3 1e-4 1e-5; do
#   run_pipinn auto m_states=3 pres_target=$PT eval_epochs=1000
# done

# --- Liu M=1 FD frozen-policy reference (all outer checkpoints ON) ----------
# run_pipinn auto m_states=1 e3b_checkpoints=1

# --- E8 timing runs (all diagnostics off) ----------------------------------
# run_pipinn auto m_states=3 timing_mode=1
# run_pinn   auto m_states=3 timing_mode=1

# --- smoke template --------------------------------------------------------
# run_pipinn auto m_states=1 batch_size=500 outer_iters=5 eval_epochs=20 \
#   val_points=2000 val_terminal_points=500 diag_points=512 test_points=2000
# run_pinn   auto m_states=1 batch_size=500 outer_iters=5 eval_epochs=20 \
#   val_points=2000 val_terminal_points=500 diag_points=512 test_points=2000

run_all_jobs

echo ""
echo "[done] all jobs finished. logs: $LOG_DIR"
echo "[done] manifest: $MANIFEST"

# Aggregate per-seed metrics into mean / std / 95% CI tables.
# Disable with AGGREGATE=0. Can also be run standalone at any time:
#   python3 aggregate_seeds.py --out-root <OUT_ROOT>
if [[ "$SWEEP_PROFILE" == "main" && "${AGGREGATE:-1}" == "1" ]]; then
  echo ""
  echo "[aggregate] computing seed statistics (mean / std / 95% CI) ..."
  aggregate_args=(--out-root "$OUT_ROOT")
  if (( ${#SEED_LIST[@]} > 0 )); then
    aggregate_args+=(--expected-seeds "$SEEDS" --min-runs "${#SEED_LIST[@]}")
    aggregate_args+=(--expected-n-assets 30 --expected-m-states "1,3,5" --expected-models "pinn,pipinn")
  fi
  if "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_seeds.py" "${aggregate_args[@]}"; then
    echo "[aggregate] summary written under: $OUT_ROOT/seed_summary"
  else
    if [[ "$STRICT_PAPER_AGGREGATION" == "1" && ${#SEED_LIST[@]} -gt 0 ]]; then
      echo "[error] paper aggregation validation failed: $OUT_ROOT" >&2
      exit 1
    fi
    echo "[warn] aggregation failed; run manually: $PYTHON_BIN $SCRIPT_DIR/aggregate_seeds.py --out-root $OUT_ROOT"
  fi
fi
