#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash Liu/tune_pipinn.sh [OUT_ROOT] [MAX_PARALLEL]
#   DEVICE_LIST="cuda:0,cuda:1,cuda:2,cuda:3" bash Liu/tune_pipinn.sh /workspace/pinn_pi_finite_horizon/outputs/my_run 4
OUT_ROOT="${1:-$(pwd)/outputs/tune_liu_$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL="${2:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUT_ROOT"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Device can be fixed or round-robin from DEVICE_LIST.
# examples:
#   DEVICE=cuda:0 bash Liu/tune_pipinn.sh
#   DEVICE_LIST="cuda:0,cuda:1" bash Liu/tune_pipinn.sh
DEVICE="${DEVICE:-cuda}"
DEVICE_LIST="${DEVICE_LIST:-}"
if [[ -n "$DEVICE_LIST" ]]; then
  IFS=',' read -ra DEVICES <<< "$DEVICE_LIST"
else
  DEVICES=("$DEVICE")
fi
GPU_CURSOR=0


# manifest 헤더 (한 번만)
MANIFEST="$OUT_ROOT/_manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf "tag\tmodel\tdevice\toverrides\tlog_path\n" > "$MANIFEST"
fi

sanitize() { echo "$1" | tr ' /:=,' '_____'; }
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

pick_device() {
  PICKED_DEVICE="${DEVICES[$((GPU_CURSOR % ${#DEVICES[@]}))]}"
  GPU_CURSOR=$((GPU_CURSOR + 1))
}

run_job() {
  local tag="$1" model="$2" overrides="$3" out_dir="$4" weight_dir="$5"; shift 5
  local log="$LOG_DIR/${tag}.log"
  local done_flag="$out_dir/_DONE"
  if [[ -f "$done_flag" ]]; then
    echo "[skip] $tag (done flag exists: $done_flag)"
    return
  fi

  # Backward-compatible skip rule using representative final figures in output dir.
  # NOTE: intentionally do NOT use weight files for skip, since they are updated during training.
  if [[ "$model" == "pinn" ]]; then
    if find "$out_dir" -type f \( -name 'loss_history_*.png' -o -name 'portfolio_w*.png' \) 2>/dev/null | grep -q .; then
      echo "[skip] $tag (final PINN figures exist in: $out_dir)"
      return
    fi
  elif [[ "$model" == "pipinn" ]]; then
    if find "$out_dir" -type f \( -name 'pi_pinn_convergence.png' -o -name 'portfolio_tauX_w*.png' \) 2>/dev/null | grep -q .; then
      echo "[skip] $tag (final PI-PINN figures exist in: $out_dir)"
      return
    fi
  fi

  local dev
  pick_device
  dev="$PICKED_DEVICE"
  echo "[run ] $tag on $dev"
  {
    printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$model" "$dev" "$overrides" "$log" >> "$MANIFEST"
  }

  (
  # Force unbuffered stdout/stderr so training prints are written to log in real time.
    if PYTHONUNBUFFERED=1 "$@" --device "$dev" >"$log" 2>&1; then
      touch "$done_flag"
      echo "[ok  ] $tag"
    else
      echo "[fail] $tag"
  fi
  ) &
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
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
  [gamma]=3.0
  [r]=0.03
  [x_range_scale]=1.0
  [dirichlet_concentration]=1.0
  [alpha_scale]=0.25
  [value_hidden]=256
  [value_depth]=3
  [batch_size]=3000
  [lr]=5e-4
  [w_terminal]=20.0
  [w_shape]=1.0
  [eval_epochs]=200
  [outer_iters]=500
)


declare -A BASE_PIPINN=(
  [n_assets]=30
  [m_states]=10
  [seed]=12
  [tau_max]=3.0
  [w_min]=0.1
  [w_max]=2.0
  [gamma]=3.0
  [r]=0.03
  [x_range_scale]=1.0
  [dirichlet_concentration]=1.0
  [alpha_scale]=0.25
  [value_hidden]=256
  [value_depth]=3
  [eval_epochs]=200
  [outer_iters]=500
  [batch_size]=3000
  [lr]=5e-4
  [w_terminal]=20.0
  [w_shape]=1.0
  [theta_init_method]=zero
  [theta_clip_abs]=3.0
  [print_every_outer]=20
  [print_every_eval]=200
)

# run_pinn <tag|auto> key=val ...
run_pinn() {
  local tag="$1"; shift
  declare -A OVR=()
  for kv in "$@"; do OVR[${kv%%=*}]=${kv#*=}; done
  if [[ "$tag" == "baseline" ]]; then tag="pinn_baseline"; fi
  if [[ "$tag" == "auto" ]]; then tag="$(auto_tag pinn "$@")"; fi

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
  local eval_epochs="${OVR[eval_epochs]:-${BASE_PINN[eval_epochs]}}"
  local outer_iters="${OVR[outer_iters]:-${BASE_PINN[outer_iters]}}"

  local run_output_root="$OUT_ROOT/pinn/$tag"
  local run_weight_root="$OUT_ROOT/weights/pinn/$tag"

  run_job "$tag" "pinn" "$*" "$run_output_root" "$run_weight_root" "$PYTHON_BIN" Liu_nd_pinn.py \
    --n-assets "$n_assets" --m-states "$m_states" --seed "$seed" \
    --tau-max "$tau_max" --w-min "$w_min" --w-max "$w_max" --gamma "$gamma" --r "$r" \
    --x-range-scale "$x_range_scale" --dirichlet-concentration "$dirc" --alpha-scale "$alpha_scale" \
    --value-hidden "$value_hidden" --value-depth "$value_depth" --batch-size "$batch_size" --lr "$lr" \
    --w-terminal "$w_terminal" --w-shape "$w_shape" --eval-epochs "$eval_epochs" \
    --outer-iters "$outer_iters" --stop-flag-path "$STOP_FLAG" \
    --output-root "$run_output_root" --weight-root "$run_weight_root"
}

# =============================================================================
# 여기부터 실제 sweep 정의
# 형식: run <tag> <field1>=<value1> <field2>=<value2> ...
# 명시하지 않은 필드는 위 BASE 값이 자동으로 쓰임
# =============================================================================

# run_pipinn <tag|auto> key=val ...
run_pipinn() {
  local tag="$1"; shift
  declare -A OVR=()
  for kv in "$@"; do OVR[${kv%%=*}]=${kv#*=}; done
  if [[ "$tag" == "baseline" ]]; then tag="pipinn_baseline"; fi
  if [[ "$tag" == "auto" ]]; then tag="$(auto_tag pipinn "$@")"; fi

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
  local theta_init_method="${OVR[theta_init_method]:-${BASE_PIPINN[theta_init_method]}}"
  local theta_clip_abs="${OVR[theta_clip_abs]:-${BASE_PIPINN[theta_clip_abs]}}"
  local print_every_outer="${OVR[print_every_outer]:-${BASE_PIPINN[print_every_outer]}}"
  local print_every_eval="${OVR[print_every_eval]:-${BASE_PIPINN[print_every_eval]}}"

  local run_output_root="$OUT_ROOT/pi-pinn/$tag"
  local run_weight_root="$OUT_ROOT/weights/pi-pinn/$tag"

  run_job "$tag" "pipinn" "$*" "$run_output_root" "$run_weight_root" "$PYTHON_BIN" Liu_nd_pi_pinn.py \
    --n-assets "$n_assets" --m-states "$m_states" --seed "$seed" \
    --tau-max "$tau_max" --w-min "$w_min" --w-max "$w_max" --gamma "$gamma" --r "$r" \
    --x-range-scale "$x_range_scale" --dirichlet-concentration "$dirc" --alpha-scale "$alpha_scale" \
    --value-hidden "$value_hidden" --value-depth "$value_depth" --outer-iters "$outer_iters" \
    --eval-epochs "$eval_epochs" --batch-size "$batch_size" --lr "$lr" \
    --w-terminal "$w_terminal" --w-shape "$w_shape" --theta-init-method "$theta_init_method" \
    --theta-clip-abs "$theta_clip_abs" --print-every-outer "$print_every_outer" \
    --print-every-eval "$print_every_eval" --stop-flag-path "$STOP_FLAG" \
    --output-root "$run_output_root" --weight-root "$run_weight_root"
}


# ==============================
# Example plans (edit freely)
# ==============================
# baseline
run_pinn baseline
run_pipinn baseline

# (A) outer × eval_epochs
run_pipinn auto batch-size=1000
run_pinn auto batch-size=1000

run_pipinn auto batch-size=2000
run_pinn auto batch-size=2000

run_pipinn auto batch-size=4000
run_pinn auto batch-size=4000

run_pipinn auto batch-size=5000
run_pinn auto batch-size=5000

# PINN example: only tweak what you need from baseline
# run_pinn auto eval_epochs=100 outer_iters=500
# run_pinn auto lr=1e-4 batch_size=3000

wait
echo "[done] all jobs finished. logs: $OUT_ROOT"

wait   # ← 이 줄 추가: 백그라운드 job이 전부 끝날 때까지 대기
echo ""
echo "[done] manifest: $MANIFEST"
# echo "[done] 집계: python3 collect_results.py $TUNE_ROOT"

# python3 "$(dirname "$0")/collect_results.py" "$TUNE_ROOT"