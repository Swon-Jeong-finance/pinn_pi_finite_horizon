#!/usr/bin/env bash
# PI-PINN 하이퍼파라미터 sweep
# 사용법: bash tune_pipinn.sh <BASE_RESOLVED_CONFIG.yaml> [TUNE_ROOT] [SAMPLE_MODE]
#   SAMPLE_MODE: oos(기본) | insample

set -euo pipefail

BASE_CFG="${1:-experiments/cov_test/ff25_pls_tau3_fixed/rank_001/outputs/ff49_stage17_rank_sweep_cv2000_curve_core_pls_fixed_pls_ret_ff5_curve_macro7_H1_k2_rolling240m_annual_const_v2_apt_pipinn_rolling240m_annual/resolved_config.yaml}"
TUNE_ROOT="${2:-$(pwd)/pinn_tune/insample/$(date +%Y%m%d)}"
# TUNE_ROOT="${2:-$(pwd)/pinn_tune/insample}"
RUN_SAMPLE="${3:-${TUNE_SAMPLE:-oos}}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"   # ← 이 줄 추가 (환경변수로 제어 가능)
GPUS="${GPUS:-cuda:0,cuda:1,cuda:2,cuda:3}"     # ← 추가: 쉼표로 구분된 GPU 목록
mkdir -p "$TUNE_ROOT"

if [[ "$RUN_SAMPLE" != "oos" && "$RUN_SAMPLE" != "insample" ]]; then
  echo "[error] SAMPLE_MODE must be one of: oos, insample (got: $RUN_SAMPLE)"
  exit 1
fi

# GPU 리스트를 배열로
IFS=',' read -ra GPU_LIST <<< "$GPUS"
GPU_COUNT=${#GPU_LIST[@]}
GPU_CURSOR=0                       # round-robin 카운터

echo "[tune] base config : $BASE_CFG"
echo "[tune] output root : $TUNE_ROOT"
echo "[tune] sample mode : $RUN_SAMPLE"

MANIFEST="$TUNE_ROOT/_manifest.tsv"

# PI-PINN이 받는 모든 하이퍼파라미터 20개를 positional로 받는다.
# 순서를 고정하기 위해 상수로 정의.
FIELDS=(outer_iters eval_epochs n_train_int n_train_bc n_val_int n_val_bc \
        p_uniform p_emp p_tau_head p_tau_near0 tau_head_window \
        lr grad_clip w_bc w_bc_dx \
        scheduler_factor scheduler_patience min_lr \
        width depth)



# manifest 헤더 (한 번만)
if [[ ! -f "$MANIFEST" ]]; then
  { printf "tag"; for f in "${FIELDS[@]}"; do printf "\t%s" "$f"; done; printf "\tdevice\toutput_dir\n"; } > "$MANIFEST"
fi

run_variant () {
  local tag=$1; shift
  if [[ $# -ne ${#FIELDS[@]} ]]; then
    echo "[error] $tag: expected ${#FIELDS[@]} args, got $#"
    return 1
  fi

  local vals=("$@")
  local out="$TUNE_ROOT/$tag"
  mkdir -p "$out"

  # 이미 완료됐으면 skip
  if [[ -f "$out/outputs/comparison_cross_modes_all_costs_summary.csv" ]] || \
     ls "$out/outputs"/*/comparison_cross_modes_all_costs_summary.csv > /dev/null 2>&1; then
    echo "[skip] $tag"
    return 0
  fi

  # 재시도 시 과거 실패 플래그 제거
  rm -f "$out/_FAILED"

  # Round-robin GPU 할당
  local device="${GPU_LIST[$((GPU_CURSOR % GPU_COUNT))]}"
  GPU_CURSOR=$((GPU_CURSOR + 1))

  # python으로 base YAML 복사 + PI-PINN 필드 override + snapshot 기록
  local cfg_path="$out/config.yaml"
  python3 - "$BASE_CFG" "$cfg_path" "$out" "$tag" "$device" "$RUN_SAMPLE" "${vals[@]}" <<'PY'
import sys, json, pathlib, datetime, yaml

base_path, cfg_path, out_dir, tag, device, sample_mode, *vals = sys.argv[1:]  # ← device/sample_mode 받기
fields = ['outer_iters','eval_epochs','n_train_int','n_train_bc','n_val_int','n_val_bc',
          'p_uniform','p_emp','p_tau_head','p_tau_near0','tau_head_window',
          'lr','grad_clip','w_bc','w_bc_dx',
          'scheduler_factor','scheduler_patience','min_lr',
          'width','depth']
int_fields = {'outer_iters','eval_epochs','n_train_int','n_train_bc','n_val_int','n_val_bc',
              'tau_head_window','scheduler_patience','width','depth'}

cfg = yaml.safe_load(pathlib.Path(base_path).read_text())
cfg['project']['output_dir'] = f"{out_dir}/outputs"
cfg['project']['name']       = f"tune_{tag}"

split = cfg.setdefault('split', {})
if str(sample_mode).strip().lower() == 'insample':
    # In-sample 평가: rolling 240m 구조 그대로 유지, 단 평가 기간을 OOS 이전으로 옮김.
    # train pool = [train_start, test_start_original) (즉 OOS를 침범하지 않음)
    import datetime
    
    original_test_start = split.get('test_start')
    if original_test_start is None:
        raise ValueError("split.test_start is required for insample mode")
    
    # 문자열/date 호환
    def _to_date(x):
        if isinstance(x, str):
            return datetime.date.fromisoformat(x[:10])
        return x
    
    train_start_d = _to_date(split['train_start'])
    oos_start_d = _to_date(original_test_start)
    
    # In-sample 평가의 시작: train_start + rolling_train_months 만큼 뒤
    # rolling240m에서는 첫 refit 시점에 학습 데이터 240개월이 있어야 함
    rolling_months = int(split.get('rolling_train_months') or 240)
    # 첫 평가 시작 = train_start + rolling_months (= 최초 full rolling window 확보 시점)
    # 1964-01 + 240m = 1984-01
    years = rolling_months // 12
    months = rolling_months % 12
    new_month = train_start_d.month + months
    new_year = train_start_d.year + years
    if new_month > 12:
        new_year += 1
        new_month -= 12
    insample_test_start = datetime.date(new_year, new_month, train_start_d.day)
    
    # 평가 종료 = OOS 시작 (OOS를 절대 침범하지 않음)
    split['test_start'] = insample_test_start.isoformat()
    split['end_date'] = oos_start_d.isoformat()
    # train_window_mode, rolling_train_months, refit_every, rebalance_every는
    # 원래 config(rolling240m_annual)의 값을 그대로 유지
    
    base_label = split.get('protocol_label') or 'rolling240m_annual'
    split['protocol_label'] = f"{base_label}_insample"

p = cfg.setdefault('pipinn', {})
p['auto_output_subdir'] = False
p['device'] = device             # ← GPU 지정

overrides = {}
for k, v in zip(fields, vals):
    p[k] = int(v) if k in int_fields else float(v)
    overrides[k] = p[k]

pathlib.Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

info = {
    'tag': tag,
    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
    'base_config': base_path,
    'device': device,            # ← 메타데이터에도 기록
    'sample_mode': sample_mode,
    'pipinn': overrides,
}
pathlib.Path(f"{out_dir}/run_info.json").write_text(json.dumps(info, indent=2))
PY

  # manifest 한 줄 (device 포함)
  { printf "%s" "$tag"; for v in "${vals[@]}"; do printf "\t%s" "$v"; done; printf "\t%s\t%s\n" "$device" "$out"; } >> "$MANIFEST"

  # 실행
  echo "[run ] $tag on $device"
  (
    if ! python3 -m dynalloc_v2.cli run --config "$cfg_path" \
          > "$out/stdout.log" 2> "$out/stderr.log"; then
      echo "[FAIL] $tag — check $out/stderr.log"
      touch "$out/_FAILED"
    else
      echo "[ok  ] $tag"
    fi
  ) &

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

# =============================================================================
# 헬퍼: baseline 복제 + 주어진 key=value들만 덮어쓰는 방식
# 이렇게 하면 축마다 20개 인자를 다 쓸 필요 없이 변화점만 명시하면 됨
# =============================================================================
declare -A BASE=(
  [outer_iters]=10       [eval_epochs]=50
  [n_train_int]=4096     [n_train_bc]=1024
  [n_val_int]=2048       [n_val_bc]=512
  [p_uniform]=0.5        [p_emp]=0.5
  [p_tau_head]=0.5       [p_tau_near0]=0.2
  [tau_head_window]=0
  [lr]=0.0005            [grad_clip]=1.0
  [w_bc]=10.0            [w_bc_dx]=3.0
  [scheduler_factor]=0.5 [scheduler_patience]=3
  [min_lr]=1.0e-05
  [width]=128            [depth]=2
  [pipinn-pde-form]=g
)

run () {
  # 사용: run <tag|auto> key1=val1 key2=val2 ...
  # tag를 'auto'로 주면 override 키=값들로 태그를 자동 생성
  local tag=$1; shift
  declare -A OVR=()
  for kv in "$@"; do
    OVR[${kv%%=*}]=${kv#*=}
  done
  # 태그 자동 생성
  if [[ "$tag" == "auto" ]]; then
    if [[ ${#OVR[@]} -eq 0 ]]; then
      tag="baseline"
    else
      # override 키를 정렬해서 재현성 확보 (같은 설정 → 같은 태그)
      local parts=()
      for k in $(printf '%s\n' "${!OVR[@]}" | sort); do
        local v=${OVR[$k]}
        # 소수점/마이너스/지수를 짧게 정리: 0.0005 → 5e-4, 1.0e-05 → 1e-5
        v=$(python3 -c "v='$v'; f=float(v); print(f'{f:g}'.replace('+0','+').replace('-0','-'))" 2>/dev/null || echo "$v")
        parts+=("${k}${v}")
      done
      tag=$(IFS=_; echo "${parts[*]}")
    fi
  fi
  # FIELDS 순서대로 인자 조립
  local args=()
  for f in "${FIELDS[@]}"; do
    if [[ -n "${OVR[$f]+x}" ]]; then
      args+=("${OVR[$f]}")
    else
      args+=("${BASE[$f]}")
    fi
  done
  run_variant "$tag" "${args[@]}"
}

# =============================================================================
# 여기부터 실제 sweep 정의
# 형식: run <tag> <field1>=<value1> <field2>=<value2> ...
# 명시하지 않은 필드는 위 BASE 값이 자동으로 쓰임
# =============================================================================

# (0) baseline 먼저 한 번
run  baseline

# # (A) outer × epochs — 시간 예산
run  auto    outer_iters=10   eval_epochs=100  pipinn-pde-form=g
run  auto   outer_iters=10   eval_epochs=200 pipinn-pde-form=g
run  auto  outer_iters=15   eval_epochs=50 pipinn-pde-form=g
run  auto  outer_iters=20   eval_epochs=50 pipinn-pde-form=g
run  auto  outer_iters=15   eval_epochs=100 pipinn-pde-form=g

# # (B) 네트워크 크기 (width × depth)
run  auto    width=64   depth=2 pipinn-pde-form=g
run  auto   width=128  depth=1 pipinn-pde-form=g
run  auto   width=128  depth=3 pipinn-pde-form=g
# # run  net_w192_d4   width=192  depth=4
run  auto   width=256  depth=2 pipinn-pde-form=g

# # (C) 학습률 + 스케줄러
run  auto       lr=1.0e-3 pipinn-pde-form=g
run  auto       lr=7.0e-4 pipinn-pde-form=g
# run  auto       lr=5.0e-2
# run  sched_f03_p5  scheduler_factor=0.3  scheduler_patience=5
# run  sched_f07_p2  scheduler_factor=0.7  scheduler_patience=2

# (D) BC 가중치
# run  bc_10         w_bc=10.0  w_bc_dx=1.5
run  auto   w_bc=20.0  w_bc_dx=3.0 pipinn-pde-form=g
run  auto   w_bc=10.0  w_bc_dx=5.0 pipinn-pde-form=g
run  auto   w_bc=10.0  w_bc_dx=1.0  pipinn-pde-form=g  # terminal-dominant

# (E) 샘플링 mixture (uniform vs empirical)
# run  auto  p_uniform=0.3  p_emp=0.7
# run  auto  p_uniform=0.5  p_emp=0.5
# run  auto  p_uniform=0.7  p_emp=0.3

# (F) tau head/near0 샘플링
# run  auto   p_tau_head=0.3  p_tau_near0=0.1
# run  auto   p_tau_head=0.5  p_tau_near0=0.2
# run  auto  p_tau_head=0.7  p_tau_near0=0.2
# run  tauhead_win6  p_tau_head=0.5  p_tau_near0=0.2  tau_head_window=6

# (G) 훈련/검증 콜로케이션 포인트 수
# run  auto    n_train_int=2048  n_train_bc=512   n_val_int=1024  n_val_bc=256
run  auto   n_train_int=4096  n_train_bc=1024  n_val_int=4096  n_val_bc=1024 pipinn-pde-form=g
run  auto    n_train_int=8192  n_train_bc=2048  n_val_int=4096  n_val_bc=1024 pipinn-pde-form=g

# (H) gradient clipping
# run  auto       pipinn-qp-solver-iters=500
# run  auto       pipinn-qp-solver-iters=1000
# run  auto       pipinn-qp-solver-iters=1000  pipinn-qp-solver-tol=1.0e-12

wait   # ← 이 줄 추가: 백그라운드 job이 전부 끝날 때까지 대기
echo ""
echo "[done] manifest: $MANIFEST"
echo "[done] 집계: python3 collect_results.py $TUNE_ROOT"

python3 "$(dirname "$0")/collect_results.py" "$TUNE_ROOT"