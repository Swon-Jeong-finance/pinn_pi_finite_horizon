#!/bin/bash
# =============================================================================
# run_eps_sweep_parallel.sh
# =============================================================================
# GPU 병렬로 epsilon sweep 실험을 실행하는 스크립트
#
# Usage:
#   ./run_eps_sweep_parallel.sh                    # 기본값 사용
#   ./run_eps_sweep_parallel.sh "0.2 0.1 0.05"     # epsilon 리스트 지정
#   ./run_eps_sweep_parallel.sh "0.2 0.1" "0 1"   # epsilon + GPU 리스트 지정
#
# =============================================================================

set -e  # 에러 시 중단

# =============================================================================
# Configuration
# =============================================================================

# Epsilon 리스트 (기본값)
DEFAULT_EPS_LIST="2.50 3.00 3.50 4.00 4.50 5.00"

# GPU 리스트 (기본값: 사용 가능한 GPU 자동 감지)
DEFAULT_GPU_LIST=""

# Python 스크립트 경로
PYTHON_SCRIPT="Liu_nd_pi_pinn_tanh_parallel.py"

# 로그 디렉토리
LOG_DIR="logs/eps_sweep_$(date +%Y%m%d_%H%M%S)"

# 기타 하이퍼파라미터 (필요시 수정)
N_ASSETS=10
M_STATES=2
OUTER_ITERS=1000
EVAL_EPOCHS=200
BATCH_SIZE=3000
SEED=12

# =============================================================================
# Parse Arguments
# =============================================================================
EPS_LIST="${1:-$DEFAULT_EPS_LIST}"
GPU_LIST="${2:-$DEFAULT_GPU_LIST}"

# GPU 리스트가 비어있으면 자동 감지
if [ -z "$GPU_LIST" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ' ')
        echo "[Info] Detected $NUM_GPUS GPUs: $GPU_LIST"
    else
        GPU_LIST="0"
        echo "[Warning] nvidia-smi not found, defaulting to GPU 0"
    fi
fi

# 배열로 변환
EPS_ARRAY=($EPS_LIST)
GPU_ARRAY=($GPU_LIST)

NUM_EPS=${#EPS_ARRAY[@]}
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================================"
echo "Epsilon Sweep Parallel Execution"
echo "============================================================"
echo "Epsilon values ($NUM_EPS): ${EPS_ARRAY[*]}"
echo "Available GPUs ($NUM_GPUS): ${GPU_ARRAY[*]}"
echo "Python script: $PYTHON_SCRIPT"
echo "Log directory: $LOG_DIR"
echo "============================================================"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# =============================================================================
# Run Experiments in Parallel
# =============================================================================
PIDS=()
EPS_TO_PID=()

for i in "${!EPS_ARRAY[@]}"; do
    EPS=${EPS_ARRAY[$i]}
    
    # Round-robin GPU 할당
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPU_ARRAY[$GPU_IDX]}
    
    EPS_FMT=$(printf "%.4f" "$EPS")
    LOG_FILE="$LOG_DIR/eps_${EPS_FMT}_gpu${GPU}.log"
    
    echo "[Starting] EPS=$EPS_FMT on GPU=$GPU (log: $LOG_FILE)"
    
    # 백그라운드에서 실행
    python3 "$PYTHON_SCRIPT" \
        --eps "$EPS_FMT" \
        --gpu "$GPU" \
        --seed "$SEED" \
        --n_assets "$N_ASSETS" \
        --m_states "$M_STATES" \
        --outer_iters "$OUTER_ITERS" \
        --eval_epochs "$EVAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    EPS_TO_PID+=("$EPS:$PID:$GPU")
    
    echo "  -> PID=$PID"
    
    # GPU당 동시 실행 제한 (선택적)
    # 같은 GPU에 다음 작업이 할당되기 전 잠시 대기
    sleep 2
done

echo ""
echo "============================================================"
echo "All jobs submitted. Waiting for completion..."
echo "============================================================"
echo ""

# =============================================================================
# Wait for All Jobs and Report Status
# =============================================================================
FAILED=0
SUCCEEDED=0

for entry in "${EPS_TO_PID[@]}"; do
    EPS=$(echo "$entry" | cut -d: -f1)
    PID=$(echo "$entry" | cut -d: -f2)
    GPU=$(echo "$entry" | cut -d: -f3)
    
    echo -n "[Waiting] EPS=$EPS (PID=$PID, GPU=$GPU)..."
    
    if wait "$PID"; then
        echo " ✓ SUCCESS"
        ((SUCCEEDED++))
    else
        EXIT_CODE=$?
        echo " ✗ FAILED (exit code: $EXIT_CODE)"
        ((FAILED++))
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Total jobs: $NUM_EPS"
echo "Succeeded: $SUCCEEDED"
echo "Failed: $FAILED"
echo "Log directory: $LOG_DIR"
echo "============================================================"

# =============================================================================
# Aggregate Results (Optional)
# =============================================================================
SUMMARY_FILE="$LOG_DIR/summary.txt"
echo "Aggregating results to $SUMMARY_FILE..."

echo "Epsilon Sweep Results - $(date)" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
printf "%-10s | %-12s | %-12s | %-12s | %-14s\n" \
    "eps" "MSE_V" "MSE_theta" "RelRMSE_V" "RelRMSE_theta" >> "$SUMMARY_FILE"
echo "---------------------------------------------" >> "$SUMMARY_FILE"

for EPS in "${EPS_ARRAY[@]}"; do
    EPS_FMT=$(printf "%.4f" "$EPS")
    # metrics 파일 찾기
    METRICS_FILE="outputs/pi-pinn/kim_omberg_${N_ASSETS}asset-${M_STATES}state_eps${EPS}/metrics_eps${EPS}.txt"
    
    if [ -f "$METRICS_FILE" ]; then
        MSE_V=$(grep "MSE_V=" "$METRICS_FILE" | cut -d= -f2)
        MSE_THETA=$(grep "MSE_theta=" "$METRICS_FILE" | cut -d= -f2)
        REL_V=$(grep "RelRMSE_V=" "$METRICS_FILE" | cut -d= -f2)
        REL_THETA=$(grep "RelRMSE_theta=" "$METRICS_FILE" | cut -d= -f2)
        
        printf "%-10s | %-12.3e | %-12.3e | %-12.3e | %-14.3e\n" \
            "$EPS" "$MSE_V" "$MSE_THETA" "$REL_V" "$REL_THETA" >> "$SUMMARY_FILE"
    else
        printf "%-10s | %-12s | %-12s | %-12s | %-14s\n" \
            "$EPS" "N/A" "N/A" "N/A" "N/A" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"
cat "$SUMMARY_FILE"

echo ""
echo "[Done] All experiments completed!"

# 실패한 작업이 있으면 비정상 종료
if [ $FAILED -gt 0 ]; then
    exit 1
fi
