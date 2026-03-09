#!/bin/bash
# Benchmark: train + relight ONE scene, report timing and resource usage
# Usage: bash run_train_and_relight_benchmark.sh [GPU_ID] [SCENE_NAME]

GPU_ID=${1:-0}
SCENE=${2:-"ceramic_vase_01_white_env_0"}
DATA_ROOT="/scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test"
METADATA_DIR="${DATA_ROOT}/metadata"
CONFIG="configs/polyhaven.json"
BENCHMARK_LOG="benchmark_${SCENE}.txt"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ── helpers ──────────────────────────────────────────────────────────────────

get_gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | head -1 | tr -d ' '
}

get_cpu_rss_mb() {
    # RSS of the most recent background python child (passed via $1 = PID)
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}'
    else
        echo "0"
    fi
}

poll_peak_resources() {
    # Polls GPU mem and CPU RSS of PID $1 every 2s, writes peak to vars
    local pid=$1
    local peak_gpu=0
    local peak_cpu=0
    while kill -0 "$pid" 2>/dev/null; do
        local gpu_now
        gpu_now=$(get_gpu_mem)
        gpu_now=${gpu_now:-0}
        local cpu_now
        cpu_now=$(get_cpu_rss_mb "$pid")
        cpu_now=${cpu_now:-0}
        (( gpu_now > peak_gpu )) && peak_gpu=$gpu_now
        (( cpu_now > peak_cpu )) && peak_cpu=$cpu_now
        sleep 2
    done
    echo "${peak_gpu} ${peak_cpu}"
}

fmt_duration() {
    local secs=$1
    local h=$((secs / 3600))
    local m=$(( (secs % 3600) / 60 ))
    local s=$((secs % 60))
    printf "%02d:%02d:%02d" $h $m $s
}

# ── start benchmark ─────────────────────────────────────────────────────────

echo "============================================================"
echo "  BENCHMARK: ${SCENE}  (GPU ${GPU_ID})"
echo "============================================================"
echo ""

E2E_START=$(date +%s)

# ── Phase 1: Training ────────────────────────────────────────────────────────

echo ">>> Phase 1: Training"
TRAIN_START=$(date +%s)

# Force retrain (remove old output if exists)
rm -rf "out/polyhaven/${SCENE}"

python train.py \
    --config "${CONFIG}" \
    --ref_mesh "${METADATA_DIR}/${SCENE}.json" \
    --out-dir "polyhaven/${SCENE}" &
TRAIN_PID=$!

# Poll resources in background
poll_peak_resources $TRAIN_PID > /tmp/bench_train_peak_$$ &
POLL_TRAIN_PID=$!

wait $TRAIN_PID
TRAIN_EXIT=$?
wait $POLL_TRAIN_PID 2>/dev/null
read TRAIN_PEAK_GPU TRAIN_PEAK_CPU < /tmp/bench_train_peak_$$
rm -f /tmp/bench_train_peak_$$

TRAIN_END=$(date +%s)
TRAIN_SECS=$((TRAIN_END - TRAIN_START))

echo ""
echo "  Training exit code : ${TRAIN_EXIT}"
echo "  Training time      : $(fmt_duration $TRAIN_SECS) (${TRAIN_SECS}s)"
echo "  Peak GPU memory    : ${TRAIN_PEAK_GPU} MiB"
echo "  Peak CPU memory    : ${TRAIN_PEAK_CPU} MiB"
echo ""

# ── Phase 2: Relighting ─────────────────────────────────────────────────────

echo ">>> Phase 2: Relighting"
RELIGHT_START=$(date +%s)

META="relight_metadata/${SCENE}.json"
if [ ! -f "${META}" ]; then
    echo "  [ERROR] Relight metadata not found: ${META}"
    RELIGHT_SECS=0
    RELIGHT_PEAK_GPU=0
    RELIGHT_PEAK_CPU=0
    RELIGHT_EXIT=1
else
    python relight.py \
        --relight-meta "${META}" \
        --scene-meta-root "${DATA_ROOT}/metadata" \
        --envmaps-root "${DATA_ROOT}/envmaps" \
        --mesh-root "out/polyhaven" \
        --tonemap "reinhard" \
        --output-dir relight_output &
    RELIGHT_PID=$!

    poll_peak_resources $RELIGHT_PID > /tmp/bench_relight_peak_$$ &
    POLL_RELIGHT_PID=$!

    wait $RELIGHT_PID
    RELIGHT_EXIT=$?
    wait $POLL_RELIGHT_PID 2>/dev/null
    read RELIGHT_PEAK_GPU RELIGHT_PEAK_CPU < /tmp/bench_relight_peak_$$
    rm -f /tmp/bench_relight_peak_$$
fi

RELIGHT_END=$(date +%s)
RELIGHT_SECS=$((RELIGHT_END - RELIGHT_START))

echo ""
echo "  Relight exit code  : ${RELIGHT_EXIT}"
echo "  Relight time       : $(fmt_duration $RELIGHT_SECS) (${RELIGHT_SECS}s)"
echo "  Peak GPU memory    : ${RELIGHT_PEAK_GPU} MiB"
echo "  Peak CPU memory    : ${RELIGHT_PEAK_CPU} MiB"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────

E2E_END=$(date +%s)
E2E_SECS=$((E2E_END - E2E_START))

OVERALL_PEAK_GPU=$(( TRAIN_PEAK_GPU > RELIGHT_PEAK_GPU ? TRAIN_PEAK_GPU : RELIGHT_PEAK_GPU ))
OVERALL_PEAK_CPU=$(( TRAIN_PEAK_CPU > RELIGHT_PEAK_CPU ? TRAIN_PEAK_CPU : RELIGHT_PEAK_CPU ))

echo "============================================================"
echo "  BENCHMARK SUMMARY: ${SCENE}"
echo "============================================================"
echo "  End-to-end time    : $(fmt_duration $E2E_SECS) (${E2E_SECS}s)"
echo "  ├─ Training        : $(fmt_duration $TRAIN_SECS) (${TRAIN_SECS}s)"
echo "  └─ Relighting      : $(fmt_duration $RELIGHT_SECS) (${RELIGHT_SECS}s)"
echo ""
echo "  Peak GPU memory    : ${OVERALL_PEAK_GPU} MiB"
echo "  ├─ Training        : ${TRAIN_PEAK_GPU} MiB"
echo "  └─ Relighting      : ${RELIGHT_PEAK_GPU} MiB"
echo ""
echo "  Peak CPU memory    : ${OVERALL_PEAK_CPU} MiB"
echo "  ├─ Training        : ${TRAIN_PEAK_CPU} MiB"
echo "  └─ Relighting      : ${RELIGHT_PEAK_CPU} MiB"
echo "============================================================"

# Write to log file
cat > "${BENCHMARK_LOG}" <<LOGEOF
scene: ${SCENE}
gpu: ${GPU_ID}
date: $(date -Iseconds)

end_to_end_secs: ${E2E_SECS}
train_secs: ${TRAIN_SECS}
relight_secs: ${RELIGHT_SECS}

train_exit: ${TRAIN_EXIT}
relight_exit: ${RELIGHT_EXIT}

peak_gpu_mib_overall: ${OVERALL_PEAK_GPU}
peak_gpu_mib_train: ${TRAIN_PEAK_GPU}
peak_gpu_mib_relight: ${RELIGHT_PEAK_GPU}

peak_cpu_mib_overall: ${OVERALL_PEAK_CPU}
peak_cpu_mib_train: ${TRAIN_PEAK_CPU}
peak_cpu_mib_relight: ${RELIGHT_PEAK_CPU}
LOGEOF

echo "Log written to ${BENCHMARK_LOG}"
