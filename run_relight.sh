#!/bin/bash
# Batch relight all scenes using relight metadata
# Usage: bash run_relight.sh [GPU_ID]
#   e.g. bash run_relight.sh 0

GPU_ID=${1:-0}
DATA_ROOT="/scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test"
MESH_ROOT="out/polyhaven"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python relight.py \
    --relight-meta-dir data_samples/relight_metadata \
    --scene-meta-root "${DATA_ROOT}/metadata" \
    --envmaps-root "${DATA_ROOT}/envmaps" \
    --mesh-root "${MESH_ROOT}" \
    --output-dir relight_output \
    --n-samples 32 \
    --probe-res 256

echo "All relighting done."
