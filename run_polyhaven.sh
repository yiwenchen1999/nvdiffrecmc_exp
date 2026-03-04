#!/bin/bash
# Batch optimize all scenes in /data/polyhaven_lvsm/test
# Usage: bash run_polyhaven.sh [GPU_ID]
#   e.g. bash run_polyhaven.sh 0

GPU_ID=${1:-0}
DATA_ROOT="/data/polyhaven_lvsm/test"
METADATA_DIR="${DATA_ROOT}/metadata"
CONFIG="configs/polyhaven.json"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

for metadata_json in "${METADATA_DIR}"/*.json; do
    scene_name=$(basename "${metadata_json}" .json)
    out_dir="polyhaven/${scene_name}"

    echo "===== Processing scene: ${scene_name} ====="

    python train.py \
        --config "${CONFIG}" \
        --ref_mesh "${metadata_json}" \
        --out-dir "${out_dir}"

    echo "===== Done: ${scene_name} ====="
    echo ""
done

echo "All scenes processed."
