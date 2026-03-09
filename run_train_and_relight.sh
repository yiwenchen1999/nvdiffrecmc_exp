#!/bin/bash
# Train and relight specified scenes end-to-end
# Usage: bash run_train_and_relight.sh [GPU_ID]

GPU_ID=${1:-0}
DATA_ROOT="/scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test"
METADATA_DIR="${DATA_ROOT}/metadata"
CONFIG="configs/polyhaven.json"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

SCENES=(
    "ceramic_vase_02_white_env_0"
    "marble_bust_01_env_2"
    "pot_enamel_01_white_env_0"
    "potted_plant_02_white_env_0"
)

# # ===== Phase 1: Train all scenes =====
# for scene in "${SCENES[@]}"; do
#     echo "========================================="
#     echo "  TRAINING: ${scene}"
#     echo "========================================="

#     if [ -f "out/polyhaven/${scene}/mesh/mesh.obj" ]; then
#         echo "  Mesh already exists, skipping training."
#     else
#         python train.py \
#             --config "${CONFIG}" \
#             --ref_mesh "${METADATA_DIR}/${scene}.json" \
#             --out-dir "polyhaven/${scene}"
#     fi
#     echo ""
# done

# ===== Phase 2: Relight all scenes =====
echo "========================================="
echo "  RELIGHTING ALL SCENES"
echo "========================================="

for scene in "${SCENES[@]}"; do
    meta="relight_metadata/${scene}.json"
    if [ ! -f "${meta}" ]; then
        echo "  [SKIP] Relight metadata not found: ${meta}"
        continue
    fi

    echo "----- Relighting: ${scene} -----"
    python relight.py \
        --relight-meta "${meta}" \
        --scene-meta-root "${DATA_ROOT}/metadata" \
        --envmaps-root "${DATA_ROOT}/envmaps" \
        --mesh-root "out/polyhaven" \
        --tonemap "reinhard" \
        --output-dir relight_output
    echo ""
done

echo "All done."
