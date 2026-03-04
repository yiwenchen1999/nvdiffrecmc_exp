conda create python=3.9 cmake=3.14.0 anaconda --prefix=/projects/vig/yiwenc/all_env/nvd
export CONDA_PKGS_DIRS=/scratch/chen.yiwe/conda_pkgs
mkdir -p $CONDA_PKGS_DIRS
# 之后再做 conda create 等操作
export PIP_CACHE_DIR=/scratch/chen.yiwe/pip_cache
mkdir -p $PIP_CACHE_DIR
conda activate /projects/vig/yiwenc/all_env/nvd
cd /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/nvdiffrecmc/nvdiffrecmc_exp

python train.py \
  --config configs/polyhaven.json \
  --ref_mesh /scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test/metadata/ceramic_vase_02_white_env_0.json \
  --out-dir polyhaven/ceramic_vase_02_white_env_0

  python relight.py \
    --relight-meta relight_metadata/ceramic_vase_02_white_env_0.json \
    --scene-meta-root /scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test/metadata \
    --envmaps-root /scratch/chen.yiwe/temp_objaverse/polyhaven_lvsm/test/envmaps \
    --mesh-root out/polyhaven \
    --output-dir relight_output