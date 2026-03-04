#!/bin/bash
set -e

# ============================================================
# Step 1: Install Miniconda (skip if already installed)
# ============================================================
if ! command -v conda &> /dev/null; then
    echo ">>> Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo ">>> Miniconda installed. Please run: source ~/.bashrc && bash setup.sh"
    exit 0
else
    echo ">>> Conda found at: $(which conda)"
    eval "$(conda shell.bash hook)"
fi

# ============================================================
# Step 2: Create conda env
# ============================================================
ENV_NAME="dmodel"

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo ">>> Env '${ENV_NAME}' exists, activating..."
else
    echo ">>> Creating conda env '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.9 -y
fi
conda activate ${ENV_NAME}

# ============================================================
# Step 3: Install PyTorch (2.1.0 + CUDA 12.1, supports H100)
# ============================================================
echo ">>> Installing PyTorch 2.1.0 + cu121..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# Step 4: Install dependencies
# ============================================================
echo ">>> Installing dependencies..."
pip install ninja imageio PyOpenGL glfw xatlas gdown

# ============================================================
# Step 5: Install nvdiffrast
# ============================================================
echo ">>> Installing nvdiffrast..."
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/

# ============================================================
# Step 6: Install tiny-cuda-nn (with C++17 fix for PyTorch 2.x)
# ============================================================
echo ">>> Installing tiny-cuda-nn..."
pip install setuptools
export TCNN_CUDA_ARCHITECTURES=90
export CXXFLAGS="-std=c++17"
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# ============================================================
# Step 7: Download freeimage for imageio
# ============================================================
echo ">>> Downloading freeimage..."
imageio_download_bin freeimage

echo ""
echo "============================================================"
echo "Setup complete! Activate with: conda activate ${ENV_NAME}"
echo "============================================================"