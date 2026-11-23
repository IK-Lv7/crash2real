#!/usr/bin/env bash
# Create the conda environment "drivestudio" (tested on WSL2, RTX A6000, driver CUDA 13).
# CUDA 11.7 and gcc 11 are installed inside the environment to build the CUDA extensions.
# Usage (repository root, after cloning drivestudio into ./drivestudio): bash scripts/setup_drivestudio_env.sh
set -eo pipefail

ENV=drivestudio
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda env list | grep -q "^$ENV " || conda create -n "$ENV" python=3.9 -y
conda activate "$ENV"

conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install -y -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11" ninja cmake

# Fix sysroot linker scripts (libc.so/libm.so) that may contain absolute paths of another conda env;
# ld resolves them relative to the sysroot, so they must read /lib64/libc.so.6 etc.
for f in "$CONDA_PREFIX"/x86_64-conda-linux-gnu/sysroot/lib64/{libc,libm}.so; do
  sed -i -E 's#/[^ ]*/x86_64-conda-linux-gnu/sysroot/lib64/\.\./usr/lib64#/usr/lib64#g; s#/[^ ]*/x86_64-conda-linux-gnu/sysroot##g' "$f"
done
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX A6000 (Ampere)
export MAX_JOBS=16

cd "$ROOT/drivestudio"
git submodule update --init --recursive
pip install "setuptools<70" wheel
pip install --no-build-isolation chumpy   # its setup.py imports pip
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast
(cd third_party/smplx && pip install -e .)
pip install pymunk shapely   # collision physics and box geometry

# AV2 preprocessing needs drivestudio's fork of av2-api (track ids), which needs Rust nightly.
command -v rustup >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
rustup default nightly-2024-01-25   # newer nightlies fail to build ethnum 1.5.0 / ahash 0.8.6
pip install git+https://github.com/ziyc/av2-api

python -c "import torch, gsplat, pytorch3d, pymunk; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gsplat', gsplat.__version__)"
