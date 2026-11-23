#!/usr/bin/env bash
# Create the conda environment "segformer" for sky masks (needs torch 1.8, so separate from drivestudio).
# drivestudio's docs ask for mmcv-full 1.2.7 built from source; we use the prebuilt 1.3.0 wheel (SegFormer's MMCV_MAX).
# Usage (repository root): bash scripts/setup_segformer_env.sh
set -eo pipefail
ENV=segformer
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV " || conda create -n "$ENV" python=3.8 -y
conda activate "$ENV"

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf gdown
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
[ -d "$ROOT/SegFormer" ] || git clone --depth 1 https://github.com/NVlabs/SegFormer.git "$ROOT/SegFormer"
(cd "$ROOT/SegFormer" && pip install .)

mkdir -p "$ROOT/SegFormer/pretrained"   # weights linked from drivestudio's docs
CKPT="$ROOT/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth"
[ -f "$CKPT" ] || gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj -O "$CKPT"
python -c "import torch, mmcv; from mmcv.ops import nms; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mmcv', mmcv.__version__)"
