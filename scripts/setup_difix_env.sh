#!/usr/bin/env bash
# Create the conda environment "difix" for the optional Difix refinement (NVIDIA License, non-commercial only).
# Separate from drivestudio because Difix3D pins old diffusers. Weights come from Hugging Face (nvidia/difix) on first use.
# Usage (repository root): bash scripts/setup_difix_env.sh
set -eo pipefail
ENV=difix
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV " || conda create -n "$ENV" python=3.10 -y
conda activate "$ENV"
[ -d "$ROOT/Difix3D" ] || git clone --depth 1 https://github.com/nv-tlabs/Difix3D.git "$ROOT/Difix3D"
pip install "numpy<2" Pillow
pip install -r "$ROOT/Difix3D/requirements.txt"
python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__)"
