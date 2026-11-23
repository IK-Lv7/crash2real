#!/usr/bin/env bash
# Reconstruct AV2 logs: preprocess -> sky masks -> train OmniRe (3DGS). One scene at a time.
# Usage (repository root): bash scripts/train_scene.sh <scene_idx>...
#   scene_idx = line number - 1 in configs/av2_logs.txt. Output: outputs/av2/<first 8 chars of the log id>_omnire
# Needs: conda envs drivestudio and segformer, AV2_DIR (the logs are read from $AV2_DIR/sensor/val/<LOG_ID>).
# About 1-3 h and 22 GB of GPU memory per scene. Do not run two GPU jobs at once (WSL crashed when we did).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$ROOT/drivestudio"
export PYTHONPATH="$(pwd)" AV2_LOG_LIST="$ROOT/configs/av2_logs.txt"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}   # WSL: cuDNN needs libcuda.so
mkdir -p data/argoverse
[ -e data/argoverse/raw ] || ln -s "${AV2_DIR:?set AV2_DIR}/sensor" data/argoverse/raw
PROC=data/argoverse/processed/training

for i in "$@"; do
  name=$(sed -n "$((i+1))p" "$AV2_LOG_LIST" | cut -c1-8)
  if [ ! -f "$PROC/$(printf %03d "$i")/sky_masks/000_0.png" ]; then
    echo "=== preprocess scene $i ($name)"
    rm -f ~/.cache/av2/*_cache.feather   # av2 caches the log list once; new logs would come out empty
    conda activate drivestudio
    python datasets/preprocess.py --data_root data/argoverse/raw --target_dir "$PROC" --dataset argoverse \
      --scene_ids "$i" --workers 1 --process_keys images lidar calib pose dynamic_masks objects
    conda activate segformer
    python datasets/tools/extract_masks.py --data_root "$PROC" --segformer_path="$ROOT/SegFormer" \
      --checkpoint="$ROOT/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth" --scene_ids "$i" --process_dynamic_mask
  fi
  if [ -f "$ROOT/outputs/av2/${name}_omnire/checkpoint_final.pth" ]; then echo "=== scene $i ($name) already trained"; continue; fi
  echo "=== train scene $i ($name)"
  conda activate drivestudio
  # Resuming from an intermediate checkpoint does not work in drivestudio e59bda4 (optimizer state); restart from scratch on failure.
  python tools/train.py --config_file "$ROOT/configs/omnire_nosmpl.yaml" --output_root "$ROOT/outputs" \
    --project av2 --run_name "${name}_omnire" dataset=argoverse/5cams data.scene_idx="$i" \
    data.pixel_source.load_smpl=False data.pixel_source.test_image_stride=10 render.render_full=False
done
