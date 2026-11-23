"""Refine rendered frames with Difix (nvidia/difix, single-step diffusion). Optional; NVIDIA License, non-commercial only.

Each frame is refined on its own (no temporal consistency). Refined details (wheels, windows, plates) are generated:
use the unrefined frames as well when evaluating or training.

Needs the `difix` environment and a Difix3D checkout (default ./Difix3D, or $DIFIX3D). The Hugging Face model needs
remote code (vae/autoencoder_kl.py); the revision is pinned to the commit we inspected.

  python -m crash2real.refine --in_dir outputs/cases/<case>/cam0 --out_dir outputs/cases/<case>/refined_cam0
"""
import argparse
import glob
import os
import sys

import torch  # noqa: F401  (initialize CUDA before diffusers)
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("DIFIX3D", os.path.join(REPO, "Difix3D")), "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pattern", default="rgb_*.png")
    ap.add_argument("--timestep", type=int, default=99, help="smaller = weaker edit and less flicker")
    ap.add_argument("--model", default="nvidia/difix")
    ap.add_argument("--revision", default="2b0c6fb5797c26b01154dfdeb19d36e5e2eaf388")
    a = ap.parse_args()

    from diffusers.utils import load_image
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(a.model, revision=a.revision, trust_remote_code=True).to("cuda")
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.in_dir, a.pattern)))
    for i, f in enumerate(files):
        img = load_image(f)
        w, h = img.size
        inp = img.resize((round(w / 8) * 8, round(h / 8) * 8), Image.LANCZOS)  # the VAE needs multiples of 8
        out = pipe("remove degradation", image=inp, num_inference_steps=1, timesteps=[a.timestep], guidance_scale=0.0).images[0]
        out.resize((w, h), Image.LANCZOS).save(os.path.join(a.out_dir, os.path.basename(f)))
        if i % 10 == 0:
            print(f"{i + 1}/{len(files)} {os.path.basename(f)}", flush=True)
    print("refined", len(files), "frames ->", a.out_dir)


if __name__ == "__main__":
    main()
