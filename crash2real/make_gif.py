"""Make a preview GIF of a generated case: camera frames (refined if available) and a top view. CPU only.

  python -m crash2real.make_gif outputs/cases/<case> --cams 0 1 --out outputs/gifs/<case>.gif
"""
import argparse
import json
import os

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .physics import DT, EGO_L, EGO_W, box_poly, ego_center, yaw_of  # noqa: E402

CAM_NAMES = {0: "front", 1: "front-left", 2: "front-right", 3: "side-left", 4: "side-right"}


def topdown(ego_rec, ego_new, part_rec, part_new, size, f, ext):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=100)
    ax.plot(ego_rec[:, 0, 3], ego_rec[:, 1, 3], ":", color="0.6", lw=1)
    ax.plot(ego_new[:, 0, 3], ego_new[:, 1, 3], "-", color="tab:blue", lw=1)
    ok = ~np.isnan(part_rec[:, 0, 3])
    ax.plot(part_rec[ok, 0, 3], part_rec[ok, 1, 3], ":", color="salmon", lw=1)
    ax.plot(part_new[:, 0], part_new[:, 1], "-", color="tab:red", lw=1)
    ec = ego_center(ego_new[f])
    for poly, col in ((box_poly(ec[0], ec[1], float(yaw_of(ego_new[f])), EGO_L, EGO_W), "tab:blue"),
                      (box_poly(part_new[f, 0], part_new[f, 1], part_new[f, 2], size[0], size[1]), "tab:red")):
        ax.fill(*poly.exterior.xy, color=col, alpha=0.6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("top view  (blue: ego / red: partner)", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def label(img, text, after=False):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    if after:
        d.rectangle([0, im.height - 18, im.width, im.height], fill=(200, 0, 0))
        d.text((4, im.height - 15), "AFTER IMPACT", fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir")
    ap.add_argument("--cams", type=int, nargs="+", default=[0])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = json.load(open(os.path.join(a.case_dir, "scenario.json")))
    Tc, t0, (f0, f1), size = s["Tc"], s["t0"], s["frames"], s["partner_size"]
    ego_rec, ego_new = np.array(s["ego_recorded"]), np.array(s["ego_new"])
    part_rec = np.array(s["partner_recorded"], dtype=float)
    full = {k: np.array(v) for k, v in s["partner_new"].items()}
    N = len(ego_rec)
    part_new = np.full((N, 3), np.nan)
    for t in range(N):
        if t >= t0:
            i = min(t - t0, len(full["x"]) - 1)
            part_new[t] = [full["x"][i], full["y"][i], full["yaw"][i]]
        elif not np.isnan(part_rec[t, 0, 3]):
            part_new[t] = [part_rec[t, 0, 3], part_rec[t, 1, 3], np.arctan2(part_rec[t, 1, 0], part_rec[t, 0, 0])]
    xs = np.concatenate([ego_new[f0:f1 + 1, 0, 3], part_new[f0:f1 + 1, 0]])
    ys = np.concatenate([ego_new[f0:f1 + 1, 1, 3], part_new[f0:f1 + 1, 1]])
    cx, cy = (np.nanmin(xs) + np.nanmax(xs)) / 2, (np.nanmin(ys) + np.nanmax(ys)) / 2
    half = max(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys)) / 2 + 4.0
    ext = (cx - half, cx + half, cy - half, cy + half)

    frames_out = []
    for f in range(f0, f1 + 1):
        tag = "t = 0.0 s (contact)" if f == Tc else f"t = {(f - Tc) * DT:+.1f} s"
        panels = []
        for cam in a.cams:
            d = os.path.join(a.case_dir, f"refined_cam{cam}")
            d = d if os.path.isdir(d) else os.path.join(a.case_dir, f"cam{cam}")
            img = np.asarray(Image.open(os.path.join(d, f"rgb_{f:03d}.png")).convert("RGB"))
            panels.append(label(img, f"{CAM_NAMES[cam]}  frame {f}  {tag}", after=f > Tc))
        h = panels[0].shape[0]
        top = np.asarray(Image.fromarray(topdown(ego_rec, ego_new, part_rec, part_new, size, f, ext)).resize((h, h)))
        im = Image.fromarray(np.hstack(panels + [top]))
        frames_out.append(im.resize((int(im.width * a.scale), int(im.height * a.scale)), Image.LANCZOS))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    frames_out[0].save(a.out, save_all=True, append_images=frames_out[1:], duration=int(DT * 1000), loop=0, optimize=True)
    print("saved", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB", len(frames_out), "frames")


if __name__ == "__main__":
    main()
