"""Render one collision case from the ego cameras and write its labels.

Output (--out_dir):
  scenario.json                    case options, contact frame, ego/partner trajectories, speed change
  cam{c}/rgb_{f}.png               rendered image (before refinement)
  cam{c}/depth_{f}.npz             depth [m] along the camera z axis (float16) and opacity
  cam{c}/mask_partner_visible_{f}.png   partner pixels not hidden by anything else
  cam{c}/mask_partner_amodal_{f}.png    partner pixels ignoring occlusion
  cam{c}/mask_vehicles_{f}.png          all rigid cars
  boxes.json                       3D boxes of every car per frame and camera (world, camera, image)
  viz/cam{c}_{f}.png               overlay for checking (red = partner box and mask, green = other cars, depth)

Labels come from the rendered geometry before refinement; refinement changes only the look.
Cameras: 0 front, 1 front-left, 2 front-right, 3 side-left, 4 side-right.

  python -m crash2real.generate --run outputs/av2/<name>_omnire <case options> --cams 0 1 3 --out_dir outputs/cases/<case>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import scene as S
from .scenario import add_case_args, case_from_args

EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(center, R, size):
    """8 corners in world coordinates. Object frame: x front, y left, z up. Order: (sx, sy, sz) in {-,+}^3, x most significant."""
    L, W, H = size
    c = np.array([[sx * L / 2, sy * W / 2, sz * H / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (R @ c.T).T + center


def project(pts_cam, K):
    """Camera points (n,3) -> pixels (n,2) and a mask of points in front. Camera frame: x right, y down, z forward."""
    uv = (K @ pts_cam.T).T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), pts_cam[:, 2] > 0.1


def bbox_clipped(cc, K, W, H, near=0.1):
    """2D box of the 8 camera-frame corners. Edges crossing z=near are cut there, so a box partly behind the camera does not shrink."""
    pts = [p for p in cc if p[2] > near]
    for i in range(8):
        for bit in (1, 2, 4):  # corners that differ in one bit share an edge
            j = i ^ bit
            if j > i and (cc[i, 2] > near) != (cc[j, 2] > near):
                t = (near - cc[i, 2]) / (cc[j, 2] - cc[i, 2])
                pts.append(cc[i] + t * (cc[j] - cc[i]))
    if not pts:
        return None
    uv = (K @ np.array(pts).T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return [float(np.clip(uv[:, 0].min(), 0, W)), float(np.clip(uv[:, 1].min(), 0, H)),
            float(np.clip(uv[:, 0].max(), 0, W)), float(np.clip(uv[:, 1].max(), 0, H))]


def car_boxes(rigid, f, idx, w2c, K, W, H):
    from pytorch3d.transforms import quaternion_to_matrix
    size = rigid.instances_size.detach().cpu().numpy()
    trans = rigid.instances_trans.detach().cpu().numpy()
    quats = rigid.instances_quats.detach().cpu()
    fv = rigid.instances_fv.detach().cpu().numpy()
    out = []
    for k in range(trans.shape[1]):
        if not fv[f, k]:
            continue
        R = quaternion_to_matrix(quats[f, k]).numpy()
        cc = (w2c[:3, :3] @ box_corners(trans[f, k], R, size[k]).T).T + w2c[:3, 3]
        uv, ok = project(cc, K)
        center_cam = w2c[:3, :3] @ trans[f, k] + w2c[:3, 3]
        if not ok.any():
            continue
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if center_cam[2] <= 0.1 and not inside.any():
            continue
        out.append(dict(rigid_index=int(k), is_partner=bool(k == idx), size_lwh=[float(x) for x in size[k]],
                        center_world=[float(x) for x in trans[f, k]], center_cam=[float(x) for x in center_cam],
                        yaw_world=float(np.arctan2(R[1, 0], R[0, 0])),
                        corners_cam=cc.round(3).tolist(), corners_uv=uv.round(1).tolist(), corners_in_front=ok.tolist(),
                        bbox_xyxy=bbox_clipped(cc, K, W, H), n_corners_in_image=int(inside.sum()),
                        distance=float(np.linalg.norm(center_cam))))
    return out


def overlay(rgb, visible, depth, alpha, boxes):
    im = Image.fromarray(rgb).convert("RGBA")
    ov = np.zeros((*rgb.shape[:2], 4), np.uint8)
    ov[visible] = (255, 0, 0, 90)
    im = Image.alpha_composite(im, Image.fromarray(ov))
    draw = ImageDraw.Draw(im)
    for b in boxes:
        uv, ok = np.array(b["corners_uv"]), np.array(b["corners_in_front"])
        color = (255, 40, 40, 255) if b["is_partner"] else (40, 220, 40, 255)
        for i, j in EDGES:
            if ok[i] and ok[j]:
                draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=2)
    d = np.clip(depth / 40.0, 0, 1)
    dep = (np.stack([d, 1 - np.abs(d - 0.5) * 2, 1 - d], -1) * 255 * (alpha[..., None] > 0.3)).astype(np.uint8)
    return np.hstack([np.asarray(im.convert("RGB")), dep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="trained scene, e.g. outputs/av2/<name>_omnire")
    add_case_args(ap)
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--viz_every", type=int, default=10, help="write an overlay every N frames (and at contact)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)

    cfg, dataset, trainer = S.load_scene(a.run)
    scene_dir = S.scene_dir_of(cfg)
    case = case_from_args(scene_dir, a, verbose=True)
    ego_orig, Tc, t0, frames, sc = case["ego_orig"], case["Tc"], case["t0"], case["frames"], case["sc"]
    rigid = trainer.models["RigidNodes"]
    idx, err = S.match_instance(rigid, scene_dir, a.instance)
    print(f"partner {a.instance} -> RigidNodes {idx} (trajectory error {err:.2f} m)")
    if a.partner_donor:
        didx, derr = S.match_instance(rigid, scene_dir, a.partner_donor)
        n_old, n_new = S.transplant(rigid, didx, idx)
        print(f"partner drawn as instance {a.partner_donor} (RigidNodes {didx}, error {derr:.2f} m): {n_old} -> {n_new} Gaussians")
    S.write_trajectory(rigid, idx, sc["full"], t0, ego=ego_orig)
    ego_new = sc["ego_new"]

    os.makedirs(os.path.join(out_dir, "viz"), exist_ok=True)
    json.dump(dict(cli={k: v for k, v in vars(a).items() if k not in ("run", "out_dir", "cams", "viz_every")}, run=os.path.abspath(a.run),
                   scene_dir=scene_dir, Tc=Tc, t0=t0, frames=[frames[0], frames[-1]], cams=a.cams, rigid_index=idx,
                   dv_ego=sc["dv_ego"], dv_partner=sc["dv_partner"], restitution=sc["restitution"], partner_size=case["size"].tolist(),
                   ego_recorded=ego_orig.tolist(), ego_new=ego_new.tolist(), partner_recorded=case["part"].tolist(),
                   partner_new={k: np.asarray(v).tolist() for k, v in sc["full"].items()}),
              open(os.path.join(out_dir, "scenario.json"), "w"))
    boxes = {"meta": {"instance": a.instance, "rigid_index": idx, "Tc": Tc, "frames": [frames[0], frames[-1]],
                      "box_corner_order": "(sx,sy,sz) in {-,+}^3, x=front y=left z=up", "camera_convention": "x right, y down, z forward (OpenCV)",
                      "note": "labels come from the rendered geometry before refinement"}, "frames": {}}
    for cam in a.cams:
        cdir = os.path.join(out_dir, f"cam{cam}")
        os.makedirs(cdir, exist_ok=True)
        cam_new = S.camera_path(dataset, cam, ego_orig, ego_new)
        data = dataset.prepare_novel_view_render_data(torch.as_tensor(cam_new, dtype=torch.float32))
        K = data[0]["cam_infos"]["intrinsics"].cpu().numpy()
        H, W = int(data[0]["cam_infos"]["height"].item()), int(data[0]["cam_infos"]["width"].item())
        for f in frames:
            o = S.render(trainer, data[f])
            rgb = (o["rgb"].float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depth, alpha = o["depth"][..., 0].float().cpu().numpy(), o["opacity"][..., 0].float().cpu().numpy()
            a_p, d_p = S.render_cars(trainer, data[f], rigid, [idx])
            a_v, _ = S.render_cars(trainer, data[f], rigid, None)
            amodal = a_p > 0.5
            visible = amodal & (d_p <= depth + 1.0)  # 1 m slack for blended expected depth
            Image.fromarray(rgb).save(os.path.join(cdir, f"rgb_{f:03d}.png"))
            np.savez_compressed(os.path.join(cdir, f"depth_{f:03d}.npz"), depth=depth.astype(np.float16), alpha=alpha.astype(np.float16))
            for name, m in (("partner_visible", visible), ("partner_amodal", amodal), ("vehicles", a_v > 0.5)):
                Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(cdir, f"mask_{name}_{f:03d}.png"))
            entries = car_boxes(rigid, f, idx, np.linalg.inv(cam_new[f]), K, W, H)
            boxes["frames"].setdefault(str(f), {})[f"cam{cam}"] = entries
            if (f - frames[0]) % a.viz_every == 0 or f == Tc:
                Image.fromarray(overlay(rgb, visible, depth, alpha, entries)).save(os.path.join(out_dir, "viz", f"cam{cam}_{f:03d}.png"))
        print(f"cam{cam}: {len(frames)} frames", flush=True)
    json.dump(boxes, open(os.path.join(out_dir, "boxes.json"), "w"))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
