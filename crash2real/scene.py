"""Load a trained drivestudio (OmniRe) scene and edit its rigid cars.

Needs the `drivestudio` environment and a drivestudio checkout (default: ./drivestudio, or $DRIVESTUDIO).
RigidNodes keeps every car's Gaussians in one array (point_ids tells which car) in the car's local
frame, plus per-frame poses (instances_trans / instances_quats). Moving a car = rewriting its poses.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVESTUDIO = os.path.abspath(os.environ.get("DRIVESTUDIO", os.path.join(REPO, "drivestudio")))
if DRIVESTUDIO not in sys.path:
    sys.path.insert(0, DRIVESTUDIO)


def scene_dir_of(cfg):
    return os.path.join(cfg.data.data_root, f"{cfg.data.scene_idx:03d}")


def load_scene(run_dir):
    """Load a trained run (outputs/av2/<name>_omnire) exactly as drivestudio's eval does."""
    from omegaconf import OmegaConf

    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    if not os.path.isabs(cfg.data.data_root):  # stored relative to the drivestudio folder
        cfg.data.data_root = os.path.join(DRIVESTUDIO, cfg.data.data_root)
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(run_dir, "checkpoint_final.pth"), load_only_model=True)
    trainer.set_eval()
    return cfg, dataset, trainer


def match_instance(rigid, scene_dir, instance):
    """AV2 instance id -> RigidNodes index, by the closest trajectory. Returns (index, mean error [m])."""
    fa = json.load(open(os.path.join(scene_dir, "instances", "instances_info.json")))[instance]["frame_annotations"]
    ego0 = np.loadtxt(os.path.join(scene_dir, "ego_pose", "000.txt"))
    fi = np.array(fa["frame_idx"])
    ref = torch.tensor((np.linalg.inv(ego0) @ np.array(fa["obj_to_world"]))[:, :3, 3], dtype=torch.float32)
    err = (rigid.instances_trans.detach().cpu()[fi] - ref[:, None, :]).norm(dim=-1).mean(0)
    k = int(err.argmin())
    return k, float(err[k])


def write_trajectory(rigid, idx, traj, t0, ego=None, own_tol=3.0):
    """Write traj (x, y, yaw from frame t0; the last pose is held) into car `idx`.

    Height and tilt come from the new position, not from the recording at the same time (a recorded car
    keeps driving, so on a slope a stopped partner would sink or float): the car's own recorded pose
    within own_tol m, else the nearest ego pose height plus the car's median height offset from the ego path."""
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
    with torch.no_grad():
        cache = rigid.__dict__.setdefault("_recorded_tracks", {})  # keep the recording if a car is written twice
        if idx not in cache:
            cache[idx] = (rigid.instances_trans[:, idx].detach().clone(), rigid.instances_quats[:, idx].detach().clone(),
                          rigid.instances_fv[:, idx].detach().clone().bool())
        trk_t, quat_t, fv_t = cache[idx]
        trk = trk_t.cpu().numpy()
        valid = np.where(fv_t.cpu().numpy() & (np.abs(trk).sum(1) > 0))[0]
        if len(valid) == 0:
            valid = np.arange(len(trk))
        R_all = quaternion_to_matrix(quat_t)
        rec_yaw = torch.atan2(R_all[:, 1, 0], R_all[:, 0, 0])
        z_ego = None
        if ego is not None:
            exy, ez = np.asarray(ego)[:, :2, 3], np.asarray(ego)[:, 2, 3]
            dd = np.linalg.norm(trk[valid, None, :2] - exy[None], axis=2)
            near, ok = dd.argmin(1), dd.min(1) < 4.0
            dz = float(np.median(trk[valid][ok, 2] - ez[near[ok]])) if ok.any() else float(rigid.instances_size[idx, 2]) / 2 - 0.3
            z_ego = lambda x, y: float(ez[np.argmin(np.linalg.norm(exy - np.array([x, y]), axis=1))] + dz)  # noqa: E731
        for t in range(t0, rigid.instances_trans.shape[0]):
            i = min(t - t0, len(traj["x"]) - 1)
            x, y = float(traj["x"][i]), float(traj["y"][i])
            dj = np.linalg.norm(trk[valid, :2] - np.array([x, y]), axis=1)
            j = int(valid[dj.argmin()])
            if dj.min() <= own_tol:
                zt, jt = float(trk[j, 2]), j
            elif z_ego is not None:
                zt, jt = z_ego(x, y), j
            else:
                zt, jt = float(trk[t, 2]), t
            rigid.instances_trans[t, idx] = torch.tensor([x, y, zt], dtype=rigid.instances_trans.dtype)
            dyaw = float(traj["yaw"][i]) - float(rec_yaw[jt])
            c, s = np.cos(dyaw), np.sin(dyaw)
            Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=R_all.dtype, device=R_all.device)
            rigid.instances_quats[t, idx] = matrix_to_quaternion(Rz @ R_all[jt])


def transplant(rigid, donor_idx, dst_idx):
    """Replace car dst's Gaussians with donor's, stretched to dst's box. dst keeps its own motion.

    Used when the partner is poorly reconstructed (seen from few directions in training).
    Note: the spherical-harmonic colors keep the donor's world view directions."""
    from torch.nn import Parameter
    if donor_idx == dst_idx:
        raise ValueError("donor and partner are the same car")
    with torch.no_grad():
        ids = rigid.point_ids[:, 0]
        src, dst = ids == donor_idx, ids == dst_idx
        if int(src.sum()) == 0:
            raise ValueError(f"donor (RigidNodes {donor_idx}) has no Gaussians")
        ratio = rigid.instances_size[dst_idx] / rigid.instances_size[donor_idx]  # box L, W, H ratio
        keep = ~dst
        new = dict(_means=rigid._means[src] * ratio,
                   _scales=rigid._scales[src] + torch.log(ratio.prod() ** (1 / 3)),
                   _quats=rigid._quats[src], _features_dc=rigid._features_dc[src],
                   _features_rest=rigid._features_rest[src], _opacities=rigid._opacities[src])
        for name, b in new.items():
            setattr(rigid, name, Parameter(torch.cat([getattr(rigid, name)[keep].detach(), b.detach().clone()], dim=0)))
        rigid.instances_fv[:, dst_idx] = 1  # draw in every frame, even if the partner was rarely observed
        new_ids = torch.full((int(src.sum()), 1), dst_idx, dtype=rigid.point_ids.dtype, device=rigid.point_ids.device)
        rigid.point_ids = torch.cat([rigid.point_ids[keep], new_ids], dim=0)
    return int(dst.sum()), int(src.sum())


def camera_path(dataset, cam, ego_orig, ego_new):
    """Camera poses that follow the edited ego: C_new = E_new @ inv(E_recorded) @ C_recorded."""
    cam_orig = dataset.pixel_source.camera_data[cam].cam_to_worlds.clone().numpy()
    return np.array([ego_new[t] @ np.linalg.inv(ego_orig[t]) @ cam_orig[t] for t in range(len(ego_orig))])


def render(trainer, item):
    """Render one prepared view (an item of dataset.prepare_novel_view_render_data). The item is copied."""
    d = {g: {k: v.clone().cuda() for k, v in item[g].items()} for g in ("cam_infos", "image_infos")}
    with torch.no_grad():
        return trainer(image_infos=d["image_infos"], camera_infos=d["cam_infos"], novel_view=True)


class _NullModel(torch.nn.Module):
    def get_gaussians(self, cam):
        return None


def render_cars(trainer, item, rigid, keep=None):
    """Render only RigidNodes (only the cars in `keep`, or all). Returns (opacity, depth) as numpy arrays.

    Other node classes are swapped for a model with no Gaussians; gaussian_classes itself is left alone
    because the renderer later looks up "Background"."""
    others = {k: trainer.models[k] for k in trainer.gaussian_classes if k != "RigidNodes"}
    fv_bak = rigid.instances_fv
    try:
        for k in others:
            trainer.models[k] = _NullModel()
        if keep is not None:
            fv = torch.zeros_like(fv_bak)
            for k in keep:
                fv[:, k] = fv_bak[:, k]
            rigid.instances_fv = fv
        o = render(trainer, item)
    finally:
        for k, m in others.items():
            trainer.models[k] = m
        rigid.instances_fv = fv_bak
    return o["opacity"][..., 0].float().cpu().numpy(), o["depth"][..., 0].float().cpu().numpy()
