"""Trajectories before the impact: the ego is retimed on its recorded path, the partner follows its recorded path.

Three planners for the partner car (all return the same dict keys):
  brake: the car ahead brakes hard from frame t_b (rear-end; with crossing cars also angle impacts)
  cutin: a car in the adjacent lane moves onto the ego path from frame t_s
  lane:  the partner stays on its recorded lane, retimed so that it first touches the ego at Tc (sideswipe)

Scene coordinates: the drivestudio world frame (first ego pose is the origin, z up), 10 Hz frames.
No model is needed; everything here runs on CPU from the preprocessed scene folder.
"""
import json
import os

import numpy as np

from .physics import DT, box_poly, ego_poly


def load_ego(scene_dir):
    """Ego poses (n,4,4) in the scene world frame, and the inverse of the first city-frame pose."""
    n = len(os.listdir(os.path.join(scene_dir, "ego_pose")))
    ego_city = np.array([np.loadtxt(os.path.join(scene_dir, "ego_pose", f"{t:03d}.txt")) for t in range(n)])
    inv0 = np.linalg.inv(ego_city[0])
    return inv0 @ ego_city, inv0


def load_tracks(scene_dir, instance):
    """Ego poses, the partner's poses (NaN where unannotated), its mean box size (L, W, H) and annotated frames."""
    ego, inv0 = load_ego(scene_dir)
    fa = json.load(open(os.path.join(scene_dir, "instances", "instances_info.json")))[instance]["frame_annotations"]
    fi = np.array(fa["frame_idx"])
    part = np.full((len(ego), 4, 4), np.nan)
    part[fi] = inv0 @ np.array(fa["obj_to_world"])
    return ego, part, np.array(fa["box_size"]).mean(0), fi


def load_all(scene_dir):
    """Ego poses and every annotated object: {instance: dict(cls, poses, size, id)}."""
    ego, inv0 = load_ego(scene_dir)
    info = json.load(open(os.path.join(scene_dir, "instances", "instances_info.json")))
    inst = {}
    for k, v in info.items():
        fa = v["frame_annotations"]
        poses = np.full((len(ego), 4, 4), np.nan)
        poses[np.array(fa["frame_idx"])] = inv0 @ np.array(fa["obj_to_world"])
        inst[k] = dict(cls=v["class_name"], poses=poses, size=np.array(fa["box_size"]).mean(0), id=v["id"])
    return ego, inst


def retime_ego(ego, ramp_start, K, ramp=10):
    """Drive the ego faster (K>1) along its recorded path. Scene time t maps to path time tau(t).

    tau(t)=t until ramp_start, then the speed factor rises from 1 to K over `ramp` frames.
    Translation is interpolated linearly, rotation by slerp. Other cars keep the recorded scene time."""
    from scipy.spatial.transform import Rotation, Slerp
    N = len(ego)
    tau = np.zeros(N)
    for t in range(1, N):
        k = 1.0 if t <= ramp_start else 1.0 + (K - 1.0) * min(1.0, (t - ramp_start) / ramp)
        tau[t] = min(N - 1.0, tau[t - 1] + k)
    out = np.empty_like(ego)
    for t in range(N):
        i = min(int(np.floor(tau[t])), N - 2)
        a = tau[t] - i
        T = np.eye(4)
        T[:3, 3] = (1 - a) * ego[i][:3, 3] + a * ego[i + 1][:3, 3]
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([ego[i][:3, :3], ego[i + 1][:3, :3]]))([a]).as_matrix()[0]
        out[t] = T
    return out, tau


def quintic(p0, v0, a0, p1, v1, a1, T):
    """Coefficients of p(t), 0<=t<=T, matching position, velocity and acceleration at both ends."""
    A = np.array([
        [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5], [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4], [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ], dtype=float)
    return np.linalg.solve(A, np.array([p0, v0, a0, p1, v1, a1], dtype=float))


def eval_poly(c, t):
    return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]*t**4 + c[5]*t**5


def _smooth(a, k=5):
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], np.ones(k) / k, mode="valid") for i in range(a.shape[1])], 1)


class RecordedPath:
    """The partner's recorded track as an arc-length path: position, heading, normal and recorded speed."""

    def __init__(self, part, step=0.05):
        fi = np.where(~np.isnan(part[:, 0, 3]))[0]
        xy = _smooth(part[fi, :2, 3], 5)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        self.fi, self.s_f = fi, np.concatenate([[0.0], np.cumsum(seg)])
        self.v_f = np.gradient(self.s_f, fi * DT)
        self.S = np.arange(0.0, self.s_f[-1], step)
        self.P = np.stack([np.interp(self.S, self.s_f, xy[:, 0]), np.interp(self.S, self.s_f, xy[:, 1])], 1)
        h = np.unwrap(np.arctan2(*np.gradient(self.P, axis=0).T[::-1]))
        k = max(3, int(1.0 / step) | 1)  # smooth the heading over about 1 m
        self.h = np.convolve(np.pad(h, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        self.step = step

    def pos(self, s):
        return np.array([np.interp(s, self.S, self.P[:, 0]), np.interp(s, self.S, self.P[:, 1])])

    def heading(self, s):
        return float(np.interp(s, self.S, self.h))

    def normal(self, s):  # left of the direction of travel
        h = self.heading(s)
        return np.array([-np.sin(h), np.cos(h)])

    def s_at_frame(self, t):
        return float(np.interp(t, self.fi, self.s_f))

    def v_at_s(self, s):
        return float(np.interp(s, self.s_f, self.v_f))


def _kinematics(xy, yaw):
    vel = np.gradient(xy, DT, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    acc = np.gradient(vel, DT, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    sp = np.linalg.norm(vel, axis=1)
    lat = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / np.maximum(sp, 0.5)
    return sp, lat, np.linalg.norm(acc, axis=1)


def _result(xy, yaw, **extra):
    sp, lat, acc = _kinematics(xy, yaw)
    d = np.array([np.cos(yaw[-1]), np.sin(yaw[-1])])
    return dict(x=xy[:, 0], y=xy[:, 1], yaw=yaw, speed=sp, lat_acc=lat, acc=acc, direction=d, end_speed=float(sp[-1]), **extra)


def partner_profile(rp, N, t_b, a_brake, v_final):
    """Arc length S(t) of a car that follows its recorded timing until t_b, then decelerates at a_brake to v_final."""
    f0, f1 = int(rp.fi[0]), int(rp.fi[-1])
    S = np.array([rp.s_at_frame(min(max(t, f0), f1)) for t in range(N)])
    tb = min(max(t_b, f0), f1)
    sb = rp.s_at_frame(tb)
    vb = max(float(np.interp(tb, rp.fi, rp.v_f)), 0.0)
    for t in range(t_b, N):
        tau = (t - t_b) * DT
        if vb <= v_final:
            s = sb + vb * tau
        else:
            ts = (vb - v_final) / a_brake
            s = sb + vb * tau - 0.5 * a_brake * tau**2 if tau <= ts else sb + vb * ts - 0.5 * a_brake * ts**2 + v_final * (tau - ts)
        S[t] = min(s, rp.S[-1] - 0.01)
    return S, vb


def plan_brake(ego, part, size, t_b, Tc, a_brake=8.0, v_final=0.0):
    """The partner brakes from t_b along its recorded path. Trajectory for frames t_b..Tc."""
    rp = RecordedPath(part)
    S, vb = partner_profile(rp, len(ego), t_b, a_brake, v_final)
    ts = np.arange(t_b, Tc + 1)
    xy = np.array([rp.pos(S[t]) for t in ts])
    yaw = np.unwrap([rp.heading(S[t]) for t in ts])
    return _result(xy, yaw, v0=float(vb))


def ego_lateral(ego_xy, p, n):
    """Closest point q on the ego path polyline to p, and the signed distance from p to q along n."""
    a, b = ego_xy[:-1], ego_xy[1:]
    ab = b - a
    u = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    q = a + u[:, None] * ab
    k = int(np.argmin(np.linalg.norm(q - p, axis=1)))
    return q[k], float(np.dot(q[k] - p, n))


def plan_cutin(ego, part, size, t_s, Tc, dur=20, brake_a=0.0, brake_delay=None):
    """Cut-in from the adjacent lane. The partner keeps its recorded speed along its path and, from t_s,
    moves sideways onto the ego path over `dur` frames (smoothstep). If brake_a>0 it then brakes hard
    from t_s+brake_delay (default: when the lane change ends). The ego keeps its recorded drive."""
    rp = RecordedPath(part)
    N = len(ego)
    if brake_a > 0:
        S = partner_profile(rp, N, t_s + (dur if brake_delay is None else brake_delay), brake_a, 0.0)[0]
    else:
        f0, f1 = int(rp.fi[0]), int(rp.fi[-1])
        S = np.array([rp.s_at_frame(min(max(t, f0), f1)) for t in range(N)])
    ego_xy = ego[:, :2, 3]
    ts = np.arange(t_s, Tc + 1)
    u = np.clip((ts - t_s) / float(dur), 0, 1)
    w = u * u * (3 - 2 * u)
    xy = []
    for t, wi in zip(ts, w):
        p, n = rp.pos(S[t]), rp.normal(S[t])
        _, d = ego_lateral(ego_xy, p, n)
        xy.append(p + wi * d * n)
    xy = np.array(xy)
    sp = _kinematics(xy, None)[0]
    vel = np.gradient(xy, DT, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    hd = np.array([rp.heading(S[t]) for t in ts])
    yaw = np.unwrap(np.where(sp > 0.5, np.arctan2(vel[:, 1], vel[:, 0]), hd))
    return _result(xy, yaw, v0=float(sp[0]))


def plan_lane(ego, part, size, t0, Tc, blend=30, delta_max=2.0, v_end=None, v_end_scale=1.0):
    """Keep the partner on its recorded lane and retime it (quintic in arc length) so that it first touches
    the ego box at Tc. A lateral offset delta (smallest needed, up to +-delta_max) is blended in over the
    last `blend` frames; its derivative is zero at Tc, so the contact heading is the lane tangent."""
    rp = RecordedPath(part)
    ego_box = ego_poly(ego[Tc])
    s0 = rp.s_at_frame(t0)
    L, W = size[0], size[1]

    def partner_at(S, delta):
        c = rp.pos(S) + delta * rp.normal(S)
        return box_poly(c[0], c[1], rp.heading(S), L, W)

    found = None  # smallest offset first: the first arc length where the box enters the ego box from outside
    for delta in sorted(np.arange(-delta_max, delta_max + 1e-9, 0.25), key=abs):
        prev_hit = None
        for S in np.arange(s0 + 2.0, rp.S[-1] - 1.0, rp.step * 2):
            if ego_box.intersection(partner_at(S, delta)).area > 1e-3:
                if prev_hit is False:
                    found = (delta, S)
                break
            prev_hit = False
        if found:
            break
    if not found:
        raise ValueError("no contact with the ego box along the recorded path (change Tc or delta_max)")
    delta_end, s_c = found

    T = (Tc - t0) * DT
    v0 = max(0.5, rp.v_at_s(s0))
    if v_end is None:
        v_end = float(np.clip(np.mean([rp.v_at_s(s_c + d) for d in (-1.0, 0.0, 1.0)]) * v_end_scale, 1.0, 8.0))
    c = quintic(s0, v0, 0.0, s_c, v_end, 0.0, T)
    ts = np.arange(0, Tc - t0 + 1) * DT
    s = eval_poly(c, ts)
    u = np.clip((ts - (T - blend * DT)) / (blend * DT), 0, 1)
    w = u * u * (3 - 2 * u)
    xy = np.array([rp.pos(si) + delta_end * wi * rp.normal(si) for si, wi in zip(s, w)])
    vel = np.gradient(xy, DT, axis=0)
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    yaw[-1] = rp.heading(s_c) + 2 * np.pi * np.round((yaw[-2] - rp.heading(s_c)) / (2 * np.pi))  # lane tangent at contact
    return _result(xy, yaw, v0=float(v0))
