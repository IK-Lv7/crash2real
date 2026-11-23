"""Build one collision case: approach -> time of impact -> impact -> post-impact motion -> ego camera motion.

`build_scenario` needs only the recorded tracks (CPU). `case_from_args` turns the shared command-line
options into the retimed ego, contact frame and frame range, so that search, generate and make_gif
all rebuild the same case from the same `cli` string.
"""
import argparse
import shlex

import numpy as np

from . import physics as P
from .physics import DT, EGO_L, EGO_OFF, EGO_W, yaw_of
from .planners import load_tracks, plan_brake, plan_cutin, plan_lane, retime_ego


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ego_after(ego, Tc, cs, ys, dv, shake_per_dv=0.0055):
    """Ego poses (n,4,4): unchanged before Tc, then the rigid post-impact motion plus a damped shake.

    shake_per_dv [rad/(m/s)] scales pitch and roll with the ego speed change. It is calibrated to the
    90th percentile of pitch/roll in real ego-involved crash videos (MM-AU). Frequencies (2.5 / 2.0 / 3.0 Hz)
    and the 0.35 s decay are assumptions."""
    out = ego.copy()
    yaw_ref = float(yaw_of(ego[Tc]))
    R_ref = ego[Tc][:3, :3]
    for t in range(Tc, len(ego)):
        i = min(t - Tc, len(cs) - 1)
        tau = (t - Tc) * DT
        Rw = _Rz(ys[i] - yaw_ref) @ R_ref
        env = np.exp(-tau / 0.35)
        pitch = shake_per_dv * dv * env * np.sin(2 * np.pi * 2.5 * tau)
        roll = 0.5 * shake_per_dv * dv * env * np.sin(2 * np.pi * 2.0 * tau + 0.6)
        bump = 0.015 * dv * env * np.sin(2 * np.pi * 3.0 * tau)
        T = np.eye(4)
        T[:3, :3] = Rw @ _Ry(pitch) @ _Rx(roll)
        T[:2, 3] = cs[i] - Rw[:2, 0] * EGO_OFF  # box center -> pose origin
        T[2, 3] = ego[Tc][2, 3] + bump
        out[t] = T
    return out


def toi_state(ego, r, Tc, size, iters=30):
    """Poses (c1, yaw1, c2, yaw2) at the instant the boxes first touch between frames Tc-1 and Tc.

    At 10 Hz a fast car is already deep inside the other box at the first overlapping frame; the
    minimum-penetration axis there can point sideways and the impulse vanishes (tunneling).
    Bisecting the time of impact avoids that."""
    def pose(a):
        e0, e1 = ego[Tc - 1], ego[Tc]
        y0, y1 = float(yaw_of(e0)), float(yaw_of(e1))
        c1 = (1 - a) * P.ego_center(e0) + a * P.ego_center(e1)
        q0, q1 = float(r["yaw"][-2]), float(r["yaw"][-1])
        c2 = np.array([(1 - a) * r["x"][-2] + a * r["x"][-1], (1 - a) * r["y"][-2] + a * r["y"][-1]])
        wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        return c1, y0 + a * wrap(y1 - y0), c2, q0 + a * wrap(q1 - q0)

    def hit(a):
        c1, y1, c2, y2 = pose(a)
        return P.obb_overlap(c1, y1, EGO_L, EGO_W, c2, y2, size[0], size[1]) is not None

    if len(r["x"]) < 2:
        return pose(1.0)
    if hit(0.0):
        return pose(0.0)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if hit(mid) else (mid, hi)
    return pose(hi)


def build_scenario(ego, part, size, t0, Tc, planner="brake", m_ego=1600.0, m_partner=1700.0, brake_a=8.0, brake_vfinal=0.0,
                   cutin_dur=20, cutin_brake_delay=None, partner_speed_scale=1.0, restitution=None, mu_contact=0.3,
                   mu_g=0.51, r_rot=1.6, engine="hybrid", verbose=False):
    """Approach, impact and post-impact motion. `ego` is the (retimed) ego; t0 is the planner start frame.

    Returns dict(full: partner x/y/yaw from t0 to the end, ego_new: ego poses with the post-impact motion,
    dv_ego, restitution, plan). engine: "hybrid" (point impulse, then pymunk) or "pymunk" (pymunk only)."""
    n_post = len(ego) - 1 - Tc
    if planner == "brake":
        r = plan_brake(ego, part, size, t0, Tc, a_brake=brake_a, v_final=brake_vfinal)
    elif planner == "cutin":
        r = plan_cutin(ego, part, size, t0, Tc, dur=cutin_dur, brake_a=brake_a, brake_delay=cutin_brake_delay)
    elif planner == "lane":
        r = plan_lane(ego, part, size, t0, Tc, v_end_scale=partner_speed_scale)
    else:
        raise ValueError(f"unknown planner {planner}")
    if P.obb_overlap(P.ego_center(ego[Tc]), float(yaw_of(ego[Tc])), EGO_L, EGO_W,
                     np.array([r["x"][-1], r["y"][-1]]), float(r["yaw"][-1]), size[0], size[1]) is None:
        raise ValueError("the ego and partner boxes do not overlap at the planned contact frame (change t0/Tc)")

    c1, yaw1, c2, yaw2 = toi_state(ego, r, Tc, size)
    r = dict(r, x=np.r_[r["x"][:-1], c2[0]], y=np.r_[r["y"][:-1], c2[1]], yaw=np.r_[r["yaw"][:-1], yaw2])
    n, depth = P.obb_overlap(c1, yaw1, EGO_L, EGO_W, c2, yaw2, size[0], size[1])
    contact = P.box_poly(*c1, yaw1, EGO_L, EGO_W).intersection(P.box_poly(c2[0], c2[1], yaw2, size[0], size[1]))
    Pc = np.array(contact.centroid.coords[0])
    tang = np.array([-n[1], n[0]])

    v1 = (ego[Tc + 1][:2, 3] - ego[Tc - 1][:2, 3]) / (2 * DT)
    v2 = r["direction"] * r["end_speed"]
    I1 = m_ego * (EGO_L**2 + EGO_W**2) / 12
    I2 = m_partner * (size[0] ** 2 + size[1] ** 2) / 12
    e = P.restitution_speed_model(max(0.0, float(np.dot(v1 - v2, n)))) if restitution is None else float(restitution)
    v1n, w1, v2n, w2, j = P.impact(c1, v1, m_ego, I1, c2, v2, m_partner, I2, Pc, n, tang, e=e, mu=mu_contact)
    cs1, ys1, cs2, ys2, dv_pm = P.simulate_pair(
        c1, yaw1, v1, m_ego, (EGO_L, EGO_W), c2, yaw2, v2, m_partner, (float(size[0]), float(size[1])),
        n, depth, n_post, e=e, mu=mu_contact, mu_g=mu_g, r_rot=r_rot,
        post=(v1n, w1, v2n, w2) if engine == "hybrid" else None)
    dv_ego = float(np.linalg.norm(v1n - v1)) if engine == "hybrid" else dv_pm
    if verbose:
        rel = np.degrees(abs((yaw2 - yaw1 + np.pi) % (2 * np.pi) - np.pi))
        print(f"impact ({planner}): ego {np.linalg.norm(v1):.1f} -> {np.linalg.norm(v1n):.1f} m/s (dv {dv_ego:.2f}), "
              f"partner {np.linalg.norm(v2):.1f} -> {np.linalg.norm(v2n):.1f} m/s, relative heading {rel:.0f} deg, restitution {e:.2f}")
        print(f"after impact: ego moves {np.linalg.norm(cs1[-1] - c1):.1f} m, turns {np.degrees(ys1[-1] - ys1[0]):+.0f} deg; "
              f"partner {np.linalg.norm(cs2[-1] - c2):.1f} m, {np.degrees(ys2[-1] - ys2[0]):+.0f} deg")
    full = dict(x=np.concatenate([r["x"], cs2[1:, 0]]), y=np.concatenate([r["y"], cs2[1:, 1]]), yaw=np.concatenate([r["yaw"], ys2[1:]]))
    return dict(plan=r, full=full, ego_new=ego_after(ego, Tc, cs1, ys1, dv_ego), dv_ego=dv_ego, restitution=e,
                impulse=float(j), dv_partner=float(np.linalg.norm(v2n - v2)))


# ---------------------------------------------------------------------------------------------
# Case options shared by search (writes them), generate and make_gif (read them)
# ---------------------------------------------------------------------------------------------
def add_case_args(ap):
    g = ap.add_argument_group("case")
    g.add_argument("--instance", required=True, help="AV2 instance id of the partner car (key of instances_info.json)")
    g.add_argument("--planner", choices=["brake", "cutin", "lane"], default="brake")
    g.add_argument("--ego_scale", type=float, default=1.0, help="ego speed factor along its recorded path (1 = as recorded)")
    g.add_argument("--ego_ramp_start", type=int, default=70, help="frame where the ego starts to speed up")
    g.add_argument("--ego_ramp", type=int, default=10, help="frames to ramp the speed factor from 1 to K")
    g.add_argument("--ego_tc_orig", type=int, default=110, help="(lane) contact where the recorded ego was at this frame")
    g.add_argument("--tc", type=int, default=None, help="contact frame")
    g.add_argument("--t0", type=int, default=None, help="planner start: braking (brake), lane change (cutin), approach (lane)")
    g.add_argument("--brake_a", type=float, default=8.0, help="partner deceleration [m/s^2] (brake, cutin)")
    g.add_argument("--brake_vfinal", type=float, default=0.0, help="partner speed after braking [m/s] (brake)")
    g.add_argument("--cutin_dur", type=int, default=20, help="frames for the lane change (cutin)")
    g.add_argument("--cutin_brake_delay", type=int, default=None, help="frames from the lane change start to braking (cutin; default cutin_dur)")
    g.add_argument("--partner_speed_scale", type=float, default=1.0, help="partner contact speed relative to the record (lane)")
    g.add_argument("--partner_donor", default=None, help="draw the partner with the Gaussians of this AV2 instance")
    g.add_argument("--m_ego", type=float, default=1600.0)
    g.add_argument("--m_partner", type=float, default=1700.0)
    g.add_argument("--f0", type=int, default=None, help="first output frame (default tc-20)")
    g.add_argument("--f1", type=int, default=None, help="last output frame (default tc+30)")
    return ap


def parse_case(cli):
    """Parse a case `cli` string (as written by search) into an argparse namespace."""
    return add_case_args(argparse.ArgumentParser()).parse_known_args(shlex.split(cli))[0]


def case_from_args(scene_dir, c, verbose=False):
    """Rebuild a case from its options. Returns dict(ego_orig, ego, part, size, Tc, t0, frames, sc)."""
    ego_orig, part, size, _ = load_tracks(scene_dir, c.instance)
    N = len(ego_orig)
    ego, Tc = ego_orig, c.ego_tc_orig
    if c.ego_scale != 1.0:
        ego, tau = retime_ego(ego_orig, c.ego_ramp_start, c.ego_scale, c.ego_ramp)
        if c.tc is None:
            if not (tau >= c.ego_tc_orig).any():
                raise SystemExit(f"the retimed ego never reaches recorded frame {c.ego_tc_orig}; pass --tc")
            Tc = int(np.argmax(tau >= c.ego_tc_orig))
    Tc = Tc if c.tc is None else c.tc
    t0 = Tc - 30 if c.t0 is None else c.t0
    f0 = Tc - 20 if c.f0 is None else c.f0
    f1 = min(N - 1, Tc + 30) if c.f1 is None else c.f1
    sc = build_scenario(ego, part, size, t0, Tc, planner=c.planner, m_ego=c.m_ego, m_partner=c.m_partner, brake_a=c.brake_a,
                        brake_vfinal=c.brake_vfinal, cutin_dur=c.cutin_dur, cutin_brake_delay=c.cutin_brake_delay,
                        partner_speed_scale=c.partner_speed_scale, verbose=verbose)
    return dict(ego_orig=ego_orig, ego=ego, part=part, size=size, Tc=Tc, t0=t0, frames=list(range(f0, f1 + 1)), sc=sc)
