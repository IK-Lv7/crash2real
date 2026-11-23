"""Find collision cases in a reconstructed scene from the recorded tracks only (CPU). Writes cases JSON for run_scene.sh.

  rear:  a car ahead of the ego brakes hard while the ego drives faster on its recorded path
         (speed factor K). Gives rear-end impacts, and angle/oblique ones with crossing cars.
  cutin: a car in the adjacent lane moves onto the ego path; the ego keeps its recorded drive.

Kept only if: the boxes first touch from outside, neither car passes through another recorded car before
the impact, the ego speed change dv >= --min_dv, and (cut-in) the partner is ahead of the ego center so
the trained cameras can see it. A diverse subset is chosen, and a donor appearance is picked per partner.

  python -m crash2real.search rear  --scene_idx 9 --run outputs/av2/<name>_omnire --out outputs/scene9/cases_rear.json
  python -m crash2real.search cutin --scene_idx 9 --run outputs/av2/<name>_omnire --out outputs/scene9/cases_cutin.json
"""
import argparse
import collections
import itertools
import json
import os

import numpy as np

from .physics import DT, EGO_L, EGO_W, box_poly, ego_center, ego_poly, obb_overlap, yaw_of
from .planners import RecordedPath, ego_lateral, load_all, load_tracks, partner_profile, plan_cutin, retime_ego
from .scenario import build_scenario, case_from_args, parse_case

VEHICLE = {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "BUS", "VEHICULAR_TRAILER", "TRUCK_CAB", "ARTICULATED_BUS", "SCHOOL_BUS"}
VEH_PREFIX = ("REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "VEHICULAR_TRAILER", "BUS")
REAR_KINDS = {"rear_end_ego_hits": "r", "angle_tbone": "t", "oblique": "o", "sideswipe_merge": "s"}  # head-on and rear-ended are not used
CUTIN_KINDS = {"cutin_front": "f", "cutin_side": "s"}


def classify(ego_T, p_T):
    """Impact type from the relative heading [deg], and how far the partner is ahead of the ego [m]."""
    rel = np.degrees(abs((float(yaw_of(p_T)) - float(yaw_of(ego_T)) + np.pi) % (2 * np.pi) - np.pi))
    ahead = float(np.dot(p_T[:2, 3] - ego_T[:2, 3], ego_T[:2, 0]))
    if rel < 25:
        kind = "rear_end_ego_hits" if ahead > 0 else "rear_ended_by_other"
    elif rel < 65:
        kind = "sideswipe_merge"
    elif rel < 115:
        kind = "angle_tbone"
    elif rel > 155:
        kind = "head_on"
    else:
        kind = "oblique"
    return kind, rel, ahead


def pose2d(x, y, yaw):
    T = np.eye(4)
    T[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T[:2, 3] = [x, y]
    return T


def worst_overlap(box, others, t, z):
    """Deepest penetration of box=(center, yaw, L, W) into any recorded car at frame t (cars within 2 m in height)."""
    c, yaw, L, W = box
    worst = 0.0
    for o in others.values():
        P = o["poses"][t]
        if np.isnan(P[0, 3]) or abs(P[2, 3] - z) > 2.0:
            continue
        r = obb_overlap(np.asarray(c), yaw, L, W, P[:2, 3], float(yaw_of(P)), o["size"][0], o["size"][1])
        if r is not None:
            worst = max(worst, float(r[1]))
    return worst


# ---------------------------------------------------------------------------------------------
# donor appearance: the best-reconstructed car of a similar size
# ---------------------------------------------------------------------------------------------
def rigid_table(run, scene):
    """Per RigidNodes car: Gaussian count, 90th percentile opacity, box size, and its AV2 id (trajectory error < 0.5 m)."""
    import torch
    ck = torch.load(os.path.join(run, "checkpoint_final.pth"), map_location="cpu", weights_only=False)["models"]["RigidNodes"]
    ids = ck["points_ids"][:, 0]
    op = torch.sigmoid(ck["_opacities"][:, 0])
    trans = ck["instances_trans"].float()
    _, inst = load_all(scene)
    rows = {}
    for k in range(ck["instances_size"].shape[0]):
        s = ids == k
        rows[k] = dict(idx=k, points=int(s.sum()), op90=float(op[s].quantile(0.9)) if s.any() else 0.0,
                       size=ck["instances_size"][k].tolist(), av2=None, err=None)
    for name, d in inst.items():
        fi = np.where(~np.isnan(d["poses"][:, 0, 3]))[0]
        if not d["cls"].startswith(VEH_PREFIX) or len(fi) == 0:
            continue
        err = (trans[fi] - torch.tensor(d["poses"][fi, :3, 3], dtype=torch.float32)[:, None, :]).norm(dim=-1).mean(0)
        k = int(err.argmin())
        if float(err[k]) < 0.5 and (rows[k]["err"] is None or float(err[k]) < rows[k]["err"]):
            rows[k].update(av2=name, err=float(err[k]))
    return rows


def pick_donor(table, partner, tol=1.35, min_op90=0.5):
    """Most Gaussians among opaque cars (op90 >= 0.5; see-through cars sit at 0.2-0.36) whose box is within 1/tol..tol.
    None if the partner itself is the best."""
    me = next((r for r in table.values() if r["av2"] == partner), None)
    if me is None:
        return None
    tgt = np.array(me["size"])
    cand = [r for r in table.values() if r["av2"] is not None and r["op90"] >= min_op90
            and not ((tgt / np.array(r["size"]) > tol).any() or (tgt / np.array(r["size"]) < 1 / tol).any())]
    if not cand:
        return None
    best = max(cand, key=lambda r: r["points"])
    return None if best["av2"] == partner else best["av2"]


# ---------------------------------------------------------------------------------------------
# rear: the car ahead brakes
# ---------------------------------------------------------------------------------------------
def lead_vehicles(ego, inst, min_frames=10, min_move=8.0):
    """Cars 3-40 m ahead of the ego within 3.5 m laterally for >= min_frames frames, moving more than min_move m."""
    out = []
    for k, d in inst.items():
        if not d["cls"].startswith(VEH_PREFIX):
            continue
        ok = ~np.isnan(d["poses"][:, 0, 3])
        rel = np.einsum("tij,tj->ti", np.linalg.inv(ego[ok]), np.c_[d["poses"][ok, :3, 3], np.ones(ok.sum())])
        ahead = (rel[:, 0] > 3) & (rel[:, 0] < 40) & (np.abs(rel[:, 1]) < 3.5)
        p = d["poses"][ok, :2, 3]
        move = float(np.linalg.norm(p[-1] - p[0])) if len(p) else 0.0
        if ahead.sum() >= min_frames and move > min_move:
            out.append((k, int(ahead.sum())))
    return [k for k, _ in sorted(out, key=lambda r: -r[1])]


def brake_contacts(ego0, inst, instances):
    """All (K, ramp, brake start, deceleration) where the ego first hits a braking lead car from outside."""
    N = len(ego0)
    egos = {}
    for K, rs in itertools.product((1.0, 1.5, 2.0, 2.5, 3.0), (20, 40, 60, 70, 80)):
        if K == 1.0 and rs != 20:
            continue
        egos[(K, rs)] = ego0 if K == 1.0 else retime_ego(ego0, rs, K, 10)[0]
    rows = []
    for k in instances:
        d = inst[k]
        rp = RecordedPath(d["poses"])
        L, W = d["size"][0], d["size"][1]
        for (K, rs), ego in egos.items():
            ev = np.linalg.norm(np.gradient(ego[:, :2, 3], DT, axis=0), axis=1)
            for t_b, a_brake in itertools.product(range(10, 140, 10), (5.0, 8.0)):
                S, _ = partner_profile(rp, N, t_b, a_brake, 0.0)
                prev, hit = False, None
                for t in range(max(t_b - 20, 1), N - 31):
                    c = rp.pos(S[t])
                    if np.hypot(*(c - ego[t, :2, 3])) > 8.0:
                        prev = False
                        continue
                    ov = ego_poly(ego[t]).intersection(box_poly(c[0], c[1], rp.heading(S[t]), L, W)).area > 1e-3
                    if ov and not prev:
                        hit = t
                        break
                    prev = ov
                if hit is None or ev[hit] > 18.0 or t_b >= hit:  # ego at most 65 km/h
                    continue
                pv = float((S[min(hit + 1, N - 1)] - S[max(hit - 1, 0)]) / (2 * DT))
                kind, rel, _ = classify(ego[hit], pose2d(*rp.pos(S[hit]), rp.heading(S[hit])))
                rows.append(dict(instance=k, K=K, ramp_start=rs, t_b=t_b, a_brake=a_brake, v_final=0.0, Tc=hit, kind=kind,
                                 rel_deg=round(rel, 1), closing=round(float(ev[hit]) - pv, 1)))
    return rows


def passes_through(scene_dir, inst, r, cache, max_depth=0.1):
    """True if the partner or the retimed ego penetrates another recorded car by more than max_depth before the impact."""
    k = r["instance"]
    if k not in cache:
        cache[k] = load_tracks(scene_dir, k)
    ego_orig, part, size, _ = cache[k]
    ego = retime_ego(ego_orig, r["ramp_start"], r["K"], 10)[0] if r["K"] != 1.0 else ego_orig
    try:
        sc = build_scenario(ego, part, size, r["t_b"], r["Tc"], planner="brake", brake_a=r["a_brake"], brake_vfinal=r["v_final"])
    except Exception:  # noqa: BLE001  (no contact at the planned frame)
        return True
    full, en = sc["full"], sc["ego_new"]
    others = {kk: v for kk, v in inst.items() if kk != k and v["size"][0] > 2.5}
    for t in range(r["Tc"]):
        if t >= r["t_b"]:
            i = min(t - r["t_b"], len(full["x"]) - 1)
            if worst_overlap((np.array([full["x"][i], full["y"][i]]), full["yaw"][i], size[0], size[1]), others, t, en[t][2, 3]) > max_depth:
                return True
        if worst_overlap((ego_center(en[t]), float(yaw_of(en[t])), EGO_L, EGO_W), others, t, en[t][2, 3]) > max_depth:
            return True
    return False


def farthest(rows, n):
    """Farthest-point order over (closing speed, relative heading, K, partner) for diversity."""
    insts = sorted({r["instance"] for r in rows})
    F = np.array([[r["closing"] / 4.0, r["rel_deg"] / 8.0, r["K"], 2.0 * insts.index(r["instance"])] for r in rows])
    chosen = [int(np.argmax([r["closing"] for r in rows]))]
    while len(chosen) < min(n, len(rows)):
        d = np.min(np.linalg.norm(F[:, None, :] - F[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return chosen


def dv_and_side(scene_dir, cli):
    c = case_from_args(scene_dir, parse_case(cli))
    e, full = c["ego"][c["Tc"]], c["sc"]["full"]
    i = min(c["Tc"] - c["t0"], len(full["x"]) - 1)
    lat = float(np.dot(np.array([full["x"][i], full["y"][i]]) - e[:2, 3], e[:2, 1]))  # partner left of the ego is positive
    return float(c["sc"]["dv_ego"]), lat


def search_rear(scene_dir, run, n=8, min_dv=1.0):
    ego0, inst = load_all(scene_dir)
    leads = lead_vehicles(ego0, inst)
    print("lead cars:", leads)
    rows = brake_contacts(ego0, inst, leads)
    print("contacts by type:", dict(collections.Counter(r["kind"] for r in rows)))
    cache = {}
    rows = [r for r in rows if r["kind"] in REAR_KINDS and not passes_through(scene_dir, inst, r, cache)]
    print("without pass-through:", dict(collections.Counter(r["kind"] for r in rows)))
    keep = []
    for j in (farthest(rows, len(rows)) if rows else []):
        r = dict(rows[j])
        r["cli"] = (f"--instance {r['instance']} --ego_scale {r['K']} --ego_ramp_start {r['ramp_start']} --tc {r['Tc']} --t0 {r['t_b']} "
                    f"--planner brake --brake_a {r['a_brake']} --brake_vfinal {r['v_final']}")
        try:
            dv, lat = dv_and_side(scene_dir, r["cli"])
        except Exception:  # noqa: BLE001
            continue
        if dv >= min_dv and not any(q["instance"] == r["instance"] and abs(q["closing"] - r["closing"]) < 0.5 and q["K"] == r["K"] for q in keep):
            keep.append(dict(r, dv=round(dv, 2), partner_lat=round(lat, 2)))
        if len(keep) >= n:
            break
    table = rigid_table(run, scene_dir)
    donors = {p: pick_donor(table, p) for p in {r["instance"] for r in keep}}
    out = []
    for i, r in enumerate(sorted(keep, key=lambda r: -r["dv"])):
        d = donors[r["instance"]]
        r["name"] = f"a{i:02d}{REAR_KINDS[r['kind']]}_i{r['instance']}_K{r['K']:g}_tb{r['t_b']}_a{r['a_brake']:g}" + (f"_d{d}" if d else "")
        r["cli"] += f" --partner_donor {d}" if d else ""
        out.append(dict(r, side="L" if r["partner_lat"] >= 0 else "R"))
    return out


# ---------------------------------------------------------------------------------------------
# cutin: a car in the adjacent lane cuts in
# ---------------------------------------------------------------------------------------------
def adjacent_frames(ego, part):
    """Frames where the partner is in the adjacent lane: heading within 30 deg, 2-6 m beside the ego path, within 25 m ahead/behind."""
    out = []
    for t in np.where(~np.isnan(part[:, 0, 3]))[0]:
        p = part[t, :2, 3]
        rel = (float(yaw_of(part[t])) - float(yaw_of(ego[t])) + np.pi) % (2 * np.pi) - np.pi
        lon = float(np.dot(p - ego[t, :2, 3], ego[t, :2, 0]))
        _, lat = ego_lateral(ego[:, :2, 3], p, ego[t, :2, 1])
        if abs(np.degrees(rel)) < 30 and 2.0 <= abs(lat) <= 6.0 and abs(lon) <= 25:
            out.append(int(t))
    return out


def search_cutin(scene_dir, run, n=8, min_dv=1.0, per_partner=3, max_lat=4.5, max_lead=4.0):
    ego_c, inst = load_all(scene_dir)
    N = len(ego_c)
    rows = []
    for k, d in inst.items():
        if d["cls"] not in VEHICLE or not (3.0 <= d["size"][0] <= 6.5):  # cars only: truck beds break up at close range
            continue
        fi = np.where(~np.isnan(d["poses"][:, 0, 3]))[0]
        if len(fi) < 20 or np.ptp(d["poses"][fi, :2, 3], axis=0).max() < 8:
            continue
        ego, part, size, _ = load_tracks(scene_dir, k)
        adj = adjacent_frames(ego, part)
        if not adj:
            continue
        others = {j: o for j, o in inst.items() if j != k}
        t_end = min(int(fi[-1]), N - 31)
        combos = [(ts, du, ba, bd) for ts, du, ba in itertools.product(sorted(set(t - t % 5 for t in adj)), (10, 15, 20, 30), (0.0, 5.0, 8.0))
                  for bd in ((None,) if ba == 0 else (du, du // 2))]
        for t_s, dur, ba, bdel in combos:
            if not {t_s, t_s + 1, t_s + 2} & set(adj) or t_end - t_s < 5:
                continue
            try:
                r = plan_cutin(ego, part, size, t_s, t_end, dur=dur, brake_a=ba, brake_delay=bdel)
            except Exception:  # noqa: BLE001
                continue
            Tc = None
            for i, t in enumerate(range(t_s, t_end + 1)):
                if ego_poly(ego[t]).intersection(box_poly(r["x"][i], r["y"][i], r["yaw"][i], size[0], size[1])).area > 1e-3:
                    Tc = t if i > 0 else None
                    break
            if Tc is None or Tc - t_s < 5 or (Tc - t_s) * DT > max_lead:
                continue
            m = Tc - t_s + 1
            if r["lat_acc"][:m].max() > max_lat:
                continue
            z = float(part[fi[np.argmin(abs(fi - Tc))], 2, 3])
            if max(worst_overlap(((r["x"][i], r["y"][i]), float(r["yaw"][i]), size[0], size[1]), others, t_s + i, z) for i in range(m - 1)) > 0.1:
                continue
            _, rel, ahead = classify(ego[Tc], pose2d(r["x"][m - 1], r["y"][m - 1], r["yaw"][m - 1]))
            if ahead < 0:  # behind the ego center: hits the rear side, which the trained cameras do not see
                continue
            kind = "cutin_front" if ahead >= (EGO_L + size[0]) / 2 - 0.5 else "cutin_side"
            cli = f"--instance {k} --tc {Tc} --t0 {t_s} --planner cutin --cutin_dur {dur} --brake_a {ba:g}" + \
                (f" --cutin_brake_delay {bdel}" if bdel is not None and bdel != dur else "")
            try:
                dv = float(build_scenario(ego, part, size, t_s, Tc, planner="cutin", brake_a=ba, cutin_dur=dur, cutin_brake_delay=bdel)["dv_ego"])
            except Exception:  # noqa: BLE001
                continue
            e = ego[Tc]
            plat = float(np.dot(np.array([r["x"][m - 1], r["y"][m - 1]]) - e[:2, 3], e[:2, 1]))
            rows.append(dict(instance=k, t_s=int(t_s), dur=dur, brake_a=ba, brake_delay=bdel, Tc=int(Tc), kind=kind, rel_deg=round(rel, 1),
                             lead_time=round((Tc - t_s) * DT, 1), dv=round(dv, 2), partner_lat=round(plat, 2), cli=cli))
    ok = [r for r in rows if r["dv"] >= min_dv]
    print(f"cut-ins: {len(rows)}, dv >= {min_dv}: {len(ok)}, by type {dict(collections.Counter(r['kind'] for r in ok))}")
    by_kind = {k: sorted([r for r in ok if r["kind"] == k], key=lambda r: -r["dv"]) for k in CUTIN_KINDS}
    keep = []
    for r in (r for grp in itertools.zip_longest(*by_kind.values()) for r in grp if r is not None):  # alternate the types
        if any(q["instance"] == r["instance"] and (abs(q["t_s"] - r["t_s"]) < 15 or q["Tc"] == r["Tc"]) for q in keep) \
                or sum(q["instance"] == r["instance"] for q in keep) >= per_partner:
            continue
        keep.append(r)
        if len(keep) >= n:
            break
    table = rigid_table(run, scene_dir) if keep else {}
    donors = {p: pick_donor(table, p) for p in {r["instance"] for r in keep}}
    out = []
    for j, r in enumerate(keep):
        d = donors[r["instance"]]
        name = f"c{j:02d}{CUTIN_KINDS[r['kind']]}_i{r['instance']}_ts{r['t_s']}_d{r['dur']}_b{r['brake_a']:g}"
        name += (f"_bd{r['brake_delay']}" if r["brake_delay"] not in (None, r["dur"]) else "") + (f"_d{d}" if d else "")
        out.append(dict(r, name=name, side="L" if r["partner_lat"] >= 0 else "R", cli=r["cli"] + (f" --partner_donor {d}" if d else "")))
    return out


def main():
    from .scene import DRIVESTUDIO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["rear", "cutin"])
    ap.add_argument("--scene_idx", type=int, required=True, help="line number - 1 in configs/av2_logs.txt")
    ap.add_argument("--run", required=True, help="trained scene (for the donor choice)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min_dv", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scene_dir = os.path.join(DRIVESTUDIO, "data/argoverse/processed/training", f"{a.scene_idx:03d}")
    cases = (search_rear if a.kind == "rear" else search_cutin)(scene_dir, a.run, n=a.n, min_dv=a.min_dv)
    for r in cases:
        print(f"  {r['name']:42s} {r['kind']:18s} Tc {r['Tc']:4d}  dv {r['dv']:5.2f} m/s  partner on the {'left' if r['side'] == 'L' else 'right'}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(scene_idx=a.scene_idx, run=a.run, cases=cases), open(a.out, "w"), indent=1)
    print("saved", len(cases), "cases ->", a.out)


if __name__ == "__main__":
    main()
