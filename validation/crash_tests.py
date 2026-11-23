"""Compare the collision model with four real crash tests. Prints measured vs. model values.

The measured numbers were read from published reports and figures (reading errors are noted per test);
the documents themselves are not included. Vehicle dimensions and some masses are assumptions.

  1. Side impact, Gidlewski, Prochowski & Wach (2019), Transport 34(3):330-338, doi:10.3846/transport.2019.9684.
     Car A (~50 km/h) hits the left side of car B (half the speed, perpendicular) at three points L_AB.
     Values read from the regression curves of Fig. 8 (+-0.3 m/s, +-15 deg/s), averaged over 0.08-0.16 s.
     L_AB is measured rearward from B's front axle; B's center of mass is ~1.0 m behind it (Fig. 2).
     Mass 1400 kg and size 4.67 x 1.78 m (Honda Accord 2000-02) are assumptions.
  2. Rear impact, NHTSA test 131031 (report v08549R003): a 1993 Cadillac DeVille (1738.2 kg, 56.58 km/h) hits a stopped
     1998 Jeep Cherokee (1860.0 kg), 381 mm offset to the left. Speed changes integrated from the CG acceleration curves
     of Appendix B. Contact point: middle of the width overlap. DeVille 5.3 x 1.9 m is assumed.
  3. Oblique offset, NHTSA report v03949R001: a moving deformable barrier (2033 kg, 95.7 km/h, 17 deg, 45% overlap) hits a
     stopped 2001 Dodge Neon (1496 kg). Measured peak yaw rate of the Neon during the impact (OCR of a table). The sign of
     the oblique angle cannot be read from the report, so both are shown. A barrier is not a car: a weak check.
  4. Front-to-side at 120 deg, RICSAC test 1 (Jones & Baum 1978, DOT HS 805 040): Chevelle Malibu 2109 kg and Ford Pinto
     1411 kg, both 8.85 m/s, tire-road friction 0.87. Positions at impact and rest read from the table images in the
     Virtual CRASH guide (vcrashusa.com/guide-chapter12-vc3). Sizes 5.0 x 1.9 and 4.1 x 1.6 m are assumed.

  python validation/crash_tests.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from crash2real.physics import box_poly, impact, obb_corners, obb_overlap, restitution_speed_model, simulate_pair  # noqa: E402


def inertia(m, L, W):
    return m * (L ** 2 + W ** 2) / 12.0


# --------------------------------------------------------------------------------------------- 1. Gidlewski side impacts
G_LEN, G_WID, G_MASS = 4.67, 1.78, 1400.0
G_TESTS = [  # L_AB [m], V_A [m/s], measured (V_B [m/s], yaw rate [deg/s], yaw [deg], X [m], Y [m])
    (0.04, 13.6, (9.2, -143, -7.5, 0.50, 0.59)),
    (1.34, 12.7, (8.55, 156, 7.8, 0.45, 0.70)),
    (2.71, 14.7, (7.2, 260, 19.0, 0.28, 0.60)),
]


def gidlewski(L_AB, VA, dt=0.01):
    s = 1.0 - L_AB  # contact point ahead of B's center
    VB, e, I = VA / 2.0, restitution_speed_model(VA), inertia(G_MASS, G_LEN, G_WID)
    c1, c2 = np.array([-(G_WID / 2 + G_LEN / 2), s]), np.zeros(2)  # A drives +x, B drives +y; B's left side faces -x
    n, t = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    post = impact(c1, np.array([VA, 0.0]), G_MASS, I, c2, np.array([0.0, VB]), G_MASS, I, np.array([-G_WID / 2, s]), n, t, e=e, mu=0.3)[:4]
    _, _, cs2, ys2, _ = simulate_pair(c1, 0.0, (VA, 0.0), G_MASS, (G_LEN, G_WID), c2, np.pi / 2, (0.0, VB), G_MASS, (G_LEN, G_WID),
                                      n, 0.0, 20, dt_out=dt, e=e, post=post)
    tt = np.arange(len(cs2)) * dt
    w = (tt >= 0.08 - 1e-9) & (tt <= 0.16 + 1e-9)
    v, yr = np.gradient(cs2, dt, axis=0), np.degrees(np.gradient(ys2, dt))
    return (np.linalg.norm(v[w], axis=1).mean(), yr[w].mean(), np.degrees(ys2[w] - ys2[0]).mean(),
            (cs2[w, 0] - cs2[0, 0]).mean(), (cs2[w, 1] - cs2[0, 1]).mean())


# --------------------------------------------------------------------------------------------- 2. NHTSA 131031 rear impact
def nhtsa_rear():
    M1, M2, V0 = 1738.2, 1860.0, 56.58 / 3.6
    L1, W1, L2, W2, off = 5.3, 1.9, 4.15, 1.77, 0.381
    c1, c2 = np.array([-(L2 / 2 + L1 / 2), off]), np.zeros(2)
    lo, hi = max(off - W1 / 2, -W2 / 2), min(off + W1 / 2, W2 / 2)
    v1n, _, v2n, _, _ = impact(c1, np.array([V0, 0.0]), M1, inertia(M1, L1, W1), c2, np.zeros(2), M2, inertia(M2, L2, W2),
                               np.array([-L2 / 2, (lo + hi) / 2]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), e=restitution_speed_model(V0))
    return v2n[0], v1n[0] - V0


# --------------------------------------------------------------------------------------------- 3. NHTSA oblique offset
def nhtsa_oblique(sign):
    M1, M2, V = 2033.0, 1496.0, 95.7 / 3.6
    LN, WN, LM, WM, ang = 4.37, 1.70, 2.0, 1.676, np.radians(17)
    yc = WN / 2 - 0.45 * WN / 2  # middle of the 45% overlap on the Neon's left
    c1, v1 = np.array([LN / 2 + LM / 2, yc]), V * np.array([-np.cos(ang), sign * np.sin(ang)])
    _, _, _, w2, _ = impact(c1, v1, M1, inertia(M1, LM, WM), np.zeros(2), np.zeros(2), M2, inertia(M2, LN, WN),
                            np.array([LN / 2, yc]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), e=restitution_speed_model(V))
    return np.degrees(w2)


# --------------------------------------------------------------------------------------------- 4. RICSAC test 1
FT, MPH = 0.3048, 0.44704
R_M1, R_M2, R_D1, R_D2, R_V = 4650 * 0.45359, 3110 * 0.45359, (5.0, 1.9), (4.1, 1.6), 19.8 * MPH
R_C1, R_Y1 = np.array([-10.8, 1.0]) * FT, np.radians(-30.0)
R_C2, R_Y2 = np.array([0.0, -5.5]) * FT, np.radians(90.0)
R_MEAS = dict(dv=(12.2, 15.6), move=(np.hypot(-2.4 + 10.8, -1.0 - 1.0), np.hypot(7.5 - 0.0, 2.5 + 5.5)), dyaw=(28.5, 15.0))


def ricsac(normal="box"):
    """normal='box': minimum-penetration axis of the overlapping boxes (as in the generator);
    'heading': the Malibu's direction of travel, with the contact point 0.3 m ahead of the Pinto's center."""
    v1, v2 = R_V * np.array([np.cos(R_Y1), np.sin(R_Y1)]), R_V * np.array([np.cos(R_Y2), np.sin(R_Y2)])
    n, depth = obb_overlap(R_C1, R_Y1, *R_D1, R_C2, R_Y2, *R_D2)
    P = np.array(box_poly(*R_C1, R_Y1, *R_D1).intersection(box_poly(*R_C2, R_Y2, *R_D2)).centroid.coords[0])
    if normal == "heading":
        n = np.array([np.cos(R_Y1), np.sin(R_Y1)])
        k1, _ = obb_corners(R_C1, R_Y1, *R_D1)
        k2, _ = obb_corners(R_C2, R_Y2, *R_D2)
        depth = float(max(0.0, min((k1 @ n).max(), (k2 @ n).max()) - max((k1 @ n).min(), (k2 @ n).min())))
        P = np.array([R_C2[0] - R_D2[1] / 2, R_C2[1] + 0.3])
    e = restitution_speed_model(max(0.0, float(np.dot(v1 - v2, n))))
    post = impact(R_C1, v1, R_M1, inertia(R_M1, *R_D1), R_C2, v2, R_M2, inertia(R_M2, *R_D2), P, n, np.array([-n[1], n[0]]), e=e)[:4]
    cs1, ys1, cs2, ys2, _ = simulate_pair(R_C1, R_Y1, v1, R_M1, R_D1, R_C2, R_Y2, v2, R_M2, R_D2, n, depth, 60, e=e, mu_g=0.87, post=post)
    return (np.linalg.norm(post[0] - v1) / MPH, np.linalg.norm(post[2] - v2) / MPH,
            np.linalg.norm(cs1[-1] - R_C1) / FT, np.linalg.norm(cs2[-1] - R_C2) / FT,
            np.degrees(ys1[-1] - ys1[0]), np.degrees(ys2[-1] - ys2[0]))


def main():
    print("1. Side impact (Gidlewski et al. 2019): struck car B, mean over 0.08-0.16 s")
    print(f"   {'L_AB':>5} {'':9}{'speed m/s':>10}{'yaw rate':>10}{'yaw deg':>9}{'X m':>7}{'Y m':>7}")
    errs = []
    for L, VA, meas in G_TESTS:
        sim = gidlewski(L, VA)
        errs.append(np.subtract(sim, meas))
        for tag, row in (("measured", meas), ("model", sim)):
            print(f"   {L if tag == 'measured' else '':>5} {tag:9}{row[0]:10.2f}{row[1]:10.0f}{row[2]:9.1f}{row[3]:7.2f}{row[4]:7.2f}")
    print("   mean |model - measured|:", "  ".join(f"{k} {v:.2f}" for k, v in zip(("speed", "yaw rate", "yaw", "X", "Y"), np.abs(errs).mean(0))))

    d2, d1 = nhtsa_rear()
    print("\n2. Rear impact (NHTSA 131031): speed change [m/s]")
    print(f"   struck car  measured +8.64  model {d2:+.2f} ({(d2 / 8.64 - 1) * 100:+.1f}%)")
    print(f"   bullet car  measured -9.32  model {d1:+.2f} ({(abs(d1) / 9.32 - 1) * 100:+.1f}%)")

    print("\n3. Oblique offset (NHTSA v03949R001): Neon peak yaw rate [deg/s]")
    print(f"   measured +199.7 (during the impact)  model {nhtsa_oblique(+1):+.0f} / {nhtsa_oblique(-1):+.0f} (the two possible angle signs)")

    print("\n4. Front-to-side (RICSAC test 1)")
    print(f"   {'':26}{'dv Malibu':>10}{'dv Pinto':>9}{'move M ft':>10}{'move P ft':>10}{'yaw M':>7}{'yaw P':>7}")
    m = R_MEAS
    print(f"   {'measured':26}{m['dv'][0]:10.1f}{m['dv'][1]:9.1f}{m['move'][0]:10.1f}{m['move'][1]:10.1f}{m['dyaw'][0]:+7.1f}{m['dyaw'][1]:+7.1f}")
    for tag, nrm in (("model, normal from boxes", "box"), ("model, normal = heading", "heading")):
        r = ricsac(nrm)
        print(f"   {tag:26}{r[0]:10.1f}{r[1]:9.1f}{r[2]:10.1f}{r[3]:10.1f}{r[4]:+7.1f}{r[5]:+7.1f}")
    print("   (speed changes in mph; the choice of the impact normal changes dv by more than 2x; rest positions do not match)")


if __name__ == "__main__":
    main()
