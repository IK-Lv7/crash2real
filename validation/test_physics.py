"""Self-checks of the collision physics (numpy + pymunk): conservation laws, analytic cases, friction cone,
symmetry and no penetration. They check that the implementation obeys mechanics, not that the parameters
match real cars (see crash_tests.py for that).

  python validation/test_physics.py     (or pytest validation/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from crash2real.physics import cross2, impact, obb_overlap, restitution_speed_model, simulate_pair  # noqa: E402

RNG = np.random.default_rng(0)


def config(rng):
    """Random setup: centers, contact point P, normal n (1->2), tangent t, mass, inertia, closing velocity."""
    ang = rng.uniform(-np.pi, np.pi)
    n = np.array([np.cos(ang), np.sin(ang)])
    t = np.array([-n[1], n[0]])
    m1, m2 = rng.uniform(1000, 2500, 2)
    I1, I2 = m1 * rng.uniform(1.5, 3.5), m2 * rng.uniform(1.5, 3.5)
    c1 = np.zeros(2)
    c2 = n * rng.uniform(3, 5) + t * rng.uniform(-1, 1)
    P = c1 + n * rng.uniform(1.5, 2.5) + t * rng.uniform(-0.9, 0.9)
    v1 = n * rng.uniform(3, 20) + t * rng.uniform(-3, 3)
    v2 = n * rng.uniform(-3, 3) + t * rng.uniform(-3, 3)
    if np.dot(v1 - v2, n) <= 0.5:
        v1 = v1 + n * 5
    return c1, v1, m1, I1, c2, v2, m2, I2, P, n, t


def test_momentum_conserved():
    for _ in range(200):
        c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
        e, mu = RNG.uniform(0, 1), RNG.uniform(0, 0.8)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=e, mu=mu)
        p0, p1 = m1 * v1 + m2 * v2, m1 * v1n + m2 * v2n
        assert np.allclose(p0, p1, rtol=1e-9, atol=1e-6), (p0, p1)


def test_angular_momentum_about_contact_point_conserved():
    for _ in range(200):
        c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
        e, mu = RNG.uniform(0, 1), RNG.uniform(0, 0.8)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=e, mu=mu)
        L0 = m1 * cross2(c1 - P, v1) + m2 * cross2(c2 - P, v2)  # no spin before the impact
        L1 = m1 * cross2(c1 - P, v1n) + I1 * w1 + m2 * cross2(c2 - P, v2n) + I2 * w2
        assert abs(L0 - L1) <= 1e-6 * max(1.0, abs(L0)) + 1e-3, (L0, L1)


def ke(v1, w1, m1, I1, v2, w2, m2, I2):
    return 0.5 * (m1 * v1 @ v1 + I1 * w1**2 + m2 * v2 @ v2 + I2 * w2**2)


def test_energy_never_increases_and_elastic_is_conserved():
    for _ in range(200):
        c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
        K0 = ke(v1, 0, m1, I1, v2, 0, m2, I2)
        e, mu = RNG.uniform(0, 1), RNG.uniform(0, 0.8)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=e, mu=mu)
        assert ke(v1n, w1, m1, I1, v2n, w2, m2, I2) <= K0 * (1 + 1e-9)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=1.0, mu=0.0)
        assert abs(ke(v1n, w1, m1, I1, v2n, w2, m2, I2) - K0) <= 1e-9 * K0


def test_central_impact_matches_analytic_formula():
    for _ in range(100):
        m1, m2 = RNG.uniform(1000, 2500, 2)
        n = np.array([1.0, 0.0]); t = np.array([0.0, 1.0])
        c1, c2, P = np.zeros(2), np.array([4.0, 0.0]), np.array([2.0, 0.0])  # contact point on the line through both centers
        u1, u2 = RNG.uniform(5, 20), RNG.uniform(-3, 3)
        e = RNG.uniform(0, 1)
        v1n, w1, v2n, w2, j = impact(c1, n * u1, m1, 3 * m1, c2, n * u2, m2, 3 * m2, P, n, t, e=e, mu=0.3)
        a1 = ((m1 - e * m2) * u1 + (1 + e) * m2 * u2) / (m1 + m2)
        a2 = ((m2 - e * m1) * u2 + (1 + e) * m1 * u1) / (m1 + m2)
        assert np.allclose(v1n, [a1, 0], atol=1e-9) and np.allclose(v2n, [a2, 0], atol=1e-9)
        assert abs(w1) < 1e-12 and abs(w2) < 1e-12


def test_equal_mass_elastic_head_on_swaps_velocities():
    n = np.array([1.0, 0.0]); t = np.array([0.0, 1.0])
    v1n, w1, v2n, w2, j = impact(np.zeros(2), n * 10, 1500, 4500, np.array([4.0, 0]), n * -4, 1500, 4500, np.array([2.0, 0]), n, t, e=1.0, mu=0.3)
    assert np.allclose(v1n, n * -4) and np.allclose(v2n, n * 10)


def test_newton_restitution_at_contact_point_without_friction():
    for _ in range(100):
        c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
        e = RNG.uniform(0, 1)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=e, mu=0.0)
        vp1 = v1n + w1 * np.array([-(P - c1)[1], (P - c1)[0]])  # contact point velocity = v + w x r
        vp2 = v2n + w2 * np.array([-(P - c2)[1], (P - c2)[0]])
        before = np.dot(v1 - v2, n)
        after = np.dot(vp1 - vp2, n)
        assert abs(after - (-e * before)) <= 1e-9 * max(1, abs(before)), (after, -e * before)


def test_friction_inside_coulomb_cone_and_normal_impulse_matches():
    for _ in range(200):
        c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
        mu = RNG.uniform(0, 0.8)
        v1n, w1, v2n, w2, j = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=0.2, mu=mu)
        J = m2 * (v2n - v2)  # impulse on body 2
        assert abs(np.dot(J, n) - j) <= 1e-6 * max(1, j)
        assert abs(np.dot(J, t)) <= mu * j * (1 + 1e-9) + 1e-9


def test_mirror_symmetry_flips_rotation():
    c1, v1, m1, I1, c2, v2, m2, I2, P, n, t = config(RNG)
    out = impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=0.3, mu=0.3)
    # mirror about n (flip the tangential components)
    mir = lambda x: np.dot(x, n) * n - np.dot(x, t) * t  # noqa: E731
    out2 = impact(mir(c1), mir(v1), m1, I1, mir(c2), mir(v2), m2, I2, mir(P), n, t, e=0.3, mu=0.3)
    assert np.allclose(mir(out[0]), out2[0], atol=1e-9) and np.allclose(mir(out[2]), out2[2], atol=1e-9)
    assert abs(out[1] + out2[1]) < 1e-9 and abs(out[3] + out2[3]) < 1e-9


def test_pymunk_post_impact_conserves_momentum_and_does_not_penetrate():
    # the ego (4.9x1.9, 12 m/s) hits a stopped car (4.65x2.08) from behind; no road friction, 3 s
    m1, m2 = 1600.0, 1700.0
    c1, c2 = np.array([-4.775, 0.0]), np.zeros(2)  # boxes just touching
    v1, v2 = np.array([12.0, 0.0]), np.zeros(2)
    cs1, ys1, cs2, ys2, _ = simulate_pair(c1, 0.0, v1, m1, (4.9, 1.9), c2, 0.0, v2, m2, (4.65, 2.08),
                                          np.array([1.0, 0.0]), 0.0, 30, e=0.3, mu_g=0.0)
    va, vb = (cs1[-1] - cs1[-2]) / 0.1, (cs2[-1] - cs2[-2]) / 0.1
    assert np.allclose(m1 * va + m2 * vb, m1 * v1, rtol=1e-3, atol=1.0), (m1 * va + m2 * vb, m1 * v1)
    for i in range(len(cs1)):
        hit = obb_overlap(cs1[i], ys1[i], 4.9, 1.9, cs2[i], ys2[i], 4.65, 2.08)
        assert hit is None or hit[1] < 0.02, (i, hit)  # pymunk keeps a few mm of contact slop
    assert 0.5 * (m1 * va @ va + m2 * vb @ vb) <= 0.5 * m1 * (v1 @ v1) * (1 + 1e-9)


def test_obb_overlap_basic():
    assert obb_overlap(np.zeros(2), 0, 4, 2, np.array([3.0, 0]), 0, 4, 2) is not None
    assert obb_overlap(np.zeros(2), 0, 4, 2, np.array([4.5, 0]), 0, 4, 2) is None
    n, d = obb_overlap(np.zeros(2), 0, 4, 2, np.array([3.0, 0]), 0, 4, 2)
    assert abs(d - 1.0) < 1e-9 and np.allclose(n, [1, 0])


def test_restitution_model_monotone_and_in_literature_range():
    vs = np.linspace(0, 25, 200)
    es = np.array([restitution_speed_model(v) for v in vs])
    assert np.all(np.diff(es) <= 1e-12)  # decreasing with closing speed
    assert es.min() >= 0.1 - 1e-12 and es.max() <= 0.4 + 1e-12  # within published ranges
    assert abs(restitution_speed_model(6.7) - 0.1) < 1e-9 and abs(restitution_speed_model(30) - 0.1) < 1e-12


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as ex:
                fails += 1
                print(f"FAIL {name}: {ex}")
    print("\nall passed" if not fails else f"\n{fails} failed")
    sys.exit(1 if fails else 0)
