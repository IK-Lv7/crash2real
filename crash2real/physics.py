"""Planar rigid-body collision of two vehicles (numpy + pymunk).

The impact itself is a single point impulse (`impact`). The motion after the impact,
including further contact and the slide to rest, is integrated with pymunk
(`simulate_pair`). This "hybrid" scheme matched real side-impact tests better than
pymunk alone (see validation/crash_tests.py).

Coordinates are the scene world frame: x, y on the road plane, yaw counter-clockwise.
Vehicles are oriented boxes. Mass, road friction and car dimensions are assumptions.
"""
import numpy as np

try:
    import pymunk
except ImportError:  # pragma: no cover
    pymunk = None

DT = 0.1  # frame interval of the recordings [s] (10 Hz)
G = 9.81
EGO_L, EGO_W, EGO_OFF = 4.9, 1.9, 1.4  # ego box length/width, box center ahead of the pose origin (assumed)


def yaw_of(T):
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def box_poly(x, y, yaw, length, width):
    from shapely import affinity
    from shapely.geometry import Polygon
    p = Polygon([(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)])
    return affinity.translate(affinity.rotate(p, yaw, origin=(0, 0), use_radians=True), x, y)


def ego_center(T):
    return T[:2, 3] + T[:2, 0] * EGO_OFF


def ego_poly(T):
    c = ego_center(T)
    return box_poly(c[0], c[1], float(yaw_of(T)), EGO_L, EGO_W)


def obb_corners(c, yaw, length, width):
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    ay = np.array([-np.sin(yaw), np.cos(yaw)])
    return np.array([c + sx * length / 2 * ax + sy * width / 2 * ay for sx in (-1, 1) for sy in (-1, 1)]), (ax, ay)


def obb_overlap(c1, y1, l1, w1, c2, y2, l2, w2):
    """Separating-axis test of two oriented boxes. None if apart, else (normal from box 1 to box 2, penetration depth)."""
    k1, ax1 = obb_corners(c1, y1, l1, w1)
    k2, ax2 = obb_corners(c2, y2, l2, w2)
    best = None
    for ax in (*ax1, *ax2):
        p1, p2 = k1 @ ax, k2 @ ax
        ov = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
        if ov <= 0:
            return None
        if best is None or ov < best[1]:
            best = (ax if np.dot(c2 - c1, ax) >= 0 else -ax, ov)
    return best


def restitution_speed_model(closing, e_low=0.4, e_min=0.1, v_lo=2.0, v_hi=6.7):
    """Coefficient of restitution as a function of the closing speed [m/s].

    Faster impacts crush more and rebound less. Piecewise linear: e_low up to v_lo,
    e_min from v_hi (15 mph), linear in between. The shape is an assumption chosen to
    stay within published ranges (0.1-0.2 at high speed, 0.1-0.4 at low speed)."""
    x = (float(closing) - v_lo) / (v_hi - v_lo)
    return float(e_low + (e_min - e_low) * min(1.0, max(0.0, x)))


def impact(c1, v1, m1, I1, c2, v2, m2, I2, P, n, t, e=0.2, mu=0.3):
    """Point impulse at contact point P. n: normal from body 1 to body 2, t: tangent.

    Normal impulse from Newton's restitution e; tangential impulse from Coulomb friction mu,
    capped at mu*j and never more than what stops the sliding at P (so friction does not add energy).
    Both bodies start without spin. Returns (v1', w1', v2', w2', normal impulse j)."""
    r1, r2 = P - c1, P - c2
    vrel_n = float(np.dot(v1 - v2, n))
    k_n = 1 / m1 + 1 / m2 + cross2(r1, n) ** 2 / I1 + cross2(r2, n) ** 2 / I2
    j = max(0.0, (1 + e) * vrel_n / k_n)
    # tangential velocity at P after the normal impulse (including the spin it creates)
    v1a, w1a = v1 - j * n / m1, cross2(r1, -j * n) / I1
    v2a, w2a = v2 + j * n / m2, cross2(r2, j * n) / I2
    vp1 = v1a + w1a * np.array([-r1[1], r1[0]])
    vp2 = v2a + w2a * np.array([-r2[1], r2[0]])
    vt = float(np.dot(vp2 - vp1, t))
    k_t = 1 / m1 + 1 / m2 + cross2(r1, t) ** 2 / I1 + cross2(r2, t) ** 2 / I2
    jt = -np.sign(vt) * min(mu * j, abs(vt) / k_t)
    J = j * n + jt * t  # applied to body 2, minus to body 1
    return v1 - J / m1, cross2(r1, -J) / I1, v2 + J / m2, cross2(r2, J) / I2, j


def simulate_pair(c1, yaw1, v1, m1, dims1, c2, yaw2, v2, m2, dims2, n, depth, steps, dt_out=DT, e=0.2, mu=0.3,
                  mu_g=0.51, r_rot=1.6, sub=100, inertia_scale=1.0, iterations=10, post=None):
    """Advance both cars for steps*dt_out seconds with pymunk.

    Road friction: a PivotJoint (translation, max force mu_g*m*g) and a GearJoint (rotation,
    max torque mu_g*m*g*r_rot) against the static body. mu_g=0.51 is a 5 m/s^2 slide;
    r_rot=1.6 m is the tire contact distance from the center (assumed).
    The partner starts separated along n by the penetration depth.
    post=(v1', w1', v2', w2'): velocities right after the impact (hybrid scheme); if None, pymunk
    resolves the impact itself.
    Returns (c1[], yaw1[], c2[], yaw2[], ego speed change 0.05 s after the start)."""
    assert pymunk is not None, "pymunk is required (pip install pymunk)"
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 1.0
    space.iterations = iterations
    space.collision_slop = 0.005
    c1 = np.asarray(c1, float)
    c2s = np.asarray(c2, float) + np.asarray(n, float) * (float(depth) + 0.002)
    bodies = []
    for (c, yaw, v, m, dims) in ((c1, yaw1, v1, m1, dims1), (c2s, yaw2, v2, m2, dims2)):
        L, W = dims
        b = pymunk.Body(m, inertia_scale * m * (L * L + W * W) / 12.0)
        b.position, b.angle, b.velocity = tuple(c), float(yaw), tuple(np.asarray(v, float))
        s = pymunk.Poly.create_box(b, size=(L, W))
        s.elasticity, s.friction = e ** 0.5, mu ** 0.5  # pymunk multiplies the two shapes' values
        space.add(b, s)
        pj = pymunk.PivotJoint(space.static_body, b, (0, 0), (0, 0))
        pj.max_bias, pj.max_force = 0.0, mu_g * m * G
        gj = pymunk.GearJoint(space.static_body, b, 0.0, 1.0)
        gj.max_bias, gj.max_force = 0.0, mu_g * m * G * r_rot
        space.add(pj, gj)
        bodies.append(b)
    b1, b2 = bodies
    if post is not None:
        pv1, pw1, pv2, pw2 = post
        b1.velocity, b1.angular_velocity = tuple(np.asarray(pv1, float)), float(pw1)
        b2.velocity, b2.angular_velocity = tuple(np.asarray(pv2, float)), float(pw2)
    h = dt_out / sub
    p1_0, p2_0 = np.array(b1.position), np.array(b2.position)
    a1_0, a2_0 = b1.angle, b2.angle
    v1_0 = np.array(b1.velocity) if post is None else np.asarray(v1, float)
    cs1, ys1, cs2, ys2 = [c1.copy()], [float(yaw1)], [np.asarray(c2, float).copy()], [float(yaw2)]
    dv1 = None
    for k in range(steps):
        for j in range(sub):
            space.step(h)
            if dv1 is None and (k * sub + j + 1) * h >= 0.05:
                dv1 = float(np.linalg.norm(np.array(b1.velocity) - v1_0))
        cs1.append(c1 + (np.array(b1.position) - p1_0))
        ys1.append(float(yaw1) + (b1.angle - a1_0))
        cs2.append(np.asarray(c2, float) + (np.array(b2.position) - p2_0))
        ys2.append(float(yaw2) + (b2.angle - a2_0))
    if dv1 is None:
        dv1 = float(np.linalg.norm(np.array(b1.velocity) - v1_0))
    return np.array(cs1), np.array(ys1), np.array(cs2), np.array(ys2), dv1
