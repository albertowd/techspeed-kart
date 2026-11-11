"""Assetto Corsa driver posture for the kart: AC's driver model sits where a posture file (.knh, the bone
tree with each bone's local matrix) and the steering animation (.ksanim, the same bones' local matrices per
frame) put its bones, placed by driver3d.ini's POSITION.

fit() makes both from the game's own driver (driver_80's bones in their rest pose, kept in
tools/driver_80_skeleton.json by tools/dev/driver_skeleton.py): it lays the back into the seat with the
head level, sits the hip joints on the seat, bends the legs so the feet reach the pedals, curls the fingers
and bends the arms so the hands grip the rim, and turns the rim lock degrees each way in the animation with
the hands going round on it.

Matrices are AC's: row vectors (p' = p @ M), translation in the last row.
"""
import json
import math
import os
import struct

import numpy as np

C = np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])        # kart (x right, y fwd, z up) <-> AC (X left, Y up, Z fwd)
REF_SW = np.array([0.0, 0.527, 0.296])                   # our steering wheel's centre in the posture's coordinates
SKELETON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver_80_skeleton.json")


# ----------------------------------------------------------------- files
class _R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def s(self):
        return self.take(self.i32()).decode("latin-1")


def read_knh(path):
    """-> (names, locals, parents, nchildren), lists in stored (depth-first) order; parents are indices
    (-1 for the root). Names repeat (the kart's own nodes), the driver's bones don't."""
    r = _R(open(path, "rb").read())
    names, local, parent, nch = [], [], [], []

    def node(par):
        i = len(names)
        names.append(r.s())
        local.append(np.array(struct.unpack("<16f", r.take(64)), float).reshape(4, 4))
        parent.append(par)
        nch.append(r.i32())
        for _ in range(nch[i]):
            node(i)

    node(-1)
    assert r.p == len(r.d)
    return names, local, parent, nch


def write_knh(path, knh):
    names, local, parent, nch = knh
    out = bytearray()
    for name, m, n in zip(names, local, nch):
        b = name.encode("latin-1")
        out += struct.pack("<i", len(b)) + b
        out += struct.pack("<16f", *np.asarray(m, np.float32).ravel())
        out += struct.pack("<i", n)
    open(path, "wb").write(bytes(out))


def write_ksanim(path, tracks):
    out = bytearray(struct.pack("<ii", 2, len(tracks)))
    for name, fr in tracks:
        b = name.encode("latin-1")
        out += struct.pack("<i", len(b)) + b + struct.pack("<i", len(fr))
        out += np.asarray(fr, "<f4").tobytes()
    open(path, "wb").write(bytes(out))


# ----------------------------------------------------------------- matrices
def _quat_to_rot(q):
    """xyzw -> row-vector rotation matrix (the transpose of the usual column one)."""
    x, y, z, w = q / np.linalg.norm(q)
    m = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return m.T


def _rot_to_quat(r):
    m = r.T
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def mat_to_frame(m):
    s = np.linalg.norm(m[:3, :3], axis=1)
    return np.r_[_rot_to_quat(m[:3, :3] / s[:, None]), m[3, :3], s]


def _translate(t):
    m = np.eye(4)
    m[3, :3] = t
    return m


def _rot_about(r, c):
    """Row-vector rotation r (3x3) about point c."""
    m = np.eye(4)
    m[:3, :3] = r
    return _translate(-np.asarray(c)) @ m @ _translate(c)


def _rotvec(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * kx @ kx).T       # row-vector form


def _align(a, b):
    """Row-vector rotation turning direction a onto direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), a @ b
    if s < 1e-9:
        return np.eye(3)
    return _rotvec(v / s * math.atan2(s, c))


def _worlds(local, parent):
    w = []
    for m, p in zip(local, parent):
        w.append(m @ w[p] if p >= 0 else m.copy())
    return w


# ----------------------------------------------------------------- wheels
D = "DRIVER:"


class Wheel:
    """A steering wheel in the posture's coordinates: centre, axis (down the column, the way AC spins it)
    and the mesh's points, read as a frame (side, top, axis) and the rim's outer edge by angle."""

    def __init__(self, centre, axis, pts):
        self.c = np.asarray(centre, float)
        ax = np.asarray(axis, float)
        self.ax = ax / np.linalg.norm(ax)
        up = np.array([0.0, 1.0, 0.0]) - self.ax * self.ax[1]
        self.up = up / np.linalg.norm(up)
        self.side = np.cross(self.ax, self.up)
        self.m = np.eye(4)
        self.m[:3, :3] = [self.side, self.up, self.ax]
        self.m[3, :3] = self.c
        d = np.asarray(pts, float) - self.c
        self.ang = np.arctan2(d @ self.side, d @ self.up)
        self.r = np.hypot(d @ self.side, d @ self.up)
        self.a = d @ self.ax

    def rim(self, ang):
        """Outer edge radius and the rim's mid-plane offset along the axis at a wheel angle (rad)."""
        near = np.abs((self.ang - ang + math.pi) % (2 * math.pi) - math.pi) < math.radians(8)
        r = self.r[near].max()
        band = near & (self.r > r - 0.025)
        return r, float(np.mean(self.a[band]))


REACH_TOL = 0.005      # m: a hand follows the rim round only while its arm reaches the grip this closely
FOLLOW_STEPS = 24      # fractions of the wheel's turn tried, from all of it down to none
RIM_HALF = 0.0175      # our grip's half thickness in the wheel's plane (kart_model.SW_GRIP)
# finger flexion (deg) at the knuckle, middle and end joints over driver_80's rest hand, whose fingers are
# nearly straight (0-25 deg a joint). Curled like this the middle finger's three joints sit on a ~3 cm
# circle: the rim's grip and the finger round it.
CURL = (55.0, 55.0, 55.0)
FINGERS = {"L": ("Index", "Middle", "Ring", "Pinkie"), "R": ("Index", "Middle", "Ring", "Pinkie")}
FINGER_NUM = {"L": (1, 2, 3), "R": (4, 5, 6)}


def _circle_centre(p1, p2, p3):
    a, b = p1 - p3, p2 - p3
    axb = np.cross(a, b)
    return p3 + np.cross(a @ a * b - b @ b * a, axb) / (2 * axb @ axb)


def _slerp(ra, rb, t):
    """Blend two row-vector rotations (unit scale)."""
    qa, qb = _rot_to_quat(ra), _rot_to_quat(rb)
    if qa @ qb < 0:
        qb = -qb
    th = math.acos(min(1.0, float(qa @ qb)))
    q = qa if th < 1e-6 else (math.sin((1 - t) * th) * qa + math.sin(t * th) * qb) / math.sin(th)
    return _quat_to_rot(q)


def _unit(r):
    return r / np.linalg.norm(r, axis=1)[:, None]


def _two_bone(w, local, parent, a, b, c, target, keep_rot, pole_bias=None):
    """Bend the chain a (upper bone) -> b (lower bone) so c's joint lands on target (clamped to reach),
    the middle joint in the plane it bent in, turned towards pole_bias (a direction) if given; c's world
    rotation becomes keep_rot (3x3). Updates `local` in place, returns how far short of the target the
    chain stays (m)."""
    ap, bp, cp = w[a][3, :3], w[b][3, :3], w[c][3, :3]
    l1, l2 = np.linalg.norm(bp - ap), np.linalg.norm(cp - bp)
    full = np.linalg.norm(target - ap)
    u = (target - ap) / full
    dist = min(full, (l1 + l2) * 0.995)
    t = ap + u * dist
    pole = (bp - ap) - u * ((bp - ap) @ u)
    pole /= np.linalg.norm(pole)
    if pole_bias is not None:
        pb = np.asarray(pole_bias, float)
        pole = pole + (pb - u * (pb @ u))
        pole /= np.linalg.norm(pole)
    x = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    mid = ap + u * x + pole * math.sqrt(max(l1 * l1 - x * x, 0.0))
    wa = w[a] @ _rot_about(_align(bp - ap, mid - ap), ap)
    wb = local[b] @ wa
    # c may hang off b through in-between bones (the forearm's twist bone): carry them along
    chain = [c]
    while parent[chain[-1]] != b:
        chain.append(parent[chain[-1]])
    chain = chain[::-1]

    def tip(wb_):
        m = wb_
        for k in chain:
            m = local[k] @ m
        return m

    wb = wb @ _rot_about(_align(tip(wb)[3, :3] - mid, t - mid), mid)
    wc = tip(wb)
    wc_new = np.eye(4)
    wc_new[:3, :3] = keep_rot
    wc_new[3, :3] = wc[3, :3]
    local[a] = wa @ np.linalg.inv(w[parent[a]])
    local[b] = wb @ np.linalg.inv(wa)
    m = wb
    for k in chain[:-1]:
        m = local[k] @ m
    local[c] = wc_new @ np.linalg.inv(m)
    return full - dist


def _angle(p, q):
    return round(math.degrees(math.acos(max(-1.0, min(1.0, p @ q / np.linalg.norm(p) / np.linalg.norm(q))))), 1)


def _frame(e1, e2):
    """Orthonormal rows from a first axis and a second one squared to it."""
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 - e1 * (e2 @ e1)
    e2 /= np.linalg.norm(e2)
    return np.array([e1, e2, np.cross(e1, e2)])


def skeleton(path=SKELETON):
    """AC's driver_80 bones in their rest pose (tools/driver_80_skeleton.json) under the nodes a posture file
    starts with (SCENE_ROOT > MODEL > FBX, as the stock cars' files): -> (names, locals, parents, nchildren)
    in stored (depth-first) order, as a .knh holds them."""
    bones = json.load(open(path))["bones"]
    names = ["SCENE_ROOT", "MODEL: driver_base_pos.fbx", "FBX: driver_base_pos.fbx"]
    local = [np.eye(4) for _ in names]
    parent = [-1, 0, 1]
    for b in bones:
        names.append(b["name"])
        local.append(np.array(b["local"], float).reshape(4, 4))
        parent.append(b["parent"] + 3 if b["parent"] >= 0 else 2)
    nch = [parent.count(k) for k in range(len(names))]
    return names, local, parent, nch


def _pitch(deg):
    """Row-vector rotation about AC's +X (the driver's left): positive tips the top forwards."""
    return _rotvec(np.array([math.radians(deg), 0.0, 0.0]))


def fit(out_knh, out_ksanim, to_ac, hip, heels, toes, wheel, scale=1.0, recline=(0.0, 0.0, 0.0, 0.0),
        hand_angle=78.0, lock=100.0, frames=101, elbow_down=0.0, elbow_out=0.0, tilts=tuple(range(-30, 105, 15)),
        hand_travel=30.0, root_move=(0.0, 0.0, 0.0), twist=0.0, knees_out=0.0,
        shoulders_fwd=0.0):
    """Write the kart's posture and steering animation, both made here from AC's own driver (skeleton()).

    to_ac(p): kart point -> the posture's own coordinates (AC axes, our wheel's centre at REF_SW).
    hip: the kart point the hip joints' centre (RIG_Center) sits on; scale: the driver's size.
    recline: degrees the pelvis, chest, neck and head lie back from the rest pose's (each bone's own turn
    in the world, so the head can stay level while the back lies in the seat); shoulders_fwd: degrees the
    collarbones swing forward about the vertical, rolling the shoulders forward towards the wheel.
    heels: kart points the ankles bend to (left, right), the knees rising between; toes: points the feet
    then point at (the pedals); knees_out spreads the knees to the sides (0: they rise straight up).

    wheel: our steering wheel (Wheel). The hands grip the rim the way a hand can: fingers curled round it
    (CURL), the knuckle row along the rim with the index finger uppermost, the palm on the rim's outer back
    and the knuckles outside it, the fingers wrapping round to its front, so the forearm comes from outside
    the rim, not over it. With the wheel straight each hand sits hand_angle degrees either side of the top;
    the animation turns the rim lock degrees each way over `frames` frames (AC plays it against the steering
    wheel's angle, driver3d.ini's LOCK). How far the knuckles tip forward round the rim (tilts, deg) is the
    one that bends the wrist least with the wheel straight. The arms are bent again in every frame;
    elbow_down and elbow_out turn the elbows down and out to the sides. In every steering frame each hand
    goes round with the rim as far as its arm reaches its grip (references/hands.mp4: the hands stay put on
    the rim to full lock) and slides on the rim only past that; hand_travel is the turn the hold is picked
    for. Returns figures for checking.
    """
    names, local, parent, nch = skeleton()
    ix = {n: k for k, n in enumerate(names) if n.startswith(D)}
    nfr, mid = frames, frames // 2                            # the middle frame: the wheel straight ahead
    root = ix[D + "DRIVER"]
    rest = [_unit(w[:3, :3]) for w in _worlds(local, parent)]

    def set_turn(loc, k, rot):
        """Bone k's world rotation -> rot (unit, row vectors), turning about its own origin."""
        w = _worlds(loc, parent)
        loc[k] = w[k] @ _rot_about(_unit(w[k][:3, :3]).T @ rot, w[k][3, :3]) @ np.linalg.inv(w[parent[k]])

    # the seated base: full size, the pelvis, chest, neck and head laid back, then moved so the hip joints sit
    # on `hip`
    # (sized at RIG_Center, the whole body's parent: the root stays square, it hands its move to POSITION)
    base = list(local)
    centre = ix[D + "RIG_Center"]
    base[centre] = base[centre].copy()
    base[centre][:3, :3] *= scale
    for bone, deg in zip(("RIG_Center", "RIG_Cest", "RIG_Nek", "RIG_Head"), recline):
        k = ix[D + bone]
        set_turn(base, k, rest[k] @ _pitch(-deg))
    for side, sgn in (("L", -1.0), ("R", 1.0)):                # each shoulder's outer end towards +Z (forward)
        k = ix[D + "RIG_Clave_" + side]
        w = _worlds(base, parent)
        set_turn(base, k, _unit(w[k][:3, :3]) @ _rotvec(np.array([0.0, sgn * math.radians(shoulders_fwd), 0.0])))
    w = _worlds(base, parent)
    base[root][3, :3] += np.asarray(to_ac(hip), float) - w[ix[D + "RIG_Center"]][3, :3]

    # finger curls: a rotation of each finger bone in its parent's space, about the hand's knuckle line
    w0 = _worlds(base, parent)
    curls, grip = {}, {}
    for side in ("L", "R"):
        hand = ix[D + "RIG_HAND_" + side]
        n1, n2, n3 = FINGER_NUM[side]

        def joint(f_, i):
            return ix[D + "HAND_%s%d" % (f_, i)]

        trial = list(base)
        # the palm's side of the hand: where the rest pose's slight curl takes the fingertips (all four: the
        # middle finger alone is nearly straight)
        palm_side = np.sign(sum(((w0[joint(f_, n3)][3, :3] - w0[joint(f_, n1)][3, :3])
                                 @ np.linalg.inv(w0[hand][:3, :3]))[0] for f_ in FINGERS[side]))
        for finger in FINGERS[side]:
            for j, ang in zip((n1, n2, n3), CURL):
                k = joint(finger, j)
                wt = _worlds(trial, parent)
                kl = (wt[ix[D + "HAND_Index%d" % n1]][3, :3] - wt[ix[D + "HAND_Pinkie%d" % n1]][3, :3])
                prow = _unit(wt[parent[k]][:3, :3])
                axis = (kl / np.linalg.norm(kl)) @ prow.T                       # in the parent's space
                best = None
                for sgn in (1, -1):                                             # the way that closes the hand
                    rp = _rotvec(axis * math.radians(ang) * sgn)
                    m = trial[k].copy()
                    m[:3, :3] = trial[k][:3, :3] @ rp
                    t2 = list(trial)
                    t2[k] = m
                    wt2 = _worlds(t2, parent)
                    tipj = joint(finger, n3)
                    palm = ((wt2[tipj][3, :3] - wt2[hand][3, :3]) @ np.linalg.inv(wt2[hand][:3, :3]))[0]
                    if best is None or palm * palm_side > best[0]:          # furthest to the palm side
                        best = (palm * palm_side, rp)
                curls[k] = best[1]
                trial[k] = trial[k].copy()
                trial[k][:3, :3] = trial[k][:3, :3] @ best[1]
        wt = _worlds(trial, parent)
        inv = np.linalg.inv(wt[hand][:3, :3])

        def hl(k):
            return (wt[k][3, :3] - wt[hand][3, :3]) @ inv                       # hand-local point

        centre = _circle_centre(*(hl(joint("Middle", j)) for j in (n1, n2, n3)))
        g = hl(joint("Pinkie", n1)) - hl(joint("Index", n1))                    # knuckle row, index to pinkie
        m = hl(joint("Middle", n1)) - centre                                    # rim centre to the knuckles
        grip[side] = dict(centre=centre, frame=_frame(g, m),
                          radius=float(np.linalg.norm(hl(joint("Middle", n1)) - centre)))
    for k, rp in curls.items():
        base[k] = base[k].copy()
        base[k][:3, :3] = base[k][:3, :3] @ rp

    # legs (the posture only; the animation leaves them alone): ankles to the heel points with the knees
    # up, feet pointed at the pedals
    post = list(base)
    short, knees = [], []
    up = np.array([0.0, 1.0, 0.0])
    for i, (side, target) in enumerate((("L", heels[0]), ("R", heels[1]))):
        w = _worlds(post, parent)
        leg, shin, ank = ix[D + "RIG_Leg_" + side], ix[D + "RIG_Shin_" + side], ix[D + "RIG_Hill_" + side]
        out = np.array([1.0 if side == "L" else -1.0, 0.0, 0.0])          # AC +X is the driver's left
        short.append(_two_bone(w, post, parent, leg, shin, ank, np.asarray(to_ac(target), float), w[ank][:3, :3],
                               pole_bias=up * 0.5 + out * knees_out))
        w = _worlds(post, parent)
        hp_, kp_, ap_ = w[leg][3, :3], w[shin][3, :3], w[ank][3, :3]
        knees.append(_angle(hp_ - kp_, ap_ - kp_))
        foot = next(k for k in range(len(names)) if parent[k] == ank)
        wa = w[ank] @ _rot_about(_align(w[foot][3, :3] - ap_, np.asarray(to_ac(toes[i]), float) - ap_), ap_)
        post[ank] = wa @ np.linalg.inv(w[parent[ank]])

    # the grips: each hand's place round the rim with the wheel straight, and the rim's turn in each frame
    # (frame 0 at full lock one way, the last at full lock the other)
    arm_bones = {side: (ix[D + "RIG_Arm_" + side], ix[D + "RIG_ForeArm_" + side], ix[D + "RIG_HAND_" + side])
                 for side in ("L", "R")}
    knuckle = {"L": ix[D + "HAND_Middle1"], "R": ix[D + "HAND_Middle4"]}
    ours = wheel
    turn = math.radians(lock) * (1.0 - 2.0 * np.arange(nfr) / (nfr - 1))
    ang0 = {"L": -math.radians(hand_angle), "R": math.radians(hand_angle)}
    down = np.array([0.0, -1.0, 0.0])

    def hand_target(side, f, pick, turned=None):
        index_up, tilt = pick
        if turned is None:
            turned = float(np.clip(turn[f], -math.radians(hand_travel), math.radians(hand_travel)))
        phi = ang0[side] + turned
        r_edge, a_mid = ours.rim(ang0[side])
        rhat = math.sin(phi) * ours.side + math.cos(phi) * ours.up
        t = math.cos(phi) * ours.side - math.sin(phi) * ours.up
        t0 = math.cos(ang0[side]) * ours.side - math.sin(ang0[side]) * ours.up
        if (t0 @ ours.up > 0) == index_up:                     # index finger uppermost with the wheel straight
            t = -t
        dirn = math.cos(math.radians(tilt)) * rhat + math.sin(math.radians(tilt)) * ours.ax
        world = _frame(t, dirn)
        rot = grip[side]["frame"].T @ world                    # hand-local axes -> world (row vectors)
        rim_pt = ours.c + rhat * (r_edge - RIM_HALF) + ours.ax * a_mid
        return rot * scale, rim_pt - grip[side]["centre"] @ (rot * scale)

    def solve(loc, side, f, tilt, turned=None):
        a, b, c = arm_bones[side]
        rot, wpos = hand_target(side, f, tilt, turned)
        w = _worlds(loc, parent)
        out = np.array([1.0 if side == "L" else -1.0, 0.0, 0.0])          # AC +X is the driver's left
        return _two_bone(w, loc, parent, a, b, c, wpos, rot, pole_bias=down * elbow_down + out * elbow_out)

    # each hand's hold: which finger is uppermost and how far the knuckles tip round the rim, the one that
    # bends the wrist least with the wheel straight and still reaches at both full locks (the two hands are
    # mirror images, so the same rule doesn't give both the same hold)
    best_tilt = {}
    for side, (a, b, c) in arm_bones.items():
        scores = []
        for pick in [(iu, tl) for iu in (True, False) for tl in tilts]:
            bend, short_, out_pen = 0.0, 0.0, 0.0
            for f in (mid, 0, nfr - 1):
                loc = list(post)
                sh = solve(loc, side, f, pick)
                short_ = max(short_, sh)
                wl = _worlds(loc, parent)
                bend_f = _angle(wl[c][3, :3] - wl[b][3, :3], wl[knuckle[side]][3, :3] - wl[c][3, :3])
                bend = max(bend, bend_f if f == mid else bend_f * 0.5)
                if f == mid:
                    # the wrist outside the rim and on the driver's side of it: the forearm comes round the
                    # rim, not over its face (a hand can't hold a wheel from the inside of the ring)
                    d = wl[c][3, :3] - ours.c
                    ax_ = d @ ours.ax
                    radial = np.linalg.norm(d - ax_ * ours.ax)
                    r_edge = ours.rim(ang0[side])[0]
                    out_pen = max(0.0, r_edge + 0.01 - radial) + max(0.0, ax_ - ours.rim(ang0[side])[1])
            scores.append((bend + 1000 * short_ + 3000 * out_pen, pick))
        best_tilt[side] = min(scores)[1]

    # every frame each hand goes round with the rim as far as its arm reaches (as a driver's do, the hand
    # going down all the way to full lock) and only slides on the rim past that: the most of the wheel's turn,
    # in steps, that leaves the hand within REACH_TOL of its grip
    bones = [n for n in names if n.startswith(D)]
    out_frames = {n: np.zeros((nfr, 10)) for n in bones}
    arm_short, bends, shorts = 0.0, [], []
    followed = {"L": [], "R": []}
    chest, neck = ix[D + "RIG_Cest"], ix[D + "RIG_Nek"]
    turn_max = float(np.max(np.abs(turn))) or 1.0
    for f in range(nfr):
        loc = list(post)
        if twist:
            # the chest turns into the turn about the spine (twist deg at full lock), bringing the shoulder
            # of the hand going over the top forwards; the legs hang on RIG_Center and stay put
            w = _worlds(loc, parent)
            spine = w[neck][3, :3] - w[chest][3, :3]
            r = _rotvec(spine / np.linalg.norm(spine) * math.radians(twist) * float(turn[f]) / turn_max)
            loc[chest] = w[chest] @ _rot_about(r, w[chest][3, :3]) @ np.linalg.inv(w[parent[chest]])
        for side, (a, b, c) in arm_bones.items():
            for k in range(FOLLOW_STEPS + 1):
                turned = float(turn[f]) * (1 - k / FOLLOW_STEPS)
                trial = list(loc)
                sh_ = solve(trial, side, f, best_tilt[side], turned)
                if sh_ <= REACH_TOL or k == FOLLOW_STEPS:
                    break
            loc[:] = trial
            followed[side].append(turned / float(turn[f]) if abs(turn[f]) > 1e-6 else 1.0)
            shorts.append((round(float(sh_) * 1000), f, side))
            arm_short = max(arm_short, sh_)
            if f == mid:
                wl = _worlds(loc, parent)
                bends.append(_angle(wl[c][3, :3] - wl[b][3, :3], wl[knuckle[side]][3, :3] - wl[c][3, :3]))
        for n in bones:
            out_frames[n][f] = mat_to_frame(loc[ix[n]])
        if f == mid:
            post_final = loc

    # the root bone square and full size, carrying the body's whole move plus root_move (driver3d.ini's
    # POSITION stays 0: the game placed the driver at minus it)
    w = _worlds(post_final, parent)
    r = post_final[root]
    assert np.allclose(r[:3, :3], np.eye(3), atol=1e-6) and np.allclose(w[parent[root]], np.eye(4)), \
        "the root bone must be square and full size to hand its move to POSITION"
    root_offset = r[3, :3].copy()
    post_final = list(post_final)
    post_final[root] = np.eye(4)
    post_final[root][3, :3] = root_offset + np.asarray(root_move, float)
    out_frames[D + "DRIVER"][:] = mat_to_frame(post_final[root])
    write_knh(out_knh, (names, post_final, parent, nch))
    write_ksanim(out_ksanim, [(n, out_frames[n]) for n in bones])
    return dict(heel_short_mm=[round(float(v) * 1000, 1) for v in short], hand_short_mm=round(float(arm_short) * 1000, 1),
                knee_deg=knees, wrist_bend_deg=bends, tilt_deg=best_tilt,
                follow_min={s: round(min(followed[s]), 2) for s in followed},
                follow={s: [round(v, 2) for v in followed[s]] for s in followed}, worst_frames=sorted(shorts)[-4:],
                root_offset=root_offset.tolist(),
                grip_radius_mm=[round(grip[s]["radius"] * scale * 1000, 1) for s in ("L", "R")],
                joints={n[len(D):]: w[k][3, :3].tolist() for n, k in ix.items()})
