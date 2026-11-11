"""Parametric Techspeed kart model.

Coordinates: x = right, y = forward, z = up (metres), origin on the ground at
mid-wheelbase. Dimensions come from the reference photos (side view scaled
with the 1.04 m wheelbase) and standard CIK kart sizes.

build(variant) returns a Scene: a list of nodes (name, parent, 4x4 frame) and
a list of parts (name, node, material, Mesh in node-local coordinates).
"""
import math
from dataclasses import dataclass, field

import numpy as np

import meshlib as ml


# ----------------------------------------------------------------- dimensions
# The layout below is drawn to a 1.04 m wheelbase. The side photos, scaled by the tyres (11" rear, 10" front:
# the only parts of known size), put the axles 1.12 m apart; each end of the kart matches the photos against
# its own axle and the middle is short (tools/dev/photo_overlay.py). build() therefore stretches the drawing
# fore and aft between WARP_Y0 and WARP_Y1 (warp_y): the front end moves forward and the rear end back by
# WB_STRETCH, the long parts crossing the middle (frame, floor, pods, protector) stretch, the rest moves whole.
# Exporters placing points from these constants pass them through warp_y as well.
WB = 1.040
Y_F, Y_R = WB / 2, -WB / 2
WB_REAL = 1.120
WB_STRETCH = (WB_REAL - WB) / 2
WARP_Y0, WARP_Y1 = -0.06, 0.22       # behind the seat front / engine, ahead of the steering supports


def warp_y(y):
    """Fore-aft stretch from the drawn 1.04 m layout to the real 1.12 m wheelbase (smooth in between)."""
    y = np.asarray(y, float)
    t = np.clip((y - WARP_Y0) / (WARP_Y1 - WARP_Y0), 0.0, 1.0)
    return y + WB_STRETCH * (2 * t * t * (3 - 2 * t) - 1)


def warp_pt(p):
    return (p[0], float(warp_y(p[1])), p[2])


_WARP_TAB = np.linspace(-1.5, 1.5, 3001)


def unwarp_y(y):
    """Inverse of warp_y: the drawn (1.04 m layout) y of a point measured on the real kart."""
    return np.interp(y, warp_y(_WARP_TAB), _WARP_TAB)


Y_FRONT, Y_REAR = float(warp_y(Y_F)), float(warp_y(Y_R))   # real axle lines
POD_OUT_REAR, POD_OUT_FRONT = 0.665, 0.615   # sidepod outer face: the kart is about as wide there as at the wheels
GX_X, GX_Y = 0.49, -0.12     # rental Honda crank centre: recoil starter outboard, PTO inboard
GX_PUMP = (GX_X - 0.075, GX_Y + 0.065, 0.262)   # the pulse fuel pump on the crankcase top, ahead of the cylinder
# v2: the right pod is cut away for the air cleaner (_engine_gx) to stand up above it (onboard shot): from ~2 cm in
# front of its hood-like lid back to the pod's end, and from just inboard of the lid (the pod top's xi + 0.02
# line) out to the kart's side; inboard of that the top runs on to the rear cover
POD_CUT_X = 0.45
POD_CUT_Y = GX_Y - 0.225 + 0.012 + 0.018
RENTAL_SPROCKET_X = 0.33     # rental chain line, between the inner rail and the engine
RENTAL_POD_R_REAR = 0.07     # the rental's right sidepod stops ahead of the engine
RECOIL_Z = 0.165             # rental Honda recoil starter (crank) height over the frame: ~0.19 m off the ground
RECOIL_R = 0.098             # recoil starter cover radius: ~0.19 m across in side-right.jpg
RECOIL_CLEAR = 0.115         # radius the right pod's lower edge keeps clear of round it
# the Techspeed rental's seat is nearly centred (the studio rear shot: its middle ~3 cm left of the kart's; the
# older karts had it well over to the left, -0.07, which this one was first modelled after)
SEAT_OFFSET_X = -0.030      # clear of the engine cover's skirt (tools/dev/check_clearance.py)
SEAT_YAW_DEG = 2.0          # seat turned a little towards the centre / steering wheel
# the driver's eyes (cockpit camera), measured on AC's driver_80 fitted in the seat (export_ac): over the seat's
# centre line, ~9 cm behind the visor's front; from the steering wheel's centre (world): back, and off the
# ground (it was on the kart's centre line 40 cm behind the wheel, inside the visor, and looked 8 deg down)
DRIVER_EYE_X, DRIVER_EYE_BACK, DRIVER_EYE_Z = SEAT_OFFSET_X + 0.003, 0.46, 0.83

# Rental steel protector v2, plan view of the two stacked tubes (x half-width, y; front +y). The tubes keep
# PROTECTOR_GAP (surface to surface) from the bodywork and wheels all round: the nose in front, the front
# tyres at full lock, the side pods, the rear tyres and the rear plastic. Shared with the BeamNG exporter.
PROTECTOR_R = 0.012          # ~24 mm tubes (the studio side shot; 32 mm looked heavy)
PROTECTOR_Z = (0.050, 0.094)  # the two tubes' centres over the frame: ~0.07 / 0.114 m off the ground (the studio side
                              # shot unprojected onto the tubes' plane; they were 3.5 cm higher). BeamNG's nodes too.
BAND_Z, BAND_T = (0.055, 0.155), 0.008
TREAD_TILE = 0.20            # floor tread plate texture repeat (m)      # rental v1's band: bottom / top over the frame, thickness
PROTECTOR_GAP = 0.02
PROTECTOR_FRONT = 1.01        # 1 cm ahead of the original measurement, clear of the nose
PROTECTOR_SIDE_F = 0.69       # beside the front wheels (tyre outer face 0.62, ~0.655 at full lock)
PROTECTOR_SIDE_P = 0.69       # along the side pods (0.61-0.657 at tube height)
PROTECTOR_SIDE = 0.708        # beside the rear wheels (tyre outer face 0.672)
PROTECTOR_REAR = -0.775       # rear line ~0.81 m behind the middle, real (the user: just behind the kart; -0.85 came from the
                              # side shot's edge, where its lens distortion is worst)
# The corners are wide bends (~15-20 cm radius; front.jpg, protector-v2-front-driving.jpg), not the tight
# ones first drawn.
# The front follows the kart's outline from above (the user; the paddock shots 514273209, 514277469): straight across
# ahead of the nose's centre, then bending back into a straight that runs ~4.5 cm outside the nose wings' diagonal bottom
# edge (NOSE_BOTTOM) to beside the front tyres, and bending again into the side. The bends are steel tube bends, ~8 cm
# radius. (It was a rounded curve after the old rounded nose.)
NOSE_HALF = 0.59
PROTECTOR_FRONT_BEND = 0.284       # where the straight front bends back into the diagonal


def _protector_front_line():
    """The protector's front tube centre line in plan, from the middle out and round into the side (x >= 0)."""
    (bx0, by0, _), (bx1, by1, _) = (0.25, 0.98, 0.0), (0.59, 0.70, 0.0)
    k = (by1 - by0) / (bx1 - bx0)
    n = np.array((-k, 1.0)) / math.hypot(k, 1.0)                 # the diagonal's outward normal
    o = np.array((bx0, by0)) + n * 0.045
    y_side = o[1] + k * (PROTECTOR_SIDE_F - o[0])                # where the diagonal reaches the side line
    poly = [(0.0, PROTECTOR_FRONT), (PROTECTOR_FRONT_BEND, PROTECTOR_FRONT), (PROTECTOR_SIDE_F, y_side),
            (PROTECTOR_SIDE_F, 0.55)]
    pts = ml.fillet([(x, y, 0.0) for x, y in poly], 0.08, steps=6)[:, :2]
    out = []
    for a_, b_ in zip(pts[:-1], pts[1:]):
        m = max(1, int(np.linalg.norm(b_ - a_) / 0.03))
        out += [tuple(a_ + (b_ - a_) * j / m) for j in range(m)]
    return out + [tuple(pts[-1])]


PROTECTOR_FRONT_LINE = _protector_front_line()


def protector_front_y(x):
    """The protector's front tube centre line at lateral x (drawn y), from the middle out to beside the front tyres."""
    q = np.array([p for p in PROTECTOR_FRONT_LINE if p[0] < PROTECTOR_SIDE_F - 0.01])
    return float(np.interp(abs(x), q[:, 0], q[:, 1]))


def _protector_front():
    return [p for p in PROTECTOR_FRONT_LINE if p[1] > 0.56]


PROTECTOR_PLAN = _protector_front() + [
                  (PROTECTOR_SIDE_F, 0.40), (PROTECTOR_SIDE_P + 0.005, 0.25),
                  (PROTECTOR_SIDE_P, 0.10), (PROTECTOR_SIDE_P, -0.20), (PROTECTOR_SIDE, -0.36), (PROTECTOR_SIDE, -0.52)]
# The rear corners (the user, the studio rear 3/4 shot 514248494): the side runs straight back past the rear tyre, bends
# (~74 deg, not 90) into a straight angled across behind the tyre, ~4 cm clear of its rear-outer corner, and that
# straight finishes the turn in a second bend into the rear straight about level with the Velopark sticker, ~0.47 m out.
# (Bending in at the tyre's middle, the angled straight ran through the tyre.)
# Bent steel tube: ~6 cm bend radii, densified so the spline through the plan keeps the straights straight. (A small
# chamfer out behind the kart with the rear line 8 cm further back, then one round curve, didn't match.)
REAR_BEND_1, REAR_BEND_2 = (PROTECTOR_SIDE, -0.70), (0.47, PROTECTOR_REAR)


def _rear_corner():
    poly = [(PROTECTOR_SIDE, -0.52), REAR_BEND_1, REAR_BEND_2, (0.35, PROTECTOR_REAR)]
    pts = ml.fillet([(x, y, 0.0) for x, y in poly], 0.06, steps=5)[:, :2]
    out = []
    for a_, b_ in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b_ - a_) / 0.03))
        out += [tuple(a_ + (b_ - a_) * k / n) for k in range(n)]
    return out[1:] + [tuple(pts[-1])]


REAR_CORNER = _rear_corner()
PROTECTOR_PLAN += REAR_CORNER
# Rental v1's band (protector-v1-top.jpg, -side-3q.jpg): the v2's plan round the nose, the front tyres and the
# rear, but between the wheels it pinches in to run just outside the side nerf bars (their outer edge ~0.63),
# behind the front tyres' sweep at full lock and ahead of the rear tyres. (It ran straight down the sides on the
# v2's line, ~7 cm off the nerf bars, their mounts reaching out to it.) Shared with the BeamNG exporter.
BAND_SIDE_P = 0.645
BAND_PLAN = _protector_front() + [
    (PROTECTOR_SIDE_F, 0.40), (0.672, 0.31), (BAND_SIDE_P, 0.22), (BAND_SIDE_P, -0.08), (0.668, -0.20),
    (PROTECTOR_SIDE, -0.33), (PROTECTOR_SIDE, -0.52),
    (PROTECTOR_SIDE - 0.006, PROTECTOR_REAR + 0.13), (PROTECTOR_SIDE - 0.045, PROTECTOR_REAR + 0.04),
    (PROTECTOR_SIDE - 0.12, PROTECTOR_REAR + 0.004), (0.35, PROTECTOR_REAR)]
# its right side, by the engine: pinched in only ahead of it, then out past the recoil starter (0.664 out, from
# 2 cm behind the crank to 22 cm behind it) with ~2.5 cm between; pinched all the way it ran through the engine
BAND_PLAN_R = _protector_front() + [
    (PROTECTOR_SIDE_F, 0.40), (0.672, 0.31), (BAND_SIDE_P, 0.20), (BAND_SIDE_P, 0.13), (0.675, 0.06),
    (0.692, -0.02), (0.695, -0.22), (PROTECTOR_SIDE, -0.33), (PROTECTOR_SIDE, -0.52),
    (PROTECTOR_SIDE - 0.006, PROTECTOR_REAR + 0.13), (PROTECTOR_SIDE - 0.045, PROTECTOR_REAR + 0.04),
    (PROTECTOR_SIDE - 0.12, PROTECTOR_REAR + 0.004), (0.35, PROTECTOR_REAR)]
RADIATOR_X, RADIATOR_Y = -0.31, -0.43
# steering arm ends (right side; the tie rods run from the column plate to here): short arms off the
# knuckles' bottoms, 5 cm behind and 3.5 cm inside the kingpins, so the rods run nearly straight across from
# the plate under the column (steering-pedals-front-axle.jpg; they ran back at an angle from a plate up by
# the pedals to long arms). ~90% Ackermann at full lock (inner 28.6, outer 21.4 deg on the BeamNG rack):
# at ~50% the front tyres fought each other at lock and the grip made them hop. Level with the kingpin's
# centre, so the rods rise a little outwards and pass over the front axle's diagonal tubes (at 0.105 they
# ran through them). Shared with the BeamNG exporter.
STEER_ARM_END = (0.39, 0.47, 0.13)
# steering column: wheel centre and the column's lower bearing on the floor. The side photos put the
# column at ~50 deg from horizontal (the wheel's face ~40 deg, square to it) and its foot just ahead of the
# kingpin line; it was 58 deg with the wheel lying back at 32. The column plate (tie rod inner ends) sits
# on the column's foot. Shared with the BeamNG exporter (the steering rack nodes).
SW_CENTER = (0.0, 0.25, 0.455)
COL_BASE = (0.0, 0.565, 0.075)
COL_PLATE_Z = COL_BASE[2] + 0.012
# Rental pedals (pedals-brake-master.jpg): tall 'TECH' plates in tube loops, nearly upright, pivoting on a
# bracket on top of the front frame rails, their tops up at the nose's inner edge; the loops ~0.18 m off the
# centre line, beside the front panel's foot (front.jpg, protector-v1-top.jpg). They were small plates down
# by the floor (10.5 cm square, then 13 x 14 cm), under the shoes' toes; the tops stay just behind the front
# panel's flared foot.
PEDAL_X = 0.18                           # plate centres either side of the column foot
PEDAL_W, PEDAL_H = 0.15, 0.17            # plate width and height
PEDAL_TILT = 12.0                        # plate leans forward from upright (deg)
PEDAL_PIVOT_Y, PEDAL_PIVOT_Z = 0.73, 0.092    # pivot (drawn y, height over the frame): on the rails' tops
PEDAL_TRAVEL = {"gas": 16.0, "brake": 16.0}   # degrees each pedal turns fully pressed (the brake was 10: it looked to press less) (AC's CSP script, BeamNG props)
_t = math.radians(PEDAL_TILT)
PEDAL_Y = PEDAL_PIVOT_Y + math.sin(_t) * (0.008 + PEDAL_H / 2)    # plate centre
PEDAL_Z = PEDAL_PIVOT_Z + math.cos(_t) * (0.008 + PEDAL_H / 2)
# tie rod inner ends: side by side on the plate hanging back from the column foot, behind the column's axis
# (steering-pedals-front-axle.jpg), so turning the column slides them across the kart. (They sat either side
# of the column, where turning it only swings them fore and aft.)
TIE_ROD_IN_X = 0.012
TIE_ROD_IN_BACK = 0.05
KINGPIN_AXIS = (0.0, -0.024, 0.095)       # top leans back (caster), as the BeamNG kingpin pivots
FLOOR_Z = 0.030                          # rental aluminium floor plate's underside over the frame as drawn

def family(variant):
    """The kart a variant is a version of ("rental_v1" -> "rental"): shared frame, engines and liveries."""
    return "rental" if variant.startswith("rental") else variant


VARIANTS = {
    "racing": dict(
        protector=False, front_brakes=True, engine="kf125",
        front_x=0.565, rear_x=0.600,
        tyre_f=(0.128, 0.125), tyre_r=(0.140, 0.185),      # radius, width
        rim_w_f=0.130, rim_w_r=0.180,
        chassis_mat="chassis_gunmetal", rim_mat="rim_black",
        wheel_style="racing",
        seat_x=-0.076, seat_yaw=5.0,      # the racing chassis keeps its seat well to the left, clear of the air box
    ),
    "rental": dict(
        protector="pipes", front_brakes=False, engine="gx",
        front_x=0.560, rear_x=0.585,
        tyre_f=(0.128, 0.125), tyre_r=(0.140, 0.185),
        rim_w_f=0.130, rim_w_r=0.180,
        chassis_mat="chassis_black", rim_mat="rim_alu",
        wheel_style="rental",
        # Honda's air cleaner box under its hood-like lid on both rental versions (the green cone filter of
        # engine-air-filter.jpg is the racing karts' fit)
        air_cleaner="box",
        # the frame rides 2 cm higher on the axles than a racing kart's (taller bearing cassettes at the
        # rear, spacers dropping the front stub axles below the kingpins): ~5 cm under the floor plate,
        # so the rental gets over kerbs instead of beaching on them
        ride=0.02,
    ),
}
# Rental with the first protector (protector-v1-*.jpg): a flat black band all round on rubber buffers instead of
# the steel tubes, no side pods and no engine cover (the Honda in the open); nose, front panel and rear cover
# as the v2's.
VARIANTS["rental_v1"] = dict(VARIANTS["rental"], protector="band", pods=False, engine_cover=False,
                             rear_cover="v1", exhaust="single")

# Livery atlas layout (u0, v0, u1, v1) in the 2048x2048 body texture; panel coords
# are (x or y across, arclength along the section) in metres.
ATLAS = {
    "nose":      (0.00, 0.00, 1.00, 0.25),
    "panel":     (0.00, 0.25, 0.30, 0.75),
    "pod_L":     (0.30, 0.25, 1.00, 0.50),
    "pod_R":     (0.30, 0.50, 1.00, 0.75),
    "rear":      (0.00, 0.75, 1.00, 1.00),
}


@dataclass
class Node:
    name: str
    parent: str = None
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))   # local -> parent


@dataclass
class Part:
    name: str
    node: str
    material: str
    mesh: ml.Mesh
    hidden: bool = False      # an alternative kept in the source but not exported (hidden from render)


@dataclass
class Scene:
    variant: str
    params: dict
    nodes: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    panels: dict = field(default_factory=dict)   # panel name -> (u_range, v_range) for the painter
    info: dict = field(default_factory=dict)     # hard points for physics exporters
    hide_new: bool = False                       # parts added while set are hidden alternatives

    @property
    def family(self):
        return family(self.variant)

    def node(self, name, parent="BODY", matrix=None):
        self.nodes.append(Node(name, parent, np.eye(4) if matrix is None else matrix))

    def add(self, name, material, mesh, node="BODY"):
        self.parts.append(Part(name, node, material, mesh, self.hide_new))

    def world(self, node_name):
        m = np.eye(4)
        by = {n.name: n for n in self.nodes}
        n = by.get(node_name)
        while n is not None:
            m = n.matrix @ m
            n = by.get(n.parent)
        return m


def mirror_pts(pts):
    return [(-x, y, z) for x, y, z in pts]


# ------------------------------------------------------------------- helpers
def _closest_on_mesh(verts, faces, p):
    """The point of a mesh's surface nearest p (faces fanned into triangles): -> (point, distance)."""
    tri = np.array([(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)])
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    ab, ac, ap = b - a, c - a, p - a
    # the point's barycentric place in each triangle's plane: inside the triangle it is the nearest point
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    q = a + ab * v[:, None] + ac * w[:, None]
    best = np.where(inside, np.linalg.norm(q - p, axis=1), np.inf)
    for s0, s1 in ((a, b), (b, c), (c, a)):              # outside: the nearest point of its nearest edge
        e = s1 - s0
        t = np.clip(((p - s0) * e).sum(1) / np.maximum((e * e).sum(1), 1e-18), 0, 1)
        qe = s0 + e * t[:, None]
        de = np.linalg.norm(qe - p, axis=1)
        better = ~inside & (de < best)
        q = np.where(better[:, None], qe, q)
        best = np.where(better, de, best)
    i = int(np.argmin(best))
    return q[i], float(best[i])


def _reach(sc, prefixes, p, into=0.003):
    """Where a tube from p meets the nearest of the named parts (as built so far, on any node): their surface
    point nearest p, `into` further on, so the tube's end sits in the part rather than short of it."""
    p = np.asarray(p, float)
    best = None
    for part in sc.parts:
        if not part.name.startswith(prefixes):
            continue
        v = np.asarray(part.mesh.verts, float)[:, :3]
        if part.node != "BODY":
            v = (np.c_[v, np.ones(len(v))] @ sc.world(part.node).T)[:, :3]
        q, d = _closest_on_mesh(v, part.mesh.faces, p)
        if best is None or d < best[1]:
            best = (q, d)
    assert best is not None, "nothing named %s to reach" % (prefixes,)
    q, d = best
    return q + (q - p) / max(d, 1e-9) * into


def _sym_tube(sc, name, mat, pts, r, bend=0.03, segs=10, both=True):
    m = ml.tube(pts, r, segs, bend)
    sc.add(name + ("_R" if both else ""), mat, m)
    if both:
        sc.add(name + "_L", mat, m.mirrored_x())


def _section_loft(section_fn, params, closed=False, cap=False):
    """Loft sections and return mesh plus per-vertex (param, arclength) for UVs."""
    secs, arcs = [], []
    for p in params:
        pts = np.asarray(section_fn(p), float)
        secs.append(pts)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arcs.append(np.r_[0, np.cumsum(seg)])
    m = ml.loft(secs, closed=closed, cap_start=cap, cap_end=cap)
    return m, np.concatenate(arcs)


def _uv_from(mesh, ucoord, vcoord, rect, u_range, v_range):
    u0, v0, u1, v1 = rect

    def f(i):
        return (u0 + (ucoord[i] - u_range[0]) / (u_range[1] - u_range[0]) * (u1 - u0),
                v0 + (vcoord[i] - v_range[0]) / (v_range[1] - v_range[0]) * (v1 - v0))

    n = len(ucoord)
    mesh.uvs = [tuple(f(i) if i < n else (u0, v0) for i in face) for face in mesh.faces]


# ------------------------------------------------------------------- wheels
def tyre_mesh(radius, width, rim_r, segs=48):
    w = width / 2
    prof = [
        (rim_r + 0.004, -w * 0.80),
        (rim_r + 0.018, -w * 0.97),
        (rim_r + (radius - rim_r) * 0.55, -w * 1.02),
        (radius - 0.016, -w * 0.99),
        (radius - 0.004, -w * 0.90),
        (radius, -w * 0.70),
        (radius, 0.0),
        (radius, w * 0.70),
        (radius - 0.004, w * 0.90),
        (radius - 0.016, w * 0.99),
        (rim_r + (radius - rim_r) * 0.55, w * 1.02),
        (rim_r + 0.018, w * 0.97),
        (rim_r + 0.004, w * 0.80),
    ]
    m = ml.lathe(prof, segs, axis=(1, 0, 0))
    m.sharp_angle = 70
    # UV: u once round the tyre (the sidewall's MG badges and labels; it repeated 8 times), v across the
    # profile from the inner bead (0) over the tread to the outer bead (1): textures.tyre
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(prof), axis=0), axis=1))]
    arc /= arc[-1]
    k = len(prof)
    uvs = []
    for f in m.faces:
        uv = []
        for idx in f:
            i, j = divmod(idx, k)
            uv.append((i / segs, arc[j]))
        # fix wrap seam
        us = [a for a, _ in uv]
        if max(us) - min(us) > 0.5:
            uv = [(a + 1 if a < 0.5 else a, b) for a, b in uv]
        uvs.append(tuple(uv))
    m.uvs = uvs
    return m


def rim_mesh(rim_r, width, style, outer_sign, segs=40):
    """Kart rim (5"), outer face towards outer_sign*x."""
    w = width / 2
    s = outer_sign
    prof = [
        (rim_r + 0.010, s * w),                 # outer flange lip
        (rim_r + 0.010, s * (w - 0.006)),
        (rim_r + 0.002, s * (w - 0.008)),
        (rim_r, s * (w - 0.020)),               # bead seat
        (rim_r - 0.008, s * (w - 0.032)),       # drop
        (rim_r - 0.008, -s * (w - 0.032)),
        (rim_r, -s * (w - 0.020)),
        (rim_r + 0.002, -s * (w - 0.008)),
        (rim_r + 0.010, -s * (w - 0.006)),
        (rim_r + 0.010, -s * w),                # inner flange
        (rim_r - 0.004, -s * w),
        (rim_r - 0.012, -s * (w - 0.025)),
        (rim_r - 0.012, s * (w - 0.016)),       # inner barrel back to the face
        (0.030, s * (w - 0.026)),               # mounting face: shallow, a polished disc nearly flush with the
        (0.030, s * (w - 0.036)),               # flange (the photos' rims are bright to the middle; 4.5 cm
        (rim_r - 0.016, s * (w - 0.026)),       # deep, the face sat in shadow and the wheels looked black)
    ]
    m = ml.lathe(prof, segs, axis=(1, 0, 0))
    # one open sheet folded round the inner flange: whichever way it faces, either the barrel or the
    # mounting face shows its back from outside, and the games draw one side only (it rendered as a hole).
    # It gets a 3 mm wall (solidify) so both sides are drawn. (A back skin of the same faces turned round did
    # that too, but the coincident faces shaded the rims near black in the renders, where the real ones are
    # bright polished discs.)
    m.modifiers = [("solidify", 0.003)]
    m.sharp_angle = 35
    parts = [m]
    # the hub the rim bolts to, filling the mounting face's bore flush with its front (the bore was open: the
    # wheel showed straight through its middle)
    face = w - 0.026                                   # the mounting face's front at the bore
    parts.append(ml.lathe([(0.0, s * (face - 0.03)), (0.034, s * (face - 0.03)), (0.034, s * face), (0.0, s * face)],
                          24, axis=(1, 0, 0)))
    # hub centre nut (~19 mm) on the axle, seated on the hub (it stood 1 cm off it)
    parts.append(ml.lathe([(0.0, s * face), (0.0095, s * face), (0.0095, s * (face + 0.012)),
                           (0.006, s * (face + 0.015)), (0.0, s * (face + 0.015))], 12, axis=(1, 0, 0)))
    # three M8 studs with ~1 cm nuts tightened down on the face (the nuts stood 1 cm off it); the face slopes
    # back from the bore, so at the studs' radius its front is ~1.5 mm further out
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.5
        c = np.array([s * (face + 0.0012), 0.036 * math.cos(a), 0.036 * math.sin(a)])
        parts.append(ml.lathe([(0.0, -0.002), (0.004, -0.002), (0.004, 0.013), (0.0, 0.013)], 8,
                              origin=c, axis=(s, 0, 0)))
        parts.append(ml.lathe([(0.0, -0.001), (0.0062, -0.001), (0.0062, 0.007), (0.0, 0.007)], 6,
                              origin=c, axis=(s, 0, 0)))
    return ml.merge(parts)


def front_hub_mesh(x_in, x_out):
    return ml.lathe([(0.0, x_in), (0.026, x_in), (0.030, x_in + 0.01), (0.030, x_out - 0.01),
                     (0.040, x_out - 0.006), (0.040, x_out), (0.0, x_out)], 20, axis=(1, 0, 0))


def disc_mesh(x, r_out, r_in, thick, axis_sign=1, vents=12):
    """Ventilated brake disc ring around the x axis at lateral position x."""
    t = thick / 2
    d = ml.lathe([(r_in, x - t), (r_out, x - t), (r_out, x + t), (r_in, x + t), (r_in, x - t)],
                 48, axis=(1, 0, 0))
    d.sharp_angle = 30
    carrier = ml.lathe([(0.026, x - 0.012), (r_in + 0.004, x - 0.004), (r_in + 0.004, x + 0.004),
                        (0.026, x + 0.012)], 24, axis=(1, 0, 0))
    return ml.merge([d, carrier])


def sprocket_mesh(x, r, thick=0.006, teeth=60):
    pts = []
    for i in range(teeth * 2):
        a = 2 * math.pi * i / (teeth * 2)
        rr = r if i % 2 == 0 else r - 0.006
        pts.append((rr * math.cos(a), rr * math.sin(a)))
    ring = ml.sweep(np.array([[x - thick / 2, 0, 0], [x + thick / 2, 0, 0]]), pts, caps=True)
    hub = ml.lathe([(0.026, x - 0.015), (0.045, x - 0.015), (0.045, x + 0.015), (0.026, x + 0.015)],
                   24, axis=(1, 0, 0))
    return ml.merge([ring, hub])


# ------------------------------------------------------------------- build
def build(variant):
    p = dict(VARIANTS[variant])
    sc = Scene(variant, p)
    # everything on BODY is modelled for a frame `ride` lower than it sits: BODY lifts it, the wheels hang
    # that much lower on it, so the axles stay on the ground at their radius
    ride = p.get("ride", 0.0)
    sc.node("BODY", parent=None, matrix=ml.translate((0.0, 0.0, ride)))

    rf, wf = p["tyre_f"]
    rr, wr = p["tyre_r"]
    fx, rx = p["front_x"], p["rear_x"]
    rim_r = 0.0635
    chassis = p["chassis_mat"]
    tube_r = 0.015

    # ---------------------------------------------------------------- frame
    rail = [(0.130, 0.80, 0.072), (0.150, 0.62, 0.055), (0.170, 0.40, 0.050),
            (0.270, 0.15, 0.050), (0.330, -0.15, 0.050), (0.370, -0.42, 0.056),
            (0.390, -0.60, 0.085)]
    _sym_tube(sc, "frame_rail", chassis, rail, tube_r, bend=0.12)
    inner = [(0.230, 0.18, 0.050), (0.215, -0.10, 0.050), (0.210, -0.40, 0.056), (0.215, -0.60, 0.085)]
    _sym_tube(sc, "frame_inner", chassis, inner, tube_r * 0.93, bend=0.10)
    sc.add("frame_cross_front", chassis, ml.tube([(-0.13, 0.80, 0.072), (0.13, 0.80, 0.072)], tube_r, 10))
    sc.add("frame_cross_mid", chassis, ml.tube([(-0.25, 0.20, 0.050), (0.25, 0.20, 0.050)], tube_r * 0.9, 10))
    sc.add("frame_cross_rear", chassis, ml.tube([(-0.37, -0.42, 0.056), (0.37, -0.42, 0.056)], tube_r, 10))
    sc.add("frame_cross_end", chassis, ml.tube([(-0.39, -0.60, 0.085), (0.39, -0.60, 0.085)], tube_r * 0.9, 10))
    sc.add("frame_diag", chassis, ml.tube([(-0.21, -0.40, 0.056), (0.33, -0.15, 0.050)], tube_r * 0.85, 10))
    # front axle: the cross tubes end in the kingpin C-yokes (front-spindle.jpg, front-brake-spindle.jpg)
    kp_x = 0.425
    kpf = _kingpin_frame((kp_x, Y_F, rf))
    web = lambda dy, dz: tuple((kpf @ np.array([-0.045, dy, dz, 1.0]))[:3])    # on the yoke's web
    _sym_tube(sc, "frame_axle", chassis, [(0.150, 0.55, 0.055), web(0.0, -0.02)], tube_r, 0)
    _sym_tube(sc, "frame_axle_diag", chassis, [(0.175, 0.36, 0.050), web(-0.012, -0.026)], tube_r * 0.9, 0)
    _sym_tube(sc, "frame_axle_up", chassis, [(0.140, 0.70, 0.062), web(0.012, 0.03)], tube_r * 0.8, 0)
    yoke, bolt = _kingpin_yoke(kpf)
    sc.add("kingpin_yoke_R", chassis, yoke)
    sc.add("kingpin_yoke_L", chassis, yoke.mirrored_x())
    sc.add("kingpin_bolt_R", "steel", bolt)
    sc.add("kingpin_bolt_L", "steel", bolt.mirrored_x())
    # front bumper tubes
    _sym_tube(sc, "bumper_front_lo", chassis, [(0.11, 0.80, 0.072), (0.20, 0.93, 0.060), (0.0001, 0.96, 0.060)], 0.011, 0.05)
    _sym_tube(sc, "bumper_front_hi", chassis, [(0.13, 0.78, 0.075), (0.17, 0.88, 0.150), (0.0001, 0.90, 0.155)], 0.010, 0.05)
    # side pod bars (nerf bars)
    nerf_rear = (0.61, -0.22, 0.110) if variant == "racing" else (0.62, 0.06, 0.110)
    nerf_ret = (0.33, -0.17, 0.050) if variant == "racing" else (0.30, 0.08, 0.050)
    _sym_tube(sc, "nerf", chassis, [(0.27, 0.15, 0.050), (0.58, 0.25, 0.100), nerf_rear, nerf_ret],
              0.011, 0.06)
    # rear bumper tubes
    _sym_tube(sc, "bumper_rear", chassis, [(0.37, -0.42, 0.056), (0.48, -0.66, 0.100), (0.0001, -0.74, 0.100)], 0.011, 0.08)
    # rear axle bearing hangers (left, centre, right)
    for x in (-0.39, 0.0, 0.39):
        plate = ml.box((x, Y_R, 0.112), (0.012, 0.10, 0.075))
        bearing = ml.lathe([(0.026, -0.012), (0.042, -0.012), (0.042, 0.012), (0.026, 0.012)], 24,
                           origin=(x, Y_R, rr - ride), axis=(1, 0, 0))
        sc.add("axle_bearing_%+.2f" % x, "black_anodized", ml.merge([plate, bearing]))
    sc.add("axle_post_c", chassis, ml.tube([(0.0, -0.42, 0.056), (0.0, Y_R, 0.090)], 0.012, 8))
    # floor tray
    tray_pts = [(-0.14, 0.74), (0.14, 0.74), (0.17, 0.40), (0.25, 0.16), (-0.25, 0.16), (-0.17, 0.40)]
    tray = ml.loft([[(x, y, 0.036) for x, y in tray_pts], [(x, y, 0.040) for x, y in tray_pts]],
                   closed=True, cap_start=True, cap_end=True)
    tray.smooth = False
    ml.project_uv(tray, (0, 0, 0), (1, 0, 0), (0, 1, 0), (-0.3, 0.3), (0.1, 0.8), (0, 0, 4, 4))
    if variant == "racing":
        sc.add("floor_tray", "carbon", tray)
    else:
        # rental: big bolted aluminium floor plate (steering-pedals-front-axle.jpg)
        # aluminium tread plate out to the kart's sides between the wheels (the onboard frames: it reaches the
        # band beside the driver and in front of the engine); it stopped at the inner rails. At the front it's a
        # wedge (steering-pedals-front-axle.jpg, pedals-brake-master.jpg): from the front cross tube, wide enough
        # for both pedals and the master cylinder, it widens back to the full width behind the front wheels,
        # staying ~5 cm inside the front tyres at full lock (it was a 32 cm tongue under half of each pedal)
        fl = [(-0.20, 0.835), (0.20, 0.835), (0.285, 0.81), (0.395, 0.635), (0.40, 0.43), (0.60, 0.345),
              (0.62, -0.05), (-0.62, -0.05), (-0.60, 0.345), (-0.40, 0.43), (-0.395, 0.635), (-0.285, 0.81)]
        floor = ml.loft([[(x, y, FLOOR_Z) for x, y in fl], [(x, y, FLOOR_Z + 0.004) for x, y in fl]],
                        closed=True, cap_start=True, cap_end=True)
        floor.smooth = False
        # the diamond tread texture repeats every TREAD_TILE metres
        ml.project_uv(floor, (0, 0, 0), (1, 0, 0), (0, 1, 0), (0.0, TREAD_TILE), (0.0, TREAD_TILE), (0, 0, 1, 1))
        # plain bolted aluminium, not diamond tread (the studio cockpit shots, 515330967 / 515935974)
        sc.add("floor_plate", "floor_alu", floor)
        bolts = [ml.lathe([(0, 0), (0.008, 0), (0.006, 0.004), (0, 0.004)], 10, origin=(x, y, 0.034),
                          axis=(0, 0, 1), closed_ends=True)
                 for x, y in ((-0.12, 0.74), (0.12, 0.74), (-0.19, 0.50), (0.19, 0.50), (-0.25, 0.79), (0.25, 0.79),
                              (-0.36, 0.62), (0.36, 0.62), (-0.37, 0.45), (0.37, 0.45), (-0.40, 0.30),
                              (0.40, 0.30), (-0.41, -0.03), (0.41, -0.03), (0.0, 0.60), (0.0, 0.20),
                              (-0.24, 0.10), (0.24, 0.10), (-0.57, 0.31), (0.57, 0.31), (-0.59, -0.02),
                              (0.59, -0.02))]
        sc.add("floor_bolts", "steel", ml.merge(bolts))

    # ---------------------------------------------------------------- steering
    sw_c = np.array(SW_CENTER)                         # steering wheel centre
    col_base = np.array(COL_BASE)                      # column foot (lower bearing)
    d = col_base - sw_c
    d /= np.linalg.norm(d)
    sc.add("steer_column", "steel" if variant == "racing" else "black_anodized",
           ml.tube([col_base, sw_c + d * 0.03], 0.0105, 12))
    if variant == "racing":
        _sym_tube(sc, "steer_support", chassis, [(0.16, 0.50, 0.055), (0.03, 0.43, 0.30)], 0.010, 0)
        sc.add("steer_support_clamp", "black_anodized",
               ml.lathe([(0.016, -0.025), (0.016, 0.025)], 16, origin=(0, 0.43, 0.30), axis=d))
    else:
        # two tubes from the frame in front of the seat up to just under the wheel (seat-cockpit.jpg)
        top = sw_c + d * 0.10
        _sym_tube(sc, "steer_support", chassis, _steer_support_path(col_base, sw_c), 0.010, 0.05)
        sc.add("steer_support_clamp", "black_anodized",
               ml.lathe([(0.017, -0.025), (0.017, 0.025)], 16, origin=tuple(top), axis=d))
    _steering_linkage(sc, col_base, d)
    arm_end = np.array(STEER_ARM_END)
    # steering wheel node. AC spins STEER_HR about its own Z axis, and the stock karts (gokart270/390)
    # have that Z pointing down the column away from the driver, with X to the car's left. export_ac
    # turns every node into AC's axes (x -> -X, y -> Z, z -> Y), which relabels the node's own axes too:
    # AC's local Z is this frame's y and AC's X is this frame's -x. So here: y down the column, x to the
    # kart's right, z towards the top of the wheel. (With z down the column, the game spun the wheel
    # about an axis across the column, and the other way round.)
    xl = np.array([1.0, 0, 0])
    yl = d
    zl = np.cross(xl, yl)
    m_sw = np.eye(4)
    m_sw[:3, 0], m_sw[:3, 1], m_sw[:3, 2], m_sw[:3, 3] = xl, yl, zl, sw_c
    sc.node("STEER_HR", "BODY", m_sw)
    sc.info["steer_wheel"] = dict(center=sw_c.tolist(), axis=yl.tolist(), column_base=col_base.tolist())
    n_before = len(sc.parts)
    _steering_wheel(sc, p["wheel_style"])   # modelled with +z towards the driver, +y to the top ...
    flip = ml.rot_axis((1, 0, 0), 90)        # ... so turn it into the node frame (z -> -y, y -> z)
    for part in sc.parts[n_before:]:
        part.mesh = part.mesh.transformed(flip)

    # ---------------------------------------------------------------- pedals
    if variant == "racing":
        for side, nm in ((1, "throttle"), (-1, "brake")):
            x = side * 0.11
            pedal = ml.merge([ml.box((x, 0.80, 0.11), (0.06, 0.012, 0.085), rot=ml.rot_axis((1, 0, 0), -25)),
                              ml.tube([(x, 0.79, 0.045), (x, 0.80, 0.075)], 0.006, 8)])
            sc.add("pedal_" + nm, "black_anodized" if nm == "brake" else "aluminium", pedal)
    else:
        _pedals_rental(sc)

    # ---------------------------------------------------------------- seat
    _seat(sc, chassis)
    if p.get("pods", True) and family(variant) == "rental":
        # v2: no tank under the steering column (the studio cockpit shots show a clear floor); the fuel is in the
        # left pod, filled through the cap on the left deck behind it (_rear_cover_rental). Its line runs from
        # the pod's rear across the floor behind the seat to the Honda's pulse pump.
        pump = np.array(GX_PUMP)
        sc.add("fuel_line", "fuel_hose", ml.tube(
            [(-0.50, -0.30, 0.10), (-0.42, -0.36, 0.075), (-0.10, -0.40, 0.068), (0.20, -0.36, 0.068),
             (0.30, -0.22, 0.09), (0.37, -0.05, 0.17), (pump[0] - 0.01, pump[1] + 0.05, pump[2] + 0.02),
             tuple(pump + (-0.022, 0.0, 0.006))], 0.004, 8, 0.05))
    else:
        _fuel_tank(sc, chassis, variant, col_base, sw_c)

    # ---------------------------------------------------------------- wheels
    corners = {"LF": (-fx, Y_F, rf, "f"), "RF": (fx, Y_F, rf, "f"),
               "LR": (-rx, Y_R, rr, "r"), "RR": (rx, Y_R, rr, "r")}
    sc.info["wheels"] = {}
    for key, (x, y, r, axle) in corners.items():
        side = 1 if x > 0 else -1
        sc.node("WHEEL_" + key, "BODY", ml.translate((x, y, r - ride)))
        radius, width = (rf, wf) if axle == "f" else (rr, wr)
        rim_w = p["rim_w_f"] if axle == "f" else p["rim_w_r"]
        tm = tyre_mesh(radius, width, rim_r)
        if side < 0:     # same mesh on the left: mirror v so the texture's inner side faces the kart, and u so
            # the sidewall lettering reads the right way round from outside
            tm.uvs = [tuple((1.0 - u, 1.0 - v) for u, v in face) for face in tm.uvs]
        sc.add("tyre_" + key, "tyre", tm, "WHEEL_" + key)
        sc.add("rim_" + key, p["rim_mat"], rim_mesh(rim_r, rim_w, p["wheel_style"], side), "WHEEL_" + key)
        sc.info["wheels"][key] = dict(center=(x, y, r), radius=radius, width=width, rim_radius=rim_r + 0.01)

    # front spindles (steer with the wheel) and hubs (spin with the wheel)
    sc.info["kingpin_x"] = kp_x
    for key in ("LF", "RF"):
        x, y, r = corners[key][:3]
        side = 1 if x > 0 else -1
        kp = np.array([side * kp_x, Y_F, r])
        sc.node("SUSP_" + key, "BODY", ml.translate(kp))
        local_arm = arm_end * (side, 1, 1) - kp
        body, bolt = _spindle(abs(x) - kp_x, ride, local_arm * (side, 1, 1))
        if side < 0:
            body, bolt = body.mirrored_x(), bolt.mirrored_x()
        sc.add("spindle_" + key, "black_anodized", body, "SUSP_" + key)
        sc.add("spindle_bolt_" + key, "steel", bolt, "SUSP_" + key)
        # front hub spins with the wheel
        hub_in, hub_out = -side * (abs(x) - kp_x - 0.035), -side * 0.03
        hub = front_hub_mesh(min(hub_in, hub_out), max(hub_in, hub_out))
        sc.add("hub_" + key, "black_anodized", hub, "WHEEL_" + key)
        if p["front_brakes"]:
            dx = -side * (abs(x) - kp_x - 0.045)
            sc.add("disc_" + key, "brake_disc", disc_mesh(dx, 0.065, 0.040, 0.010), "WHEEL_" + key)
            cal = ml.box((side * (abs(x) - kp_x - 0.045), -0.035, 0.045 - ride), (0.035, 0.045, 0.05))
            cal.modifiers = [("bevel", 0.005, 2)]
            sc.add("caliper_" + key, "caliper", cal, "SUSP_" + key)

    # rear axle (halves spin with each rear wheel), brake disc on the left, sprocket on the right
    for key in ("LR", "RR"):
        x, y, r = corners[key][:3]
        side = 1 if x > 0 else -1
        ax = ml.lathe([(0.0, 0.0), (0.025, 0.0), (0.025, side * 0.0), (0.025, -side * (abs(x) - 0.01)),
                       (0.0, -side * (abs(x) - 0.01))], 20, axis=(1, 0, 0), closed_ends=False)
        hub = ml.lathe([(0.0, side * 0.0), (0.040, side * 0.0), (0.040, -side * 0.09), (0.028, -side * 0.10),
                        (0.0, -side * 0.10)], 20, axis=(1, 0, 0))
        sc.add("axle_" + key, "steel", ax, "WHEEL_" + key)
        sc.add("rear_hub_" + key, "black_anodized", hub, "WHEEL_" + key)
    lx = corners["LR"][0]
    sc.add("disc_rear", "brake_disc", disc_mesh(-0.12 - lx, 0.100, 0.065, 0.016), "WHEEL_LR")
    cal = ml.box((-0.12, Y_R + 0.06, rr - ride + 0.085), (0.05, 0.06, 0.045))
    cal.modifiers = [("bevel", 0.006, 2)]
    sc.add("caliper_rear", "caliper", cal)
    sc.add("caliper_rear_mount", "black_anodized", ml.box((-0.12, Y_R + 0.06, rr - ride + 0.03), (0.012, 0.05, 0.09)))
    # chain line: outboard of the 125; on the rental the Honda's PTO faces inboard
    sprocket_x = 0.46 if variant == "racing" else RENTAL_SPROCKET_X
    sc.add("sprocket_axle", "sprocket", sprocket_mesh(sprocket_x - corners["RR"][0], 0.090, teeth=p.get("teeth", 70)), "WHEEL_RR")
    sc.info["sprocket"] = dict(x=sprocket_x, radius=0.090)

    # ---------------------------------------------------------------- bodywork
    if variant == "racing":
        _nose(sc, variant)
        _front_panel(sc, variant)
        _sidepods(sc, variant)
    else:
        _nose_rental(sc)
        _front_panel_rental(sc)
        if p.get("pods", True):
            _sidepods_rental(sc)
    if variant == "racing":
        _rear_bumper(sc, variant)
    else:
        _rear_cover_rental(sc)
        if p.get("engine_cover", True):
            _engine_cover_rental(sc)
    if p["protector"] == "pipes":
        _protector(sc, chassis)
    elif p["protector"] == "band":
        _protector_band(sc, chassis)

    # ---------------------------------------------------------------- engine
    if p["engine"] == "kf125":
        _engine_125(sc, chassis, sprocket_x)
    else:
        _engine_gx(sc, chassis, sprocket_x)

    sc.info["steer_lock_deg"] = 100
    _stretch(sc)
    return sc


# ------------------------------------------------------------------- front axle and steering
def _kingpin_frame(kp):
    """Frame on a kingpin: x out to the right, z up the kingpin bolt (leaning back for caster), y forward."""
    a = np.asarray(KINGPIN_AXIS, float)
    a /= np.linalg.norm(a)
    return ml.frame(kp, (1.0, 0.0, 0.0), (0.0, a[2], -a[1]))


def _kingpin_yoke(kpf):
    """Right-hand C-yoke welded on the end of the front axle tubes: two ears round the spindle boss and the
    web joining them, and the kingpin bolt with its washer stack and nuts (front-spindle.jpg)."""
    parts = []
    for z0, z1 in ((0.033, 0.041), (-0.041, -0.033)):
        parts.append(ml.lathe([(0.0, z0), (0.021, z0), (0.021, z1), (0.0, z1)], 20, axis=(0, 0, 1), closed_ends=True))
        parts.append(ml.box((-0.023, 0.0, (z0 + z1) / 2), (0.046, 0.042, z1 - z0)))
    # the web stops 4 mm behind the kingpin (the axle tubes meet it within 12 mm): with the inner wheel near
    # full lock the steering arm swings in behind it, and it ran through the web's back edge in game
    parts.append(ml.box((-0.044, 0.0085, 0.0), (0.008, 0.025, 0.082)))
    yoke = ml.merge(parts).transformed(kpf)
    yoke.sharp_angle = 50
    # M8 kingpin bolt: ~1 cm nuts (13 mm across the corners) on thin washers; they were 19 mm nuts on a
    # 34 mm washer, the biggest thing on the front corner in game
    bolt = ml.merge([
        ml.lathe([(0.0, -0.052), (0.004, -0.052), (0.004, 0.052), (0.0, 0.052)], 10, axis=(0, 0, 1), closed_ends=True),
        ml.lathe([(0.0, 0.041), (0.0085, 0.041), (0.0085, 0.0435), (0.0, 0.0435)], 16, axis=(0, 0, 1), closed_ends=True),
        ml.lathe([(0.0, 0.0435), (0.0065, 0.0435), (0.0065, 0.0505), (0.0, 0.0505)], 6, axis=(0, 0, 1), closed_ends=True),
        ml.lathe([(0.0, -0.0505), (0.0065, -0.0505), (0.0065, -0.041), (0.0, -0.041)], 6, axis=(0, 0, 1), closed_ends=True),
    ]).transformed(kpf)
    bolt.smooth = False
    return yoke, bolt


def _spindle(reach, ride, arm):
    """Right front spindle in the kingpin's node frame (front-spindle.jpg, front-brake-spindle.jpg): the
    boss between the yoke's ears, a tapered stub axle out to the wheel (dropped `ride` below the kingpin by
    the rental's spacers), and the flat steering arm running back and in to the tie rod's rod end, with
    the rod end's bolt. reach: kingpin to wheel centre; arm: rod end centre relative to the kingpin."""
    kpf = _kingpin_frame((0.0, 0.0, 0.0))
    boss = ml.lathe([(0.0, -0.031), (0.016, -0.031), (0.016, 0.031), (0.0, 0.031)], 20, axis=(0, 0, 1),
                    closed_ends=True).transformed(kpf)
    stub = ml.lathe([(0.0, 0.010), (0.019, 0.010), (0.019, 0.03), (0.0145, 0.07), (0.0125, 0.075),
                     (0.0125, reach - 0.03), (0.0, reach - 0.03)], 20, origin=(0, 0, -ride), axis=(1, 0, 0),
                    closed_ends=True)
    gusset = ml.box((0.014, 0.0, -ride / 2), (0.022, 0.016, ride + 0.03))
    # steering arm: a flat plate just under the rod end, from the back of the boss to a rounded end
    z = arm[2] - 0.0085
    s = np.array([0.0, -0.010, z])
    e = np.array([arm[0], arm[1], z])
    d = e - s
    length = float(np.linalg.norm(d))
    plate = ml.box((0, 0, 0), (length, 0.018, 0.006)).transformed(
        ml.frame((s + e) / 2, d, (-d[1], d[0], 0.0)))
    tip = ml.lathe([(0.0, -0.003), (0.0095, -0.003), (0.0095, 0.003), (0.0, 0.003)], 16, origin=tuple(e),
                   axis=(0, 0, 1), closed_ends=True)
    body = ml.merge([boss, stub, gusset, plate, tip])
    body.sharp_angle = 50
    bolt = ml.merge([
        ml.lathe([(0.0, -0.02), (0.004, -0.02), (0.004, 0.012), (0.0, 0.012)], 8, origin=tuple(arm), axis=(0, 0, 1),
                 closed_ends=True),
        ml.lathe([(0.0, 0.006), (0.0058, 0.006), (0.0058, 0.012), (0.0, 0.012)], 6, origin=tuple(arm), axis=(0, 0, 1),
                 closed_ends=True),
        ml.lathe([(0.0, -0.021), (0.0058, -0.021), (0.0058, -0.0145), (0.0, -0.0145)], 6, origin=tuple(arm),
                 axis=(0, 0, 1), closed_ends=True),
    ])
    bolt.smooth = False
    return body, bolt


def _rod_end_eye(c):
    """Rod end eye (heim joint) on a vertical bolt at c: a ring round the ball."""
    return ml.lathe([(0.004, -0.0045), (0.011, -0.0045), (0.011, 0.0045), (0.004, 0.0045), (0.004, -0.0045)], 16,
                    origin=tuple(c), axis=(0, 0, 1))


TIE_ROD_STUB = 0.078        # the inner half's threaded rod, out from the plate's rod end
TIE_ROD_GAP = 0.042         # the outer half's tube stops this short of the plate's rod end


def _steering_linkage(sc, col_base, d):
    """Column foot, the plate on it and the two tie rods out to the steering arms
    (steering-pedals-front-axle.jpg, pedals-brake-master.jpg). The plate and each tie rod hang on their own
    nodes (STEER_PLATE, TIEROD_LF/RF) so the BeamNG flexbodies can follow the steering rack and arms.

    Each tie rod is two pieces, as the real one (a tube with a threaded rod and rod end screwed into each
    end): the inner half (TIEROD_xx: rod end on the plate and a short threaded rod) and the outer half
    (TIEROD_xx_OUT: rod end on the steering arm, lock nut and the tube, its end TIE_ROD_GAP short of the
    plate's rod end, over the threaded rod). In AC the plate stays on the chassis (hung on the steering wheel
    node it was thrown forward past the nose in game) and the outer half on the knuckle is aimed at the
    plate's rod end (export_ac), so through the lock (+/-25 deg) the tube slides +/-27 mm along the threaded
    rod and swings up to 3 deg round it: the 4.0 mm clearance between them hides that up to the stub's end."""
    # lower bearing block bolted to the floor plate (stays with the frame)
    sc.add("steer_bearing", "black_anodized", ml.box(col_base + (0, 0, -0.006), (0.05, 0.034, 0.028)))
    plate_c = np.array([0.0, col_base[1], COL_PLATE_Z])
    sc.node("STEER_PLATE", "BODY", ml.translate(plate_c))
    # the plate: round the column's foot and back under it to the two rod ends, side by side
    holes = [np.array([sx * TIE_ROD_IN_X, -TIE_ROD_IN_BACK, 0.0]) for sx in (1, -1)]
    back = TIE_ROD_IN_BACK + 0.016
    tab = ml.box((0, (0.016 - back) / 2, 0), (2 * TIE_ROD_IN_X + 0.026, 0.016 + back, 0.006))
    tab.modifiers = [("bevel", 0.004, 2)]
    plate = ml.merge([tab]
                     + [ml.lathe([(0.0, -0.003), (0.0105, -0.003), (0.0105, 0.003), (0.0, 0.003)], 16, origin=tuple(h),
                                 axis=(0, 0, 1), closed_ends=True) for h in holes]
                     + [ml.lathe([(0.016, -0.004), (0.016, 0.026)], 16, origin=(0, 0, -0.012 + 0.003),
                                 axis=tuple(-d), closed_ends=True)])
    plate.sharp_angle = 50
    sc.add("steer_plate", "black_anodized", plate, "STEER_PLATE")
    bolts = ml.merge([ml.lathe([(0.0, -0.006), (0.004, -0.006), (0.004, 0.02), (0.0, 0.02)], 8, origin=tuple(h),
                               axis=(0, 0, 1), closed_ends=True) for h in holes]
                     + [ml.lathe([(0.0, 0.014), (0.007, 0.014), (0.007, 0.02), (0.0, 0.02)], 6, origin=tuple(h),
                                 axis=(0, 0, 1), closed_ends=True) for h in holes])
    bolts.smooth = False
    sc.add("steer_plate_bolts", "steel", bolts, "STEER_PLATE")
    # tie rods: rod ends on top of the plate and on the steering arms, a tube with a hex at its outer end,
    # the threaded rod into it with a lock nut by the outer rod end
    for side in (1, -1):
        key = "RF" if side > 0 else "LF"
        p_in = plate_c + holes[0 if side > 0 else 1] + (0, 0, 0.0075)
        p_out = np.array(STEER_ARM_END) * (side, 1, 1)
        v = p_out - p_in
        length = float(np.linalg.norm(v))
        up = np.cross((0, 0, 1.0), v)
        # node x along each half, from its own rod end towards the other (export_ac aims it there)
        sc.node("TIEROD_" + key, "BODY", ml.frame(p_in, v, up))
        sc.node("TIEROD_%s_OUT" % key, "BODY", ml.frame(p_out, -v, -up))
        stub = ml.lathe([(0.0, 0.009), (0.0065, 0.009), (0.0065, 0.033), (0.0085, 0.033), (0.0085, 0.039),
                         (0.0045, 0.039), (0.0045, TIE_ROD_STUB), (0.0, TIE_ROD_STUB)],
                        12, origin=(0, 0, 0), axis=(1.0, 0, 0), closed_ends=True)
        inner = ml.merge([stub, _rod_end_eye((0, 0, 0))])
        inner.sharp_angle = 50
        sc.add("tie_rod_" + key, "steel", inner, "TIEROD_" + key)
        t_end = length - TIE_ROD_GAP
        tube = ml.lathe([(0.0, 0.009), (0.0065, 0.009), (0.0065, 0.033), (0.0095, 0.033), (0.0095, 0.045),
                         (0.0085, 0.045), (0.0085, t_end), (0.0, t_end)],
                        12, origin=(0, 0, 0), axis=(1.0, 0, 0), closed_ends=True)
        outer = ml.merge([tube, _rod_end_eye((0, 0, 0))])
        outer.sharp_angle = 50
        sc.add("tie_rod_out_" + key, "steel", outer, "TIEROD_%s_OUT" % key)


_STRETCH_PARTS = ("frame_rail", "frame_inner", "floor", "protector", "sidepod", "pod_top", "pod_caps", "pod_inner",
                  "nerf", "fuel_line", "brake_line")


def _stretch(sc):
    """Apply warp_y: long parts crossing the middle stretch, everything else (and every node) moves whole."""
    seen = set()
    for part in sc.parts:
        v = part.mesh.verts
        if part.node != "BODY" or id(v) in seen:
            continue
        seen.add(id(v))
        if part.name.startswith(_STRETCH_PARTS):
            v[:, 1] = warp_y(v[:, 1])
        else:
            c = (v[:, 1].min() + v[:, 1].max()) / 2
            v[:, 1] += float(warp_y(c)) - c
    for n in sc.nodes:
        if n.parent == "BODY":
            n.matrix[1, 3] = float(warp_y(n.matrix[1, 3]))
    for w in sc.info.get("wheels", {}).values():
        w["center"] = warp_pt(w["center"])
    sw = sc.info.get("steer_wheel")
    if sw:
        sw["center"], sw["column_base"] = list(warp_pt(sw["center"])), list(warp_pt(sw["column_base"]))
    seat = sc.info.get("seat")
    if seat:
        seat["bottom"], seat["top"] = list(warp_pt(seat["bottom"])), list(warp_pt(seat["top"]))


# ------------------------------------------------------------------- parts
def _steering_wheel(sc, style):
    n = "STEER_HR"
    if style == "racing":
        # rounded-rectangle wheel, red top grip
        k = 48
        ring = ml.superellipse(0, 0, 0.160, 0.125, n=3.2, k=k, plane="xz")[:, [0, 2, 1]]
        ring[:, 2] = 0
        top = [i for i in range(k) if ring[i, 1] > 0.07]
        rest = [i for i in range(k) if ring[i, 1] <= 0.07]
        full_closed = ml.sweep(ring, ml.circle(0.016, 12), closed_path=True)
        # split faces by ring index
        faces_top, faces_rest, uv_t, uv_r = [], [], [], []
        for f, uv in zip(full_closed.faces, full_closed.uvs):
            ring_i = f[0] // 12
            (faces_top if ring_i in top else faces_rest).append(f)
            (uv_t if ring_i in top else uv_r).append(uv)
        sc.add("steer_grip_red", "grip_red", ml.Mesh(full_closed.verts, faces_top, uv_t), n)
        sc.add("steer_grip", "grip_black", ml.Mesh(full_closed.verts, faces_rest, uv_r), n)
        spokes = [ml.tube([(0, 0, 0.0), (sx * 0.13, sy, -0.004)], 0.011, 8) for sx, sy in ((1, 0.02), (-1, 0.02))]
        spokes.append(ml.tube([(0, 0, 0.0), (0, -0.11, -0.004)], 0.011, 8))
        sc.add("steer_spokes", "black_anodized", ml.merge(spokes), n)
    else:
        _steering_wheel_rental(sc, n)
        return
    hub = ml.lathe([(0, -0.035), (0.032, -0.035), (0.032, 0.008), (0.024, 0.012), (0, 0.012)], 20,
                   axis=(0, 0, 1))
    sc.add("steer_hub", "aluminium" if style == "racing" else "black_anodized", hub, n)
    logo = ml.box((0, 0, 0.013), (0.045, 0.030, 0.002))
    ml.project_uv(logo, (0, 0, 0), (1, 0, 0), (0, 1, 0), (-0.0225, 0.0225), (-0.015, 0.015), (0, 0, 1, 1))
    sc.add("steer_logo", "logo_badge", logo, n)


# rental steering wheel (steering-wheel.jpg, seat-cockpit.jpg), in the wheel's own frame: +y to the top, +z
# towards the driver, centre on the column. A rounded rectangle ~0.31 x 0.25 m on the grip's centre line,
# flat across the top and bottom and a little narrower at the bottom; a thick oval suede grip in four pieces
# split by black collars at the corners, a moulded block where the top spoke meets it; one flat black plate
# with a spoke straight up and two to the lower sides, drilled through along each spoke, three aluminium
# bolts in a triangle at the hub. (It was a thin round ring of 48 segments with the spokes upside down.)
SW_HALF = (0.155, 0.132)          # rim centre line half width (at the sides' widest) / height
SW_GRIP = (0.0175, 0.0135)        # grip section half sizes: in the wheel's plane / along the column
SW_SPOKES = ((90, 2), (207, 3), (333, 3))      # spoke angle (deg from +x) and its lightening holes
SW_TOP, SW_BOTTOM = 0.70, 0.52    # the straight top and bottom grips' half lengths, as fractions of SW_HALF[0]


def _sw_path(k=160):
    """Rim centre line: closed (k, 3) points evenly spaced, counter-clockwise from +x.

    The rental wheel (515417588, steering-wheel.jpg): a straight top grip ~70% of the width and a narrower straight
    bottom one (~half), joined by sides bowed right out in one long arc each, widest a little above the middle, with
    tight rounded corners where they meet. (A superellipse made the sides flat-ish and the top round.)"""
    a, b = SW_HALF
    # each side an elliptical arc through the corner points, its centre ~1 cm above the middle
    p_top, p_bot = (1 - SW_TOP ** 2) ** 0.5, (1 - SW_BOTTOM ** 2) ** 0.5
    c = 2 * b / (p_top + p_bot)
    y0 = b - p_top * c
    ys = np.linspace(b, -b, 40)
    side = [(a * math.sqrt(max(0.0, 1 - ((y - y0) / c) ** 2)), y, 0.0) for y in ys]
    loop = [(0.0, b, 0.0)] + side + [(0.0, -b, 0.0)] + [(-x, y, z) for x, y, z in reversed(side)] + [(0.0, b, 0.0)]
    pts = ml.fillet(loop, 0.028, steps=8)[:-1, :2]
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s_len = np.r_[0, np.cumsum(seg)]
    # start at +x (the right side's widest point) and run counter-clockwise (up first)
    i0 = int(np.argmax(pts[:, 0]))
    pts = np.roll(pts, -i0, axis=0)
    if pts[1, 1] < pts[0, 1]:
        pts = np.r_[pts[:1], pts[1:][::-1]]
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s_len = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, s_len[-1], k, endpoint=False)
    px = np.interp(u, s_len, np.r_[pts[:, 0], pts[0, 0]])
    py = np.interp(u, s_len, np.r_[pts[:, 1], pts[0, 1]])
    return np.c_[px, py, np.zeros(k)]


def _newell(p):
    n = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
    return n


def _sw_sweep(path, half, closed, n_sec=24, expo=2.4):
    """Oval section swept along a path in the wheel's plane (counter-clockwise): the section's first axis in
    the plane, out from the wheel's centre, the second along the column."""
    m = len(path)
    tang = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    if not closed:
        tang[0], tang[-1] = path[1] - path[0], path[-1] - path[-2]
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    out = np.c_[tang[:, 1], -tang[:, 0], np.zeros(m)]          # clockwise of a ccw tangent: outwards
    zz = np.array([0, 0, 1.0])
    sec = [(half[0] * math.copysign(abs(math.cos(q)) ** (2 / expo), math.cos(q)),
            half[1] * math.copysign(abs(math.sin(q)) ** (2 / expo), math.sin(q)))
           for q in np.linspace(0, 2 * math.pi, n_sec, endpoint=False)]
    verts = np.array([p + o * a_ + zz * c_ for p, o in zip(path, out) for a_, c_ in sec])
    faces, uvs = [], []
    rings = m if closed else m - 1
    for i in range(rings):
        i2 = (i + 1) % m
        for j in range(n_sec):
            j2 = (j + 1) % n_sec
            faces.append((i * n_sec + j, i2 * n_sec + j, i2 * n_sec + j2, i * n_sec + j2))
            uvs.append(((j / n_sec, i / m), (j / n_sec, (i + 1) / m), ((j + 1) / n_sec, (i + 1) / m),
                        ((j + 1) / n_sec, i / m)))
    caps = []
    if not closed:
        caps = [tuple(j for j in range(n_sec)), tuple((m - 1) * n_sec + j for j in range(n_sec))]
    # the sides face out of the section's centre line, the end caps out along the path
    fixed = []
    for f in faces:
        mid = verts[list(f)].mean(0)
        i = f[0] // n_sec
        fixed.append(f if _newell(verts[list(f)]) @ (mid - path[i]) > 0 else tuple(reversed(f)))
    for f, d in zip(caps, (-1, 1)):
        t_ = tang[0] if d < 0 else tang[-1]
        fixed.append(f if _newell(verts[list(f)]) @ (t_ * d) > 0 else tuple(reversed(f)))
        uvs.append(tuple((0.5, 0.5) for _ in f))
    return ml.Mesh(verts, fixed, [tuple(u) if len(u) == len(f) else tuple((0.5, 0.5) for _ in f)
                                  for f, u in zip(fixed, uvs)])


def _sw_plate(path):
    """The flat spoke plate (z = 0 here) drilled through along each spoke: a star-shaped middle round the
    hub with concave fillets between the spokes' roots, then each spoke a run of cells, the drilled ones an
    annulus from the cell's outline to its hole. Faces towards +z."""
    ang_path = np.arctan2(path[:, 1], path[:, 0])
    r_path = np.hypot(path[:, 0], path[:, 1])
    r0 = 0.040                                    # the middle's radius at the spokes' roots
    verts, faces = [np.zeros(3)], []

    def add(p):
        verts.append(np.array([p[0], p[1], 0.0]))
        return len(verts) - 1

    def width(r):                                 # spoke width tapering from the hub to the rim
        return 0.044 - 0.016 * min(max((r - r0) / 0.09, 0.0), 1.0)

    def frame(deg):
        a = math.radians(deg)
        return np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])

    # the middle
    sp = sorted(SW_SPOKES)
    ring, roots = [], {}
    for idx, (deg, _) in enumerate(sp):
        u, v = frame(deg)
        h = width(r0) / 2
        right, left = u * r0 - v * h, u * r0 + v * h
        roots[deg] = (len(ring), len(ring) + 1)
        ring += [right, left]
        u2, v2 = frame(sp[(idx + 1) % len(sp)][0])
        nxt_right = u2 * r0 - v2 * h
        a1, a2 = math.atan2(left[1], left[0]), math.atan2(nxt_right[1], nxt_right[0])
        while a2 <= a1:
            a2 += 2 * math.pi
        for t in np.linspace(0, 1, 9)[1:-1]:
            aa = a1 + (a2 - a1) * t
            ring.append(np.array([math.cos(aa), math.sin(aa)]) * (r0 - 0.012 * math.sin(math.pi * t)))
    ids = [add(p) for p in ring]
    for i in range(len(ids)):
        faces.append((0, ids[i], ids[(i + 1) % len(ids)]))
    # the spokes: cells along each, every other one drilled. Each cut across the spoke keeps its points
    # (right to left), shared by the cells either side of it
    for deg, n_holes in SW_SPOKES:
        u, v = frame(deg)
        near = np.argmin(np.abs((ang_path - math.radians(deg) + math.pi) % (2 * math.pi) - math.pi))
        r_end = r_path[near] - 0.004               # inside the grip
        wts = [1.0 if c % 2 == 0 else 1.7 for c in range(2 * n_holes + 1)]
        cuts = r0 + (r_end - r0) * np.r_[0, np.cumsum(wts)] / sum(wts)
        corner = [(ids[roots[deg][0]], ids[roots[deg][1]])]
        for r in cuts[1:]:
            corner.append((add(u * r - v * width(r) / 2), add(u * r + v * width(r) / 2)))
        edge = [[a, b] for a, b in corner]
        for c in range(1, len(cuts) - 2, 2):        # drilled cells
            ra, rb = cuts[c], cuts[c + 1]
            quad = [verts[corner[c][0]][:2], verts[corner[c + 1][0]][:2], verts[corner[c + 1][1]][:2],
                    verts[corner[c][1]][:2]]
            qid = [corner[c][0], corner[c + 1][0], corner[c + 1][1], corner[c][1]]
            cen = u * (ra + rb) / 2
            hole = min(width(rb) / 2, (rb - ra) / 2) * 0.66
            angs = set(np.round(np.linspace(0, 2 * math.pi, 20, endpoint=False), 9))
            for q in quad:
                d = q - cen
                angs.add(round(math.atan2(d @ v, d @ u) % (2 * math.pi), 9))
            angs = sorted(angs)
            outline, inner = [], []
            for ang in angs:
                dvec = u * math.cos(ang) + v * math.sin(ang)
                best = None
                for e in range(4):                  # ray from the hole's centre to the cell's edges
                    p1, p2 = quad[e], quad[(e + 1) % 4]
                    m_ = np.array([dvec, p1 - p2]).T
                    if abs(np.linalg.det(m_)) < 1e-12:
                        continue
                    tt, ss = np.linalg.solve(m_, p1 - cen)
                    if tt > 0 and -1e-7 <= ss <= 1 + 1e-7 and (best is None or tt < best[0]):
                        best = (tt, e)
                pt = cen + dvec * best[0]
                hit = [k_ for k_ in range(4) if np.linalg.norm(quad[k_] - pt) < 1e-7]
                if hit:
                    vid = qid[hit[0]]
                else:
                    vid = add(pt)
                    if best[1] == 3:
                        edge[c].append(vid)
                    elif best[1] == 1:
                        edge[c + 1].append(vid)
                outline.append(vid)
                inner.append(add(cen + dvec * hole))
            for q in range(len(angs)):
                q2 = (q + 1) % len(angs)
                faces.append((outline[q], outline[q2], inner[q2], inner[q]))
        for e in edge:
            e.sort(key=lambda i: verts[i][:2] @ v)  # right -> left
        for c in range(0, len(cuts) - 1, 2):        # solid cells
            faces.append(tuple(edge[c] + edge[c + 1][::-1]))
    mesh = ml.Mesh(np.array(verts), faces, [tuple((0.5, 0.5) for _ in f) for f in faces], smooth=False)
    mesh.faces = [f if _newell(mesh.verts[list(f)])[2] > 0 else tuple(reversed(f)) for f in mesh.faces]
    return mesh


def _steering_wheel_rental(sc, n):
    path = _sw_path()
    grip = _sw_sweep(path, SW_GRIP, closed=True)
    grip.modifiers = [("subsurf", 1)]
    sc.add("steer_grip", "grip_black", grip, n)
    # black collars splitting the grip at the four corners, and the block over the top spoke
    k = len(path)
    collars = []
    for cx, cy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
        # at the ends of the straight top and bottom grips (515417588)
        tgt = np.array([cx * SW_HALF[0] * (SW_TOP if cy > 0 else SW_BOTTOM), cy * SW_HALF[1]])
        i = int(np.argmin(np.linalg.norm(path[:, :2] - tgt, axis=1)))
        # a straight sleeve ~1.6 cm long along the rim there (swept along the rim's own points it folded through itself
        # on the tight corners and came out inside out)
        t = path[(i + 1) % k] - path[(i - 1) % k]
        t /= np.linalg.norm(t)
        seg = np.array([path[i] - t * 0.008, path[i], path[i] + t * 0.008])
        collars.append(_sw_sweep(seg, (SW_GRIP[0] * 1.10, SW_GRIP[1] * 1.12), closed=False, n_sec=20))
    sc.add("steer_collars", "black_plastic", ml.merge(collars), n)
    top = path[(path[:, 1] > 0)]
    top_y = float(top[np.argmin(np.abs(top[:, 0])), 1])
    block = ml.box((0.0, top_y - 0.018, -0.002), (0.046, 0.034, 0.026))
    block.modifiers = [("bevel", 0.006, 2)]
    sc.add("steer_grip_block", "grip_black", block, n)
    # the spoke plate, 4 mm, set a little towards the column
    plate = _sw_plate(path).transformed(ml.translate((0, 0, -0.008)))
    plate.modifiers = [("solidify", 0.004)]
    sc.add("steer_spokes", "black_anodized", plate, n)
    # countersunk aluminium bolts with hex sockets
    bolts, sockets = [], []
    for a in (90, 210, 330):
        c = (0.022 * math.cos(math.radians(a)), 0.022 * math.sin(math.radians(a)), -0.004)
        bolts.append(ml.lathe([(0, 0), (0.0085, 0), (0.0085, 0.0015), (0.0065, 0.0035), (0, 0.0035)], 20,
                              origin=c, axis=(0, 0, 1), closed_ends=True))
        sock = ml.lathe([(0, 0.0036), (0.0032, 0.0036)], 6, origin=c, axis=(0, 0, 1))
        sock.faces = [f if _newell(sock.verts[list(f)])[2] > 0 else tuple(reversed(f)) for f in sock.faces]
        sockets.append(sock)
    sc.add("steer_bolts", "aluminium", ml.merge(bolts), n)
    sc.add("steer_bolt_sockets", "engine_black", ml.merge(sockets), n)
    # boss on the column just behind the plate (which runs 8-12 mm behind the rim's mid-plane)
    hub = ml.lathe([(0, -0.045), (0.024, -0.045), (0.024, -0.016), (0.030, -0.014), (0.030, -0.0125), (0, -0.0125)], 24,
                   axis=(0, 0, 1), closed_ends=True)
    sc.add("steer_hub", "black_anodized", hub, n)


SEAT_BACK_TOP = 0.41       # seat back top over the frame: ~0.40 m off the ground, a little over the rear deck (rear-3q.jpg,
                           # side-right.jpg: low and laid back; at 0.48 it stood tall and upright over the pods).
                           # The studio rear shot has its top ~8 cm over the rear deck: 0.43, laid back as before


def _seat_z(z):
    """The seat was drawn with its back topping out at 0.56: squash everything above the pan to SEAT_BACK_TOP."""
    return z if z <= 0.08 else 0.08 + (z - 0.08) * (SEAT_BACK_TOP - 0.08) / (0.56 - 0.08)


def nrm_i(nrm, i):
    """The seat spine's in-plane normal at row i as a 3D (x, y, z) vector."""
    return np.array([0.0, nrm[i, 0], nrm[i, 1]])


def _seat(sc, chassis):
    """Rental bucket seat (the studio cockpit shots 515935974 / 515330967, seat-cockpit.jpg, seat-adjust-bracket.jpg):
    a deep moulded bucket laid well back, its pan low with a raised thigh lip at the front, hip walls flaring out
    towards a thick rolled rim, the back's sides curling forward round the shoulders, two bolts with big washers
    in the back and one in each hip wall. The back's top stays where the side and rear studio shots put it (~0.44 m
    off the ground, ~0.36 m behind the middle); the pan came forward under it, so the back lies ~36 deg from
    upright (it was ~24 deg: too upright and too plain a shell)."""
    # spine (drawn y, z over the frame, written in the unsquashed space _seat_z maps to SEAT_BACK_TOP)
    # (the back's top ~0.36 m behind the middle, real, where the side and rear studio shots and the outdoor shot put
    # it; the hips come forward with it so the back keeps its ~30 deg lean)
    spine = np.array([(0.235, 0.107), (0.17, 0.064), (0.08, 0.043), (0.0, 0.041), (-0.05, 0.066),
                      (-0.105, 0.14), (-0.165, 0.248), (-0.23, 0.356), (-0.285, 0.464), (-0.335, 0.56)])
    spine[:, 1] = [_seat_z(z) for z in spine[:, 1]]
    spine = ml.catmull(spine, 4)
    m = len(spine)
    tang = np.gradient(spine, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.c_[tang[:, 1], -tang[:, 0]]            # rotate -90: points up/forward (into the seat)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))]
    s /= s[-1]
    # half widths along the spine: narrow between the thighs at the front lip, widest at the hips, the back a
    # little narrower, flaring again at the shoulders
    width = np.interp(s, [0, 0.12, 0.35, 0.55, 0.8, 1.0], [0.150, 0.170, 0.192, 0.185, 0.188, 0.200])
    # wall heights from the rim line (seat-cockpit.jpg, seat-adjust-bracket.jpg): a deep tub, the rim ~0.24 m
    # over the frame round the hips and sloping down to the thigh lip; along the back the sides curl forward
    # ~20 cm low down, closing to ~11 cm at the top (fixed wall depths left a long shallow chute)
    z_rim = np.interp(-spine[:, 0], [-0.235, -0.17, -0.08, 0.0, 0.07, 0.13], [0.11, 0.15, 0.19, 0.225, 0.245, 0.265])
    cap = np.interp(s, [0, 0.45, 0.6, 0.8, 1.0], [0.25, 0.25, 0.20, 0.14, 0.085])   # small wings at the top
    rl = np.maximum(0.03, (z_rim - spine[:, 1]) / np.maximum(nrm[:, 1], 0.25))
    back = _smoothstep_v(0.40, 0.55, s)      # up the back the sides are the cap's depth (it is over the rim line)
    wall = np.minimum(cap, rl) * (1 - back) + cap * back
    k = np.ones(7) / 7.0                       # smoothed along the seat: no step where the hips meet the back
    wall = np.convolve(np.r_[[wall[0]] * 3, wall, [wall[-1]] * 3], k, mode="valid")
    flare = np.interp(s, [0, 0.35, 1.0], [0.10, 0.28, 0.20])     # walls lean out towards the rim
    ts = np.linspace(-1, 1, 21)
    secs = []
    for i in range(m):
        sec = []
        expo = 3.4 + (1.35 - 3.4) * _smoothstep(0.30, 0.60, s[i])      # the back a round U
        for t in ts:
            depth = wall[i] * abs(t) ** expo
            x = width[i] * math.sin(t * math.pi / 2) + math.copysign(flare[i] * depth, t)
            y = spine[i, 0] + nrm[i, 0] * depth
            z = spine[i, 1] + nrm[i, 1] * depth
            sec.append((x, y, z))
        secs.append(sec)
    shell = ml.loft(secs, closed=False)
    shell.modifiers = [("solidify", 0.005), ("subsurf", 1)]
    shell.sharp_angle = 80
    # the rolled rim: a thick round lip all along the shell's free edge (sides, the back's top, the thigh lip)
    edge = ([secs[i][-1] for i in range(m)] + [secs[-1][j] for j in range(len(ts) - 2, 0, -1)]
            + [secs[i][0] for i in range(m - 1, -1, -1)] + [secs[0][j] for j in range(1, len(ts) - 1)])
    edge = np.array(edge)
    keep = np.r_[True, np.linalg.norm(np.diff(edge, axis=0), axis=1) > 0.004]
    rim = ml.sweep(ml.catmull(edge[keep], 2, closed=True), ml.circle(0.011, 10), closed_path=True)
    # the seat sits nearly in the middle and is yawed a little towards the centre, facing the steering wheel
    pivot = np.array([0.0, -0.05, 0.0])
    sx, yaw = sc.params.get("seat_x", SEAT_OFFSET_X), sc.params.get("seat_yaw", SEAT_YAW_DEG)
    place = ml.translate(pivot + (sx, 0, 0)) @ ml.rot_axis((0, 0, 1), -yaw) @ ml.translate(-pivot)
    sc.add("seat", "seat", shell.transformed(place))
    sc.add("seat_rim", "seat", rim.transformed(place))

    def on_seat(pt):
        return (place @ np.r_[pt[0], pt[1], _seat_z(pt[2]), 1.0])[:3]

    def placed(pt):                                   # a point given in final (squashed) seat coordinates
        return (place @ np.r_[pt, 1.0])[:3]

    sc.info["seat"] = dict(bottom=on_seat((0, -0.02, 0.041)).tolist(), top=on_seat((0, -0.335, 0.56)).tolist())
    # mounting bolts with big washers: two in the back, one low in each hip wall (seat-cockpit.jpg)
    bolts = []
    for sx, i in ((-0.06, int(m * 0.72)), (0.06, int(m * 0.72)), (-0.15, int(m * 0.40)), (0.15, int(m * 0.40))):
        j = int(np.argmin([abs(q[0] - sx) for q in secs[i]]))
        if abs(sx) > 0.1:
            j = int(np.argmin([abs(q[0] - sx) + (0 if q[2] > secs[i][len(ts) // 2][2] + 0.05 else 1) for q in secs[i]]))
        a, b = np.array(secs[i][j]), np.array(secs[i][j + (1 if j < len(ts) - 1 else -1)])
        c = np.array(secs[min(i + 1, m - 1)][j])
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n) + 1e-12
        mid = np.array(secs[i][len(ts) // 2])
        if n @ (mid + nrm_i(nrm, i) * 0.2 - a) < 0:
            n = -n
        o = a + n * 0.0005
        washer = ml.lathe([(0.0, 0.0), (0.017, 0.0), (0.017, 0.0025), (0.0, 0.0025)], 16, origin=tuple(o),
                          axis=tuple(n), closed_ends=True)
        head = ml.lathe([(0.0, 0.002), (0.008, 0.002), (0.007, 0.007), (0.0, 0.008)], 10, origin=tuple(o),
                        axis=tuple(n), closed_ends=True)
        bolts += [ml.orient_outward(washer, o + n * 0.00125), ml.orient_outward(head, o + n * 0.004)]
    sc.add("seat_bolts", "steel", ml.merge(bolts).transformed(place))
    # the seat's cradle and its struts (515935974, 514248494, seat-cockpit.jpg): a tube hoop hugging the rolled rim
    # ~3 cm under it on the outside, from each hip's front round behind the back, held off the frame by uprights:
    # one behind each back corner from the inner rails, one under each hip from the side rails. (The struts ran
    # from the rails into the shell's back and pan, and the hoop floated away from the rim.)
    i0 = int(np.searchsorted(s, 0.14))
    cradle_l, cradle_r = [], []
    for i in range(i0, m):
        for j, out in ((0, cradle_l), (len(ts) - 1, cradle_r)):
            e = np.array(secs[i][j])
            w = e - np.array(secs[i][j - 1 if j else 1])           # up the wall to the rim
            w /= np.linalg.norm(w) + 1e-12
            out.append(e - w * 0.032 + (math.copysign(0.013, e[0]), 0.0, 0.0))
    back = []
    for j in range(len(ts) - 3, 2, -1):                            # behind the back's top, right to left
        e = np.array(secs[-1][j])
        back.append(e - np.r_[0.0, nrm[-1]] * 0.018 - (0.0, 0.0, 0.032) * np.array([0, 0, 1.0]))
    path = np.array(cradle_r + back + cradle_l[::-1])
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 0.012]
    path = ml.catmull(path[keep], 3)
    path = np.array([placed(q) for q in path])
    if sc.family == "rental":
        sc.add("seat_frame", "protector", ml.tube(path, 0.011, 10))
    # uprights from the frame to the cradle
    for side in (1, -1):
        sfx = "_R" if side > 0 else "_L"
        hoop = path[path[:, 0] * side > 0]
        top_b = hoop[np.argmin(np.linalg.norm(hoop[:, :2] - (side * 0.15, -0.33), axis=1))]
        top_h = hoop[np.argmin(np.linalg.norm(hoop[:, :2] - (side * 0.25, -0.06), axis=1))]
        sc.add("seat_strut_hi" + sfx, chassis,
               ml.tube([(side * 0.212, -0.40, 0.056), (side * 0.205, -0.385, 0.14), top_b - (0, 0, 0.006)], 0.009, 8, 0.05))
        sc.add("seat_strut_lo" + sfx, chassis,
               ml.tube([(side * 0.29, -0.05, 0.050), (side * 0.285, -0.055, 0.10), top_h - (0, 0, 0.006)], 0.009, 8, 0.04))
    if sc.family == "rental":
        # slotted seat-adjust strap on the left (seat-adjust-bracket.jpg)
        a, b = placed((-0.24, -0.12, 0.20)), np.array([-0.40, -0.40, 0.31])
        mid = (a + b) / 2
        d = (b - a) / np.linalg.norm(b - a)
        strap = ml.box((0, 0, 0), (0.035, np.linalg.norm(b - a), 0.004))
        strap = strap.transformed(ml.frame(mid, np.cross(d, (0, 0, 1)), d))
        sc.add("seat_strap", "black_anodized", strap)


def _nose(sc, variant):
    half = 0.52

    def sec(x):
        a = abs(x) / half
        yf = 1.035 - 0.12 * a ** 2
        yb = 0.79 - 0.13 * a ** 2.2
        zt = 0.165 + 0.11 * a ** 4
        zc = 0.035 + 0.01 * a
        ym = (yb + yf) / 2
        # back lip, flat top sloping down to a crisp front edge, near-vertical front face
        return [(x, yb + 0.015, zc + 0.03), (x, yb, zt - 0.05), (x, yb + 0.004, zt - 0.012), (x, yb + 0.02, zt),
                (x, ym, zt - 0.012), (x, yf - 0.04, zt - 0.04), (x, yf - 0.012, zt - 0.055), (x, yf, zt - 0.075),
                (x, yf + 0.004, zc + 0.05), (x, yf - 0.004, zc + 0.01), (x, yf - 0.025, zc)]

    xs = np.r_[np.linspace(-half, -0.3, 6), np.linspace(-0.26, 0.26, 9), np.linspace(0.3, half, 6)]
    m, arc = _section_loft(sec, xs, cap=True)
    vrow = arc.reshape(len(xs), -1).mean(0)
    vrow = vrow[-1] - vrow          # v grows upwards on the front face (rows[0] = back lip)
    _uv_from(m, np.repeat(-xs, 11), np.tile(vrow, len(xs)), ATLAS["nose"], (-half, half), (0.0, float(vrow[0])))
    m.modifiers = [("subsurf", 1), ("solidify", 0.004)]
    m.sharp_angle = 85
    sc.add("nose", "body_livery", m)
    sc.panels["nose"] = dict(u=(-half, half), v=(0.0, float(vrow[0])), rows=vrow.tolist())
    # mount brackets
    _sym_tube(sc, "nose_mount", "black_plastic", [(0.2, 0.90, 0.155), (0.2, 0.95, 0.16)], 0.008, 0)


def _column_z(y, col_base, sw_c):
    """Height of the steering column's axis at y."""
    t = (col_base[1] - y) / (col_base[1] - sw_c[1])
    return col_base[2] + t * (sw_c[2] - col_base[2])


def _steer_support_path(col_base, sw_c):
    """Rental steering support tube (right one; the left mirrors it): frame in front of the seat, up, then
    forward to the column 10 cm under the wheel (seat-cockpit.jpg)."""
    d = (col_base - sw_c) / np.linalg.norm(col_base - sw_c)        # down the column
    top = sw_c + d * 0.10
    return [(0.095, 0.20, 0.05), (0.085, 0.24, 0.20), (0.022, float(top[1]), float(top[2]))]


def _fuel_tank_back(col_base, sw_c, bottom, gap):
    """The rental tank's back, facing the driver: it fills the space between the two steering support
    tubes, its outline following them 2 cm inside (wide at the floor, narrowing as they lean in towards the
    column) and its back face in their plane, its top under the column like the rest of the tank. Built
    from horizontal slices; it runs forward into the tank's body."""
    path = np.array(_steer_support_path(col_base, sw_c))
    zs_p = path[:, 2]

    def tube_at(z):
        return np.interp(z, zs_p, path[:, 0]), np.interp(z, zs_p, path[:, 1])

    def col_y(z):         # where the column passes `gap` over height z
        return col_base[1] - (z + gap - col_base[2]) / (sw_c[2] - col_base[2]) * (col_base[1] - sw_c[1])

    # as high as the tubes and the column leave it 3.5 cm of depth
    zz = np.linspace(bottom + 0.1, zs_p[-1], 200)
    top_z = float(zz[np.nonzero([col_y(z) - tube_at(z)[1] - 0.004 >= 0.035 for z in zz])[0].max()])
    zs = np.r_[bottom, bottom + 0.004, np.linspace(bottom + 0.02, top_z - 0.03, 9), top_z - 0.012, top_z]
    secs = []
    for i, z in enumerate(zs):
        tx, ty = tube_at(min(z, top_z))
        hw = tx - 0.020
        y_r = ty + 0.004
        y_f = min(0.33, col_y(z))      # into the body, short of its front strap
        # rounded edges at the floor and along the top
        shrink = 1.0
        if i <= 1:
            shrink = 0.90 + 0.10 * i
        elif z > top_z - 0.031:
            shrink = 1.0 - 0.10 * (z - (top_z - 0.031)) / 0.031
        hw *= shrink
        depth = (y_f - y_r) / 2
        y_r += depth * (1 - shrink)
        sec = ml.superellipse(0.0, (y_r + y_f) / 2, hw, (y_f - y_r) / 2, n=4, k=32, y=z, plane="xz")
        secs.append(sec[:, [0, 2, 1]])          # slices across z: (x, y, z)
    back = ml.loft(secs, closed=True, cap_start=True, cap_end=True)
    # slices run upwards: the loft's winding comes out mirrored, so turn the faces outward
    v = back.verts
    f0 = back.faces[len(back.faces) // 3]
    nrm = np.cross(v[f0[1]] - v[f0[0]], v[f0[2]] - v[f0[0]])
    c = v.mean(0)
    if nrm @ (v[list(f0)].mean(0) - c) < 0:
        back.faces = [tuple(reversed(f)) for f in back.faces]
        back.uvs = [tuple(reversed(u)) for u in back.uvs]
    return back


def _fuel_tank(sc, chassis, variant, col_base, sw_c):
    """White plastic tank on the floor plate between the legs, under the steering column: a wedge whose
    top follows the column 3.5 cm below it, so the column, its support tubes and the tie rods all pass
    clear. Held by a clamp on the column, two short brackets to the steering supports and two straps
    over the top down to the floor plate. The rental feeds the Honda carburettor from the bottom."""
    y0, y1, bottom, gap = 0.305, 0.465, 0.042, 0.035

    def sec_dims(y):
        t = (y - y0) / (y1 - y0)
        end = min(t, 1 - t)
        shrink = 1.0 if end > 0.08 else 0.82 + 0.18 * end / 0.08      # rounded ends
        hw = (0.062 + 0.014 * math.sin(math.pi * t)) * shrink
        top = _column_z(y, col_base, sw_c) - gap
        mid, half = (top + bottom) / 2, (top - bottom) / 2 * (0.94 + 0.06 * shrink)
        return hw, mid, half

    ys = [y0, y0 + 0.006, y0 + 0.02, 0.35, 0.385, 0.42, y1 - 0.02, y1 - 0.006, y1]
    secs = []
    for y in ys:
        hw, mid, half = sec_dims(y)
        secs.append(ml.superellipse(0.0, mid, hw, half, n=4, k=32, y=y, plane="xz"))
    tank = ml.loft(secs, closed=True, cap_start=True, cap_end=True)
    if family(variant) == "rental":
        tank = ml.merge([tank, _fuel_tank_back(col_base, sw_c, bottom, gap)])
    sc.add("fuel_tank", "tank_plastic", tank)
    cap_y = 0.335
    cap_z = _column_z(cap_y, col_base, sw_c) - gap
    sc.add("fuel_tank_cap", "black_plastic", ml.lathe([(0, 0), (0.016, 0), (0.016, 0.022), (0, 0.022)], 16,
                                                      origin=(0.045, cap_y, cap_z - 0.002), axis=(0, 0, 1),
                                                      closed_ends=True))
    # clamp on the column with a tab down onto the tank top
    d = (sw_c - col_base) / np.linalg.norm(sw_c - col_base)
    cy = 0.40
    c = np.array([0.0, cy, _column_z(cy, col_base, sw_c)])
    tab_top, tab_bot = c[2] - 0.014, c[2] - gap
    mount = [ml.lathe([(0.016, -0.015), (0.016, 0.015)], 16, origin=tuple(c), axis=tuple(d), closed_ends=True),
             ml.box((0.0, cy, (tab_top + tab_bot) / 2), (0.03, 0.006, tab_top - tab_bot))]
    # brackets from the tank to the steering support tubes (the rental's tank back sits between them)
    if family(variant) != "rental":
        for sx in (1, -1):        # support tube (0.16, 0.50, 0.055) -> (0.03, 0.43, 0.30)
            t = (0.50 - 0.455) / 0.07
            sp = (sx * (0.16 - 0.13 * t), 0.455, 0.055 + 0.245 * t)
            hw, mid, half = sec_dims(0.455)
            mount.append(ml.tube([(sx * (hw - 0.01), 0.455, mid + half - 0.004), sp], 0.005, 8))
    sc.add("fuel_tank_mount", "black_anodized", ml.merge(mount))
    # rubber straps over the tank, bolted to the floor plate
    straps = []
    for y in (0.35, 0.43):
        hw, mid, half = sec_dims(y)
        arc = ml.superellipse(0.0, mid, hw + 0.004, half + 0.004, n=4, k=32, y=y, start=-10, end=190, plane="xz")
        pts = [(hw + 0.012, y, 0.037)] + [tuple(p) for p in arc] + [(-hw - 0.012, y, 0.037)]
        straps.append(ml.tube(pts, 0.0035, 6))
    sc.add("fuel_tank_straps", "black_plastic", ml.merge(straps))
    if family(variant) == "rental":
        pump = np.array(GX_PUMP)                                     # the pulse fuel pump (see _engine_gx)
        line = [(0.06, 0.33, 0.05), (0.10, 0.30, 0.046), (0.20, 0.22, 0.05), (0.30, 0.14, 0.10), (0.38, 0.02, 0.17),
                (pump[0] - 0.01, pump[1] + 0.05, pump[2] + 0.02), tuple(pump + (-0.022, 0.0, 0.006))]
        sc.add("fuel_line", "fuel_hose", ml.tube(line, 0.004, 8, 0.05))


def _panel_volume(spine_ctrl, width_tab, bulge_tab, depth_tab):
    """Front panel as a moulded plastic part with volume: the printed, convex front face and a back skin
    depth_tab metres behind it (deeper on the stem that wraps the column), closed round the edges and at
    both ends. Returns the mesh, the arclength of each section along the spine, the total length, the
    widths, the section size and how many of its points are on the front face."""
    spine = ml.catmull(np.array(spine_ctrl), 4)
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))]
    total = s_[-1]
    sn = s_ / total
    width = np.interp(sn, *width_tab)
    bulge = np.interp(sn, *bulge_tab)
    depth = np.interp(sn, *depth_tab)
    tang = np.gradient(spine, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])          # out of the printed face: forward and up
    if nrm[len(nrm) // 2, 0] < 0:
        nrm = -nrm
    ts = np.linspace(-1, 1, 13)

    def sec(i):
        y, z = spine[i]
        w, b, dp = width[i], bulge[i], depth[i]
        ny, nz = nrm[i]
        front = [(w * t, y + b * (1 - t * t) * ny, z + b * (1 - t * t) * nz) for t in ts]
        back = [((w - 0.004) * t, y + (0.35 * b * (1 - t * t) - dp) * ny, z + (0.35 * b * (1 - t * t) - dp) * nz)
                for t in ts[::-1]]
        edge_r = (w + 0.003, y - 0.5 * dp * ny, z - 0.5 * dp * nz)
        edge_l = (-w - 0.003, y - 0.5 * dp * ny, z - 0.5 * dp * nz)
        return front + [edge_r] + back + [edge_l]

    secs = [sec(i) for i in range(len(spine))]
    k = len(secs[0])
    m = ml.loft(secs, closed=True, cap_start=True, cap_end=True)
    return m, s_, total, width, k, len(ts)


def _panel_uvs(m, s_, width, k, nf, rect, u_range, v_range, plain):
    """Livery UVs on the front face (u = viewer left to right, v along the spine); every other face on a
    plain spot of the panel's base colour, so the back shows no mirrored print."""
    u0, v0, u1, v1 = rect

    def f(u, v):
        return (u0 + (u - u_range[0]) / (u_range[1] - u_range[0]) * (u1 - u0),
                v0 + (v - v_range[0]) / (v_range[1] - v_range[0]) * (v1 - v0))

    ts = np.linspace(-1, 1, nf)
    uvs = []
    for face in m.faces:
        cols = [vi % k for vi in face]
        if len(face) == 4 and max(cols) < nf and max(cols) - min(cols) == 1:
            uvs.append(tuple(f(-width[vi // k] * ts[vi % k], s_[vi // k]) for vi in face))
        else:
            uvs.append(tuple(f(*plain) for _ in face))
    m.uvs = uvs


def _front_panel(sc, variant):
    # spine from the nose up to the top of the number plate, leaning towards the driver
    m, s, total, width, k, nf = _panel_volume(
        [(0.86, 0.15), (0.78, 0.22), (0.66, 0.34), (0.56, 0.45), (0.50, 0.53), (0.47, 0.555)],
        ([0, 0.25, 0.45, 0.65, 0.85, 1.0], [0.085, 0.060, 0.075, 0.14, 0.155, 0.14]),
        ([0, 0.5, 1.0], [0.035, 0.03, 0.02]),
        ([0, 0.15, 0.35, 0.55, 0.8, 1.0], [0.010, 0.025, 0.04, 0.03, 0.022, 0.018]))
    _panel_uvs(m, s, width, k, nf, ATLAS["panel"], (-0.16, 0.16), (0.0, total), (0.155, total * 0.985))
    m.modifiers = [("subsurf", 2)]       # moulded plastic with rounded edges
    m.sharp_angle = 85
    sc.add("front_panel", "body_livery", m)
    sc.panels["panel"] = dict(u=(-0.16, 0.16), v=(0.0, float(total)),
                              width=[(float(a), float(b)) for a, b in zip(s, width)])
    sc.add("front_panel_bracket", "black_anodized",
           ml.tube([(0, 0.62, 0.30), (0, 0.55, 0.33)], 0.007, 8))


def _sidepods(sc, variant):
    y0, y1 = -0.34, 0.33

    def sec(y, side):
        t = (y - y0) / (y1 - y0)            # 0 rear .. 1 front
        x_out = POD_OUT_REAR - (POD_OUT_REAR - POD_OUT_FRONT) * t   # in line with the tyres' outer faces
        x_in = 0.45 + 0.03 * t
        z_top = 0.265 - 0.13 * t ** 1.2
        z_bot = 0.04
        pts = [(x_in + 0.02, y, z_bot + 0.02), (x_out - 0.03, y, z_bot), (x_out, y, z_bot + 0.03),
               (x_out + 0.005, y, (z_bot + z_top) / 2), (x_out - 0.005, y, z_top - 0.02), (x_out - 0.04, y, z_top),
               ((x_out + x_in) / 2, y, z_top - 0.015), (x_in, y, z_top - 0.035)]
        return [(side * a, b, c) for a, b, c in pts]

    for side, key in ((1, "pod_R"), (-1, "pod_L")):
        y_rear = RENTAL_POD_R_REAR if (family(variant) == "rental" and side > 0) else y0
        ys = np.linspace(y_rear, y1, 12)
        secs_fn = (lambda y, s=side: sec(y, s))
        m, arc = _section_loft(secs_fn, ys, cap=True)
        if side < 0:
            m.faces = [tuple(reversed(f)) for f in m.faces]
        # planar side projection; u runs rear->front on the right pod and front->rear
        # on the left pod so artwork reads correctly from either side
        n = len(ys) * 8
        ucoord = m.verts[:n, 1] * side
        _uv_from(m, ucoord, m.verts[:n, 2], ATLAS[key], (-0.36, 0.36), (0.02, 0.29))
        m.modifiers = [("subsurf", 2), ("solidify", 0.004)]
        m.sharp_angle = 85
        sc.add("sidepod_" + ("R" if side > 0 else "L"), "body_livery", m)
        sc.panels[key] = dict(u=(-0.36, 0.36), v=(0.02, 0.29), front=side,
                              top=[(float(y), float(sec(y, 1)[5][2])) for y in ys])



def _limit(prev, cur, nxt):
    """Cubic B-spline limit point: where a subdivision surface passes a control point."""
    return (np.asarray(prev) + 4 * np.asarray(cur) + np.asarray(nxt)) / 6.0


def _smoothstep_v(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


# the nose's centre back lip and the front panel's foot on it: 3.5 cm further forward than first drawn, so
# the bigger pedals can come forward under the panel's flared foot (protector-v1-top.jpg: the pedals sit
# between the nose's fins, beside the panel's foot)
NOSE_LIP_Y = 0.87


# Rental v2 nose wings (drawn y, z over the frame), from the paddock shots 514273209 (kart 1 from ahead) and 514277469
# (kart 1 from ahead and right) and the studio side shot: each wing a straight blade running diagonally in plan from
# the centre's front corner back and out to a tall squared-off tip just ahead of the front tyre's inner half. Its
# top edge rises in a straight line to the tip, with a flat strip on top; the inner face is upright, the outer face
# (the logo) leans out a little down to the bottom edge, which runs straight from the centre's front corner back to
# the tyre. The tip's back edge drops forward to ~0.2 m. (They were rounded pods with flat tops and tail fins.)
NOSE_RIDGE = ((0.20, 0.935, 0.222), (0.52, 0.645, 0.340))     # the wing's top inner edge: front, tip
NOSE_BOTTOM = ((0.25, 0.980, 0.050), (0.59, 0.700, 0.190))   # the outer face's bottom edge: front, tip
NOSE_STRIP = (0.040, 0.038, -0.008)                            # the top strip: outer edge from the ridge
NOSE_CENTRE = 0.235                                            # the centre piece's half width (straight front)


def _nose_rental(sc):
    """Rental nose (the paddock shots 514273209 and 514277469, front.jpg, the studio front and side shots): one moulding
    of flat faces and crisp edges, its plan lines straight and bending smoothly where they meet. The centre runs straight
    across: the TECHSPEED front face leaning back under a crisp crest, a deck behind rising a little to the lip ahead of
    the pedals, the front panel's foot resting on it (_front_panel_rental). Out to each side the same edges bend back into
    a wing (NOSE_RIDGE, NOSE_BOTTOM): the lip becomes the wing's upright inner face and its top inner edge, the deck its
    narrow top strip, the crest its outer top edge and the front face its outer face, all running diagonally back to a
    tall tip just ahead of the front tyre. The 11 profile points are the livery painter's rows. (The v1 has the same
    nose.)"""
    half = 0.59
    (r0, r1), (b0, b1) = np.array(NOSE_RIDGE), np.array(NOSE_BOTTOM)
    so = np.array(NOSE_STRIP)

    def lin(p, q, x):                                 # a feature line's (y, z) at x, along the straight from p to q
        t = (x - p[0]) / (q[0] - p[0])
        return p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t

    def smin(a, b, r):                                # the smaller of a and b, bending round where they cross
        return 0.5 * (a + b - math.sqrt((a - b) ** 2 + r * r))

    def smax(a, b, r):
        return 0.5 * (a + b + math.sqrt((a - b) ** 2 + r * r))

    def sec(x):
        ax = abs(x)
        w = _smoothstep(0.20, 0.30, ax)               # 0 the centre's profile .. 1 the wing's
        yr, zr = lin(r0, r1, ax)                      # the wing's top inner edge (the ridge)
        yo, zo = lin(r0, r1, ax - so[0])              # its top outer edge
        yo, zo = yo + so[1], zo + so[2]
        ybd, _ = lin(b0, b1, ax)                      # the outer face's bottom edge
        zbd = b0[2] + (b1[2] - b0[2]) * _smoothstep(0.35, b1[0], ax)
        yl, zl = smin(NOSE_LIP_Y, yr, 0.035), smax(0.235, zr, 0.02)       # the lip / the ridge
        yc, zc = smin(0.985, yo, 0.035), smax(0.205, zo, 0.02)            # the crest / the outer top edge
        yb, zb = smin(0.959, ybd, 0.035), 0.100 + (zbd - 0.100) * w       # the bottom edge
        zib = 0.130 + 0.065 * _smoothstep(0.42, 0.52, ax)                 # the wing's inner face foot
        L = lambda c, g: (c[0] + (g[0] - c[0]) * w, c[1] + (g[1] - c[1]) * w)      # noqa: E731
        p0 = L((yl + 0.012, 0.110), (yl - 0.004, zib))
        p1 = L((yl, zl - 0.045), (yl - 0.002, (zl + zib) / 2))
        p2 = L((yl + 0.004, zl - 0.010), (yl - 0.001, zl - 0.012))
        p3 = L((yl + 0.020, zl), (yl + 0.001, zl))
        p6 = (yc - 0.006, zc + 0.001)
        p7 = (yc, zc - 0.010)
        p4 = (p3[0] + (p6[0] - p3[0]) * 0.45, p3[1] + (p6[1] - p3[1]) * 0.45)
        p5 = (p3[0] + (p6[0] - p3[0]) * 0.80, p3[1] + (p6[1] - p3[1]) * 0.80 + 0.003)
        p10 = (yb, zb)
        p9 = (p10[0] + (p7[0] - p10[0]) * 0.12, p10[1] + (p7[1] - p10[1]) * 0.12)
        p8 = (p10[0] + (p7[0] - p10[0]) * 0.55 + 0.004, p10[1] + (p7[1] - p10[1]) * 0.55)
        pts = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
        # the tip: past the ridge's end the inner face and the strip close onto the outer edge, then that onto the bottom
        e1 = _smoothstep(r1[0] - 0.03, r1[0] + 0.01, ax)
        e2 = _smoothstep(r1[0] + 0.01, b1[0], ax)
        pts = [(p[0] + (p6[0] - p[0]) * e1, p[1] + (p6[1] - p[1]) * e1) if k < 6 else p for k, p in enumerate(pts)]
        pts = [(p[0] + (p10[0] - p[0]) * e2, p[1] + (p10[1] - p[1]) * e2) for p in pts]
        return [(x, py, pz) for py, pz in pts]

    side = np.r_[0.0, 0.06, 0.12, 0.16, np.linspace(0.18, 0.32, 10), np.linspace(0.34, 0.50, 7),
                 np.linspace(0.52, half - 0.006, 5)]
    xs = np.r_[-side[:0:-1], side]
    m, arc = _section_loft(sec, xs, cap=True)
    vrow = arc.reshape(len(xs), -1).mean(0)
    vrow = vrow[-1] - vrow
    _uv_from(m, np.repeat(-xs, 11), np.tile(vrow, len(xs)), ATLAS["nose"], (-half, half), (0.0, float(vrow[0])))
    m.modifiers = [("solidify", 0.004)]           # crisp: no smoothing (it was a rounded subsurf moulding)
    m.sharp_angle = 40                            # the bends shade smooth, the profile's edges stay sharp
    sc.add("nose", "body_livery", m)
    sc.panels["nose"] = dict(u=(-half, half), v=(0.0, float(vrow[0])), rows=vrow.tolist())
    # (down onto the front bumper's low tube: it ended inside the hollow nose, on nothing)
    _sym_tube(sc, "nose_mount", "black_plastic",
              [(0.20, 0.86, 0.14), _reach(sc, ("bumper_front_lo_R",), (0.20, 0.93, 0.08))], 0.008, 0)


def _front_panel_rental(sc):
    """Rental front panel (front.jpg, front-panel-nose.jpg): wide plate on top, narrow waist, flared foot over the
    nose's centre. A separate moulding from the nose (_nose_rental)."""
    # the front face's centre line from the studio side shot (its silhouette unprojected onto the centre plane, real y
    # less the 4 cm the front end moves with the wheelbase stretch, z over the frame, less the face's ~2.5 cm bulge):
    # bowed forward, its foot running on down over the nose's centre to ~5 cm short of the nose's front edge (it was
    # an S ~7 cm further back in the middle, the foot ending 11 cm behind the nose's front)
    photo = [(0.364, 0.559), (0.442, 0.533), (0.519, 0.494), (0.597, 0.448), (0.674, 0.396), (0.751, 0.347),
             (0.828, 0.302), (0.885, 0.254), (0.925, 0.200)]
    # the photo's real y less the 4 cm the front end moves with the wheelbase stretch (it took 1.2 cm off and the whole
    # panel sat ~3 cm forward of the photo's), z over the frame less the face's bulge. Only a slight extra bow (at 3.5 cm
    # it was too bent), the top ending ~3 cm under the steering wheel's top instead of level with it (the user), and the
    # foot lying on the nose's centre deck, its flared end down to ~4 cm short of the crest (the studio front shot and
    # the paddock shots: a separate moulding resting on top of the nose; it stopped 2.5 cm over the deck, 10 cm back)
    top_z, foot = 0.505, (0.945, 0.226)
    line = np.array([(y - WB_STRETCH, z - 0.020) for y, z in photo])          # top first
    k_ = int(np.argmax(line[:, 1] < top_z))                                   # cut the top at top_z
    t0 = (line[k_ - 1, 1] - top_z) / (line[k_ - 1, 1] - line[k_, 1])
    top = line[k_ - 1] + (line[k_] - line[k_ - 1]) * t0
    pts = np.array([foot] + [tuple(q) for q in line[::-1] if foot[1] + 0.03 < q[1] < top_z] + [tuple(top)])
    a_, b_ = pts[-1], pts[0]
    chord = b_ - a_
    nrm_ = np.array([chord[1], -chord[0]]) / np.linalg.norm(chord)
    if nrm_[0] < 0:
        nrm_ = -nrm_                                                     # forward and up
    t_ = [float(np.dot(q - a_, chord) / np.dot(chord, chord)) for q in pts]
    spine = [tuple(q + nrm_ * 0.012 * math.sin(math.pi * min(max(t, 0.0), 1.0))) for q, t in zip(pts, t_)]
    m, s_, total, width, k, nf = _panel_volume(
        spine,
        # half widths along the spine (0 the foot, 1 the top), read off the studio front shot through its refitted camera:
        # the wide head reaches about halfway down, a short neck ~13 cm across, the foot flaring to ~31 cm (the head was
        # only the top quarter and the neck 8 cm)
        ([0, 0.06, 0.14, 0.21, 0.30, 0.42, 0.52, 0.85, 1.0], [0.155, 0.13, 0.085, 0.066, 0.075, 0.12, 0.155, 0.16, 0.14]),
        ([0, 0.5, 1.0], [0.02, 0.03, 0.02]),
        ([0, 0.12, 0.3, 0.45, 0.62, 0.8, 1.0], [0.010, 0.022, 0.042, 0.04, 0.026, 0.018, 0.015]))
    _panel_uvs(m, s_, width, k, nf, ATLAS["panel"], (-0.16, 0.16), (0.0, total), (0.155, total * 0.985))
    m.modifiers = [("subsurf", 2)]       # moulded plastic with rounded edges
    m.sharp_angle = 85
    sc.add("front_panel", "body_livery", m)
    sc.panels["panel"] = dict(u=(-0.16, 0.16), v=(0.0, float(total)),
                              width=[(float(a), float(b)) for a, b in zip(s_, width)])
    # support rod from the steering column's clamp up and forward (~35 deg) to the back of the panel's upper
    # part (side-right.jpg); it hung in the air once the column foot moved forward
    sw, cb = np.array(SW_CENTER), np.array(COL_BASE)
    d = (cb - sw) / np.linalg.norm(cb - sw)
    a = sw + d * ((0.335 - sw[1]) / d[1])   # clamped on the column ...
    b = _reach(sc, ("front_panel",), (0.0, 0.463, 0.442), 0.002)   # ... to a tab on the panel's back face
    tab = ml.box((0, 0, 0), (0.03, 0.005, 0.035)).transformed(ml.frame(b + (0, 0.003, 0.002), (1, 0, 0), (0, -0.64, 0.77)))
    clamp = ml.lathe([(0.015, -0.012), (0.015, 0.012)], 16, origin=tuple(a), axis=tuple(d), closed_ends=True)
    sc.add("front_panel_bracket", "black_anodized", ml.merge([ml.tube([a, b], 0.007, 8), tab, clamp]))
    # the panel's lower part is held by a flat steel strap up from the floor plate between the pedals, bolted to its
    # back (steering-wheel.jpg, the user): a 2.5 cm bar with a foot tab on the floor and a top tab flat on the panel
    top = _reach(sc, ("front_panel",), (0.0, 0.70, 0.30), 0.0)
    nrm_top = top - _reach(sc, ("front_panel",), (0.0, 0.70, 0.30), -0.01)
    nrm_top /= np.linalg.norm(nrm_top)                        # into the panel
    tab_c = top - nrm_top * 0.003                              # the top tab, flat on the panel's back
    tang = np.cross((1.0, 0, 0), nrm_top)
    tang /= np.linalg.norm(tang)
    if tang[2] < 0:
        tang = -tang                                          # up the panel
    tab_lo, tab_hi = tab_c - tang * 0.022, tab_c + tang * 0.022
    bend = tab_lo - nrm_top * 0.010
    foot_pt = np.array((0.0, 0.665, FLOOR_Z + 0.006))
    strap = []
    for p0, p1 in ((foot_pt + (0, -0.035, 0), foot_pt), (foot_pt, bend), (bend, tab_lo), (tab_lo, tab_hi)):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        strap.append(ml.box((0.0, 0.0, 0.0), (0.025, ln + 0.004, 0.004)).transformed(
            ml.frame((p0 + p1) / 2, (1.0, 0, 0), d / ln)))
    sc.add("front_panel_strap", "black_anodized", ml.merge(strap))
    for o, ax, nm in ((tab_c - nrm_top * 0.002, -nrm_top, "top"), (foot_pt + (0, -0.022, 0.002), (0, 0, 1.0), "foot")):
        sc.add("front_panel_strap_bolt_" + nm, "aluminium",
               ml.lathe([(0.0, 0.0), (0.0075, 0.0), (0.0065, 0.003), (0.004, 0.005), (0.0, 0.0055)], 12,
                        origin=tuple(o), axis=tuple(ax), closed_ends=True))


# Right pod's top over the frame (BODY z) against the real kart's y, read off side-right.jpg on a metre grid
# (tools/dev/photo_overlay.py's scale): a rounded front end ~0.17 m tall behind the front wheel, the top rising
# gently to ~0.33 m at the engine's front, then steeply over the fan shroud to the cover's lip at ~0.44 m off
# the ground, level back to the rear cover. It was a low pointed tip with the roof only 0.29 m off the ground
# at the engine's front, and the shroud came through it.
POD_R_TOP = ([0.335, 0.32, 0.30, 0.27, 0.22, 0.15, 0.05, -0.05, -0.10, -0.15, -0.20, -0.27, -0.40, -0.54],
             [0.12, 0.15, 0.19, 0.225, 0.245, 0.265, 0.29, 0.315, 0.35, 0.38, 0.405, 0.42, 0.42, 0.405])
POD_R_PAINT_END = -0.20       # real y where the right pod's painted shell ends; the roof behind it is plain black
POD_R_ROOF_XO = 0.465         # the right pod roof's outer edge over the engine
POD_R_END = -0.52             # real y where the right pod's roof ends
POD_R_ROOF_XI = 0.25          # the right pod roof's inner edge over the engine, where it turns down into the skirt
POD_R_OPEN = (-0.045, -0.11)    # real y where the pod's side panel ends and its lower edge climbs, at the shroud


# Left pod's top, real y: the same rounded front as the right pod (POD_R_TOP), then rising gently to the left
# fender (0.335 m off the ground) instead of climbing over an engine. Its front was a flat-topped wedge 0.24 m
# off the ground at the tip.
# The left pod's fittings (drawn y, |x|; seat-adjust-bracket.jpg, sidepod-ballast-fuel.jpg, 514252952, 515336126). The
# ballast holder is not part of the pod: a tall black box standing on the floor against the pod's inner wall, between it
# and the seat, up to the pod top's inner edge: its rear and front ends and outer and inner walls. The weights (a long
# steel one at the front, then white, yellow and red ones) stand in it, handles up, under a slotted steel strap hinged at
# its rear end, hooked at the front under a T-handle. The fuel filler's hole, at the pod's rear end: centre, radius
# across and along, depth (514252952: a round hole straight down into the pod's top, no raised border, the aluminium cap
# standing down inside it with a second disc, the breather, beside it). The tank is in this pod.
BALLAST_BOX = (-0.150, 0.200, (0.468, 0.405))
POD_L_BOWL = (0.548, -0.255, 0.062, 0.062, 0.060)
POD_L_TOP = (POD_R_TOP[0][:6] + [0.05, -0.05, -0.15, -0.25, -0.40],
             POD_R_TOP[1][:6] + [0.283, 0.296, 0.305, 0.311, 0.315])


def _rental_pod_top(y, side):
    """Top of a rental pod (BODY z) at drawn y, from POD_R_TOP / POD_L_TOP."""
    ys, zs = POD_R_TOP if side > 0 else POD_L_TOP
    yr = float(warp_y(y))
    return float(np.interp(-yr, [-a for a in ys], zs))

def _sidepods_rental(sc):
    """Rental side pods (protector-v2-side.jpg, protector-v2-front-row.jpg, onboard shot).
    Left: outer face in line with the tyres, smooth top rising into the left fender.
    Right: one shell from the front to the rear cover that widens towards the back and covers the
    whole engine; its side panel is full height at the front and its lower edge then climbs along
    the ramp tube, leaving the engine open underneath. Red TECHSPEED band along both tops,
    louvred inner wall over the cockpit."""
    y1 = 0.30
    for side, key in ((1, "pod_R"), (-1, "pod_L")):
        right = side > 0
        # the right pod's roof ends ~0.46 m behind the middle (real), where the rear cover's tower starts (the side and
        # rear studio shots: behind it the engine's top and the filter show; it ran on to 0.54 over them)
        y0 = float(unwarp_y(POD_R_END)) if right else -0.34
        ys = (np.r_[np.linspace(y0, -0.04, 16), np.linspace(0.0, 0.14, 6), np.linspace(0.18, y1, 6)] if right
              else np.r_[np.linspace(y0, -0.17, 19), np.linspace(-0.155, 0.14, 21), np.linspace(0.17, y1, 9)])
        if right:       # a row on the air cleaner cut's front edge
            ys = np.unique(np.r_[[y for y in ys if abs(y - POD_CUT_Y) > 0.008], POD_CUT_Y])

        def geo(y, right=right):
            zt = _rental_pod_top(y, 1 if right else -1)
            if right:
                xo = 0.62 + 0.045 * _smoothstep(0.20, 0.0, y)            # out over the recoil starter
                # the painted pod carries on up and back over the shroud's front to ~0.2 m behind the middle, at the
                # roof's height (the studio side and outdoor shots: its rear edge runs up parallel to the protector
                # hoop's diagonal); behind, only the black roof over the clutch side stays, inboard of ~0.46, and the
                # shroud's top is open (no skirt hangs over it)
                yr_ = float(warp_y(y))
                xo = xo + (POD_R_ROOF_XO - xo) * _smoothstep(POD_R_PAINT_END + 0.02, POD_R_PAINT_END - 0.05, yr_)
                # in over the engine: one moulding with the roof over the clutch and chain, its top running on
                # inboard to the skirt over the seat's side (514400663, 515935974: the roof and the louvred skirt
                # are the pod itself; it stopped at 0.43 over a separate, lower cover and looked like two parts)
                xi = 0.47 - (0.47 - POD_R_ROOF_XI) * _smoothstep(0.10, -0.10, y)
                # full-height panel ahead of the engine; at the fan shroud's front the lower edge climbs
                # steeply to a lip just under the top, leaving the whole shroud, recoil starter,
                # carburettor and filter open below the roof (side-right.jpg)
                yr = float(warp_y(y))
                zb = 0.10 + (zt - 0.05 - 0.10) * _smoothstep(-POD_R_OPEN[0], -POD_R_OPEN[1], -yr)
                dy = y - GX_Y                     # and arches over the recoil starter, which stays in view
                if abs(dy) < RECOIL_CLEAR:
                    zb = max(zb, RECOIL_Z + math.sqrt(RECOIL_CLEAR ** 2 - dy * dy))
            else:
                xo, xi, zb = 0.635 - 0.015 * (y - y0) / (y1 - y0), 0.47, 0.10     # a narrow rail (sidepod-ballast-fuel.jpg)
            return xo, xi, zt, zb

        def side_sec(y):
            xo, xi, zt, zb = geo(y)
            zb = min(zb, zt - 0.035)
            pts = [(xi + 0.01, y, zb + 0.012), (xo - 0.02, y, zb), (xo, y, zb + 0.02), (xo + 0.004, y, (zb + zt) / 2),
                   (xo, y, zt - 0.025), (xo - 0.012, y, zt)]
            if right and y < 0.0:        # over the engine: no floor across, just the outer lip
                pts[0] = (xo - 0.012, y, zb + 0.012)
            return [(side * a, b, c) for a, b, c in pts]

        def top_sec(y):
            xo, xi, zt, zb = geo(y)
            # the right pod's top rises to its inner edge beside the seat, up to the seat's rim (the studio cockpit and
            # outdoor shots: it dropped 4 cm towards the seat)
            lift = 0.06 * _smoothstep(0.15, -0.05, float(warp_y(y))) if right else 0.0
            pts = [(xo - 0.012, y, zt), (xo - 0.05, y, zt + 0.004 + 0.3 * lift), ((xo + xi) / 2, y, zt - 0.008 + 0.7 * lift),
                   (xi + 0.02, y, zt - 0.026 + lift), (xi, y, zt - 0.04 + lift)]
            if not right:
                # the left pod's top (514252952, seat-adjust-bracket.jpg): a rounded rim outboard, the top sloping a little
                # down to the inner edge, and at the rear end the fuel filler's round hole (POD_L_BOWL) going straight
                # down, its edge only just rounded (sampled ~1 cm across so the hole stays round)
                mid = np.linspace(0.600, 0.485, 13)
                pts = ([(xo - 0.012, 0.0), (xo - 0.035, 0.004)] + [(float(q), 0.003 - 0.015 * (0.600 - q) / 0.115) for q in mid]
                       + [(xi + 0.004, -0.022), (xi, -0.04)])
                bx, by, ax_, ay_, depth = POD_L_BOWL
                out = []
                for k_, (px, dz) in enumerate(pts):
                    rr = math.hypot((px - bx) / ax_, (y - by) / ay_)
                    if k_ < len(pts) - 1:
                        dz += -depth * _smoothstep(1.0, 0.86, rr)
                    out.append((side * px, y, zt + dz))
                return out
            return [(side * a, b, c) for a, b, c in pts]

        m, _ = _section_loft(side_sec, ys, cap=False)
        if side < 0:
            m.faces = [tuple(reversed(f)) for f in m.faces]
        n = len(ys) * 6
        _uv_from(m, m.verts[:n, 1] * side, m.verts[:n, 2], ATLAS[key], (-0.52, 0.36), (0.06, 0.46))
        if right:       # the outer side ends at the air cleaner cut
            keep = [i for i, f in enumerate(m.faces) if m.verts[list(f)].mean(0)[1] > POD_CUT_Y]
            # (the real cone-filter kart has it cut too: from behind, nothing stands over the carburettor and filter
            # outboard of the louvres; the roof's inner part runs on back to POD_R_END)
            m.faces, m.uvs = [m.faces[i] for i in keep], [m.uvs[i] for i in keep]
        # the pods are closed hollow mouldings: cap both ends over the whole section (side + top + inner); the
        # right one's rear cap closes the air cleaner cut's front face
        caps = []
        for y, front in ((ys[-1], True), (POD_CUT_Y if right else ys[0], False)):
            ring = side_sec(y) + top_sec(y)[1:]
            v = np.array(ring)
            f = tuple(range(len(ring)))
            cap = ml.Mesh(v, [f], [tuple((0.5, 0.5) for _ in f)], smooth=False)
            c = v.mean(0)
            nrm = np.cross(v[1] - v[0], v[2] - v[0])
            want = np.array([0, 1.0, 0]) if front else np.array([0, -1.0, 0])
            if nrm @ want < 0:
                cap.faces = [tuple(reversed(f))]
            caps.append(cap)
        sc.add("pod_caps_" + ("R" if right else "L"), "black_plastic", ml.merge(caps))
        m.modifiers = [("subsurf", 1), ("solidify", 0.004)]
        m.sharp_angle = 85
        if right:
            # behind the painted shell's end the roof's side skirt is plain black plastic (the studio side shot)
            yc = [float(warp_y(m.verts[list(f)].mean(0)[1])) for f in m.faces]
            m.faces, m.uvs = ([f for f, q in zip(m.faces, yc) if q >= POD_R_PAINT_END - 0.03],
                              [u for u, q in zip(m.uvs, yc) if q >= POD_R_PAINT_END - 0.03])
        sc.add("sidepod_" + ("R" if right else "L"), "body_livery", m)
        sc.panels[key] = dict(u=(-0.52, 0.36), v=(0.06, 0.46), front=side, kind="rental",
                              top=[(float(y), float(geo(y)[2])) for y in ys],
                              bottom=[(float(y), float(min(geo(y)[3], geo(y)[2] - 0.035))) for y in ys])

        top, _ = _section_loft(top_sec, ys, cap=False)
        if side < 0:
            top.faces = [tuple(reversed(f)) for f in top.faces]
        k = len(top_sec(ys[0]))
        u = [((y1 - y) if right else (y - y0)) / 0.64 for y in ys]     # letters' tops face outwards
        # v by the distance in from the outer edge, scaled to the pod's width at the front (there it's the row
        # order 1, 0.85, 0.5, 0.15, 0): where the right pod's roof widens over the engine its inner part
        # stays black and the red decal on its outer part (514400663; one v per row stretched the decal across)
        def vrow(idx):
            xo_ = geo(ys[idx // k])[0]
            if right and float(warp_y(ys[idx // k])) < POD_R_PAINT_END:
                return 0.0                       # the roof over the engine is plain black
            if not right:                        # the left pod's inner edge and the fuel bowl are black
                q = top.verts[idx]
                bx, by, ax_, ay_, _ = POD_L_BOWL
                if math.hypot((abs(q[0]) - bx) / ax_, (q[1] - by) / ay_) < 1.03:
                    return 0.95                  # (the black past the decal's outer edge: from 0 the walls crossed the
                    # gold pinstripe's row and came out streaked)
                if abs(q[0]) < 0.49:
                    return 0.0
            return max(0.0, min(1.0, 1.0 - (xo_ - abs(top.verts[idx][0])) / 0.15 * 0.95))
        def in_hole(f):                          # a face of the left pod's fuel filler hole: black at every corner
            if right:
                return False
            c = top.verts[list(f)].mean(0)
            bx, by, ax_, ay_, _ = POD_L_BOWL
            return math.hypot((abs(c[0]) - bx) / ax_, (c[1] - by) / ay_) < 1.16
        top.uvs = [tuple((u[idx // k], 0.95 if in_hole(f) else vrow(idx)) for idx in f) for f in top.faces]
        if right:
            # cut away from the air cleaner out to the side (onboard shot): the top ran through the box
            c = [top.verts[list(f)].mean(0) for f in top.faces]
            keep = [i for i, q in enumerate(c) if not (q[0] > POD_CUT_X and q[1] < POD_CUT_Y)]
            top.faces, top.uvs = [top.faces[i] for i in keep], [top.uvs[i] for i in keep]
        top.modifiers = [("subsurf", 1), ("solidify", 0.004)]
        top.sharp_angle = 85
        sc.add("pod_top_" + ("R" if right else "L"), "pod_stripe", top)

        # inner wall with louvres, over the cockpit only
        yw = [y for y in ys if y >= 0.0] if right else list(ys)
        wall = ml.loft([[top_sec(y)[-1], side_sec(y)[0]] for y in yw], closed=False)
        wall.smooth = False
        f0 = wall.faces[0]       # single sided: it must face the cockpit (it showed through on the right)
        if np.cross(wall.verts[f0[1]] - wall.verts[f0[0]], wall.verts[f0[2]] - wall.verts[f0[0]])[0] * side > 0:
            wall.faces = [tuple(reversed(f)) for f in wall.faces]
            wall.uvs = [tuple(reversed(u)) for u in wall.uvs]
        sc.add("pod_inner_" + ("R" if right else "L"), "black_plastic", wall)
        slots = []
        for kk in range(6):
            y = 0.12 + 0.027 * kk
            hi, lo = np.array(top_sec(y)[-1]), np.array(side_sec(y)[0])
            tall = min(0.07, 0.6 * (hi[2] - lo[2]))
            if tall < 0.03:           # the right pod's low front: no room for louvres
                continue
            mid = lo + (hi - lo) * 0.55 + (-side * 0.003, 0, 0)
            slots.append(ml.box((0, 0, 0), (0.004, 0.009, tall)).transformed(ml.frame(mid, (side, 0, 0), (0, 1, 0))))
        if slots:
            sc.add("pod_louvres_" + ("R" if right else "L"), "grip_black", ml.merge(slots))
        if not right:
            _pod_left_fittings(sc, top_sec)


def _pod_left_fittings(sc, top_sec):
    """By the left pod (seat-adjust-bracket.jpg, sidepod-ballast-fuel.jpg, 514252952): the fuel filler in its hole, and the
    ballast holder against its inner wall with its strap and latch (BALLAST_BOX, POD_L_BOWL). Exported closed and empty;
    the strap swung open with the weights standing in the box is kept in the source as a hidden alternative
    (pod_ballast_*)."""
    ty0, ty1, (wo, wi) = BALLAST_BOX
    bx, by, ax_, ay_, depth = POD_L_BOWL

    def top_at(x, y):                                # the pod top's height (BODY z) at (x, y), from its section
        q = np.array(top_sec(y))
        o = np.argsort(q[:, 0])
        return float(np.interp(x, q[o, 0], q[o, 2]))

    # down in the hole: the polished aluminium cap on a welded neck rising from its floor, toward the hole's front edge, its
    # knurled top standing ~1.5 cm over the pod's top (514252952; it sat 2 cm under it), and a second aluminium disc
    # behind it on the floor (the breather)
    floor_z = top_at(-bx, by)
    cx, cy = -(bx + 0.008), by + 0.030
    h = top_at(cx, by + ay_ + 0.01) - floor_z + 0.004 + 0.015          # to ~1.5 cm over the pod's top
    sc.add("fuel_cap", "aluminium", ml.lathe(
        [(0.0, 0.0), (0.017, 0.0), (0.017, h - 0.028), (0.026, h - 0.026), (0.027, h - 0.008), (0.025, h - 0.002),
         (0.019, h), (0.0, h)], 28, origin=(cx, cy, floor_z - 0.004), axis=(0, 0, 1), closed_ends=True))
    sc.add("fuel_vent", "aluminium", ml.lathe(
        [(0.0, 0.0), (0.022, 0.0), (0.022, 0.020), (0.020, 0.024), (0.0, 0.024)], 24,
        origin=(cx + 0.020, cy - 0.058, floor_z - 0.004), axis=(0, 0, 1), closed_ends=True))

    # the ballast holder: a box open at the top, on the floor plate, its top just under the pod top's inner edge
    def box_top(y):
        return top_sec(y)[-1][2] - 0.004
    zb = FLOOR_Z + 0.004
    yy = np.linspace(ty0, ty1, 8)
    u_secs = [[(-wo, y, box_top(y)), (-wo, y, zb), (-wi, y, zb), (-wi, y, box_top(y))] for y in yy]
    hold = ml.loft(u_secs, closed=False)
    ends = []
    for y in (ty0, ty1):
        v = np.array([(-wo, y, zb), (-wi, y, zb), (-wi, y, box_top(y)), (-wo, y, box_top(y))])
        ends.append(ml.Mesh(v, [(0, 1, 2, 3)], [((0, 0), (1, 0), (1, 1), (0, 1))], smooth=False))
    hold = ml.merge([hold] + ends)
    hold.smooth = False
    hold.modifiers = [("solidify", 0.003)]
    sc.add("pod_ballast_box", "black_plastic", hold)

    # the strap: a 4 cm steel flat bar along the box's top from the hinge at its rear end, slotted (a long slot at each
    # end, short ones between), its front end hooked under the T-handle
    xc = -(wo + wi) / 2
    y_hinge, y_front = ty0 + 0.008, ty1 + 0.004
    z_hinge, z_front = box_top(ty0) + 0.007, box_top(ty1) + 0.007
    length = y_front - y_hinge
    solid = [(0.0, 0.05), (0.33, 0.38), (0.45, 0.50), (0.57, 0.62), (0.95, 1.0)]   # along it, from the hinge

    def strap_mesh(ang):
        ms = []
        for side_x in (-0.0135, 0.0135):             # the bar either side of its narrow slots
            ms.append(ml.box((xc + side_x, y_hinge + length / 2, 0.0), (0.016, length, 0.004)))
        for a_, b_ in solid:                          # and between them
            ms.append(ml.box((xc, y_hinge + length * (a_ + b_) / 2, 0.0), (0.012, length * (b_ - a_), 0.004)))
        ms.append(ml.box((xc, y_front + 0.010, 0.004), (0.016, 0.022, 0.004)))      # the hook's tongue
        m = ml.merge(ms)
        v = m.verts.copy()
        # lying on the box's top all along (it rises to the back with the pod), then turned up on the hinge
        v[:, 2] += np.array([box_top(min(max(y, ty0), ty1)) + 0.007 for y in v[:, 1]])
        c, s_ = math.cos(ang), math.sin(ang)
        dy, dz = v[:, 1] - y_hinge, v[:, 2] - z_hinge
        v[:, 1], v[:, 2] = y_hinge + dy * c - dz * s_, z_hinge + dy * s_ + dz * c
        m.verts = v
        return m

    hinge = ml.lathe([(0.005, -0.022), (0.005, 0.022)], 12, origin=(xc, y_hinge, z_hinge - 0.004), axis=(1, 0, 0),
                     closed_ends=True)
    sc.add("pod_ballast_lid", "black_anodized", ml.merge([strap_mesh(0.0), hinge]))
    # the T-handle at the box's front end that the strap's hook goes under: a post and a cross bar
    tz = box_top(ty1)
    latch = [ml.box((xc, ty1 + 0.028, tz - 0.002), (0.03, 0.026, 0.008)),
             ml.lathe([(0.0045, 0.0), (0.0045, 0.030)], 10, origin=(xc, ty1 + 0.028, tz), axis=(0, 0, 1), closed_ends=True),
             ml.lathe([(0.006, -0.022), (0.006, 0.022)], 10, origin=(xc, ty1 + 0.028, tz + 0.034), axis=(1, 0, 0),
                      closed_ends=True)]
    sc.add("pod_ballast_latch", "steel", ml.merge(latch))

    # hidden alternative: the strap swung up on its hinge, and the weights standing in the box, handles up
    sc.hide_new = True
    sc.add("pod_ballast_strap_open", "black_anodized", strap_mesh(math.radians(65)))
    wts = {"steel": [], "tank_plastic": [], "oil_cap": [], "engine_red": []}
    y = ty1 - 0.006
    for ln, mat in ((0.17, "steel"), (0.055, "tank_plastic"), (0.055, "oil_cap"), (0.055, "engine_red")):
        yc = y - ln / 2
        zt_ = box_top(yc) - 0.022
        wts[mat].append(ml.box((xc, yc, (zb + zt_) / 2), (0.054, ln - 0.002, zt_ - zb)))
        wts["steel"].append(ml.tube([(xc, yc - 0.015, zt_ - 0.002), (xc, yc - 0.015, zt_ + 0.014),
                                     (xc, yc + 0.015, zt_ + 0.014), (xc, yc + 0.015, zt_ - 0.002)], 0.002, 6))
        y -= ln + 0.002
    for mat, ms in wts.items():
        sc.add("pod_ballast_" + mat, mat, ml.merge(ms))
    sc.hide_new = False


def _rear_bumper(sc, variant):
    big = family(variant) == "rental"
    half = 0.70 if not big else 0.72

    def sec(x):
        a = abs(x) / half
        yb = -0.80 + 0.06 * a ** 2 - (0.03 if big else 0)
        yf = -0.66 + 0.03 * a ** 2
        zt = (0.25 if not big else 0.33) - (0.04 if not big else 0.07) * a ** 2
        if big and a < 0.35:
            zt += 0.02
        zb = 0.06
        return [(x, yf, zb + 0.03), (x, yf, zt - 0.03), (x, yf - 0.03, zt), (x, (yf + yb) / 2, zt + 0.005),
                (x, yb + 0.02, zt - 0.01), (x, yb, zt - 0.05), (x, yb, zb + 0.02), (x, yb + 0.03, zb)]

    xs = np.linspace(-half, half, 21)
    m, arc = _section_loft(sec, xs, cap=True)
    vrow = arc.reshape(len(xs), -1).mean(0)
    vrow = vrow[-1] - vrow          # v grows upwards on the back face
    _uv_from(m, np.repeat(xs, 8), np.tile(vrow, len(xs)), ATLAS["rear"], (-half, half), (0.0, float(vrow[0])))
    m.modifiers = [("subsurf", 2), ("solidify", 0.004)]
    m.sharp_angle = 85
    sc.add("rear_bumper", "body_livery", m)
    sc.panels["rear"] = dict(u=(-half, half), v=(0.0, float(vrow[0])), rows=vrow.tolist())


def _pedals_rental(sc):
    """Square 'TECH' pedal plates on tube loops, aluminium master cylinder (pedals-brake-master.jpg)."""
    for side, nm in ((1, "throttle"), (-1, "brake")):
        x = side * PEDAL_X
        py = PEDAL_Y
        tilt = ml.rot_axis((1, 0, 0), -PEDAL_TILT)
        plate = ml.box((0, 0, 0), (PEDAL_W, 0.006, PEDAL_H), rot=tilt).transformed(ml.translate((x, py, PEDAL_Z)))
        # TECH reads left to right from the seat on both (it was mirrored on the throttle)
        t = math.radians(PEDAL_TILT)
        ml.project_uv(plate, (x, py, PEDAL_Z), (1.0, 0, 0), (0, math.sin(t), math.cos(t)), (-PEDAL_W / 2, PEDAL_W / 2),
                      (-PEDAL_H / 2, PEDAL_H / 2), (0, 0, 1, 1))
        # the print on the face towards the driver only: the back and the edges in the plate's plain black (the
        # projection went through, so from the front the pedals read TECH mirrored)
        fy = [np.cross(plate.verts[f[1]] - plate.verts[f[0]], plate.verts[f[2]] - plate.verts[f[0]])[1] for f in plate.faces]
        face = int(np.argmin(fy))
        plate.uvs = [uv if i == face else tuple((0.02, 0.02) for _ in uv) for i, uv in enumerate(plate.uvs)]
        # plate and loop hang on their own node at the pivot (PEDAL_GAS / PEDAL_BRAKE, axes square to the kart):
        # a CSP car script turns it with the throttle and brake (export_ac)
        node = "PEDAL_GAS" if nm == "throttle" else "PEDAL_BRAKE"
        pivot = np.array([x, PEDAL_PIVOT_Y, PEDAL_PIVOT_Z])
        sc.node(node, "BODY", ml.translate(pivot))
        sc.add("pedal_" + nm, "pedal_plate", plate.transformed(ml.translate(-pivot)), node)
        # the tube loop round the plate's sides and over its top, from the pivot at the floor
        lx, top = PEDAL_W / 2 + 0.007, PEDAL_Z + (PEDAL_H / 2 + 0.012) * math.cos(t)
        ty = py + (PEDAL_H / 2 + 0.012) * math.sin(t)
        pvy, pvz = PEDAL_PIVOT_Y, PEDAL_PIVOT_Z                 # the loop's feet on the pivot
        loop = [(x - lx, pvy, pvz), (x - lx, ty, top), (x + lx, ty, top), (x + lx, pvy, pvz)]
        sc.add("pedal_frame_" + nm, "protector", ml.tube(loop, 0.007, 8, 0.02).transformed(ml.translate(-pivot)), node)
        sc.add("pedal_pivot_" + nm, "black_anodized",
               ml.tube([(x - lx - 0.01, pvy, pvz), (x + lx + 0.01, pvy, pvz)], 0.008, 8))
    # brake master cylinder (pedals-brake-master.jpg, steering-pedals-front-axle.jpg): a short aluminium body
    # lying fore and aft between the two pedals, a little towards the brake and behind them, on a bracket off
    # the floor, its small clear reservoir standing on top, pulled by a threaded pushrod angled out to the
    # brake pedal's loop, and the brake hose leaving its back for the rear caliper. (It sat right behind the
    # brake pedal with a reservoir as wide as the pedal's edge, almost touching it.)
    mx, my, mz = -0.030, PEDAL_PIVOT_Y - 0.02, 0.078    # front end of the body (pushrod side)
    body = ml.lathe([(0.0, 0.0), (0.0115, 0.0), (0.0115, 0.008), (0.0145, 0.010), (0.0145, 0.054), (0.012, 0.058),
                     (0.012, 0.066), (0.0, 0.066)], 16, origin=(mx, my, mz), axis=(0, -1, 0), closed_ends=True)
    lug = ml.box((mx, my - 0.026, mz - 0.019), (0.020, 0.030, 0.010))          # mounting lug under the body
    bracket = ml.box((mx, my - 0.026, 0.044), (0.036, 0.046, 0.020))          # on the floor plate
    sc.add("brake_master", "aluminium", ml.merge([body, lug]))
    sc.add("brake_master_bracket", "black_anodized", bracket)
    res = (mx, my - 0.028, mz + 0.0145)
    sc.add("brake_master_cap", "tank_plastic",
           ml.lathe([(0.0, 0.0), (0.0085, 0.0), (0.0095, 0.005), (0.0095, 0.024), (0.0, 0.024)], 16, origin=res,
                    axis=(0, 0, 1), closed_ends=True))
    sc.add("brake_master_lid", "black_plastic",
           ml.lathe([(0.0, 0.024), (0.0105, 0.024), (0.0105, 0.030), (0.0, 0.030)], 16, origin=res, axis=(0, 0, 1),
                    closed_ends=True))
    # pushrod out of the body's nose, angled out to a clevis on the brake pedal loop's inner leg
    leg = np.array([-PEDAL_X + PEDAL_W / 2 + 0.007, PEDAL_PIVOT_Y + 0.04 * math.sin(math.radians(PEDAL_TILT)),
                    PEDAL_PIVOT_Z + 0.04 * math.cos(math.radians(PEDAL_TILT))])
    sc.add("brake_pushrod", "steel", ml.merge([
        ml.tube([(mx, my, mz), (leg[0] + 0.012, leg[1] - 0.006, leg[2])], 0.0035, 8),
        ml.box((leg[0] + 0.009, leg[1] - 0.004, leg[2]), (0.014, 0.012, 0.012))]))
    # brake hose from the back of the body along the left inner rail to the rear caliper
    sc.add("brake_line", "black_plastic", ml.tube(
        [(mx, my - 0.066, mz - 0.004), (mx - 0.01, my - 0.10, 0.050), (-0.15, 0.42, 0.046), (-0.255, 0.16, 0.068),
         (-0.255, -0.36, 0.072), (-0.17, -0.44, 0.13), (-0.125, Y_R + 0.06, 0.185)], 0.0035, 8, 0.04))


# v2's exhaust opening in the rear cover's face: half width (the middle four section columns, 0.0943) and its top
# edge; the bottom edge is the dipped row under it (~0.124). The twin tips' centre height, central in it.
# the twin tips sit right under the number plate (the user, the studio rear and rear 3/4 shots): 2.5 cm higher than
# first drawn, with the opening they come out of (EXH_DZ raises it in the middle of the rear face only)
EXH_DZ = 0.025
EXH_HOLE = (0.095, 0.192 + EXH_DZ)
EXH_TIP_Z = 0.158 + EXH_DZ
# v2 rear cover, measured on the photos (rear.jpg with its camera solved from the protector's rails, the tyres
# and the steering wheel; side-right.jpg scaled by the tyres), in drawn coordinates over the frame (photo z less
# the 2 cm ride, photo y plus the 4 cm wheelbase stretch):
# - the top across the back (rear view), |x| -> z: the lip under the plate low in the middle, a ridge along each
#   side of the deck, a valley outboard of it (the latches), the hump over the tyre, rounding down over its
#   shoulder at the end
REAR_TOP = ((0.0, 0.322), (0.24, 0.322), (0.258, 0.324), (0.274, 0.340), (0.29, 0.369), (0.31, 0.369),
            (0.325, 0.351), (0.345, 0.326), (0.377, 0.308), (0.408, 0.322), (0.44, 0.330), (0.471, 0.335),
            (0.503, 0.341), (0.534, 0.343), (0.566, 0.341), (0.585, 0.337), (0.60, 0.326))
# The outer corners stay high and turn over tightly into the turned-down lip (the studio rear 3/4 shot: boxy corners;
# they rounded down to 0.295 at the end and looked bulbous).
# its ends ~0.60 out (rear.jpg: the rear tyres' outer halves stand out past it; it was 0.64, over the whole tyre)
REAR_HALF = 0.60
# - over the tyre (side view) the fender follows the tyre round the axle, ~21.5 cm out over the top and behind,
#   closing in to ~19.5 cm where it meets the upright rear face 18.5 cm behind the axle
REAR_FENDER_R = 0.215
REAR_FACE_Y = -0.705     # the rear face, upright, ~4.5 cm behind the tyres
REAR_PLATE = (0.235, 0.336, 0.082)  # v2: the number plate's bottom, top (just over the lip) and half width: 16.5 x
                                    # 10 cm (rear.jpg, measured on its solved camera; it was 22 x 12 cm)
# v2 rear cover section rows the livery is laid out on (textures._rear_fleet_v2)
REAR_ROWS = dict(deck_front=2, dish=4, lip=7, groove=9, band_top=10, band_bottom=11, hole_top=12)


def _rear_v2_section(sc, x, half, deck_front):
    """One section (constant x) of the v2 rear cover, 16 points from the front edge of the deck over the top to the
    bottom edge of the rear face (REAR_ROWS names them for the painter): 0-2 the deck's front edge, 3-5 the deck,
    6 the rim along the back, 7 the lip, 8-15 the rear face down to the edge turning under it. Built from the curves
    measured on the photos (REAR_TOP, REAR_FENDER_R, REAR_FACE_Y). In the middle the deck (a flat panel, the web
    address on it) runs back to a crisp lip and the face drops nearly upright, gently convex, with no step across it
    (protector-v2-rear-left.jpg, the rear-row photo); a recess round the plate and the exhaust in the middle only.
    Over each tyre the section is a fender: the same rows taken round the axle on an arc, from over the tyre's top
    down behind it into the upright face (side-right.jpg). Between them, the ridge along the deck's side and the
    valley outboard of it come from the top line."""
    ax = abs(x)
    yb = REAR_FACE_Y
    yf = deck_front(x)
    ztop = float(np.interp(ax, [q[0] for q in REAR_TOP], [q[1] for q in REAR_TOP]))
    # the narrow ridge along each side of the deck (rear-cover-right.jpg, rear-3q.jpg), as high at the back as
    # the rear view shows it, a little higher still at its front
    ridge = _smoothstep(0.255, 0.285, ax) * (1.0 - _smoothstep(0.315, 0.345, ax))
    zt = ztop
    over_tyre = ax > 0.46
    # the front faces: over the tyre the fender's front edge sweeps down ahead of it to ~0.25 (side-right.jpg; at 0.30
    # it stood tall and boxy there), the tyre's top being ~6 cm further back
    zf = 0.25 if over_tyre else 0.22
    if not over_tyre and x > 0:                     # and over the chain cover (its top ~0.235 at the axle sprocket)
        zf += 0.04 * _smoothstep(0.255, 0.285, x) * (1 - _smoothstep(0.38, 0.42, x))
    # the deck's front edge: rising from the lip to the seat in the middle (rear-3q.jpg: the web address panel
    # slopes up to the seat), the ridges along its sides higher still at their front (rear-cover-right.jpg)
    ztf = zt + 0.030 * (1.0 - _smoothstep(0.24, 0.30, ax)) + 0.012 * ridge
    if x > 0.26:                                     # rises to meet the right pod shell over the engine
        # (to ~0.37 m off the ground: the rear studio shot shows the engine's top over it; at 0.425 it stood up as a
        # wall behind the louvres)
        # outboard of a low shelf ~0.33-0.42 out where the latches sit (the user: they're on the lower area beside
        # the deck, not on top; triangulated 0.31 m off the ground)
        ztf = ztf + (0.35 - ztf) * _smoothstep(0.44, 0.50, x) * (1 - _smoothstep(0.60, 0.64, x))
    sunk = 1.0 - _smoothstep(0.24, 0.30, ax)        # the flat deck panel between the ridges, sunk ~2.5 cm
    ym = (yf + yb) / 2

    def pocket(w, depth):                           # the recess in the middle, its wall over two 3 cm columns
        return depth * (1.0 - _smoothstep(w - 0.06, w, ax))
    g = 1.0 - _smoothstep(0.10, 0.15, ax)           # the sill under the exhaust opening dips ~1.5 cm
    deck = [(yf + 0.012, zf), (yf, ztf - 0.04), (yf - 0.02, ztf),
            (yf - 0.06, ztf - 0.025 * sunk), (ym, (ztf + zt) / 2 - 0.028 * sunk), (yb + 0.065, zt - 0.020 * sunk),
            (yb + 0.022, zt + 0.003), (yb + 0.002, zt - 0.014),     # the lip: a tight round, no overhang
            # the face: upright, ~1 cm convex over its height, turning under at the bottom. In the middle a recess
            # (rear.jpg): its top edge across at the plate's middle height, ~50 cm wide, narrowing to the tunnel
            # round the tips ~28 cm wide; the sloping sides of its outline are painted (textures._rear_fleet_v2):
            # narrowing row by row in the geometry folded the face (it was a narrow box round the plate)
            # (halfway between the lip and the next row: at a fixed 4.5 cm under the lip it fell below that row
            # where the top dips, ~0.31 in the valleys, and the face folded there)
            (yb - 0.008, 0.276 + (zt - 0.014 - 0.276) * 0.5),
            (yb - 0.010, 0.276),
            # one even depth from its top edge down past the tips (a narrower tunnel under a wider recess left a
            # step across the face that the smoothing turned into lumps)
            (yb - 0.011 + pocket(0.26, 0.030), 0.250),
            (yb - 0.010 + pocket(0.26, 0.030), 0.212 + EXH_DZ * g),
            # round the tips: clear of the silencer's end plate ~2 cm behind the face; the opening between these
            # two rows a little taller than the tips, as the smoothing draws its edge in (EXH_HOLE)
            (yb - 0.008 + pocket(0.26, 0.016), EXH_HOLE[1] - EXH_DZ * (1 - g) + 0.006 * g),
            (yb - 0.004 + pocket(0.26, 0.010), 0.140 + (EXH_DZ - 0.016) * g),
            (yb + 0.005, 0.118 - 0.008 * g), (yb + 0.022, 0.108 - 0.010 * g)]
    # over the tyre: the rows from the deck's rear over the top and down behind the tyre on an arc round the axle,
    # closing in from REAR_FENDER_R over the top to where it meets the upright face (side-right.jpg); the arc's top
    # follows the top line, rounding down over the tyre's shoulder at the end (REAR_TOP)
    b = _smoothstep(0.45, 0.52, ax)
    if b <= 0:
        return [(x, py, pz) for py, pz in deck]
    cy, cz = Y_R, _axle_z(sc)
    rtop = zt - cz
    fender = list(deck[:3])
    fender[1] = (yf, zt - 0.035)
    fender[2] = (yf - 0.02, zt)
    # rows 3-12 on the arc from over the tyre's top round to the rear face, the radius closing from the top line's
    # height to ~19.6 cm where it meets the face (side-right.jpg: ~21.5 cm out over the top and behind, 19.6 cm at
    # the face, 6.5 cm over the axle); its z falling all the way, so the face below has no fold (rows 13-15)
    for th in np.linspace(90, 158, 10):
        r = float(np.interp(th, [90, 130, 160], [rtop, min(rtop, REAR_FENDER_R - 0.002), 0.196]))
        fender.append((cy + r * math.cos(math.radians(th)), cz + r * math.sin(math.radians(th))))
    # the arch stops a little over the axle with a short lip turned in under it: the tyre's back shows below it
    # (rear.jpg / the studio rear shots: only between the tyres does the face run down to the bottom; it covered
    # the tyres' backs down to 0.11)
    for th, r, dy in ((165, 0.196, 0.0), (171, 0.196, 0.0), (171, 0.180, 0.004)):
        fender.append((cy + r * math.cos(math.radians(th)) + dy, cz + r * math.sin(math.radians(th))))
    return [(x, d[0] + (f[0] - d[0]) * b, d[1] + (f[1] - d[1]) * b) for d, f in zip(deck, fender)]


def _rear_cover_rental(sc):
    """Rental rear cover. v2 (rear.jpg, rear-3q.jpg, rear-cover-right.jpg, protector-v2-rear-left.jpg): a dished
    deck behind the seat inside a raised lip along its back; under the lip the rear face steps in to a groove
    across it, bulges out again in a band carrying the four stickers, and steps in to a lower recess where the
    twin exhaust tips come out under the plate; a hump over each rear tyre ~4.5 cm over the middle, rounding
    down over the tyre, whose outer edge stands out past the cover; rubber T-latches on the humps; an open bay
    on the right for the engine; the fuel filler in the left deck. Its ends stop ~4 cm inside the tyres' outer
    faces, outboard of the engine's air box. (It was a smooth tub, level across the back; at 0.58 half-width it
    looked too narrow.)"""
    v1 = sc.params.get("rear_cover") == "v1"
    half = 0.64 if v1 else REAR_HALF
    # v1 (the rental with the band, onboard frames): the rear face ~4 cm lower with a rounder top lip, and
    # with no side pod there the deck doesn't rise over the engine on the right
    V1_ROUND_X, V1_ROUND_Z = 0.50, 0.29     # v1: where the ends start rounding down, and the lip they round to

    def v1_ztop(x):                         # v1 deck height: rising over the tyres
        return 0.325 + 0.045 * _smoothstep(0.34, 0.52, abs(x))

    def deck_front(x):
        ax = abs(x)
        centre, fender = -0.42, -0.345
        y = centre + (fender - centre) * _smoothstep(0.24, 0.40, ax)
        if x > 0:   # engine bay: the right pod shell covers it and meets the cover here. The cover's tower beside
            # the deck reaches forward to ~0.47 m behind the middle (real), just behind the carburettor: its upper
            # latch is there (the rear and rear 3/4 studio shots, triangulated); it was cut back to 0.54
            y = y + (float(unwarp_y(-0.47)) - y) * _smoothstep(0.20, 0.26, x) * (1 - _smoothstep(0.60, 0.63, x))
        return y

    def sec(x):
        if not v1 and abs(x) > half:
            # the turned-down lip along each end: the end section again, 3.5 mm in and 3 cm lower (515336126: a
            # thick rolled edge round the tyre's arch; the ends were cut off thin)
            return [(math.copysign(half - 0.004, x), py, pz - 0.030) for _, py, pz in
                    _rear_v2_section(sc, math.copysign(half, x), half, deck_front)]
        if not v1:
            return _rear_v2_section(sc, x, half, deck_front)
        a = abs(x) / half
        yb = -0.705                                  # straight rear face, clear of the tyres
        yf = deck_front(x)
        zt_rear, zt_front = (0.27, 0.33) if v1 else (0.31, 0.35)
        zt = zt_rear - 0.02 * a ** 3
        ztf = zt_front - 0.02 * a ** 3
        over_tyre = abs(x) > 0.46
        zf = 0.30 if over_tyre else 0.22            # fender front faces stay above the tyre
        if not over_tyre and x > 0:                 # and over the chain cover (its top ~0.235 at the axle sprocket)
            zf += 0.04 * _smoothstep(0.255, 0.285, x) * (1 - _smoothstep(0.38, 0.42, x))
        if v1:
            # (onboard frames) a flat top panel behind the seat out to a rear edge curved in plan, and below it
            # a shelf right across, standing out behind, then the lower face down to just over the band. In the
            # middle the edge drops to the shelf in a vertical step (the lettering); out to the sides the shelf
            # line rises and meets the edge in a slope facing back and up (the orange decals). (It was a soft
            # step to the deck; then a step in the middle only, the sides straight down.)
            sd = _smoothstep(0.26, 0.42, abs(x))                       # 0: the vertical step, 1: the side slopes
            # (rear view) the sides curve up over the tyres, the round decals on their rear slopes
            ztop, zl, d = v1_ztop(x), 0.255, 0.065
            yr = -0.700 + 0.020 * (abs(x) / half) ** 2                   # the rear edge, curved in plan
            zle = zl + 0.030 * sd                                         # the shelf line, rising to the sides
            face_mid = (yr * (1 - sd) + (yr - 0.5 * d) * sd, (zl + 0.012) * (1 - sd) + ((ztop - 0.015 + zle) / 2) * sd)
            shelf = (yr - 0.5 * d * (1 - sd) - 0.85 * d * sd, zl * (1 - sd) + (zle + 0.010) * sd)
            pts = [(x, yf + 0.012, zf), (x, yf, ztop - 0.04), (x, yf - 0.02, ztop), (x, (yf + yr) / 2, ztop + 0.004),
                   (x, yr + 0.012, ztop), (x, yr, ztop - 0.015), (x, *face_mid), (x, *shelf),
                   (x, yr - d + 0.006, zle), (x, yr - d, zle - 0.022), (x, yr - d, 0.18), (x, yr - d + 0.018, 0.165)]
            # over the tyre the ends round down in a quarter circle (rear view) to a thin lip just over the tyre
            # (the section was cut off straight down at the ends: square corners)
            t = min(max((abs(x) - V1_ROUND_X) / (half - V1_ROUND_X), 0.0), 1.0)
            f = 1.0 - 0.80 * (1.0 - math.sqrt(1.0 - t * t))
            return [(px, py, pz if pz <= V1_ROUND_Z else V1_ROUND_Z + (pz - V1_ROUND_Z) * f) for px, py, pz in pts]

    xs = np.r_[np.linspace(-half, -0.36, 6), np.linspace(-0.33, 0.33, 15), np.linspace(0.36, half, 6)]
    if not v1:  # 3 cm columns across the middle for the recess's edges (the exhaust opening's are still at +-0.09),
        # ~3.5 cm over the ridges, valleys and humps outboard
        side = np.r_[np.linspace(0.36, half, 9), half + 0.001]
        xs = np.r_[-side[::-1], np.linspace(-0.33, 0.33, 23), side]
    if v1:      # denser over the tyres for the rounded ends
        side = np.r_[0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.59, 0.615, 0.632, half]
        xs = np.r_[-side[::-1], np.linspace(-0.33, 0.33, 15), side]
    m, arc = _section_loft(sec, xs, cap=False)          # open sides: the tyres show past the cover
    vcol = arc.reshape(len(xs), -1)
    vcol = vcol[:, -1:] - vcol                          # each section's own length up from its bottom edge
    vrow = vcol.mean(0)
    npt = len(sec(0.0))
    if v1:
        vtop = float(vrow[0])
        _uv_from(m, np.repeat(xs, npt), np.tile(vrow, len(xs)), ATLAS["rear"], (-half, half), (0.0, vtop))
    else:
        # v by each section's own length, so the print keeps its shape where the sections differ (the round
        # stickers on the band, taller at the sides, came out stretched with one length for all)
        vtop = float(vcol.max())
        _uv_from(m, np.repeat(xs, npt), vcol.ravel(), ATLAS["rear"], (-half, half), (0.0, vtop))
    if not v1:
        # the opening the twin tips come out of, under the plate (exhaust.jpg, rear.jpg): the face between rows 7 and
        # 8 across the middle four columns is left open, the subsurf rounds its corners and the solidify's black rim
        # lines it, the silencer showing behind. The tips went through the face, which showed inside them.
        c = [m.verts[list(f)].mean(0) for f in m.faces]
        keep = [i for i, q in enumerate(c) if not (abs(q[0]) < EXH_HOLE[0] and 0.13 + EXH_DZ < q[2] < 0.19 + EXH_DZ and q[1] < -0.67)]
        m.faces, m.uvs = [m.faces[i] for i in keep], [m.uvs[i] for i in keep]
    m.modifiers = [("subsurf", 1), ("solidify", 0.004)]     # one level keeps the steps and creases crisp (two
    # rounded the v2's back into a smooth tub)
    m.sharp_angle = 85
    sc.add("rear_cover", "body_livery", m)
    sc.panels["rear"] = dict(u=(-half, half), v=(0.0, vtop), rows=vrow.tolist(),
                             kind="rental_cover_v1" if v1 else "rental_cover",
                             marks={k: float(vrow[i]) for k, i in ({} if v1 else REAR_ROWS).items()},
                             cols=[(float(x), [float(c) for c in col]) for x, col in zip(xs, vcol)],
                             colz=[(float(x), [float(q[2]) for q in sec(x)]) for x in xs])
    if not v1:
        # the number plate: a flat yellow plate hanging on the rear face over the exhaust (rear.jpg, rear-3q.jpg),
        # wide at the top with a narrower tab under it. Its face shows the front panel's number plate (the same
        # yellow and number; textures._panel_fleet): painted on the cover behind it, it showed round its edges
        z0, z1, pw = REAR_PLATE
        k = 0.105 / pw                                      # plate metres -> the front plate's (21 cm wide)
        pp = sc.panels["panel"]
        tot = pp["v"][1]
        pv = (tot * 0.61 + tot * 0.93) / 2                  # the front plate's middle (textures._panel_fleet)
        ph = (z1 - z0) * k                                  # its height on the front plate, keeping the number's shape
        yp = REAR_FACE_Y - 0.018                       # hung just behind the lip's round, over the recess (at -0.012 it cut into the lip)
        plate = []
        # a shield (rear.jpg): rounded top corners, the sides tapering in over its lowest 5.5 cm to 13 cm across at
        # the bottom (it was a square plate on a narrower tab); one convex outline
        rc, zs = 0.025, z0 + 0.055
        outline = [(-0.76 * pw, z0), (0.76 * pw, z0), (pw, zs)]
        outline += [(pw - rc + rc * math.cos(math.radians(a)), z1 - rc + rc * math.sin(math.radians(a))) for a in (0, 30, 60, 90)]
        outline += [(-pw + rc + rc * math.cos(math.radians(a)), z1 - rc + rc * math.sin(math.radians(a))) for a in (90, 120, 150, 180)]
        outline += [(-pw, zs)]
        for za, zb in ((z0, z1),):
            pm = ml.loft([[(x, yp, z) for x, z in outline], [(x, yp + 0.004, z) for x, z in outline]],
                         closed=True, cap_start=True, cap_end=True)
            pm.smooth = False
            u0, v0, u1, v1_ = ATLAS["panel"]
            pu = lambda u: u0 + (u - pp["u"][0]) / (pp["u"][1] - pp["u"][0]) * (u1 - u0)          # noqa: E731
            pvv = lambda v: v0 + v / tot * (v1_ - v0)                                               # noqa: E731
            uvs = []
            for f in pm.faces:
                pts = pm.verts[list(f)]
                if np.allclose(pts[:, 1], yp):            # the face looking back: the front plate's print
                    uvs.append(tuple((pu(-0.006 + px * k), pvv(pv + ((pz - z0) / (z1 - z0) - 0.5) * ph))
                                     for px, _, pz in pts))
                else:                                     # edges and back: the plate's yellow
                    uvs.append(tuple((pu(-0.09), pvv(pv - 0.03)) for _ in f))
            pm.uvs = uvs
            plate.append(ml.orient_outward(pm, np.array([0.0, yp + 0.002, (za + zb) / 2])))
        sc.add("rear_plate", "body_livery", ml.merge(plate))
    for x in ((-0.045, 0.045) if not v1 else ()):       # the v1 has one exhaust under the cover (_engine_gx)
        # a closed ring section: the tube's inside wall too (it had only the outside and the lip, so in game,
        # which draws one side of each face, the tips were see-through from behind), and a sooty plug
        # 4 cm in, so a look down the tip doesn't show the silencer's shiny can. ~5 cm across, 9 cm apart,
        # standing ~7 cm out behind the cover's face (exhaust.jpg scaled by the plate, rear.jpg; they were
        # 4.4 cm, 8 cm apart, 2.5 cm out)
        o = (x, -0.775, EXH_TIP_Z)
        sc.add("exhaust_tip_%s" % ("L" if x < 0 else "R"), "exhaust",
               ml.lathe([(0.0228, 0), (0.025, 0), (0.025, 0.105), (0.0228, 0.105), (0.0228, 0)], 18, origin=o,
                        axis=(0, 1, 0.0)))
        plug = ml.lathe([(0.0, 0.04), (0.0233, 0.04)], 18, origin=o, axis=(0, 1, 0.0))
        v, f = plug.verts, plug.faces[0]
        if np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])[1] > 0:       # face out of the tip (-y)
            plug.faces = [tuple(reversed(ff)) for ff in plug.faces]
            plug.uvs = [tuple(reversed(u)) for u in plug.uvs]
        sc.add("exhaust_soot_%s" % ("L" if x < 0 else "R"), "engine_black", plug)
    if not v1:
        # the flat end plate on the silencer's back that the two tips are welded into (exhaust.jpg), inside the opening
        # (sooty: bright, it showed as a silver box behind the tips; exhaust.jpg shows a dark recess round them)
        sc.add("exhaust_tip_plate", "engine_black", ml.box((0.0, -0.6825, EXH_TIP_Z), (0.158, 0.005, 0.060)))
    if v1:                                              # the v1's tank isn't in the cover: no filler cap
        return
    # rubber T-latches holding the cover down, on the humps (protector-v2-rear-left.jpg: one on the left;
    # rear-cover-right.jpg: two on the right by the engine), each a base pad on the shell and a T-handle
    latches = []
    # down in the cover's depressions, the valleys between the deck's ridges and the humps over the tyres (the user;
    # the studio rear and rear 3/4 shots, 515631146): the left one and the right's lower one mirror each other near the
    # back, ~0.64 m behind the middle over the round stickers; the right's upper one further forward at the foot of the
    # engine tower (triangulated 0.38 out, 0.49 back). (The left one stood forward on the ridge behind the seat.)
    for x, y in ((-0.385, float(unwarp_y(-0.64))), (0.380, float(unwarp_y(-0.49))), (0.385, float(unwarp_y(-0.64)))):
        # square to the shell where it sits (on the humps' slopes); 8 mm into the shell's cage, which the
        # smoothing pulls in over the humps (on the cage they stood 2 cm proud)
        q = _reach(sc, ("rear_cover",), (x, y, 0.60), 0.008)
        qx = _reach(sc, ("rear_cover",), (x + 0.01, y, 0.60), 0.008)
        qy = _reach(sc, ("rear_cover",), (x, y + 0.01, 0.60), 0.008)
        up = np.cross(qx - q, qy - q)
        up /= np.linalg.norm(up)
        fwd = (qy - q) / np.linalg.norm(qy - q)
        F = ml.frame(q, np.cross(fwd, up), fwd)
        for c, size in (((0, 0, 0.004), (0.032, 0.05, 0.008)), ((0, 0, 0.016), (0.012, 0.016, 0.020)),
                        ((0, 0, 0.028), (0.036, 0.016, 0.010))):
            latches.append(ml.box(c, size).transformed(F))
    sc.add("rear_cover_latches", "grip_black", ml.merge(latches))


def _plate_frame(c, t, n):
    """Transform from a box modelled at c (x along, y through, z up) to one lying along t, facing n."""
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = t, n, np.cross(t, n)
    if m[2, 2] < 0:
        m[:3, 2] = -m[:3, 2]
        m[:3, 1] = -m[:3, 1]
    c = np.asarray(c, float)
    return ml.translate(c) @ m @ ml.translate(-c)


def _protector(sc, chassis):
    """Rental steel protector v2, measured from side-right.jpg, front.jpg and rear.jpg: two stacked
    tubes ~5 cm apart (centres 0.08 / 0.13 m) all round, laid out by PROTECTOR_PLAN; one long arch on
    each half of the rear that rises from the rail near the centre, runs out across the back and comes
    down round the corner (it ran on forward along the side); a low rounded arch at the engine on each
    side (it was a steep ramp peaking at 0.335). Nothing rises at the front wheels."""
    R = PROTECTOR_R
    LOW, HIGH = PROTECTOR_Z
    SIDE, REAR = PROTECTOR_SIDE, PROTECTOR_REAR
    plan = PROTECTOR_PLAN

    def ring(z):
        # level all round: from the side both tubes are straight lines at one height (the user; a 4 cm rise round the
        # back came from reading the lower tube on the upper one's line in the rear shot)
        pts = [(x, y, z) for x, y in plan]
        loop = pts + [(-x, y, z) for x, y, _ in reversed(pts[1:])]
        return ml.catmull(np.array(loop), 6, closed=True)

    low_path, high_path = ring(LOW), ring(HIGH)
    sc.add("protector_lo", "protector", ml.sweep(low_path, ml.circle(R, 12), closed_path=True))
    sc.info["protector_curves"] = [("lo", low_path.tolist()), ("hi", high_path.tolist())]
    sc.add("protector_hi", "protector", ml.sweep(high_path, ml.circle(R, 12), closed_path=True))

    def near(path, x, y):
        d = (path[:, 0] - x) ** 2 + (path[:, 1] - y) ** 2
        return path[int(np.argmin(d))]

    hoops, side_hoops = [], []
    for s_ in (1, -1):
        # a rounded arch that wraps the corner (515631146, 514422718, the studio rear and rear 3/4 shots; the user: "round
        # on the back"): up from the top tube 0.145 out in a wide bend, its top running out across the back 0.25 over
        # the frame and round the protector's own corner curve, then down in another wide bend onto the top tube on the
        # side, just ahead of the corner. (It was a box in the rear plane with tight corners, its inner leg at 0.26.)
        # (the rear shot unprojected: inner leg 0.145 out, top 0.277 m off the ground; outline-checked over the rear shot)
        top = 0.257
        a = near(high_path, s_ * 0.145, REAR)
        # (inner bend wide and smooth into the upright leg, the user)
        # the outer part follows the protector's own corner in plan (the user: from above, the same path as the tubes
        # under it), level, then comes down onto the top tube just ahead of the corner's end
        # (round the second bend and along the angled straight, coming down gently to land on it ~0.65 out: the user,
        # the rear shot)
        sel = [k for k in range(len(high_path)) if 0.46 < high_path[k][0] * s_ < 0.655 and high_path[k][1] < -0.69]
        hp = high_path[sel]
        hp = hp[np.argsort(np.abs(hp[:, 0]))]                   # out along the curve
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(hp[:, :2], axis=0), axis=1))]
        drop = _smoothstep_v(arc[-1] - 0.13, arc[-1], arc)      # down onto the angled straight, ~0.65 out
        # (eased 1.2 cm back over the top, where it doesn't meet the tube, to clear the rear cover's corner; it came to 8 mm)
        outer = [(q[0], q[1] - 0.012 * (1 - d), top + (q[2] - top) * d) for q, d in zip(hp, drop)]
        arch = np.array([a, (s_ * 0.147, a[1] - 0.002, top - 0.10), (s_ * 0.156, REAR - 0.003, top - 0.05),
                         (s_ * 0.185, REAR - 0.006, top - 0.013), (s_ * 0.25, REAR - 0.010, top), (s_ * 0.45, REAR - 0.012, top)]
                        + outer)
        hoops.append(ml.tube(ml.catmull(arch, 6), R, 10))
        sc.info.setdefault("protector_curves", []).append(("rear%+d" % s_, ml.catmull(arch, 6).tolist()))
        # the side hoop, the same on both sides (514252952 and 515336126 show the left)
        f1 = near(high_path, s_ * PROTECTOR_SIDE_P, unwarp_y(0.215))
        xr = s_ * PROTECTOR_SIDE_P
        # traced over the studio side shot (tools/dev/photo_trace.py, the tube's centre line unprojected onto its plane):
        # an upright rear leg 0.28 m behind the middle, a broad round bend over the top ~0.276 m off the ground, its
        # highest point ~0.185 m behind the middle, running straight on into the diagonal down to the top tube under the
        # pod's front (outline-checked over the studio side and outdoor shots: it peaked 1.5 cm higher in a tighter bend)
        f0 = near(high_path, s_ * SIDE, unwarp_y(-0.285))
        ramp = [f0] + [(xr, float(unwarp_y(yr)), zw - 0.02) for yr, zw in
                       ((-0.281, 0.17), (-0.279, 0.21), (-0.270, 0.240), (-0.250, 0.262), (-0.220, 0.273),
                        (-0.185, 0.276), (-0.150, 0.272), (-0.110, 0.262), (-0.060, 0.242), (0.039, 0.201),
                        (0.150, 0.155))] + [f1]
        hoops.append(ml.tube(ramp, R * 0.85, 10, 0.02))
        side_hoops.append((s_, ramp))
        sc.info.setdefault("protector_curves", []).append(("side%+d" % s_, [list(map(float, q)) for q in ramp]))
    sc.add("protector_hoops", "protector", ml.merge(hoops))

    posts = []
    spacers = [(0.0, PROTECTOR_FRONT), (0.30, protector_front_y(0.30)), (0.55, protector_front_y(0.55)),
               (PROTECTOR_SIDE_F, 0.70), (PROTECTOR_SIDE_F, 0.36), (PROTECTOR_SIDE_P, 0.0), (SIDE, -0.40),
               (SIDE - 0.045, REAR + 0.04), (0.50, REAR)]
    for x, y in spacers:
        for s_ in ((1, -1) if x else (1,)):
            posts.append(ml.tube([near(low_path, s_ * x, y), near(high_path, s_ * x, y)], 0.012, 8))
    sc.add("protector_posts", "protector", ml.merge(posts))

    # bolted splice plates where the tube sections join: flat bars on the outside across both tubes, with button
    # head bolts (rear.jpg / the studio front and rear shots, 515250929 close-up): a long one at the rear centre
    # (5 bolts), short ones each side of it at the rear (2), one at the front centre (4), and one under each front
    # wheel and by each pod on the sides
    plates, bolts = [], []

    def plate(x, y, length, n_bolts):
        i = int(np.argmin((low_path[:, 0] - x) ** 2 + (low_path[:, 1] - y) ** 2))
        p0 = low_path[i]
        t = low_path[(i + 1) % len(low_path)] - low_path[i - 1]
        t[2] = 0.0
        t /= np.linalg.norm(t)
        n = np.cross(t, (0, 0, 1.0))
        if n @ (p0[0], p0[1], 0.0) < 0:        # out of the kart
            n = -n
        c = np.array([p0[0], p0[1], (LOW + HIGH) / 2]) + n * (R + 0.003)
        plates.append(ml.box(tuple(c), (length, 0.005, 0.026)).transformed(_plate_frame(c, t, n)))
        for k in range(n_bolts):
            u = (k - (n_bolts - 1) / 2) * (length - 0.03) / max(n_bolts - 1, 1)
            o = c + t * u + n * 0.003
            bolts.append(ml.lathe([(0.0, 0.0), (0.0075, 0.0), (0.0065, 0.003), (0.004, 0.005), (0.0, 0.0055)], 10,
                                  origin=tuple(o), axis=tuple(n), closed_ends=True))
        return c, t, n

    plate(0.0, REAR, 0.42, 5)
    # the front one is the protector's mount (514401612, the user: the support is bolted there): a backing plate behind
    # the two tubes, the four bolts through both with nuts behind, and the two arms from the front bumper's corners
    # coming in onto it. (The arms went to the lower tube ~0.26 out and the plate held nothing.)
    fc, ft, fn = plate(0.0, PROTECTOR_FRONT + 0.1, 0.17, 4)
    back = fc - fn * (2 * R + 0.006 + 0.005)
    plates.append(ml.box(tuple(back), (0.17, 0.005, 0.034)).transformed(_plate_frame(back, ft, fn)))
    for k in range(4):
        o = back + ft * ((k - 1.5) * 0.14 / 3) - fn * 0.0025
        bolts.append(ml.lathe([(0.0, 0.0), (0.0065, 0.0), (0.0065, 0.005), (0.0, 0.005)], 6, origin=tuple(o),
                              axis=tuple(-fn), closed_ends=True))
        bolts.append(ml.tube([o + fn * 0.002, o + fn * (2 * R + 0.02)], 0.003, 6))     # the bolt's shank through
    for s_ in (1, -1):
        plate(s_ * 0.45, REAR, 0.11, 2)
        # (the studio side shot: beside the front wheel; there's none along the pod). On the side straight, ending ahead of
        # the bend into the front's diagonal: at 0.677 it sat on the bend, a straight bar standing off the curving tube
        plate(s_ * PROTECTOR_SIDE_F, float(unwarp_y(0.61)), 0.13, 2)
    sc.add("protector_plates", "black_anodized", ml.merge(plates))
    sc.add("protector_bolts", "steel", ml.merge(bolts))

    # the protector is held off the frame on big black rubber buffers (514248494, 515768936, 514252952, 515631146):
    # steel arms out from the frame (the nerf bars at the sides, the frame's end at the back) to a fat rubber cylinder
    # ~6.5 cm across, 7 cm long, bolted against the protector's lower tube. (They were thin steel tubes straight to the tube.)
    mounts, rubbers = [], []

    def buffer(frame_pt, x, y, path=None):
        q = near(low_path if path is None else path, x, y)
        d = np.array([q[0] - frame_pt[0], q[1] - frame_pt[1], 0.0])
        d /= np.linalg.norm(d)
        r_end = q - d * (R + 0.002)                    # the buffer's face against the tube
        r_start = r_end - d * 0.070
        mounts.append(ml.tube([frame_pt, (r_start[0], r_start[1], frame_pt[2]), r_start - d * 0.004], 0.011, 8, 0.02))
        rubbers.append(ml.lathe([(0.0, 0.0), (0.028, 0.0), (0.033, 0.008), (0.033, 0.062), (0.028, 0.070), (0.0, 0.070)],
                                18, origin=tuple(r_start), axis=tuple(d), closed_ends=True))

    for s_ in (1, -1):
        # on each side one under the side hoop's front foot and one under its rear leg, below the engine's red shroud
        # (the studio side shot: 0.24 m ahead of and 0.28 m behind the middle, real; the rear one sat ahead of the engine).
        # The rear one's arm comes off the engine mount bar on the right, off the frame rail on the left
        buffer((s_ * 0.58, 0.20, 0.10), s_ * PROTECTOR_SIDE_P, float(unwarp_y(0.237)))
        yb_ = float(unwarp_y(-0.281))
        buffer((0.57, GX_Y - 0.11, 0.05) if s_ > 0 else (-0.35, yb_, 0.055), s_ * SIDE, yb_)
        # (at the back against the upper tube, showing between the tubes and the cover in the rear 3/4 shot)
        buffer((s_ * 0.39, -0.60, 0.11), s_ * 0.45, REAR, high_path)
        # from the front bumper's rounded corner (the corner point itself is 1 cm off the fillet) in to the backing plate
        # at the front centre, ending on a welded tab flat against it
        tab = back + ft * (s_ * 0.062) - fn * 0.004
        mounts.append(ml.tube([_reach(sc, ("bumper_front_lo",), (s_ * 0.20, 0.93, 0.06)),
                               tab - fn * 0.03 + ft * (s_ * 0.03), tab], 0.011, 8, 0.02))
    sc.add("protector_mounts", "protector", ml.merge(mounts))
    sc.add("protector_rubbers", "grip_black", ml.merge(rubbers))
    # rubber bush mounts from each side hoop's diagonal in to its pod's side (514252952: a short black cylinder on a
    # stub, on the pod's lower side)
    bushes = []
    for s_, ramp in side_hoops:
        for yr in (0.10,):
            yd = float(unwarp_y(yr))
            path = ml.catmull(np.array(ramp, float), 8)
            q = path[int(np.argmin(np.abs(path[:, 1] - yd)))]
            a_ = np.array([s_ * (abs(q[0]) - R * 0.6), q[1], q[2]])
            b_ = _reach(sc, ("sidepod_" + ("R" if s_ > 0 else "L"),), (s_ * 0.60, q[1], q[2]), 0.002)  # on the pod
            ln = abs(a_[0] - b_[0])
            bushes.append(ml.lathe([(0, 0), (0.019, 0), (0.023, 0.005), (0.023, ln * 0.6), (0.019, ln * 0.65),
                                    (0, ln * 0.65)], 16, origin=tuple(b_), axis=(s_, 0, 0), closed_ends=True))
            bushes.append(ml.tube([a_, b_ + (s_ * ln * 0.5, 0, 0)], 0.007, 8))
    sc.add("protector_bushes", "black_plastic", ml.merge(bushes))

def _protector_band(sc, chassis):
    """Rental protector v1 (protector-v1-side-3q.jpg, -rear-3q.jpg, -rear-axle.jpg, -top.jpg): a flat black
    band ~10 cm tall all round, on the v2's plan pinched in between the wheels (BAND_PLAN), bolted to brackets off the
    frame at the sides and the front and pressed by two rubber buffers at the back, round bolt heads along
    its outer face."""
    # the right side (BAND_PLAN_R) stays out past the engine; the left is pinched in all along (BAND_PLAN)
    pts = [(x, y, 0.0) for x, y in BAND_PLAN_R]
    left = [(x, y, 0.0) for x, y in BAND_PLAN]
    loop = pts + [(-x, y, 0.0) for x, y, _ in reversed(left[1:])]
    path = ml.catmull(np.array(loop), 6, closed=True)
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6]]
    n = len(path)
    tang = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    out = np.c_[tang[:, 1], -tang[:, 0], np.zeros(n)]
    centre = path.mean(0)
    flip = np.einsum("ij,ij->i", out, path - centre) < 0
    out[flip] *= -1
    z0, z1, th = BAND_Z[0], BAND_Z[1], BAND_T
    prof = [(-th / 2, z0), (th / 2, z0), (th / 2, z1), (-th / 2, z1)]
    verts = np.array([p_ + o * a + np.array([0.0, 0.0, c]) for p_, o in zip(path, out) for a, c in prof])
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    faces, uvs = [], []
    k = len(prof)
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((i * k + j, i * k + j2, i2 * k + j2, i2 * k + j))
            v1, v2 = length[i] * 4, (length[i2] if i2 else length[-1]) * 4
            uvs.append(((j / k, v1), ((j + 1) / k, v1), ((j + 1) / k, v2), (j / k, v2)))
    band = ml.orient_outward(ml.Mesh(verts, faces, uvs), centre + np.array([0, 0, (z0 + z1) / 2]))
    band.sharp_angle = 60
    sc.add("protector_band", "black_plastic", band)

    def near(x, y):
        i = int(np.argmin((path[:, 0] - x) ** 2 + (path[:, 1] - y) ** 2))
        return path[i], out[i]

    zm = (z0 + z1) / 2
    # brackets off the frame to the band's inner face (the v2's mounts), bolted through
    mounts, bolts = [], []
    for s_ in (1, -1):
        for frm, (bx, by) in (((s_ * 0.58, 0.20, 0.10), (s_ * BAND_SIDE_P, 0.20)),
                              ((s_ * 0.62, 0.06, 0.11), (s_ * BAND_SIDE_P, 0.06)),
                              ((s_ * 0.20, 0.93, 0.06), (s_ * 0.26, protector_front_y(0.26)))):
            q, o = near(bx, by)
            if frm[1] > 0.9:                         # from the front bumper's rounded corner, not 1 cm off it
                frm = _reach(sc, ("bumper_front_lo",), frm)
            mounts.append(ml.tube([frm, q - o * (th / 2 + 0.006) + np.array([0, 0, zm])], 0.012, 8))
        # rear: brackets back from the frame's end to two rubber buffers against the band
        q, o = near(s_ * 0.33, PROTECTOR_REAR)
        inner = q - o * (th / 2) + np.array([0, 0, zm])
        mounts.append(ml.tube([(s_ * 0.33, -0.60, 0.085), inner - o * 0.06], 0.012, 8))
        sc.add("protector_buffer_" + ("R" if s_ > 0 else "L"), "grip_black",
               ml.lathe([(0.0, 0.0), (0.03, 0.0), (0.034, 0.03), (0.03, 0.058), (0.0, 0.058)], 16,
                        origin=tuple(inner - o * 0.058), axis=tuple(o), closed_ends=True))
    sc.add("protector_mounts", chassis, ml.merge(mounts))
    # round bolt heads along the outer face, ~40 cm apart
    for a in np.arange(0.0, length[-1], 0.40):
        i = int(np.searchsorted(length, a))
        i = min(i, n - 1)
        bolts.append(ml.lathe([(0.0, 0.0), (0.011, 0.0), (0.009, 0.004), (0.0, 0.006)], 12,
                              origin=tuple(path[i] + out[i] * th / 2 + np.array([0, 0, zm])), axis=tuple(out[i]),
                              closed_ends=True))
    sc.add("protector_band_bolts", "grip_black", ml.merge(bolts))


def _axle_z(sc):
    """Rear axle height in BODY coordinates (the frame rides `ride` above where it is modelled)."""
    return sc.params["tyre_r"][0] - sc.params.get("ride", 0.0)


def _chain(sc, x, p1, r1, p2, r2):
    """Chain loop between two sprockets in the yz plane at lateral x."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    pts = [p1 + nrm * r1, p2 + nrm * r2]
    arc2 = [p2 + r2 * np.array([math.cos(a), math.sin(a)]) for a in
            np.linspace(math.atan2(nrm[1], nrm[0]), math.atan2(nrm[1], nrm[0]) - math.pi, 12)]
    pts2 = [p2 - nrm * r2, p1 - nrm * r1]
    arc1 = [p1 + r1 * np.array([math.cos(a), math.sin(a)]) for a in
            np.linspace(math.atan2(-nrm[1], -nrm[0]), math.atan2(-nrm[1], -nrm[0]) - math.pi, 8)]
    loop = [pts[0]] + arc2 + [pts2[1]] + arc1
    path = np.array([(x, a, b) for a, b in loop])
    prof = [(-0.004, -0.006), (0.004, -0.006), (0.004, 0.006), (-0.004, 0.006)]
    chain = ml.sweep(path, [(b, a) for a, b in prof], closed_path=True)
    chain.smooth = False
    sc.add("chain", "chain", chain)


def _engine_125(sc, chassis, sprocket_x):
    """Water-cooled 125cc direct-drive 2-stroke (TaG style), right side."""
    ex, ey = 0.27, -0.30
    case = ml.box((ex, ey, 0.135), (0.17, 0.17, 0.13))
    case.modifiers = [("bevel", 0.02, 3)]
    sc.add("engine_case", "engine_alu", case)
    cyl = ml.box((ex, ey + 0.01, 0.255), (0.10, 0.10, 0.11))
    cyl.modifiers = [("bevel", 0.012, 2)]
    head = ml.lathe([(0, 0), (0.058, 0), (0.058, 0.03), (0.045, 0.045), (0, 0.045)], 24,
                    origin=(ex, ey + 0.01, 0.31), axis=(0, 0, 1), closed_ends=True)
    sc.add("engine_cylinder", "engine_alu", cyl)
    sc.add("engine_head", "engine_black", head)
    sc.add("engine_plug", "black_plastic", ml.tube([(ex, ey + 0.01, 0.355), (ex, ey + 0.01, 0.39)], 0.009, 8))
    sc.add("engine_mount", "engine_black", ml.box((ex, ey, 0.065), (0.20, 0.18, 0.016)))
    # clutch drum + engine sprocket outboard
    sc.add("clutch", "engine_black", ml.lathe([(0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)], 24,
                                             origin=(ex + 0.085, ey, 0.13), axis=(1, 0, 0), closed_ends=True))
    sc.add("sprocket_engine", "steel", sprocket_mesh(0, 0.028, teeth=11).transformed(
        ml.translate((sprocket_x, ey, 0.13))))
    _chain(sc, sprocket_x, (ey, 0.13), 0.028, (Y_R, _axle_z(sc)), 0.090)
    # carburettor + airbox (twin intake) beside the seat
    sc.add("carb", "engine_alu", ml.lathe([(0, 0), (0.025, 0), (0.025, 0.07), (0, 0.07)], 16,
                                          origin=(ex - 0.06, ey + 0.04, 0.16), axis=(-1, 0.4, 0), closed_ends=True))
    airbox = ml.box((0.27, -0.03, 0.22), (0.12, 0.20, 0.13))
    airbox.modifiers = [("bevel", 0.03, 3)]
    sc.add("airbox", "engine_alu", airbox)
    sc.add("airbox_inlets", "black_plastic", ml.merge([
        ml.lathe([(0.018, 0), (0.024, 0), (0.024, 0.04), (0.018, 0.04)], 16, origin=(0.27 + dx, 0.07, 0.22), axis=(0, 1, 0))
        for dx in (-0.03, 0.03)]))
    sc.add("intake_hose", "black_plastic", ml.tube([(0.27, -0.13, 0.2), (0.23, -0.22, 0.18), (ex - 0.07, ey + 0.08, 0.17)], 0.02, 10, 0.03))
    # expansion chamber behind the seat to the left, silencer
    path = ml.catmull(np.array([(ex, ey - 0.04, 0.27), (ex - 0.01, -0.44, 0.29), (0.17, -0.60, 0.31),
                                (0.0, -0.645, 0.31), (-0.15, -0.63, 0.30)]), 8)
    rad = np.interp(np.linspace(0, 1, len(path)), [0, 0.15, 0.45, 0.75, 1.0], [0.020, 0.035, 0.052, 0.042, 0.018])
    pipe = ml.sweep(path, ml.circle(1.0, 16), caps=True, scale=rad)
    sc.add("exhaust_pipe", "exhaust", pipe)
    sil = ml.lathe([(0, 0), (0.038, 0), (0.042, 0.02), (0.042, 0.20), (0.03, 0.22), (0.015, 0.22), (0.015, 0.25), (0, 0.25)],
                   20, origin=(-0.15, -0.63, 0.30), axis=(-1.0, -0.15, 0.0), closed_ends=True)
    sc.add("silencer", "exhaust", sil)
    # radiator on the left side, behind the seat back (clear of the seat)
    rx, ry = RADIATOR_X, RADIATOR_Y
    core = ml.box((rx, ry, 0.34), (0.24, 0.05, 0.27))
    core.modifiers = [("bevel", 0.006, 1)]
    ml.project_uv(core, (0, 0, 0), (1, 0, 0), (0, 0, 1), (rx - 0.12, rx + 0.12), (0.20, 0.48), (0, 0, 1, 1))
    sc.add("radiator_core", "radiator", core)
    frame_r = ml.merge([ml.box((rx, ry, 0.205), (0.26, 0.06, 0.02)), ml.box((rx, ry, 0.475), (0.26, 0.06, 0.02)),
                        ml.box((rx - 0.125, ry, 0.34), (0.012, 0.06, 0.28)), ml.box((rx + 0.125, ry, 0.34), (0.012, 0.06, 0.28))])
    sc.add("radiator_frame", "aluminium", frame_r)
    shroud = ml.box((rx, ry + 0.035, 0.27), (0.25, 0.012, 0.14))
    ml.project_uv(shroud, (0, 0, 0), (-1, 0, 0), (0, 0, 1), (-rx - 0.125, -rx + 0.125), (0.20, 0.34), (0, 0, 1, 1))
    sc.add("radiator_shroud", "radiator_shroud", shroud)
    sc.add("radiator_cap", "black_plastic", ml.tube([(rx - 0.02, ry, 0.485), (rx - 0.02, ry, 0.52)], 0.012, 10))
    sc.add("radiator_stays", chassis, ml.merge([
        ml.tube([(-0.37, -0.42, 0.056), (rx - 0.10, ry, 0.21)], 0.008, 8),
        ml.tube([(-0.21, -0.40, 0.056), (rx + 0.10, ry, 0.21)], 0.008, 8),
        ml.tube([(rx + 0.11, ry, 0.47), (rx + 0.16, ry - 0.06, 0.53), (-0.05, -0.44, 0.52)], 0.007, 8, 0.03)]))
    sc.add("coolant_hoses", "black_plastic", ml.merge([
        ml.tube([(rx + 0.13, ry, 0.44), (0.0, -0.47, 0.42), (ex - 0.04, ey - 0.02, 0.33)], 0.012, 10, 0.08),
        ml.tube([(rx + 0.13, ry + 0.03, 0.24), (-0.10, -0.36, 0.26), (0.10, -0.36, 0.18), (ex - 0.08, ey - 0.03, 0.12)],
                0.012, 10, 0.06)]))
    sc.info["engine"] = dict(center=(ex, ey, 0.2), kind="kf125")


def _engine_cover_rental(sc):
    """The right pod's skirt over the seat's side (engine-air-filter.jpg, 514400663): the pod's roof runs on inboard
    over the clutch, chain and sprocket (_sidepods_rental, POD_R_ROOF_XI) and turns down at its inner edge into
    this skirt facing the seat; behind the cylinder head a louvred wall across under the roof faces back. Only
    the engine's outer side stays open."""
    x_skirt = POD_R_ROOF_XI - 0.002
    ys = np.linspace(0.0, float(unwarp_y(POD_R_END)), 14)
    # the louvres face back (the studio rear and rear 3/4 shots, engine close-up 514400663): a slotted wall
    # under the roof just behind the cylinder head, from the skirt out to the carburettor (triangulated:
    # x 0.24-0.46, z 0.31-0.40 off the ground); behind it the roof is open underneath and the engine's top
    # shows. (The slots were on an inboard skirt running the roof's whole length.)
    y_lv = float(unwarp_y(-0.36))

    def z_top(y):                                    # the roof's inner edge (the pod top's last row, with its lift)
        return _rental_pod_top(max(y, -0.50), 1) - 0.04 + 0.06 * _smoothstep(0.15, -0.05, float(warp_y(y)))

    def z_bottom(y):
        zb = min(0.235, _rental_pod_top(max(y, -0.50), 1) - 0.14)
        return zb + (z_top(y) - 0.012 - zb) * _smoothstep(y_lv + 0.005, y_lv - 0.025, y)

    skirt = ml.loft([[(x_skirt, y, z_bottom(y)), (x_skirt, y, (z_bottom(y) + z_top(y)) / 2), (x_skirt, y, z_top(y) + 0.004)]
                     for y in ys], closed=False)
    skirt.smooth = False
    f0 = skirt.faces[0]                              # faces the seat (-x); the solidify grows it into the pod
    if np.cross(skirt.verts[f0[1]] - skirt.verts[f0[0]], skirt.verts[f0[2]] - skirt.verts[f0[0]])[0] > 0:
        skirt.faces = [tuple(reversed(f)) for f in skirt.faces]
        skirt.uvs = None if skirt.uvs is None else [tuple(reversed(u)) for u in skirt.uvs]
    skirt.modifiers = [("solidify", 0.004)]
    sc.add("engine_cover_R", "black_plastic", skirt)
    x_skirt = POD_R_ROOF_XI
    # the louvred wall: from the skirt out to just inboard of the carburettor, from 0.28 up to the roof
    # its top follows the roof's underside, which rises from the curl at the skirt to the flat top outboard
    x0, x1, zb = x_skirt, 0.442, 0.285
    roof = lambda x: z_top(y_lv) + 0.031 * _smoothstep(POD_R_ROOF_XI, POD_R_ROOF_XI + 0.17, x) - 0.008  # noqa: E731
    xs = np.linspace(x0, x1, 9)
    wall = ml.loft([[(x, y_lv + dy, zb) for x in xs] + [(x, y_lv + dy, roof(x)) for x in xs[::-1]]
                    for dy in (-0.0025, 0.0025)], closed=True, cap_start=True, cap_end=True)
    wall.smooth = False
    sc.add("engine_cover_wall", "black_plastic", wall)
    zt = roof(x0 + 0.03)
    slots = [ml.box((x, y_lv - 0.003, zb + 0.45 * (zt - zb)), (0.009, 0.004, 0.62 * (zt - zb)))
             for x in np.linspace(x0 + 0.03, x1 - 0.03, 7)]
    sc.add("engine_cover_louvres", "grip_black", ml.merge(slots))


def _chain_guard(sc, sprocket_x, p1, r1):
    """Black plastic chain cover over the whole run, engine sprocket to axle sprocket: a top skin
    following the chain loop's upper half ~2.5 cm out from it, from ahead of the clutch drum round to
    behind the axle sprocket, with a side wall down its inboard edge; a strap at each end down to the
    frame. (It was a short arc round the axle sprocket only.)"""
    c1, c2 = np.asarray(p1, float), np.array([Y_R, _axle_z(sc)])
    R1, R2 = 0.068, 0.112               # clear of the clutch drum (r 0.055) and the axle sprocket's teeth
    w, t, depth = 0.026, 0.0025, 0.045  # half width over the chain, skin, inboard wall
    x = sprocket_x - 0.004
    L = float(np.linalg.norm(c2 - c1))
    u = (c2 - c1) / L
    perp = np.array([u[1], -u[0]]) if u[0] < 0 else np.array([-u[1], u[0]])      # the upward normal
    sa = (R1 - R2) / L
    nrm = math.sqrt(1 - sa * sa) * perp + sa * u          # common outer tangent: n.(c2 - c1) = R1 - R2
    a_n = math.atan2(nrm[1], nrm[0])

    def outline(dr):                     # the guard's line in (y, z), dr in from the skin
        # angles go from the front (+y, 0) up over the top to the back (+-pi): increasing from -35 deg
        a1 = np.linspace(math.radians(-35), a_n, 10)
        a2 = np.linspace(a_n, math.radians(158), 12)
        pts = [c1 + (R1 - dr) * np.array([math.cos(a), math.sin(a)]) for a in a1]
        pts += [c2 + (R2 - dr) * np.array([math.cos(a), math.sin(a)]) for a in a2]
        return pts

    skin = ml.sweep(np.array([(x, y, z) for y, z in outline(0.0)]), [(-w, -t), (w, -t), (w, t), (-w, t)])
    wall = ml.sweep(np.array([(x - w + t, y, z) for y, z in outline(depth / 2)]),
                    [(-t, -depth / 2), (t, -depth / 2), (t, depth / 2), (-t, depth / 2)])
    skin.smooth = wall.smooth = False
    sc.add("chain_guard", "black_plastic", ml.merge([skin, wall]))
    # straps from its two ends down to the frame: the rail ahead, the end cross tube behind
    o = outline(0.0)
    f0, r0 = np.array([x, *o[0]]), np.array([x, *o[-1]])
    sc.add("chain_guard_bracket", "black_anodized", ml.merge([
        ml.tube([f0, (x, f0[1] + 0.004, 0.064)], 0.006, 8),
        ml.tube([r0, (x, -0.600, 0.098)], 0.006, 8)]))


def _engine_gx(sc, chassis, sprocket_x):
    """Honda GX270/340/390 without its tank (engine-air-filter.jpg, side-right.jpg): crank across the
    kart, red fan shroud and recoil starter facing OUTWARD, PTO/clutch and chain on the inboard side,
    green cone filter pointing back, slotted heat shield over the cylinder, exhaust to twin tips at
    the rear centre."""
    ex, ey = GX_X, GX_Y
    base_z = 0.06
    sc.add("engine_mount", "engine_black", ml.box((ex - 0.03, ey, base_z), (0.24, 0.26, 0.014)))
    # the plate sits on the right rail and two bars out to the side
    sc.add("engine_mount_bars", chassis, ml.merge([
        # (out to the plate's outer edge: they ran 3 cm past it, into the air)
        ml.tube([(0.34, ey - 0.11, 0.05), (ex + 0.08, ey - 0.11, 0.05)], 0.011, 8),
        ml.tube([(0.34, ey + 0.11, 0.05), (ex + 0.08, ey + 0.11, 0.05)], 0.011, 8)]))
    # output shaft from the crankcase to the clutch
    sc.add("engine_pto", "steel", ml.lathe([(0, 0), (0.012, 0), (0.012, ex - 0.09 - sprocket_x - 0.04),
                                           (0, ex - 0.09 - sprocket_x - 0.04)], 12,
                                          origin=(sprocket_x + 0.045, ey, base_z + 0.085), axis=(1, 0, 0), closed_ends=True))
    # sized from side-right.jpg (tyres as the scale): red fan shroud ~0.25 m tall, recoil starter centre
    # ~0.19 m off the ground, engine top ~0.33 m. Shapes from Honda's own pictures in references/engine
    # (honda-gx*-recoil-side.jpg, -three-quarter.png, -dimensions.png) and rental/engine-air-filter.jpg: a
    # rounded cast crankcase on a flanged base, a square-finned cylinder leaning back with its head and the
    # "OHV" rocker cover, and the red fan housing round the flywheel and up over the cylinder's outer side
    # (it was a round red ball on a box).
    crank = np.array([ey, RECOIL_Z])                      # crank centre in (y, z)
    xo = ex + 0.137                                       # the fan housing's outer face

    def outline_yz(centre, half, n=3.0, k=32):
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        c, sn = np.cos(a), np.sin(a)
        return np.c_[centre[0] + half[0] * np.sign(c) * np.abs(c) ** (2 / n),
                     centre[1] + half[1] * np.sign(sn) * np.abs(sn) ** (2 / n)]

    def prism_x(outline, xs, scales, about):
        about = np.asarray(about, float)
        secs = [[(x, *(about + (q - about) * sc_)) for q in outline] for x, sc_ in zip(xs, scales)]
        return ml.loft(secs, closed=True, cap_start=True, cap_end=True)

    def slab(c, u, w, hu, hw, a0, a1, axis, n=4.0, k=24, taper=1.0):
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        cs, sn = np.cos(a), np.sin(a)
        ring = [(np.sign(x_) * abs(x_) ** (2 / n), np.sign(y_) * abs(y_) ** (2 / n)) for x_, y_ in zip(cs, sn)]
        secs = []
        for t, sc_ in ((a0, 1.0), (a1, taper)):
            secs.append([tuple(c + axis * t + u * hu * sc_ * x_ + w * hw * sc_ * y_) for x_, y_ in ring])
        return ml.loft(secs, closed=True, cap_start=True, cap_end=True)

    # crankcase: rounded casting on a flanged base, oil filler caps low at the front on both sides
    # sized as the GX390's (honda-gx390-dimensions.png): the cast body as big as the fan housing in front of it
    # (it was ~24 cm by 17 cm, the cylinder ~12 cm across the fins, and looked small behind the housing)
    case = prism_x(outline_yz((ey, base_z + 0.100), (0.136, 0.096), n=3.0),
                   [ex - 0.088, ex - 0.080, ex + 0.060, ex + 0.068], [0.90, 1.0, 1.0, 0.94], (ey, base_z + 0.100))
    case.modifiers = [("subsurf", 1)]
    sc.add("engine_case", "engine_alu", case)
    feet = [ml.box((ex - 0.01, ey + dy, base_z + 0.012), (0.17, 0.05, 0.018)) for dy in (-0.095, 0.095)]
    for dx in (-0.075, 0.055):
        for dy in (-0.105, 0.105):
            feet.append(ml.lathe([(0.0, 0.0), (0.009, 0.0), (0.009, 0.026), (0.0, 0.026)], 12,
                                 origin=(ex + dx, ey + dy, base_z + 0.003), axis=(0, 0, 1), closed_ends=True))
    sc.add("engine_feet", "engine_alu", ml.merge(feet))
    # the drive-side crankcase cover: a raised boss round the output shaft and its ring of bolts
    pto_c = (ex - 0.088, ey, base_z + 0.085)
    sc.add("engine_pto_cover", "engine_alu", ml.lathe([(0.0, 0.0), (0.072, 0.0), (0.066, 0.012), (0.034, 0.016),
                                                       (0.030, 0.026), (0.0, 0.026)], 28, origin=pto_c,
                                                      axis=(-1, 0, 0), closed_ends=True))
    sc.add("engine_pto_bolts", "steel", ml.merge([
        ml.lathe([(0, 0), (0.0055, 0), (0.0055, 0.006), (0, 0.006)], 6,
                 origin=(pto_c[0] + 0.002, pto_c[1] + 0.0825 * math.cos(a), pto_c[2] + 0.0825 * math.sin(a)),
                 axis=(-1, 0, 0), closed_ends=True) for a in np.linspace(0.3, 2 * math.pi + 0.3, 8, endpoint=False)
        if pto_c[2] + 0.0825 * math.sin(a) > base_z + 0.02]))
    sc.add("engine_oil_caps", "oil_cap", ml.merge([
        ml.lathe([(0.0, 0.0), (0.014, 0.0), (0.014, 0.012), (0.009, 0.018), (0.0, 0.018)], 14,
                 origin=(ex + dx, ey + 0.120, base_z + 0.050), axis=(0, 1, 0), closed_ends=True) for dx in (-0.06, 0.04)]))
    # cylinder leaning back, square cooling fins, the head and the "OHV" rocker cover with its bolts
    tilt = math.radians(35)
    cyl_axis = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    u_ax = np.array([1.0, 0.0, 0.0])
    w_ax = np.cross(cyl_axis, u_ax)                       # across the cylinder in the kart's side view
    cyl_o = np.array([ex - 0.015, ey - 0.040, base_z + 0.120])
    barrel = ml.lathe([(0, 0), (0.052, 0), (0.052, 0.112), (0, 0.112)], 20, origin=tuple(cyl_o), axis=cyl_axis,
                      closed_ends=True)
    fins = [slab(cyl_o, u_ax, w_ax, 0.074, 0.071, 0.014 + 0.0145 * k, 0.0175 + 0.0145 * k, cyl_axis) for k in range(7)]
    head = slab(cyl_o, u_ax, w_ax, 0.077, 0.075, 0.112, 0.146, cyl_axis, n=3.0, taper=0.97)
    head.modifiers = [("bevel", 0.006, 2)]
    sc.add("engine_cylinder", "engine_alu", ml.merge([barrel] + fins))
    sc.add("engine_head", "engine_alu", head)
    rocker = slab(cyl_o, u_ax, w_ax, 0.058, 0.064, 0.146, 0.176, cyl_axis, n=3.2, taper=0.88)
    rocker.modifiers = [("subsurf", 1)]
    sc.add("engine_rocker_cover", "engine_alu", rocker)
    # its four bolts sit down in the cover's top, well inside the corners its smoothing rounds off (at the
    # square corners they floated over it in game)
    top = cyl_o + cyl_axis * 0.176
    sc.add("engine_rocker_bolts", "steel", ml.merge([
        ml.lathe([(0, 0), (0.0055, 0), (0.0055, 0.008), (0, 0.008)], 6, origin=tuple(top + u_ax * du + w_ax * dw - cyl_axis * 0.007),
                 axis=cyl_axis, closed_ends=True) for du in (-0.030, 0.030) for dw in (-0.034, 0.034)]))
    # spark plug boot on the head's inboard side
    plug = cyl_o + cyl_axis * 0.128 - u_ax * 0.077
    # exhaust port on the same face, below and behind the plug (the header came out of the plug boot)
    port = cyl_o + cyl_axis * 0.122 - u_ax * 0.077 - w_ax * 0.042
    boot_end = plug - u_ax * 0.05 + cyl_axis * 0.04
    sc.add("engine_plug_boot", "black_plastic", ml.tube([plug, plug - u_ax * 0.03 + cyl_axis * 0.012, boot_end],
                                                        0.010, 10, 0.02))
    # red fan housing, outlined from Honda's recoil-side drawing (honda-gx390-dimensions.png, scaled to the
    # crank; a little inside it where the v2's right pod wraps it): ~14 cm ahead of the crank, ~17 cm over it
    # with a flat top where the (here absent) tank sits,
    # over the cylinder's outer side behind, and down round the flywheel to the base; edges rolled. (It was a
    # 12.4 cm circle with a tail up to the head.)
    rel = [(0.095, -0.104), (0.128, -0.080), (0.142, -0.025), (0.142, 0.050), (0.126, 0.114), (0.086, 0.157),
           (0.020, 0.173), (-0.060, 0.172), (-0.118, 0.158), (-0.152, 0.115), (-0.162, 0.040), (-0.152, -0.040),
           (-0.118, -0.088), (-0.060, -0.106), (0.020, -0.110)]
    loop = ml.catmull(np.array([(0.0, y_, z_) for y_, z_ in rel]), 4, closed=True)
    shroud_outline = np.c_[crank[0] + loop[:, 1], crank[1] + loop[:, 2]]
    shroud = prism_x(shroud_outline, [ex + 0.060, ex + 0.068, ex + 0.112, ex + 0.127, xo],
                     [0.95, 1.0, 1.0, 0.95, 0.82], crank)              # a broad roll round the outer face
    shroud.modifiers = [("subsurf", 1)]
    sc.add("engine_shroud", "engine_red", shroud)
    # the lead from the plug boot to the coil under the fan housing (the boot ended in the air)
    sc.add("engine_plug_lead", "black_plastic", ml.tube([boot_end, _reach(sc, ("engine_shroud",), boot_end, 0.004)],
                                                        0.0035, 8))
    # domed recoil starter cover with radial slots (texture), black centre
    rc = (xo, ey, RECOIL_Z)            # on the crank
    rs = RECOIL_R / 0.106
    recoil = ml.lathe([(0.0, 0.034), (0.03 * rs, 0.032), (0.07 * rs, 0.024), (0.098 * rs, 0.012), (RECOIL_R, 0.0)], 36,
                      origin=rc, axis=(1, 0, 0), up=(0, 0, 1))
    recoil.uvs = [tuple((0.5 + (recoil.verts[k][1] - rc[1]) / (0.22 * rs), 0.5 + (recoil.verts[k][2] - rc[2]) / (0.22 * rs))
                        for k in face) for face in recoil.faces]
    sc.add("engine_recoil", "recoil_cover", recoil)
    grille = ml.lathe([(0, 0), (0.028, 0), (0.028, 0.004), (0, 0.004)], 24,
                      origin=(rc[0] + 0.033, rc[1], rc[2]), axis=(1, 0, 0), closed_ends=True)
    sc.add("engine_recoil_grille", "engine_black", grille)
    # recoil starter T-grip on the housing's top front (honda-gx390-recoil-side.jpg, -three-quarter.png): a 10 cm oval
    # grip, fuller in the middle and rounded at the ends, on a tapered neck where the rope comes out of the
    # housing's guide. (It was two thin 8-sided tubes.)
    # on the housing's top behind the crank: over its front the v2's right pod roof comes down to ~1 cm off it
    hz = RECOIL_Z + 0.172
    g_c = np.array([xo - 0.040, ey - 0.075, hz + 0.013])          # grip centre; the grip runs fore and aft
    secs = []
    for t in np.linspace(-0.050, 0.050, 11):
        f = abs(t) / 0.050
        k = (1.0 - 0.18 * f * f) * (math.sqrt(max(0.0, 1 - f ** 8)) if f > 0.9 else 1.0)
        k = max(k, 0.25)
        secs.append([(g_c[0] + 0.0115 * k * math.cos(a), g_c[1] + t, g_c[2] + 0.0155 * k * math.sin(a))
                     for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
    grip = ml.loft(secs, closed=True, cap_start=True, cap_end=True)
    grip.modifiers = [("subsurf", 1)]
    neck = ml.lathe([(0.0, 0.0), (0.011, 0.0), (0.0085, 0.012), (0.007, 0.020), (0.0, 0.020)], 14,
                    origin=(g_c[0], g_c[1] - 0.004, hz - 0.012), axis=(0, 0, 1), closed_ends=True)
    guide = ml.lathe([(0.0, 0.0), (0.016, 0.0), (0.016, 0.008), (0.0, 0.008)], 16,
                     origin=(g_c[0], g_c[1] - 0.004, hz - 0.016), axis=(0, 0, 1), closed_ends=True)
    sc.add("engine_recoil_handle", "black_plastic", ml.merge([grip, neck, guide]))
    # carburettor behind the head, its float bowl underneath
    carb_c = np.array([ex - 0.012, ey - 0.195, base_z + 0.195])
    carb = ml.box(tuple(carb_c), (0.052, 0.045, 0.046))
    carb.modifiers = [("bevel", 0.008, 2)]
    sc.add("carb", "engine_alu", ml.merge([carb, ml.lathe([(0, 0), (0.022, 0), (0.024, 0.026), (0, 0.03)], 16,
                                                         origin=tuple(carb_c - (0, 0, 0.052)), axis=(0, 0, 1),
                                                         closed_ends=True)]))
    # engine top (the v1 onboard frames): a round black pulse fuel pump on the crankcase top, yellow fuel
    # lines from it to the carb and its pulse line to the crankcase, the governor's rod to the carb's
    # throttle lever, and the throttle cable from the pedal
    pump = np.array(GX_PUMP)
    sc.add("fuel_pump", "engine_black", ml.merge([
        ml.lathe([(0, 0), (0.028, 0), (0.030, 0.006), (0.030, 0.014), (0.026, 0.020), (0, 0.020)], 24,
                 origin=tuple(pump - (0, 0, 0.010)), axis=(0, 0, 1), closed_ends=True),
        ml.lathe([(0, 0), (0.008, 0), (0.008, 0.006), (0, 0.006)], 6, origin=tuple(pump + (0, 0, 0.010)),
                 axis=(0, 0, 1), closed_ends=True)]
        + [ml.lathe([(0, 0), (0.004, 0), (0.004, 0.016), (0, 0.016)], 8, origin=tuple(pump + d), axis=tuple(d / np.linalg.norm(d)),
                    closed_ends=True) for d in (np.array([-0.026, 0.0, 0.006]), np.array([0.0, -0.026, 0.006]),
                                                np.array([0.018, 0.018, 0.0]))]))
    sc.add("fuel_lines_engine", "fuel_hose_yellow", ml.merge([
        ml.tube([pump + (0.0, -0.040, 0.006), pump + (0.010, -0.075, 0.030), carb_c + (-0.010, 0.025, 0.035),
                 carb_c + (-0.018, 0.010, 0.012)], 0.004, 8, 0.03),
        ml.tube([pump + (0.031, 0.031, 0.0), pump + (0.055, 0.050, 0.010), (ex + 0.02, ey + 0.03, base_z + 0.19)],
                0.0035, 8, 0.03)]))
    gov = np.array([ex - 0.060, ey + 0.010, base_z + 0.205])          # governor arm on the crankcase top
    lever = carb_c + np.array([-0.032, 0.012, 0.018])                 # the carb's throttle lever
    sc.add("governor_linkage", "steel", ml.merge([
        ml.tube([gov - (0, 0, 0.012), gov + (0, 0, 0.020)], 0.003, 6),
        ml.tube([gov + (0, 0, 0.018), lever], 0.0018, 6),
        ml.box(tuple(lever), (0.006, 0.022, 0.012))]))
    sc.add("throttle_cable", "black_plastic", ml.tube(
        [lever + (-0.004, 0.012, 0.0), lever + (-0.04, 0.06, 0.01), (0.36, 0.02, 0.17), (0.26, 0.20, 0.07),
         (0.20, 0.45, 0.06), (PEDAL_X + 0.04, PEDAL_PIVOT_Y - 0.03, PEDAL_PIVOT_Z + 0.01),
         _reach(sc, ("pedal_frame_throttle",), (PEDAL_X + 0.04, PEDAL_PIVOT_Y - 0.01, PEDAL_PIVOT_Z + 0.01), 0.002)], 0.0028, 6, 0.06))
    def air_box():
        # the air cleaner as the Techspeed rentals carry it (the v1 onboard frames): Honda's box shell standing
        # behind the cylinder head on a neck from the carb, under a bigger lid that overhangs it all round and
        # runs on back like a hood, open underneath; plain black, no decal, and no tank in front of it.
        # (It was Honda's complete box with its knob, before that a squat block.)
        y0, y1, z0, z1 = ey - 0.225, ey - 0.345, base_z + 0.205, base_z + 0.365
        bx0, bx1 = ex - 0.015, ex + 0.095
        cy, cx = (y0 + y1) / 2, (bx0 + bx1) / 2
        shell = ml.box((cx, cy, (z0 + z1) / 2), (bx1 - bx0, y0 - y1, z1 - z0))
        shell.modifiers = [("bevel", 0.010, 2)]
        neck = ml.box(tuple(carb_c + (0.0, -0.030, 0.0)), (0.050, 0.040, 0.042))
        neck.modifiers = [("bevel", 0.006, 2)]
        sc.add("air_cleaner", "black_plastic", ml.merge([shell, neck]))
        # the lid: a box with no bottom (a hood), 2 cm over the shell's sides and 2.5 cm past its back (the
        # v2's rear cover rises behind it), a step
        # round its top
        lx0, lx1, ly0, ly1 = bx0 - 0.018, bx1 + 0.018, y0 + 0.012, y1 - 0.025
        lz0, lz1 = z1 - 0.055, z1 + 0.030
        lid = ml.box(((lx0 + lx1) / 2, (ly0 + ly1) / 2, (lz0 + lz1) / 2), (lx1 - lx0, ly0 - ly1, lz1 - lz0))
        lid.faces = [f for f in lid.faces if not all(lid.verts[i][2] < (lz0 + lz1) / 2 for i in f)]
        if lid.uvs is not None:
            lid.uvs = None
        lid.modifiers = [("solidify", 0.004), ("bevel", 0.006, 2)]
        top = ml.box(((lx0 + lx1) / 2, (ly0 + ly1) / 2 + 0.006, lz1 + 0.006), (lx1 - lx0 - 0.03, ly0 - ly1 - 0.04, 0.012))
        top.modifiers = [("bevel", 0.005, 2)]
        sc.add("air_cleaner_lid", "black_plastic", ml.merge([lid, top]))
        sc.add("air_cleaner_stud", "steel", ml.lathe([(0, 0), (0.0035, 0), (0.0035, 0.022), (0, 0.022)], 8,
                                                     origin=(cx, (ly0 + ly1) / 2 + 0.006, lz1 + 0.010), axis=(0, 0, 1),
                                                     closed_ends=True))
    def air_cone():
        # pod-style cylindrical filter on a rubber neck (engine-air-filter.jpg, rear-3q.jpg): out of the carb at the
        # back of the engine, pointing back and out (level), its end clear ahead of the rear
        # tyre. A polished elbow carries it ~12 cm out from the carb, so its end ring stands nearly flush with the
        # rear tyre's outer face (triangulated from the side, rear and rear 3/4 studio photos: the ring at
        # ~(0.68, -0.44, 0.29) real; it hung just off the carb, 10 cm further in). A chrome clamp on the rubber neck, the
        # wide chrome base on the carb side, green pleats one diameter all along (~10 cm before the 15% scale below; it
        # was drawn as a cone, which looked wrong in every photo), and a flat chrome end ring round a recessed pleated
        # centre.
        # it points mostly straight out (the side shot sees its end ring face-on and round, the rear shot its length),
        # ~15% bigger than first drawn, its ring ~0.70 out, 0.46 m behind the middle, ~0.30 m up (it pointed 45 deg
        # back and read small and low)
        # its axis mostly out to the side, a little back and up (the user: it faces the side; aimed 45 deg back it ran
        # into the rear tyre), the end ring ~0.69 out, 0.40 m behind the middle and 0.29 m up, real, where the side,
        # rear and rear 3/4 shots put it, clear ahead of the tyre's tread
        # (candidate rings drawn over the side, rear and rear 3/4 shots: the end ring ~9 cm across, nearly face on to the
        # side camera, so the axis is mostly out to the side; its centre ~0.70 out, 0.445 m behind the middle, 0.29 m
        # up, real. At 1.15x the filter was ~12 cm across and its bulk hung low)
        f_axis = np.array([0.97, -0.20, 0.10])
        f_axis /= np.linalg.norm(f_axis)
        F_SCALE = 0.92
        ring_c = np.array([0.70, float(unwarp_y(-0.445)), 0.285])         # (0.305 m off the ground, the user: higher)
        f_org = ring_c - f_axis * 0.1075 * F_SCALE
        at = lambda a: tuple(f_org + f_axis * a)
        elbow_in = carb_c + np.array([0.030, -0.004, 0.0])
        sc.add("air_filter_elbow", "aluminium", ml.tube(
            [elbow_in, elbow_in + (0.030, -0.010, 0.004), f_org - f_axis * 0.050, f_org - f_axis * 0.034], 0.019, 16, 0.03))
        n_before = len(sc.parts)
        sc.add("air_filter_coupler", "black_plastic",
               ml.lathe([(0.0, -0.035), (0.020, -0.035), (0.021, 0.004), (0.026, 0.010), (0.0, 0.010)], 20,
                        origin=tuple(f_org), axis=f_axis, closed_ends=True))
        # rings: closed loop profiles (closed_ends would fill their holes with discs)
        clamp = ml.lathe([(0.0205, -0.020), (0.0235, -0.020), (0.0235, -0.008), (0.0205, -0.008), (0.0205, -0.020)], 20,
                         origin=tuple(f_org), axis=f_axis)
        flange = ml.lathe([(0.020, 0.004), (0.0555, 0.004), (0.0555, 0.013), (0.020, 0.013), (0.020, 0.004)], 32,
                          origin=tuple(f_org), axis=f_axis)
        # a cylinder, not a cone (the user, with the studio close-ups): the end ring at the pleats' full diameter
        ring = ml.lathe([(0.026, 0.104), (0.0535, 0.104), (0.0535, 0.111), (0.026, 0.111), (0.026, 0.104)], 32,
                        origin=tuple(f_org), axis=f_axis)
        sc.add("air_filter_cap", "steel", ml.merge([clamp, flange, ring]))
        # pleats: a star section swept along the axis, one diameter from the base to the end ring
        pleats = 40
        star = [(math.cos(2 * math.pi * i / (2 * pleats)) * (1.0 if i % 2 == 0 else 0.88),
                 math.sin(2 * math.pi * i / (2 * pleats)) * (1.0 if i % 2 == 0 else 0.88)) for i in range(2 * pleats)]
        a_s = np.linspace(0.012, 0.105, 6)
        cone = ml.sweep(np.array([at(a) for a in a_s]), star, caps=False, scale=[0.0510 for a in a_s])
        cone.sharp_angle = 20
        # recessed centre inside the end ring: the pleats turned in to a dished middle
        a_i = np.linspace(0.107, 0.080, 4)
        inner = ml.sweep(np.array([at(a) for a in a_i]), star, caps=False,
                         scale=[0.026 - 0.008 * (0.107 - a) / 0.027 for a in a_i])
        inner = ml.orient_outward(inner, at(0.09))
        inner.faces = [tuple(reversed(f)) for f in inner.faces]     # seen from inside the ring
        inner.uvs = [tuple(reversed(u)) for u in inner.uvs]
        dish = ml.lathe([(0.0, 0.080), (0.0185, 0.080)], 20, origin=tuple(f_org), axis=f_axis)
        if np.cross(dish.verts[dish.faces[0][1]] - dish.verts[dish.faces[0][0]],
                    dish.verts[dish.faces[0][2]] - dish.verts[dish.faces[0][0]]) @ f_axis < 0:
            dish.faces = [tuple(reversed(f)) for f in dish.faces]
        sc.add("air_filter", "air_filter", ml.merge([cone, inner, dish]))
        grow = ml.translate(f_org) @ np.diag([F_SCALE, F_SCALE, F_SCALE, 1.0]) @ ml.translate(-f_org)
        for part in sc.parts[n_before:]:
            part.mesh = part.mesh.transformed(grow)
    # both air cleaners are built; the variant's choice is exported, the other stays in the source, hidden
    choice = sc.params.get("air_cleaner", "box")
    for kind, make in (("box", air_box), ("cone", air_cone)):
        sc.hide_new = kind != choice
        make()
    sc.hide_new = False
    # the header out of the port inboard, over the chain cover's top (~0.22 high there) just behind the clutch, down
    # and back inboard of the cover's wall and under the v2 engine cover's edge, under the rear cover's front edge
    # (0.22) and over the axle (it ran back by the carb and down through the chain and its guard)
    header = [port, (0.372, -0.272, 0.262), (0.330, -0.300, 0.272), (0.272, -0.345, 0.262), (0.258, -0.420, 0.218),
              (0.242, -0.500, 0.177)]
    if sc.params.get("exhaust") == "single":
        # v1 (the onboard frames): one exhaust, as the racing kart's, under the rear cover: the header down and back
        # behind the axle to a silencer lying across the kart, its single tailpipe straight on out of its left end,
        # finishing under the cover and opening to the left (it turned back and showed under the cover's rear face)
        sc.add("exhaust_header", "exhaust", ml.tube(header + [(0.200, -0.585, 0.150), (0.150, -0.652, 0.126),
                                                              (0.12, -0.66, 0.12)], 0.016, 10, 0.03))
        sc.add("muffler", "exhaust", ml.lathe(
            [(0, 0), (0.024, 0), (0.040, 0.025), (0.040, 0.195), (0.026, 0.22), (0, 0.22)], 20,
            origin=(0.12, -0.66, 0.12), axis=(-1, 0, 0), closed_ends=True))
        tail = [(-0.095, -0.66, 0.12), (-0.175, -0.66, 0.12)]
        sc.add("exhaust_tailpipe", "exhaust", ml.tube(tail, 0.013, 12, caps=False))     # open: the soot plug shows
        # an open end with a sooty plug 3 cm in (see the v2's tips)
        o = tail[-1]
        sc.add("exhaust_tip_C", "exhaust", ml.lathe([(0.011, 0), (0.013, 0), (0.013, 0.012), (0.011, 0.012), (0.011, 0)], 16,
                                                    origin=(o[0] + 0.012, o[1], o[2]), axis=(-1, 0, 0)))
        plug = ml.lathe([(0.0, 0.0), (0.0115, 0.0)], 16, origin=(o[0] + 0.03, o[1], o[2]), axis=(-1, 0, 0))
        v, f = plug.verts, plug.faces[0]
        if np.cross(v[f[1]] - v[f[0]], v[f[2]] - v[f[0]])[0] > 0:       # face out of the pipe (-x)
            plug.faces = [tuple(reversed(ff)) for ff in plug.faces]
            plug.uvs = [tuple(reversed(u)) for u in plug.uvs]
        sc.add("exhaust_soot_C", "engine_black", plug)
    else:
        # exhaust: header under the cover to the twin tips at the rear centre
        sc.add("exhaust_header", "exhaust", ml.tube(header + [(0.200, -0.580, 0.162 + EXH_DZ * 0.5),
                                                              (0.130, -0.626, 0.157 + EXH_DZ),
                                                              (0.08, -0.630, 0.155 + EXH_DZ)], 0.016, 10, 0.03))
        # silencer inside the rear cover, feeding the twin tips in the rear face
        sc.add("muffler", "exhaust", ml.lathe([(0, 0), (0.045, 0), (0.045, 0.16), (0, 0.16)], 20,
                                              origin=(-0.08, -0.635, 0.15 + EXH_DZ), axis=(1, 0, 0), closed_ends=True))
    # clutch drum and engine sprocket on the inboard side, chain to the axle
    sc.add("clutch", "engine_black", ml.lathe([(0, 0), (0.055, 0), (0.055, 0.04), (0, 0.04)], 24,
                                             origin=(sprocket_x + 0.006, ey, base_z + 0.085), axis=(1, 0, 0), closed_ends=True))
    sc.add("sprocket_engine", "steel", sprocket_mesh(0, 0.035, teeth=18).transformed(
        ml.translate((sprocket_x, ey, base_z + 0.085))))
    _chain(sc, sprocket_x, (ey, base_z + 0.085), 0.035, (Y_R, _axle_z(sc)), 0.090)
    _chain_guard(sc, sprocket_x, (ey, base_z + 0.085), 0.035)
    sc.info["engine"] = dict(center=(ex, ey, 0.2), kind="gx")
