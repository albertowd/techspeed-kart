"""BeamNG.drive converter: source .blend files -> vehicles/techspeed_kart/.

    python tools/export_beamng.py

One vehicle with two chassis parts (racing / rental), engine parts per
car in cars.py, four .pc configs and shared livery slots.
Axes: kart (x right, y fwd, z up) -> BeamNG (X left, -Y fwd, Z up): (-x, -y, z).
"""
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blender_bridge as bb  # noqa: E402
import collada  # noqa: E402
from cars import ACTIVE_CARS, BUILD_VARIANTS, CARS, built_skins  # noqa: E402
from export_ac import CLUTCH_THERMALS, hardpoints  # noqa: E402
from kart_model import DRIVER_EYE_BACK, DRIVER_EYE_X, DRIVER_EYE_Z, COL_BASE, COL_PLATE_Z, FLOOR_Z, PEDAL_TRAVEL, SEAT_OFFSET_X, STEER_ARM_END, TIE_ROD_IN_BACK, TIE_ROD_IN_X, Y_F, Y_R, warp_y  # noqa: E402
from kart_model import PROTECTOR_Z, BAND_SIDE_P, family, VARIANTS, protector_front_y, PROTECTOR_FRONT, PROTECTOR_REAR, PROTECTOR_SIDE, PROTECTOR_SIDE_F, PROTECTOR_SIDE_P  # noqa: E402
from materials import MATERIALS  # noqa: E402
from textures import skins_dir, skins_for as _all_skins  # noqa: E402


def skins_for(variant):
    """The liveries this build carries (cars.built_skins: the whole fleet in a release, the first one locally)."""
    every = _all_skins(variant)
    return {k: every[k] for k in built_skins(every)}

ROOT = os.path.dirname(HERE)
MODEL = "techspeed_kart"
VDIR = "/vehicles/%s/" % MODEL
B = np.diag([-1.0, -1.0, 1.0])
RAYS = 16            # tyre rays per wheel: fewer and the small tyre rolls on its corners (10 rays left
                     # a 6 mm step between contact nodes, which kicked the kart)
HUB_NODE = 0.12      # kg per generated hub node
TYRE_NODE = 0.115    # kg per generated tyre node, as the stock 10" race tyre. No stock tyre goes under 0.10:
                     # at 0.07 the unbraked front wheels spun up on their own and walked off the rims in game
TYRE_PSI = 16
# grip curve per compound, BeamNG style: mu = frictionCoef * (fullLoad + (noLoad - fullLoad) * exp(-slope * N))
# per contact node; the kart's tread nodes carry ~150-400 N. The rental's hard compound peaks ~1.2 there
# (times tyre_grip), between the stock 10" street tyre (~1.05) and the stock 10" race tyre (~1.65). It was
# ~1.5, then 1.2: both hopped at full lock, before the knuckle ears and steering dampers; then ~1.05, which
# pushed wide. It slides at its static friction like the stock tyres do: a sliding drop makes the tyre
# stick-slip at the limit, which hops.
COMPOUND = {
    "rental": dict(mu=1.0, no_load=1.4, full_load=0.8, sliding=None, stribeck=2.0),   # None: = frictionCoef
    "racing": dict(mu=1.12, no_load=1.55, full_load=0.75, sliding=0.92, stribeck=1.6),
}
# force (N) past which a beam bends for good (beamDeform). Hard cornering over a kerb (2 g lateral, 3 g
# vertical) loads the busiest frame beam to ~2.6 kN (the axle beams at the outer rear hub), so the frame
# only bends in a crash. The stock pigeon's frame (a ~3x heavier car) runs 10-40 kN. The knuckles and the
# steering rack never bend: a bent one would leave the kart undrivable over the next kerb.
FRAME_DEFORM = 12000         # steel tubes and the raised truss that stands in for their bending stiffness
AXLE_DEFORM = 20000          # rear hubs on the axle bearings and the frame
BUMPER_DEFORM = 6000         # perimeter protector and bumpers: the first thing to take a knock
# the rental's rear brake in BeamNG: 75 Nm a wheel (cars.py's 150 Nm on the one rear disc) is more than a
# rear tyre holds once braking moves weight forwards (~70 Nm at 0.6 g), and the game's brake input goes from
# nothing to full in a blink, so the solid rear axle locked under a second into every stop. Two thirds of it
# stops at ~0.4 g with the rear tyres a third short of locking; a slide still needs a hard stab on low grip.
BRAKE_SCALE = 0.67
# air drag, as AC's aero.ini has it (Cd 0.85 on 0.55 m2: kart and driver). The kart had no drag surfaces at
# all, only the tyres' own: at the governor the rev limiter cuts in and nothing held the speed, so the 9hp
# ran on to ~90 km/h wherever the track fell away. One plate across the driver's chest and the frame rails
# under the seat carries it (no collision, no lift), its coefficient set from its frontal area.
DRAG_AREA = 0.85 * 0.55      # m2, Cd x frontal area
STEER_STOP = 8.0     # degrees past full lock where the knuckle steering stops engage
STEER_DAMP = 850     # Ns/m, steering damper per front wheel (see chassis_part)
CAM_PITCH = 6        # degrees the driver camera looks down by default: the road ahead with the wheel's top in view
#                      (at 18 from a camera 3 cm lower it looked down at the wheel)
OMEGA_DT = 1.7       # per-node stiffness budget at the 2 kHz step; the stock Bolide chassis runs 1.75-2.1
                     # by the same measure (tools/dev/bng_stability.py)
K_MAX = 8001000      # N/m cap for frame beams
PLASTIC_NODE = 0.4   # kg per bodywork node
KNUCKLE_EAR = 0.06   # m, the stub axle's kingpin ears outboard of the kingpin axis
KNUCKLE_NODE = 0.75  # kg per kingpin ear node
RACK_NODE = 0.4      # kg per steering rack end (column plate, rod end)
KNUCKLE_DAMP = 0.3   # knuckle beams' damping as a share of the frame's (~0.1 of critical)
# kg added to the frame nodes the front wheels hang on (kingpin C-brackets and kingpins, the column's lower
# bracket where the tie rods meet, the side bars the steering dampers pull on, the column and front rails
# the steering-wheel prop and camera frame hang on), taken from the rest of the
# frame: a node's beams can only be as stiff (and damped) as its mass allows at the 2 kHz step, and on
# 0.65 kg nodes the kingpins and tie rods were the soft spot
HEAVY_NODES = {"kt": 1.5, "kb": 1.5, "sc": 1.5, "n1": 1.0, "up": 0.5, "co": 0.5, "c2": 0.5}
ENGINE_MOUNT = ("c4r", "c5r", "n2r", "n1r")   # frame nodes carrying the engine mass
PLASTIC_K = 12000    # N/m bodywork mounts: soft, so the plastics move on bumps while racing, as the real
                     # covers and pods do (at 40 kN/m they stood nearly still, ~100 Hz)
PLASTIC_DAMP = 27    # Ns/m per mount beam, the same share of critical as before (50 at 40 kN/m): rubber
                     # grommets, not bells (at 4 they rang ~1 s at 14-30 Hz and shook the whole kart after a kerb)
MOUNT_K = 100000     # N/m steering-wheel prop and camera-frame beams (were 30 kN/m at damp 3: 15-24 Hz,
MOUNT_DAMP = 40      # ringing 1-2 s, so the wheel and the driver's view kept shaking)
# the driver's upper body: a seated person is a soft, heavily damped mass (resonance ~4-6 Hz, ~0.3-0.5 of
# critical), and in a kart with no suspension it is what calms the chassis after a kerb. With all of the
# driver bolted rigidly to the frame the kart bounced on its barely damped tyres at 12-20 Hz for ~1 s.
DRIVER_BODY = 30.0   # kg of the driver on two torso nodes, taken from the seat and the truss shares
DRIVER_BODY_SEAT = 20.0  # kg of it from the seat nodes: the truss keeps most of its mass, it is what the
                         # rear hubs are braced to (lighter truss nodes, softer beams: 2.3 deg rear camber)
DRIVER_BODY_K = 5000     # N/m per torso mount beam (10 per node, to the seat and the truss): ~4-7 Hz
DRIVER_BODY_DAMP = 300   # Ns/m per mount beam: body modes ~0.3-1 of critical, the kart's bounce ~0.1-0.25
PEDAL_NODES = {"PEDAL_GAS": "pedal_gas", "PEDAL_BRAKE": "pedal_brake"}    # prop meshes, turned by the inputs
PLASTIC_PARTS = {"nose", "front_panel", "sidepod_L", "sidepod_R", "rear_bumper", "rear_cover",
                 "pod_top_L", "pod_top_R", "pod_inner_L", "pod_inner_R",
                 "pod_louvres_L", "pod_louvres_R", "engine_cover_R", "engine_cover_louvres"}


def bng(p):
    return (B @ np.asarray(p, float)).tolist()


def r3(p):
    return [round(float(v), 4) for v in p]


# ----------------------------------------------------------------- meshes
def mesh_sets(variant, nodes, baked):
    """Group baked parts into BeamNG meshes. Returns {mesh_name: (subs, kind, key)}."""
    pre = "tskart_%s_" % variant
    groups = {}
    for b in baked:
        node = b.node
        if node == "BODY":
            base = b.name.rsplit("_", 1)[0] if b.name.rsplit("_", 1)[-1].isdigit() else b.name   # "<part>_<material>"
            key = pre + ("plastic" if b.name in PLASTIC_PARTS or base in PLASTIC_PARTS else "body")
        elif node == "STEER_HR":
            key = pre + "steeringwheel"
        elif node in PEDAL_NODES:
            key = pre + PEDAL_NODES[node]
        elif node.startswith("WHEEL_"):
            w = node[6:]
            key = pre + ("tire_" if b.material == "tyre" else "wheel_") + w
        elif node.startswith("SUSP_"):
            key = pre + "spindle_" + node[5:]
        elif node.startswith("TIEROD_"):             # both halves of a tie rod (TIEROD_xx, TIEROD_xx_OUT)
            key = pre + "tierod_" + node[7:9]
        elif node == "STEER_PLATE":
            key = pre + "steerplate"
        else:
            key = pre + "body"
        groups.setdefault(key, []).append(b)
    out = {}
    for key, parts in groups.items():
        local = key.endswith("steeringwheel")
        pedal = key[len(pre):] in PEDAL_NODES.values()
        subs = {}
        for b in parts:
            if pedal:
                # pedal props hang on the steering wheel's frame with its baseRotation, so they are laid out as
                # the wheel is: in STEER_HR's axes (the PEDAL_ nodes are square to the kart), then the same swap
                r = bb.node_world(nodes, "STEER_HR")[:3, :3]
                r = r / np.linalg.norm(r, axis=0)
                swap = lambda v: np.c_[-v[:, 0], -v[:, 2], -v[:, 1]]
                pos, nrm = swap(b.pos @ r), swap(b.nrm @ r)
            elif local:
                # prop mesh in its own frame, laid out like the game's common steer_01a: centred on the
                # column, +z towards the driver, +y to the bottom of the wheel, +x to the left. STEER_HR has
                # +x to the right, +y down the column and +z to the top (see kart_model: AC's convention)
                swap = lambda v: np.c_[-v[:, 0], -v[:, 2], -v[:, 1]]
                pos, nrm = swap(b.pos), swap(b.nrm)
            else:
                w = bb.node_world(nodes, b.node)
                pos = b.pos @ w[:3, :3].T + w[:3, 3]
                nrm = b.nrm @ w[:3, :3].T
                pos, nrm = pos @ B.T, nrm @ B.T
            mat = mat_name(variant, b.material)
            subs.setdefault(mat, []).append((pos, nrm, b.uv, b.tris))
        merged = []
        for mat, items in subs.items():
            off, P, N, U, T = 0, [], [], [], []
            for p, n, u, t in items:
                P.append(p), N.append(n), U.append(u), T.append(t + off)
                off += len(p)
            merged.append((mat, np.vstack(P), np.vstack(N), np.vstack(U), np.vstack(T)))
        out[key] = merged
    return out


def mat_name(variant, m):
    return "tskart_%s_livery" % variant if m == "body_livery" else "tskart_" + m


def materials_json(variants):
    mats = {}

    def stage(spec, tex):
        st = {"metallicFactor": spec["metal"], "roughnessFactor": spec["rough"]}
        if tex:
            st["baseColorMap"] = VDIR + tex
        else:
            st["baseColorFactor"] = [*spec["color"], 1]
        return st

    for name, spec in MATERIALS.items():
        if name == "body_livery":
            continue
        mats["tskart_" + name] = dict(name="tskart_" + name, mapTo="tskart_" + name, **{"class": "Material"},
                                      Stages=[stage(spec, spec.get("texture")), {}, {}, {}],
                                      dynamicCubemap=True, version=1.5)
    spec = MATERIALS["body_livery"]
    for v in variants:
        base = "tskart_%s_livery" % v
        skins = list(skins_for(v))
        mats[base] = dict(name=base, mapTo=base, **{"class": "Material"},
                          Stages=[stage(spec, "%s/skins/%s/body_livery.png" % (v, skins[0])), {}, {}, {}],
                          dynamicCubemap=True, version=1.5)
        for i, s in enumerate(skins):
            nm = "%s.skin.s%d" % (base, i + 1)
            mats[nm] = dict(name=nm, mapTo=nm, **{"class": "Material"},
                            Stages=[stage(spec, "%s/skins/%s/body_livery.png" % (v, s)), {}, {}, {}],
                            dynamicCubemap=True, version=1.5)
    # the liveries flagged worn come on worn tyres (no red line): a skin variant of the shared tyre material. The
    # slots are shared by the variants (sN), so the flag must agree between them (the fleets: every third number)
    tspec = MATERIALS["tyre"]
    worn = {i + 1 for v in variants for i, sk in enumerate(skins_for(v).values()) if sk.get("worn")}
    for n in sorted(worn):
        nm = "tskart_tyre.skin.s%d" % n
        mats[nm] = dict(name=nm, mapTo=nm, **{"class": "Material"}, Stages=[stage(tspec, "tyre_used.png"), {}, {}, {}],
                        dynamicCubemap=True, version=1.5)
    return mats


# ----------------------------------------------------------------- jbeam
def chassis_part(variant, car, hp, nodes):
    W = hp["wheels"]
    fx, rx = W["RF"]["center"][0], W["RR"]["center"][0]
    yF, yR = Y_F, Y_R       # layout coordinates: add() stretches them to the real wheelbase (kart_model.warp_y)
    rf, rr = W["RF"]["radius"], W["RR"]["radius"]
    kp_x = W["RF"]["center"][0] - hp["kingpin_inboard"]
    protector = family(variant) == "rental"
    band = VARIANTS[variant]["protector"] == "band"            # v1: the band has no raised rear arches
    pods = VARIANTS[variant].get("pods", True)
    pre = "tskart_%s_" % variant

    N = []          # (id, kart pos, groups, weight, extra)
    # the layout below is the frame as drawn in kart_model: 1.04 m wheelbase, stretched to the real 1.12 m
    # by warp_y, and ridden `lift` higher on the axles (rental). Points taken from the wheels or the
    # measured source (world=True) are already where they belong.
    lift = hp["lift"]

    def add(nid, p, groups=("kart_chassis",), w=1.0, world=False, **kw):
        if not world:
            p = (p[0], float(warp_y(p[1])), p[2] + lift)
        N.append((nid, tuple(p), list(groups), w, kw))

    def pair(base, p, **kw):
        x, y, z = p
        add(base + "r", (abs(x), y, z), **kw)
        add(base + "l", (-abs(x), y, z), **kw)

    # ---- rigid frame (steel tubes, floor, seat, engine, protector): weights assigned below
    frame = [("c1", (0.13, 0.80, 0.07)), ("c2", (0.16, yF, 0.055)), ("c3", (0.27, 0.15, 0.05)),
             ("c4", (0.33, -0.15, 0.05)), ("c5", (0.38, -0.45, 0.06)), ("c6", (0.39, yR - 0.08, 0.085))]
    for nid, p in frame:
        pair(nid, p)
    pair("n1", (0.58, 0.25, 0.10))
    pair("n2", (0.61, -0.22, 0.11))
    pair("ra", (0.20, hp["y_rear"], rr), world=True)          # axle bearings, on the axle line
    add("rac", (0.0, hp["y_rear"], rr), world=True)
    pair("kt", (kp_x, yF - 0.012, rf + 0.045))
    pair("kb", (kp_x, yF + 0.012, rf - 0.050))
    # steering rack: the column plate's tie rod ends slide sideways (see the steering section). "sc": the
    # hydros' ends beside the frame rails; "rf"/"rv": the links each rack end hangs on, ahead on the front
    # cross tube and up by the column, both in the rack end's own fore-aft plane so it can only slide across
    pair("sc", (0.155, COL_BASE[1] - TIE_ROD_IN_BACK, COL_PLATE_Z))
    pair("rf", (TIE_ROD_IN_X, 0.80, 0.072))
    pair("rv", (TIE_ROD_IN_X, 0.35, 0.40), groups=("kart_chassis", "steeringrack"), collision=False)
    add("col", (0.0, 0.30, 0.42))
    pair("sb", (0.19, -0.30, 0.34))                   # seat back (kart_model.SEAT_BACK_TOP)
    pair("fb", (0.20, 0.93, 0.07))                    # front bumper tubes
    pair("bb", (0.45, -0.68, 0.10))                   # rear bumper tubes
    add("ref", (0.0, 0.15, 0.05))
    add("up", (0.0, 0.15, 0.32), collision=False)
    # the aluminium floor plate's underside (kart_model: 3 cm under the frame as drawn): the frame nodes sit on
    # the tube centres 2 cm higher, so over a tall kerb the plate sank into it before anything touched and the
    # kart never bottomed. Metal nodes here scrape and spark on the ground like the real plate.
    if protector:
        pair("fpa", (0.14, 0.76, FLOOR_Z))
        pair("fpb", (0.30, 0.38, FLOOR_Z))
        pair("fpc", (0.40, 0.0, FLOOR_Z))
        add("fpm", (0.0, 0.40, FLOOR_Z), groups=("kart_chassis", "fuel"))      # under the tank: carries the fuel
    if protector:
        # between the protector's two stacked tubes (measured layout); higher at the rear arches (v2); on the v1's
        # band, at its middle all round
        pf, pr, ps, psf, psp = PROTECTOR_FRONT, PROTECTOR_REAR, PROTECTOR_SIDE, PROTECTOR_SIDE_F, PROTECTOR_SIDE_P
        if band:                                     # the v1's band pinches in between the wheels (kart_model.BAND_PLAN), on
            psp = BAND_SIDE_P + 0.02                 # the right only ahead of the engine: between the two sides' lines
        zm = 0.105 if band else sum(PROTECTOR_Z) / 2       # between the v2's tubes (kart_model.PROTECTOR_Z)
        for i, p in enumerate(((0.30, protector_front_y(0.30), zm), (0.58, protector_front_y(0.58), zm), (psf, 0.55, zm), (psp, 0.05, zm),
                               (ps, -0.45, zm), ((ps - 0.045, pr + 0.04) if band else (0.60, pr + 0.035)) + (zm if band else 0.18,),
                               (0.30, pr, zm if band else 0.18))):
            pair("p%d" % i, p)
    # raised truss: the frame nodes all lie close to one plane, so beams between them alone leave it free
    # to bend and twist out of that plane. A second layer of invisible nodes ~0.3 m up (not in the
    # flexbody, no collision) turns the frame into a 3D truss that only the plastics can move against.
    truss = []
    for i, p in enumerate(((0.22, 0.85, 0.30), (0.35, 0.40, 0.34), (0.45, -0.10, 0.34), (0.45, -0.60, 0.34))):
        pair("t%d" % i, p, groups=("kart_truss",), collision=False)
        truss += ["t%dr" % i, "t%dl" % i]
    # the engine is bolted to the right-hand rails: its mass goes on those frame nodes, no nodes of its own
    eng_mass = 31.0 if family(variant) == "rental" else 18.0
    frame_ids = [n[0] for n in N]

    # ---- plastic bodywork: light nodes on soft mounts, the only thing that visibly shakes
    plastic = []

    def plas(nid, p):
        add(nid, p, groups=("kart_plastic",), w=PLASTIC_NODE)
        plastic.append(nid)

    plas("pnc", (0.0, 0.96, 0.14))
    for s_, x in (("r", 1), ("l", -1)):
        plas("pnm" + s_, (x * 0.24, 0.92, 0.17))
        plas("pnw" + s_, (x * 0.52, 0.80, 0.25))
        # the pods' rounded fronts (kart_model.POD_R_TOP / POD_L_TOP), the same on both sides (none on the v1)
        if pods:
            plas("psf" + s_, (x * 0.62, 0.26, 0.15 if protector else 0.20))
            plas("psm" + s_, (x * 0.63, 0.05, 0.24 if protector else 0.30))
            plas("psr" + s_, (x * 0.60, -0.28, 0.40 if x > 0 else 0.34))
        plas("prw" + s_, (x * 0.56, -0.62, 0.34))
    plas("ppt", (0.0, 0.50, 0.50))
    plas("prc", (0.0, -0.66 if protector else -0.79, 0.30 if protector else 0.22))

    # ---- steering wheel prop frame (light, soft). Same layout as the game's own cars (pigeon + steer_01a):
    # idX straight back from the wheel centre, idY straight down; baseRotation then tilts the wheel.
    sw = hp["steer"]
    c = sw[:3, 3]
    col_dir = sw[:3, 1] / np.linalg.norm(sw[:3, 1])            # down the column, away from the driver
    steer_tilt = math.degrees(math.acos(max(-1.0, min(1.0, -col_dir[2]))))   # wheel face from horizontal
    for nid, p in (("stw", c), ("stwx", c + (0, -0.12, 0)), ("stwy", c + (0, 0, -0.12))):
        add(nid, tuple(p), groups=("kart_steer",), w=0.5, collision=False, world=True)
    # driver camera aim: the occupant camera looks from cam_back through cam_ref, cam_left to its left.
    # The game gives the driver camera no default pitch, and a kart driver's eyes sit so far above the
    # low wheel that a level view misses it, so the aim frame is tilted down by CAM_PITCH.
    t = math.radians(CAM_PITCH)
    cam_ref = np.array([0.0, 0.0, 0.40])
    for nid, p in (("cam_ref", cam_ref), ("cam_back", cam_ref + 0.3 * np.array([0.0, -math.cos(t), math.sin(t)])),
                   ("cam_left", cam_ref + (-0.3, 0.0, 0.0))):
        add(nid, tuple(p), groups=("kart_cam",), w=0.5, collision=False)

    # ---- knuckles and hubs
    arm, hub_nodes = {}, {}
    for key, side in (("FR", 1), ("FL", -1), ("RR", 1), ("RL", -1)):
        ac_key = {"FR": "RF", "FL": "LF", "RR": "RR", "RL": "LR"}[key]
        cx, cy, cz = W[ac_key]["center"]
        s_ = key[1].lower()
        o, i_ = "w%s_o" % key.lower(), "w%s_i" % key.lower()
        add(o, (cx + side * 0.04, cy, cz), groups=("wheelhub_" + key,), w=1.0, world=True)
        add(i_, (cx - side * 0.04, cy, cz), groups=("wheelhub_" + key,), w=1.0, world=True)
        hub_nodes[key] = (o, i_)
        if key[0] == "F":
            a = "sa" + s_
            add(a, (side * STEER_ARM_END[0], STEER_ARM_END[1], STEER_ARM_END[2]), groups=("knuckle_" + key, "tierod_" + key),
                w=1.0)
            arm[key] = a
            # the stub axle's upper and lower ears round the kingpin bolt: with only the hub and arm nodes the
            # knuckle was a flat triangle at axle height, hinged on two pivots 3 and 6.5 cm off its plane, and
            # a cornering load cambered the wheel ~10 deg on it ("a joint not holding"). Two ears level with
            # the pivots make it a 3D body, as the pigeon's 4-node hub (tools/dev/bng_wheel_load.py). Ears
            # staggered fore and aft measured worse: they turn the kingpin box into diagonals. Toe needs a
            # node off the axle line ahead of it (the caliper mount), against the steering arm behind
            for e, p in (("ku", (kp_x + KNUCKLE_EAR, yF - 0.012, rf + 0.045)),
                         ("kd", (kp_x + KNUCKLE_EAR, yF + 0.012, rf - 0.050)),
                         ("kf", (kp_x + KNUCKLE_EAR, yF + 0.07, rf - lift))):
                # the ears also carry the tie rod's flexbody: BeamNG needs its nodes spread in 3D ("VY node
                # not found" with just the rack end and the arm, and the tie rods were not drawn)
                add(e + s_, (side * p[0], p[1], p[2]), groups=("knuckle_" + key, "tierod_" + key), w=KNUCKLE_NODE)
            for nid, p, g, w, kw in N:
                if nid in ("kt" + s_, "kb" + s_, o, i_):
                    g.append("knuckle_" + key)

    # ---- steering rack: the tie rods' inner rod ends on the column plate
    for key, s_, x in (("FR", "r", TIE_ROD_IN_X), ("FL", "l", -TIE_ROD_IN_X)):
        add("rk" + s_, (x, COL_BASE[1] - TIE_ROD_IN_BACK, COL_PLATE_Z + 0.0075), groups=("steeringrack", "tierod_" + key), w=RACK_NODE,
            collision=False)

    # ---- the driver's torso on a soft, damped mount over the seat (see DRIVER_BODY)
    torso = []
    for s_, x in (("r", 1), ("l", -1)):
        add("drv" + s_, (SEAT_OFFSET_X + x * 0.15, -0.15, 0.53), groups=("kart_driver",), w=DRIVER_BODY / 2,
            collision=False)
        torso.append("drv" + s_)

    # ---- mass: everything not fixed above goes into the frame; the driver sits on the seat nodes
    wheel_mass = 4 * (RAYS * 2 * HUB_NODE + RAYS * 2 * TYRE_NODE)
    cam_mass = 0.5
    fixed_mass = (sum(w for nid, _, _, w, _ in N if nid not in frame_ids and nid not in torso)
                  + wheel_mass + cam_mass)
    driver = 80.0 if family(variant) == "rental" else 75.0
    seat_ids = [n for n in frame_ids if n[:2] in ("c3", "c4", "c5", "sb")]
    heavy = {n: HEAVY_NODES[n[:2]] for n in frame_ids if n[:2] in HEAVY_NODES}
    rest = car["mass"] - fixed_mass - driver - eng_mass - sum(heavy.values())
    # driver: 35% on the seat nodes, 45% on the raised truss (about where the body's mass sits) and 20%
    # over the whole frame, so no structural node is light, less the DRIVER_BODY on the damped torso nodes
    # (DRIVER_BODY_SEAT of it off the seat, the rest off the truss); the engine's mass on the rails it is
    # bolted to
    # the driver sits on the seat, SEAT_OFFSET_X to the left, against the engine on the right: each of the
    # driver's shares leans that way (heavier on the left nodes) so its centre lands under the seat. Spread
    # evenly, the driver sat on the centre line and the kart's centre of gravity was 9 cm right of it, so a
    # right turn had to lift the engine's rear wheel and the kart pushed on.
    xs = {nid: p[0] for nid, p, _, _, _ in N}

    def lean(ids, n):
        sx, sxx = sum(xs[i] for i in ids), sum(xs[i] ** 2 for i in ids)
        return 1.0 + (SEAT_OFFSET_X * len(ids) - sx) / sxx * xs[n]

    W8 = {n: rest / len(frame_ids) + 0.2 * driver / len(frame_ids) * lean(frame_ids, n)
          + ((0.35 * driver - DRIVER_BODY_SEAT) / len(seat_ids) * lean(seat_ids, n) if n in seat_ids else 0)
          + ((0.45 * driver - (DRIVER_BODY - DRIVER_BODY_SEAT)) / len(truss) * lean(truss, n) if n in truss else 0)
          + (eng_mass / len(ENGINE_MOUNT) if n in ENGINE_MOUNT else 0)
          + heavy.get(n, 0)
          for n in frame_ids}
    N = [(nid, p, g, W8.get(nid, w), kw) for nid, p, g, w, kw in N]
    total = sum(w for _, _, _, w, _ in N) + wheel_mass + cam_mass
    cg_x = sum(w * p[0] for _, p, _, w, _ in N) / sum(w for _, _, _, w, _ in N)
    print("  %s centre of gravity %+.1f cm across (+ right, without wheels)" % (variant, cg_x * 100))
    print("  %s mass: target %.0f kg, nodes %.1f kg (frame nodes %.1f-%.1f kg)" %
          (variant, car["mass"], total, min(W8.values()), max(W8.values())))
    mass = {nid: w for nid, _, _, w, _ in N}
    pos = {nid: np.array(p) for nid, p, _, _, _ in N}

    # ---- beam topology
    stiff = set()
    struct = [n for n in frame_ids if n not in truss]
    for a in struct:
        for d, b in sorted((np.linalg.norm(pos[a] - pos[b]), b) for b in struct if b != a)[:8]:
            if d < 0.9:
                stiff.add(tuple(sorted((a, b))))
        for d, b in sorted((np.linalg.norm(pos[a] - pos[b]), b) for b in truss)[:3]:
            stiff.add(tuple(sorted((a, b))))
    for i, a in enumerate(truss):              # the truss layer braced in itself
        for b in truss[i + 1:]:
            stiff.add(tuple(sorted((a, b))))
    # knuckle beams: a tight cluster of stiff beams on light nodes. Damped like the frame the explicit step
    # can't hold it ((omega*dt)^2 + 2*c*dt/m over 4, the kart exploded at spawn), so they get KNUCKLE_DAMP of
    # the frame's damping (the pigeon's hub beams: 4 MN/m, damp 50) and the stock links' 500 Hz low-pass;
    # stable even with the low-pass ignored (tools/dev/bng_step_check.py --unfiltered on the chassis)
    knuckle_beams = set()
    for key, s_ in (("FR", "r"), ("FL", "l")):
        o, i_ = hub_nodes[key]
        body = [o, i_, arm[key], "ku" + s_, "kd" + s_, "kf" + s_]       # the knuckle, braced in itself
        for n1 in body:
            for n2 in ("kt" + s_, "kb" + s_):               # hinged on the kingpin pivots
                knuckle_beams.add((n1, n2))
        for j, n1 in enumerate(body):
            for n2 in body[j + 1:]:
                knuckle_beams.add((n1, n2))
    stiff |= knuckle_beams
    for s_ in ("r", "l"):     # rack ends on their two links each, the rack itself and the tie rods
        for b in ("rv" + s_, "rf" + s_, "sa" + s_):
            stiff.add(tuple(sorted(("rk" + s_, b))))
    stiff.add(("rkl", "rkr"))
    for key, s_ in (("RR", "r"), ("RL", "l")):
        o, i_ = hub_nodes[key]
        # the frame round the rear hubs lies nearly in the axle's plane, so it alone barely holds the hub
        # against camber (~14 deg under a cornering load); the raised truss above stands in for the stiff
        # axle shaft in its bearings
        for n1 in (o, i_):
            for n2 in ("c5" + s_, "c6" + s_, "ra" + s_, "n2" + s_, "rac", "c4" + s_, "t2" + s_, "t3" + s_):
                stiff.add((n1, n2))
        stiff.add((o, i_))
    # ---- steering: a rack, as the stock cars have (pigeon). The two rod ends on the column plate (rkr/rkl)
    # slide across the kart, driven by two crossed hydros, and fixed-length tie rods push the steering arms,
    # so the tie rod meshes can follow them (they stood still while the wheels turned). The stroke gives the
    # front wheels steer_lock / steer_ratio degrees at full lock on average. A positive steering input
    # turns right: the arms (behind the kingpins) and so the rack move to the left.
    # The arms swing through a wide arc and lose leverage towards ~60 deg, where a knuckle can go over centre
    # and never come back (the "broken axle" with the wheel flailing), so each knuckle also gets push-only
    # stops (|SUPPORT, as the stock cars' bump stops) that engage STEER_STOP degrees past full lock.
    lock = math.radians(car["steer_lock"] / car["steer_ratio"])
    grid = np.radians(np.arange(-70.0, 70.05, 0.05))
    kn = {}
    for key, s_ in (("FL", "l"), ("FR", "r")):
        kb, kt = pos["kb" + s_], pos["kt" + s_]
        ax = (kt - kb) / np.linalg.norm(kt - kb)

        def turned(n, ang, kb=kb, ax=ax):
            k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            r = np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k
            return kb + r @ (pos[n] - kb)

        a, rk = arm[key], "rk" + s_
        tie = np.linalg.norm(pos[a] - pos[rk])
        arc = np.array([turned(a, g) for g in grid])

        def angle(u, arc=arc, rk=rk, tie=tie):
            """Knuckle angle with the rack moved u along x: the tie rod length root nearest straight ahead."""
            err = np.linalg.norm(arc - (pos[rk] + (u, 0.0, 0.0)), axis=1) - tie
            roots = [i for i in range(len(grid) - 1) if err[i] * err[i + 1] <= 0]
            if not roots:                    # the rack past the arm's reach: beyond any lock
                return math.copysign(math.pi / 2, -u)
            return grid[min(roots, key=lambda i: abs(grid[i]))]

        kn[key] = (turned, angle)

    def mean_lock(u):
        return np.mean([(abs(kn[k][1](u)) + abs(kn[k][1](-u))) / 2 for k in kn])

    lo, hi = 0.0, 0.10
    for _ in range(40):                      # rack stroke that gives the target lock
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_lock(mid) < lock else (lo, mid)
    stroke = (lo + hi) / 2
    hydro_pairs = []
    for rk, sc_, sign in (("rkr", "scl", -1), ("rkl", "scr", 1)):
        hydro_pairs.append((rk, sc_, sign * round(stroke / np.linalg.norm(pos[rk] - pos[sc_]), 4)))
    print("  %s steering rack: stroke +/-%.1f mm" % (variant, stroke * 1000))
    stops = []
    for key, s_ in (("FL", "l"), ("FR", "r")):
        turned, angle = kn[key]
        a = arm[key]
        th_a, th_b = angle(stroke), angle(-stroke)
        full = {1: max(th_a, th_b), -1: min(th_a, th_b)}
        print("  %s steer %s: knuckle %+.1f / %+.1f deg at full lock" % (variant, key, math.degrees(full[1]), math.degrees(full[-1])))
        for d in (1, -1):
            th = full[d] + d * math.radians(STEER_STOP)
            rng = np.linspace(full[-d] + -d * math.radians(STEER_STOP), th + d * math.radians(10), 80)
            cands = []
            for n in (hub_nodes[key][0], hub_nodes[key][1], a):
                for b in frame_ids:
                    if b[0] == "p":            # frame only: the protector takes the knocks
                        continue
                    l = np.array([np.linalg.norm(turned(n, g) - pos[b]) for g in rng])
                    if l[0] > 0.6 or l.min() < 0.08 or np.any(np.diff(l) >= 0):
                        continue               # must shorten all the way through the travel into the stop
                    lt = np.linalg.norm(turned(n, th) - pos[b])
                    lev = abs(np.linalg.norm(turned(n, th + 0.01) - pos[b]) - lt) / 0.01
                    cands.append((lev, n, b, lt / np.linalg.norm(pos[n] - pos[b])))
            cands.sort(reverse=True)
            used = set()
            for lev, n, b, prec in cands:
                if b in used:
                    continue
                used.add(b)
                stops.append((n, b, round(prec, 4)))
                if len(used) == 2:
                    break
    soft = []
    for pnode in plastic:                      # each plastic node hangs on its 4 nearest frame nodes
        for d, b in sorted((np.linalg.norm(pos[pnode] - pos[b]), b) for b in frame_ids)[:4]:
            soft.append((pnode, b))
    for pa in plastic:                         # and on its 2 nearest plastic neighbours
        for d, b in sorted((np.linalg.norm(pos[pa] - pos[b]), b) for b in plastic if b != pa)[:2]:
            soft.append(tuple(sorted((pa, b))))
    soft = list(dict.fromkeys(soft))
    stw_pairs = [(n, b) for n in ("stw", "stwx", "stwy") for b in ("col", "scr", "scl", "c2r", "c2l", "up")]
    stw_pairs += [(n, b) for n in ("cam_ref", "cam_back", "cam_left") for b in ("c3r", "c3l", "c2r", "c2l", "up", "col")]
    stw_pairs += [("cam_ref", "cam_back"), ("cam_ref", "cam_left"), ("cam_back", "cam_left")]
    stw_pairs += [("stw", "stwx"), ("stw", "stwy"), ("stwx", "stwy")]

    # ---- stiffness: as stiff as each node's mass allows at the 2 kHz step (omega*dt <= OMEGA_DT)
    deg = {}
    for a, b in list(stiff) + [(h[0], h[1]) for h in hydro_pairs] + [(t[0], t[1]) for t in stops]:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    kcap = (OMEGA_DT / 0.0005) ** 2

    def k_of(a, b):
        return min(kcap * mass[a] / deg[a], kcap * mass[b] / deg[b], K_MAX)

    def row(a, b, k, extra=None, damp=1.0):
        mm = min(mass[a] / deg.get(a, 1), mass[b] / deg.get(b, 1))
        # ~0.3 of critical damping, capped so c*dt/m stays under 0.8 for the explicit step (stock cars reach 1.1+)
        opts = {"beamSpring": int(k), "beamDamp": round(damp * min(0.6 * math.sqrt(k * mm), 1600 * mm), 1)}
        if extra:
            opts.update(extra)
        return [a, b, opts]

    def deform(a, b):
        if any(n.startswith("rk") for n in (a, b)):
            return "FLT_MAX"
        if any(n[:2] in ("wr", "ra") for n in (a, b)):
            return AXLE_DEFORM
        if any((n[0] == "p" and n[1:2].isdigit()) or n[:2] in ("fb", "bb") for n in (a, b)):
            return BUMPER_DEFORM
        return FRAME_DEFORM

    beams = [row(a, b, k_of(a, b), {"dampCutoffHz": 500, "beamDeform": "FLT_MAX"}, KNUCKLE_DAMP) if (a, b) in knuckle_beams
             else row(a, b, k_of(a, b), {"beamDeform": deform(a, b)}) for a, b in sorted(stiff)]
    plastic_beams = [[a, b, {"beamSpring": PLASTIC_K, "beamDamp": PLASTIC_DAMP}] for a, b in soft]
    stw_beams = [[a, b, {"beamSpring": MOUNT_K, "beamDamp": MOUNT_DAMP}] for a, b in stw_pairs]
    driver_beams = [[t, b, {"beamSpring": DRIVER_BODY_K, "beamDamp": DRIVER_BODY_DAMP}] for t in torso
                    for b in ("sbr", "sbl", "c3r", "c3l", "c4r", "c4l", "t1r", "t1l", "t2r", "t2l")]
    driver_beams.append([torso[0], torso[1], {"beamSpring": DRIVER_BODY_K, "beamDamp": DRIVER_BODY_DAMP}])
    # steering damper: the knuckle twists about its kingpin on the tie-rod link at ~15 Hz with under 1% damping.
    # A rolling tyre damps that, a sliding one doesn't, so at full lock at speed the front wheels shimmied
    # and kicked. In a real kart the driver's arms and the column take it out. A pure damper from the outer hub
    # node (the longest lever, ~0.2 m) back to the side bar node behind it, close to the hub's path when it
    # steers, gives ~0.45 of critical (tools/dev/bng_steer_mode.py); 850 Ns/m on ~1 kg nodes is c*dt/m ~0.43
    # at the 2 kHz step, low-passed like the stock suspension links.
    damper_beams = [[hub_nodes[k][0], "n1" + k[1].lower(), {"beamSpring": 0, "beamDamp": STEER_DAMP,
                                                             "dampCutoffHz": 500}] for k in ("FR", "FL")]
    stop_beams = [row(a, b, k_of(a, b), {"beamType": "|SUPPORT", "beamPrecompression": p, "beamLongBound": 4})
                  for a, b, p in stops]
    hydro_rows = [row(a, b, k_of(a, b), {"factor": f, "steeringWheelLock": car["steer_lock"], "inRate": 2.5,
                                          "outRate": 2.5}) for a, b, f in hydro_pairs]
    ks = [r[2]["beamSpring"] for r in beams]
    print("  %s frame beams: %d, %.0f-%.0f kN/m" % (variant, len(beams), min(ks) / 1000, max(ks) / 1000))

    def node_rows():
        rows = [["id", "posX", "posY", "posZ"]]
        rows += [{"selfCollision": False}, {"collision": True}, {"frictionCoef": 0.6},
                 {"nodeMaterial": "|NM_METAL"}]
        for nid, p, g, w, kw in N:
            opts = {"group": g, "nodeWeight": round(w, 3)}
            if kw.get("collision") is False:
                opts["collision"] = False
            rows.append([nid, *r3(bng(p)), opts])
        return rows

    hydros = [["id1:", "id2:"],
              {"beamPrecompression": 1.0, "beamType": "|NORMAL", "beamLongBound": 1, "beamShortBound": 1},
              {"beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
              *hydro_rows]

    cmp_ = COMPOUND[family(variant)]

    def wheel_rows(axle):
        rows = []
        r = rf if axle == "F" else rr
        k1 = "RF" if axle == "F" else "RR"
        width = W[k1]["width"]
        front_brake = car["brake_torque"] * car["brake_front_share"] / 2
        rear_brake = car["brake_torque"] * (1 - car["brake_front_share"]) / 2
        bt = front_brake if axle == "F" else rear_brake
        rows += [{"brakeTorque": round(bt * BRAKE_SCALE, 1)}, {"parkingTorque": 0}, {"brakeSpring": 50},
                 {"enableBrakeThermals": False},
                 {"hasTire": True}, {"radius": round(r, 4)}, {"tireWidth": round(width * 0.92, 4)},
                 {"hubRadius": round(W[k1]["rim_radius"], 4)}, {"hubWidth": round(width * 0.85, 4)},
                 {"wheelOffset": 0.0}, {"numRays": RAYS},
                 # rim: rigid next to the tyre (the stock 10" wheel runs 701k-901k on 0.3 kg nodes)
                 {"hubTreadBeamSpring": 181000, "hubTreadBeamDamp": 5}, {"hubPeripheryBeamSpring": 181000, "hubPeripheryBeamDamp": 5},
                 {"hubSideBeamSpring": 201000, "hubSideBeamDamp": 5}, {"hubReinfBeamSpring": 151000, "hubReinfBeamDamp": 5},
                 {"hubNodeWeight": HUB_NODE}, {"hubNodeMaterial": "|NM_METAL"}, {"hubFrictionCoef": 0.5},
                 # carcass laid out like the stock 165/70R10 race tyre (pigeon), scaled to the kart's size:
                 # a pre-tensioned tread ring and periphery hold the shape. A kart tyre's short sidewall
                 # carries load on its own, so it is stiff in compression too (the pigeon's is 255 N/m per
                 # psi and leaves the load to the air): ~100-180 kN/m once squashed past the pre-tension,
                 # 2-3 mm under the kart's weight (tools/dev/bng_tyre_load.py). The front tyre has only 5.5 cm
                 # of sidewall: softer than 60 kN/m (with 30 kN/m L-beams) a sharp kerb punched a tread node
                 # past the rim and the tyre turned inside out in game (tools/dev/bng_wheel_roll.py kerb runs)
                 {"enableTireLbeams": True}, {"enableTireSideReinfBeams": False}, {"enableTreadReinfBeams": True},
                 {"enableTirePeripheryReinfBeams": True}, {"enableTireReinfBeams": False},
                 {"wheelSideBeamSpring": 60000, "wheelSideBeamDamp": 28},
                 {"wheelSideBeamSpringExpansion": 171000, "wheelSideBeamDampExpansion": 23},
                 {"wheelSideTransitionZone": 0.1, "wheelSideBeamPrecompression": 0.99},
                 # damping low-passed like the stock tyre; the unfiltered part (sidewalls) stays light for
                 # the tyre nodes: above ~1.2 c*dt/m the wheel blows up at spawn (tools/dev/bng_wheel_modes.py)
                 {"wheelTreadBeamSpring": 66000, "wheelTreadBeamDamp": 38, "wheelTreadBeamDampCutoffHz": 500,
                  "wheelTreadBeamPrecompression": 0.99},
                 {"wheelTreadReinfBeamSpring": 41000, "wheelTreadReinfBeamDamp": 32, "wheelTreadReinfBeamDampCutoffHz": 500,
                  "wheelTreadReinfBeamPrecompression": 0.99},
                 {"wheelPeripheryBeamSpring": 36000, "wheelPeripheryBeamDamp": 19, "wheelPeripheryBeamDampCutoffHz": 500,
                  "wheelPeripheryBeamPrecompression": 0.99},
                 {"wheelPeripheryReinfBeamSpring": 36000, "wheelPeripheryReinfBeamDamp": 19,
                  "wheelPeripheryReinfBeamDampCutoffHz": 500, "wheelPeripheryReinfBeamPrecompression": 0.99},
                 # L-beams across the sidewall: the stock tyre's lateral support
                 {"wheelReinfBeamSpring": 30000, "wheelReinfBeamDamp": 77, "wheelReinfBeamDampCutoffHz": 500,
                  "wheelReinfBeamPrecompression": 0.99},
                 {"nodeWeight": TYRE_NODE}, {"nodeMaterial": "|NM_RUBBER"}, {"triangleCollision": False},
                 {"pressurePSI": TYRE_PSI}, {"dragCoef": 4},
                 {"frictionCoef": round(cmp_["mu"] * car["tyre_grip"], 3)}, {"slidingFrictionCoef": round(cmp_["sliding"] or cmp_["mu"] * car["tyre_grip"], 3)},
                 {"stribeckExponent": cmp_["stribeck"]}, {"treadCoef": 1.0},
                 {"noLoadCoef": cmp_["no_load"]}, {"loadSensitivitySlope": 0.0006}, {"fullLoadCoef": cmp_["full_load"]},
                 {"softnessCoef": 0.6},
                 {"skinDragCoef": 6}]
        return rows

    pw = [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"]]
    pw += wheel_rows("F")
    pw.append(["FR", "wheelhub_FR", "tire_FR", hub_nodes["FR"][0], hub_nodes["FR"][1], 9999, "c2r", 1, {}])
    pw.append(["FL", "wheelhub_FL", "tire_FL", hub_nodes["FL"][0], hub_nodes["FL"][1], 9999, "c2l", -1, {}])
    pw += wheel_rows("R")
    # nodeArm must sit off the axle line (it is the torque-reaction lever)
    pw.append(["RR", "wheelhub_RR", "tire_RR", hub_nodes["RR"][0], hub_nodes["RR"][1], 9999, "c5r", 1, {}])
    pw.append(["RL", "wheelhub_RL", "tire_RL", hub_nodes["RL"][0], hub_nodes["RL"][1], 9999, "c5l", -1, {}])

    flex = [["mesh", "[group]:", "nonFlexMaterials"],
            [pre + "body", ["kart_chassis"]],
            [pre + "plastic", ["kart_plastic"]],
            # the column plate slides with the rack ends; the rack links' upper ends give its frame depth
            [pre + "steerplate", ["steeringrack"]]]
    for key in ("FR", "FL"):
        ac_key = {"FR": "RF", "FL": "LF"}[key]
        flex.append([pre + "spindle_" + ac_key, ["knuckle_" + key]])
        flex.append([pre + "tierod_" + ac_key, ["tierod_" + key, "steeringrack"]])
    for key in ("FR", "FL", "RR", "RL"):
        ac_key = {"FR": "RF", "FL": "LF", "RR": "RR", "RL": "LR"}[key]
        flex.append([pre + "wheel_" + ac_key, ["wheel_" + key, "wheelhub_" + key]])
        flex.append([pre + "tire_" + ac_key, ["wheel_" + key, "tire_" + key]])

    eye = np.array([DRIVER_EYE_X, hp["steer"][1, 3] - DRIVER_EYE_BACK, DRIVER_EYE_Z])   # the driver's eyes
    def drag_plate():
        tris = [("drvl", "drvr", "c4r"), ("drvl", "c4r", "c4l")]
        frontal = sum(abs(np.cross(pos[b] - pos[a], pos[c] - pos[a])[1]) / 2 for a, b, c in tris)
        coef = round(100 * DRAG_AREA / frontal, 1)
        print("  %s drag plate: %.2f m2 frontal, dragCoef %.0f%% (Cd x A %.2f m2)" % (variant, frontal, coef, DRAG_AREA))
        return [["id1:", "id2:", "id3:"], {"triangleType": "NONCOLLIDABLE", "dragCoef": coef, "liftCoef": 0},
                *[list(t) for t in tris], {"dragCoef": 100, "liftCoef": 100, "triangleType": "NORMAL"}]

    part = {
        "information": {"authors": "", "name": "Techspeed %s chassis" % variant.replace("_", " ").title().replace("V1", "v1"),
                        "value": 4000},
        "slotType": "techspeed_chassis",
        "slots2": [["name", "allowTypes", "denyTypes", "default", "description"],
                   ["techspeed_engine_" + family(variant), ["techspeed_engine_" + family(variant)], [],
                    "techspeed_engine_%s" % ("gx390" if family(variant) == "rental" else "125"), "Engine", {"coreSlot": True}],
                   ["paint_design", ["paint_design"], [], "techspeed_livery_1", "Livery"]],
        "refNodes": [["ref:", "back:", "left:", "up:"], ["ref", "rac", "c3l", "up"]],
        # an open kart: the driver's camera hears the outside sound (no onboard muffling: isInside false on the camera,
        # no cabin filter); the engine sound sits on the engine node, so up beside it in the seat it's louder
        "sounds": {"cabinFilterCoef": 0},
        "cameraExternal": {"distance": 3.0, "distanceMin": 1.5, "offset": {"x": 0, "y": 0.0, "z": 0.45}, "fov": 65},
        "camerasInternal": [["type", "x", "y", "z", "fov", "id1:", "id2:", "id3:", "id4:", "id5:", "id6:"],
                            {"nodeWeight": 0.5}, {"selfCollision": False}, {"collision": False},
                            {"beamSpring": 60000, "beamDamp": 50},
                            ["driver", *r3(bng(eye)), 72, "c3r", "c3l", "sbr", "sbl", "c2r", "c2l",
                             {"isInside": False, "beamDeform": 5001000, "beamStrength": "FLT_MAX",
                              "idRef:": "cam_ref", "idBack:": "cam_back", "idLeft:": "cam_left"}]],
        "nodes": node_rows(),
        "beams": [["id1:", "id2:"],
                  {"beamType": "|NORMAL", "beamPrecompression": 1, "beamLongBound": 1, "beamShortBound": 1},
                  {"beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
                  *beams,
                  {"beamDeform": 20000, "beamStrength": "FLT_MAX"},
                  *plastic_beams, *stw_beams, *damper_beams,
                  {"beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"}, *stop_beams, *driver_beams,
                  {"beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"}],
        "hydros": hydros,
        "triangles": drag_plate(),
        "pressureWheels": pw,
        "flexbodies": flex,
        "props": [["func", "mesh", "idRef:", "idX:", "idY:", "baseRotation", "rotation", "translation",
                   "min", "max", "offset", "multiplier"],
                  ["steering", pre + "steeringwheel", "stw", "stwx", "stwy",
                   {"x": -round(steer_tilt, 1), "y": 180, "z": 90},
                   {"x": 0, "y": 0, "z": 1}, {"x": 0, "y": 0, "z": 0}, -1000, 1000, 0, 1],
                  # the pedals: on the wheel's frame, laid out in its axes (mesh_sets), turned about their pivots
                  # (the prop's origin) around the kart's lateral axis, the wheel mesh's x (to the left): a
                  # positive turn about it takes the plate's top forwards. The travel goes in the rotation, the
                  # input (0-1) is clamped to min/max after the multiplier (as the atv's throttle): with the travel
                  # as the multiplier and max 1 the pedals turned 1 degree
                  ["throttle", pre + "pedal_gas", "stw", "stwx", "stwy", {"x": -round(steer_tilt, 1), "y": 180, "z": 90},
                   {"x": PEDAL_TRAVEL["gas"], "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, 0, 1, 0, 1],
                  ["brake", pre + "pedal_brake", "stw", "stwx", "stwy", {"x": -round(steer_tilt, 1), "y": 180, "z": 90},
                   {"x": PEDAL_TRAVEL["brake"], "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 0}, 0, 1, 0, 1]],
        "energyStorage": [["type", "name"], ["fuelTank", "mainTank"]],
        "mainTank": {"energyType": "gasoline", "fuelCapacity": car["fuel_max"], "startingFuelCapacity": car["fuel_l"],
                     "fuel": {"[engineGroup]:": ["fuel"]}},
    }
    return part


def engine_part(car_id, car, variant):
    tq = [["rpm", "torque"]] + [[r, t] for r, t in car["torque"]]
    ratio = car["chain"][1] / car["chain"][0]
    two_stroke = variant == "racing"
    sample = "I3_2" if not two_stroke else "I3_3"
    clutch = {"engageRPMStart": car["clutch_engage"][0], "engageRPMEnd": car["clutch_engage"][1],
              "additionalEngineInertia": 0.005}
    if not two_stroke:
        # a rental clutch only holds a little over the engine's peak torque, so it slips rather than bogs the
        # Honda, and the slip heats a ~1.2 kg drum and shoes cooling in open air (time constant ~40 s).
        # A standing start on the tall chain puts ~27 kJ into it (+45 C); flat out through slow corners it
        # climbs past the warning at 150 C and from 200 C loses grip and slips more, damaged at 320 C.
        # (export_ac.CLUTCH_THERMALS: the AC build's CSP script runs the same model)
        th = CLUTCH_THERMALS
        clutch.update({"lockTorque": round(max(t for _, t in car["torque"]) * car["clutch_lock"], 1),
                       "clutchMass": th["mass"], "clutchSpecificHeat": th["cp"], "coolingCoef": th["cool"] / 20.0,
                       "warningTemp": th["warn"], "maxSafeClutchTemp": th["safe"], "clutchOverheatRange": th["span"],
                       "maxClutchTemp": th["ruin"]})
    return {
        "information": {"authors": "", "name": "%s (%s)" % (car["engine"], car["bhp"]), "value": 1500},
        "slotType": "techspeed_engine_" + family(variant),
        "powertrain": [["type", "name", "inputName", "inputIndex"],
                       ["combustionEngine", "mainEngine", "dummy", 0],
                       # direct drive, as the real kart: no gearbox, so no neutral and no reverse, only the one
                       # chain ratio; the centrifugal clutch on the engine engages as the throttle revs it up and
                       # lets go when the pedal comes back (it fed a CVT "gearbox" with R/N/D modes)
                       ["centrifugalClutch", "clutch", "mainEngine", 1, {"uiName": "Centrifugal Clutch"}],
                       ["differential", "rearAxle", "clutch", 1,
                        {"diffType": "locked", "lockTorque": 50000, "gearRatio": round(ratio, 4), "uiName": "Chain + solid axle",
                         "defaultVirtualInertia": 0.05, "friction": 0.4, "dynamicFriction": 0.0003}],
                       ["shaft", "wheelaxleRR", "rearAxle", 1, {"connectedWheel": "RR", "friction": 0.2}],
                       ["shaft", "wheelaxleRL", "rearAxle", 2, {"connectedWheel": "RL", "friction": 0.2}]],
        "mainEngine": {
            "torque": tq, "idleRPM": car["idle"], "maxRPM": car["limiter"], "revLimiterRPMDrop": 200,
            "inertia": car["engine_inertia"], "friction": 1.2 if not two_stroke else 0.8,
            "dynamicFriction": 0.0004 if not two_stroke else 0.00012,
            "engineBrakeTorque": 2.5 if not two_stroke else 3.0,
            "burnEfficiency": [[0, 0.12], [0.05, 0.2], [0.4, 0.28], [0.7, 0.3], [1, 0.27]],
            "energyStorage": "mainTank", "requiredEnergyType": "gasoline",
            "thermalsEnabled": False,
            "torqueReactionNodes:": ["c4r", "c5r", "c5l"],     # heavy frame rails under the engine
            "uiName": "Engine", "soundConfig": "soundConfig", "soundConfigExhaust": "soundConfigExhaust",
            "starterSample": "event:>Engine>Starter>i4_2011_eng", "starterSampleExhaust": "event:>Engine>Starter>i4_2011_exh",
            "shutOffSampleEngine": "event:>Engine>Shutoff>i4_2011_eng", "shutOffSampleExhaust": "event:>Engine>Shutoff>i4_2011_exh",
            "starterVolume": 0.5, "starterVolumeExhaust": 0.5, "shutOffVolumeEngine": 0.5, "shutOffVolumeExhaust": 0.5,
            "idleRPMStartRate": 2, "idleRPMStartCoef": 1,
        },
        "soundConfig": {"sampleName": sample + "_engine", "intakeMuffling": 0.8, "mainGain": -4,
                        "onLoadGain": 1.0, "offLoadGain": 0.6, "maxLoadMix": 0.7, "minLoadMix": 0,
                        "lowShelfGain": -6, "lowShelfFreq": 160, "highShelfGain": 4, "highShelfFreq": 6000,
                        "eqLowGain": 0, "eqLowFreq": 250, "eqLowWidth": 0.1, "eqHighGain": 2, "eqHighFreq": 900,
                        "eqHighWidth": 0.1, "fundamentalFrequencyCylinderCount": 1, "eqFundamentalGain": -2},
        "soundConfigExhaust": {"sampleName": sample + "_exhaust", "mainGain": -2, "onLoadGain": 1.0, "offLoadGain": 0.3,
                               "maxLoadMix": 0.8, "minLoadMix": 0, "lowShelfGain": -6, "lowShelfFreq": 120,
                               "highShelfGain": 3, "highShelfFreq": 5000, "eqLowGain": -2, "eqLowFreq": 400,
                               "eqLowWidth": 0.1, "eqHighGain": 3, "eqHighFreq": 1500, "eqHighWidth": 0.2,
                               "fundamentalFrequencyCylinderCount": 1, "eqFundamentalGain": -2},
        "clutch": clutch,
        # with no "gearbox" device the vehicle controller runs its pass-through "dummy" shift logic: throttle and
        # brake straight through, no gear to select (game lua: vehicleController.lua, shiftLogic/dummy.lua)
        "vehicleController": {"shiftLogicName": "dummy"},
    }


# ----------------------------------------------------------------- export
def export(out_root):
    vdir = os.path.join(out_root, "vehicles", MODEL)
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
    os.makedirs(vdir)
    jb = {MODEL: {
        "information": {"authors": "", "name": "Techspeed Kart"},
        "slotType": "main",
        "slots2": [["name", "allowTypes", "denyTypes", "default", "description"],
                   ["techspeed_chassis", ["techspeed_chassis"], [], "techspeed_chassis_rental", "Chassis", {"coreSlot": True}]],
        "controller": [["fileName"], ["vehicleController", {}]],
        "scaledragCoef": 1,
        "variables": [["name", "type", "unit", "category", "default", "min", "max", "title", "description"]],
    }}
    variants = []
    for variant in BUILD_VARIANTS:
        car_ids = [c for c in CARS if CARS[c]["variant"] == variant]
        car0 = CARS[car_ids[0]]
        bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
        nodes, baked = bb.collect()
        hp = hardpoints(nodes, baked)
        meshes = mesh_sets(variant, nodes, baked)
        mats = sorted({s[0] for subs in meshes.values() for s in subs})
        # the steering wheel prop is laid out like steer_01a but kept in the vehicle's own DAE, so (as the
        # stock cars do with theirs, e.g. etk800_steer) its node sits where the wheel is: the game draws a
        # vehicle-DAE prop at its node position, carried by the ref frame. At the origin it was drawn under
        # the floor tray. Only the translation counts; the orientation comes from baseRotation.
        place = {}
        for k in meshes:
            if k.endswith("steeringwheel"):
                m4 = np.eye(4)
                m4[:3, 3] = r3(bng(hp["steer"][:3, 3]))
                place[k] = m4
            for node, name in PEDAL_NODES.items():      # the pedals' props sit at their pivots
                if k.endswith(name):
                    m4 = np.eye(4)
                    m4[:3, 3] = r3(bng(bb.node_world(nodes, node)[:3, 3]))
                    place[k] = m4
        collada.write(os.path.join(vdir, "tskart_%s.dae" % variant), [(k, v, place.get(k)) for k, v in meshes.items()],
                      mats)
        jb["techspeed_chassis_" + variant] = chassis_part(variant, car0, hp, nodes)
        for cid in car_ids:
            eng_key = cid.split("_")[-1]
            ename = {"9hp": "gx270", "11hp": "gx340", "13hp": "gx390", "125": "125"}[eng_key]
            jb["techspeed_engine_" + ename] = engine_part(cid, CARS[cid], variant)
        tdir = os.path.join(ROOT, "source", variant, "textures")
        for skin in skins_for(variant):
            shutil.copytree(os.path.join(skins_dir(variant), skin), os.path.join(vdir, variant, "skins", skin),
                            ignore=shutil.ignore_patterns("preview.jpg"))
        for fn in os.listdir(tdir):
            if fn.endswith(".png"):
                shutil.copy(os.path.join(tdir, fn), vdir)
        variants.append(variant)
    # liveries: shared slots, each variant mapping s1..sN to its own artwork (the rental fleets: kart numbers)
    names = {v: list(skins_for(v)) for v in variants}
    for i in range(max(len(n) for n in names.values())):
        nums = sorted({names[v][i].split("_")[0].lstrip("0") for v in variants if i < len(names[v])})
        jb["techspeed_livery_%d" % (i + 1)] = {"information": {"authors": "", "name": "Kart #%s" % "/".join(nums), "value": 100},
                                               "slotType": "paint_design", "skinName": "s%d" % (i + 1)}
    json.dump(jb, open(os.path.join(vdir, MODEL + ".jbeam"), "w"), indent=1)
    json.dump(materials_json(variants), open(os.path.join(vdir, "main.materials.json"), "w"), indent=1)

    for cid in ACTIVE_CARS:
        pc = cid.replace("techspeed_", "")                   # rental_9hp, rental_v1_9hp, ...
        car = CARS[cid]
        eng = {"9hp": "gx270", "11hp": "gx340", "13hp": "gx390", "125": "125"}[cid.split("_")[-1]]
        parts = {"techspeed_chassis": "techspeed_chassis_" + car["variant"],
                 "techspeed_engine_" + family(car["variant"]): "techspeed_engine_" + eng,
                 "paint_design": "techspeed_livery_1"}
        json.dump({"format": 2, "model": MODEL, "parts": parts}, open(os.path.join(vdir, pc + ".pc"), "w"), indent=2)
        peak_kw = max(t * r * 2 * math.pi / 60 for r, t in car["torque"]) / 1000
        json.dump({"Configuration": car["name"].replace("Techspeed ", ""), "Description": "%s, %s, %d kg with driver." %
                   (car["engine"], car["bhp"], car["mass"]), "Config Type": "Factory", "Drivetrain": "RWD",
                   "Transmission": "Direct drive", "Fuel Type": "Gasoline", "Induction Type": "NA", "Propulsion": "ICE",
                   "Power": round(peak_kw * 1.341, 1), "Weight": car["mass"], "Value": 6000 if family(car["variant"]) == "rental" else 9000,
                   "Population": 0}, open(os.path.join(vdir, "info_%s.json" % pc), "w"), indent=2)
    json.dump({"Name": "Techspeed Kart", "Brand": "Techspeed", "Author": "", "Country": "Brazil", "Body Style": "Kart",
               "Type": "Car", "Description": "Techspeed rental kart, built from photos of the real kart.",
               "default_pc": "rental_13hp", "Years": {"min": 2015, "max": 2026},
               "paints": {"Black": {"baseColor": [0.02, 0.02, 0.02, 1.2], "metallic": 0, "roughness": 0.5,
                                    "clearcoat": 0, "clearcoatRoughness": 0}}},
              open(os.path.join(vdir, "info.json"), "w"), indent=2)
    print("BeamNG: wrote", vdir)
    return vdir


if __name__ == "__main__":
    export(os.path.join(ROOT, "build", "beamng"))
