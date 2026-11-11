"""Assetto Corsa converter: source .blend/.fbx -> content/cars/<car_id>/.

    python tools/export_ac.py [car_id ...]

Axes: kart (x right, y fwd, z up) -> AC (X left, Y up, Z fwd): (-x, z, y).
The kn5 origin is on the ground under the CoG; GRAPHICS_OFFSET drops it by
the CoG height.
"""
import io
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blender_bridge as bb  # noqa: E402
import kn5  # noqa: E402
from cars import ACTIVE_CARS, CARS, built_skins  # noqa: E402
from textures import skins_dir  # noqa: E402
import ac_driver  # noqa: E402
from kart_model import ATLAS, family, DRIVER_EYE_BACK, DRIVER_EYE_X, DRIVER_EYE_Z, FLOOR_Z, KINGPIN_AXIS, PEDAL_TRAVEL, PEDAL_X, PEDAL_Y, PEDAL_Z, SEAT_OFFSET_X, warp_y  # noqa: E402
from materials import MATERIALS  # noqa: E402

ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "reference", "GoKart_0.5c")     # the old GoKart mod: its sound banks (sfx())
ONBOARD_ENGINE_VOLUME = 1.4     # CSP multiplier on the cockpit engine (the exterior event played up close)
# The centrifugal clutch's heat, the same numbers as the BeamNG build (export_beamng.engine_part, which BeamNG's
# centrifugalClutch.lua turns into the same model): a ~1.2 kg drum and shoes, cooling to the air (~40 s time constant),
# a warning at 150 C, from 200 C losing grip down to half at 300 C, ruined (a quarter grip) at 320 C.
CLUTCH_THERMALS = dict(mass=1.2, cp=490.0, cool=20.0 * 0.75, warn=150.0, safe=200.0, span=100.0, ruin=320.0)
CLUTCH_SCRIPT = """\
-- Techspeed rental kart (Custom Shaders Patch car physics script): direct chain drive, one gear, no neutral, no
-- reverse, and a centrifugal clutch on the engine that heats up when it slips.
--
-- The clutch's grip rises with engine rpm from %(rpm0)d to %(rpm1)d (the shoes fly out), up to %(lock).1f Nm: the
-- throttle revs the engine into it and it lets go as the engine drops back to idle. Until the kart is fast enough
-- for the engine to reach %(rpm1)d rpm in gear, it slips, and slipping turns torque x slip speed into heat in the drum.
-- A slow corner or a standing start keeps it slipping longer and heats it more; it cools to the air. From %(safe).0f C
-- it loses grip (and slips more), down to half at %(safe).0f + %(span).0f C; at %(ruin).0f C it's ruined until a reset.
-- The same model as the BeamNG build. The temperature is in carPh.controllerInputs[0] (CPHYS_SCRIPT_0) and on the
-- Lua debug app.
local carPh = ac.accessCarPhysics()

local LOCK, RPM0, RPM1 = %(lock).3f, %(rpm0)d, %(rpm1)d
local HEAT_CAP = %(mass).3f * %(cp).1f                      -- J/K
local COOL, SAFE, SPAN, RUIN, WARN = %(cool).2f, %(safe).1f, %(span).1f, %(ruin).1f, %(warn).1f
local tAmb = 25
local temp = tAmb
local ruined = false

local function grip()
  if ruined then return 0.25 end
  return math.min(1, math.max(0.5, 1 - 0.5 * (temp - SAFE) / SPAN))
end

-- engine and gearbox speeds in rad/s; returns the torque back on the engine and the torque into the gearbox
ac.replaceClutch(function (engineVelocity, rootVelocity, engineInertia, rootInertia, clutchAmount, dt)
  local rpm = engineVelocity * (30 / math.pi)
  local x = math.min(1, math.max(0, (rpm - RPM0) / (RPM1 - RPM0)))
  local capacity = LOCK * x * x * (3 - 2 * x) * grip()
  local slip = engineVelocity - rootVelocity
  local torque = capacity * slip / (math.abs(slip) + 4)
  temp = temp + (math.abs(torque * slip) - COOL * (temp - tAmb)) * dt / HEAT_CAP
  if temp >= RUIN then temp, ruined = RUIN, true end
  return -torque, torque
end)

function script.update(dt)
  carPh.gearUp, carPh.gearDown, carPh.requestedGearIndex = false, false, -1         -- no gear to select
  ac.overrideSpecificValue(ac.CarPhysicsValueID.DrivetrainEngagedGear, 2)           -- 0 R, 1 N, 2 first
  ac.overrideSpecificValue(ac.CarPhysicsValueID.DrivetrainOpenThreshold, 0)         -- the clutch above always runs
  tAmb = carPh.ambientTemperature or tAmb
  carPh.controllerInputs[0] = temp
  carPh.controllerInputs[1] = grip()
  if __monitored then
    ac.debug('clutch temperature (C)', temp)
    ac.debug('clutch grip', grip())
    ac.debug('clutch hot', temp >= WARN)
  end
end

function script.reset()
  temp, ruined = tAmb, false
end
"""
C = np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]])


# ----------------------------------------------------------------- geometry info
def hardpoints(nodes, baked):
    """Measure the things physics needs from the (possibly edited) source."""
    hp = {"wheels": {}}
    for key in ("LF", "RF", "LR", "RR"):
        w = bb.node_world(nodes, "WHEEL_" + key)
        c = w[:3, 3]
        tyre = [b for b in baked if b.node == "WHEEL_" + key and b.material == "tyre"]
        p = np.vstack([t.pos for t in tyre])
        radius = float(np.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2).max())
        width = float(p[:, 0].max() - p[:, 0].min())
        rim = [b for b in baked if b.node == "WHEEL_" + key and b.material.startswith("rim")]
        rp = np.vstack([r.pos for r in rim]) if rim else p
        rim_r = float(np.sqrt(rp[:, 1] ** 2 + rp[:, 2] ** 2).max())
        hp["wheels"][key] = dict(center=c, radius=radius, width=width, rim_radius=min(rim_r, radius - 0.03))
    W = hp["wheels"]
    hp["y_front"] = (W["LF"]["center"][1] + W["RF"]["center"][1]) / 2
    hp["y_rear"] = (W["LR"]["center"][1] + W["RR"]["center"][1]) / 2
    hp["wheelbase"] = hp["y_front"] - hp["y_rear"]
    hp["track_f"] = W["RF"]["center"][0] - W["LF"]["center"][0]
    hp["track_r"] = W["RR"]["center"][0] - W["LR"]["center"][0]
    hp["steer"] = bb.node_world(nodes, "STEER_HR")
    kp = bb.node_world(nodes, "SUSP_RF")[:3, 3]
    hp["kingpin_inboard"] = W["RF"]["center"][0] - kp[0]
    allp = np.vstack([b.pos @ bb.node_world(nodes, b.node)[:3, :3].T + bb.node_world(nodes, b.node)[:3, 3] for b in baked])
    hp["bbox"] = (allp.min(0), allp.max(0))
    hp["lift"] = float(bb.node_world(nodes, "BODY")[2, 3])      # frame ride height over the stock layout
    return hp


def tangents(pos, nrm, uv, tris):
    t = np.zeros_like(pos)
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    w0, w1, w2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    for k in range(3):
        np.add.at(t, tris[:, k], ft)
    t -= nrm * (t * nrm).sum(1)[:, None]
    ln = np.linalg.norm(t, axis=1)
    fallback = np.cross(nrm, [0, 1, 0])
    bad = ln < 1e-8
    t[bad] = fallback[bad]
    ln = np.linalg.norm(t, axis=1)
    return t / np.maximum(ln, 1e-9)[:, None]


def split_mesh(pos, nrm, uv, tris, limit=65000):
    """Yield sub-meshes under the kn5 16-bit index limit."""
    if len(pos) <= limit:
        yield pos, nrm, uv, tris
        return
    start = 0
    while start < len(tris):
        used, idx_map, chunk = {}, [], []
        i = start
        while i < len(tris):
            new = [v for v in tris[i] if v not in used]
            if len(used) + len(new) > limit:
                break
            for v in new:
                used[v] = len(used)
            chunk.append([used[v] for v in tris[i]])
            i += 1
        order = np.array(sorted(used, key=used.get))
        yield pos[order], nrm[order], uv[order], np.array(chunk)
        start = i


# ----------------------------------------------------------------- materials / textures
# flat colours: AC shades textures as they are, with no gamma, under a strong ambient light: encoded like
# sRGB (2.2) the black frame (3 %) became a 20 % grey texture and the frame, yokes and column read light grey
# in game. 1.5 keeps light colours nearly as they were and dark ones dark.
AC_TEXTURE_GAMMA = 1.5


def color_png(rgb):
    img = Image.new("RGB", (8, 8), tuple(int(max(0, min(1, c)) ** (1 / AC_TEXTURE_GAMMA) * 255) for c in rgb))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def ac_material(name, tex_name):
    spec = MATERIALS.get(name, dict(color=(0.5, 0.5, 0.5), metal=0.0, rough=0.5))
    metal, rough = spec["metal"], spec["rough"]
    zero2, zero3, zero4 = (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    shiny = 1 - rough
    # bare metal: little ambient/diffuse, the look comes from the reflection (at 0.35/0.45 on a 75-80 % grey
    # texture the steel and aluminium parts came out flat white in game; at 0.18/0.22 with a third of the sky
    # reflected the polished rims still read as white paint)
    props = [("ksAmbient", 0.45 if metal < 0.5 else 0.12), ("ksDiffuse", 0.55 if metal < 0.5 else 0.14),
             ("ksSpecular", 0.1 + 0.6 * shiny), ("ksSpecularEXP", 5 + 180 * shiny ** 2),
             ("ksEmissive", 0.0), ("ksAlphaRef", 0.0)]
    shader = "ksPerPixel"
    if metal > 0.4 or shiny > 0.65:
        shader = "ksPerPixelReflection"
        # reflections: strong only for bare polished metal (the rims ~0.6 of the sky at grazing angles), a mild
        # clear-coat for paint/plastic
        props += [("fresnelC", 0.02 + 0.25 * metal * shiny ** 2 + 0.35 * metal * shiny ** 4), ("fresnelEXP", 4.0),
                  ("fresnelMaxLevel", 0.06 + 0.45 * metal * shiny ** 2 + 0.45 * metal * shiny ** 4), ("isAdditive", 0.0)]
    return kn5.Material(name, shader, 0, False, 0,
                        [(n, float(v), zero2, zero3, zero4) for n, v in props],
                        [("txDiffuse", 0, tex_name)])


# ----------------------------------------------------------------- kn5 writers
def build_kn5(nodes, baked, y_cog, tex_dir, root_name):
    k = kn5.Kn5()
    textures, mats = {}, {}

    def tex_for(mat):
        spec = MATERIALS.get(mat, {})
        if spec.get("texture"):
            fn = spec["texture"]
            if fn not in textures:
                textures[fn] = open(os.path.join(tex_dir, fn), "rb").read()
            return fn
        fn = "c_%s.png" % mat
        if fn not in textures:
            textures[fn] = color_png(spec.get("color", (0.5, 0.5, 0.5)))
        return fn

    root = kn5.Node(root_name, 1, matrix=np.eye(4, dtype=np.float32))
    shift = np.array([0, 0, -y_cog])
    # game nodes, for AC's live suspension (car.ini USE_ANIMATED_SUSPENSIONS=0), as the SDK's pipeline
    # document and bmw_m3_e30 have it: AC moves and steers SUSP_xx, which must sit exactly on WHEEL_xx (axes
    # square), and spins WHEEL_xx; both hang on the root. HUB_xx nodes only exist for suspension animated
    # from a baked ksanim, and AC never moved them here: the spindles and the tie rods' aim points hung on
    # them and stood still while the wheels steered. So the source's front knuckle node SUSP_xx (on the
    # kingpin) is moved to the wheel centre and its meshes re-expressed there. AC turns a node so its +X
    # points at DIR_<node> (as bmw_m3_e30's steerL / DIR_steerL; with -X, as the SDK document reads, the
    # rods swung round to point forwards in game). Each tie rod is two halves (kart_model._steering_linkage):
    # the inner one stays on the chassis with the column plate, the outer one hangs on SUSP_xx by the
    # steering arm and is aimed at the plate's rod end (DIR_ on the root, the bmw layout), its tube sliding
    # over the inner half's threaded rod. (One rigid rod overshot or fell short of the plate as the wheels
    # turned; with the plate and the inner halves hung on STEER_HR, the game threw the plate forward past the
    # nose and the rods followed it.)
    # The rest stays flattened under the root at its world transform.
    world = {}
    for name, (parent, local) in nodes.items():
        if name.startswith("preview_"):
            continue
        w = bb.node_world(nodes, name)
        m = np.eye(4)
        m[:3, :3] = (C @ w[:3, :3] @ C.T).T          # row-vector convention: rows are the local axes
        m[3, :3] = C @ w[:3, 3] + shift
        world[name] = m
    remap = {}                                       # node -> (old world, new world) for its meshes
    for key in ("LF", "RF", "LR", "RR"):
        susp = np.eye(4)
        susp[3] = world["WHEEL_" + key][3]
        if "SUSP_" + key in world:
            remap["SUSP_" + key] = (world["SUSP_" + key], susp)
        world["SUSP_" + key] = susp
    parents = {}
    for name, (parent, local) in nodes.items():
        if name.startswith("DIR_"):
            parents[name] = parent
    for key in ("LF", "RF"):
        tie_in, tie = "TIEROD_" + key, "TIEROD_%s_OUT" % key
        if tie_in not in world or tie not in world:
            continue
        target = world[tie_in][3, :3].copy()
        if world[tie][0, :3] @ (target - world[tie][3, :3]) < 0:     # turn it half round about its z
            turned = world[tie].copy()
            turned[:3, :3] = np.diag([-1.0, -1.0, 1.0]) @ world[tie][:3, :3]
            remap[tie] = (world[tie], turned)
            world[tie] = turned
        aim = "DIR_" + tie
        world[aim] = np.eye(4)
        world[aim][3, :3] = target
        parents[tie] = "SUSP_" + key
    ac_nodes = {}
    for name in world:
        p = parents.get(name)
        local = world[name] @ np.linalg.inv(world[p]) if p else world[name]
        ac_nodes[name] = kn5.Node(name, 1, matrix=local.astype(np.float32))
    for name, n in ac_nodes.items():
        (ac_nodes[parents[name]].children if name in parents else root.children).append(n)
    for b in baked:
        if b.material not in mats:
            mats[b.material] = len(k.materials)
            k.materials.append(ac_material(b.material, tex_for(b.material)))
        pos = b.pos @ C.T
        nrm = b.nrm @ C.T
        uv = np.c_[b.uv[:, 0], -b.uv[:, 1]]
        if b.node in remap:
            w_old, w_new = remap[b.node]
            m = w_old @ np.linalg.inv(w_new)
            pos = pos @ m[:3, :3] + m[3, :3]
            nrm = nrm @ m[:3, :3]
        for i, (p, n_, u, t) in enumerate(split_mesh(pos, nrm, uv, b.tris)):
            tg = tangents(p, n_, u, t)
            mesh = kn5.Node(b.name if i == 0 else "%s_%d" % (b.name, i), 2)
            mesh.verts = np.c_[p, n_, u, tg].astype(np.float32)
            mesh.indices = t.reshape(-1).astype(np.uint16)
            mesh.material_id = mats[b.material]
            mesh.cast_shadows = True
            ac_nodes.get(b.node, ac_nodes["BODY"]).children.append(mesh)
    k.root = root
    k.textures = list(textures.items())
    return k


def build_collider(hp, y_cog, protector):
    lo, hi = hp["bbox"]
    if protector:
        x0, x1, y0, y1 = lo[0], hi[0], lo[1], hi[1]
    else:
        x0, x1 = lo[0] + 0.02, hi[0] - 0.02
        y0, y1 = lo[1] + 0.01, hi[1] - 0.01
    # bottom at the rental's floor plate, so the plate is what grounds over a tall kerb (it stood 2 cm higher,
    # and the plate sank into the kerb before the body touched)
    z0, z1 = (FLOOR_Z if protector else 0.036) + hp["lift"], 0.30 + hp["lift"]
    # octagonal prism in plan view (cut corners) - AC needs a closed mesh
    c = 0.18
    plan = [(x0 + c, y0), (x1 - c, y0), (x1, y0 + c), (x1, y1 - c), (x1 - c, y1), (x0 + c, y1), (x0, y1 - c), (x0, y0 + c)]
    v = [(x, y, z0) for x, y in plan] + [(x, y, z1) for x, y in plan]
    f = []
    n = len(plan)
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i)]
    for i in range(1, n - 1):
        f += [(0, i + 1, i), (n, n + i, n + i + 1)]
    pos = np.array(v) @ C.T + [0, 0, -y_cog]
    tris = np.array(f)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    cen = pos.mean(0)
    if ((pos[tris].mean(1) - cen) * fn).sum(1).mean() < 0:
        tris = tris[:, ::-1]
    nrm = pos - cen
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    k = kn5.Kn5()
    k.materials = [kn5.Material("GL", "GL")]
    # same layout as Kunos colliders: root -> transform "Collider" -> mesh "Collider"
    root = kn5.Node("FBX: collider.FBX", 1, matrix=np.eye(4, dtype=np.float32))
    holder = kn5.Node("Collider", 1, matrix=np.eye(4, dtype=np.float32))
    m = kn5.Node("Collider", 2)
    m.verts = np.c_[pos, nrm, np.zeros((len(pos), 2)), np.zeros((len(pos), 3))].astype(np.float32)
    m.indices = tris.reshape(-1).astype(np.uint16)
    m.cast_shadows = False
    holder.children.append(m)
    root.children.append(holder)
    k.root = root
    return k


# ----------------------------------------------------------------- data files
def ini(sections):
    out = []
    for sec, items in sections:
        out.append("[%s]" % sec)
        for k, v in items:
            out.append("%s=%s" % (k, v))
        out.append("")
    return "\n".join(out)


def fmt(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        return ",".join(fmt(x) for x in v)
    if isinstance(v, float):
        return ("%.5f" % v).rstrip("0").rstrip(".") if v != 0 else "0"
    return str(v)


FRONT_ARM_IN = 0.20          # AC front wishbones' and steering rod's car ends, from the centre line (m)


def data_files(car_id, car, hp):
    W = hp["wheels"]
    rf, rr = W["LF"]["radius"], W["LR"]["radius"]
    lift = hp["lift"]
    cg_h, fw = car["cg_height"] + lift, car["front_weight"]      # driver, engine and frame ride up with it
    wb = hp["wheelbase"]
    y_cog = hp["y_front"] - (1 - fw) * wb
    ratio = car["chain"][1] / car["chain"][0]
    files = {}
    eyes_kart = np.array([DRIVER_EYE_X, hp["steer"][1, 3] - DRIVER_EYE_BACK, DRIVER_EYE_Z])     # world: lift included

    def ac(p):          # kart point -> AC coordinates relative to the CoG
        q = C @ np.asarray(p, float)
        return q + [0, -cg_h, -y_cog]

    def ac_model(p):    # kart point -> the kn5's coordinates (origin on the ground under the CoG): what AC places
        q = C @ np.asarray(p, float)    # the cameras in (from the CoG they sat cg_h too low, under the wheel's top)
        return q + [0, 0, -y_cog]

    eyes = ac_model(eyes_kart)
    files["car.ini"] = ini([
        # CSP extended physics: only then does CSP run the car physics script (data/script.lua: one gear, the
        # centrifugal clutch and its heat). With VERSION=1 the script was never loaded (custom_shaders_patch.log).
        ("HEADER", [("VERSION", "extended-2")]),
        ("INFO", [("SCREEN_NAME", car["name"])]),
        ("BASIC", [("GRAPHICS_OFFSET", fmt((0.0, -cg_h, 0.0))), ("GRAPHICS_PITCH_ROTATION", 0),
                   ("TOTALMASS", car["mass"]),
                   ("INERTIA", fmt((hp["track_r"] + 0.1, 0.55, wb + 0.75)))]),
        ("GRAPHICS", [("DRIVEREYES", fmt(eyes)), ("ONBOARD_EXPOSURE", 33), ("OUTBOARD_EXPOSURE", 31),
                      ("ON_BOARD_PITCH_ANGLE", -15), ("BUMPER_CAMERA_POS", fmt(ac_model((0, hp["y_front"] + 0.45, 0.30)))),
                      ("BONNET_CAMERA_POS", fmt(ac_model((0, hp["y_front"] - 0.1, 0.62)))), ("MIRROR_POSITION", fmt(eyes)),
                      ("SHAKE_MUL", 2.0), ("USE_ANIMATED_SUSPENSIONS", 0), ("FUEL_LIGHT_MIN_LITERS", 1),
                      ("BUMPER_CAMERA_PITCH", 0), ("BONNET_CAMERA_PITCH", 0), ("CHASE_CAMERA_PITCH", 0)]),
        ("CONTROLS", [("FFMULT", car["ffb"]), ("STEER_ASSIST", 0), ("STEER_LOCK", car["steer_lock"]),
                      ("STEER_RATIO", car["steer_ratio"]), ("LINEAR_STEER_ROD_RATIO", 0.0019)]),
        ("FUEL", [("CONSUMPTION", car["fuel_consumption"]), ("FUEL", car["fuel_l"]), ("MAX_FUEL", car["fuel_max"])]),
        ("FUELTANK", [("POSITION", fmt(ac((0, 0.37, 0.22 + lift))))]),
        ("RIDE", [("PICKUP_FRONT_HEIGHT", fmt(-cg_h + 0.03 + lift)), ("PICKUP_REAR_HEIGHT", fmt(-cg_h + 0.03 + lift))]),
        ("PIT_STOP", [("TYRE_CHANGE_TIME_SEC", 10), ("FUEL_LITER_TIME_SEC", 0.5), ("BODY_REPAIR_TIME_SEC", 1),
                      ("ENGINE_REPAIR_TIME_SEC", 1), ("SUSP_REPAIR_TIME_SEC", 1)]),
    ])
    # AC's race engineer plans the AI's fuel and pit stops from this file, not from CONSUMPTION: without it every kart
    # reckoned a tank lasted one lap ("Max laps with max fuel:1" in log.txt), pitted and dropped out. The figure is the
    # burn AC applies (rpm * throttle * CONSUMPTION / 1000 l/s) over a lap averaged on the thirsty side: 90% of the
    # governor, 85% throttle, at 65% of the top speed (~0.0195 km/h per rpm on the 24/65 chain), so the AI never
    # starts short: ~6.6 km/l for the 9hp, ~5.4 for the 13hp (~20 laps of vp_1500 on the 6 l tank).
    rpm, gas = 0.90 * car["limiter"], 0.85
    l_per_h = rpm * gas * car["fuel_consumption"] / 1000 * 3600
    files["fuel_cons.ini"] = ini([("FUEL_EVAL", [("KM_PER_LITER", "%.1f" % (0.65 * car["limiter"] * 0.0195 / l_per_h))])])

    # engine: power.lut holds torque after chain losses
    eff = 0.92
    files["power.lut"] = "\n".join("%d|%.2f" % (r, t * eff) for r, t in car["torque"]) + "\n"
    files["engine.ini"] = ini([
        ("HEADER", [("VERSION", 1), ("POWER_CURVE", "power.lut"), ("COAST_CURVE", "FROM_COAST_REF")]),
        ("ENGINE_DATA", [("ALTITUDE_SENSITIVITY", 0.1), ("INERTIA", car["engine_inertia"]),
                         ("LIMITER", car["limiter"]), ("LIMITER_HZ", 25), ("MINIMUM", car["idle"])]),
        ("COAST_REF", [("RPM", car["limiter"]), ("TORQUE", 3 if family(car["variant"]) == "rental" else 4), ("NON_LINEARITY", 0)]),
        ("DAMAGE", [("TURBO_BOOST_THRESHOLD", 10), ("TURBO_DAMAGE_K", 0),
                    ("RPM_THRESHOLD", car["limiter"] + 300), ("RPM_DAMAGE_K", 1)]),
    ])
    opts = car["chain_options"]
    files["final.rto"] = "\n".join("%d//%d|%.4f" % (a, b, b / a) for a, b in opts) + "\n"
    # Direct drive, as the real kart: one gear, no neutral, no reverse, and a centrifugal clutch with a temperature.
    # AC itself always has N and R (GEAR_R stays defined for it) and no clutch heat, so a Custom Shaders Patch car
    # physics script (data/script.lua) holds the drivetrain in first, swallows the shift inputs and replaces AC's
    # clutch with the same centrifugal clutch and thermal model as the BeamNG build (CLUTCH_SCRIPT). Without CSP the
    # script doesn't run: the forced autoclutch below engages between clutch_engage's rpm, no heat, N and R as before.
    lock = max(t for _, t in car["torque"]) * car["clutch_lock"]
    files["script.lua"] = CLUTCH_SCRIPT % dict(lock=lock, rpm0=car["clutch_engage"][0], rpm1=car["clutch_engage"][1],
                                               **CLUTCH_THERMALS)
    files["drivetrain.ini"] = ini([
        ("HEADER", [("VERSION", 3)]),
        ("TRACTION", [("TYPE", "RWD")]),
        ("GEARS", [("COUNT", 1), ("GEAR_R", fmt(-1.0)), ("GEAR_1", 1.0), ("FINAL", fmt(ratio))]),
        ("DIFFERENTIAL", [("POWER", 1.0), ("COAST", 1.0), ("PRELOAD", 2000)]),
        ("GEARBOX", [("CHANGE_UP_TIME", 100), ("CHANGE_DN_TIME", 100), ("AUTO_CUTOFF_TIME", 0),
                     ("SUPPORTS_SHIFTER", 0), ("VALID_SHIFT_RPM_WINDOW", 20000), ("CONTROLS_WINDOW_GAIN", 0.4),
                     ("INERTIA", 0.002)]),
        ("CLUTCH", [("MAX_TORQUE", int(max(t for _, t in car["torque"]) * car["clutch_lock"]))]),
        ("AUTOCLUTCH", [("UPSHIFT_PROFILE", "NONE"), ("DOWNSHIFT_PROFILE", "NONE"), ("USE_ON_CHANGES", 0),
                        ("MIN_RPM", car["clutch_engage"][0]), ("MAX_RPM", car["clutch_engage"][1]), ("FORCED_ON", 1)]),
        ("AUTOBLIP", [("ELECTRONIC", 0), ("POINT_0", 10), ("POINT_1", 150), ("POINT_2", 180), ("LEVEL", 0.0)]),
        ("DOWNSHIFT_PROTECTION", [("ACTIVE", 0)]),
    ])

    # suspension: rigid kart chassis emulated with stiff DWB + anti-roll bars (flex)
    kpi = hp["kingpin_inboard"]
    susp = [("HEADER", [("VERSION", 1)]),
            ("BASIC", [("WHEELBASE", fmt(wb)), ("CG_LOCATION", fmt(fw))]),
            ("ARB", [("FRONT", 40000), ("REAR", 60000 if car["variant"] == "racing" else 70000)])]
    # hub masses: on the ~1.2 MN/m "rigid" springs a 4 kg front hub rang at ~87 Hz, near what AC's 333 Hz step
    # holds, and the front wheels shook in game with the engine idling (the stock karts: 9 / 16 kg)
    for axle, r, track, hub in (("FRONT", rf, hp["track_f"], 9.0), ("REAR", rr, hp["track_r"], 9.0)):
        if axle == "FRONT":
            # the steering axis through the hub points on the modelled kingpin (kart_model.KINGPIN_AXIS: upright
            # across the kart, top leaning back). It had 11 deg of inclination the model hasn't, and in game the
            # spindle rose and dropped out of its yoke as the wheel steered
            ka = np.asarray(KINGPIN_AXIS, float)
            back = 0.100 * -ka[1] / ka[2]
            top, bot = (kpi, 0.100, -back), (kpi, -0.100, back)
            # AC's rack pushes the same way whatever the layout, so a tie rod behind the axle (as the real
            # kart's) steers the car the wrong way round; like the stock karts (gokart270: z +0.08) the
            # physics rod sits ahead of the axle, its hub end outboard of the kingpin for Ackermann
            # WBTYRE x is measured inboard from the hub (the SDK's formula_k: 0.12 inside a 0.76 m half
            # track, its wishbones' car ends at 0.50). The wishbones and the rod: ~22 cm long and level, car
            # ends at ARM_IN from the centre line, so the hub can only turn about the kingpin. With the car
            # ends at 0.45 (just outboard of the 0.425 kingpin) the "wishbones" were 2.5 cm long and pointed
            # the wrong way and the steering rod was 5 mm long: in game the spindle rode up and down its
            # yoke as it steered, one way on each lock.
            car_top, car_bot = (FRONT_ARM_IN, top[1]), (FRONT_ARM_IN, bot[1])
            steer_car, steer_tyre = (FRONT_ARM_IN, 0.0, 0.08), (kpi - 0.02, 0.0, 0.085)
            toe = 0.0
        else:
            top, bot = (0.10, 0.10, 0.0), (0.10, -0.06, 0.0)
            steer_car, steer_tyre = (0.60, -0.05, -0.06), (0.14, -0.05, -0.07)
            car_top, car_bot = (0.45, 0.08), (0.45, -0.10)
            toe = 0.0
        susp.append((axle, [
            ("TYPE", "DWB"), ("BASEY", fmt(r - cg_h)), ("TRACK", fmt(track)), ("ROD_LENGTH", 0.0),
            ("HUB_MASS", hub),
            ("WBCAR_TOP_FRONT", fmt((*car_top, 0.25))), ("WBCAR_TOP_REAR", fmt((*car_top, -0.25))),
            ("WBCAR_BOTTOM_FRONT", fmt((*car_bot, 0.25))), ("WBCAR_BOTTOM_REAR", fmt((*car_bot, -0.25))),
            ("WBTYRE_TOP", fmt(top)), ("WBTYRE_BOTTOM", fmt(bot)),
            ("WBCAR_STEER", fmt(steer_car)), ("WBTYRE_STEER", fmt(steer_tyre)),
            ("TOE_OUT", toe), ("STATIC_CAMBER", -0.3 if axle == "FRONT" else 0.0),
            ("SPRING_RATE", 1200000 if axle == "FRONT" else 1250000), ("PROGRESSIVE_SPRING_RATE", 0),
            ("BUMP_STOP_RATE", 60000), ("BUMPSTOP_UP", 0.01), ("BUMPSTOP_DN", 0.01), ("PACKER_RANGE", 0.12),
            ("DAMP_BUMP", 450), ("DAMP_FAST_BUMP", 450), ("DAMP_FAST_BUMPTHRESHOLD", 0.5),
            ("DAMP_REBOUND", 500), ("DAMP_FAST_REBOUND", 500), ("DAMP_FAST_REBOUNDTHRESHOLD", 0.5)]))
    susp.append(("GRAPHICS_OFFSETS", [(k, 0) for k in ("WHEEL_LF", "SUSP_LF", "WHEEL_RF", "SUSP_RF",
                                                        "WHEEL_LR", "SUSP_LR", "WHEEL_RR", "SUSP_RR")]))
    susp.append(("DAMAGE", [("MIN_VELOCITY", 25), ("GAIN", 0.0001), ("MAX_DAMAGE", 0.03), ("DEBUG_LOG", 0)]))
    files["suspensions.ini"] = ini(susp)

    # tyres
    g = car["tyre_grip"]
    load_f = car["mass"] * 9.81 * fw / 2
    load_r = car["mass"] * 9.81 * (1 - fw) / 2
    compound = "Rental Hard" if family(car["variant"]) == "rental" else "Kart Medium"
    short = "H" if family(car["variant"]) == "rental" else "M"

    def tyre(key, axle, load):
        w = W[key]
        return [("NAME", compound), ("SHORT_NAME", short), ("WIDTH", fmt(w["width"] * 0.85)),
                ("RADIUS", fmt(w["radius"])), ("RIM_RADIUS", fmt(w["rim_radius"])),
                ("ANGULAR_INERTIA", 0.10 if axle == "f" else 0.16), ("DAMP", 600), ("RATE", 70000 if axle == "f" else 85000),
                ("DY0", fmt(1.62 * g)), ("DY1", -0.045), ("DX0", fmt(1.60 * g)), ("DX1", -0.025),
                ("WEAR_CURVE", "wear.lut"), ("SPEED_SENSITIVITY", 0.0045), ("RELAXATION_LENGTH", 0.035),
                ("ROLLING_RESISTANCE_0", 8), ("ROLLING_RESISTANCE_1", 0.0009), ("ROLLING_RESISTANCE_SLIP", 500),
                ("FLEX", 0.0006), ("CAMBER_GAIN", 0.04), ("DCAMBER_0", 1.2), ("DCAMBER_1", -12),
                ("FRICTION_LIMIT_ANGLE", 7.5), ("XMU", 0.28), ("PRESSURE_STATIC", 12), ("PRESSURE_SPRING_GAIN", 1500),
                ("PRESSURE_FLEX_GAIN", 1.3), ("PRESSURE_RR_GAIN", 1.1), ("PRESSURE_D_GAIN", 0.006),
                ("PRESSURE_IDEAL", 14 if car["variant"] == "racing" else 15), ("FZ0", int(load)),
                ("LS_EXPX", 0.88), ("LS_EXPY", 0.88), ("DX_REF", 1.62), ("DY_REF", 1.62), ("FLEX_GAIN", 0.19),
                ("FALLOFF_LEVEL", 0.90), ("FALLOFF_SPEED", 3), ("CX_MULT", 1.0), ("RADIUS_ANGULAR_K", 0.01),
                ("BRAKE_DX_MOD", 0.03)]

    def thermal():
        return [("SURFACE_TRANSFER", 0.02), ("PATCH_TRANSFER", 0.002), ("CORE_TRANSFER", 0.003),
                ("INTERNAL_CORE_TRANSFER", 0.007), ("FRICTION_K", 0.6), ("ROLLING_K", 1.2),
                ("PERFORMANCE_CURVE", "tcurve.lut"), ("GRAIN_GAMMA", 2), ("GRAIN_GAIN", 0), ("BLISTER_GAMMA", 1.5),
                ("BLISTER_GAIN", 0), ("COOL_FACTOR", 1.6), ("SURFACE_ROLLING_K", 8)]

    files["tyres.ini"] = ini([("HEADER", [("VERSION", 10)]), ("COMPOUND_DEFAULT", [("INDEX", 0)]),
                              ("FRONT", tyre("LF", "f", load_f)), ("REAR", tyre("LR", "r", load_r)),
                              ("THERMAL_FRONT", thermal()), ("THERMAL_REAR", thermal())])
    files["wear.lut"] = "0|100\n100|100\n300|99\n600|98\n1000|96\n"
    files["tcurve.lut"] = "0|0.88\n20|0.95\n40|0.99\n55|1.0\n75|1.0\n95|0.95\n120|0.85\n"

    files["brakes.ini"] = ini([("HEADER", [("VERSION", 1)]),
                               ("DATA", [("MAX_TORQUE", car["brake_torque"]), ("FRONT_SHARE", car["brake_front_share"]),
                                         ("HANDBRAKE_TORQUE", 0), ("COCKPIT_ADJUSTABLE", 0), ("ADJUST_STEP", 0)])])
    files["aero.ini"] = ini([("HEADER", [("VERSION", 2)]),
                             ("WING_0", [("NAME", "BODY"), ("CHORD", 1.0), ("SPAN", 0.55), ("POSITION", "0,0.2,0"),
                                         ("LUT_AOA_CL", "aero_cl.lut"), ("LUT_GH_CL", ""), ("CL_GAIN", 0),
                                         ("LUT_AOA_CD", "aero_cd.lut"), ("LUT_GH_CD", ""), ("CD_GAIN", 1), ("ANGLE", 0),
                                         ("ZONE_FRONT_CL", 0), ("ZONE_FRONT_CD", 0), ("ZONE_REAR_CL", 0),
                                         ("ZONE_REAR_CD", 0), ("ZONE_LEFT_CL", 0), ("ZONE_LEFT_CD", 0),
                                         ("ZONE_RIGHT_CL", 0), ("ZONE_RIGHT_CD", 0)])])
    files["aero_cl.lut"] = "-180|0\n0|0\n180|0\n"
    files["aero_cd.lut"] = "-180|0.85\n0|0.85\n180|0.85\n"

    files["setup.ini"] = ini([
        ("DISPLAY_METHOD", [("SHOW_CLICKS", 0)]),
        ("FINAL_GEAR_RATIO", [("RATIOS", "final.rto"), ("NAME", "Sprockets"), ("POS_X", 0.5), ("POS_Y", 0), ("HELP", "HELP_REAR_GEAR")]),
        ("FUEL", [("SHOW_CLICKS", 0), ("TAB", "GENERIC"), ("NAME", "Fuel"), ("MIN", 1), ("MAX", car["fuel_max"]), ("STEP", 1),
                  ("POS_X", 0.5), ("POS_Y", 1), ("HELP", "HELP_FUEL")]),
    ] + [("PRESSURE_" + w, [("SHOW_CLICKS", 0), ("TAB", "TYRES"), ("NAME", "Pressure " + w), ("MIN", 8), ("MAX", 22),
                            ("STEP", 1), ("POS_X", i % 2), ("POS_Y", 2 + 2 * (i // 2)), ("HELP", "HELP_%s_PRESSURE" % w)])
         for i, w in enumerate(("LF", "RF", "LR", "RR"))]
      + ([("FRONT_BIAS", [("SHOW_CLICKS", 0), ("TAB", "GENERIC"), ("NAME", "Brake Bias"), ("MIN", 20), ("MAX", 45),
                          ("STEP", 1), ("POS_X", 0.5), ("POS_Y", 2), ("HELP", "HELP_BRAKE_BIAS")])]
         if car["brake_front_share"] > 0 else []))

    files["electronics.ini"] = ini([("ABS", [("SLIP_RATIO_LIMIT", 0.1), ("CURVE", ""), ("PRESENT", 0), ("ACTIVE", 0), ("RATE_HZ", 100)]),
                                    ("TRACTION_CONTROL", [("SLIP_RATIO_LIMIT", 0.1), ("CURVE", ""), ("PRESENT", 0), ("ACTIVE", 0),
                                                          ("RATE_HZ", 100), ("MIN_SPEED_KMH", 20)])])
    files["ai.ini"] = ini([("HEADER", [("VERSION", 3)]), ("GEARS", [("UP", car["limiter"] + 1000), ("DOWN", 0)]),
                           ("LOOKAHEAD", [("BASE", 12), ("SPEED_GAIN", 0.9), ("GAS_BRAKE_LOOKAHEAD", 2)]),
                           ("STEER", [("STEER_GAIN", 1.2)]),
                           ("PEDALS", [("GASGAIN", 2.0), ("BRAKE_HINT", 0.95), ("TRAIL_HINT", 0.6)])])
    files["lods.ini"] = ini([("LOD_0", [("FILE", car_id + ".kn5"), ("IN", 0), ("OUT", 1500)])])
    files["ambient_shadows.ini"] = ini([("SETTINGS", [("WIDTH", fmt(float(hp["bbox"][1][0] - hp["bbox"][0][0]) / 2 + 0.05)),
                                                      ("LENGTH", fmt(float(hp["bbox"][1][1] - hp["bbox"][0][1]) / 2 + 0.1))])])
    files["blurred_objects.ini"] = ""
    files["lights.ini"] = ""
    files["mirrors.ini"] = ""
    files["dash_cam.ini"] = ini([("DASH_CAM", [("POS", fmt(eyes + [0, 0.05, 0.05])), ("EXP", 27)])])
    files["escmode.ini"] = ini([("SETTINGS", [("POSITION", "0,1.2,3.0"), ("FOV", 45)])])
    files["sounds.ini"] = ini([("SKIDS", [("ENTRY_POINT", 0.35), ("MIX_VOLUME", 0.6), ("PITCH_BASE", 0.6), ("PITCH_GAIN", 0.15)]),
                               ("WIND", [("SPEED_GAIN_0", 0), ("SPEED_GAIN_1", 0.0002), ("VOLUME_GAIN", 0.7),
                                         ("PITCH_REFERENCE", 50), ("PITCH_GAIN", 0.3)]),
                               ("TYRE_ROLLING", [("SPEED_GAIN_0", 0.0015), ("SPEED_GAIN_1", 0.0001), ("VOLUME_GAIN", 0.8),
                                                 ("PITCH_REFERENCE", 45), ("PITCH_GAIN", 0.2)]),
                               ("BODY_WORK", [("MIN", 0.2), ("GAIN", 1)]), ("GEAR_EXT", [("DISTANCE_SCALE", 5)]),
                               ("ENGINE", [("POSITION", "rear")])])
    sw = hp["steer"][:3, 3]
    # the driver's whole placement is on the posture's root bone (driver_posture); POSITION stays 0: in game AC
    # put the driver at minus POSITION (12 cm up and back off the seat with the body's move in it, behind the
    # seat with the wheel alignment in it)
    files["driver3d.ini"] = ini([("MODEL", [("NAME", "driver_80"), ("POSITION", "0,0,0")]),
                                 ("STEER_ANIMATION", [("NAME", "steer.ksanim"), ("LOCK", car["steer_lock"])]),
                                 ("SHIFT_ANIMATION", [("BLEND_TIME", 200), ("POSITIVE_TIME", 300), ("STATIC_TIME", 10),
                                                      ("NEGATIVE_TIME", 300), ("PRELOAD_RPM", car["limiter"]),
                                                      ("INVERT_SHIFTING_HANDS", 0)])])
    cams = []
    for i, (pos, fwd) in enumerate((((0.9, 0.55, 0.9), (-0.45, -0.25, -0.86)), ((-0.9, 0.55, 0.9), (0.45, -0.25, -0.86)),
                                     ((0.0, 0.45, 1.3), (0.0, -0.15, -0.99)), ((0.6, 0.35, -1.2), (-0.3, -0.1, 0.95)))):
        f = np.array(fwd) / np.linalg.norm(fwd)
        up = np.array([0, 1.0, 0]) - f * f[1]
        up /= np.linalg.norm(up)
        cams.append(("CAMERA_%d" % i, [("POSITION", fmt(pos)), ("FORWARD", fmt(f)), ("UP", fmt(up)), ("FOV", 56), ("EXPOSURE", 35)]))
    files["cameras.ini"] = ini(cams)
    return files, y_cog


# ----------------------------------------------------------------- misc assets
# the driver in the seat (ac_driver.fit, from AC's own driver_80): the hip joints on the seat's centre line
# over the pan, lying in the low, laid-back seat with the head level: the pelvis tipped back DRIVER_RECLINE[0]
# deg from driver_80's rest pose, the chest less (the back curving forward off the pelvis), the neck and head
# back up towards upright, the shoulders rolled forward to the wheel. driver_80 at full size stood a head
# taller than the driver in protector-v2-side-driving.jpg (helmet top ~0.95 m off the ground, eyes ~0.82):
# at 0.92 the helmet tops out at ~0.94 and the eyes are within 3 cm of the cockpit camera (DRIVER_EYE_*).
# The thighs sit ~6 cm into the pan's raised sides, as before, nothing else in the seat; the knees ~5 cm
# clear of the steering column and the fuel tank, the feet on the pedals, the hands round the rim.
DRIVER_HIP = (SEAT_OFFSET_X, 0.065, 0.120)        # hip joints' centre, kart (drawn) axes over the frame, m
DRIVER_SCALE = 0.92
DRIVER_RECLINE = (24.0, 10.0, 15.0, 5.0)          # pelvis, chest, neck, head: deg back from driver_80's rest
# each hand's place round the rim with the wheel straight, either side of the top (deg): just over quarter to
# three, where the rental wheel's grips are (steering-wheel.jpg)
DRIVER_HANDS = 78.0
STEER_FRAMES = 101                                 # frames in the steering animation, full lock to full lock
DRIVER_KNEES_OUT = 1.3                             # knees spread round the steering column and the fuel tank
DRIVER_SHOULDERS_FWD = 30.0                        # collarbones swung forward (deg): shoulders rolled to the wheel
# elbows: down under the grips, a little out, relaxed, as protector-v2-front-driving.jpg's driver: ~14 cm under
# the shoulders, the forearms rising to the hands. (At 1/2 they stood out level with the wrists, 13 cm outside
# the shoulders, the forearms coming down onto the grips in game; at -1/3 the arms went straight out to the
# sides; with no outward pull at all they hung straight down into the body.)
DRIVER_ELBOWS_DOWN = 4.0
DRIVER_ELBOWS_OUT = 0.5
# chest turned into the turn about the spine at full lock (deg; negative: towards the hand going over the
# top): with it the hand going over the top stays on the rim for ~2/3 of full lock (1/3 without), the other
# hand all the way (references/hands.mp4: both hands go round with the wheel)
DRIVER_TWIST = -30.0
FOOT_PITCH = 33.0                                  # degrees the ankle-to-ball bone rises: sole flat on the plate
FOOT_BACK, FOOT_DZ = 0.14, 0.0                     # ankle behind / over the pedal plate's centre (m)




def cog_y(car, hp):
    """Fore-aft place of the CoG (kart y): the kn5 origin."""
    return hp["y_front"] - (1 - car["front_weight"]) * hp["wheelbase"]


def driver_posture(hp, sw_pts, out_knh, out_ksanim, y_cog, steer_lock):
    sw = hp["steer"][:3, 3]
    lift = hp["lift"]
    to_ac = lambda p: C @ (np.asarray(p, float) - sw) + ac_driver.REF_SW      # noqa: E731
    # ours: AC spins STEER_HR about its z, which is the model frame's y (down the column; reference-ac-steer-node)
    ours = ac_driver.Wheel(to_ac(sw), C @ hp["steer"][:3, 1], [to_ac(p) for p in sw_pts])
    py = float(warp_y(PEDAL_Y))
    # the soles flat on the tall, nearly upright pedal plates (kart_model._pedals_rental). driver_80's
    # sole doesn't lie along its ankle-to-ball bone: measured as the shoe's biggest flat face (~12 x 26 cm),
    # it lies on the plate (0-20 mm off, nothing through; the rest is an 8 deg sideways tilt) with that bone
    # at FOOT_PITCH and the ankle FOOT_BACK behind the plate's centre, the heel at the plate's bottom and the
    # toes ~10 cm over its top. (With the bone at 63-77 deg only the heel end of the sole touched the plate
    # and the toes stood off it; on the older, lower plates the feet pivoted on the heels on the floor.)
    ank_y, ank_z = py - FOOT_BACK, PEDAL_Z + FOOT_DZ + lift
    rise = (0.0, math.cos(math.radians(FOOT_PITCH)), math.sin(math.radians(FOOT_PITCH)))
    heels = [(sx * PEDAL_X, ank_y, ank_z) for sx in (-1, 1)]
    toes = [np.add(h, rise) for h in heels]
    hip = (DRIVER_HIP[0], float(warp_y(DRIVER_HIP[1])), DRIVER_HIP[2] + lift)
    return ac_driver.fit(out_knh, out_ksanim, to_ac, hip, heels, toes, ours, scale=DRIVER_SCALE, recline=DRIVER_RECLINE,
                         hand_angle=DRIVER_HANDS, lock=steer_lock, frames=STEER_FRAMES,
                         elbow_down=DRIVER_ELBOWS_DOWN, elbow_out=DRIVER_ELBOWS_OUT,
                         root_move=C @ sw + [0, 0, -y_cog] - ac_driver.REF_SW, twist=DRIVER_TWIST,
                         knees_out=DRIVER_KNEES_OUT, shoulders_fwd=DRIVER_SHOULDERS_FWD)


# Pedal travel at full throttle / brake (deg; the plates' tops go forward). Original AC can't animate pedals
# or the driver's feet (the SDK document: "Animating the pedals is not supported yet"), so a Custom Shaders
# Patch car script does it: it turns the pedal nodes (kart_model._pedals_rental: PEDAL_GAS / PEDAL_BRAKE at the
# pivots) and the driver's ankles by the same angle, so the soles stay on the plates. Without CSP the pedals
# and feet stay still.


def csp_pedals(out, knh_path):
    """extension/ext_config.ini + pedals.lua: pedals and feet follow the throttle and brake (CSP)."""
    names, local, parent, _ = ac_driver.read_knh(knh_path)
    w = ac_driver._worlds(local, parent)
    axes = {}
    for side in ("L", "R"):
        k = names.index(ac_driver.D + "RIG_Hill_" + side)
        rot = np.asarray(w[k], float)[:3, :3]
        a = np.array([1.0, 0.0, 0.0]) @ np.linalg.inv(rot)        # the car's X (across) in the ankle's own axes
        axes[side] = a / np.linalg.norm(a)
    ext = os.path.join(out, "extension")
    os.makedirs(ext, exist_ok=True)
    open(os.path.join(ext, "ext_config.ini"), "w", newline="\r\n").write(
        "[SCRIPT_0]\n; pedals and the driver's feet follow the throttle and brake\nSCRIPT = pedals.lua\n\n"
        "[AUDIO_VOLUME]\n; open kart: the cockpit plays the exterior events (sfx/GUIDs.txt), the engine a little louder\n"
        "; there, the driver sitting right beside it\nENGINE_INT = %.2f\n" % ONBOARD_ENGINE_VOLUME)
    v = lambda a: "vec3(%.5f, %.5f, %.5f)" % tuple(a)                     # noqa: E731
    open(os.path.join(ext, "pedals.lua"), "w", newline="\n").write("""-- Pedals and the driver's feet follow the throttle and brake.
-- Original AC doesn't animate pedals or feet; this Custom Shaders Patch car script turns the pedal nodes (at
-- their pivots, axes square to the car) and the driver's ankles by the same angle, so the soles stay on the
-- plates: the plates' tops and the toes go forward.
-- Each node's resting matrix is kept from the first frame and every frame the node is set to that matrix
-- turned about the car's X (absolute, never relative: the game doesn't reset the leg bones each frame, so
-- turning them a little more every frame spun the feet round).
local pedalGas, pedalBrake = ac.findNodes('?PEDAL_GAS'), ac.findNodes('?PEDAL_BRAKE')
local ankleRight, ankleLeft = ac.findNodes('?RIG_Hill_R'), ac.findNodes('?RIG_Hill_L')
local across = vec3(1, 0, 0)
local acrossRight, acrossLeft = %s, %s   -- the car's X in each ankle's own axes
local gasTravel, brakeTravel = math.rad(%.1f), math.rad(%.1f)
local rest = {}

local function turn(ref, axis, angle)
  if #ref == 0 then return end
  local m = ref:getTransformationRaw()
  if not rest[ref] then rest[ref] = m:clone() end
  m:set(mat4x4.rotation(angle, axis):mul(rest[ref]))
end

function script.update(dt)
  local gas, brake = car.gas * gasTravel, car.brake * brakeTravel
  turn(pedalGas, across, gas)
  turn(pedalBrake, across, brake)
  ac.updateDriverModel()
  turn(ankleRight, acrossRight, gas)
  turn(ankleLeft, acrossLeft, brake)
  ac.debug('pedal nodes (gas, brake)', #pedalGas .. ', ' .. #pedalBrake)
  ac.debug('ankle nodes (right, left)', #ankleRight .. ', ' .. #ankleLeft)
end
""" % (v(axes["R"]), v(axes["L"]), PEDAL_TRAVEL["gas"], PEDAL_TRAVEL["brake"]))


def shadow_png(path, w, h, blur):
    img = Image.new("L", (128, 128), 255)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([64 - w, 64 - h, 64 + w, 64 + h], radius=min(w, h) // 2, fill=40)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    Image.merge("RGBA", (img, img, img, Image.new("L", img.size, 255))).convert("RGB").save(path)


def sfx(car_id, car, out):
    """Re-use the old mod's FMOD bank (4-stroke 390 / 2-stroke 125) under our car id."""
    src = "gokart390" if family(car["variant"]) == "rental" else "gokart125"
    sdir = os.path.join(REF, src, "sfx")
    if not os.path.isdir(sdir):
        return False
    os.makedirs(os.path.join(out, "sfx"), exist_ok=True)
    shutil.copy(os.path.join(sdir, src + ".bank"), os.path.join(out, "sfx", car_id + ".bank"))
    guids = open(os.path.join(sdir, "GUIDs.txt"), encoding="latin-1").read()
    guids = guids.replace("event:/cars/%s/" % src, "event:/cars/%s/" % car_id)
    guids = guids.replace("bank:/%s" % src, "bank:/%s" % car_id)       # (CSP: "bank ... is not registered in GUIDs")
    # The kart is open: the onboard camera hears what the outside cameras hear. Each *_int event path (engine, gear,
    # skid) points at its *_ext event's GUID, so AC plays the exterior sound in the cockpit too; CSP's [AUDIO_VOLUME]
    # (ext_config.ini) lifts the engine there, the driver sitting right beside it.
    lines = guids.splitlines()
    ext = {}
    for line in lines:
        p = line.split()
        if len(p) == 2 and p[1].endswith("_ext"):
            ext[p[1][:-4]] = p[0]
    out_lines = []
    for line in lines:
        p = line.split()
        if len(p) == 2 and p[1].endswith("_int") and p[1][:-4] in ext:
            line = "%s %s" % (ext[p[1][:-4]], p[1])
        out_lines.append(line)
    open(os.path.join(out, "sfx", "GUIDs.txt"), "w", encoding="latin-1").write("\n".join(out_lines) + "\n")
    return True


def export(car_id, out_root):
    car = CARS[car_id]
    src = os.path.join(ROOT, "source", car["source"], car["source"] + ".blend")
    meta = bb.open_source(src)
    nodes, baked = bb.collect()
    hp = hardpoints(nodes, baked)
    out = os.path.join(out_root, car_id)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data"))
    # driver posture + steering animation: AC's own driver seated in this kart, on its pedals and wheel
    os.makedirs(os.path.join(out, "animations"))
    sw_parts = [b for b in baked if b.node == "STEER_HR"]
    sw_w = bb.node_world(nodes, "STEER_HR")
    sw_pts = np.vstack([b.pos @ sw_w[:3, :3].T + sw_w[:3, 3] for b in sw_parts])
    fit = driver_posture(hp, sw_pts, os.path.join(out, "driver_base_pos.knh"),
                         os.path.join(out, "animations", "steer.ksanim"), cog_y(car, hp), car["steer_lock"])
    print("  driver posture: hands short %.1f mm, heels short %s mm, knees %s deg, wrists %s deg, grip tilt %s deg" % (
        fit["hand_short_mm"], fit["heel_short_mm"], fit["knee_deg"], fit["wrist_bend_deg"], fit["tilt_deg"]))
    csp_pedals(out, os.path.join(out, "driver_base_pos.knh"))
    files, y_cog = data_files(car_id, car, hp)
    for name, text in files.items():
        open(os.path.join(out, "data", name), "w", newline="\r\n").write(text)

    tex_dir = os.path.join(ROOT, "source", car["source"], "textures")
    k = build_kn5(nodes, baked, y_cog, tex_dir, car_id)
    kn5.write(k, os.path.join(out, car_id + ".kn5"))
    kn5.write(build_collider(hp, y_cog, family(car["variant"]) == "rental"), os.path.join(out, "collider.kn5"))

    has_sfx = sfx(car_id, car, out)

    # shadows
    lo, hi = hp["bbox"]
    shadow_png(os.path.join(out, "body_shadow.png"), 40, 52, 8)
    for i in range(4):
        shadow_png(os.path.join(out, "tyre_%d_shadow.png" % i), 34, 48, 6)

    # skins
    skins_src = skins_dir(car["variant"])            # painted by tools/paint_skins.py (build/skins/<variant>)
    for skin in built_skins(os.listdir(skins_src)):
        sd = os.path.join(out, "skins", skin)
        os.makedirs(sd)
        shutil.copy(os.path.join(skins_src, skin, "body_livery.png"), sd)
        if os.path.exists(os.path.join(skins_src, skin, "tyre.png")):      # worn-tyre liveries
            shutil.copy(os.path.join(skins_src, skin, "tyre.png"), sd)
        num = skin.split("_")[0].lstrip("0") or "0"
        json.dump(dict(skinname=skin.replace("_", " ").title(), drivername="", country="Brazil",
                       team="Techspeed", number=int(num), priority=1), open(os.path.join(sd, "ui_skin.json"), "w"), indent=2)
        prev = os.path.join(skins_src, skin, "preview.jpg")
        if os.path.exists(prev):
            shutil.copy(prev, sd)
        # the skin's icon in AC's selector: the front panel's number plate (it was cut from the side pods' print,
        # all black on the v1, which has none)
        livery = Image.open(os.path.join(skins_src, skin, "body_livery.png")).convert("RGB")
        u0, v0, u1, v1 = ATLAS["panel"]
        n = livery.size[0]
        cx, cy, half = (u0 + u1) / 2 * n, (1 - (v0 + 0.77 * (v1 - v0))) * n, 0.13 * n
        livery.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((64, 64), Image.LANCZOS).save(
            os.path.join(sd, "livery.png"))

    # ui
    os.makedirs(os.path.join(out, "ui"))
    tq = [[str(r), "%.1f" % t] for r, t in car["torque"]]
    pw = [[str(r), "%.1f" % (t * r * 2 * math.pi / 60 / 745.7)] for r, t in car["torque"]]
    # in AC it is the Techspeed Rental, never "a kart" (the name, description and tags say only what it is called)
    desc = ("%s (%s, %s), as run at Velopark. Built from photos of the real ones; physics adapted from the GoKart 0.5c "
            "mod and Honda data." % (car["name"], car["engine"], car["bhp"]))
    json.dump(dict(name=car["name"], brand="Techspeed", description=desc, tags=["rwd", "techspeed", car["variant"]],
                   **{"class": "race" if car["variant"] == "racing" else "street"},
                   specs=dict(bhp=car["bhp"], torque="%dNm" % max(t for _, t in car["torque"]), weight="%dkg*" % car["mass"],
                              topspeed=car["top_kmh"] + "km/h", acceleration="-", pwratio="-"),
                   torqueCurve=tq, powerCurve=pw, country="Brazil", author="", version="1.0", year=2026),
              open(os.path.join(out, "ui", "ui_car.json"), "w"), indent=2)
    badge = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(badge)
    d.ellipse([4, 4, 124, 124], fill=(200, 16, 22, 255))
    d.text((64, 64), "TS", fill=(255, 255, 255, 255), anchor="mm",
           font=__import__("PIL.ImageFont", fromlist=["x"]).truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", "ariblk.ttf"), 54))
    badge.save(os.path.join(out, "ui", "badge.png"))
    badge.save(os.path.join(out, "logo.png"))
    print("AC: wrote", out, "(sfx reused)" if has_sfx else "(no sfx)")
    return out


if __name__ == "__main__":
    out_root = os.path.join(ROOT, "build", "ac", "content", "cars")
    if not sys.argv[1:] and os.path.isdir(out_root):
        for old in os.listdir(out_root):          # drop cars no longer built (the parked racing kart)
            if old not in ACTIVE_CARS:
                shutil.rmtree(os.path.join(out_root, old))
    for cid in (sys.argv[1:] or list(ACTIVE_CARS)):
        export(cid, out_root)
