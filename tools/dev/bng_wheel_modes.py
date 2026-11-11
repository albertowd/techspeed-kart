"""Stability check of a BeamNG pressureWheel. It rebuilds the wheel's generated nodes and beams the way
lua/common/jbeam/sections/wheelTypes/pressureWheels.lua does, then takes the highest vibration
frequency from the mass-scaled stiffness matrix. omega*dt at the 2 kHz step must stay well under 2;
the anisotropic sidewall beams are counted at their (stiffer) expansion rate.

    python tools/dev/bng_wheel_modes.py [jbeam] [part]     # every wheel of the kart part
    python tools/dev/bng_wheel_modes.py --pigeon            # the stock pigeon race tyre, for comparison
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DT = 0.0005


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def perpendicular(a):
    v = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0])
    return v / np.linalg.norm(v)


def build(w, n1, n2, axle_mass):
    """Nodes (pos, mass) and beams (a, b, k, family) of one pressureWheel. Axle nodes are 0 and 1, the
    tyre nodes start at index 2, then the hub nodes. family is the jbeam key stem (wheelTreadBeam, ...)."""
    n1, n2 = np.array(n1, float), np.array(n2, float)
    axis = (n2 - n1) / np.linalg.norm(n2 - n1)
    mid = (n1 + n2) / 2 + axis * w.get("wheelOffset", 0)
    rays = w.get("numRays", 10)
    pos, mass = [n1, n2], [axle_mass, axle_mass]
    half = rot(axis, math.pi / rays)
    step = rot(axis, 2 * math.pi / rays)
    # tyre nodes
    t1, t2 = mid - 0.5 * w["tireWidth"] * axis, mid + 0.5 * w["tireWidth"] * axis
    v2 = perpendicular(axis) * w["radius"]
    v1 = half @ v2
    tire0 = len(pos)
    for _ in range(rays):
        pos.append(t1 + v2), mass.append(w["nodeWeight"])
        v2 = step @ v2
        pos.append(t2 + v1), mass.append(w["nodeWeight"])
        v1 = step @ v1
    # hub nodes
    h1, h2 = mid - 0.5 * w["hubWidth"] * axis, mid + 0.5 * w["hubWidth"] * axis
    v = perpendicular(axis) * w["hubRadius"]
    hub0 = len(pos)
    for _ in range(rays):
        pos.append(h2 + v), mass.append(w["hubNodeWeight"])
        v = half @ v
        pos.append(h1 + v), mass.append(w["hubNodeWeight"])
        v = half @ v
    B = []
    hs = w["hubSideBeamSpring"]
    hr = w.get("hubReinfBeamSpring", hs)
    side_k = max(w.get("wheelSideBeamSpring", 0), w.get("wheelSideBeamSpringExpansion", 0))
    sider_k = max(w.get("wheelSideReinfBeamSpring", 0), w.get("wheelSideReinfBeamSpringExpansion", 0))
    for i in range(rays):
        i2, nx = 2 * i, 2 * ((i + 1) % rays)
        oh, ih, noh, nih = hub0 + i2, hub0 + i2 + 1, hub0 + nx, hub0 + nx + 1
        it, ot, nit, not_ = tire0 + i2, tire0 + i2 + 1, tire0 + nx, tire0 + nx + 1
        nnit = tire0 + 2 * ((i + 2) % rays)
        B += [(oh, ih, w["hubTreadBeamSpring"], "hub"), (ih, noh, w["hubTreadBeamSpring"], "hub"),
              (oh, noh, w["hubPeripheryBeamSpring"], "hub"), (ih, nih, w["hubPeripheryBeamSpring"], "hub"),
              (oh, 0, hr, "hub"), (oh, 1, hs, "hub"), (ih, 0, hs, "hub"), (ih, 1, hr, "hub")]
        tr = "wheelTreadBeam"
        B += [(it, ot, w[tr + "Spring"], tr), (ot, nit, w[tr + "Spring"], tr)]
        if w.get("enableTreadReinfBeams"):
            tr = "wheelTreadReinfBeam"
            B += [(it, not_, w[tr + "Spring"], tr), (ot, nnit, w[tr + "Spring"], tr)]
        pe = "wheelPeripheryBeam"
        B += [(it, nit, w[pe + "Spring"], pe), (ot, not_, w[pe + "Spring"], pe)]
        sd = "wheelSideBeam"
        B += [(oh, ot, side_k, sd), (ot, noh, side_k, sd), (ih, it, side_k, sd), (ih, nit, side_k, sd)]
        reinf = w.get("wheelReinfBeamSpring", 0)
        if w.get("enableTireReinfBeams") or w.get("enableTireLbeams"):
            B += [(it, oh, reinf, "wheelReinfBeam"), (ih, ot, reinf, "wheelReinfBeam")]
        if w.get("enableTireSideReinfBeams"):
            sr = "wheelSideReinfBeam"
            B += [(oh, not_, sider_k, sr), (ot, hub0 + 2 * ((i + 2) % rays), sider_k, sr),
                  (it, nih, sider_k, sr), (ih, nnit, sider_k, sr)]
        if w.get("enableTirePeripheryReinfBeams"):
            nnot = nnit + 1
            pr = "wheelPeripheryReinfBeam"
            B += [(it, nnit, w[pr + "Spring"], pr), (ot, nnot, w[pr + "Spring"], pr)]
    return np.array(pos), np.array(mass), B, (tire0, hub0)


def omega_dt(pos, mass, beams):
    n = len(pos)
    K = np.zeros((3 * n, 3 * n))
    for a, b, k, *_ in beams:
        if k <= 0:
            continue
        d = pos[b] - pos[a]
        u = d / np.linalg.norm(d)
        kk = k * np.outer(u, u)
        for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            K[3 * p:3 * p + 3, 3 * q:3 * q + 3] += s * kk
    mi = 1 / np.sqrt(np.repeat(mass, 3))
    w2 = np.linalg.eigvalsh(K * mi[:, None] * mi[None, :])
    return math.sqrt(max(w2[-1], 0)) * DT


def damping(w, rays_nodes=None):
    """Largest unfiltered damping sum per tyre node, c*dt/m (beams with a DampCutoffHz are filtered in
    game and left out). The old, stable kart tyre was ~1.0; keep it about there."""
    per_node = {  # beams of each family touching one tyre node
        "wheelTreadBeam": 4, "wheelTreadReinfBeam": 4 if w.get("enableTreadReinfBeams") else 0,
        "wheelPeripheryBeam": 2, "wheelPeripheryReinfBeam": 2 if w.get("enableTirePeripheryReinfBeams") else 0,
        "wheelSideBeam": 2, "wheelReinfBeam": 1 if (w.get("enableTireLbeams") or w.get("enableTireReinfBeams")) else 0,
        "wheelSideReinfBeam": 2 if w.get("enableTireSideReinfBeams") else 0}
    c = 0.0
    for fam, n in per_node.items():
        if not n or w.get(fam + "DampCutoffHz"):
            continue
        c += n * max(w.get(fam + "Damp", 0), w.get(fam + "DampExpansion", 0))
    return c * DT / w["nodeWeight"]


def table_rows(section):
    cur, out = {}, []
    for r in section[1:]:
        if isinstance(r, dict):
            cur = {**cur, **r}
        else:
            out.append((r, dict(cur)))
    return out


PIGEON = dict(numRays=16, radius=0.242, tireWidth=0.145, hubRadius=0.145, hubWidth=0.13, nodeWeight=0.115,
              hubNodeWeight=0.3, hubTreadBeamSpring=701000, hubPeripheryBeamSpring=701000, hubSideBeamSpring=901000,
              enableTreadReinfBeams=True, enableTirePeripheryReinfBeams=True, enableTireLbeams=True,
              wheelSideBeamSpring=22 * 255, wheelSideBeamSpringExpansion=261000, wheelReinfBeamSpring=16000,
              wheelTreadBeamSpring=121000, wheelTreadReinfBeamSpring=81000, wheelPeripheryBeamSpring=61000,
              wheelPeripheryReinfBeamSpring=61000)


def kart_wheels(path=None, part="techspeed_chassis_rental"):
    """[(name, wheel options, node1 pos, node2 pos, axle node mass)] of a built kart jbeam part."""
    path = path or os.path.join(ROOT, "build", "beamng", "vehicles", "techspeed_kart", "techspeed_kart.jbeam")
    jb = json.load(open(path, encoding="utf8"))[part]
    nodes = {}
    for r, o in table_rows(jb["nodes"]):
        extra = r[4] if len(r) > 4 and isinstance(r[4], dict) else {}
        nodes[r[0]] = (np.array(r[1:4], float), extra.get("nodeWeight", o.get("nodeWeight", 25)))
    out = []
    for r, o in table_rows(jb["pressureWheels"]):
        w = {**o, **(r[-1] if isinstance(r[-1], dict) else {})}
        n1, n2 = nodes[r[3]], nodes[r[4]]
        out.append((r[0], w, n1[0], n2[0], min(n1[1], n2[1])))
    return out


def main():
    if "--pigeon" in sys.argv:
        w = PIGEON
        p, m, b, _ = build(w, (0, -0.07, 0.3), (0, 0.07, 0.3), 3.0)
        print("pigeon 165/70R10 race: omega*dt %.2f" % omega_dt(p, m, b))
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    part = args[1] if len(args) > 1 else "techspeed_chassis_rental"
    for name, w, n1, n2, am in kart_wheels(args[0] if args else None, part):
        p, m, b, _ = build(w, n1, n2, am)
        print("%s %s: omega*dt %.2f, tyre damping c*dt/m %.2f" % (part, name, omega_dt(p, m, b), damping(w)))


if __name__ == "__main__":
    main()
