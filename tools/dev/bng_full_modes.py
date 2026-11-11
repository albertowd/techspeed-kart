"""Highest vibration mode of the whole kart: the chassis part's nodes, beams and hydros together with the
nodes and beams the game generates for every pressureWheel (bng_wheel_modes.build), joined at the
wheels' axle nodes. The wheel check alone holds each axle node with only the wheel's own beams; in
game the axle nodes also carry the spindle beams, and the two add up. omega*dt at the 2 kHz step must
stay under 2; stock cars sit at about 1.75 or below.

    python tools/dev/bng_full_modes.py [jbeam] [part]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_modes as bm  # noqa: E402
import bng_wheel_modes as wm  # noqa: E402

import json  # noqa: E402


def assemble(path, part):
    jb = json.load(open(path, encoding="utf8"))[part]
    pos, mass, names = [], [], []
    idx = {}
    for r, o in bm.rows(jb["nodes"]):
        idx[r[0]] = len(pos)
        pos.append(np.array(r[1:4], float)), mass.append(o.get("nodeWeight", 1.0)), names.append(r[0])
    beams = []
    for r, o in list(bm.rows(jb["beams"])) + list(bm.rows(jb.get("hydros", [["h"]]))):
        if r[0] in idx and r[1] in idx:
            beams.append((idx[r[0]], idx[r[1]], float(o.get("beamSpring", 4300000))))
    for name, w, n1, n2, _ in wm.kart_wheels(path, part):
        row = next(r for r, _ in wm.table_rows(jb["pressureWheels"]) if r[0] == name)
        a1, a2 = idx[row[3]], idx[row[4]]
        p, m, b, _ = wm.build(w, n1, n2, 1.0)
        base = len(pos)
        local = {0: a1, 1: a2}
        for i in range(2, len(p)):
            local[i] = base + i - 2
            pos.append(p[i]), mass.append(m[i]), names.append("%s_%s%d" % (name, "t" if i < 2 + 2 * w["numRays"] else "h", i))
        beams += [(local[a], local[b_], k) for a, b_, k, *_ in b]
    return np.array(pos), np.array(mass), beams, names


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(wm.ROOT, "build", "beamng", "vehicles", "techspeed_kart",
                                             "techspeed_kart.jbeam")
    part = args[1] if len(args) > 1 else "techspeed_chassis_rental"
    pos, mass, beams, names = assemble(path, part)
    n = len(pos)
    K = np.zeros((3 * n, 3 * n))
    for a, b, k in beams:
        if k <= 0:
            continue
        d = pos[b] - pos[a]
        u = d / np.linalg.norm(d)
        kk = k * np.outer(u, u)
        for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            K[3 * p:3 * p + 3, 3 * q:3 * q + 3] += s * kk
    mi = 1 / np.sqrt(np.repeat(mass, 3))
    w2, vec = np.linalg.eigh(K * mi[:, None] * mi[None, :])
    print("%s: %d nodes (%.1f kg), %d beams" % (part, n, mass.sum(), len(beams)))
    for i in range(1, 6):
        v = (vec[:, -i] * mi).reshape(-1, 3)
        amp = np.linalg.norm(v, axis=1)
        top = np.argsort(-amp)[:4]
        print("  omega*dt %.2f  moves: %s" % (np.sqrt(max(w2[-i], 0)) * wm.DT, ", ".join(names[t] for t in top)))


if __name__ == "__main__":
    main()
