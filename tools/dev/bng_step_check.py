"""Explicit-step stability of the whole kart, damping included: the chassis part's nodes, beams and hydros
with every pressureWheel's generated nodes and beams (as bng_full_modes), stepped like the game's
integrator (semi-implicit Euler at 2 kHz). Prints the largest growth factor per step of any vibration;
above 1 it grows every step and the kart explodes at spawn.

The undamped check (bng_full_modes, omega*dt) is not enough: for one node the step is stable only while
(omega*dt)^2 + 2*c*dt/m < 4, so stiff beams that are fine alone blow up once the same node carries a lot of
damping. That is how the kart failed to spawn on 2026-11-08 with every omega*dt check passing.

Beams with a dampCutoffHz (knuckle, steering dampers, the tyre's tread/periphery/L-beams) are low-passed
in game and left out of the damping here; --unfiltered counts them at full rate, a worst case that even
the last kart that spawned fine fails (in its tyres). Calibrated on that kart: it sits at 1.0000.

    python tools/dev/bng_step_check.py [part] [--unfiltered] [--jbeam path]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_modes as bm  # noqa: E402
import bng_wheel_modes as wm  # noqa: E402


TOL = 1e-4


def assemble(path, part, filtered):
    jb = json.load(open(path, encoding="utf8"))[part]
    pos, mass, names, idx = [], [], [], {}
    for r, o in bm.rows(jb["nodes"]):
        idx[r[0]] = len(pos)
        pos.append(np.array(r[1:4], float))
        mass.append(o.get("nodeWeight", 25))
        names.append(r[0])
    beams = []        # (a, b, k, c)
    for r, o in list(bm.rows(jb["beams"])) + list(bm.rows(jb.get("hydros", [["h"]]))):
        if r[0] not in idx or r[1] not in idx:
            continue
        support = "SUPPORT" in str(o.get("beamType", ""))
        k = 0.0 if support else float(o.get("beamSpring", 4300000))      # stops touch only past full lock
        c = float(o.get("beamDamp", 580))
        if filtered and o.get("dampCutoffHz"):
            c = 0.0
        beams.append((idx[r[0]], idx[r[1]], k, 0.0 if support else c))
    for name, w, n1, n2, _ in wm.kart_wheels(path, part):
        row = next(r for r, _ in wm.table_rows(jb["pressureWheels"]) if r[0] == name)
        a1, a2 = idx[row[3]], idx[row[4]]
        p, m, b, _ = wm.build(w, n1, n2, 1.0)
        local = {0: a1, 1: a2}
        for i in range(2, len(p)):
            local[i] = len(pos)
            pos.append(p[i])
            mass.append(m[i])
            names.append("%s_%d" % (name, i))
        for a, b_, k, fam in b:
            if fam == "hub":
                c = w.get("hubTreadBeamDamp", 0)
            else:
                c = max(w.get(fam + "Damp", 0), w.get(fam + "DampExpansion", 0))
                if filtered and w.get(fam + "DampCutoffHz"):
                    c = 0.0
            beams.append((local[a], local[b_], k, c))
    return np.array(pos), np.array(mass), beams, names


def matrices(pos, beams):
    n = len(pos)
    K, C = np.zeros((3 * n, 3 * n)), np.zeros((3 * n, 3 * n))
    for a, b, k, c in beams:
        u = pos[b] - pos[a]
        u /= np.linalg.norm(u)
        uu = np.outer(u, u)
        for M, v in ((K, k), (C, c)):
            if v <= 0:
                continue
            for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                M[3 * p:3 * p + 3, 3 * q:3 * q + 3] += s * v * uu
    return K, C


def growth(pos, mass, beams, dt=wm.DT):
    """Spectral radius of one semi-implicit Euler step and the nodes of the worst mode."""
    K, C = matrices(pos, beams)
    mi = np.repeat(1 / mass, 3)
    A, B = mi[:, None] * K, mi[:, None] * C
    m = len(mi)
    I = np.eye(m)
    T = np.zeros((2 * m, 2 * m))
    T[:m, :m] = I - dt * dt * A          # x' = x + dt v'
    T[:m, m:] = dt * (I - dt * B)
    T[m:, :m] = -dt * A                  # v' = v + dt (-A x - B v)
    T[m:, m:] = I - dt * B
    ev, V = np.linalg.eig(T)
    j = int(np.argmax(abs(ev)))
    amp = np.linalg.norm(V[:m, j].reshape(-1, 3), axis=1)
    return abs(ev[j]), np.argsort(-amp)[:5], ev


def main():
    args = sys.argv[1:]
    filtered = "--unfiltered" not in args
    path = os.path.join(wm.ROOT, "build", "beamng", "vehicles", "techspeed_kart", "techspeed_kart.jbeam")
    if "--jbeam" in args:
        path = args[args.index("--jbeam") + 1]
    built = json.load(open(path, encoding="utf8"))
    parts = [a for a in args if a.startswith("techspeed_")] or [p for p in built if p.startswith("techspeed_chassis_")]
    for part in parts:
        pos, mass, beams, names = assemble(path, part, filtered)
        rho, top, ev = growth(pos, mass, beams)
        bad = int((abs(ev) > 1 + TOL).sum())       # rigid-body modes sit at 1 within round-off
        print("%s: growth per step %.4f (%s), %d growing modes; worst moves %s"
              % (part, rho, "OK" if rho <= 1 + TOL else "UNSTABLE", bad, ", ".join(names[t] for t in top)))


if __name__ == "__main__":
    main()
