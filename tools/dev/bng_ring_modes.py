"""How long the kart rings after a bump: the vibration modes of the whole kart (chassis part and the
generated pressureWheels, as bng_full_modes) standing on its tyres (the lowest tread nodes of each tyre
held), up to 30 Hz, with each mode's damping ratio and which beams supply it (modal damping,
zeta = phi' C phi / 2 omega, per beam family) and, with -s, which beams flex.

After a kerb the kart once kept shaking for ~1 s: the camera frame, steering-wheel prop and plastics on
soft mounts damped at 3-4 Ns/m (zeta ~0.005) and the frame bouncing on its barely damped tyres
(zeta ~0.015). Aim for zeta >= ~0.05 everywhere; a seated driver damps a real kart's chassis, see
DRIVER_BODY in export_beamng.py. The fore-aft mode near 4 Hz is the wheels rolling, an artefact of
holding the tread nodes.

    python tools/dev/bng_ring_modes.py [part] [-s] [-n Hz]      # -n: the nodes a mode moves
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_modes as bm  # noqa: E402
import bng_wheel_modes as wm  # noqa: E402


def tag(a, b, o):
    """Beam family of a chassis beam, from its node names."""
    if o.get("beamSpring", 1) == 0:
        return "steer damper"
    for pre, name in (("stw", "steering wheel prop"), ("cam", "camera"), ("drv", "driver")):
        if a.startswith(pre) or b.startswith(pre):
            return name
    if any(x.startswith("p") and not x[1].isdigit() for x in (a, b)):
        return "plastic"
    if o.get("dampCutoffHz"):
        return "knuckle"
    return "frame"


def where(nm):
    """What a node belongs to, for the 'moves' column."""
    if nm[:3] in ("FR_", "FL_", "RR_", "RL_"):
        return "tyres"
    if nm.startswith("p") and not nm[1].isdigit():
        return "plastic"
    for pre, name in (("drv", "driver"), ("stw", "steer wheel"), ("cam", "camera")):
        if nm.startswith(pre):
            return name
    return "frame"


def main():
    args = [a for a in sys.argv[1:]]
    flags = {"-s": "-s" in args}
    probe = float(args[args.index("-n") + 1]) if "-n" in args else None
    pos_args = [a for i, a in enumerate(args) if not a.startswith("-") and (i == 0 or args[i - 1] != "-n")]
    part = pos_args[0] if pos_args else "techspeed_chassis_rental"
    path = pos_args[1] if len(pos_args) > 1 else os.path.join(wm.ROOT, "build", "beamng", "vehicles",
                                                               "techspeed_kart", "techspeed_kart.jbeam")
    jb = json.load(open(path, encoding="utf8"))[part]
    pos, mass, names, idx = [], [], [], {}
    for r, o in bm.rows(jb["nodes"]):
        idx[r[0]] = len(pos)
        pos.append(np.array(r[1:4], float))
        mass.append(o.get("nodeWeight", 25))
        names.append(r[0])
    beams = []
    for r, o in list(bm.rows(jb["beams"])) + list(bm.rows(jb.get("hydros", [["h"]]))):
        if r[0] in idx and r[1] in idx:
            sup = "SUPPORT" in str(o.get("beamType", ""))          # the steering stops: past full lock only
            beams.append((idx[r[0]], idx[r[1]], 0 if sup else float(o.get("beamSpring", 4300000)),
                          0 if sup else float(o.get("beamDamp", 580)), tag(r[0], r[1], o)))
    for name, w, n1, n2, _ in wm.kart_wheels(path, part):
        row = next(r for r, _ in wm.table_rows(jb["pressureWheels"]) if r[0] == name)
        p, m, b, _ = wm.build(w, n1, n2, 1.0)
        local = {0: idx[row[3]], 1: idx[row[4]]}
        for i in range(2, len(p)):
            local[i] = len(pos)
            pos.append(p[i])
            mass.append(m[i])
            names.append("%s_%d" % (name, i))
        for a, b_, k, fam in b:
            c = (w.get("hubTreadBeamDamp", 0) if fam == "hub"
                 else max(w.get(fam + "Damp", 0), w.get(fam + "DampExpansion", 0)))
            beams.append((local[a], local[b_], k, c, "tyre " + fam))
    pos, mass, n = np.array(pos), np.array(mass), len(pos)
    fams = sorted({b[4] for b in beams})
    K = np.zeros((3 * n, 3 * n))
    C = {f: np.zeros((3 * n, 3 * n)) for f in fams}
    Kfam = {f: np.zeros((3 * n, 3 * n)) for f in fams}
    for a, b, k, c, f in beams:
        u = pos[b] - pos[a]
        u /= np.linalg.norm(u)
        uu = np.outer(u, u)
        for M, v in ((K, k), (C[f], c), (Kfam[f], k)):
            if v <= 0:
                continue
            for p_, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                M[3 * p_:3 * p_ + 3, 3 * q:3 * q + 3] += s * v * uu
    held = set()
    for wn in ("FR", "FL", "RR", "RL"):
        t = [i for i, nm in enumerate(names) if nm.startswith(wn + "_")]
        held |= set(sorted(t, key=lambda i: pos[i][2])[:4])
    free = [i for i in range(n) if i not in held]
    dof = np.array([3 * i + j for i in free for j in range(3)])
    mi = 1 / np.sqrt(np.repeat(mass[free], 3))
    w2, Ph = np.linalg.eigh(K[np.ix_(dof, dof)] * mi[:, None] * mi[None, :])
    print("%s on its tyres, modes 1-30 Hz (tau: time to decay to 37%%)" % part)
    for j in range(len(w2)):
        om = np.sqrt(max(w2[j], 0))
        f = om / 2 / np.pi
        if f < 1 or f > 30:
            continue
        ph = Ph[:, j] * mi
        contrib = {fa: ph @ C[fa][np.ix_(dof, dof)] @ ph / (2 * om) for fa in fams}
        z = sum(contrib.values())
        e = (Ph[:, j] ** 2).reshape(-1, 3).sum(1)
        moves = {}
        for k_, i in enumerate(free):
            moves[where(names[i])] = moves.get(where(names[i]), 0) + e[k_]
        print("  %5.1f Hz  zeta %.3f  tau %5.2f s | moves %s | damping from %s" % (
            f, z, 1 / max(z * om, 1e-9),
            ", ".join("%s %.0f%%" % (c_, 100 * v) for c_, v in sorted(moves.items(), key=lambda x: -x[1])[:2]),
            ", ".join("%s %.3f" % t for t in sorted(contrib.items(), key=lambda x: -x[1])[:3])))
        if flags["-s"]:
            strain = {fa: ph @ Kfam[fa][np.ix_(dof, dof)] @ ph for fa in fams}
            st = sum(strain.values())
            print("          flexes: " + ", ".join("%s %.0f%%" % (a_, 100 * b_ / st)
                                                    for a_, b_ in sorted(strain.items(), key=lambda x: -x[1])[:4]))
    if probe is not None:
        j = int(np.argmin(abs(np.sqrt(np.maximum(w2, 0)) / 2 / np.pi - probe)))
        e = (Ph[:, j] ** 2).reshape(-1, 3).sum(1)
        d = (Ph[:, j] * mi).reshape(-1, 3)
        print("mode %.1f Hz, kinetic energy by node:" % (np.sqrt(w2[j]) / 2 / np.pi))
        for k_ in np.argsort(-e)[:12]:
            print("   %-8s %4.0f%%  %.1f kg  dir %s" % (names[free[k_]], 100 * e[k_], mass[free[k_]],
                                                     np.round(d[k_] / np.linalg.norm(d[k_]), 2)))


if __name__ == "__main__":
    main()
