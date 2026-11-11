"""How far each wheel gives against the frame under a cornering load: the chassis part's beams and hydros
together with the beams the game generates for every pressureWheel (as bng_full_modes), the frame held
fixed, and a lateral plus a vertical load on each tyre's contact patch. Prints the camber and toe the
axle nodes, the rim and the tread ring take, and where the give is (strain energy by beam).

The front knuckles once cambered ~10 deg and the rear hubs ~14 deg at 600 N + 600 N: the wheels shook
in hard corners like a loose joint. Now ~1.3 / ~1.1 deg at the axle, ~2.5 deg at the tread with the tyre.

    python tools/dev/bng_wheel_load.py [part] [lateral N] [vertical N]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_modes as bm  # noqa: E402
import bng_wheel_modes as wm  # noqa: E402

KNUCKLE = ("sa", "wf", "wr", "ku", "kd", "kf", "rk")    # node prefixes free to move with the wheels (and the rack)


def assemble(path, part):
    jb = json.load(open(path, encoding="utf8"))[part]
    pos, names, idx = [], [], {}
    for r, o in bm.rows(jb["nodes"]):
        idx[r[0]] = len(pos)
        pos.append(np.array(r[1:4], float))
        names.append(r[0])
    beams = []
    for r, o in list(bm.rows(jb["beams"])) + list(bm.rows(jb.get("hydros", [["h"]]))):
        # the steering stops only push once past full lock
        if r[0] in idx and r[1] in idx and "SUPPORT" not in str(o.get("beamType", "")):
            beams.append((idx[r[0]], idx[r[1]], float(o.get("beamSpring", 4300000)), "chassis"))
    wheels = {}
    for name, w, n1, n2, _ in wm.kart_wheels(path, part):
        row = next(r for r, _ in wm.table_rows(jb["pressureWheels"]) if r[0] == name)
        a1, a2 = idx[row[3]], idx[row[4]]
        p, _, b, _ = wm.build(w, n1, n2, 1.0)
        local = {0: a1, 1: a2}
        for i in range(2, len(p)):
            local[i] = len(pos)
            pos.append(p[i])
            names.append("%s_%d" % (name, i))
        nt = 2 + 2 * w["numRays"]
        beams += [(local[a], local[b_], k, fam) for a, b_, k, fam in b]
        wheels[name] = (a1, a2, [local[i] for i in range(2, nt)], [local[i] for i in range(nt, len(p))])
    return np.array(pos), names, beams, wheels


def stiffness(pos, beams):
    n = len(pos)
    K = np.zeros((3 * n, 3 * n))
    for a, b, k, _ in beams:
        if k <= 0:
            continue
        u = pos[b] - pos[a]
        u /= np.linalg.norm(u)
        kk = k * np.outer(u, u)
        for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            K[3 * p:3 * p + 3, 3 * q:3 * q + 3] += s * kk
    return K + np.eye(3 * n) * 1e-3


def rotation(pos, U, group):
    """Best-fit small rotation (deg about x, y, z) of a group of nodes."""
    c = pos[group].mean(0)
    A, b = np.zeros((3 * len(group), 6)), np.zeros(3 * len(group))
    for k, (p, d) in enumerate(zip(pos[group] - c, U[group])):
        A[3 * k:3 * k + 3, :3] = np.eye(3)
        A[3 * k:3 * k + 3, 3:] = -np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        b[3 * k:3 * k + 3] = d
    return np.degrees(np.linalg.lstsq(A, b, rcond=None)[0][3:])


def main():
    args = sys.argv[1:]
    part = args[0] if args else "techspeed_chassis_rental"
    f_lat = float(args[1]) if len(args) > 1 else 600.0
    f_vert = float(args[2]) if len(args) > 2 else 600.0
    path = os.path.join(wm.ROOT, "build", "beamng", "vehicles", "techspeed_kart", "techspeed_kart.jbeam")
    pos, names, beams, wheels = assemble(path, part)
    n = len(pos)
    K = stiffness(pos, beams)
    moving = {i for i, nm in enumerate(names) if nm.startswith(KNUCKLE)}
    for a1, a2, tread, hub in wheels.values():
        moving |= set(tread) | set(hub)
    dof = [3 * i + j for i in sorted(moving) for j in range(3)]
    Kf = K[np.ix_(dof, dof)]
    print("%s: %.0f N lateral (inboard on the tyre) + %.0f N vertical at each contact patch, frame fixed"
          % (part, f_lat, f_vert))
    print("  deg of camber / toe       axle    rim  tread   contact moves")
    for name, (a1, a2, tread, hub) in wheels.items():
        axis = pos[a1] - pos[a2]
        axis /= np.linalg.norm(axis)
        mid = (pos[a1] + pos[a2]) / 2
        # contact patch: tread nodes within 40 deg of the bottom, weighted so the load centre sits right
        # under the axle on both tread rings, whatever the nodes' phase
        rel = pos[tread] - mid
        radial = rel - np.outer(rel @ axis, axis)
        low = [t for t, r_, q in zip(tread, rel, radial) if r_[2] < -np.cos(np.radians(40)) * np.linalg.norm(q)]
        side = np.array([np.sign((pos[i] - mid) @ axis) for i in low])
        dy = np.array([pos[i][1] - mid[1] for i in low])
        A = np.vstack([side > 0, side < 0, dy * (side > 0), dy * (side < 0)]).astype(float)
        wts = A.T @ np.linalg.solve(A @ A.T + 1e-12 * np.eye(4), np.array([0.5, 0.5, 0, 0]))
        F = np.zeros(3 * n)
        for i, wt in zip(low, wts):
            F[3 * i:3 * i + 3] += (-axis * f_lat + np.array([0, 0, f_vert])) * wt
        # the wheel spins freely about its axle: keep that zero-energy mode out of the load and the answer
        spin = np.zeros(3 * n)
        for i in tread + hub:
            spin[3 * i:3 * i + 3] = np.cross(axis, pos[i] - pos[a1])
        spin /= np.linalg.norm(spin)
        F -= spin * (spin @ F)
        u = np.zeros(3 * n)
        u[dof] = np.linalg.solve(Kf, F[dof])
        u -= spin * (spin @ u)
        U = u.reshape(-1, 3)
        rots = [rotation(pos, U, g) for g in ([a1, a2], hub, tread)]
        contact = np.mean([np.linalg.norm(U[i]) for i in low]) * 1000
        print("  %s camber %6.2f %6.2f %6.2f   %.1f mm" % ((name,) + tuple(abs(r[1]) for r in rots) + (contact,)))
        print("     toe    %6.2f %6.2f %6.2f" % tuple(abs(r[2]) for r in rots))
        E = {}
        for a, b, k, fam in beams:
            if k <= 0:
                continue
            e = pos[b] - pos[a]
            dl = (U[b] - U[a]) @ (e / np.linalg.norm(e))
            key = fam if fam != "chassis" else "%s-%s" % (names[a], names[b])
            E[key] = E.get(key, 0) + 0.5 * k * dl * dl
        tot = sum(E.values())
        print("     give: " + ", ".join("%s %.0f%%" % (k, 100 * v / tot)
                                     for k, v in sorted(E.items(), key=lambda x: -x[1])[:6]))


if __name__ == "__main__":
    main()
