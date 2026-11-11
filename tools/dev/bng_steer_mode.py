"""Steering mode of a front wheel: the left knuckle and its wheel twisting about the kingpin on the tie-rod
link, with the frame held still. Prints the modes with the most kingpin twist, their frequency and damping
ratio, with and without the steering damper beams (outer hub node to the side bar). A sliding tyre doesn't
damp this; at ~0.4% of critical the front wheels kicked at full lock at speed. The push-only steering stops
are left out (they only touch past full lock).

    python tools/dev/bng_steer_mode.py [techspeed_chassis_rental|techspeed_chassis_racing]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_modes as bm, bng_wheel_modes as wm, bng_full_modes as fm
path = os.path.join(wm.ROOT, "build", "beamng", "vehicles", "techspeed_kart", "techspeed_kart.jbeam")
part = sys.argv[1] if len(sys.argv) > 1 else "techspeed_chassis_rental"
jb = json.load(open(path))[part]
pos, mass, beams, names = fm.assemble(path, part)
idx = {n: i for i, n in enumerate(names)}
sup = set()
for r, o in bm.rows(jb["beams"]):
    if o.get("beamType") == "|SUPPORT":
        sup |= {(idx[r[0]], idx[r[1]]), (idx[r[1]], idx[r[0]])}
beams = [b for b in beams if (b[0], b[1]) not in sup]
damp = []
for r, o in list(bm.rows(jb["beams"])) + list(bm.rows(jb.get("hydros", [["h"]]))):
    if r[0] in idx and r[1] in idx:
        if o.get("beamType") != "|SUPPORT":
            damp.append((idx[r[0]], idx[r[1]], float(o.get("beamDamp", 0)), r))
print("damper rows:", [(r[0], r[1], c) for a, b, c, r in damp if r[0].startswith("wf") and r[1].startswith("n1")])
# the knuckle as the exporter groups it (hub, arm, ears and kingpin pivots), and the wheel's own nodes
knuckle = {r[0] for r, o in bm.rows(jb["nodes"]) if "knuckle_FL" in (o.get("group") or [])}
free = [i for i, nm in enumerate(names) if nm in knuckle or nm.startswith("FL_")]
fi = {g: k for k, g in enumerate(free)}
m = len(free)
def mat(items):
    M_ = np.zeros((3 * m, 3 * m))
    for a, b, k in items:
        if k <= 0 or (a not in fi and b not in fi): continue
        d = pos[b] - pos[a]; u = d / np.linalg.norm(d); kk = k * np.outer(u, u)
        for p, q, s in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
            if p in fi and q in fi:
                M_[3*fi[p]:3*fi[p]+3, 3*fi[q]:3*fi[q]+3] += s * kk
    return M_
K = mat(beams)
Minv = np.repeat(1 / mass[free], 3)
kp = pos[idx["kbl"]]; ax = pos[idx["ktl"]] - kp; ax /= np.linalg.norm(ax)
yaw = np.array([np.cross(ax, pos[g] - kp) for g in free]).ravel()      # rigid rotation about the kingpin
for label, cdamp in (("no damper", 0.0), ("damper", 1.0)):
    C = mat([(a, b, c * (cdamp if r[0].startswith("wf") and r[1].startswith("n1") else 1)) for a, b, c, r in damp])
    A = np.zeros((6 * m, 6 * m))
    A[:3*m, 3*m:] = np.eye(3 * m)
    A[3*m:, :3*m] = -Minv[:, None] * K
    A[3*m:, 3*m:] = -Minv[:, None] * C
    ev, V = np.linalg.eig(A)
    out = []
    for j in np.where(ev.imag > 1)[0]:
        v = V[:3*m, j]
        w8 = np.repeat(mass[free], 3)
        share = abs(np.vdot(yaw * w8, v)) / np.sqrt((w8 * yaw ** 2).sum() * (w8 * abs(v) ** 2).sum())
        out.append((share, ev[j].imag / 2 / np.pi, -ev[j].real / abs(ev[j])))
    out.sort(reverse=True)
    print(label, ["%.1f Hz zeta %.3f (yaw share %.2f)" % (f, z, s) for s, f, z in out[:3]])
