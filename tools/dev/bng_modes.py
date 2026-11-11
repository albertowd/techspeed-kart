"""Modal check of a BeamNG chassis part: assemble the linear stiffness of its beams (hydros included),
lump the node masses and list the lowest vibration modes. A stiff frame has six rigid-body modes
near 0 Hz and then nothing below a few tens of Hz; extra near-zero modes are mechanisms (parts that
can move without stretching any beam), which show in game as a frame that bends.

    python tools/dev/bng_modes.py [jbeam] [part] [--groups kart_chassis,...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def rows(section):
    """Yield (row, opts) from a jbeam table, carrying header-less option dicts forward."""
    cur = {}
    for r in section[1:]:
        if isinstance(r, dict):
            cur = {**cur, **r}
            continue
        extra = r[-1] if isinstance(r[-1], dict) else {}
        yield r, {**cur, **extra}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else os.path.join(ROOT, "build", "beamng", "vehicles", "techspeed_kart",
                                             "techspeed_kart.jbeam")
    part = args[1] if len(args) > 1 else "techspeed_chassis_rental"
    groups = None
    for a in sys.argv[1:]:
        if a.startswith("--groups="):
            groups = set(a.split("=", 1)[1].split(","))
    jb = json.load(open(path, encoding="utf8"))[part]

    pos, mass, grp = {}, {}, {}
    for r, o in rows(jb["nodes"]):
        pos[r[0]] = np.array(r[1:4], float)
        mass[r[0]] = o.get("nodeWeight", 1.0)
        g = o.get("group", [])
        grp[r[0]] = set([g] if isinstance(g, str) else g)
    keep = [n for n in pos if groups is None or grp[n] & groups]
    idx = {n: i for i, n in enumerate(keep)}
    n = len(keep)
    K = np.zeros((3 * n, 3 * n))
    beams = list(rows(jb["beams"])) + list(rows(jb.get("hydros", [["h"]])))
    used = 0
    for r, o in beams:
        a, b = r[0], r[1]
        if a not in idx or b not in idx:
            continue
        k = float(o.get("beamSpring", 4300000))
        d = pos[b] - pos[a]
        L = np.linalg.norm(d)
        u = d / L
        kk = k * np.outer(u, u)
        ia, ib = 3 * idx[a], 3 * idx[b]
        K[ia:ia + 3, ia:ia + 3] += kk
        K[ib:ib + 3, ib:ib + 3] += kk
        K[ia:ia + 3, ib:ib + 3] -= kk
        K[ib:ib + 3, ia:ia + 3] -= kk
        used += 1
    m = np.repeat([mass[x] for x in keep], 3)
    Mi = 1 / np.sqrt(m)
    w2, vec = np.linalg.eigh(K * Mi[:, None] * Mi[None, :])
    f = np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi)
    print("%s: %d nodes, %d beams, %.1f kg" % (part, n, used, sum(mass[x] for x in keep)))
    soft = int((f < 1.0).sum())
    print("modes under 1 Hz: %d (6 are rigid-body); highest mode %.0f Hz, omega*dt %.2f at 2 kHz (must stay < 2)"
          % (soft, f[-1], 2 * np.pi * f[-1] * 0.0005))
    for i in range(min(len(f), 16)):
        v = (vec[:, i] * Mi).reshape(-1, 3)
        amp = np.linalg.norm(v, axis=1)
        top = np.argsort(-amp)[:4]
        print("  %2d  %8.2f Hz   moves: %s" % (i, f[i], ", ".join(keep[t] for t in top)))


if __name__ == "__main__":
    main()
