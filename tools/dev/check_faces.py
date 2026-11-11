"""Find parts the games would draw inside out. Blender renders both sides of every face, so the source
renders never show it; AC and BeamNG draw the front side only and an inside-out part shows as a hole.

Closed parts: flagged when their volume is negative (faces point in). Open parts: flagged when few faces
point away from the part's centre, a rough hint only (pod_inner_L is a flat wall facing the cockpit and is
right). An open sheet folded back on itself (the rim) can't be right either way round and needs a back
skin; this check can't see that.

    python tools/dev/check_faces.py [rental|racing ...]
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import blender_bridge as bb
for variant in sys.argv[1:] or ["rental", "racing"]:
    bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
    nodes, baked = bb.collect()
    for b in baked:
        p, t = b.pos, b.tris
        # weld by position
        key = np.round(p, 5)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        tw = inv[t]
        e = np.sort(np.r_[tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]], axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        closed = (cnt == 2).mean()
        vol = np.einsum("ij,ij->i", p[t[:, 0]], np.cross(p[t[:, 1]], p[t[:, 2]])).sum() / 6
        fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        c = p[t].mean(1)
        away = (np.einsum("ij,ij->i", fn, c - p.mean(0)) > 0).mean()
        flag = vol < 0 if closed > 0.98 else away < 0.4
        if flag:
            print("%-7s %-26s closed %.2f vol %+.2e  away %.2f" % (variant, b.name, closed, vol, away))
