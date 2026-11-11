"""Clearance between the seat and the engine / radiator parts, for both variants.

    python tools/dev/check_clearance.py [min_mm]

Uses the parametric meshes (before modifiers; the seat shell's 5 mm thickness is added).
Reports the closest seat point to every part near it and fails if any is under min_mm.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kart_model  # noqa: E402

MIN_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
NEAR = ("engine", "airbox", "carb", "intake", "radiator", "coolant", "exhaust", "silencer", "muffler",
        "air_filter", "clutch", "fuel_tank", "chain", "sprocket")


def surface_points(mesh, step=0.006):
    """Vertices plus points sampled on the faces, so big flat faces count too."""
    v = mesh.verts
    pts = [v]
    for f in mesh.faces:
        tri = [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
        for a, b, c in tri:
            pa, pb, pc = v[a], v[b], v[c]
            n = int(max(np.linalg.norm(pb - pa), np.linalg.norm(pc - pa)) / step) + 1
            if n < 2:
                continue
            uu, ww = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
            m = uu + ww <= 1
            pts.append(pa + uu[m, None] * (pb - pa) + ww[m, None] * (pc - pa))
    return np.vstack(pts)


def main():
    ok = True
    for variant in ("racing", "rental"):
        sc = kart_model.build(variant)
        seat = next(p for p in sc.parts if p.name == "seat")
        sp = surface_points(seat.mesh)
        print(variant)
        for p in sc.parts:
            if p.node != "BODY" or not any(k in p.name for k in NEAR):
                continue
            q = surface_points(p.mesh)
            lo, hi = q.min(0) - 0.1, q.max(0) + 0.1
            cand = sp[np.all((sp > lo) & (sp < hi), axis=1)]
            if len(cand) == 0:
                continue
            # nearest distance via chunked brute force
            best = np.inf
            for i in range(0, len(cand), 400):
                d = np.linalg.norm(cand[i:i + 400, None, :] - q[None, :, :], axis=2)
                best = min(best, d.min())
            gap_mm = best * 1000 - 5.0          # seat shell thickness
            flag = "" if gap_mm >= MIN_MM else "   <-- TOO CLOSE"
            if gap_mm < MIN_MM:
                ok = False
            print("  %-22s %6.0f mm%s" % (p.name, gap_mm, flag))
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
