"""Static load test of a BeamNG pressureWheel: how far the tyre squashes under the kart's weight.

The wheel is rebuilt as the game builds it (bng_wheel_modes.build), the rim held fixed, the tyre
inflated (gauge pressure on the closed hub/tread/sidewall surface, as pressureWheels.lua lays out its
pressure triangles) and pressed onto a flat, frictionless floor. Sidewall beams are anisotropic
(compression spring below their rest length, expansion spring above it); every beam's rest length is
its built length times its precompression. Solved by damped relaxation.

    python tools/dev/bng_tyre_load.py              # every wheel of the built rental kart
    python tools/dev/bng_tyre_load.py --pigeon     # the stock pigeon 10" race tyre at 22 psi, for comparison
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_wheel_modes as wm  # noqa: E402

PSI = 6894.76


def setup(w, n1, n2):
    pos, mass, beams, (tire0, hub0) = wm.build(w, n1, n2, 1.0)
    rays = w.get("numRays", 10)
    fam = [b[3] for b in beams]
    A = np.array([b[0] for b in beams])
    Bn = np.array([b[1] for b in beams])
    L = np.linalg.norm(pos[Bn] - pos[A], axis=1)
    pre = np.array([w.get(f + "Precompression", 1.0) if f != "hub" else 1.0 for f in fam])
    kc = np.array([b[2] if f != "wheelSideBeam" else w.get("wheelSideBeamSpring", 0) for b, f in zip(beams, fam)], float)
    ke = np.array([b[2] if f != "wheelSideBeam" else w.get("wheelSideBeamSpringExpansion", b[2]) for b, f in zip(beams, fam)],
                  float)
    tris = []
    for i in range(rays):
        i2, nx = 2 * i, 2 * ((i + 1) % rays)
        oh, ih, noh, nih = hub0 + i2, hub0 + i2 + 1, hub0 + nx, hub0 + nx + 1
        it, ot, nit, not_ = tire0 + i2, tire0 + i2 + 1, tire0 + nx, tire0 + nx + 1
        tris += [(ih, noh, oh), (ih, nih, noh), (it, ot, nit), (nit, ot, not_), (ot, oh, noh), (ot, noh, not_),
                 (ih, it, nit), (nih, ih, nit)]
    free = np.zeros(len(pos), bool)
    free[tire0:hub0] = True
    axis = (np.asarray(n2, float) - np.asarray(n1, float))
    axis /= np.linalg.norm(axis)
    centre = (np.asarray(n1, float) + np.asarray(n2, float)) / 2
    return dict(pos=pos, A=A, B=Bn, L0=L * pre, kc=kc, ke=ke, tris=np.array(tris), free=free, axis=axis,
                centre=centre, mid_r=(w["radius"] + w["hubRadius"]) / 2, m=w["nodeWeight"], p=w.get("pressurePSI", 10) * PSI,
                R=w["radius"], fam=fam)


def forces(S, x, floor=None, kfloor=2e6):
    d = x[S["B"]] - x[S["A"]]
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    ext = L - S["L0"]
    k = np.where(ext < 0, S["kc"], S["ke"])
    f = (k * ext)[:, None] * u
    F = np.zeros_like(x)
    np.add.at(F, S["A"], f)
    np.add.at(F, S["B"], -f)
    # pressure: p * area along the normal pointing out of the air volume, a third to each corner
    t = S["tris"]
    p0, p1, p2 = x[t[:, 0]], x[t[:, 1]], x[t[:, 2]]
    n = 0.5 * np.cross(p1 - p0, p2 - p0)
    c = (p0 + p1 + p2) / 3
    rel = c - S["centre"]
    radial = rel - (rel @ S["axis"])[:, None] * S["axis"]
    ring = S["centre"] + radial / np.linalg.norm(radial, axis=1)[:, None] * S["mid_r"]
    sgn = np.sign(np.einsum("ij,ij->i", n, c - ring))
    fp = (S["p"] * sgn)[:, None] * n / 3
    for j in range(3):
        np.add.at(F, t[:, j], fp)
    reaction = 0.0
    if floor is not None:                  # floor below the wheel: z >= floor
        pen = floor - x[:, 2]
        hit = (pen > 0) & S["free"]
        F[hit, 2] += kfloor * pen[hit]
        reaction = kfloor * pen[hit].sum()
    F[~S["free"]] = 0
    return F, reaction


def relax(S, x, floor=None, steps=30000):
    v = np.zeros_like(x)
    dt, m = 1.0e-4, S["m"]
    c = 2 * m * 600.0                       # heavy viscous damping, only to settle
    reaction = 0.0
    for _ in range(steps):
        F, reaction = forces(S, x, floor)
        v += dt * (F - c * v) / m
        v[~S["free"]] = 0
        x = x + dt * v
    return x, reaction


def curve(w, n1, n2, deflections=(0.002, 0.004, 0.006, 0.008, 0.010, 0.014)):
    S = setup(w, n1, n2)
    x, _ = relax(S, S["pos"].copy())
    tz = x[S["free"], 2]
    centre_z = S["centre"][2]
    bottom = tz.min()
    free_r = centre_z - bottom
    pts = []
    for d in deflections:
        x2, F = relax(S, x.copy(), floor=bottom + d, steps=30000)
        pts.append((d, F))
    return free_r, pts


def at_load(pts, load):
    d = [0.0] + [p[0] for p in pts]
    f = [0.0] + [p[1] for p in pts]
    return float(np.interp(load, f, d))


def report(name, w, n1, n2, load):
    free_r, pts = curve(w, n1, n2)
    print("%s: %d rays, %.0f psi, built radius %.3f, inflated %.4f" % (name, w.get("numRays", 10),
                                                                        w.get("pressurePSI", 10), w["radius"], free_r))
    print("   squash mm: " + "  ".join("%4.0f" % (d * 1000) for d, _ in pts))
    print("   load N:    " + "  ".join("%4.0f" % F for _, F in pts))
    k = (pts[2][1] - pts[0][1]) / (pts[2][0] - pts[0][0])
    print("   stiffness 2-6 mm: %.0f kN/m; under %.0f N static load it squashes %.1f mm" %
          (k / 1000, load, at_load(pts, load) * 1000))


def main():
    if "--pigeon" in sys.argv:
        w = dict(wm.PIGEON, pressurePSI=22, wheelSideBeamPrecompression=0.99, wheelReinfBeamPrecompression=0.99,
                 wheelTreadBeamPrecompression=0.99, wheelTreadReinfBeamPrecompression=0.99,
                 wheelPeripheryBeamPrecompression=0.99, wheelPeripheryReinfBeamPrecompression=0.99)
        # the pigeon weighs ~500 kg; about a quarter on each rear wheel
        report("pigeon 10x6 race", w, (0, -0.07, 0.3), (0, 0.07, 0.3), 500 * 9.81 / 4)
        return
    sys.path.insert(0, os.path.join(wm.ROOT, "tools"))
    from cars import CARS  # noqa: E402
    car = CARS.get("rental_9hp") or next(iter(CARS.values()))
    g = car["mass"] * 9.81
    seen = set()
    for name, w, n1, n2, _ in wm.kart_wheels():
        if name[0] in seen:
            continue
        seen.add(name[0])
        share = car.get("front_weight", 0.42) if name[0] == "F" else 1 - car.get("front_weight", 0.42)
        report(name, w, n1, n2, g * share / 2)


if __name__ == "__main__":
    main()
