"""Dynamic test of a BeamNG pressureWheel: the wheel rebuilt as the game builds it (bng_wheel_modes),
inflated, loaded by its share of the kart and rolled along a flat floor at the game's 2 kHz step, free
to spin (no brake, like the rental's front wheels). A sound wheel settles, rolls at speed / radius and
keeps the tyre centred on the rim; an unstable one gains vibration energy, spins up or walks off the rim.

Beams follow the game: spring plus damping on the beam's rate of stretch, the damping rate low-passed
where the family sets a DampCutoffHz; sidewalls anisotropic (compression / expansion spring and damping);
rest lengths times precompression. The floor stops tyre nodes and grips them with Coulomb friction.

    python tools/dev/bng_wheel_roll.py                 # front and rear wheels of the built rental kart
    python tools/dev/bng_wheel_roll.py --pigeon        # the stock pigeon race tyre, for comparison
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bng_wheel_modes as wm  # noqa: E402
import bng_tyre_load as tl  # noqa: E402

DT = wm.DT


def kerb(height=0.03, period=0.25, start=1.0, count=8):
    """Rumble strip: rounded bumps across the path from x = start; returns (h(x), dh/dx(x))."""
    def h(x):
        u = (x - start) / period
        on = (u >= 0) & (u < count)
        s_ = np.sin(np.pi * (u % 1.0))
        return np.where(on, height * s_ ** 2, 0.0), np.where(on, height * 2 * s_ * np.cos(np.pi * (u % 1.0)) * np.pi / period, 0.0)
    return h


def simulate(w, n1, n2, load, speed=8.0, seconds=1.5, mu=1.0, report=True, label="", side=0.0, drop=0.0,
             bumps=None):
    """bumps: kerb() profile laid on the floor (None: flat)."""
    # lay the axle along +y at a fixed height so the wheel rolls along +x
    half = np.linalg.norm(np.asarray(n2, float) - np.asarray(n1, float)) / 2
    S = tl.setup(w, (0, -half, 0.3), (0, half, 0.3))
    pos, fam = S["pos"], S["fam"]
    n = len(pos)
    tire = S["free"].copy()
    mass = np.full(n, w["hubNodeWeight"])
    mass[tire] = w["nodeWeight"]
    axle_m = load / 9.81 / 2                     # the load carried by the two axle nodes
    mass[0] = mass[1] = axle_m
    # beam damping (compression / expansion) and low-pass factor per beam
    cc, ce, lp = [], [], []
    for f in fam:
        if f == "hub":
            c = w.get("hubTreadBeamDamp", 0)
            cc.append(c), ce.append(c), lp.append(1.0)
            continue
        c = w.get(f + "Damp", 0)
        cc.append(c), ce.append(w.get(f + "DampExpansion", c))
        hz = w.get(f + "DampCutoffHz")
        lp.append(1.0 if not hz else DT / (DT + 1 / (2 * math.pi * hz)))
    cc, ce, lp = np.array(cc, float), np.array(ce, float), np.array(lp)

    x = pos.copy()
    # inflate with the rim held, then drop onto the floor
    x, _ = tl.relax(S, x, steps=8000)
    R_free = S["centre"][2] - x[tire, 2].min()
    floor = x[tire, 2].min() - 0.0005 - drop          # drop: start this far above the floor
    v = np.zeros_like(x)
    v[:, 0] = speed
    # spin the wheel up to rolling speed: v = omega x r about the axle
    axis = S["axis"]
    rel = x - S["centre"]
    omega = speed / R_free                        # rolling forwards (+x) on a floor below (-z)
    v += omega * np.cross(axis, rel)
    v[0] = v[1] = (speed, 0, 0)
    rate_f = np.zeros(len(S["A"]))
    A, B = S["A"], S["B"]
    steps = int(seconds / DT)
    rec = []
    centre0 = x[[0, 1]].mean(0)
    for k in range(steps):
        d = x[B] - x[A]
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        ext = L - S["L0"]
        rate = np.einsum("ij,ij->i", v[B] - v[A], u)
        rate_f += lp * (rate - rate_f)
        comp = ext < 0
        kk = np.where(comp, S["kc"], S["ke"])
        c = np.where(comp, cc, ce)
        f = (kk * ext + c * rate_f)[:, None] * u
        F = np.zeros_like(x)
        np.add.at(F, A, f)
        np.add.at(F, B, -f)
        # pressure
        Fp, _ = tl.forces(dict(S, kc=S["kc"] * 0, ke=S["ke"] * 0, free=np.ones(n, bool), centre=x[[0, 1]].mean(0)), x)
        F += Fp
        F[:, 2] -= mass * 9.81
        # floor: stop the node, then Coulomb friction from the impulse it took
        v += DT * F / mass[:, None]
        xn = x + DT * v
        if bumps is None:
            hz, slope = np.full(n, floor), np.zeros(n)
        else:
            hz, slope = bumps(xn[:, 0])
            hz = hz + floor
        pen = (xn[:, 2] < hz) & tire
        if pen.any():
            nrm = np.column_stack([-slope[pen], np.zeros(pen.sum()), np.ones(pen.sum())])
            nrm /= np.linalg.norm(nrm, axis=1)[:, None]
            # impulse along the surface normal that puts the node back on the surface
            J = ((hz[pen] - x[pen, 2]) / DT - v[pen, 2]) / nrm[:, 2]
            J = np.maximum(J, 0)
            v[pen] += J[:, None] * nrm
            vt = v[pen] - np.einsum("ij,ij->i", v[pen], nrm)[:, None] * nrm
            s = np.linalg.norm(vt, axis=1)
            dv = np.minimum(s, mu * J)
            v[pen] -= vt / np.maximum(s, 1e-9)[:, None] * dv[:, None]
        # axle nodes: carried along at the kart's speed, free to move up and down only
        v[0, :2] = v[1, :2] = (speed, side)          # side: sideways speed of the axle, as in a slip angle
        vz = v[[0, 1], 2].mean()
        v[0, 2] = v[1, 2] = vz
        x = x + DT * v
        if k % 20 == 0:
            c_ax = x[[0, 1]].mean(0)
            relt = x[tire] - c_ax
            # spin: angular velocity of the tyre about the axle, from the tread nodes
            rp = relt - np.outer(relt @ axis, axis)                  # radial part: tread nodes sit off-centre
            vrel = v[tire] - v[[0, 1]].mean(0)
            w_t = np.cross(rp, vrel) @ axis / np.einsum("ij,ij->i", rp, rp)
            lat = (relt @ axis).mean()                               # tyre walk along the axle
            # vibration: tyre node motion left after taking out the wheel's spin about the axle
            vib = 0.5 * (mass[tire, None] * (vrel - w_t.mean() * np.cross(axis, rp)) ** 2).sum()
            rmin = np.sqrt(np.einsum("ij,ij->i", rp, rp)).min()             # tyre squashed towards the rim
            rec.append((k * DT, c_ax[2] - floor, w_t.mean(), lat, vib, rmin))
    rec = np.array(rec)
    if report:
        late = rec[rec[:, 0] > seconds * 0.5]
        spin = late[:, 2].mean() * R_free / speed
        print("%s: ride height %.4f (free %.4f), roll ratio %.3f (1 = rolling true), tyre walk %.1f mm, "
              "vibration %.2f J -> %.2f J, tread closest to the axle %.4f (rim %.4f)"
              % (label, late[:, 1].mean(), R_free, spin, abs(late[:, 3]).max() * 1000, rec[5:25, 4].mean(),
                 late[:, 4].mean(), rec[:, 5].min(), w["hubRadius"]))
    return rec


def main():
    speed = 8.0
    if "--pigeon" in sys.argv:
        w = dict(wm.PIGEON, pressurePSI=22, hubNodeWeight=0.3, wheelSideBeamDamp=25, wheelSideBeamDampExpansion=30,
                 wheelReinfBeamDamp=150, wheelReinfBeamDampCutoffHz=500, wheelTreadBeamDamp=52,
                 wheelTreadBeamDampCutoffHz=500, wheelTreadReinfBeamDamp=45, wheelTreadReinfBeamDampCutoffHz=500,
                 wheelPeripheryBeamDamp=25, wheelPeripheryBeamDampCutoffHz=500, wheelPeripheryReinfBeamDamp=25,
                 wheelPeripheryReinfBeamDampCutoffHz=500, hubTreadBeamDamp=10,
                 wheelSideTransitionZone=0.1, wheelSideBeamPrecompression=0.99, wheelReinfBeamPrecompression=0.99,
                 wheelTreadBeamPrecompression=0.99, wheelTreadReinfBeamPrecompression=0.99,
                 wheelPeripheryBeamPrecompression=0.99, wheelPeripheryReinfBeamPrecompression=0.99)
        simulate(w, (0, -0.07, 0.3), (0, 0.07, 0.3), 500 * 9.81 / 4, speed, label="pigeon 10x6 race")
        return
    sys.path.insert(0, os.path.join(wm.ROOT, "tools"))
    from cars import CARS  # noqa: E402
    car = CARS.get("rental_9hp") or next(iter(CARS.values()))
    seen = set()
    for name, w, n1, n2, _ in wm.kart_wheels():
        if name[0] in seen:
            continue
        seen.add(name[0])
        share = car.get("front_weight", 0.42) if name[0] == "F" else 1 - car.get("front_weight", 0.42)
        simulate(w, n1, n2, car["mass"] * 9.81 * share / 2, speed, label=name)
        simulate(w, n1, n2, car["mass"] * 9.81 * share / 2, 15.0, seconds=1.2, bumps=kerb(), label=name + " kerb 15 m/s")


if __name__ == "__main__":
    main()
