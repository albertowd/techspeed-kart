"""Find a 3D point (model coordinates) from its pixels in two or more reference photos, through the cameras
tools/dev/photo_match.py fits.

    python tools/dev/photo_triangulate.py side:268,433 rear:1068,457 [rear3q:895,500 ...]
"""
import glob
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import photo_match as pm  # noqa: E402


def cameras(names):
    cams = {}
    for n in names:
        v = pm.VIEWS[n]
        w, h = Image.open(glob.glob(os.path.join(pm.ROOT, "references", "rental", v["photo"] + "*.jpg"))[0]).size
        size = (int(pm.GRID_W), int(round(h * pm.GRID_W / w)))
        cams[n] = (pm.solve(v, size), size)
    return cams


def proj(cam, size, X):
    pc = (np.asarray(X) - cam["C"]) @ cam["R"]
    return np.array([size[0] / 2 + cam["f"] * pc[0] / -pc[2], size[1] / 2 - cam["f"] * pc[1] / -pc[2]])


def triangulate(obs, cams):
    X = np.array([0.3, 0.0, 0.3])
    for _ in range(60):
        r, J = [], []
        for n, px in obs:
            cam, size = cams[n]
            p = proj(cam, size, X)
            r.extend(p - px)
            for k in range(3):
                d = np.zeros(3)
                d[k] = 1e-6
            J.extend(np.array([(proj(cam, size, X + np.eye(3)[k] * 1e-6) - p) / 1e-6 for k in range(3)]).T)
        r, J = np.array(r), np.array(J)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + step
        if np.abs(step).max() < 1e-7:
            break
    return X, math.sqrt(r @ r / len(obs))


if __name__ == "__main__":
    obs = []
    for a in sys.argv[1:]:
        n, xy = a.split(":")
        obs.append((n, np.array([float(c) for c in xy.split(",")])))
    X, rms = triangulate(obs, cameras({n for n, _ in obs}))
    print("point", np.round(X, 3), "rms %.1f px" % rms)
