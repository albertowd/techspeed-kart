"""Where a pixel of a reference photo lands on a known plane (x, y or z = const, model coordinates), through the
camera tools/dev/photo_match.py fits: for parts of known side position, e.g. the protector's side tubes.

    python tools/dev/photo_unproject.py side x=0.69 385,545 440,450 690,530
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import photo_triangulate as pt  # noqa: E402


def unproject(cam, size, px, axis, value):
    d_cam = np.array([(px[0] - size[0] / 2) / cam["f"], -(px[1] - size[1] / 2) / cam["f"], -1.0])
    d = cam["R"] @ d_cam
    t = (value - cam["C"][axis]) / d[axis]
    return cam["C"] + t * d


if __name__ == "__main__":
    view, plane = sys.argv[1], sys.argv[2]
    axis, value = "xyz".index(plane[0]), float(plane.split("=")[1])
    cam, size = pt.cameras([view])[view]
    for a in sys.argv[3:]:
        px = [float(c) for c in a.split(",")]
        print(a, np.round(unproject(cam, size, px, axis, value), 3))
