"""Draw model curves (lists of 3D points, model coordinates, ground at z = 0) over a reference photo through the camera
tools/dev/photo_match.py fits, to compare a part's centre line with the photo without reading pixels off a grid.

    python tools/dev/photo_trace.py <view> <out.jpg> <crop x0,y0,x1,y1 | full> <name>=x,y,z;x,y,z;... ...
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import photo_match as pm  # noqa: E402
import photo_triangulate as pt  # noqa: E402

COLOURS = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 160, 0), (255, 255, 0)]


def trace(view, out, crop, curves):
    v = pm.VIEWS[view]
    photo = Image.open(glob.glob(os.path.join(pm.ROOT, "references", "rental", v["photo"] + "*.jpg"))[0]).convert("RGB")
    cam, size = pt.cameras([view])[view]
    photo = photo.resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(photo)
    for k, (name, pts) in enumerate(curves):
        px = [tuple(pt.proj(cam, size, p)) for p in pts]
        d.line(px, fill=COLOURS[k % len(COLOURS)], width=2)
        d.text(px[0], name, fill=COLOURS[k % len(COLOURS)])
    if crop != "full":
        x0, y0, x1, y1 = [int(c) for c in crop.split(",")]
        photo = photo.crop((x0, y0, x1, y1))
        photo = photo.resize((1200, int(photo.size[1] * 1200 / photo.size[0])), Image.LANCZOS)
    photo.save(out, quality=90)


if __name__ == "__main__":
    curves = []
    for a in sys.argv[4:]:
        name, s = a.split("=", 1)
        curves.append((name, [tuple(float(c) for c in q.split(",")) for q in s.split(";")]))
    trace(sys.argv[1], sys.argv[2], sys.argv[3], curves)
