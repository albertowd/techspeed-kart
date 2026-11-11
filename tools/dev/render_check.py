"""Check renders of a source model after a visual change: front and rear three-quarter views and a
top view, written to build/renders/<variant>_<view>.jpg.

    python tools/dev/render_check.py [rental|racing ...]
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blender_bridge as bb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "build", "renders")
VIEWS = [  # name, camera, target, lens, ortho scale
    ("front34", (-1.85, 2.35, 1.05), (0, 0.0, 0.22), 50, None),
    ("rear34", (1.9, -2.3, 1.15), (0, -0.05, 0.2), 50, None),
    ("top", (0, 0.07, 4.0), (0, 0.07, 0), 50, 3.3),
    ("cockpit", (0.42, -0.30, 0.95), (0.0, 0.42, 0.25), 35, None),
    ("side", (-3.2, 0.0, 0.10), (0, 0.0, 0.16), 50, 2.6),      # ground level: ride height, axles, spindles
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for variant in sys.argv[1:] or ["rental"]:
        bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
        for name, cam, target, lens, ortho in VIEWS:
            tmp = os.path.join(OUT, "_%s_%s.png" % (variant, name))
            bb.render(tmp, cam, target, lens=lens, size=(1400, 900), samples=32, ortho=ortho)
            out = os.path.join(OUT, "%s_%s.jpg" % (variant, name))
            Image.open(tmp).convert("RGB").save(out, quality=92)
            os.remove(tmp)
            print("render", out)


if __name__ == "__main__":
    main()
