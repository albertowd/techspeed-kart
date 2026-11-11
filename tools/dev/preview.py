"""Render previews of a source model: python preview.py <source.blend|fbx> <outdir> [views...]"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import blender_bridge as bb  # noqa: E402

src, outdir = sys.argv[1], sys.argv[2]
views = sys.argv[3:] or ["side", "front", "top", "q34", "rear34"]
os.makedirs(outdir, exist_ok=True)
meta = bb.open_source(src)
name = os.path.splitext(os.path.basename(src))[0]
CAMS = {
    "side":     ((4.0, 0.0, 0.30), (0, 0, 0.25), dict(ortho=2.3)),     # kart's right side
    "front":    ((0.0, 4.0, 0.35), (0, 0, 0.25), dict(ortho=1.8)),
    "top":      ((0.0, 0.0, 4.0), (0, 0.001, 0), dict(ortho=2.3)),
    "q34":      ((-1.8, 2.3, 1.3), (0, 0.1, 0.2), dict(lens=50)),
    "rear34":   ((-1.9, -2.2, 1.4), (0, -0.1, 0.2), dict(lens=50)),
    "front34r": ((1.6, 2.4, 0.7), (0, 0.1, 0.25), dict(lens=50)),
    "cockpit":  ((0.0, -0.45, 0.95), (0, 0.6, 0.25), dict(lens=24)),
    "engine_rr": ((1.5, -1.5, 0.95), (0.40, -0.20, 0.22), dict(lens=45)),   # as rental/engine-air-filter.jpg
    "engine_low": ((1.25, -1.15, 0.62), (0.48, -0.28, 0.27), dict(lens=50)),
    "engine_top": ((0.30, -0.25, 2.6), (0.30, -0.249, 0.0), dict(ortho=1.1)),
}
for v in views:
    pos, tgt, kw = CAMS[v]
    bb.render(os.path.join(outdir, "%s_%s.png" % (name, v)), pos, tgt, samples=24, size=(1000, 700), **kw)
