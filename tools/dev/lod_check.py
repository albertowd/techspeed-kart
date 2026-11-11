"""Render the Assetto Corsa LODs (export_ac.AC_LODS) of a source model side by side, each from the same three-quarter
view, with its triangle count: build/renders/<variant>_lods.jpg.

    python tools/dev/lod_check.py [rental|rental_v1 ...]

Each LOD is baked as export_ac bakes it (blender_bridge.collect) and put back in the scene as plain objects with the
source's materials, so the render shows exactly the meshes the kn5 gets.
"""
import os
import sys

import bpy
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import blender_bridge as bb  # noqa: E402
from export_ac import AC_LODS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "build", "renders")
CAM, TARGET = (-1.85, 2.35, 1.05), (0, 0.0, 0.22)


def show(nodes, baked):
    """Plain objects for the baked meshes, at their nodes' places; returns them."""
    made = []
    for b in baked:
        ob = bb.baked_object("lod_" + b.name, b)
        ob.matrix_world = bb.Matrix(bb.node_world(nodes, b.node).tolist())
        bpy.context.scene.collection.objects.link(ob)
        made.append(ob)
    return made


def main():
    os.makedirs(OUT, exist_ok=True)
    for variant in sys.argv[1:] or ["rental"]:
        bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
        source = [o for o in bpy.data.objects if o.type == "MESH" and not o.hide_render]
        tiles = []
        for suffix, lo, hi, reduction in AC_LODS:
            nodes, baked = bb.collect(reduction)
            for o in source:
                o.hide_render = True
            made = show(nodes, baked)
            path = os.path.join(OUT, "%s_lod%s.png" % (variant, suffix or "_A"))
            bb.render(path, CAM, TARGET, size=(800, 500), samples=24)
            for o in made:
                bpy.data.objects.remove(o)
            for o in source:
                o.hide_render = False
            im = Image.open(path).convert("RGB")
            ImageDraw.Draw(im).text((12, 10), "LOD%s  %d-%d m  %d triangles" % (
                suffix or "_A", lo, hi, sum(len(b.tris) for b in baked)), fill=(0, 0, 0))
            tiles.append(im)
            os.remove(path)
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * 2, h * 2))
        for i, im in enumerate(tiles):
            sheet.paste(im, ((i % 2) * w, (i // 2) * h))
        out = os.path.join(OUT, variant + "_lods.jpg")
        sheet.save(out, quality=90)
        print("wrote", out)


if __name__ == "__main__":
    main()
