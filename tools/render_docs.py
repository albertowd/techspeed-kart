"""Showcase renders for the README and the overtake.gg page.

    python tools/render_docs.py

Writes docs/screens/<n>-<name>.jpg from the source models and docs/liveries.jpg from the first four liveries'
previews of each version. The liveries it shows are painted into build/skins (tools/paint_skins.py) as needed.
"""
import os
import sys

import bpy
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blender_bridge as bb  # noqa: E402
import paint_skins  # noqa: E402
import textures  # noqa: E402
from render_previews import skin_previews, swap_skin_textures  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "screens")
SIZE = (1600, 900)

SHOTS = [
    ("1-rental", "rental", "01_techspeed", (-1.85, 2.35, 1.05), (0, 0.0, 0.22), 50),
    ("2-rental-v1", "rental_v1", "07_techspeed", (-1.85, 2.35, 1.05), (0, 0.0, 0.22), 50),
    ("3-rental-rear", "rental", "12_techspeed", (1.9, -2.3, 1.15), (0, -0.05, 0.2), 50),
    ("4-rental-v1-side", "rental_v1", "23_techspeed", (3.4, 0.35, 0.55), (0, 0.0, 0.25), 60),
]


GRID = 4                                           # liveries per version in docs/liveries.jpg


def set_skin(variant, skin):
    swap_skin_textures(os.path.join(textures.skins_dir(variant), skin), os.path.join(ROOT, "source", variant, "textures"))


def main():
    os.makedirs(OUT, exist_ok=True)
    grid = {v: sorted(textures.skins_for(v))[:GRID] for v in ("rental", "rental_v1")}
    for v in grid:
        paint_skins.paint(v, sorted(set(grid[v]) | {s for _, sv, s, *_ in SHOTS if sv == v}))
    current = None
    for name, variant, skin, cam, target, lens in SHOTS:
        if current != variant:
            bb.open_source(os.path.join(ROOT, "source", variant, variant + ".blend"))
            current = variant
        set_skin(variant, skin)
        tmp = os.path.join(OUT, "_" + name + ".png")
        bb.render(tmp, cam, target, lens=lens, size=SIZE, samples=96)
        Image.open(tmp).convert("RGB").save(os.path.join(OUT, name + ".jpg"), quality=90)
        os.remove(tmp)
        print("wrote", name)
    # livery grid: the first 4 liveries of the rental v2 and v1
    tiles = []
    for variant, skins in grid.items():
        skin_previews(variant, skins)
        for skin in skins:
            p = os.path.join(textures.skins_dir(variant), skin, "preview.jpg")
            if os.path.exists(p):
                tiles.append(Image.open(p).convert("RGB").resize((511, 287), Image.LANCZOS))
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * 511, rows * 287), (230, 230, 232))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * 511, (i // cols) * 287))
        grid.save(os.path.join(ROOT, "docs", "liveries.jpg"), quality=88)
        print("wrote liveries")


if __name__ == "__main__":
    main()
