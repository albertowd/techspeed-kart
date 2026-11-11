"""Preview renders.

    python tools/render_previews.py <rental|rental_v1> -> build/skins/<variant>/<skin>/preview.jpg
    python tools/render_previews.py --beamng-thumbs   -> <config>.jpg in build/beamng/vehicles/techspeed_kart
"""
import os
import sys

import bpy
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blender_bridge as bb  # noqa: E402
from cars import ACTIVE_CARS, BUILD_VARIANTS, built_skins  # noqa: E402
from textures import skins_dir as skins_dir_of  # noqa: E402

ROOT = os.path.dirname(HERE)
CAM = ((-1.95, 2.5, 1.15), (0, 0.0, 0.2))      # (pulled back: the longer nose was cut off at the edge)


def livery_image():
    for img in bpy.data.images:
        if img.name.lower().startswith("body_livery") or img.filepath.replace("\\", "/").lower().endswith("/body_livery.png"):
            return img
    return None


def swap_skin_textures(skin_dir, base_dir):
    """Point every image node at the skin's copy of its texture (livery, worn tyres), else the base one."""
    for stem in ("body_livery", "tyre"):
        path = os.path.join(skin_dir, stem + ".png")
        if not os.path.exists(path):
            path = os.path.join(base_dir, stem + ".png")
        new = bpy.data.images.load(path, check_existing=False)
        for mat in bpy.data.materials:
            if mat.node_tree:
                for nd in mat.node_tree.nodes:
                    if nd.type == "TEX_IMAGE" and nd.image is not None and nd.image.name.startswith(stem):
                        nd.image = new


def skin_previews(variant, skins=None):
    """Render preview.jpg for the liveries painted in build/skins/<variant> (tools/paint_skins.py), or the named ones."""
    src = os.path.join(ROOT, "source", variant, variant + ".blend")
    bb.open_source(src)
    img = livery_image()
    skins_dir = skins_dir_of(variant)
    for skin in skins or built_skins(os.listdir(skins_dir)):
        if img is not None:
            swap_skin_textures(os.path.join(skins_dir, skin), os.path.join(ROOT, "source", variant, "textures"))
        tmp = os.path.join(skins_dir, skin, "_preview.png")
        bb.render(tmp, CAM[0], CAM[1], lens=50, size=(1022, 575), samples=48)
        Image.open(tmp).convert("RGB").save(os.path.join(skins_dir, skin, "preview.jpg"), quality=92)
        os.remove(tmp)
        print("preview", skin)


def beamng_thumbs():
    vdir = os.path.join(ROOT, "build", "beamng", "vehicles", "techspeed_kart")
    for cid, car in ACTIVE_CARS.items():
        pc, variant = cid.replace("techspeed_", ""), car["variant"]
        skins_dir = skins_dir_of(variant)
        if variant not in BUILD_VARIANTS:
            continue
        first = sorted(os.listdir(skins_dir))[0]
        p = os.path.join(skins_dir, first, "preview.jpg")
        if os.path.exists(p):
            Image.open(p).resize((500, 281), Image.LANCZOS).save(os.path.join(vdir, pc + ".jpg"), quality=90)
    p = os.path.join(vdir, "rental_13hp.jpg")
    if os.path.exists(p):
        Image.open(p).save(os.path.join(vdir, "default.jpg"))


if __name__ == "__main__":
    if sys.argv[1] == "--beamng-thumbs":
        beamng_thumbs()
    else:
        skin_previews(sys.argv[1])
